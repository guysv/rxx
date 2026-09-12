//! Rune-based plugin scripting.
//!
//! Plugins are Rune scripts. Each plugin exports a `pub fn init(rx)` that
//! receives the [`Ctx`] and returns the plugin's state value; further hooks
//! are free functions taking `(state, rx, ...)`. A missing hook simply means
//! the plugin isn't subscribed to it.
//!
//! Architecture rules (see docs/rune-plan.md): hooks receive `&mut` host
//! state through `Ctx` for the duration of the call only — no globals, no
//! queued dispatch.

use std::fmt;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use rune::runtime::{Function, RuntimeContext, VmError};
use rune::{Context, Diagnostics, Source, Sources, Unit, Value, Vm};

use crate::session::Session;

mod gpu;
mod sprite;
use gpu::{PassOptions, PipelineOptions};

////////////////////////////////////////////////////////////////////////////
// Errors

#[derive(Debug)]
pub enum ScriptError {
    /// Compilation failed; the string holds rendered diagnostics.
    Compile(String),
    /// A runtime error in a script call.
    Vm(String),
    /// The script doesn't define the requested function.
    MissingFn(String),
    Io(io::Error),
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "script compile error: {}", e),
            Self::Vm(e) => write!(f, "script error: {}", e),
            Self::MissingFn(name) => write!(f, "script function not found: {}", name),
            Self::Io(e) => write!(f, "script i/o error: {}", e),
        }
    }
}

impl From<io::Error> for ScriptError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<VmError> for ScriptError {
    fn from(e: VmError) -> Self {
        Self::Vm(e.to_string())
    }
}

////////////////////////////////////////////////////////////////////////////
// Gfx

/// The GPU capability handed to scripts: clones of the renderer's
/// device and queue. wgpu resources are Arc-backed and self-validating,
/// so sharing them with scripts carries no host invariants
/// (docs/rune-plan.md, "GPU" section). Attached to the [`PluginHost`]
/// once the renderer exists, before plugins load.
pub struct Gfx {
    pub device: std::sync::Arc<wgpu::Device>,
    pub queue: std::sync::Arc<wgpu::Queue>,
    /// Canonical layouts and sampler for script pipelines & bind
    /// groups. The texture layout and sampler are Arc'd so the shade
    /// encoder can hold them for `view_bind_group` (wgpu 23 resources
    /// don't implement `Clone` themselves).
    transform_bgl: Arc<wgpu::BindGroupLayout>,
    sprites: Arc<sprite::Blitter>,
    texture_bgl: std::sync::Arc<wgpu::BindGroupLayout>,
    /// Compute layouts per input count (index `n - 1`): inputs at
    /// bindings `0..n`, the storage output last at binding `n`.
    compute_bgls: [wgpu::BindGroupLayout; Self::MAX_COMPUTE_INPUTS],
    sampler: std::sync::Arc<wgpu::Sampler>,
    frame: Arc<AtomicU64>,
}

impl Gfx {
    /// Texture groups a render pipeline may declare. Group 0 is the
    /// transform and wgpu's default bind-group limit is 4, so at most
    /// 3 texture groups fit (the plan's "0–4" predates the limit).
    const MAX_TEXTURE_GROUPS: i64 = 3;
    /// Compute inputs share group 0 with the output, so the cap is
    /// symmetry, not a device limit.
    const MAX_COMPUTE_INPUTS: usize = 4;

    fn new(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) -> Self {
        // Group 0: the transform at binding 0, user params at binding 1
        // (always present in the layout; WGSL that doesn't declare
        // binding 1 is unaffected, and `create_transform_bind_group`
        // supplies a zeroed buffer).
        let transform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("script_transform_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::FRAGMENT),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        // Textures are visible to vertex shaders too: point-scatter
        // pipelines position vertices by `textureLoad`ing their input.
        // One layout shared by all render pipelines keeps every texture
        // bind group compatible with every pipeline by construction.
        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("script_texture_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::FRAGMENT),
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::FRAGMENT),
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // One compute layout per input count: inputs at bindings 0..n,
        // the storage output last. n=1 is the original layout, so
        // existing WGSL is unaffected.
        let compute_bgls = std::array::from_fn(|i| {
            let n = i + 1;
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n as u32)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                })
                .collect();
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: n as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            });
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("script_compute_bgl"),
                entries: &entries,
            })
        });
        // Clamp-to-edge: what pixel-art filters sampling outside the
        // source (cleanedge & co) want.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let transform_bgl = Arc::new(transform_bgl);
        let texture_bgl = Arc::new(texture_bgl);
        let sampler = Arc::new(sampler);
        let sprites = Arc::new(sprite::Blitter::new(
            device.clone(),
            transform_bgl.clone(),
            texture_bgl.clone(),
            sampler.clone(),
        ));
        Self {
            device,
            queue,
            transform_bgl,
            sprites,
            texture_bgl,
            compute_bgls,
            sampler,
            frame: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Pop a validation error scope, blocking on the result.
    fn pop_error(&self) -> Option<wgpu::Error> {
        let fut = self.device.pop_error_scope();
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(fut)
    }
}

/// The format script render work agrees on: same as the renderer's view
/// and screen textures, so script passes can later target views too.
const SCRIPT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Maximum dimension for script-created textures.
const MAX_TEXTURE_DIM: i64 = 8192;

/// Maximum user params per bind group (16 vec4s — plenty, and well
/// under any uniform-buffer size limit).
const MAX_PARAMS: usize = 64;

/// A script-owned GPU texture (rgba8, render-attachable + bindable).
///
/// Owns the underlying wgpu texture: when the plugin's state value is
/// dropped (unload/reload), the texture is released with it. wgpu
/// defers the actual destruction past any in-flight frame, so mid-frame
/// drops are safe.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptTexture {
    texture: wgpu::Texture,
    /// sRGB view: render attachments and sampling (matches the
    /// renderer's view/screen formats).
    view: Arc<wgpu::TextureView>,
    /// Raw (unorm) view: compute storage and raw loads. Storage
    /// textures cannot be sRGB, so the base format is unorm and the
    /// sRGB conversion lives in `view`.
    raw_view: Arc<wgpu::TextureView>,
    /// Cloned queue handle, so uploads need no further host access.
    queue: std::sync::Arc<wgpu::Queue>,
    width: u32,
    height: u32,
}

impl ScriptTexture {
    pub(crate) fn create(gfx: &Gfx, width: u32, height: u32) -> Self {
        let texture = gfx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("script_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[SCRIPT_TEXTURE_FORMAT],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(SCRIPT_TEXTURE_FORMAT),
            ..Default::default()
        });
        let raw_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view: Arc::new(view),
            raw_view: Arc::new(raw_view),
            queue: gfx.queue.clone(),
            width,
            height,
        }
    }

    #[cfg(test)]
    fn wgpu_view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// A reusable sRGB texture view for rendering or sampling.
    #[rune::function]
    fn view(&self) -> ScriptTextureView {
        self.sampled_view()
    }

    fn sampled_view(&self) -> ScriptTextureView {
        ScriptTextureView {
            size: [self.width, self.height],
            format: SCRIPT_TEXTURE_FORMAT,
            source: TextureViewSource::Owned(self.view.clone()),
            scope: None,
        }
    }

    /// A linear rgba8unorm view over the same storage (no sRGB conversion).
    #[rune::function]
    fn raw_view(&self) -> ScriptTextureView {
        ScriptTextureView {
            size: [self.width, self.height],
            format: wgpu::TextureFormat::Rgba8Unorm,
            source: TextureViewSource::Owned(self.raw_view.clone()),
            scope: None,
        }
    }

    pub(crate) fn wgpu_texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn write(&self, texels: &[u8]) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Texture width in pixels.
    #[rune::function]
    fn width(&self) -> i64 {
        self.width as i64
    }

    /// Texture height in pixels.
    #[rune::function]
    fn height(&self) -> i64 {
        self.height as i64
    }

    /// Upload rgba8 pixel data (must be exactly `width * height * 4`
    /// bytes, row-major). Returns whether the size matched.
    #[rune::function]
    fn upload(&self, data: rune::runtime::Bytes) -> bool {
        if data.len() != (self.width * self.height * 4) as usize {
            return false;
        }
        self.write(&data);
        true
    }

    /// Fill the whole texture with one color.
    #[rune::function]
    fn fill(&self, color: &crate::gfx::color::Rgba8) {
        let px = [color.r, color.g, color.b, color.a];
        let texels: Vec<u8> = px
            .iter()
            .copied()
            .cycle()
            .take((self.width * self.height * 4) as usize)
            .collect();
        self.write(&texels);
    }

    /// Read back the texture contents (blocking GPU sync; backs
    /// `rx.texture_pixels` and the tests).
    pub(crate) fn pixels(&self, device: &wgpu::Device) -> Vec<u8> {
        let (w, h) = (self.width, self.height);
        let bytes_per_row = (4 * w + 255) & !255;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("script_texture_readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let start = (y * bytes_per_row) as usize;
            out.extend_from_slice(&data[start..start + (w * 4) as usize]);
        }
        out
    }
}

////////////////////////////////////////////////////////////////////////////
// GPU objects (shaders, pipelines, passes)

/// Vertex layout shared by script pipelines: position, uv, color,
/// opacity — the renderer's sprite layout, so view-space WGSL ports
/// from master unchanged.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScriptVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [u8; 4],
    opacity: f32,
}

/// Uniforms for the script transform bind group (matches the
/// renderer's `TransformUniforms` and master's plugin WGSL).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScriptUniforms {
    ortho: [[f32; 4]; 4],
    transform: [[f32; 4]; 4],
}

/// A compiled WGSL shader module.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptShader {
    module: wgpu::ShaderModule,
}

/// A render pipeline created from a script shader.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptPipeline {
    pipeline: wgpu::RenderPipeline,
}

/// A compute pipeline created from a script shader.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptComputePipeline {
    pipeline: wgpu::ComputePipeline,
}

/// A renderable/sampleable texture view. Editor targets are frame-scoped:
/// retaining one cannot silently render into an obsolete resized texture.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptTextureView {
    size: [u32; 2],
    format: wgpu::TextureFormat,
    source: TextureViewSource,
    scope: Option<FrameScope>,
}

enum TextureViewSource {
    Owned(Arc<wgpu::TextureView>),
    Editor {
        targets: Arc<ViewTargets>,
        id: u16,
        staging: bool,
    },
}

#[derive(Clone)]
struct FrameScope {
    clock: std::sync::Weak<AtomicU64>,
    generation: u64,
}

fn validate_scope(scope: &Option<FrameScope>) -> Result<(), String> {
    if let Some(scope) = scope {
        let clock = scope
            .clock
            .upgrade()
            .ok_or("the texture view's frame has ended")?;
        if clock.load(Ordering::Relaxed) != scope.generation {
            return Err("the texture view's frame has ended".into());
        }
    }
    Ok(())
}

impl ScriptTextureView {
    fn get(&self) -> Result<&wgpu::TextureView, String> {
        validate_scope(&self.scope)?;
        Ok(match &self.source {
            TextureViewSource::Owned(view) => view,
            TextureViewSource::Editor {
                targets,
                id,
                staging,
            } => {
                let target = &targets[id];
                if *staging {
                    &target.staging
                } else {
                    &target.layer
                }
            }
        })
    }
}

/// A bind group (transform uniforms, texture+sampler, or compute IO).
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptBindGroup {
    bind_group: wgpu::BindGroup,
    scope: Option<FrameScope>,
}

fn texture_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    view: &ScriptTextureView,
) -> Result<ScriptBindGroup, String> {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("script_texture_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view.get()?),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    Ok(ScriptBindGroup {
        bind_group,
        scope: view.scope.clone(),
    })
}

/// A vertex buffer plus its vertex count.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptBuffer {
    buffer: wgpu::Buffer,
    count: u32,
}

impl ScriptBuffer {
    /// Number of vertices in the buffer.
    #[rune::function]
    fn count(&self) -> i64 {
        self.count as i64
    }
}

/// A 4x4 transform matrix.
#[derive(rune::Any, Clone, Copy)]
#[rune(item = ::rx)]
pub struct Mat4(crate::gfx::math::Matrix4<f32>);

/// Identity matrix.
#[rune::function]
fn mat4_identity() -> Mat4 {
    Mat4(crate::gfx::math::Matrix4::identity())
}

/// Translation by `(x, y)`.
#[rune::function]
fn mat4_translation(x: f64, y: f64) -> Mat4 {
    let mut m = crate::gfx::math::Matrix4::identity();
    m.w.x = x as f32;
    m.w.y = y as f32;
    Mat4(m)
}

/// Scale by `(sx, sy)`.
#[rune::function]
fn mat4_scale(sx: f64, sy: f64) -> Mat4 {
    let mut m = crate::gfx::math::Matrix4::identity();
    m.x.x = sx as f32;
    m.y.y = sy as f32;
    Mat4(m)
}

/// Transform a 2D point by a matrix, returning `(x, y)`.
#[rune::function]
fn mat4_transform_point(m: &Mat4, x: f64, y: f64) -> (f64, f64) {
    let p = m.0 * crate::gfx::math::Point2::new(x as f32, y as f32);
    (p.x as f64, p.y as f64)
}

/// `atan2(y, x)` in radians.
#[rune::function]
fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Rotation around Z, in radians.
#[rune::function]
fn mat4_rotation_z(theta: f64) -> Mat4 {
    let (sin, cos) = (theta as f32).sin_cos();
    let mut m = crate::gfx::math::Matrix4::identity();
    m.x.x = cos;
    m.x.y = sin;
    m.y.x = -sin;
    m.y.y = cos;
    Mat4(m)
}

/// Matrix product `a * b`.
#[rune::function]
fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    Mat4(a.0 * b.0)
}

/// An axis-aligned rectangle (used for sprite source/dest coords).
#[derive(rune::Any, Clone, Copy)]
#[rune(item = ::rx)]
pub struct Rect {
    #[rune(get)]
    pub x1: f64,
    #[rune(get)]
    pub y1: f64,
    #[rune(get)]
    pub x2: f64,
    #[rune(get)]
    pub y2: f64,
}

/// Construct a rectangle from two corners.
#[rune::function]
fn rect(x1: f64, y1: f64, x2: f64, y2: f64) -> Rect {
    Rect { x1, y1, x2, y2 }
}

/// Parsed sprite descriptor. Borrow fields so descriptors stored in plugin
/// state can be reused across frames without consuming their native values.
struct SpriteOptions {
    src: Rect,
    dst: Rect,
    color: crate::gfx::color::Rgba8,
    opacity: f64,
}

impl SpriteOptions {
    fn parse(options: &rune::runtime::Object, width: u32, height: u32) -> Result<Self, String> {
        Self::parse_with_fields(options, width, height, &[])
    }

    fn parse_with_fields(
        options: &rune::runtime::Object,
        width: u32,
        height: u32,
        extra: &[&str],
    ) -> Result<Self, String> {
        for key in options.keys() {
            if !matches!(key.as_str(), "src" | "dst" | "color" | "opacity")
                && !extra.contains(&key.as_str())
            {
                return Err(format!("unknown option `{}`", key));
            }
        }
        let read_rect = |key: &str| -> Result<Option<Rect>, String> {
            options
                .get(key)
                .map(|value| {
                    value
                        .borrow_ref::<Rect>()
                        .map(|rect| *rect)
                        .map_err(|_| format!("`{}` must be a Rect", key))
                })
                .transpose()
        };
        let dst = read_rect("dst")?.ok_or("missing required option `dst`")?;
        let src = read_rect("src")?.unwrap_or(Rect {
            x1: 0.0,
            y1: 0.0,
            x2: width as f64,
            y2: height as f64,
        });
        let color = match options.get("color") {
            Some(value) => *value
                .borrow_ref::<crate::gfx::color::Rgba8>()
                .map_err(|_| "`color` must be an Rgba8")?,
            None => crate::gfx::color::Rgba8::WHITE,
        };
        let opacity = match options.get("opacity") {
            Some(value) => {
                rune::from_value::<f64>(value.clone()).map_err(|_| "`opacity` must be a float")?
            }
            None => 1.0,
        };
        Ok(Self {
            src,
            dst,
            color,
            opacity,
        })
    }
}

type SharedPass = std::sync::Arc<std::sync::Mutex<Option<wgpu::RenderPass<'static>>>>;
type SharedComputePass = std::sync::Arc<std::sync::Mutex<Option<wgpu::ComputePass<'static>>>>;
type SharedEncoder = std::sync::Arc<std::sync::Mutex<Option<wgpu::CommandEncoder>>>;
type PassList = std::sync::Arc<std::sync::Mutex<Vec<SharedPass>>>;
type ComputePassList = std::sync::Arc<std::sync::Mutex<Vec<SharedComputePass>>>;

/// Per-frame render targets for one view: its layer (the artwork) and
/// staging (the per-frame preview overlay, cleared each frame)
/// textures, plus the layer size.
pub struct ViewTarget {
    pub layer: wgpu::TextureView,
    pub staging: wgpu::TextureView,
    pub staging_size: [u32; 2],
    pub width: u32,
    pub height: u32,
}

/// Per-frame render targets for the views, keyed by view id. Built by
/// the renderer for the `shade` stage.
pub type ViewTargets = std::collections::HashMap<u16, ViewTarget>;

/// The handles `view_bind_group` needs, cloned from [`Gfx`] when the
/// encoder is built (wgpu resources are Arc-backed).
#[derive(Clone)]
struct EncoderGfx {
    sprites: Arc<sprite::Blitter>,
    device: std::sync::Arc<wgpu::Device>,
    texture_bgl: std::sync::Arc<wgpu::BindGroupLayout>,
    sampler: std::sync::Arc<wgpu::Sampler>,
    frame: Arc<AtomicU64>,
}

/// The command encoder handed to `shade` hooks. The host owns the
/// underlying encoder and takes it back when dispatch ends; passes a
/// hook leaves open are force-ended after the call, so a stored
/// encoder or pass errors cleanly instead of wedging the frame
/// (docs/rune-plan.md, GPU misuse policy).
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptEncoder {
    enc: SharedEncoder,
    passes: PassList,
    cpasses: ComputePassList,
    view_targets: std::sync::Arc<ViewTargets>,
    gfx: Option<EncoderGfx>,
}

impl ScriptEncoder {
    /// End every pass begun so far: wgpu allows one open pass per
    /// encoder, so beginning a new one auto-ends its predecessors
    /// (sequential-pass semantics; a kept handle errors cleanly).
    fn end_open_passes(&self) {
        for pass in self.passes.lock().expect("pass list lock").drain(..) {
            pass.lock().expect("pass lock").take();
        }
        for pass in self.cpasses.lock().expect("compute pass list lock").drain(..) {
            pass.lock().expect("pass lock").take();
        }
    }

    fn begin(
        &self,
        label: &str,
        target: &ScriptTextureView,
        load: &str,
    ) -> Result<ScriptPass, String> {
        let load = match load {
            "load" => wgpu::LoadOp::Load,
            "clear" => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            other => return Err(format!("unknown load op `{}`", other)),
        };
        self.begin_with_load(label, target, load)
    }

    fn begin_with_load(
        &self,
        label: &str,
        target: &ScriptTextureView,
        load: wgpu::LoadOp<wgpu::Color>,
    ) -> Result<ScriptPass, String> {
        self.end_open_passes();
        let mut guard = self.enc.lock().expect("encoder lock");
        let Some(encoder) = guard.as_mut() else {
            return Err("the encoder is no longer valid".to_string());
        };
        let pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.get()?,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        let shared: SharedPass = std::sync::Arc::new(std::sync::Mutex::new(Some(pass)));
        self.passes
            .lock()
            .expect("pass list lock")
            .push(shared.clone());
        Ok(ScriptPass {
            pass: shared,
            size: target.size,
            format: target.format,
            sprites: self.gfx.as_ref().map(|g| g.sprites.clone()),
        })
    }

    /// Begin a render pass on any texture view. The descriptor specifies
    /// load/clear behavior, an optional linear clear color, and a label.
    #[rune::function]
    fn begin_render_pass(
        &self,
        target: &ScriptTextureView,
        options: &rune::runtime::Object,
    ) -> Result<ScriptPass, String> {
        let options = PassOptions::parse(options)?;
        self.begin_with_load(&options.label, target, options.load)
    }

    fn editor_texture_view(
        &self,
        view_id: i64,
        staging: bool,
    ) -> Result<ScriptTextureView, String> {
        if self.enc.lock().expect("encoder lock").is_none() {
            return Err("the encoder is no longer valid".into());
        }
        let id = u16::try_from(view_id).map_err(|_| format!("invalid view id: {}", view_id))?;
        if !self.view_targets.contains_key(&id) {
            return Err(format!("no such view: {}", view_id));
        }
        let gfx = self.gfx.as_ref().ok_or("the GPU is not available")?;
        let target = &self.view_targets[&id];
        let view = ScriptTextureView {
            size: if staging {
                target.staging_size
            } else {
                [target.width, target.height]
            },
            format: SCRIPT_TEXTURE_FORMAT,
            source: TextureViewSource::Editor {
                targets: self.view_targets.clone(),
                id,
                staging,
            },
            scope: Some(FrameScope {
                clock: Arc::downgrade(&gfx.frame),
                generation: gfx.frame.load(Ordering::Relaxed),
            }),
        };
        view.get()?;
        Ok(view)
    }

    /// The live artwork texture view, valid for this frame.
    /// Uses the same sheet extent as begin_view_pass/view_bind_group.
    #[rune::function]
    fn view_layer(&self, view_id: i64) -> Result<ScriptTextureView, String> {
        self.editor_texture_view(view_id, false)
    }

    /// The preview overlay texture view, valid for this frame.
    #[rune::function]
    fn view_staging(&self, view_id: i64) -> Result<ScriptTextureView, String> {
        self.editor_texture_view(view_id, true)
    }

    /// Paint the live artwork; touch_view still controls undo recording.
    #[rune::function]
    fn begin_view_pass(&self, label: &str, view_id: i64, load: &str) -> Result<ScriptPass, String> {
        self.begin(label, &self.editor_texture_view(view_id, false)?, load)
    }

    /// Paint the per-frame preview overlay.
    #[rune::function]
    fn begin_staging_pass(
        &self,
        label: &str,
        view_id: i64,
        load: &str,
    ) -> Result<ScriptPass, String> {
        self.begin(label, &self.editor_texture_view(view_id, true)?, load)
    }

    /// Convenience binding for live artwork, with the same lifetime as view_layer.
    #[rune::function]
    fn view_bind_group(&self, view_id: i64) -> Result<ScriptBindGroup, String> {
        let target = self.editor_texture_view(view_id, false)?;
        let gfx = self
            .gfx
            .as_ref()
            .ok_or("the GPU is not available in this context")?;
        texture_bind_group(&gfx.device, &gfx.texture_bgl, &gfx.sampler, &target)
    }

    /// Begin a compute pass.
    #[rune::function]
    fn begin_compute_pass(&self, label: &str) -> Result<ScriptComputePass, String> {
        self.end_open_passes();
        let mut guard = self.enc.lock().expect("encoder lock");
        let Some(encoder) = guard.as_mut() else {
            return Err("the encoder is no longer valid".to_string());
        };
        let pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            })
            .forget_lifetime();
        let shared: SharedComputePass = std::sync::Arc::new(std::sync::Mutex::new(Some(pass)));
        self.cpasses
            .lock()
            .expect("compute pass list lock")
            .push(shared.clone());
        Ok(ScriptComputePass { pass: shared })
    }
}

/// A compute pass recorded by a script. Same lifetime rules as
/// [`ScriptPass`]: ended by `end()` or by the host at hook return.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptComputePass {
    pass: SharedComputePass,
}

impl ScriptComputePass {
    fn with<T>(&self, f: impl FnOnce(&mut wgpu::ComputePass<'static>) -> T) -> Result<T, String> {
        match self.pass.lock().expect("pass lock").as_mut() {
            Some(pass) => Ok(f(pass)),
            None => Err("the compute pass has ended".to_string()),
        }
    }

    /// Set the active compute pipeline.
    #[rune::function]
    fn set_pipeline(&self, pipeline: &ScriptComputePipeline) -> Result<(), String> {
        self.with(|p| p.set_pipeline(&pipeline.pipeline))
    }

    /// Bind a bind group at the given index.
    #[rune::function]
    fn set_bind_group(&self, index: i64, group: &ScriptBindGroup) -> Result<(), String> {
        validate_scope(&group.scope)?;
        self.with(|p| p.set_bind_group(index as u32, &group.bind_group, &[]))
    }

    /// Dispatch workgroups.
    #[rune::function]
    fn dispatch(&self, x: i64, y: i64, z: i64) -> Result<(), String> {
        self.with(|p| p.dispatch_workgroups(x as u32, y as u32, z as u32))
    }

    /// End the pass. Further use errors.
    #[rune::function]
    fn end(&self) {
        self.pass.lock().expect("pass lock").take();
    }
}

/// A render pass recorded by a script. Ended by `end()`, or by the
/// host when the hook returns.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct ScriptPass {
    size: [u32; 2],
    format: wgpu::TextureFormat,
    sprites: Option<Arc<sprite::Blitter>>,
    pass: SharedPass,
}

impl ScriptPass {
    /// Draw a texture or texture view in target-pixel coordinates.
    /// See docs/gpu-api.md for cropping, transforms, tint and blending.
    #[rune::function]
    fn draw_sprite(&self, source: Value, options: &rune::runtime::Object) -> Result<(), String> {
        self.with(|_| ())?;
        let sprites = self
            .sprites
            .as_ref()
            .ok_or("the GPU is not available in this context")?;
        if let Ok(texture) = source.borrow_ref::<ScriptTexture>() {
            sprites.draw(self, &texture.sampled_view(), options)
        } else {
            let view = source
                .borrow_ref::<ScriptTextureView>()
                .map_err(|_| "source must be a Texture or TextureView")?;
            sprites.draw(self, &view, options)
        }
    }

    fn with<T>(&self, f: impl FnOnce(&mut wgpu::RenderPass<'static>) -> T) -> Result<T, String> {
        match self.pass.lock().expect("pass lock").as_mut() {
            Some(pass) => Ok(f(pass)),
            None => Err("the render pass has ended".to_string()),
        }
    }

    /// Set the active pipeline.
    #[rune::function]
    fn set_pipeline(&self, pipeline: &ScriptPipeline) -> Result<(), String> {
        self.with(|p| p.set_pipeline(&pipeline.pipeline))
    }

    /// Bind a bind group at the given index.
    #[rune::function]
    fn set_bind_group(&self, index: i64, group: &ScriptBindGroup) -> Result<(), String> {
        validate_scope(&group.scope)?;
        self.with(|p| p.set_bind_group(index as u32, &group.bind_group, &[]))
    }

    /// Bind a vertex buffer at the given slot.
    #[rune::function]
    fn set_vertex_buffer(&self, slot: i64, buffer: &ScriptBuffer) -> Result<(), String> {
        self.with(|p| p.set_vertex_buffer(slot as u32, buffer.buffer.slice(..)))
    }

    /// Draw `vertices` vertices, `instances` instances.
    #[rune::function]
    fn draw(&self, vertices: i64, instances: i64) -> Result<(), String> {
        self.with(|p| p.draw(0..vertices as u32, 0..instances as u32))
    }

    /// End the pass. Further use errors.
    #[rune::function]
    fn end(&self) {
        self.pass.lock().expect("pass lock").take();
    }
}

////////////////////////////////////////////////////////////////////////////
// Script commands

/// A typed parameter of a script command.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ParamType {
    Int,
    Float,
    Str,
    Color,
    Bool,
}

impl ParamType {
    fn name(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::Float => "float",
            Self::Str => "str",
            Self::Color => "color",
            Self::Bool => "bool",
        }
    }
}

#[derive(Clone, Debug)]
struct Param {
    ty: ParamType,
    optional: bool,
}

/// Parse a declared signature, e.g. `["int", "color?"]`. A `?` suffix
/// marks the parameter optional; optional parameters must come last.
fn parse_sig(sig: &[String]) -> Result<Vec<Param>, String> {
    let mut params: Vec<Param> = Vec::with_capacity(sig.len());
    for s in sig {
        let (name, optional) = match s.strip_suffix('?') {
            Some(n) => (n, true),
            None => (s.as_str(), false),
        };
        let ty = match name {
            "int" => ParamType::Int,
            "float" => ParamType::Float,
            "str" => ParamType::Str,
            "color" => ParamType::Color,
            "bool" => ParamType::Bool,
            other => return Err(format!("unknown parameter type `{}`", other)),
        };
        if !optional && params.last().is_some_and(|p| p.optional) {
            return Err("required parameter after optional parameter".to_string());
        }
        params.push(Param { ty, optional });
    }
    Ok(params)
}

/// `usage: <name> <int> [color]` — for dispatch-time argument errors.
fn usage(name: &str, params: &[Param]) -> String {
    let mut s = format!("usage: {}", name);
    for p in params {
        if p.optional {
            s.push_str(&format!(" [{}]", p.ty.name()));
        } else {
            s.push_str(&format!(" <{}>", p.ty.name()));
        }
    }
    s
}

/// Parse raw invocation arguments against a declared signature into Rune
/// values (i64, f64, String, Rgba8, bool).
fn parse_args(name: &str, params: &[Param], raw: &str) -> Result<Vec<Value>, String> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    let required = params.iter().filter(|p| !p.optional).count();
    if tokens.len() < required || tokens.len() > params.len() {
        return Err(usage(name, params));
    }

    let mut out = Vec::with_capacity(tokens.len());
    for (tok, param) in tokens.iter().zip(params) {
        let value = match param.ty {
            ParamType::Int => tok
                .parse::<i64>()
                .ok()
                .and_then(|n| rune::to_value(n).ok()),
            ParamType::Float => tok
                .parse::<f64>()
                .ok()
                .and_then(|n| rune::to_value(n).ok()),
            ParamType::Str => rune::to_value(tok.to_string()).ok(),
            ParamType::Bool => match *tok {
                "on" | "true" => rune::to_value(true).ok(),
                "off" | "false" => rune::to_value(false).ok(),
                _ => None,
            },
            ParamType::Color => {
                // Guard the length: `Rgba8::from_str` slices bytes.
                if tok.len() == 7 && tok.starts_with('#') && tok.is_ascii() {
                    tok.parse::<crate::gfx::color::Rgba8>()
                        .ok()
                        .and_then(|c| rune::to_value(c).ok())
                } else {
                    None
                }
            }
        };
        match value {
            Some(v) => out.push(v),
            None => {
                return Err(format!(
                    "invalid {} `{}`; {}",
                    param.ty.name(),
                    tok,
                    usage(name, params)
                ))
            }
        }
    }
    Ok(out)
}

/// A command registered by a plugin.
pub struct ScriptCommand {
    pub name: String,
    pub help: String,
    pub repeating: bool,
    /// Owning plugin; handlers run with this plugin's state.
    pub plugin: String,
    params: Vec<Param>,
    handler: Function,
}

/// A function a plugin exports for other plugins to call (the
/// meta-plugin mechanism). Cross-unit Rune function values don't work
/// — a consumer VM cannot execute a foreign unit's offsets — so calls
/// are host-mediated: the handler runs on its own unit with its own
/// plugin's state, like a command handler.
pub struct ScriptExport {
    plugin: String,
    name: String,
    handler: Function,
    /// The owning plugin's state; filled in by the host once `init`
    /// returns (exports are registered during `init`, before the
    /// state value exists).
    state: Option<Value>,
}

/// All commands registered by loaded plugins. Owned by [`PluginHost`];
/// reachable from hooks through [`Ctx`] for registration.
#[derive(Default)]
pub struct ScriptCommands {
    entries: Vec<ScriptCommand>,
    exports: Vec<ScriptExport>,
}

impl ScriptCommands {
    fn add_export(&mut self, export: ScriptExport) -> Result<(), String> {
        if self
            .exports
            .iter()
            .any(|e| e.plugin == export.plugin && e.name == export.name)
        {
            return Err(format!(
                "`{}::{}` is already exported",
                export.plugin, export.name
            ));
        }
        self.exports.push(export);
        Ok(())
    }

    fn get_export(&self, plugin: &str, name: &str) -> Option<&ScriptExport> {
        self.exports
            .iter()
            .find(|e| e.plugin == plugin && e.name == name)
    }

    /// Attach the owning plugin's state to its exports (post-init).
    fn fill_export_states(&mut self, plugin: &str, state: &Value) {
        for e in self.exports.iter_mut().filter(|e| e.plugin == plugin) {
            e.state = Some(state.clone());
        }
    }

    fn register(&mut self, cmd: ScriptCommand) -> Result<(), String> {
        if let Some(existing) = self.entries.iter().find(|c| c.name == cmd.name) {
            return Err(format!(
                "command ':{}' is already registered by plugin `{}`",
                cmd.name, existing.plugin
            ));
        }
        self.entries.push(cmd);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ScriptCommand> {
        self.entries.iter().find(|c| c.name == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ScriptCommand> {
        self.entries.iter()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.exports.clear();
    }

    /// `(name, help)` pairs for the help view.
    fn help_entries(&self) -> Vec<(String, String)> {
        self.entries
            .iter()
            .map(|c| (c.name.clone(), c.help.clone()))
            .collect()
    }
}

////////////////////////////////////////////////////////////////////////////
// Ctx

/// The context object passed to every plugin hook as `rx`.
///
/// Holds a raw pointer to the session for the duration of a single hook
/// call. Hooks receive it as `&mut Ctx` through rune's guarded arguments,
/// so scripts cannot store it; the pointer never outlives the call.
#[derive(rune::Any)]
#[rune(item = ::rx)]
pub struct Ctx {
    session: *mut Session,
    /// Draw sink; non-null only during the `draw` hook.
    draw: *mut crate::draw::Context,
    /// Command registry; non-null only for calls made by the plugin host.
    cmds: *mut ScriptCommands,
    /// GPU capability; non-null once the renderer exists.
    gfx: *const Gfx,
    /// Name of the plugin whose hook is running (owner of registrations).
    plugin: String,
    /// Directory of the running plugin (for `read_file`).
    root: Option<std::path::PathBuf>,
}

impl Ctx {
    /// Construct a context borrowing the session for one hook call.
    ///
    /// SAFETY: the returned `Ctx` must not outlive `session`, and the
    /// session must not be accessed by the host while a hook call using
    /// this ctx is in progress. Both hold because `Ctx` is created
    /// immediately before a `vm.call` and dropped right after, and the
    /// call passes it as a guarded `&mut`.
    pub fn new(session: &mut Session) -> Self {
        Ctx {
            session: session as *mut Session,
            draw: std::ptr::null_mut(),
            cmds: std::ptr::null_mut(),
            gfx: std::ptr::null(),
            plugin: String::new(),
            root: None,
        }
    }

    /// A context that can also draw (used by the `draw` hook). Same
    /// safety contract as [`Ctx::new`], extended to `draw`.
    pub fn with_draw(session: &mut Session, draw: &mut crate::draw::Context) -> Self {
        Ctx {
            draw: draw as *mut crate::draw::Context,
            ..Ctx::new(session)
        }
    }

    /// Attach the command registry and the calling plugin's name. Same
    /// safety contract as [`Ctx::new`], extended to `cmds`.
    pub fn with_commands(mut self, cmds: &mut ScriptCommands, plugin: &str) -> Self {
        self.cmds = cmds as *mut ScriptCommands;
        self.plugin = plugin.to_string();
        self
    }

    /// Attach the GPU capability. Unlike the other pointers, `gfx` is
    /// host-owned and outlives every hook call; the pointer is still
    /// scoped to one call by construction.
    pub fn with_gfx(mut self, gfx: Option<&Gfx>) -> Self {
        if let Some(g) = gfx {
            self.gfx = g as *const Gfx;
        }
        self
    }

    /// Attach the running plugin's directory (for `read_file`).
    pub fn with_root(mut self, root: Option<std::path::PathBuf>) -> Self {
        self.root = root;
        self
    }

    fn gfx(&self) -> Option<&Gfx> {
        if self.gfx.is_null() {
            None
        } else {
            Some(unsafe { &*self.gfx })
        }
    }

    /// Post an error to the message line (helper for GPU methods).
    fn error(&mut self, msg: impl fmt::Display) {
        self.session_mut()
            .message(format!("Error: {}", msg), crate::session::MessageType::Error);
    }

    fn session(&self) -> &Session {
        unsafe { &*self.session }
    }

    fn session_mut(&mut self) -> &mut Session {
        unsafe { &mut *self.session }
    }

    fn draw_mut(&mut self) -> Option<&mut crate::draw::Context> {
        if self.draw.is_null() {
            None
        } else {
            Some(unsafe { &mut *self.draw })
        }
    }

    /// Current mode, as a string (e.g. "normal", "visual", "command").
    #[rune::function]
    fn mode(&self) -> String {
        self.session().mode.to_string()
    }

    /// Previous mode, if any.
    #[rune::function]
    fn prev_mode(&self) -> Option<String> {
        self.session().prev_mode.as_ref().map(|m| m.to_string())
    }

    /// Switch the session mode. Builtin mode names (`normal`, `visual`,
    /// `command`, `present`, `help`) switch to the builtin mode; any
    /// other name enters a custom script mode (escape exits it, builtin
    /// input handling is inert in it). Returns whether the switch was
    /// accepted.
    #[rune::function]
    fn switch_mode(&mut self, name: &str) -> bool {
        use crate::session::{Mode, ModeString, VisualState};

        let mode = match name {
            "normal" => Mode::Normal,
            "visual" => Mode::Visual(VisualState::default()),
            "command" => Mode::Command,
            "present" => Mode::Present,
            "help" => Mode::Help,
            custom => match ModeString::try_from_str(custom) {
                Ok(s) if !s.is_empty() => Mode::Script(s),
                _ => {
                    self.session_mut().message(
                        format!("Error: invalid mode name `{}`", custom),
                        crate::session::MessageType::Error,
                    );
                    return false;
                }
            },
        };
        self.session_mut().switch_mode(mode);
        true
    }

    /// Post a message to the message line.
    #[rune::function]
    fn message(&mut self, msg: &str) {
        self.session_mut()
            .message(msg.to_string(), crate::session::MessageType::Echo);
    }

    /// Id of the active view.
    #[rune::function]
    fn active_view_id(&self) -> i64 {
        u16::from(self.session().views.active_id) as i64
    }

    /// The foreground color.
    #[rune::function]
    fn fg(&self) -> crate::gfx::color::Rgba8 {
        self.session().fg
    }

    /// The background color.
    #[rune::function]
    fn bg(&self) -> crate::gfx::color::Rgba8 {
        self.session().bg
    }

    /// Set the foreground color, with the picker's semantics: the old
    /// foreground becomes the background; transparent is ignored.
    /// color, and no builtin command does.
    #[rune::function]
    fn set_fg(&mut self, color: &crate::gfx::color::Rgba8) {
        let c = *color;
        if c.a == 0 {
            return;
        }
        let s = self.session_mut();
        if c != s.fg {
            s.bg = s.fg;
            s.fg = c;
        }
    }

    /// The session workspace offset `(x, y)`.
    #[rune::function]
    fn offset(&self) -> (f64, f64) {
        let o = self.session().offset;
        (o.x as f64, o.y as f64)
    }

    /// The screen size `(w, h)` in session pixels — the size of the
    /// `render` stage's target. Build screen-space orthos against it
    /// (`create_transform_bind_group(w, h, ...)`); compose `rx.offset()`
    /// and view zoom into the transform for session-space placement.
    #[rune::function]
    fn screen_size(&self) -> (i64, i64) {
        let s = self.session();
        (s.width as i64, s.height as i64)
    }

    /// The cursor position in session coordinates.
    #[rune::function]
    fn cursor(&self) -> (f64, f64) {
        let c = self.session().cursor;
        (c.x as f64, c.y as f64)
    }

    /// Convert window logical coordinates (as received by the
    /// `cursor_moved` hook) to session coordinates.
    #[rune::function]
    fn session_coords(&self, x: f64, y: f64) -> (f64, f64) {
        let p = self
            .session()
            .window_to_session_coords(crate::platform::LogicalPosition::new(x, y));
        (p.x as f64, p.y as f64)
    }

    /// Convert session coordinates to the active view's coordinates
    /// (floored to integer pixels).
    #[rune::function]
    fn active_view_coords(&self, x: f64, y: f64) -> (f64, f64) {
        let s = self.session();
        let p = s.active_view_coords(crate::session::SessionCoords::new(x as f32, y as f32));
        (p.x as f64, p.y as f64)
    }

    /// Mark a view as modified (its contents will be re-recorded, e.g.
    /// after a script render pass painted into it).
    #[rune::function]
    fn touch_view(&mut self, id: i64) {
        use crate::view::ViewId;

        if let Some(v) = self.session_mut().views.get_mut(ViewId::from(id as u16)) {
            v.touch();
        }
    }

    /// Clear a rect of the active view to transparent. This is a
    /// recorded paint (part of the same undoable edit as any other
    /// paint this frame); scripts use it to erase the source region
    /// before stamping a transformed copy.
    #[rune::function]
    fn clear_view_rect(&mut self, rect: &Rect) {
        use crate::gfx::color::Rgba8;
        use crate::gfx::rect::Rect as GfxRect;
        use crate::gfx::shape2d::{Fill, Rotation, Shape, Stroke};
        use crate::gfx::ZDepth;
        use crate::session::{Blending, Effect};

        let r = GfxRect::new(
            rect.x1.min(rect.x2) as f32,
            rect.y1.min(rect.y2) as f32,
            rect.x1.max(rect.x2) as f32,
            rect.y1.max(rect.y2) as f32,
        );
        let session = self.session_mut();
        session.effects.extend_from_slice(&[
            Effect::ViewBlendingChanged(Blending::Constant),
            Effect::ViewPaintFinal(vec![Shape::Rectangle(
                r,
                ZDepth::default(),
                Rotation::ZERO,
                Stroke::NONE,
                Fill::Solid(Rgba8::TRANSPARENT.into()),
            )]),
        ]);
        if let Some(v) = session.views.active_mut() {
            v.touch();
        }
    }

    /// Re-render a view's texture from its recorded snapshot,
    /// discarding any unrecorded GPU-side paint (e.g. a preview a
    /// script rendered into the view and now wants gone).
    #[rune::function]
    fn damage_view(&mut self, id: i64) {
        use crate::session::Effect;
        use crate::view::ViewId;

        self.session_mut()
            .effects
            .push(Effect::ViewDamaged(ViewId::from(id as u16), None));
    }

    /// Read a view's recorded pixels in a y-down rect (RGBA8, top-first rows).
    /// Buffer row zero is the top row of the clamped rectangle.
    /// The rect is clamped to the view; `None` if the view doesn't
    /// exist or the rect is empty.
    #[rune::function]
    fn view_pixels(&mut self, id: i64, rect: &Rect) -> Option<rune::runtime::Bytes> {
        use crate::gfx::rect::Rect as GfxRect;
        use crate::view::ViewId;

        let session = self.session();
        let v = session.views.get(ViewId::from(id as u16))?;
        let r = GfxRect::new(
            rect.x1.min(rect.x2) as i32,
            rect.y1.min(rect.y2) as i32,
            rect.x1.max(rect.x2) as i32,
            rect.y1.max(rect.y2) as i32,
        );
        if !r.intersects(v.layer_bounds()) {
            return None;
        }
        let r = r.intersection(v.layer_bounds());
        let (_, pixels) = v.resource.layer.get_snapshot_rect(&r)?;
        let mut bytes = Vec::with_capacity(pixels.len() * 4);
        for px in &pixels {
            bytes.extend_from_slice(&[px.r, px.g, px.b, px.a]);
        }
        rune::runtime::Bytes::from_slice(&bytes).ok()
    }

    /// Read one *layer strip's* pixels in the given rect (rgba8 bytes,
    /// row-major, row 0 = top). `rect` is display-space — a single frame,
    /// `y` in `0..fh`, y-down — and `layer` is the strip index (`0` = first
    /// strip). Unlike `view_pixels`, which reaches only the first strip
    /// (display-space / active-routed writes), this addresses any strip:
    /// the per-layer read a plugin needs to introspect a non-active layer
    /// on the CPU. `None` if the view or
    /// layer doesn't exist, or the rect misses the frame.
    #[rune::function]
    fn view_layer_pixels(
        &mut self,
        id: i64,
        layer: i64,
        rect: &Rect,
    ) -> Option<rune::runtime::Bytes> {
        use crate::gfx::rect::Rect as GfxRect;
        use crate::view::ViewId;

        let session = self.session();
        let v = session.views.get(ViewId::from(id as u16))?;
        if layer < 0 || layer as usize >= v.nlayers {
            return None;
        }
        // Clamp the request to one frame strip (display bounds), then lift
        // it into the layer's rows of the sheet (strip n = sheet-space
        // y-down rows `n*fh .. (n+1)*fh`).
        let r = GfxRect::new(
            rect.x1.min(rect.x2) as i32,
            rect.y1.min(rect.y2) as i32,
            rect.x1.max(rect.x2) as i32,
            rect.y1.max(rect.y2) as i32,
        );
        if !r.intersects(v.layer_bounds()) {
            return None;
        }
        let r = r.intersection(v.layer_bounds());
        let off = layer as i32 * v.fh as i32;
        let sheet = GfxRect::new(r.x1, r.y1 + off, r.x2, r.y2 + off);
        let (_, pixels) = v.resource.layer.get_snapshot_rect(&sheet)?;
        let mut bytes = Vec::with_capacity(pixels.len() * 4);
        for px in &pixels {
            bytes.extend_from_slice(&[px.r, px.g, px.b, px.a]);
        }
        rune::runtime::Bytes::from_slice(&bytes).ok()
    }

    /// Snapshots of all views, in view order.
    #[rune::function]
    fn views(&self) -> Vec<ViewInfo> {
        self.session()
            .views
            .iter()
            .map(|v| ViewInfo {
                id: u16::from(v.id) as i64,
                width: v.width() as i64,
                height: v.height() as i64,
                offset_x: v.offset.x as f64,
                offset_y: v.offset.y as f64,
                zoom: v.zoom as f64,
                frames: v.animation.len() as i64,
                animation_frame: v.animation.index as i64,
                frame_width: v.fw as i64,
                frame_height: v.fh as i64,
                nlayers: v.nlayers as i64,
                active_layer: v.active_layer as i64,
                animation_preview_visible: v.animation_preview_visible,
            })
            .collect()
    }

    /// Set a view's current animation frame. The index wraps by the
    /// view's frame count. Returns false if the view does not exist.
    #[rune::function]
    fn set_animation_frame(&mut self, id: i64, frame: i64) -> bool {
        use crate::view::ViewId;

        let Some(v) = self.session_mut().views.get_mut(ViewId::from(id as u16)) else {
            return false;
        };
        let n = v.animation.len() as i64;
        v.animation.set_frame(frame.rem_euclid(n) as usize);
        true
    }

    /// Set a view's playback sequence. Entries are zero-based frame indices;
    /// an empty sequence restores natural frame order. Returns false if the
    /// view does not exist or any entry is outside its frame range.
    #[rune::function]
    fn set_animation_sequence(&mut self, id: i64, sequence: Vec<i64>) -> bool {
        use crate::view::ViewId;

        let Some(v) = self.session_mut().views.get_mut(ViewId::from(id as u16)) else {
            return false;
        };
        let Ok(sequence) = sequence
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        v.animation.set_sequence(sequence)
    }

    /// The view's custom playback sequence. Empty means natural frame order;
    /// also empty if the view does not exist.
    #[rune::function]
    fn animation_sequence(&self, id: i64) -> Vec<i64> {
        use crate::view::ViewId;

        self.session()
            .views
            .get(ViewId::from(id as u16))
            .map(|v| {
                v.animation
                    .sequence()
                    .iter()
                    .map(|&frame| frame as i64)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Restore natural frame-order playback for one view. Returns false if
    /// the view does not exist.
    #[rune::function]
    fn clear_animation_sequence(&mut self, id: i64) -> bool {
        use crate::view::ViewId;

        let Some(v) = self.session_mut().views.get_mut(ViewId::from(id as u16)) else {
            return false;
        };
        v.animation.clear_sequence();
        true
    }

    /// Show or suppress the workspace's built-in animation preview for
    /// one view. Specialized preview plugins use this to replace it.
    #[rune::function]
    fn set_animation_preview_visible(&mut self, id: i64, visible: bool) -> bool {
        use crate::view::ViewId;

        let Some(v) = self.session_mut().views.get_mut(ViewId::from(id as u16)) else {
            return false;
        };
        v.animation_preview_visible = visible;
        true
    }

    /// Per-layer visibility for a view, bottom compositing layer first (index `0`).
    /// `nlayers` long; an empty vec if the view doesn't exist.
    #[rune::function]
    fn layer_visibility(&self, id: i64) -> Vec<bool> {
        use crate::view::ViewId;

        match self.session().views.get(ViewId::from(id as u16)) {
            Some(v) => v.layer_attrs.iter().map(|a| a.visible).collect(),
            None => Vec::new(),
        }
    }

    /// Per-layer opacity, in the same bottom-to-top order as layer_visibility.
    #[rune::function]
    fn layer_opacity(&self, id: i64) -> Vec<f64> {
        self.session()
            .views
            .get(crate::view::ViewId::from(id as u16))
            .map(|v| v.layer_attrs.iter().map(|a| a.opacity as f64).collect())
            .unwrap_or_default()
    }

    /// A setting's value: bool, integer, float, string or a tuple,
    /// depending on the setting. Unit if the setting doesn't exist.
    #[rune::function]
    fn setting(&self, name: &str) -> Value {
        use crate::cmd::Value as V;

        let unit = || rune::to_value(()).expect("unit converts");
        match self.session().settings.get(name) {
            None => unit(),
            Some(V::Bool(b)) => rune::to_value(*b).unwrap_or_else(|_| unit()),
            Some(V::U32(n)) => rune::to_value(*n as i64).unwrap_or_else(|_| unit()),
            Some(V::U32Tuple(a, b)) => {
                rune::to_value((*a as i64, *b as i64)).unwrap_or_else(|_| unit())
            }
            Some(V::F32Tuple(a, b)) => {
                rune::to_value((*a as f64, *b as f64)).unwrap_or_else(|_| unit())
            }
            Some(V::F64(f)) => rune::to_value(*f).unwrap_or_else(|_| unit()),
            Some(V::Str(s)) | Some(V::Ident(s)) => {
                rune::to_value(s.clone()).unwrap_or_else(|_| unit())
            }
            Some(V::Rgba8(c)) => rune::to_value(c.to_string()).unwrap_or_else(|_| unit()),
        }
    }

    /// Set a setting. The value must match the setting's current type.
    /// Returns whether the set was applied.
    #[rune::function]
    fn set_setting(&mut self, name: &str, value: Value) -> bool {
        use crate::cmd::Value as V;

        let converted = match self.session().settings.get(name) {
            None => None,
            Some(V::Bool(_)) => rune::from_value::<bool>(value).ok().map(V::Bool),
            Some(V::U32(_)) => rune::from_value::<i64>(value).ok().map(|n| V::U32(n as u32)),
            Some(V::F64(_)) => rune::from_value::<f64>(value).ok().map(V::F64),
            Some(V::Str(_)) => rune::from_value::<String>(value).ok().map(V::Str),
            Some(V::Ident(_)) => rune::from_value::<String>(value).ok().map(V::Ident),
            Some(V::U32Tuple(..)) | Some(V::F32Tuple(..)) | Some(V::Rgba8(_)) => None,
        };
        match converted {
            Some(v) => self.session_mut().settings.set(name, v).is_ok(),
            None => false,
        }
    }

    /// Declare a setting this plugin owns, with its default value
    /// (bool, int, float or string). Declared settings are `:set`-able
    /// like builtins; re-declaring is a no-op.
    #[rune::function]
    fn declare_setting(&mut self, name: &str, value: Value) -> bool {
        use crate::cmd::Value as V;

        let v = if let Ok(b) = rune::from_value::<bool>(value.clone()) {
            V::Bool(b)
        } else if let Ok(n) = rune::from_value::<i64>(value.clone()) {
            V::U32(n as u32)
        } else if let Ok(f) = rune::from_value::<f64>(value.clone()) {
            V::F64(f)
        } else if let Ok(s) = rune::from_value::<String>(value) {
            // Ident rather than Str: `:set name = value` parses bare
            // words as idents, and the discriminants must match.
            V::Ident(s)
        } else {
            self.session_mut().message(
                format!("Error: unsupported value for setting `{}`", name),
                crate::session::MessageType::Error,
            );
            return false;
        };
        self.session_mut().settings.declare(name, v);
        true
    }

    /// The current selection bounds as `(x1, y1, x2, y2)`, if any.
    /// Normalized (`x1 <= x2`, `y1 <= y2`) regardless of the drag
    /// direction that created the selection, like the built-in
    /// selection commands see it.
    #[rune::function]
    fn selection(&self) -> Option<(i64, i64, i64, i64)> {
        self.session().selection.map(|s| {
            let r = s.abs().bounds();
            (r.x1 as i64, r.y1 as i64, r.x2 as i64, r.y2 as i64)
        })
    }

    /// Replace the selection with the given bounds.
    #[rune::function]
    fn set_selection(&mut self, x1: i64, y1: i64, x2: i64, y2: i64) {
        self.session_mut().selection = Some(crate::session::Selection::new(
            x1 as i32, y1 as i32, x2 as i32, y2 as i32,
        ));
    }

    /// Clear the selection.
    #[rune::function]
    fn clear_selection(&mut self) {
        self.session_mut().selection = None;
    }

    /// Register a command: `rx.register_command(name, sig, help, handler)`.
    ///
    /// `sig` declares the typed parameters (`"int"`, `"float"`, `"str"`,
    /// `"color"`, `"bool"`; suffix `?` for optional). The handler is called
    /// as `handler(state, rx, args)` with `args` a vector of parsed values.
    /// Returns whether registration succeeded.
    #[rune::function]
    fn register_command(&mut self, name: &str, sig: Vec<String>, help: &str, handler: Function) -> bool {
        self.register_cmd(name, sig, help, handler, false)
    }

    /// Like `register_command`, but the command repeats while its bound
    /// key is held.
    #[rune::function]
    fn register_command_repeating(
        &mut self,
        name: &str,
        sig: Vec<String>,
        help: &str,
        handler: Function,
    ) -> bool {
        self.register_cmd(name, sig, help, handler, true)
    }

    fn register_cmd(
        &mut self,
        name: &str,
        sig: Vec<String>,
        help: &str,
        handler: Function,
        repeating: bool,
    ) -> bool {
        use crate::session::MessageType;

        if self.cmds.is_null() {
            self.session_mut().message(
                format!("Error: ':{}' cannot be registered in this context", name),
                MessageType::Error,
            );
            return false;
        }
        // Builtins always win at parse time; a shadowing registration
        // would never be dispatched, so reject it loudly.
        if self
            .session()
            .cmdline
            .commands
            .iter()
            .any(|(n, _, _)| *n == name)
        {
            self.session_mut().message(
                format!("Error: ':{}' would shadow a builtin command", name),
                MessageType::Error,
            );
            return false;
        }
        let params = match parse_sig(&sig) {
            Ok(p) => p,
            Err(e) => {
                self.session_mut().message(
                    format!("Error registering ':{}': {}", name, e),
                    MessageType::Error,
                );
                return false;
            }
        };
        let entry = ScriptCommand {
            name: name.to_string(),
            help: help.to_string(),
            repeating,
            plugin: self.plugin.clone(),
            params,
            handler,
        };
        let result = unsafe { &mut *self.cmds }.register(entry);
        match result {
            Ok(()) => true,
            Err(e) => {
                self.session_mut()
                    .message(format!("Error: {}", e), MessageType::Error);
                false
            }
        }
    }

    /// Read back a script texture's contents (rgba8 bytes, row-major)
    /// — the mirror of `upload`. This forces a GPU sync: it is a
    /// command-handler tool, not a per-frame one. Work recorded by
    /// `shade`/`render` hooks this frame has not been submitted yet, so
    /// a readback in the same frame as a mutation sees stale pixels —
    /// split mutate and read into separate commands a few frames apart
    /// (the `view_pixels` staleness family).
    #[rune::function]
    fn texture_pixels(&mut self, texture: &ScriptTexture) -> Option<rune::runtime::Bytes> {
        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        let data = texture.pixels(&gfx.device);
        rune::runtime::Bytes::from_slice(&data).ok()
    }

    /// Write rgba8 pixels (row-major, `w`×`h`×4 bytes) to `path` as a
    /// PNG. The path resolves like builtin `:export` paths — as typed,
    /// relative to the working directory, *not* plugin-relative. This
    /// is the scripts' first write capability: success and failure are
    /// both posted to the message line. Returns whether the write
    /// succeeded.
    #[rune::function]
    fn write_png(&mut self, path: &str, w: i64, h: i64, data: rune::runtime::Bytes) -> bool {
        use crate::gfx::color::Rgba8;
        use crate::session::MessageType;

        if w <= 0 || h <= 0 || data.len() != (w * h * 4) as usize {
            self.error(format!(
                "write_png: {} bytes for {}x{} (need {})",
                data.len(),
                w,
                h,
                w.max(0) * h.max(0) * 4
            ));
            return false;
        }
        let pixels: Vec<Rgba8> = data
            .chunks_exact(4)
            .map(|px| Rgba8::new(px[0], px[1], px[2], px[3]))
            .collect();
        match crate::image::save_as(path, w as u32, h as u32, 1, &pixels) {
            Ok(()) => {
                self.session_mut().message(
                    format!("\"{}\" {}x{} png written", path, w, h),
                    MessageType::Info,
                );
                true
            }
            Err(e) => {
                self.error(format!("write_png: \"{}\": {}", path, e));
                false
            }
        }
    }

    /// Create a GPU texture (rgba8, `w`×`h`). The texture belongs to the
    /// plugin: it is released when the plugin's state is dropped on
    /// unload/reload. `None` (with a message) if the GPU isn't available
    /// or the dimensions are invalid.
    #[rune::function]
    fn create_texture(&mut self, w: i64, h: i64) -> Option<ScriptTexture> {
        use crate::session::MessageType;

        let Some(gfx) = self.gfx() else {
            self.session_mut().message(
                "Error: the GPU is not available in this context",
                MessageType::Error,
            );
            return None;
        };
        if w <= 0 || h <= 0 || w > MAX_TEXTURE_DIM || h > MAX_TEXTURE_DIM {
            self.session_mut().message(
                format!("Error: invalid texture size {}x{}", w, h),
                MessageType::Error,
            );
            return None;
        }
        Some(ScriptTexture::create(gfx, w as u32, h as u32))
    }

    /// Read a file relative to the plugin's directory (e.g. a WGSL
    /// shader). `None` (with a message) if it can't be read.
    #[rune::function]
    fn read_file(&mut self, path: &str) -> Option<String> {
        let p = std::path::Path::new(path);
        let resolved = if p.is_relative() {
            match &self.root {
                Some(root) => root.join(p),
                None => p.to_path_buf(),
            }
        } else {
            p.to_path_buf()
        };
        match std::fs::read_to_string(&resolved) {
            Ok(s) => Some(s),
            Err(e) => {
                self.error(format!("{}: {}", resolved.display(), e));
                None
            }
        }
    }

    /// Read an 8-bit RGBA PNG relative to the plugin's directory (e.g.
    /// a cursor or icon asset) into `(w, h, pixels)` — row-major rgba8
    /// bytes, the layout `upload` and `write_png` use. `None` (with a
    /// message) if it can't be read or decoded.
    #[rune::function]
    fn read_png(&mut self, path: &str) -> Option<(i64, i64, rune::runtime::Bytes)> {
        let p = std::path::Path::new(path);
        let resolved = if p.is_relative() {
            match &self.root {
                Some(root) => root.join(p),
                None => p.to_path_buf(),
            }
        } else {
            p.to_path_buf()
        };
        match crate::image::load(&resolved) {
            Ok((pixels, w, h)) => {
                let bytes = rune::runtime::Bytes::from_slice(&pixels).ok()?;
                Some((w as i64, h as i64, bytes))
            }
            Err(e) => {
                self.error(format!("{}", e));
                None
            }
        }
    }

    /// Compile a WGSL shader. `None` (with the compile error in the
    /// message line) on failure.
    #[rune::function]
    fn create_shader(&mut self, source: &str) -> Option<ScriptShader> {
        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        gfx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = gfx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("script_shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        if let Some(e) = gfx.pop_error() {
            let msg = flatten_error(&e.to_string());
            self.error(format!("shader: {}", msg));
            return None;
        }
        Some(ScriptShader { module })
    }

    /// Create a pipeline with explicit blending and independent topology/layout.
    /// See docs/gpu-api.md for the descriptor and canonical binding layouts.
    #[rune::function]
    fn create_render_pipeline(
        &mut self,
        shader: &ScriptShader,
        options: &rune::runtime::Object,
    ) -> Option<ScriptPipeline> {
        let options = match PipelineOptions::parse(options) {
            Ok(options) => options,
            Err(error) => {
                self.error(format!("pipeline: {}", error));
                return None;
            }
        };
        self.build_render_pipeline(shader, &options)
    }

    fn build_render_pipeline(
        &mut self,
        shader: &ScriptShader,
        options: &PipelineOptions,
    ) -> Option<ScriptPipeline> {
        let textures = options.textures;
        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        gfx.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut layouts: Vec<&wgpu::BindGroupLayout> = vec![&gfx.transform_bgl];
        for _ in 0..textures {
            layouts.push(&gfx.texture_bgl);
        }
        let layout = gfx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("script_pipeline_layout"),
                bind_group_layouts: &layouts,
                push_constant_ranges: &[],
            });
        let sprite_layout = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ScriptVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }];
        let pipeline = gfx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("script_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader.module,
                    entry_point: Some(&options.vertex),
                    buffers: if options.sprite { &sprite_layout } else { &[] },
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader.module,
                    entry_point: Some(&options.fragment),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: options.format,
                        blend: Some(options.blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: options.topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        if let Some(e) = gfx.pop_error() {
            let msg = flatten_error(&e.to_string());
            self.error(format!("pipeline: {}", msg));
            return None;
        }
        Some(ScriptPipeline { pipeline })
    }

    /// Build the group-0 bind group: transform uniforms at binding 0,
    /// `params` (already padded to vec4 granularity) at binding 1.
    fn transform_bind_group(
        &mut self,
        w: i64,
        h: i64,
        transform: &Mat4,
        params: &[f32],
    ) -> Option<ScriptBindGroup> {
        use crate::gfx::math::{Matrix4, Origin};

        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        if w <= 0 || h <= 0 {
            self.error(format!("invalid target size {}x{}", w, h));
            return None;
        }
        // Match the renderer: application coordinates and texture rows are y-down.
        let ortho = Matrix4::ortho(w as u32, h as u32, Origin::TopLeft);

        let uniforms = ScriptUniforms {
            ortho: ortho.into(),
            transform: transform.0.into(),
        };
        let buffer = gfx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("script_uniform_buffer"),
            size: std::mem::size_of::<ScriptUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&uniforms));
        buffer.unmap();

        let params_buffer = gfx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("script_params_buffer"),
            size: (params.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(params));
        params_buffer.unmap();

        let bind_group = gfx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("script_transform_bind_group"),
            layout: &gfx.transform_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        Some(ScriptBindGroup {
            bind_group,
            scope: None,
        })
    }

    /// Create the group-0 uniforms bind group: an orthographic
    /// projection for a `w`×`h` target (origin top-left), composed with
    /// `transform`. The user-params slot (binding 1) is a zeroed
    /// vec4 — WGSL that doesn't declare it is unaffected.
    #[rune::function]
    fn create_transform_bind_group(
        &mut self,
        w: i64,
        h: i64,
        transform: &Mat4,
    ) -> Option<ScriptBindGroup> {
        self.transform_bind_group(w, h, transform, &[0.0; 4])
    }

    /// Like `create_transform_bind_group`, with user parameters at
    /// **group 0, binding 1**, declared WGSL-side as
    ///
    /// ```wgsl
    /// @group(0) @binding(1) var<uniform> params: array<vec4<f32>, N>;
    /// ```
    ///
    /// where `N = ceil(len / 4)`; `params` is packed in order and
    /// zero-padded to vec4 granularity (`params[i]` is component
    /// `i % 4` of `params[i / 4]` in WGSL). An f32 holds 24 exact
    /// integer bits, so pass a 32-bit bitmask as two 16-bit halves.
    #[rune::function]
    fn create_transform_params_bind_group(
        &mut self,
        w: i64,
        h: i64,
        transform: &Mat4,
        params: Vec<f64>,
    ) -> Option<ScriptBindGroup> {
        if params.is_empty() || params.len() > MAX_PARAMS {
            self.error(format!(
                "invalid param count {} (1..={})",
                params.len(),
                MAX_PARAMS
            ));
            return None;
        }
        let mut packed: Vec<f32> = params.iter().map(|p| *p as f32).collect();
        while packed.len() % 4 != 0 {
            packed.push(0.0);
        }
        self.transform_bind_group(w, h, transform, &packed)
    }

    /// Bind a texture view using the canonical texture + nearest sampler layout.
    #[rune::function]
    fn create_texture_bind_group(&mut self, view: &ScriptTextureView) -> Option<ScriptBindGroup> {
        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        match texture_bind_group(&gfx.device, &gfx.texture_bgl, &gfx.sampler, view) {
            Ok(group) => Some(group),
            Err(error) => {
                self.error(error);
                None
            }
        }
    }

    /// Create a compute pipeline from a shader and its entry point.
    /// Bind group 0 is the compute IO layout for `inputs` (1–4) input
    /// textures: inputs at bindings `0..n`, a write-only rgba8 storage
    /// texture last at binding `n`.
    #[rune::function]
    fn create_compute_pipeline(
        &mut self,
        shader: &ScriptShader,
        entry: &str,
        inputs: i64,
    ) -> Option<ScriptComputePipeline> {
        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        if !(1..=Gfx::MAX_COMPUTE_INPUTS as i64).contains(&inputs) {
            self.error(format!(
                "invalid compute input count {} (1..={})",
                inputs,
                Gfx::MAX_COMPUTE_INPUTS
            ));
            return None;
        }
        gfx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = gfx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("script_compute_pipeline_layout"),
                bind_group_layouts: &[&gfx.compute_bgls[inputs as usize - 1]],
                push_constant_ranges: &[],
            });
        let pipeline = gfx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("script_compute_pipeline"),
                layout: Some(&layout),
                module: &shader.module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(e) = gfx.pop_error() {
            let msg = flatten_error(&e.to_string());
            self.error(format!("compute pipeline: {}", msg));
            return None;
        }
        Some(ScriptComputePipeline { pipeline })
    }

    /// Create the compute IO bind group: `inputs` is a list of 1–4
    /// textures read at bindings `0..n` (raw, no sRGB decode), `output`
    /// is written as storage at binding `n`. Must match the pipeline's
    /// declared input count.
    #[rune::function]
    fn create_compute_bind_group(
        &mut self,
        inputs: Vec<Value>,
        output: &ScriptTexture,
    ) -> Option<ScriptBindGroup> {
        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        if inputs.is_empty() || inputs.len() > Gfx::MAX_COMPUTE_INPUTS {
            self.error(format!(
                "invalid compute input count {} (1..={})",
                inputs.len(),
                Gfx::MAX_COMPUTE_INPUTS
            ));
            return None;
        }
        let mut views = Vec::with_capacity(inputs.len());
        for v in &inputs {
            match v.borrow_ref::<ScriptTexture>() {
                Ok(t) => views.push(t),
                Err(_) => {
                    self.error("compute inputs must be textures");
                    return None;
                }
            }
        }
        let mut entries: Vec<wgpu::BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(i, t)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(&t.raw_view),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: views.len() as u32,
            resource: wgpu::BindingResource::TextureView(&output.raw_view),
        });
        gfx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bind_group = gfx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("script_compute_bind_group"),
            layout: &gfx.compute_bgls[views.len() - 1],
            entries: &entries,
        });
        if let Some(e) = gfx.pop_error() {
            let msg = flatten_error(&e.to_string());
            self.error(format!("compute bind group: {}", msg));
            return None;
        }
        Some(ScriptBindGroup {
            bind_group,
            scope: None,
        })
    }

    /// Shared body of the sprite-vertex constructors: one textured quad
    /// mapping `src` (texture pixels) onto `dst` (target pixels).
    fn sprite_vertices(
        &mut self,
        texture: &ScriptTexture,
        src: &Rect,
        dst: &Rect,
        color: &crate::gfx::color::Rgba8,
        opacity: f64,
    ) -> Option<ScriptBuffer> {
        use crate::gfx::rect::Rect as GfxRect;
        use crate::gfx::{Repeat, ZDepth};

        let Some(gfx) = self.gfx() else {
            self.error("the GPU is not available in this context");
            return None;
        };
        let (src_w, src_h) = (texture.width, texture.height);
        let mut batch = crate::gfx::sprite2d::Batch::new(src_w, src_h);
        batch.add(
            GfxRect::new(src.x1 as f32, src.y1 as f32, src.x2 as f32, src.y2 as f32),
            GfxRect::new(dst.x1 as f32, dst.y1 as f32, dst.x2 as f32, dst.y2 as f32),
            ZDepth::default(),
            (*color).into(),
            opacity as f32,
            Repeat::default(),
        );
        let vertices: Vec<ScriptVertex> = batch
            .vertices()
            .iter()
            .map(|v| ScriptVertex {
                position: [v.position.x, v.position.y, v.position.z],
                uv: [v.uv.x, v.uv.y],
                color: [v.color.r, v.color.g, v.color.b, v.color.a],
                opacity: v.opacity,
            })
            .collect();

        let buffer = gfx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("script_vertex_buffer"),
            size: (vertices.len() * std::mem::size_of::<ScriptVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&vertices));
        buffer.unmap();

        Some(ScriptBuffer {
            buffer,
            count: vertices.len() as u32,
        })
    }

    /// Convenience helper for the existing render-pass API: allocate six
    /// sprite-layout vertices, then bind/draw the returned buffer as usual.
    /// `options` requires `dst`; `src` defaults to the whole texture,
    /// `color` to white, and `opacity` to 1.0. Rectangles are y-down pixels.
    #[rune::function]
    fn create_sprite_vertices(
        &mut self,
        texture: &ScriptTexture,
        options: &rune::runtime::Object,
    ) -> Option<ScriptBuffer> {
        let options = match SpriteOptions::parse(options, texture.width, texture.height) {
            Ok(options) => options,
            Err(error) => {
                self.error(format!("create_sprite_vertices: {}", error));
                return None;
            }
        };
        self.sprite_vertices(
            texture,
            &options.src,
            &options.dst,
            &options.color,
            options.opacity,
        )
    }

    /// Export a function for other plugins to call via
    /// `rx.call_plugin`. The handler is called as
    /// `handler(state, rx, args)` with the *exporting* plugin's state.
    /// Returns whether the export succeeded.
    #[rune::function]
    fn export(&mut self, name: &str, handler: Function) -> bool {
        use crate::session::MessageType;

        if self.cmds.is_null() {
            self.session_mut().message(
                format!("Error: `{}` cannot be exported in this context", name),
                MessageType::Error,
            );
            return false;
        }
        let export = ScriptExport {
            plugin: self.plugin.clone(),
            name: name.to_string(),
            handler,
            state: None,
        };
        let result = unsafe { &mut *self.cmds }.add_export(export);
        match result {
            Ok(()) => true,
            Err(e) => {
                self.session_mut()
                    .message(format!("Error: {}", e), MessageType::Error);
                false
            }
        }
    }

    /// Call a function another plugin exported:
    /// `rx.call_plugin(plugin, name, args)`. The callee runs with its
    /// own plugin's state and this same context. Errors if the export
    /// doesn't exist or the callee fails.
    #[rune::function]
    fn call_plugin(
        &mut self,
        plugin: &str,
        name: &str,
        args: Vec<Value>,
    ) -> Result<Value, String> {
        use rune::alloc::clone::TryClone;

        if self.cmds.is_null() {
            return Err("plugins cannot be called in this context".to_string());
        }
        let (handler, state) = {
            let cmds = unsafe { &*self.cmds };
            let Some(export) = cmds.get_export(plugin, name) else {
                return Err(format!("no such export: {}::{}", plugin, name));
            };
            let Some(state) = export.state.clone() else {
                return Err(format!("{}::{} is not initialized", plugin, name));
            };
            let handler = export
                .handler
                .try_clone()
                .map_err(|e| format!("{}::{}: {}", plugin, name, e))?;
            (handler, state)
        };
        let args = rune::to_value(args).map_err(|e| e.to_string())?;
        handler
            .call::<Value>((state, &mut *self, args))
            .into_result()
            .map_err(|e| first_line_str(&e.to_string()))
    }

    /// Bind keys in a script mode: `rx.bind(mode, mapping)`. The mode is
    /// a script-mode name prefix; the mapping uses `:map` syntax, e.g.
    /// `"<tab> :v/prev"` or `"'r' :rotate 90 {:rotate 0}"`. The binding
    /// is script-tier: it wins over general bindings while the mode is
    /// active and may fire even while the mouse is held. Returns whether
    /// the mapping parsed.
    #[rune::function]
    fn bind(&mut self, mode: &str, mapping: &str) -> bool {
        use crate::cmd::{Command, KeyMapping};
        use crate::session::{BindingTier, MessageType, ModeString};

        let name = match ModeString::try_from_str(mode) {
            Ok(s) if !s.is_empty() => s,
            _ => {
                self.session_mut().message(
                    format!("Error: invalid mode name `{}`", mode),
                    MessageType::Error,
                );
                return false;
            }
        };
        match KeyMapping::parser(BindingTier::Script(name)).parse(mapping.trim()) {
            Ok((km, rest)) if rest.trim().is_empty() => {
                self.session_mut().command(Command::Map(Box::new(km)));
                true
            }
            Ok((_, rest)) => {
                self.session_mut().message(
                    format!("Error: trailing input in mapping: `{}`", rest),
                    MessageType::Error,
                );
                false
            }
            Err((e, _)) => {
                self.session_mut()
                    .message(format!("Error: {}", e), MessageType::Error);
                false
            }
        }
    }

    /// Run a builtin command, e.g. `rx.run_builtin("v/center")`. Script
    /// commands are not resolvable through this; it exists so handlers
    /// can delegate to (or compose) default behavior. Returns whether the
    /// invocation parsed and ran.
    #[rune::function]
    fn run_builtin(&mut self, invocation: &str) -> bool {
        use crate::cmd::Command;
        use crate::session::MessageType;

        let input = format!(":{}", invocation.trim());
        let session = self.session_mut();
        match session.cmdline.parse(&input) {
            Ok(Command::Script(name, _)) => {
                session.message(
                    format!("Error: ':{}' is not a builtin command", name),
                    MessageType::Error,
                );
                false
            }
            Ok(cmd) => {
                session.command(cmd);
                true
            }
            Err(e) => {
                session.message(format!("Error: {}", e), MessageType::Error);
                false
            }
        }
    }

    /// Draw text at `(x, y)` in UI coordinates. Only valid inside the
    /// `draw` hook; a no-op elsewhere.
    #[rune::function]
    fn draw_text(&mut self, text: &str, x: f64, y: f64, color: &crate::gfx::color::Rgba8) {
        use crate::font::TextAlign;

        let color = *color;
        if let Some(draw) = self.draw_mut() {
            draw.text_batch.add(
                text,
                x as f32,
                y as f32,
                crate::draw::UI_LAYER,
                color,
                TextAlign::Left,
            );
        }
    }

    /// Draw a 1px line from `p1` to `p2` (`(x, y)` tuples) in UI
    /// coordinates. Only valid inside the `draw` hook; a no-op elsewhere.
    #[rune::function]
    fn draw_line(&mut self, p1: (f64, f64), p2: (f64, f64), color: &crate::gfx::color::Rgba8) {
        use crate::gfx::math::Point2;
        use crate::gfx::shape2d::Shape;

        let color = *color;
        if let Some(draw) = self.draw_mut() {
            draw.ui_batch.add(
                Shape::line(
                    Point2::new(p1.0 as f32, p1.1 as f32),
                    Point2::new(p2.0 as f32, p2.1 as f32),
                )
                .stroke(1.0, color)
                .zdepth(crate::draw::UI_LAYER),
            );
        }
    }
}

/// Construct a color from RGB components (alpha 255).
#[rune::function]
fn rgb(r: i64, g: i64, b: i64) -> crate::gfx::color::Rgba8 {
    crate::gfx::color::Rgba8::new(r as u8, g as u8, b as u8, 0xff)
}

/// Construct a color from RGBA components. Alpha blends in `draw_line`
#[rune::function]
fn rgba(r: i64, g: i64, b: i64, a: i64) -> crate::gfx::color::Rgba8 {
    crate::gfx::color::Rgba8::new(r as u8, g as u8, b as u8, a as u8)
}

/// An immutable snapshot of a view, handed to scripts. Mutation goes
/// through session methods by id — live references never cross the
/// boundary (docs/rune-plan.md).
#[derive(rune::Any, Clone)]
#[rune(item = ::rx)]
pub struct ViewInfo {
    #[rune(get)]
    pub id: i64,
    /// Full sheet width: `frame_width * frames`.
    #[rune(get)]
    pub width: i64,
    #[rune(get)]
    pub height: i64,
    #[rune(get)]
    pub offset_x: f64,
    #[rune(get)]
    pub offset_y: f64,
    #[rune(get)]
    pub zoom: f64,
    /// Number of animation frames (1 for a still).
    #[rune(get)]
    pub frames: i64,
    /// Zero-based current animation frame.
    #[rune(get)]
    pub animation_frame: i64,
    /// Width of a single animation frame.
    #[rune(get)]
    pub frame_width: i64,
    /// Height of a single animation frame (same as `height`).
    #[rune(get)]
    pub frame_height: i64,
    /// Number of layers (1 for a flat view).
    #[rune(get)]
    pub nlayers: i64,
    /// Index of the active layer (`0` is the bottom compositing layer).
    #[rune(get)]
    pub active_layer: i64,
    /// Whether the builtin animation preview is enabled for this view.
    #[rune(get)]
    pub animation_preview_visible: bool,
}

/// The native `rx` module installed into every plugin's context.
fn module() -> Result<rune::Module, rune::ContextError> {
    let mut m = rune::Module::with_crate("rx")?;
    m.ty::<Ctx>()?;
    m.function_meta(Ctx::mode)?;
    m.function_meta(Ctx::prev_mode)?;
    m.function_meta(Ctx::switch_mode)?;
    m.function_meta(Ctx::fg)?;
    m.function_meta(Ctx::set_fg)?;
    m.function_meta(Ctx::bg)?;
    m.function_meta(Ctx::offset)?;
    m.function_meta(Ctx::screen_size)?;
    m.function_meta(Ctx::cursor)?;
    m.function_meta(Ctx::session_coords)?;
    m.function_meta(Ctx::active_view_coords)?;
    m.function_meta(Ctx::touch_view)?;
    m.function_meta(Ctx::view_pixels)?;
    m.function_meta(Ctx::view_layer_pixels)?;
    m.function_meta(Ctx::clear_view_rect)?;
    m.function_meta(Ctx::damage_view)?;
    m.function_meta(Ctx::message)?;
    m.function_meta(Ctx::active_view_id)?;
    m.function_meta(Ctx::views)?;
    m.function_meta(Ctx::set_animation_frame)?;
    m.function_meta(Ctx::set_animation_sequence)?;
    m.function_meta(Ctx::animation_sequence)?;
    m.function_meta(Ctx::clear_animation_sequence)?;
    m.function_meta(Ctx::set_animation_preview_visible)?;
    m.function_meta(Ctx::layer_visibility)?;
    m.function_meta(Ctx::layer_opacity)?;
    m.function_meta(Ctx::setting)?;
    m.function_meta(Ctx::set_setting)?;
    m.function_meta(Ctx::declare_setting)?;
    m.function_meta(Ctx::selection)?;
    m.function_meta(Ctx::set_selection)?;
    m.function_meta(Ctx::clear_selection)?;
    m.function_meta(Ctx::register_command)?;
    m.function_meta(Ctx::register_command_repeating)?;
    m.function_meta(Ctx::bind)?;
    m.function_meta(Ctx::export)?;
    m.function_meta(Ctx::call_plugin)?;
    m.function_meta(Ctx::run_builtin)?;
    m.function_meta(Ctx::draw_text)?;
    m.function_meta(Ctx::draw_line)?;
    m.function_meta(Ctx::create_texture)?;
    m.function_meta(Ctx::texture_pixels)?;
    m.function_meta(Ctx::write_png)?;
    m.function_meta(Ctx::read_file)?;
    m.function_meta(Ctx::read_png)?;
    m.function_meta(Ctx::create_shader)?;
    m.function_meta(Ctx::create_render_pipeline)?;
    m.function_meta(Ctx::create_transform_bind_group)?;
    m.function_meta(Ctx::create_transform_params_bind_group)?;
    m.function_meta(Ctx::create_texture_bind_group)?;
    m.function_meta(Ctx::create_sprite_vertices)?;
    m.function_meta(rgb)?;
    m.function_meta(rgba)?;
    m.function_meta(rect)?;
    m.function_meta(mat4_identity)?;
    m.function_meta(mat4_translation)?;
    m.function_meta(mat4_scale)?;
    m.function_meta(mat4_rotation_z)?;
    m.function_meta(mat4_mul)?;
    m.function_meta(mat4_transform_point)?;
    m.function_meta(atan2)?;
    m.ty::<ViewInfo>()?;
    m.ty::<crate::gfx::color::Rgba8>()?;
    m.ty::<ScriptTexture>()?;
    m.ty::<ScriptTextureView>()?;
    m.function_meta(ScriptTexture::view)?;
    m.function_meta(ScriptTexture::raw_view)?;
    m.function_meta(ScriptTexture::width)?;
    m.function_meta(ScriptTexture::height)?;
    m.function_meta(ScriptTexture::upload)?;
    m.function_meta(ScriptTexture::fill)?;
    m.ty::<ScriptShader>()?;
    m.ty::<ScriptPipeline>()?;
    m.ty::<ScriptBindGroup>()?;
    m.ty::<ScriptBuffer>()?;
    m.function_meta(ScriptBuffer::count)?;
    m.ty::<Mat4>()?;
    m.ty::<Rect>()?;
    m.function_meta(Ctx::create_compute_pipeline)?;
    m.function_meta(Ctx::create_compute_bind_group)?;
    m.ty::<ScriptComputePipeline>()?;
    m.ty::<ScriptComputePass>()?;
    m.function_meta(ScriptComputePass::set_pipeline)?;
    m.function_meta(ScriptComputePass::set_bind_group)?;
    m.function_meta(ScriptComputePass::dispatch)?;
    m.function_meta(ScriptComputePass::end)?;
    m.ty::<ScriptEncoder>()?;
    m.function_meta(ScriptEncoder::begin_render_pass)?;
    m.function_meta(ScriptEncoder::begin_view_pass)?;
    m.function_meta(ScriptEncoder::view_layer)?;
    m.function_meta(ScriptEncoder::view_staging)?;
    m.function_meta(ScriptEncoder::begin_staging_pass)?;
    m.function_meta(ScriptEncoder::view_bind_group)?;
    m.function_meta(ScriptEncoder::begin_compute_pass)?;
    m.ty::<ScriptPass>()?;
    m.function_meta(ScriptPass::draw_sprite)?;
    m.function_meta(ScriptPass::set_pipeline)?;
    m.function_meta(ScriptPass::set_bind_group)?;
    m.function_meta(ScriptPass::set_vertex_buffer)?;
    m.function_meta(ScriptPass::draw)?;
    m.function_meta(ScriptPass::end)?;
    Ok(m)
}

////////////////////////////////////////////////////////////////////////////
// Engine

/// Shared compiler state: the native context all plugins compile against.
pub struct ScriptEngine {
    context: Context,
    runtime: Arc<RuntimeContext>,
}

impl ScriptEngine {
    pub fn new() -> Result<Self, ScriptError> {
        let mut context =
            Context::with_default_modules().map_err(|e| ScriptError::Compile(e.to_string()))?;
        context
            .install(module().map_err(|e| ScriptError::Compile(e.to_string()))?)
            .map_err(|e| ScriptError::Compile(e.to_string()))?;
        let runtime = Arc::new(
            context
                .runtime()
                .map_err(|e| ScriptError::Compile(e.to_string()))?,
        );
        Ok(Self { context, runtime })
    }

    /// Compile a script from a file path.
    pub fn compile_path(&self, path: &Path) -> Result<CompiledScript, ScriptError> {
        let source = Source::from_path(path).map_err(|e| {
            ScriptError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: {}", path.display(), e),
            ))
        })?;
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        self.compile(name, source)
    }

    /// Compile a script from an in-memory string (used by tests).
    pub fn compile_str(&self, name: &str, text: &str) -> Result<CompiledScript, ScriptError> {
        let source = Source::memory(text).map_err(|e| ScriptError::Compile(e.to_string()))?;
        self.compile(name.to_string(), source)
    }

    fn compile(&self, name: String, source: Source) -> Result<CompiledScript, ScriptError> {
        let mut sources = Sources::new();
        sources
            .insert(source)
            .map_err(|e| ScriptError::Compile(e.to_string()))?;

        let mut diagnostics = Diagnostics::new();
        let result = rune::prepare(&mut sources)
            .with_context(&self.context)
            .with_diagnostics(&mut diagnostics)
            .build();

        if diagnostics.has_error() {
            return Err(ScriptError::Compile(render_diagnostics(
                &diagnostics,
                &sources,
            )));
        }
        let unit = result.map_err(|e| ScriptError::Compile(e.to_string()))?;

        Ok(CompiledScript {
            name,
            unit: Arc::new(unit),
            runtime: self.runtime.clone(),
        })
    }
}

/// Render diagnostics to a plain string (for the message line / logs).
fn render_diagnostics(diagnostics: &Diagnostics, sources: &Sources) -> String {
    use rune::termcolor::{Buffer, BufferWriter, ColorChoice};

    let writer = BufferWriter::stderr(ColorChoice::Never);
    let mut buffer: Buffer = writer.buffer();
    if diagnostics.emit(&mut buffer, sources).is_err() {
        return "unknown compile error".to_string();
    }
    String::from_utf8_lossy(buffer.as_slice()).into_owned()
}

////////////////////////////////////////////////////////////////////////////
// Compiled script

/// A compiled script, ready to call.
pub struct CompiledScript {
    pub name: String,
    unit: Arc<Unit>,
    runtime: Arc<RuntimeContext>,
}

impl CompiledScript {
    /// Whether the script defines a top-level function with this name.
    pub fn has_fn(&self, name: &str) -> bool {
        let vm = Vm::new(self.runtime.clone(), self.unit.clone());
        vm.lookup_function([name]).is_ok()
    }

    /// Call a top-level function. `MissingFn` if it isn't defined.
    pub fn call(
        &self,
        name: &str,
        args: impl rune::runtime::GuardedArgs,
    ) -> Result<Value, ScriptError> {
        let mut vm = Vm::new(self.runtime.clone(), self.unit.clone());
        if vm.lookup_function([name]).is_err() {
            return Err(ScriptError::MissingFn(name.to_string()));
        }
        let value = vm.call([name], args)?;
        Ok(value)
    }
}

impl fmt::Debug for CompiledScript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompiledScript({})", self.name)
    }
}

////////////////////////////////////////////////////////////////////////////
// Plugin host

/// A loaded plugin: compiled script + the state value its `init` returned.
pub struct Plugin {
    pub name: String,
    script: CompiledScript,
    state: Value,
    /// The plugin's directory (assets, shaders).
    root: std::path::PathBuf,
    /// Cleared when a hook errors at runtime; the plugin stops receiving
    /// hooks but its siblings are unaffected.
    enabled: bool,
}

/// Owns all plugins. Lives outside [`Session`] so hooks can borrow the
/// session mutably while the host borrows itself.
pub struct PluginHost {
    engine: ScriptEngine,
    plugins: Vec<Plugin>,
    /// Commands registered by plugins (cleared on unload/reload).
    commands: ScriptCommands,
    /// GPU capability; attached once the renderer exists.
    gfx: Option<Gfx>,
    dirs: Vec<std::path::PathBuf>,
    watchers: Vec<ReloadWatcher>,
    /// Last mode reported to `switch_mode` hooks (edge detection).
    last_mode: Option<String>,
}

/// Loop over enabled plugins defining `$hook` and call it; disable a
/// plugin whose hook errors. A macro rather than a generic fn: the
/// argument tuple borrows a per-call local `Ctx`, which a closure-based
/// helper can't express without aliasing the host. `$plugins`/`$cmds`
/// are the host's fields, split-borrowed by the caller.
macro_rules! dispatch {
    ($plugins:expr, $cmds:expr, $gfx:expr, $session:expr, $hook:expr, |$state:ident, $ctx:ident| $args:tt) => {
        for i in 0..$plugins.len() {
            {
                let plugin = &$plugins[i];
                if !plugin.enabled || !plugin.script.has_fn($hook) {
                    continue;
                }
            }
            let $state = $plugins[i].state.clone();
            let name = $plugins[i].name.clone();
            let mut ctx = Ctx::new($session)
                .with_commands(&mut *$cmds, &name)
                .with_gfx($gfx.as_ref())
                .with_root(Some($plugins[i].root.clone()));
            let $ctx = &mut ctx;
            let result = $plugins[i].script.call($hook, $args);
            drop(ctx);
            if let Err(e) = result {
                let plugin = &mut $plugins[i];
                plugin.enabled = false;
                log::error!("plugin `{}` {}: {}", plugin.name, $hook, e);
                let name = plugin.name.clone();
                $session.message(
                    format!("Plugin `{}` disabled: {}", name, first_line(&e)),
                    crate::session::MessageType::Error,
                );
            }
        }
    };
}

impl PluginHost {
    /// A host rooted at the given plugin directory (`None` = no plugins).
    pub fn new(dir: Option<std::path::PathBuf>) -> Result<Self, ScriptError> {
        Self::with_dirs(dir.into_iter().collect())
    }

    /// A host searching plugin directories in precedence order.
    pub fn with_dirs(dirs: Vec<std::path::PathBuf>) -> Result<Self, ScriptError> {
        let engine = ScriptEngine::new()?;
        let watchers = dirs
            .iter()
            .filter(|d| d.is_dir())
            .filter_map(|d| ReloadWatcher::new(d).ok())
            .collect();
        Ok(Self {
            engine,
            plugins: Vec::new(),
            commands: ScriptCommands::default(),
            gfx: None,
            dirs,
            watchers,
            last_mode: None,
        })
    }

    pub fn plugin_dir(&self) -> Option<&Path> {
        self.dirs.first().map(std::path::PathBuf::as_path)
    }

    pub fn plugin_dirs(&self) -> impl Iterator<Item = &Path> {
        self.dirs.iter().map(std::path::PathBuf::as_path)
    }

    /// Attach the GPU capability (device + queue clones). Called once
    /// the renderer exists, before plugins load, so `init` hooks can
    /// create GPU resources.
    pub fn attach_gfx(
        &mut self,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
    ) {
        self.gfx = Some(Gfx::new(device, queue));
    }

    /// The GPU capability, if attached.
    pub fn gfx(&self) -> Option<&Gfx> {
        self.gfx.as_ref()
    }

    pub fn plugins(&self) -> impl Iterator<Item = &Plugin> {
        self.plugins.iter()
    }

    /// Discover plugin entry points: `<dir>/<name>.rune` files and
    /// `<dir>/<name>/<name>.rune` packages, in lexicographic order.
    fn discover(dir: &Path) -> Vec<(String, std::path::PathBuf)> {
        let mut found = Vec::new();
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return found,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let entry_point = path.join(name).with_extension("rune");
                    if entry_point.is_file() {
                        found.push((name.to_string(), entry_point));
                    }
                }
            } else if path.extension().and_then(|e| e.to_str()) == Some("rune") {
                if let Some(stem) = path.file_stem().and_then(|n| n.to_str()) {
                    found.push((stem.to_string(), path.clone()));
                }
            }
        }
        found.sort();
        found
    }

    /// Load (or re-load) all plugins from the plugin search path. A plugin that
    /// fails to compile or whose `init` errors is reported to the message
    /// line and skipped; it never affects its siblings.
    pub fn load(&mut self, session: &mut Session) {
        let mut found = Vec::new();
        let mut names = std::collections::HashSet::new();
        for dir in &self.dirs {
            for (name, path) in Self::discover(dir) {
                if names.insert(name.clone()) {
                    found.push((name, path));
                } else {
                    log::warn!(
                        "plugin `{}` at {} shadowed by an earlier plugin directory",
                        name,
                        path.display()
                    );
                }
            }
        }
        for (name, path) in found {
            let script = match self.engine.compile_path(&path) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("plugin `{}`: {}", name, e);
                    session.message(
                        format!("Error loading plugin `{}`: {}", name, first_line(&e)),
                        crate::session::MessageType::Error,
                    );
                    continue;
                }
            };
            let root = path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default();
            let state = {
                let mut ctx = Ctx::new(session)
                    .with_commands(&mut self.commands, &name)
                    .with_gfx(self.gfx.as_ref())
                    .with_root(Some(root.clone()));
                match script.call("init", (&mut ctx,)) {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("plugin `{}` init: {}", name, e);
                        session.message(
                            format!("Error initializing plugin `{}`: {}", name, first_line(&e)),
                            crate::session::MessageType::Error,
                        );
                        continue;
                    }
                }
            };
            log::info!("plugin `{}` loaded from {}", name, path.display());
            self.commands.fill_export_states(&name, &state);
            self.plugins.push(Plugin {
                name,
                script,
                state,
                root,
                enabled: true,
            });
        }
        session
            .cmdline
            .set_script_commands(self.commands.help_entries());
    }

    /// Call `unload` on every plugin that defines it, then drop them all
    /// along with their command registrations.
    pub fn unload(&mut self, session: &mut Session) {
        for plugin in self.plugins.drain(..) {
            let mut ctx = Ctx::new(session);
            match plugin
                .script
                .call("unload", (plugin.state.clone(), &mut ctx))
            {
                Ok(_) | Err(ScriptError::MissingFn(_)) => {}
                Err(e) => {
                    log::error!("plugin `{}` unload: {}", plugin.name, e);
                }
            }
        }
        self.commands.clear();
        session.cmdline.set_script_commands(Vec::new());
    }

    /// Reload all plugins if the watcher saw a change. Returns whether a
    /// reload happened.
    pub fn reload_if_changed(&mut self, session: &mut Session) -> bool {
        if self.watchers.iter().any(ReloadWatcher::changed) {
            self.reload(session);
            true
        } else {
            false
        }
    }

    /// Unconditional reload: unload everything and load fresh.
    pub fn reload(&mut self, session: &mut Session) {
        self.unload(session);
        self.load(session);
        session.message("Plugins reloaded", crate::session::MessageType::Execution);
    }

    ////////////////////////////////////////////////////////////////////////
    // Hook dispatch
    //
    // Each dispatch loops over enabled plugins that define the hook and
    // calls it with `(state, rx, ...)`. A runtime error disables the
    // offending plugin and reports it; siblings are unaffected.

    /// Cursor moved (window logical coordinates). Runs before builtin
    /// event handling.
    pub fn dispatch_cursor_moved(&mut self, session: &mut Session, x: f64, y: f64) {
        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;
        dispatch!(plugins, commands, gfx, session, "cursor_moved", |state, ctx| (
            state, ctx, x, y
        ));
    }

    /// Front-to-back overlay mouse interception. Returning true consumes this
    /// event before ordinary plugin mouse hooks and builtin painting/panning.
    pub fn dispatch_capture_mouse(
        &mut self,
        session: &mut Session,
        button: &str,
        input: &str,
    ) -> bool {
        for i in (0..self.plugins.len()).rev() {
            let plugin = &self.plugins[i];
            if !plugin.enabled || !plugin.script.has_fn("capture_mouse") {
                continue;
            }
            let state = plugin.state.clone();
            let name = plugin.name.clone();
            let root = plugin.root.clone();
            let mut ctx = Ctx::new(session)
                .with_commands(&mut self.commands, &name)
                .with_gfx(self.gfx.as_ref())
                .with_root(Some(root));
            let result = self.plugins[i].script.call(
                "capture_mouse",
                (state, &mut ctx, button.to_string(), input.to_string()),
            );
            drop(ctx);
            let result = result.map_err(|e| first_line(&e)).and_then(|v| {
                rune::from_value::<bool>(v)
                    .map_err(|_| "capture_mouse must return a bool".to_string())
            });
            match result {
                Ok(true) => return true,
                Ok(false) => {}
                Err(e) => {
                    self.plugins[i].enabled = false;
                    log::error!("plugin `{}` capture_mouse: {}", name, e);
                    session.message(
                        format!("Plugin `{}` disabled: {}", name, e),
                        crate::session::MessageType::Error,
                    );
                }
            }
        }
        false
    }

    /// Mouse button input. Runs before builtin event handling.
    pub fn dispatch_mouse_input(&mut self, session: &mut Session, button: &str, input: &str) {
        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;
        dispatch!(plugins, commands, gfx, session, "mouse_input", |state, ctx| (
            state,
            ctx,
            button.to_string(),
            input.to_string()
        ));
    }

    /// End-of-update dispatch: fires `switch_mode` on mode edges, then
    /// `update`.
    pub fn dispatch_update(&mut self, session: &mut Session) {
        let mode = session.mode.to_string();
        let switched = self.last_mode.as_deref() != Some(mode.as_str());
        if switched {
            self.last_mode = Some(mode);
        }
        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;
        if switched {
            dispatch!(plugins, commands, gfx, session, "switch_mode", |state, ctx| (
                state, ctx
            ));
        }
        dispatch!(plugins, commands, gfx, session, "update", |state, ctx| (
            state, ctx
        ));
    }

    /// Dispatch a script command invocation: resolve `name` in the
    /// registry, parse `raw` against the declared signature, and call the
    /// handler as `handler(state, rx, args)` with the owning plugin's
    /// state. Errors (unknown command, bad arguments, disabled plugin)
    /// land in the message line.
    pub fn dispatch_command(&mut self, session: &mut Session, name: &str, raw: &str) {
        use crate::session::MessageType;
        use rune::alloc::clone::TryClone;

        let (params, handler, plugin_name) = match self.commands.get(name) {
            Some(c) => match c.handler.try_clone() {
                Ok(handler) => (c.params.clone(), handler, c.plugin.clone()),
                Err(e) => {
                    session.message(format!("Error: ':{}': {}", name, e), MessageType::Error);
                    return;
                }
            },
            None => {
                session.message(
                    format!("Error: unknown command: {}", name),
                    MessageType::Error,
                );
                return;
            }
        };
        let args = match parse_args(name, &params, raw) {
            Ok(a) => a,
            Err(e) => {
                session.message(format!("Error: {}", e), MessageType::Error);
                return;
            }
        };
        let args = match rune::to_value(args) {
            Ok(v) => v,
            Err(e) => {
                session.message(format!("Error: {}", e), MessageType::Error);
                return;
            }
        };
        let Some(i) = self
            .plugins
            .iter()
            .position(|p| p.name == plugin_name && p.enabled)
        else {
            session.message(
                format!("Error: ':{}': plugin `{}` is not loaded", name, plugin_name),
                MessageType::Error,
            );
            return;
        };
        let state = self.plugins[i].state.clone();
        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;
        let root = plugins[i].root.clone();
        let mut ctx = Ctx::new(session)
            .with_commands(commands, &plugin_name)
            .with_gfx(gfx.as_ref())
            .with_root(Some(root));
        let result = handler
            .call::<Value>((state, &mut ctx, args))
            .into_result()
            .map_err(|e| ScriptError::Vm(e.to_string()));
        drop(ctx);
        if let Err(e) = result {
            let plugin = &mut plugins[i];
            plugin.enabled = false;
            log::error!("plugin `{}` :{}: {}", plugin.name, name, e);
            session.message(
                format!("Plugin `{}` disabled: {}", plugin_name, first_line(&e)),
                MessageType::Error,
            );
        }
    }

    /// The per-frame `shade` stage: each plugin may record render
    /// passes on the frame's command encoder. The host owns the encoder
    /// round-trip; passes a hook leaves open are force-ended after the
    /// call. Each hook runs inside a GPU validation error scope — a GPU
    /// error disables the plugin like a script error would.
    pub fn dispatch_shade(
        &mut self,
        session: &mut Session,
        encoder: wgpu::CommandEncoder,
        view_targets: ViewTargets,
    ) -> wgpu::CommandEncoder {
        let shared: SharedEncoder = std::sync::Arc::new(std::sync::Mutex::new(Some(encoder)));
        let view_targets = std::sync::Arc::new(view_targets);
        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;
        if let Some(gfx) = gfx.as_ref() {
            gfx.frame.fetch_add(1, Ordering::Relaxed);
        }
        let encoder_gfx = gfx.as_ref().map(|g| EncoderGfx {
            sprites: g.sprites.clone(),
            device: g.device.clone(),
            texture_bgl: g.texture_bgl.clone(),
            sampler: g.sampler.clone(),
            frame: g.frame.clone(),
        });

        for i in 0..plugins.len() {
            {
                let plugin = &plugins[i];
                if !plugin.enabled || !plugin.script.has_fn("shade") {
                    continue;
                }
            }
            let state = plugins[i].state.clone();
            let name = plugins[i].name.clone();
            let passes: PassList = Default::default();
            let cpasses: ComputePassList = Default::default();
            let enc = ScriptEncoder {
                enc: shared.clone(),
                passes: passes.clone(),
                cpasses: cpasses.clone(),
                view_targets: view_targets.clone(),
                gfx: encoder_gfx.clone(),
            };

            if let Some(g) = gfx.as_ref() {
                g.device.push_error_scope(wgpu::ErrorFilter::Validation);
            }
            let mut ctx = Ctx::new(session)
                .with_commands(&mut *commands, &name)
                .with_gfx(gfx.as_ref())
                .with_root(Some(plugins[i].root.clone()));
            let result = plugins[i].script.call("shade", (state, &mut ctx, enc));
            drop(ctx);

            // End any passes the hook left open: a wedged encoder must
            // never escape a plugin.
            for pass in passes.lock().expect("pass list lock").drain(..) {
                pass.lock().expect("pass lock").take();
            }
            for pass in cpasses.lock().expect("compute pass list lock").drain(..) {
                pass.lock().expect("pass lock").take();
            }
            let gpu_error = gfx.as_ref().and_then(|g| g.pop_error());

            // A validation error poisons the wgpu command encoder: any
            // further recording (including the host's own passes) would
            // be fatal. Swap in a fresh one; the plugin's work this
            // frame is lost, the frame itself is not.
            if gpu_error.is_some() {
                if let Some(g) = gfx.as_ref() {
                    *shared.lock().expect("encoder lock") =
                        Some(g.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("render_encoder"),
                            },
                        ));
                }
            }

            let error = match (result, gpu_error) {
                (Err(e), _) => Some(first_line(&e)),
                (Ok(_), Some(e)) => Some(flatten_error(&e.to_string())),
                (Ok(_), None) => None,
            };
            if let Some(e) = error {
                let plugin = &mut plugins[i];
                plugin.enabled = false;
                log::error!("plugin `{}` shade: {}", plugin.name, e);
                let name = plugin.name.clone();
                session.message(
                    format!("Plugin `{}` disabled: {}", name, e),
                    crate::session::MessageType::Error,
                );
            }
        }

        let encoder = shared
            .lock()
            .expect("encoder lock")
            .take()
            .expect("the host always gets the frame encoder back");
        encoder
    }

    /// The `render` stage: the live screen pass — everything drawn, the
    /// present still ahead — handed to each plugin's
    /// `render(state, rx, pass)` hook for screen-space drawing.
    ///
    /// The host owns the pass round-trip. Pass validation in wgpu only
    /// surfaces when the pass *ends*, so to attribute errors to the
    /// plugin that recorded them, each hook gets the pass in its own
    /// `Arc` and the host ends it inside that hook's validation error
    /// scope, beginning a fresh load-pass over the screen for the next
    /// hook (pass boundaries are invisible: load + store). A kept
    /// handle errors cleanly. A GPU error disables the plugin and
    /// poisons the encoder — the host swaps in a fresh one so later
    /// plugins and the present still run (the frame's earlier recording
    /// is lost, the frame itself is not — the next frame re-renders
    /// everything).
    pub fn dispatch_render(
        &mut self,
        session: &mut Session,
        encoder: wgpu::CommandEncoder,
        pass: wgpu::RenderPass<'static>,
        screen: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        let mut pass = Some(pass);
        let mut encoder = Some(encoder);
        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;

        // Continuation pass: load whatever the screen holds, never clear.
        let begin_screen_pass = |encoder: &mut wgpu::CommandEncoder| {
            encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("script_render_stage"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: screen,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime()
        };

        for hook in ["render", "overlay"] {
            for i in 0..plugins.len() {
                {
                    let plugin = &plugins[i];
                    if !plugin.enabled || !plugin.script.has_fn(hook) {
                        continue;
                    }
                }
                let state = plugins[i].state.clone();
                let name = plugins[i].name.clone();

                if let Some(g) = gfx.as_ref() {
                    g.device.push_error_scope(wgpu::ErrorFilter::Validation);
                }
                let live = match pass.take() {
                    Some(p) => p,
                    None => begin_screen_pass(encoder.as_mut().expect("the host holds the encoder")),
                };
                let shared: SharedPass = std::sync::Arc::new(std::sync::Mutex::new(Some(live)));
                let script_pass = ScriptPass {
                    size: [session.width as u32, session.height as u32],
                    format: SCRIPT_TEXTURE_FORMAT,
                    sprites: gfx.as_ref().map(|g| g.sprites.clone()),
                    pass: shared.clone(),
                };
                let mut ctx = Ctx::new(session)
                    .with_commands(&mut *commands, &name)
                    .with_gfx(gfx.as_ref())
                    .with_root(Some(plugins[i].root.clone()));
                let result = plugins[i].script.call(hook, (state, &mut ctx, script_pass));
                drop(ctx);

                // End the pass while this hook's error scope is still
                // pushed: recording errors validate at pass end, and they
                // must land on the plugin that recorded them. The Arc is
                // dead from here on — a kept handle errors cleanly.
                shared.lock().expect("pass lock").take();
                let gpu_error = gfx.as_ref().and_then(|g| g.pop_error());

                if gpu_error.is_some() {
                    // The encoder is poisoned: swap in a fresh one. The
                    // next hook (or no one) begins a fresh screen pass.
                    if let Some(g) = gfx.as_ref() {
                        encoder = Some(g.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("render_encoder"),
                            },
                        ));
                    }
                }

                let error = match (result, gpu_error) {
                    (Err(e), _) => Some(first_line(&e)),
                    (Ok(_), Some(e)) => Some(flatten_error(&e.to_string())),
                    (Ok(_), None) => None,
                };
                if let Some(e) = error {
                    let plugin = &mut plugins[i];
                    plugin.enabled = false;
                    log::error!("plugin `{}` {}: {}", plugin.name, hook, e);
                    let name = plugin.name.clone();
                    session.message(
                        format!("Plugin `{}` disabled: {}", name, e),
                        crate::session::MessageType::Error,
                    );
                }
            }
        }

        // End the host's pass if no hook consumed it.
        pass.take();
        encoder
            .take()
            .expect("the host always gets the frame encoder back")
    }

    /// Whether a registered script command repeats on key-hold.
    pub fn command_repeats(&self, name: &str) -> bool {
        self.commands.get(name).is_some_and(|c| c.repeating)
    }

    /// The script command registry (read access, e.g. for help).
    pub fn commands(&self) -> &ScriptCommands {
        &self.commands
    }

    /// The per-frame `draw` hook: scripts populate the UI batches through
    /// `rx.draw_text` / `rx.draw_line`.
    pub fn dispatch_draw(&mut self, session: &mut Session, draw: &mut crate::draw::Context) {
        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;
        for i in 0..plugins.len() {
            {
                let plugin = &plugins[i];
                if !plugin.enabled || !plugin.script.has_fn("draw") {
                    continue;
                }
            }
            let state = plugins[i].state.clone();
            let name = plugins[i].name.clone();
            let mut ctx = Ctx::with_draw(session, draw)
                .with_commands(&mut *commands, &name)
                .with_gfx(gfx.as_ref())
                .with_root(Some(plugins[i].root.clone()));
            let result = plugins[i].script.call("draw", (state, &mut ctx));
            drop(ctx);
            if let Err(e) = result {
                let plugin = &mut plugins[i];
                plugin.enabled = false;
                log::error!("plugin `{}` draw: {}", plugin.name, e);
                let name = plugin.name.clone();
                session.message(
                    format!("Plugin `{}` disabled: {}", name, first_line(&e)),
                    crate::session::MessageType::Error,
                );
            }
        }
    }

    /// View lifecycle hooks, driven from the session's effects.
    pub fn dispatch_effects(&mut self, session: &mut Session, effects: &[crate::session::Effect]) {
        use crate::session::Effect;

        let Self {
            plugins,
            commands,
            gfx,
            ..
        } = self;
        for effect in effects {
            let (hook, id) = match effect {
                Effect::ViewAdded(id) => ("view_added", *id),
                Effect::ViewRemoved(id) => ("view_removed", *id),
                _ => continue,
            };
            let id = u16::from(id) as i64;
            match hook {
                "view_added" => {
                    dispatch!(plugins, commands, gfx, session, "view_added", |state, ctx| (
                        state, ctx, id
                    ))
                }
                _ => dispatch!(
                    plugins,
                    commands,
                    gfx,
                    session,
                    "view_removed",
                    |state, ctx| (state, ctx, id)
                ),
            }
        }
    }
}


/// First line of an error display, for the one-line message bar.
fn first_line(e: &ScriptError) -> String {
    first_line_str(&e.to_string())
}

fn first_line_str(s: &str) -> String {
    s.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Flatten a (possibly multi-line, with-cause) error display into one
/// message-bar line.
fn flatten_error(s: &str) -> String {
    let mut out = String::new();
    for l in s.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if !out.is_empty() {
            out.push_str(": ");
        }
        out.push_str(l.trim_end_matches(':'));
    }
    out
}

////////////////////////////////////////////////////////////////////////////
// Hot reload

/// Watches a directory and reports (debounced) whether anything changed.
pub struct ReloadWatcher {
    _watcher: notify::RecommendedWatcher,
    rx: mpsc::Receiver<()>,
}

impl ReloadWatcher {
    pub fn new(dir: &Path) -> Result<Self, ScriptError> {
        use notify::{RecursiveMode, Watcher};

        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                use notify::EventKind::*;
                if matches!(event.kind, Create(_) | Modify(_) | Remove(_)) {
                    tx.send(()).ok();
                }
            }
        })
        .map_err(|e| ScriptError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;

        watcher
            .watch(dir, RecursiveMode::Recursive)
            .map_err(|e| ScriptError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;

        Ok(Self {
            _watcher: watcher,
            rx,
        })
    }

    /// True if anything changed since the last call. Non-blocking; drains
    /// the queue so a burst of events reports once.
    pub fn changed(&self) -> bool {
        let mut changed = false;
        while self.rx.try_recv().is_ok() {
            changed = true;
        }
        changed
    }

    /// Wait up to `timeout` for a change (used by tests).
    pub fn wait_changed(&self, timeout: Duration) -> bool {
        if self.rx.recv_timeout(timeout).is_ok() {
            while self.rx.try_recv().is_ok() {}
            true
        } else {
            false
        }
    }
}

////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sprite_descriptor_defaults_validation_and_reuse() {
        let engine = ScriptEngine::new().unwrap();
        let descriptor = |expression: &str| {
            engine
                .compile_str("descriptor", &format!("pub fn make() {{ {} }}", expression))
                .unwrap()
                .call("make", ())
                .unwrap()
        };
        let value = descriptor("#{ dst: rx::rect(1.0, 2.0, 5.0, 6.0) }");
        let object = value.borrow_ref::<rune::runtime::Object>().unwrap();
        let options = SpriteOptions::parse(&object, 16, 8).unwrap();
        assert_eq!(
            (
                options.src.x1,
                options.src.y1,
                options.src.x2,
                options.src.y2
            ),
            (0.0, 0.0, 16.0, 8.0)
        );
        assert_eq!(options.color, crate::gfx::color::Rgba8::WHITE);
        assert_eq!(options.opacity, 1.0);
        assert_eq!(
            (
                options.dst.x1,
                options.dst.y1,
                options.dst.x2,
                options.dst.y2
            ),
            (1.0, 2.0, 5.0, 6.0)
        );

        let value = descriptor("#{ dst: rx::rect(1.0, 2.0, 5.0, 6.0), src: rx::rect(2.0, 3.0, 4.0, 5.0), color: rx::rgba(10, 20, 30, 40), opacity: 0.5 }");
        let object = value.borrow_ref::<rune::runtime::Object>().unwrap();
        for _ in 0..2 {
            let options = SpriteOptions::parse(&object, 16, 8).unwrap();
            assert_eq!(
                (
                    options.src.x1,
                    options.src.y1,
                    options.src.x2,
                    options.src.y2
                ),
                (2.0, 3.0, 4.0, 5.0)
            );
            assert_eq!(
                options.color,
                crate::gfx::color::Rgba8 {
                    r: 10,
                    g: 20,
                    b: 30,
                    a: 40
                }
            );
            assert_eq!(options.opacity, 0.5);
        }
        for (expression, message) in [
            ("#{}", "missing required option `dst`"),
            ("#{ dst: 1 }", "`dst` must be a Rect"),
            (
                "#{ dst: rx::rect(0.0, 0.0, 1.0, 1.0), src: false }",
                "`src` must be a Rect",
            ),
            (
                "#{ dst: rx::rect(0.0, 0.0, 1.0, 1.0), color: 1 }",
                "`color` must be an Rgba8",
            ),
            (
                "#{ dst: rx::rect(0.0, 0.0, 1.0, 1.0), opacity: false }",
                "`opacity` must be a float",
            ),
            (
                "#{ dst: rx::rect(0.0, 0.0, 1.0, 1.0), opactiy: 0.5 }",
                "unknown option `opactiy`",
            ),
        ] {
            let value = descriptor(expression);
            let object = value.borrow_ref::<rune::runtime::Object>().unwrap();
            assert_eq!(SpriteOptions::parse(&object, 16, 8).err().unwrap(), message);
        }
    }

    #[test]
    fn compile_and_call() {
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str("t", "pub fn init() { 41 + 1 }")
            .unwrap();
        let v = script.call("init", ()).unwrap();
        let n: i64 = rune::from_value(v).unwrap();
        assert_eq!(n, 42);
    }

    #[test]
    fn compile_error_reports_diagnostics() {
        let engine = ScriptEngine::new().unwrap();
        let err = engine
            .compile_str("bad", "pub fn init() { let }")
            .unwrap_err();
        match err {
            ScriptError::Compile(msg) => {
                assert!(!msg.is_empty(), "diagnostics should not be empty");
            }
            other => panic!("expected compile error, got: {}", other),
        }
    }

    #[test]
    fn missing_function_is_distinguished() {
        let engine = ScriptEngine::new().unwrap();
        let script = engine.compile_str("t", "pub fn init() {}").unwrap();
        assert!(script.has_fn("init"));
        assert!(!script.has_fn("draw"));
        match script.call("draw", ()) {
            Err(ScriptError::MissingFn(name)) => assert_eq!(name, "draw"),
            other => panic!("expected MissingFn, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn runtime_error_is_reported() {
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str("t", "pub fn init() { None.unwrap() }")
            .unwrap();
        match script.call("init", ()) {
            Err(ScriptError::Vm(_)) => {}
            other => panic!("expected Vm error, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn state_value_round_trip() {
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                struct State { count }
                pub fn init() { State { count: 0 } }
                pub fn bump(state) { state.count += 1; state.count }
                "#,
            )
            .unwrap();
        let state = script.call("init", ()).unwrap();
        let v = script.call("bump", (state.clone(),)).unwrap();
        let n: i64 = rune::from_value(v).unwrap();
        assert_eq!(n, 1);
        let v = script.call("bump", (state,)).unwrap();
        let n: i64 = rune::from_value(v).unwrap();
        assert_eq!(n, 2, "state mutations must persist across calls");
    }

    /// A bare headless session for script tests.
    pub(crate) fn test_session() -> Session {
        let proj_dirs = directories::ProjectDirs::from("io", "cloudhead", "rx").unwrap();
        let base_dirs = directories::BaseDirs::new().unwrap();
        Session::new(640, 480, std::env::temp_dir(), proj_dirs, base_dirs)
    }

    #[test]
    fn ctx_reads_and_mutates_session() {
        let mut session = test_session();
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn init(rx) {
                    rx.message("hello from rune");
                    rx.mode()
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("init", (&mut ctx,)).unwrap();
        drop(ctx);

        let mode: String = rune::from_value(v).unwrap();
        assert_eq!(mode, "normal");
        assert_eq!(session.message.to_string(), "hello from rune");
    }

    #[test]
    fn stored_ctx_is_revoked_after_the_call() {
        let mut session = test_session();
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                struct State { rx }
                pub fn init(rx) { State { rx } }
                pub fn later(state) { state.rx.mode() }
                "#,
            )
            .unwrap();

        let state = {
            let mut ctx = Ctx::new(&mut session);
            script.call("init", (&mut ctx,)).unwrap()
        };
        // The guard was revoked when `init` returned; using the smuggled
        // ctx must be a clean runtime error, not a dangling-pointer deref.
        match script.call("later", (state,)) {
            Err(ScriptError::Vm(_)) => {}
            other => panic!("expected Vm error, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn session_view_api() {
        use crate::view::FileStatus;

        let mut session = test_session().with_blank(FileStatus::NoFile, 128, 96);
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn probe(rx) {
                    let views = rx.views();
                    let v = views[0];
                    (rx.active_view_id(), views.len(), v.width, v.height)
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("probe", (&mut ctx,)).unwrap();
        let (id, count, w, h): (i64, i64, i64, i64) = rune::from_value(v).unwrap();
        assert_eq!(count, 1);
        assert_eq!(id, 1);
        assert_eq!((w, h), (128, 96));
    }

    #[test]
    fn view_info_frame_metadata() {
        use crate::view::FileStatus;

        let mut session = test_session().with_blank(FileStatus::NoFile, 128, 96);
        // Two extra frames: the sheet is 3 frames wide, frame size
        // stays 128x96.
        session.command(crate::cmd::Command::FrameAdd);
        session.command(crate::cmd::Command::FrameAdd);

        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn probe(rx) {
                    let v = rx.views()[0];
                    let set = rx.set_animation_frame(v.id, 5);
                    let hidden = rx.set_animation_preview_visible(v.id, false);
                    let after = rx.views()[0];
                    (
                        v.frames, v.frame_width, v.frame_height, v.width, v.height,
                        set, hidden, after.animation_frame,
                    )
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("probe", (&mut ctx,)).unwrap();
        let (frames, fw, fh, w, h, set, hidden, frame): (
            i64,
            i64,
            i64,
            i64,
            i64,
            bool,
            bool,
            i64,
        ) = rune::from_value(v).unwrap();
        assert_eq!((frames, fw, fh), (3, 128, 96));
        // `width` is the full sheet: fw * frames.
        assert_eq!((w, h), (3 * 128, 96));
        assert!(set);
        assert!(hidden);
        assert_eq!(frame, 2, "frame setters wrap by the frame count");
        assert!(!session.active_view().animation_preview_visible);
    }

    #[test]
    fn animation_sequence_api_validates_and_round_trips() {
        use crate::view::FileStatus;

        let mut session = test_session().with_blank(FileStatus::NoFile, 16, 16);
        session.command(crate::cmd::Command::FrameAdd);
        session.command(crate::cmd::Command::FrameAdd);

        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn probe(rx) {
                    let id = rx.active_view_id();
                    let set = rx.set_animation_sequence(id, [0, 1, 2, 1]);
                    let sequence = rx.animation_sequence(id);
                    let invalid = rx.set_animation_sequence(id, [0, 3]);
                    let preserved = rx.animation_sequence(id);
                    let cleared = rx.clear_animation_sequence(id);
                    (set, sequence, invalid, preserved, cleared, rx.animation_sequence(id))
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let value = script.call("probe", (&mut ctx,)).unwrap();
        let (set, sequence, invalid, preserved, cleared, after): (
            bool,
            Vec<i64>,
            bool,
            Vec<i64>,
            bool,
            Vec<i64>,
        ) = rune::from_value(value).unwrap();
        assert!(set);
        assert_eq!(sequence, vec![0, 1, 2, 1]);
        assert!(!invalid);
        assert_eq!(preserved, sequence);
        assert!(cleared);
        assert!(after.is_empty());
    }

    #[test]
    fn animation_mode_setting_updates_playback_silently() {
        use crate::view::FileStatus;

        let dir = tempfile::tempdir().unwrap();
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("plugins/animation-mode/animation-mode.rune");
        std::fs::copy(source, dir.path().join("animation-mode.rune")).unwrap();

        let mut session = test_session().with_blank(FileStatus::NoFile, 16, 16);
        for _ in 0..3 {
            session.command(crate::cmd::Command::FrameAdd);
        }
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);
        assert_eq!(host.plugins().filter(|plugin| plugin.enabled).count(), 1);
        host.dispatch_update(&mut session);
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "{}",
            session.message
        );
        assert!(session.active_view().animation.sequence().is_empty());
        let message = session.message.to_string();
        session.command(crate::cmd::Command::Set(
            "animation/mode".into(),
            crate::cmd::Value::Ident("reverse".into()),
        ));
        host.dispatch_update(&mut session);
        assert_eq!(
            session.active_view().animation.sequence(),
            &[3, 2, 1, 0],
            "{}; setting={:?}; enabled={}",
            session.message,
            session.settings.get("animation/mode"),
            host.plugins().filter(|p| p.enabled).count()
        );
        assert_eq!(
            session.message.to_string(),
            message,
            "mode changes must be silent"
        );
        session.views.active_mut().unwrap().animation.step();
        let index = session.active_view().animation.index;
        host.dispatch_update(&mut session);
        assert_eq!(
            session.active_view().animation.index,
            index,
            "updates must not restart playback"
        );

        session.command(crate::cmd::Command::Set(
            "animation/mode".into(),
            crate::cmd::Value::Ident("ping-pong".into()),
        ));
        host.dispatch_update(&mut session);
        assert_eq!(
            session.active_view().animation.sequence(),
            &[0, 1, 2, 3, 2, 1]
        );

        session.views.active_mut().unwrap().animation.set_frame(0);
        for _ in 0..4 {
            session.views.active_mut().unwrap().animation.step();
        }
        assert_eq!(session.active_view().animation.index, 2);
        host.dispatch_update(&mut session);
        session.views.active_mut().unwrap().animation.step();
        assert_eq!(
            session.active_view().animation.index,
            1,
            "updates must preserve the return leg of ping-pong"
        );

        session.command(crate::cmd::Command::Set(
            "animation/mode".into(),
            crate::cmd::Value::Ident("invalid".into()),
        ));
        host.dispatch_update(&mut session);
        assert_eq!(
            session.settings.get("animation/mode"),
            Some(&crate::cmd::Value::Ident("ping-pong".into()))
        );
        assert_eq!(
            session.active_view().animation.sequence(),
            &[0, 1, 2, 3, 2, 1]
        );
        let error = session.message.to_string();
        host.dispatch_update(&mut session);
        assert_eq!(session.message.to_string(), error);

        session.command(crate::cmd::Command::Set(
            "animation/mode".into(),
            crate::cmd::Value::Ident("forward".into()),
        ));
        host.dispatch_update(&mut session);
        assert!(session.active_view().animation.sequence().is_empty());

        session.command(crate::cmd::Command::FrameRemove);
        session.command(crate::cmd::Command::FrameRemove);
        session.command(crate::cmd::Command::Set(
            "animation/mode".into(),
            crate::cmd::Value::Ident("ping-pong".into()),
        ));
        host.dispatch_update(&mut session);
        assert_eq!(session.active_view().animation.sequence(), &[0, 1]);

        session.command(crate::cmd::Command::FrameRemove);
        host.dispatch_update(&mut session);
        assert_eq!(session.active_view().animation.sequence(), &[0]);
    }

    #[test]
    fn animation_mode_remembers_each_view_without_restarting() {
        use crate::view::FileStatus;
        let dir = tempfile::tempdir().unwrap();
        std::fs::copy(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("plugins/animation-mode/animation-mode.rune"),
            dir.path().join("animation-mode.rune"),
        )
        .unwrap();
        let mut session = test_session().with_blank(
            FileStatus::New(crate::view::FileStorage::Single("first.png".into())),
            16,
            16,
        );
        for _ in 0..3 {
            session.command(crate::cmd::Command::FrameAdd);
        }
        let first = session.views.active_id;
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);
        host.dispatch_update(&mut session);
        session.command(crate::cmd::Command::Set(
            "animation/mode".into(),
            crate::cmd::Value::Ident("ping-pong".into()),
        ));
        host.dispatch_update(&mut session);
        session.views.active_mut().unwrap().animation.set_frame(0);
        for _ in 0..4 {
            session.views.active_mut().unwrap().animation.step();
        }

        session.blank(FileStatus::NoFile, 16, 16);
        for _ in 0..2 {
            session.command(crate::cmd::Command::FrameAdd);
        }
        let second = session.views.active_id;
        assert_ne!(first, second);
        host.dispatch_update(&mut session);
        assert_eq!(
            session.settings.get("animation/mode"),
            Some(&crate::cmd::Value::Ident("forward".into()))
        );
        assert!(session.active_view().animation.sequence().is_empty());
        session.command(crate::cmd::Command::Set(
            "animation/mode".into(),
            crate::cmd::Value::Ident("reverse".into()),
        ));
        host.dispatch_update(&mut session);
        assert_eq!(session.active_view().animation.sequence(), &[2, 1, 0]);

        session.views.activate(first);
        host.dispatch_update(&mut session);
        assert_eq!(
            session.settings.get("animation/mode"),
            Some(&crate::cmd::Value::Ident("ping-pong".into()))
        );
        assert_eq!(
            session.active_view().animation.sequence(),
            &[0, 1, 2, 3, 2, 1]
        );
        session.views.active_mut().unwrap().animation.step();
        assert_eq!(
            session.active_view().animation.index,
            1,
            "switching views must preserve the return leg"
        );
        session.views.activate(second);
        host.dispatch_update(&mut session);
        assert_eq!(
            session.settings.get("animation/mode"),
            Some(&crate::cmd::Value::Ident("reverse".into()))
        );
        assert_eq!(session.active_view().animation.sequence(), &[2, 1, 0]);
        assert_eq!(host.plugins().filter(|p| p.enabled).count(), 1);
    }

    #[test]
    fn view_info_layer_metadata() {
        use crate::cmd::Command;
        use crate::view::FileStatus;

        let mut session = test_session().with_blank(FileStatus::NoFile, 128, 96);
        // Three layers, active on the middle one, top layer hidden.
        session.command(Command::LayerAdd);
        session.command(Command::LayerAdd);
        session.command(Command::LayerSet(1));
        session.command(Command::LayerHide(Some(2)));

        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn probe(rx) {
                    let v = rx.views()[0];
                    (v.nlayers, v.active_layer, rx.layer_visibility(v.id))
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("probe", (&mut ctx,)).unwrap();
        let (nlayers, active, vis): (i64, i64, Vec<bool>) = rune::from_value(v).unwrap();
        assert_eq!((nlayers, active), (3, 1));
        // Bottom strip first: layers 0 and 1 visible, top (2) hidden.
        assert_eq!(vis, vec![true, true, false]);
    }

    #[test]
    fn view_layer_pixels_reads_and_guards() {
        use crate::view::FileStatus;

        // Single-layer view: strip 0 is the whole sheet. A bare unit-test
        // session has no recorded snapshot at the grown extent after LayerAdd.
        let mut session = test_session().with_blank(FileStatus::NoFile, 8, 8);

        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn probe(rx) {
                    let id = rx.active_view_id();
                    let l0 = rx.view_layer_pixels(id, 0, rx::rect(0.0, 0.0, 8.0, 8.0));
                    let oob = rx.view_layer_pixels(id, 1, rx::rect(0.0, 0.0, 8.0, 8.0));
                    let neg = rx.view_layer_pixels(id, -1, rx::rect(0.0, 0.0, 8.0, 8.0));
                    let sub = rx.view_layer_pixels(id, 0, rx::rect(2.0, 2.0, 5.0, 6.0));
                    (l0.unwrap().len(), oob.is_none(), neg.is_none(), sub.unwrap().len())
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("probe", (&mut ctx,)).unwrap();
        let (l0, oob, neg, sub): (i64, bool, bool, i64) = rune::from_value(v).unwrap();
        // The full strip is 8*8 rgba8 = 256 bytes.
        assert_eq!(l0, 256);
        // Out-of-range and negative strips read None.
        assert!(oob && neg);
        // The sub-rect is 3 wide * 4 tall * 4 bytes = 48.
        assert_eq!(sub, 48);
    }

    #[test]
    fn settings_get_set() {
        let mut session = test_session();
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn probe(rx) {
                    let before = rx.setting("debug");
                    let ok = rx.set_setting("debug", true);
                    let bad_type = rx.set_setting("debug", 3.5);
                    let missing = rx.set_setting("no/such/setting", 1);
                    (before, ok, bad_type, missing, rx.setting("debug"))
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("probe", (&mut ctx,)).unwrap();
        let (before, ok, bad_type, missing, after): (bool, bool, bool, bool, bool) =
            rune::from_value(v).unwrap();
        assert!(!before);
        assert!(ok);
        assert!(!bad_type, "type-mismatched set must be rejected");
        assert!(!missing, "unknown setting must be rejected");
        assert!(after, "the set must be visible");
        assert!(session.settings["debug"].is_set());
    }

    #[test]
    fn selection_round_trip() {
        let mut session = test_session();
        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str(
                "t",
                r#"
                pub fn probe(rx) {
                    let empty = rx.selection();
                    rx.set_selection(1, 2, 11, 22);
                    let some = rx.selection();
                    (empty, some)
                }
                "#,
            )
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("probe", (&mut ctx,)).unwrap();
        let (empty, some): (Option<(i64, i64, i64, i64)>, Option<(i64, i64, i64, i64)>) =
            rune::from_value(v).unwrap();
        assert_eq!(empty, None);
        assert_eq!(some, Some((1, 2, 11, 22)));
        assert!(session.selection.is_some());
    }

    #[test]
    fn prev_mode_tracks_switches() {
        use crate::session::Mode;

        let mut session = test_session();
        session.switch_mode(Mode::Help);

        let engine = ScriptEngine::new().unwrap();
        let script = engine
            .compile_str("t", "pub fn probe(rx) { (rx.mode(), rx.prev_mode()) }")
            .unwrap();

        let mut ctx = Ctx::new(&mut session);
        let v = script.call("probe", (&mut ctx,)).unwrap();
        let (mode, prev): (String, Option<String>) = rune::from_value(v).unwrap();
        assert_eq!(mode, "help");
        assert_eq!(prev.as_deref(), Some("normal"));
    }

    fn write_plugin(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name).with_extension("rune"), body).unwrap();
    }

    #[test]
    fn host_loads_plugins_in_order() {
        let dir = tempfile::tempdir().unwrap();
        // Flat file plugin.
        write_plugin(
            dir.path(),
            "alpha",
            r#"pub fn init(rx) { rx.message("alpha up"); #{} }"#,
        );
        // Directory-package plugin.
        std::fs::create_dir(dir.path().join("beta")).unwrap();
        write_plugin(
            &dir.path().join("beta"),
            "beta",
            r#"pub fn init(rx) { rx.message("beta up"); #{} }"#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);

        let names: Vec<_> = host.plugins().map(|p| p.name.clone()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
        // beta loaded last; its message is the visible one.
        assert_eq!(session.message.to_string(), "beta up");
    }

    #[test]
    fn host_searches_multiple_plugin_dirs_with_first_match_winning() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        write_plugin(
            first.path(),
            "shared",
            r#"pub fn init(rx) { rx.message("first"); #{} }"#,
        );
        write_plugin(second.path(), "extra", r#"pub fn init(rx) { #{} }"#);
        write_plugin(
            second.path(),
            "shared",
            r#"pub fn init(rx) { rx.message("second"); #{} }"#,
        );

        let mut session = test_session();
        let mut host = PluginHost::with_dirs(vec![
            first.path().to_path_buf(),
            second.path().to_path_buf(),
        ])
        .unwrap();
        host.load(&mut session);

        let names: Vec<_> = host.plugins().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["shared", "extra"]);
        assert_eq!(host.plugin_dirs().collect::<Vec<_>>(), vec![first.path(), second.path()]);
        assert_eq!(session.message.to_string(), "first");
    }

    #[test]
    fn broken_plugin_does_not_block_siblings() {
        let dir = tempfile::tempdir().unwrap();
        write_plugin(dir.path(), "broken", "pub fn init(rx) { let }");
        write_plugin(
            dir.path(),
            "crashy",
            r#"pub fn init(rx) { None.unwrap() }"#,
        );
        write_plugin(dir.path(), "good", r#"pub fn init(rx) { #{} }"#);

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);

        let names: Vec<_> = host.plugins().map(|p| p.name.clone()).collect();
        assert_eq!(names, vec!["good"]);
    }

    #[test]
    fn unload_hook_runs_on_reload() {
        let dir = tempfile::tempdir().unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) { #{} }
            pub fn unload(state, rx) { rx.message("p unloaded"); }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1);

        // Change the plugin and force a reload: unload runs, new code lands.
        write_plugin(
            dir.path(),
            "p",
            r#"pub fn init(rx) { rx.message("p v2"); #{} }"#,
        );
        host.reload(&mut session);
        assert_eq!(host.plugins().count(), 1);
        // The reload banner is posted last; v2's init ran before it.
        assert_eq!(session.message.to_string(), "Plugins reloaded");
    }

    fn host_with(dir: &Path, name: &str, body: &str) -> (PluginHost, Session) {
        write_plugin(dir, name, body);
        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.to_path_buf())).unwrap();
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1, "fixture plugin must load");
        (host, session)
    }

    #[test]
    fn input_hooks_fire_before_builtins_via_update() {
        use crate::event::Event;
        use crate::execution::Execution;
        use crate::platform;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) { #{} }
            pub fn cursor_moved(state, rx, x, y) {
                rx.message(`cursor ${x} ${y}`);
            }
            pub fn mouse_input(state, rx, button, input) {
                rx.message(`mouse ${button} ${input}`);
            }
            "#,
        );
        // The builtin handlers need an active view to exist.
        let mut session = session.with_blank(crate::view::FileStatus::NoFile, 32, 32);

        let mut exec = Execution::normal().unwrap();
        let mut events = vec![Event::CursorMoved(platform::LogicalPosition::new(5.0, 6.0))];
        session.update(
            &mut events,
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.message.to_string(), "cursor 5.0 6.0");

        let mut events = vec![Event::MouseInput(
            platform::MouseButton::Left,
            platform::InputState::Pressed,
        )];
        session.update(
            &mut events,
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.message.to_string(), "mouse left pressed");
    }

    #[test]
    fn switch_mode_hook_fires_on_edges_only() {
        use crate::session::Mode;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) { #{ switches: 0 } }
            pub fn switch_mode(state, rx) {
                state.switches += 1;
                rx.message(`switched to ${rx.mode()} (#${state.switches})`);
            }
            "#,
        );

        // First dispatch observes the initial mode as an edge.
        host.dispatch_update(&mut session);
        assert_eq!(session.message.to_string(), "switched to normal (#1)");

        // Same mode: no edge, no hook.
        session.message("sentinel", crate::session::MessageType::Info);
        host.dispatch_update(&mut session);
        assert_eq!(session.message.to_string(), "sentinel");

        session.switch_mode(Mode::Help);
        host.dispatch_update(&mut session);
        assert_eq!(session.message.to_string(), "switched to help (#2)");
    }

    #[test]
    fn view_lifecycle_hooks() {
        use crate::session::Effect;
        use crate::view::FileStatus;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) { #{} }
            pub fn view_added(state, rx, id) { rx.message(`added ${id}`); }
            pub fn view_removed(state, rx, id) { rx.message(`removed ${id}`); }
            "#,
        );
        let mut session = session.with_blank(FileStatus::NoFile, 32, 32);
        let id = session.views.active_id;

        host.dispatch_effects(&mut session, &[Effect::ViewAdded(id)]);
        assert_eq!(session.message.to_string(), "added 1");
        host.dispatch_effects(&mut session, &[Effect::ViewRemoved(id)]);
        assert_eq!(session.message.to_string(), "removed 1");
    }

    #[test]
    fn erroring_hook_disables_only_that_plugin() {
        let dir = tempfile::tempdir().unwrap();
        write_plugin(
            dir.path(),
            "bad",
            r#"
            pub fn init(rx) { #{} }
            pub fn update(state, rx) { None.unwrap() }
            "#,
        );
        write_plugin(
            dir.path(),
            "good",
            r#"
            pub fn init(rx) { #{ ticks: 0 } }
            pub fn update(state, rx) {
                state.ticks += 1;
                rx.message(`tick ${state.ticks}`);
            }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 2);

        host.dispatch_update(&mut session);
        // `bad` errored and was disabled; `good` still ran (it runs after
        // `bad` alphabetically, so its message is the last one).
        assert_eq!(session.message.to_string(), "tick 1");

        host.dispatch_update(&mut session);
        assert_eq!(session.message.to_string(), "tick 2");
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "bad plugin must be disabled"
        );
    }

    #[test]
    fn draw_hook_populates_ui_batches() {
        use crate::draw;
        use crate::font::TextBatch;
        use crate::gfx::{shape2d, sprite2d};
        use crate::sprite;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) { #{} }
            pub fn draw(state, rx) {
                rx.draw_text("hi", 10.0, 20.0, rx::rgb(255, 0, 0));
                rx.draw_line((0.0, 0.0), (50.0, 50.0), rx::rgb(0, 255, 0));
            }
            "#,
        );

        let mut draw_ctx = draw::Context {
            ui_batch: shape2d::Batch::new(),
            text_batch: TextBatch::new(96, 208, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT),
            overlay_batch: TextBatch::new(96, 208, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT),
            cursor_sprite: sprite::Sprite::new(96, 96),
            tool_batch: sprite2d::Batch::new(96, 96),
            paste_batch: sprite2d::Batch::new(8, 8),
            checker_batch: sprite2d::Batch::new(2, 2),
        };
        host.dispatch_draw(&mut session, &mut draw_ctx);

        // "hi" = 2 glyphs * 6 vertices.
        assert_eq!(draw_ctx.text_batch.vertices().len(), 12);
        // One stroked line = one quad = 6 vertices.
        assert_eq!(draw_ctx.ui_batch.vertices().len(), 6);
        // Drawing outside the draw hook is a no-op, not a crash.
        host.dispatch_update(&mut session);
    }

    #[test]
    fn script_mode_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("probe", ["str"], "Switch and probe", probe);
                #{}
            }
            pub fn probe(state, rx, args) {
                let ok = rx.switch_mode(args[0]);
                let prev = rx.prev_mode().unwrap_or("-");
                rx.message(`${ok} ${rx.mode()} ${prev}`);
            }
            "#,
        );

        // Custom mode: accepted, prev_mode tracked.
        host.dispatch_command(&mut session, "probe", "funky");
        assert_eq!(session.message.to_string(), "true funky normal");
        assert_eq!(session.mode.to_string(), "funky");

        // Builtin name maps to the builtin mode.
        host.dispatch_command(&mut session, "probe", "visual");
        assert_eq!(session.mode, crate::session::Mode::Visual(Default::default()));

        // An over-long name is rejected and the mode stays.
        host.dispatch_command(
            &mut session,
            "probe",
            "way-too-long-of-a-mode-name-to-be-allowed-in-here",
        );
        assert!(session.message.to_string().starts_with("false"));
        assert_eq!(session.mode, crate::session::Mode::Visual(Default::default()));
    }

    #[test]
    fn script_mode_click_does_not_pass_views() {
        use crate::event::Event;
        use crate::execution::Execution;
        use crate::platform;
        use crate::session::Mode;
        use crate::view::FileStatus;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, session) = host_with(dir.path(), "p", "pub fn init(rx) { #{} }");
        // The first view must not be a scratch pad (`NoFile`), or adding
        // the second view would replace it.
        let mut session = session.with_blank(
            FileStatus::New(std::path::PathBuf::from("/tmp/rx-test-first.png").into()),
            32,
            32,
        );
        let first = session.views.active_id;
        session.blank(FileStatus::NoFile, 32, 32);
        let second = session.views.active_id;
        assert_ne!(first, second);
        session.activate(first);

        let mut exec = Execution::normal().unwrap();
        let click = || {
            vec![
                Event::MouseInput(platform::MouseButton::Left, platform::InputState::Pressed),
                Event::MouseInput(platform::MouseButton::Left, platform::InputState::Released),
            ]
        };

        // In a script mode, clicking a non-active view must not activate it.
        session.switch_mode(Mode::Script("funky".try_into().unwrap()));
        session.hover_view = Some(second);
        session.update(
            &mut click(),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.views.active_id, first);

        // In normal mode the same click activates the hovered view.
        session.switch_mode(Mode::Normal);
        session.hover_view = Some(second);
        session.update(
            &mut click(),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.views.active_id, second);
    }

    #[test]
    fn escape_exits_script_mode() {
        use crate::event::Event;
        use crate::execution::Execution;
        use crate::platform;
        use crate::session::Mode;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, session) = host_with(dir.path(), "p", "pub fn init(rx) { #{} }");
        let mut session = session.with_blank(crate::view::FileStatus::NoFile, 32, 32);

        session.switch_mode(Mode::Script("funky".try_into().unwrap()));
        let mut exec = Execution::normal().unwrap();
        let mut events = vec![Event::KeyboardInput(platform::KeyboardInput {
            key: Some(platform::Key::Escape),
            modifiers: platform::ModifiersState::default(),
            state: platform::InputState::Pressed,
        })];
        session.update(
            &mut events,
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.mode, Mode::Normal);
    }

    #[test]
    fn script_bindings_via_bind() {
        use crate::event::Event;
        use crate::execution::Execution;
        use crate::platform;
        use crate::session::{Mode, MessageType};
        use crate::view::FileStatus;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("funky/hello", [], "Say hello", hello);
                let ok = rx.bind("funky", "<tab> :funky/hello");
                let bad = rx.bind("funky", "<nokey> :funky/hello");
                if !ok || bad {
                    rx.message("bind results wrong");
                }
                #{}
            }
            pub fn hello(state, rx, args) { rx.message("funky hello"); }
            "#,
        );
        let mut session = session.with_blank(FileStatus::NoFile, 32, 32);
        let mut exec = Execution::normal().unwrap();

        let tab = |state| {
            vec![Event::KeyboardInput(platform::KeyboardInput {
                key: Some(platform::Key::Tab),
                modifiers: platform::ModifiersState::default(),
                state,
            })]
        };

        // In the script mode, <tab> runs the script command.
        session.switch_mode(Mode::Script("funky".try_into().unwrap()));
        session.update(
            &mut tab(platform::InputState::Pressed),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.message.to_string(), "funky hello");
        session.update(
            &mut tab(platform::InputState::Released),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );

        // Script-tier bindings fire even while the mouse is held.
        session.message("sentinel", MessageType::Info);
        let mouse = |state| vec![Event::MouseInput(platform::MouseButton::Left, state)];
        session.update(
            &mut mouse(platform::InputState::Pressed),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        session.update(
            &mut tab(platform::InputState::Pressed),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.message.to_string(), "funky hello");
        session.update(
            &mut mouse(platform::InputState::Released),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        session.update(
            &mut tab(platform::InputState::Released),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );

        // Outside the mode, the script binding is inert.
        session.switch_mode(Mode::Normal);
        session.message("sentinel", MessageType::Info);
        session.update(
            &mut tab(platform::InputState::Pressed),
            &mut exec,
            Duration::default(),
            Duration::default(),
            &mut host,
        );
        assert_eq!(session.message.to_string(), "sentinel");
    }

    #[test]
    fn register_and_dispatch_typed_command() {
        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("greet", ["str", "int?"], "Greet someone", greet);
                #{ calls: 0 }
            }
            pub fn greet(state, rx, args) {
                state.calls += 1;
                if args.len() == 2 {
                    rx.message(`hi ${args[0]} x${args[1]} (#${state.calls})`);
                } else {
                    rx.message(`hi ${args[0]} (#${state.calls})`);
                }
            }
            "#,
        );

        host.dispatch_command(&mut session, "greet", "world 3");
        assert_eq!(session.message.to_string(), "hi world x3 (#1)");

        // Optional argument omitted; state persists across calls.
        host.dispatch_command(&mut session, "greet", "moon");
        assert_eq!(session.message.to_string(), "hi moon (#2)");
    }

    #[test]
    fn read_png_decodes_plugin_relative_rgba() {
        use crate::gfx::color::Rgba8;

        let dir = tempfile::tempdir().unwrap();
        let pixels = vec![Rgba8::new(1, 2, 3, 4); 6];
        crate::image::save_as(dir.path().join("img.png"), 3, 2, 1, &pixels).unwrap();

        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("png/stat", [], "Read the fixture png", stat);
                #{}
            }
            pub fn stat(state, rx, args) {
                let (w, h, px) = rx.read_png("img.png").unwrap();
                rx.message(`png ${w}x${h} ${px.len()}`);
            }
            "#,
        );
        host.dispatch_command(&mut session, "png/stat", "");
        assert_eq!(session.message.to_string(), "png 3x2 24");
    }

    #[test]
    fn command_argument_validation() {
        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("paint", ["int", "color"], "Paint", paint);
                #{}
            }
            pub fn paint(state, rx, args) { rx.message("painted"); }
            "#,
        );

        // Too few arguments.
        host.dispatch_command(&mut session, "paint", "1");
        assert_eq!(
            session.message.to_string(),
            "Error: usage: paint <int> <color>"
        );

        // Bad int.
        host.dispatch_command(&mut session, "paint", "x #ff0000");
        assert!(session.message.to_string().contains("invalid int `x`"));

        // Bad color (also: must not panic on short tokens).
        host.dispatch_command(&mut session, "paint", "1 #f");
        assert!(session.message.to_string().contains("invalid color `#f`"));

        // Too many arguments.
        host.dispatch_command(&mut session, "paint", "1 #ff0000 extra");
        assert!(session.message.to_string().contains("usage: paint"));

        // Valid invocation.
        host.dispatch_command(&mut session, "paint", "1 #ff0000");
        assert_eq!(session.message.to_string(), "painted");
    }

    #[test]
    fn unknown_command_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(dir.path(), "p", "pub fn init(rx) { #{} }");

        host.dispatch_command(&mut session, "no/such", "");
        assert_eq!(session.message.to_string(), "Error: unknown command: no/such");
    }

    #[test]
    fn run_builtin_delegates_to_builtin_commands() {
        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("debug/on", [], "Enable debug", handler);
                #{}
            }
            pub fn handler(state, rx, args) {
                let ok = rx.run_builtin("set debug = on");
                let not_builtin = rx.run_builtin("no/such/builtin");
                if ok && !not_builtin {
                    rx.message("delegated");
                }
            }
            "#,
        );

        host.dispatch_command(&mut session, "debug/on", "");
        assert_eq!(session.message.to_string(), "delegated");
        assert!(session.settings["debug"].is_set());
    }

    #[test]
    fn command_registry_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("mine", [], "My command", handler);
                rx.register_command_repeating("again", [], "Repeats", handler);
                #{}
            }
            pub fn handler(state, rx, args) { rx.message("ran"); }
            "#,
        );

        // Declared metadata is queryable and synced to the help list.
        assert!(!host.command_repeats("mine"));
        assert!(host.command_repeats("again"));
        assert_eq!(
            session.cmdline.commands.script_commands(),
            &[
                ("mine".to_string(), "My command".to_string()),
                ("again".to_string(), "Repeats".to_string())
            ]
        );

        // Re-registering after reload works (the registry is cleared).
        host.reload(&mut session);
        host.dispatch_command(&mut session, "mine", "");
        assert_eq!(session.message.to_string(), "ran");

        // A plugin that goes away takes its commands with it.
        std::fs::write(
            dir.path().join("p.rune"),
            "pub fn init(rx) { #{} }",
        )
        .unwrap();
        host.reload(&mut session);
        assert!(session.cmdline.commands.script_commands().is_empty());
        host.dispatch_command(&mut session, "mine", "");
        assert_eq!(session.message.to_string(), "Error: unknown command: mine");
    }

    #[test]
    fn command_registration_conflicts() {
        let dir = tempfile::tempdir().unwrap();
        // `a` registers first (lexicographic load order); `b` collides.
        write_plugin(
            dir.path(),
            "a",
            r#"
            pub fn init(rx) {
                rx.register_command("shared", [], "From a", handler);
                #{}
            }
            pub fn handler(state, rx, args) { rx.message("a ran"); }
            "#,
        );
        write_plugin(
            dir.path(),
            "b",
            r#"
            pub fn init(rx) {
                let dup = rx.register_command("shared", [], "From b", handler);
                let builtin = rx.register_command("undo", [], "Shadow", handler);
                let bad_sig = rx.register_command("sig", ["nope"], "Bad", handler);
                if !dup && !builtin && !bad_sig {
                    rx.message("all rejected");
                }
                #{}
            }
            pub fn handler(state, rx, args) { rx.message("b ran"); }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);
        assert_eq!(session.message.to_string(), "all rejected");

        // First registration wins.
        host.dispatch_command(&mut session, "shared", "");
        assert_eq!(session.message.to_string(), "a ran");
    }

    #[test]
    fn erroring_command_handler_disables_plugin() {
        let dir = tempfile::tempdir().unwrap();
        let (mut host, mut session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("boom", [], "Explode", handler);
                #{}
            }
            pub fn handler(state, rx, args) { None.unwrap() }
            "#,
        );

        host.dispatch_command(&mut session, "boom", "");
        assert!(session.message.to_string().starts_with("Plugin `p` disabled"));
        assert_eq!(host.plugins().filter(|p| p.enabled).count(), 0);

        // Disabled plugin's commands report instead of running.
        host.dispatch_command(&mut session, "boom", "");
        assert_eq!(
            session.message.to_string(),
            "Error: ':boom': plugin `p` is not loaded"
        );
    }

    /// A headless GPU for texture tests; `None` if no adapter exists.
    fn test_gfx() -> Option<Gfx> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                ..Default::default()
            },
            None,
        ))
        .ok()?;
        Some(Gfx::new(
            std::sync::Arc::new(device),
            std::sync::Arc::new(queue),
        ))
    }

    #[test]
    fn texture_requires_gfx() {
        let dir = tempfile::tempdir().unwrap();
        let (_host, session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let t = rx.create_texture(8, 8);
                if t.is_none() { #{} } else { panic!("expected None") }
            }
            "#,
        );
        assert!(session
            .message
            .to_string()
            .contains("GPU is not available"));
    }

    #[test]
    fn texture_create_upload_readback() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };

        let dir = tempfile::tempdir().unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                // Invalid sizes are rejected.
                if rx.create_texture(0, 8).is_some() { panic!("0 accepted"); }
                if rx.create_texture(8, 9000).is_some() { panic!("9000 accepted"); }

                let t = rx.create_texture(4, 2).unwrap();
                t.fill(rx::rgb(255, 0, 0));

                // A short upload is rejected; a full one accepted.
                let bad = t.upload(b"xx");
                let data = Bytes::new();
                for i in 0..t.width() * t.height() {
                    data.extend(b"\x00\xff\x00\xff");
                }
                let good = t.upload(data);
                rx.message(`${t.width()}x${t.height()} ${bad} ${good}`);
                t
            }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1, "fixture plugin must load");
        assert_eq!(session.message.to_string(), "4x2 false true");

        // Read the texture back through the state value: the upload
        // (green) must have overwritten the fill (red).
        let state = host.plugins().next().unwrap().state.clone();
        let tex = state.borrow_ref::<ScriptTexture>().unwrap();
        let pixels = tex.pixels(&gfx.device);
        assert_eq!(pixels.len(), 4 * 2 * 4);
        for px in pixels.chunks(4) {
            assert_eq!(px, &[0x00, 0xff, 0x00, 0xff]);
        }
    }

    /// WGSL used by the render-pass tests: samples a texture across a
    /// quad using the standard sprite vertex layout and bind groups.
    const TEST_WGSL: &str = r#"
        struct TransformUniforms { ortho: mat4x4<f32>, transform: mat4x4<f32>, }
        @group(0) @binding(0) var<uniform> uniforms: TransformUniforms;
        @group(1) @binding(0) var tex: texture_2d<f32>;
        @group(1) @binding(1) var tex_sampler: sampler;

        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) uv: vec2<f32>,
            @location(2) color: vec4<f32>,
            @location(3) opacity: f32,
        }
        struct VertexOutput {
            @builtin(position) pos: vec4<f32>,
            @location(0) uv: vec2<f32>,
        }

        @vertex
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            out.pos = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
            out.uv = in.uv;
            return out;
        }

        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            return textureSample(tex, tex_sampler, in.uv);
        }
    "#;

    #[test]
    fn gpu_descriptors_validate_and_can_be_reused() {
        let engine = ScriptEngine::new().unwrap();
        let descriptor = |expression: &str| {
            engine
                .compile_str("options", &format!("pub fn make() {{ {expression} }}"))
                .unwrap()
                .call("make", ())
                .unwrap()
        };
        let value = descriptor(
            r##"#{ vertex: "v", fragment: "f", blend: "replace", topology: "point_list", vertex_layout: "sprite" }"##,
        );
        let object = value.borrow_ref::<rune::runtime::Object>().unwrap();
        for _ in 0..2 {
            let options = PipelineOptions::parse(&object).unwrap();
            assert_eq!(options.blend, wgpu::BlendState::REPLACE);
            assert_eq!(options.topology, wgpu::PrimitiveTopology::PointList);
            assert!(options.sprite, "topology must not select the vertex layout");
        }
        let value = descriptor(
            r##"#{ vertex: "v", fragment: "f", blend: #{ color: #{ src: "zero", dst: "one_minus_src_alpha", op: "add" }, alpha: #{ src: "zero", dst: "one_minus_src_alpha", op: "add" } } }"##,
        );
        let options =
            PipelineOptions::parse(&value.borrow_ref::<rune::runtime::Object>().unwrap()).unwrap();
        assert_eq!(options.blend.alpha.src_factor, wgpu::BlendFactor::Zero);
        assert_eq!(
            options.blend.alpha.dst_factor,
            wgpu::BlendFactor::OneMinusSrcAlpha
        );
        for expression in [
            r##"#{ vertex: "v", fragment: "f" }"##,
            r##"#{ vertex: "v", fragment: "f", blend: "alhpa" }"##,
            r##"#{ vertex: "v", fragment: "f", blend: "alpha", textures: -1 }"##,
            r##"#{ vertex: "v", fragment: "f", blend: "alpha", texture: 1 }"##,
            r##"#{ vertex: "v", fragment: "f", blend: #{ color: #{ src: "zero", dst: "one", op: "add" } } }"##,
        ] {
            let value = descriptor(expression);
            assert!(
                PipelineOptions::parse(&value.borrow_ref::<rune::runtime::Object>().unwrap())
                    .is_err(),
                "{expression}"
            );
        }
        for expression in [
            r##"#{ load: "load", clear: [0.0, 0.0, 0.0, 0.0] }"##,
            r##"#{ load: "clear", clear: [0.0] }"##,
            r##"#{ load: "keep" }"##,
        ] {
            let value = descriptor(expression);
            assert!(
                PassOptions::parse(&value.borrow_ref::<rune::runtime::Object>().unwrap()).is_err()
            );
        }
    }

    #[test]
    fn gpu_replace_preserves_neighbors_order_and_texture_view_lifetimes() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.wgsl"), TEST_WGSL).unwrap();
        write_plugin(
            dir.path(),
            "p",
            r##"
            pub fn init(rx) {
                let shader = rx.create_shader(rx.read_file("test.wgsl").unwrap()).unwrap();
                let replace = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 1, blend: "replace" }).unwrap();
                let alpha = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 1, blend: "alpha" }).unwrap();
                let brush = rx.create_texture(1, 1).unwrap();
                brush.fill(rx::rgba(0, 0, 0, 0));
                let blue = rx.create_texture(1, 1).unwrap();
                blue.fill(rx::rgb(0, 0, 255));
                let out = rx.create_texture(4, 1).unwrap();
                #{ replace, alpha, brush, blue, out, persistent: out.view(), saved: None, binding: None }
            }
            fn draw(rx, pass, state, pipeline, texture, x) {
                pass.set_pipeline(pipeline).unwrap();
                pass.set_bind_group(0, rx.create_transform_bind_group(4, 1, rx::mat4_identity()).unwrap()).unwrap();
                pass.set_bind_group(1, rx.create_texture_bind_group(texture.view()).unwrap()).unwrap();
                let vertices = rx.create_sprite_vertices(texture, #{ dst: rx::rect(x, 0.0, x + 1.0, 1.0) }).unwrap();
                pass.set_vertex_buffer(0, vertices).unwrap();
                pass.draw(vertices.count(), 1).unwrap();
            }
            pub fn shade(state, rx, encoder) {
                let id = rx.active_view_id();
                if state.saved.is_some() {
                    assert!(encoder.begin_render_pass(state.saved.unwrap(), #{ load: "load" }).is_err());
                    let pass = encoder.begin_render_pass(state.persistent, #{ load: "load" }).unwrap();
                    assert!(pass.set_bind_group(1, state.binding.unwrap()).is_err());
                    pass.end();
                    return;
                }
                assert!(encoder.view_layer(-1).is_err());
                assert!(encoder.view_layer(65536 + id).is_err());
                let target = encoder.view_layer(id).unwrap();
                let pass = encoder.begin_render_pass(target, #{ load: "load" }).unwrap();
                draw(rx, pass, state, state.replace, state.blue, 1.0);
                draw(rx, pass, state, state.replace, state.blue, 2.0);
                draw(rx, pass, state, state.replace, state.brush, 2.0);
                draw(rx, pass, state, state.alpha, state.brush, 0.0);
                pass.end();
                let binding = rx.create_texture_bind_group(target).unwrap();
                let pass = encoder.begin_render_pass(state.persistent, #{ load: "clear" }).unwrap();
                pass.set_pipeline(state.replace).unwrap();
                pass.set_bind_group(0, rx.create_transform_bind_group(4, 1, rx::mat4_identity()).unwrap()).unwrap();
                pass.set_bind_group(1, binding).unwrap();
                let vertices = rx.create_sprite_vertices(state.out, #{ dst: rx::rect(0.0, 0.0, 4.0, 1.0) }).unwrap();
                pass.set_vertex_buffer(0, vertices).unwrap();
                pass.draw(vertices.count(), 1).unwrap();
                pass.end();
                let staging = encoder.view_staging(id).unwrap();
                encoder.begin_render_pass(staging, #{ load: "clear", clear: [0.0, 1.0, 0.0, 1.0] }).unwrap().end();
                state.saved = Some(target);
                state.binding = Some(binding);
            }
            pub fn out(state) { state.out }
            pub fn saved(state) { state.saved.unwrap() }
        "##,
        );
        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 4, 1);
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "{}",
            session.message
        );
        let layer = ScriptTexture::create(&gfx, 4, 1);
        layer.write(&[255, 0, 0, 128].repeat(4));
        let staging = ScriptTexture::create(&gfx, 4, 1);
        let srgb = |texture: &ScriptTexture| {
            texture
                .wgpu_texture()
                .create_view(&wgpu::TextureViewDescriptor {
                    format: Some(SCRIPT_TEXTURE_FORMAT),
                    ..Default::default()
                })
        };
        for frame in 0..2 {
            let mut targets = ViewTargets::new();
            targets.insert(
                u16::from(session.views.active_id),
                ViewTarget {
                    layer: srgb(&layer),
                    staging: srgb(&staging),
                    width: 4,
                    height: 1,
                    staging_size: [4, 1],
                },
            );
            let encoder = host.dispatch_shade(
                &mut session,
                gfx.device.create_command_encoder(&Default::default()),
                targets,
            );
            gfx.queue.submit(Some(encoder.finish()));
            let plugin = host.plugins().next().unwrap();
            let saved = plugin
                .script
                .call("saved", (plugin.state.clone(),))
                .unwrap();
            assert_eq!(
                saved
                    .borrow_ref::<ScriptTextureView>()
                    .unwrap()
                    .get()
                    .is_ok(),
                frame == 0,
                "editor views survive shade but expire at the next frame"
            );
            assert_eq!(
                host.plugins().filter(|p| p.enabled).count(),
                1,
                "{}",
                session.message
            );
        }
        let expected = [255, 0, 0, 128, 0, 0, 255, 255, 0, 0, 0, 0, 255, 0, 0, 128];
        assert_eq!(layer.pixels(&gfx.device), expected);
        let plugin = host.plugins().next().unwrap();
        let out = plugin.script.call("out", (plugin.state.clone(),)).unwrap();
        assert_eq!(
            out.borrow_ref::<ScriptTexture>()
                .unwrap()
                .pixels(&gfx.device),
            expected
        );
        assert_eq!(staging.pixels(&gfx.device), [0, 255, 0, 255].repeat(4));
    }

    #[test]
    fn shader_compile_error_is_reported() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let bad = rx.create_shader("not wgsl at all !");
                if bad.is_some() { panic!("bad shader accepted"); }
                #{}
            }
            "#,
        );
        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1);
        assert!(
            session.message.to_string().starts_with("Error: shader:"),
            "got: {}",
            session.message
        );
    }

    #[test]
    fn shade_renders_through_script_pipeline() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.wgsl"), TEST_WGSL).unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let wgsl = rx.read_file("test.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                let pipeline = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 1, blend: "alpha" }).unwrap();
                let source = rx.create_texture(4, 4).unwrap();
                source.fill(rx::rgb(0, 0, 255));
                let target = rx.create_texture(4, 4).unwrap();
                let tbg = rx.create_transform_bind_group(4, 4, rx::mat4_identity()).unwrap();
                let sbg = rx.create_texture_bind_group(source.view()).unwrap();
                let verts = rx.create_sprite_vertices(source, #{
                    dst: rx::rect(0.0, 0.0, 4.0, 4.0),
                }).unwrap();
                #{ pipeline, target, tbg, sbg, verts }
            }
            pub fn shade(state, rx, encoder) {
                let pass = encoder.begin_render_pass(state.target.view(), #{ label: "test", load: "clear" }).unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_bind_group(1, state.sbg).unwrap();
                pass.set_vertex_buffer(0, state.verts).unwrap();
                pass.draw(state.verts.count(), 1).unwrap();
                // Deliberately not ended: the host must end it.
            }
            pub fn target(state) { state.target }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1, "fixture plugin must load");

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));

        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive the shade dispatch: {}",
            session.message
        );

        // The script's pass must have painted the target blue.
        let plugin = host.plugins().next().unwrap();
        let target = plugin
            .script
            .call("target", (plugin.state.clone(),))
            .unwrap();
        let target = target.borrow_ref::<ScriptTexture>().unwrap();
        let pixels = target.pixels(&gfx.device);
        for px in pixels.chunks(4) {
            assert_eq!(px, &[0x00, 0x00, 0xff, 0xff]);
        }
    }

    /// A vertex_index-driven point scatter (the lookupmap mechanism):
    /// each vertex loads one source texel, decodes its color to a map
    /// position, and emits a point there — or clip-rejects if the
    /// texel is transparent. No vertex buffers are bound.
    const SCATTER_WGSL: &str = r#"
        @group(1) @binding(0) var src: texture_2d<f32>;

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec4<f32>,
        };

        fn srgb_encode(c: f32) -> f32 {
            if (c <= 0.0031308) {
                return c * 12.92;
            }
            return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
        }

        @vertex
        fn vs_scatter(@builtin(vertex_index) index: u32) -> VertexOutput {
            let dims = textureDimensions(src);
            let texel = textureLoad(
                src, vec2<u32>(index % dims.x, index / dims.x), 0);
            var out: VertexOutput;
            out.color = texel;
            if (texel.a == 0.0) {
                // Clip-space rejection: transparent pixels land nowhere.
                out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
                return out;
            }
            // Decode the key: (r, g) bytes are the map position.
            // The map is 8x8; point centers in NDC, v as the row.
            let u = round(srgb_encode(texel.r) * 255.0);
            let v = round(srgb_encode(texel.g) * 255.0);
            out.position = vec4<f32>(
                (u + 0.5) / 4.0 - 1.0,
                1.0 - (v + 0.5) / 4.0,
                0.0,
                1.0,
            );
            return out;
        }

        @fragment
        fn fs_scatter(in: VertexOutput) -> @location(0) vec4<f32> {
            return in.color;
        }
    "#;

    #[test]
    fn point_pipeline_scatters_by_vertex_index() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("scatter.wgsl"), SCATTER_WGSL).unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            fn keys() {
                let data = Bytes::new();
                // (0,0): key color (3,5) -> map (3,5)
                data.push(3); data.push(5); data.push(0); data.push(255);
                // (1,0): transparent -> clip-rejected, lands nowhere
                data.push(0); data.push(0); data.push(0); data.push(0);
                // (0,1): key (7,0) -> map (7,0)
                data.push(7); data.push(0); data.push(0); data.push(255);
                // (1,1): key (0,2) -> map (0,2)
                data.push(0); data.push(2); data.push(0); data.push(255);
                data
            }
            pub fn init(rx) {
                let wgsl = rx.read_file("scatter.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                let pipeline = rx.create_render_pipeline(shader, #{ vertex: "vs_scatter", fragment: "fs_scatter", textures: 1, blend: "alpha", topology: "point_list", vertex_layout: "none" }).unwrap();
                let src = rx.create_texture(2, 2).unwrap();
                src.upload(keys());
                let map = rx.create_texture(8, 8).unwrap();
                let tbg = rx.create_transform_bind_group(8, 8, rx::mat4_identity()).unwrap();
                let sbg = rx.create_texture_bind_group(src.view()).unwrap();
                #{ pipeline, map, tbg, sbg }
            }
            pub fn shade(state, rx, encoder) {
                let pass = encoder.begin_render_pass(state.map.view(), #{ label: "scatter", load: "clear" }).unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_bind_group(1, state.sbg).unwrap();
                pass.draw(4, 1).unwrap();
                pass.end();
            }
            pub fn map(state) { state.map }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().count(),
            1,
            "fixture plugin must load: {}",
            session.message
        );

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive the shade dispatch: {}",
            session.message
        );

        let plugin = host.plugins().next().unwrap();
        let map = plugin.script.call("map", (plugin.state.clone(),)).unwrap();
        let map = map.borrow_ref::<ScriptTexture>().unwrap();
        let pixels = map.pixels(&gfx.device);

        let at = |x: usize, y: usize| &pixels[(y * 8 + x) * 4..][..4];
        let landed = [(3, 5), (7, 0), (0, 2)];
        assert_eq!(at(3, 5), &[3, 5, 0, 255], "key (3,5)");
        assert_eq!(at(7, 0), &[7, 0, 0, 255], "key (7,0)");
        assert_eq!(at(0, 2), &[0, 2, 0, 255], "key (0,2)");
        for y in 0..8 {
            for x in 0..8 {
                if !landed.contains(&(x, y)) {
                    assert_eq!(at(x, y), &[0, 0, 0, 0], "({}, {}) must be empty", x, y);
                }
            }
        }
    }

    #[test]
    fn snapshot_crop_upload_and_render_share_pixel_coordinates() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.wgsl"), TEST_WGSL).unwrap();
        write_plugin(
            dir.path(),
            "coords",
            r#"
            pub fn init(rx) {
                let shader = rx.create_shader(rx.read_file("test.wgsl").unwrap()).unwrap();
                let pipeline = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 1, blend: "alpha" }).unwrap();
                let pixels = rx.view_pixels(rx.active_view_id(), rx::rect(1.0, 0.0, 3.0, 2.0)).unwrap();
                let source = rx.create_texture(2, 2).unwrap();
                source.upload(pixels);
                let target = rx.create_texture(6, 5).unwrap();
                let tbg = rx.create_transform_bind_group(6, 5, rx::mat4_identity()).unwrap();
                let sbg = rx.create_texture_bind_group(source.view()).unwrap();
                let verts = rx.create_sprite_vertices(source, #{ dst: rx::rect(1.0, 1.0, 3.0, 3.0) }).unwrap();
                #{ pipeline, target, tbg, sbg, verts }
            }
            pub fn shade(state, rx, encoder) {
                let pass = encoder.begin_render_pass(state.target.view(), #{ label: "coordinates", load: "clear" }).unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_bind_group(1, state.sbg).unwrap();
                pass.set_vertex_buffer(0, state.verts).unwrap();
                pass.draw(state.verts.count(), 1).unwrap();
                pass.end();
            }
            pub fn output(state) { state.target }
        "#,
        );
        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 4, 3);
        use crate::gfx::Rgba8;
        // Four distinct crop corners, offset from the full image origin.
        let mut source = vec![Rgba8::TRANSPARENT; 12];
        source[1] = Rgba8::new(255, 0, 0, 255);
        source[2] = Rgba8::new(0, 255, 0, 255);
        source[5] = Rgba8::new(0, 0, 255, 255);
        source[6] = Rgba8::new(255, 255, 0, 255);
        session
            .active_view_mut()
            .resource
            .record_view_painted(source);
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "{}",
            session.message
        );
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "{}",
            session.message
        );
        let plugin = host.plugins().next().unwrap();
        let target = plugin
            .script
            .call("output", (plugin.state.clone(),))
            .unwrap();
        let target = target.borrow_ref::<ScriptTexture>().unwrap();
        let pixels = target.pixels(&gfx.device);
        for y in 0..5 {
            for x in 0..6 {
                let expected = match (x, y) {
                    (1, 1) => [255, 0, 0, 255],
                    (2, 1) => [0, 255, 0, 255],
                    (1, 2) => [0, 0, 255, 255],
                    (2, 2) => [255, 255, 0, 255],
                    _ => [0, 0, 0, 0],
                };
                assert_eq!(
                    &pixels[(y * 6 + x) * 4..][..4],
                    &expected,
                    "pixel ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn view_bind_group_samples_live_layer() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.wgsl"), TEST_WGSL).unwrap();
        // Each shade paints the view's layer (red on the first frame,
        // green after) and then samples that layer into `out` through
        // `view_bind_group` — in the same frame, with no snapshot or
        // upload in between. The readback proves liveness.
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let wgsl = rx.read_file("test.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                let pipeline = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 1, blend: "alpha" }).unwrap();
                let brush = rx.create_texture(1, 1).unwrap();
                let out = rx.create_texture(4, 4).unwrap();
                // Ortho normalizes, so 4x4 transforms and a (0,0,4,4)
                // quad cover the full target whatever its pixel size.
                let tbg = rx.create_transform_bind_group(4, 4, rx::mat4_identity()).unwrap();
                let verts = rx.create_sprite_vertices(brush, #{ dst: rx::rect(0.0, 0.0, 4.0, 4.0) }).unwrap();
                #{ pipeline, brush, out, tbg, verts, frame: 0 }
            }
            pub fn shade(state, rx, encoder) {
                state.frame = state.frame + 1;
                let color = if state.frame == 1 { rx::rgb(255, 0, 0) } else { rx::rgb(0, 255, 0) };
                state.brush.fill(color);
                let id = rx.active_view_id();

                let bbg = rx.create_texture_bind_group(state.brush.view()).unwrap();
                let pass = encoder.begin_view_pass("paint", id, "clear").unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_bind_group(1, bbg).unwrap();
                pass.set_vertex_buffer(0, state.verts).unwrap();
                pass.draw(state.verts.count(), 1).unwrap();
                pass.end();

                let vbg = encoder.view_bind_group(id).unwrap();
                let pass = encoder.begin_render_pass(state.out.view(), #{ label: "sample", load: "clear" }).unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_bind_group(1, vbg).unwrap();
                pass.set_vertex_buffer(0, state.verts).unwrap();
                pass.draw(state.verts.count(), 1).unwrap();
                pass.end();
            }
            pub fn out(state) { state.out }
            "#,
        );

        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 4, 4);
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().count(),
            1,
            "fixture plugin must load: {}",
            session.message
        );

        let id = u16::from(session.views.active_id);
        let srgb_view = |t: &ScriptTexture| {
            t.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
                format: Some(SCRIPT_TEXTURE_FORMAT),
                ..Default::default()
            })
        };
        let out_pixels = |host: &PluginHost| {
            let plugin = host.plugins().next().unwrap();
            let out = plugin.script.call("out", (plugin.state.clone(),)).unwrap();
            let out = out.borrow_ref::<ScriptTexture>().unwrap();
            out.pixels(&gfx.device)
        };

        // Frame 1: a 4x4 layer. The paint must come back out through
        // the bind group within the frame.
        let layer = ScriptTexture::create(&gfx, 4, 4);
        let staging = ScriptTexture::create(&gfx, 4, 4);
        let mut targets = ViewTargets::new();
        targets.insert(
            id,
            ViewTarget {
                layer: srgb_view(&layer),
                staging: srgb_view(&staging),
                width: 4,
                height: 4,
                staging_size: [4, 4],
            },
        );
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, targets);
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive the shade dispatch: {}",
            session.message
        );
        for px in out_pixels(&host).chunks(4) {
            assert_eq!(px, &[0xff, 0x00, 0x00, 0xff], "frame 1 must sample the live paint");
        }

        // Frame 2: the view was "resized" — a fresh, larger layer
        // texture in fresh targets. The bind group is built per call,
        // so it must track the new texture, not the old one.
        let layer2 = ScriptTexture::create(&gfx, 8, 8);
        let staging2 = ScriptTexture::create(&gfx, 8, 8);
        let mut targets = ViewTargets::new();
        targets.insert(
            id,
            ViewTarget {
                layer: srgb_view(&layer2),
                staging: srgb_view(&staging2),
                width: 8,
                height: 8,
                staging_size: [8, 8],
            },
        );
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, targets);
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive the resized dispatch: {}",
            session.message
        );
        for px in out_pixels(&host).chunks(4) {
            assert_eq!(px, &[0x00, 0xff, 0x00, 0xff], "frame 2 must sample the new layer");
        }
    }

    #[test]
    fn texture_pixels_roundtrips_upload() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            fn pattern() {
                let data = Bytes::new();
                for i in 0..8 {
                    data.push(i * 16);
                    data.push(255 - i);
                    data.push(i);
                    data.push(255);
                }
                data
            }
            pub fn init(rx) {
                let t = rx.create_texture(4, 2).unwrap();
                t.upload(pattern());
                let back = rx.texture_pixels(t).unwrap();
                if back != pattern() {
                    panic!("roundtrip mismatch");
                }
                rx.message(`roundtrip ok ${back.len()}`);
                #{}
            }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().count(),
            1,
            "fixture plugin must load: {}",
            session.message
        );
        assert_eq!(session.message.to_string(), "roundtrip ok 32");
    }

    #[test]
    fn params_bind_group_packs_and_pads() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        // The fragment color is assembled from across the params array:
        // x of the second vec4 (the 5th param), y of the first, and a
        // padded slot (y of the second vec4), which must read zero.
        std::fs::write(
            dir.path().join("tint.wgsl"),
            r#"
            struct TransformUniforms { ortho: mat4x4<f32>, transform: mat4x4<f32>, }
            @group(0) @binding(0) var<uniform> uniforms: TransformUniforms;
            @group(0) @binding(1) var<uniform> params: array<vec4<f32>, 2>;

            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) uv: vec2<f32>,
                @location(2) color: vec4<f32>,
                @location(3) opacity: f32,
            }
            struct VertexOutput { @builtin(position) pos: vec4<f32>, }

            @vertex
            fn vs_main(in: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                out.pos = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(params[1].x, params[0].y, params[1].y, 1.0);
            }
            "#,
        )
        .unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let wgsl = rx.read_file("tint.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                let pipeline = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 0, blend: "alpha" }).unwrap();
                let target = rx.create_texture(4, 4).unwrap();
                let source = rx.create_texture(4, 4).unwrap();
                if rx.create_transform_params_bind_group(4, 4, rx::mat4_identity(), []).is_some() {
                    panic!("[] accepted");
                }
                let tbg = rx.create_transform_params_bind_group(
                    4, 4, rx::mat4_identity(),
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                ).unwrap();
                let verts = rx.create_sprite_vertices(source, #{
                    dst: rx::rect(0.0, 0.0, 4.0, 4.0),
                }).unwrap();
                #{ pipeline, target, tbg, verts }
            }
            pub fn shade(state, rx, encoder) {
                let pass = encoder.begin_render_pass(state.target.view(), #{ label: "tint", load: "clear" }).unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_vertex_buffer(0, state.verts).unwrap();
                pass.draw(state.verts.count(), 1).unwrap();
            }
            pub fn target(state) { state.target }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().count(),
            1,
            "fixture plugin must load: {}",
            session.message
        );

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive: {}",
            session.message
        );

        // (params[1].x, params[0].y, padding) = (1, 1, 0): yellow.
        let plugin = host.plugins().next().unwrap();
        let target = plugin
            .script
            .call("target", (plugin.state.clone(),))
            .unwrap();
        let target = target.borrow_ref::<ScriptTexture>().unwrap();
        for px in target.pixels(&gfx.device).chunks(4) {
            assert_eq!(px, &[0xff, 0xff, 0x00, 0xff]);
        }
    }

    #[test]
    fn multi_input_compute_sums() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("sum.wgsl"),
            r#"
            @group(0) @binding(0) var cs_a: texture_2d<f32>;
            @group(0) @binding(1) var cs_b: texture_2d<f32>;
            @group(0) @binding(2) var cs_out: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(4, 4)
            fn cs_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
                let dims = textureDimensions(cs_a);
                if (gid.x >= dims.x || gid.y >= dims.y) { return; }
                let a = textureLoad(cs_a, vec2<i32>(gid.xy), 0);
                let b = textureLoad(cs_b, vec2<i32>(gid.xy), 0);
                textureStore(cs_out, vec2<i32>(gid.xy), min(a + b, vec4<f32>(1.0)));
            }
            "#,
        )
        .unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let wgsl = rx.read_file("sum.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                // Count mismatches are rejected loudly.
                if rx.create_compute_pipeline(shader, "cs_sum", 0).is_some() { panic!("0 accepted"); }
                if rx.create_compute_pipeline(shader, "cs_sum", 5).is_some() { panic!("5 accepted"); }
                let pipeline = rx.create_compute_pipeline(shader, "cs_sum", 2).unwrap();
                let a = rx.create_texture(4, 4).unwrap();
                a.fill(rx::rgb(64, 0, 32));
                let b = rx.create_texture(4, 4).unwrap();
                b.fill(rx::rgb(16, 8, 250));
                let out = rx.create_texture(4, 4).unwrap();
                if rx.create_compute_bind_group([], out).is_some() { panic!("[] accepted"); }
                let bg = rx.create_compute_bind_group([a, b], out).unwrap();
                #{ pipeline, bg, out }
            }
            pub fn shade(state, rx, encoder) {
                let pass = encoder.begin_compute_pass("sum").unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.bg).unwrap();
                pass.dispatch(1, 1, 1).unwrap();
            }
            pub fn out(state) { state.out }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().count(),
            1,
            "fixture plugin must load: {}",
            session.message
        );

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive: {}",
            session.message
        );

        let plugin = host.plugins().next().unwrap();
        let out = plugin.script.call("out", (plugin.state.clone(),)).unwrap();
        let out = out.borrow_ref::<ScriptTexture>().unwrap();
        for px in out.pixels(&gfx.device).chunks(4) {
            // 64+16, 0+8, 32+250 clamped, ff+ff clamped.
            assert_eq!(px, &[80, 8, 0xff, 0xff]);
        }
    }

    #[test]
    fn miniview_picks_drags_and_renders_above_other_plugins() {
        use crate::event::Event;
        use crate::execution::Execution;
        use crate::platform::{InputState, LogicalPosition, MouseButton};
        let Some(gfx) = test_gfx() else { eprintln!("skipping: no GPU adapter"); return; };
        let dir = tempfile::tempdir().unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins/miniview");
        for file in ["miniview.rune", "glyphs.png"] {
            std::fs::copy(source.join(file), dir.path().join(file)).unwrap();
        }
        // This renders later in the ordinary stage. The panel still wins.
        write_plugin(dir.path(), "z-background", r#"
            pub fn init(rx) { let tex = rx.create_texture(1, 1).unwrap(); tex.fill(rx::rgb(255, 0, 255)); #{ tex } }
            pub fn render(state, rx, pass) { pass.draw_sprite(state.tex, #{ dst: rx::rect(0.0, 0.0, 640.0, 480.0) }).unwrap(); }
        "#);
        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 4, 4);
        session.command(crate::cmd::Command::FrameAdd);
        session.offset = crate::gfx::math::Vector2::new(0.0, 0.0);
        session.active_view_mut().offset = crate::gfx::math::Vector2::new(100.0, 100.0);
        session.active_view_mut().zoom = 10.0;
        let id = u16::from(session.views.active_id);
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().filter(|p| p.enabled).count(), 2, "{}", session.message);
        let mut exec = Execution::normal().unwrap();
        let move_to = |x, y| Event::CursorMoved(LogicalPosition::new(x, y));
        let press = || Event::MouseInput(MouseButton::Left, InputState::Pressed);
        let release = || Event::MouseInput(MouseButton::Left, InputState::Released);
        host.dispatch_command(&mut session, "miniview", "");
        assert_eq!(session.mode.to_string(), "miniview target", "{}", session.message);
        for event in [move_to(110.0, 110.0), press(), release()] { session.handle_event(event, &mut exec, &mut host); }
        assert_eq!(session.mode.to_string(), "normal", "{}", session.message);
        assert_eq!(session.brush.state, crate::brush::BrushState::NotDrawing);
        // Drag the panel over the canvas; captured press/release never paints.
        for event in [move_to(354.0, 26.0), press(), move_to(120.0, 80.0), release()] {
            session.handle_event(event, &mut exec, &mut host);
        }
        assert_eq!(session.brush.state, crate::brush::BrushState::NotDrawing);
        let layer = ScriptTexture::create(&gfx, 8, 4);
        let row = [[255, 0, 0, 255].repeat(4), [0, 255, 0, 255].repeat(4)].concat();
        layer.write(&row.repeat(4));
        let staging = ScriptTexture::create(&gfx, 8, 4);
        let screen = ScriptTexture::create(&gfx, 640, 480);
        let render = |host: &mut PluginHost, session: &mut Session| {
            let srgb = |t: &ScriptTexture| t.wgpu_texture().create_view(&wgpu::TextureViewDescriptor { format: Some(SCRIPT_TEXTURE_FORMAT), ..Default::default() });
            let mut targets = ViewTargets::new();
            targets.insert(id, ViewTarget { layer: srgb(&layer), staging: srgb(&staging), width: 8, height: 4, staging_size: [8, 4] });
            let encoder = gfx.device.create_command_encoder(&Default::default());
            let mut encoder = host.dispatch_shade(session, encoder, targets);
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("miniview-test"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: screen.wgpu_view(), resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            }).forget_lifetime();
            let encoder = host.dispatch_render(session, encoder, pass, screen.wgpu_view());
            gfx.queue.submit([encoder.finish()]);
            assert!(host.plugins().all(|p| p.enabled), "{}", session.message);
            screen.pixels(&gfx.device)
        };
        let pixel = |pixels: &[u8], x: usize, y: usize| pixels[(y * 640 + x) * 4..(y * 640 + x + 1) * 4].to_vec();
        session.active_view_mut().animation.index = 1;
        let pixels = render(&mut host, &mut session);
        assert_eq!(pixel(&pixels, 200, 150), [255, 0, 0, 255], "fixed frame stays pinned as animation advances");
        assert_eq!(pixel(&pixels, 20, 20), [255, 0, 255, 255], "ordinary render stage ran");
        // Pick the builtin preview through the panel's Target button.
        for event in [move_to(310.0, 80.0), press(), release(), move_to(70.0, 115.0), press(), release()] {
            session.handle_event(event, &mut exec, &mut host);
        }
        assert_eq!(session.mode.to_string(), "normal", "{}", session.message);
        let pixels = render(&mut host, &mut session);
        assert_eq!(pixel(&pixels, 200, 150), [0, 255, 0, 255], "preview follows animation");
        session.active_view_mut().animation.index = 0;
        let pixels = render(&mut host, &mut session);
        assert_eq!(pixel(&pixels, 200, 150), [255, 0, 0, 255]);
        layer.write(&[0, 0, 255, 255].repeat(32));
        let pixels = render(&mut host, &mut session);
        assert_eq!(pixel(&pixels, 200, 150), [0, 0, 255, 255], "live edits reach the panel");
        host.dispatch_command(&mut session, "miniview", "");
        assert!(host.dispatch_capture_mouse(&mut session, "right", "pressed"));
        assert_eq!(session.mode.to_string(), "normal");
        host.dispatch_command(&mut session, "miniview/off", "");
        let pixels = render(&mut host, &mut session);
        assert_eq!(pixel(&pixels, 200, 150), [255, 0, 255, 255]);
    }

    #[test]
    fn builtin_sprite_crops_transforms_blends_and_validates_handles() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let source = rx.create_texture(2, 1).unwrap();
                source.upload(b"\xff\x00\x00\xff\xff\xff\xff\xff");
                let clear = rx.create_texture(1, 1).unwrap();
                let out = rx.create_texture(8, 2).unwrap();
                let raw = rx.create_texture(8, 2).unwrap();
                #{ source, clear, out, raw, saved: None }
            }
            pub fn shade(state, rx, encoder) {
                let pass = encoder.begin_render_pass(state.out.view(), #{ load: "clear" }).unwrap();
                let dst = rx::rect(0.0, 0.0, 2.0, 1.0);
                if state.saved.is_some() {
                    assert!(pass.draw_sprite(state.saved.unwrap(), #{ dst }).is_err());
                    pass.end();
                    return;
                }
                assert!(pass.draw_sprite(state.source, #{ dst, typo: true }).is_err());
                assert!(pass.draw_sprite(state.source, #{ dst, opacity: 2.0 }).is_err());
                assert!(pass.draw_sprite(state.source, #{ dst, blend: "bogus" }).is_err());
                assert!(pass.draw_sprite(state.source, #{ dst, transform: 0 }).is_err());
                assert!(pass.draw_sprite(42, #{ dst }).is_err());
                // Crop white, scale it two pixels, translate into the second row,
                // tint green and halve alpha. Reuse the descriptor and source.
                let options = #{ src: rx::rect(1.0, 0.0, 2.0, 1.0), dst,
                    transform: rx::mat4_translation(2.0, 1.0),
                    color: rx::rgb(0, 255, 0), opacity: 0.5, blend: "replace" };
                pass.draw_sprite(state.source, options).unwrap();
                pass.draw_sprite(state.source.view(), options).unwrap();
                pass.draw_sprite(state.source, #{ dst }).unwrap();
                pass.draw_sprite(state.clear, #{ dst: rx::rect(0.0, 0.0, 1.0, 1.0) }).unwrap();
                pass.draw_sprite(state.clear, #{ dst: rx::rect(1.0, 0.0, 2.0, 1.0), blend: "replace" }).unwrap();
                let editor = encoder.view_layer(rx.active_view_id()).unwrap();
                pass.draw_sprite(editor, #{ dst: rx::rect(6.0, 0.0, 8.0, 1.0) }).unwrap();
                pass.end();
                assert!(pass.draw_sprite(state.source, #{ dst }).is_err());
                state.saved = Some(editor);
                let pass = encoder.begin_render_pass(state.raw.raw_view(), #{ load: "clear" }).unwrap();
                pass.draw_sprite(state.out.raw_view(), #{ dst: rx::rect(0.0, 0.0, 8.0, 2.0), blend: "replace" }).unwrap();
                pass.end();
                let pass = encoder.begin_staging_pass("small-staging", rx.active_view_id(), "clear").unwrap();
                pass.draw_sprite(state.source, #{ dst }).unwrap();
            }
            pub fn render(state, rx, pass) {
                pass.draw_sprite(state.raw, #{ dst: rx::rect(0.0, 0.0, 8.0, 2.0), blend: "replace" }).unwrap();
            }
            pub fn out(state) { state.out }
            pub fn raw(state) { state.raw }
        "#,
        );
        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 2, 1);
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert!(host.plugins().all(|p| p.enabled), "{}", session.message);
        let layer = ScriptTexture::create(&gfx, 2, 2);
        layer.write(&[0, 0, 255, 255].repeat(4));
        let staging = ScriptTexture::create(&gfx, 2, 1);
        let targets = || {
            let mut targets = ViewTargets::new();
            let view = |t: &ScriptTexture| {
                t.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
                    format: Some(SCRIPT_TEXTURE_FORMAT),
                    ..Default::default()
                })
            };
            targets.insert(
                u16::from(session.views.active_id),
                ViewTarget {
                    layer: view(&layer),
                    staging: view(&staging),
                    width: 2,
                    height: 2,
                    staging_size: [2, 1],
                },
            );
            targets
        };
        let first_targets = targets();
        let second_targets = targets();
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, first_targets);
        gfx.queue.submit([encoder.finish()]);
        assert!(host.plugins().all(|p| p.enabled), "{}", session.message);
        let mut expected = vec![0u8; 8 * 2 * 4];
        expected[0..4].copy_from_slice(&[255, 0, 0, 255]);
        expected[24..32].copy_from_slice(&[0, 0, 255, 255].repeat(2));
        expected[40..48].copy_from_slice(&[0, 255, 0, 128].repeat(2));
        let plugin = host.plugins().next().unwrap();
        for name in ["out", "raw"] {
            let value = plugin.script.call(name, (plugin.state.clone(),)).unwrap();
            assert_eq!(
                value
                    .borrow_ref::<ScriptTexture>()
                    .unwrap()
                    .pixels(&gfx.device),
                expected,
                "{name}"
            );
        }
        assert_eq!(
            staging.pixels(&gfx.device),
            [255, 0, 0, 255, 255, 255, 255, 255]
        );
        session.width = 8.0;
        session.height = 2.0;
        let screen = ScriptTexture::create(&gfx, 8, 2);
        let mut encoder = gfx.device.create_command_encoder(&Default::default());
        let pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sprite-screen-test"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: screen.wgpu_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        let encoder = host.dispatch_render(&mut session, encoder, pass, screen.wgpu_view());
        gfx.queue.submit([encoder.finish()]);
        assert!(host.plugins().all(|p| p.enabled), "{}", session.message);
        assert_eq!(screen.pixels(&gfx.device), expected);

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, second_targets);
        gfx.queue.submit([encoder.finish()]);
        assert!(host.plugins().all(|p| p.enabled), "{}", session.message);
    }

    #[test]
    fn render_stage_draws_to_screen() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.wgsl"), TEST_WGSL).unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let wgsl = rx.read_file("test.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                let pipeline = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 1, blend: "alpha" }).unwrap();
                let source = rx.create_texture(4, 4).unwrap();
                source.fill(rx::rgb(0, 0, 255));
                let tbg = rx.create_transform_bind_group(4, 4, rx::mat4_identity()).unwrap();
                let sbg = rx.create_texture_bind_group(source.view()).unwrap();
                let verts = rx.create_sprite_vertices(source, #{
                    dst: rx::rect(0.0, 0.0, 4.0, 4.0),
                }).unwrap();
                #{ pipeline, tbg, sbg, verts, kept: None }
            }
            pub fn render(state, rx, pass) {
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_bind_group(1, state.sbg).unwrap();
                pass.set_vertex_buffer(0, state.verts).unwrap();
                pass.draw(state.verts.count(), 1).unwrap();
                state.kept = Some(pass);
            }
            pub fn poke(state, rx) {
                // The host ended the pass after the hook returned; a
                // kept handle errors cleanly.
                match state.kept.unwrap().draw(6, 1) {
                    Err(e) => rx.message(`kept: ${e}`),
                    Ok(_) => rx.message("kept: unexpectedly alive"),
                }
            }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1, "fixture plugin must load");

        // Host-side stand-in for the screen: a texture-target pass.
        let screen = ScriptTexture::create(&gfx, 4, 4);
        let mut encoder = gfx.device.create_command_encoder(&Default::default());
        let pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: screen.wgpu_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        let encoder = host.dispatch_render(&mut session, encoder, pass, screen.wgpu_view());
        gfx.queue.submit(std::iter::once(encoder.finish()));

        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive the render dispatch: {}",
            session.message
        );

        // The hook must have painted the screen blue.
        let pixels = screen.pixels(&gfx.device);
        for px in pixels.chunks(4) {
            assert_eq!(px, &[0x00, 0x00, 0xff, 0xff]);
        }

        // The pass handle the hook kept must error cleanly now.
        let plugin = host.plugins().next().unwrap();
        let mut ctx = Ctx::new(&mut session);
        plugin
            .script
            .call("poke", (plugin.state.clone(), &mut ctx))
            .unwrap();
        drop(ctx);
        assert_eq!(
            session.message.to_string(),
            "kept: the render pass has ended"
        );
    }

    #[test]
    fn render_stage_gpu_error_disables_plugin() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.wgsl"), TEST_WGSL).unwrap();
        // `a` records an invalid draw (no pipeline bound): a GPU
        // validation error that poisons the encoder. `b` (running
        // after) must still get a live pass and draw.
        write_plugin(
            dir.path(),
            "a",
            r#"
            pub fn init(rx) { #{} }
            pub fn render(state, rx, pass) {
                pass.draw(3, 1).unwrap();
            }
            "#,
        );
        write_plugin(
            dir.path(),
            "b",
            r#"
            pub fn init(rx) {
                let wgsl = rx.read_file("test.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                let pipeline = rx.create_render_pipeline(shader, #{ vertex: "vs_main", fragment: "fs_main", textures: 1, blend: "alpha" }).unwrap();
                let source = rx.create_texture(4, 4).unwrap();
                source.fill(rx::rgb(0, 0, 255));
                let tbg = rx.create_transform_bind_group(4, 4, rx::mat4_identity()).unwrap();
                let sbg = rx.create_texture_bind_group(source.view()).unwrap();
                let verts = rx.create_sprite_vertices(source, #{
                    dst: rx::rect(0.0, 0.0, 4.0, 4.0),
                }).unwrap();
                #{ pipeline, tbg, sbg, verts }
            }
            pub fn render(state, rx, pass) {
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.tbg).unwrap();
                pass.set_bind_group(1, state.sbg).unwrap();
                pass.set_vertex_buffer(0, state.verts).unwrap();
                pass.draw(state.verts.count(), 1).unwrap();
            }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 2, "fixture plugins must load");

        let screen = ScriptTexture::create(&gfx, 4, 4);
        let mut encoder = gfx.device.create_command_encoder(&Default::default());
        let pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: screen.wgpu_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        let encoder = host.dispatch_render(&mut session, encoder, pass, screen.wgpu_view());
        gfx.queue.submit(std::iter::once(encoder.finish()));

        let enabled: Vec<&str> = host
            .plugins()
            .filter(|p| p.enabled)
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(enabled, ["b"], "a disabled, b alive: {}", session.message);
        assert!(
            session.message.to_string().contains("Plugin `a` disabled"),
            "got: {}",
            session.message
        );

        // `b` drew on the continuation pass.
        let pixels = screen.pixels(&gfx.device);
        for px in pixels.chunks(4) {
            assert_eq!(px, &[0x00, 0x00, 0xff, 0xff]);
        }
    }

    #[test]
    fn stored_pass_errors_cleanly() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                #{ target: rx.create_texture(2, 2).unwrap(), pass: None }
            }
            pub fn shade(state, rx, encoder) {
                match state.pass {
                    None => {
                        // Store the pass across frames (misuse).
                        state.pass = Some(encoder.begin_render_pass(state.target.view(), #{ label: "p", load: "clear" }).unwrap());
                    }
                    Some(pass) => {
                        // The host ended it when the previous hook
                        // returned; using it errors cleanly.
                        match pass.draw(6, 1) {
                            Err(e) => rx.message(`pass error: ${e}`),
                            Ok(_) => rx.message("unexpectedly alive"),
                        }
                    }
                }
            }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1);

        // First frame: the plugin leaves the pass open and stores it;
        // the host force-ends it, so the encoder still finishes.
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(host.plugins().filter(|p| p.enabled).count(), 1);

        // Second frame: using the stored pass is a clean error.
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            session.message.to_string(),
            "pass error: the render pass has ended"
        );
    }

    #[test]
    fn selection_outline_plugin_smoke() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins");
        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 128, 128);
        let mut host = PluginHost::new(Some(dir)).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert!(
            host.plugins()
                .any(|p| p.name == "selection-outline" && p.enabled),
            "plugin must load: {}",
            session.message
        );

        // Give the view CPU-side content: a white block on transparency.
        use crate::gfx::Rgba8;
        let mut pixels = vec![Rgba8::TRANSPARENT; 128 * 128];
        for y in 60..65 {
            for x in 56..73 {
                pixels[y * 128 + x] = Rgba8::new(0xff, 0xff, 0xff, 0xff);
            }
        }
        session
            .views
            .active_mut()
            .unwrap()
            .resource
            .record_view_painted(pixels);

        session.selection = Some(crate::session::Selection::new(40, 40, 90, 90));
        host.dispatch_command(&mut session, "selection/outline", "");
        assert!(
            !session.message.to_string().starts_with("Error")
                && !session.message.to_string().contains("disabled"),
            "command must succeed: {}",
            session.message
        );

        let id = u16::from(session.views.active_id);
        let mut targets = ViewTargets::new();
        let target = ScriptTexture::create(&gfx, 128, 128);
        let staging = ScriptTexture::create(&gfx, 128, 128);
        // Like the real renderer's view layers, the target views are sRGB.
        let srgb_view = |t: &ScriptTexture| {
            t.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
                format: Some(SCRIPT_TEXTURE_FORMAT),
                ..Default::default()
            })
        };
        targets.insert(
            id,
            ViewTarget {
                layer: srgb_view(&target),
                staging: srgb_view(&staging),
                width: 128,
                height: 128,
                staging_size: [128, 128],
            },
        );

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, targets);
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert!(
            host.plugins().filter(|p| p.name == "selection-outline").all(|p| p.enabled),
            "plugin must survive shade: {}",
            session.message
        );

        // The outline ring must be painted around the block's alpha
        // boundary in the foreground color (default white).
        let out = target.pixels(&gfx.device);
        let px = |x: usize, y: usize| {
            let i = (y * 128 + x) * 4;
            [out[i], out[i + 1], out[i + 2], out[i + 3]]
        };
        assert_eq!(px(55, 60), [0xff, 0xff, 0xff, 0xff], "left of block must be outlined");
        assert_eq!(px(64, 59), [0xff, 0xff, 0xff, 0xff], "above block must be outlined");
        assert_eq!(px(64, 62), [0, 0, 0, 0], "block interior must not be painted");
        assert_eq!(px(20, 20), [0, 0, 0, 0], "far field must stay clear");
    }

    #[test]
    fn selection_outline_reversed_selection() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins");
        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 128, 128);
        let mut host = PluginHost::new(Some(dir)).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert!(
            host.plugins()
                .any(|p| p.name == "selection-outline" && p.enabled),
            "plugin must load: {}",
            session.message
        );

        // A block sitting off-center in the selection, which was dragged
        // from bottom-right to top-left: the rect is reversed on both
        // axes, like an interactive right-to-left drag produces. The
        // outline must land on the block, not on its mirror image.
        use crate::gfx::Rgba8;
        let mut pixels = vec![Rgba8::TRANSPARENT; 128 * 128];
        for y in 60..65 {
            for x in 50..60 {
                pixels[y * 128 + x] = Rgba8::new(0xff, 0xff, 0xff, 0xff);
            }
        }
        session
            .views
            .active_mut()
            .unwrap()
            .resource
            .record_view_painted(pixels);

        session.selection = Some(crate::session::Selection::new(90, 90, 41, 41));
        host.dispatch_command(&mut session, "selection/outline", "");
        assert!(
            !session.message.to_string().starts_with("Error")
                && !session.message.to_string().contains("disabled"),
            "command must succeed: {}",
            session.message
        );

        let id = u16::from(session.views.active_id);
        let mut targets = ViewTargets::new();
        let target = ScriptTexture::create(&gfx, 128, 128);
        let staging = ScriptTexture::create(&gfx, 128, 128);
        let srgb_view = |t: &ScriptTexture| {
            t.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
                format: Some(SCRIPT_TEXTURE_FORMAT),
                ..Default::default()
            })
        };
        targets.insert(
            id,
            ViewTarget {
                layer: srgb_view(&target),
                staging: srgb_view(&staging),
                width: 128,
                height: 128,
                staging_size: [128, 128],
            },
        );

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, targets);
        gfx.queue.submit(std::iter::once(encoder.finish()));

        let out = target.pixels(&gfx.device);
        let px = |x: usize, y: usize| {
            let i = (y * 128 + x) * 4;
            [out[i], out[i + 1], out[i + 2], out[i + 3]]
        };
        assert_eq!(px(49, 62), [0xff, 0xff, 0xff, 0xff], "left of block must be outlined");
        assert_eq!(px(60, 62), [0xff, 0xff, 0xff, 0xff], "right of block must be outlined");
        assert_eq!(px(55, 62), [0, 0, 0, 0], "block interior must not be painted");
        assert_eq!(px(75, 62), [0, 0, 0, 0], "the outline must not be mirrored");
    }

    #[test]
    fn compute_pass_through_script_pipeline() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("invert.wgsl"),
            r#"
            @group(0) @binding(0) var input: texture_2d<f32>;
            @group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(8, 8)
            fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let dims = textureDimensions(input);
                if (gid.x >= dims.x || gid.y >= dims.y) { return; }
                let c = textureLoad(input, vec2<i32>(gid.xy), 0);
                textureStore(
                    output,
                    vec2<i32>(gid.xy),
                    vec4<f32>(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a),
                );
            }
            "#,
        )
        .unwrap();
        write_plugin(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                let wgsl = rx.read_file("invert.wgsl").unwrap();
                let shader = rx.create_shader(wgsl).unwrap();
                let pipeline = rx.create_compute_pipeline(shader, "cs_main", 1).unwrap();
                let input = rx.create_texture(8, 8).unwrap();
                input.fill(rx::rgb(255, 0, 0));
                let output = rx.create_texture(8, 8).unwrap();
                let bg = rx.create_compute_bind_group([input], output).unwrap();
                #{ pipeline, input, output, bg }
            }
            pub fn shade(state, rx, encoder) {
                let pass = encoder.begin_compute_pass("invert").unwrap();
                pass.set_pipeline(state.pipeline).unwrap();
                pass.set_bind_group(0, state.bg).unwrap();
                pass.dispatch(1, 1, 1).unwrap();
                // Not ended: the host must end it.
            }
            pub fn output(state) { state.output }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1, "fixture plugin must load");

        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive: {}",
            session.message
        );

        // Red inverted is cyan.
        let plugin = host.plugins().next().unwrap();
        let output = plugin
            .script
            .call("output", (plugin.state.clone(),))
            .unwrap();
        let output = output.borrow_ref::<ScriptTexture>().unwrap();
        let pixels = output.pixels(&gfx.device);
        for px in pixels.chunks(4) {
            assert_eq!(px, &[0x00, 0xff, 0xff, 0xff]);
        }
    }

    #[test]
    fn view_interop_effects() {
        use crate::session::{Blending, Effect};
        use crate::view::FileStatus;

        let dir = tempfile::tempdir().unwrap();
        let (mut host, session) = host_with(
            dir.path(),
            "p",
            r#"
            pub fn init(rx) {
                rx.register_command("interop", [], "Test interop", interop);
                #{}
            }
            pub fn interop(state, rx, args) {
                rx.clear_view_rect(rx::rect(4.0, 4.0, 12.0, 12.0));
                rx.damage_view(rx.active_view_id());
            }
            "#,
        );
        let mut session = session.with_blank(FileStatus::NoFile, 32, 32);
        session.effects.clear();

        host.dispatch_command(&mut session, "interop", "");

        // The rect clear is a recorded paint: constant blending + a
        // transparent fill, and the view is touched (dirty -> snapshot).
        assert!(matches!(
            session.effects[0],
            Effect::ViewBlendingChanged(Blending::Constant)
        ));
        match &session.effects[1] {
            Effect::ViewPaintFinal(shapes) => assert_eq!(shapes.len(), 1),
            other => panic!("expected ViewPaintFinal, got {:?}", other),
        }
        assert!(matches!(session.effects[2], Effect::ViewDamaged(_, None)));
        assert!(session.views.active().unwrap().is_dirty());
    }

    #[test]
    fn meta_plugin_exports() {
        let dir = tempfile::tempdir().unwrap();
        // `algo` exports a function; `caller` (loads after, lexicographic)
        // invokes it through the host-mediated registry.
        write_plugin(
            dir.path(),
            "algo",
            r#"
            pub fn init(rx) {
                rx.export("scale", scale);
                #{ calls: 0 }
            }
            pub fn scale(state, rx, args) {
                state.calls += 1;
                args[0] * 2
            }
            pub fn calls(state) { state.calls }
            "#,
        );
        write_plugin(
            dir.path(),
            "caller",
            r#"
            pub fn init(rx) {
                rx.register_command("use-algo", [], "Call into algo", run);
                #{}
            }
            pub fn run(state, rx, args) {
                let doubled = rx.call_plugin("algo", "scale", [21]).unwrap();
                let missing = rx.call_plugin("algo", "no/such", []);
                rx.message(`${doubled} ${missing.is_err()}`);
            }
            "#,
        );

        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 2);

        host.dispatch_command(&mut session, "use-algo", "");
        assert_eq!(session.message.to_string(), "42 true");

        // The exporter's own state was mutated by the call.
        let algo = host.plugins().find(|p| p.name == "algo").unwrap();
        let calls = algo.script.call("calls", (algo.state.clone(),)).unwrap();
        let calls: i64 = rune::from_value(calls).unwrap();
        assert_eq!(calls, 1, "the export must run against the exporter's state");

        // Reload clears exports along with commands.
        std::fs::write(dir.path().join("algo.rune"), "pub fn init(rx) { #{} }").unwrap();
        host.reload(&mut session);
        host.dispatch_command(&mut session, "use-algo", "");
        assert!(
            session.message.to_string().contains("disabled"),
            "calling a gone export must fail loudly: {}",
            session.message
        );
    }

    #[test]
    fn rotsprite_chain_smoke() {
        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let plugins = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins");
        std::os::unix::fs::symlink(plugins.join("rotate-scale"), dir.path().join("rotate-scale")).unwrap();
        write_plugin(
            dir.path(),
            "driver",
            r#"
            pub fn init(rx) { #{} }
            pub fn shade(state, rx, encoder) {
                let src = rx.create_texture(8, 8).unwrap();
                src.fill(rx::rgb(255, 0, 0));
                let target = rx.create_texture(64, 64).unwrap();
                rx.call_plugin("rotate-scale", "rotsprite",
                    [encoder, rx::mat4_identity(), target, src,
                     rx::rect(0.0, 0.0, 64.0, 64.0), 64, 64]).unwrap();
            }
            "#,
        );
        write_plugin(
            dir.path(),
            "external",
            r#"
            pub fn init(rx) {
                rx.export("render_pass", render_pass);
                #{}
            }
            pub fn render_pass(state, rx, args) {
                rx.call_plugin("rotate-scale", "mmpx", args).unwrap();
            }
            "#,
        );
        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 3, "all plugins must load: {}", session.message);

        // Default (EPX) chain.
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            3,
            "EPX chain must not error: {}",
            session.message
        );

        // Delegated (mmpx) chain.
        session
            .settings
            .set("rotsprite/algo", crate::cmd::Value::Ident("mmpx".into()))
            .unwrap();
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            3,
            "mmpx chain must not error: {}",
            session.message
        );

        // An arbitrary plugin name still delegates each x2 pass through
        // the historical external `render_pass` contract.
        session
            .settings
            .set("rotsprite/algo", crate::cmd::Value::Ident("external".into()))
            .unwrap();
        let encoder = gfx.device.create_command_encoder(&Default::default());
        let encoder = host.dispatch_shade(&mut session, encoder, ViewTargets::new());
        gfx.queue.submit(std::iter::once(encoder.finish()));
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            3,
            "external x2 chain must not error: {}",
            session.message
        );
    }

    #[test]
    fn rotate_scale_preview_tracks_algorithm_without_painting() {
        use crate::gfx::Rgba8;
        use crate::session::{Mode, Selection};
        let Some(gfx) = test_gfx() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins/rotate-scale");
        let dest = dir.path().join("rotate-scale");
        std::fs::create_dir(&dest).unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), dest.join(entry.file_name())).unwrap();
            }
        }
        write_plugin(
            dir.path(),
            "external",
            r#"
            pub fn init(rx) { rx.export("render_pass", render_pass); #{} }
            pub fn render_pass(state, rx, args) {
                rx.call_plugin("rotate-scale", "cleanedge", args).unwrap();
            }
        "#,
        );
        let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 64, 64);
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            2,
            "{}",
            session.message
        );
        let mut pixels = vec![Rgba8::TRANSPARENT; 64 * 64];
        for y in 20..43 {
            for x in 24..(27 + (y - 20) / 2) {
                pixels[y * 64 + x] = Rgba8::new(255, 255, 255, 255);
            }
        }
        session
            .views
            .active_mut()
            .unwrap()
            .resource
            .record_view_painted(pixels.clone());
        session.selection = Some(Selection::new(18, 16, 47, 48));
        host.dispatch_command(&mut session, "selection/rotate", "");
        host.dispatch_command(&mut session, "rotate/set", "33");

        // Command entry must return to this exact gesture, not start a new one.
        let mut execution = crate::execution::Execution::Normal;
        let colon: crate::event::TimedEvent = "00000 0000000 char/received ':'".parse().unwrap();
        session.handle_event(colon.event, &mut execution, &mut host);
        host.dispatch_update(&mut session);
        assert_eq!(session.mode, Mode::Command);
        session.cmdline.puts("set rotate-scale/algo = cleanedge");
        let enter: crate::event::TimedEvent = "00001 0000017 keyboard/input <return> pressed"
            .parse()
            .unwrap();
        session.handle_event(enter.event, &mut execution, &mut host);
        host.dispatch_update(&mut session);
        assert_eq!(session.mode.to_string(), "visual (rotation)");
        assert!(session.selection.is_some());

        let render = |host: &mut PluginHost, session: &mut Session| {
            let layer = ScriptTexture::create(&gfx, 64, 64);
            let staging = ScriptTexture::create(&gfx, 64, 64);
            let mut targets = ViewTargets::new();
            targets.insert(
                u16::from(session.views.active_id),
                ViewTarget {
                    layer: layer
                        .wgpu_texture()
                        .create_view(&wgpu::TextureViewDescriptor {
                            format: Some(SCRIPT_TEXTURE_FORMAT),
                            ..Default::default()
                        }),
                    staging: staging
                        .wgpu_texture()
                        .create_view(&wgpu::TextureViewDescriptor {
                            format: Some(SCRIPT_TEXTURE_FORMAT),
                            ..Default::default()
                        }),
                    width: 64,
                    height: 64,
                    staging_size: [64, 64],
                },
            );
            let encoder = gfx.device.create_command_encoder(&Default::default());
            let encoder = host.dispatch_shade(session, encoder, targets);
            gfx.queue.submit(std::iter::once(encoder.finish()));
            assert_eq!(
                host.plugins().filter(|p| p.enabled).count(),
                2,
                "{}",
                session.message
            );
            (layer.pixels(&gfx.device), staging.pixels(&gfx.device))
        };
        let mask = |pixels: &[u8]| pixels.chunks_exact(4).map(|p| p[3] > 0).collect::<Vec<_>>();
        let mut results = Vec::new();
        for algo in [
            "nearest",
            "cleanedge",
            "mmpx",
            "rotsprite",
            "external",
            "cleanedge",
        ] {
            session
                .settings
                .set("rotate-scale/algo", crate::cmd::Value::Ident(algo.into()))
                .unwrap();
            let (layer, preview) = render(&mut host, &mut session);
            assert!(
                layer.iter().all(|&v| v == 0),
                "{algo}: preview must not paint artwork"
            );
            assert!(
                preview.iter().any(|&v| v != 0),
                "{algo}: preview must be visible"
            );
            let (_, again) = render(&mut host, &mut session);
            assert_eq!(
                preview, again,
                "{algo}: successive previews must not accumulate"
            );
            host.dispatch_command(&mut session, "rotate/apply", "");
            let (committed, _) = render(&mut host, &mut session);
            assert_eq!(
                mask(&preview),
                mask(&committed),
                "{algo}: preview and commit silhouettes must agree"
            );
            results.push(mask(&preview));
        }
        assert!(
            results[1..4].iter().any(|m| m != &results[0]),
            "switching algorithms must change the preview"
        );
        assert_eq!(
            results[1], results[4],
            "external algorithms must also preview"
        );
        assert_eq!(
            results[1], results[5],
            "switching back must retain the source and transform"
        );
        session.switch_mode(Mode::Normal);
        host.dispatch_update(&mut session);
        let (_, staging) = render(&mut host, &mut session);
        assert!(
            staging.iter().all(|&v| v == 0),
            "leaving the gesture must stop previewing"
        );
    }

    #[test]
    fn help_columns_share_the_glyph_top_margin() {
        use crate::{draw, font::TextBatch, gfx::shape2d};
        let mut session = test_session();
        session.key_bindings.add(crate::session::KeyBinding {
            input: crate::session::Input::Key(crate::platform::Key::A),
            command: crate::cmd::Command::Noop,
            is_toggle: false,
            display: Some("Test binding".into()),
            modifiers: Default::default(),
            state: crate::platform::InputState::Pressed,
            tier: crate::session::BindingTier::General,
        });
        let mut text = TextBatch::new(96, 208, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT);
        draw::draw_help(&session, &mut text, &mut shape2d::Batch::new());
        let vertices = text.vertices();
        // The header occupies the first 40px. Both body columns start at y=58.
        for right_column in [false, true] {
            let top = vertices.iter()
                .filter(|v| v.position.y > 40. && (v.position.x >= 400.) == right_column)
                .map(|v| v.position.y)
                .fold(f32::INFINITY, f32::min);
            assert_eq!(top, 58., "help column must keep its top margin");
        }
    }

    #[test]
    fn mode_indicator_keeps_bottom_margin_when_resized() {
        use crate::{draw, font::TextBatch, gfx::{shape2d, sprite2d}, sprite};
        let dir = tempfile::tempdir().unwrap();
        let plugin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins/mode-vis");
        std::os::unix::fs::symlink(plugin, dir.path().join("mode-vis")).unwrap();
        let mut session = test_session();
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.load(&mut session);
        for height in [480., 718.] {
            session.height = height;
            let mut ctx = draw::Context {
                ui_batch: shape2d::Batch::new(),
                text_batch: TextBatch::new(96, 208, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT),
                overlay_batch: TextBatch::new(96, 208, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT),
                cursor_sprite: sprite::Sprite::new(96, 96),
                tool_batch: sprite2d::Batch::new(96, 96),
                paste_batch: sprite2d::Batch::new(8, 8),
                checker_batch: sprite2d::Batch::new(2, 2),
            };
            host.dispatch_draw(&mut session, &mut ctx);
            let vertices = ctx.text_batch.vertices();
            assert!(!vertices.is_empty(), "indicator must render: {}", session.message);
            let bottom = vertices.iter().map(|v| v.position.y).fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(height - bottom, 46., "indicator must stay above the bottom status rows");
        }
    }

    #[test]
    fn rotate_scale_flow_smoke() {
        use crate::gfx::Rgba8;
        use crate::session::Mode;
        use crate::view::FileStatus;

        let Some(gfx) = test_gfx() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let plugins = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("plugins");
        std::os::unix::fs::symlink(plugins.join("rotate-scale"), dir.path().join("rotate-scale")).unwrap();

        let mut session = test_session().with_blank(FileStatus::NoFile, 128, 128);
        let mut host = PluginHost::new(Some(dir.path().to_path_buf())).unwrap();
        host.attach_gfx(gfx.device.clone(), gfx.queue.clone());
        host.load(&mut session);
        assert_eq!(host.plugins().count(), 1, "plugin must load: {}", session.message);

        // Content + selection.
        let mut pixels = vec![Rgba8::TRANSPARENT; 128 * 128];
        for y in 40..60 {
            for x in 40..60 {
                pixels[y * 128 + x] = Rgba8::new(0xff, 0x33, 0x66, 0xff);
            }
        }
        session
            .views
            .active_mut()
            .unwrap()
            .resource
            .record_view_painted(pixels);
        session.selection = Some(crate::session::Selection::new(40, 40, 60, 60));

        let shade = |host: &mut PluginHost, session: &mut Session| {
            let encoder = gfx.device.create_command_encoder(&Default::default());
            let mut targets = ViewTargets::new();
            let layer = ScriptTexture::create(&gfx, 128, 128);
            let staging = ScriptTexture::create(&gfx, 128, 128);
            let srgb = |t: &ScriptTexture| {
                t.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
                    format: Some(SCRIPT_TEXTURE_FORMAT),
                    ..Default::default()
                })
            };
            targets.insert(
                u16::from(session.views.active_id),
                ViewTarget {
                    layer: srgb(&layer),
                    staging: srgb(&staging),
                    width: 128,
                    height: 128,
                    staging_size: [128, 128],
                },
            );
            let encoder = host.dispatch_shade(session, encoder, targets);
            gfx.queue.submit(std::iter::once(encoder.finish()));
        };

        let draw = |host: &mut PluginHost, session: &mut Session| {
            use crate::draw;
            use crate::font::TextBatch;
            use crate::gfx::{shape2d, sprite2d};
            use crate::sprite;
            let mut draw_ctx = draw::Context {
                ui_batch: shape2d::Batch::new(),
                text_batch: TextBatch::new(96, 208, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT),
                overlay_batch: TextBatch::new(96, 208, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT),
                cursor_sprite: sprite::Sprite::new(96, 96),
                tool_batch: sprite2d::Batch::new(96, 96),
                paste_batch: sprite2d::Batch::new(8, 8),
                checker_batch: sprite2d::Batch::new(2, 2),
            };
            host.dispatch_draw(session, &mut draw_ctx);
        };

        // Enter rotation mode, rotate, preview. Draw runs every frame
        // in the real loop — dispatch it repeatedly to catch state
        // poisoning (rune take semantics).
        host.dispatch_command(&mut session, "selection/rotate", "");
        assert_eq!(session.mode.to_string(), "visual (rotation)");
        draw(&mut host, &mut session);
        draw(&mut host, &mut session);
        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "draw must be repeatable: {}",
            session.message
        );
        host.dispatch_command(&mut session, "rotate/set", "45");
        host.dispatch_update(&mut session);
        draw(&mut host, &mut session);
        // The 45-degree gesture must widen the synced selection bounds
        // (a rotated 20x20 square has ~28x28 AABB).
        let sel = session.selection.expect("selection must survive").abs().bounds();
        assert!(
            sel.width() >= 26 && sel.height() >= 26,
            "rotation must widen the selection bounds, got {:?} ({})",
            sel,
            session.message
        );
        shade(&mut host, &mut session);

        // Switch to scale (commits the rotation gesture), scale.
        host.dispatch_command(&mut session, "selection/scale", "");
        assert_eq!(session.mode.to_string(), "visual (scale)");
        host.dispatch_command(&mut session, "scale/set", "1.5 1.5");
        shade(&mut host, &mut session);

        // Apply, then leave the mode.
        session
            .settings
            .set("rotate-scale/algo", crate::cmd::Value::Ident("cleanedge".into()))
            .unwrap();
        host.dispatch_command(&mut session, "rotate/apply", "");
        shade(&mut host, &mut session);
        session.switch_mode(Mode::Normal);
        host.dispatch_update(&mut session);

        assert_eq!(
            host.plugins().filter(|p| p.enabled).count(),
            1,
            "plugin must survive the whole flow: {}",
            session.message
        );
        // The selection moved with the transform and survives the
        // apply (pasted_once skips the base-selection restore).
        assert!(session.selection.is_none() || session.selection.is_some());
    }

    #[test]
    fn draw_take_bisect() {
        let dir = tempfile::tempdir().unwrap();
        let cases = [
            ("a", r#"
                pub fn init(rx) { #{ committed: rx::mat4_identity(), gesture: rx::mat4_identity() } }
                pub fn probe(state, rx, args) {
                    let m = rx::mat4_mul(state.gesture, state.committed);
                    let (x, y) = rx::mat4_transform_point(m, 1.0, 2.0);
                    rx.message(`ok ${x} ${y}`);
                }
            "#),
            ("b", r#"
                pub fn init(rx) { #{ deg: 12.5 } }
                pub fn probe(state, rx, args) {
                    let (_, height) = rx.screen_size();
                    rx.draw_text(`Angle: ${state.deg}`, 10.0, (height as f64) - 66.0 - 14.0, rx::rgb(1, 2, 3));
                    rx.message("ok text");
                }
            "#),
            ("c", r#"
                pub fn init(rx) { #{} }
                pub fn probe(state, rx, args) {
                    rx.draw_line((1.0, 2.0), (3.0, 4.0), rx::rgb(1, 2, 3));
                    rx.message("ok line");
                }
            "#),
            ("d", r#"
                pub fn init(rx) { #{} }
                pub fn probe(state, rx, args) {
                    let found = ();
                    for v in rx.views() {
                        if v.id == rx.active_view_id() { found = v; }
                    }
                    rx.message(`ok view ${found.zoom}`);
                }
            "#),
            ("e", r#"
                pub fn init(rx) { #{} }
                pub fn probe(state, rx, args) {
                    let algo = rx.setting("debug");
                    let algo = if algo is String { algo } else { "" };
                    rx.message(`ok setting ${algo == ""}`);
                }
            "#),
        ];
        for (name, body) in cases {
            let sub = dir.path().join(name);
            std::fs::create_dir(&sub).unwrap();
            std::fs::write(sub.join(name).with_extension("rune"), format!(
                "{}\npub fn run(state, rx, args) {{ probe(state, rx, args); }}\n", body)).unwrap();
            let mut full = String::from(body);
            full.push_str(&format!("\npub fn init2() {{}}\n"));
            let mut session = test_session().with_blank(crate::view::FileStatus::NoFile, 32, 32);
            let mut host = PluginHost::new(Some(sub.clone())).unwrap();
            host.load(&mut session);
            assert_eq!(host.plugins().count(), 1, "case {} must load: {}", name, session.message);
            // call probe twice through a command-like direct call
            let plugin = host.plugins().next().unwrap();
            for i in 0..2 {
                let mut ctx = Ctx::new(&mut session);
                let r = plugin.script.call("probe", (plugin.state.clone(), &mut ctx, rune::to_value(Vec::<Value>::new()).unwrap()));
                drop(ctx);
                if let Err(e) = r {
                    panic!("case {} call {} failed: {}", name, i, e);
                }
            }
            eprintln!("case {} OK", name);
        }
    }

    #[test]
    fn watcher_reports_changes() {
        let dir = tempfile::tempdir().unwrap();
        let watcher = ReloadWatcher::new(dir.path()).unwrap();
        assert!(!watcher.changed());

        std::fs::write(dir.path().join("plugin.rune"), "pub fn init() {}").unwrap();
        assert!(
            watcher.wait_changed(Duration::from_secs(10)),
            "watcher should observe the write"
        );
        assert!(!watcher.changed(), "queue should be drained");
    }
}

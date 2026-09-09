//! wgpu-based renderer for rx.

use crate::cmd::Axis;
use crate::draw;
use crate::execution::Execution;
use crate::font::TextBatch;
use crate::platform::{self, LogicalSize};
use crate::renderer;
use crate::session::{self, Blending, Effect, Session};
use crate::sprite;
use crate::util;
use crate::view::resource::ViewResource;
use crate::view::{View, ViewId, ViewOp};
use crate::{data, data::Assets, image};

use crate::gfx::{shape2d, sprite2d, Origin, Rgba, Rgba8, ZDepth};
use crate::gfx::{Matrix4, Rect, Repeat, Vector2};

use bytemuck::{Pod, Zeroable};
#[cfg(feature = "glfw")]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::io;
use std::time;

type M44 = [[f32; 4]; 4];

/// Project pixel coordinates into WebGPU NDC. TopLeft maps pixel (0, 0)
/// to NDC (-1, +1): framebuffer rows and application coordinates are y-down.
fn ortho_wgpu(w: u32, h: u32, origin: Origin) -> Matrix4<f32> {
    Matrix4::ortho(w, h, origin)
}

/// Vertex for sprite rendering (text, sprites, views).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Sprite2dVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [u8; 4],
    pub opacity: f32,
}

/// Vertex for shape rendering (UI, brush strokes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Shape2dVertex {
    pub position: [f32; 3],
    pub angle: f32,
    pub center: [f32; 2],
    pub color: [u8; 4],
}

/// Vertex for cursor rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Cursor2dVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// Uniform buffer for transform matrices.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TransformUniforms {
    ortho: M44,
    transform: M44,
}

/// Uniform buffer for cursor shader.
/// Note: Must match WGSL std140 layout - vec3 requires 16-byte alignment and size
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CursorUniforms {
    ortho: M44,         // 64 bytes
    scale: f32,         // 4 bytes
    _padding: [f32; 7], // 28 bytes to reach 96 total (WGSL alignment)
}

/// Render texture (like a framebuffer). Used for both render targets and source textures (font, cursors, etc.).
struct Texture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: [u32; 2],
    format: wgpu::TextureFormat,
}

impl Texture {
    fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            size: [width, height],
            format,
        }
    }

    /// Create a texture with optional initial pixel data (e.g. font atlas, cursors, checker).
    fn new_with_data(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        data: Option<&[u8]>,
    ) -> Self {
        let tex = Self::new(device, width, height, format);
        if let Some(data) = data {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        tex
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        *self = Self::new(device, width, height, format);
    }

    /// Resize keeping current format (e.g. for paste texture).
    fn resize_same_format(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.resize(device, width, height, self.format);
    }

    #[allow(dead_code)]
    fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_layer"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }

    fn upload(&self, queue: &wgpu::Queue, texels: &[u8]) {
        let [w, h] = self.size;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_part(&self, queue: &wgpu::Queue, offset: [u32; 2], size: [u32; 2], texels: &[u8]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * size[0]),
                rows_per_image: Some(size[1]),
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
    }

    /// Read back pixel data from the target (blocking). Creates its own encoder and submit.
    fn pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Rgba8> {
        let [w, h] = self.size;
        let bytes_per_row = (4 * w + 255) & !255;
        let buffer_size = (bytes_per_row * h) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
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
        queue.submit(std::iter::once(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            let row_start = (y * bytes_per_row) as usize;
            for x in 0..w {
                let offset = row_start + (x * 4) as usize;
                pixels.push(Rgba8::new(
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ));
            }
        }
        pixels
    }
}

/// Per-layer data for a view.
struct LayerData {
    texture: Texture,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

/// Build a static vertex buffer from a sprite batch.
fn sprite_vertex_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    batch: &sprite2d::Batch,
    label: &str,
) -> (wgpu::Buffer, u32) {
    let verts: Vec<Sprite2dVertex> = batch
        .vertices()
        .iter()
        .map(|v| Sprite2dVertex {
            position: [v.position.x, v.position.y, v.position.z],
            uv: [v.uv.x, v.uv.y],
            color: [v.color.r, v.color.g, v.color.b, v.color.a],
            opacity: v.opacity,
        })
        .collect();

    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (verts.len() * std::mem::size_of::<Sprite2dVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&verts));

    (vertex_buffer, verts.len() as u32)
}

/// One quad per layer strip, all mapped onto the display footprint,
/// with per-strip opacity. The bottom layer comes first so upper layers
/// alpha-blend over it. Batch src rects are y-down texture coordinates:
/// strip n covers rows `n*h .. (n+1)*h`.
fn strip_batch(w: u32, h: u32, sheet_h: u32, alphas: &[f32]) -> sprite2d::Batch {
    let mut batch = sprite2d::Batch::new(w, sheet_h);
    for (n, alpha) in alphas.iter().enumerate() {
        let y2 = ((n as u32 + 1) * h) as f32;
        batch.add(
            Rect::new(0., y2 - h as f32, w as f32, y2),
            Rect::origin(w as f32, h as f32),
            ZDepth::default(),
            Rgba::TRANSPARENT,
            *alpha,
            Repeat::default(),
        );
    }
    batch
}

impl LayerData {
    /// `h` is the display (strip) height; `sheet_h` is the texture
    /// height, one strip per layer.
    fn new(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        sheet_h: u32,
        pixels: Option<&[Rgba8]>,
        queue: &wgpu::Queue,
    ) -> Self {
        let texture = Texture::new(device, w, sheet_h, wgpu::TextureFormat::Rgba8UnormSrgb);

        debug_assert!(sheet_h % h == 0, "the sheet is a whole number of strips");
        let nlayers = (sheet_h / h) as usize;

        let batch = strip_batch(w, h, sheet_h, &vec![1.; nlayers]);
        let (vertex_buffer, vertex_count) =
            sprite_vertex_buffer(device, queue, &batch, "layer_vertex_buffer");

        // Upload initial pixels if provided
        if let Some(pixels) = pixels {
            let aligned = util::align_u8(pixels);
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                aligned,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(sheet_h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: sheet_h,
                    depth_or_array_layers: 1,
                },
            );
        }

        Self {
            texture,
            vertex_buffer,
            vertex_count,
        }
    }

    #[allow(dead_code)]
    fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        self.texture.clear(encoder);
    }

    fn upload(&self, queue: &wgpu::Queue, texels: &[u8]) {
        self.texture.upload(queue, texels);
    }

    fn upload_part(&self, queue: &wgpu::Queue, offset: [u32; 2], size: [u32; 2], texels: &[u8]) {
        self.texture.upload_part(queue, offset, size, texels);
    }

    /// Snapshot of the layer pixels (blocking readback). Matches the GL `pixels()` API.
    fn pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Rgba8> {
        self.texture.pixels(device, queue)
    }
}

/// Per-view rendering data.
struct ViewData {
    layer: LayerData,
    staging_texture: Texture,
    staging_vertex_buffer: wgpu::Buffer,
    staging_vertex_count: u32,
    /// The per-strip opacities baked into the layer vertex buffer
    /// (visibility, opacity, dim — see `update_view_layer_attrs`).
    strip_alphas: Vec<f32>,
    anim_vertex_buffer: Option<wgpu::Buffer>,
    anim_vertex_count: u32,
    layer_vertex_buffer: Option<wgpu::Buffer>,
    layer_vertex_count: u32,
}

impl ViewData {
    /// `h` is the *display* height (one strip): the staging texture
    /// covers the workspace footprint. `sheet_h` is the *sheet* height
    /// (one strip per layer): the layer texture holds all strips. The
    /// two are equal for single-layer views.
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        sheet_h: u32,
        pixels: Option<&[Rgba8]>,
    ) -> Self {
        let staging_texture = Texture::new(device, w, h, wgpu::TextureFormat::Rgba8UnormSrgb);
        let layer = LayerData::new(device, w, h, sheet_h, pixels, queue);

        // The staging texture maps 1:1 onto the display footprint.
        let staging_batch = sprite2d::Batch::singleton(
            w,
            h,
            Rect::origin(w as f32, h as f32),
            Rect::origin(w as f32, h as f32),
            ZDepth::default(),
            Rgba::TRANSPARENT,
            1.,
            Repeat::default(),
        );
        let (staging_vertex_buffer, staging_vertex_count) =
            sprite_vertex_buffer(device, queue, &staging_batch, "staging_vertex_buffer");

        let nlayers = (sheet_h / h) as usize;
        Self {
            layer,
            staging_texture,
            staging_vertex_buffer,
            staging_vertex_count,
            strip_alphas: vec![1.; nlayers],
            anim_vertex_buffer: None,
            anim_vertex_count: 0,
            layer_vertex_buffer: None,
            layer_vertex_count: 0,
        }
    }
}

/// The wgpu renderer.
pub struct Renderer {
    pub win_size: LogicalSize,

    // Core wgpu state. Arc'd so handles can be shared with the script
    // host (wgpu 23 doesn't impl Clone on these; 24+ does).
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    /// `None` when running surface-less (dummy platform, e.g. headless tests);
    /// everything still renders to `screen_texture`, only presentation is skipped.
    surface: Option<wgpu::Surface<'static>>,
    surface_config: wgpu::SurfaceConfiguration,

    // Draw context (same as GL version)
    draw_ctx: draw::Context,
    scale_factor: f64,
    scale: f64,
    blending: Blending,

    // Render textures
    screen_texture: Texture,

    // Batches
    staging_batch: shape2d::Batch,
    final_batch: shape2d::Batch,

    // Textures
    font: Texture,
    cursors: Texture,
    checker: Texture,
    paste: Texture,

    // Sampler
    sampler: wgpu::Sampler,

    // Pipelines
    sprite_pipeline: wgpu::RenderPipeline,
    /// Surface-format twin of `sprite_pipeline`, for the debug/replay
    /// overlay text drawn into the present pass (the swapchain) rather
    /// than the Rgba8 `screen_texture`.
    overlay_pipeline: wgpu::RenderPipeline,
    shape_pipeline: wgpu::RenderPipeline,
    shape_replace_pipeline: wgpu::RenderPipeline,
    cursor_pipeline: wgpu::RenderPipeline,
    screen_pipeline: wgpu::RenderPipeline,

    // Bind group layouts
    transform_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    screen_bind_group_layout: wgpu::BindGroupLayout,
    cursor_bind_group_layout: wgpu::BindGroupLayout,
    cursor_texture_bind_group_layout: wgpu::BindGroupLayout,

    // Uniform buffers
    transform_buffer: wgpu::Buffer,
    cursor_uniform_buffer: wgpu::Buffer,

    // Per-view data
    view_data: BTreeMap<ViewId, ViewData>,

    // Paste buffer for yank/paste operations
    paste_pixels: Vec<Rgba8>,
    paste_size: (u32, u32),

    // Paste outputs for final pass rendering (like GL's paste_outputs)
    paste_outputs: Vec<(wgpu::Buffer, u32)>,
}

#[derive(Debug)]
pub enum RendererError {
    Initialization(String),
    Surface(wgpu::SurfaceError),
    Device(wgpu::RequestDeviceError),
}

impl From<wgpu::SurfaceError> for RendererError {
    fn from(e: wgpu::SurfaceError) -> Self {
        Self::Surface(e)
    }
}

impl From<RendererError> for io::Error {
    fn from(err: RendererError) -> io::Error {
        io::Error::new(io::ErrorKind::Other, err)
    }
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::Initialization(msg) => write!(f, "initialization error: {}", msg),
            Self::Surface(e) => write!(f, "surface error: {}", e),
            Self::Device(e) => write!(f, "device error: {}", e),
        }
    }
}

impl Error for RendererError {
    fn description(&self) -> &str {
        "Renderer error"
    }

    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}

fn text_batch([w, h]: [u32; 2]) -> TextBatch {
    TextBatch::new(w, h, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT)
}

impl<'a> renderer::Renderer<'a> for Renderer {
    type Error = RendererError;

    fn new(
        win: &mut platform::backend::Window,
        win_size: LogicalSize,
        scale_factor: f64,
        assets: Assets<'a>,
    ) -> io::Result<Self> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface from window handle. The dummy platform has no window
        // handle; we run surface-less and skip presentation entirely.
        // SAFETY: The window handle is valid and the window will outlive the surface
        // because both are stored in the main loop and the renderer is dropped before
        // the window.
        #[cfg(feature = "glfw")]
        let surface = Some(unsafe {
            let raw_display_handle = win
                .display_handle()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
                .as_raw();
            let raw_window_handle = win
                .window_handle()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
                .as_raw();
            let target = wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            };
            instance
                .create_surface_unsafe(target)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
        });
        #[cfg(not(feature = "glfw"))]
        let surface: Option<wgpu::Surface<'static>> = {
            let _ = &win;
            None
        };

        // Request adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: surface.as_ref(),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No suitable adapter found"))?;

        // Request device and queue
        let (device, queue): (wgpu::Device, wgpu::Queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rx_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));

        // Configure surface
        let physical = win_size.to_physical(scale_factor);
        let (surface_format, alpha_mode) = match &surface {
            Some(surface) => {
                let caps = surface.get_capabilities(&adapter);
                let format = caps
                    .formats
                    .iter()
                    .find(|f| f.is_srgb())
                    .copied()
                    .unwrap_or(caps.formats[0]);
                (format, caps.alpha_modes[0])
            }
            None => (
                wgpu::TextureFormat::Rgba8UnormSrgb,
                wgpu::CompositeAlphaMode::Auto,
            ),
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: physical.width as u32,
            height: physical.height as u32,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        if let Some(surface) = &surface {
            surface.configure(&device, &surface_config);
        }

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Load textures
        let (font_img, font_w, font_h) = image::read(assets.glyphs)?;
        let (cursors_img, cursors_w, cursors_h) = image::read(data::CURSORS)?;
        let (checker_w, checker_h) = (2, 2);
        let (paste_w, paste_h) = (8, 8);

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let font = Texture::new_with_data(&device, &queue, font_w, font_h, format, Some(&font_img));
        let cursors = Texture::new_with_data(
            &device,
            &queue,
            cursors_w,
            cursors_h,
            format,
            Some(&cursors_img),
        );
        let checker = Texture::new_with_data(
            &device,
            &queue,
            checker_w,
            checker_h,
            format,
            Some(&draw::CHECKER),
        );
        let paste = Texture::new_with_data(&device, &queue, paste_w, paste_h, format, None);

        // Create screen render target
        let screen_texture = Texture::new(
            &device,
            win_size.width as u32,
            win_size.height as u32,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );

        // Create bind group layouts
        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("transform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let screen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("screen_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let cursor_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cursor_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let cursor_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cursor_texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        // VERTEX needed for textureDimensions() call in vs_main
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create uniform buffers
        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transform_buffer"),
            size: std::mem::size_of::<TransformUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cursor_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cursor_uniform_buffer"),
            size: std::mem::size_of::<CursorUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load shaders and create pipelines
        let sprite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("data/sprite.wgsl").into()),
        });

        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shape_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("data/shape.wgsl").into()),
        });

        let cursor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cursor_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("data/cursor.wgsl").into()),
        });

        let screen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screen_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("data/screen.wgsl").into()),
        });

        // Sprite pipeline
        let sprite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sprite_pipeline_layout"),
                bind_group_layouts: &[&transform_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sprite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite_pipeline"),
            layout: Some(&sprite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sprite_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Sprite2dVertex>() as u64,
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
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sprite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Surface-format twin of the sprite pipeline: the present pass
        // (the swapchain, `surface_format`) draws the debug/replay overlay
        // text, which `sprite_pipeline` (Rgba8, for `screen_texture`)
        // can't target — a windowed replay would hit a pipeline/pass
        // format mismatch otherwise.
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_pipeline"),
            layout: Some(&sprite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sprite_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Sprite2dVertex>() as u64,
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
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sprite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Shape pipeline
        let shape_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shape_pipeline_layout"),
                bind_group_layouts: &[&transform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let shape_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shape_pipeline"),
            layout: Some(&shape_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shape_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Shape2dVertex>() as u64,
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
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Unorm8x4,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shape_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Shape replace pipeline (for Blending::Constant - no alpha blending, just replace)
        let shape_replace_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shape_replace_pipeline"),
                layout: Some(&shape_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shape_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Shape2dVertex>() as u64,
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
                                format: wgpu::VertexFormat::Float32,
                            },
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Unorm8x4,
                            },
                        ],
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shape_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // Cursor pipeline
        let cursor_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cursor_pipeline_layout"),
                bind_group_layouts: &[&cursor_bind_group_layout, &cursor_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let cursor_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cursor_pipeline"),
            layout: Some(&cursor_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cursor_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Cursor2dVertex>() as u64,
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
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cursor_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Screen pipeline (fullscreen quad)
        let screen_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("screen_pipeline_layout"),
                bind_group_layouts: &[&screen_bind_group_layout],
                push_constant_ranges: &[],
            });

        let screen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen_pipeline"),
            layout: Some(&screen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &screen_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &screen_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create draw context
        let draw_ctx = draw::Context {
            ui_batch: shape2d::Batch::new(),
            text_batch: text_batch(font.size),
            overlay_batch: text_batch(font.size),
            cursor_sprite: sprite::Sprite::new(cursors_w, cursors_h),
            tool_batch: sprite2d::Batch::new(cursors_w, cursors_h),
            paste_batch: sprite2d::Batch::new(paste_w, paste_h),
            checker_batch: sprite2d::Batch::new(checker_w, checker_h),
        };

        Ok(Renderer {
            win_size,
            device,
            queue,
            surface,
            surface_config,
            draw_ctx,
            scale_factor,
            scale: 1.0,
            blending: Blending::Alpha,
            screen_texture,
            staging_batch: shape2d::Batch::new(),
            final_batch: shape2d::Batch::new(),
            font,
            cursors,
            checker,
            paste,
            sampler,
            sprite_pipeline,
            overlay_pipeline,
            shape_pipeline,
            shape_replace_pipeline,
            cursor_pipeline,
            screen_pipeline,
            transform_bind_group_layout,
            texture_bind_group_layout,
            screen_bind_group_layout,
            cursor_bind_group_layout,
            cursor_texture_bind_group_layout,
            transform_buffer,
            cursor_uniform_buffer,
            view_data: BTreeMap::new(),
            paste_pixels: Vec::new(),
            paste_size: (0, 0),
            paste_outputs: Vec::new(),
        })
    }

    fn init(&mut self, effects: Vec<Effect>, session: &Session) {
        self.handle_effects(effects, session).unwrap();
    }

    fn frame(
        &mut self,
        session: &mut Session,
        execution: &mut Execution,
        effects: Vec<session::Effect>,
        avg_frametime: &time::Duration,
        plugins: &mut crate::script::PluginHost,
    ) -> Result<(), RendererError> {
        if session.state != session::State::Running {
            return Ok(());
        }

        self.staging_batch.clear();
        self.final_batch.clear();
        self.paste_outputs.clear();

        self.handle_effects(effects, session).unwrap();

        self.update_view_animations(session);
        self.update_view_composites(session);
        self.update_view_layer_attrs(session);

        // Get surface texture (surface-less mode renders to `screen_texture` only)
        let output = match &self.surface {
            Some(surface) => Some(surface.get_current_texture()?),
            None => None,
        };
        let surface_view = output.as_ref().map(|output| {
            output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        // Prepare draw context
        self.draw_ctx.clear();
        self.draw_ctx.draw(session, avg_frametime, execution);

        // Script draw hooks add to the UI batches on top of the builtins.
        plugins.dispatch_draw(session, &mut self.draw_ctx);

        let [screen_w, screen_h] = self.screen_texture.size;
        let ortho: M44 = ortho_wgpu(screen_w, screen_h, Origin::TopLeft).into();
        let identity: M44 = Matrix4::identity().into();

        // Create vertex buffers for this frame
        let ui_vertices = self.create_shape_vertices(&self.draw_ctx.ui_batch.vertices());
        let text_vertices = self.create_sprite_vertices(&self.draw_ctx.text_batch.vertices());
        let tool_vertices = self.create_sprite_vertices(&self.draw_ctx.tool_batch.vertices());
        let checker_vertices = self.create_sprite_vertices(&self.draw_ctx.checker_batch.vertices());
        let cursor_verts = self.draw_ctx.cursor_sprite.vertices();

        // Create uniform bind group with ortho and identity transform
        let uniforms = TransformUniforms {
            ortho,
            transform: identity,
        };
        self.queue
            .write_buffer(&self.transform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let transform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform_bind_group"),
            layout: &self.transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.transform_buffer.as_entire_binding(),
            }],
        });

        // Texture bind groups
        let font_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("font_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.font.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let cursors_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cursors_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.cursors.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let checker_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("checker_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.checker.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Get active view for brush stroke rendering
        let v = session
            .views
            .active()
            .expect("there must always be an active view");
        let view_data = self
            .view_data
            .get(&v.id)
            .expect("view must have associated view data");

        // Render brush strokes to view staging buffer
        let view_ortho: M44 = ortho_wgpu(v.width(), v.fh, Origin::TopLeft).into();

        // Create staging vertex buffer from staging_batch
        let staging_vertices = self.create_shape_vertices(&self.staging_batch.vertices());
        let paste_vertices = self.create_sprite_vertices(&self.draw_ctx.paste_batch.vertices());

        // Always run staging pass for active view so the staging target is cleared every frame
        // (otherwise after ESC the paste preview ghost persists until next view modification).
        // Create uniform buffer for view ortho
        let view_uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_uniform_buffer"),
            size: std::mem::size_of::<TransformUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        let staging_uniforms = TransformUniforms {
            ortho: view_ortho,
            transform: identity,
        };
        view_uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&staging_uniforms));
        view_uniform_buffer.unmap();

        let staging_transform_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("staging_transform_bind_group"),
                layout: &self.transform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_uniform_buffer.as_entire_binding(),
                }],
            });

        // Render to staging target (clear every frame so ghost disappears on ESC)
        {
            let mut staging_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("staging_brush_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_data.staging_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw staging brush strokes
            if let Some((buffer, count)) = staging_vertices {
                staging_pass.set_pipeline(&self.shape_pipeline);
                staging_pass.set_bind_group(0, &staging_transform_bind_group, &[]);
                staging_pass.set_vertex_buffer(0, buffer.slice(..));
                staging_pass.draw(0..count, 0..1);
            }

            // Draw paste preview (paste texture)
            if let Some((buffer, count)) = paste_vertices {
                let paste_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("staging_paste_bind_group"),
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.paste.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                staging_pass.set_pipeline(&self.sprite_pipeline);
                staging_pass.set_bind_group(0, &staging_transform_bind_group, &[]);
                staging_pass.set_bind_group(1, &paste_bind_group, &[]);
                staging_pass.set_vertex_buffer(0, buffer.slice(..));
                staging_pass.draw(0..count, 0..1);
            }
        }

        // Render final brush strokes and paste outputs to layer target
        let final_vertices = self.create_shape_vertices(&self.final_batch.vertices());
        let has_paste_outputs = !self.paste_outputs.is_empty();

        if final_vertices.is_some() || has_paste_outputs {
            let view_uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("final_uniform_buffer"),
                size: std::mem::size_of::<TransformUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            // The final pass targets the sheet-sized layer texture, while
            // shapes are display-space. Translate down by one strip per
            // layer index to route writes into the active layer's rows.
            let final_ortho: M44 =
                ortho_wgpu(v.width(), v.sheet_height(), Origin::TopLeft).into();
            let routed = Matrix4::from_translation(
                Vector2::new(0., (v.active_layer as u32 * v.fh) as f32).extend(0.),
            );
            let final_uniforms = TransformUniforms {
                ortho: final_ortho,
                transform: routed.into(),
            };
            view_uniform_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&final_uniforms));
            view_uniform_buffer.unmap();

            let final_transform_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("final_transform_bind_group"),
                    layout: &self.transform_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: view_uniform_buffer.as_entire_binding(),
                    }],
                });

            // Render to layer target (don't clear - preserve existing pixels)
            let mut final_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("final_brush_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_data.layer.texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw final brush strokes
            if let Some((buffer, count)) = final_vertices {
                // Use replace pipeline for Blending::Constant, otherwise alpha blending
                if self.blending == Blending::Constant {
                    final_pass.set_pipeline(&self.shape_replace_pipeline);
                } else {
                    final_pass.set_pipeline(&self.shape_pipeline);
                }
                final_pass.set_bind_group(0, &final_transform_bind_group, &[]);
                final_pass.set_vertex_buffer(0, buffer.slice(..));
                final_pass.draw(0..count, 0..1);
            }

            // Draw paste outputs (rendered quads with paste texture)
            if has_paste_outputs {
                let paste_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("final_paste_bind_group"),
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.paste.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                final_pass.set_pipeline(&self.sprite_pipeline);
                final_pass.set_bind_group(0, &final_transform_bind_group, &[]);
                final_pass.set_bind_group(1, &paste_bind_group, &[]);

                for (buffer, count) in &self.paste_outputs {
                    final_pass.set_vertex_buffer(0, buffer.slice(..));
                    final_pass.draw(0..*count, 0..1);
                }
            }
        }

        // Script `shade` stage: plugins record their own passes after
        // view content, before screen composition. Each view's layer
        // texture is exposed as a named render target.
        let mut view_targets = crate::script::ViewTargets::new();
        for (id, vd) in &self.view_data {
            let tex = &vd.layer.texture;
            view_targets.insert(
                u16::from(*id),
                crate::script::ViewTarget {
                    layer: tex.texture.create_view(&Default::default()),
                    staging: vd.staging_texture.texture.create_view(&Default::default()),
                    width: tex.size[0],
                    height: tex.size[1],
                },
            );
        }
        let mut encoder = plugins.dispatch_shade(session, encoder, view_targets);

        // Render to screen framebuffer. The pass is handed to the
        // script `render` stage at the end of the block (lifetime
        // erased so the encoder stays movable for the hand-off).
        let screen_pass = {
            let bg = Rgba::from(session.settings["background"].to_rgba8());
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("screen_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.screen_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: bg.r as f64,
                                g: bg.g as f64,
                                b: bg.b as f64,
                                a: bg.a as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();

            // Draw checkers if enabled
            if session.settings["checker"].is_set() {
                if let Some((buffer, count)) = &checker_vertices {
                    pass.set_pipeline(&self.sprite_pipeline);
                    pass.set_bind_group(0, &transform_bind_group, &[]);
                    pass.set_bind_group(1, &checker_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..*count, 0..1);
                }
            }

            // Render views
            for (id, view_data) in &self.view_data {
                if let Some(view) = session.views.get(*id) {
                    let transform = Matrix4::from_translation(
                        (session.offset + view.offset).extend(*draw::VIEW_LAYER),
                    ) * Matrix4::from_nonuniform_scale(view.zoom, view.zoom, 1.0);

                    let view_uniforms = TransformUniforms {
                        ortho,
                        transform: transform.into(),
                    };

                    // Create a temporary buffer for view transform
                    let view_transform_buffer =
                        self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("view_transform_buffer"),
                            size: std::mem::size_of::<TransformUniforms>() as u64,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: true,
                        });
                    view_transform_buffer
                        .slice(..)
                        .get_mapped_range_mut()
                        .copy_from_slice(bytemuck::bytes_of(&view_uniforms));
                    view_transform_buffer.unmap();

                    let view_transform_bind_group =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("view_transform_bind_group"),
                            layout: &self.transform_bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: view_transform_buffer.as_entire_binding(),
                            }],
                        });

                    let view_texture_bind_group =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("view_texture_bind_group"),
                            layout: &self.texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &view_data.layer.texture.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });

                    let staging_bind_group =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("staging_bind_group"),
                            layout: &self.texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &view_data.staging_texture.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });

                    // Layer strips composite bottom-up (six vertices per
                    // strip quad), with the staging overlay at the active
                    // layer's depth: in-progress strokes preview correctly
                    // occluded by the layers above.
                    let count = view_data.layer.vertex_count;
                    let split = ((view.active_layer as u32 + 1) * 6).min(count);

                    pass.set_pipeline(&self.sprite_pipeline);
                    pass.set_bind_group(0, &view_transform_bind_group, &[]);
                    pass.set_bind_group(1, &view_texture_bind_group, &[]);
                    pass.set_vertex_buffer(0, view_data.layer.vertex_buffer.slice(..));
                    pass.draw(0..split, 0..1);

                    pass.set_bind_group(1, &staging_bind_group, &[]);
                    pass.set_vertex_buffer(0, view_data.staging_vertex_buffer.slice(..));
                    pass.draw(0..view_data.staging_vertex_count, 0..1);

                    if split < count {
                        pass.set_bind_group(1, &view_texture_bind_group, &[]);
                        pass.set_vertex_buffer(0, view_data.layer.vertex_buffer.slice(..));
                        pass.draw(split..count, 0..1);
                    }
                }
            }

            // Render UI shapes
            if let Some((buffer, count)) = &ui_vertices {
                pass.set_pipeline(&self.shape_pipeline);
                pass.set_bind_group(0, &transform_bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..*count, 0..1);
            }

            // Render text
            if let Some((buffer, count)) = &text_vertices {
                pass.set_pipeline(&self.sprite_pipeline);
                pass.set_bind_group(0, &transform_bind_group, &[]);
                pass.set_bind_group(1, &font_bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..*count, 0..1);
            }

            // Render tool sprites
            if let Some((buffer, count)) = &tool_vertices {
                pass.set_pipeline(&self.sprite_pipeline);
                pass.set_bind_group(0, &transform_bind_group, &[]);
                pass.set_bind_group(1, &cursors_bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..*count, 0..1);
            }

            // Render view animations if enabled
            if session.settings["animation"].is_set() {
                for (id, view_data) in &self.view_data {
                    if let Some(view) = session.views.get(*id) {
                        // Only render animations for views with more than one frame
                        if view.animation.len() > 1 && view_data.anim_vertex_count > 0 {
                            if let Some(ref anim_buffer) = view_data.anim_vertex_buffer {
                                // Create animation transform with translation
                                let anim_transform = Matrix4::from_translation(
                                    Vector2::new(0., view.zoom).extend(0.),
                                );
                                let anim_uniforms = TransformUniforms {
                                    ortho,
                                    transform: anim_transform.into(),
                                };

                                let anim_uniform_buffer =
                                    self.device.create_buffer(&wgpu::BufferDescriptor {
                                        label: Some("anim_uniform_buffer"),
                                        size: std::mem::size_of::<TransformUniforms>() as u64,
                                        usage: wgpu::BufferUsages::UNIFORM
                                            | wgpu::BufferUsages::COPY_DST,
                                        mapped_at_creation: true,
                                    });
                                anim_uniform_buffer
                                    .slice(..)
                                    .get_mapped_range_mut()
                                    .copy_from_slice(bytemuck::bytes_of(&anim_uniforms));
                                anim_uniform_buffer.unmap();

                                let anim_bind_group =
                                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("anim_transform_bind_group"),
                                        layout: &self.transform_bind_group_layout,
                                        entries: &[wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: anim_uniform_buffer.as_entire_binding(),
                                        }],
                                    });

                                // Bind layer texture for animation
                                let layer_bind_group =
                                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("anim_layer_bind_group"),
                                        layout: &self.texture_bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::TextureView(
                                                    &view_data.layer.texture.view,
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::Sampler(
                                                    &self.sampler,
                                                ),
                                            },
                                        ],
                                    });

                                pass.set_pipeline(&self.sprite_pipeline);
                                pass.set_bind_group(0, &anim_bind_group, &[]);
                                pass.set_bind_group(1, &layer_bind_group, &[]);
                                pass.set_vertex_buffer(0, anim_buffer.slice(..));
                                pass.draw(0..view_data.anim_vertex_count, 0..1);
                            }
                        }
                    }
                }
            }

            // Render help overlay if in help mode
            if session.mode == session::Mode::Help {
                let mut help_shape_batch = shape2d::Batch::new();
                let mut help_text_batch = text_batch(self.font.size);
                draw::draw_help(session, &mut help_text_batch, &mut help_shape_batch);

                // Draw help shape (background)
                if let Some((buffer, count)) =
                    self.create_shape_vertices(&help_shape_batch.vertices())
                {
                    pass.set_pipeline(&self.shape_pipeline);
                    pass.set_bind_group(0, &transform_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..count, 0..1);
                }

                // Draw help text
                if let Some((buffer, count)) =
                    self.create_sprite_vertices(&help_text_batch.vertices())
                {
                    pass.set_pipeline(&self.sprite_pipeline);
                    pass.set_bind_group(0, &transform_bind_group, &[]);
                    pass.set_bind_group(1, &font_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..count, 0..1);
                }
            }

            pass
        };

        // Script `render` stage: the live screen pass — everything
        // drawn, before present — handed to each plugin's `render`
        // hook for screen-space drawing.
        let mut encoder =
            plugins.dispatch_render(session, encoder, screen_pass, &self.screen_texture.view);

        // Render screen to surface (final pass). Skipped surface-less: cursor and
        // overlay only ever hit the swapchain, so digests are unaffected.
        if let Some(surface_view) = &surface_view {
            let screen_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("screen_bind_group"),
                layout: &self.screen_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.screen_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.screen_pipeline);
            pass.set_bind_group(0, &screen_bind_group, &[]);
            pass.draw(0..6, 0..1);

            // Render cursor
            if !cursor_verts.is_empty() {
                // Create cursor vertex buffer (sprite::Vertex is a tuple struct)
                let cursor_vertices: Vec<Cursor2dVertex> = cursor_verts
                    .iter()
                    .map(|v| Cursor2dVertex {
                        position: [v.0.x, v.0.y, v.0.z],
                        uv: [v.1.x, v.1.y],
                    })
                    .collect();

                let cursor_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cursor_vertex_buffer"),
                    size: (cursor_vertices.len() * std::mem::size_of::<Cursor2dVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                cursor_buffer
                    .slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(&cursor_vertices));
                cursor_buffer.unmap();

                // Create cursor uniforms - use screen_texture size since cursor positions are in that coordinate space
                let [cursor_w, cursor_h] = self.screen_texture.size;
                let cursor_ortho: M44 = ortho_wgpu(cursor_w, cursor_h, Origin::TopLeft).into();
                let ui_scale = session.settings["scale"].to_f64();
                let pixel_ratio = platform::pixel_ratio(self.scale_factor);
                let cursor_uniforms = CursorUniforms {
                    ortho: cursor_ortho,
                    scale: (ui_scale * pixel_ratio) as f32,
                    _padding: [0.0; 7],
                };
                self.queue.write_buffer(
                    &self.cursor_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&cursor_uniforms),
                );

                let cursor_uniform_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cursor_uniform_bind_group"),
                        layout: &self.cursor_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.cursor_uniform_buffer.as_entire_binding(),
                        }],
                    });

                let cursor_texture_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cursor_texture_bind_group"),
                        layout: &self.cursor_texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.cursors.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.screen_texture.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });

                pass.set_pipeline(&self.cursor_pipeline);
                pass.set_bind_group(0, &cursor_uniform_bind_group, &[]);
                pass.set_bind_group(1, &cursor_texture_bind_group, &[]);
                pass.set_vertex_buffer(0, cursor_buffer.slice(..));
                pass.draw(0..cursor_vertices.len() as u32, 0..1);
            }

            // Render debug/overlay text if debug setting is on or execution is not normal
            if session.settings["debug"].is_set() || !execution.is_normal() {
                let overlay_vertices =
                    self.create_sprite_vertices(&self.draw_ctx.overlay_batch.vertices());
                if let Some((buffer, count)) = overlay_vertices {
                    // TopLeft ortho, matching the screen pass and the
                    // cursor in this same present pass. The glyph quads'
                    // sprite pixel coordinates use this
                    // origin; a BottomLeft projection (the old GL default)
                    // renders the text vertically flipped / mirrored.
                    let [overlay_w, overlay_h] = self.screen_texture.size;
                    let overlay_ortho: M44 =
                        ortho_wgpu(overlay_w, overlay_h, Origin::TopLeft).into();
                    let overlay_uniforms = TransformUniforms {
                        ortho: overlay_ortho,
                        transform: identity,
                    };

                    let overlay_uniform_buffer =
                        self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("overlay_uniform_buffer"),
                            size: std::mem::size_of::<TransformUniforms>() as u64,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: true,
                        });
                    overlay_uniform_buffer
                        .slice(..)
                        .get_mapped_range_mut()
                        .copy_from_slice(bytemuck::bytes_of(&overlay_uniforms));
                    overlay_uniform_buffer.unmap();

                    let overlay_bind_group =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("overlay_transform_bind_group"),
                            layout: &self.transform_bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: overlay_uniform_buffer.as_entire_binding(),
                            }],
                        });

                    // The present pass is the swapchain (surface format),
                    // so the overlay text uses the surface-format twin.
                    pass.set_pipeline(&self.overlay_pipeline);
                    pass.set_bind_group(0, &overlay_bind_group, &[]);
                    pass.set_bind_group(1, &font_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..count, 0..1);
                }
            }
        }

        // Record a snapshot of every dirty view — not just the active
        // one: scripts edit other views through view passes +
        // touch_view (the v3 cross-view recording fix, `e540903`).
        let should_record: Vec<_> = session
            .views
            .iter()
            .filter(|v| v.is_dirty())
            .map(|v| (v.id, v.is_resized()))
            .collect();

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Some(output) = output {
            output.present();
        }

        for (view_id, was_resized) in should_record {
            let view_data = self
                .view_data
                .get(&view_id)
                .expect("view must have associated view data");
            let pixels = view_data.layer.pixels(&self.device, &self.queue);

            if let Some(v) = session.views.get_mut(view_id) {
                if was_resized {
                    v.resource.record_view_resized(pixels, v.extent());
                } else {
                    v.resource.record_view_painted(pixels);
                }
            }
        }

        // Record snapshots if needed
        if !execution.is_normal() {
            let [w, h] = self.screen_texture.size;
            let texels = self.read_screen_pixels(w, h);
            // Debug aid: dump each *distinct* frame as PNG when RX_DUMP_FRAMES
            // is set to a directory (used to triage digest mismatches). Frames
            // are deduped by hash, mirroring the digest recorder's stale check.
            if let Ok(dir) = std::env::var("RX_DUMP_FRAMES") {
                use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
                static FRAME_NO: AtomicUsize = AtomicUsize::new(0);
                static LAST_HASH: AtomicU64 = AtomicU64::new(0);
                let hash = seahash::hash(util::align_u8(&texels));
                if LAST_HASH.swap(hash, Ordering::SeqCst) != hash {
                    let n = FRAME_NO.fetch_add(1, Ordering::SeqCst);
                    let path = std::path::Path::new(&dir).join(format!("{:04}.png", n));
                    crate::image::save_as(&path, w, h, 1, &texels).ok();
                }
            }
            execution.record(&texels).ok();
        }

        Ok(())
    }

    fn handle_scale_factor_changed(&mut self, scale_factor: f64) {
        self.scale_factor = scale_factor;
        self.handle_resized(self.win_size);
    }
}

impl Renderer {
    /// Shared handles to the device and queue for the script GPU
    /// capability.
    pub fn gpu_handles(&self) -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
        (self.device.clone(), self.queue.clone())
    }

    pub fn handle_resized(&mut self, size: platform::LogicalSize) {
        let physical = size.to_physical(self.scale_factor);

        if physical.width as u32 > 0 && physical.height as u32 > 0 {
            self.surface_config.width = physical.width as u32;
            self.surface_config.height = physical.height as u32;
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, &self.surface_config);
            }
        }

        self.win_size = size;
        self.handle_session_scale_changed(self.scale);
    }

    pub fn handle_session_scale_changed(&mut self, scale: f64) {
        self.scale = scale;
        let w = (self.win_size.width / scale) as u32;
        let h = (self.win_size.height / scale) as u32;

        if w > 0 && h > 0 {
            self.screen_texture
                .resize(&self.device, w, h, wgpu::TextureFormat::Rgba8UnormSrgb);
        }
    }

    fn handle_effects(
        &mut self,
        mut effects: Vec<Effect>,
        session: &Session,
    ) -> Result<(), RendererError> {
        for eff in effects.drain(..) {
            match eff {
                Effect::SessionResized(size) => {
                    self.handle_resized(size);
                }
                Effect::SessionScaled(scale) => {
                    self.handle_session_scale_changed(scale);
                }
                Effect::ViewActivated(_) => {}
                Effect::ViewAdded(id) => {
                    if let Some((s, pixels)) = session.views.get_snapshot_safe(id) {
                        let (w, h, sheet_h) = (s.width(), s.extent.fh, s.height());
                        self.view_data.insert(
                            id,
                            ViewData::new(&self.device, &self.queue, w, h, sheet_h, Some(pixels)),
                        );
                    }
                }
                Effect::ViewRemoved(id) => {
                    self.view_data.remove(&id);
                }
                Effect::ViewOps(id, ops) => {
                    self.handle_view_ops(session.view(id), &ops)?;
                }
                Effect::ViewDamaged(id, Some(extent)) => {
                    self.handle_view_resized(session.view(id), extent.width(), extent.height())?;
                }
                Effect::ViewDamaged(id, None) => {
                    self.handle_view_damaged(session.view(id))?;
                }
                Effect::ViewBlendingChanged(blending) => {
                    self.blending = blending;
                }
                Effect::ViewPaintDraft(shapes) => {
                    shapes.into_iter().for_each(|s| self.staging_batch.add(s));
                }
                Effect::ViewPaintFinal(shapes) => {
                    shapes.into_iter().for_each(|s| self.final_batch.add(s));
                }
                Effect::ViewTouched(_) => {}
            }
        }
        Ok(())
    }

    fn handle_view_ops(
        &mut self,
        v: &View<ViewResource>,
        ops: &[ViewOp],
    ) -> Result<(), RendererError> {
        for op in ops {
            match op {
                ViewOp::Resize(w, h) => {
                    self.resize_view(v, *w, *h)?;
                }
                ViewOp::Clear(color) => {
                    // Clear the *active layer's* strip only (the whole
                    // texture for single-layer views, as before). A pass
                    // load-op clear can't be scissored, so the strip is
                    // overwritten with an upload.
                    let view_data = self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data");
                    let [tw, _] = view_data.layer.texture.size;
                    let strip_y = v.active_layer as u32 * v.fh;
                    let texels: Vec<u8> = [color.r, color.g, color.b, color.a]
                        .iter()
                        .cycle()
                        .take((tw * v.fh) as usize * 4)
                        .cloned()
                        .collect();
                    view_data
                        .layer
                        .upload_part(&self.queue, [0, strip_y], [tw, v.fh], &texels);
                }
                ViewOp::Blit(src, dst) => {
                    // Get pixels from the view's CPU snapshot
                    if let Some((_, pixels)) =
                        v.resource.layer.get_snapshot_rect(&src.map(|n| n as i32))
                    {
                        let view_data = self
                            .view_data
                            .get_mut(&v.id)
                            .expect("views must have associated view data");
                        let texels = util::align_u8(&pixels);
                        // Sheet coordinates are texture row coordinates.
                        view_data.layer.upload_part(
                            &self.queue,
                            [dst.x1 as u32, dst.y1 as u32],
                            [src.width() as u32, src.height() as u32],
                            texels,
                        );
                    }
                }
                ViewOp::Yank(src) => {
                    // `src` is a display-space rect; the snapshot read routes
                    // to the active layer's strip (y-down sheet coords).
                    let src = *src + Vector2::new(0, (v.active_layer as u32 * v.fh) as i32);
                    if let Some((_, pixels)) = v.resource.layer.get_snapshot_rect(&src) {
                        let (w, h) = (src.width() as u32, src.height() as u32);

                        // Resize paste texture if needed
                        let [paste_w, paste_h] = self.paste.size;
                        if paste_w != w || paste_h != h {
                            self.paste.resize_same_format(&self.device, w, h);
                        }

                        // Upload to paste texture
                        let body = util::align_u8(&pixels);
                        self.queue.write_texture(
                            wgpu::ImageCopyTexture {
                                texture: &self.paste.texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            body,
                            wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(4 * w),
                                rows_per_image: Some(h),
                            },
                            wgpu::Extent3d {
                                width: w,
                                height: h,
                                depth_or_array_layers: 1,
                            },
                        );

                        // Update paste_pixels/paste_size for compatibility
                        self.paste_pixels = pixels;
                        self.paste_size = (w, h);
                    }
                }
                ViewOp::Flip(src, dir) => {
                    // `src` is display-space; read from the active layer's
                    // strip, flip on the CPU, and write back through the
                    // paste machinery (routed by the final-pass transform).
                    let src = *src + Vector2::new(0, (v.active_layer as u32 * v.fh) as i32);
                    if let Some((_, mut pixels)) = v.resource.layer.get_snapshot_rect(&src) {
                        let (w, h) = (src.width() as u32, src.height() as u32);

                        match dir {
                            Axis::Vertical => {
                                let len = pixels.len();
                                let (front, back) = pixels.split_at_mut(len / 2);
                                for (front_row, back_row) in front
                                    .chunks_exact_mut(w as usize)
                                    .zip(back.rchunks_exact_mut(w as usize))
                                {
                                    front_row.swap_with_slice(back_row);
                                }
                            }
                            Axis::Horizontal => {
                                pixels
                                    .chunks_exact_mut(w as usize)
                                    .for_each(|row| row.reverse());
                            }
                        }

                        // Resize paste texture if needed
                        let [paste_w, paste_h] = self.paste.size;
                        if paste_w != w || paste_h != h {
                            self.paste.resize_same_format(&self.device, w, h);
                        }

                        // Upload to paste texture
                        let body = util::align_u8(&pixels);
                        self.queue.write_texture(
                            wgpu::ImageCopyTexture {
                                texture: &self.paste.texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            body,
                            wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(4 * w),
                                rows_per_image: Some(h),
                            },
                            wgpu::Extent3d {
                                width: w,
                                height: h,
                                depth_or_array_layers: 1,
                            },
                        );

                        // Also update paste_pixels/paste_size for Paste op compatibility
                        self.paste_pixels = pixels;
                        self.paste_size = (w, h);
                    }
                }
                ViewOp::Paste(dst) => {
                    // Create a sprite batch for the paste quad (like GL's paste_outputs)
                    let [paste_w, paste_h] = self.paste.size;
                    if paste_w > 0 && paste_h > 0 {
                        let batch = sprite2d::Batch::singleton(
                            paste_w,
                            paste_h,
                            Rect::origin(paste_w as f32, paste_h as f32),
                            dst.map(|n| n as f32),
                            ZDepth::default(),
                            Rgba::TRANSPARENT,
                            1.,
                            Repeat::default(),
                        );

                        let vertices = batch.vertices();
                        if !vertices.is_empty() {
                            let sprite_vertices: Vec<Sprite2dVertex> = vertices
                                .iter()
                                .map(|v| Sprite2dVertex {
                                    position: [v.position.x, v.position.y, v.position.z],
                                    uv: [v.uv.x, v.uv.y],
                                    color: [v.color.r, v.color.g, v.color.b, v.color.a],
                                    opacity: v.opacity,
                                })
                                .collect();

                            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("paste_output_buffer"),
                                size: (sprite_vertices.len()
                                    * std::mem::size_of::<Sprite2dVertex>())
                                    as u64,
                                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: true,
                            });
                            buffer
                                .slice(..)
                                .get_mapped_range_mut()
                                .copy_from_slice(bytemuck::cast_slice(&sprite_vertices));
                            buffer.unmap();

                            self.paste_outputs
                                .push((buffer, sprite_vertices.len() as u32));
                        }
                    }
                }
                ViewOp::SetPixel(rgba, x, y) => {
                    let view_data = self
                        .view_data
                        .get_mut(&v.id)
                        .expect("views must have associated view data");
                    let texels = &[rgba.r, rgba.g, rgba.b, rgba.a];
                    // `y` is a display-space row; offset it into the active
                    // layer's strip (no-op for single-layer views).
                    let strip = v.active_layer as u32 * v.fh;
                    view_data.layer.upload_part(
                        &self.queue,
                        [*x as u32, strip + *y as u32],
                        [1, 1],
                        texels,
                    );
                }
            }
        }
        Ok(())
    }

    fn handle_view_damaged(&mut self, view: &View<ViewResource>) -> Result<(), RendererError> {
        let view_data = self
            .view_data
            .get_mut(&view.id)
            .expect("views must have associated view data");
        let (_, pixels) = view.layer.current_snapshot();
        view_data.layer.upload(&self.queue, util::align_u8(pixels));

        // Clear staging target so old draft strokes don't appear on top
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clear_staging_encoder"),
            });
        view_data.staging_texture.clear(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn handle_view_resized(
        &mut self,
        view: &View<ViewResource>,
        vw: u32,
        vh: u32,
    ) -> Result<(), RendererError> {
        self.resize_view(view, vw, vh)
    }

    fn resize_view(
        &mut self,
        view: &View<ViewResource>,
        vw: u32,
        vh: u32,
    ) -> Result<(), RendererError> {
        let previous = view.resource.extent;
        let copy_w = u32::min(previous.width(), vw);
        let copy_h = u32::min(previous.fh, view.fh);

        // Each layer keeps its own top-left anchor when the frame height
        // changes. Copying the sheet as one rectangle would leave later
        // layers at their old row offsets, crossing the new strip boundaries.
        let view_data = ViewData::new(&self.device, &self.queue, vw, view.fh, vh, None);
        let layers = usize::min(previous.nlayers, (vh / view.fh) as usize);
        for n in 0..layers as u32 {
            let old_y = n * previous.fh;
            let src = Rect::new(0, old_y as i32, copy_w as i32, (old_y + copy_h) as i32);
            if let Some((_, texels)) = view.layer.get_snapshot_rect(&src) {
                view_data.layer.upload_part(
                    &self.queue,
                    [0, n * view.fh],
                    [copy_w, copy_h],
                    util::align_u8(&texels),
                );
            }
        }

        self.view_data.insert(view.id, view_data);
        Ok(())
    }

    /// Re-bake layer presentation attributes (visibility, opacity, and
    /// the `layers/dim` factor on inactive layers) into each view's
    /// strip vertex buffer, when they changed. Hidden strips stay in
    /// the buffer at opacity zero, so the vertex count — and the
    /// staging z-order split — never shifts.
    fn update_view_layer_attrs(&mut self, session: &Session) {
        let dim = session.settings["layers/dim"].to_f64() as f32;

        for (id, vd) in self.view_data.iter_mut() {
            if let Some(view) = session.views.get(*id) {
                let alphas: Vec<f32> = view
                    .layer_attrs
                    .iter()
                    .enumerate()
                    .map(|(n, a)| match (a.visible, n == view.active_layer) {
                        (false, _) => 0.,
                        (true, true) => a.opacity,
                        (true, false) => a.opacity * dim,
                    })
                    .collect();

                if alphas != vd.strip_alphas
                    && alphas.len() * 6 == vd.layer.vertex_count as usize
                {
                    let [w, sheet_h] = vd.layer.texture.size;
                    let batch = strip_batch(w, view.fh, sheet_h, &alphas);
                    let verts: Vec<Sprite2dVertex> = batch
                        .vertices()
                        .iter()
                        .map(|v| Sprite2dVertex {
                            position: [v.position.x, v.position.y, v.position.z],
                            uv: [v.uv.x, v.uv.y],
                            color: [v.color.r, v.color.g, v.color.b, v.color.a],
                            opacity: v.opacity,
                        })
                        .collect();
                    self.queue
                        .write_buffer(&vd.layer.vertex_buffer, 0, bytemuck::cast_slice(&verts));
                    vd.strip_alphas = alphas;
                }
            }
        }
    }

    fn update_view_animations(&mut self, session: &Session) {
        if !session.settings["animation"].is_set() {
            return;
        }
        for v in session.views.iter() {
            let batch = draw::draw_view_animation(session, v);
            let vertices = batch.vertices();

            if let Some(vd) = self.view_data.get_mut(&v.id) {
                if vertices.is_empty() {
                    vd.anim_vertex_buffer = None;
                    vd.anim_vertex_count = 0;
                } else {
                    let sprite_vertices: Vec<Sprite2dVertex> = vertices
                        .iter()
                        .map(|v| Sprite2dVertex {
                            position: [v.position.x, v.position.y, v.position.z],
                            uv: [v.uv.x, v.uv.y],
                            color: [v.color.r, v.color.g, v.color.b, v.color.a],
                            opacity: v.opacity,
                        })
                        .collect();

                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("anim_vertex_buffer"),
                        size: (sprite_vertices.len() * std::mem::size_of::<Sprite2dVertex>())
                            as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    buffer
                        .slice(..)
                        .get_mapped_range_mut()
                        .copy_from_slice(bytemuck::cast_slice(&sprite_vertices));
                    buffer.unmap();

                    vd.anim_vertex_buffer = Some(buffer);
                    vd.anim_vertex_count = sprite_vertices.len() as u32;
                }
            }
        }
    }

    fn update_view_composites(&mut self, session: &Session) {
        for v in session.views.iter() {
            let batch = draw::draw_view_composites(session, v);
            let vertices = batch.vertices();

            if let Some(vd) = self.view_data.get_mut(&v.id) {
                if vertices.is_empty() {
                    vd.layer_vertex_buffer = None;
                    vd.layer_vertex_count = 0;
                } else {
                    let sprite_vertices: Vec<Sprite2dVertex> = vertices
                        .iter()
                        .map(|v| Sprite2dVertex {
                            position: [v.position.x, v.position.y, v.position.z],
                            uv: [v.uv.x, v.uv.y],
                            color: [v.color.r, v.color.g, v.color.b, v.color.a],
                            opacity: v.opacity,
                        })
                        .collect();

                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("composite_vertex_buffer"),
                        size: (sprite_vertices.len() * std::mem::size_of::<Sprite2dVertex>())
                            as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    buffer
                        .slice(..)
                        .get_mapped_range_mut()
                        .copy_from_slice(bytemuck::cast_slice(&sprite_vertices));
                    buffer.unmap();

                    vd.layer_vertex_buffer = Some(buffer);
                    vd.layer_vertex_count = sprite_vertices.len() as u32;
                }
            }
        }
    }

    /// Create a vertex buffer from sprite2d vertices.
    fn create_sprite_vertices(&self, vertices: &[sprite2d::Vertex]) -> Option<(wgpu::Buffer, u32)> {
        if vertices.is_empty() {
            return None;
        }

        let verts: Vec<Sprite2dVertex> = vertices
            .iter()
            .map(|v| Sprite2dVertex {
                position: [v.position.x, v.position.y, v.position.z],
                uv: [v.uv.x, v.uv.y],
                color: [v.color.r, v.color.g, v.color.b, v.color.a],
                opacity: v.opacity,
            })
            .collect();

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_vertex_buffer"),
            size: (verts.len() * std::mem::size_of::<Sprite2dVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&verts));
        buffer.unmap();

        Some((buffer, verts.len() as u32))
    }

    /// Create a vertex buffer from shape2d vertices.
    fn create_shape_vertices(&self, vertices: &[shape2d::Vertex]) -> Option<(wgpu::Buffer, u32)> {
        if vertices.is_empty() {
            return None;
        }

        let verts: Vec<Shape2dVertex> = vertices
            .iter()
            .map(|v| Shape2dVertex {
                position: [v.position.x, v.position.y, v.position.z],
                angle: v.angle,
                center: [v.center.x, v.center.y],
                color: [v.color.r, v.color.g, v.color.b, v.color.a],
            })
            .collect();

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape_vertex_buffer"),
            size: (verts.len() * std::mem::size_of::<Shape2dVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&verts));
        buffer.unmap();

        Some((buffer, verts.len() as u32))
    }

    /// Read pixels from the screen texture (blocking).
    pub fn read_screen_pixels(&self, width: u32, height: u32) -> Vec<Rgba8> {
        self.read_texture_pixels(&self.screen_texture.texture, width, height)
    }

    /// Read pixels from a texture (blocking). Creates its own encoder and submit.
    fn read_texture_pixels(&self, texture: &wgpu::Texture, width: u32, height: u32) -> Vec<Rgba8> {
        let bytes_per_row = (4 * width + 255) & !255; // Align to 256 bytes
        let buffer_size = (bytes_per_row * height) as u64;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let row_bytes = (width * 4) as usize;
        let mut pixels = vec![Rgba8::default(); (width * height) as usize];
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut pixels);

        // Copy row-wise to strip the 256-byte row padding (memcpy per row,
        // fast even in unoptimized builds — this runs once per frame).
        for y in 0..height as usize {
            let src = &data[y * bytes_per_row as usize..][..row_bytes];
            dst[y * row_bytes..][..row_bytes].copy_from_slice(src);
        }

        pixels
    }
}

#[cfg(all(test, not(feature = "glfw")))]
mod coordinate_tests {
    use super::*;
    use crate::renderer::Renderer as _;
    use crate::view::FileStatus;

    fn frame(
        renderer: &mut Renderer,
        session: &mut Session,
        plugins: &mut crate::script::PluginHost,
    ) {
        let mut execution = Execution::Normal;
        let effects = session.update(
            &mut vec![],
            &mut execution,
            time::Duration::ZERO,
            time::Duration::ZERO,
            plugins,
        );
        renderer
            .frame(
                session,
                &mut execution,
                effects,
                &time::Duration::ZERO,
                plugins,
            )
            .unwrap();
        session.cleanup();
    }

    fn pixels(session: &Session) -> Vec<Rgba8> {
        session
            .active_view()
            .resource
            .layer
            .current_snapshot()
            .1
            .to_vec()
    }

    #[test]
    fn core_paint_layers_paste_undo_and_save_use_top_first_rows() {
        let (mut window, _) =
            platform::init("coordinates", 64, 64, &[], platform::GraphicsContext::None).unwrap();
        let mut renderer = Renderer::new(
            &mut window,
            LogicalSize::new(64., 64.),
            1.,
            Assets::new(crate::data::GLYPHS),
        )
        .unwrap();
        let dirs = directories::ProjectDirs::from("io", "cloudhead", "rx").unwrap();
        let base = directories::BaseDirs::new().unwrap();
        let mut session = Session::new(64, 64, std::env::temp_dir(), dirs, base).with_blank(
            FileStatus::NoFile,
            4,
            3,
        );
        session.transition(session::State::Running);
        let mut plugins = crate::script::PluginHost::new(None).unwrap();
        let (device, queue) = renderer.gpu_handles();
        plugins.attach_gfx(device, queue);
        frame(&mut renderer, &mut session, &mut plugins);
        let red = Rgba8::new(255, 0, 0, 255);
        let green = Rgba8::new(0, 255, 0, 255);
        let blue = Rgba8::new(0, 0, 255, 255);
        let clear = Rgba8::TRANSPARENT;
        {
            let view = session.active_view_mut();
            view.paint_color(red, 0, 0);
            view.paint_color(blue, 3, 2);
            view.touch();
        }
        frame(&mut renderer, &mut session, &mut plugins);
        let mut bottom = vec![clear; 12];
        bottom[0] = red;
        bottom[11] = blue;
        assert_eq!(pixels(&session), bottom);
        assert_eq!(
            *session
                .active_view()
                .color_at(crate::view::ViewCoords::new(0, 0))
                .unwrap(),
            red
        );

        session.active_view_mut().extend_layer();
        frame(&mut renderer, &mut session, &mut plugins);
        assert_eq!(pixels(&session), [bottom.clone(), vec![clear; 12]].concat());
        session.active_view_mut().activate_layer(1);
        session
            .effects
            .push(Effect::ViewPaintFinal(vec![shape2d::Shape::Rectangle(
                Rect::new(1., 0., 2., 1.),
                ZDepth::default(),
                shape2d::Rotation::ZERO,
                shape2d::Stroke::NONE,
                shape2d::Fill::Solid(green.into()),
            )]));
        session.active_view_mut().touch();
        frame(&mut renderer, &mut session, &mut plugins);
        let mut top = vec![clear; 12];
        top[1] = green;
        assert_eq!(pixels(&session), [bottom.clone(), top.clone()].concat());
        assert_eq!(
            *session
                .active_view()
                .color_at(crate::view::ViewCoords::new(1, 0))
                .unwrap(),
            green
        );

        // Yank/paste addresses the active layer using the same local rows.
        session.active_view_mut().yank(Rect::new(1, 0, 2, 1));
        session.active_view_mut().paste(Rect::new(2, 2, 3, 3));
        frame(&mut renderer, &mut session, &mut plugins);
        top[10] = green;
        assert_eq!(pixels(&session), [bottom.clone(), top.clone()].concat());

        session
            .active_view_mut()
            .restore_snapshot(session::Direction::Backward);
        frame(&mut renderer, &mut session, &mut plugins);
        top[10] = clear;
        assert_eq!(pixels(&session), [bottom.clone(), top.clone()].concat());
        session
            .active_view_mut()
            .restore_snapshot(session::Direction::Forward);
        frame(&mut renderer, &mut session, &mut plugins);
        top[10] = green;
        assert_eq!(pixels(&session), [bottom, top.clone()].concat());

        let path = tempfile::tempdir().unwrap();
        let file = path.path().join("layer.png");
        session
            .active_view()
            .resource
            .save(Rect::new(0, 3, 4, 6), &file)
            .unwrap();
        let (bytes, w, h) = crate::image::load(&file).unwrap();
        assert_eq!((w, h), (4, 3));
        assert_eq!(bytes, util::align_u8(&top));

        // Changing frame height must move whole strips, preserving each
        // layer's local rows rather than reinterpreting the old sheet.
        let before_resize = pixels(&session);
        session.active_view_mut().resize_frames(4, 4);
        frame(&mut renderer, &mut session, &mut plugins);
        let mut resized = before_resize[..12].to_vec();
        resized.extend_from_slice(&[clear; 4]);
        resized.extend_from_slice(&before_resize[12..]);
        resized.extend_from_slice(&[clear; 4]);
        assert_eq!(pixels(&session), resized);
        session.active_view_mut().restore_snapshot(session::Direction::Backward);
        frame(&mut renderer, &mut session, &mut plugins);
        assert_eq!(pixels(&session), before_resize);
    }
}

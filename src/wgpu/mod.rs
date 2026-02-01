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
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::io;
use std::time;

type M44 = [[f32; 4]; 4];

/// Create an orthographic projection matrix corrected for wgpu's coordinate system.
/// wgpu uses Y-down in clip space (Y=-1 at top, Y=+1 at bottom), opposite of OpenGL.
fn ortho_wgpu(w: u32, h: u32, origin: Origin) -> Matrix4<f32> {
    let mut m = Matrix4::ortho(w, h, origin);
    // Flip Y axis to account for wgpu's inverted clip space
    m.y.y = -m.y.y;
    m.w.y = -m.w.y;
    m
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
    ortho: M44,           // 64 bytes
    scale: f32,           // 4 bytes
    _padding: [f32; 7],   // 28 bytes to reach 96 total (WGSL alignment)
}

/// Bundle of texture, view, and sampler for a texture resource.
struct TextureBundle {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: [u32; 2],
}

impl TextureBundle {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, data: Option<&[u8]>) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        if let Some(data) = data {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
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
                size,
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            size: [width, height],
        }
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        self.texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        self.view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.size = [width, height];
    }
}

/// Render target texture (like a framebuffer).
struct RenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: [u32; 2],
}

impl RenderTarget {
    fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_target"),
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
        }
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) {
        *self = Self::new(device, width, height, format);
    }
}

/// Per-layer data for a view.
struct LayerData {
    target: RenderTarget,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

impl LayerData {
    fn new(device: &wgpu::Device, w: u32, h: u32, pixels: Option<&[Rgba8]>, queue: &wgpu::Queue) -> Self {
        let target = RenderTarget::new(device, w, h, wgpu::TextureFormat::Rgba8UnormSrgb);

        // Create a quad vertex buffer for rendering this layer
        let batch = sprite2d::Batch::singleton(
            w,
            h,
            Rect::origin(w as f32, h as f32),
            Rect::origin(w as f32, h as f32),
            ZDepth::default(),
            Rgba::TRANSPARENT,
            1.,
            Repeat::default(),
        );

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
            label: Some("layer_vertex_buffer"),
            size: (verts.len() * std::mem::size_of::<Sprite2dVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&verts));

        // Upload initial pixels if provided
        if let Some(pixels) = pixels {
            let aligned = util::align_u8(pixels);
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &target.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                aligned,
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

        Self {
            target,
            vertex_buffer,
            vertex_count: verts.len() as u32,
        }
    }

    #[allow(dead_code)]
    fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_layer"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target.view,
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
        let [w, h] = self.target.size;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.target.texture,
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
                texture: &self.target.texture,
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
}

/// Per-view rendering data.
struct ViewData {
    layer: LayerData,
    staging_target: RenderTarget,
    anim_vertex_buffer: Option<wgpu::Buffer>,
    anim_vertex_count: u32,
    layer_vertex_buffer: Option<wgpu::Buffer>,
    layer_vertex_count: u32,
}

impl ViewData {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, pixels: Option<&[Rgba8]>) -> Self {
        let staging_target = RenderTarget::new(device, w, h, wgpu::TextureFormat::Rgba8UnormSrgb);
        let layer = LayerData::new(device, w, h, pixels, queue);

        Self {
            layer,
            staging_target,
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

    // Core wgpu state
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // Draw context (same as GL version)
    draw_ctx: draw::Context,
    scale_factor: f64,
    scale: f64,
    blending: Blending,

    // Render targets
    screen_target: RenderTarget,

    // Batches
    staging_batch: shape2d::Batch,
    final_batch: shape2d::Batch,

    // Textures
    font: TextureBundle,
    cursors: TextureBundle,
    checker: TextureBundle,
    paste: TextureBundle,

    // Sampler
    sampler: wgpu::Sampler,

    // Pipelines
    sprite_pipeline: wgpu::RenderPipeline,
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

        // Create surface from window handle
        // SAFETY: The window handle is valid and the window will outlive the surface
        // because both are stored in the main loop and the renderer is dropped before
        // the window.
        let surface = unsafe {
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
        };

        // Request adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No suitable adapter found"))?;

        // Request device and queue
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rx_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        // Configure surface
        let physical = win_size.to_physical(scale_factor);
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: physical.width as u32,
            height: physical.height as u32,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

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

        let font = TextureBundle::new(&device, &queue, font_w, font_h, Some(&font_img));
        let cursors = TextureBundle::new(&device, &queue, cursors_w, cursors_h, Some(&cursors_img));
        let checker = TextureBundle::new(&device, &queue, checker_w, checker_h, Some(&draw::CHECKER));
        let paste = TextureBundle::new(&device, &queue, paste_w, paste_h, None);

        // Create screen render target
        let screen_target = RenderTarget::new(
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
        let shape_replace_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            screen_target,
            staging_batch: shape2d::Batch::new(),
            final_batch: shape2d::Batch::new(),
            font,
            cursors,
            checker,
            paste,
            sampler,
            sprite_pipeline,
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

        // Get surface texture
        let output = self.surface.get_current_texture()?;
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        // Prepare draw context
        self.draw_ctx.clear();
        self.draw_ctx.draw(session, avg_frametime, execution);

        let [screen_w, screen_h] = self.screen_target.size;
        let ortho: M44 = ortho_wgpu(screen_w, screen_h, Origin::TopLeft).into();
        let identity: M44 = Matrix4::identity().into();

        // Create vertex buffers for this frame
        let ui_vertices = self.create_shape_vertices(&self.draw_ctx.ui_batch.vertices());
        let text_vertices = self.create_sprite_vertices(&self.draw_ctx.text_batch.vertices());
        let tool_vertices = self.create_sprite_vertices(&self.draw_ctx.tool_batch.vertices());
        let checker_vertices = self.create_sprite_vertices(&self.draw_ctx.checker_batch.vertices());
        let cursor_verts = self.draw_ctx.cursor_sprite.vertices();

        // Create uniform bind group with ortho and identity transform
        let uniforms = TransformUniforms { ortho, transform: identity };
        self.queue.write_buffer(&self.transform_buffer, 0, bytemuck::bytes_of(&uniforms));

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

        // Render to staging target if we have staging shapes or paste preview
        if staging_vertices.is_some() || paste_vertices.is_some() {
            // Create uniform buffer for view ortho
            let view_uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_uniform_buffer"),
                size: std::mem::size_of::<TransformUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            let staging_uniforms = TransformUniforms { ortho: view_ortho, transform: identity };
            view_uniform_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&staging_uniforms));
            view_uniform_buffer.unmap();

            let staging_transform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("staging_transform_bind_group"),
                layout: &self.transform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view_uniform_buffer.as_entire_binding(),
                }],
            });

            // Render to staging target
            let mut staging_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("staging_brush_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_data.staging_target.view,
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
            let final_uniforms = TransformUniforms { ortho: view_ortho, transform: identity };
            view_uniform_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&final_uniforms));
            view_uniform_buffer.unmap();

            let final_transform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    view: &view_data.layer.target.view,
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

        // Render to screen framebuffer
        {
            let bg = Rgba::from(session.settings["background"].to_rgba8());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.screen_target.view,
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
            });

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
                    let view_transform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
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
                                        &view_data.layer.target.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });

                    pass.set_pipeline(&self.sprite_pipeline);
                    pass.set_bind_group(0, &view_transform_bind_group, &[]);
                    pass.set_bind_group(1, &view_texture_bind_group, &[]);
                    pass.set_vertex_buffer(0, view_data.layer.vertex_buffer.slice(..));
                    pass.draw(0..view_data.layer.vertex_count, 0..1);

                    // Also render staging buffer
                    let staging_bind_group =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("staging_bind_group"),
                            layout: &self.texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &view_data.staging_target.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });

                    pass.set_bind_group(1, &staging_bind_group, &[]);
                    pass.draw(0..view_data.layer.vertex_count, 0..1);
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

                                let anim_uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some("anim_uniform_buffer"),
                                    size: std::mem::size_of::<TransformUniforms>() as u64,
                                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                                    mapped_at_creation: true,
                                });
                                anim_uniform_buffer
                                    .slice(..)
                                    .get_mapped_range_mut()
                                    .copy_from_slice(bytemuck::bytes_of(&anim_uniforms));
                                anim_uniform_buffer.unmap();

                                let anim_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("anim_transform_bind_group"),
                                    layout: &self.transform_bind_group_layout,
                                    entries: &[wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: anim_uniform_buffer.as_entire_binding(),
                                    }],
                                });

                                // Bind layer texture for animation
                                let layer_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("anim_layer_bind_group"),
                                    layout: &self.texture_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                &view_data.layer.target.view,
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(&self.sampler),
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
                if let Some((buffer, count)) = self.create_shape_vertices(&help_shape_batch.vertices()) {
                    pass.set_pipeline(&self.shape_pipeline);
                    pass.set_bind_group(0, &transform_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..count, 0..1);
                }

                // Draw help text
                if let Some((buffer, count)) = self.create_sprite_vertices(&help_text_batch.vertices()) {
                    pass.set_pipeline(&self.sprite_pipeline);
                    pass.set_bind_group(0, &transform_bind_group, &[]);
                    pass.set_bind_group(1, &font_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..count, 0..1);
                }
            }
        }

        // Render screen to surface (final pass)
        {
            let screen_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("screen_bind_group"),
                layout: &self.screen_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.screen_target.view),
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
                    view: &surface_view,
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

                // Create cursor uniforms - use screen_target size since cursor positions are in that coordinate space
                let [cursor_w, cursor_h] = self.screen_target.size;
                let cursor_ortho: M44 = ortho_wgpu(cursor_w, cursor_h, Origin::TopLeft).into();
                let ui_scale = session.settings["scale"].to_f64();
                let pixel_ratio = platform::pixel_ratio(self.scale_factor);
                let cursor_uniforms = CursorUniforms {
                    ortho: cursor_ortho,
                    scale: (ui_scale * pixel_ratio) as f32,
                    _padding: [0.0; 7],
                };
                self.queue.write_buffer(&self.cursor_uniform_buffer, 0, bytemuck::bytes_of(&cursor_uniforms));

                let cursor_uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cursor_uniform_bind_group"),
                    layout: &self.cursor_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.cursor_uniform_buffer.as_entire_binding(),
                    }],
                });

                let cursor_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cursor_texture_bind_group"),
                    layout: &self.cursor_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.cursors.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.screen_target.view),
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
                let overlay_vertices = self.create_sprite_vertices(&self.draw_ctx.overlay_batch.vertices());
                if let Some((buffer, count)) = overlay_vertices {
                    // Use BottomLeft ortho for overlay (like GL)
                    let [overlay_w, overlay_h] = self.screen_target.size;
                    let overlay_ortho: M44 = ortho_wgpu(overlay_w, overlay_h, Origin::BottomLeft).into();
                    let overlay_uniforms = TransformUniforms {
                        ortho: overlay_ortho,
                        transform: identity,
                    };

                    let overlay_uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
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

                    let overlay_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_transform_bind_group"),
                        layout: &self.transform_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: overlay_uniform_buffer.as_entire_binding(),
                        }],
                    });

                    pass.set_pipeline(&self.sprite_pipeline);
                    pass.set_bind_group(0, &overlay_bind_group, &[]);
                    pass.set_bind_group(1, &font_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..count, 0..1);
                }
            }
        }

        // Track if we need to create a snapshot (painted this frame or view was resized)
        let needs_snapshot = !self.final_batch.vertices().is_empty();
        let active_view = session.views.active();
        let should_record = active_view
            .filter(|v| v.is_dirty() && (needs_snapshot || v.is_resized()))
            .map(|v| (v.id, v.is_resized()));

        // If we need to record a snapshot, add the copy command to this encoder BEFORE finishing
        // This ensures the copy runs after all render passes that wrote to the layer texture.
        let snapshot_readback = if let Some((view_id, _)) = should_record {
            let view_data = self
                .view_data
                .get(&view_id)
                .expect("view must have associated view data");
            let [w, h] = view_data.layer.target.size;
            let bytes_per_row = (4 * w + 255) & !255; // Align to 256 bytes
            let buffer_size = (bytes_per_row * h) as u64;

            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snapshot_readback_buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &view_data.layer.target.texture,
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

            Some((buffer, w, h, bytes_per_row))
        } else {
            None
        };

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Create snapshot for dirty views: record_view_resized when resized, else record_view_painted
        // The copy was already encoded above, now we just need to map and read
        if let Some((view_id, was_resized)) = should_record {
            if let Some((buffer, w, h, bytes_per_row)) = snapshot_readback {
                let pixels = self.read_buffer_pixels(&buffer, w, h, bytes_per_row);

                if let Some(v) = session.views.get_mut(view_id) {
                    if was_resized {
                        v.resource.record_view_resized(pixels, v.extent());
                    } else {
                        v.resource.record_view_painted(pixels);
                    }
                }
            }
        }

        // Record snapshots if needed
        if !execution.is_normal() {
            let [w, h] = self.screen_target.size;
            let texels = self.read_screen_pixels(w, h);
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
    pub fn handle_resized(&mut self, size: platform::LogicalSize) {
        let physical = size.to_physical(self.scale_factor);

        if physical.width as u32 > 0 && physical.height as u32 > 0 {
            self.surface_config.width = physical.width as u32;
            self.surface_config.height = physical.height as u32;
            self.surface.configure(&self.device, &self.surface_config);
        }

        self.win_size = size;
        self.handle_session_scale_changed(self.scale);
    }

    pub fn handle_session_scale_changed(&mut self, scale: f64) {
        self.scale = scale;
        let w = (self.win_size.width / scale) as u32;
        let h = (self.win_size.height / scale) as u32;

        if w > 0 && h > 0 {
            self.screen_target.resize(&self.device, w, h, wgpu::TextureFormat::Rgba8UnormSrgb);
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
                        let (w, h) = (s.width(), s.height());
                        self.view_data.insert(
                            id,
                            ViewData::new(&self.device, &self.queue, w, h, Some(pixels)),
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
                    let view_data = self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data");
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("clear_encoder"),
                        });
                    {
                        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("clear_view"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view_data.layer.target.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: color.r as f64 / 255.0,
                                        g: color.g as f64 / 255.0,
                                        b: color.b as f64 / 255.0,
                                        a: color.a as f64 / 255.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    }
                    self.queue.submit(std::iter::once(encoder.finish()));
                }
                ViewOp::Blit(src, dst) => {
                    // Get pixels from the view's CPU snapshot
                    if let Some((_, pixels)) = v.resource.layer.get_snapshot_rect(&src.map(|n| n as i32)) {
                        let view_data = self
                            .view_data
                            .get_mut(&v.id)
                            .expect("views must have associated view data");
                        let texels = util::align_u8(&pixels);
                        view_data.layer.upload_part(
                            &self.queue,
                            [dst.x1 as u32, dst.y1 as u32],
                            [src.width() as u32, src.height() as u32],
                            texels,
                        );
                    }
                }
                ViewOp::Yank(src) => {
                    // Get pixels from the view's CPU snapshot (like GL)
                    if let Some((_, pixels)) = v.resource.layer.get_snapshot_rect(&src.map(|n| n as i32)) {
                        let (w, h) = (src.width() as u32, src.height() as u32);

                        // Resize paste texture if needed
                        let [paste_w, paste_h] = self.paste.size;
                        if paste_w != w || paste_h != h {
                            self.paste.resize(&self.device, w, h);
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
                    // Get pixels from the view's CPU snapshot and flip in CPU
                    if let Some((_, mut pixels)) = v.resource.layer.get_snapshot_rect(&src.map(|n| n as i32)) {
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
                            self.paste.resize(&self.device, w, h);
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
                                size: (sprite_vertices.len() * std::mem::size_of::<Sprite2dVertex>()) as u64,
                                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: true,
                            });
                            buffer
                                .slice(..)
                                .get_mapped_range_mut()
                                .copy_from_slice(bytemuck::cast_slice(&sprite_vertices));
                            buffer.unmap();

                            self.paste_outputs.push((buffer, sprite_vertices.len() as u32));
                        }
                    }
                }
                ViewOp::SetPixel(rgba, x, y) => {
                    let view_data = self
                        .view_data
                        .get_mut(&v.id)
                        .expect("views must have associated view data");
                    let texels = &[rgba.r, rgba.g, rgba.b, rgba.a];
                    view_data.layer.upload_part(
                        &self.queue,
                        [*x as u32, *y as u32],
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
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_staging"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_data.staging_target.view,
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
        let (ew, eh) = {
            let extent = view.resource.extent;
            (extent.width(), extent.height())
        };

        let tw = u32::min(ew, vw);
        let th = u32::min(eh, vh);

        let view_data = ViewData::new(&self.device, &self.queue, vw, vh, None);

        if let Some((_, texels)) = view.layer.get_snapshot_rect(&Rect::origin(tw as i32, th as i32))
        {
            let texels = util::align_u8(&texels);
            view_data.layer.upload_part(
                &self.queue,
                [0, vh - th],
                [tw, th],
                texels,
            );
        }

        self.view_data.insert(view.id, view_data);
        Ok(())
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
                        size: (sprite_vertices.len() * std::mem::size_of::<Sprite2dVertex>()) as u64,
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
                        size: (sprite_vertices.len() * std::mem::size_of::<Sprite2dVertex>()) as u64,
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
    fn create_sprite_vertices(
        &self,
        vertices: &[sprite2d::Vertex],
    ) -> Option<(wgpu::Buffer, u32)> {
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
    fn create_shape_vertices(
        &self,
        vertices: &[shape2d::Vertex],
    ) -> Option<(wgpu::Buffer, u32)> {
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
        self.read_texture_pixels(&self.screen_target.texture, width, height)
    }

    /// Read pixels from a buffer that has already been filled via copy_texture_to_buffer.
    /// This assumes the copy command has already been submitted.
    fn read_buffer_pixels(&self, buffer: &wgpu::Buffer, width: u32, height: u32, bytes_per_row: u32) -> Vec<Rgba8> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            let row_start = (y * bytes_per_row) as usize;
            for x in 0..width {
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
        let mut pixels = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            let row_start = (y * bytes_per_row) as usize;
            for x in 0..width {
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

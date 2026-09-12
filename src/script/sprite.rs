//! Built-in sprite drawing shared by all plugin passes.
use super::*;
use std::collections::HashMap;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

pub(super) struct Blitter {
    device: Arc<wgpu::Device>,
    transform_bgl: Arc<wgpu::BindGroupLayout>,
    texture_bgl: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
    pipelines: Mutex<HashMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>>,
}

impl Blitter {
    pub(super) fn new(
        device: Arc<wgpu::Device>,
        transform_bgl: Arc<wgpu::BindGroupLayout>,
        texture_bgl: Arc<wgpu::BindGroupLayout>,
        sampler: Arc<wgpu::Sampler>,
    ) -> Self {
        Self {
            device,
            transform_bgl,
            texture_bgl,
            sampler,
            pipelines: Mutex::new(HashMap::new()),
        }
    }

    fn pipeline(&self, format: wgpu::TextureFormat, replace: bool) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("builtin_sprite"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../wgpu/data/sprite.wgsl").into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("builtin_sprite"),
                bind_group_layouts: &[&self.transform_bgl, &self.texture_bgl],
                push_constant_ranges: &[],
            });
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("builtin_sprite"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ScriptVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Unorm8x4, 3 => Float32],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(if replace { wgpu::BlendState::REPLACE } else { wgpu::BlendState::ALPHA_BLENDING }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        })
    }

    pub(super) fn draw(
        &self,
        pass: &ScriptPass,
        source: &ScriptTextureView,
        options: &rune::runtime::Object,
    ) -> Result<(), String> {
        source.get()?;
        let sprite = SpriteOptions::parse_with_fields(
            options,
            source.size[0],
            source.size[1],
            &["transform", "blend"],
        )?;
        let transform = match options.get("transform") {
            Some(value) => *value
                .borrow_ref::<Mat4>()
                .map_err(|_| "`transform` must be a Mat4")?,
            None => Mat4(crate::gfx::math::Matrix4::identity()),
        };
        let replace = match options.get("blend") {
            Some(value) => match &*value
                .borrow_string_ref()
                .map_err(|_| "`blend` must be a string")?
            {
                "alpha" => false,
                "replace" => true,
                _ => return Err("`blend` must be `alpha` or `replace`".into()),
            },
            None => false,
        };
        for rect in [sprite.src, sprite.dst] {
            if [rect.x1, rect.y1, rect.x2, rect.y2]
                .iter()
                .any(|n| !n.is_finite() || !(*n as f32).is_finite())
            {
                return Err("sprite rectangles must contain finite coordinates".into());
            }
        }
        if !sprite.opacity.is_finite() || !(0.0..=1.0).contains(&sprite.opacity) {
            return Err("`opacity` must be between 0.0 and 1.0".into());
        }
        let matrix: [[f32; 4]; 4] = transform.0.into();
        if matrix.iter().flatten().any(|n| !n.is_finite()) {
            return Err("`transform` must contain finite values".into());
        }
        if pass.size.contains(&0) {
            return Err("sprite target must have nonzero dimensions".into());
        }
        let ortho = crate::gfx::math::Matrix4::ortho(
            pass.size[0],
            pass.size[1],
            crate::gfx::Origin::TopLeft,
        );
        let uniforms = ScriptUniforms {
            ortho: ortho.into(),
            transform: matrix,
        };
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sprite_transform"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sprite_params"),
                contents: &[0; 16],
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let transform_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_transform"),
            layout: &self.transform_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let texture_group =
            texture_bind_group(&self.device, &self.texture_bgl, &self.sampler, source)?;
        let mut batch = crate::gfx::sprite2d::Batch::new(source.size[0], source.size[1]);
        let rect = |r: Rect| {
            crate::gfx::rect::Rect::new(r.x1 as f32, r.y1 as f32, r.x2 as f32, r.y2 as f32)
        };
        batch.add(
            rect(sprite.src),
            rect(sprite.dst),
            crate::gfx::ZDepth::default(),
            sprite.color.into(),
            sprite.opacity as f32,
            crate::gfx::Repeat::default(),
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
        let vertices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sprite_vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut pipelines = self.pipelines.lock().expect("sprite pipeline cache lock");
        let pipeline = pipelines
            .entry((pass.format, replace))
            .or_insert_with(|| self.pipeline(pass.format, replace));
        pass.with(|p| {
            p.set_pipeline(pipeline);
            p.set_bind_group(0, &transform_group, &[]);
            p.set_bind_group(1, &texture_group.bind_group, &[]);
            p.set_vertex_buffer(0, vertices.slice(..));
            p.draw(0..6, 0..1);
        })
    }
}

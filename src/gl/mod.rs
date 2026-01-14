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
use crate::view::{View, ViewExtent, ViewId, ViewOp, ViewState};
use crate::{data, data::Assets, image};

use crate::gfx::{Origin, Point, Rgba, Rgba8, ZDepth, color, shape2d, sprite2d};
use crate::gfx::{Matrix4, Rect, Repeat, Vector2};

use luminance::context::GraphicsContext;
use luminance::depth_test::DepthComparison;
use luminance::framebuffer::Framebuffer;
use luminance::pipeline::{PipelineState, TextureBinding};
use luminance::pixel;
use luminance::render_state::RenderState;
use luminance::shader::{Program, Uniform};
use luminance::tess::{Mode, Tess, TessBuilder};
use luminance::texture::{Dim2, GenMipmaps, MagFilter, MinFilter, Sampler, Texture, Wrap};
use luminance::{
    blending::{self, Equation, Factor},
    pipeline::PipelineError,
};

use luminance_derive::{Semantics, UniformInterface, Vertex};
use luminance_gl::gl33;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::io;
use std::mem;
use std::time;

type Backend = gl33::GL33;
type M44 = [[f32; 4]; 4];

const SAMPLER: Sampler = Sampler {
    wrap_r: Wrap::Repeat,
    wrap_s: Wrap::Repeat,
    wrap_t: Wrap::Repeat,
    min_filter: MinFilter::Nearest,
    mag_filter: MagFilter::Nearest,
    depth_comparison: None,
};

#[derive(UniformInterface)]
struct Sprite2dInterface {
    tex: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    ortho: Uniform<M44>,
    transform: Uniform<M44>,
}

#[derive(Copy, Clone, Debug, Semantics)]
pub enum VertexSemantics {
    #[sem(name = "position", repr = "[f32; 3]", wrapper = "VertexPosition")]
    Position,
    #[sem(name = "uv", repr = "[f32; 2]", wrapper = "VertexUv")]
    Uv,
    #[sem(name = "color", repr = "[u8; 4]", wrapper = "VertexColor")]
    Color,
    #[sem(name = "opacity", repr = "[f32; 1]", wrapper = "VertexOpacity")]
    Opacity,
    #[sem(name = "angle", repr = "[f32; 1]", wrapper = "VertexAngle")]
    Angle,
    #[sem(name = "center", repr = "[f32; 2]", wrapper = "VertexCenter")]
    Center,
}

#[repr(C)]
#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
#[rustfmt::skip]
struct Sprite2dVertex {
    #[allow(dead_code)] position: VertexPosition,
    #[allow(dead_code)] uv: VertexUv,
    #[vertex(normalized = "true")]
    #[allow(dead_code)] color: VertexColor,
    #[allow(dead_code)] opacity: VertexOpacity,
}

////////////////////////////////////////////////////////////

#[derive(UniformInterface)]
struct Shape2dInterface {
    ortho: Uniform<M44>,
    transform: Uniform<M44>,
}

#[repr(C)]
#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
#[rustfmt::skip]
struct Shape2dVertex {
    #[allow(dead_code)] position: VertexPosition,
    #[allow(dead_code)] angle: VertexAngle,
    #[allow(dead_code)] center: VertexCenter,
    #[vertex(normalized = "true")]
    #[allow(dead_code)] color: VertexColor,
}

#[derive(UniformInterface)]
struct Cursor2dInterface {
    cursor: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    framebuffer: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    ortho: Uniform<M44>,
    scale: Uniform<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Vertex)]
#[vertex(sem = "VertexSemantics")]
#[rustfmt::skip]
struct Cursor2dVertex {
    #[allow(dead_code)] position: VertexPosition,
    #[allow(dead_code)] uv: VertexUv,
}

#[derive(UniformInterface)]
struct Screen2dInterface {
    framebuffer: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
}

#[derive(UniformInterface)]
struct Lookuptex2dInterface {
    tex: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    ltex: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    ltexim: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    // ltexreg: Uniform<V2>,
    lt_tfw: Uniform<u32>,
    frame_mask: Uniform<u32>,
    ortho: Uniform<M44>,
    transform: Uniform<M44>,
}

#[derive(UniformInterface)]
struct Lookupmap2dInterface {
    ortho: Uniform<M44>,
    transform: Uniform<M44>,
    tex: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    lt_tfw: Uniform<u32>,
}

#[derive(UniformInterface)]
struct Lookupquery2dInterface {
    ltexim: Uniform<TextureBinding<Dim2, pixel::NormUnsigned>>,
    pixel_coords: Uniform<[i32; 2]>,
}

pub struct Renderer {
    pub win_size: LogicalSize,

    ctx: Context,
    draw_ctx: draw::Context,
    scale_factor: f64,
    scale: f64,
    present_fb: Framebuffer<Backend, Dim2, (), ()>,
    screen_fb: Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F>,
    render_st: RenderState,
    pipeline_st: PipelineState,
    blending: Blending,

    staging_batch: shape2d::Batch,
    final_batch: shape2d::Batch,

    font: Texture<Backend, Dim2, pixel::SRGBA8UI>,
    cursors: Texture<Backend, Dim2, pixel::SRGBA8UI>,
    checker: Texture<Backend, Dim2, pixel::SRGBA8UI>,
    paste: Texture<Backend, Dim2, pixel::SRGBA8UI>,
    paste_outputs: Vec<Tess<Backend, Sprite2dVertex>>,

    sprite2d: Program<Backend, VertexSemantics, (), Sprite2dInterface>,
    shape2d: Program<Backend, VertexSemantics, (), Shape2dInterface>,
    cursor2d: Program<Backend, VertexSemantics, (), Cursor2dInterface>,
    screen2d: Program<Backend, VertexSemantics, (), Screen2dInterface>,
    lookuptex2d: Program<Backend, VertexSemantics, (), Lookuptex2dInterface>,
    lookupmap2d: Program<Backend, VertexSemantics, (), Lookupmap2dInterface>,
    lookupquery2d: Program<Backend, VertexSemantics, (), Lookupquery2dInterface>,

    view_data: BTreeMap<ViewId, RefCell<ViewData>>,
}

struct LayerData {
    fb: RefCell<Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F>>,
    lt_im: RefCell<Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F>>,
    lookup_anim_fb: RefCell<Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F>>,
    lt_tfw: u32,
    lt_tfh: u32,
    w: u32,
    _h: u32,
    tess: Tess<Backend, Sprite2dVertex>,
    lt_tess: Tess<Backend, ()>,
}

impl LayerData {
    fn new(w: u32, h: u32, tfw: u32, tfh: u32, pixels: Option<&[Rgba8]>, ctx: &mut Context) -> Self {
        println!("w: {}, h: {}, tfw: {}, tfh: {}", w, h, tfw, tfh);
        println!("view orth");
        println!("{}", Matrix4::ortho(4096, 4096, Origin::BottomLeft));
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
            .map(|v| unsafe { mem::transmute(*v) })
            .collect();
        let tess = TessBuilder::new(ctx)
            .set_vertices(verts)
            .set_mode(Mode::Triangle)
            .build()
            .unwrap();

        let lt_tess = TessBuilder::new(ctx)
            .set_vertex_nb((h * w) as usize)
            .set_mode(Mode::Point)
            .build()
            .unwrap();

        let mut fb: Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F> =
            Framebuffer::new(ctx, [w, h], 0, self::SAMPLER).unwrap();
        let mut lt_im: Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F> =
            Framebuffer::new(ctx, [256 * 16, 256 * 16], 0, self::SAMPLER).unwrap();
                    // Create intermediate framebuffer for lookup animation output (sized to full spritesheet)
        let mut lookup_anim_fb: Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F> =
        Framebuffer::new(ctx, [w, h], 0, self::SAMPLER).unwrap();
    
        lookup_anim_fb
            .color_slot()
            .clear(GenMipmaps::No, (0, 0, 0, 0))
            .unwrap();

        fb.color_slot().clear(GenMipmaps::No, (0, 0, 0, 0)).unwrap();
        lt_im
            .color_slot()
            .clear(GenMipmaps::No, (0, 0, 0, 0))
            .unwrap();

        if let Some(pixels) = pixels {
            let aligned = util::align_u8(pixels);
            fb.color_slot().upload_raw(GenMipmaps::No, aligned).unwrap();
        }

        Self {
            fb: RefCell::new(fb),
            lt_im: RefCell::new(lt_im),
            lt_tfw: tfw,
            lt_tfh: tfh,
            w,
            _h: h,
            tess,
            lt_tess,
            lookup_anim_fb: RefCell::new(lookup_anim_fb),
        }
    }

    fn clear(&mut self) -> Result<(), RendererError> {
        self.fb
            .borrow_mut()
            .color_slot()
            .clear(GenMipmaps::No, (0, 0, 0, 0))
            .map_err(RendererError::Texture)
    }

    fn upload_part(
        &mut self,
        offset: [u32; 2],
        size: [u32; 2],
        texels: &[u8],
    ) -> Result<(), RendererError> {
        self.fb
            .borrow_mut()
            .color_slot()
            .upload_part_raw(GenMipmaps::No, offset, size, texels)
            .map_err(RendererError::Texture)
    }

    fn upload(&mut self, texels: &[u8]) -> Result<(), RendererError> {
        self.fb
            .borrow_mut()
            .color_slot()
            .upload_raw(GenMipmaps::No, texels)
            .map_err(RendererError::Texture)
    }

    fn pixels(&mut self) -> Vec<Rgba8> {
        let texels = self
            .fb
            .borrow_mut()
            .color_slot()
            .get_raw_texels()
            .expect("getting raw texels never fails");
        Rgba8::align(&texels).to_vec()
    }

    fn lookup_anim_pixels(&mut self) -> Vec<Rgba8> {
        let texels = self
            .lookup_anim_fb
            .borrow_mut()
            .color_slot()
            .get_raw_texels()
            .expect("getting raw texels never fails");
        Rgba8::align(&texels).to_vec()
    }
}

struct ViewData {
    layer: LayerData,
    staging_fb: Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F>,
    anim_tess: Option<Tess<Backend, Sprite2dVertex>>,
    anim_lt_tess: Option<Tess<Backend, Sprite2dVertex>>,
    lt_fb_tess: Option<Tess<Backend, Sprite2dVertex>>,
    lookup_layer_tess: Vec<(ViewId, Tess<Backend, Sprite2dVertex>)>,
    layer_tess: Option<Tess<Backend, Sprite2dVertex>>,
}

impl ViewData {
    fn new(w: u32, h: u32, tfw: u32, tfh: u32, pixels: Option<&[Rgba8]>, ctx: &mut Context) -> Self {
        let mut staging_fb: Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F> =
            Framebuffer::new(ctx, [w, h], 0, self::SAMPLER).unwrap();

        staging_fb
            .color_slot()
            .clear(GenMipmaps::No, (0, 0, 0, 0))
            .unwrap();

        Self {
            layer: LayerData::new(w, h, tfw, tfh, pixels, ctx),
            staging_fb,
            anim_tess: None,
            anim_lt_tess: None,
            lt_fb_tess: None,
            lookup_layer_tess: Vec::new(),
            layer_tess: None,
        }
    }
}

struct Context {
    ctx: Backend,
}

unsafe impl GraphicsContext for Context {
    type Backend = self::Backend;

    fn backend(&mut self) -> &mut Self::Backend {
        &mut self.ctx
    }
}

impl Context {
    fn program<T>(&mut self, vert: &str, frag: &str) -> Program<Backend, VertexSemantics, (), T>
    where
        T: luminance::shader::UniformInterface<Backend>,
    {
        self.new_shader_program()
            .from_strings(vert, None, None, frag)
            .unwrap()
            .ignore_warnings()
    }

    fn tessellation<T, S>(&mut self, verts: &[T]) -> Tess<Backend, S>
    where
        S: luminance::vertex::Vertex + Sized,
    {
        let (head, body, tail) = unsafe { verts.align_to::<S>() };

        assert!(head.is_empty());
        assert!(tail.is_empty());

        TessBuilder::new(self)
            .set_vertices(body)
            .set_mode(Mode::Triangle)
            .build()
            .unwrap()
    }
}

#[derive(Debug)]
pub enum RendererError {
    Initialization,
    Texture(luminance::texture::TextureError),
    Framebuffer(luminance::framebuffer::FramebufferError),
    Pipeline(luminance::pipeline::PipelineError),
    State(luminance_gl::gl33::StateQueryError),
}

impl From<luminance::pipeline::PipelineError> for RendererError {
    fn from(other: luminance::pipeline::PipelineError) -> Self {
        Self::Pipeline(other)
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
            Self::Initialization => write!(f, "initialization error"),
            Self::Texture(e) => write!(f, "texture error: {}", e),
            Self::Framebuffer(e) => write!(f, "framebuffer error: {}", e),
            Self::Pipeline(e) => write!(f, "pipeline error: {}", e),
            Self::State(e) => write!(f, "state error: {}", e),
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

impl<'a> renderer::Renderer<'a> for Renderer {
    type Error = RendererError;

    fn new(
        win: &mut platform::backend::Window,
        win_size: LogicalSize,
        scale_factor: f64,
        assets: Assets<'a>,
    ) -> io::Result<Self> {
        use RendererError as Error;

        gl::load_with(|s| win.get_proc_address(s) as *const _);

        let ctx = Backend::new().map_err(Error::State)?;
        let mut ctx = Context { ctx };

        let (font_img, font_w, font_h) = image::read(assets.glyphs)?;
        let (cursors_img, cursors_w, cursors_h) = image::read(data::CURSORS)?;
        let (checker_w, checker_h) = (2, 2);
        let (paste_w, paste_h) = (8, 8);

        let mut font =
            Texture::new(&mut ctx, [font_w, font_h], 0, self::SAMPLER).map_err(Error::Texture)?;
        let mut cursors = Texture::new(&mut ctx, [cursors_w, cursors_h], 0, self::SAMPLER)
            .map_err(Error::Texture)?;
        let paste =
            Texture::new(&mut ctx, [paste_w, paste_h], 0, self::SAMPLER).map_err(Error::Texture)?;
        let mut checker = Texture::new(&mut ctx, [checker_w, checker_h], 0, self::SAMPLER)
            .map_err(Error::Texture)?;

        font.upload_raw(GenMipmaps::No, &font_img)
            .map_err(Error::Texture)?;
        cursors
            .upload_raw(GenMipmaps::No, &cursors_img)
            .map_err(Error::Texture)?;
        checker
            .upload_raw(GenMipmaps::No, &draw::CHECKER)
            .map_err(Error::Texture)?;

        let sprite2d = ctx.program::<Sprite2dInterface>(
            include_str!("data/sprite.vert"),
            include_str!("data/sprite.frag"),
        );
        let shape2d = ctx.program::<Shape2dInterface>(
            include_str!("data/shape.vert"),
            include_str!("data/shape.frag"),
        );
        let cursor2d = ctx.program::<Cursor2dInterface>(
            include_str!("data/cursor.vert"),
            include_str!("data/cursor.frag"),
        );
        let screen2d = ctx.program::<Screen2dInterface>(
            include_str!("data/screen.vert"),
            include_str!("data/screen.frag"),
        );
        let lookuptex2d = ctx.program::<Lookuptex2dInterface>(
            include_str!("data/lookuptex.vert"),
            include_str!("data/lookuptex.frag"),
        );
        let lookupmap2d = ctx.program::<Lookupmap2dInterface>(
            include_str!("data/lookupmap.vert"),
            include_str!("data/lookupmap.frag"),
        );

        let lookupquery2d = ctx.program::<Lookupquery2dInterface>(
            include_str!("data/lookupquery.vert"),
            include_str!("data/lookupquery.frag"),
        );

        let physical = win_size.to_physical(scale_factor);
        let present_fb =
            Framebuffer::back_buffer(&mut ctx, [physical.width as u32, physical.height as u32])
                .map_err(Error::Framebuffer)?;
        let screen_fb = Framebuffer::new(
            &mut ctx,
            [win_size.width as u32, win_size.height as u32],
            0,
            self::SAMPLER,
        )
        .map_err(Error::Framebuffer)?;

        let render_st = RenderState::default()
            .set_blending(blending::Blending {
                equation: Equation::Additive,
                src: Factor::SrcAlpha,
                dst: Factor::SrcAlphaComplement,
            })
            .set_depth_test(Some(DepthComparison::LessOrEqual));
        let pipeline_st = PipelineState::default()
            .set_clear_color([0., 0., 0., 0.])
            .enable_srgb(true)
            .enable_clear_depth(true)
            .enable_clear_color(true);

        let draw_ctx = draw::Context {
            ui_batch: shape2d::Batch::new(),
            text_batch: self::text_batch(font.size()),
            overlay_batch: self::text_batch(font.size()),
            cursor_sprite: sprite::Sprite::new(cursors_w, cursors_h),
            tool_batch: sprite2d::Batch::new(cursors_w, cursors_h),
            paste_batch: sprite2d::Batch::new(paste_w, paste_h),
            checker_batch: sprite2d::Batch::new(checker_w, checker_h),
        };

        Ok(Renderer {
            ctx,
            draw_ctx,
            win_size,
            scale_factor,
            scale: 1.0,
            blending: Blending::Alpha,
            present_fb,
            screen_fb,
            render_st,
            pipeline_st,
            sprite2d,
            shape2d,
            cursor2d,
            screen2d,
            lookuptex2d,
            lookupmap2d,
            lookupquery2d,
            font,
            cursors,
            checker,
            paste,
            paste_outputs: Vec::new(),
            staging_batch: shape2d::Batch::new(),
            final_batch: shape2d::Batch::new(),
            view_data: BTreeMap::new(),
        })
    }

    fn init(&mut self, effects: Vec<Effect>, session: &mut Session) {
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

        self.handle_effects(effects, session).unwrap();
        self.update_view_animations(session);
        self.update_view_composites(session);

        let [screen_w, screen_h] = self.screen_fb.size();
        let ortho: M44 = Matrix4::ortho(screen_w, screen_h, Origin::TopLeft).into();
        let identity: M44 = Matrix4::identity().into();

        let Self {
            draw_ctx,
            font,
            cursors,
            checker,
            sprite2d,
            shape2d,
            cursor2d,
            screen2d,
            lookuptex2d,
            lookupmap2d,
            scale_factor,
            present_fb,
            blending,
            screen_fb,
            render_st,
            pipeline_st,
            paste,
            paste_outputs,
            view_data,
            ..
        } = self;

        draw_ctx.clear();
        draw_ctx.draw(session, avg_frametime, execution);

        let text_tess = self
            .ctx
            .tessellation::<_, Sprite2dVertex>(&draw_ctx.text_batch.vertices());
        let overlay_tess = self
            .ctx
            .tessellation::<_, Sprite2dVertex>(&draw_ctx.overlay_batch.vertices());
        let ui_tess = self
            .ctx
            .tessellation::<_, Shape2dVertex>(&draw_ctx.ui_batch.vertices());
        let tool_tess = self
            .ctx
            .tessellation::<_, Sprite2dVertex>(&draw_ctx.tool_batch.vertices());
        let cursor_tess = self
            .ctx
            .tessellation::<_, Cursor2dVertex>(&draw_ctx.cursor_sprite.vertices());
        let checker_tess = self
            .ctx
            .tessellation::<_, Sprite2dVertex>(&draw_ctx.checker_batch.vertices());
        let screen_tess = TessBuilder::<Backend, ()>::new(&mut self.ctx)
            .set_vertex_nb(6)
            .set_mode(Mode::Triangle)
            .build()
            .unwrap();

        let paste_tess = if draw_ctx.paste_batch.is_empty() {
            None
        } else {
            Some(
                self.ctx
                    .tessellation::<_, Sprite2dVertex>(&draw_ctx.paste_batch.vertices()),
            )
        };
        let staging_tess = if self.staging_batch.is_empty() {
            None
        } else {
            Some(
                self.ctx
                    .tessellation::<_, Shape2dVertex>(&self.staging_batch.vertices()),
            )
        };
        let final_tess = if self.final_batch.is_empty() {
            None
        } else {
            let final_vertices = self.final_batch.clone().vertices();
            Some(self.ctx.tessellation::<_, Shape2dVertex>(&final_vertices))
        };

        let help_tess = if session.mode == session::Mode::Help {
            let mut win = shape2d::Batch::new();
            let mut text = self::text_batch(font.size());
            draw::draw_help(session, &mut text, &mut win);

            let win_tess = self
                .ctx
                .tessellation::<_, Shape2dVertex>(win.vertices().as_slice());
            let text_tess = self
                .ctx
                .tessellation::<_, Sprite2dVertex>(text.vertices().as_slice());
            Some((win_tess, text_tess))
        } else {
            None
        };

        // Precompute miniview tessellation and geometry
        let mut miniview_tess: Option<Tess<Backend, Sprite2dVertex>> = None;
        let mut miniview_id_and_pos: Option<(ViewId, f32, f32)> = None;
        if let Some((mini_id, miniview)) = &session.miniview {
            if let Some(mini_view) = session.views.get(*mini_id) {
                let dst_x = miniview.offset.x;
                let dst_y = miniview.offset.y;
                let target_w = mini_view.width() as f32 * miniview.zoom;
                let target_h = mini_view.fh as f32 * miniview.zoom;

                let batch = sprite2d::Batch::singleton(
                    mini_view.width(),
                    mini_view.fh,
                    Rect::origin(mini_view.width() as f32, mini_view.fh as f32),
                    Rect::new(dst_x, dst_y, dst_x + target_w, dst_y + target_h),
                    draw::VIEW_LAYER,
                    Rgba::TRANSPARENT,
                    1.0,
                    Repeat::default(),
                );
                miniview_tess = Some(self.ctx.tessellation::<_, Sprite2dVertex>(&batch.vertices()));
                miniview_id_and_pos = Some((*mini_id, dst_x, dst_y));
            }
        }

        let mut builder = self.ctx.new_pipeline_gate();
        let v = session
            .views
            .active()
            .expect("there must always be an active view");
        let view_ortho = Matrix4::ortho(v.width(), v.fh, Origin::TopLeft);
        let view_ortho_lookup = Matrix4::ortho(4096, 4096, Origin::BottomLeft);

        {
            let v_data = view_data.get(&v.id).unwrap().borrow();
            let l_data = &v_data.layer;

            // Render to view staging buffer.
            builder.pipeline::<PipelineError, _, _, _, _>(
                &v_data.staging_fb,
                pipeline_st,
                |pipeline, mut shd_gate| {
                    // Render staged brush strokes.
                    if let Some(tess) = staging_tess {
                        shd_gate.shade(shape2d, |mut iface, uni, mut rdr_gate| {
                            iface.set(&uni.ortho, view_ortho.into());
                            iface.set(&uni.transform, identity);

                            rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&tess))
                        })?;
                    }
                    // Render staging paste buffer.
                    if let Some(tess) = paste_tess {
                        let bound_paste = pipeline
                            .bind_texture(paste)
                            .expect("binding textures never fails. qed.");
                        shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                            iface.set(&uni.ortho, view_ortho.into());
                            iface.set(&uni.transform, identity);
                            iface.set(&uni.tex, bound_paste.binding());

                            rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&tess))
                        })?;
                    }
                    Ok(())
                },
            );

            // Render to view final buffer.
            builder.pipeline::<PipelineError, _, _, _, _>(
                &*l_data.fb.borrow(),
                &pipeline_st.clone().enable_clear_color(false),
                |pipeline, mut shd_gate| {
                    let bound_paste = pipeline
                        .bind_texture(paste)
                        .expect("binding textures never fails. qed.");

                    // Render final brush strokes.
                    if let Some(tess) = final_tess {
                        shd_gate.shade(shape2d, |mut iface, uni, mut rdr_gate| {
                            iface.set(&uni.ortho, view_ortho.into());
                            iface.set(&uni.transform, identity);

                            let render_st = if blending == &Blending::Constant {
                                render_st.clone().set_blending(blending::Blending {
                                    equation: Equation::Additive,
                                    src: Factor::One,
                                    dst: Factor::Zero,
                                })
                            } else {
                                render_st.clone()
                            };

                            rdr_gate.render(&render_st, |mut tess_gate| tess_gate.render(&tess))
                        })?;
                    }
                    if !paste_outputs.is_empty() {
                        shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                            iface.set(&uni.ortho, view_ortho.into());
                            iface.set(&uni.transform, identity);
                            iface.set(&uni.tex, bound_paste.binding());

                            for out in paste_outputs.drain(..) {
                                rdr_gate
                                    .render(render_st, |mut tess_gate| tess_gate.render(&out))?;
                            }
                            Ok(())
                        })?;
                    }
                    Ok(())
                },
            );

            // Render to lookup texture intermediate map buffer.
            builder.pipeline::<PipelineError, _, _, _, _>(
                &*l_data.lt_im.borrow(),
                pipeline_st,
                |pipeline, mut shd_gate| {
                    let mut fb = l_data.fb.borrow_mut();
                    let bound_lt_im = pipeline
                        .bind_texture(fb.color_slot())
                        .expect("binding textures never fails. qed.");

                    shd_gate.shade(lookupmap2d, |mut iface, uni, mut rdr_gate| {
                        iface.set(&uni.tex, bound_lt_im.binding());
                        iface.set(&uni.ortho, view_ortho_lookup.into());
                        iface.set(&uni.transform, identity);
                        iface.set(&uni.lt_tfw, l_data.lt_tfw);

                        rdr_gate
                            .render(render_st, |mut tess_gate| tess_gate.render(&l_data.lt_tess))
                    })?;
                    Ok(())
                },
            );
        }

        // Render lookup texture animations to intermediate framebuffers
        if session.settings["animation"].is_set() {
            let frame_mask = session.settings["lookup/framemask"].to_u64() as u32;
            for (id, _rcv) in view_data.iter() {
                let v = view_data.get(&id).unwrap();
                let view = session.views.get(*id).unwrap();
                let Some(ltid) = view.lookuptexture() else {
                    continue;
                };
                let rcltv = view_data.get(&ltid).unwrap();
                // if let Some(tess) = rcltv.borrow().lt_fb_tess {
                // Render to intermediate framebuffer
                builder.pipeline::<PipelineError, _, _, _, _>(
                    &*v.borrow().layer.lookup_anim_fb.borrow(),
                    pipeline_st,
                    |pipeline, mut shd_gate| {
                        let v_inner = v.borrow();
                        let Some(tess) = &v_inner.lt_fb_tess else {
                            return Ok(());
                        };
                        let lookup_anim_ortho: M44 = Matrix4::ortho(v_inner.layer.w, v_inner.layer._h, Origin::TopLeft).into();
                        
                        let mut main_fb = v_inner.layer.fb.borrow_mut();
                        let bound_layer = pipeline
                            .bind_texture(main_fb.color_slot())
                            .expect("binding textures never fails");
                        
                        let ltv_inner = rcltv.borrow();
                        let mut lookup_fb = ltv_inner.layer.fb.borrow_mut();
                        let mut lookup_lt_im = ltv_inner.layer.lt_im.borrow_mut();
                        let lookup_layer = pipeline
                            .bind_texture(lookup_fb.color_slot())
                            .expect("binding textures never fails");
                        let lookup_map = pipeline
                            .bind_texture(lookup_lt_im.color_slot())
                            .expect("binding textures never fails");
                        
                        shd_gate.shade(lookuptex2d, |mut iface, uni, mut rdr_gate| {
                            iface.set(&uni.ortho, lookup_anim_ortho);
                            iface.set(&uni.transform, identity);
                            iface.set(&uni.tex, bound_layer.binding());
                            iface.set(&uni.ltex, lookup_layer.binding());
                            iface.set(&uni.ltexim, lookup_map.binding());
                            iface.set(&uni.lt_tfw, ltv_inner.layer.lt_tfw);
                            iface.set(&uni.frame_mask, frame_mask);
                            
                            rdr_gate.render(render_st, |mut tess_gate| {
                                tess_gate.render(tess)
                            })
                        })?;
                        
                        Ok(())
                    },
                );
            }
        }

        // Render to screen framebuffer.
        let bg = Rgba::from(session.settings["background"].to_rgba8());
        let screen_st = &pipeline_st
            .clone()
            .set_clear_color([bg.r, bg.g, bg.b, bg.a]);
        builder.pipeline::<PipelineError, _, _, _, _>(
            screen_fb,
            screen_st,
            |pipeline, mut shd_gate| {
                // Draw view checkers to screen framebuffer.
                if session.settings["checker"].is_set() {
                    shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                        let bound_checker = pipeline
                            .bind_texture(checker)
                            .expect("binding textures never fails");

                        iface.set(&uni.ortho, ortho);
                        iface.set(&uni.transform, identity);
                        iface.set(&uni.tex, bound_checker.binding());

                        rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&checker_tess))
                    })?;
                }

                for (id, rcv) in view_data.iter() {
                    let mut v = rcv.borrow_mut();
                    if let Some(view) = session.views.get(*id) {
                        let transform =
                            Matrix4::from_translation(
                                (session.offset + view.offset).extend(*draw::VIEW_LAYER),
                            ) * Matrix4::from_nonuniform_scale(view.zoom, view.zoom, 1.0);

                        // Render views.
                        shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                            let mut fb = v.layer.fb.borrow_mut();
                            let bound_view = pipeline
                                .bind_texture(fb.color_slot())
                                .expect("binding textures never fails");

                            iface.set(&uni.ortho, ortho);
                            iface.set(&uni.transform, transform.into());
                            iface.set(&uni.tex, bound_view.binding());

                            rdr_gate.render(render_st, |mut tess_gate| {
                                tess_gate.render(&v.layer.tess)
                            })?;
                            drop(fb);

                            // TODO: We only need to render this on the active view.
                            let staging_texture = v.staging_fb.color_slot();
                            let bound_view_staging = pipeline
                                .bind_texture(staging_texture)
                                .expect("binding textures never fails");

                            iface.set(&uni.tex, bound_view_staging.binding());
                            rdr_gate.render(render_st, |mut tess_gate| {
                                tess_gate.render(&v.layer.tess)
                            })?;

                            Ok(())
                        })?;
                    }
                }

                // Render miniview (read-only overlay) before UI
                if let (Some((mini_id, _dx, _dy)), Some(mini_tess)) = (miniview_id_and_pos, miniview_tess.as_ref()) {
                    if let Some(vd_rc) = view_data.get(&mini_id) {
                        let vdb = vd_rc.borrow_mut();
                        shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                            let mut fb = vdb.layer.fb.borrow_mut();
                            let bound_view = pipeline
                                .bind_texture(fb.color_slot())
                                .expect("binding textures never fails");

                            iface.set(&uni.ortho, ortho);
                            iface.set(&uni.transform, identity);
                            iface.set(&uni.tex, bound_view.binding());

                            let res = rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&*mini_tess));
                            drop(fb);
                            res
                        })?;
                        drop(vdb);
                    }
                }

                // LookupSampling overlay for miniview is drawn via draw.rs batch

                // Render UI.
                shd_gate.shade(shape2d, |mut iface, uni, mut rdr_gate| {
                    iface.set(&uni.ortho, ortho);
                    iface.set(&uni.transform, identity);

                    rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&ui_tess))
                })?;

                // Composite lookup animations to screen using sprite2d
                shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                    iface.set(&uni.ortho, ortho);
                    iface.set(&uni.transform, identity);
                    
                    if session.settings["animation"].is_set() {
                        for (id, v) in view_data.iter() {
                            let v_borrow = v.borrow();
                            let Some(view) = session.views.get(*id) else {
                                continue;
                            };
                            
                            // Render the main lookup animation
                            match (&v_borrow.anim_lt_tess, Some(view)) {
                                (Some(tess), Some(view)) if !view.lookuptexture().is_none() => {
                                    let mut lookup_anim_fb = v_borrow.layer.lookup_anim_fb.borrow_mut();
                                    let bound_lookup_anim = pipeline
                                        .bind_texture(lookup_anim_fb.color_slot())
                                        .expect("binding textures never fails");
                                    
                                    let t = Matrix4::from_translation(
                                        Vector2::new(0., view.zoom).extend(0.),
                                    );
                                    
                                    iface.set(&uni.tex, bound_lookup_anim.binding());
                                    iface.set(&uni.transform, t.into());
                                    rdr_gate.render(render_st, |mut tess_gate| {
                                        tess_gate.render(tess)
                                    })?;
                                }
                                _ => (),
                            }

                            // Render lookup layered animations
                            for (llid, tess) in v_borrow.lookup_layer_tess.iter() {
                                let ltv = view_data.get(&llid).unwrap().borrow();
                                let mut lookup_anim_fb = ltv.layer.lookup_anim_fb.borrow_mut();
                                let bound_lookup_anim = pipeline
                                    .bind_texture(lookup_anim_fb.color_slot())
                                    .expect("binding textures never fails");
                                
                                let t = Matrix4::from_translation(
                                    Vector2::new(0., view.zoom).extend(0.),
                                );
                                
                                iface.set(&uni.tex, bound_lookup_anim.binding());
                                iface.set(&uni.transform, t.into());
                                rdr_gate.render(render_st, |mut tess_gate| {
                                    tess_gate.render(tess)
                                })?;
                            }
                        }
                    }
                    
                    Ok(())
                })?;

                // Render text, tool & view animations.
                shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                    iface.set(&uni.ortho, ortho);
                    iface.set(&uni.transform, identity);

                    // Render view animations.
                    if session.settings["animation"].is_set() {
                        for (id, v) in view_data.iter() {
                            let v = &mut *v.borrow_mut();
                            match (&v.anim_tess, session.views.get(*id)) {
                                (Some(tess), Some(view)) if view.animation.len() > 1 => {
                                    let mut fb = v.layer.fb.borrow_mut();
                                    let bound_layer = pipeline
                                        .bind_texture(fb.color_slot())
                                        .expect("binding textures never fails");
                                    let t = Matrix4::from_translation(
                                        Vector2::new(0., view.zoom).extend(0.),
                                    );

                                    // Render layer animation.
                                    iface.set(&uni.tex, bound_layer.binding());
                                    iface.set(&uni.transform, t.into());
                                    rdr_gate.render(render_st, |mut tess_gate| {
                                        tess_gate.render(tess)
                                    })?;
                                }
                                _ => (),
                            }
                        }
                    }

                    {
                        let bound_font = pipeline
                            .bind_texture(font)
                            .expect("binding textures never fails");
                        iface.set(&uni.tex, bound_font.binding());
                        iface.set(&uni.transform, identity);

                        // Render text.
                        rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&text_tess))?;
                    }
                    {
                        let bound_tool = pipeline
                            .bind_texture(cursors)
                            .expect("binding textures never fails");
                        iface.set(&uni.tex, bound_tool.binding());

                        // Render tool.
                        rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&tool_tess))?;
                    }
                    Ok(())
                })?;


                // Render help.
                if let Some((win_tess, text_tess)) = help_tess {
                    shd_gate.shade(shape2d, |_iface, _uni, mut rdr_gate| {
                        rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&win_tess))
                    })?;
                    shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                        let bound_font = pipeline
                            .bind_texture(font)
                            .expect("binding textures never fails");

                        iface.set(&uni.tex, bound_font.binding());
                        iface.set(&uni.ortho, ortho);
                        iface.set(&uni.transform, identity);

                        rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&text_tess))
                    })?;
                }
                Ok(())
            },
        );

        // Render to back buffer.
        builder.pipeline::<PipelineError, _, _, _, _>(
            present_fb,
            pipeline_st,
            |pipeline, mut shd_gate| {
                // Render screen framebuffer.
                let bound_screen = pipeline
                    .bind_texture(screen_fb.color_slot())
                    .expect("binding textures never fails");
                shd_gate.shade(screen2d, |mut iface, uni, mut rdr_gate| {
                    iface.set(&uni.framebuffer, bound_screen.binding());

                    rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&screen_tess))
                })?;

                if session.settings["debug"].is_set() || !execution.is_normal() {
                    let bound_font = pipeline
                        .bind_texture(font)
                        .expect("binding textures never fails");

                    shd_gate.shade(sprite2d, |mut iface, uni, mut rdr_gate| {
                        iface.set(&uni.tex, bound_font.binding());
                        iface.set(
                            &uni.ortho,
                            Matrix4::ortho(screen_w, screen_h, Origin::BottomLeft).into(),
                        );

                        rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&overlay_tess))
                    })?;
                }

                // Render cursor.
                let bound_cursors = pipeline
                    .bind_texture(cursors)
                    .expect("binding textures never fails");
                shd_gate.shade(cursor2d, |mut iface, uni, mut rdr_gate| {
                    let ui_scale = session.settings["scale"].to_f64();
                    let pixel_ratio = platform::pixel_ratio(*scale_factor);

                    iface.set(&uni.cursor, bound_cursors.binding());
                    iface.set(&uni.framebuffer, bound_screen.binding());
                    iface.set(&uni.ortho, ortho);
                    iface.set(&uni.scale, (ui_scale * pixel_ratio) as f32);

                    rdr_gate.render(render_st, |mut tess_gate| tess_gate.render(&cursor_tess))
                })
            },
        );

                // If any view is dirty, record a snapshot of it.
        for v in session.views.iter_mut() {
            if v.is_dirty() {
                let id = v.id;
                let state = v.state;
                let is_resized = v.is_resized();
                let extent = v.extent();

                if let Some(vd_rc) = view_data.get(&id) {
                    let mut v_data = vd_rc.borrow_mut();

                    match state {
                        ViewState::Dirty(_) if is_resized => {
                            v.record_view_resized(v_data.layer.pixels(), extent);
                        }
                        ViewState::Dirty(_) => {
                            v.record_view_painted(v_data.layer.pixels());
                        }
                        ViewState::Okay | ViewState::Damaged(_) => {}
                    }
                }
            }
        }

        if !self.final_batch.is_empty() {
            session.cursor_dirty();
        }

        if !execution.is_normal() {
            let texels = screen_fb
                .color_slot()
                .get_raw_texels()
                .expect("binding textures never fails");
            let texels = Rgba8::align(&texels);

            execution.record(texels).ok();
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

        self.present_fb = Framebuffer::back_buffer(
            &mut self.ctx,
            [physical.width as u32, physical.height as u32],
        )
        .expect("binding textures never fails");

        self.win_size = size;
        self.handle_session_scale_changed(self.scale);
    }

    pub fn handle_session_scale_changed(&mut self, scale: f64) {
        self.scale = scale;
        self.screen_fb = Framebuffer::new(
            &mut self.ctx,
            [
                (self.win_size.width / scale) as u32,
                (self.win_size.height / scale) as u32,
            ],
            0,
            self::SAMPLER,
        )
        .unwrap();
    }

    fn handle_effects(
        &mut self,
        mut effects: Vec<Effect>,
        session: &mut Session,
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
                    // FIXME: This should be done when the view is added in the ViewManager.
                    if let Some((s, pixels)) = session.views.get_snapshot_safe(id) {
                        let (w, h, tfw, tfh) = (s.width(), s.height(), s.extent.fw, s.extent.fh);

                        self.view_data.insert(
                            id,
                            RefCell::new(ViewData::new(w, h, tfw, tfh, Some(pixels), &mut self.ctx)),
                        );
                    }
                }
                Effect::ViewRemoved(id) => {
                    self.view_data.remove(&id);
                    if let Some((vid, _)) = &session.miniview {
                        if *vid == id {
                            session.miniview = None;
                        }
                    }
                }
                Effect::ViewOps(id, ops) => {
                    self.handle_view_ops(session, id, &ops)?;
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
                Effect::LookupTextureQuery(id, color, view_coords, source_view_id) => {
                    self.handle_lookup_texture_query(session, id, color, view_coords, source_view_id)?;
                }
            }
        }
        Ok(())
    }

    fn handle_view_ops(
        &mut self,
        session: &mut Session,
        id: ViewId,
        ops: &[ViewOp],
    ) -> Result<(), RendererError> {
        let v = session.view(id);
        use RendererError as Error;

        for op in ops {
            match op {
                ViewOp::Resize(w, h) => {
                    self.resize_view(v, *w, *h)?;
                }
                ViewOp::Clear(color) => {
                    let view = self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data")
                        .borrow_mut();

                    view.layer
                        .fb
                        .borrow_mut()
                        .color_slot()
                        .clear(GenMipmaps::No, (color.r, color.g, color.b, color.a))
                        .map_err(Error::Texture)?;
                }
                ViewOp::Blit(src, dst) => {
                    let view = self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data")
                        .borrow_mut();

                    let (_, texels) = v.layer.get_snapshot_rect(&src.map(|n| n as i32)).unwrap(); // TODO: Handle this nicely?
                    let texels = util::align_u8(&texels);

                    view.layer
                        .fb
                        .borrow_mut()
                        .color_slot()
                        .upload_part_raw(
                            GenMipmaps::No,
                            [dst.x1 as u32, dst.y1 as u32],
                            [src.width() as u32, src.height() as u32],
                            texels,
                        )
                        .map_err(Error::Texture)?;
                }
                ViewOp::Yank(src) => {
                    let (_, pixels) = v.layer.get_snapshot_rect(&src.map(|n| n)).unwrap();
                    let (w, h) = (src.width() as u32, src.height() as u32);
                    let [paste_w, paste_h] = self.paste.size();

                    if paste_w != w || paste_h != h {
                        self.paste = Texture::new(&mut self.ctx, [w, h], 0, self::SAMPLER)
                            .map_err(Error::Texture)?;
                    }
                    let body = util::align_u8(&pixels);

                    self.paste
                        .upload_raw(GenMipmaps::No, body)
                        .map_err(Error::Texture)?;
                }
                ViewOp::Flip(src, dir) => {
                    let (_, mut pixels) = v.layer.get_snapshot_rect(&src.map(|n| n)).unwrap();
                    let (w, h) = (src.width() as u32, src.height() as u32);
                    let [paste_w, paste_h] = self.paste.size();

                    if paste_w != w || paste_h != h {
                        self.paste = Texture::new(&mut self.ctx, [w, h], 0, self::SAMPLER)
                            .map_err(Error::Texture)?;
                    }

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

                    let body = util::align_u8(&pixels);

                    self.paste
                        .upload_raw(GenMipmaps::No, body)
                        .map_err(Error::Texture)?;
                }
                ViewOp::Paste(dst) => {
                    let [paste_w, paste_h] = self.paste.size();
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

                    self.paste_outputs.push(
                        self.ctx
                            .tessellation::<_, Sprite2dVertex>(batch.vertices().as_slice()),
                    );
                }
                ViewOp::SetPixel(rgba, x, y) => {
                    let layer = &mut self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data")
                        .borrow_mut()
                        .layer;
                    let texels = &[*rgba];
                    let texels = util::align_u8(texels);
                    layer
                        .fb
                        .borrow_mut()
                        .color_slot()
                        .upload_part_raw(GenMipmaps::No, [*x as u32, *y as u32], [1, 1], texels)
                        .map_err(Error::Texture)?;
                }
                ViewOp::Shade(rect, color, target_color) => {
                    let view = &mut self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data")
                        .borrow_mut();
                    
                    // TODO: fix this, looks ugly
                    let mut myrect = rect.clone();
                    myrect.y1 = view.layer.lt_tfh as i32 - myrect.y1;
                    myrect.y2 = view.layer.lt_tfh as i32 - myrect.y2;
                    mem::swap(&mut myrect.y1, &mut myrect.y2);
                    // mem::swap(&mut myrect.x1, &mut myrect.x2);
                    println!("Shade: {:?}", myrect);

                    let (_, orig_pixels) = v.layer.get_snapshot_rect(&rect.map(|n| n as i32)).unwrap(); // TODO: Handle this nicely?

                    // Convert base color to HSV to get hue for color grid
                    let base_rgba: Rgba = (*color).into();
                    let base_hsv: crate::gfx::color::Hsv = base_rgba.into();
                    
                    // Generate color grid using the base color's hue
                    let color_grid: Vec<Rgba8> = color::generate_color_grid(
                        base_hsv.h, 
                        rect.width() as u32, 
                        rect.height() as u32, 
                        1.0, // gamma for saturation
                        1.0, // gamma for value
                        0.3, // s_start: start with low saturation
                        0.7, // s_end: end with full saturation
                        0.3, // v_start: start with moderate brightness
                        1.0  // v_end: end with full brightness
                    ).collect();
                    
                    // Create texels using the color grid
                    let mut texels = Vec::with_capacity(rect.width() as usize * rect.height() as usize * 4);
                    for y in 0..rect.height() {
                        for x in 0..rect.width() {
                            let grid_index = (y * rect.width() + x) as usize;
                            let shaded_color = color_grid[grid_index];
                            let orig_color = orig_pixels[(y * rect.width()) as usize + x as usize];
                            
                            if orig_color == *target_color {
                                // Use the grid color directly (as per user's edit)
                                texels.extend_from_slice(&[shaded_color.r, shaded_color.g, shaded_color.b, shaded_color.a]);
                            } else {
                                texels.extend_from_slice(&[orig_color.r, orig_color.g, orig_color.b, orig_color.a]);
                            }
                        }
                    }
                    
                    let texels = util::align_u8(&texels);
                    view.layer
                        .fb
                        .borrow_mut()
                        .color_slot()
                        .upload_part_raw(
                            GenMipmaps::No,
                            [myrect.x1 as u32, myrect.y1 as u32],
                            [myrect.width() as u32, myrect.height() as u32],
                            texels,
                        )
                        .map_err(Error::Texture)?;
                }
                ViewOp::LookupTextureImDump => {
                    let layer = &mut self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data")
                        .borrow_mut()
                        .layer;
                    let mut lt_im = layer.lt_im.borrow_mut();
                    let texels = lt_im.color_slot().get_raw_texels().unwrap();
                    let pixels = Rgba8::align(&texels).to_vec();
                    println!("lt_im");
                    for (i, pixel) in pixels.iter().enumerate() {
                        // use v.fw and v.fh to decode i into x and y
                        let x = i as u32 % 4096;
                        let y = i as u32 / 4096;
                        let r = x % 256;
                        let g = y % 256;
                        let b = ((x / 256) << 4) + (y / 256);
                        let idx_pixel = Rgba8::new(r as u8, g as u8, b as u8, 255);
                        if pixel.a != 0 {
                            println!("pixel {}, {}: {} ({} => {},{})", x, y, pixel, idx_pixel, pixel.r, pixel.g);
                        }
                    }
                    let mut fb = layer.fb.borrow_mut();
                    let texels = fb.color_slot().get_raw_texels().unwrap();
                    let pixels = Rgba8::align(&texels).to_vec();
                    println!("fb");
                    for (i, pixel) in pixels.iter().enumerate() {
                        let x = i as u32 % layer.w;
                        let y = i as u32 / layer.w;
                        if pixel.a != 0 {
                            println!("pixel {}, {}: {}", x, y, pixel);
                        }
                    }

                    let mut fb = layer.lookup_anim_fb.borrow_mut();
                    let texels = fb.color_slot().get_raw_texels().unwrap();
                    let pixels = Rgba8::align(&texels).to_vec();
                    println!("lookup_anim_fb");
                    for (i, pixel) in pixels.iter().enumerate() {
                        let x = i as u32 % layer.w;
                        let y = i as u32 / layer.w;
                        if pixel.a != 0 {
                            println!("pixel {}, {}: {}", x, y, pixel);
                        }
                    }
                }
                ViewOp::LookupTextureExport(path, frame_mask_override) => {
                    let v_inner = self
                        .view_data
                        .get(&v.id)
                        .expect("views must have associated view data")
                        .borrow();
                    
                    let Some(ltid) = v.lookuptexture() else {
                        eprintln!("View {} does not have a lookup texture", v.id);
                        continue;
                    };
                    
                    let Some(tess) = &v_inner.lt_fb_tess else {
                        eprintln!("View {} does not have lookup texture tessellation", v.id);
                        continue;
                    };
                    
                    let rcltv = self.view_data.get(&ltid)
                        .expect("lookup texture view must have associated view data");
                    
                    let frame_mask = frame_mask_override.unwrap_or_else(|| {
                        session.settings["lookup/framemask"].to_u64() as u32
                    });
                    let lookup_anim_ortho: M44 = Matrix4::ortho(v_inner.layer.w, v_inner.layer._h, Origin::TopLeft).into();
                    let identity: M44 = Matrix4::identity().into();
                    
                    let render_st = RenderState::default()
                        .set_blending(blending::Blending {
                            equation: Equation::Additive,
                            src: Factor::SrcAlpha,
                            dst: Factor::SrcAlphaComplement,
                        })
                        .set_depth_test(Some(DepthComparison::LessOrEqual));
                    let pipeline_st = PipelineState::default()
                        .set_clear_color([0., 0., 0., 0.])
                        .enable_srgb(true)
                        .enable_clear_depth(true)
                        .enable_clear_color(true);
                    
                    let mut builder = self.ctx.new_pipeline_gate();
                    let lookuptex2d = &mut self.lookuptex2d;
                    builder.pipeline::<PipelineError, _, _, _, _>(
                        &*v_inner.layer.lookup_anim_fb.borrow(),
                        &pipeline_st,
                        |pipeline, mut shd_gate| {
                            let mut main_fb = v_inner.layer.fb.borrow_mut();
                            let bound_layer = pipeline
                                .bind_texture(main_fb.color_slot())
                                .expect("binding textures never fails");
                            
                            let ltv_inner = rcltv.borrow();
                            let mut lookup_fb = ltv_inner.layer.fb.borrow_mut();
                            let mut lookup_lt_im = ltv_inner.layer.lt_im.borrow_mut();
                            let lookup_layer = pipeline
                                .bind_texture(lookup_fb.color_slot())
                                .expect("binding textures never fails");
                            let lookup_map = pipeline
                                .bind_texture(lookup_lt_im.color_slot())
                                .expect("binding textures never fails");
                            
                            shd_gate.shade(lookuptex2d, |mut iface, uni, mut rdr_gate| {
                                iface.set(&uni.ortho, lookup_anim_ortho);
                                iface.set(&uni.transform, identity);
                                iface.set(&uni.tex, bound_layer.binding());
                                iface.set(&uni.ltex, lookup_layer.binding());
                                iface.set(&uni.ltexim, lookup_map.binding());
                                iface.set(&uni.lt_tfw, ltv_inner.layer.lt_tfw);
                                iface.set(&uni.frame_mask, frame_mask);
                                
                                rdr_gate.render(&render_st, |mut tess_gate| {
                                    tess_gate.render(tess)
                                })
                            })?;
                            
                            Ok(())
                        },
                    );
                    
                    let mut lookup_anim_fb = v_inner.layer.lookup_anim_fb.borrow_mut();
                    let texels = lookup_anim_fb.color_slot().get_raw_texels().unwrap();
                    let pixels = Rgba8::align(&texels).to_vec();
                    
                    // Save using the existing image module
                    if let Err(e) = image::save_as(&path, v_inner.layer.w, v_inner.layer._h, 1, &pixels) {
                        eprintln!("Failed to save lookup texture export to {}: {}", path, e);
                    } else {
                        println!("Lookup texture exported to: {}", path);
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_view_damaged(&mut self, view: &View<ViewResource>) -> Result<(), RendererError> {
        let layer = &mut self
            .view_data
            .get(&view.id)
            .expect("views must have associated view data")
            .borrow_mut()
            .layer;

        let (_, pixels) = view.layer.current_snapshot();

        layer.clear()?;
        layer.upload(util::align_u8(pixels))?;

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
        // View size changed. Re-create view resources.
        let (ew, eh) = {
            let extent = view.resource.extent;
            (extent.width(), extent.height())
        };

        // Ensure not to transfer more data than can fit in the view buffer.
        let tw = u32::min(ew, vw);
        let th = u32::min(eh, vh);
        let mut view_data = ViewData::new(vw, vh, vw / view.animation.len() as u32, vh, None, &mut self.ctx);
        let trect = Rect::origin(tw as i32, th as i32);
        // The following sequence of commands will try to copy a rect that isn't contained
        // in the snapshot, hence we must skip the uploading in that case:
        //
        //     :f/add
        //     :f/remove
        //     :undo
        //
        if let Some((_, texels)) = view.layer.get_snapshot_rect(&trect) {
            let texels = util::align_u8(&texels);
            let l = &mut view_data.layer;

            l.upload_part([0, vh - th], [tw, th], texels)?;
        }

        self.view_data.insert(view.id, RefCell::new(view_data));

        Ok(())
    }

    fn update_view_animations(&mut self, s: &Session) {
        if !s.settings["animation"].is_set() {
            return;
        }
        // TODO: Does this need to run if the view has only one frame?
        for v in s.views.iter() {
            if v.is_lookuptexture() {
                if let Some(vd) = self.view_data.get(&v.id) {
                    vd.borrow_mut().anim_tess = None;
                    vd.borrow_mut().anim_lt_tess = None;
                }
                continue;
            }
            // FIXME: When `v.animation.val()` doesn't change, we don't need
            // to re-create the buffer.
            let batch = draw::draw_view_animation(s, v);
            if let Some(vd) = self.view_data.get(&v.id) {
                vd.borrow_mut().anim_tess = Some(
                    self.ctx
                        .tessellation::<_, Sprite2dVertex>(batch.vertices().as_slice()),
                );
            }

            // lookup-texture animation enabled
            if let Some(_) = v.lookuptexture() {
                let ltbatch = draw::draw_view_lookuptexture_fb(s, v);
                let ltfbbatch = draw::draw_view_lookuptexture_animation(v);
                if let Some(vd) = self.view_data.get(&v.id) {
                    vd.borrow_mut().anim_lt_tess = Some(
                        self.ctx
                            .tessellation::<_, Sprite2dVertex>(ltbatch.vertices().as_slice()),
                    );
                    vd.borrow_mut().lt_fb_tess = Some(
                        self.ctx
                            .tessellation::<_, Sprite2dVertex>(ltfbbatch.vertices().as_slice()),
                    );
                }
            }

            if !v.lookup_layers().is_empty() {
                let vd = self.view_data.get(&v.id).unwrap();
                vd.borrow_mut().lookup_layer_tess.clear();
                let batch = draw::draw_view_lookuptexture_layer(s, v, v);
                vd.borrow_mut().lookup_layer_tess.push((
                    v.id,
                    self.ctx
                        .tessellation::<_, Sprite2dVertex>(batch.vertices().as_slice()),
                ));
                for llid in v.lookup_layers() {
                    let ltv = s.views.get(*llid).unwrap();
                    let batch = draw::draw_view_lookuptexture_layer(s, ltv, v);
                    vd.borrow_mut().lookup_layer_tess.push((
                        *llid,
                        self.ctx
                            .tessellation::<_, Sprite2dVertex>(batch.vertices().as_slice()),
                    ));
                }
            }

        }
    }

    fn update_view_composites(&mut self, s: &Session) {
        for v in s.views.iter() {
            let batch = draw::draw_view_composites(s, v);

            if let Some(vd) = self.view_data.get(&v.id) {
                vd.borrow_mut().layer_tess = Some(
                    self.ctx
                        .tessellation::<_, Sprite2dVertex>(batch.vertices().as_slice()),
                );
            }
        }
    }

    fn handle_lookup_texture_query(&mut self, session: &mut Session, id: ViewId, color: Rgba8, view_coords: Point<ViewExtent, f32>, source_view_id: ViewId) -> Result<(), RendererError> {
        println!("view_coords: {}, {}", view_coords.x, view_coords.y);
        let lookup_layer = &mut self
            .view_data
            .get(&id)
            .expect("views must have associated view data")
            .borrow_mut()
            .layer;
        

        let mut fb: Framebuffer<Backend, Dim2, pixel::SRGBA8UI, pixel::Depth32F> =
            Framebuffer::new(&mut self.ctx, [1, 1], 0, self::SAMPLER).unwrap();

        let tess: Tess<Backend, Sprite2dVertex> = TessBuilder::new(&mut self.ctx)
            .set_vertex_nb(1)
            .set_mode(Mode::Point)
            .build()
            .unwrap();
        
        let render_st = RenderState::default()
            .set_blending(blending::Blending {
                equation: Equation::Additive,
                src: Factor::SrcAlpha,
                dst: Factor::SrcAlphaComplement,
            })
            .set_depth_test(Some(DepthComparison::LessOrEqual));
        let pipeline_st = PipelineState::default()
            .set_clear_color([0., 0., 0., 0.])
            .enable_srgb(true)
            .enable_clear_depth(true)
            .enable_clear_color(true);

        // decode color into pixel_coords
        let pixel_coords = [
            color.r as i32 + (color.b as i32 >> 4) * 256,
            color.g as i32 + (color.b as i32 & 15) * 256
        ];

        let mut builder = self.ctx.new_pipeline_gate();
        builder.pipeline::<PipelineError, _, _, _, _>(
            &fb,
            &pipeline_st,
            |pipeline, mut shd_gate| {
                let mut lt_im = lookup_layer.lt_im.borrow_mut();
                let bound_lookup_layer = pipeline
                    .bind_texture(lt_im.color_slot())
                    .expect("binding textures never fails");
                shd_gate.shade(&mut self.lookupquery2d, |mut iface, uni, mut rdr_gate| {
                    // iface.set(&uni., view_ortho_lookup.into());
                    // iface.set(&uni.transform, identity);
                    // iface.set(&uni.tex, bound_lookup_layer.binding());
                    iface.set(&uni.ltexim, bound_lookup_layer.binding());
                    iface.set(&uni.pixel_coords, pixel_coords);
                    rdr_gate.render(&render_st, |mut tess_gate| tess_gate.render(&tess))
                })?;
                Ok(())
            }
        );

        let texels = fb.color_slot().get_raw_texels().unwrap();
        let pixels = Rgba8::align(&texels).to_vec();
        println!("lookup query: {}, {}", pixels[0].r, pixels[0].g);

        let cursor_x = pixels[0].r as u32;
        let cursor_y = pixels[0].g as u32;

        // Check if we're already in LookupSampling mode - if so, don't add to other_queries
        // (mouse clicks in LookupSampling mode should pop from cartridge instead)
        let is_lookup_sampling = matches!(session.mode, session::Mode::Visual(session::VisualState::LookupSampling));
        
        if let Some(mut res) = session.lookup_result.take() {
            if res.view_id == id {
                // Only add to other_queries if not already in LookupSampling mode
                // (initial query or query from outside LookupSampling mode)
                if !is_lookup_sampling {
                    res.other_queries.push((cursor_x, cursor_y));
                }
                session.lookup_result = Some(res);
            } else {
                session.lookup_result = Some(session::LookupResult {
                    view_id: id,
                    cursor_x,
                    cursor_y,
                    pixel_x: view_coords.x as i32,
                    pixel_y: view_coords.y as i32,
                    source_view_id: source_view_id,
                    other_queries: vec![],
                    cartridge: std::collections::VecDeque::new(),
                });
            }
        } else {
            session.lookup_result = Some(session::LookupResult {
                view_id: id,
                cursor_x,
                cursor_y,
                pixel_x: view_coords.x as i32,
                pixel_y: view_coords.y as i32,
                source_view_id: source_view_id,
                other_queries: vec![],
                cartridge: std::collections::VecDeque::new(),
            });
        }

        println!("session.lookup_result: {:?}", session.lookup_result);

        session.switch_mode(session::Mode::Visual(session::VisualState::LookupSampling));

        Ok(())
    }
}

fn text_batch([w, h]: [u32; 2]) -> TextBatch {
    TextBatch::new(w, h, draw::GLYPH_WIDTH, draw::GLYPH_HEIGHT)
}

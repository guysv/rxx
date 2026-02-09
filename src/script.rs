//! Rhai script loading and event-handler lifecycle.
//!
//! Follows the [event-handler pattern](https://rhai.rs/book/patterns/events-1.html):
//! one main script with `init(session, renderer)`; custom Scope + CallFnOptions (eval_ast false,
//! rewind_scope false) so variables defined in `init()` persist.

use crate::draw::{self, USER_LAYER};
use crate::gfx::color::Rgba;
use crate::gfx::math::{Point2, Vector2};
use crate::gfx::rect::Rect;
use crate::gfx::shape2d::{self, Line, Rotation, Shape, Stroke};
use crate::gfx::sprite2d;
use crate::gfx::ZDepth;
use crate::gfx::{Repeat, Rgba8};
use crate::session::{Effect, MessageType, Session};
use crate::view::{View, ViewId, ViewResource};
use crate::wgpu::{
    self, Color, Operations, RenderPassColorAttachment, RenderPassDescriptor,
    RenderTexture, RenderTextureHandle, StoreOp, TextureFormat,
};
use ::wgpu::LoadOp;

use rhai::{Array, CallFnOptions, Dynamic, Engine, ImmutableString, Scope, AST};

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::SystemTime;

/// Type alias for the user batches shared between script and session.
/// (shape batch, sprite batch for text). Sprite batch is created lazily with font size.
pub type UserBatch = (Rc<RefCell<shape2d::Batch>>, Rc<RefCell<Option<sprite2d::Batch>>>);

/// Read-only view handle exposed to Rhai scripts.
#[derive(Debug, Clone)]
struct ScriptView {
    id: ViewId,
    offset: Vector2<f32>,
    frame_width: f32,
    frame_height: f32,
    zoom: f32,
}

impl From<&View<ViewResource>> for ScriptView {
    fn from(view: &View<ViewResource>) -> Self {
        ScriptView { id: view.id, offset: view.offset, frame_width: view.fw as f32, frame_height: view.fh as f32, zoom: view.zoom }
    }
}

/// Script-held sprite batch (singleton or built via API). Has vertices() to compile.
#[derive(Clone, Debug)]
pub struct ScriptSpriteBatch {
    pub(crate) batch: sprite2d::Batch,
}

/// Opaque vertex list from ScriptSpriteBatch.vertices(); passed to create_vertex_buffer_from_sprite_vertices.
#[derive(Clone)]
pub struct ScriptSpriteVertexList {
    pub(crate) vertices: Vec<sprite2d::Vertex>,
}

impl std::fmt::Debug for ScriptSpriteVertexList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScriptSpriteVertexList {{ vertices: {:?} }}", self.vertices)
    }
}

impl std::fmt::Display for ScriptSpriteVertexList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScriptSpriteVertexList {{ vertices: {:?} }}", self.vertices)
    }
}

/// State for the main Rhai script (event-handler style).
/// Built in lib.rs and passed to renderer.frame() and draw_ctx.draw().
pub struct ScriptState {
    /// Path to the main Rhai script.
    pub script_path: Option<PathBuf>,
    /// Mtime of the script file after last successful load (for hot-reload dedup).
    pub script_mtime: Option<SystemTime>,
    /// Rhai engine for the main script.
    pub script_engine: Option<Engine>,
    /// Custom scope so variables defined in init() persist.
    pub script_scope: Option<Scope<'static>>,
    /// Compiled script AST.
    pub script_ast: Option<Rc<RefCell<AST>>>,
    /// User batches for script's draw() event (shape + sprite for text), shared with Rhai closures.
    pub script_user_batch: UserBatch,
    /// Script-created render textures (owned here so renderer stays stateless for script resources).
    pub script_render_textures: BTreeMap<u64, Rc<RefCell<RenderTexture>>>,
    /// Next id for script_render_textures.
    pub next_script_texture_id: u64,
}

impl ScriptState {
    pub fn new() -> Self {
        Self {
            script_path: None,
            script_mtime: None,
            script_engine: None,
            script_scope: None,
            script_ast: None,
            script_user_batch: (
                Rc::new(RefCell::new(shape2d::Batch::new())),
                Rc::new(RefCell::new(None)),
            ),
            script_render_textures: BTreeMap::new(),
            next_script_texture_id: 0,
        }
    }

    /// Store a render texture and return its script handle id. Used by the renderer when creating script textures.
    pub fn add_render_texture(&mut self, texture: RenderTexture) -> u64 {
        let id = self.next_script_texture_id;
        self.next_script_texture_id = self.next_script_texture_id.saturating_add(1);
        self.script_render_textures.insert(id, Rc::new(RefCell::new(texture)));
        id
    }

    /// Ensure the user sprite batch exists (created with font texture size). Call from renderer each frame.
    pub fn ensure_user_sprite_batch(&mut self, w: u32, h: u32) {
        if self.script_user_batch.1.borrow().is_none() {
            *self.script_user_batch.1.borrow_mut() = Some(sprite2d::Batch::new(w, h));
        }
    }

    pub fn set_path(&mut self, path: PathBuf) {
        self.script_path = Some(path);
    }

    /// Populate engine, scope, ast, mtime after a successful load. Used by load_script.
    fn apply_loaded_script(
        &mut self,
        engine: Engine,
        scope: Scope<'static>,
        ast: AST,
        path: &PathBuf,
    ) {
        self.script_engine = Some(engine);
        self.script_scope = Some(scope);
        self.script_ast = Some(Rc::new(RefCell::new(ast)));
        self.script_mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
    }
}

/// Load or reload the main Rhai script: compile, create scope, call init(session, renderer).
/// Takes script_state_handle so that init() can call create_render_texture (which borrows it)
/// without RefCell double-borrow. Returns an error message for the caller to display.
pub fn load_script(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    session_handle: &Rc<RefCell<Session>>,
    renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
) -> Result<(), String> {
    let (path, shape_batch, sprite_batch) = {
        let state = script_state_handle.borrow();
        let path = match state.script_path.as_ref() {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        if !path.exists() {
            return Err(format!("Script not found: {}", path.display()));
        }
        (
            path,
            state.script_user_batch.0.clone(),
            state.script_user_batch.1.clone(),
        )
    };
    let mut engine = Engine::new();
    register_draw_primitives(&mut engine, shape_batch, sprite_batch);
    register_session_handle(&mut engine);
    register_renderer_handle(&mut engine, renderer_handle, script_state_handle);
    register_wgpu_types(&mut engine);

    let script_commands: Rc<RefCell<Vec<(String, String)>>> =
        Rc::new(RefCell::new(Vec::new()));
    register_command_api(&mut engine, script_commands.clone());

    let ast = compile_file(&engine, &path)
        .map_err(|e| format!("Script compile error: {}", e))?;
    let mut scope = Scope::new();
    register_constants(&mut scope);
    call_init(&engine, &mut scope, &ast, session_handle, renderer_handle)
        .map_err(|e| format!("Script init error: {}", e))?;

    let cmds = script_commands.borrow().clone();
    session_handle.borrow_mut().set_script_commands(cmds);

    script_state_handle
        .borrow_mut()
        .apply_loaded_script(engine, scope, ast, &path);
    Ok(())
}

/// Reload the main Rhai script (recompile and call init again). Errors are ignored.
pub fn reload_script(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    session_handle: &Rc<RefCell<Session>>,
    renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
) {
    let _ = load_script(script_state_handle, session_handle, renderer_handle);
}

impl ScriptState {
    /// True if the script file on disk is newer than the last load (or we never stored mtime).
    pub fn script_file_modified_since_load(&self) -> bool {
        let path = match &self.script_path {
            Some(p) => p,
            None => return false,
        };
        let current = match std::fs::metadata(path).ok().and_then(|m| m.modified().ok()) {
            Some(t) => t,
            None => return false,
        };
        match self.script_mtime {
            Some(last) => current > last,
            None => true,
        }
    }

    /// Path to the main Rhai script, if one is loaded.
    pub fn script_path(&self) -> Option<&PathBuf> {
        self.script_path.as_ref()
    }


    /// Traverse effects: call script's `view_added` / `view_removed` handlers and run script
    /// commands (with session messages on error). Returns effects not consumed for the renderer.
    pub fn call_view_effects(
        &mut self,
        effects: &[Effect],
        session_handle: &Rc<RefCell<Session>>,
    ) -> Vec<Effect> {
        let mut renderer_effects = Vec::new();
        let mut script_commands = Vec::new();
        {
            let (engine, scope, ast) = match (
                self.script_engine.as_ref(),
                self.script_scope.as_mut(),
                self.script_ast.as_ref(),
            ) {
                (Some(e), Some(s), Some(a)) => (e, s, a),
                _ => return effects.to_vec(),
            };
            for eff in effects {
                match eff {
                    Effect::ViewAdded(id) => {
                        let _ = call_view_added(engine, scope, &ast.borrow(), id.raw() as i64);
                        renderer_effects.push(eff.clone());
                    }
                    Effect::ViewRemoved(id) => {
                        let _ = call_view_removed(engine, scope, &ast.borrow(), id.raw() as i64);
                        renderer_effects.push(eff.clone());
                    }
                    Effect::RunScriptCommand(name, args) => script_commands.push((name.clone(), args.clone())),
                    other => renderer_effects.push(other.clone()),
                }
            }
        }
        for (name, args) in script_commands {
            match self.call_script_command(&name, args) {
                Ok(true) => {}
                Ok(false) => {
                    session_handle.borrow_mut().message(
                        format!("Script command '{}' has no handler cmd_{}", name, name),
                        MessageType::Error,
                    );
                }
                Err(e) => {
                    session_handle.borrow_mut().message(
                        format!("Script command '{}' error: {}", name, e),
                        MessageType::Error,
                    );
                }
            }
        }
        renderer_effects
    }

    /// Call the script's `draw()` event handler.
    /// The script's draw primitives (e.g. `draw_line`, `draw_text`) mutate the user batches directly.
    pub fn call_draw_event(&mut self) -> Result<(), Box<rhai::EvalAltResult>> {
        self.script_user_batch.0.borrow_mut().clear();
        if let Some(ref mut batch) = *self.script_user_batch.1.borrow_mut() {
            batch.clear();
        }
        let (engine, scope, ast) = match (
            self.script_engine.as_ref(),
            self.script_scope.as_mut(),
            self.script_ast.as_ref(),
        ) {
            (Some(e), Some(s), Some(a)) => (e, s, a),
            _ => return Ok(()),
        };
        call_draw(engine, scope, &ast.borrow())
    }

    pub fn call_shade_event(&mut self, encoder: &Rc<RefCell<wgpu::Encoder>>) -> Result<(), Box<rhai::EvalAltResult>> {
        let (engine, scope, ast) = match (
            self.script_engine.as_ref(),
            self.script_scope.as_mut(),
            self.script_ast.as_ref(),
        ) {
            (Some(e), Some(s), Some(a)) => (e, s, a),
            _ => return Ok(()),
        };
        call_shade(engine, scope, &ast.borrow(), encoder)
    }

    pub fn call_render_event(&mut self, script_pass: ScriptPass) -> Result<(), Box<rhai::EvalAltResult>> {
        let (engine, scope, ast) = match (
            self.script_engine.as_ref(),
            self.script_scope.as_mut(),
            self.script_ast.as_ref(),
        ) {
            (Some(e), Some(s), Some(a)) => (e, s, a),
            _ => return Ok(()),
        };
        call_render(engine, scope, &ast.borrow(), script_pass)
    }

    /// Get the user shape batch vertices for rendering.
    pub fn user_batch_vertices(&self) -> Vec<crate::gfx::shape2d::Vertex> {
        self.script_user_batch.0.borrow().vertices()
    }

    /// Check if the user shape batch is empty.
    pub fn user_batch_is_empty(&self) -> bool {
        self.script_user_batch.0.borrow().is_empty()
    }

    /// Get the user sprite batch vertices for rendering (text). Empty if batch not yet created.
    pub fn user_sprite_batch_vertices(&self) -> Vec<crate::gfx::sprite2d::Vertex> {
        self.script_user_batch
            .1
            .borrow()
            .as_ref()
            .map(|b| b.vertices())
            .unwrap_or_default()
    }

    /// Check if the user sprite batch is empty or not created.
    pub fn user_sprite_batch_is_empty(&self) -> bool {
        self.script_user_batch
            .1
            .borrow()
            .as_ref()
            .map(|b| b.is_empty())
            .unwrap_or(true)
    }

    /// Call a script command handler `cmd_<name>(args)`.
    /// Returns Ok(true) if handler was found and called, Ok(false) if no handler exists.
    pub fn call_script_command(
        &mut self,
        name: &str,
        args: Vec<String>,
    ) -> Result<bool, Box<rhai::EvalAltResult>> {
        let (engine, scope, ast) = match (
            self.script_engine.as_ref(),
            self.script_scope.as_mut(),
            self.script_ast.as_ref(),
        ) {
            (Some(e), Some(s), Some(a)) => (e, s, a),
            _ => return Ok(false),
        };

        let handler_name = format!("cmd_{}", name.replace('/', "_"));
        let rhai_args: Array = args.into_iter().map(Dynamic::from).collect();

        match engine.call_fn::<()>(scope, &ast.borrow(), &handler_name, (rhai_args,)) {
            Ok(()) => Ok(true),
            Err(e) => Err(e),
        }
    }
}

impl Default for ScriptState {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile a script file into an AST.
pub fn compile_file(engine: &Engine, path: &Path) -> Result<AST, Box<rhai::EvalAltResult>> {
    engine.compile_file(path.into())
}

/// Register session handle type so scripts can use it in init(session, renderer).
/// Exposes width, height, offset_x, offset_y from the session.
pub fn register_session_handle(engine: &mut Engine) {
    engine
        .register_type_with_name::<Rc<RefCell<Session>>>("Session")
        .register_get("width", |s: &mut Rc<RefCell<Session>>| s.borrow().width as f64)
        .register_get("height", |s: &mut Rc<RefCell<Session>>| s.borrow().height as f64)
        .register_get("offset_x", |s: &mut Rc<RefCell<Session>>| s.borrow().offset.x as f64)
        .register_get("offset_y", |s: &mut Rc<RefCell<Session>>| s.borrow().offset.y as f64)
        .register_get("active_view_id", |s: &mut Rc<RefCell<Session>>| s.borrow().views.active_id.raw() as i64)
        .register_type_with_name::<ScriptView>("View")
        .register_get("id", |v: &mut ScriptView| v.id.raw() as i64)
        .register_get("offset", |v: &mut ScriptView| Vector2::new(v.offset.x as f32, v.offset.y as f32))
        .register_get("frame_width", |v: &mut ScriptView| v.frame_width)
        .register_get("frame_height", |v: &mut ScriptView| v.frame_height)
        .register_get("zoom", |v: &mut ScriptView| v.zoom)
        .register_fn("views", |s: &mut Rc<RefCell<Session>>| {
            s.borrow()
                .views
                .iter()
                .map(|v| Dynamic::from(ScriptView::from(v)))
                .collect::<Array>()
        })
        .register_fn("view", |s: &mut Rc<RefCell<Session>>, id: i64| {
            let id = ViewId(id as u16);
            Dynamic::from(ScriptView::from(s.borrow().view(id)))
        });
}

/// Load op for script render pass color attachments.
#[derive(Clone)]
pub enum ScriptLoadOp {
    Load,
    Clear { r: f64, g: f64, b: f64, a: f64 },
}

/// Store op for script render pass color attachments.
#[derive(Clone)]
pub enum ScriptStoreOp {
    Store,
}

/// Script-side color attachment: handle + load/store ops.
#[derive(Clone)]
pub struct ScriptColorAttachment {
    pub handle: RenderTextureHandle,
    pub load_op: ScriptLoadOp,
    pub store_op: ScriptStoreOp,
}

/// Unified pass type for script: wraps the wgpu pass and renderer so pass methods can resolve handles
/// without borrowing ScriptState (which is already borrowed to call render()).
#[derive(Clone)]
pub struct ScriptPass {
    pass: Rc<RefCell<wgpu::Pass<'static>>>,
    renderer: Rc<RefCell<wgpu::Renderer>>,
}

impl ScriptPass {
    pub fn new(pass: Rc<RefCell<wgpu::Pass<'static>>>, renderer: Rc<RefCell<wgpu::Renderer>>) -> Self {
        Self { pass, renderer }
    }

    pub fn set_pipeline(&mut self, handle: i64) -> Result<(), Box<rhai::EvalAltResult>> {
        let id = handle as u64;
        let renderer_ref = self.renderer.borrow();
        let pipeline = renderer_ref.get_script_pipeline(id).ok_or_else(|| {
            rhai::EvalAltResult::ErrorRuntime(
                format!("script pipeline handle not found: {}", id).into(),
                rhai::Position::NONE,
            )
        })?;
        self.pass.borrow_mut().set_pipeline(pipeline);
        Ok(())
    }

    pub fn set_bind_group(&mut self, index: i64, handle: i64) -> Result<(), Box<rhai::EvalAltResult>> {
        let id = handle as u64;
        let renderer_ref = self.renderer.borrow();
        let bind_group = renderer_ref.get_script_bind_group(id).ok_or_else(|| {
            rhai::EvalAltResult::ErrorRuntime(
                format!("script bind group handle not found: {}", id).into(),
                rhai::Position::NONE,
            )
        })?;
        self.pass.borrow_mut().set_bind_group(index as u32, bind_group, &[]);
        Ok(())
    }

    pub fn set_vertex_buffer(&mut self, slot: i64, handle: i64) -> Result<(), Box<rhai::EvalAltResult>> {
        let id = handle as u64;
        let renderer_ref = self.renderer.borrow();
        let buffer = renderer_ref.get_script_buffer(id).ok_or_else(|| {
            rhai::EvalAltResult::ErrorRuntime(
                format!("script buffer handle not found: {}", id).into(),
                rhai::Position::NONE,
            )
        })?;
        self.pass.borrow_mut().set_vertex_buffer(slot as u32, buffer.slice(..));
        Ok(())
    }

    pub fn draw(
        &mut self,
        vertex_count: i64,
        instance_count: i64,
        first_vertex: i64,
        first_instance: i64,
    ) {
        let vertices = (first_vertex as u32)..(first_vertex as u32 + vertex_count as u32);
        let instances = (first_instance as u32)..(first_instance as u32 + instance_count as u32);
        self.pass.borrow_mut().draw(vertices, instances);
    }
}

/// Register renderer handle type so scripts can use it in init(session, renderer).
/// Exposes create_render_texture, view_render_texture, begin_render_pass.
/// Script-created textures are stored in script_state_handle.
pub fn register_renderer_handle(
    engine: &mut Engine,
    _renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
    script_state_handle: &Rc<RefCell<ScriptState>>,
) {
    let script_state_create = script_state_handle.clone();
    let script_state_for_texture_bind_group = script_state_handle.clone();
    let script_state_for_compute_texture = script_state_handle.clone();
    engine
        .register_type_with_name::<TextureFormat>("TextureFormat")
        .register_type_with_name::<RenderTextureHandle>("RenderTextureHandle")
        .register_type_with_name::<ScriptLoadOp>("LoadOp")
        .register_type_with_name::<ScriptStoreOp>("StoreOp")
        .register_type_with_name::<ScriptColorAttachment>("ColorAttachment")
        .register_type_with_name::<ScriptPass>("ScriptPass")
        .register_fn("load_load", || ScriptLoadOp::Load)
        .register_fn("load_clear", |r: f64, g: f64, b: f64, a: f64| ScriptLoadOp::Clear { r, g, b, a })
        .register_fn("store_store", || ScriptStoreOp::Store)
        .register_fn(
            "color_attachment",
            |handle: RenderTextureHandle, load_op: ScriptLoadOp, store_op: ScriptStoreOp| {
                ScriptColorAttachment {
                    handle,
                    load_op,
                    store_op,
                }
            },
        )
        .register_fn(
            "set_pipeline",
            |pass: &mut ScriptPass, handle: i64| -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_pipeline(handle)
            },
        )
        .register_fn(
            "set_bind_group",
            |pass: &mut ScriptPass, index: i64, handle: i64| -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_bind_group(index, handle)
            },
        )
        .register_fn(
            "set_vertex_buffer",
            |pass: &mut ScriptPass, slot: i64, handle: i64| -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_vertex_buffer(slot, handle)
            },
        )
        .register_fn(
            "draw",
            |pass: &mut ScriptPass,
             vertex_count: i64,
             instance_count: i64,
             first_vertex: i64,
             first_instance: i64| {
                pass.draw(vertex_count, instance_count, first_vertex, first_instance);
            },
        )
        .register_type_with_name::<Rc<RefCell<wgpu::Renderer>>>("Renderer")
        .register_fn(
            "create_render_texture",
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, width: i64, height: i64| {
                let mut script_state = script_state_create.borrow_mut();
                r.borrow_mut().create_render_texture(&mut *script_state, width as u32, height as u32)
            },
        )
        .register_fn(
            "create_compute_texture",
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, width: i64, height: i64, format: TextureFormat| {
                let mut script_state = script_state_for_compute_texture.borrow_mut();
                r.borrow_mut().create_compute_texture(&mut *script_state, width as u32, height as u32, format)
            },
        )
        .register_fn(
            "view_render_texture",
            |r: &mut Rc<RefCell<wgpu::Renderer>>, view_id: i64| -> Dynamic {
                let id = ViewId(view_id as u16);
                match r.borrow().view_render_texture(id) {
                    Some(h) => Dynamic::from(h),
                    None => Dynamic::UNIT,
                }
            },
        )
        .register_fn(
            "create_texture_sampler_bind_group",
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, handle: RenderTextureHandle| {
                let script_state = script_state_for_texture_bind_group.borrow();
                r.borrow_mut()
                    .create_texture_sampler_bind_group(&*script_state, handle)
                    .expect("create_texture_sampler_bind_group failed") as i64
            },
        )
        .register_fn(
            "create_shader_module",
            |r: &mut Rc<RefCell<wgpu::Renderer>>, wgsl_source: ImmutableString| {
                r.borrow_mut().create_shader_module(None, wgsl_source.as_str()) as i64
            },
        )
        .register_fn(
            "create_render_pipeline",
            |r: &mut Rc<RefCell<wgpu::Renderer>>,
             shader_handle: i64,
             vs_entry: ImmutableString,
             fs_entry: ImmutableString| {
                r.borrow_mut().create_render_pipeline(
                    shader_handle as u64,
                    vs_entry.as_str(),
                    fs_entry.as_str(),
                ) as i64
            },
        )
        .register_fn(
            "create_render_pipeline_with_texture",
            |r: &mut Rc<RefCell<wgpu::Renderer>>,
             shader_handle: i64,
             vs_entry: ImmutableString,
             fs_entry: ImmutableString| {
                r.borrow_mut().create_render_pipeline_with_texture(
                    shader_handle as u64,
                    vs_entry.as_str(),
                    fs_entry.as_str(),
                ) as i64
            },
        )
        .register_fn(
            "create_buffer",
            |r: &mut Rc<RefCell<wgpu::Renderer>>, size: i64, usage_str: ImmutableString| {
                r.borrow_mut().create_buffer(size as u64, usage_str.as_str(), None) as i64
            },
        )
        .register_fn(
            "create_buffer",
            |r: &mut Rc<RefCell<wgpu::Renderer>>,
             size: i64,
             usage_str: ImmutableString,
             data: Array| {
                let bytes: Vec<u8> = data
                    .iter()
                    .map(|v| v.clone().cast::<i64>().clamp(0, 255) as u8)
                    .collect();
                let slice = if bytes.is_empty() { None } else { Some(bytes.as_slice()) };
                r.borrow_mut().create_buffer(size as u64, usage_str.as_str(), slice) as i64
            },
        )
        .register_fn(
            "create_transform_bind_group",
            |r: &mut Rc<RefCell<wgpu::Renderer>>, buffer_handle: i64| {
                r.borrow_mut().create_transform_bind_group(buffer_handle as u64) as i64
            },
        )
        .register_fn(
            "create_identity_transform_bind_group",
            |r: &mut Rc<RefCell<wgpu::Renderer>>| {
                r.borrow_mut().create_identity_transform_bind_group() as i64
            },
        )
        .register_fn(
            "create_view_transform_bind_group",
            |r: &mut Rc<RefCell<wgpu::Renderer>>,
             translation_x: f64,
             translation_y: f64,
             zoom: f64| {
                r.borrow_mut().create_view_transform_bind_group(
                    translation_x as f32,
                    translation_y as f32,
                    zoom as f32,
                ) as i64
            },
        )
        .register_type_with_name::<ScriptSpriteBatch>("ScriptSpriteBatch")
        .register_fn(
            "sprite_singleton_batch",
            |_r: &mut Rc<RefCell<wgpu::Renderer>>,
             w: i64,
             h: i64,
             src: Rect<f32>,
             dst: Rect<f32>,
             zdepth: f64,
             color: Rgba8| {
                let batch = sprite2d::Batch::singleton(
                    w as u32,
                    h as u32,
                    src,
                    dst,
                    crate::gfx::ZDepth(zdepth as f32),
                    Rgba::from(color),
                    1.0,
                    Repeat::default(),
                );
                ScriptSpriteBatch { batch }
            },
        )
        .register_fn("vertices", |batch: &mut ScriptSpriteBatch| ScriptSpriteVertexList {
            vertices: batch.batch.vertices(),
        })
        .register_type_with_name::<ScriptSpriteVertexList>("ScriptSpriteVertexList")
        .register_fn("to_string", |list: &mut ScriptSpriteVertexList| list.to_string())
        .register_fn("to_debug", |list: &mut ScriptSpriteVertexList| format!("{list:?}"))
        .register_fn(
            "create_vertex_buffer_from_sprite_vertices",
            |r: &mut Rc<RefCell<wgpu::Renderer>>, list: ScriptSpriteVertexList| {
                r.borrow_mut()
                    .create_vertex_buffer_from_sprite_vertices(&list.vertices) as i64
            },
        )
        .register_fn(
            "create_fullscreen_triangle_vertex_buffer",
            |r: &mut Rc<RefCell<wgpu::Renderer>>| {
                r.borrow_mut().create_fullscreen_triangle_vertex_buffer() as i64
            },
        )
        .register_raw_fn(
            "begin_render_pass",
            [
                TypeId::of::<Rc<RefCell<wgpu::Renderer>>>(),
                TypeId::of::<Rc<RefCell<wgpu::Encoder>>>(),
                TypeId::of::<ImmutableString>(),
                TypeId::of::<Array>(),
            ],
            move |_ctx, args| {
                use std::cell::Ref;
                let r = args[0]
                    .clone()
                    .try_cast::<Rc<RefCell<wgpu::Renderer>>>()
                    .ok_or_else(|| {
                        rhai::EvalAltResult::ErrorMismatchDataType(
                            "Renderer".into(),
                            args[0].type_name().into(),
                            rhai::Position::NONE,
                        )
                    })?;
                let encoder = args[1]
                    .clone()
                    .try_cast::<Rc<RefCell<wgpu::Encoder>>>()
                    .ok_or_else(|| {
                        rhai::EvalAltResult::ErrorMismatchDataType(
                            "Encoder".into(),
                            args[1].type_name().into(),
                            rhai::Position::NONE,
                        )
                    })?;
                let label = args[2].clone().into_immutable_string().unwrap_or_default();
                let attachments = args[3]
                    .clone()
                    .try_cast::<Array>()
                    .ok_or_else(|| {
                        rhai::EvalAltResult::ErrorMismatchDataType(
                            "Array".into(),
                            args[3].type_name().into(),
                            rhai::Position::NONE,
                        )
                    })?;
                let renderer = r.borrow();
                let mut texture_refs: Vec<Ref<RenderTexture>> = Vec::new();
                let mut ops_list: Vec<(LoadOp<Color>, StoreOp)> = Vec::new();
                for att in attachments.iter() {
                    let Some(att) = att.clone().try_cast::<ScriptColorAttachment>() else {
                        continue;
                    };
                    if let RenderTextureHandle::ViewLayer(vid) = att.handle {
                        let vd = match renderer.view_data.get(&vid) {
                            Some(v) => v,
                            None => continue,
                        };
                        let load_op = match &att.load_op {
                            ScriptLoadOp::Load => LoadOp::Load,
                            ScriptLoadOp::Clear { r, g, b, a } => LoadOp::Clear(Color {
                                r: *r,
                                g: *g,
                                b: *b,
                                a: *a,
                            }),
                        };
                        let store_op = match &att.store_op {
                            ScriptStoreOp::Store => StoreOp::Store,
                        };
                        texture_refs.push(vd.layer.texture.borrow());
                        ops_list.push((load_op, store_op));
                    }
                }
                if texture_refs.is_empty() {
                    return Ok(Dynamic::UNIT);
                }
                let color_attachments: Vec<Option<RenderPassColorAttachment>> = texture_refs
                    .iter()
                    .zip(ops_list.iter())
                    .map(|(r, (load_op, store_op))| {
                        let ops = Operations {
                            load: load_op.clone(),
                            store: *store_op,
                        };
                        Some(RenderPassColorAttachment {
                            view: r.view(),
                            resolve_target: None,
                            ops,
                        })
                    })
                    .collect();
                let descriptor = RenderPassDescriptor {
                    label: Some(label.as_str()),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                };
                let mut encoder_mut = encoder.borrow_mut();
                let pass = encoder_mut.begin_render_pass(&descriptor);
                let pass_handle = Rc::new(RefCell::new(pass.forget_lifetime()));
                let script_pass = ScriptPass::new(pass_handle, r.clone());
                Ok(Dynamic::from(script_pass))
            },
        );
}

/// Register Vector2<f32> and Rgba8 for script use. vec2(x,y), rgb8(r,g,b), rgb8(r,g,b,a).
fn register_draw_types(engine: &mut Engine) {
    engine
        .register_type_with_name::<Vector2<f32>>("Vector2")
        .register_get("x", |v: &mut Vector2<f32>| v.x as f64)
        .register_get("y", |v: &mut Vector2<f32>| v.y as f64)
        .register_fn("vec2", |x: f64, y: f64| Vector2::new(x as f32, y as f32))
        .register_fn("+", |v1: Vector2<f32>, v2: Vector2<f32>| v1 + v2);

    engine
        .register_type_with_name::<Rgba8>("Rgba8")
        .register_get("r", |c: &mut Rgba8| c.r as i64)
        .register_get("g", |c: &mut Rgba8| c.g as i64)
        .register_get("b", |c: &mut Rgba8| c.b as i64)
        .register_get("a", |c: &mut Rgba8| c.a as i64)
        .register_fn("rgb8", |r: i64, g: i64, b: i64| {
            Rgba8::new(
                r.clamp(0, 255) as u8,
                g.clamp(0, 255) as u8,
                b.clamp(0, 255) as u8,
                255,
            )
        })
        .register_fn("rgb8", |r: i64, g: i64, b: i64, a: i64| {
            Rgba8::new(
                r.clamp(0, 255) as u8,
                g.clamp(0, 255) as u8,
                b.clamp(0, 255) as u8,
                a.clamp(0, 255) as u8,
            )
        });

    engine
        .register_type_with_name::<Rect<f32>>("Rect")
        .register_get("x1", |r: &mut Rect<f32>| r.x1 as f64)
        .register_get("y1", |r: &mut Rect<f32>| r.y1 as f64)
        .register_get("x2", |r: &mut Rect<f32>| r.x2 as f64)
        .register_get("y2", |r: &mut Rect<f32>| r.y2 as f64)
        .register_fn("rect", |x1: f64, y1: f64, x2: f64, y2: f64| {
            Rect::new(x1 as f32, y1 as f32, x2 as f32, y2 as f32)
        })
        .register_fn("+", |r: Rect<f32>, v: Vector2<f32>| r + v)
        .register_fn("*", |r: Rect<f32>, v: f32| r * v);

    engine
        .register_type_with_name::<ZDepth>("ZDepth")
        .register_fn("zdepth", |z: f64| ZDepth(z as f32));

    engine
        .register_type_with_name::<Repeat>("Repeat")
        .register_get("x", |r: &mut Repeat| r.x as f64)
        .register_get("y", |r: &mut Repeat| r.y as f64)
        .register_fn("repeat", |x: f64, y: f64| Repeat::new(x as f32, y as f32));
}

/// Register draw primitives on the engine. Call this once when loading a script.
/// The batches are shared; `draw_line` adds shapes, `draw_text` adds text sprites.
pub fn register_draw_primitives(
    engine: &mut Engine,
    shape_batch: Rc<RefCell<shape2d::Batch>>,
    sprite_batch: Rc<RefCell<Option<sprite2d::Batch>>>,
) {
    register_draw_types(engine);

    let shape_batch_line = shape_batch.clone();
    engine.register_fn(
        "draw_line",
        move |p1: Vector2<f32>, p2: Vector2<f32>| {
            let shape = Shape::Line(
                Line::new(Point2::new(p1.x, p1.y), Point2::new(p2.x, p2.y)),
                USER_LAYER,
                Rotation::ZERO,
                Stroke::new(1.0, Rgba::WHITE),
            );
            shape_batch_line.borrow_mut().add(shape);
        },
    );
    engine.register_fn(
        "draw_line",
        move |p1: Vector2<f32>, p2: Vector2<f32>, color: Rgba8| {
            let color: Rgba = color.into();
            let shape = Shape::Line(
                Line::new(Point2::new(p1.x, p1.y), Point2::new(p2.x, p2.y)),
                USER_LAYER,
                Rotation::ZERO,
                Stroke::new(1.0, color),
            );
            shape_batch.borrow_mut().add(shape);
        },
    );

    const FONT_OFFSET: usize = 32;
    let gw = draw::GLYPH_WIDTH;
    let gh = draw::GLYPH_HEIGHT;

    let sprite_batch_text = sprite_batch.clone();
    engine.register_fn("draw_text", move |pos: Vector2<f32>, text: &str| {
        if let Some(ref mut batch) = *sprite_batch_text.borrow_mut() {
            let mut sx = pos.x;
            let sy = pos.y;
            for c in text.bytes() {
                let i = c as usize - FONT_OFFSET;
                let tx = (i % 16) as f32 * gw;
                let ty = (i / 16) as f32 * gh;
                batch.add(
                    Rect::new(tx, ty, tx + gw, ty + gh),
                    Rect::new(sx, sy, sx + gw, sy + gh),
                    USER_LAYER,
                    Rgba::WHITE,
                    1.0,
                    Repeat::default(),
                );
                sx += gw;
            }
        }
    });
    engine.register_fn("draw_text", move |pos: Vector2<f32>, text: &str, color: Rgba8| {
        let color: Rgba = color.into();
        if let Some(ref mut batch) = *sprite_batch.borrow_mut() {
            let mut sx = pos.x;
            let sy = pos.y;
            for c in text.bytes() {
                let i = c as usize - FONT_OFFSET;
                let tx = (i % 16) as f32 * gw;
                let ty = (i / 16) as f32 * gh;
                batch.add(
                    Rect::new(tx, ty, tx + gw, ty + gh),
                    Rect::new(sx, sy, sx + gw, sy + gh),
                    USER_LAYER,
                    color,
                    1.0,
                    Repeat::default(),
                );
                sx += gw;
            }
        }
    });
}

pub fn register_wgpu_types(engine: &mut Engine) {
    engine.register_type_with_name::<Rc<RefCell<wgpu::Encoder>>>("Encoder");
}

/// Register the `register_command(name, help)` function for scripts to register custom commands.
/// Commands are collected in the provided `Rc<RefCell<Vec<(String, String)>>>`.
fn register_command_api(engine: &mut Engine, commands: Rc<RefCell<Vec<(String, String)>>>) {
    engine.register_fn("register_command", move |name: &str, help: &str| {
        commands.borrow_mut().push((name.to_string(), help.to_string()));
    });
}

fn register_constants(scope: &mut Scope) {
    scope.push_constant("TEXTURE_FORMAT_RGBA8_UNORM", TextureFormat::Rgba8Unorm);
}

/// Call the script's `init(session, renderer)` function with options so that new variables
/// introduced in the scope are retained (rewind_scope false) and the AST
/// is not re-evaluated (eval_ast false).
///
/// If the script does not define `init`, this is a no-op (no error).
pub fn call_init(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    session_handle: &Rc<RefCell<Session>>,
    renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let options = CallFnOptions::new()
        .eval_ast(false)
        .rewind_scope(false);

    let session = session_handle.clone();
    let renderer = renderer_handle.clone();
    match engine.call_fn_with_options::<()>(options, scope, ast, "init", (session, renderer)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `draw()` event handler.
/// Script uses global_session (set in init); no session argument.
///
/// If the script does not define `draw`, this is a no-op (no error).
pub fn call_draw(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "draw", ()) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `shade(encoder)` event handler.

pub fn call_shade(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    encoder: &Rc<RefCell<wgpu::Encoder>>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "shade", (encoder.clone(),)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

pub fn call_render(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    script_pass: ScriptPass,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "render", (script_pass,)) {
        Ok(()) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `view_added(view_id)` handler.
fn call_view_added(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    view_id: i64,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "view_added", (view_id,)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `view_removed(view_id)` handler.
fn call_view_removed(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    view_id: i64,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "view_removed", (view_id,)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

fn is_function_not_found(e: &Box<rhai::EvalAltResult>) -> bool {
    use rhai::EvalAltResult;
    matches!(&**e, EvalAltResult::ErrorFunctionNotFound(_, _))
}

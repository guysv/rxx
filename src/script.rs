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
use crate::gfx::{Point, sprite2d};
use crate::gfx::ZDepth;
use crate::gfx::{Repeat, Rgba8};
use crate::platform::{InputState, LogicalDelta, MouseButton};
use crate::session::{Effect, MessageType, Mode, ModeString, ScriptEffect, Session, VisualState};
use crate::view::{View, ViewExtent, ViewId, ViewResource};
use crate::wgpu::{self, Texture};
use ::wgpu as wgpu_types;

use rhai::{Array, CallFnOptions, Dynamic, Engine, ImmutableString, Scope, AST};

use std::any::TypeId;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::SystemTime;

/// Type alias for the user batches shared between script and session.
/// (shape batch, sprite batch for text). Sprite batch is created lazily with font size.
#[allow(dead_code)]
pub type UserBatch = (
    Rc<RefCell<shape2d::Batch>>,
    Rc<RefCell<Option<sprite2d::Batch>>>,
);

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
        ScriptView {
            id: view.id,
            offset: view.offset,
            frame_width: view.fw as f32,
            frame_height: view.fh as f32,
            zoom: view.zoom,
        }
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
        write!(
            f,
            "ScriptSpriteVertexList {{ vertices: {:?} }}",
            self.vertices
        )
    }
}

impl std::fmt::Display for ScriptSpriteVertexList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScriptSpriteVertexList {{ vertices: {:?} }}",
            self.vertices
        )
    }
}

/// Per-plugin state: one Rhai engine, scope, and AST per .rxx file.
pub struct LoadedPlugin {
    /// Path to the .rxx script file.
    pub path: PathBuf,
    /// Mtime of the script file after last successful load (for hot-reload).
    pub mtime: Option<SystemTime>,
    pub engine: Engine,
    pub scope: Scope<'static>,
    pub ast: Rc<RefCell<AST>>,
}

/// State for Rhai plugins (event-handler style). Each plugin has its own engine and ScriptState.
/// Built in lib.rs and passed to renderer.frame() and draw_ctx.draw().
pub struct ScriptState {
    /// Plugin directory (where *.rxx were discovered).
    pub plugin_dir: Option<PathBuf>,
    /// Loaded plugins, one per .rxx file.
    pub plugins: Vec<LoadedPlugin>,
}

/// Discover *.rxx files in a directory. Returns paths sorted by name for deterministic load order.
pub fn discover_rxx(plugin_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(plugin_dir)
        .map_err(|e| format!("Plugin dir read error: {}", e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map(|ext| ext == "rxx").unwrap_or(false))
        .collect();
    paths.sort_by_cached_key(|p| p.file_name().unwrap_or_default().to_owned());
    Ok(paths)
}

/// Load a single plugin from a path. Returns the loaded plugin and its registered commands.
fn load_one_plugin(
    path: &Path,
    session_handle: &Rc<RefCell<Session>>,
    renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
) -> Result<(LoadedPlugin, Vec<(String, String)>), String> {
    if !path.exists() {
        return Err(format!("Plugin not found: {}", path.display()));
    }
    let (shape_batch, sprite_batch) = renderer_handle.borrow().user_batch();

    let mut engine = Engine::new();
    register_draw_primitives(&mut engine, shape_batch, sprite_batch);
    register_session_handle(&mut engine);
    register_renderer_handle(&mut engine, renderer_handle);
    register_wgpu_types(&mut engine);

    let script_commands: Rc<RefCell<Vec<(String, String)>>> = Rc::new(RefCell::new(Vec::new()));
    register_command_api(&mut engine, script_commands.clone());

    let ast = compile_file(&engine, path).map_err(|e| format!("Script compile error: {}", e))?;
    let mut scope = Scope::new();
    register_constants(&mut scope);
    call_init(&engine, &mut scope, &ast, session_handle, renderer_handle)
        .map_err(|e| format!("Script init error: {}", e))?;

    let cmds = script_commands.borrow().clone();
    let mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
    let plugin = LoadedPlugin {
        path: path.to_path_buf(),
        mtime,
        engine,
        scope,
        ast: Rc::new(RefCell::new(ast)),
    };
    Ok((plugin, cmds))
}

impl ScriptState {
    pub fn new() -> Self {
        Self {
            plugin_dir: None,
            plugins: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn set_plugin_dir(&mut self, dir: PathBuf) {
        self.plugin_dir = Some(dir);
    }
}

/// Load or reload all plugins from the plugin directory. Discovers *.rxx, loads each,
/// merges script commands, and replaces ScriptState. Returns an error message for the caller.
pub fn load_plugins(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    session_handle: &Rc<RefCell<Session>>,
    renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
    plugin_dir: PathBuf,
) -> Result<(), String> {
    if !plugin_dir.is_dir() {
        return Err(format!("Plugin dir is not a directory: {}", plugin_dir.display()));
    }
    let paths = discover_rxx(&plugin_dir)?;
    let mut all_commands = Vec::new();
    let mut plugins = Vec::with_capacity(paths.len());
    for path in &paths {
        match load_one_plugin(path, session_handle, renderer_handle) {
            Ok((plugin, cmds)) => {
                all_commands.extend(cmds);
                plugins.push(plugin);
            }
            Err(e) => return Err(format!("{}: {}", path.display(), e)),
        }
    }
    session_handle.borrow_mut().set_script_commands(all_commands);
    let mut state = script_state_handle.borrow_mut();
    state.plugin_dir = Some(plugin_dir);
    state.plugins = plugins;
    Ok(())
}

/// Reload all plugins from the current plugin directory. Errors are ignored.
pub fn reload_plugins(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    session_handle: &Rc<RefCell<Session>>,
    renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
) {
    let plugin_dir = script_state_handle.borrow().plugin_dir.clone();
    if let Some(dir) = plugin_dir {
        let _ = load_plugins(script_state_handle, session_handle, renderer_handle, dir);
    }
}

impl ScriptState {
    /// True if any plugin file on disk is newer than the last load, or the set of .rxx files changed.
    pub fn script_file_modified_since_load(&self) -> bool {
        let dir = match &self.plugin_dir {
            Some(d) => d,
            None => return false,
        };
        let current_paths = match discover_rxx(dir) {
            Ok(p) => p,
            Err(_) => return false,
        };
        if current_paths.len() != self.plugins.len() {
            return true;
        }
        for (i, path) in current_paths.iter().enumerate() {
            if let Some(plugin) = self.plugins.get(i) {
                if plugin.path != *path {
                    return true;
                }
                let current = match std::fs::metadata(path).ok().and_then(|m| m.modified().ok()) {
                    Some(t) => t,
                    None => return true,
                };
                if plugin.mtime.map(|last| current > last).unwrap_or(true) {
                    return true;
                }
            } else {
                return true;
            }
        }
        false
    }

    /// Plugin directory, if one is set.
    pub fn plugin_dir(&self) -> Option<&PathBuf> {
        self.plugin_dir.as_ref()
    }

    /// Traverse effects: call each plugin's `view_added` / `view_removed` handlers and run script
    /// commands (try each plugin until one handles it). Returns effects not consumed for the renderer.
    pub fn call_view_effects(
        &mut self,
        effects: &[Effect],
        session_handle: &Rc<RefCell<Session>>,
    ) -> Vec<Effect> {
        let mut renderer_effects = Vec::new();
        let mut script_commands = Vec::new();
        for eff in effects {
            match eff {
                Effect::ViewAdded(id) => {
                    for plugin in &mut self.plugins {
                        let _ = call_view_added(
                            &plugin.engine,
                            &mut plugin.scope,
                            &plugin.ast.borrow(),
                            id.raw() as i64,
                        );
                    }
                    renderer_effects.push(eff.clone());
                }
                Effect::ViewRemoved(id) => {
                    for plugin in &mut self.plugins {
                        let _ = call_view_removed(
                            &plugin.engine,
                            &mut plugin.scope,
                            &plugin.ast.borrow(),
                            id.raw() as i64,
                        );
                    }
                    renderer_effects.push(eff.clone());
                }
                Effect::ScriptEffect(ScriptEffect::RunScriptCommand(name, args)) => {
                    script_commands.push((name.clone(), args.clone()));
                }
                Effect::ScriptEffect(ScriptEffect::MouseInput(state, button, p)) => {
                    for plugin in &mut self.plugins {
                        let _ = call_mouse_input(
                            &plugin.engine,
                            &mut plugin.scope,
                            &plugin.ast.borrow(),
                            state,
                            button,
                            p,
                        );
                    }
                    renderer_effects.push(eff.clone());
                }
                Effect::ScriptEffect(ScriptEffect::MouseWheel(delta)) => {
                    for plugin in &mut self.plugins {
                        let _ = call_mouse_wheel(
                            &plugin.engine,
                            &mut plugin.scope,
                            &plugin.ast.borrow(),
                            delta,
                        );
                    }
                    renderer_effects.push(eff.clone());
                }
                Effect::ScriptEffect(ScriptEffect::CursorMoved(p)) => {
                    for plugin in &mut self.plugins {
                        let _ = call_cursor_moved(
                            &plugin.engine,
                            &mut plugin.scope,
                            &plugin.ast.borrow(),
                            p,
                        );
                    }
                    renderer_effects.push(eff.clone());
                }
                other => renderer_effects.push(other.clone()),
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

    /// Call each plugin's `draw()` event handler. User batch is cleared once before the first plugin.
    /// The script's draw primitives (e.g. `draw_line`, `draw_text`) mutate the user batches directly.
    pub fn call_draw_event(
        &mut self,
        user_batch: &(
            Rc<RefCell<shape2d::Batch>>,
            Rc<RefCell<Option<sprite2d::Batch>>>,
        ),
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        user_batch.0.borrow_mut().clear();
        if let Some(ref mut b) = *user_batch.1.borrow_mut() {
            b.clear();
        }
        for plugin in &mut self.plugins {
            call_draw(&plugin.engine, &mut plugin.scope, &plugin.ast.borrow())?;
        }
        Ok(())
    }

    pub fn call_shade_event(
        &mut self,
        encoder: &Rc<RefCell<wgpu_types::CommandEncoder>>,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        for plugin in &mut self.plugins {
            call_shade(
                &plugin.engine,
                &mut plugin.scope,
                &plugin.ast.borrow(),
                encoder,
            )?;
        }
        Ok(())
    }

    pub fn call_render_event(
        &mut self,
        script_pass: ScriptPass,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        for plugin in &mut self.plugins {
            call_render(
                &plugin.engine,
                &mut plugin.scope,
                &plugin.ast.borrow(),
                script_pass.clone(),
            )?;
        }
        Ok(())
    }

    /// Call a script command handler `cmd_<name>(args)` on each plugin in turn.
    /// Returns Ok(true) if any plugin handled it, Ok(false) if none.
    pub fn call_script_command(
        &mut self,
        name: &str,
        args: Vec<String>,
    ) -> Result<bool, Box<rhai::EvalAltResult>> {
        let handler_name = format!("cmd_{}", name.replace('/', "_"));
        let rhai_args: Array = args.into_iter().map(Dynamic::from).collect();
        for plugin in &mut self.plugins {
            match plugin.engine.call_fn::<()>(
                &mut plugin.scope,
                &plugin.ast.borrow(),
                &handler_name,
                (rhai_args.clone(),),
            ) {
                Ok(()) => return Ok(true),
                Err(ref e) if is_function_not_found(e, &handler_name) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(false)
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
        .register_get("width", |s: &mut Rc<RefCell<Session>>| {
            s.borrow().width as f64
        })
        .register_get("height", |s: &mut Rc<RefCell<Session>>| {
            s.borrow().height as f64
        })
        .register_get("offset_x", |s: &mut Rc<RefCell<Session>>| {
            s.borrow().offset.x as f64
        })
        .register_get("offset_y", |s: &mut Rc<RefCell<Session>>| {
            s.borrow().offset.y as f64
        })
        .register_get("active_view_id", |s: &mut Rc<RefCell<Session>>| {
            s.borrow().views.active_id.raw() as i64
        })
        .register_get("mode", |s: &mut Rc<RefCell<Session>>| {
            s.borrow().mode.to_string()
        })
        .register_fn("switch_mode", |s: &mut Rc<RefCell<Session>>, mode: Mode| {
            s.borrow_mut().switch_mode(mode);
        })
        .register_fn("script_mode", |name: String| {
            Mode::ScriptMode(ModeString::try_from_str(name.as_str())
                .expect("Failed to convert string to ModeString"))
        })
        .register_type_with_name::<Mode>("Mode")
        .register_type_with_name::<ScriptView>("View")
        .register_get("id", |v: &mut ScriptView| v.id.raw() as i64)
        .register_get("offset", |v: &mut ScriptView| {
            Vector2::new(v.offset.x as f32, v.offset.y as f32)
        })
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
        })
        .register_type_with_name::<InputState>("InputState")
        .register_fn("to_string", |state: InputState| {
            format!("{:?}", state)
        })
        .register_type_with_name::<MouseButton>("MouseButton")
        .register_fn("to_string", |button: MouseButton| {
            format!("{:?}", button)
        })
        .register_type_with_name::<Point<ViewExtent, f32>>("Point")
        .register_fn("to_string", |p: Point<ViewExtent, f32>| {
            format!("{:?}", p)
        })
        .register_type_with_name::<LogicalDelta>("LogicalDelta")
        .register_fn("to_string", |delta: LogicalDelta| {
            format!("{:?}", delta)
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
    pub handle: Rc<RefCell<Texture>>,
    pub load_op: ScriptLoadOp,
    pub store_op: ScriptStoreOp,
}

/// Script-side binding type for bind group layout entries (no string parsing).
#[derive(Clone)]
pub enum ScriptBindingType {
    UniformBuffer,
    Texture2dFilterable,
    SamplerFiltering,
}

/// Script-side bind group layout entry: binding index, visibility (ShaderStages bits), type.
#[derive(Clone)]
pub struct ScriptBindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: u32,
    pub ty: ScriptBindingType,
}

/// Script-side bind group entry for create_bind_group: binding index + resource.
#[derive(Clone)]
pub enum ScriptBindGroupEntry {
    Buffer {
        binding: u32,
        buffer: Rc<RefCell<wgpu_types::Buffer>>,
    },
    Texture {
        binding: u32,
        texture: Rc<RefCell<Texture>>,
    },
    SamplerDefault { binding: u32 },
}

/// Unified pass type for script: wraps the wgpu pass and renderer so pass methods can resolve handles.
#[derive(Clone)]
pub struct ScriptPass {
    pass: Rc<RefCell<wgpu_types::RenderPass<'static>>>,
}

impl ScriptPass {
    pub fn new(
        pass: Rc<RefCell<wgpu_types::RenderPass<'static>>>,
    ) -> Self {
        Self {
            pass,
        }
    }

    pub fn set_pipeline(&mut self, handle: Rc<RefCell<wgpu_types::RenderPipeline>>) -> Result<(), Box<rhai::EvalAltResult>> {
        let pipeline = handle.borrow();
        self.pass.borrow_mut().set_pipeline(&pipeline);
        Ok(())
    }

    pub fn set_bind_group(
        &mut self,
        index: i64,
        handle: Rc<RefCell<wgpu_types::BindGroup>>,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        // let id = handle as u64;
        // let storage_ref = self.script_storage.borrow();
        // let bind_group = storage_ref.get_script_bind_group(id).ok_or_else(|| {
        //     rhai::EvalAltResult::ErrorRuntime(
        //         format!("script bind group handle not found: {}", id).into(),
        //         rhai::Position::NONE,
        //     )
        // })?;
        let bind_group = handle.borrow();
        self.pass
            .borrow_mut()
            .set_bind_group(index as u32, &*bind_group, &[]);
        Ok(())
    }

    pub fn set_vertex_buffer(
        &mut self,
        slot: i64,
        handle: Rc<RefCell<wgpu_types::Buffer>>,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        let buffer = handle.borrow();
        self.pass
            .borrow_mut()
            .set_vertex_buffer(slot as u32, buffer.slice(..));
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
) {
    engine
        .register_type_with_name::<wgpu_types::TextureFormat>("TextureFormat")
        .register_type_with_name::<Rc<RefCell<Texture>>>("TextureHandle")
        .register_type_with_name::<ScriptLoadOp>("LoadOp")
        .register_type_with_name::<ScriptStoreOp>("StoreOp")
        .register_type_with_name::<ScriptColorAttachment>("ColorAttachment")
        .register_type_with_name::<ScriptPass>("ScriptPass")
        .register_fn("load_load", || ScriptLoadOp::Load)
        .register_fn("load_clear", |r: f64, g: f64, b: f64, a: f64| {
            ScriptLoadOp::Clear { r, g, b, a }
        })
        .register_fn("store_store", || ScriptStoreOp::Store)
        .register_fn(
            "color_attachment",
            |handle: Rc<RefCell<Texture>>, load_op: ScriptLoadOp, store_op: ScriptStoreOp| {
                ScriptColorAttachment {
                    handle,
                    load_op,
                    store_op,
                }
            },
        )
        .register_type_with_name::<ScriptBindingType>("BindingType")
        .register_type_with_name::<ScriptBindGroupLayoutEntry>("BindGroupLayoutEntry")
        .register_type_with_name::<ScriptBindGroupEntry>("BindGroupEntry")
        .register_type_with_name::<Rc<RefCell<wgpu_types::BindGroupLayout>>>("BindGroupLayoutHandle")
        .register_fn("binding_uniform_buffer", || ScriptBindingType::UniformBuffer)
        .register_fn("binding_texture_2d", || ScriptBindingType::Texture2dFilterable)
        .register_fn("binding_sampler_filtering", || ScriptBindingType::SamplerFiltering)
        .register_fn(
            "layout_entry",
            |binding: i64, visibility: i64, ty: ScriptBindingType| ScriptBindGroupLayoutEntry {
                binding: binding as u32,
                visibility: visibility as u32,
                ty,
            },
        )
        .register_fn(
            "bind_buffer",
            |binding: i64, buffer: Rc<RefCell<wgpu_types::Buffer>>| {
                ScriptBindGroupEntry::Buffer {
                    binding: binding as u32,
                    buffer,
                }
            },
        )
        .register_fn(
            "bind_texture",
            |binding: i64, texture: Rc<RefCell<Texture>>| ScriptBindGroupEntry::Texture {
                binding: binding as u32,
                texture,
            },
        )
        .register_fn("bind_sampler_default", |binding: i64| {
            ScriptBindGroupEntry::SamplerDefault {
                binding: binding as u32,
            }
        })
        .register_fn(
            "set_pipeline",
            |pass: &mut ScriptPass, handle: Rc<RefCell<wgpu_types::RenderPipeline>>| -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_pipeline(handle.clone())
            },
        )
        .register_fn(
            "set_bind_group",
            |pass: &mut ScriptPass,
             index: i64,
             handle: Rc<RefCell<wgpu_types::BindGroup>>|
             -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_bind_group(index, handle.clone())
            },
        )
        .register_fn(
            "set_vertex_buffer",
            |pass: &mut ScriptPass,
             slot: i64,
             handle: Rc<RefCell<wgpu_types::Buffer>>|
             -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_vertex_buffer(slot, handle.clone())
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
        .register_fn("create_render_texture", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, width: i64, height: i64| {
                let texture = r.borrow_mut()
                    .create_render_texture(width as u32, height as u32);
                Rc::new(RefCell::new(texture))
            }
        })
        .register_fn("create_compute_texture", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  width: i64,
                  height: i64| {
                let texture = r.borrow_mut().create_compute_texture(
                    width as u32,
                    height as u32,
                    1,
                );
                Rc::new(RefCell::new(texture))
            }
        })
        .register_fn("create_compute_texture", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  width: i64,
                  height: i64,
                  depth: i64| {
                let texture = r.borrow_mut().create_compute_texture(
                    width as u32,
                    height as u32,
                    depth as u32,
                );
                Rc::new(RefCell::new(texture))
            }
        })
        .register_fn("view_render_texture", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, view_id: i64| -> Dynamic {
                let id = ViewId(view_id as u16);
                match r.borrow().view_render_texture(id) {
                    Some(h) => Dynamic::from(h),
                    None => Dynamic::UNIT,
                }
            }
        })
        .register_fn("create_texture_sampler_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, handle: Rc<RefCell<Texture>>| {
                let texture = handle.borrow();
                let bind_group = r.borrow_mut()
                    .create_texture_sampler_bind_group(&texture);
                Rc::new(RefCell::new(bind_group))
            }
        })
        .register_fn("create_shader_module", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, wgsl_source: ImmutableString| {
                let module = r.borrow_mut()
                    .create_shader_module(None, wgsl_source.as_str());
                Rc::new(RefCell::new(module))
            }
        })
        .register_fn("create_render_pipeline", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  module_handle: Rc<RefCell<wgpu_types::ShaderModule>>,
                  vs_entry: ImmutableString,
                  fs_entry: ImmutableString| {
                let pipeline = r.borrow_mut().create_render_pipeline(
                    &module_handle.borrow(),
                    vs_entry.as_str(),
                    fs_entry.as_str(),
                );
                Rc::new(RefCell::new(pipeline))
            }
        })
        .register_fn("create_render_pipeline_with_texture", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  module_handle: Rc<RefCell<wgpu_types::ShaderModule>>,
                  vs_entry: ImmutableString,
                  fs_entry: ImmutableString| {
                let pipeline = r.borrow_mut().create_render_pipeline_with_texture(
                    &module_handle.borrow(),
                    vs_entry.as_str(),
                    fs_entry.as_str(),
                );
                Rc::new(RefCell::new(pipeline))
            }
        })
        .register_fn("create_bind_group_layout", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, entries: Array| {
                let layout_entries: Vec<ScriptBindGroupLayoutEntry> = entries
                    .iter()
                    .filter_map(|d| d.clone().try_cast::<ScriptBindGroupLayoutEntry>())
                    .collect();
                let layout = r
                    .borrow_mut()
                    .create_bind_group_layout_from_entries(&layout_entries);
                Rc::new(RefCell::new(layout))
            }
        })
        .register_fn("create_render_pipeline", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  module_handle: Rc<RefCell<wgpu_types::ShaderModule>>,
                  vs_entry: ImmutableString,
                  fs_entry: ImmutableString,
                  bind_group_layouts: Array| {
                use std::ops::Deref;
                let handles: Vec<Rc<RefCell<wgpu_types::BindGroupLayout>>> = bind_group_layouts
                    .iter()
                    .filter_map(|d| d.clone().try_cast::<Rc<RefCell<wgpu_types::BindGroupLayout>>>())
                    .collect();
                let ref_guards: Vec<std::cell::Ref<wgpu_types::BindGroupLayout>> =
                    handles.iter().map(|h| h.borrow()).collect();
                let layout_refs: Vec<&wgpu_types::BindGroupLayout> =
                    ref_guards.iter().map(Deref::deref).collect();
                let pipeline = r.borrow_mut().create_render_pipeline_with_layouts(
                    &module_handle.borrow(),
                    vs_entry.as_str(),
                    fs_entry.as_str(),
                    &layout_refs,
                );
                Rc::new(RefCell::new(pipeline))
            }
        })
        .register_fn("create_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  layout_handle: Rc<RefCell<wgpu_types::BindGroupLayout>>,
                  entries: Array| {
                let bind_entries: Vec<ScriptBindGroupEntry> = entries
                    .iter()
                    .filter_map(|d| d.clone().try_cast::<ScriptBindGroupEntry>())
                    .collect();
                let layout = layout_handle.borrow();
                let bind_group = r
                    .borrow_mut()
                    .create_bind_group_from_entries(&layout, &bind_entries);
                Rc::new(RefCell::new(bind_group))
            }
        })
        .register_fn("create_buffer", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, size: i64, usage_str: ImmutableString| {
                let buffer = r.borrow_mut().create_buffer(size as u64, usage_str.as_str(), None);
                Rc::new(RefCell::new(buffer))
            }
        })
        .register_fn("create_buffer", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  size: i64,
                  usage_str: ImmutableString,
                  data: Array| {
                let bytes: Vec<u8> = data
                    .iter()
                    .map(|v| v.clone().cast::<i64>().clamp(0, 255) as u8)
                    .collect();
                let slice = if bytes.is_empty() {
                    None
                } else {
                    Some(bytes.as_slice())
                };
                let buffer = r.borrow_mut()
                    .create_buffer(size as u64, usage_str.as_str(), slice);
                Rc::new(RefCell::new(buffer))
            }
        })
        .register_fn("create_transform_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, buffer_handle: Rc<RefCell<wgpu_types::Buffer>>| {
                let buffer = buffer_handle.borrow();
                let bind_group = r.borrow_mut()
                    .create_transform_bind_group(&buffer);
                Rc::new(RefCell::new(bind_group))
            }
        })
        .register_fn("create_identity_transform_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>| {
                let bind_group = r.borrow_mut()
                    .create_identity_transform_bind_group();
                Rc::new(RefCell::new(bind_group))
            }
        })
        .register_fn("create_view_transform_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  translation_x: f64,
                  translation_y: f64,
                  zoom: f64| {
                let bind_group = r.borrow_mut().create_view_transform_bind_group(
                    translation_x as f32,
                    translation_y as f32,
                    zoom as f32,
                );
                Rc::new(RefCell::new(bind_group))
            }
        })
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
        .register_fn("vertices", |batch: &mut ScriptSpriteBatch| {
            ScriptSpriteVertexList {
                vertices: batch.batch.vertices(),
            }
        })
        .register_type_with_name::<ScriptSpriteVertexList>("ScriptSpriteVertexList")
        .register_fn("to_string", |list: &mut ScriptSpriteVertexList| {
            list.to_string()
        })
        .register_fn("to_debug", |list: &mut ScriptSpriteVertexList| {
            format!("{list:?}")
        })
        .register_fn("create_vertex_buffer_from_sprite_vertices", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, list: ScriptSpriteVertexList| {
                let buffer = r.borrow_mut()
                    .create_vertex_buffer_from_sprite_vertices(&list.vertices);
                Rc::new(RefCell::new(buffer))
            }
        })
        .register_fn("create_fullscreen_triangle_vertex_buffer", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>| {
                let buffer = r.borrow_mut()
                    .create_fullscreen_triangle_vertex_buffer();
                Rc::new(RefCell::new(buffer))
            }
        })
        .register_raw_fn(
            "begin_compute_pass",
            [
                TypeId::of::<Rc<RefCell<wgpu_types::CommandEncoder>>>(),
                TypeId::of::<ImmutableString>(),
            ],
            {
                move |_ctx, args| {
                    let encoder = args[0]
                        .clone()
                        .try_cast::<Rc<RefCell<wgpu_types::CommandEncoder>>>()
                        .ok_or_else(|| {
                            rhai::EvalAltResult::ErrorMismatchDataType(
                                "Encoder".into(),
                                args[0].type_name().into(),
                                rhai::Position::NONE,
                            )
                        })?;
                    let label = args[1].clone().into_immutable_string().unwrap_or_default();
                    let descriptor = wgpu_types::ComputePassDescriptor {
                        label: Some(label.as_str()),
                        timestamp_writes: None,
                    };
                    let mut encoder_mut = encoder.borrow_mut();
                    let pass = encoder_mut.begin_compute_pass(&descriptor);
                    let pass_handle = Rc::new(RefCell::new(pass.forget_lifetime()));
                    Ok(Dynamic::from(pass_handle))
                }
            },
        )
        .register_raw_fn(
            "begin_render_pass",
            [
                TypeId::of::<Rc<RefCell<wgpu_types::CommandEncoder>>>(),
                TypeId::of::<ImmutableString>(),
                TypeId::of::<Array>(),
            ],
            {
                move |_ctx, args| {
                    use std::cell::Ref;
                    let encoder = args[0]
                        .clone()
                        .try_cast::<Rc<RefCell<wgpu_types::CommandEncoder>>>()
                        .ok_or_else(|| {
                            rhai::EvalAltResult::ErrorMismatchDataType(
                                "Encoder".into(),
                                args[0].type_name().into(),
                                rhai::Position::NONE,
                            )
                        })?;
                    let label = args[1].clone().into_immutable_string().unwrap_or_default();
                    let attachments = args[2].clone().try_cast::<Array>().ok_or_else(|| {
                        rhai::EvalAltResult::ErrorMismatchDataType(
                            "Array".into(),
                            args[2].type_name().into(),
                            rhai::Position::NONE,
                        )
                    })?;
                    // Collect handles first so they outlive the Ref borrows in texture_refs.
                    let mut handles: Vec<Rc<RefCell<Texture>>> = Vec::new();
                    let mut ops_list: Vec<(
                        wgpu_types::LoadOp<wgpu_types::Color>,
                        wgpu_types::StoreOp,
                    )> = Vec::new();
                    for att in attachments.iter() {
                        let Some(att) = att.clone().try_cast::<ScriptColorAttachment>() else {
                            continue;
                        };
                        let load_op = match &att.load_op {
                            ScriptLoadOp::Load => wgpu_types::LoadOp::Load,
                            ScriptLoadOp::Clear { r, g, b, a } => {
                                wgpu_types::LoadOp::Clear(wgpu_types::Color {
                                    r: *r,
                                    g: *g,
                                    b: *b,
                                    a: *a,
                                })
                            }
                        };
                        let store_op = match &att.store_op {
                            ScriptStoreOp::Store => wgpu_types::StoreOp::Store,
                        };
                        handles.push(att.handle.clone());
                        ops_list.push((load_op, store_op));
                    }
                    let texture_refs: Vec<Ref<Texture>> =
                        handles.iter().map(|h| h.borrow()).collect();
                    if texture_refs.is_empty() {
                        return Ok(Dynamic::UNIT);
                    }
                    let color_attachments: Vec<Option<wgpu_types::RenderPassColorAttachment>> =
                        texture_refs
                            .iter()
                            .zip(ops_list.iter())
                            .map(|(r, (load_op, store_op))| {
                                let ops = wgpu_types::Operations {
                                    load: load_op.clone(),
                                    store: *store_op,
                                };
                                Some(wgpu_types::RenderPassColorAttachment {
                                    view: r.view(),
                                    resolve_target: None,
                                    ops,
                                })
                            })
                            .collect();
                    let descriptor = wgpu_types::RenderPassDescriptor {
                        label: Some(label.as_str()),
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    };
                    let mut encoder_mut = encoder.borrow_mut();
                    let pass = encoder_mut.begin_render_pass(&descriptor);
                    let pass_handle = Rc::new(RefCell::new(pass.forget_lifetime()));
                    let script_pass = ScriptPass::new(pass_handle);
                    Ok(Dynamic::from(script_pass))
                }
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
    engine.register_fn("draw_line", move |p1: Vector2<f32>, p2: Vector2<f32>| {
        let shape = Shape::Line(
            Line::new(Point2::new(p1.x, p1.y), Point2::new(p2.x, p2.y)),
            USER_LAYER,
            Rotation::ZERO,
            Stroke::new(1.0, Rgba::WHITE),
        );
        shape_batch_line.borrow_mut().add(shape);
    });
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
    engine.register_fn(
        "draw_text",
        move |pos: Vector2<f32>, text: &str, color: Rgba8| {
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
        },
    );
}

pub fn register_wgpu_types(engine: &mut Engine) {
    engine.register_type_with_name::<Rc<RefCell<wgpu_types::CommandEncoder>>>("CommandEncoder");
}

/// Register the `register_command(name, help)` function for scripts to register custom commands.
/// Commands are collected in the provided `Rc<RefCell<Vec<(String, String)>>>`.
fn register_command_api(engine: &mut Engine, commands: Rc<RefCell<Vec<(String, String)>>>) {
    engine.register_fn("register_command", move |name: &str, help: &str| {
        commands
            .borrow_mut()
            .push((name.to_string(), help.to_string()));
    });
}

fn register_constants(scope: &mut Scope) {
    scope.push_constant(
        "TEXTURE_FORMAT_RGBA8_UNORM",
        wgpu_types::TextureFormat::Rgba8Unorm,
    );
    scope.push_constant("SHADER_STAGE_VERTEX", 1_i64);
    scope.push_constant("SHADER_STAGE_FRAGMENT", 2_i64);
    scope.push_constant("SHADER_STAGE_COMPUTE", 4_i64);
    scope.push_constant("MODE_NORMAL", Mode::Normal);
    scope.push_constant("MODE_VISUAL", Mode::Visual(VisualState::default()));
    scope.push_constant("MODE_COMMAND", Mode::Command);
    scope.push_constant("MODE_PRESENT", Mode::Present);
    scope.push_constant("MODE_HELP", Mode::Help);
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
    let options = CallFnOptions::new().eval_ast(false).rewind_scope(false);

    let session = session_handle.clone();
    let renderer = renderer_handle.clone();
    match engine.call_fn_with_options::<()>(options, scope, ast, "init", (session, renderer)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "init") => Ok(()),
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
        Err(ref e) if is_function_not_found(e, "draw") => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `shade(encoder)` event handler.

pub fn call_shade(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    encoder: &Rc<RefCell<wgpu_types::CommandEncoder>>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "shade", (encoder.clone(),)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "shade") => Ok(()),
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
        Err(ref e) if is_function_not_found(e, "render") => Ok(()),
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
        Err(ref e) if is_function_not_found(e, "view_added") => Ok(()),
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
        Err(ref e) if is_function_not_found(e, "view_removed") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_mouse_input(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    state: &InputState,
    button: &MouseButton,
    p: &Point<ViewExtent, f32>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "mouse_input", (state.clone(), button.clone(), p.clone())) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "mouse_input") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_mouse_wheel(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    delta: &LogicalDelta,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "mouse_wheel", (delta.clone(),)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "mouse_wheel") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_cursor_moved(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    p: &Point<ViewExtent, f32>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "cursor_moved", (p.clone(),)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "cursor_moved") => Ok(()),
        Err(e) => Err(e),
    }
}

fn is_function_not_found(e: &Box<rhai::EvalAltResult>, function_name: &str) -> bool {
    use rhai::EvalAltResult;
    matches!(&**e, EvalAltResult::ErrorFunctionNotFound(name, _) if name == function_name)
}

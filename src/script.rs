//! Rhai script loading and event-handler lifecycle.
//!
//! Follows the [event-handler pattern](https://rhai.rs/book/patterns/events-1.html):
//! one main script with `init()` and optional `unload()`; `session` and `renderer` are in scope as globals; custom Scope +
//! CallFnOptions (eval_ast false, rewind_scope false) so variables defined in `init()` persist.
//! `unload()` is called on each plugin just before unloading/reloading.

use crate::draw::{self, USER_LAYER};
use crate::gfx::color::Rgba;
use crate::gfx::math::{Matrix4, Point2, Vector2, Vector3};
use crate::gfx::rect::Rect;
use crate::gfx::shape2d::{self, Fill, Line, Rotation, Shape, Stroke};
use crate::gfx::{Point, sprite2d};
use crate::gfx::ZDepth;
use crate::gfx::{Repeat, Rgba8};
use crate::platform::{InputState, Key, LogicalDelta, MouseButton};
use crate::cmd::{Command, Value};
use crate::session::{Blending, Effect, MessageType, Mode, ModeString, ScriptEffect, Session, SessionCoords, VisualState};
use crate::view::{View, ViewExtent, ViewId, ViewResource};
use crate::wgpu::{self, Texture};
use ::wgpu as wgpu_types;

use rhai::{Array, CallFnOptions, Dynamic, Engine, ImmutableString, Scope, AST};

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::SystemTime;

/// Script file basename (file stem) for use as plugin key, e.g. "rotate-scale" from "rotate-scale.rxx".
fn plugin_basename(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

/// Opaque plugin function handle returned by `session.get_plugin_fn(addr)`.
/// The target is resolved and invoked when the handle is called.
#[derive(Clone, Debug)]
pub struct PluginFnRef {
    pub plugin_key: String,
    pub fn_name: String,
}

/// Parse "<plugin>/<function>" into a normalized plugin function handle.
/// Accepts plugin file names with extension (eg. "cleanedge.rxx/render_pass")
/// and normalizes to plugin basename ("cleanedge").
fn parse_plugin_fn_ref(addr: &str) -> Result<PluginFnRef, Box<rhai::EvalAltResult>> {
    let trimmed = addr.trim();
    let (plugin_raw, fn_raw) = trimmed
        .rsplit_once('/')
        .ok_or_else(|| format!("Invalid plugin function address '{}'", addr))?;
    let plugin_key = Path::new(plugin_raw)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let fn_name = fn_raw.trim().to_string();
    if plugin_key.is_empty() || fn_name.is_empty() {
        return Err(format!("Invalid plugin function address '{}'", addr).into());
    }
    Ok(PluginFnRef { plugin_key, fn_name })
}

/// Type alias for the user batches shared between script and session.
/// (shape batch, sprite batch for text). Sprite batch is created lazily with font size.
#[allow(dead_code)]
pub type UserBatch = (
    Rc<RefCell<shape2d::Batch>>,
    Rc<RefCell<Option<sprite2d::Batch>>>,
);

/// Dispatch a command with script-first semantics.
///
/// Must be called when `session_handle` is **not** borrowed, so that script
/// handlers can access the session. If a script handles the command
/// (`call_script_command` returns `Ok(true)`), the built-in is skipped.
/// Otherwise, the session is borrowed and the built-in runs.
pub fn dispatch_command(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    session_handle: &Rc<RefCell<Session>>,
    cmd: Command,
) {
    // Try script handlers first (session is NOT borrowed here).
    if let Some((name, args)) = cmd.to_invocation() {
        match call_script_command(script_state_handle, &name, args) {
            Ok(true) => return, // Script handled it.
            Ok(false) => {}     // Fall through to built-in.
            Err(e) => {
                session_handle.borrow_mut().message(
                    format!("Script command '{}' error: {}", name, e),
                    MessageType::Error,
                );
                // Fall through to built-in on error.
            }
        }
    }

    // For ScriptCommand (unknown to parser) that no script handled: error.
    if let Command::ScriptCommand(ref name, _) = cmd {
        session_handle.borrow_mut().message(
            format!(
                "Unknown command: '{}' (no script handler cmd_{})",
                name,
                name.replace('/', "_")
            ),
            MessageType::Error,
        );
        return;
    }

    // Script didn't handle it — run the built-in.
    session_handle.borrow_mut().command(cmd);
}

/// Read-only view handle exposed to Rhai scripts.
#[derive(Debug, Clone)]
struct ScriptView {
    id: ViewId,
    offset: Vector2<f64>,
    frame_width: f64,
    frame_height: f64,
    zoom: f64,
}

impl From<&View<ViewResource>> for ScriptView {
    fn from(view: &View<ViewResource>) -> Self {
        ScriptView {
            id: view.id,
            offset: view.offset.into(),
            frame_width: view.fw as f64,
            frame_height: view.fh as f64,
            zoom: view.zoom as f64,
        }
    }
}

/// Read-only setting value exposed to Rhai scripts. Returned by `session.get_setting(name)`.
#[derive(Clone, Debug)]
pub struct ScriptSettingValue(pub(crate) Option<Value>);

impl ScriptSettingValue {
    pub fn is_present(&mut self) -> bool {
        self.0.is_some()
    }

    /// Returns a bool if the setting is present and a bool, otherwise `()`.
    pub fn as_bool(&mut self) -> Dynamic {
        match self.0.as_ref().and_then(Value::try_is_set) {
            Some(v) => Dynamic::from(v),
            None => Dynamic::UNIT,
        }
    }

    /// Returns a float if the setting is present and a float, otherwise `()`.
    pub fn as_f64(&mut self) -> Dynamic {
        match self.0.as_ref().and_then(Value::try_to_f64) {
            Some(v) => Dynamic::from(v),
            None => Dynamic::UNIT,
        }
    }

    /// Returns an integer if the setting is present and an integer, otherwise `()`.
    pub fn as_int(&mut self) -> Dynamic {
        match self.0.as_ref().and_then(Value::try_to_u64).map(|u| u as i64) {
            Some(v) => Dynamic::from(v),
            None => Dynamic::UNIT,
        }
    }

    /// Returns a string if the setting is present and a string/ident, otherwise `()`.
    pub fn as_string(&mut self) -> Dynamic {
        match &self.0 {
            Some(Value::Str(s)) | Some(Value::Ident(s)) => Dynamic::from(s.clone()),
            _ => Dynamic::UNIT,
        }
    }

    /// Returns a color if the setting is present and a color, otherwise `()`.
    pub fn as_rgba8(&mut self) -> Dynamic {
        match self.0.as_ref().and_then(Value::try_to_rgba8) {
            Some(v) => Dynamic::from(v),
            None => Dynamic::UNIT,
        }
    }

    /// Returns `[a, b]` if the setting is present and a tuple, otherwise `()`.
    pub fn as_tuple(&mut self) -> Dynamic {
        match &self.0 {
            Some(Value::U32Tuple(a, b)) => {
                Dynamic::from(vec![Dynamic::from(*a as i64), Dynamic::from(*b as i64)])
            }
            Some(Value::F32Tuple(a, b)) => {
                Dynamic::from(vec![Dynamic::from(*a as f64), Dynamic::from(*b as f64)])
            }
            _ => Dynamic::UNIT,
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

impl ScriptSpriteVertexList {
    fn extents(&self) -> Option<(f32, f32, f32, f32)> {
        let mut it = self.vertices.iter();
        let first = it.next()?;
        let mut min_x = first.position.x;
        let mut min_y = first.position.y;
        let mut max_x = first.position.x;
        let mut max_y = first.position.y;
        for v in it {
            min_x = min_x.min(v.position.x);
            min_y = min_y.min(v.position.y);
            max_x = max_x.max(v.position.x);
            max_y = max_y.max(v.position.y);
        }
        Some((min_x, min_y, max_x, max_y))
    }

    pub fn width(&self) -> f64 {
        self.extents()
            .map(|(min_x, _, max_x, _)| (max_x - min_x) as f64)
            .unwrap_or(0.0)
    }

    pub fn height(&self) -> f64 {
        self.extents()
            .map(|(_, min_y, _, max_y)| (max_y - min_y) as f64)
            .unwrap_or(0.0)
    }
}

/// Per-plugin state: one Rhai engine and scope per .rxx file. AST is stored separately in
/// `LoadedPluginEntry` so we can borrow plugin (engine/scope) and ast independently when calling into scripts.
pub struct LoadedPlugin {
    /// Path to the .rxx script file.
    pub path: PathBuf,
    /// Mtime of the script file after last successful load (for hot-reload).
    pub mtime: Option<SystemTime>,
    pub engine: Engine,
    pub scope: Scope<'static>,
}

/// A loaded plugin plus its AST in separate ref-counted cells so plugin code can run while
/// other plugins are accessible (borrow plugin for engine/scope and ast independently).
pub struct LoadedPluginEntry {
    pub plugin: Rc<RefCell<LoadedPlugin>>,
    pub ast: Rc<RefCell<AST>>,
}

/// State for Rhai plugins (event-handler style). Each plugin has its own engine and ScriptState.
/// Built in lib.rs and passed to renderer.frame() and draw_ctx.draw().
/// Plugins are wrapped in `Rc<RefCell<>>` so that plugin code can access other plugins via the script state handle.
pub struct ScriptState {
    /// Plugin directory (where *.rxx were discovered).
    pub plugin_dir: Option<PathBuf>,
    /// Loaded plugins keyed by script file basename (file stem), e.g. "rotate-scale".
    pub plugins: HashMap<String, LoadedPluginEntry>,
    /// Effects queued by plugins.
    pub effects: Rc<RefCell<Vec<Effect>>>,
}

thread_local! {
    static CURRENT_PLUGIN_KEY: RefCell<Option<String>> = RefCell::new(None);
}

struct CurrentPluginScopeGuard {
    prev_key: Option<String>,
}

impl CurrentPluginScopeGuard {
    fn new(plugin_key: &str) -> Self {
        let prev_key = CURRENT_PLUGIN_KEY.with(|slot| slot.replace(Some(plugin_key.to_string())));
        Self { prev_key }
    }
}

impl Drop for CurrentPluginScopeGuard {
    fn drop(&mut self) {
        CURRENT_PLUGIN_KEY.with(|slot| {
            slot.replace(self.prev_key.take());
        });
    }
}

fn call_with_plugin_context<T, F>(plugin_key: &str, plugin: &mut LoadedPlugin, f: F) -> T
where
    F: FnOnce(&mut LoadedPlugin) -> T,
{
    let _guard = CurrentPluginScopeGuard::new(plugin_key);
    f(plugin)
}

fn invoke_plugin_fn_ref(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    plugin_fn: &PluginFnRef,
    args: Vec<Dynamic>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let caller_plugin = CURRENT_PLUGIN_KEY
        .with(|slot| slot.borrow().clone())
        .unwrap_or_else(|| "<unknown>".to_string());
    let (target_plugin_rc, target_ast_rc) = {
        let state = script_state_handle.borrow();
        let entry = state.plugins.get(&plugin_fn.plugin_key).ok_or_else(|| {
            format!(
                "Plugin '{}' not found (requested from '{}')",
                plugin_fn.plugin_key, caller_plugin
            )
        })?;
        (entry.plugin.clone(), entry.ast.clone())
    };
    let ast = target_ast_rc.borrow();
    let mut target_plugin = target_plugin_rc.try_borrow_mut().map_err(|_| {
        format!(
            "Plugin '{}' is already executing",
            plugin_fn.plugin_key
        )
    })?;
    {
        let plugin = &mut *target_plugin;
        plugin
            .engine
            .call_fn::<()>(&mut plugin.scope, &*ast, &plugin_fn.fn_name, args)?;
    }
    Ok(())
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
    script_state_handle: &Rc<RefCell<ScriptState>>,
) -> Result<(LoadedPlugin, Rc<RefCell<AST>>, Vec<(String, String)>), String> {
    if !path.exists() {
        return Err(format!("Plugin not found: {}", path.display()));
    }
    let (shape_batch, sprite_batch) = renderer_handle.borrow().user_batch();

    let mut engine = Engine::new();
    engine.set_max_expr_depths(10_000, 10_000);
    register_draw_primitives(&mut engine, shape_batch, sprite_batch);
    register_session_handle(
        &mut engine,
        script_state_handle.borrow().effects.clone(),
        script_state_handle.clone(),
    );
    register_renderer_handle(&mut engine);
    register_wgpu_types(&mut engine);
    register_script_queue(&mut engine);
    let plugin_dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    register_system_api(&mut engine, plugin_dir);

    let script_commands: Rc<RefCell<Vec<(String, String)>>> = Rc::new(RefCell::new(Vec::new()));
    register_command_api(&mut engine, script_commands.clone());

    let ast = compile_file(&engine, path).map_err(|e| format!("Script compile error: {}", e))?;
    let mut scope = Scope::new();
    register_constants(&mut scope);
    scope.push("session", Dynamic::from(session_handle.clone()));
    scope.push("renderer", Dynamic::from(renderer_handle.clone()));
    scope.push("script_queue", Dynamic::from(ScriptQueue::new()));
    call_init(&engine, &mut scope, &ast)
        .map_err(|e| format!("Script init error: {}", e))?;

    let cmds = script_commands.borrow().clone();
    let mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
    let plugin = LoadedPlugin {
        path: path.to_path_buf(),
        mtime,
        engine,
        scope,
    };
    let ast_rc = Rc::new(RefCell::new(ast));
    Ok((plugin, ast_rc, cmds))
}

impl ScriptState {
    pub fn new() -> Self {
        Self {
            plugin_dir: None,
            plugins: HashMap::new(),
            effects: Rc::new(RefCell::new(Vec::new())),
        }
    }

    #[allow(dead_code)]
    pub fn set_plugin_dir(&mut self, dir: PathBuf) {
        self.plugin_dir = Some(dir);
    }
}

/// Load or reload all plugins from the plugin directory. Discovers *.rxx, loads each,
/// merges script commands, and replaces ScriptState. Returns an error message for the caller.
/// Calls `unload()` on each existing plugin just before replacing them.
pub fn load_plugins(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    session_handle: &Rc<RefCell<Session>>,
    renderer_handle: &Rc<RefCell<wgpu::Renderer>>,
    plugin_dir: PathBuf,
) -> Result<(), String> {
    if !plugin_dir.is_dir() {
        return Err(format!("Plugin dir is not a directory: {}", plugin_dir.display()));
    }
    // Call unload() on each plugin before replacing them.
    {
        let state = script_state_handle.borrow();
        for entry in state.plugins.values() {
            let ast = entry.ast.borrow();
            let mut plugin = entry.plugin.borrow_mut();
            let _ = call_unload(&mut plugin, &*ast);
        }
    }
    let paths = discover_rxx(&plugin_dir)?;
    let mut all_commands = Vec::new();
    let mut plugins = HashMap::with_capacity(paths.len());
    for path in &paths {
        match load_one_plugin(path, session_handle, renderer_handle, script_state_handle) {
            Ok((plugin, ast_rc, cmds)) => {
                all_commands.extend(cmds);
                plugins.insert(
                    plugin_basename(path),
                    LoadedPluginEntry {
                        plugin: Rc::new(RefCell::new(plugin)),
                        ast: ast_rc,
                    },
                );
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
        for path in &current_paths {
            let key = plugin_basename(path);
            if let Some(entry) = self.plugins.get(&key) {
                let plugin = entry.plugin.borrow();
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
}

/// Traverse effects: call each plugin's event handlers (`view_added`, `view_removed`,
/// `mouse_input`, etc.). Returns effects not consumed for the renderer.
/// Takes `script_state_handle` so plugin code can access other plugins without holding a borrow.
pub fn call_view_effects(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    effects: &[Effect],
) -> Vec<Effect> {
    let mut keys: Vec<_> = script_state_handle.borrow().plugins.keys().cloned().collect();
    keys.sort();
    let mut renderer_effects = Vec::new();
    for eff in effects {
        match eff {
            Effect::ViewAdded(id) => {
                for k in &keys {
                    if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
                        let ast = entry.ast.borrow();
                        let mut plugin = entry.plugin.borrow_mut();
                        let _ = call_with_plugin_context(k, &mut plugin, |plugin| {
                            call_view_added(plugin, &*ast, id.raw() as i64)
                        });
                    }
                }
                renderer_effects.push(eff.clone());
            }
            Effect::ViewRemoved(id) => {
                for k in &keys {
                    if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
                        let ast = entry.ast.borrow();
                        let mut plugin = entry.plugin.borrow_mut();
                        let _ = call_with_plugin_context(k, &mut plugin, |plugin| {
                            call_view_removed(plugin, &*ast, id.raw() as i64)
                        });
                    }
                }
                renderer_effects.push(eff.clone());
            }
            Effect::ScriptEffect(ScriptEffect::MouseInput(state, button, p)) => {
                for k in &keys {
                    if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
                        let ast = entry.ast.borrow();
                        let mut plugin = entry.plugin.borrow_mut();
                        let _ = call_with_plugin_context(k, &mut plugin, |plugin| {
                            call_mouse_input(plugin, &*ast, state, button, p)
                        });
                    }
                }
                renderer_effects.push(eff.clone());
            }
            Effect::ScriptEffect(ScriptEffect::MouseWheel(delta)) => {
                for k in &keys {
                    if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
                        let ast = entry.ast.borrow();
                        let mut plugin = entry.plugin.borrow_mut();
                        let _ = call_with_plugin_context(k, &mut plugin, |plugin| {
                            call_mouse_wheel(plugin, &*ast, delta)
                        });
                    }
                }
                renderer_effects.push(eff.clone());
            }
            Effect::ScriptEffect(ScriptEffect::CursorMoved(p)) => {
                for k in &keys {
                    if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
                        let ast = entry.ast.borrow();
                        let mut plugin = entry.plugin.borrow_mut();
                        if let Err(e) = call_with_plugin_context(k, &mut plugin, |plugin| {
                            call_cursor_moved(plugin, &*ast, p)
                        }) {
                            error!("Script command 'cursor_moved' error: {}", e);
                        }
                    }
                }
                renderer_effects.push(eff.clone());
            }
            Effect::ScriptEffect(ScriptEffect::SwitchMode) => {
                for k in &keys {
                    if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
                        let ast = entry.ast.borrow();
                        let mut plugin = entry.plugin.borrow_mut();
                        let _ = call_with_plugin_context(k, &mut plugin, |plugin| {
                            call_switch_mode(plugin, &*ast)
                        });
                    }
                }
            }
            other => renderer_effects.push(other.clone()),
        }
    }
    {
        let state = script_state_handle.borrow_mut();
        let mut script_state_effects = state.effects.borrow_mut();
        renderer_effects.append(&mut script_state_effects);
        script_state_effects.clear();
    }
    renderer_effects
}

/// Call each plugin's `draw()` event handler. User batch is cleared once before the first plugin.
/// The script's draw primitives (e.g. `draw_line`, `draw_text`) mutate the user batches directly.
pub fn call_draw_event(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    user_batch: &(
        Rc<RefCell<shape2d::Batch>>,
        Rc<RefCell<Option<sprite2d::Batch>>>,
    ),
) -> Result<(), Box<rhai::EvalAltResult>> {
    user_batch.0.borrow_mut().clear();
    if let Some(ref mut b) = *user_batch.1.borrow_mut() {
        b.clear();
    }
    let mut keys: Vec<_> = script_state_handle.borrow().plugins.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
            let ast = entry.ast.borrow();
            let mut plugin = entry.plugin.borrow_mut();
            call_with_plugin_context(k, &mut plugin, |plugin| call_draw(plugin, &*ast))?;
        }
    }
    Ok(())
}

/// Call each plugin's `shade(encoder)` event handler.
pub fn call_shade_event(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    encoder: &Rc<RefCell<wgpu_types::CommandEncoder>>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let mut keys: Vec<_> = script_state_handle.borrow().plugins.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
            let ast = entry.ast.borrow();
            let mut plugin = entry.plugin.borrow_mut();
            call_with_plugin_context(k, &mut plugin, |plugin| call_shade(plugin, &*ast, encoder))?;
        }
    }
    Ok(())
}

/// Call each plugin's `render(pass)` event handler.
pub fn call_render_event(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    script_pass: ScriptRenderPass,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let mut keys: Vec<_> = script_state_handle.borrow().plugins.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        if let Some(entry) = script_state_handle.borrow().plugins.get(k) {
            let ast = entry.ast.borrow();
            let mut plugin = entry.plugin.borrow_mut();
            call_with_plugin_context(k, &mut plugin, |plugin| {
                call_render(plugin, &*ast, script_pass.clone())
            })?;
        }
    }
    Ok(())
}

/// Call a script command handler `cmd_<name>(args)` on each plugin in turn.
///
/// A handler can:
/// - Return `true`  → command is handled, skip built-in.
/// - Return `false` → command is not handled, try next plugin / built-in.
/// - Return nothing  → treated as `false` (not handled).
///
/// Returns `Ok(true)` if any plugin handled it, `Ok(false)` if none did.
pub fn call_script_command(
    script_state_handle: &Rc<RefCell<ScriptState>>,
    name: &str,
    args: Vec<String>,
) -> Result<bool, Box<rhai::EvalAltResult>> {
    let handler_name = format!("cmd_{}", name.replace('/', "_"));
    let rhai_args: Array = args.into_iter().map(Dynamic::from).collect();
    let mut keys: Vec<_> = script_state_handle.borrow().plugins.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let state = script_state_handle.borrow();
        let (plugin_rc, ast_rc) = match state.plugins.get(k) {
            Some(entry) => (entry.plugin.clone(), entry.ast.clone()),
            None => continue,
        };
        drop(state);
        let ast = ast_rc.borrow();
        let mut plugin = plugin_rc.borrow_mut();
        match call_with_plugin_context(k, &mut plugin, |plugin| {
            call_script_command_plugin(plugin, &*ast, &handler_name, &rhai_args)
        }) {
            Ok(Some(true)) => return Ok(true),
            Ok(_) => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(false)
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

/// Register session handle type so scripts can use the global `session`.
/// Exposes width, height, offset_x, offset_y from the session.
pub fn register_session_handle(
    engine: &mut Engine,
    script_effects_queue: Rc<RefCell<Vec<Effect>>>,
    script_state_handle: Rc<RefCell<ScriptState>>,
) {
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
            s.borrow().mode
        })
        .register_get("prev_mode", |s: &mut Rc<RefCell<Session>>| {
            match s.borrow().prev_mode {
                Some(m) => Dynamic::from(m),
                None => Dynamic::UNIT
            }
        })
        .register_fn("switch_mode", |s: &mut Rc<RefCell<Session>>, mode: Mode| {
            s.borrow_mut().switch_mode(mode);
        })
        .register_fn("touch_active_view", |s: &mut Rc<RefCell<Session>>| {
            s.borrow_mut().active_view_mut().touch();
        })
        .register_fn("run_builtin", |s: &mut Rc<RefCell<Session>>, invocation: &str| {
            s.borrow_mut().run_builtin(invocation);
        })
        .register_fn("script_mode", |name: String| {
            Mode::ScriptMode(ModeString::try_from_str(name.as_str())
                .expect("Failed to convert string to ModeString"))
        })
        .register_get("selection", |s: &mut Rc<RefCell<Session>>| {
            match s.borrow().selection {
                Some(s) => { Dynamic::from(s.abs().bounds()) },
                None => Dynamic::UNIT
            }
        })
        .register_get("cursor", |s: &mut Rc<RefCell<Session>>| {
            Vector2::new(s.borrow().cursor.x as f64, s.borrow().cursor.y as f64)
        })
        .register_get("keys_pressed", |s: &mut Rc<RefCell<Session>>| {
            s.borrow()
                .keys_pressed()
                .into_iter()
                .map(Dynamic::from)
                .collect::<Array>()
        })
        .register_fn("key_pressed", |s: &mut Rc<RefCell<Session>>, key: Key| {
            s.borrow().key_pressed(key)
        })
        .register_fn("active_view_coords", |s: &mut Rc<RefCell<Session>>, p: Vector2<f64>| {
            let coords = s.borrow().active_view_coords(SessionCoords::new(p.x as f32, p.y as f32));
            Vector2::new(coords.x as f64, coords.y as f64)
        })
        .register_fn("active_view_coords", |s: &mut Rc<RefCell<Session>>, p: Point<Session, f64>| {
            let coords = s.borrow().active_view_coords(SessionCoords::new(p.x as f32, p.y as f32));
            Vector2::new(coords.x as f64, coords.y as f64)
        })
        .register_fn("active_view_sub_coords", |s: &mut Rc<RefCell<Session>>, p: Vector2<f64>, percision: i64| {
            let coords = s.borrow().active_view_sub_coords(SessionCoords::new(p.x as f32, p.y as f32), percision as u32);
            Vector2::new(coords.x as f64, coords.y as f64)
        })
        .register_fn("active_view_sub_coords", |s: &mut Rc<RefCell<Session>>, p: Point<Session, f64>, percision: i64| {
            let coords = s.borrow().active_view_sub_coords(SessionCoords::new(p.x as f32, p.y as f32), percision as u32);
            Vector2::new(coords.x as f64, coords.y as f64)
        })
        .register_type_with_name::<Mode>("Mode")
        .register_fn("to_string", |mode: Mode| mode.to_string())
        .register_fn("==", |a: Mode, b: Mode| a == b)
        .register_fn("!=", |a: Mode, b: Mode| a != b)
        .register_type_with_name::<ScriptView>("View")
        .register_get("id", |v: &mut ScriptView| v.id.raw() as i64)
        .register_get("offset", |v: &mut ScriptView| {
            Vector2::new(v.offset.x as f64, v.offset.y as f64)
        })
        .register_get("frame_width", |v: &mut ScriptView| v.frame_width)
        .register_get("frame_height", |v: &mut ScriptView| v.frame_height)
        .register_get("zoom", |v: &mut ScriptView| v.zoom as f64)
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
        .register_fn("get_setting", |s: &mut Rc<RefCell<Session>>, name: &str| {
            ScriptSettingValue(s.borrow().settings.get(name).cloned())
        })
        .register_fn("init_setting", |s: &mut Rc<RefCell<Session>>, name: &str, value: &str| {
            s.borrow_mut()
                .settings
                .init(name, Value::Str(value.to_string()))
        })
        .register_fn("get_plugin_fn", |_: &mut Rc<RefCell<Session>>, addr: &str| {
            parse_plugin_fn_ref(addr)
        })
        .register_fn("get_plugin_fn", |_: &mut Rc<RefCell<Session>>, addr: Option<String>| {
            match addr {
                Some(s) => parse_plugin_fn_ref(s.as_str()),
                None => Err("Plugin function address is missing".into()),
            }
        })
        .register_type_with_name::<PluginFnRef>("PluginFnRef")
        .register_fn("invoke", {
            let script_state_handle = script_state_handle.clone();
            move |plugin_fn: &mut PluginFnRef, call_args: Array| {
                invoke_plugin_fn_ref(&script_state_handle, plugin_fn, call_args)
            }
        })
        .register_type_with_name::<ScriptSettingValue>("ScriptSettingValue")
        .register_fn("is_present", ScriptSettingValue::is_present)
        .register_fn("as_bool", ScriptSettingValue::as_bool)
        .register_fn("as_f64", ScriptSettingValue::as_f64)
        .register_fn("as_int", ScriptSettingValue::as_int)
        .register_fn("as_string", ScriptSettingValue::as_string)
        .register_fn("as_rgba8", ScriptSettingValue::as_rgba8)
        .register_fn("as_tuple", ScriptSettingValue::as_tuple)
        .register_type_with_name::<InputState>("InputState")
        .register_fn("to_string", |state: InputState| {
            format!("{:?}", state)
        })
        .register_fn("==", |a: InputState, b: InputState| a == b)
        .register_type_with_name::<Key>("Key")
        .register_fn("to_string", |key: Key| key.to_string())
        .register_fn("==", |a: Key, b: Key| a == b)
        .register_type_with_name::<MouseButton>("MouseButton")
        .register_fn("to_string", |button: MouseButton| {
            format!("{:?}", button)
        })
        .register_fn("==", |a: MouseButton, b: MouseButton| a == b)
        .register_type_with_name::<Point<ViewExtent, f64>>("Point")
        .register_fn("to_string", |p: Point<ViewExtent, f64>| {
            format!("{:?}", p)
        })
        .register_get("x", |p: &mut Point<ViewExtent, f64>| p.point.x as f64)
        .register_get("y", |p: &mut Point<ViewExtent, f64>| p.point.y as f64)
        .register_type_with_name::<LogicalDelta>("LogicalDelta")
        .register_fn("to_string", |delta: LogicalDelta| {
            format!("{:?}", delta)
        })
        .register_fn("queue_effect", {
            let script_effects_queue = script_effects_queue.clone();
            move |effect: Effect| {
                script_effects_queue.borrow_mut().push(effect);
            }
        })
        .register_fn("queue_active_view_rect_clear", {
            let script_effects_queue = script_effects_queue.clone();
            move |rect: Rect<i32>| {
                let mut effects_queue = script_effects_queue.borrow_mut();
                effects_queue.push(Effect::ViewBlendingChanged(Blending::Constant));
                effects_queue.push(Effect::ViewPaintFinal(vec![Shape::Rectangle(
                    rect.into(),
                    ZDepth::default(),
                    Rotation::ZERO,
                    Stroke::NONE,
                    Fill::Solid(Rgba8::TRANSPARENT.into()),
                )]));
            }
        })
        .register_type_with_name::<Effect>("Effect")
        .register_fn("effect_view_paint_final", |shapes: &[Shape]| Effect::ViewPaintFinal(shapes.to_vec()))
        .register_fn("effect_view_damaged", |id: i64| Effect::ViewDamaged(ViewId(id as u16), None));
}

/// Queue for passing messages from command handlers to `shade(encoder)`.
/// Scripts push from `cmd_*(args)` and drain in `shade()`.
#[derive(Clone, Debug)]
pub struct ScriptQueue {
    inner: Rc<RefCell<Vec<Dynamic>>>,
}

impl ScriptQueue {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn push(&mut self, msg: Dynamic) {
        self.inner.borrow_mut().push(msg);
    }

    /// Remove and return the next message, or `()` if empty.
    pub fn pop(&mut self) -> Dynamic {
        let mut v = self.inner.borrow_mut();
        if v.is_empty() {
            Dynamic::UNIT
        } else {
            v.remove(0)
        }
    }

    pub fn len(&mut self) -> i64 {
        self.inner.borrow().len() as i64
    }

    pub fn is_empty(&mut self) -> bool {
        self.inner.borrow().is_empty()
    }

    /// Remove and return all messages as an array, clearing the queue.
    pub fn take_all(&mut self) -> Array {
        let mut v = self.inner.borrow_mut();
        v.drain(..).collect()
    }
}

impl Default for ScriptQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Register the shade queue type so scripts can use `script_queue.push(msg)` and drain in `shade()`.
pub fn register_script_queue(engine: &mut Engine) {
    engine
        .register_type_with_name::<ScriptQueue>("ShadeQueue")
        .register_fn("push", ScriptQueue::push)
        .register_fn("pop", ScriptQueue::pop)
        .register_fn("len", ScriptQueue::len)
        .register_fn("is_empty", ScriptQueue::is_empty)
        .register_fn("take_all", ScriptQueue::take_all);
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
    /// Texture view in native format (Rgba8Unorm). Use for compute passes; Texture uses sRGB view.
    TextureRaw {
        binding: u32,
        texture: Rc<RefCell<Texture>>,
    },
    Sampler {
        binding: u32,
        sampler: Rc<RefCell<wgpu_types::Sampler>>,
    },
    SamplerDefault { binding: u32 },
}

const SAMPLER_ADDRESS_CLAMP_TO_EDGE: i64 = 0;
const SAMPLER_ADDRESS_REPEAT: i64 = 1;
const SAMPLER_ADDRESS_MIRROR_REPEAT: i64 = 2;

const SAMPLER_FILTER_NEAREST: i64 = 0;
const SAMPLER_FILTER_LINEAR: i64 = 1;

fn script_sampler_address_mode(mode: i64) -> wgpu_types::AddressMode {
    match mode {
        SAMPLER_ADDRESS_REPEAT => wgpu_types::AddressMode::Repeat,
        SAMPLER_ADDRESS_MIRROR_REPEAT => wgpu_types::AddressMode::MirrorRepeat,
        _ => wgpu_types::AddressMode::ClampToEdge,
    }
}

fn script_sampler_filter_mode(mode: i64) -> wgpu_types::FilterMode {
    match mode {
        SAMPLER_FILTER_LINEAR => wgpu_types::FilterMode::Linear,
        _ => wgpu_types::FilterMode::Nearest,
    }
}

/// Render pass type for script: wraps the wgpu render pass so pass methods can resolve handles.
#[derive(Clone)]
pub struct ScriptRenderPass {
    pass: Rc<RefCell<wgpu_types::RenderPass<'static>>>,
}

impl ScriptRenderPass {
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

/// Compute pass type for script: wraps the wgpu compute pass so pass methods can resolve handles.
#[derive(Clone)]
pub struct ScriptComputePass {
    pass: Rc<RefCell<wgpu_types::ComputePass<'static>>>,
}

impl ScriptComputePass {
    pub fn new(pass: Rc<RefCell<wgpu_types::ComputePass<'static>>>) -> Self {
        Self { pass }
    }

    pub fn set_pipeline(
        &mut self,
        handle: Rc<RefCell<wgpu_types::ComputePipeline>>,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        let pipeline = handle.borrow();
        self.pass.borrow_mut().set_pipeline(&pipeline);
        Ok(())
    }

    pub fn set_bind_group(
        &mut self,
        index: i64,
        handle: Rc<RefCell<wgpu_types::BindGroup>>,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        let bind_group = handle.borrow();
        self.pass
            .borrow_mut()
            .set_bind_group(index as u32, &*bind_group, &[]);
        Ok(())
    }

    pub fn dispatch_workgroups(
        &mut self,
        x: i64,
        y: i64,
        z: i64,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        self.pass
            .borrow_mut()
            .dispatch_workgroups(x as u32, y as u32, z as u32);
        Ok(())
    }
}

/// Register renderer handle type so scripts can use the global `renderer`.
/// Exposes create_render_texture, view_render_texture, begin_render_pass.
/// Script-created textures are stored in script_state_handle.
pub fn register_renderer_handle(
    engine: &mut Engine,
) {
    engine
        .register_type_with_name::<wgpu_types::TextureFormat>("TextureFormat")
        .register_type_with_name::<Rc<RefCell<Texture>>>("TextureHandle")
        .register_type_with_name::<Rc<RefCell<wgpu_types::Sampler>>>("SamplerHandle")
        .register_type_with_name::<ScriptLoadOp>("LoadOp")
        .register_type_with_name::<ScriptStoreOp>("StoreOp")
        .register_type_with_name::<ScriptColorAttachment>("ColorAttachment")
        .register_type_with_name::<ScriptRenderPass>("ScriptRenderPass")
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
        .register_fn(
            "bind_texture_raw",
            |binding: i64, texture: Rc<RefCell<Texture>>| ScriptBindGroupEntry::TextureRaw {
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
            "bind_sampler",
            |binding: i64, sampler: Rc<RefCell<wgpu_types::Sampler>>| {
                ScriptBindGroupEntry::Sampler {
                    binding: binding as u32,
                    sampler,
                }
            },
        )
        .register_fn(
            "set_pipeline",
            |pass: &mut ScriptRenderPass, handle: Rc<RefCell<wgpu_types::RenderPipeline>>| -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_pipeline(handle.clone())
            },
        )
        .register_fn(
            "set_bind_group",
            |pass: &mut ScriptRenderPass,
             index: i64,
             handle: Rc<RefCell<wgpu_types::BindGroup>>|
             -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_bind_group(index, handle.clone())
            },
        )
        .register_fn(
            "set_vertex_buffer",
            |pass: &mut ScriptRenderPass,
             slot: i64,
             handle: Rc<RefCell<wgpu_types::Buffer>>|
             -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_vertex_buffer(slot, handle.clone())
            },
        )
        .register_fn(
            "draw",
            |pass: &mut ScriptRenderPass,
             vertex_count: i64,
             instance_count: i64,
             first_vertex: i64,
             first_instance: i64| {
                pass.draw(vertex_count, instance_count, first_vertex, first_instance);
            },
        )
        .register_type_with_name::<ScriptComputePass>("ScriptComputePass")
        .register_fn(
            "set_pipeline",
            |pass: &mut ScriptComputePass,
             handle: Rc<RefCell<wgpu_types::ComputePipeline>>|
             -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_pipeline(handle.clone())
            },
        )
        .register_fn(
            "set_bind_group",
            |pass: &mut ScriptComputePass,
             index: i64,
             handle: Rc<RefCell<wgpu_types::BindGroup>>|
             -> Result<(), Box<rhai::EvalAltResult>> {
                pass.set_bind_group(index, handle.clone())
            },
        )
        .register_fn(
            "dispatch_workgroups",
            |pass: &mut ScriptComputePass, x: i64, y: i64, z: i64| -> Result<(), Box<rhai::EvalAltResult>> {
                pass.dispatch_workgroups(x, y, z)
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
        .register_fn("ensure_render_texture_size", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  target: Rc<RefCell<Texture>>,
                  width: i64,
                  height: i64| {
                r.borrow_mut().ensure_render_texture_size(
                    &mut target.borrow_mut(),
                    width as u32,
                    height as u32,
                );
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
        .register_fn("view_staging_texture", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, view_id: i64| -> Dynamic {
                let id = ViewId(view_id as u16);
                match r.borrow().view_staging_texture(id) {
                    Some(h) => Dynamic::from(h),
                    None => Dynamic::UNIT,
                }
            }
        })
        .register_fn("upload_selection_to_texture", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  session: Rc<RefCell<Session>>,
                  target: Rc<RefCell<Texture>>|
                  -> bool {
                r.borrow_mut()
                    .upload_selection_to_texture(&session.borrow(), &mut target.borrow_mut())
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
        .register_fn("create_texture_sampler_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  texture_handle: Rc<RefCell<Texture>>,
                  sampler_handle: Rc<RefCell<wgpu_types::Sampler>>| {
                let texture = texture_handle.borrow();
                let sampler = sampler_handle.borrow();
                let bind_group = r.borrow_mut()
                    .create_texture_sampler_bind_group_with_sampler(&texture, &sampler);
                Rc::new(RefCell::new(bind_group))
            }
        })
        .register_fn("create_sampler", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, address_mode: i64, filter_mode: i64| {
                let address = script_sampler_address_mode(address_mode);
                let filter = script_sampler_filter_mode(filter_mode);
                let sampler = r.borrow()
                    .create_sampler(address, address, address, filter, filter, filter);
                Rc::new(RefCell::new(sampler))
            }
        })
        .register_fn("create_sampler", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  address_mode_u: i64,
                  address_mode_v: i64,
                  address_mode_w: i64,
                  mag_filter: i64,
                  min_filter: i64,
                  mipmap_filter: i64| {
                let sampler = r.borrow().create_sampler(
                    script_sampler_address_mode(address_mode_u),
                    script_sampler_address_mode(address_mode_v),
                    script_sampler_address_mode(address_mode_w),
                    script_sampler_filter_mode(mag_filter),
                    script_sampler_filter_mode(min_filter),
                    script_sampler_filter_mode(mipmap_filter),
                );
                Rc::new(RefCell::new(sampler))
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
        .register_type_with_name::<Rc<RefCell<wgpu_types::ComputePipeline>>>("ComputePipelineHandle")
        .register_fn("create_compute_pipeline", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  module_handle: Rc<RefCell<wgpu_types::ShaderModule>>| {
                let pipeline = r.borrow_mut().create_compute_pipeline(
                    &module_handle.borrow(),
                    None,
                );
                Rc::new(RefCell::new(pipeline))
            }
        })
        .register_fn("create_compute_pipeline", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  module_handle: Rc<RefCell<wgpu_types::ShaderModule>>,
                  entry_point: ImmutableString| {
                let ep = if entry_point.is_empty() {
                    None
                } else {
                    Some(entry_point.as_str())
                };
                let pipeline = r.borrow_mut().create_compute_pipeline(
                    &module_handle.borrow(),
                    ep,
                );
                Rc::new(RefCell::new(pipeline))
            }
        })
        .register_fn("create_compute_pipeline_with_layouts", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  module_handle: Rc<RefCell<wgpu_types::ShaderModule>>,
                  entry_point: ImmutableString,
                  bind_group_layouts: Array| {
                use std::ops::Deref;
                let ep = if entry_point.is_empty() {
                    None
                } else {
                    Some(entry_point.as_str())
                };
                let handles: Vec<Rc<RefCell<wgpu_types::BindGroupLayout>>> = bind_group_layouts
                    .iter()
                    .filter_map(|d| d.clone().try_cast::<Rc<RefCell<wgpu_types::BindGroupLayout>>>())
                    .collect();
                let ref_guards: Vec<std::cell::Ref<wgpu_types::BindGroupLayout>> =
                    handles.iter().map(|h| h.borrow()).collect();
                let layout_refs: Vec<&wgpu_types::BindGroupLayout> =
                    ref_guards.iter().map(Deref::deref).collect();
                let pipeline = r.borrow_mut().create_compute_pipeline_with_layouts(
                    &module_handle.borrow(),
                    ep,
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
        .register_fn("create_ortho_transform_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>, width: f64, height: f64| {
                let bind_group = r.borrow_mut()
                    .create_ortho_transform_bind_group(width as u32, height as u32);
                Rc::new(RefCell::new(bind_group))
            }
        })
        .register_fn("create_ortho_custom_transform_bind_group", {
            move |r: &mut Rc<RefCell<wgpu::Renderer>>,
                  width: f64,
                  height: f64,
                  transform: Matrix4<f32>| {
                let bind_group = r.borrow_mut()
                    .create_ortho_custom_transform_bind_group(
                        width as u32,
                        height as u32,
                        transform,
                    );
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
             src: Rect<f64>,
             dst: Rect<f64>,
             zdepth: f64,
             color: Rgba8,
             alpha: f64| {
                let batch = sprite2d::Batch::singleton(
                    w as u32,
                    h as u32,
                    src.into(),
                    dst.into(),
                    crate::gfx::ZDepth(zdepth as f32),
                    Rgba::from(color),
                    alpha as f32,
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
        .register_fn("width", |list: &mut ScriptSpriteVertexList| {
            list.width()
        })
        .register_fn("height", |list: &mut ScriptSpriteVertexList| {
            list.height()
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
                    let script_compute_pass = ScriptComputePass::new(pass_handle);
                    Ok(Dynamic::from(script_compute_pass))
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
                    let script_pass = ScriptRenderPass::new(pass_handle);
                    Ok(Dynamic::from(script_pass))
                }
            },
        );
}

/// Register Vector2<f64> and Rgba8 for script use. vec2(x,y), rgb8(r,g,b), rgb8(r,g,b,a).
fn register_draw_types(engine: &mut Engine) {
    engine
        .register_type_with_name::<Point2<f64>>("Point2_f64")
        .register_get("x", |p: &mut Point2<f64>| p.x as f64)
        .register_get("y", |p: &mut Point2<f64>| p.y as f64)
        .register_fn("to_string", |p: Point2<f64>| format!("{:?}", p));
    
    engine
        .register_type_with_name::<Point2<i32>>("Point2_i32")
        .register_get("x", |p: &mut Point2<i32>| p.x as f64)
        .register_get("y", |p: &mut Point2<i32>| p.y as f64)
        .register_fn("to_string", |p: Point2<i32>| format!("{:?}", p));

    engine
        .register_type_with_name::<Vector2<f64>>("Vector2")
        .register_get("x", |v: &mut Vector2<f64>| v.x as f64)
        .register_get("y", |v: &mut Vector2<f64>| v.y as f64)
        .register_fn("vec2", |x: f64, y: f64| Vector2::new(x as f64, y as f64))
        .register_fn("+", |v1: Vector2<f64>, v2: Vector2<f64>| v1 + v2)
        .register_fn("-", |v1: Vector2<f64>, v2: Vector2<f64>| v1 - v2)
        .register_fn("*", |v1: Vector2<f64>, v2: f64| v1 * v2)
        .register_fn("==", |v1: Vector2<f64>, v2: Vector2<f64>| v1 == v2)
        .register_fn("to_string", |v: Vector2<f64>| format!("{:?}", v));

    engine
        .register_type_with_name::<Matrix4<f32>>("Mat4")
        .register_fn("mat4_identity", || Matrix4::<f32>::identity())
        .register_fn("mat4_translation", |x: f64, y: f64| {
            Matrix4::<f32>::from_translation(Vector3::new(x as f32, y as f32, 0.0))
        })
        .register_fn("mat4_rotation_z", |angle: f64| {
            Matrix4::<f32>::from_rotation_z(angle as f32)
        })
        .register_fn("mat4_scale", |sx: f64, sy: f64| {
            Matrix4::<f32>::from_nonuniform_scale(sx as f32, sy as f32, 1.0)
        })
        .register_fn("*", |a: Matrix4<f32>, b: Matrix4<f32>| a * b)
        .register_fn("to_string", |m: Matrix4<f32>| format!("{:?}", m));

    engine
        .register_type_with_name::<Point<Session, f64>>("SessionPoint")
        .register_get("x", |p: &mut Point<Session, f64>| p.x as f64)
        .register_get("y", |p: &mut Point<Session, f64>| p.y as f64)
        .register_fn("to_string", |p: Point<Session, f64>| { let p: SessionCoords = p.into(); format!("{:?}", p) })
        .register_fn("to_string", |p: Point<Session, f32>| format!("{:?}", p))
        .register_fn("to_vec2", |p: &mut Point<Session, f64>| { let p: Vector2<f64> = (*p).into(); p})
        .register_fn("==", |p1: Point<Session, f64>, p2: Point<Session, f64>| p1 == p2)
        .register_fn("==", |p1: Point<Session, f64>, p2: Vector2<f64>| { let p1: Vector2<f64> = p1.into(); p1 == p2});

    engine
        .register_type_with_name::<Point<ViewExtent, f64>>("ViewPoint")
        .register_get("x", |p: &mut Point<ViewExtent, f64>| p.x as f64)
        .register_get("y", |p: &mut Point<ViewExtent, f64>| p.y as f64)
        .register_fn("to_string", |p: Point<ViewExtent, f64>| format!("{:?}", p))
        .register_fn("to_string", |p: Point<ViewExtent, f32>| format!("{:?}", p));

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
        .register_type_with_name::<Rect<f64>>("Rect_f64")
        .register_fn("to_string", |r: Rect<f64>| format!("{:?}", r))
        .register_get("x1", |r: &mut Rect<f64>| r.x1 as f64)
        .register_get("y1", |r: &mut Rect<f64>| r.y1 as f64)
        .register_get("x2", |r: &mut Rect<f64>| r.x2 as f64)
        .register_get("y2", |r: &mut Rect<f64>| r.y2 as f64)
        .register_get("width", |r: &mut Rect<f64>| r.width())
        .register_get("height", |r: &mut Rect<f64>| r.height())
        .register_type_with_name::<Rect<i32>>("Rect_i32")
        .register_fn("to_string", |r: Rect<i32>| format!("{:?}", r))
        .register_get("x1", |r: &mut Rect<i32>| r.x1 as f64)
        .register_get("y1", |r: &mut Rect<i32>| r.y1 as f64)
        .register_get("x2", |r: &mut Rect<i32>| r.x2 as f64)
        .register_get("y2", |r: &mut Rect<i32>| r.y2 as f64)
        .register_fn("center", |r: &mut Rect<i32>| { let p = r.center(); Vector2::new(p.x as f64, p.y as f64)})
        .register_fn("to_rect_f64", |r: &mut Rect<i32>| {let r: Rect<f64> = r.clone().into(); r})
        .register_fn("rect", |x1: f64, y1: f64, x2: f64, y2: f64| {
            Rect::new(x1 as f64, y1 as f64, x2 as f64, y2 as f64)
        })
        .register_fn("+", |r: Rect<f64>, v: Vector2<f64>| r + v)
        .register_fn("*", |r: Rect<f64>, v: f64| r * v);

    engine
        .register_type_with_name::<ZDepth>("ZDepth")
        .register_fn("zdepth", |z: f64| ZDepth(z as f32));

    engine
        .register_type_with_name::<Repeat>("Repeat")
        .register_get("x", |r: &mut Repeat| r.x as f64)
        .register_get("y", |r: &mut Repeat| r.y as f64)
        .register_fn("repeat", |x: f64, y: f64| Repeat::new(x as f32, y as f32));

    engine
        .register_type_with_name::<Shape>("Shape")
        .register_fn("shape_rectangle", |rect: Rect<f64>, depth: ZDepth, rotation: Rotation, stroke: Stroke, fill: Fill| Shape::Rectangle(rect.into(), depth, rotation, stroke, fill));
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
    engine.register_fn("draw_line", move |p1: Vector2<f64>, p2: Vector2<f64>| {
        let shape = Shape::Line(
            Line::new(Point2::new(p1.x as f32, p1.y as f32), Point2::new(p2.x as f32, p2.y as f32)),
            USER_LAYER,
            Rotation::ZERO,
            Stroke::new(1.0, Rgba::WHITE),
        );
        shape_batch_line.borrow_mut().add(shape);
    });
    engine.register_fn(
        "draw_line",
        move |p1: Vector2<f64>, p2: Vector2<f64>, color: Rgba8| {
            let color: Rgba = color.into();
            let shape = Shape::Line(
                Line::new(Point2::new(p1.x as f32, p1.y as f32), Point2::new(p2.x as f32, p2.y as f32)),
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
    engine.register_fn("draw_text", move |pos: Vector2<f64>, text: &str| {
        if let Some(ref mut batch) = *sprite_batch_text.borrow_mut() {
            let mut sx = pos.x as f32;
            let sy = pos.y as f32;
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
        move |pos: Vector2<f64>, text: &str, color: Rgba8| {
            let color: Rgba = color.into();
            if let Some(ref mut batch) = *sprite_batch.borrow_mut() {
                let mut sx = pos.x as f32;
                let sy = pos.y as f32;
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

/// Register the `register_command(name, help)` function for scripts.
///
/// This is used **only** for help-view content and command-line auto-completion.
/// It does *not* affect command dispatch: scripts handle commands by defining
/// `cmd_<name>(args)` functions, which are tried before built-in implementations
/// regardless of whether the command was registered here.
fn register_command_api(engine: &mut Engine, commands: Rc<RefCell<Vec<(String, String)>>>) {
    engine.register_fn("register_command", move |name: &str, help: &str| {
        commands
            .borrow_mut()
            .push((name.to_string(), help.to_string()));
    });
}

fn register_system_api(engine: &mut Engine, plugin_dir: PathBuf) {
    engine.register_fn("read_file", move |path: &str| {
        let p = Path::new(path);
        let path = if p.is_relative() {
            plugin_dir.join(p)
        } else {
            p.to_path_buf()
        };
        let contents = std::fs::read_to_string(&path).unwrap();
        contents
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
    scope.push_constant("SAMPLER_ADDRESS_CLAMP_TO_EDGE", SAMPLER_ADDRESS_CLAMP_TO_EDGE);
    scope.push_constant("SAMPLER_ADDRESS_REPEAT", SAMPLER_ADDRESS_REPEAT);
    scope.push_constant("SAMPLER_ADDRESS_MIRROR_REPEAT", SAMPLER_ADDRESS_MIRROR_REPEAT);
    scope.push_constant("SAMPLER_FILTER_NEAREST", SAMPLER_FILTER_NEAREST);
    scope.push_constant("SAMPLER_FILTER_LINEAR", SAMPLER_FILTER_LINEAR);
    scope.push_constant("MODE_NORMAL", Mode::Normal);
    scope.push_constant("MODE_VISUAL", Mode::Visual(VisualState::default()));
    scope.push_constant("MODE_COMMAND", Mode::Command);
    scope.push_constant("MODE_PRESENT", Mode::Present);
    scope.push_constant("MODE_HELP", Mode::Help);
    scope.push_constant("MOUSE_BUTTON_LEFT", MouseButton::Left);
    scope.push_constant("MOUSE_BUTTON_RIGHT", MouseButton::Right);
    scope.push_constant("MOUSE_BUTTON_MIDDLE", MouseButton::Middle);
    scope.push_constant("INPUT_STATE_PRESSED", InputState::Pressed);
    scope.push_constant("INPUT_STATE_RELEASED", InputState::Released);
    scope.push_constant("INPUT_STATE_REPEATED", InputState::Repeated);
    scope.push_constant("EFFECT_VIEW_BLENDING_CHANGED_CONSTANT", Effect::ViewBlendingChanged(Blending::Constant));
    scope.push_constant("EFFECT_VIEW_BLENDING_CHANGED_ALPHA", Effect::ViewBlendingChanged(Blending::Alpha));
    scope.push_constant("KEY_CTRL", Key::Control);
    scope.push_constant("KEY_SHIFT", Key::Shift);
    scope.push_constant("KEY_ALT", Key::Alt);
}

/// Call the script's `unload()` function. Invoked on each plugin just before
/// unloading/reloading so scripts can clean up (e.g. release resources).
///
/// If the script does not define `unload`, this is a no-op (no error).
pub fn call_unload(
    plugin: &mut LoadedPlugin,
    ast: &AST,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "unload", ()) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "unload") => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `init()` function with options so that new variables
/// introduced in the scope are retained (rewind_scope false) and the AST
/// is not re-evaluated (eval_ast false).
/// `session` and `renderer` are already in scope (registered before this call).
///
/// If the script does not define `init`, this is a no-op (no error).
pub fn call_init(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let options = CallFnOptions::new().eval_ast(false).rewind_scope(false);

    match engine.call_fn_with_options::<()>(options, scope, ast, "init", ()) {
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
    plugin: &mut LoadedPlugin,
    ast: &AST,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "draw", ()) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "draw") => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `shade(encoder)` event handler.
pub fn call_shade(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    encoder: &Rc<RefCell<wgpu_types::CommandEncoder>>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "shade", (encoder.clone(),)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "shade") => Ok(()),
        Err(e) => Err(e),
    }
}

pub fn call_render(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    script_pass: ScriptRenderPass,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "render", (script_pass,)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "render") => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `view_added(view_id)` handler.
fn call_view_added(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    view_id: i64,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "view_added", (view_id,)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "view_added") => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `view_removed(view_id)` handler.
fn call_view_removed(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    view_id: i64,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "view_removed", (view_id,)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "view_removed") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_mouse_input(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    state: &InputState,
    button: &MouseButton,
    p: &Point<Session, f32>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let p: Point<Session, f64> = p.clone().into();
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "mouse_input", (state.clone(), button.clone(), p)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "mouse_input") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_mouse_wheel(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    delta: &LogicalDelta,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "mouse_wheel", (delta.clone(),)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "mouse_wheel") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_cursor_moved(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    p: &Point<Session, f32>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let p: Point<Session, f64> = p.clone().into();
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "cursor_moved", (p,)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "cursor_moved") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_switch_mode(
    plugin: &mut LoadedPlugin,
    ast: &AST,
) -> Result<(), Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<()>(&mut plugin.scope, ast, "switch_mode", ()) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e, "switch_mode") => Ok(()),
        Err(e) => Err(e),
    }
}

fn call_script_command_plugin(
    plugin: &mut LoadedPlugin,
    ast: &AST,
    handler_name: &str,
    rhai_args: &Array,
) -> Result<Option<bool>, Box<rhai::EvalAltResult>> {
    match plugin.engine.call_fn::<Dynamic>(
        &mut plugin.scope,
        ast,
        handler_name,
        (rhai_args.clone(),),
    ) {
        Ok(val) => Ok(Some(val.as_bool().unwrap_or(false))),
        Err(ref e) if is_function_not_found(e, handler_name) => Ok(None),
        Err(e) => Err(e),
    }
}

fn is_function_not_found(e: &Box<rhai::EvalAltResult>, function_name: &str) -> bool {
    use rhai::EvalAltResult;
    matches!(&**e, EvalAltResult::ErrorFunctionNotFound(name, _) if name == function_name)
}

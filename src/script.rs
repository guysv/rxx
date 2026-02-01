//! Rhai script loading and event-handler lifecycle.
//!
//! Follows the [event-handler pattern](https://rhai.rs/book/patterns/events-1.html):
//! one main script with `init(session)`; custom Scope + CallFnOptions (eval_ast false,
//! rewind_scope false) so variables defined in `init()` persist.

use crate::draw::{self, USER_LAYER};
use crate::gfx::color::Rgba;
use crate::gfx::math::{Point2, Vector2};
use crate::gfx::rect::Rect;
use crate::gfx::shape2d::{self, Line, Rotation, Shape, Stroke};
use crate::gfx::sprite2d;
use crate::gfx::{Repeat, Rgba8};
use crate::session::{Effect, MessageType, Session};
use crate::view::{View, ViewId, ViewResource};

use rhai::{Array, CallFnOptions, Dynamic, Engine, Scope, AST};

use std::cell::RefCell;
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
}

impl From<&View<ViewResource>> for ScriptView {
    fn from(view: &View<ViewResource>) -> Self {
        ScriptView { id: view.id, offset: view.offset }
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
        }
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

    /// Load or reload the main Rhai script: compile, create scope, call init(session).
    /// Returns an error message for the caller to display (e.g. via session.message).
    pub fn load_script(&mut self, session_handle: &Rc<RefCell<Session>>) -> Result<(), String> {
        let path = match &self.script_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        if !path.exists() {
            return Err(format!("Script not found: {}", path.display()));
        }
        let mut engine = Engine::new();
        register_draw_primitives(
            &mut engine,
            self.script_user_batch.0.clone(),
            self.script_user_batch.1.clone(),
        );
        register_session_handle(&mut engine);

        // Collector for script commands registered during init()
        let script_commands: Rc<RefCell<Vec<(String, String)>>> =
            Rc::new(RefCell::new(Vec::new()));
        register_command_api(&mut engine, script_commands.clone());

        let ast = compile_file(&engine, &path)
            .map_err(|e| format!("Script compile error: {}", e))?;
        let mut scope = Scope::new();
        call_init(&engine, &mut scope, &ast, session_handle)
            .map_err(|e| format!("Script init error: {}", e))?;

        // Apply collected script commands to session's command line
        let cmds = script_commands.borrow().clone();
        session_handle.borrow_mut().set_script_commands(cmds);

        self.script_engine = Some(engine);
        self.script_scope = Some(scope);
        self.script_ast = Some(Rc::new(RefCell::new(ast)));
        self.script_mtime = std::fs::metadata(&path).ok().and_then(|m| m.modified().ok());
        Ok(())
    }

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

    /// Reload the main Rhai script: recompile and call init(session) again.
    /// Errors are ignored (e.g. for hot-reload); use load_script() for explicit feedback.
    pub fn reload_script(&mut self, session_handle: &Rc<RefCell<Session>>) {
        let _ = self.load_script(session_handle);
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

/// Register session handle type so scripts can use it in init(session).
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

/// Register Vector2<f32> and Rgba8 for script use. vec2(x,y), rgb8(r,g,b), rgb8(r,g,b,a).
fn register_draw_types(engine: &mut Engine) {
    engine
        .register_type_with_name::<Vector2<f32>>("Vector2")
        .register_get("x", |v: &mut Vector2<f32>| v.x as f64)
        .register_get("y", |v: &mut Vector2<f32>| v.y as f64)
        .register_fn("vec2", |x: f64, y: f64| Vector2::new(x as f32, y as f32));

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

/// Register the `register_command(name, help)` function for scripts to register custom commands.
/// Commands are collected in the provided `Rc<RefCell<Vec<(String, String)>>>`.
fn register_command_api(engine: &mut Engine, commands: Rc<RefCell<Vec<(String, String)>>>) {
    engine.register_fn("register_command", move |name: &str, help: &str| {
        commands.borrow_mut().push((name.to_string(), help.to_string()));
    });
}

/// Call the script's `init(session)` function with options so that new variables
/// introduced in the scope are retained (rewind_scope false) and the AST
/// is not re-evaluated (eval_ast false).
///
/// If the script does not define `init`, this is a no-op (no error).
pub fn call_init(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    session_handle: &Rc<RefCell<Session>>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let options = CallFnOptions::new()
        .eval_ast(false)
        .rewind_scope(false);

    let session = session_handle.clone();
    match engine.call_fn_with_options::<()>(options, scope, ast, "init", (session,)) {
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

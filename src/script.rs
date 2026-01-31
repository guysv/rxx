//! Rhai script loading and event-handler lifecycle.
//!
//! Follows the [event-handler pattern](https://rhai.rs/book/patterns/events-1.html):
//! one main script with `init()`; custom Scope + CallFnOptions (eval_ast false,
//! rewind_scope false) so variables defined in `init()` persist.

use crate::draw::USER_LAYER;
use crate::gfx::color::Rgba;
use crate::gfx::math::Point2;
use crate::gfx::shape2d::{self, Line, Rotation, Shape, Stroke};
use crate::session::Session;

use rhai::{CallFnOptions, Engine, Scope, AST};

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::SystemTime;

/// Type alias for the user batch shared between script and session.
pub type UserBatch = Rc<RefCell<shape2d::Batch>>;

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
    /// User batch for script's draw() event, shared with Rhai closures.
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
            script_user_batch: Rc::new(RefCell::new(shape2d::Batch::new())),
        }
    }

    pub fn set_path(&mut self, path: PathBuf) {
        self.script_path = Some(path);
    }

    /// Load or reload the main Rhai script: compile, create scope, call init().
    /// Returns an error message for the caller to display (e.g. via session.message).
    pub fn load_script(&mut self) -> Result<(), String> {
        let path = match &self.script_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        if !path.exists() {
            return Err(format!("Script not found: {}", path.display()));
        }
        let mut engine = Engine::new();
        register_draw_primitives(&mut engine, self.script_user_batch.clone());
        register_session_handle(&mut engine);
        let ast = compile_file(&engine, &path)
            .map_err(|e| format!("Script compile error: {}", e))?;
        let mut scope = Scope::new();
        call_init(&engine, &mut scope, &ast)
            .map_err(|e| format!("Script init error: {}", e))?;
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

    /// Reload the main Rhai script: recompile and call init() again.
    /// Errors are ignored (e.g. for hot-reload); use load_script() for explicit feedback.
    pub fn reload_script(&mut self) {
        let _ = self.load_script();
    }

    /// Call the script's `draw()` event handler.
    /// The script's draw primitives (e.g. `draw_line`) mutate the user batch directly.
    pub fn call_draw_event(&mut self, session_handle: &Rc<RefCell<Session>>) -> Result<(), Box<rhai::EvalAltResult>> {
        self.script_user_batch.borrow_mut().clear();
        let (engine, scope, ast) = match (
            self.script_engine.as_ref(),
            self.script_scope.as_mut(),
            self.script_ast.as_ref(),
        ) {
            (Some(e), Some(s), Some(a)) => (e, s, a),
            _ => return Ok(()),
        };
        call_draw(engine, scope, &ast.borrow(), session_handle)
    }

    /// Get the user batch vertices for rendering.
    pub fn user_batch_vertices(&self) -> Vec<crate::gfx::shape2d::Vertex> {
        self.script_user_batch.borrow().vertices()
    }

    /// Check if the user batch is empty.
    pub fn user_batch_is_empty(&self) -> bool {
        self.script_user_batch.borrow().is_empty()
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

/// Register session handle type so scripts can use it in draw(session).
/// Exposes width, height, offset_x, offset_y from the session.
pub fn register_session_handle(engine: &mut Engine) {
    engine
        .register_type_with_name::<Rc<RefCell<Session>>>("Session")
        .register_get("width", |s: &mut Rc<RefCell<Session>>| s.borrow().width as f64)
        .register_get("height", |s: &mut Rc<RefCell<Session>>| s.borrow().height as f64)
        .register_get("offset_x", |s: &mut Rc<RefCell<Session>>| s.borrow().offset.x as f64)
        .register_get("offset_y", |s: &mut Rc<RefCell<Session>>| s.borrow().offset.y as f64);
}

/// Register draw primitives on the engine. Call this once when loading a script.
/// The batch is shared; `draw_line` adds shapes directly to it.
pub fn register_draw_primitives(engine: &mut Engine, user_batch: UserBatch) {
    engine.register_fn("draw_line", move |x1: f64, y1: f64, x2: f64, y2: f64| {
        let shape = Shape::Line(
            Line::new(
                Point2::new(x1 as f32, y1 as f32),
                Point2::new(x2 as f32, y2 as f32),
            ),
            USER_LAYER,
            Rotation::ZERO,
            Stroke::new(1.0, Rgba::WHITE),
        );
        user_batch.borrow_mut().add(shape);
    });
}

/// Call the script's `init()` function with options so that new variables
/// introduced in the scope are retained (rewind_scope false) and the AST
/// is not re-evaluated (eval_ast false).
///
/// If the script does not define `init`, this is a no-op (no error).
pub fn call_init(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let options = CallFnOptions::new()
        .eval_ast(false)
        .rewind_scope(false);

    match engine.call_fn_with_options::<()>(options, scope, ast, "init", ()) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Call the script's `draw()` event handler.
/// Unlike `init`, this uses the default call_fn (scope is rewound after the call).
///
/// If the script does not define `draw`, this is a no-op (no error).
pub fn call_draw(
    engine: &Engine,
    scope: &mut Scope,
    ast: &AST,
    session_handle: &Rc<RefCell<Session>>,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let session = session_handle.clone();
    match engine.call_fn::<()>(scope, ast, "draw", (session,)) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

fn is_function_not_found(e: &Box<rhai::EvalAltResult>) -> bool {
    use rhai::EvalAltResult;
    matches!(&**e, EvalAltResult::ErrorFunctionNotFound(_, _))
}

//! Rhai script loading and event-handler lifecycle.
//!
//! Follows the [event-handler pattern](https://rhai.rs/book/patterns/events-1.html):
//! one main script with `init()`; custom Scope + CallFnOptions (eval_ast false,
//! rewind_scope false) so variables defined in `init()` persist.

use crate::draw::USER_LAYER;
use crate::gfx::color::Rgba;
use crate::gfx::math::Point2;
use crate::gfx::shape2d::{self, Line, Rotation, Shape, Stroke};

use rhai::{CallFnOptions, Engine, Scope, AST};

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

/// Type alias for the user batch shared between script and session.
pub type UserBatch = Rc<RefCell<shape2d::Batch>>;

/// Compile a script file into an AST.
pub fn compile_file(engine: &Engine, path: &Path) -> Result<AST, Box<rhai::EvalAltResult>> {
    engine.compile_file(path.into())
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
) -> Result<(), Box<rhai::EvalAltResult>> {
    match engine.call_fn::<()>(scope, ast, "draw", ()) {
        Ok(()) => Ok(()),
        Err(ref e) if is_function_not_found(e) => Ok(()),
        Err(e) => Err(e),
    }
}

fn is_function_not_found(e: &Box<rhai::EvalAltResult>) -> bool {
    use rhai::EvalAltResult;
    matches!(&**e, EvalAltResult::ErrorFunctionNotFound(_, _))
}

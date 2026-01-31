#![deny(clippy::all)]
#![allow(
    clippy::collapsible_if,
    clippy::many_single_char_names,
    clippy::expect_fun_call,
    clippy::useless_format,
    clippy::new_without_default,
    clippy::cognitive_complexity,
    clippy::comparison_chain,
    clippy::type_complexity,
    clippy::or_fun_call,
    clippy::nonminimal_bool,
    clippy::single_match,
    clippy::large_enum_variant
)]

pub mod data;
pub mod execution;
pub mod gfx;
pub mod logger;
pub mod session;

mod alloc;
mod autocomplete;
mod brush;
mod cmd;
mod script;
mod color;
mod draw;
mod event;
mod flood;
mod font;
mod history;
mod wgpu;
mod image;
mod io;
mod palette;
mod parser;
mod pixels;
mod platform;
mod renderer;
mod sprite;
mod timer;
mod view;

#[macro_use]
pub mod util;

use cmd::Value;
use event::Event;
use execution::{DigestMode, Execution, ExecutionMode};
use platform::{WindowEvent, WindowHint};
use renderer::Renderer;
use script::ScriptState;
use session::*;
use timer::FrameTimer;
use view::FileStatus;

#[macro_use]
extern crate log;

use directories as dirs;

use std::alloc::System;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Program version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[global_allocator]
pub static ALLOCATOR: alloc::Allocator = alloc::Allocator::new(System);

#[derive(Debug)]
pub struct Options<'a> {
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
    pub headless: bool,
    pub source: Option<PathBuf>,
    pub script: Option<PathBuf>,
    pub exec: ExecutionMode,
    pub glyphs: &'a [u8],
    pub debug: bool,
}

impl<'a> Default for Options<'a> {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            headless: false,
            resizable: true,
            source: None,
            script: None,
            exec: ExecutionMode::Normal,
            glyphs: data::GLYPHS,
            debug: false,
        }
    }
}

pub fn init<P: AsRef<Path>>(paths: &[P], options: Options<'_>) -> std::io::Result<()> {
    use std::io;

    debug!("options: {:?}", options);

    let hints = &[
        WindowHint::Resizable(options.resizable),
        WindowHint::Visible(!options.headless),
    ];
    let (mut win, mut events) = platform::init(
        "rx",
        options.width,
        options.height,
        hints,
        platform::GraphicsContext::None,
    )?;

    let scale_factor = win.scale_factor();
    let win_size = win.size();
    let (win_w, win_h) = (win_size.width as u32, win_size.height as u32);

    info!("framebuffer size: {}x{}", win_size.width, win_size.height);
    info!("scale factor: {}", scale_factor);

    let assets = data::Assets::new(options.glyphs);
    let proj_dirs = dirs::ProjectDirs::from("io", "cloudhead", "rx")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "config directory not found"))?;
    let base_dirs = dirs::BaseDirs::new()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "home directory not found"))?;
    let cwd = std::env::current_dir()?;
    let session = Session::new(win_w, win_h, cwd, proj_dirs, base_dirs)
        .with_blank(
            FileStatus::NoFile,
            Session::DEFAULT_VIEW_W,
            Session::DEFAULT_VIEW_H,
        )
        .init(options.source.clone())?;
    let session_handle = Rc::new(RefCell::new(session));

    let mut script_state = ScriptState::new();
    if let Some(path) = options.script.clone() {
        script_state.set_path(path);
        if let Err(e) = script_state.load_script(&session_handle) {
            log::error!("Error loading script: {}", e);
        }
    }

    let mut session = session_handle.borrow_mut();
    if options.debug {
        session
            .settings
            .set("debug", Value::Bool(true))
            .expect("'debug' is a bool'");
    }

    let mut execution = match options.exec {
        ExecutionMode::Normal => Execution::normal(),
        ExecutionMode::Replay(path, digest) => Execution::replaying(path, digest),
        ExecutionMode::Record(path, digest, gif) => {
            Execution::recording(path, digest, win_w as u16, win_h as u16, gif)
        }
    }?;

    // When working with digests, certain settings need to be overwritten
    // to ensure things work correctly.
    match &execution {
        Execution::Replaying { digest, .. } | Execution::Recording { digest, .. }
            if digest.mode != DigestMode::Ignore =>
        {
            session
                .settings
                .set("animation", Value::Bool(false))
                .expect("'animation' is a bool");
        }
        _ => {}
    }

    let wait_events = execution.is_normal() || execution.is_recording();

    let script_reload_pending: Option<Arc<AtomicBool>> =
        script_state.script_path().map(|_| Arc::new(AtomicBool::new(false)));

    if let (Some(script_path), Some(pending)) = (
        script_state.script_path().cloned(),
        script_reload_pending.as_ref(),
    ) {
        let pending_clone = Arc::clone(pending);
        let script_canonical = script_path.canonicalize().ok();
        let watch_dir = script_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        thread::spawn(move || {
            use notify::{RecursiveMode, Watcher};
            let pending_cb = Arc::clone(&pending_clone);
            let mut watcher = match notify::recommended_watcher(move |res: Result<notify::Event, _>| {
                if let Ok(event) = res {
                    let is_our_file = event.paths.iter().any(|p| {
                        p.canonicalize().ok().as_ref() == script_canonical.as_ref()
                    });
                    let is_modify = matches!(
                        event.kind,
                        notify::EventKind::Modify(_) | notify::EventKind::Create(_)
                    );
                    if is_our_file && is_modify {
                        pending_cb.store(true, Ordering::Relaxed);
                        unsafe { glfw::ffi::glfwPostEmptyEvent() };
                    }
                }
            }) {
                Ok(w) => w,
                Err(e) => {
                    log::warn!("script watcher: {}", e);
                    return;
                }
            };
            if watcher.watch(&watch_dir, RecursiveMode::NonRecursive).is_err() {
                log::warn!("script watcher: failed to watch {}", watch_dir.display());
                return;
            }
            let (_tx, rx) = std::sync::mpsc::channel::<()>();
            let _ = rx.recv();
        });
    }

    let mut renderer: wgpu::Renderer = Renderer::new(&mut win, win_size, scale_factor, assets)?;

    if let Err(e) = session.edit(paths) {
        session.message(format!("Error loading path(s): {}", e), MessageType::Error);
    }
    // Make sure our session ticks once before anything is rendered.
    let effects = session.update(
        &mut vec![],
        &mut execution,
        Duration::default(),
        Duration::default(),
    );
    renderer.init(effects, &session);

    let mut render_timer = FrameTimer::new();
    let mut update_timer = FrameTimer::new();
    let mut session_events = Vec::with_capacity(16);
    let mut last = Instant::now();
    let mut resized = false;
    let mut hovering = false;
    let mut delta;

    drop(session);
    while !win.is_closing() {
        let mut session = session_handle.borrow_mut();
        match session.animation_delay() {
            Some(delay) if session.is_running() => {
                // How much time is left until the next animation frame?
                let remaining = delay - session.accumulator;
                // If more than 1ms remains, let's wait.
                if remaining.as_millis() > 1 {
                    events.wait_timeout(remaining);
                } else {
                    events.poll();
                }
            }
            _ if wait_events => events.wait(),
            _ => events.poll(),
        }

        if let Some(ref flag) = script_reload_pending {
            if flag.swap(false, Ordering::Relaxed) && script_state.script_file_modified_since_load()
            {
                drop(session);
                script_state.reload_script(&session_handle);
                session = session_handle.borrow_mut();
            }
        }

        for event in events.flush() {
            if event.is_input() {
                debug!("event: {:?}", event);
            }

            match event {
                WindowEvent::Resized(size) => {
                    if size.is_zero() {
                        // On certain operating systems, the window size will be set to
                        // zero when the window is minimized. Since a zero-sized framebuffer
                        // is not valid, we pause the session until the window is restored.
                        session.transition(State::Paused);
                    } else {
                        resized = true;
                        session.transition(State::Running);
                    }
                }
                WindowEvent::CursorEntered { .. } => {
                    if win.is_focused() {
                        win.set_cursor_visible(false);
                    }
                    hovering = true;
                }
                WindowEvent::CursorLeft { .. } => {
                    win.set_cursor_visible(true);

                    hovering = false;
                }
                WindowEvent::Minimized => {
                    session.transition(State::Paused);
                }
                WindowEvent::Restored => {
                    if win.is_focused() {
                        session.transition(State::Running);
                    }
                }
                WindowEvent::Focused(true) => {
                    session.transition(State::Running);

                    if hovering {
                        win.set_cursor_visible(false);
                    }
                }
                WindowEvent::Focused(false) => {
                    win.set_cursor_visible(true);
                    session.transition(State::Paused);
                }
                WindowEvent::RedrawRequested => {
                    drop(session);
                    render_timer.run(|avg| {
                        renderer
                            .frame(&session_handle, &mut script_state, &mut execution, vec![], &avg)
                            .unwrap_or_else(|err| {
                                log::error!("{}", err);
                            });
                    });
                    win.present();
                    session = session_handle.borrow_mut();
                }
                WindowEvent::ScaleFactorChanged(factor) => {
                    renderer.handle_scale_factor_changed(factor);
                }
                WindowEvent::CloseRequested => {
                    session.quit(ExitReason::Normal);
                }
                WindowEvent::CursorMoved { position } => {
                    session_events.push(Event::CursorMoved(position));
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    session_events.push(Event::MouseInput(button, state));
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    session_events.push(Event::MouseWheel(delta));
                }
                WindowEvent::KeyboardInput(input) => match input {
                    // Intercept `<insert>` key for pasting.
                    //
                    // Reading from the clipboard causes the loop to wake up for some strange
                    // reason I cannot comprehend. So we only read from clipboard when we
                    // need to paste.
                    platform::KeyboardInput {
                        key: Some(platform::Key::Insert),
                        state: platform::InputState::Pressed,
                        modifiers: platform::ModifiersState { shift: true, .. },
                    } => {
                        session_events.push(Event::Paste(win.clipboard()));
                    }
                    _ => session_events.push(Event::KeyboardInput(input)),
                },
                WindowEvent::ReceivedCharacter(c, mods) => {
                    session_events.push(Event::ReceivedCharacter(c, mods));
                }
                _ => {}
            };
        }

        if resized {
            // Instead of responded to each resize event by creating a new framebuffer,
            // we respond to the event *once*, here.
            resized = false;
            session.handle_resized(win.size());
        }

        delta = last.elapsed();
        last += delta;

        // If we're paused, we want to keep the timer running to not get a
        // "jump" when we unpause, but skip session updates and rendering.
        if session.state == State::Paused {
            continue;
        }

        let effects =
            update_timer.run(|avg| session.update(&mut session_events, &mut execution, delta, avg));

        if let Some(path) = session.take_pending_script_path() {
            script_state.set_path(path);
            match script_state.load_script(&session_handle) {
                Ok(()) => session.message("Script loaded".to_string(), MessageType::Info),
                Err(e) => session.message(e, MessageType::Error),
            }
        }

        script_state.call_view_effects(&effects);

        // Handle script command effects before passing to renderer
        let mut renderer_effects = Vec::new();
        for eff in effects {
            match eff {
                Effect::RunScriptCommand(name, args) => {
                    match script_state.call_script_command(&name, args) {
                        Ok(true) => {} // Handler found and called
                        Ok(false) => {
                            session.message(
                                format!("Script command '{}' has no handler cmd_{}", name, name),
                                MessageType::Error,
                            );
                        }
                        Err(e) => {
                            session.message(
                                format!("Script command '{}' error: {}", name, e),
                                MessageType::Error,
                            );
                        }
                    }
                }
                other => renderer_effects.push(other),
            }
        }

        drop(session);
        render_timer.run(|avg| {
            renderer
                    .frame(&session_handle, &mut script_state, &mut execution, renderer_effects, &avg)
                    .unwrap_or_else(|err| {
                        log::error!("{}", err);
                    });
            });
        session = session_handle.borrow_mut();

        session.cleanup();
        win.present();

        match &session.state {
            State::Closing(ExitReason::Normal) => {
                return Ok(());
            }
            State::Closing(ExitReason::Error(e)) => {
                return Err(io::Error::new(io::ErrorKind::Other, e.clone()));
            }
            _ => {}
        }
    }

    Ok(())
}

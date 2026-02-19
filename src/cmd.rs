use crate::autocomplete::{self, Autocomplete, FileCompleter, FileCompleterOpts};
use crate::brush::BrushMode;
use crate::history::History;
use crate::parser::*;
use crate::platform;
use crate::session::{BindingTier, Direction, Input, Mode, ModeString, PanState, Tool, VisualState};

use memoir::traits::Parse;
use memoir::*;

use crate::gfx::Rect;
use crate::gfx::Rgba8;

use std::fmt;
use std::path::Path;

pub const COMMENT: char = '-';

#[derive(Clone, PartialEq, Debug)]
pub enum Op {
    Incr,
    Decr,
    Set(f32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Axis {
    Horizontal,
    Vertical,
}

/// User command. Most of the interactions available to
/// the user are modeled as commands that are processed
/// by the session.
#[derive(PartialEq, Debug, Clone)]
pub enum Command {
    // Brush
    Brush,
    BrushSet(BrushMode),
    BrushToggle(BrushMode),
    BrushSize(Op),
    BrushUnset(BrushMode),

    #[allow(dead_code)]
    Crop(Rect<u32>),
    ChangeDir(Option<String>),
    Echo(Value),

    // Files
    Edit(Vec<String>),
    EditFrames(Vec<String>),
    Export(Option<u32>, String),
    Write(Option<String>),
    WriteFrames(Option<String>),
    WriteQuit,
    Quit,
    QuitAll,
    ForceQuit,
    ForceQuitAll,
    Source(Option<String>),
    SetPluginDir(String),
    OpenPluginDir,

    // Frames
    FrameAdd,
    FrameClone(i32),
    FrameRemove,
    FramePrev,
    FrameNext,
    FrameResize(u32, u32),

    // Palette
    PaletteAdd(Rgba8),
    PaletteClear,
    PaletteGradient(Rgba8, Rgba8, usize),
    PaletteSample,
    PaletteSort,
    PaletteWrite(String),

    // Navigation
    Pan(i32, i32),
    Zoom(Op),

    PaintColor(Rgba8, i32, i32),
    PaintForeground(i32, i32),
    PaintBackground(i32, i32),
    PaintPalette(usize, i32, i32),
    PaintLine(Rgba8, i32, i32, i32, i32),

    // Selection
    SelectionMove(i32, i32),
    SelectionResize(i32, i32),
    SelectionOffset(i32, i32),
    SelectionExpand,
    SelectionPaste,
    SelectionYank,
    SelectionCut,
    SelectionFill(Option<Rgba8>),
    SelectionErase,
    SelectionJump(Direction),
    SelectionFlip(Axis),

    // Settings
    Set(String, Value),
    Toggle(String),
    Reset,
    Map(Box<KeyMapping>),
    MapClear,

    Slice(Option<usize>),
    Fill(Option<Rgba8>),

    SwapColors,

    Mode(Mode),
    Tool(Tool),
    ToolPrev,

    Undo,
    Redo,

    // View
    ViewCenter,
    ViewNext,
    ViewPrev,

    Noop,

    /// Script command registered by Rhai init(). (name, args)
    ScriptCommand(String, Vec<String>),
}

impl Command {
    pub fn repeats(&self) -> bool {
        matches!(
            self,
            Self::Zoom(_)
                | Self::BrushSize(_)
                | Self::Pan(_, _)
                | Self::Undo
                | Self::Redo
                | Self::ViewNext
                | Self::ViewPrev
                | Self::SelectionMove(_, _)
                | Self::SelectionJump(_)
                | Self::SelectionResize(_, _)
                | Self::SelectionOffset(_, _)
        )
    }

    /// Return the canonical `(name, args)` invocation for this command, using
    /// the same names as the command-line parser (i.e. what the user types after `:`).
    /// Returns `None` for `Noop`.
    pub fn to_invocation(&self) -> Option<(String, Vec<String>)> {
        let (name, args) = match self {
            Self::Noop => return None,

            // Brush
            Self::Brush => ("brush", vec![]),
            Self::BrushSet(m) => ("brush/set", vec![format!("{}", m)]),
            Self::BrushToggle(m) => ("brush/toggle", vec![format!("{}", m)]),
            Self::BrushSize(Op::Incr) => ("brush/size", vec!["+".into()]),
            Self::BrushSize(Op::Decr) => ("brush/size", vec!["-".into()]),
            Self::BrushSize(Op::Set(s)) => ("brush/size", vec![format!("{}", s)]),
            Self::BrushUnset(m) => ("brush/unset", vec![format!("{}", m)]),

            // Files
            Self::Edit(paths) => ("e", paths.clone()),
            Self::EditFrames(paths) => ("e/frames", paths.clone()),
            Self::Export(None, path) => ("export", vec![path.clone()]),
            Self::Export(Some(s), path) => ("export", vec![format!("@{}x", s), path.clone()]),
            Self::Write(None) => ("w", vec![]),
            Self::Write(Some(path)) => ("w", vec![path.clone()]),
            Self::WriteFrames(None) => ("w/frames", vec![]),
            Self::WriteFrames(Some(dir)) => ("w/frames", vec![dir.clone()]),
            Self::WriteQuit => ("wq", vec![]),
            Self::Quit => ("q", vec![]),
            Self::QuitAll => ("qa", vec![]),
            Self::ForceQuit => ("q!", vec![]),
            Self::ForceQuitAll => ("qa!", vec![]),
            Self::Source(None) => ("source", vec![]),
            Self::Source(Some(path)) => ("source", vec![path.clone()]),
            Self::SetPluginDir(path) => ("plugin-dir", vec![path.clone()]),
            Self::OpenPluginDir => ("plugin-dir/open", vec![]),
            Self::ChangeDir(None) => ("cd", vec![]),
            Self::ChangeDir(Some(dir)) => ("cd", vec![dir.clone()]),

            // Navigation
            Self::Pan(x, y) => ("pan", vec![format!("{}", x), format!("{}", y)]),
            Self::Zoom(Op::Incr) => ("zoom", vec!["+".into()]),
            Self::Zoom(Op::Decr) => ("zoom", vec!["-".into()]),
            Self::Zoom(Op::Set(z)) => ("zoom", vec![format!("{}", z)]),

            // Mode / Tool
            Self::Mode(m) => ("mode", vec![format!("{}", m)]),
            Self::Tool(Tool::Brush) => ("brush", vec![]),
            Self::Tool(Tool::FloodFill) => ("flood", vec![]),
            Self::Tool(Tool::Sampler) => ("sampler", vec![]),
            Self::Tool(Tool::Pan(_)) => ("tool", vec!["pan".into()]),
            Self::ToolPrev => ("tool/prev", vec![]),

            // Frames
            Self::FrameAdd => ("f/add", vec![]),
            Self::FrameClone(i) => ("f/clone", vec![format!("{}", i)]),
            Self::FrameRemove => ("f/remove", vec![]),
            Self::FramePrev => ("f/prev", vec![]),
            Self::FrameNext => ("f/next", vec![]),
            Self::FrameResize(w, h) => ("f/resize", vec![format!("{}", w), format!("{}", h)]),

            // Palette
            Self::PaletteAdd(c) => ("p/add", vec![format!("{}", c)]),
            Self::PaletteClear => ("p/clear", vec![]),
            Self::PaletteGradient(cs, ce, n) => (
                "p/gradient",
                vec![format!("{}", cs), format!("{}", ce), format!("{}", n)],
            ),
            Self::PaletteSample => ("p/sample", vec![]),
            Self::PaletteSort => ("p/sort", vec![]),
            Self::PaletteWrite(path) => ("p/write", vec![path.clone()]),

            // Selection
            Self::SelectionMove(x, y) => {
                ("selection/move", vec![format!("{}", x), format!("{}", y)])
            }
            Self::SelectionResize(x, y) => {
                ("selection/resize", vec![format!("{}", x), format!("{}", y)])
            }
            Self::SelectionOffset(x, y) => {
                ("selection/offset", vec![format!("{}", x), format!("{}", y)])
            }
            Self::SelectionExpand => ("selection/expand", vec![]),
            Self::SelectionPaste => ("selection/paste", vec![]),
            Self::SelectionYank => ("selection/yank", vec![]),
            Self::SelectionCut => ("selection/cut", vec![]),
            Self::SelectionFill(None) => ("selection/fill", vec![]),
            Self::SelectionFill(Some(c)) => ("selection/fill", vec![format!("{}", c)]),
            Self::SelectionErase => ("selection/erase", vec![]),
            Self::SelectionJump(Direction::Forward) => ("selection/jump", vec!["+".into()]),
            Self::SelectionJump(Direction::Backward) => ("selection/jump", vec!["-".into()]),
            Self::SelectionFlip(Axis::Horizontal) => ("selection/flip", vec!["x".into()]),
            Self::SelectionFlip(Axis::Vertical) => ("selection/flip", vec!["y".into()]),

            // Paint
            Self::PaintColor(rgba, x, y) => (
                "paint/color",
                vec![format!("{}", rgba), format!("{}", x), format!("{}", y)],
            ),
            Self::PaintForeground(x, y) => {
                ("paint/fg", vec![format!("{}", x), format!("{}", y)])
            }
            Self::PaintBackground(x, y) => {
                ("paint/bg", vec![format!("{}", x), format!("{}", y)])
            }
            Self::PaintPalette(i, x, y) => {
                ("paint/p", vec![format!("{}", i), format!("{}", x), format!("{}", y)])
            }
            Self::PaintLine(rgba, x1, y1, x2, y2) => (
                "paint/line",
                vec![
                    format!("{}", rgba),
                    format!("{}", x1),
                    format!("{}", y1),
                    format!("{}", x2),
                    format!("{}", y2),
                ],
            ),

            // Settings
            Self::Set(s, v) => ("set", vec![format!("{}", s), "=".into(), format!("{}", v)]),
            Self::Toggle(s) => ("toggle", vec![s.clone()]),
            Self::Reset => ("reset!", vec![]),
            Self::Map(_) | Self::MapClear => ("map/clear!", vec![]),
            Self::Echo(v) => ("echo", vec![format!("{}", v)]),

            Self::Fill(None) => ("v/fill", vec![]),
            Self::Fill(Some(c)) => ("v/fill", vec![format!("{}", c)]),
            Self::Slice(None) => ("slice", vec![]),
            Self::Slice(Some(n)) => ("slice", vec![format!("{}", n)]),
            Self::SwapColors => ("swap", vec![]),
            Self::Undo => ("undo", vec![]),
            Self::Redo => ("redo", vec![]),

            // View
            Self::ViewCenter => ("v/center", vec![]),
            Self::ViewNext => ("v/next", vec![]),
            Self::ViewPrev => ("v/prev", vec![]),

            // Crop is currently dead code
            Self::Crop(_) => return None,

            // Script command: already has (name, args)
            Self::ScriptCommand(name, args) => (name.as_str(), args.clone()),
        };
        Some((name.to_string(), args))
    }
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Brush => write!(f, "Reset brush"),
            Self::BrushSet(m) => write!(f, "Set brush mode to `{}`", m),
            Self::BrushToggle(m) => write!(f, "Toggle `{}` brush mode", m),
            Self::BrushSize(Op::Incr) => write!(f, "Increase brush size"),
            Self::BrushSize(Op::Decr) => write!(f, "Decrease brush size"),
            Self::BrushSize(Op::Set(s)) => write!(f, "Set brush size to {}", s),
            Self::BrushUnset(m) => write!(f, "Unset brush `{}` mode", m),
            Self::Crop(_) => write!(f, "Crop view"),
            Self::ChangeDir(_) => write!(f, "Change the current working directory"),
            Self::Echo(_) => write!(f, "Echo a value"),
            Self::Edit(_) => write!(f, "Edit path(s)"),
            Self::EditFrames(_) => write!(f, "Edit path(s) as animation frames"),
            Self::Fill(Some(c)) => write!(f, "Fill view with {color}", color = c),
            Self::Fill(None) => write!(f, "Fill view with background color"),
            Self::ForceQuit => write!(f, "Quit view without saving"),
            Self::ForceQuitAll => write!(f, "Quit all views without saving"),
            Self::Map(_) => write!(f, "Map a key combination to a command"),
            Self::MapClear => write!(f, "Clear all key mappings"),
            Self::Mode(Mode::Help) => write!(f, "Toggle help"),
            Self::Mode(m) => write!(f, "Switch to {} mode", m),
            Self::FrameAdd => write!(f, "Add a blank frame to the view"),
            Self::FrameClone(i) => write!(f, "Clone frame {} and add it to the view", i),
            Self::FrameRemove => write!(f, "Remove the last frame of the view"),
            Self::FramePrev => write!(f, "Navigate to previous frame"),
            Self::FrameNext => write!(f, "Navigate to next frame"),
            Self::Noop => write!(f, "No-op"),
            Self::PaletteAdd(c) => write!(f, "Add {color} to palette", color = c),
            Self::PaletteClear => write!(f, "Clear palette"),
            Self::PaletteGradient(cs, ce, n) => {
                write!(f, "Create {n} colors gradient from {cs} to {ce}")
            }
            Self::PaletteSample => write!(f, "Sample palette from view"),
            Self::PaletteSort => write!(f, "Sort palette colors"),
            Self::Pan(x, 0) if *x > 0 => write!(f, "Pan workspace right"),
            Self::Pan(x, 0) if *x < 0 => write!(f, "Pan workspace left"),
            Self::Pan(0, y) if *y > 0 => write!(f, "Pan workspace up"),
            Self::Pan(0, y) if *y < 0 => write!(f, "Pan workspace down"),
            Self::Pan(x, y) => write!(f, "Pan workspace by {},{}", x, y),
            Self::Quit => write!(f, "Quit active view"),
            Self::QuitAll => write!(f, "Quit all views"),
            Self::Redo => write!(f, "Redo view edit"),
            Self::FrameResize(_, _) => write!(f, "Resize active view frame"),
            Self::Tool(Tool::Pan(_)) => write!(f, "Pan tool"),
            Self::Tool(Tool::Brush) => write!(f, "Brush tool"),
            Self::Tool(Tool::Sampler) => write!(f, "Color sampler tool"),
            Self::Tool(Tool::FloodFill) => write!(f, "Flood fill tool"),
            Self::ToolPrev => write!(f, "Switch to previous tool"),
            Self::Set(s, v) => write!(f, "Set {setting} to {val}", setting = s, val = v),
            Self::Slice(Some(n)) => write!(f, "Slice view into {} frame(s)", n),
            Self::Slice(None) => write!(f, "Reset view slices"),
            Self::Source(_) => write!(f, "Source an rx script (eg. a palette)"),
            Self::SetPluginDir(_) => write!(f, "Set plugin directory (loads *.rxx)"),
            Self::OpenPluginDir => write!(f, "Open plugin directory in file explorer"),
            Self::SwapColors => write!(f, "Swap foreground & background colors"),
            Self::Toggle(s) => write!(f, "Toggle {setting} on/off", setting = s),
            Self::Undo => write!(f, "Undo view edit"),
            Self::ViewCenter => write!(f, "Center active view"),
            Self::ViewNext => write!(f, "Go to next view"),
            Self::ViewPrev => write!(f, "Go to previous view"),
            Self::Write(None) => write!(f, "Write view to disk"),
            Self::Write(Some(_)) => write!(f, "Write view to disk as..."),
            Self::WriteQuit => write!(f, "Write file to disk and quit"),
            Self::Zoom(Op::Incr) => write!(f, "Zoom in view"),
            Self::Zoom(Op::Decr) => write!(f, "Zoom out view"),
            Self::Zoom(Op::Set(z)) => write!(f, "Set view zoom to {:.1}", z),
            Self::Reset => write!(f, "Reset all settings to default"),
            Self::SelectionFill(None) => write!(f, "Fill selection with foreground color"),
            Self::SelectionYank => write!(f, "Yank (copy) selection"),
            Self::SelectionCut => write!(f, "Cut selection"),
            Self::SelectionPaste => write!(f, "Paste selection"),
            Self::SelectionExpand => write!(f, "Expand selection to frame"),
            Self::SelectionOffset(1, 1) => write!(f, "Outset selection"),
            Self::SelectionOffset(-1, -1) => write!(f, "Inset selection"),
            Self::SelectionOffset(x, y) => write!(f, "Offset selection by {:2},{:2}", x, y),
            Self::SelectionMove(x, 0) if *x > 0 => write!(f, "Move selection right"),
            Self::SelectionMove(x, 0) if *x < 0 => write!(f, "Move selection left"),
            Self::SelectionMove(0, y) if *y > 0 => write!(f, "Move selection up"),
            Self::SelectionMove(0, y) if *y < 0 => write!(f, "Move selection down"),
            Self::SelectionJump(Direction::Forward) => {
                write!(f, "Move selection forward by one frame")
            }
            Self::SelectionJump(Direction::Backward) => {
                write!(f, "Move selection backward by one frame")
            }
            Self::SelectionErase => write!(f, "Erase selection contents"),
            Self::SelectionFlip(Axis::Horizontal) => write!(f, "Flip selection horizontally"),
            Self::SelectionFlip(Axis::Vertical) => write!(f, "Flip selection vertically"),
            Self::PaintColor(_, x, y) => write!(f, "Paint {:2},{:2}", x, y),
            Self::ScriptCommand(name, _) => write!(f, "Script command: {}", name),
            _ => write!(f, "..."),
        }
    }
}

impl From<Command> for String {
    fn from(cmd: Command) -> Self {
        match cmd {
            Command::Brush => format!("brush"),
            Command::BrushSet(m) => format!("brush/set {}", m),
            Command::BrushSize(Op::Incr) => format!("brush/size +"),
            Command::BrushSize(Op::Decr) => format!("brush/size -"),
            Command::BrushSize(Op::Set(s)) => format!("brush/size {}", s),
            Command::BrushUnset(m) => format!("brush/unset {}", m),
            Command::Echo(_) => unimplemented!(),
            Command::Edit(_) => unimplemented!(),
            Command::Fill(Some(c)) => format!("v/fill {}", c),
            Command::Fill(None) => format!("v/fill"),
            Command::ForceQuit => format!("q!"),
            Command::ForceQuitAll => format!("qa!"),
            Command::Map(_) => format!("map <key> <command> {{<command>}}"),
            Command::Mode(m) => format!("mode {}", m),
            Command::FrameAdd => format!("f/add"),
            Command::FrameClone(i) => format!("f/clone {}", i),
            Command::FrameRemove => format!("f/remove"),
            Command::Export(None, path) => format!("export {}", path),
            Command::Export(Some(s), path) => format!("export @{}x {}", s, path),
            Command::Noop => format!(""),
            Command::ScriptCommand(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    format!("{} {}", name, args.join(" "))
                }
            }
            Command::PaletteAdd(c) => format!("p/add {}", c),
            Command::PaletteClear => format!("p/clear"),
            Command::PaletteWrite(_) => format!("p/write"),
            Command::PaletteSample => format!("p/sample"),
            Command::PaletteGradient(cs, ce, n) => format!("p/gradient {} {} {}", cs, ce, n),
            Command::Pan(x, y) => format!("pan {} {}", x, y),
            Command::Quit => format!("q"),
            Command::Redo => format!("redo"),
            Command::FrameResize(w, h) => format!("f/resize {} {}", w, h),
            Command::Set(s, v) => format!("set {} = {}", s, v),
            Command::Slice(Some(n)) => format!("slice {}", n),
            Command::Slice(None) => format!("slice"),
            Command::Source(Some(path)) => format!("source {}", path),
            Command::SetPluginDir(path) => format!("plugin-dir {}", path),
            Command::OpenPluginDir => format!("plugin-dir/open"),
            Command::SwapColors => format!("swap"),
            Command::Toggle(s) => format!("toggle {}", s),
            Command::Undo => format!("undo"),
            Command::ViewCenter => format!("v/center"),
            Command::ViewNext => format!("v/next"),
            Command::ViewPrev => format!("v/prev"),
            Command::Write(None) => format!("w"),
            Command::Write(Some(path)) => format!("w {}", path),
            Command::WriteQuit => format!("wq"),
            Command::Zoom(Op::Incr) => format!("v/zoom +"),
            Command::Zoom(Op::Decr) => format!("v/zoom -"),
            Command::Zoom(Op::Set(z)) => format!("v/zoom {}", z),
            _ => unimplemented!(),
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

#[derive(PartialEq, Debug, Clone)]
pub struct KeyMapping {
    pub input: Input,
    pub press: Command,
    pub release: Option<Command>,
    pub repeats: Option<bool>,
    pub tier: BindingTier,
}

impl KeyMapping {
    pub fn parser(tier: BindingTier) -> Parser<KeyMapping> {
        // Prevent stack overflow.
        let press = Parser::new(
            move |input| Commands::default().parser().parse(input),
            "<cmd>",
        );

        // Prevent stack overflow.
        let release = Parser::new(
            move |input| {
                if let Some(i) = input.bytes().position(|c| c == b'}') {
                    match Commands::default().parser().parse(&input[..i]) {
                        Ok((cmd, rest)) if rest.is_empty() => Ok((cmd, &input[i..])),
                        Ok((_, rest)) => {
                            Err((format!("expected {:?}, got {:?}", '}', rest).into(), rest))
                        }
                        Err(err) => Err(err),
                    }
                } else {
                    Err(("unclosed '{' delimiter".into(), input))
                }
            },
            "<cmd>",
        );

        let character = between('\'', '\'', character())
            .map(Input::Character)
            .skip(whitespace())
            .then(press.clone())
            .map(|(input, press)| ((input, press), None));
        let key = param::<platform::Key>()
            .map(Input::Key)
            .skip(whitespace())
            .then(press)
            .skip(optional(whitespace()))
            .then(optional(between('{', '}', release)));

        character
            .or(key)
            .map(move |((input, press), release)| KeyMapping {
                input,
                press,
                release,
                repeats: None,
                tier: tier.clone(),
            })
            .label("<key> <cmd>") // TODO: We should provide the full command somehow.
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    Bool(bool),
    U32(u32),
    U32Tuple(u32, u32),
    F32Tuple(f32, f32),
    F64(f64),
    Str(String),
    Ident(String),
    Rgba8(Rgba8),
}

impl Value {
    pub fn is_set(&self) -> bool {
        if let Value::Bool(b) = self {
            return *b;
        }
        panic!("expected {:?} to be a `bool`", self);
    }

    /// Non-panicking: returns `Some(b)` for `Bool(b)`, `None` otherwise.
    pub fn try_is_set(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn to_f64(&self) -> f64 {
        if let Value::F64(n) = self {
            return *n;
        }
        panic!("expected {:?} to be a `float`", self);
    }

    /// Non-panicking: returns `Some(n)` for `F64(n)`, `None` otherwise.
    pub fn try_to_f64(&self) -> Option<f64> {
        match self {
            Value::F64(n) => Some(*n),
            _ => None,
        }
    }

    pub fn to_u64(&self) -> u64 {
        if let Value::U32(n) = self {
            return *n as u64;
        }
        panic!("expected {:?} to be a `uint`", self);
    }

    /// Non-panicking: returns `Some(n)` for `U32(n)`, `None` otherwise.
    pub fn try_to_u64(&self) -> Option<u64> {
        match self {
            Value::U32(n) => Some(*n as u64),
            _ => None,
        }
    }

    pub fn to_rgba8(&self) -> Rgba8 {
        if let Value::Rgba8(rgba8) = self {
            return *rgba8;
        }
        panic!("expected {:?} to be a `Rgba8`", self);
    }

    /// Non-panicking: returns `Some(c)` for `Rgba8(c)`, `None` otherwise.
    pub fn try_to_rgba8(&self) -> Option<Rgba8> {
        match self {
            Value::Rgba8(rgba8) => Some(*rgba8),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Bool(_) => "on / off",
            Self::U32(_) => "positive integer, eg. 32",
            Self::F64(_) => "float, eg. 1.33",
            Self::U32Tuple(_, _) => "two positive integers, eg. 32, 48",
            Self::F32Tuple(_, _) => "two floats , eg. 32.17, 48.29",
            Self::Str(_) => "string, eg. \"fnord\"",
            Self::Rgba8(_) => "color, eg. #ffff00",
            Self::Ident(_) => "identifier, eg. fnord",
        }
    }
}

impl From<Value> for (u32, u32) {
    fn from(other: Value) -> (u32, u32) {
        if let Value::U32Tuple(x, y) = other {
            return (x, y);
        }
        panic!("expected {:?} to be a `(u32, u32)`", other);
    }
}

impl From<Value> for f32 {
    fn from(other: Value) -> f32 {
        if let Value::F64(x) = other {
            return x as f32;
        }
        panic!("expected {:?} to be a `f64`", other);
    }
}

impl From<Value> for f64 {
    fn from(other: Value) -> f64 {
        if let Value::F64(x) = other {
            return x;
        }
        panic!("expected {:?} to be a `f64`", other);
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(true) => "on".fmt(f),
            Value::Bool(false) => "off".fmt(f),
            Value::U32(u) => u.fmt(f),
            Value::F64(x) => x.fmt(f),
            Value::U32Tuple(x, y) => write!(f, "{},{}", x, y),
            Value::F32Tuple(x, y) => write!(f, "{},{}", x, y),
            Value::Str(s) => s.fmt(f),
            Value::Rgba8(c) => c.fmt(f),
            Value::Ident(i) => i.fmt(f),
        }
    }
}

impl Parse for Value {
    fn parser() -> Parser<Self> {
        let str_val = quoted().map(Value::Str).label("<string>");
        let rgba8_val = color().map(Value::Rgba8);
        let u32_tuple_val = tuple::<u32>(natural(), natural()).map(|(x, y)| Value::U32Tuple(x, y));
        let u32_val = natural::<u32>().map(Value::U32);
        let f64_tuple_val =
            tuple::<f32>(rational(), rational()).map(|(x, y)| Value::F32Tuple(x, y));
        let f64_val = rational::<f64>().map(Value::F64).label("0.0 .. 4096.0");
        let bool_val = string("on")
            .value(Value::Bool(true))
            .or(string("off").value(Value::Bool(false)))
            .label("on/off");
        let ident_val = identifier().map(Value::Ident);

        greediest(vec![
            rgba8_val,
            u32_tuple_val,
            f64_tuple_val,
            u32_val,
            f64_val,
            bool_val,
            ident_val,
            str_val,
        ])
        .label("<value>")
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct CommandLine {
    /// The history of commands entered.
    pub history: History,
    /// Command auto-complete.
    pub autocomplete: Autocomplete<CommandCompleter>,
    /// Input cursor position.
    pub cursor: usize,
    /// Parser.
    pub parser: Parser<Command>,
    /// Commands.
    pub commands: Commands,
    /// The current input string displayed to the user.
    input: String,
    /// File extensions supported.
    extensions: Vec<String>,
}

impl CommandLine {
    const MAX_INPUT: usize = 256;

    pub fn new<P: AsRef<Path>>(cwd: P, history_path: P, extensions: &[&str]) -> Self {
        let cmds = Commands::default();

        Self {
            input: String::with_capacity(Self::MAX_INPUT),
            cursor: 0,
            parser: cmds.line_parser(),
            commands: cmds,
            history: History::new(history_path, 1024),
            autocomplete: Autocomplete::new(CommandCompleter::new(cwd, extensions)),
            extensions: extensions.iter().map(|e| (*e).into()).collect(),
        }
    }

    pub fn set_cwd(&mut self, path: &Path) {
        let exts: Vec<_> = self.extensions.iter().map(|s| s.as_str()).collect();
        self.autocomplete = Autocomplete::new(CommandCompleter::new(path, exts.as_slice()));
    }

    /// Set script commands registered by Rhai init() and rebuild the parser.
    pub fn set_script_commands(&mut self, cmds: Vec<(String, String)>) {
        self.commands.set_script_commands(cmds);
        self.parser = self.commands.line_parser();
    }

    /// Get script commands (name, help) for help display.
    pub fn script_commands(&self) -> &[(String, String)] {
        self.commands.script_commands()
    }

    pub fn parse(&self, input: &str) -> Result<Command, Error> {
        match self.parser.parse(input) {
            Ok((cmd, _)) => Ok(cmd),
            Err((err, _)) => Err(err),
        }
    }

    pub fn input(&self) -> String {
        self.input.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.input.is_empty()
    }

    pub fn history_prev(&mut self) {
        let prefix = self.prefix();

        if let Some(entry) = self.history.prev(&prefix).map(str::to_owned) {
            self.replace(&entry);
        }
    }

    pub fn history_next(&mut self) {
        let prefix = self.prefix();

        if let Some(entry) = self.history.next(&prefix).map(str::to_owned) {
            self.replace(&entry);
        } else {
            self.reset();
        }
    }

    pub fn completion_next(&mut self) {
        let prefix = self.prefix();

        if let Some((completion, range)) = self.autocomplete.next(&prefix, self.cursor) {
            // Replace old completion with new one.
            self.cursor = range.start + completion.len();
            self.input.replace_range(range, &completion);
        }
    }

    pub fn cursor_backward(&mut self) -> Option<char> {
        if let Some(c) = self.peek_back() {
            let cursor = self.cursor - c.len_utf8();

            // Don't allow deleting the `:` prefix of the command.
            if c != ':' || cursor > 0 {
                self.cursor = cursor;
                self.autocomplete.invalidate();
                return Some(c);
            }
        }
        None
    }

    pub fn cursor_forward(&mut self) -> Option<char> {
        if let Some(c) = self.input[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
            self.autocomplete.invalidate();
            Some(c)
        } else {
            None
        }
    }

    pub fn cursor_back(&mut self) {
        if self.cursor > 1 {
            self.cursor = 1;
            self.autocomplete.invalidate();
        }
    }

    pub fn cursor_front(&mut self) {
        self.cursor = self.input.len();
    }

    pub fn putc(&mut self, c: char) {
        if self.input.len() + c.len_utf8() > self.input.capacity() {
            return;
        }
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.autocomplete.invalidate();
    }

    pub fn puts(&mut self, s: &str) {
        // TODO: Check capacity.
        self.input.push_str(s);
        self.cursor += s.len();
        self.autocomplete.invalidate();
    }

    pub fn delc(&mut self) {
        match self.peek_back() {
            // Don't allow deleting the ':' unless it's the last remaining character.
            Some(c) if self.cursor > 1 || self.input.len() == 1 => {
                self.cursor -= c.len_utf8();
                self.input.remove(self.cursor);
                self.autocomplete.invalidate();
            }
            _ => {}
        }
    }

    pub fn clear(&mut self) {
        self.cursor = 0;
        self.input.clear();
        self.history.reset();
        self.autocomplete.invalidate();
    }

    ////////////////////////////////////////////////////////////////////////////

    fn replace(&mut self, s: &str) {
        // We don't re-assign `input` here, because it
        // has a fixed capacity we want to preserve.
        self.input.clear();
        self.input.push_str(s);
        self.autocomplete.invalidate();
    }

    fn reset(&mut self) {
        self.clear();
        self.putc(':');
    }

    fn prefix(&self) -> String {
        self.input[..self.cursor].to_owned()
    }

    #[cfg(test)]
    fn peek(&self) -> Option<char> {
        self.input[self.cursor..].chars().next()
    }

    fn peek_back(&self) -> Option<char> {
        self.input[..self.cursor].chars().next_back()
    }
}

pub struct Commands {
    commands: Vec<(&'static str, &'static str, Parser<Command>)>,
    /// Script commands registered by Rhai init(). (name, help)
    script_commands: Vec<(String, String)>,
}

impl Commands {
    pub fn new() -> Self {
        Self {
            commands: vec![(
                "#",
                "Add color to palette",
                color().map(Command::PaletteAdd),
            )],
            script_commands: Vec::new(),
        }
    }

    /// Set/replace script commands registered by Rhai init().
    pub fn set_script_commands(&mut self, cmds: Vec<(String, String)>) {
        self.script_commands = cmds;
    }

    /// Get script command names (for autocomplete).
    pub fn script_command_names(&self) -> Vec<String> {
        self.script_commands
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Get script commands (name, help) for help display.
    pub fn script_commands(&self) -> &[(String, String)] {
        &self.script_commands
    }

    pub fn parser(&self) -> Parser<Command> {
        use std::iter;

        let noop = expect(|s| s.is_empty(), "<empty>").value(Command::Noop);
        let commands = self.commands.iter().map(|(_, _, v)| v.clone());

        // Combine: static commands and noop
        let all_choices: Vec<Parser<Command>> = commands
            .chain(iter::once(noop))
            .collect();

        // Fallback: any unknown command is treated as a script command
        let script_fallback: Parser<Command> = Parser::new(
            |input: &str| {
                // Extract command name up to first whitespace
                let (name, rest) = match input.find(char::is_whitespace) {
                    Some(idx) => (&input[..idx], &input[idx..]),
                    None => (input, ""),
                };

                if name.is_empty() {
                    return Err(("expected <command>".into(), input));
                }

                let args: Vec<String> = rest
                    .trim_start()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                Ok((Command::ScriptCommand(name.to_string(), args), ""))
            },
            "<script-cmd>",
        );

        symbol(':')
            .then(
                choice(all_choices).or(script_fallback),
            )
            .map(|(_, cmd)| cmd)
    }

    pub fn line_parser(&self) -> Parser<Command> {
        self.parser()
            .skip(optional(whitespace()))
            .skip(optional(comment()))
            .end()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(&'static str, &'static str, Parser<Command>)> {
        self.commands.iter()
    }

    ///////////////////////////////////////////////////////////////////////////

    fn command<F>(mut self, name: &'static str, help: &'static str, f: F) -> Self
    where
        F: Fn(Parser<String>) -> Parser<Command>,
    {
        let cmd = peek(
            string(name)
                .followed_by(hush(whitespace()) / end())
                .skip(optional(whitespace())),
        )
        .label(name);

        self.commands.push((name, help, f(cmd)));
        self
    }
}

impl Default for Commands {
    fn default() -> Self {
        Self::new()
            .command("q", "Quit view", |p| p.value(Command::Quit))
            .command("qa", "Quit all views", |p| p.value(Command::QuitAll))
            .command("q!", "Force quit view", |p| p.value(Command::ForceQuit))
            .command("qa!", "Force quit all views", |p| {
                p.value(Command::ForceQuitAll)
            })
            .command("export", "Export view", |p| {
                p.then(optional(scale().skip(whitespace())).then(path()))
                    .map(|(_, (scale, path))| Command::Export(scale, path))
            })
            .command("wq", "Write & quit view", |p| p.value(Command::WriteQuit))
            .command("x", "Write & quit view", |p| p.value(Command::WriteQuit))
            .command("w", "Write view", |p| {
                p.then(optional(path()))
                    .map(|(_, path)| Command::Write(path))
            })
            .command("w/frames", "Write view as individual frames", |p| {
                p.then(optional(path()))
                    .map(|(_, dir)| Command::WriteFrames(dir))
            })
            .command("e", "Edit path(s)", |p| {
                p.then(paths()).map(|(_, paths)| Command::Edit(paths))
            })
            .command("e/frames", "Edit frames as view", |p| {
                p.then(paths()).map(|(_, paths)| Command::EditFrames(paths))
            })
            .command("help", "Display help", |p| {
                p.value(Command::Mode(Mode::Help))
            })
            .command("set", "Set setting to value", |p| {
                p.then(setting())
                    .skip(optional(whitespace()))
                    .then(optional(
                        symbol('=')
                            .skip(optional(whitespace()))
                            .then(Value::parser())
                            .map(|(_, v)| v),
                    ))
                    .map(|((_, k), v)| Command::Set(k, v.unwrap_or(Value::Bool(true))))
            })
            .command("unset", "Set setting to `off`", |p| {
                p.then(setting())
                    .map(|(_, k)| Command::Set(k, Value::Bool(false)))
            })
            .command("toggle", "Toggle setting", |p| {
                p.then(setting()).map(|(_, k)| Command::Toggle(k))
            })
            .command("echo", "Echo setting or value", |p| {
                p.then(Value::parser()).map(|(_, v)| Command::Echo(v))
            })
            .command("slice", "Slice view into <n> frames", |p| {
                p.then(optional(natural::<usize>().label("<n>")))
                    .map(|(_, n)| Command::Slice(n))
            })
            .command(
                "source",
                "Source an rx script (eg. palette or config)",
                |p| p.then(optional(path())).map(|(_, p)| Command::Source(p)),
            )
            .command("plugin-dir", "Set plugin directory (loads *.rxx Rhai scripts)", |p| {
                p.then(path()).map(|(_, path)| Command::SetPluginDir(path))
            })
            .command("plugin-dir/open", "Open plugin directory in file explorer", |p| {
                p.value(Command::OpenPluginDir)
            })
            .command("cd", "Change current directory", |p| {
                p.then(optional(path())).map(|(_, p)| Command::ChangeDir(p))
            })
            .command("zoom", "Zoom view", |p| {
                p.then(
                    peek(rational::<f32>().label("<level>"))
                        .try_map(|z| {
                            if z >= 1.0 {
                                Ok(Command::Zoom(Op::Set(z)))
                            } else {
                                Err("zoom level must be >= 1.0")
                            }
                        })
                        .or(symbol('+')
                            .value(Command::Zoom(Op::Incr))
                            .or(symbol('-').value(Command::Zoom(Op::Decr)))
                            .or(fail("couldn't parse zoom parameter")))
                        .label("+/-"),
                )
                .map(|(_, cmd)| cmd)
            })
            .command("brush/size", "Set brush size", |p| {
                p.then(
                    natural::<usize>()
                        .label("<size>")
                        .map(|z| Command::BrushSize(Op::Set(z as f32)))
                        .or(symbol('+')
                            .value(Command::BrushSize(Op::Incr))
                            .or(symbol('-').value(Command::BrushSize(Op::Decr)))
                            .or(fail("couldn't parse brush size parameter")))
                        .label("+/-"),
                )
                .map(|(_, cmd)| cmd)
            })
            .command(
                "brush/set",
                "Set brush mode, eg. `xsym` for x-symmetry",
                |p| {
                    p.then(param::<BrushMode>())
                        .map(|(_, m)| Command::BrushSet(m))
                },
            )
            .command("brush/unset", "Unset brush mode", |p| {
                p.then(param::<BrushMode>())
                    .map(|(_, m)| Command::BrushUnset(m))
            })
            .command("brush/toggle", "Toggle brush mode", |p| {
                p.then(param::<BrushMode>())
                    .map(|(_, m)| Command::BrushToggle(m))
            })
            .command("brush", "Switch to brush", |p| {
                p.value(Command::Tool(Tool::Brush))
            })
            .command("flood", "Switch to flood fill tool", |p| {
                p.value(Command::Tool(Tool::FloodFill))
            })
            .command("mode", "Set session mode, eg. `visual` or `normal`", |p| {
                p.then(param::<Mode>()).map(|(_, m)| Command::Mode(m))
            })
            .command("visual", "Set session mode to visual", |p| {
                p.map(|_| Command::Mode(Mode::Visual(VisualState::default())))
            })
            .command("sampler/off", "Switch the sampler tool off", |p| {
                p.value(Command::ToolPrev)
            })
            .command("sampler", "Switch to the sampler tool", |p| {
                p.value(Command::Tool(Tool::Sampler))
            })
            .command("v/next", "Activate the next view", |p| {
                p.value(Command::ViewNext)
            })
            .command("v/prev", "Activate the previous view", |p| {
                p.value(Command::ViewPrev)
            })
            .command("v/center", "Center the active view", |p| {
                p.value(Command::ViewCenter)
            })
            .command("v/clear", "Clear the active view", |p| {
                p.value(Command::Fill(Some(Rgba8::TRANSPARENT)))
            })
            .command("v/fill", "Fill the active view", |p| {
                p.then(optional(color())).map(|(_, c)| Command::Fill(c))
            })
            .command("pan", "Switch to the pan tool", |p| {
                p.then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|(_, (x, y))| Command::Pan(x, y))
            })
            .command("map", "Map keys to a command in all modes", |p| {
                p.then(KeyMapping::parser(BindingTier::General))
                .map(|(_, km)| Command::Map(Box::new(km)))
            })
            .command("map/visual", "Map keys to a command in visual mode", |p| {
                p.then(KeyMapping::parser(BindingTier::ModeSpecific(vec![
                    Mode::Visual(VisualState::selecting()),
                    Mode::Visual(VisualState::Pasting),
                ])))
                .map(|(_, km)| Command::Map(Box::new(km)))
            })
            .command("map/normal", "Map keys to a command in normal mode", |p| {
                p.then(KeyMapping::parser(BindingTier::ModeSpecific(vec![Mode::Normal])))
                    .map(|(_, km)| Command::Map(Box::new(km)))
            })
            .command("map/help", "Map keys to a command in help mode", |p| {
                p.then(KeyMapping::parser(BindingTier::ModeSpecific(vec![Mode::Help])))
                    .map(|(_, km)| Command::Map(Box::new(km)))
            })
            .command("map/script", "Map keys to a command in a script mode", |p| {
                p.then(Parser::new(
                    |input: &str| {
                        let input = input.trim_start();
                        let (repeats, rest) = optional(string("repeats").skip(whitespace()))
                            .map(|v| v.is_some())
                            .parse(input)?;
                        let (name_str, rest) = quoted().parse(rest.trim_start())
                            .map_err(|(e, _)| (e, input))?;
                        let name = ModeString::try_from_str(&name_str)
                            .map_err(|e| (format!("{}", e).into(), rest))?;
                        let rest = rest.trim_start();
                        let (mut km, rest) =
                            KeyMapping::parser(BindingTier::Script(name))
                                .parse(rest)?;
                        km.repeats = Some(repeats);
                        Ok((km, rest))
                    },
                    "[repeats] <script-mode> <key> <cmd>",
                ))
                .map(|(_, km)| Command::Map(Box::new(km)))
            })
            .command("map/clear!", "Clear all key mappings", |p| {
                p.value(Command::MapClear)
            })
            .command("p/add", "Add a color to the palette", |p| {
                p.then(color()).map(|(_, rgba)| Command::PaletteAdd(rgba))
            })
            .command("p/clear", "Clear the color palette", |p| {
                p.value(Command::PaletteClear)
            })
            .command("p/gradient", "Add a gradient to the palette", |p| {
                p.then(tuple::<Rgba8>(
                    color().label("<from>"),
                    color().label("<to>"),
                ))
                .skip(whitespace())
                .then(natural::<usize>().label("<count>"))
                .map(|((_, (cs, ce)), n)| Command::PaletteGradient(cs, ce, n))
            })
            .command(
                "p/sample",
                "Sample palette colors from the active view",
                |p| p.value(Command::PaletteSample),
            )
            .command("p/sort", "Sort the palette colors", |p| {
                p.value(Command::PaletteSort)
            })
            .command("p/write", "Write the color palette to a file", |p| {
                p.then(path()).map(|(_, path)| Command::PaletteWrite(path))
            })
            .command("undo", "Undo the last edit", |p| p.value(Command::Undo))
            .command("redo", "Redo the last edit", |p| p.value(Command::Redo))
            .command("f/add", "Add a blank frame to the active view", |p| {
                p.value(Command::FrameAdd)
            })
            .command("f/clone", "Clone a frame and add it to the view", |p| {
                p.then(optional(integer::<i32>().label("<index>")))
                    .map(|(_, index)| Command::FrameClone(index.unwrap_or(-1)))
            })
            .command(
                "f/remove",
                "Remove the last frame from the active view",
                |p| p.value(Command::FrameRemove),
            )
            .command("f/prev", "Navigate to previous frame", |p| {
                p.value(Command::FramePrev)
            })
            .command("f/next", "Navigate to next frame", |p| {
                p.value(Command::FrameNext)
            })
            .command("f/resize", "Resize the active view frame(s)", |p| {
                p.then(tuple::<u32>(
                    natural().label("<width>"),
                    natural().label("<height>"),
                ))
                .map(|(_, (w, h))| Command::FrameResize(w, h))
            })
            .command("tool", "Switch tool", |p| {
                p.then(word().label("pan/brush/sampler/.."))
                    .try_map(|(_, t)| match t.as_str() {
                        "pan" => Ok(Command::Tool(Tool::Pan(PanState::default()))),
                        "brush" => Ok(Command::Tool(Tool::Brush)),
                        "sampler" => Ok(Command::Tool(Tool::Sampler)),
                        _ => Err(format!("unknown tool {:?}", t)),
                    })
            })
            .command("tool/prev", "Switch to previous tool", |p| {
                p.value(Command::ToolPrev)
            })
            .command("swap", "Swap foreground and background colors", |p| {
                p.value(Command::SwapColors)
            })
            .command("reset!", "Reset all settings to defaults", |p| {
                p.value(Command::Reset)
            })
            .command("selection/move", "Move selection", |p| {
                p.then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|(_, (x, y))| Command::SelectionMove(x, y))
            })
            .command("selection/resize", "Resize selection", |p| {
                p.then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|(_, (x, y))| Command::SelectionResize(x, y))
            })
            .command("selection/yank", "Yank/copy selection content", |p| {
                p.value(Command::SelectionYank)
            })
            .command("selection/cut", "Cut selection content", |p| {
                p.value(Command::SelectionCut)
            })
            .command("selection/paste", "Paste into selection", |p| {
                p.value(Command::SelectionPaste)
            })
            .command("selection/expand", "Expand selection", |p| {
                p.value(Command::SelectionExpand)
            })
            .command("selection/erase", "Erase selection contents", |p| {
                p.value(Command::SelectionErase)
            })
            .command("selection/offset", "Offset selection bounds", |p| {
                p.then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|(_, (x, y))| Command::SelectionOffset(x, y))
            })
            .command("selection/jump", "Translate selection by one frame", |p| {
                p.then(param::<Direction>())
                    .map(|(_, dir)| Command::SelectionJump(dir))
            })
            .command("selection/fill", "Fill selection with color", |p| {
                p.then(optional(color()))
                    .map(|(_, rgba)| Command::SelectionFill(rgba))
            })
            .command("selection/flip", "Flip selection", |p| {
                p.then(word().label("x/y"))
                    .try_map(|(_, t)| match t.as_str() {
                        "x" => Ok(Command::SelectionFlip(Axis::Horizontal)),
                        "y" => Ok(Command::SelectionFlip(Axis::Vertical)),
                        _ => Err(format!("unknown axis {:?}, must be 'x' or 'y'", t)),
                    })
            })
            .command("paint/color", "Paint color", |p| {
                p.then(color())
                    .skip(whitespace())
                    .then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|((_, rgba), (x, y))| Command::PaintColor(rgba, x, y))
            })
            .command("paint/line", "Draw a line between two points", |p| {
                p.then(color())
                    .skip(whitespace())
                    .then(tuple::<i32>(
                        integer().label("<x1>"),
                        integer().label("<y1>"),
                    ))
                    .skip(whitespace())
                    .then(tuple::<i32>(
                        integer().label("<x2>"),
                        integer().label("<y2>"),
                    ))
                    .map(|(((_, color), (x1, y1)), (x2, y2))| {
                        Command::PaintLine(color, x1, y1, x2, y2)
                    })
            })
            .command("paint/fg", "Paint foreground color", |p| {
                p.then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|(_, (x, y))| Command::PaintForeground(x, y))
            })
            .command("paint/bg", "Paint background color", |p| {
                p.then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|(_, (x, y))| Command::PaintBackground(x, y))
            })
            .command("paint/p", "Paint palette color", |p| {
                p.then(natural::<usize>())
                    .skip(whitespace())
                    .then(tuple::<i32>(integer().label("<x>"), integer().label("<y>")))
                    .map(|((_, i), (x, y))| Command::PaintPalette(i, x, y))
            })
    }
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct CommandCompleter {
    file_completer: FileCompleter,
}

impl CommandCompleter {
    fn new<P: AsRef<Path>>(cwd: P, exts: &[&str]) -> Self {
        Self {
            file_completer: FileCompleter::new(cwd, exts),
        }
    }
}

impl autocomplete::Completer for CommandCompleter {
    type Options = ();

    fn complete(&self, input: &str, _opts: ()) -> Vec<String> {
        let p = Commands::default().parser();

        match p.parse(input) {
            Ok((cmd, _)) => match cmd {
                Command::ChangeDir(path) | Command::WriteFrames(path) => self.complete_path(
                    path.as_ref(),
                    input,
                    FileCompleterOpts { directories: true },
                ),
                Command::Source(path) | Command::Write(path) => {
                    self.complete_path(path.as_ref(), input, Default::default())
                }
                Command::SetPluginDir(path) => {
                    self.complete_path(Some(&path), input, Default::default())
                }
                Command::Edit(paths) | Command::EditFrames(paths) => {
                    self.complete_path(paths.last(), input, Default::default())
                }
                _ => vec![],
            },
            Err(_) => vec![],
        }
    }
}

impl CommandCompleter {
    fn complete_path(
        &self,
        path: Option<&String>,
        input: &str,
        opts: FileCompleterOpts,
    ) -> Vec<String> {
        use crate::autocomplete::Completer;

        let empty = "".to_owned();
        let path = path.unwrap_or(&empty);

        // If there's whitespace between the path and the cursor, don't complete the path.
        // Instead, complete as if the input was empty.
        match input.chars().next_back() {
            Some(c) if c.is_whitespace() => self.file_completer.complete("", opts),
            _ => self.file_completer.complete(path, opts),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{fs, fs::File};

    #[test]
    fn test_command_completer() {
        let tmp = tempfile::tempdir().unwrap();

        for file_name in &["one.png", "two.png", "three.png"] {
            let path = tmp.path().join(file_name);
            File::create(path).unwrap();
        }

        let cc = CommandCompleter::new(tmp.path(), &["png"]);
        let mut auto = Autocomplete::new(cc);

        assert_eq!(auto.next(":e |", 3), Some(("one.png".to_owned(), 3..3)));
        auto.invalidate();
        assert_eq!(
            auto.next(":e |one.png", 3),
            Some(("one.png".to_owned(), 3..3))
        );

        auto.invalidate();
        assert_eq!(
            auto.next(":e one.png | two.png", 11),
            Some(("one.png".to_owned(), 11..11))
        );
        assert_eq!(
            auto.next(":e one.png one.png| two.png", 20),
            Some(("three.png".to_owned(), 11..18))
        );
        assert_eq!(
            auto.next(":e one.png three.png| two.png", 18),
            Some(("two.png".to_owned(), 11..20))
        );

        fs::create_dir(tmp.path().join("assets")).unwrap();
        for file_name in &["four.png", "five.png", "six.png"] {
            let path = tmp.path().join("assets").join(file_name);
            File::create(path).unwrap();
        }

        auto.invalidate();
        assert_eq!(
            auto.next(":e assets/|", 10),
            Some(("five.png".to_owned(), 10..10))
        );
    }

    #[test]
    fn test_command_line() {
        let tmp = tempfile::tempdir().unwrap();

        fs::create_dir(tmp.path().join("assets")).unwrap();
        for file_name in &["one.png", "two.png", "three.png"] {
            let path = tmp.path().join(file_name);
            File::create(path).unwrap();
        }
        for file_name in &["four.png", "five.png"] {
            let path = tmp.path().join("assets").join(file_name);
            File::create(path).unwrap();
        }

        let mut cli = CommandLine::new(tmp.path(), &tmp.path().join(".history"), &["png"]);

        cli.puts(":e one");
        cli.completion_next();
        assert_eq!(cli.input(), ":e one.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e one.png");

        cli.clear();
        cli.puts(":e ");
        cli.completion_next();
        assert_eq!(cli.input(), ":e assets");

        cli.completion_next();
        assert_eq!(cli.input(), ":e one.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e three.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e two.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e assets");

        cli.putc('/');
        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/four.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png");

        cli.putc(' ');
        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png assets");
        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png one.png");

        cli.putc(' ');
        cli.putc('t');
        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png one.png three.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png one.png two.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png one.png three.png");

        for _ in 0..10 {
            cli.cursor_backward();
        }
        cli.putc(' ');
        cli.putc('o');
        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png one.png one.png three.png");

        cli.clear();
        cli.puts(":e assets");
        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/");

        cli.clear();
        cli.puts(":e asset");

        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/");

        cli.completion_next();
        assert_eq!(cli.input(), ":e assets/five.png");
    }

    #[test]
    fn test_command_line_change_dir() {
        let tmp = tempfile::tempdir().unwrap();

        fs::create_dir(tmp.path().join("assets")).unwrap();
        for file_name in &["four.png", "five.png"] {
            let path = tmp.path().join("assets").join(file_name);
            File::create(path).unwrap();
        }

        let mut cli = CommandLine::new(tmp.path(), Path::new("/dev/null"), &["png"]);

        cli.set_cwd(tmp.path().join("assets/").as_path());
        cli.puts(":e ");

        cli.completion_next();
        assert_eq!(cli.input(), ":e five.png");

        cli.completion_next();
        assert_eq!(cli.input(), ":e four.png");
    }

    #[test]
    fn test_command_line_cd() {
        let tmp = tempfile::tempdir().unwrap();

        fs::create_dir(tmp.path().join("assets")).unwrap();
        fs::create_dir(tmp.path().join("assets").join("1")).unwrap();
        fs::create_dir(tmp.path().join("assets").join("2")).unwrap();
        File::create(tmp.path().join("assets").join("rx.png")).unwrap();

        let mut cli = CommandLine::new(tmp.path(), Path::new("/dev/null"), &["png"]);

        cli.clear();
        cli.puts(":cd assets/");

        cli.completion_next();
        assert_eq!(cli.input(), ":cd assets/1");

        cli.completion_next();
        assert_eq!(cli.input(), ":cd assets/2");

        cli.completion_next();
        assert_eq!(cli.input(), ":cd assets/1");
    }

    #[test]
    fn test_command_line_cursor() {
        let mut cli = CommandLine::new("/dev/null", "/dev/null", &[]);

        cli.puts(":echo");
        cli.delc();
        assert_eq!(cli.input(), ":ech");
        cli.delc();
        assert_eq!(cli.input(), ":ec");
        cli.delc();
        assert_eq!(cli.input(), ":e");
        cli.delc();
        assert_eq!(cli.input(), ":");
        cli.delc();
        assert_eq!(cli.input(), "");

        cli.clear();
        cli.puts(":e");

        assert_eq!(cli.peek(), None);
        cli.cursor_backward();

        assert_eq!(cli.peek(), Some('e'));
        cli.cursor_backward();

        assert_eq!(cli.peek(), Some('e'));
        assert_eq!(cli.peek_back(), Some(':'));

        cli.delc();
        assert_eq!(cli.input(), ":e");

        cli.clear();
        cli.puts(":echo");

        assert_eq!(cli.peek(), None);
        cli.cursor_back();

        assert_eq!(cli.peek(), Some('e'));
        assert_eq!(cli.peek_back(), Some(':'));

        cli.cursor_front();
        assert_eq!(cli.peek(), None);
        assert_eq!(cli.peek_back(), Some('o'));
    }

    #[test]
    fn test_parser() {
        let p = Commands::default().line_parser();

        assert_eq!(
            p.parse(":set foo = value"),
            Ok((
                Command::Set("foo".to_owned(), Value::Ident(String::from("value"))),
                ""
            ))
        );
        assert_eq!(
            p.parse(":set scale = 1.0"),
            Ok((Command::Set("scale".to_owned(), Value::F64(1.0)), ""))
        );
        assert_eq!(
            p.parse(":set foo=value"),
            Ok((
                Command::Set("foo".to_owned(), Value::Ident(String::from("value"))),
                ""
            ))
        );
        assert_eq!(
            p.parse(":set foo"),
            Ok((Command::Set("foo".to_owned(), Value::Bool(true)), ""))
        );

        assert_eq!(
            param::<platform::Key>()
                .parse("<hello>")
                .unwrap_err()
                .0
                .to_string(),
            "unknown key <hello>"
        );

        assert_eq!(p.parse(":").unwrap(), (Command::Noop, ""));
    }

    #[test]
    fn test_echo_command() {
        let p = Commands::default().line_parser();

        p.parse(":echo 42").unwrap();
        p.parse(":echo \"hello.\"").unwrap();
        p.parse(":echo \"\"").unwrap();
    }

    #[test]
    fn test_zoom_command() {
        let p = Commands::default().line_parser();

        assert!(p.parse(":zoom -").is_ok());
        assert!(p.parse(":zoom 3.0").is_ok());
        assert!(p.parse(":zoom -1.0").is_err());
    }

    #[test]
    fn test_vfill_commands() {
        let p = Commands::default().line_parser();

        p.parse(":v/fill").unwrap();
        p.parse(":v/fill #ff00ff").unwrap();
    }

    // #[test]
    // fn test_unknown_command() {
    //     let p = Commands::default().line_parser();

    //     let (err, rest) = p.parse(":fnord").unwrap_err();
    //     assert_eq!(rest, "fnord");
    //     assert_eq!(err.to_string(), "unknown command: fnord");

    //     let (err, rest) = p.parse(":mode fnord").unwrap_err();
    //     assert_eq!(rest, "fnord");
    //     assert_eq!(err.to_string(), "unknown mode: fnord");
    // }

    #[test]
    fn test_keymapping_parser() {
        let p = string("map")
            .skip(whitespace())
            .then(KeyMapping::parser(BindingTier::General));

        let (_, rest) = p.parse("map <tab> :q! {:q}").unwrap();
        assert_eq!(rest, "");

        let (_, rest) = p
            .parse("map <tab> :brush/set erase {:brush/unset erase}")
            .unwrap();
        assert_eq!(rest, "");

        let (_, rest) = p.parse("map <ctrl> :tool sampler {:tool/prev}").unwrap();
        assert_eq!(rest, "");
    }

    #[test]
    fn test_map_script_parser() {
        let p = Commands::default().line_parser();
        let (cmd, rest) = p.parse(r#":map/script "visual (rotation)" <tab> :v/prev"#).unwrap();
        assert_eq!(rest, "");
        match &cmd {
            Command::Map(km) => {
                assert_eq!(km.input, Input::Key(platform::Key::Tab));
                assert_eq!(km.repeats, Some(false));
                assert!(matches!(
                    &km.tier,
                    BindingTier::Script(name) if name.as_str() == "visual (rotation)"
                ));
            }
            _ => panic!("expected Command::Map, got {:?}", cmd),
        }
    }

    #[test]
    fn test_map_script_repeats_parser() {
        let p = Commands::default().line_parser();
        let (cmd, rest) = p
            .parse(r#":map/script repeats "visual (rotation)" <tab> :v/prev"#)
            .unwrap();
        assert_eq!(rest, "");
        match &cmd {
            Command::Map(km) => {
                assert_eq!(km.input, Input::Key(platform::Key::Tab));
                assert_eq!(km.repeats, Some(true));
                assert!(matches!(
                    &km.tier,
                    BindingTier::Script(name) if name.as_str() == "visual (rotation)"
                ));
            }
            _ => panic!("expected Command::Map, got {:?}", cmd),
        }
    }

    #[test]
    fn tes_value_parser() {
        let p = Value::parser();

        assert_eq!(p.parse("1.0 2.0").unwrap(), (Value::F32Tuple(1.0, 2.0), ""));
        assert_eq!(p.parse("1.0").unwrap(), (Value::F64(1.0), ""));
        assert_eq!(p.parse("1").unwrap(), (Value::U32(1), ""));
        assert_eq!(p.parse("1 2").unwrap(), (Value::U32Tuple(1, 2), ""));
        assert_eq!(p.parse("on").unwrap(), (Value::Bool(true), ""));
        assert_eq!(p.parse("off").unwrap(), (Value::Bool(false), ""));
        assert_eq!(
            p.parse("#ff00ff").unwrap(),
            (Value::Rgba8(Rgba8::new(0xff, 0x0, 0xff, 0xff)), "")
        );
    }

    #[test]
    fn test_parser_errors() {
        let p = Commands::default().line_parser();

        let (err, _) = p
            .parse(":map <ctrl> :tool sampler {:tool/prev")
            .unwrap_err();
        assert_eq!(err.to_string(), "unclosed '{' delimiter".to_string());

        let (err, _) = p.parse(":map <ctrl> :tool sampler :tool/prev").unwrap_err();
        assert_eq!(
            err.to_string(),
            "extraneous input found: :tool/prev".to_string()
        );

        assert!(p.parse(":map repeats <tab> :q!").is_err());
        assert!(p.parse(":map/normal repeats <tab> :q!").is_err());
        assert!(p.parse(":map/visual repeats <tab> :q!").is_err());
        assert!(p.parse(":map/help repeats <tab> :q!").is_err());
    }
}

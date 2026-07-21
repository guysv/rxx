use rx::execution::{DigestMode, ExecutionMode};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_derive::Deserialize;

#[macro_use]
extern crate lazy_static;

#[derive(Deserialize)]
struct Config {
    window: WindowConfig,
    assets: AssetConfig,
    #[serde(default)]
    plugins: PluginConfig,
}

#[derive(Deserialize)]
struct WindowConfig {
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct AssetConfig {
    glyphs: PathBuf,
}

#[derive(Deserialize, Default)]
struct PluginConfig {
    load: Vec<String>,
}

lazy_static! {
    /// Windowed (glfw) runs spawn real windows and graphics contexts,
    /// which are not thread-safe, so they are serialized. Headless runs
    /// are surface-less and can run in parallel.
    pub static ref MUTEX: Mutex<()> = Mutex::new(());
}

#[test]
fn simple() {
    test("simple");
}

#[test]
fn resize() {
    test("resize");
}

#[test]
fn visual() {
    test("visual");
}

#[test]
fn palette() {
    test("palette");
}

#[test]
fn snapshots() {
    test("snapshots");
}

#[test]
fn saving() {
    test("saving");
}

#[test]
fn views() {
    test("views");
}

#[test]
fn yank_paste() {
    test("yank-paste");
}

#[test]
fn brush_basic() {
    test("brush-basic");
}

#[test]
fn brush_advanced() {
    test("brush-advanced");
}

#[test]
fn frames() {
    test("frames");
}

#[test]
fn animation_delay() {
    test("animation-delay");
}

#[test]
fn layers() {
    test("layers");
}

#[test]
fn layers_active() {
    test("layers-active");
}

#[test]
fn layers_attrs() {
    test("layers-attrs");
}

#[test]
fn ui() {
    test("ui");
}

#[test]
fn grid() {
    test("grid");
}

#[test]
fn flood() {
    test("flood");
}

#[test]
fn plugin_load() {
    test("plugin-load");
}

#[test]
fn mode_vis() {
    plugin_test("mode-vis");
}

#[test]
#[ignore = "layer-status UX is WIP (3d card stack, click-to-toggle); digest not stable yet"]
fn layer_status() {
    plugin_test("layer-status");
}

#[test]
fn easymetric() {
    plugin_test("easymetric");
}

#[test]
fn cmd_script() {
    test("cmd-script");
}

#[test]
fn script_mode() {
    test("script-mode");
}

#[test]
fn script_bind() {
    test("script-bind");
}

#[test]
fn script_settings() {
    test("script-settings");
}

#[test]
fn script_hooks() {
    test("script-hooks");
}

#[test]
fn script_view() {
    test("script-view");
}

#[test]
fn script_gpu() {
    test("script-gpu");
}

#[test]
fn script_render() {
    test("script-render");
}

#[test]
fn script_multitex() {
    test("script-multitex");
}

#[test]
fn script_params() {
    test("script-params");
}

#[test]
fn script_export() {
    let png = Path::new("/tmp/rx-script-export.png");
    fs::remove_file(png).ok();
    test("script-export");

    // The replay's :ex/save must have written the composited 8x8 PNG
    // (the digest asserts the message line; this asserts the file).
    let data = fs::read(png).expect("the export replay writes the png");
    assert_eq!(&data[..8], b"\x89PNG\r\n\x1a\n", "png signature");
    assert_eq!(&data[16..24], &[0, 0, 0, 8, 0, 0, 0, 8], "8x8 ihdr");
}

#[test]
fn selection_outline() {
    plugin_test("selection-outline");
}

#[test]
fn rotate_scale() {
    plugin_test("rotate-scale");
}

#[test]
fn source() {
    test("source");
}

#[test]
fn mouse() {
    test("mouse");
}

#[test]
fn visual_mouse() {
    test("visual-mouse");
}

#[test]
fn organize_views() {
    test("organize-views");
}

////////////////////////////////////////////////////////////////////////////////

fn test(name: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name);
    if let Err(e) = run(name, &path) {
        panic!("test '{}' failed with: {}", name, e);
    }
}

/// Sample-plugin e2e tests live with the plugin they exercise, in
/// `plugins/<name>/test/` (test.toml, test.rx, test.events, test.digest,
/// and optionally a driver `*.rune` script). The plugins to load are
/// declared in test.toml's `[plugins]` section.
fn plugin_test(name: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("plugins")
        .join(name)
        .join("test");
    if let Err(e) = run(name, &path) {
        panic!("test '{}' failed with: {}", name, e);
    }
}

fn run(name: &str, path: &Path) -> io::Result<()> {
    // The `saving` test writes and re-opens this file; make sure it's
    // not there when the test runs. Scoped to `saving` so concurrent
    // tests don't delete it mid-run.
    if name == "saving" {
        fs::remove_file("/tmp/rx.png").ok();
    }

    // Replay derives the .events/.digest names from the directory name,
    // so the config and source files follow the same convention.
    let stem: PathBuf = path
        .file_name()
        .map(PathBuf::from)
        .expect("test path has a directory name");
    let cfg: Config = {
        let path = path.join(&stem).with_extension("toml");
        let cfg = fs::read_to_string(&path)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {}", path.display(), e)))?;
        toml::from_str(&cfg)?
    };
    let glyphs = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(&cfg.assets.glyphs))
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {}", path.display(), e)))?;

    let glyphs = glyphs.as_slice();

    // Tests that declare `[plugins] load = [...]` get those sample plugins
    // staged into a scratch directory, together with any driver `*.rune`
    // scripts sitting next to the test files. Other tests load whatever is
    // in their `plugins/` subdirectory, if present.
    let plugin_dir = if cfg.plugins.load.is_empty() {
        Some(path.join("plugins")).filter(|p| p.is_dir())
    } else {
        Some(stage_plugins(name, path, &cfg.plugins.load)?)
    };

    let options = rx::Options {
        resizable: false,
        headless: true,
        source: Some(path.join(&stem).with_extension("rx")),
        plugin_dir,
        plugin_dirs: Vec::new(),
        width: cfg.window.width,
        height: cfg.window.height,
        exec: ExecutionMode::Replay(path.to_path_buf(), DigestMode::Verify),
        glyphs,
        debug: false,
    };

    {
        let _guard = if cfg!(feature = "glfw") {
            Some(MUTEX.lock())
        } else {
            None
        };
        rx::init::<&str>(&[], options)
    }
}

/// Assemble a plugin directory for a test under `target/plugin-tests/<name>`:
/// a copy of each listed sample plugin's package, plus any `*.rune` driver
/// scripts from the test directory itself. Plugin packages are flat, so only
/// top-level files are copied — which also keeps their `test/` dirs out.
fn stage_plugins(name: &str, test_dir: &Path, load: &[String]) -> io::Result<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stage = manifest.join("target").join("plugin-tests").join(name);

    fs::remove_dir_all(&stage).ok();
    fs::create_dir_all(&stage)?;

    for plugin in load {
        let src = manifest.join("plugins").join(plugin);
        let dst = stage.join(plugin);
        fs::create_dir(&dst)?;
        for entry in fs::read_dir(&src)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                fs::copy(entry.path(), dst.join(entry.file_name()))?;
            }
        }
    }
    for entry in fs::read_dir(test_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("rune") {
            fs::copy(&path, stage.join(entry.file_name()))?;
        }
    }
    Ok(stage)
}

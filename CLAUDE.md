# rx development notes

## Build & test

```sh
cargo test --no-default-features
```

is the whole verified suite (unit + replay + doc tests), headless. The
default `glfw` feature opens real windows and needs a display server
(WindowServer on macOS) — it SIGTRAPs in headless shells. Headless GPU
(wgpu) renders and digests deterministically.

## Replay tests

Two homes, one harness (`tests/main.rs`):

- `tests/<name>/` — scripting-core tests, run via `test("<name>")`.
- `plugins/<name>/test/` — each sample plugin's e2e test (file stem is
  always `test`), run via `plugin_test("<name>")`.

Each test dir holds `<stem>.toml` (window size, glyphs, plugin deps),
`<stem>.rx` (sourced at startup, *before* plugins load — plugin
commands can be `map`ped there, but plugin settings can only be `:set`
from events), `<stem>.events` (input recording), `<stem>.digest` (one
SeaHash64 hex line per *unique* rendered frame). The `.events` and
`.digest` filenames derive from the directory name (`src/execution.rs`).

Plugin deps are declared in the toml:

```toml
[plugins]
load = ["rotate-scale"]
```

The harness stages each listed plugin's top-level files, plus any
`*.rune` driver scripts sitting next to the test files, into
`target/plugin-tests/<name>/`. Tests without a `load` list use their
own `plugins/` fixture subdir, if present.

## Re-recording a digest

When rendering legitimately changes:

```sh
scripts/record-digest tests/<name>
scripts/record-digest plugins/<name>/test
```

The script stages plugins like the harness, replays headless with
`--record-digests`, and handles the one big gotcha: **rx does not exit
when the replay ends** — it drops into the normal main loop, so the
process must be killed after the `replaying: digest saved` log line.

A digest locks in whatever rendered, *including fixture errors*, so
the script refuses runs whose log contains `error` or `disabled` and
prints the unique-frame count — a healthy test has tens of frames;
2–3 means the interesting states never rendered. Re-run the test
afterwards to confirm: `cargo test --no-default-features --test main <name>`.

## Hand-writing .events

Plain text, one event per line: `FFFFF DDDDDDD <event>` (5-digit
frame, 7-digit milliseconds, both monotonic). Event kinds
(`src/event.rs`): `keyboard/input <key> pressed|released`,
`char/received 'c'`, `cursor/moved <x> <y>`,
`mouse/input [right|middle] pressed|released`, `paste '...'`.
Commands are typed as `char/received ':'` followed by one char per
line and a `<return>` press/release.

## Pitfalls that cost us digests

- Session/view/drawing coordinates and pixel rows are y-down from the
  top-left. Layer n starts at sheet row n*fh; layer order is independent
  of row direction. Reversed drags can still reverse selection corners;
  normalize before using a selection as a draw target.
- The `update` hook fires every frame — message booleans/edges from
  it, never counters, or every frame becomes a unique digest line.
- `switch_mode` fires on edges, including an initial `normal` edge and
  `command`-mode entries; derive expected sequences from a record run.
- `view_pixels` reads the *recorded snapshot*: a readback right after
  a mutation in the same command sees stale pixels. Split mutate and
  read into separate commands a few frames apart.
- A left click over the view paints. Park the cursor off-view (the
  default 128×128 view is centered in the 1278×718 test window; e.g.
  100,100 is safe) before mouse-hook experiments.

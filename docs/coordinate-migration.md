# Top-left coordinate migration

The core uses y-down coordinates throughout. Pixel (x, y) is stored at
`pixels[y * width + x]`. Rectangles use half-open bounds; the first row of a
cropped buffer is the clamped rectangle's top row. Session coordinates apply
interface scale to window logical coordinates and floor to pixels. View
coordinates additionally remove pan, view offset, and zoom. Drawing uses the
same origin; positive matrix rotations appear clockwise.

Layer n occupies sheet rows `n * fh .. (n + 1) * fh`. Layer 0 still composites
first; storage direction does not determine compositing order. Resizing frame
height preserves each layer's top-left contents and moves subsequent strips
to their new row offsets. GPU textures, snapshots, uploads, saved images and
script byte buffers all keep top-first rows. There is no legacy coordinate
mode or script-only inversion.

The migration updates core pixel indexing and cropping, mouse coordinates,
flood fill, sprite UV mapping, renderer and script projections, layer routing,
blits, resize, palette indexing, UI placement and directional key bindings.
LUT sampler/paint operations and layer-status
hit testing use the same coordinates. The website API documentation, replay
fixtures, generated demo packages, videos and posters are migrated too.

## Verification

Verified across the workspace:
- `rxx`: 109 unit tests, 38 standard replays, 18 doc tests, and the
  normally ignored layer-status replay run explicitly.
- `rxx-master-preview-fix` and `rxx-security-fixes`: 105 unit tests,
  38 standard replays, 18 doc tests, and layer-status run explicitly, each.
- Windowed/default-feature builds checked with `cargo check`.
- External LUT replay and semantic assertions; asset-generator fidelity.
- Six refreshed demo captures, including regenerated website videos/posters.


Run `cargo test --no-default-features` in each rxx worktree. GPU tests require
Metal access on macOS. The layer-status replay is historically ignored for UX
work; run it explicitly with
`cargo test --no-default-features --test main layer_status -- --ignored`.

New tests cover:
- Asymmetric multi-layer coordinates, cropped rows and coordinate bounds.
- Snapshot crop -> texture upload -> GPU drawing at an offset destination,
  with four different corner colors and an otherwise transparent target.
- Actual core GPU paint, layer growth, active-layer drawing and sampling,
  yank/paste, undo/redo, PNG saving, frame-height resize and resize undo.

The external LUT plugin has `python3 scripts/test.py`, which checks its replay
and an ordered sequence of semantic observations for lookup, sampling, paint,
erase, undo, export and shading. `ms-re/verify-fidelity.sh` checks generated
asset fidelity; its existing top-first pixel data needs no coordinate conversion.

The replay recorder now isolates plugin loading to match the harness and
cleans the saving fixture's output. The cmd-script, frames and resize fixtures
intentionally exercise invalid commands; their expected error messages were
reviewed when regenerating digests.

Historical design plans in this directory describe the implementation at their
original milestones. Coordinate descriptions in those plans are superseded by
this document.

## Layout regression follow-up

The initial migration missed bottom-anchored plugin text. Mode-vis,
rotate-scale and LUT now place their HUDs relative to the
session height, including glyph height. LUT's canvas label stays below its
view. The sampler sprite retains its hotspot, and j/k navigate down/up
through the now downward-stacked views. These are layout anchors in the
shared y-down coordinate system, not an API compatibility conversion.

A regression test dispatches the actual mode-vis plugin at two session
heights and checks the emitted glyphs retain their bottom margin. Pixel
coordinate tests and regenerated digests alone did not catch this omission.

A second position audit restored the help body's glyph-top margin. The
website mode-vis tutorial and test HUD snippets now use bottom anchors too.
A help-layout test checks the first glyph row in both populated columns.
Explicit-coordinate GPU fixtures remain explicit; LUT's documented one-pixel
upward preview offset and centered cursor offsets are intentional.

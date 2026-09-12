# Miniview

A draggable preview panel drawn above the workspace and other plugins' previews.
It is a self-contained Rune plugin: `miniview.rune` plus the editor's bundled
`glyphs.png` atlas. No custom shader is needed.

## Use

Run `:miniview` to show the panel and enter target selection. Move the pointer
over a view frame or preview; a pink outline shows the target. Click to attach
it. The panel's **Target** button enters the same tool to change an attachment.
**Escape** or **right-click** cancels selection and keeps the previous target.

- Click a frame in a view's horizontal sheet to pin that frame. Miniview shows
  its visible layers composited with their opacity, and follows artwork edits.
- Click the editor's animation preview to follow its current frame and playback.

Drag the title bar to move the panel. Drag its bottom-right grip to resize it.
The panel stays in screen coordinates as the workspace pans and zooms. Artwork
is fitted with its aspect ratio preserved, using integer enlargement when it
fits, nearest-neighbor sampling, and a black background behind transparent pixels. The panel uses a one-pixel
outline and compact text controls, matching the editor’s wireframe UI. Its title
shows the pinned frame number or `live` for animation playback.

Click **x** or run `:miniview/off` to hide the panel. `:miniview/show` restores it
with its target and position. `:miniview/select` is an alias for `:miniview`.
`miniview/width` and `miniview/height` settings also control its size (defaults
280 by 240). The panel starts hidden and does not open until commanded.

Closing the source view detaches the target. If a pinned frame is removed, the display falls back to the last remaining frame. Reloading
plugins resets the panel and attachment. This version provides one panel.

## Installation

The editor loads this folder automatically when run from the rxx repository.
For an external installation, copy the entire folder into a configured plugin
directory, or pass its path with `--plugin-dir /path/to/miniview`. It requires the
current sprite, `overlay`, `capture_mouse`, and view metadata APIs documented in
[the GPU API guide](../../docs/gpu-api.md).

## Verification

`cargo test --no-default-features miniview` runs the GPU interaction test and
recorded UI replay. The GPU test checks target selection without painting,
dragging, fixed versus live frames, live edits, cancellation, hiding, and drawing
above a later-rendering plugin. The replay covers the Target button and resizing.

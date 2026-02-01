# wgpu/mod.rs vs glmod.rs — Comparison & Bug Checklist

This document compares the old luminance/GL renderer (`/tmp/glmod.rs`) with the new wgpu renderer (`src/wgpu/mod.rs`) to pinpoint missing or differing behavior that can cause bugs.

---

## 1. Snapshot / dirty view recording (high impact)

**GL:** At the end of `frame()`, snapshots are taken only when the **view is dirty** (`v.is_dirty()`), and both paint and resize are handled:

- `ViewState::Dirty(_)` + `is_resized()` → `record_view_resized(v_data.layer.pixels(), extent)`
- `ViewState::Dirty(_)` and not resized → `record_view_painted(v_data.layer.pixels())`

**wgpu:** Snapshot is taken only when `!self.final_batch.vertices().is_empty()` (`needs_snapshot`):

- Always calls `record_view_painted(pixels)` only.
- Never checks `v.is_dirty()` or `ViewState`.
- Never calls `record_view_resized` (resize is never recorded into history).

**Bugs this can cause:**

- History/undo can be wrong: paints may be recorded when the view wasn’t actually dirty, or not recorded when it was.
- Resize is never pushed to the snapshot layer, so undo/redo and extent may diverge from GL.

**Fix direction:** Mirror GL: only record when the active view `v.is_dirty()`, and branch on `is_resized()` to call either `record_view_resized(..., extent)` or `record_view_painted(...)`.

---

## 2. Paste: render vs upload (behavior / correctness)

**GL:** Paste is **rendered** into the layer:

- `ViewOp::Paste(dst)` builds a sprite batch and pushes a **tessellation** into `paste_outputs`.
- In the “Render to view final buffer” pass, those tessellations are drawn with the **paste texture** (and blending), so the paste appears as a quad on the layer.

**wgpu:** Paste is **uploaded** as raw pixels:

- `ViewOp::Paste(dst)` uses `view_data.layer.upload_part(&self.queue, [dst_x, dst_y], [paste_w, paste_h], &bytes)`.
- No rendering of the paste texture; no blending.

**Bugs this can cause:**

- Coordinate mix-ups (e.g. Y flip) in `upload_part` will misplace the paste.
- Yank in wgpu reads from GPU with a custom Y loop (`tex_y = layer_h - 1 - (src.y1 as u32 + (src_h - 1 - y))`); if the layer’s logical vs texture Y convention doesn’t match the rest of the app, yank/paste can be wrong or inconsistent with GL.

**Fix direction:** Either keep upload but align coordinate and row order with the rest of the pipeline (and document), or reintroduce a “paste texture → layer” draw pass to match GL and get blending.

---

## 3. ViewOps: Blit and Flip (missing)

**GL:**

- `ViewOp::Blit(src, dst)`: reads from **view’s CPU snapshot** (`v.layer.get_snapshot_rect(&src)`), then `view.layer.fb.color_slot().upload_part_raw(...)` to the GPU framebuffer.
- `ViewOp::Flip(src, dir)`: reads from snapshot, flips in CPU (vertical/horizontal), re-uploads to the paste texture (or equivalent).

**wgpu:**

- `ViewOp::Blit` → `// TODO: Implement blit`
- `ViewOp::Flip` → `// TODO: Implement flip`

**Bugs:** Any use of blit or flip in the app will do nothing or behave incorrectly in wgpu.

**Fix direction:** Implement Blit by getting pixels from `view.resource.layer.get_snapshot_rect(&src)` and then uploading to the layer target (equivalent of GL’s upload_part_raw). Implement Flip by reading the rect from the layer (or snapshot), flipping in CPU, then uploading back (or to paste) to match GL.

---

## 4. Blending mode (Constant) (medium impact)

**GL:** When rendering **final** brush strokes to the view layer, if `blending == Blending::Constant`, it uses a different render state:

- `src: Factor::One`, `dst: Factor::Zero` (replace, no alpha blending).

**wgpu:** No branch on `self.blending`; all shape/sprite rendering uses the same pipeline (alpha blending).

**Bugs:** With `Blending::Constant`, strokes should replace the destination; in wgpu they will blend, so colors and opacity will differ from GL.

**Fix direction:** Add a second shape (or pipeline) for “replace” blending and use it when `self.blending == Blending::Constant` for the final brush pass.

---

## 5. Help overlay (missing)

**GL:** When `session.mode == session::Mode::Help`:

- Builds `help_tess`: a shape tessellation for the help window and a sprite tessellation for help text.
- In the “Render to screen framebuffer” pass, draws both (shape then sprite with font texture).

**wgpu:** No handling of `session::Mode::Help`; no help tessellations, no help pass.

**Bugs:** Help screen never appears.

**Fix direction:** In the screen pass, if `session.mode == session::Mode::Help`, build and draw equivalent geometry (shape + text sprites) using the same layout as GL.

---

## 6. Debug / overlay text (missing)

**GL:** After drawing the screen texture to the back buffer, if `session.settings["debug"].is_set() || !execution.is_normal()`:

- Uses a **different ortho**: `Matrix4::ortho(screen_w, screen_h, Origin::BottomLeft)` (vs normal TopLeft).
- Draws `overlay_tess` with the font texture (FPS / debug text).

**wgpu:** No use of `overlay_tess`; no debug/overlay text pass; no BottomLeft ortho.

**Bugs:** Debug overlay and non-normal execution overlay are not shown.

**Fix direction:** After the screen quad, if debug or !is_normal(), run a pass that uses BottomLeft ortho and draws overlay text (reuse or mirror GL’s overlay_tess construction and draw).

---

## 7. View animations and composites (stubbed)

**GL:**

- `update_view_animations(session)`: for each view, builds `draw::draw_view_animation(s, v)` batch and stores tessellation in `vd.anim_tess`. In the screen pass, when `session.settings["animation"].is_set()`, draws each view’s `anim_tess` (layer texture + translation).
- `update_view_composites(session)`: builds `draw::draw_view_composites(s, v)` and stores in `vd.layer_tess` (used conceptually for composite overlay; in the snippet the composite tess isn’t drawn in the same way as anim — confirm in full GL flow).

**wgpu:**

- `update_view_animations` and `update_view_composites` are empty: `// TODO: Implement view animations` / `// TODO: Implement view composites`.

**Bugs:** Animated views (multi-frame) and any composite overlays will not render as in GL.

**Fix direction:** Port animation/composite batch building from draw.rs and the GL frame pass: create vertex buffers from those batches and draw them in the same order and with the same transforms as GL (layer texture, translation, etc.).

---

## 8. Staging paste to view (order and clearing)

**GL:** For the **active** view:

1. “Render to view staging buffer”: clear, then draw staging brush tess + **paste tess** (paste texture) with view ortho.
2. “Render to view final buffer”: **no clear** (load existing), then draw final brush tess + **paste_outputs** (tessellations from `ViewOp::Paste`).

So staging shows draft + current paste preview; final shows committed strokes + pastes.

**wgpu:**

- Staging: clear, then draw staging shapes only (no paste texture draw to staging).
- Final: load, then draw final shapes only (no paste_outputs; Paste is done via upload_part only).

So:

- Paste **preview** in the staging area is missing.
- Paste **result** is done by upload only; no blended paste quad. If you later add a paste draw pass, it should match GL’s order (staging vs final).

**Fix direction:** If you want staging to show the paste texture like GL, add a draw of the paste texture (or paste quads) in the staging pass. Keep final pass semantics aligned with GL (either upload only or add a paste draw step).

---

## 9. Cursor scale (pixel ratio)

**GL:** Cursor uniform uses:

- `platform::pixel_ratio(*scale_factor)` for the scale (and `ui_scale * pixel_ratio`).

**wgpu:** Cursor uses `self.scale` only:

- `scale: self.scale as f32` (and no pixel_ratio).

**Bugs:** On HiDPI, cursor size/position may be wrong compared to GL.

**Fix direction:** Use the same scale as GL: e.g. `(ui_scale * platform::pixel_ratio(scale_factor)) as f32` (and pass scale_factor where needed).

---

## 10. Readback / async (correctness and ordering)

**wgpu:** After `encoder.finish()` and `output.present()`:

- If `needs_snapshot`, it calls `read_texture_pixels(&view_data.layer.target.texture, ...)` and then `record_view_painted(pixels)`.
- `read_texture_pixels` submits a copy encoder and then `device.poll(Maintain::Wait)` + `map_async` to read back.

**Potential bugs:**

- The GPU work for the frame is submitted in `submit(encoder.finish())`; the readback uses a **separate** encoder and submit. Without a formal dependency, the readback might run before the render pass has finished writing to the layer (undefined behavior or wrong pixels). In GL, `get_raw_texels()` is done after the pipeline, so ordering is implicit.
- Snapshot is taken even when the view isn’t dirty (see §1).

**Fix direction:** Ensure the readback copy is submitted **after** the same frame’s render (e.g. single encoder: render passes then copy_texture_to_buffer, then one submit). Only run snapshot when the view is dirty and use the right record (paint vs resize).

---

## 11. Yank coordinate and row order

**wgpu ViewOp::Yank:** Builds `paste_pixels` with:

- `tex_y = layer_h - 1 - (src.y1 as u32 + (src_h - 1 - y))` and row iteration `for y in 0..src_h`.

This is a specific Y-flip and origin convention. If the rest of the stack (view coords, layer upload, GL comparison) uses a different convention, yank will be flipped or shifted.

**Fix direction:** Align with view/session coordinate system (e.g. Origin::TopLeft vs BottomLeft) and with how the layer texture is filled; add a test that compares a yank/paste cycle to GL or to a reference.

---

## 12. Screen size and ortho

**GL:** Screen framebuffer size comes from `self.screen_fb.size()` (logical size after scale). Ortho uses that and `Origin::TopLeft`.

**wgpu:** Screen size is `self.screen_target.size` and ortho uses `ortho_wgpu(..., Origin::TopLeft)` (with Y flip for clip space). Same idea, but any mismatch in when/how `screen_target` is resized (e.g. handle_session_scale_changed) will cause wrong layout; worth checking that logical size and scale factor are applied the same way as GL.

---

## Summary table

| Area                     | GL behavior                    | wgpu behavior              | Risk / action                          |
|--------------------------|--------------------------------|----------------------------|----------------------------------------|
| Snapshot / dirty         | Only when dirty, paint+resize  | When final_batch non-empty, paint only | High – fix recording logic             |
| Paste                    | Render quads to layer          | upload_part only           | Medium – coordinates + optional draw   |
| Blit                     | Implemented                    | TODO                       | High if used – implement               |
| Flip                     | Implemented                    | TODO                       | High if used – implement               |
| Blending::Constant       | Replace blending in final pass| Not used                   | Medium – add replace blend path        |
| Help overlay             | Drawn                          | Not drawn                  | Medium – add help pass                 |
| Debug overlay            | Drawn (BottomLeft ortho)       | Not drawn                  | Low–medium – add overlay pass          |
| View animations          | Updated and drawn             | TODO                       | Medium – port animation/composite     |
| Staging paste            | Paste texture drawn to staging| Not drawn                  | Low–medium – optional staging paste   |
| Cursor scale             | pixel_ratio(scale_factor)     | scale only                 | Medium – use pixel_ratio               |
| Readback ordering        | After pipeline                 | Separate submit            | Medium – same encoder, then snapshot   |
| Yank Y convention        | N/A (GPU readback)            | Custom Y formula           | Medium – verify vs rest of pipeline    |

Recommended order to fix for maximum stability: (1) snapshot/dirty and readback ordering, (2) Blit/Flip if the app uses them, (3) Blending::Constant, (4) cursor scale and yank/paste coordinates, (5) help and debug overlay, (6) animations and composites.

---

## Fixes applied

### §1 Snapshot / dirty view recording

**Fix:** Snapshot recording now mirrors GL behavior.

- **When to record:** Snapshot is taken only when the active view is dirty (`v.is_dirty()`) **and** either something was painted this frame (`needs_snapshot`) or the view was resized (`v.is_resized()`). So resize-only frames are recorded too.
- **What to call:** If the view was resized (`v.is_resized()`), call `record_view_resized(pixels, v.extent())` so `ViewResource.extent` and the snapshot match the new size. Otherwise call `record_view_painted(pixels)`.

**Effect:** After `:f/resize 8 8`, the first paint no longer panics: the extent used for the snapshot matches the read-back pixel buffer size (e.g. 8×8), and undo/redo and extent stay in sync with the view.

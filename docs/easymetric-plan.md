> Historical milestone notes: coordinate conventions here are superseded by
> [the top-left core migration](coordinate-migration.md). Current coordinates and
> pixel rows are y-down; layer n begins at sheet row n*fh.

# Easymetric as a plugin (v2): plan for phases 40+

This document covers the work after `docs/layers-plan.md` (P35–P39, layers
in core, complete through P38): a reimplementation of Oroshibu's
**EasyMetric** as `plugins/easymetric/`. It takes **P40+** (the parked v1
claimed P35–P41; layers landed in those slots, so this renumbers).

**This is a v2 and a near-total rewrite of the parked plan, because the v1
was built on a wrong premise.** v1 described a *polygon modeler* — build
vertex/face geometry, project it dimetrically, bake — with a command set
of `em/add`/`extrude`/`nudge`/`select`. That is not what EasyMetric is.
EasyMetric is a **reconstruct-from-three-views tool**: the artist paints
what an object looks like from the front, side, and top, and the plugin
carves the 3D form out of the intersection of those silhouettes and shows
it rotating. There is no geometry to edit — you paint. Everything below
follows from that correction. (The abandoned v1 lives in `git stash` —
as the record of the wrong turn.)

As before there is **no reference branch and no published source** — the
spec is observed behavior from the announcement thread (bsky
`oroshibu.bsky.social/post/3lqrxxuh6as2g`: three painted views, a rotating
3D reconstruction, per-layer detail, outlines, dithering) plus a
firsthand reading of the demos. This is a redesign from behavior, not a
port, so where rx's shape suggests a better mechanism (live view
textures, real layers, the multi-view workspace) we take rx's mechanism.

**Out of scope** (for the whole plan): geometry persistence of any kind
(the mesh — here, the carved volume — lives and dies with plugin state;
the painted *views* persist for free, they're just pixels), model export
(.obj, Minecraft), and frame-based object animation (deferred to an open
question — the rotation *is* the animation for now).

## Status: P40–P44 complete — Part L shipped

All of Part L (the reconstructor, `plugins/easymetric/`) is built, green
(`cargo test --no-default-features`), and exercised by one evolving replay
(`plugins/easymetric/test/`, 243 unique frames): layout + carve + preview
(P40), free rotation (P41), triplanar texture / per-part colour / shading
+ dither (P42, landing the `view_layer_pixels` core read), bake (P43), and
outlines (P44). The one planned host addition, `view_layer_pixels`,
shipped in P42; nothing else in the render path needed a core change, as
the GPU-architecture section predicted.

P40 landed as `plugins/easymetric/` (`easymetric.rune` + `easymetric.wgsl`
+ `test/`, replay `plugins/easymetric/test/`, 11 unique frames). The
quadrant layout, `em/box`, the `draw`-hook guides + HUD, the GPU space
carve (fragment raycast over the raw sheet via `view_bind_group`), the
fixed canonical-dimetric preview pane (`render` hook), and `em/on`/`off`/
`status`/`scan` all work; the carve math is verified exact by the probes
(`voxels 7680` for a 20×16×24 box, `scan front 480 side 384 top 320`).
Findings and deviations:

- **The params uniform is sized to the floats passed, padded to a vec4
  multiple — *not* a fixed 64-float buffer.** Passing 12 floats yields a
  48-byte buffer, so the WGSL must declare `array<vec4<f32>, 3>` to match
  (a `, 4>` declaration is a "buffer is bound with size 48 where the
  precedent (`array<vec4, 1>` ← 1 or 4 floats) reads as "fixed budget";
  it is not.
- **Flat-shaded blob, not triplanar.** P40 shades each hit by face normal
  (a fixed light) over a flat base — colour/texture is explicitly P42.
  The DDA already yields the exact face normal, so triplanar/shade slot in
  without re-architecting the march.
- **`em/status`/`em/scan` read the active layer via `view_pixels`** (the
  single-layer degenerate carve, recomputed CPU-side). The GPU carve
  unions all strips from day one, but per-*strip* CPU probes need
  `view_layer_pixels` (lands in P42); the P40 test is single-layer, where
  the two coincide.
- **Ambient toggle confirmed.** No script mode, no mouse capture: the
  silhouettes are painted with the ordinary brush (here, a test-driver
  `t/fill` recording exact rects), and `em/on`/`off` are the whole
  lifecycle. The em commands stay reachable from the command line; the
  replay maps single keys only to stay compact.

P41 (free rotation) followed in the same plugin (replay extended to 16
unique frames). `em/rotate dyaw dpitch` (repeating, for held-key spin) and
`em/reset` mutate the camera angle; the raycast already took yaw/pitch as
params, so generalising the pose was free — no shader change beyond what
P40 shipped. Findings:

- **Rotation was a no-op on the render path.** P40 passed the canonical
  yaw/pitch through the param buffer and built the camera basis in WGSL
  from them, so P41 is purely the two commands + an angle readout; the
  "the raycast generalizes to an arbitrary rotation matrix" work was
  already done. Pitch is clamped to ±89° so `cross(world_up, forward)`
  never collapses at the poles.
- **The same-content-hash dedup is real and load-bearing.** Re-rendering
  every frame (no edge-triggering) is digest-clean: the 120-frame static
  stretch after the last rotate adds zero unique frames, and `em/reset`
  reproduces the canonical pose *bit-identically* (its frame dedups with
  the pre-rotation cube). Edge-triggering stays a pure CPU optimisation,
  unbuilt — nothing in the digest depends on it.

P42 (texture, colour & shading) landed with its one core addition,
`view_layer_pixels(id, layer, rect)` (the replay grew to a two-layer
carve, 150 unique frames). The shader gained triplanar sampling, a
per-layer attr texture (group 2), the topmost-strip arbitration, and the
shade/dither knobs; the plugin gained `em/textured`, `em/paint`, and the
`easymetric/textured`/`shade`/`dither` settings. Verified end to end: a
textured left-half box (a distinct colour per silhouette proves the
per-face select — top green, front teal, side purple) beside a flat
orange right-half colored part, the two unioned with the topmost strip
winning the seam. Findings:

- **`view_layer_pixels` is `view_pixels` lifted by `layer*fh`.**
  `view_pixels` reaches only the *bottom* strip (`layer_bounds()` clamps
  to rows `0..fh`); the new call clamps the same way then offsets the
  sheet rect up one strip, reusing `get_snapshot_rect`'s y-flip. It needs
  a *recorded snapshot at the grown extent* — a bare unit-test session
  after `LayerAdd` has none (the strip read returns `None`), so the unit
  test covers single-layer + range/clamp guards and the replay covers the
  multi-strip offset on real snapshots (the `scan` probe reads the
  *non-active* strip, the thing the API exists for).
- **`Bytes`, not a Rune array, for `upload`.** Building the attr buffer
  with a plain `[]` fails `upload`'s `Bytes` arg (`Expected Bytes, found
  Vec`); `use std::bytes::Bytes;` + `Bytes::new()`/`.push()` is the build
  so there was no from-scratch precedent).
- **The triplanar colour round-trips for free**, as the GPU-architecture
  section predicted: `textureLoad` sRGB-decodes the sheet, the sRGB target
  re-encodes on write, so a painted byte survives the carve unchanged —
  (attr `rgb`) round-trips the same way.
- **A colored part only appears once all three of its silhouettes are
  painted** — the carve is their AND, so during the replay's incremental
  paint of layer 1 the preview shows just layer 0 until layer 1's third
  fill lands, then the colored half pops in. Expected, not a bug; worth
  knowing when reading the digest's frame progression.
- **Digit keys mis-bind command args; letters are the safe map keys.**
  `map 1 :t/fill ...` ran `t/fill` with mangled args (a "usage" error)
  while every letter map worked, so the silhouette fills are *typed*
  unique-frame count (each typed char is a distinct command-line frame).

P43 (bake) landed: `em/bake [frame]` stamps the render into an output
view (replay grew to 209 frames; the undo-roundtrip probe is the
assertion — `213,213,213,255` present, `0,0,0,0` after `:undo`). Findings:

- **The carve must read a *fixed* canvas, not the active view.** Baking
  creates/switches to an output view; if the carve followed the active
  view it would reconstruct from the blank output (verified: empty pane,
  blank stamp). The fix is structural — `em/on` records the canvas id and
  every hook (`shade`/`render`/`draw`/`status`/`scan`/`textured`/`paint`)
  reads *that* view, so the active view is free to move. This is the right
  model anyway: EasyMetric reconstructs from one canvas, not "wherever the
  cursor is."
- **View creation replaces a `NoFile` scratch-pad active view**
  (`add_view`, `session.rs:1709`). The replay's canvas is exactly that, so
  `em/bake`'s on-demand `:e` *destroyed the canvas* — the silhouettes
  vanished and the carve found nothing. Two consequences: the test names
  the canvas (`:e em-canvas.png`) up front so it is no longer a scratch
  pad, and `em/bake` guards (if the canvas is gone after creating the
  output, it bails loudly instead of baking blank). A real user with a
  scratch-pad canvas hits the same wall — name/`:w` first, or set
  `easymetric/output` to a pre-made view.
- **`em/probe` reads the output by id** (`view_layer_pixels`), so the
  undo step needs no view switch: `em/bake` leaves the output active, so a
  bare `:undo` reverts the bake while `em/probe` still reads it by id.
  redraw idiom) — the pane is a render target (row 0 = top), the view dst
  is y-up, and `load` composes over existing pixels so spin + bake across
  frames accretes a turnaround. Same-encoder ordering lets the bake pass
  sample the pane the raycast pass wrote earlier in the same `shade`.

P44 (outlines) completed Part L: a third pipeline `fs_outline` edge-detects
SCRATCH_A (the raycast output) into SCRATCH_B (`pane_b`), which the render
hook and the bake now read — so the outline lands in both for free.
Findings:

- **The outline pass always runs; `off` is a passthrough.** Rather than
  branch the pipeline, SCRATCH_A → SCRATCH_B every frame and style 0
  returns the pixel unchanged, so render/bake unconditionally read
  SCRATCH_B and the outline-or-not decision is one shader uniform. Inner
  border (outline rides opaque edge pixels), so the sprite never grows.
- **Colour+coverage edge-detect catches the seams**, as predicted: a
  transparent neighbour is the silhouette, a >0.12 RGB delta to an opaque
  neighbour is an interior seam (verified by the darkened-teal edge texels
  the shade style produces at the face boundaries). MRT face-id stays the
  unbuilt optional fix; nothing in the suite needs it.
- **`set_setting` consumes its String argument.** `em/outline` built the
  message *after* `set_setting(next)` and read the moved value — "Cannot
  read, value is M-…", which disables the plugin. Format the message
  before the set (the same move-discipline rotate-scale's header warns
  about for `Option`/`unwrap`).
- **Same param-buffer size across entry points.** All three fragment
  passes share `params: array<vec4<f32>, 3>`, so the outline pass must
  also bind a 12-float (3×vec4) params buffer — a shorter one is the P40
  "bound with size N where the shader expects 64/48" validation error.

## The feature, in one loop

Partition the canvas into three orthographic quadrants → paint the
silhouettes with the **normal brush** → the plugin carves the voxel union
→ spin it freely in a live preview → **bake the chosen angle to a pixel
sprite.**

### 1. Layout — the splits are the bounding box

The canvas carries two split lines: a vertical split at column `X` and a
horizontal split at row `Z` (measuring from the shared corner). They
partition the canvas into four quadrants of *deliberately unequal* size,
because the splits encode the object's bounding box `X × Y × Z`:

```
+----------+--------+
|  TOP     |        |   top   = X × Y   (looking down −Z)
|  (X×Y)   | unused |
+----------+--------+
|  FRONT   |  SIDE  |   front = X × Z   (looking down +Y)
|  (X×Z)   |  (Y×Z) |   side  = Y × Z   (looking down +X)
+----------+--------+
```

The views are **axis-aligned and edge-sharing**, the classic drafting
layout: front and top share width `X`, front and side share height `Z`.
The top-right quadrant is the pictorial slot of real multiview drawing —
here it stays empty. The two split coordinates are `X` and `Y`
horizontally / `Z` and `Y` vertically, so setting them *is* setting the
model dimensions: `em/box X Y Z` moves the splits, the `draw` hook paints
the guide lines, and resizing the box is resizing the splits.

### 2. Carve — space intersection

A voxel `(x, y, z)` in the `X × Y × Z` grid is **solid** iff it is opaque
in all three silhouettes that project onto it:

```
solid(x,y,z) = front[x, z] ∧ side[y, z] ∧ top[x, y]
```

This is visual-hull space carving. It captures far more than a convex
blob (any shape whose three axis silhouettes agree) but cannot represent
concavities hidden from all three axes or detached internal voids — a
known and acceptable limit, the same one the technique has on paper.

### 3. Layers are parts, unioned — and they ride the raw sheet

Each rx layer carries its own three silhouettes, hence its own carved
voxel set; the final scene is the **union of every layer's carve**. You
add a horn or a tail by adding a layer (`:layer/add`) and painting its
three quadrants. This is the headline feature, and it maps onto the strips
we just built with no friction:

**the carve reads the raw stacked sheet.** `view_bind_group` exposes the
texture as-is — all layer strips stacked vertically (the layers-plan P39
raw-sheet exposure was a *limitation* (it wanted the composite); for
EasyMetric it is exactly right — one bind group hands the carve shader
**every layer's pixels at once**, indexed by strip. The shader walks
strips (layers) × the three quadrants, unions the solids, and is live by
construction: paint any silhouette on any layer and the reconstruction
updates mid-stroke. Single-layer is the degenerate case (one strip, one
part), so the existing suite's 1-layer views exercise the whole path.

When two layers fill the same voxel, **the topmost strip wins the color**
(highest layer index — matches composite order); overlap is incidental to
a union of parts, so the rule just needs to be deterministic.

### 4. Color & texture — triplanar projection mapping

The three silhouettes are not just shape — they are three **planar
projection textures**, one per axis:

- **front** view (XZ) textures every ±Y face (front/back),
- **side** view (YZ) textures every ±X face (left/right),
- **top** view (XY) textures every ±Z face (top/bottom).

This is **triplanar mapping** — texturing without UVs. On smooth meshes
triplanar must blend the three projections across the surface normal and
blurs at the seam; here every voxel face is axis-aligned, so each face has
exactly one owning view and the select is **hard, blend-free, exact**.
Paint detail into a silhouette and it lands on the matching faces; rotate
the model and each view's texture turns into sight as its faces face the
camera — "paint the object from three sides, watch it wrap the model."

Three properties make this nearly free:

- **Opacity is occupancy, RGB is texture.** One opaque silhouette pixel
  both carves the voxel solid *and* supplies its surface color.
- **Every surface face is guaranteed an opaque texel.** A voxel is solid
  only because all three projecting pixels are opaque (the carve is their
  AND), so each candidate face-texture is opaque by construction — there is
  no missing-texel / transparent-texture case, ever.
- **Texture resolution = voxel resolution**, one texel per face (the front
  silhouette is X×Z, the grid X×Y×Z). The render is flat-shaded blocks,
  crisp at every snap angle, no minification.

This is why the v1 "textured mode & atlas" phase **evaporates**: no atlas,
no `em/unwrap`, no UV layout, no second view to manage — the textures *are*
the three images the artist already paints. Textured is the natural
default; colored is the special case, the exact inversion of v1.

**Colored ↔ textured is per part (per layer), with a global default**
(`easymetric/textured`); `em/textured` toggles the active layer:

- **textured** — faces sample their owning view's RGB (painted detail
  shows);
- **colored** — ignore RGB, fill the part with a flat color (the captured
  foreground, re-grabbed by `em/paint`) — the blocking-out look.

So a textured body can carry a flat-colored accessory on another layer.
The flag is plugin state keyed by layer, with the same index-stability
caveat as the carve (reorder/merge/flatten permute strips; the flag does
not chase them — see open questions).

**Orientation shading is an independent knob** (`easymetric/shade`),
orthogonal to colored/textured: darken each face by its normal so form
reads, with an ordered-dither threshold (`easymetric/dither`, the feature
list's "dithering"). You can light a textured model or leave a colored one
flat — they do not gate each other.

### 5. Spin — free rotation, drawn in the render hook

The reconstruction renders by a **full-quad fragment raycast** (a render
pipeline, not compute — see GPU architecture): per output pixel, march a
ray into the `X × Y × Z` grid at the current rotation, keep the first
solid voxel, triplanar-sample its owning view. `O(pixels × depth × layers)`
is nothing at pixel-art sizes. Rotation is **free and continuous** — a
command (`em/rotate dyaw dpitch`, or a held-key spin) drives the angle;
the render re-runs edge-triggered on geometry edits *and* on angle change.

The preview is that render output (a scratch script texture) **drawn as a
screen-space quad in the `render` hook** — ephemeral, digest-clean, and
placeable anywhere, including the top-right animation-preview corner, by
coordinate math. It is *not* a view written each frame: drawing the
preview through `begin_view_pass` would record an undoable edit every
frame (undo spam, digest churn). rx's render hook lands in the screen
animation preview is" needs **no host addition** — this dissolves what v1
filed as an open question.

Because free-spin voxel rendering is nearest-neighbor at arbitrary angles,
it is **crunchy, not clean**: edges are guaranteed exact 2:1/vertical only
at the axis and dimetric snap angles, not in general. The v1 "no cleanup
pass" promise does not survive the move to free rotation, and this plan
does not make it.

### 6. Bake — commit the render to a sprite

`em/bake` stamps the render output (the scratch texture) into an **output
frame** (a `begin_view_pass` blit into a chosen frame of an output view —
never the source quadrants, which would eat the input). The payoff is a
2D pixel-art sprite of the posed model. The natural extension falls out of
free spin: **spin + bake across frames builds a direction sprite-sheet** —
rotate, bake into frame 0; rotate, bake into frame 1; the frame strip
fills with a turnaround. Bake composes over existing pixels (load, not
clear), so a baked model stamps over prior art like a brush stroke.

## GPU architecture

plugin, the load-bearing precedent: all of EasyMetric's GPU work is

### Render, not compute (a settled decision)

The carve+render must read the **live** source sheet every frame
(`encoder.view_bind_group(id)` — unrecorded, mid-stroke paint included).
That bind group is a render-pipeline texture group; compute cannot take a
live view (`create_compute_bind_group` requires owned `ScriptTexture`
inputs, and a view's texture is sRGB with no raw view and empty
`view_formats`, `wgpu/mod.rs:96-117`). Feeding a view to compute *is* an
addable core change (`view_formats: [unorm]` + a `raw_view` + a
`view_compute_bind_group`), but it buys nothing here:

- per-output-pixel raycasting is a wash between a full-quad fragment shader
  and a compute dispatch;
- compute's multi-output and transparent-write edges don't apply — the
  compute API is single-output too (`create_compute_pipeline`, one storage
  binding), and the raycast draws once over a `clear` target so the
  blend-can't-write-transparency pitfall never bites;
- a raw compute read of an sRGB view reintroduces the P28 encode/round
  wrinkle, where the fragment path samples sRGB-decoded and re-encodes on
  write — colors round-trip for free.

So render is the natural fit, not a contortion to avoid touching core.
Compute stays unused; revisit only if a future pass is genuinely
gather/scatter-bound.

### Pipelines (created in `init`)

Three render pipelines (`rx.create_render_pipeline(shader, vs, fs,
textures)`), one WGSL module (`easymetric.wgsl`):

| Pipeline | Entry | Textures | Role |
|---|---|---|---|
| `raycast` | `vs_quad` / `fs_raycast` | 2 | full-target quad; per fragment march the grid at the current rotation, carve from the live sheet, triplanar-sample the hit face, shade, write `rgb = color, a = coverage` |
| `outline` | `vs_quad` / `fs_outline` | 1 | edge-detect the raycast output, composite the outline |

### Bind groups & texture buffers (the resource graph)

```
SOURCE sheet (a real view, not owned)       ATTR texture (owned, ~nlayers×1 rgba8)
 read live: view_bind_group(id)              per-layer: rgb = colored fill, a = textured flag
        │ group 1                            re-uploaded only on a layer mode/color change
        │                                          │ group 2
        ▼                                          ▼
   raycast ── group 0: transform_params ───────────┘
        │     ortho(target) in the mat4 slot; box X/Y/Z,
        │     rotation (packed 3×3), quadrant rects, shade/
        │     dither/outline flags — ≤64 floats, vec4-packed
        ▼
   SCRATCH_A (owned, preview-sized rgba8)   ← begin_render_pass(.., "clear")
        │ group 1
        ▼
   outline → SCRATCH_B (or ping-pong A)
        │
   ┌────┴───────────────────────────┐
 render hook                      em/bake
 show → screen quad               show → begin_view_pass(OUTPUT) + touch_view
 (preview pane, ephemeral)        (one recorded, undoable edit)
```

- **Source sheet** — the live carve input; all layer strips × three
  quadrants in one texture. Not owned.
- **Attr texture** — per-layer textured flag + colored fill. *Must* be a
  texture, not params: with `fh = 128` a sheet holds ~64 strips, and 64
  colors + 64 flags overflow the 64-float param budget. Globals ride in
  params; per-layer data rides here.
- **Scratch A/B** — owned script textures sized to the preview pane; the
  render targets. `texture_pixels` reads them back for `em/status` probes
  (synchronous — split the readback a frame from the mutating pass).
- **Output view** — a real view (created via `run_builtin`); the *only*
  thing baked into, and the only recorded edit EasyMetric makes.

### Passes per frame (`shade`) + the preview (`render`)

- `shade`: (1) if a layer's mode/color changed, re-upload the attr
  texture; (2) `begin_render_pass(SCRATCH_A, "clear")` → `raycast`; (3)
  `begin_render_pass(SCRATCH_B, "clear")` → `outline`. Edge-triggered:
  skipped when neither the sheet nor the angle changed (CPU economy;
  same-content frames already hash identically).
- `render`: `show` SCRATCH_B as a screen-space quad at the preview pane
  position — the preview, ephemeral and digest-clean.
- `em/bake`: `begin_view_pass(OUTPUT, "load")` → `show` SCRATCH_B, then
  `rx.touch_view(output)` — the single recorded, undoable edit.

### The one core addition (and two optional ones)

`view_layer_pixels(id, layer, rect)` (specced in layers P39) — a CPU
per-layer read for test probes and `em/status`/`em/scan`, since
`view_pixels` only reaches the active layer. Built in P42. **Nothing else
in the render path needs a core change.**

- *Optional, P44 robustness:* multi-target output (MRT render, or
  multi-output compute) would let `raycast` emit a face/part-id attachment
  beside color, making interior-seam outlines exact. But with shading on,
  adjacent faces differ in brightness and triplanar gives them
  different-view colors, so single-target color+coverage edge-detect
  catches essentially every seam — build that first, add MRT only if a
  real model shows misses.
- *Optional, carve-cost escape hatch:* if `O(pixels × depth × layers)`
  ever stutters, add an edit-triggered pass baking occupancy (+ owning
  layer) into a flattened-3D **volume** script texture (Z-slices tiled in
  2D, ≤8192); the per-frame raycast then marches one volume texel per step
  and samples a quadrant only at the hit. Occupancy is rotation-
  independent, so the volume rebuilds on paint, not on spin. Measure first.

## Host API: this plan is *not* zero-additions

v1's headline was "zero host API additions." v2 retracts that — and that
retraction is the point. The layers plan deferred the per-layer read API
(`view_layer_pixels`) and per-layer opacity exposure "until a plugin needs
one." **EasyMetric is that plugin**, so this plan is what drives the
layers script tier (P39) from "read side partially landed" to done. The
render path needs exactly **one** addition (the optional ones — MRT,
volume — are catalogued under GPU architecture):

- **`view_layer_pixels(id, layer, rect)`.** The GPU carve rides the
  raw-sheet bind group and needs nothing, but *introspection and tests*
  need to read a specific (non-active) layer on the CPU to assert the
  per-layer carve is correct — `view_pixels` only reaches the active
  layer (routed). The layers plan already specced this exact 4-arg call;
  EasyMetric is the customer that lands it. A Part-L finding, fixed
  host-side with a test in P42.

The preview-into-the-host-slot question v1 carried is **closed**, not
deferred: the `render` hook draws screen-space, so the preview pane needs
tier (render pipelines, params, staging/view passes, `run_builtin` view
creation) plus the layers read tier (`ViewInfo.nlayers`/`active_layer`,
`layer_visibility`, `view_bind_group` raw-sheet access).

## Command & key UX

The mesh-editing command set from v1 is **deleted in full** — there is no
geometry to add, extrude, nudge, or select. What remains is small, and the
biggest structural call is the first one:

### Not a capturing script mode

and owns the mouse. EasyMetric must do the **opposite**: painting the
silhouettes is ordinary normal-mode painting with every existing brush
tool. So EasyMetric is an **ambient toggle**, not a mode:

- `em/on` / `em/off` flip plugin state. While on: the `draw` hook paints
  the quadrant guides + HUD, the preview pane re-renders, and a handful of
  commands are live. While off: dormant, painting unaffected either way.
- No mouse capture, no `switch_mode` lifecycle. The toggle is the
  lifecycle; `em/off` (or plugin unload / `view_removed`) is cleanup.

This is the plan's key divergence from every prior sample, and it exists
because the input *is* painting — the plugin observes and reconstructs, it
does not intercept.

### The command surface

| Command | Form | Notes |
|---|---|---|
| `em/on` / `em/off` | toggle | enter/leave the reconstruction overlay |
| `em/box x y z` | explicit | set bounding box = move the splits |
| `em/rotate dyaw dpitch` | explicit (keys bind literals) | spin the preview; repeatable, held-key spins |
| `em/reset` | bare | snap rotation back to the canonical dimetric pose |
| `em/textured` | bare = toggle active part | colored ↔ textured for the active layer |
| `em/paint color?` | bare = fg | set a colored part's flat fill |
| `em/bake [frame]` | bare = current frame | stamp the render into the output view |
| `em/outline str?` | `off`/`solid`/`shade`, bare = cycle | outline style |
| `em/status` / `em/scan` | bare | counts (voxels, parts/layers, angle) / per-layer probe |

Layers are managed with rx's existing `:layer/*` — each layer is a part,
no easymetric-specific layer commands. The command line is unreachable
while painting, so every argument-taking command has a bare/cursor form
(the v1 dual-form rule survives); keys bind the bare forms. `em/status`
and `em/scan` are written in **P40** and used by every later phase's
replay — they plus the message line and HUD are the entire feedback and
digest-probe channel.

### Settings

`:set`-able outside the overlay and from events in tests:

- `easymetric/output` (int, bake target view id; 0 = create on demand)
- `easymetric/textured` (int, 0/1 = default new parts colored/textured)
- `easymetric/outline` (string: `off`/`solid`/`shade`)
- `easymetric/dither` (int, 0 = off, 1–4 = Bayer level)
- `easymetric/shade` (int, 0/1 = orientation shading off/on)

Command toggles (`em/outline`) write back through `rx.set_setting` so the
setting is always the truth.

## Phase plan

Plugin layout: `plugins/easymetric/easymetric.rune` (entry) +
`easymetric.wgsl` (the three pipelines) + `test/` (the harness stems are
`test.{toml,rx,events,digest}`, per `CLAUDE.md`). Each phase ends green:
`cargo test --no-default-features` passes, and each ships a replay in
`plugins/easymetric/test/` (testing policy and digest rules inherited from
`docs/rune-plan.md` unchanged). Geometry is driven by *painting* —
recorded brush events into the quadrants — with explicit commands
(`em/box`, `em/rotate n`) for the deterministic state changes.

### Part L — the reconstructor (`plugins/easymetric/`)

**P40. Layout, carve, preview.**
The quadrant layout and `em/box`, the `draw`-hook guides + HUD, the space
carve, and the fragment raycast at the **fixed canonical dimetric angle**
into SCRATCH_A, drawn as a preview pane in the `render` hook (the `show`
blit). The carve reads the raw sheet and unions all layer strips from the
start (single-layer is the degenerate case), so the multi-layer path
exists day one even though parts get their own polish later. `em/on`/
`em/off`, `em/status`/`em/scan`. The rendered blob is the digest surface
— this pins the multiview math and the carve before any rotation or color
work.
- Test: replay painting three silhouettes of a simple box into the
  quadrants; the preview pane shows the carved solid at the canonical
  angle; `em/status` probes voxel/part counts in the message line.

**P41. Free rotation.**
`em/rotate dyaw dpitch` (repeatable; `em/reset` to the canonical pose);
the fragment raycast generalizes to an arbitrary rotation matrix (passed
in the param buffer); render re-runs edge-triggered on angle change as
well as edits.
- Test: replay rotating a carved box through several angles — digest-
  distinct poses; a static stretch after a rotate proves the unchanged
  re-render adds no unique frames (the P34 same-content-hashes finding).

**P42. Texture, color & shading.**
Triplanar texture sampling (each face ← its owning view's RGB), the
per-part colored↔textured toggle (`em/textured`, `em/paint`, default
`easymetric/textured`), the topmost-layer arbitration rule, and the
independent orientation-shading knob (`easymetric/shade`) + dither
(`easymetric/dither`). This is where `view_layer_pixels` lands — the test
probes that assert per-layer carve and texture correctness read specific
layers on the CPU through it.
- Test: replay painting a textured silhouette (a face with detail) and a
  flat-colored part on a second layer; rotate to show the detail wrapping
  the right faces and the colored part flat; toggle `em/textured` and
  `easymetric/shade` for digest-distinct states; a `view_layer_pixels`
  probe asserts the non-active layer's silhouette.

**P43. Bake.**
`em/bake [frame]`: blit SCRATCH_B (the current render) into an output
frame via the `show` pipeline (`begin_view_pass` + `touch_view`),
composing over existing pixels (load, not clear). Spin + bake across
frames = a direction sprite-sheet.
  the stamped sprite via `view_pixels` probes a few frames later, undo
  back to pre-bake; a second variant bakes three angles into three frames
  and probes the turnaround.

**P44. Outlines.**
The `outline` fragment pass over the raycast output (SCRATCH_A): a
color+coverage edge-detect — coverage transitions give the silhouette,
color deltas give interior seams (shading + triplanar make adjacent faces
differ, so same-view same-color seams are the only miss, and MRT face-id
is the optional fix). `solid` and `shade` styles, `em/outline` cycling +
setting write-back. Outlines land in both the preview and the bake (same
SCRATCH_B, so for free) — this answers the most-liked request in the
thread.
- Test: replay cycling all three styles over the P40 box; the silhouette
  outline and interior part seams (where two layers' parts meet) are the
  digest's job.

## Open questions

- **Carve cost at scale.** `O(pixels × depth × layers)` raycast is free at
  pixel-art sizes; `em/status` posts the voxel/part counts, profile if a
  real model ever stutters (the Part-I "measure first" posture). The
  escape hatch is the edit-triggered occupancy **volume** (see GPU
  architecture); the natural ceiling is the texture limit on `X × Y × Z`
  projected sizes.
- **Color arbitration on overlap.** Topmost layer wins. If artists want
  additive/painted overlaps between parts this is the first knob to grow;
  unmotivated until someone stacks parts that genuinely fight.
- **Texture depth-smearing.** A silhouette pixel `(x,z)` is shared by every
  voxel along Y at that column, so the front texture is constant along
  depth — back/interior faces exposed by carving repeat the column's color.
  Inherent to the technique (you cannot paint depth detail from one
  silhouette); name it, don't fix it.
- **Texture seam consistency.** Where a surface turns a corner from a +Y
  face to a +X face the color jumps from front-view to side-view texels;
  continuity is the artist's job (paint matching colors at the model's
  edges, the implicit-atlas-seam discipline). A future seam-preview
  affordance could help, but the mechanism does not need it.
- **Per-part flag stability under structural ops.** The colored/textured
  flag and a colored part's fill color are keyed by layer index, which
  reorder/merge/flatten permute and drop without the flags chasing (the
  non-persistence posture shared with the carve). Acceptable while the
  model is ephemeral; a stable per-layer id is the host fix if it bites.
- **Object animation (frames).** rx frames could each hold a full
  three-view set, making frame N a keyframe of the *object* (a walk cycle),
  distinct from rotating one static object. Defer until the static case
  ships and someone asks — the rotation turnaround is the animation story
  for now, and it already uses frames (spin + bake).
- **Concavities and internal voids.** Space carving cannot represent shapes
  hidden from all three axes. Acceptable (the technique's paper limit);
  more views or per-layer carving would extend it, out of scope here.
- **Editing the splits with the mouse.** `em/box` sets the bounding box by
  command; dragging the guide lines would be nicer but needs the overlay to
  claim drag events without becoming a capturing mode. Deferred; `em/box`
  is the contract.
- **User rebinding.** Overlay commands are plugin-side bindings in normal
  mode; collisions with builtin normal-mode keys are the risk (the ambient-
  toggle cost). Default: bind only on `em/on`, restore on `em/off`; a
  binding-config mechanism is the real fix if it bites.

## Appendix: reference map

| Area | Reference |
|---|---|
| Feature spec (observed) | bsky thread `oroshibu.bsky.social/post/3lqrxxuh6as2g` (three views, rotating reconstruction, per-layer detail, colored + textured modes, outlines, dithering); demo readings |
| Triplanar mapping (the texture mechanism) | standard UV-free texturing; exact here because voxel faces are axis-aligned (one owning view per face, no blend) |
| Why v2 exists | the abandoned v1 in `git stash` (polygon-modeler premise, now superseded) |
| Layers in core (lineage; read tier, routing, raw-sheet convention) | `docs/layers-plan.md` (P35–P39); read surface = `ViewInfo.nlayers`/`active_layer`, `layer_visibility`; raw sheet strip `n` = rows `n*fh..(n+1)*fh` |
| Per-layer read API this plan lands | `view_layer_pixels(id, layer, rect)` (specced in layers P39, built here in P42) |
| Why not compute | `create_compute_bind_group` needs owned `ScriptTexture` inputs (`script.rs:2061`); view textures are sRGB, no raw view (`wgpu/mod.rs:96-117`) — see GPU architecture |
| Params / dither & rotation uniforms (≤64 floats) | `create_transform_params_bind_group`, P25 |
| Bake one-renderer pattern | `plugins/rotate-scale/rotate-scale.rune` (view-pass apply, `<return>`/command commits) |
| View creation / multi-view display | `run_builtin` view creation; rx workspace shows views side by side |
| Digest rules & events format | `docs/rune-plan.md`, `CLAUDE.md` |

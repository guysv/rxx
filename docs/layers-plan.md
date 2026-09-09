> Historical milestone notes: coordinate conventions here are superseded by
> [the top-left core migration](coordinate-migration.md). Current coordinates and
> pixel rows are y-down; layer n begins at sheet row n*fh.

# Layers in core: plan for phases 35+

**layer support in the editor core**, with the script API tier on top. It
takes the P35+ numbering; the easymetric plan (`docs/easymetric-plan.md`,
revived at P40+) renumbered for the wait — and landed better for it, since
its "per-layer geometry" maps onto the real layers built here.

**Out of scope: persistence.** Frames set the precedent — rx has no
frame manifest either: a save writes the sheet PNG as-is and `:slice`
reinterprets it on load. Layers inherit exactly that posture: `:w`
writes the sheet (layers stacked vertically, frames horizontally — a
valid PNG everywhere), and reinterpreting a tall sheet on load is the
user's move, not a format's. A manifest/flatten-on-export design is
deliberately unplanned; if it ever happens it's its own document.

The design is **layers as vertical strips of the view sheet**, exactly as
frames are horizontal strips: one texture per view, sheet height grows to
`fh × nlayers`, the workspace displays the strips composited in place, and
writes route to the active layer's strip by a y-offset. The alternative —
per-layer pixel buffers (`Vec<LayerResource>`) — was considered and
rejected: it forces the undo redesign (one history coordinating N snapshot
stacks, absolute snapshot references, detach-don't-delete structural
edits, grouped multi-layer edits), a persistence format, and per-layer GPU
plumbing. The strips model dissolves all three: one buffer means one
snapshot stack, so every cross-layer operation is atomic by construction
and the existing history machinery is already correct; the saved PNG *is*
the layered document; and the renderer keeps one texture per view.

Upstream rx had layers (2020–21: `7259758` adds the module, `1ccaa57`
"work with layers instead of views", removed by `7c937a6` and deleted in
`6202914`). That arc is the reference, not the template, and its two
recorded fumbles are this plan's regression tests: undoing `LayerAdded`
dropped the resource so redo recreated a *blank* layer (the TODO in
`13485b1` admits the data loss), and layer-add initially just truncated
history instead of recording an edit (`8c9989e`). In the strips model
both bugs are unrepresentable — layer add/remove are `ViewResized` edits,
whose snapshots carry the pixels through undo/redo today.

--no-default-features` passes, each phase adds unit tests or a recorded
replay in `tests/` (policy and digest rules inherited from
`docs/rune-plan.md` unchanged), and the existing suite doubles as the
migration proof — with `nlayers = 1` every offset is zero and every quad
count collapses to today's, so **no existing digest may change in any
phase**.

## Status: P35–P38 complete; P39 deferred

P35–P38 landed, one commit each on `layers` (replays:
`tests/layers`, `tests/layers-active`, `tests/layers-attrs`).
Findings and deviations, in phase order:

- **P35**: as planned — zero digest churn; the audit's only compiler
  catch was `brush::expand`'s extent destructuring (display-space by
  design).
- **P36**: the final-pass projection is effectively **y-up** despite
  the `TopLeft` ortho origin: display shapes land in the *bottom*
  strip with no transform at all, and the planned "translate down by
  a strip" moved strokes *up* a strip (caught frame-by-frame via
  `RX_DUMP_FRAMES` before recording — the digest would have locked
  the bug in). `ViewOp::Blit` dst rects and `SetPixel` rows turned
  out to be raw GPU offsets that only worked at single-strip heights;
  they now convert y-up sheet coords explicitly, which also fixed
  `f/clone` content placement on layered sheets. The staging texture
  needed its own display-sized quad (it shared the layer quad).
- **P37**: routing moved entirely **renderer-side**, not the planned
  session-emission choke point: the final-pass transform routes all
  GPU writes (brush, fills, erases, paste stamps) and the renderer op
  arms translate the CPU reads (yank/flip/SetPixel) — one subsystem
  owns the conversion, and `Effect` payloads stay display-space.
  `v/clear` became a byte-exact strip upload (a load-op clear can't
  be scissored), which exposed an sRGB rounding artifact in the old
  pass-clear — the `flood` digest re-recorded over a ±1-byte fill
  difference. Flood fill slices the active strip from the snapshot
  (seed, bounds, and output all display-space). Replay-flow finding:
  `selection/paste` only fires in visual-*pasting* state, so
  cross-layer paste switches layers inside visual mode.
- **P38**: merge/flatten/reorder are CPU sheet rewrites **recorded
  directly into the resource** (`record_view_*` + `damaged`) — the
  undo-restore path reused for forward edits. Single-undo-step
  merge/flatten fell out with zero new ViewOps, simpler than the
  planned renderer ops. Attrs re-bake into the strip vertex buffer
  only on change; hidden strips stay as zero-opacity quads so the
  staging z-split never shifts. Test-authoring finding: mode-specific
  default bindings (`h`/`l` = frames) beat General-tier `map`s — test
  keys must dodge them.
- **P39 (script tier): read side landed incrementally as plugins
  needed it.** First the `layer-status` overlay plugin
  (`plugins/layer-status/`) drove `ViewInfo.nlayers`/`active_layer` and
  `layer_visibility(id)` (per-layer `visible`, bottom strip first). Then
  EasyMetric (`docs/easymetric-plan.md`, P42) drove the last specced read,
  **`view_layer_pixels(id, layer, rect)`** — `view_pixels` lifted by
  `layer*fh` into a chosen strip, since `view_pixels` reaches only the
  bottom strip and a plugin needed to introspect a non-active layer on the
  CPU. Current semantics: `view_pixels`/`clear_view_rect` stay
  display-space (bottom-strip reads / routed writes), `view_layer_pixels`
  reads any strip, `begin_view_pass`/`view_bind_group` see the raw sheet,
  per-layer *opacity* exposure stays unbuilt (no customer yet), and the
  (the degeneracy the suite proves).

## The design, in five mechanisms

Verified against the current code; references are the load-bearing sites.

1. **Two coordinate spaces, one conversion point.** *Display space* is
   what the user sees and points at: `fw × nframes` wide, `fh` tall —
   hit-testing (`View::rect`/`contains`, `src/view.rs:370-382`),
   `view_coords` (`src/session.rs:1413`), the brush and all its modes,
   selections, hover. *Sheet space* is the texture: `fh × nlayers` tall —
   snapshots, GPU texture, undo, save. The conversion is one translate,
   `(0, active_layer × fh)`, applied at a single choke point: effect
   emission in the session (`src/session.rs:1019-1052`). Draft shapes go
   to the staging batch untranslated (staging is display-space); final
   shapes, yank, paste, and `clear_view_rect` are translated as they are
   pushed. The renderer passes, the readback-and-snapshot flow
   (`src/wgpu/mod.rs:1921-1950`), and the history machinery are untouched.

2. **Routing has an exact precedent.** The `Multi` brush mode already
   duplicates strokes across frames by offsetting copies `(i × fw, 0)` in
   `brush::expand` (`src/brush.rs:249-256`). Layer routing is the same
   operation rotated 90° — but applied *after* the brush, at emission, so
   XSym/YSym mirror math (`src/brush.rs:234-247`) and Multi's frame fan
   stay in display space and compose with routing for free.

3. **Stacked display is more quads in the same batch.** The view is
   already drawn as one quad per frame (`draw::draw_view_composites`,
   `src/draw.rs:733-748`; buffer at `src/wgpu/mod.rs:2389-2427`). Layered
   display: one quad per *(layer, frame)*, same dst positions, src rects
   walking down the strips, drawn bottom-up in layer order. Per-layer
   opacity and dim-others ride the sprite-vertex opacity already in the
   format. The animation preview (`src/draw.rs:720-731`) generalizes
   identically: N stacked src rects per frame.

4. **Exact stroke preview falls out of quad ordering.** Drafts render to
   the separate staging texture and composite via their own quad; in
   layered display the order becomes: layers below active → staging quad
   → layers above. The in-progress stroke previews at the active layer's
   depth, correctly occluded. The staging texture stays *display-sized*
   while the layer texture grows to sheet size — the one place
   `ViewData::new` (`src/wgpu/mod.rs:417-436`) stops using a single
   (w, h) for both.

5. **Erase is already solved.** Erase strokes emit
   `Effect::ViewBlendingChanged(Constant)` (replace blend) with
   `Rgba8::TRANSPARENT` (`src/session.rs:1028-1051`, `:1992-1996`).
   Per-layer erase works the moment routing exists; alpha-blend-can't-
   write-transparency remains a *plugin* limitation only.

## Decisions already made (from the design discussions)

- **Strips over per-layer buffers** — the trade is snapshot memory
  (full-sheet snapshots scale with layer count) and a hard ceiling
  (`fh × nlayers ≤ 8192`, the texture limit) in exchange for undo,
  atomicity, and persistence staying solved. At rx scale, taken.
- **`extent.height()` becomes `fh × nlayers`; `View::height()` (display)
  stays `fh`.** Today these coincide; un-conflating every call site is
  the one real refactor knot and gets its own phase (P35).
- **Merge and flatten are single `ViewResized` edits.**
  `record_view_resized(pixels, extent)` takes the new pixel buffer
  (`src/view/resource.rs:48`), so "composite strip B into A, drop B" is
  expressible as one resize-with-pixels edit — one undo step, pixels
  preserved both directions. No grouped-edit machinery needed, ever.
- **Reorder is a physical strip swap** recorded as a paint — keeps "the
  sheet is the document" pure and undo-covered. Metadata-only ordering
  rejected (sheet layout would lie about composite order).
- **Eyedropper reads the composite**: walk strips top-down at the
  pixel, first non-transparent wins (`color_at` site,
  `src/session.rs:1223-1230`). Matches what the user sees.
- **Visibility and opacity are not undoable** (view-state, like zoom);
  pixel ops and structural ops (add/remove/reorder/merge/flatten) are.
  Stated once, enforced by which commands record edits.
- **Script compat by routing**: view-addressed script calls keep their
  display-space semantics, routed through the active layer — existing
  plugins keep "what you paint is what you read" unchanged. The raw
  sheet stays reachable (`begin_view_pass`, `view_bind_group`) with the
  strip convention documented. 1-layer views are bit-identical.

## Phase plan

### Part J — core strips

**P35. The display/sheet split.**
Pure refactor, zero behavior change, `nlayers` fixed at 1. `ViewExtent`
(`src/view.rs:50-90`) gains `nlayers`; `extent.height()` becomes
`fh * nlayers`; new helpers `layer(n) -> Rect` (strip rect, mirroring
`frame(n)` at `:80`) and `to_layer(p)` (mirroring `to_frame` at `:87`).
Audit **every** `height()` / `rect()` / extent call site into display
vs sheet buckets: `View::rect`/`contains` and all hit-testing stay
display; texture allocation, snapshot recording, `ViewState::Dirty/
Damaged(Option<ViewExtent>)` resize plumbing, and save paths go sheet.
`ViewData::new` takes separate staging (display) and layer (sheet)
sizes — equal for now. `layer_bounds()` (`src/view.rs:442`) is the
display-clamp and finally earns its name.
- Test: the entire existing suite green with **zero digest changes**
  (the degeneracy proof — this is the phase's real test); unit tests on
  extent math (strip rects, `to_layer`, the 8192 ceiling check).

**P36. Layer lifecycle & stacked display.**
`View::extend_layer`/`shrink_layer` mirroring the frame machinery
(`extend`/`shrink`/`extend_clone`, `src/view.rs:252-283`);
`Command::LayerAdd`/`LayerRemove`/`LayerDup` registered as `:layer/add`,
`:layer/remove`, `:layer/dup` (the `f/add` pattern, `src/cmd.rs:993-1006`,
handlers `src/session.rs:2773-2800`). Add/remove record as `ViewResized`
edits — undoable with pixels preserved by the existing snapshot path.
New layers are transparent; `dup` copies the active strip. `:layer/add`
refuses past the texture ceiling with a loud message (the `MAX_VIEWS`
posture). Display goes stacked: per-(layer, frame) quads bottom-up in
`draw_view_composites`, N-strip compositing in the animation preview.
No routing yet — painting still lands in strip 0 (bottom), which is the
degenerate active layer.
- Test: replay adding two layers over painted content; digest shows
  stacking; the **upstream-bug regression**: paint, `:layer/add`, paint
  layer 0, undo undo, redo redo — content identical both directions
  (the `13485b1` blank-redo failure, now impossible).

**P37. The active layer & routed writes.**
`View::active_layer`; `:layer/next`/`:layer/prev` (repeatable, the
`v/next` pattern) and `:layer/set <n>`; the status bar gains `L2/4`.
The emission choke point translates final shapes, yank, paste, and
`clear_view_rect` by `(0, active_layer * fh)`; drafts stay untranslated;
the screen pass orders quads below-active → staging → above-active.
Eyedropper walks the composite. Brush modes verified composing with
routing (Multi fans frames *within* the active layer; XSym/YSym mirror
in display space first). `SelectionJump` (`src/session.rs:3060-3073`)
needs no change — selection lives in display space and stays put across
layer switches; the stale `TODO: Test this across layers` at `:3061`
gets its test and dies.
- Test: replay on a 2-layer view — paint each layer, erase on top
  reveals bottom (the Constant-blend-through-routing proof), eyedropper
  picks the composite color, yank → `:layer/next` → paste moves pixels
  across layers, stroke preview occludes correctly under an upper
  layer, undo steps back stroke-by-stroke regardless of layer.

**P38. Layer attributes & structural ops.**
Visibility (`:layer/hide [n]`, `:layer/show [n]`, `:layer/solo`),
opacity (`:layer/opacity <f> [n]`), and a `layers/dim` setting (dim
inactive layers at composite time — the minimal-UI "where will my brush
land" affordance); all composite-time quad parameters, none recorded as
edits. Reorder (`:layer/up`/`:layer/down`): physical strip swap,
recorded paint, undoable. `:layer/merge` (composite active into the
strip below, shrink) and `:layer/flatten` — each a single
resize-with-pixels edit, one undo step.
- Test: replay cycling visibility/solo/opacity/dim — digest-distinct
  states; merge then undo restores both strips exactly; flatten of a
  3-layer view matches the composite the screen showed (probe via
  eyedropper messages).

### Part K — the script tier

**P39. The script API tier.**
`ViewInfo` grows `layers` and `active_layer` (the P27 pattern: expose
what `View` has). Routed-compat semantics made contractual and
documented: `view_pixels`/`clear_view_rect` rects stay display-space,
routed through the active layer; `begin_view_pass`/`view_bind_group`
address the raw sheet with the strip convention documented (strip n =
rows `n*fh..(n+1)*fh`, y-up rects / y-down bytes as ever). New
layer-addressed read: `view_layer_pixels(id, layer, rect)` (4 args —
inside the arity cap). Layer lifecycle reaches scripts through
`run_builtin("layer/add")` etc. — no new mutation API until a plugin
needs one. No new hooks: `update` + `ViewInfo` covers layer-switch
detection edge-wise. `docs/script-api.md` gains the layers section.
- Test: replay with a fixture plugin reading routed pixels across
  `:layer/next` switches and a sheet-space read through
  `view_layer_pixels`; the full plugin suite green on 1-layer views
  (degeneracy, again).

## Open questions

- **Composite texture for shader reads.** Plugins sampling a layered
  view via `view_bind_group` see the raw sheet, not the composite; the
  A derived, never-recorded composite texture per view (recomposited
  when dirty) is the fix — defer-by-measurement until a plugin actually
  needs to sample a layered view as-seen. Document the limitation in
  P39 meanwhile.
- **Blend modes.** The composite is currently fixed alpha-over quads.
  Modes (multiply, screen, …) would move compositing into a shader —
  deliberately out of Part J/K; the quad path doesn't foreclose it.
- **Exploded display.** The upstream spatial presentation (strips spread
  out in the workspace, hover-activates) could return as a view-mode
  toggle on top of this model — all presentation, no document change.
  Not planned until someone misses it.
- **Snapshot memory.** Full-sheet snapshots scale with `nlayers`;
  compression should eat the unchanged strips. Measure on a real
  layered session before inventing per-strip snapshots; the natural cap
  (8192/fh) bounds the worst case meanwhile.
- **Default keybindings** for `:layer/next`/`:layer/prev` — pick at
  implementation time against the default config's free keys; the
  commands are the contract, bindings are user-remappable.
- **Active layer after undo of structural edits.** Undo of
  `:layer/remove` resurrects a strip; the active index must land
  somewhere defined. Default: clamp to the nearest valid index;
  revisit only if it surprises in use.

## Appendix: reference map

| Area | Reference |
|---|---|
| Upstream layer arc (reference, not template) | `7259758` (module), `1ccaa57`, `13485b1` (blank-redo TODO), `8c9989e` (truncate hack), `7c937a6` (removal), `6202914` (module deleted) |
| Multi-mode stroke fan (routing precedent) | `src/brush.rs:249-256`; mirrors at `:234-247` |
| Effect emission choke point (routing site) | `src/session.rs:1019-1052` |
| Erase = Constant blend + transparent | `src/session.rs:1028-1051`, `:1992-1996` |
| ViewExtent (grow in P35) | `src/view.rs:50-90` |
| Frame lifecycle to mirror (P36) | `src/view.rs:252-283` (`extend`/`shrink`/`extend_clone`), `src/cmd.rs:993-1006`, `src/session.rs:2773-2800` |
| Slice precedent (if layer reinterpretation is ever wanted) | `src/view.rs:320`, `src/cmd.rs:822` |
| Per-frame display quads (stack in P36) | `src/draw.rs:733-748`, buffers `src/wgpu/mod.rs:2389-2427`; anim preview `src/draw.rs:720-731` |
| ViewData / staging texture (size split, P35) | `src/wgpu/mod.rs:408-436` |
| Staging/final passes (quad order, P37) | `src/wgpu/mod.rs:1349-1485` |
| Readback → snapshot recording (unchanged) | `src/wgpu/mod.rs:1921-1950` |
| History/resize edits (reused throughout) | `src/view/resource.rs:48-120` |
| Eyedropper (composite walk, P37) | `src/session.rs:1223-1230` |
| SelectionJump + stale layers TODO (P37) | `src/session.rs:3060-3073` |
| Hit test / display rect | `src/view.rs:370-382`, `src/session.rs:1215-1221` |
| Script touchpoints (P39) | `src/script.rs:1334` (`view_pixels`), `:656-676` (passes), `:687` (`view_bind_group`) |
| Modified-indicator via EditId (survives as-is) | `src/view/resource.rs:36-46` |

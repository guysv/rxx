# Script GPU drawing, pipelines and texture views

## Built-in sprite drawing

Use `pass.draw_sprite(source, options).unwrap()` to draw a texture without
creating a shader, pipeline, bind groups, or vertex buffer. `source` accepts a
script texture directly or any texture-view handle described below.

```rust
pub fn render(state, rx, pass) {
    pass.draw_sprite(state.icons, #{
        src: rx::rect(0.0, 0.0, 16.0, 16.0),
        dst: rx::rect(12.0, 12.0, 44.0, 44.0),
    }).unwrap();
}
```

Coordinates are pixels, with the origin at the target's top-left. The pass
supplies the target dimensions and format automatically, including screen,
artwork, staging, and script-owned targets. Sampling is nearest-neighbor with
clamp-to-edge addressing. No CPU readback is involved. The host caches pipelines
across draws and plugins; uniforms and quad buffers are allocated per draw.

| Option | Meaning / default |
| --- | --- |
| `dst` | Required destination `rx::rect(x1, y1, x2, y2)` in target pixels |
| `src` | Source rectangle in texture pixels; defaults to the whole texture |
| `transform` | Optional `rx::Mat4`, applied to destination geometry before projection; defaults to identity |
| `color` | Multiplicative sRGB tint, `rx::rgb(...)` or `rx::rgba(...)`; defaults to opaque white |
| `opacity` | Float in 0.0–1.0; defaults to 1.0 |
| `blend` | `"alpha"` (default) or `"replace"` |

Tint RGB is converted from sRGB to linear and multiplied into sampled RGB.
Output alpha is `sampled_alpha * tint_alpha * opacity`; opacity does not darken
RGB. This differs from the core renderer's color-replacement tint and from the
old layer-status shader's multiplication of opacity into RGB as well as alpha.
Transparent pixels retain destination contents with alpha blending; replacement
writes all channels, including transparent pixels. A raw source view bypasses
sRGB sampling conversion; a raw target view bypasses sRGB output conversion.

Descriptors can be reused. Unknown keys, wrong field types, non-finite geometry
or transforms, invalid opacity/blend, ended passes, and expired editor views
return `Err`. The lower-level pipeline API remains available for custom shaders,
blend equations, and other sampling or geometry requirements.

For example, draw live artwork into a separate preview texture:

```rust
pub fn shade(state, rx, encoder) {
    let source = encoder.view_layer(rx.active_view_id()).unwrap();
    let pass = encoder.begin_render_pass(state.preview.view(), #{
        load: "clear",
    }).unwrap();
    pass.draw_sprite(source, #{
        dst: rx::rect(0.0, 0.0, 128.0, 128.0),
    }).unwrap();
    pass.end();
}
```

The source artwork is the full sheet, including layer strips. Crop `src` to
select a frame or layer; this helper does not composite layers automatically.
Editor source handles acquired in `shade` may also be used in that frame's
`render` hook, but must be reacquired next frame.

For an erase, draw a transparent texture with `blend: "replace"` into a pass
opened with `load: "load"`. Call `rx.touch_view(id)` through the usual edit path
when the operation should be recorded for undo.

`draw_sprite` changes the active pipeline, bind groups 0–1, and vertex-buffer
slot 0. Before returning to custom drawing in the same pass, rebind its pipeline,
bind groups, and vertex buffer. It preserves viewport/scissor state. Sampling
from the same texture subresource as the pass target is still invalid.

Layer-status uses this API for its icons; rotate-scale uses it for ordinary
nearest-neighbor drawing; the external LUT plugin uses it for preview/cursor
sprites and paint/erase operations. Custom filter and lookup shaders remain.

## Floating overlays and mouse capture

`overlay(state, rx, pass)` draws after **all** ordinary `render` hooks, above
workspace views, built-in UI, and plugin previews. It receives the same sprite
pass API and frame-scoped editor texture handles. Overlay hooks run in plugin
load order; later overlays appear above earlier ones. Pass lifetime and GPU error
handling are the same as for `render`.

An overlay can implement `capture_mouse(state, rx, button, input) -> bool`.
Capture hooks run in reverse plugin load order, before ordinary `mouse_input`
hooks and builtin handling. Returning `true` consumes that button event. A panel
should capture both press and release for gestures that start on it, including
releases outside its bounds, but allow release of a canvas-started gesture through.
Return `false` for events the panel does not own. This hook does not intercept
cursor motion, wheel events, or keyboard input. Invalid return values/errors
are reported and disable the offending plugin.

`rx.views()` now includes `animation_preview_visible`. `rx.layer_opacity(id)`
returns bottom-to-top layer opacities, alongside `rx.layer_visibility(id)`. These
allow a preview to mirror the visible artwork composite rather than individual
storage strips. See [miniview](../plugins/miniview/README.md) for a complete panel
using these APIs and a target-selection mode.

## Custom pipelines

`create_render_pipeline(shader, options)` replaces the positional constructor
and `create_point_pipeline`. Blend behavior is required. Unknown descriptor keys
and invalid values report an error before GPU pipeline creation. Descriptors can
be reused.

```rust
let pipeline = rx.create_render_pipeline(shader, #{
    vertex: "vs_main",
    fragment: "fs_main",
    textures: 1,
    blend: "replace",
}).unwrap();
```

| Field | Values / default |
| --- | --- |
| `vertex`, `fragment` | Required WGSL entry-point names |
| `blend` | Required: `"alpha"`, `"replace"`, `"premultiplied_alpha"`, or equations below |
| `textures` | 0–3; default 0 |
| `topology` | `"triangle_list"` (default), `"triangle_strip"`, `"point_list"`, `"line_list"`, `"line_strip"` |
| `vertex_layout` | `"sprite"` (default), or `"none"` for vertex-index-generated geometry |
| `format` | `"rgba8unorm-srgb"` (default), or `"rgba8unorm"` for raw texture views |

Topology and vertex layout are independent. The former point constructor is:

```rust
rx.create_render_pipeline(shader, #{
    vertex: "vs_scatter", fragment: "fs_scatter", textures: 1,
    topology: "point_list", vertex_layout: "none", blend: "replace",
})
```

Separate color and alpha equations are supported. For example, destination-out
on **premultiplied** contents:

```rust
blend: #{
    color: #{ src: "zero", dst: "one_minus_src_alpha", op: "add" },
    alpha: #{ src: "zero", dst: "one_minus_src_alpha", op: "add" },
}
```

Factors: `zero`, `one`, `src`, `one_minus_src`, `src_alpha`,
`one_minus_src_alpha`, `dst`, `one_minus_dst`, `dst_alpha`,
`one_minus_dst_alpha`, `src_alpha_saturated`. Operations: `add`, `subtract`,
`reverse_subtract`, `min`, `max`. wgpu validates unsupported combinations.

This is a focused API, not a complete wgpu binding: group 0 remains the
transform/params layout; groups 1–3 remain texture + nearest sampler layouts.
There is one color attachment, all RGBA channels are writable, and there are
no custom vertex-buffer layouts, depth attachments, or multisampling settings.

## Common texture-view handles

- `texture.view()` returns a persistent sRGB `ScriptTextureView`.
- `texture.raw_view()` returns a persistent linear rgba8unorm view of the same
  storage. Select the matching pipeline format when rendering to it.
- `encoder.view_layer(id)` returns the live artwork sheet's texture view.
- `encoder.view_staging(id)` returns the preview overlay's texture view.

Editor views and bind groups derived from them are valid for the current
frame, including the screen render hook. Obtain fresh handles each frame;
retained handles error after the next shade stage starts. This prevents silently
using an old texture after a view resize. Script-owned views can be retained.

Layer targeting retains the existing sheet semantics: it does not introduce
new routing to a particular layer strip. Coordinates and transforms still
select the desired region of the sheet.

The same handle works as a render attachment or sampled input:

```rust
let target = encoder.view_layer(id).unwrap();
let binding = rx.create_texture_bind_group(target).unwrap();
let pass = encoder.begin_render_pass(output.view(), #{
    label: "sample-artwork",
    load: "clear",
}).unwrap();
```

Sampling the same texture subresource that a pass renders into remains a GPU
validation error. Texture views expose no CPU pixel access or ownership transfer.

`begin_render_pass(target, options)` accepts `load: "load"` or `"clear"`.
`label` defaults to `"script_pass"`. With `"clear"`, optional
`clear: [r, g, b, a]` specifies linear floats in 0–1; default transparent black.
Contents are always stored. Beginning another pass ends the previous one.

The convenient methods remain, implemented on the same internal pass path:

```rust
encoder.begin_view_pass("paint", id, "load")
encoder.begin_staging_pass("preview", id, "clear")
encoder.view_bind_group(id)
```

`create_texture_bind_group` now takes a texture **view**, so old calls become
`rx.create_texture_bind_group(texture.view())`. Compute bindings still take
owned script textures and keep their existing API.

## Erasing

Use a replacement pipeline to draw transparent fragments directly to the
artwork, with `load: "load"`. Alpha blending leaves existing pixels unchanged
when source alpha is zero. Clearing and reconstructing the entire layer is
unnecessary and can overwrite intervening edits or alter translucent pixels.

`rx.touch_view(id)` still controls undo recording. The auxiliary
`rx.clear_view_rect(rect)` also remains available for the active view.

## Migration and tradeoff

All sample plugins and replay fixtures use the descriptor and texture-view APIs.
The previous positional pipeline constructor and point constructor are removed;
external plugins need the same mechanical migration. View/staging conveniences
are unchanged. No general-purpose auxiliary library is introduced yet.

The descriptors are more verbose, and frame-scoped views require lifetime
tracking. In exchange, erasing is an ordinary draw, rendering and sampling share
one target representation, point topology is independent of vertex layout, and
blend behavior is visible at construction instead of hidden in the host.

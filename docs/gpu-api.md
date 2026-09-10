# Script GPU pipelines and texture views

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

// mmpx.wgsl - MMPX x2 fragment pass
// Based on libretro glsl-shaders mmpx.glsl (MIT)

struct TransformUniforms {
    ortho: mat4x4<f32>,
    transform: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: TransformUniforms;

@group(1) @binding(0)
var tex: texture_2d<f32>;

@group(1) @binding(1)
var tex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) opacity: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) opacity: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.opacity = in.opacity;
    return out;
}

fn luma(c: vec4<f32>) -> f32 {
    return dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn same(b: vec4<f32>, a0: vec4<f32>) -> bool {
    return all(b == a0);
}

fn notsame(b: vec4<f32>, a0: vec4<f32>) -> bool {
    return any(b != a0);
}

fn all_eq2(b: vec4<f32>, a0: vec4<f32>, a1: vec4<f32>) -> bool {
    return same(b, a0) && same(b, a1);
}

fn all_eq3(b: vec4<f32>, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>) -> bool {
    return same(b, a0) && same(b, a1) && same(b, a2);
}

fn all_eq4(b: vec4<f32>, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>) -> bool {
    return same(b, a0) && same(b, a1) && same(b, a2) && same(b, a3);
}

fn any_eq3(b: vec4<f32>, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>) -> bool {
    return same(b, a0) || same(b, a1) || same(b, a2);
}

fn none_eq2(b: vec4<f32>, a0: vec4<f32>, a1: vec4<f32>) -> bool {
    return notsame(b, a0) && notsame(b, a1);
}

fn none_eq4(b: vec4<f32>, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>) -> bool {
    return notsame(b, a0) && notsame(b, a1) && notsame(b, a2) && notsame(b, a3);
}

fn sample_clamped(px: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    let clamped = clamp(px, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(tex, clamped, 0);
}

@fragment
fn fs_mmpx(in: VertexOutput) -> @location(0) vec4<f32> {
    let src_dims_u = textureDimensions(tex, 0);
    let src_dims = vec2<i32>(i32(src_dims_u.x), i32(src_dims_u.y));
    let src_dims_f = vec2<f32>(f32(src_dims.x), f32(src_dims.y));
    let src_px = vec2<i32>(floor(in.uv * src_dims_f));

    let e = sample_clamped(src_px + vec2<i32>(0, 0), src_dims);
    let a = sample_clamped(src_px + vec2<i32>(-1, -1), src_dims);
    let b = sample_clamped(src_px + vec2<i32>(0, -1), src_dims);
    let c = sample_clamped(src_px + vec2<i32>(1, -1), src_dims);
    let d = sample_clamped(src_px + vec2<i32>(-1, 0), src_dims);
    let f = sample_clamped(src_px + vec2<i32>(1, 0), src_dims);
    let g = sample_clamped(src_px + vec2<i32>(-1, 1), src_dims);
    let h = sample_clamped(src_px + vec2<i32>(0, 1), src_dims);
    let i = sample_clamped(src_px + vec2<i32>(1, 1), src_dims);

    var j = e;
    var k = e;
    var l = e;
    var m = e;

    if same(e, a) && same(e, b) && same(e, c) && same(e, d) && same(e, f) && same(e, g) && same(e, h) && same(e, i) {
        return e;
    }

    let p = sample_clamped(src_px + vec2<i32>(0, -2), src_dims);
    let q = sample_clamped(src_px + vec2<i32>(-2, 0), src_dims);
    let r = sample_clamped(src_px + vec2<i32>(2, 0), src_dims);
    let s = sample_clamped(src_px + vec2<i32>(0, 2), src_dims);

    let bl = luma(b);
    let dl = luma(d);
    let el = luma(e);
    let fl = luma(f);
    let hl = luma(h);

    if (same(d, b) && notsame(d, h) && notsame(d, f)) &&
       ((el >= dl) || same(e, a)) &&
       any_eq3(e, a, c, g) &&
       ((el < dl) || notsame(a, d) || notsame(e, p) || notsame(e, q)) {
        j = d;
    }
    if (same(b, f) && notsame(b, d) && notsame(b, h)) &&
       ((el >= bl) || same(e, c)) &&
       any_eq3(e, a, c, i) &&
       ((el < bl) || notsame(c, b) || notsame(e, p) || notsame(e, r)) {
        k = b;
    }
    if (same(h, d) && notsame(h, f) && notsame(h, b)) &&
       ((el >= hl) || same(e, g)) &&
       any_eq3(e, a, g, i) &&
       ((el < hl) || notsame(g, h) || notsame(e, s) || notsame(e, q)) {
        l = h;
    }
    if (same(f, h) && notsame(f, b) && notsame(f, d)) &&
       ((el >= fl) || same(e, i)) &&
       any_eq3(e, c, g, i) &&
       ((el < fl) || notsame(i, h) || notsame(e, r) || notsame(e, s)) {
        m = f;
    }

    if (notsame(e, f) && all_eq4(e, c, i, d, q) && all_eq2(f, b, h)) &&
       notsame(f, sample_clamped(src_px + vec2<i32>(3, 0), src_dims)) {
        k = f;
        m = f;
    }
    if (notsame(e, d) && all_eq4(e, a, g, f, r) && all_eq2(d, b, h)) &&
       notsame(d, sample_clamped(src_px + vec2<i32>(-3, 0), src_dims)) {
        j = d;
        l = d;
    }
    if (notsame(e, h) && all_eq4(e, g, i, b, p) && all_eq2(h, d, f)) &&
       notsame(h, sample_clamped(src_px + vec2<i32>(0, 3), src_dims)) {
        l = h;
        m = h;
    }
    if (notsame(e, b) && all_eq4(e, a, c, h, s) && all_eq2(b, d, f)) &&
       notsame(b, sample_clamped(src_px + vec2<i32>(0, -3), src_dims)) {
        j = b;
        k = b;
    }

    if (bl < el) && all_eq4(e, g, h, i, s) && none_eq4(e, a, d, c, f) {
        j = b;
        k = b;
    }
    if (hl < el) && all_eq4(e, a, b, c, p) && none_eq4(e, d, g, i, f) {
        l = h;
        m = h;
    }
    if (fl < el) && all_eq4(e, a, d, g, q) && none_eq4(e, b, c, i, h) {
        k = f;
        m = f;
    }
    if (dl < el) && all_eq4(e, c, f, i, r) && none_eq4(e, b, a, g, h) {
        j = d;
        l = d;
    }

    if notsame(h, b) {
        if notsame(h, a) && notsame(h, e) && notsame(h, c) {
            if all_eq3(h, g, f, r) &&
               none_eq2(h, d, sample_clamped(src_px + vec2<i32>(2, -1), src_dims)) {
                l = m;
            }
            if all_eq3(h, i, d, q) &&
               none_eq2(h, f, sample_clamped(src_px + vec2<i32>(-2, -1), src_dims)) {
                m = l;
            }
        }
        if notsame(b, i) && notsame(b, g) && notsame(b, e) {
            if all_eq3(b, a, f, r) &&
               none_eq2(b, d, sample_clamped(src_px + vec2<i32>(2, 1), src_dims)) {
                j = k;
            }
            if all_eq3(b, c, d, q) &&
               none_eq2(b, f, sample_clamped(src_px + vec2<i32>(-2, 1), src_dims)) {
                k = j;
            }
        }
    }

    if notsame(f, d) {
        if notsame(d, i) && notsame(d, e) && notsame(d, c) {
            if all_eq3(d, a, h, s) &&
               none_eq2(d, b, sample_clamped(src_px + vec2<i32>(1, 2), src_dims)) {
                j = l;
            }
            if all_eq3(d, g, b, p) &&
               none_eq2(d, h, sample_clamped(src_px + vec2<i32>(1, -2), src_dims)) {
                l = j;
            }
        }
        if notsame(f, e) && notsame(f, a) && notsame(f, g) {
            if all_eq3(f, c, h, s) &&
               none_eq2(f, b, sample_clamped(src_px + vec2<i32>(-1, 2), src_dims)) {
                k = m;
            }
            if all_eq3(f, i, b, p) &&
               none_eq2(f, h, sample_clamped(src_px + vec2<i32>(-1, -2), src_dims)) {
                m = k;
            }
        }
    }

    let out_px = vec2<i32>(floor(in.clip_position.xy));
    let even_x = (out_px.x & 1) == 0;
    let even_y = (out_px.y & 1) == 0;

    var color = m;
    if even_x && even_y {
        color = j;
    } else if (!even_x) && even_y {
        color = k;
    } else if even_x && (!even_y) {
        color = l;
    }
    return color;
}

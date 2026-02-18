// rotsprite.wgsl - EPX upscaling + final sampling pass

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

fn eq_rgba(a: vec4<f32>, b: vec4<f32>) -> bool {
    return all(a == b);
}

fn sample_clamped(px: vec2<i32>, dims: vec2<i32>) -> vec4<f32> {
    let clamped = clamp(px, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(tex, clamped, 0);
}

@fragment
fn fs_epx(in: VertexOutput) -> @location(0) vec4<f32> {
    let src_dims_u = textureDimensions(tex, 0);
    let src_dims = vec2<i32>(i32(src_dims_u.x), i32(src_dims_u.y));
    let src_dims_f = vec2<f32>(f32(src_dims.x), f32(src_dims.y));

    // in.uv maps destination pixel to source UV domain. floor() repeats source pixels 2x.
    let src_px = vec2<i32>(floor(in.uv * src_dims_f));
    let p = sample_clamped(src_px, src_dims);
    let a = sample_clamped(src_px + vec2<i32>(0, -1), src_dims);
    let b = sample_clamped(src_px + vec2<i32>(1, 0), src_dims);
    let c = sample_clamped(src_px + vec2<i32>(-1, 0), src_dims);
    let d = sample_clamped(src_px + vec2<i32>(0, 1), src_dims);

    let out_px = vec2<i32>(floor(in.clip_position.xy));
    let even_x = (out_px.x & 1) == 0;
    let even_y = (out_px.y & 1) == 0;

    var color = p;

    if even_x && even_y {
        if eq_rgba(c, a) && !eq_rgba(c, d) && !eq_rgba(a, b) {
            color = a;
        }
    } else if !even_x && even_y {
        if eq_rgba(a, b) && !eq_rgba(a, c) && !eq_rgba(b, d) {
            color = b;
        }
    } else if even_x && !even_y {
        if eq_rgba(d, c) && !eq_rgba(d, b) && !eq_rgba(c, a) {
            color = c;
        }
    } else {
        if eq_rgba(b, d) && !eq_rgba(b, a) && !eq_rgba(d, c) {
            color = d;
        }
    }

    return color;
}

@fragment
fn fs_final(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(tex, tex_sampler, in.uv);
    return vec4<f32>(color.rgb, color.a * in.opacity);
}

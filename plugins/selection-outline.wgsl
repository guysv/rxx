// selection-outline.wgsl - alpha-boundary outline mask

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
    @location(1) color: vec4<f32>,
    @location(2) opacity: f32,
}

fn linearize(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let lower = srgb / vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.color = vec4<f32>(linearize(in.color.rgb), in.color.a);
    out.opacity = in.opacity;
    return out;
}

fn alpha_at(px: vec2<i32>, dims: vec2<i32>) -> f32 {
    let clamped = clamp(px, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(tex, clamped, 0).a;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(tex, 0);
    let dims = vec2<i32>(i32(dims_u.x), i32(dims_u.y));
    let dims_f = vec2<f32>(f32(dims.x), f32(dims.y));
    let px = vec2<i32>(floor(in.uv * dims_f));

    let center_alpha = alpha_at(px, dims);
    if center_alpha > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let up = alpha_at(px + vec2<i32>(0, -1), dims);
    let right = alpha_at(px + vec2<i32>(1, 0), dims);
    let down = alpha_at(px + vec2<i32>(0, 1), dims);
    let left = alpha_at(px + vec2<i32>(-1, 0), dims);
    let boundary_alpha = max(max(up, right), max(down, left));

    if boundary_alpha <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    return vec4<f32>(in.color.rgb, boundary_alpha * in.color.a * in.opacity);
}

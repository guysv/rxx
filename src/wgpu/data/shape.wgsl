// Shape shader for rendering UI and brush strokes.

struct TransformUniforms {
    ortho: mat4x4<f32>,
    transform: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: TransformUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) angle: f32,
    @location(2) center: vec2<f32>,
    @location(3) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn rotation2d(angle: f32) -> mat2x2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return mat2x2<f32>(c, -s, s, c);
}

fn rotate(position: vec2<f32>, around: vec2<f32>, angle: f32) -> vec2<f32> {
    let m = rotation2d(angle);
    let rotated = m * (position - around);
    return rotated + around;
}

// Convert an sRGB color to linear space.
fn linearize(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let lower = srgb / vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let r = rotate(in.position.xy, in.center, in.angle);
    out.color = vec4<f32>(linearize(in.color.rgb), in.color.a);
    out.clip_position = uniforms.ortho * uniforms.transform * vec4<f32>(r, in.position.z, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

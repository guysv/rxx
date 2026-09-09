// A minimal textured-quad pipeline for the layer-status overlay: the
// sprite vertices from `create_sprite_vertices` (group 0 = ortho +
// transform, group 1 = the icon atlas) drawn as a screen-space quad.

struct TransformUniforms { ortho: mat4x4<f32>, transform: mat4x4<f32>, }
@group(0) @binding(0) var<uniform> uniforms: TransformUniforms;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) opacity: f32,
}
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) opacity: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.opacity = in.opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv) * in.color * in.opacity;
}

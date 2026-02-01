// Cursor shader for rendering the mouse cursor with color inversion.

struct CursorUniforms {
    ortho: mat4x4<f32>,
    scale: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: CursorUniforms;

@group(1) @binding(0)
var cursor_tex: texture_2d<f32>;
@group(1) @binding(1)
var framebuffer_tex: texture_2d<f32>;
@group(1) @binding(2)
var tex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) scale: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;
    out.scale = uniforms.scale;

    // ortho_wgpu already handles Y-flip, so use position directly
    out.clip_position = uniforms.ortho * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let fb_size = vec2<f32>(textureDimensions(framebuffer_tex));
    let fb_coord = in.clip_position.xy / fb_size / in.scale;
    let fb_texel = textureSample(
        framebuffer_tex,
        tex_sampler,
        vec2<f32>(fb_coord.x, 1.0 - fb_coord.y)
    );

    let texel = textureSample(cursor_tex, tex_sampler, in.uv);

    if (texel.a > 0.0) {
        return vec4<f32>(
            1.0 - fb_texel.r,
            1.0 - fb_texel.g,
            1.0 - fb_texel.b,
            1.0
        );
    } else {
        discard;
    }
}

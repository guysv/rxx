fn init(session_in, renderer_in) {
    let session = session_in;
    let renderer = renderer_in;
    let lookup_views = #{};
    register_command("lt/on", "Add active view to lookup set");
    register_command("lt/off", "Remove active view from lookup set");

    // Demo render texture APIs
    let created_compute_handle = renderer.create_compute_texture(64, 64, TEXTURE_FORMAT_RGBA8_UNORM);
    print(`create_compute_texture(64, 64) => ${created_compute_handle}`);
    let created_compute_handle_3d = renderer.create_compute_texture(64, 64, 64, TEXTURE_FORMAT_RGBA8_UNORM);
    print(`created_compute_handle_3d(64, 64, 64) => ${created_compute_handle}`);
    let created_handle = renderer.create_render_texture(64, 64);
    print(`create_render_texture(64, 64) => ${created_handle}`);

    // Demo shader / pipeline / bind group / buffer APIs (for render(pass))
    // Simple vertex-color shader (sprite vertex layout, no texture).
    let script_wgsl = `
// Sprite shader for rendering text, sprites, and views.

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
    out.color = vec4<f32>(linearize(in.color.rgb), in.color.a);
    out.uv = in.uv;
    out.opacity = in.opacity;
    out.clip_position = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, tex_sampler, in.uv);
    return vec4<f32>(
        mix(texel.rgb, in.color.rgb, in.color.a),
        texel.a * in.opacity
    );
}
`;
    let script_shader = renderer.create_shader_module(script_wgsl);
    let script_pipeline = renderer.create_render_pipeline_with_texture(script_shader, "vs_main", "fs_main");
    let script_bind_group = renderer.create_identity_transform_bind_group();
    print(`script pipeline/bind group/vertex buffer created`);
    
    print(`script initialized`);
}

fn draw() {
    let line_height = 18.0;
    let green = rgb8(0, 255, 0);
    for view in session.views() {
        let key = `${view.id}`;
        if key in lookup_views {
            let x = session.offset_x + view.offset.x;
            let y = session.offset_y + view.offset.y - 2.0 * line_height;
            draw_text(vec2(x, y), "LUT", green);
        }
    }
}

fn view_added(view_id) {
}

fn view_removed(view_id) {
    info(`view removed: ${view_id}`);
    lookup_views.remove(`${view_id}`);
}

fn cmd_lt_on(args) {
    let vid = session.active_view_id;
    lookup_views[`${vid}`] = true;
    let view = session.view(vid);
    print(`view: ${view.id}, offset: ${view.offset.x}, ${view.offset.y}`);
}

fn cmd_lt_off(args) {
    let vid = session.active_view_id;
    lookup_views.remove(`${vid}`);
    print(`view removed from lookup: ${vid}`);
}

fn shade(encoder) {
    // print(`shade called with encoder`);
}

fn render(pass) {
    let vid = session.active_view_id;
    let view = session.view(vid);
    if view != () {
        let x = session.offset_x + view.offset.x;
        let y = session.offset_y + view.offset.y;
        let zoom = view.zoom;
        let fw = view.frame_width.to_float();
        let fh = view.frame_height.to_float();
        let batch = renderer.sprite_singleton_batch(
            view.frame_width.to_int(), view.frame_height.to_int(),
            rect(0.0, 0.0, fw, fh),
            rect(-fw, 0.0, 0.0, fh),
            0.0,
            rgb8(0, 0, 0, 0));
        let vertices = batch.vertices();
        // debug(`vertices: ${vertices}`);
        let vertex_buffer = renderer.create_vertex_buffer_from_sprite_vertices(vertices);
        // let vertex_buffer = renderer.create_fullscreen_triangle_vertex_buffer();
        let view_bind_group = renderer.create_view_transform_bind_group(x, y, zoom.to_float());
        let view_handle = renderer.view_render_texture(vid);
        let view_tex_bind_group = renderer.create_texture_sampler_bind_group(view_handle);
        pass.set_pipeline(script_pipeline);
        pass.set_bind_group(0, view_bind_group);
        pass.set_bind_group(1, view_tex_bind_group);
        pass.set_vertex_buffer(0, vertex_buffer);
        pass.draw(6, 1, 0, 0);
    }
}
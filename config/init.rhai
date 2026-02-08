fn init(session_in, renderer_in) {
    let session = session_in;
    let renderer = renderer_in;
    let lookup_views = #{};
    register_command("lt/on", "Add active view to lookup set");
    register_command("lt/off", "Remove active view from lookup set");

    // Demo render texture APIs
    let created_handle = renderer.create_render_texture(64, 64);
    print(`create_render_texture(64, 64) => ${created_handle}`);
    let view_handle = renderer.view_render_texture(session.active_view_id);
    print(`view_render_texture(${session.active_view_id}) => ${view_handle}`);

    // Demo shader / pipeline / bind group / buffer APIs (for render(pass))
    let script_wgsl = `
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
    return m * (position - around) + around;
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
    let r = rotate(in.position.xy, in.center, in.angle);
    out.color = vec4<f32>(linearize(in.color.rgb), in.color.a);
    out.clip_position = uniforms.ortho * uniforms.transform * vec4<f32>(r, in.position.z, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
`;
    let script_shader = renderer.create_shader_module(script_wgsl);
    let script_pipeline = renderer.create_render_pipeline(script_shader, "vs_main", "fs_main");
    let script_bind_group = renderer.create_identity_transform_bind_group();
    let script_vertex_buffer = renderer.create_fullscreen_triangle_vertex_buffer();
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
    pass.set_pipeline(script_pipeline);
    pass.set_bind_group(0, script_bind_group);
    pass.set_vertex_buffer(0, script_vertex_buffer);
    pass.draw(3, 1, 0, 0);
}
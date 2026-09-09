// easymetric.wgsl — the space carve + voxel preview.
//
// EasyMetric reconstructs a 3D form from three painted orthographic
// silhouettes (front/side/top) sharing the canvas with the bounding-box
// splits as their borders (see easymetric.rune). A voxel is solid iff it
// is opaque in all three silhouettes that project onto it
// (visual-hull space carving):
//
//   solid(x,y,z) = front[x,z] AND side[y,z] AND top[x,y]
//
// The carve reads the *raw stacked sheet* (view_bind_group, group 1): all
// layer strips at once, strip n = sheet rows [n*fh .. (n+1)*fh). Every
// layer is a part; the scene is the union of every strip's carve, and the
// *topmost* strip filling a voxel wins its colour.
//
// `fs_raycast` is a full-target quad: per output pixel it marches a ray
// into the X*Y*Z grid (orthographic, at the rotation in params) with a
// DDA, keeps the first solid voxel, and shades it:
//   - triplanar texture (P42): the hit face samples its owning view's RGB
//     (front view textures +/-Y faces, side +/-X, top +/-Z) — UV-free,
//     blend-free, exact, because every voxel face is axis-aligned;
//   - or a flat colour, per part (the attr texture, group 2);
//   - orientation shading (an independent knob) darkens by face normal,
//     with an optional ordered (Bayer) dither.
//
// params (group 0, binding 1):
//   [0] = (X, Y, Z, nlayers)         box dims + strip count
//   [1] = (fw, fh, pane_w, pane_h)   frame size + preview-pane size
//   [2] = (yaw, pitch, shade, dither) camera angle (rad), shade 0/1,
//                                     dither Bayer level 0..4
//
// attr texture (group 2, nlayers x 1): per-strip rgb = colored fill,
// a = textured flag (a > 0.5 => sample the view RGB).

struct TransformUniforms { ortho: mat4x4<f32>, transform: mat4x4<f32>, }
@group(0) @binding(0) var<uniform> uniforms: TransformUniforms;
@group(0) @binding(1) var<uniform> params: array<vec4<f32>, 3>;
@group(1) @binding(0) var sheet_tex: texture_2d<f32>;
@group(1) @binding(1) var sheet_samp: sampler;
@group(2) @binding(0) var attr_tex: texture_2d<f32>;
@group(2) @binding(1) var attr_samp: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) opacity: f32,
}
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_quad(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

// Opacity of a sheet texel. Texel rows are y-down (row 0 = top); the
// carve only reads alpha, so the sheet's sRGB encoding is irrelevant.
fn opaque_at(px: vec2<i32>) -> bool {
    return textureLoad(sheet_tex, px, 0).a > 0.5;
}

// The texel rows of strip `s` for voxel `v` (view coords are y-down;
// texel_row = strip_base + view_y):
//   front[x,z]: view (x, z)   side[y,z]: view (X+y, z)   top[x,y]: (x, Z+y)
fn front_texel(v: vec3<i32>, base: i32, sheetH: i32) -> vec2<i32> {
    return vec2<i32>(v.x, base + v.z);
}
fn side_texel(v: vec3<i32>, bx: i32, base: i32, sheetH: i32) -> vec2<i32> {
    return vec2<i32>(bx + v.y, base + v.z);
}
fn top_texel(v: vec3<i32>, bz: i32, base: i32, sheetH: i32) -> vec2<i32> {
    return vec2<i32>(v.x, base + bz + v.y);
}

// The *topmost* strip whose three silhouettes all cover the voxel, or -1
// if none — occupancy (any strip) and colour arbitration (highest index
// wins) in one walk.
fn hit_strip(v: vec3<i32>, bx: i32, bz: i32, fh: i32, nlayers: i32, sheetH: i32) -> i32 {
    for (var s = nlayers - 1; s >= 0; s = s - 1) {
        let base = s * fh;
        let f = opaque_at(front_texel(v, base, sheetH));
        let sd = opaque_at(side_texel(v, bx, base, sheetH));
        let tp = opaque_at(top_texel(v, bz, base, sheetH));
        if (f && sd && tp) {
            return s;
        }
    }
    return -1;
}

// 4x4 Bayer ordered-dither threshold in [0,1) for an output pixel.
fn bayer4(p: vec2<i32>) -> f32 {
    var m = array<f32, 16>(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0,
    );
    let idx = (p.y & 3) * 4 + (p.x & 3);
    return m[idx] / 16.0;
}

@fragment
fn fs_raycast(in: VertexOutput) -> @location(0) vec4<f32> {
    let bx = i32(params[0].x);
    let by = i32(params[0].y);
    let bz = i32(params[0].z);
    let nlayers = i32(params[0].w);
    let fh = i32(params[1].y);
    let pw = params[1].z;
    let ph = params[1].w;
    let yaw = params[2].x;
    let pitch = params[2].y;
    let shade_on = params[2].z > 0.5;
    let dither_lvl = i32(params[2].w);
    let sheetH = i32(textureDimensions(sheet_tex).y);

    let dim = vec3<f32>(params[0].x, params[0].y, params[0].z);

    // Camera basis from yaw/pitch. `forward` points from the box centre
    // toward the camera; the ray marches along -forward.
    let cp = cos(pitch);
    let sp = sin(pitch);
    let cyw = cos(yaw);
    let syw = sin(yaw);
    let forward = vec3<f32>(cp * syw, sp, cp * cyw);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), forward));
    let up = cross(forward, right);
    let d = -forward;

    let center = dim * 0.5;
    let radius = max(0.5 * length(dim), 0.001);
    let scale = 0.40 * min(pw, ph) / radius;

    // Output pixel -> orthographic ray origin on the image plane.
    let a = in.uv.x * pw - pw * 0.5;
    let b = in.uv.y * ph - ph * 0.5;
    let dist = dim.x + dim.y + dim.z + 4.0;
    let o = center + (a / scale) * right - (b / scale) * up + dist * forward;

    // Slab test against the box AABB. Inf from a zero-d axis is fine:
    // min/max give -inf (inside the slab) or +inf (a miss).
    let inv = vec3<f32>(1.0) / d;
    let t0 = (vec3<f32>(0.0) - o) * inv;
    let t1 = (dim - o) * inv;
    let tsm = min(t0, t1);
    let tbg = max(t0, t1);
    let tnear = max(max(tsm.x, tsm.y), tsm.z);
    let tfar = min(min(tbg.x, tbg.y), tbg.z);
    if (tfar < max(tnear, 0.0)) {
        return vec4<f32>(0.0);
    }

    // DDA from the entry point through the voxel grid.
    let p = o + (max(tnear, 0.0) + 1e-4) * d;
    var voxel = clamp(vec3<i32>(floor(p)), vec3<i32>(0), vec3<i32>(bx - 1, by - 1, bz - 1));
    let stepi = vec3<i32>(sign(d));
    let stepf = vec3<f32>(stepi);
    let tDelta = abs(inv);
    var tMax = (vec3<f32>(voxel) + max(stepf, vec3<f32>(0.0)) - p) * inv;

    // Entry-face normal: the slab axis that gave tnear, facing -d.
    var normal: vec3<f32>;
    if (tsm.x >= tsm.y && tsm.x >= tsm.z) {
        normal = vec3<f32>(-stepf.x, 0.0, 0.0);
    } else if (tsm.y >= tsm.z) {
        normal = vec3<f32>(0.0, -stepf.y, 0.0);
    } else {
        normal = vec3<f32>(0.0, 0.0, -stepf.z);
    }

    var strip = -1;
    let maxSteps = bx + by + bz + 3;
    for (var i = 0; i < maxSteps; i = i + 1) {
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= bx || voxel.y >= by || voxel.z >= bz) {
            break;
        }
        strip = hit_strip(voxel, bx, bz, fh, nlayers, sheetH);
        if (strip >= 0) {
            break;
        }
        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            voxel.x += stepi.x;
            tMax.x += tDelta.x;
            normal = vec3<f32>(-stepf.x, 0.0, 0.0);
        } else if (tMax.y <= tMax.z) {
            voxel.y += stepi.y;
            tMax.y += tDelta.y;
            normal = vec3<f32>(0.0, -stepf.y, 0.0);
        } else {
            voxel.z += stepi.z;
            tMax.z += tDelta.z;
            normal = vec3<f32>(0.0, 0.0, -stepf.z);
        }
    }
    if (strip < 0) {
        return vec4<f32>(0.0);
    }

    // Triplanar texture vs flat colour, per part (attr texture).
    let base = strip * fh;
    var texel: vec2<i32>;
    if (abs(normal.y) > 0.5) {
        texel = front_texel(voxel, base, sheetH);
    } else if (abs(normal.x) > 0.5) {
        texel = side_texel(voxel, bx, base, sheetH);
    } else {
        texel = top_texel(voxel, bz, base, sheetH);
    }
    let attr = textureLoad(attr_tex, vec2<i32>(strip, 0), 0);
    var color: vec3<f32>;
    if (attr.a > 0.5) {
        // textured: the owning view's RGB. textureLoad sRGB-decodes and
        // the sRGB target re-encodes, so the byte round-trips for free.
        color = textureLoad(sheet_tex, texel, 0).rgb;
    } else {
        color = attr.rgb;
    }

    // Orientation shading: an independent knob. Darken by face normal,
    // optionally ordered-dithered into bands.
    if (shade_on) {
        let lightdir = normalize(vec3<f32>(0.5, 0.9, 0.4));
        var lambert = 0.45 + 0.55 * max(dot(normal, lightdir), 0.0);
        if (dither_lvl > 0) {
            let pc = vec2<i32>(i32(in.uv.x * pw), i32(in.uv.y * ph));
            let bands = f32(dither_lvl) + 1.0;
            lambert = clamp(floor(lambert * bands + bayer4(pc)) / bands, 0.0, 1.0);
        }
        color = color * lambert;
    }
    return vec4<f32>(color, 1.0);
}

// Outline pass: edge-detect the raycast output (bound at group 1 as
// sheet_tex) and composite an outline. Coverage transitions (opaque next
// to transparent) give the silhouette; colour deltas between opaque
// neighbours give interior seams — shading + triplanar make adjacent
// faces differ, so a single colour+coverage pass catches essentially
// every seam (MRT face-id is the optional exact fix, deferred).
//   params[0].x: style — 0 off (passthrough), 1 solid (black), 2 shade
//   (darken the existing colour).
// The outline rides the opaque edge pixels (an inner border), so the
// sprite never grows.
@fragment
fn fs_outline(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(sheet_tex);
    let pc = vec2<i32>(in.uv * vec2<f32>(dims));
    let c = textureLoad(sheet_tex, pc, 0);
    let style = i32(params[0].x);
    if (style == 0 || c.a < 0.5) {
        return c;
    }
    let l = textureLoad(sheet_tex, pc + vec2<i32>(-1, 0), 0);
    let r = textureLoad(sheet_tex, pc + vec2<i32>(1, 0), 0);
    let u = textureLoad(sheet_tex, pc + vec2<i32>(0, -1), 0);
    let d = textureLoad(sheet_tex, pc + vec2<i32>(0, 1), 0);
    let thr = 0.12;
    var edge = l.a < 0.5 || r.a < 0.5 || u.a < 0.5 || d.a < 0.5;
    edge = edge ||
        (l.a > 0.5 && distance(c.rgb, l.rgb) > thr) ||
        (r.a > 0.5 && distance(c.rgb, r.rgb) > thr) ||
        (u.a > 0.5 && distance(c.rgb, u.rgb) > thr) ||
        (d.a > 0.5 && distance(c.rgb, d.rgb) > thr);
    if (!edge) {
        return c;
    }
    if (style == 1) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(c.rgb * 0.4, c.a);
}

// Plain textured passthrough: the preview-pane draw (render hook) and
// the test driver use this.
@fragment
fn fs_show(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(sheet_tex, sheet_samp, in.uv);
}

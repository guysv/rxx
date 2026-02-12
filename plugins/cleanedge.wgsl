/*** MIT LICENSE
Copyright (c) 2022 torcado

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
***/

// Clean Edge — pixel-art edge-smoothing shader.
// Ported from torcado's cleanEdge GDShader to WGSL.
//
// Vertex input matches Sprite2dVertex (position, uv, color, opacity).
// Texture bind group matches texture_bind_group_layout (@group(1)).

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Vertex
// ---------------------------------------------------------------------------

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

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;
    out.color = in.color;
    out.opacity = in.opacity;
    out.clip_position = uniforms.ortho * uniforms.transform * vec4<f32>(in.position, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// The color with the highest priority.  Other colors are tested based on
// distance to this color to determine overlap priority.
const HIGHEST_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

// How close two colors must be to be considered "similar".
const SIMILAR_THRESHOLD: f32 = 0.0;

// Edge line width.
const LINE_WIDTH: f32 = 1.0;

// Enable 2:1 slopes (otherwise only 45-degree diagonals).
const SLOPE: bool = true;

// Clean up small-detail slope transitions (only effective when SLOPE is true).
const CLEANUP: bool = true;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn similar(col1: vec4<f32>, col2: vec4<f32>) -> bool {
    return (col1.a == 0.0 && col2.a == 0.0) || distance(col1, col2) <= SIMILAR_THRESHOLD;
}

fn similar3(col1: vec4<f32>, col2: vec4<f32>, col3: vec4<f32>) -> bool {
    return similar(col1, col2) && similar(col2, col3);
}

fn similar4(col1: vec4<f32>, col2: vec4<f32>, col3: vec4<f32>, col4: vec4<f32>) -> bool {
    return similar(col1, col2) && similar(col2, col3) && similar(col3, col4);
}

fn higher(this_col: vec4<f32>, other_col: vec4<f32>) -> bool {
    if similar(this_col, other_col) { return false; }
    if this_col.a == other_col.a {
        return distance(this_col.rgb, HIGHEST_COLOR) < distance(other_col.rgb, HIGHEST_COLOR);
    }
    return this_col.a > other_col.a;
}

// Color distance.
fn cd(col1: vec4<f32>, col2: vec4<f32>) -> f32 {
    return distance(col1, col2);
}

fn dist_to_line(test_pt: vec2<f32>, pt1: vec2<f32>, pt2: vec2<f32>, dir: vec2<f32>) -> f32 {
    let line_dir = pt2 - pt1;
    let perp_dir = vec2<f32>(line_dir.y, -line_dir.x);
    let dir_to_pt1 = pt1 - test_pt;
    return select(-1.0, 1.0, dot(perp_dir, dir) > 0.0) * dot(normalize(perp_dir), dir_to_pt1);
}

// ---------------------------------------------------------------------------
// Slice distance — based on down-forward direction
// ---------------------------------------------------------------------------

fn slice_dist(
    point_in: vec2<f32>,
    main_dir: vec2<f32>,
    point_dir: vec2<f32>,
    ub: vec4<f32>, u: vec4<f32>, uf: vec4<f32>, uff: vec4<f32>,
    b: vec4<f32>,  c: vec4<f32>, f: vec4<f32>,  ff: vec4<f32>,
    db: vec4<f32>, d: vec4<f32>, df: vec4<f32>, dff: vec4<f32>,
    ddb: vec4<f32>, dd: vec4<f32>, ddf: vec4<f32>,
) -> vec4<f32> {
    // Clamped range prevents inaccurate identity (no change) result.
    let min_w = select(0.0, 0.45, SLOPE);
    let max_w = select(1.4, 1.142, SLOPE);
    let lw = max(min_w, min(max_w, LINE_WIDTH));
    let point = main_dir * (point_in - 0.5) + 0.5;

    // Edge detection.
    let dist_against = 4.0 * cd(f, d) + cd(uf, c) + cd(c, db) + cd(ff, df) + cd(df, dd);
    let dist_towards = 4.0 * cd(c, df) + cd(u, f) + cd(f, dff) + cd(b, d) + cd(d, ddf);
    var should_slice = (dist_against < dist_towards)
        || ((dist_against < dist_towards + 0.001) && !higher(c, f));

    // Checkerboard edge case.
    if similar4(f, d, b, u) && similar4(uf, df, db, ub) && !similar(c, f) {
        should_slice = false;
    }
    if !should_slice { return vec4<f32>(-1.0); }

    var dist: f32 = 1.0;
    var flip = false;
    let ctr = vec2<f32>(0.5, 0.5);

    // -------------------------------------------------------------------
    // Lower shallow 2:1 slant
    // -------------------------------------------------------------------
    if SLOPE && similar3(f, d, db) && !similar3(f, d, b) && !similar(uf, db) {
        if !(similar(c, df) && higher(c, f)) {
            if higher(c, f) { flip = true; }
            if similar(u, f) && !similar(c, df) && !higher(c, u) { flip = true; }
        }

        if flip {
            dist = lw - dist_to_line(point,
                ctr + vec2<f32>(1.5, -1.0) * point_dir,
                ctr + vec2<f32>(-0.5, 0.0) * point_dir, -point_dir);
        } else {
            dist = dist_to_line(point,
                ctr + vec2<f32>(1.5, 0.0) * point_dir,
                ctr + vec2<f32>(-0.5, 1.0) * point_dir, point_dir);
        }

        // Cleanup slant transitions.
        if CLEANUP && !flip && similar(c, uf)
            && !(similar3(c, uf, uff) && !similar3(c, uf, ff) && !similar(d, uff)) {
            let dist2 = dist_to_line(point,
                ctr + vec2<f32>(2.0, -1.0) * point_dir,
                ctr + vec2<f32>(0.0, 1.0) * point_dir, point_dir);
            dist = min(dist, dist2);
        }

        dist -= lw / 2.0;
        if dist <= 0.0 { return select(d, f, cd(c, f) <= cd(c, d)); }
        return vec4<f32>(-1.0);
    }

    // -------------------------------------------------------------------
    // Forward steep 2:1 slant
    // -------------------------------------------------------------------
    else if SLOPE && similar3(uf, f, d) && !similar3(u, f, d) && !similar(uf, db) {
        if !(similar(c, df) && higher(c, d)) {
            if higher(c, d) { flip = true; }
            if similar(b, d) && !similar(c, df) && !higher(c, d) { flip = true; }
        }

        if flip {
            dist = lw - dist_to_line(point,
                ctr + vec2<f32>(0.0, -0.5) * point_dir,
                ctr + vec2<f32>(-1.0, 1.5) * point_dir, -point_dir);
        } else {
            dist = dist_to_line(point,
                ctr + vec2<f32>(1.0, -0.5) * point_dir,
                ctr + vec2<f32>(0.0, 1.5) * point_dir, point_dir);
        }

        // Cleanup slant transitions.
        if CLEANUP && !flip && similar(c, db)
            && !(similar3(c, db, ddb) && !similar3(c, db, dd) && !similar(f, ddb)) {
            let dist2 = dist_to_line(point,
                ctr + vec2<f32>(1.0, 0.0) * point_dir,
                ctr + vec2<f32>(-1.0, 2.0) * point_dir, point_dir);
            dist = min(dist, dist2);
        }

        dist -= lw / 2.0;
        if dist <= 0.0 { return select(d, f, cd(c, f) <= cd(c, d)); }
        return vec4<f32>(-1.0);
    }

    // -------------------------------------------------------------------
    // 45-degree diagonal
    // -------------------------------------------------------------------
    else if similar(f, d) {
        if similar(c, df) && higher(c, f) {
            // Single pixel diagonal along neighbors — don't flip.
            if !similar(c, dd) && !similar(c, ff) {
                // Line against triple-color stripe edge case.
                flip = true;
            }
        } else {
            if higher(c, f) { flip = true; }
            if !similar(c, b) && similar4(b, f, d, u) { flip = true; }
        }

        // Single pixel 2:1 slope.
        if ((similar(f, db) && similar3(u, f, df))
            || (similar(uf, d) && similar3(b, d, df))) && !similar(c, df) {
            flip = true;
        }

        if flip {
            dist = lw - dist_to_line(point,
                ctr + vec2<f32>(1.0, -1.0) * point_dir,
                ctr + vec2<f32>(-1.0, 1.0) * point_dir, -point_dir);
        } else {
            dist = dist_to_line(point,
                ctr + vec2<f32>(1.0, 0.0) * point_dir,
                ctr + vec2<f32>(0.0, 1.0) * point_dir, point_dir);
        }

        // Cleanup slant transitions.
        if SLOPE && CLEANUP {
            if !flip && similar3(c, uf, uff) && !similar3(c, uf, ff) && !similar(d, uff) {
                let dist2 = dist_to_line(point,
                    ctr + vec2<f32>(1.5, 0.0) * point_dir,
                    ctr + vec2<f32>(-0.5, 1.0) * point_dir, point_dir);
                dist = max(dist, dist2);
            }
            if !flip && similar3(ddb, db, c) && !similar3(dd, db, c) && !similar(ddb, f) {
                let dist2 = dist_to_line(point,
                    ctr + vec2<f32>(1.0, -0.5) * point_dir,
                    ctr + vec2<f32>(0.0, 1.5) * point_dir, point_dir);
                dist = max(dist, dist2);
            }
        }

        dist -= lw / 2.0;
        if dist <= 0.0 { return select(d, f, cd(c, f) <= cd(c, d)); }
        return vec4<f32>(-1.0);
    }

    // -------------------------------------------------------------------
    // Far corner of shallow slant
    // -------------------------------------------------------------------
    else if SLOPE && similar3(ff, df, d) && !similar3(ff, df, c) && !similar(uff, d) {
        if !(similar(f, dff) && higher(f, ff)) {
            if higher(f, ff) { flip = true; }
            if similar(uf, ff) && !similar(f, dff) && !higher(f, uf) { flip = true; }
        }

        if flip {
            dist = lw - dist_to_line(point,
                ctr + vec2<f32>(2.5, -1.0) * point_dir,
                ctr + vec2<f32>(0.5, 0.0) * point_dir, -point_dir);
        } else {
            dist = dist_to_line(point,
                ctr + vec2<f32>(2.5, 0.0) * point_dir,
                ctr + vec2<f32>(0.5, 1.0) * point_dir, point_dir);
        }

        dist -= lw / 2.0;
        if dist <= 0.0 { return select(df, ff, cd(f, ff) <= cd(f, df)); }
        return vec4<f32>(-1.0);
    }

    // -------------------------------------------------------------------
    // Far corner of steep slant
    // -------------------------------------------------------------------
    else if SLOPE && similar3(f, df, dd) && !similar3(c, df, dd) && !similar(f, ddb) {
        if !(similar(d, ddf) && higher(d, dd)) {
            if higher(d, dd) { flip = true; }
            if similar(db, dd) && !similar(d, ddf) && !higher(d, dd) { flip = true; }
        }

        if flip {
            dist = lw - dist_to_line(point,
                ctr + vec2<f32>(0.0, 0.5) * point_dir,
                ctr + vec2<f32>(-1.0, 2.5) * point_dir, -point_dir);
        } else {
            dist = dist_to_line(point,
                ctr + vec2<f32>(1.0, 0.5) * point_dir,
                ctr + vec2<f32>(0.0, 2.5) * point_dir, point_dir);
        }

        dist -= lw / 2.0;
        if dist <= 0.0 { return select(dd, df, cd(d, df) <= cd(d, dd)); }
        return vec4<f32>(-1.0);
    }

    return vec4<f32>(-1.0);
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(tex, 0);
    let size = vec2<f32>(f32(dims.x), f32(dims.y)) + 0.0001; // rounding-error fix
    var px = in.uv * size;
    let local = fract(px);
    px = ceil(px);

    let point_dir = round(local) * 2.0 - 1.0;

    // Neighbor pixels.
    // Up, Down, Forward, and Back — relative to quadrant of current
    // location within pixel.

    let uub = textureSample(tex, tex_sampler, (px + vec2<f32>(-1.0, -2.0) * point_dir) / size);
    let uu  = textureSample(tex, tex_sampler, (px + vec2<f32>( 0.0, -2.0) * point_dir) / size);
    let uuf = textureSample(tex, tex_sampler, (px + vec2<f32>( 1.0, -2.0) * point_dir) / size);

    let ubb = textureSample(tex, tex_sampler, (px + vec2<f32>(-2.0, -1.0) * point_dir) / size);
    let ub  = textureSample(tex, tex_sampler, (px + vec2<f32>(-1.0, -1.0) * point_dir) / size);
    let u   = textureSample(tex, tex_sampler, (px + vec2<f32>( 0.0, -1.0) * point_dir) / size);
    let uf  = textureSample(tex, tex_sampler, (px + vec2<f32>( 1.0, -1.0) * point_dir) / size);
    let uff = textureSample(tex, tex_sampler, (px + vec2<f32>( 2.0, -1.0) * point_dir) / size);

    let bb  = textureSample(tex, tex_sampler, (px + vec2<f32>(-2.0,  0.0) * point_dir) / size);
    let b   = textureSample(tex, tex_sampler, (px + vec2<f32>(-1.0,  0.0) * point_dir) / size);
    let c   = textureSample(tex, tex_sampler, (px + vec2<f32>( 0.0,  0.0) * point_dir) / size);
    let f   = textureSample(tex, tex_sampler, (px + vec2<f32>( 1.0,  0.0) * point_dir) / size);
    let ff  = textureSample(tex, tex_sampler, (px + vec2<f32>( 2.0,  0.0) * point_dir) / size);

    let dbb = textureSample(tex, tex_sampler, (px + vec2<f32>(-2.0,  1.0) * point_dir) / size);
    let db  = textureSample(tex, tex_sampler, (px + vec2<f32>(-1.0,  1.0) * point_dir) / size);
    let d   = textureSample(tex, tex_sampler, (px + vec2<f32>( 0.0,  1.0) * point_dir) / size);
    let df  = textureSample(tex, tex_sampler, (px + vec2<f32>( 1.0,  1.0) * point_dir) / size);
    let dff = textureSample(tex, tex_sampler, (px + vec2<f32>( 2.0,  1.0) * point_dir) / size);

    let ddb = textureSample(tex, tex_sampler, (px + vec2<f32>(-1.0,  2.0) * point_dir) / size);
    let dd  = textureSample(tex, tex_sampler, (px + vec2<f32>( 0.0,  2.0) * point_dir) / size);
    let ddf = textureSample(tex, tex_sampler, (px + vec2<f32>( 1.0,  2.0) * point_dir) / size);

    var col = c;

    // Corner, back, and up slices.
    // (Slices from neighbor pixels will only ever reach these 3 quadrants.)
    let c_col = slice_dist(local, vec2<f32>( 1.0,  1.0), point_dir,
        ub, u, uf, uff, b, c, f, ff, db, d, df, dff, ddb, dd, ddf);
    let b_col = slice_dist(local, vec2<f32>(-1.0,  1.0), point_dir,
        uf, u, ub, ubb, f, c, b, bb, df, d, db, dbb, ddf, dd, ddb);
    let u_col = slice_dist(local, vec2<f32>( 1.0, -1.0), point_dir,
        db, d, df, dff, b, c, f, ff, ub, u, uf, uff, uub, uu, uuf);

    if c_col.r >= 0.0 { col = c_col; }
    if b_col.r >= 0.0 { col = b_col; }
    if u_col.r >= 0.0 { col = u_col; }

    return vec4<f32>(col.rgb, col.a * in.opacity);
}

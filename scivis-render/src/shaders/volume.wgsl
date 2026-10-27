// volume.wgsl — GPU ray-march of a 3D stack (wgpu backend). Mirrors the glow
// backend's volume shader: MIP, alpha-DVR, or isosurface compositing, per-channel
// LUT, nearest/linear/cubic sampling, view-consistent sample spacing + per-pixel
// jitter, and the Y/Z flip that matches the 2D movie orientation.
//
// The channels are packed four to a 3D texture (`vol_a` holds channels 0..3 in
// its components, `vol_b` channels 4 and 5), so one fetch returns every channel
// at a point. Texels hold *normalized* samples for integer sources (raw/255,
// raw/65535) and the raw value for float sources, so the per-channel window
// applies directly. Every format used is filterable, so linear/cubic use the
// hardware sampler; "nearest" snaps the coord to a texel center.
//
// Two accelerations, both exact (they change no pixel):
//   * Empty-space skipping. `occ` holds one texel per BRICK^3 voxels: zero
//     where nothing in or next to the brick can survive the current windows.
//     A ray reaching an empty brick jumps straight to the first sample past it,
//     staying on the same sample lattice it would have walked.
//   * Saturation. Once every channel of a MIP has reached the top of its
//     window, nothing further along the ray can change the pixel.
// And one approximation, MIP only: where a voxel is smaller than a pixel (the
// whole volume far away, or the view rendered at a fraction of its size while
// it moves), the ray reads a coarser mip level — each level the *maximum* of
// the one below, so nothing bright is lost — and steps accordingly.

// ---------------------------------------------------------------------------
// Isosurface appearance (render mode 2). Both are compile-time constants: they
// are a fixed look, not per-frame state, so they cost no uniform slot and stay
// in step with the glow backend's copy of the same two values.
//
//   SURFACE_GRAY     Albedo the surface is scaled to. 1.0 is a blown-out white
//                    for a plain grayscale stack; lower reads as softer gray.
//                    (Blender's classic default base color is 0.8.)
//   SURFACE_CONTRAST Split between ambient and camera-headlight diffuse:
//                    shade = (1 - CONTRAST) + CONTRAST * diff. 0.0 is flat and
//                    unshaded, 1.0 falls all the way to black at silhouettes.
// ---------------------------------------------------------------------------
const SURFACE_GRAY: f32 = 0.6;
const SURFACE_CONTRAST: f32 = 0.85;

// data = (window.x, window.y, enabled, albedo_t); packed in a vec4 so the uniform
// array element is 16-byte aligned.
struct VolChannel {
    data: vec4<f32>,
};

struct VolParams {
    channels: array<VolChannel, 6>,
    cam_eye: vec4<f32>,
    cam_forward: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    box_he: vec4<f32>,
    misc: vec4<f32>,   // tan_half_fov, aspect, density, iso
    gain: vec4<f32>,   // brightness, unused x3
    modes: vec4<i32>,  // num_channels, render_mode (0=MIP,1=alpha,2=surface), interp (0=nearest,1=linear,2=cubic), max steps (<= 0: MAX_STEPS)
    // (size of one pixel at unit distance, highest mip level a MIP may read,
    //  brick size in voxels, skip empty bricks: 1 or 0)
    accel: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: VolParams;
@group(0) @binding(1) var vol_a: texture_3d<f32>;
@group(0) @binding(2) var vol_b: texture_3d<f32>;
@group(0) @binding(3) var occ: texture_3d<f32>;
@group(0) @binding(4) var lut_tex: texture_2d_array<f32>;
@group(0) @binding(5) var samp: sampler;

// Upper bound on ray-march samples. High enough that the largest volume the
// memory budget admits (~1600-voxel diagonal) still gets half-voxel sampling;
// the loops run `n` iterations (n <= MAX_STEPS), so ordinary volumes never pay
// for the headroom.
const MAX_STEPS: i32 = 4096;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    var out: VertexOut;
    let p = pos[vi];
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

// Fast tricubic B-spline at mip level `lvl`: 8 hardware-linear taps (Sigg &
// Hadwiger), every component at once.
fn sample_cubic(tex: texture_3d<f32>, coord: vec3<f32>, lvl: i32) -> vec4<f32> {
    let n = vec3<f32>(textureDimensions(tex, lvl));
    let l = f32(lvl);
    let cg = coord * n - 0.5;
    let idx = floor(cg);
    let f = cg - idx;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 / 6.0) * (-f3 + 3.0 * f2 - 3.0 * f + vec3<f32>(1.0));
    let w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + vec3<f32>(4.0));
    let w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + vec3<f32>(1.0));
    let w3 = (1.0 / 6.0) * f3;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let h0 = (w1 / g0 - 0.5 + idx) / n;
    let h1 = (w3 / g1 + 1.5 + idx) / n;
    let s000 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h0.y, h0.z), l);
    let s100 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h0.y, h0.z), l);
    let s010 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h1.y, h0.z), l);
    let s110 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h1.y, h0.z), l);
    let s001 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h0.y, h1.z), l);
    let s101 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h0.y, h1.z), l);
    let s011 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h1.y, h1.z), l);
    let s111 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h1.y, h1.z), l);
    let x00 = mix(s100, s000, g0.x);
    let x10 = mix(s110, s010, g0.x);
    let x01 = mix(s101, s001, g0.x);
    let x11 = mix(s111, s011, g0.x);
    let y0 = mix(x10, x00, g0.y);
    let y1 = mix(x11, x01, g0.y);
    return mix(y1, y0, g0.z);
}

// Every component of `tex` at `tc`, level `lvl`, with the chosen filter.
fn sample3d(tex: texture_3d<f32>, tc: vec3<f32>, lvl: i32) -> vec4<f32> {
    if (P.modes.z == 2) {
        return sample_cubic(tex, tc, lvl);
    }
    if (P.modes.z == 0) {
        // Nearest: snap to the texel center so the linear sampler returns it exactly.
        let dim = vec3<f32>(textureDimensions(tex, lvl));
        let c = (floor(tc * dim) + vec3<f32>(0.5)) / dim;
        return textureSampleLevel(tex, samp, c, f32(lvl));
    }
    return textureSampleLevel(tex, samp, tc, f32(lvl));
}

// The six channels at a point: one fetch of each packed texture that holds an
// enabled channel. Unfetched components read as 0.
struct Raw {
    a: vec4<f32>,  // channels 0..3
    b: vec4<f32>,  // channels 4, 5
};

fn fetch(tc: vec3<f32>, lvl: i32, need_a: bool, need_b: bool) -> Raw {
    var r: Raw;
    r.a = vec4<f32>(0.0);
    r.b = vec4<f32>(0.0);
    if (need_a) { r.a = sample3d(vol_a, tc, lvl); }
    if (need_b) { r.b = sample3d(vol_b, tc, lvl); }
    return r;
}

fn norm_ch(v: f32, idx: i32) -> f32 {
    let win = P.channels[idx].data.xy;
    return clamp((v - win.x) / max(win.y - win.x, 1e-6), 0.0, 1.0);
}
fn lut_col(t: f32, idx: i32) -> vec3<f32> {
    return textureSampleLevel(lut_tex, samp, vec2<f32>(t, 0.5), idx, 0.0).rgb;
}
// Whether channel `idx` is present and switched on.
fn on(idx: i32) -> bool {
    return idx < P.modes.x && P.channels[idx].data.z > 0.5;
}
// Where to sample this channel's LUT for the isosurface albedo. Not always the
// top entry: a LUT that peaks early and ends dark would paint a black surface.
fn albedo_t(idx: i32) -> f32 {
    return P.channels[idx].data.w;
}

// World point -> volume texcoord, with Y and Z flipped so the volume matches the
// 2D movie: image row 0 at the top (Y), movie frame 0 nearest the default camera (Z).
fn vol_coord(p: vec3<f32>) -> vec3<f32> {
    let he = P.box_he.xyz;
    let tc = (p + he) * (0.5 / he);
    return clamp(vec3<f32>(tc.x, 1.0 - tc.y, 1.0 - tc.z), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Fade to 0 within ~1 texel of a box face (wgpu core lacks ClampToBorder), so
// faces don't smear the boundary voxel.
fn edge_fade(tc: vec3<f32>, dim: vec3<f32>) -> f32 {
    let d = min(tc, vec3<f32>(1.0) - tc) * dim;
    return clamp(min(d.x, min(d.y, d.z)), 0.0, 1.0);
}

// Whether the brick at `tc` may contribute (see the header). `dims0` is the
// volume's level-0 size in voxels.
fn occupied(tc: vec3<f32>, dims0: vec3<f32>) -> bool {
    let grid = vec3<f32>(textureDimensions(occ));
    let b = min(floor(tc * dims0 / P.accel.z), grid - vec3<f32>(1.0));
    return textureLoad(occ, vec3<i32>(b), 0).r > 0.5;
}

// How many steps of `dt` take the ray from `tc` (moving `dtc` in texture
// coordinates per unit of t) to the first sample past the brick it is in: at
// least one, so the march always advances, and at most `n` + 1, past the end.
// The bound keeps the conversion to an integer finite: a ray that only grazes
// the box has a span, and so a `dt`, of next to nothing, and an unbounded
// count could wrap the step counter round to where it started.
fn steps_past_brick(tc: vec3<f32>, dtc: vec3<f32>, dims0: vec3<f32>, dt: f32, n: i32) -> i32 {
    let bs = P.accel.z;
    let b = floor(tc * dims0 / bs);
    let lo = b * bs / dims0;
    let hi = min((b + vec3<f32>(1.0)) * bs, dims0) / dims0;
    var t = 1e30;
    if (dtc.x > 1e-9) { t = min(t, (hi.x - tc.x) / dtc.x); }
    if (dtc.x < -1e-9) { t = min(t, (lo.x - tc.x) / dtc.x); }
    if (dtc.y > 1e-9) { t = min(t, (hi.y - tc.y) / dtc.y); }
    if (dtc.y < -1e-9) { t = min(t, (lo.y - tc.y) / dtc.y); }
    if (dtc.z > 1e-9) { t = min(t, (hi.z - tc.z) / dtc.z); }
    if (dtc.z < -1e-9) { t = min(t, (lo.z - tc.z) / dtc.z); }
    return i32(min(floor(max(t, 0.0) / dt), f32(n))) + 1;
}

// Composite scalar field for the isosurface: the max windowed value across every
// enabled channel (edge-faded, matching MIP), plus the *full-intensity* LUT color
// of whichever channel is the maximum there. Using the LUT's top color (not the
// color at the crossing value) as the surface albedo keeps the surface equally
// bright at every threshold — the iso level chooses only where the surface is, and
// the lighting term does the shading. Returned as rgb = surface color, a = field.
fn field_col(tc: vec3<f32>, need_a: bool, need_b: bool) -> vec4<f32> {
    let fade = edge_fade(tc, vec3<f32>(textureDimensions(vol_a, 0)));
    let r = fetch(tc, 0, need_a, need_b);
    var f = 0.0;
    var col = vec3<f32>(0.0);
    if (on(0)) { let t = norm_ch(r.a.x, 0) * fade; if (t > f) { f = t; col = lut_col(albedo_t(0), 0); } }
    if (on(1)) { let t = norm_ch(r.a.y, 1) * fade; if (t > f) { f = t; col = lut_col(albedo_t(1), 1); } }
    if (on(2)) { let t = norm_ch(r.a.z, 2) * fade; if (t > f) { f = t; col = lut_col(albedo_t(2), 2); } }
    if (on(3)) { let t = norm_ch(r.a.w, 3) * fade; if (t > f) { f = t; col = lut_col(albedo_t(3), 3); } }
    if (on(4)) { let t = norm_ch(r.b.x, 4) * fade; if (t > f) { f = t; col = lut_col(albedo_t(4), 4); } }
    if (on(5)) { let t = norm_ch(r.b.y, 5) * fade; if (t > f) { f = t; col = lut_col(albedo_t(5), 5); } }
    return vec4<f32>(col, f);
}

fn field_scalar(tc: vec3<f32>, need_a: bool, need_b: bool) -> f32 {
    return field_col(tc, need_a, need_b).a;
}

// World-space gradient of the field (central differences, one voxel apart). It
// points toward increasing intensity; used (normalized, two-sided) as the surface
// normal for shading.
fn field_grad(wp: vec3<f32>, voxel: vec3<f32>, need_a: bool, need_b: bool) -> vec3<f32> {
    let ex = vec3<f32>(voxel.x, 0.0, 0.0);
    let ey = vec3<f32>(0.0, voxel.y, 0.0);
    let ez = vec3<f32>(0.0, 0.0, voxel.z);
    return vec3<f32>(
        field_scalar(vol_coord(wp + ex), need_a, need_b) - field_scalar(vol_coord(wp - ex), need_a, need_b),
        field_scalar(vol_coord(wp + ey), need_a, need_b) - field_scalar(vol_coord(wp - ey), need_a, need_b),
        field_scalar(vol_coord(wp + ez), need_a, need_b) - field_scalar(vol_coord(wp - ez), need_a, need_b),
    );
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let tan = P.misc.x;
    let aspect = P.misc.y;
    let density = P.misc.z;
    let iso = P.misc.w;
    let he = P.box_he.xyz;

    let rd = normalize(P.cam_forward.xyz
        + in.ndc.x * aspect * tan * P.cam_right.xyz
        + in.ndc.y * tan * P.cam_up.xyz);
    let ro = P.cam_eye.xyz;

    let inv = 1.0 / rd;
    let ta = (-he - ro) * inv;
    let tb = (he - ro) * inv;
    let tmn = min(ta, tb);
    let tmx = max(ta, tb);
    var t0 = max(max(tmn.x, tmn.y), tmn.z);
    let t1 = min(min(tmx.x, tmx.y), tmx.z);
    if (t1 < max(t0, 0.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    t0 = max(t0, 0.0);

    // Which packed textures hold a channel that is on: the other is not read.
    let need_a = on(0) || on(1) || on(2) || on(3);
    let need_b = on(4) || on(5);

    let dims0 = vec3<f32>(textureDimensions(vol_a, 0));
    let voxel0 = 2.0 * he / max(dims0, vec3<f32>(1.0));
    let ard = max(abs(rd), vec3<f32>(1e-6));

    // The mip level this ray reads (MIP only): the coarsest whose voxels are
    // still no wider than a pixel, across the ray, at the ray's nearest sample —
    // so no part of the ray is read coarser than the screen can show. Across
    // the ray, because along it the step decides, and a level's maximum keeps
    // whatever it covers.
    var lvl = 0;
    if (P.modes.y == 0 && P.accel.y > 0.5) {
        let foot = t0 * P.accel.x;
        let lat = voxel0 * sqrt(max(vec3<f32>(1.0) - rd * rd, vec3<f32>(0.0)));
        let lateral = max(lat.x, max(lat.y, lat.z));
        if (lateral > 0.0 && foot > lateral) {
            lvl = i32(clamp(floor(log2(foot / lateral)), 0.0, P.accel.y));
        }
    }
    let dims = vec3<f32>(textureDimensions(vol_a, lvl));
    let voxel = 2.0 * he / max(dims, vec3<f32>(1.0));

    // Half a voxel along every axis the ray crosses, measured along the ray:
    // a ray down through a stack of thick slices needs a step per half slice,
    // not per half of the fine in-plane pixel. Then capped by the host's
    // per-frame budget (`max_steps`), which coarsens the spacing instead.
    let along = voxel / ard;
    let base_step = max(0.5 * min(along.x, min(along.y, along.z)), 1e-4);
    let span = t1 - t0;
    let cap = select(MAX_STEPS, min(P.modes.w, MAX_STEPS), P.modes.w > 0);
    let n = clamp(i32(span / base_step) + 1, 1, cap);
    let dt = span / f32(n);

    // Jitter the start within one step: decorrelates slice-aligned banding.
    let jitter = fract(sin(dot(in.clip_position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let p0 = ro + rd * (t0 + jitter * dt);
    let dp = rd * dt;
    // Sample i is at p0 + i * dp, worked out afresh each step rather than
    // accumulated, so a ray that skips lands exactly where one that walked would.
    // Not for a ray of no length (a corner or an edge of the box, exactly):
    // it has one sample, and no step to count bricks in.
    let skip = P.accel.w > 0.5 && dt > 0.0;
    let dtc = vec3<f32>(rd.x, -rd.y, -rd.z) * (0.5 / he);

    if (P.modes.y == 1) {
        // Alpha DVR (ImageJ 3D Viewer "Volume"): emission-absorption, front-to-back.
        // The edge fade applies to the *windowed* value (fading raw values breaks
        // float data whose window doesn't start at 0). Disabled channels are
        // skipped entirely (uniform branch, no sampling cost).
        var col = vec3<f32>(0.0);
        var acc = 0.0;
        var i = 0;
        loop {
            if (i >= n) { break; }
            let p = p0 + dp * f32(i);
            let tc = vol_coord(p);
            if (skip && !occupied(tc, dims0)) {
                i = i + steps_past_brick(tc, dtc, dims0, dt, n);
                continue;
            }
            let fade = edge_fade(tc, dims);
            let r = fetch(tc, 0, need_a, need_b);
            var emit = vec3<f32>(0.0);
            var wsum = 0.0;
            if (on(0)) { let t = norm_ch(r.a.x, 0) * fade; emit += t * lut_col(t, 0); wsum += t; }
            if (on(1)) { let t = norm_ch(r.a.y, 1) * fade; emit += t * lut_col(t, 1); wsum += t; }
            if (on(2)) { let t = norm_ch(r.a.z, 2) * fade; emit += t * lut_col(t, 2); wsum += t; }
            if (on(3)) { let t = norm_ch(r.a.w, 3) * fade; emit += t * lut_col(t, 3); wsum += t; }
            if (on(4)) { let t = norm_ch(r.b.x, 4) * fade; emit += t * lut_col(t, 4); wsum += t; }
            if (on(5)) { let t = norm_ch(r.b.y, 5) * fade; emit += t * lut_col(t, 5); wsum += t; }
            let a = 1.0 - exp(-density * wsum * dt);
            var albedo = vec3<f32>(0.0);
            if (wsum > 1e-6) { albedo = emit / wsum; }
            col += (1.0 - acc) * a * albedo;
            acc += (1.0 - acc) * a;
            if (acc > 0.995) { break; }
            i = i + 1;
        }
        // Clamped *after* the gain, so a gain above 1 brightens what is there
        // rather than being rounded away first.
        return vec4<f32>(clamp(col * P.gain.x, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }

    if (P.modes.y == 2) {
        // Isosurface: walk the ray until the composite field first reaches `iso`,
        // refine the crossing between the two bracketing samples, then shade the
        // hit from the field gradient with a camera headlight (two-sided, so both
        // sides of the surface are lit and silhouettes fall off to the ambient term).
        var prev_f = 0.0;
        var skipped = false;
        var i = 0;
        loop {
            if (i >= n) { break; }
            let p = p0 + dp * f32(i);
            let tc = vol_coord(p);
            if (skip && !occupied(tc, dims0)) {
                i = i + steps_past_brick(tc, dtc, dims0, dt, n);
                skipped = true;
                continue;
            }
            let fc = field_col(tc, need_a, need_b);
            let f = fc.a;
            if (f >= iso) {
                // The crossing is refined against the sample before, which a
                // skip jumped over: it is read now, so the hit lands where a
                // ray that walked would have put it. (Below the threshold is
                // all that is known of it otherwise.)
                var before = prev_f;
                if (skipped) { before = field_scalar(vol_coord(p - dp), need_a, need_b); }
                let frac = clamp((iso - before) / max(f - before, 1e-6), 0.0, 1.0);
                let hit = p - dp * (1.0 - frac);
                let g = field_grad(hit, voxel0, need_a, need_b);
                var nrm = vec3<f32>(0.0, 0.0, 1.0);
                let glen = length(g);
                if (glen > 1e-6) { nrm = g / glen; }
                let diff = abs(dot(nrm, -rd));
                let shade = (1.0 - SURFACE_CONTRAST) + SURFACE_CONTRAST * diff;
                return vec4<f32>(clamp(fc.rgb * SURFACE_GRAY * shade, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
            }
            prev_f = f;
            skipped = false;
            i = i + 1;
        }
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // MIP: per-channel maximum of the *windowed* value (norm_ch is monotonic, so
    // this equals windowing the raw max — but stays correct for float data with
    // negative values, where a raw-space max seeded at 0 would be wrong), then
    // color + sum. Disabled channels are skipped entirely.
    var m0 = 0.0; var m1 = 0.0; var m2 = 0.0; var m3 = 0.0; var m4 = 0.0; var m5 = 0.0;
    var i = 0;
    loop {
        if (i >= n) { break; }
        let p = p0 + dp * f32(i);
        let tc = vol_coord(p);
        if (skip && !occupied(tc, dims0)) {
            i = i + steps_past_brick(tc, dtc, dims0, dt, n);
            continue;
        }
        // One fine voxel at the faces, whatever level is read: a coarse
        // level's texel would fade a thin stack's every slice.
        let fade = edge_fade(tc, dims0);
        let r = fetch(tc, lvl, need_a, need_b);
        if (on(0)) { m0 = max(m0, norm_ch(r.a.x, 0) * fade); }
        if (on(1)) { m1 = max(m1, norm_ch(r.a.y, 1) * fade); }
        if (on(2)) { m2 = max(m2, norm_ch(r.a.z, 2) * fade); }
        if (on(3)) { m3 = max(m3, norm_ch(r.a.w, 3) * fade); }
        if (on(4)) { m4 = max(m4, norm_ch(r.b.x, 4) * fade); }
        if (on(5)) { m5 = max(m5, norm_ch(r.b.y, 5) * fade); }
        // Every channel at the top of its window: nothing further can raise it.
        if ((!on(0) || m0 >= 1.0) && (!on(1) || m1 >= 1.0) && (!on(2) || m2 >= 1.0)
            && (!on(3) || m3 >= 1.0) && (!on(4) || m4 >= 1.0) && (!on(5) || m5 >= 1.0)) {
            break;
        }
        i = i + 1;
    }
    var color = vec3<f32>(0.0);
    if (on(0)) { color += lut_col(m0, 0); }
    if (on(1)) { color += lut_col(m1, 1); }
    if (on(2)) { color += lut_col(m2, 2); }
    if (on(3)) { color += lut_col(m3, 3); }
    if (on(4)) { color += lut_col(m4, 4); }
    if (on(5)) { color += lut_col(m5, 5); }
    return vec4<f32>(clamp(color * P.gain.x, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

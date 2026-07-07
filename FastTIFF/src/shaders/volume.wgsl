// volume.wgsl — GPU ray-march of a 3D stack (wgpu backend). Mirrors the glow
// backend's volume shader: MIP or alpha-DVR compositing, per-channel LUT,
// nearest/linear/cubic sampling, view-consistent sample spacing + per-pixel
// jitter, and the Y/Z flip that matches the 2D movie orientation.
//
// All volume channels are R16Float 3D textures holding *normalized* samples
// (raw/255, raw/65535, or the raw float), so the per-channel window applies
// directly. R16Float is core-filterable, so linear/cubic use the hardware
// sampler; "nearest" snaps the coord to a texel center.

// data = (window.x, window.y, enabled, unused); packed in a vec4 so the uniform
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
    misc: vec4<f32>,   // tan_half_fov, aspect, density, unused
    modes: vec4<i32>,  // num_channels, render_mode (0=MIP,1=alpha), interp (0=nearest,1=linear,2=cubic), unused
};

@group(0) @binding(0) var<uniform> P: VolParams;
@group(0) @binding(1) var vol0: texture_3d<f32>;
@group(0) @binding(2) var vol1: texture_3d<f32>;
@group(0) @binding(3) var vol2: texture_3d<f32>;
@group(0) @binding(4) var vol3: texture_3d<f32>;
@group(0) @binding(5) var vol4: texture_3d<f32>;
@group(0) @binding(6) var vol5: texture_3d<f32>;
@group(0) @binding(7) var lut_tex: texture_2d_array<f32>;
@group(0) @binding(8) var samp: sampler;

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

// Fast tricubic B-spline: 8 hardware-linear taps (Sigg & Hadwiger).
fn sample_cubic(tex: texture_3d<f32>, coord: vec3<f32>) -> f32 {
    let n = vec3<f32>(textureDimensions(tex));
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
    let s000 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h0.y, h0.z), 0.0).r;
    let s100 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h0.y, h0.z), 0.0).r;
    let s010 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h1.y, h0.z), 0.0).r;
    let s110 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h1.y, h0.z), 0.0).r;
    let s001 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h0.y, h1.z), 0.0).r;
    let s101 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h0.y, h1.z), 0.0).r;
    let s011 = textureSampleLevel(tex, samp, vec3<f32>(h0.x, h1.y, h1.z), 0.0).r;
    let s111 = textureSampleLevel(tex, samp, vec3<f32>(h1.x, h1.y, h1.z), 0.0).r;
    let x00 = mix(s100, s000, g0.x);
    let x10 = mix(s110, s010, g0.x);
    let x01 = mix(s101, s001, g0.x);
    let x11 = mix(s111, s011, g0.x);
    let y0 = mix(x10, x00, g0.y);
    let y1 = mix(x11, x01, g0.y);
    return mix(y1, y0, g0.z);
}

fn sample3d(tex: texture_3d<f32>, tc: vec3<f32>) -> f32 {
    if (P.modes.z == 2) {
        return sample_cubic(tex, tc);
    }
    if (P.modes.z == 0) {
        // Nearest: snap to the texel center so the linear sampler returns it exactly.
        let dim = vec3<f32>(textureDimensions(tex));
        let c = (floor(tc * dim) + vec3<f32>(0.5)) / dim;
        return textureSampleLevel(tex, samp, c, 0.0).r;
    }
    return textureSampleLevel(tex, samp, tc, 0.0).r;
}

fn raw0(tc: vec3<f32>) -> f32 { return sample3d(vol0, tc); }
fn raw1(tc: vec3<f32>) -> f32 { return sample3d(vol1, tc); }
fn raw2(tc: vec3<f32>) -> f32 { return sample3d(vol2, tc); }
fn raw3(tc: vec3<f32>) -> f32 { return sample3d(vol3, tc); }
fn raw4(tc: vec3<f32>) -> f32 { return sample3d(vol4, tc); }
fn raw5(tc: vec3<f32>) -> f32 { return sample3d(vol5, tc); }

fn norm_ch(v: f32, idx: i32) -> f32 {
    let win = P.channels[idx].data.xy;
    return clamp((v - win.x) / max(win.y - win.x, 1e-6), 0.0, 1.0);
}
fn lut_col(t: f32, idx: i32) -> vec3<f32> {
    return textureSampleLevel(lut_tex, samp, vec2<f32>(t, 0.5), idx, 0.0).rgb;
}
fn enabled(idx: i32) -> f32 {
    return P.channels[idx].data.z;
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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let tan = P.misc.x;
    let aspect = P.misc.y;
    let density = P.misc.z;
    let nc = P.modes.x;
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

    // Even sample spacing from the actual voxel size (view-independent).
    let tdim = vec3<f32>(textureDimensions(vol0));
    let voxel = 2.0 * he / max(tdim, vec3<f32>(1.0));
    let base_step = max(min(voxel.x, min(voxel.y, voxel.z)) * 0.5, 1e-4);
    let span = t1 - t0;
    let n = clamp(i32(span / base_step) + 1, 1, MAX_STEPS);
    let dt = span / f32(n);

    // Jitter the start within one step: decorrelates slice-aligned banding.
    let jitter = fract(sin(dot(in.clip_position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let dp = rd * dt;
    var p = ro + rd * (t0 + jitter * dt);

    if (P.modes.y == 1) {
        // Alpha DVR (ImageJ 3D Viewer "Volume"): emission-absorption, front-to-back.
        // The edge fade applies to the *windowed* value (fading raw values breaks
        // float data whose window doesn't start at 0). Disabled channels are
        // skipped entirely (uniform branch, no sampling cost).
        var col = vec3<f32>(0.0);
        var acc = 0.0;
        for (var i = 0; i < n; i = i + 1) {
            let tc = vol_coord(p);
            let fade = edge_fade(tc, tdim);
            var emit = vec3<f32>(0.0);
            var wsum = 0.0;
            if (enabled(0) > 0.5) { let t = norm_ch(raw0(tc), 0) * fade; emit += t * lut_col(t, 0); wsum += t; }
            if (nc > 1 && enabled(1) > 0.5) { let t = norm_ch(raw1(tc), 1) * fade; emit += t * lut_col(t, 1); wsum += t; }
            if (nc > 2 && enabled(2) > 0.5) { let t = norm_ch(raw2(tc), 2) * fade; emit += t * lut_col(t, 2); wsum += t; }
            if (nc > 3 && enabled(3) > 0.5) { let t = norm_ch(raw3(tc), 3) * fade; emit += t * lut_col(t, 3); wsum += t; }
            if (nc > 4 && enabled(4) > 0.5) { let t = norm_ch(raw4(tc), 4) * fade; emit += t * lut_col(t, 4); wsum += t; }
            if (nc > 5 && enabled(5) > 0.5) { let t = norm_ch(raw5(tc), 5) * fade; emit += t * lut_col(t, 5); wsum += t; }
            let a = 1.0 - exp(-density * wsum * dt);
            var albedo = vec3<f32>(0.0);
            if (wsum > 1e-6) { albedo = emit / wsum; }
            col += (1.0 - acc) * a * albedo;
            acc += (1.0 - acc) * a;
            if (acc > 0.995) { break; }
            p += dp;
        }
        return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }

    // MIP: per-channel maximum of the *windowed* value (norm_ch is monotonic, so
    // this equals windowing the raw max — but stays correct for float data with
    // negative values, where a raw-space max seeded at 0 would be wrong), then
    // color + sum. Disabled channels are skipped entirely.
    var m0 = 0.0; var m1 = 0.0; var m2 = 0.0; var m3 = 0.0; var m4 = 0.0; var m5 = 0.0;
    for (var i = 0; i < n; i = i + 1) {
        let tc = vol_coord(p);
        let fade = edge_fade(tc, tdim);
        if (enabled(0) > 0.5) { m0 = max(m0, norm_ch(raw0(tc), 0) * fade); }
        if (nc > 1 && enabled(1) > 0.5) { m1 = max(m1, norm_ch(raw1(tc), 1) * fade); }
        if (nc > 2 && enabled(2) > 0.5) { m2 = max(m2, norm_ch(raw2(tc), 2) * fade); }
        if (nc > 3 && enabled(3) > 0.5) { m3 = max(m3, norm_ch(raw3(tc), 3) * fade); }
        if (nc > 4 && enabled(4) > 0.5) { m4 = max(m4, norm_ch(raw4(tc), 4) * fade); }
        if (nc > 5 && enabled(5) > 0.5) { m5 = max(m5, norm_ch(raw5(tc), 5) * fade); }
        p += dp;
    }
    var color = enabled(0) * lut_col(m0, 0);
    if (nc > 1) { color += enabled(1) * lut_col(m1, 1); }
    if (nc > 2) { color += enabled(2) * lut_col(m2, 2); }
    if (nc > 3) { color += enabled(3) * lut_col(m3, 3); }
    if (nc > 4) { color += enabled(4) * lut_col(m4, 4); }
    if (nc > 5) { color += enabled(5) * lut_col(m5, 5); }
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

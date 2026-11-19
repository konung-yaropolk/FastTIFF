// volume.frag.glsl - GPU ray-march of a 3D stack (glow backend). The twin
// of `volume.wgsl`, whose header says more.
//
// A fullscreen pass that, per pixel, builds a camera ray and marches it
// through the packed 3D textures. Three compositing modes (`u_mode`):
// maximum-intensity projection (order-independent), emission-absorption alpha
// compositing (the ImageJ 3D Viewer's "Volume" look), and an opaque
// gradient-shaded isosurface. The camera arrives as an explicit basis
// (eye/forward/right/up + fov), avoiding any matrix inverse in-shader.
//
// Sample spacing is derived from the actual voxel size (via `textureSize`),
// half a voxel along every axis the ray crosses - this, plus a per-pixel
// jitter of the ray start, is what kills the slice-aligned banding that
// showed up when looking down an axis. Keeping every sample strictly inside
// the box (jitter < one step) plus a zero texture border avoids the edge
// artifacts.
//
// The same accelerations as the WGSL twin: channels packed four to a texture,
// empty bricks skipped on the sample lattice, a MIP stopping once every
// channel is saturated, and a MIP reading a coarser (max-reduced) mip level
// where a voxel is smaller than a pixel.
//
// Not a complete shader on its own: the `#version` line and the constants in
// capitals (the channel count, the step cap MAX_STEPS, the isosurface look
// SURFACE_GRAY and SURFACE_CONTRAST, the MODE_* and INTERP_* codes) are
// written in ahead of it by `volume_glsl` in `scivis-render/src/shaders.rs`,
// from the Rust values the WGSL copy is given too. Their docs are there.
in vec2 v_ndc;
out vec4 frag_color;

uniform sampler3D vol_a;     // channels 0..3, packed into the components
uniform sampler3D vol_b;     // channels 4, 5
uniform sampler3D occ;       // one texel per brick: may a ray need to sample there
uniform sampler2D lut_tex;
uniform vec3 cam_eye;
uniform vec3 cam_forward;
uniform vec3 cam_right;
uniform vec3 cam_up;
uniform float tan_half_fov;
uniform float aspect;
uniform vec3 box_he;         // volume box half-extents (scaled dims)
uniform vec2 ch_window[MAX_CHANNELS];   // per-channel min, max in sampled-texture units
uniform float ch_enabled[MAX_CHANNELS];
// Where to sample each channel's LUT for the isosurface albedo. Not always
// the top entry: a LUT that peaks early and ends dark (a contrast-stretched
// palette) would otherwise paint a black surface.
uniform float ch_albedo[MAX_CHANNELS];
uniform int num_channels;
uniform int u_mode;          // MODE_MIP, MODE_ALPHA or MODE_SURFACE
uniform float u_density;     // alpha-DVR opacity scale (higher = more solid)
uniform float u_iso;         // isosurface threshold in windowed units (0..1)
uniform float u_brightness;  // gain on the composited colour (MIP and alpha DVR)
uniform int u_interp;        // INTERP_*: nearest and linear are the GL filter's; cubic is in-shader
uniform int u_max_steps;     // steps per ray at most (<= 0: MAX_STEPS)
// (size of one pixel at unit distance, highest mip level a MIP may read,
//  brick size in voxels, skip empty bricks: 1 or 0)
uniform vec4 u_accel;
// The viewport's height in pixels. gl_FragCoord counts rows from the bottom
// and the WGSL twin's position from the top; with this the jitter below counts
// them from the top too (exactly so where the viewport starts at the bottom
// row of its target, as a capture's and the reduced render's do).
uniform float u_height;

// NOTE: every texture read in this shader uses textureLod (explicit LOD) or
// texelFetch. The march loop's trip count and the early return are
// per-fragment, i.e. non-uniform control flow, where implicit-LOD texture() is
// undefined in GLSL.

// Fast tricubic B-spline reconstruction at mip level `lvl`: 8 hardware-linear
// taps (Sigg & Hadwiger), every component at once. Used when u_interp is
// INTERP_CUBIC (the GL filter is LINEAR then, so each tap is a bilinear-in-3D
// fetch).
vec4 sampleCubic(sampler3D tex, vec3 coord, int lvl) {
    vec3 n = vec3(textureSize(tex, lvl));
    float l = float(lvl);
    vec3 cg = coord * n - 0.5;
    vec3 idx = floor(cg);
    vec3 f = cg - idx;
    vec3 f2 = f * f;
    vec3 f3 = f2 * f;
    vec3 w0 = (1.0 / 6.0) * (-f3 + 3.0 * f2 - 3.0 * f + 1.0);
    vec3 w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + 4.0);
    vec3 w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0);
    vec3 w3 = (1.0 / 6.0) * f3;
    vec3 g0 = w0 + w1;
    vec3 g1 = w2 + w3;
    vec3 h0 = (w1 / g0 - 0.5 + idx) / n;
    vec3 h1 = (w3 / g1 + 1.5 + idx) / n;
    vec4 s000 = textureLod(tex, vec3(h0.x, h0.y, h0.z), l);
    vec4 s100 = textureLod(tex, vec3(h1.x, h0.y, h0.z), l);
    vec4 s010 = textureLod(tex, vec3(h0.x, h1.y, h0.z), l);
    vec4 s110 = textureLod(tex, vec3(h1.x, h1.y, h0.z), l);
    vec4 s001 = textureLod(tex, vec3(h0.x, h0.y, h1.z), l);
    vec4 s101 = textureLod(tex, vec3(h1.x, h0.y, h1.z), l);
    vec4 s011 = textureLod(tex, vec3(h0.x, h1.y, h1.z), l);
    vec4 s111 = textureLod(tex, vec3(h1.x, h1.y, h1.z), l);
    vec4 x00 = mix(s100, s000, g0.x);
    vec4 x10 = mix(s110, s010, g0.x);
    vec4 x01 = mix(s101, s001, g0.x);
    vec4 x11 = mix(s111, s011, g0.x);
    vec4 y0 = mix(x10, x00, g0.y);
    vec4 y1 = mix(x11, x01, g0.y);
    return mix(y1, y0, g0.z);
}

// Every component at a point: GL-filtered fetch (nearest/linear) or the
// tricubic above.
vec4 smp(sampler3D tex, vec3 c, int lvl) {
    if (u_interp == INTERP_CUBIC) {
        return sampleCubic(tex, c, lvl);
    }
    return textureLod(tex, c, float(lvl));
}

// The six channels at a point: one fetch of each packed texture that holds an
// enabled channel. Unfetched components read as 0.
void fetch(vec3 tc, int lvl, bool need_a, bool need_b, out vec4 a, out vec4 b) {
    a = vec4(0.0);
    b = vec4(0.0);
    if (need_a) { a = smp(vol_a, tc, lvl); }
    if (need_b) { b = smp(vol_b, tc, lvl); }
}

float norm_ch(float v, int idx) {
    vec2 mm = ch_window[idx];
    return clamp((v - mm.x) / max(mm.y - mm.x, 1e-6), 0.0, 1.0);
}
vec3 lut_col(float t, int idx) {
    return textureLod(lut_tex, vec2(t, (float(idx) + 0.5) / float(MAX_CHANNELS)), 0.0).rgb;
}
// Whether channel `idx` is present and switched on.
bool on(int idx) {
    return idx < num_channels && ch_enabled[idx] > 0.5;
}

// World point -> volume texcoord, with Y and Z flipped so the volume matches the
// 2D movie: image row 0 stays at the top (Y), and movie frame 0 is the slice
// nearest the default camera (Z). X already matches.
vec3 vol_coord(vec3 p) {
    vec3 tc = (p + box_he) * (0.5 / box_he);
    return clamp(vec3(tc.x, 1.0 - tc.y, 1.0 - tc.z), 0.0, 1.0);
}

// Whether the brick at `tc` may contribute. `dims0` is the level-0 size.
bool occupied(vec3 tc, vec3 dims0) {
    vec3 grid = vec3(textureSize(occ, 0));
    vec3 b = min(floor(tc * dims0 / u_accel.z), grid - 1.0);
    return texelFetch(occ, ivec3(b), 0).r > 0.5;
}

// How many steps of `dt` take the ray from `tc` (moving `dtc` in texture
// coordinates per unit of t) to the first sample past the brick it is in: at
// least one, so the march always advances, and at most `n` + 1. See the WGSL
// twin: GLSL leaves int() of an infinity undefined, and drivers wrap it.
int steps_past_brick(vec3 tc, vec3 dtc, vec3 dims0, float dt, int n) {
    float bs = u_accel.z;
    vec3 b = floor(tc * dims0 / bs);
    vec3 lo = b * bs / dims0;
    vec3 hi = min((b + 1.0) * bs, dims0) / dims0;
    float t = 1e30;
    if (dtc.x > 1e-9) t = min(t, (hi.x - tc.x) / dtc.x);
    if (dtc.x < -1e-9) t = min(t, (lo.x - tc.x) / dtc.x);
    if (dtc.y > 1e-9) t = min(t, (hi.y - tc.y) / dtc.y);
    if (dtc.y < -1e-9) t = min(t, (lo.y - tc.y) / dtc.y);
    if (dtc.z > 1e-9) t = min(t, (hi.z - tc.z) / dtc.z);
    if (dtc.z < -1e-9) t = min(t, (lo.z - tc.z) / dtc.z);
    return int(min(floor(max(t, 0.0) / dt), float(n))) + 1;
}

// Composite scalar field for the isosurface: the max windowed value across every
// enabled channel, plus (via `col`) the *full-intensity* LUT color of whichever
// channel is the maximum there. Using the LUT's top color (not the color at the
// crossing value) as the surface albedo keeps the surface equally bright at every
// threshold - the iso level chooses only where the surface is, the lighting term
// does the shading. Returns the field value in [0,1].
float field_col(vec3 tc, bool need_a, bool need_b, out vec3 col) {
    vec4 a;
    vec4 b;
    fetch(tc, 0, need_a, need_b, a, b);
    float f = 0.0;
    col = vec3(0.0);
    if (on(0)) { float t = norm_ch(a.x, 0); if (t > f) { f = t; col = lut_col(ch_albedo[0], 0); } }
    if (on(1)) { float t = norm_ch(a.y, 1); if (t > f) { f = t; col = lut_col(ch_albedo[1], 1); } }
    if (on(2)) { float t = norm_ch(a.z, 2); if (t > f) { f = t; col = lut_col(ch_albedo[2], 2); } }
    if (on(3)) { float t = norm_ch(a.w, 3); if (t > f) { f = t; col = lut_col(ch_albedo[3], 3); } }
    if (on(4)) { float t = norm_ch(b.x, 4); if (t > f) { f = t; col = lut_col(ch_albedo[4], 4); } }
    if (on(5)) { float t = norm_ch(b.y, 5); if (t > f) { f = t; col = lut_col(ch_albedo[5], 5); } }
    return f;
}

float field_scalar(vec3 tc, bool need_a, bool need_b) {
    vec3 ignore;
    return field_col(tc, need_a, need_b, ignore);
}

// World-space gradient of the field (central differences, one voxel apart). It
// points toward increasing intensity; used (normalized, two-sided) as the
// surface normal for shading.
vec3 field_grad(vec3 wp, vec3 voxel, bool need_a, bool need_b) {
    return vec3(
        field_scalar(vol_coord(wp + vec3(voxel.x, 0.0, 0.0)), need_a, need_b) - field_scalar(vol_coord(wp - vec3(voxel.x, 0.0, 0.0)), need_a, need_b),
        field_scalar(vol_coord(wp + vec3(0.0, voxel.y, 0.0)), need_a, need_b) - field_scalar(vol_coord(wp - vec3(0.0, voxel.y, 0.0)), need_a, need_b),
        field_scalar(vol_coord(wp + vec3(0.0, 0.0, voxel.z)), need_a, need_b) - field_scalar(vol_coord(wp - vec3(0.0, 0.0, voxel.z)), need_a, need_b)
    );
}

void main() {
    vec3 rd = normalize(cam_forward
        + v_ndc.x * aspect * tan_half_fov * cam_right
        + v_ndc.y * tan_half_fov * cam_up);
    vec3 ro = cam_eye;

    // Slab intersection with the axis-aligned box [-box_he, box_he].
    vec3 inv = 1.0 / rd;
    vec3 ta = (-box_he - ro) * inv;
    vec3 tb = (box_he - ro) * inv;
    vec3 tmin = min(ta, tb);
    vec3 tmax = max(ta, tb);
    float t0 = max(max(tmin.x, tmin.y), tmin.z);
    float t1 = min(min(tmax.x, tmax.y), tmax.z);
    if (t1 < max(t0, 0.0)) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    t0 = max(t0, 0.0);

    // Which packed textures hold a channel that is on: the other is not read.
    bool need_a = on(0) || on(1) || on(2) || on(3);
    bool need_b = on(4) || on(5);

    vec3 dims0 = vec3(textureSize(vol_a, 0));
    vec3 voxel0 = 2.0 * box_he / max(dims0, vec3(1.0));
    vec3 ard = max(abs(rd), vec3(1e-6));

    // The mip level this ray reads (MIP only); see the WGSL twin.
    int lvl = 0;
    if (u_mode == MODE_MIP && u_accel.y > 0.5) {
        float foot = t0 * u_accel.x;
        vec3 lat = voxel0 * sqrt(max(vec3(1.0) - rd * rd, vec3(0.0)));
        float lateral = max(lat.x, max(lat.y, lat.z));
        if (lateral > 0.0 && foot > lateral) {
            lvl = int(clamp(floor(log2(foot / lateral)), 0.0, u_accel.y));
        }
    }
    vec3 voxel = 2.0 * box_he / max(vec3(textureSize(vol_a, lvl)), vec3(1.0));

    // Half a voxel along every axis the ray crosses, measured along the ray,
    // then capped by the host's per-frame budget. See the WGSL twin.
    vec3 along = voxel / ard;
    float base_step = max(0.5 * min(along.x, min(along.y, along.z)), 1e-4);
    float span = t1 - t0;
    int cap = (u_max_steps > 0) ? min(u_max_steps, MAX_STEPS) : MAX_STEPS;
    int n = clamp(int(span / base_step) + 1, 1, cap);
    float dt = span / float(n);

    // Jitter the start within one step: decorrelates slice-aligned banding and
    // keeps all n samples strictly inside [t0, t1]. Hashed from the pixel's
    // position from the top left, as in the WGSL twin, so that both backends
    // sample a pixel's ray at the same points and draw the same picture: from
    // the bottom left, every ray sampled elsewhere along its step.
    vec2 pixel = vec2(gl_FragCoord.x, u_height - gl_FragCoord.y);
    float jitter = fract(sin(dot(pixel, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 p0 = ro + rd * (t0 + jitter * dt);
    vec3 dp = rd * dt;
    // Sample i is at p0 + i * dp, worked out afresh each step, so a ray that
    // skips lands exactly where one that walked would.
    // Not for a ray of no length, which has no step to count bricks in.
    bool skip = u_accel.w > 0.5 && dt > 0.0;
    vec3 dtc = vec3(rd.x, -rd.y, -rd.z) * (0.5 / box_he);

    if (u_mode == MODE_ALPHA) {
        // Alpha DVR (ImageJ 3D Viewer "Volume"): emission-absorption, front-to-back.
        // Disabled channels are skipped entirely (uniform branch, no sampling cost).
        vec3 col = vec3(0.0);
        float acc = 0.0;
        int i = 0;
        while (i < n) {
            vec3 tc = vol_coord(p0 + dp * float(i));
            if (skip && !occupied(tc, dims0)) {
                i += steps_past_brick(tc, dtc, dims0, dt, n);
                continue;
            }
            vec4 a;
            vec4 b;
            fetch(tc, 0, need_a, need_b, a, b);
            vec3 emit = vec3(0.0);
            float wsum = 0.0;
            if (on(0)) { float t = norm_ch(a.x, 0); emit += t * lut_col(t, 0); wsum += t; }
            if (on(1)) { float t = norm_ch(a.y, 1); emit += t * lut_col(t, 1); wsum += t; }
            if (on(2)) { float t = norm_ch(a.z, 2); emit += t * lut_col(t, 2); wsum += t; }
            if (on(3)) { float t = norm_ch(a.w, 3); emit += t * lut_col(t, 3); wsum += t; }
            if (on(4)) { float t = norm_ch(b.x, 4); emit += t * lut_col(t, 4); wsum += t; }
            if (on(5)) { float t = norm_ch(b.y, 5); emit += t * lut_col(t, 5); wsum += t; }
            float al = 1.0 - exp(-u_density * wsum * dt);
            vec3 albedo = (wsum > 1e-6) ? emit / wsum : vec3(0.0);
            col += (1.0 - acc) * al * albedo;
            acc += (1.0 - acc) * al;
            if (acc > ALPHA_OPAQUE) break;
            i += 1;
        }
        frag_color = vec4(clamp(col * u_brightness, vec3(0.0), vec3(1.0)), 1.0);
    } else if (u_mode == MODE_SURFACE) {
        // Isosurface: walk the ray until the composite field first reaches u_iso,
        // refine the crossing between the two bracketing samples, then shade the
        // hit from the field gradient with a camera headlight (two-sided, so both
        // faces are lit and silhouettes fall off to the ambient term).
        float prev_f = 0.0;
        bool skipped = false;
        vec3 hit_col = vec3(0.0);
        bool hit = false;
        int i = 0;
        while (i < n) {
            vec3 p = p0 + dp * float(i);
            vec3 tc = vol_coord(p);
            if (skip && !occupied(tc, dims0)) {
                i += steps_past_brick(tc, dtc, dims0, dt, n);
                skipped = true;
                continue;
            }
            vec3 fcol;
            float f = field_col(tc, need_a, need_b, fcol);
            if (f >= u_iso) {
                // Refined against the sample before, read now if a skip
                // jumped over it; see the WGSL twin.
                float before = skipped ? field_scalar(vol_coord(p - dp), need_a, need_b) : prev_f;
                float frac = clamp((u_iso - before) / max(f - before, 1e-6), 0.0, 1.0);
                vec3 hp = p - dp * (1.0 - frac);
                vec3 g = field_grad(hp, voxel0, need_a, need_b);
                vec3 nrm = (length(g) > 1e-6) ? normalize(g) : vec3(0.0, 0.0, 1.0);
                float diff = abs(dot(nrm, -rd));
                float shade = (1.0 - SURFACE_CONTRAST) + SURFACE_CONTRAST * diff;
                hit_col = clamp(fcol * SURFACE_GRAY * shade, vec3(0.0), vec3(1.0));
                hit = true;
                break;
            }
            prev_f = f;
            skipped = false;
            i += 1;
        }
        frag_color = vec4(hit ? hit_col : vec3(0.0), 1.0);
    } else {
        // MIP: per-channel maximum of the *windowed* value (norm_ch is monotonic,
        // so this equals windowing the raw max - but stays correct for float data
        // with negative values, where a raw-space max seeded at 0 would be wrong),
        // then color + sum. Disabled channels are skipped entirely.
        float m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0, m5 = 0.0;
        int i = 0;
        while (i < n) {
            vec3 tc = vol_coord(p0 + dp * float(i));
            if (skip && !occupied(tc, dims0)) {
                i += steps_past_brick(tc, dtc, dims0, dt, n);
                continue;
            }
            vec4 a;
            vec4 b;
            fetch(tc, lvl, need_a, need_b, a, b);
            if (on(0)) m0 = max(m0, norm_ch(a.x, 0));
            if (on(1)) m1 = max(m1, norm_ch(a.y, 1));
            if (on(2)) m2 = max(m2, norm_ch(a.z, 2));
            if (on(3)) m3 = max(m3, norm_ch(a.w, 3));
            if (on(4)) m4 = max(m4, norm_ch(b.x, 4));
            if (on(5)) m5 = max(m5, norm_ch(b.y, 5));
            // Every channel at the top of its window: nothing further can raise it.
            if ((!on(0) || m0 >= 1.0) && (!on(1) || m1 >= 1.0) && (!on(2) || m2 >= 1.0)
                && (!on(3) || m3 >= 1.0) && (!on(4) || m4 >= 1.0) && (!on(5) || m5 >= 1.0)) {
                break;
            }
            i += 1;
        }
        vec3 color = vec3(0.0);
        if (on(0)) color += lut_col(m0, 0);
        if (on(1)) color += lut_col(m1, 1);
        if (on(2)) color += lut_col(m2, 2);
        if (on(3)) color += lut_col(m3, 3);
        if (on(4)) color += lut_col(m4, 4);
        if (on(5)) color += lut_col(m5, 5);
        frag_color = vec4(clamp(color * u_brightness, vec3(0.0), vec3(1.0)), 1.0);
    }
}

//! The 3D volume camera: navigation styles, the orbit/fly state and its
//! mutations, and the ray-march basis derivation the GPU params are built from.
//!
//! Everything here is plain math over `f32` — no toolkit types. Pointer deltas
//! arrive as `(dx, dy)` in screen pixels, so a frontend's job is only to read
//! input and call the matching method; the eframe app's `app/camera.rs` does
//! exactly that, and a browser frontend would do the same with pointer events.

use stack_renderer::{VolumeParams, VolumeRender, MAX_CHANNELS};

/// How mouse/keyboard drive the 3D camera, modeled on familiar 3D apps. The
/// first three orbit a pivot (differing only in which button/modifier does what);
/// `WasdFly` is a first-person free-fly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NavMode {
    Cad,
    Blender,
    Maya,
    WasdFly,
}

impl NavMode {
    pub fn label(self) -> &'static str {
        match self {
            NavMode::Cad => "CAD",
            NavMode::Blender => "Blender",
            NavMode::Maya => "Maya",
            NavMode::WasdFly => "Minecraft Spectator",
        }
    }

    /// One-line control hint shown under the selector.
    pub fn help(self) -> &'static str {
        match self {
            NavMode::Cad => "Left-drag: orbit · Middle-drag: pan · Scroll: zoom",
            NavMode::Blender => "Middle-drag: orbit · Shift+Middle: pan · Scroll: zoom",
            NavMode::Maya => "Alt+Left: orbit · Alt+Middle: pan · Alt+Right / Scroll: zoom",
            NavMode::WasdFly => "Left-drag: look · WASD: move · Space/Shift: up/down · Scroll: fly",
        }
    }

    /// Whether this mode is a first-person free-fly (vs. orbiting a pivot).
    pub fn is_fly(self) -> bool {
        matches!(self, NavMode::WasdFly)
    }

    /// Whether Space/Shift add vertical movement in this mode. Not Blender,
    /// where Shift is the pan modifier.
    pub fn has_vertical_keys(self) -> bool {
        self.is_fly() || matches!(self, NavMode::Cad | NavMode::Maya)
    }
}

/// What an orbit drag rotates around, for the orbit nav modes (CAD/Blender/Maya
/// and the free-fly right-drag).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrbitPoint {
    /// The volume's geometric center (a turntable): the view re-centers on the
    /// box and spins around it. The default.
    VolumeCenter,
    /// Whatever the view center is aimed at — the point where the focal axis
    /// enters the box (the original mechanics), good for inspecting a feature.
    ScreenCenter,
}

/// Orbit camera distance (eye→pivot) bounds. `MIN = 0` lets the re-pivot put the
/// pivot right at the eye (rotate in place) when the eye is inside the volume;
/// wheel/dolly keep a small floor (`UNSTICK`) so they never sit exactly on the
/// pivot and can back out of a radius-0 orbit.
pub const VOL_DIST_MIN: f32 = 0.0;
pub const VOL_DIST_MAX: f32 = 300.0;
pub const VOL_DIST_UNSTICK: f32 = 0.02;

/// Fly speed: cross the volume's longest axis (length 1.0 in box space) in ~5 s.
/// Callers scale it by the real frame time, so it's frame-rate independent.
pub const FLY_UNITS_PER_SEC: f32 = 0.2;
/// Base distance one wheel notch flies along the focal axis.
pub const FLY_WHEEL: f32 = 0.15;
/// Radians one arrow-key press rotates by.
pub const KEY_ROT: f32 = 0.04;

pub fn vol_dist_clamped(dist: f32) -> f32 {
    dist.clamp(VOL_DIST_MIN, VOL_DIST_MAX)
}

/// The live 3D camera. Orbit modes keep the eye implicit at
/// `target - forward*dist`; the free-fly mode stores it directly in `fly_pos`.
/// [`CameraState::sync_for_nav`] converts between the two so switching modes
/// never makes the view jump.
#[derive(Clone, Copy, Debug)]
pub struct CameraState {
    /// Orbit angles (radians). Yaw spins around the vertical axis, pitch tilts.
    pub yaw: f32,
    pub pitch: f32,
    /// Orbit radius (eye→pivot); 0 = rotate in place around the eye.
    pub dist: f32,
    /// Orbit pivot (world space). Panning translates this, so orbit modes can
    /// slide the volume off-center.
    pub target: [f32; 3],
    /// Free-fly eye position (world space), for [`NavMode::WasdFly`].
    pub fly_pos: [f32; 3],
    /// The volume box half-extents from the last sync (mirrors the shader's
    /// `box_he`), cached so the orbit re-pivot can ray-cast against the box
    /// without the stack dimensions on hand.
    pub box_he: [f32; 3],
    pub nav: NavMode,
    /// What an orbit drag rotates around. Persists across files (a preference).
    pub orbit_point: OrbitPoint,
}

impl Default for CameraState {
    fn default() -> Self {
        let mut cam = CameraState {
            yaw: 0.7,
            pitch: 0.5,
            dist: 3.0,
            target: [0.0, 0.0, 0.0],
            fly_pos: [0.0, 0.0, 3.0],
            box_he: [0.5, 0.5, 0.5],
            nav: NavMode::Cad,
            orbit_point: OrbitPoint::VolumeCenter,
        };
        cam.reset();
        cam
    }
}

impl CameraState {
    /// Reset to a default three-quarter view looking at the origin. Used on load
    /// and by the Reset-position button; leaves `nav`/`orbit_point` alone, since
    /// those are preferences rather than view state.
    pub fn reset(&mut self) {
        self.yaw = 0.7;
        self.pitch = 0.5;
        self.dist = 3.0;
        self.target = [0.0, 0.0, 0.0];
        // Free-fly eye starts where the orbit eye would be, looking at the origin.
        let (forward, _, _) = volume_basis(self.yaw, self.pitch);
        self.fly_pos = [-forward[0] * self.dist, -forward[1] * self.dist, -forward[2] * self.dist];
    }

    /// The orthonormal camera basis (`forward`, `right`, `up`) for the current
    /// orientation.
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        volume_basis(self.yaw, self.pitch)
    }

    /// The eye's world position for the given look direction: `fly_pos` in the
    /// free-fly mode, else `target - forward*dist` (the orbit eye).
    pub fn eye(&self, forward: [f32; 3]) -> [f32; 3] {
        if self.nav.is_fly() {
            self.fly_pos
        } else {
            let dist = vol_dist_clamped(self.dist);
            [
                self.target[0] - forward[0] * dist,
                self.target[1] - forward[1] * dist,
                self.target[2] - forward[2] * dist,
            ]
        }
    }

    /// Rotate the orbit/look by a pointer delta (screen pixels).
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.54, 1.54);
    }

    /// Pan the orbit pivot in the camera's screen plane by a pointer delta
    /// (grab-and-drag: the scene follows the cursor).
    pub fn pan(&mut self, dx: f32, dy: f32, right: [f32; 3], up: [f32; 3], pan_speed: f32) {
        let (dx, dy) = (dx * pan_speed, dy * pan_speed);
        let t = self.target;
        self.target = [
            t[0] + up[0] * dy - right[0] * dx,
            t[1] + up[1] * dy - right[1] * dx,
            t[2] + up[2] * dy - right[2] * dx,
        ];
    }

    /// The pan speed that makes the scene track the cursor 1:1, for a viewport
    /// `panel_h` pixels tall. Floors the radius so panning still works when
    /// rotating in place.
    pub fn pan_speed(&self, panel_h: f32) -> f32 {
        let dist = vol_dist_clamped(self.dist);
        2.0 * dist.max(0.1) * TAN_HALF_FOV / panel_h.max(1.0)
    }

    /// Set the orbit pivot to where the camera's focal axis first enters the
    /// volume box, keeping the eye where it is (the orbit radius becomes that
    /// entry distance). Called when an orbit drag begins, so the rotation centers
    /// on what's under the view. When the eye is inside the box the entry distance
    /// is 0, so the pivot lands on the eye itself — the camera rotates in place.
    /// If the focal ray misses the box, the pivot falls to the focal-axis point
    /// nearest the box center (still on the axis, so the eye never jumps).
    pub fn repivot_to_focal(&mut self) {
        let (forward, _, _) = self.basis();
        let eye = self.eye(forward);
        let t = focal_box_entry(eye, forward, self.box_he).unwrap_or_else(|| {
            (-(eye[0] * forward[0] + eye[1] * forward[1] + eye[2] * forward[2])).max(0.0)
        });
        self.target = [eye[0] + forward[0] * t, eye[1] + forward[1] * t, eye[2] + forward[2] * t];
        // Radius = eye->pivot distance, so the eye doesn't move (t = 0 inside).
        self.dist = vol_dist_clamped(t);
    }

    /// Re-pivot at an orbit drag's start. Only the screen-center mode moves the
    /// pivot (to the focal box entry); the volume-center mode orbits rigidly
    /// around the origin (see `orbit_center`) and must NOT touch the look-at
    /// here, or an off-center view would snap back to center on every orbit.
    pub fn repivot(&mut self) {
        if self.orbit_point == OrbitPoint::ScreenCenter {
            self.repivot_to_focal();
        }
    }

    /// Apply an orbit drag per the orbit-point setting.
    pub fn orbit_drag(&mut self, dx: f32, dy: f32) {
        match self.orbit_point {
            OrbitPoint::VolumeCenter => self.orbit_center(dx, dy),
            OrbitPoint::ScreenCenter => self.orbit(dx, dy),
        }
    }

    /// Orbit rigidly around the volume center (the origin): apply the camera's
    /// orientation change to the look-at target as well, so the eye
    /// (= `target - forward*dist`) and the target rotate by the same rotation
    /// about the origin. The framing — which may be off-center after a pan — is
    /// preserved: the volume center stays fixed on screen, and the camera never
    /// re-aims at it. Exact (the two orthonormal bases give the rotation
    /// directly), so no drift accumulates.
    pub fn orbit_center(&mut self, dx: f32, dy: f32) {
        let (f0, r0, u0) = self.basis();
        self.orbit(dx, dy); // update yaw/pitch (same as the screen-center orbit)
        let (f1, r1, u1) = self.basis();
        // Re-express the target in the old camera basis, then rebuild it in the
        // new one — i.e. rotate it about the origin by the basis change.
        let t = self.target;
        let (a, b, c) = (
            t[0] * r0[0] + t[1] * r0[1] + t[2] * r0[2],
            t[0] * u0[0] + t[1] * u0[1] + t[2] * u0[2],
            t[0] * f0[0] + t[1] * f0[1] + t[2] * f0[2],
        );
        self.target = [
            a * r1[0] + b * u1[0] + c * f1[0],
            a * r1[1] + b * u1[1] + c * f1[1],
            a * r1[2] + b * u1[2] + c * f1[2],
        ];
    }

    /// Rotate the view while keeping the eye fixed (first-person "mouse look"):
    /// the pivot swings to stay `dist` ahead of the eye along the new direction.
    pub fn look_in_place(&mut self, dx: f32, dy: f32) {
        let (forward, _, _) = self.basis();
        let eye = self.eye(forward);
        self.orbit(dx, dy);
        let (fwd, _, _) = self.basis();
        let dist = vol_dist_clamped(self.dist);
        self.target = [eye[0] + fwd[0] * dist, eye[1] + fwd[1] * dist, eye[2] + fwd[2] * dist];
    }

    /// Orbit the free-fly eye around the current pivot (used by the free-fly
    /// mode's right-drag): rotate, then place `fly_pos` on the orbit sphere.
    pub fn orbit_fly(&mut self, dx: f32, dy: f32) {
        self.orbit(dx, dy);
        let (fwd, _, _) = self.basis();
        let dist = vol_dist_clamped(self.dist);
        self.fly_pos = [
            self.target[0] - fwd[0] * dist,
            self.target[1] - fwd[1] * dist,
            self.target[2] - fwd[2] * dist,
        ];
    }

    /// Dolly in/out by a vertical drag delta (down = out). Floors the radius so
    /// it can back out of a radius-0 (in-place) orbit.
    pub fn dolly(&mut self, dy: f32) {
        self.dist = vol_dist_clamped(self.dist.max(VOL_DIST_UNSTICK) * (1.0 + dy * 0.005));
    }

    /// Rotate by an arrow-key nudge. Applied like a mouse drag delta so the keys
    /// match the pointer's sense of rotation (yaw negated, pitch not) — without
    /// the negation the left/right keys spin the camera the wrong way.
    pub fn key_rotate(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx;
        self.pitch = (self.pitch + dy).clamp(-1.54, 1.54);
    }

    /// WASD/Space/Shift translation, in every mode: fly moves the eye, orbit
    /// modes move the pivot. `mv` is `(strafe, up, forward)` in −1..1.
    pub fn translate(&mut self, mv: [f32; 3], speed: f32) {
        let (forward, right, _) = self.basis();
        if self.nav.is_fly() {
            self.fly_pos = translate3(self.fly_pos, forward, right, mv, speed);
        } else {
            self.target = translate3(self.target, forward, right, mv, speed);
        }
    }

    /// Wheel: a linear fly along the focal axis (not a zoom). In fly mode it
    /// moves the eye; in orbit modes it moves the whole camera (eye + pivot).
    /// Speed is spectator-slow inside the box and grows with the eye's distance
    /// from the box, so far views approach fast and near ones creep.
    pub fn wheel_fly(&mut self, wheel: f32, scroll_speed: f32) {
        let (forward, _, _) = self.basis();
        if self.nav.is_fly() {
            for (p, f) in self.fly_pos.iter_mut().zip(forward) {
                *p += f * wheel * FLY_WHEEL * scroll_speed;
            }
        } else {
            let eye = self.eye(forward);
            let to_box = focal_box_entry(eye, forward, self.box_he)
                .unwrap_or_else(|| (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt());
            let m = wheel * (to_box * 0.15).max(FLY_WHEEL) * scroll_speed;
            self.target = [
                self.target[0] + forward[0] * m,
                self.target[1] + forward[1] * m,
                self.target[2] + forward[2] * m,
            ];
        }
    }

    /// Keep the view continuous when switching between a free-fly and an orbit
    /// mode: the two store the eye differently, so re-derive one from the other
    /// (same eye position + look direction, so nothing on screen jumps). Call
    /// after assigning a new `nav`, passing whether the *previous* mode was fly.
    pub fn sync_for_nav(&mut self, was_fly: bool) {
        let now_fly = self.nav.is_fly();
        if was_fly == now_fly {
            return;
        }
        let (forward, _, _) = self.basis();
        let dist = vol_dist_clamped(self.dist);
        if now_fly {
            // orbit -> fly: put the free eye where the orbit eye is.
            self.fly_pos = [
                self.target[0] - forward[0] * dist,
                self.target[1] - forward[1] * dist,
                self.target[2] - forward[2] * dist,
            ];
        } else {
            // fly -> orbit: pivot sits `dist` ahead of the eye along the look dir.
            self.target = [
                self.fly_pos[0] + forward[0] * dist,
                self.fly_pos[1] + forward[1] * dist,
                self.fly_pos[2] + forward[2] * dist,
            ];
        }
    }
}

/// The camera basis (eye + orthonormal forward/right/up) and volume-box
/// half-extents the ray-march shader consumes.
#[derive(Clone, Copy, Debug)]
pub struct VolumeCamera {
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub tan_half_fov: f32,
    pub box_he: [f32; 3],
}

/// Vertical field of view is fixed at 45°; the shader takes its tangent.
const TAN_HALF_FOV: f32 = 0.414_213_57; // tan(22.5°)

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Translate `base` by motion `mv` = (strafe, up, forward) relative to the look
/// basis (`forward`/`right`, with world-Y as up).
pub fn translate3(base: [f32; 3], forward: [f32; 3], right: [f32; 3], mv: [f32; 3], speed: f32) -> [f32; 3] {
    [
        base[0] + (forward[0] * mv[2] + right[0] * mv[0]) * speed,
        base[1] + (forward[1] * mv[2] + right[1] * mv[0]) * speed + mv[1] * speed,
        base[2] + (forward[2] * mv[2] + right[2] * mv[0]) * speed,
    ]
}

pub fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Near intersection distance of the ray `ro + t*rd` with the axis-aligned box
/// `[-he, he]` (a slab test). `None` if the ray misses the box ahead of the eye;
/// clamped to ≥ 0, so it's 0 when the eye is already inside the box.
pub fn focal_box_entry(ro: [f32; 3], rd: [f32; 3], he: [f32; 3]) -> Option<f32> {
    let mut t0 = f32::NEG_INFINITY;
    let mut t1 = f32::INFINITY;
    for i in 0..3 {
        if rd[i].abs() < 1e-9 {
            // Ray parallel to this slab: a miss unless the eye is between its faces.
            if ro[i] < -he[i] || ro[i] > he[i] {
                return None;
            }
        } else {
            let inv = 1.0 / rd[i];
            let mut ta = (-he[i] - ro[i]) * inv;
            let mut tb = (he[i] - ro[i]) * inv;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
        }
    }
    if t1 < t0.max(0.0) {
        return None;
    }
    Some(t0.max(0.0))
}

/// Orthonormal camera basis (`forward`, `right`, `up`) for an orientation. At
/// `yaw = pitch = 0` the camera looks along -Z with +Y up; yaw spins around the
/// world vertical, pitch tilts. Shared by `volume_camera` and the pan/fly input
/// math so both agree on which way "right"/"up"/"forward" point.
pub fn volume_basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let pitch = pitch.clamp(-1.54, 1.54); // ~±88°, avoid the pole singularity
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let sph = [cp * sy, sp, cp * cy]; // origin -> orbit eye
    let forward = norm3([-sph[0], -sph[1], -sph[2]]);
    let right = norm3(cross(forward, [0.0, 1.0, 0.0]));
    let up = norm3(cross(right, forward));
    (forward, right, up)
}

/// Camera basis + eye + volume-box half-extents for the ray-marcher. Orbit modes
/// place the eye at `target - forward*dist` (looking at the pivot); the free-fly
/// mode uses `fly_pos` directly. The box's largest scaled axis is 0.5, and
/// `scale` (per-axis voxel size) folds in so anisotropic voxels render with
/// correct proportions regardless of the (subsampled) texture size.
pub fn volume_camera(cam: &CameraState, scale: [f32; 3], dims: (u32, u32, u32)) -> VolumeCamera {
    let (forward, right, up) = volume_basis(cam.yaw, cam.pitch);
    let eye = cam.eye(forward);

    let phys = [dims.0 as f32 * scale[0], dims.1 as f32 * scale[1], dims.2 as f32 * scale[2]];
    let m = phys[0].max(phys[1]).max(phys[2]).max(1e-6);
    let box_he = [
        (0.5 * phys[0] / m).max(1e-3),
        (0.5 * phys[1] / m).max(1e-3),
        (0.5 * phys[2] / m).max(1e-3),
    ];
    VolumeCamera { eye, forward, right, up, tan_half_fov: TAN_HALF_FOV, box_he }
}

/// Assemble the ray-march uniforms from the camera, the per-channel windows,
/// and the volume's render settings.
///
/// `windows` is one `(min, max, is_float, enabled)` tuple per channel, already
/// in the sampled texture's units: raw for a float channel, else the 0..65535
/// display window divided by 65535 (both U8 and U16 volumes are unorm-normalized
/// — see [`stack_renderer::VolumeKind`]).
pub fn build_volume_params(
    cam: &VolumeCamera,
    channels: &[(f32, f32, bool, bool)],
    aspect: f32,
    render: VolumeRender,
    density: f32,
    iso: f32,
) -> VolumeParams {
    let n = channels.len().min(MAX_CHANNELS);
    let mut windows = [0.0f32; MAX_CHANNELS * 2];
    let mut enabled = [0.0f32; MAX_CHANNELS];
    let mut is_float = [0.0f32; MAX_CHANNELS];
    for (c, &(lo, hi, float, on)) in channels.iter().take(MAX_CHANNELS).enumerate() {
        windows[c * 2] = lo;
        windows[c * 2 + 1] = hi;
        enabled[c] = if on { 1.0 } else { 0.0 };
        is_float[c] = if float { 1.0 } else { 0.0 };
    }
    VolumeParams {
        num_channels: n as i32,
        windows,
        enabled,
        is_float,
        render_mode: render.shader_mode(),
        density,
        iso,
        eye: cam.eye,
        forward: cam.forward,
        right: cam.right,
        up: cam.up,
        tan_half_fov: cam.tan_half_fov,
        aspect,
        box_he: cam.box_he,
    }
}

#[cfg(test)]
#[path = "camera_tests.rs"]
mod tests;

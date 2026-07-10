//! 3D navigation and camera math for the volume view: the `NavMode` styles
//! (CAD / Blender / Maya / free-fly), the per-frame input -> camera driver,
//! and the ray-march camera basis derivation the GPU params are built from.
//! Split from `app.rs`; operates directly on `ViewerApp`'s camera fields.

use super::*;
use crate::render;

/// How mouse/keyboard drive the 3D camera, modeled on familiar 3D apps. The
/// first three orbit a pivot (differing only in which button/modifier does what);
/// `WasdFly` is a first-person free-fly.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum NavMode {
    Cad,
    Blender,
    Maya,
    WasdFly,
}

impl NavMode {
    pub(super) fn label(self) -> &'static str {
        match self {
            NavMode::Cad => "CAD",
            NavMode::Blender => "Blender",
            NavMode::Maya => "Maya",
            NavMode::WasdFly => "Minecraft Spectator",
        }
    }

    /// One-line control hint shown under the selector.
    pub(super) fn help(self) -> &'static str {
        match self {
            NavMode::Cad => "Left-drag: orbit · Middle-drag: pan · Scroll: zoom",
            NavMode::Blender => "Middle-drag: orbit · Shift+Middle: pan · Scroll: zoom",
            NavMode::Maya => "Alt+Left: orbit · Alt+Middle: pan · Alt+Right / Scroll: zoom",
            NavMode::WasdFly => "Left-drag: look · WASD: move · Space/Shift: up/down · Scroll: fly",
        }
    }

    /// Whether this mode is a first-person free-fly (vs. orbiting a pivot).
    pub(super) fn is_fly(self) -> bool {
        matches!(self, NavMode::WasdFly)
    }
}

impl ViewerApp {
    /// Reset the 3D camera to a default three-quarter view looking at the origin.
    /// Used on load and by the Reset-position button.
    pub(super) fn reset_volume_camera(&mut self) {
        self.vol_yaw = 0.7;
        self.vol_pitch = 0.5;
        self.vol_dist = 3.0;
        self.vol_target = [0.0, 0.0, 0.0];
        // Free-fly eye starts where the orbit eye would be, looking at the origin.
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        self.vol_fly_pos = [-forward[0] * self.vol_dist, -forward[1] * self.vol_dist, -forward[2] * self.vol_dist];
    }

    /// Rotate the orbit/look by a pointer delta (screen pixels).
    fn vol_orbit(&mut self, delta: egui::Vec2) {
        self.vol_yaw -= delta.x * 0.01;
        self.vol_pitch = (self.vol_pitch + delta.y * 0.01).clamp(-1.54, 1.54);
    }

    /// Pan the orbit pivot in the camera's screen plane by a pointer delta
    /// (grab-and-drag: the scene follows the cursor).
    fn vol_pan(&mut self, delta: egui::Vec2, right: [f32; 3], up: [f32; 3], pan_speed: f32) {
        let (dx, dy) = (delta.x * pan_speed, delta.y * pan_speed);
        let t = self.vol_target;
        self.vol_target = [
            t[0] + up[0] * dy - right[0] * dx,
            t[1] + up[1] * dy - right[1] * dx,
            t[2] + up[2] * dy - right[2] * dx,
        ];
    }

    /// The eye's world position for the given look direction: `fly_pos` in the
    /// free-fly mode, else `target - forward*dist` (the orbit eye).
    fn current_eye(&self, forward: [f32; 3]) -> [f32; 3] {
        if self.nav_mode.is_fly() {
            self.vol_fly_pos
        } else {
            let dist = vol_dist_clamped(self.vol_dist);
            [
                self.vol_target[0] - forward[0] * dist,
                self.vol_target[1] - forward[1] * dist,
                self.vol_target[2] - forward[2] * dist,
            ]
        }
    }

    /// Set the orbit pivot to where the camera's focal axis first enters the
    /// volume box, keeping the eye where it is (the orbit radius becomes that
    /// entry distance). Called when an orbit drag begins, so the rotation centers
    /// on what's under the view. When the eye is inside the box the entry distance
    /// is 0, so the pivot lands on the eye itself — the camera rotates in place.
    /// If the focal ray misses the box, the pivot falls to the focal-axis point
    /// nearest the box center (still on the axis, so the eye never jumps).
    fn repivot_to_focal(&mut self) {
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let eye = self.current_eye(forward);
        let t = focal_box_entry(eye, forward, self.vol_box_he).unwrap_or_else(|| {
            (-(eye[0] * forward[0] + eye[1] * forward[1] + eye[2] * forward[2])).max(0.0)
        });
        self.vol_target = [eye[0] + forward[0] * t, eye[1] + forward[1] * t, eye[2] + forward[2] * t];
        // Radius = eye->pivot distance, so the eye doesn't move (t = 0 inside).
        self.vol_dist = vol_dist_clamped(t);
    }

    /// Rotate the view while keeping the eye fixed (first-person "mouse look"):
    /// the pivot swings to stay `dist` ahead of the eye along the new direction.
    fn vol_look_in_place(&mut self, delta: egui::Vec2) {
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let eye = self.current_eye(forward);
        self.vol_orbit(delta);
        let (fwd, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let dist = vol_dist_clamped(self.vol_dist);
        self.vol_target = [eye[0] + fwd[0] * dist, eye[1] + fwd[1] * dist, eye[2] + fwd[2] * dist];
    }

    /// Orbit the free-fly eye around the current pivot (used by the free-fly
    /// mode's right-drag): rotate, then place `fly_pos` on the orbit sphere.
    fn vol_orbit_fly(&mut self, delta: egui::Vec2) {
        self.vol_orbit(delta);
        let (fwd, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let dist = vol_dist_clamped(self.vol_dist);
        self.vol_fly_pos = [
            self.vol_target[0] - fwd[0] * dist,
            self.vol_target[1] - fwd[1] * dist,
            self.vol_target[2] - fwd[2] * dist,
        ];
    }

    /// Keep the view continuous when switching between a free-fly and an orbit
    /// mode: the two store the eye differently, so re-derive one from the other
    /// (same eye position + look direction, so nothing on screen jumps).
    pub(super) fn sync_camera_for_nav(&mut self, was_fly: bool) {
        let now_fly = self.nav_mode.is_fly();
        if was_fly == now_fly {
            return;
        }
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let dist = vol_dist_clamped(self.vol_dist);
        if now_fly {
            // orbit -> fly: put the free eye where the orbit eye is.
            self.vol_fly_pos = [
                self.vol_target[0] - forward[0] * dist,
                self.vol_target[1] - forward[1] * dist,
                self.vol_target[2] - forward[2] * dist,
            ];
        } else {
            // fly -> orbit: pivot sits `dist` ahead of the eye along the look dir.
            self.vol_target = [
                self.vol_fly_pos[0] + forward[0] * dist,
                self.vol_fly_pos[1] + forward[1] * dist,
                self.vol_fly_pos[2] + forward[2] * dist,
            ];
        }
    }

    /// Apply this frame's mouse/keyboard to the 3D camera per the active nav mode.
    /// Returns whether the camera is actively moving (so the caller keeps
    /// repainting while a drag or a held key continues).
    pub(super) fn drive_volume_camera(&mut self, ui: &egui::Ui, response: &egui::Response, panel_rect: egui::Rect) -> bool {
        const KEY_ROT: f32 = 0.04;
        // Fly speed: cross the volume's longest axis (length 1.0 in box space) in
        // ~5 s. Time-based (× dt) so it's frame-rate independent.
        const FLY_UNITS_PER_SEC: f32 = 0.2;
        const FLY_WHEEL: f32 = 0.15;

        let mut animating = false;
        let (forward, right, up) = volume_basis(self.vol_yaw, self.vol_pitch);
        let panel_h = panel_rect.height().max(1.0);
        let dist = vol_dist_clamped(self.vol_dist);
        let tan = (45.0f32.to_radians() * 0.5).tan();
        // Pan speed floors the radius so panning still works when rotating in place.
        let pan_speed = 2.0 * dist.max(0.1) * tan / panel_h;
        let hovered = ui.rect_contains_pointer(panel_rect);
        // Clamp the frame time so a long stall (or the first frame) can't teleport.
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);

        // Keyboard + wheel (wheel only while the pointer is over the canvas).
        let (alt, shift, wheel, wasd, space, arrows) = ui.input(|i| {
            let wheel = if hovered {
                i.events.iter().fold(0.0_f32, |a, e| match e {
                    egui::Event::MouseWheel { unit, delta, .. } => {
                        a + match unit {
                            egui::MouseWheelUnit::Point => delta.y / 50.0,
                            _ => delta.y,
                        }
                    }
                    _ => a,
                })
            } else {
                0.0
            };
            let wasd = [
                i.key_down(egui::Key::A),
                i.key_down(egui::Key::D),
                i.key_down(egui::Key::W),
                i.key_down(egui::Key::S),
            ];
            let arrows = [
                i.key_down(egui::Key::ArrowLeft),
                i.key_down(egui::Key::ArrowRight),
                i.key_down(egui::Key::ArrowUp),
                i.key_down(egui::Key::ArrowDown),
            ];
            (i.modifiers.alt, i.modifiers.shift, wheel, wasd, i.key_down(egui::Key::Space), arrows)
        });

        let d = response.drag_delta();
        let moved = d != egui::Vec2::ZERO;
        let drag_l = response.dragged_by(egui::PointerButton::Primary);
        let drag_m = response.dragged_by(egui::PointerButton::Middle);
        let drag_r = response.dragged_by(egui::PointerButton::Secondary);
        let start_l = response.drag_started_by(egui::PointerButton::Primary);
        let start_m = response.drag_started_by(egui::PointerButton::Middle);
        let start_r = response.drag_started_by(egui::PointerButton::Secondary);

        // Mouse drag → orbit / pan / dolly, mapped per navigation style. Orbit
        // modes re-pivot to where the focal axis enters the volume when the orbit
        // drag begins, so you rotate around what's centered in view.
        match self.nav_mode {
            NavMode::Cad => {
                if start_l {
                    self.repivot_to_focal();
                }
                if drag_l && moved {
                    self.vol_orbit(d);
                    animating = true;
                }
                if drag_m && moved {
                    self.vol_pan(d, right, up, pan_speed);
                    animating = true;
                }
                if drag_r && moved {
                    // Right-drag looks around from a fixed eye (first-person).
                    self.vol_look_in_place(d);
                    animating = true;
                }
            }
            NavMode::Blender => {
                if start_m && !shift {
                    self.repivot_to_focal();
                }
                if drag_m && moved {
                    if shift {
                        self.vol_pan(d, right, up, pan_speed);
                    } else {
                        self.vol_orbit(d);
                    }
                    animating = true;
                }
            }
            NavMode::Maya => {
                if alt && start_l {
                    self.repivot_to_focal();
                }
                if alt && moved {
                    if drag_l {
                        self.vol_orbit(d);
                        animating = true;
                    } else if drag_m {
                        self.vol_pan(d, right, up, pan_speed);
                        animating = true;
                    } else if drag_r {
                        // Alt+Right vertical drag dollies (down = out). Floors the
                        // radius so it can back out of a radius-0 (in-place) orbit.
                        self.vol_dist =
                            vol_dist_clamped(self.vol_dist.max(VOL_DIST_UNSTICK) * (1.0 + d.y * 0.005));
                        animating = true;
                    }
                }
            }
            NavMode::WasdFly => {
                if drag_l && moved {
                    self.vol_orbit(d); // mouse-look (first-person)
                    animating = true;
                }
                if start_r {
                    // Right-drag orbits: pivot on where the view enters the box.
                    self.repivot_to_focal();
                }
                if drag_r && moved {
                    self.vol_orbit_fly(d);
                    animating = true;
                }
            }
        }

        // WASD translation, in every mode: fly moves the eye, orbit modes move
        // the pivot. Space/Shift add vertical movement in the fly, CAD and Maya
        // modes — not Blender, where Shift is the pan modifier.
        if hovered {
            let mut mv = [0.0f32; 3];
            if wasd[0] {
                mv[0] -= 1.0;
            }
            if wasd[1] {
                mv[0] += 1.0;
            }
            if wasd[2] {
                mv[2] += 1.0;
            }
            if wasd[3] {
                mv[2] -= 1.0;
            }
            let vertical_keys = self.nav_mode.is_fly() || matches!(self.nav_mode, NavMode::Cad | NavMode::Maya);
            if vertical_keys {
                if space {
                    mv[1] += 1.0;
                }
                if shift {
                    mv[1] -= 1.0;
                }
            }
            if mv != [0.0; 3] {
                let speed = FLY_UNITS_PER_SEC * dt * self.move_speed;
                if self.nav_mode.is_fly() {
                    self.vol_fly_pos = translate3(self.vol_fly_pos, forward, right, mv, speed);
                } else {
                    self.vol_target = translate3(self.vol_target, forward, right, mv, speed);
                }
                animating = true;
            }
        }

        // Arrow keys orbit/look in every mode (a keyboard fallback).
        if hovered {
            let mut arot = egui::Vec2::ZERO;
            if arrows[0] {
                arot.x -= KEY_ROT;
            }
            if arrows[1] {
                arot.x += KEY_ROT;
            }
            if arrows[2] {
                arot.y -= KEY_ROT;
            }
            if arrows[3] {
                arot.y += KEY_ROT;
            }
            if arot != egui::Vec2::ZERO {
                // Apply `arot` like a mouse drag delta so the keys match the
                // pointer's sense of rotation (see `vol_orbit`): yaw is negated,
                // pitch is not. Without the negation the left/right keys spin the
                // camera the wrong way (vertical, which isn't negated, was fine).
                self.vol_yaw -= arot.x;
                self.vol_pitch = (self.vol_pitch + arot.y).clamp(-1.54, 1.54);
                animating = true;
            }
        }

        // Wheel: a linear fly along the focal axis (not a zoom). In fly mode it
        // moves the eye; in orbit modes it moves the whole camera (eye + pivot).
        // Speed is spectator-slow inside the box and grows with the eye's
        // distance from the box, so far views approach fast and near ones creep.
        if wheel.abs() > 0.01 {
            if self.nav_mode.is_fly() {
                for (p, f) in self.vol_fly_pos.iter_mut().zip(forward) {
                    *p += f * wheel * FLY_WHEEL * self.scroll_speed;
                }
            } else {
                let eye = self.current_eye(forward);
                let to_box = focal_box_entry(eye, forward, self.vol_box_he)
                    .unwrap_or_else(|| (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt());
                let m = wheel * (to_box * 0.15).max(FLY_WHEEL) * self.scroll_speed;
                self.vol_target = [
                    self.vol_target[0] + forward[0] * m,
                    self.vol_target[1] + forward[1] * m,
                    self.vol_target[2] + forward[2] * m,
                ];
            }
        }

        animating
    }
}

/// The 3D camera control state, snapshotted from `ViewerApp` each frame and fed
/// to the params builder. Bundled so the plumbing stays a single argument.
#[derive(Clone, Copy)]
pub(super) struct VolumeCam {
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    pub(super) dist: f32,
    pub(super) target: [f32; 3],
    pub(super) fly_pos: [f32; 3],
    pub(super) nav: NavMode,
    pub(super) scale: [f32; 3],
    pub(super) aspect: f32,
    pub(super) render: render::VolumeRender,
    pub(super) density: f32,
}

/// The camera basis (eye + orthonormal forward/right/up) and volume-box
/// half-extents the ray-march shader consumes.
pub(super) struct VolumeCamera {
    pub(super) eye: [f32; 3],
    pub(super) forward: [f32; 3],
    pub(super) right: [f32; 3],
    pub(super) up: [f32; 3],
    pub(super) tan_half_fov: f32,
    pub(super) box_he: [f32; 3],
}

pub(super) fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Translate `base` by motion `mv` = (strafe, up, forward) relative to the look
/// basis (`forward`/`right`, with world-Y as up).
pub(super) fn translate3(base: [f32; 3], forward: [f32; 3], right: [f32; 3], mv: [f32; 3], speed: f32) -> [f32; 3] {
    [
        base[0] + (forward[0] * mv[2] + right[0] * mv[0]) * speed,
        base[1] + (forward[1] * mv[2] + right[1] * mv[0]) * speed + mv[1] * speed,
        base[2] + (forward[2] * mv[2] + right[2] * mv[0]) * speed,
    ]
}

pub(super) fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Orbit camera distance (eye→pivot) bounds. `MIN = 0` lets the re-pivot put the
/// pivot right at the eye (rotate in place) when the eye is inside the volume;
/// wheel/dolly keep a small floor (`UNSTICK`) so they never sit exactly on the
/// pivot and can back out of a radius-0 orbit.
pub(super) const VOL_DIST_MIN: f32 = 0.0;
pub(super) const VOL_DIST_MAX: f32 = 300.0;
pub(super) const VOL_DIST_UNSTICK: f32 = 0.02;

pub(super) fn vol_dist_clamped(dist: f32) -> f32 {
    dist.clamp(VOL_DIST_MIN, VOL_DIST_MAX)
}

/// Near intersection distance of the ray `ro + t*rd` with the axis-aligned box
/// `[-he, he]` (a slab test). `None` if the ray misses the box ahead of the eye;
/// clamped to ≥ 0, so it's 0 when the eye is already inside the box.
pub(super) fn focal_box_entry(ro: [f32; 3], rd: [f32; 3], he: [f32; 3]) -> Option<f32> {
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
pub(super) fn volume_basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
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
/// mode uses `fly_pos` directly. The box's largest scaled axis is 0.5.
pub(super) fn volume_camera(view: VolumeCam, dims: (u32, u32, u32)) -> VolumeCamera {
    let (forward, right, up) = volume_basis(view.yaw, view.pitch);
    let dist = vol_dist_clamped(view.dist);
    let eye = if view.nav.is_fly() {
        view.fly_pos
    } else {
        [
            view.target[0] - forward[0] * dist,
            view.target[1] - forward[1] * dist,
            view.target[2] - forward[2] * dist,
        ]
    };

    let scale = view.scale;
    let phys = [dims.0 as f32 * scale[0], dims.1 as f32 * scale[1], dims.2 as f32 * scale[2]];
    let m = phys[0].max(phys[1]).max(phys[2]).max(1e-6);
    let box_he = [
        (0.5 * phys[0] / m).max(1e-3),
        (0.5 * phys[1] / m).max(1e-3),
        (0.5 * phys[2] / m).max(1e-3),
    ];
    let tan_half_fov = (45.0f32.to_radians() * 0.5).tan();
    VolumeCamera { eye, forward, right, up, tan_half_fov, box_he }
}

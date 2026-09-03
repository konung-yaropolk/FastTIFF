//! egui input → 3D camera. This module reads the pointer, wheel and keyboard
//! and calls the matching [`fast_tiff_viewer::camera::CameraState`] method; all
//! the camera *math* lives there, so a different frontend reuses it by writing
//! its own version of just this file.

use super::*;
use fast_tiff_viewer::camera::{CameraState, NavMode, FLY_UNITS_PER_SEC, KEY_ROT};

/// Vertical-drag units one whole pinch is worth.
///
/// `dolly` is written for a mouse, where the input is a distance in points; a
/// pinch gives a *ratio* instead. This converts one into the other, and its
/// value is simply what makes a comfortable two-finger pinch move about as far
/// as the equivalent drag would.
const PINCH_DOLLY: f32 = 260.0;

impl ViewerApp {
    /// Apply this frame's mouse/keyboard to the 3D camera per the active nav mode.
    /// Returns whether the camera is actively moving (so the caller keeps
    /// repainting while a drag or a held key continues).
    pub(super) fn drive_volume_camera(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        panel_rect: egui::Rect,
    ) -> bool {
        let mut animating = false;
        let hovered = ui.rect_contains_pointer(panel_rect);
        // Clamp the frame time so a long stall (or the first frame) can't teleport.
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let pan_speed = self.core.volume.cam.pan_speed(panel_rect.height());
        let (_, right, up) = self.core.volume.cam.basis();
        // Snapshot the speed preferences before borrowing the camera mutably.
        let (move_speed, scroll_speed) = (self.move_speed, self.scroll_speed);

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
            (
                i.modifiers.alt,
                i.modifiers.shift,
                wheel,
                wasd,
                i.key_down(egui::Key::Space),
                arrows,
            )
        });

        // A gesture, if one is running. Two fingers pan and pinch; one finger
        // arrives as an ordinary drag and orbits through the mouse paths below,
        // which is the mapping every 3D viewer on a tablet uses.
        let touch = ui.input(|i| i.multi_touch()).filter(|t| t.num_touches >= 2);

        let d = response.drag_delta();
        let moved = d != egui::Vec2::ZERO && touch.is_none();
        let drag_l = response.dragged_by(egui::PointerButton::Primary);
        let drag_m = response.dragged_by(egui::PointerButton::Middle);
        let drag_r = response.dragged_by(egui::PointerButton::Secondary);
        let start_l = response.drag_started_by(egui::PointerButton::Primary);
        let start_m = response.drag_started_by(egui::PointerButton::Middle);
        let start_r = response.drag_started_by(egui::PointerButton::Secondary);

        let cam: &mut CameraState = &mut self.core.volume.cam;

        // Gesture first, and to the exclusion of the drag paths: egui
        // synthesises a pointer from the first touch, so without `moved` being
        // suppressed above a two-finger pan would also orbit.
        //
        // Deliberately the same two fingers for both pan and dolly rather than
        // a mode switch — pinching to close in on something while sliding it
        // into view is one motion of the hand, and splitting it across two
        // gestures is what makes touch 3D feel like driving a mouse badly.
        if let Some(t) = touch {
            if t.translation_delta != egui::Vec2::ZERO {
                cam.pan(
                    t.translation_delta.x,
                    t.translation_delta.y,
                    right,
                    up,
                    pan_speed,
                );
                animating = true;
            }
            // A pinch open means "bring it closer". `dolly` takes the vertical
            // drag convention (down is out), so the ratio is turned into a
            // signed amount in those units.
            let pinch = t.zoom_delta;
            if (pinch - 1.0).abs() > 0.001 {
                cam.dolly(-(pinch - 1.0) * PINCH_DOLLY);
                animating = true;
            }
        }

        // Mouse drag → orbit / pan / dolly, mapped per navigation style. Orbit
        // modes re-pivot to where the focal axis enters the volume when the orbit
        // drag begins, so you rotate around what's centered in view.
        match cam.nav {
            NavMode::Cad => {
                if start_l {
                    cam.repivot();
                }
                if drag_l && moved {
                    cam.orbit_drag(d.x, d.y);
                    animating = true;
                }
                if drag_m && moved {
                    cam.pan(d.x, d.y, right, up, pan_speed);
                    animating = true;
                }
                if drag_r && moved {
                    // Right-drag looks around from a fixed eye (first-person).
                    cam.look_in_place(d.x, d.y);
                    animating = true;
                }
            }
            NavMode::Blender => {
                if start_m && !shift {
                    cam.repivot();
                }
                if drag_m && moved {
                    if shift {
                        cam.pan(d.x, d.y, right, up, pan_speed);
                    } else {
                        cam.orbit_drag(d.x, d.y);
                    }
                    animating = true;
                }
            }
            NavMode::Maya => {
                if alt && start_l {
                    cam.repivot();
                }
                if alt && moved {
                    if drag_l {
                        cam.orbit_drag(d.x, d.y);
                        animating = true;
                    } else if drag_m {
                        cam.pan(d.x, d.y, right, up, pan_speed);
                        animating = true;
                    } else if drag_r {
                        // Alt+Right vertical drag dollies (down = out).
                        cam.dolly(d.y);
                        animating = true;
                    }
                }
            }
            NavMode::WasdFly => {
                if drag_l && moved {
                    cam.orbit(d.x, d.y); // mouse-look (first-person)
                    animating = true;
                }
                if start_r {
                    // Right-drag orbits the free eye around the box-entry point.
                    // Fly is first-person, so it always uses the focal pivot
                    // (the orbit-point setting governs the orbit modes above).
                    cam.repivot_to_focal();
                }
                if drag_r && moved {
                    cam.orbit_fly(d.x, d.y);
                    animating = true;
                }
            }
        }

        // WASD translation, in every mode: fly moves the eye, orbit modes move
        // the pivot. Space/Shift add vertical movement where the mode allows it.
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
            if cam.nav.has_vertical_keys() {
                if space {
                    mv[1] += 1.0;
                }
                if shift {
                    mv[1] -= 1.0;
                }
            }
            if mv != [0.0; 3] {
                cam.translate(mv, FLY_UNITS_PER_SEC * dt * move_speed);
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
                cam.key_rotate(arot.x, arot.y);
                animating = true;
            }
        }

        // Wheel: a linear fly along the focal axis (not a zoom).
        if wheel.abs() > 0.01 {
            cam.wheel_fly(wheel, scroll_speed);
        }

        animating
    }
}

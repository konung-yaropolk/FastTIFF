//! The 3D coordinate-box overlay: a white bounding box around the volume with
//! x/y/z tick coordinates (pixels, or calibrated length when the file carries a
//! physical unit). Drawn with the egui painter *on top of* the GPU ray-march,
//! projected with the same pinhole camera the shader uses
//! (`rd = forward + ndc.x·aspect·tan·right + ndc.y·tan·up`), so it lines up with
//! the rendered volume on both the wgpu and glow backends.

use super::camera::{volume_camera, VolumeCam, VolumeCamera};
use super::*;

impl ViewerApp {
    /// Draw the coordinate box for the current volume view into `painter` (which
    /// the caller has clipped to the volume rect). No-op without a loaded stack.
    pub(super) fn draw_coord_box(&self, painter: &egui::Painter, rect: egui::Rect) {
        let Some(loaded) = &self.stack else { return };
        let f0 = loaded.tiff.frames.first();
        let (Some(w), Some(h)) = (f0.map(|f| f.width), f0.map(|f| f.height)) else { return };
        let slices = loaded.tiff.meta.slices.max(1);
        let d = if slices > 1 { slices as u32 } else { loaded.tiff.meta.frames.max(1) as u32 };
        let dims = (w, h, d);

        // Rebuild the exact camera the shader used this frame (box half-extents
        // fold in the voxel scale, matching what's on screen).
        let cam = volume_camera(
            VolumeCam {
                yaw: self.vol_yaw,
                pitch: self.vol_pitch,
                dist: self.vol_dist,
                target: self.vol_target,
                fly_pos: self.vol_fly_pos,
                nav: self.nav_mode,
                scale: self.vol_scale,
                aspect: self.vol_aspect,
                render: self.vol_render,
                density: self.vol_density,
            },
            dims,
        );

        let unit = loaded.tiff.meta.unit.as_deref().filter(|u| !u.is_empty());
        draw_box(painter, rect, &cam, self.vol_aspect, dims, self.vol_scale, unit);
    }
}

/// Project a world-space point to screen pixels via the shader's pinhole model.
/// `None` when the point is at/behind the camera plane.
fn project(cam: &VolumeCamera, aspect: f32, rect: egui::Rect, p: [f32; 3]) -> Option<egui::Pos2> {
    let v = [p[0] - cam.eye[0], p[1] - cam.eye[1], p[2] - cam.eye[2]];
    let depth = v[0] * cam.forward[0] + v[1] * cam.forward[1] + v[2] * cam.forward[2];
    if depth <= 1e-4 {
        return None;
    }
    let a = v[0] * cam.right[0] + v[1] * cam.right[1] + v[2] * cam.right[2];
    let b = v[0] * cam.up[0] + v[1] * cam.up[1] + v[2] * cam.up[2];
    let ndc_x = a / (depth * cam.tan_half_fov * aspect);
    let ndc_y = b / (depth * cam.tan_half_fov);
    Some(egui::pos2(
        rect.center().x + ndc_x * rect.width() * 0.5,
        rect.center().y - ndc_y * rect.height() * 0.5, // ndc.y is up; screen y is down
    ))
}

/// The 8 box corners, indexed by bits (bit0=x, bit1=y, bit2=z): set bit = +half,
/// clear = -half. Corner 0 = the min corner (the coordinate origin end).
fn corner(he: [f32; 3], i: usize) -> [f32; 3] {
    [
        if i & 1 != 0 { he[0] } else { -he[0] },
        if i & 2 != 0 { he[1] } else { -he[1] },
        if i & 4 != 0 { he[2] } else { -he[2] },
    ]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// A "nice" tick step (1 / 2 / 5 × 10^k) giving roughly `target` intervals
/// across `span` — the standard axis-labelling heuristic (matplotlib / ImageJ's
/// 3D Viewer): so a ~102 unit axis ticks every 10, a ~204 one every 20, and a
/// ~1.04 one every 0.1.
fn nice_step(span: f32, target: f32) -> f32 {
    if span <= 0.0 || !span.is_finite() {
        return 1.0;
    }
    let raw = span / target.max(1.0);
    let mag = 10f32.powf(raw.log10().floor());
    let norm = raw / mag; // in [1, 10)
    let factor = if norm < 1.5 {
        1.0
    } else if norm < 3.0 {
        2.0
    } else if norm < 7.0 {
        5.0
    } else {
        10.0
    };
    factor * mag
}

/// Decimal places needed to print values stepped by `step` (0.1 → 1, 10 → 0).
fn step_decimals(step: f32) -> usize {
    (-step.log10().floor()).max(0.0) as usize
}

/// Draw `text` with a 1px dark halo (so it stays legible over bright volume
/// pixels), returning nothing.
fn label(painter: &egui::Painter, pos: egui::Pos2, align: egui::Align2, text: &str, font: &egui::FontId) {
    for off in [egui::vec2(1.0, 1.0), egui::vec2(-1.0, 1.0), egui::vec2(1.0, -1.0), egui::vec2(-1.0, -1.0)] {
        painter.text(pos + off, align, text, font.clone(), egui::Color32::from_black_alpha(180));
    }
    painter.text(pos, align, text, font.clone(), egui::Color32::WHITE);
}

#[allow(clippy::too_many_arguments)]
fn draw_box(
    painter: &egui::Painter,
    rect: egui::Rect,
    cam: &VolumeCamera,
    aspect: f32,
    dims: (u32, u32, u32),
    scale: [f32; 3],
    unit: Option<&str>,
) {
    let he = cam.box_he;
    let corners: [[f32; 3]; 8] = std::array::from_fn(|i| corner(he, i));
    let proj: Vec<Option<egui::Pos2>> = corners.iter().map(|&c| project(cam, aspect, rect, c)).collect();

    // 12 edges: corner pairs differing in exactly one axis bit. Skip an edge if
    // either endpoint is behind the camera (rare — only when zoomed inside).
    let stroke = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(230));
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            let j = i ^ bit;
            if i < j {
                if let (Some(a), Some(b)) = (proj[i], proj[j]) {
                    painter.line_segment([a, b], stroke);
                }
            }
        }
    }

    // Screen centroid of the box, so labels/ticks can be pushed outward (away
    // from the box) instead of overlapping the faces.
    let (mut sum, mut n) = (egui::Vec2::ZERO, 0.0f32);
    for p in proj.iter().flatten() {
        sum += p.to_vec2();
        n += 1.0;
    }
    let centroid = if n > 0.0 { (sum / n).to_pos2() } else { rect.center() };
    let outward = |p: egui::Pos2, perp: egui::Vec2| -> egui::Vec2 {
        if (p - centroid).dot(perp) < 0.0 {
            -perp
        } else {
            perp
        }
    };

    let font = egui::FontId::proportional(11.0);

    // Shared "0" at the origin corner (corner 0 = the min corner of all axes).
    if let Some(o) = proj[0] {
        label(painter, o, egui::Align2::RIGHT_BOTTOM, "0", &font);
    }

    // Nice-number ticks along each axis (x -> corner 1, y -> 2, z -> 4). The
    // step targets ~10 ticks over the axis's *display* extent (pixels, or
    // calibrated length). Only the final tick carries the unit, to avoid
    // crowding — matching the ImageJ 3D Viewer's axis labels.
    let suffix: &str = match unit {
            Some(u) => u,
            None => "px",
        };
    let axes: [(usize, String, u32, f32); 3] =
        [(1, format!("x, {suffix}"), dims.0, scale[0]), (2, format!("y, {suffix}"), dims.1, scale[1]), (4, format!("z, {suffix}"), dims.2, scale[2])];

    for (end_idx, ax_label, npix, sc) in axes {
        let calibrated = unit.is_some();
        let extent = if calibrated { npix as f32 * sc } else { npix as f32 };
        if extent <= 0.0 || !extent.is_finite() {
            continue;
        }
        let step = nice_step(extent, 10.0);
        let dec = step_decimals(step);

        // Screen perpendicular of this edge, for the tick marks + label offset.
        let perp = match (proj[0], proj[end_idx]) {
            (Some(a), Some(b)) if a.distance(b) > 1.0 => {
                let dir = (b - a) / a.distance(b);
                egui::vec2(-dir.y, dir.x)
            }
            _ => egui::vec2(0.0, -1.0),
        };

        let count = (extent / step).floor() as i32;
        for k in 1..=count {
            let value = k as f32 * step;
            let world = lerp3(corners[0], corners[end_idx], value / extent);
            let Some(p) = project(cam, aspect, rect, world) else { continue };
            painter.line_segment([p - perp * 3.0, p + perp * 3.0], stroke);
            let text = format!("{value:.dec$}");
            label(painter, p + outward(p, perp) * 9.0, egui::Align2::CENTER_CENTER, &text, &font);
        }

        // Axis letter beyond the far corner.
        if let Some(p) = proj[end_idx] {
            label(painter, p + outward(p, perp) * 18.0, egui::Align2::CENTER_CENTER, &ax_label, &font);
        }
    }
}

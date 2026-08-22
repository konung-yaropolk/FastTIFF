//! Planning which part of an over-large frame stays resident on the GPU.
//!
//! Three properties have to hold for *every* view of *every* frame on *every*
//! device, and getting any of them wrong is a visible bug rather than a
//! slowdown:
//!
//! 1. the window fits the device — otherwise the texture is rejected, which is
//!    fatal, which is the crash this whole path exists to stop;
//! 2. the window covers what is on screen — otherwise the user sees the edge of
//!    the resident data, which looks like the image is torn;
//! 3. the UV mapping puts the visible region exactly where it was asked for —
//!    otherwise the picture is subtly displaced, which is worse than an obvious
//!    failure because it looks like data.
//!
//! The sweeps at the bottom check all three across the parameter space. The
//! named tests above them are the cases worth reading.

use fast_tiff_viewer::roi::{extract, plan, Roi, MAX_ROI_BYTES};

/// A viewport large enough that it never drives the stride — so the tests below
/// exercise the texture-budget path unless they say otherwise.
const HUGE_PANEL: [f32; 2] = [100_000.0, 100_000.0];

/// Shorthand: the window to upload for a view.
fn resident(w: u32, h: u32, off: [f32; 2], scale: [f32; 2], max_axis: u32, bpt: usize) -> Roi {
    plan(w, h, off, scale, HUGE_PANEL, max_axis, bpt).resident
}

/// The file this feature exists for, and a common device limit.
const ANDROMEDA: (u32, u32) = (40_000, 12_788);
/// Three 8-bit channels, the Andromeda case.
const RGB8: usize = 3;

/// Everything visible, i.e. zoomed out to fit.
const WHOLE: ([f32; 2], [f32; 2]) = ([0.0, 0.0], [1.0, 1.0]);

#[test]
fn a_frame_that_fits_stays_whole_and_full_resolution() {
    let roi = resident(2048, 1024, WHOLE.0, WHOLE.1, 16_384, RGB8);
    assert_eq!(roi, Roi { x: 0, y: 0, w: 2048, h: 1024, stride: 1 });
    assert_eq!(roi.texture_size(), (2048, 1024));
}

/// The point of the whole exercise: zoomed in, a gigapixel frame is shown at
/// the file's own resolution. Stride 1 means no detail was thrown away.
#[test]
fn zoomed_in_the_giant_mosaic_is_full_resolution() {
    let (w, h) = ANDROMEDA;
    // A ~2000 x 1200 window near the middle — roughly 100% zoom on a laptop.
    let uv_off = [0.5, 0.4];
    let uv_scale = [2000.0 / w as f32, 1200.0 / h as f32];
    let roi = resident(w, h, uv_off, uv_scale, 32_768, RGB8);

    assert_eq!(roi.stride, 1, "at this zoom nothing needs to be discarded");
    let (tw, th) = roi.texture_size();
    assert!(tw <= 32_768 && th <= 32_768, "{tw}x{th} does not fit");
    // And it holds more than the screen needs, so panning does not re-upload.
    assert!(roi.w > 2000 && roi.h > 1200, "no margin was left for panning: {roi:?}");
}

/// Zoomed out, the same frame becomes an overview: coarse, but whole.
#[test]
fn zoomed_out_the_giant_mosaic_becomes_a_coarse_overview() {
    let (w, h) = ANDROMEDA;
    let roi = resident(w, h, WHOLE.0, WHOLE.1, 32_768, RGB8);

    assert!(roi.stride > 1, "40000 pixels cannot be shown at full resolution at once");
    assert_eq!((roi.x, roi.y), (0, 0));
    assert_eq!((roi.w, roi.h), (w, h), "the overview should cover the whole frame");
    let (tw, th) = roi.texture_size();
    assert!(tw <= 32_768 && th <= 32_768);
}

/// Panning by a little must not move the resident window, or every drag frame
/// would re-upload. This is what the margin is for.
#[test]
fn a_small_pan_is_served_by_the_window_already_resident() {
    let (w, h) = ANDROMEDA;
    let scale = [2000.0 / w as f32, 1200.0 / h as f32];
    let before = plan(w, h, [0.5, 0.4], scale, HUGE_PANEL, 32_768, RGB8).resident;
    // Pan by a quarter of the visible width. What matters is whether the window
    // already on the GPU still covers what the new view *needs* — not whether
    // it matches the new margined window, which moves with every pixel of pan.
    let after = plan(w, h, [0.5 + scale[0] * 0.25, 0.4], scale, HUGE_PANEL, 32_768, RGB8);
    assert!(before.serves(&after.required), "{before:?} should already cover {:?}", after.required);
}

/// Panning a long way must move it, or the view would run off the resident
/// data — the failure that looks like the image is torn.
#[test]
fn a_large_pan_needs_a_new_window() {
    let (w, h) = ANDROMEDA;
    let scale = [2000.0 / w as f32, 1200.0 / h as f32];
    let before = plan(w, h, [0.1, 0.4], scale, HUGE_PANEL, 32_768, RGB8).resident;
    let after = plan(w, h, [0.8, 0.4], scale, HUGE_PANEL, 32_768, RGB8);
    assert!(
        !before.serves(&after.required),
        "a pan across the frame cannot be served from {before:?}"
    );
}

/// Zooming in has to re-cut even when the coarse window already covers the
/// area. This is the difference between the feature working and not: the
/// overview covers everything, so a containment test that ignored resolution
/// would say it serves every view, and zooming in would never sharpen.
#[test]
fn a_coarse_window_does_not_serve_a_view_that_wants_full_resolution() {
    let coarse = Roi { x: 0, y: 0, w: 40_000, h: 12_788, stride: 4 };
    let sharp = Roi { x: 1_000, y: 1_000, w: 2_000, h: 1_000, stride: 1 };
    assert!(
        coarse.x <= sharp.x && coarse.x + coarse.w >= sharp.x + sharp.w,
        "the coarse window does contain the area, which is what makes this the trap"
    );
    assert!(!coarse.serves(&sharp), "covering the area is not enough — the resolution differs");
    // ...and the same window does serve a view at its own resolution.
    let same_res = Roi { x: 1_000, y: 1_000, w: 2_000, h: 1_000, stride: 4 };
    assert!(coarse.serves(&same_res));
}

/// The end-to-end version of the same thing, through the planner: zooming into
/// a frame that opened coarse must ask for a finer window than the one on
/// screen.
#[test]
fn zooming_in_asks_for_a_finer_window_than_the_overview() {
    let (w, h) = ANDROMEDA;
    let overview = plan(w, h, WHOLE.0, WHOLE.1, HUGE_PANEL, 32_768, RGB8).resident;
    let zoomed = plan(w, h, [0.5, 0.4], [2000.0 / w as f32, 1200.0 / h as f32], HUGE_PANEL, 32_768, RGB8);
    assert!(overview.stride > zoomed.required.stride, "zooming in should sharpen");
    assert!(!overview.serves(&zoomed.required), "the overview must not satisfy a zoomed-in view");
}

/// The sampling grid must not shift as the window moves. If it did, panning
/// would resample the same detail at a different phase each time and fine
/// structure would crawl.
#[test]
fn the_window_origin_is_aligned_to_the_sampling_grid() {
    let (w, h) = ANDROMEDA;
    for i in 0..40 {
        let uv = [i as f32 / 40.0, 0.3];
        let roi = resident(w, h, uv, [0.05, 0.2], 4096, RGB8);
        assert_eq!(roi.x % roi.stride, 0, "x {} not aligned to stride {}", roi.x, roi.stride);
        assert_eq!(roi.y % roi.stride, 0, "y {} not aligned to stride {}", roi.y, roi.stride);
    }
}

/// A float stack costs four times an 8-bit one per texel, so the same view has
/// to be coarser to stay inside the same VRAM budget. Getting this wrong is how
/// you exhaust VRAM on exactly the machines least able to spare it.
#[test]
fn wider_channels_buy_a_coarser_window_for_the_same_budget() {
    let (w, h) = ANDROMEDA;
    let cheap = resident(w, h, WHOLE.0, WHOLE.1, 32_768, 3); // 3 x u8
    let dear = resident(w, h, WHOLE.0, WHOLE.1, 32_768, 12); // 3 x f32
    assert!(dear.stride >= cheap.stride, "float channels should not be finer: {dear:?} vs {cheap:?}");
    for roi in [cheap, dear] {
        let (tw, th) = roi.texture_size();
        assert!((tw as usize) * (th as usize) * 3 <= MAX_ROI_BYTES || roi.stride > 1);
    }
}

/// The UV mapping is what puts the picture where the frontend asked for it.
/// A window covering the whole frame at stride 1 must be the identity, or every
/// ordinary image is displaced.
#[test]
fn a_full_window_maps_uvs_unchanged() {
    let roi = Roi::full(2048, 1024);
    let (off, scale) = roi.map_uv(2048, 1024, [0.25, 0.5], [0.5, 0.25]);
    assert!((off[0] - 0.25).abs() < 1e-6 && (off[1] - 0.5).abs() < 1e-6, "{off:?}");
    assert!((scale[0] - 0.5).abs() < 1e-6 && (scale[1] - 0.25).abs() < 1e-6, "{scale:?}");
}

/// A partial window has to rescale the request onto its own texture: the
/// visible region should land at the same image pixels it names, expressed in
/// texture coordinates.
#[test]
fn a_partial_window_maps_the_visible_region_onto_its_texture() {
    let (w, h) = (40_000u32, 12_788u32);
    let roi = Roi { x: 10_000, y: 2_000, w: 4_000, h: 2_000, stride: 1 };
    // Ask for image pixels 11000..13000 horizontally.
    let uv_off = [11_000.0 / w as f32, 2_500.0 / h as f32];
    let uv_scale = [2_000.0 / w as f32, 1_000.0 / h as f32];
    let (off, scale) = roi.map_uv(w, h, uv_off, uv_scale);

    // In texture space that is (11000-10000)/4000 = 0.25, spanning 2000/4000.
    assert!((off[0] - 0.25).abs() < 1e-4, "x offset {} should be 0.25", off[0]);
    assert!((scale[0] - 0.5).abs() < 1e-4, "x scale {} should be 0.5", scale[0]);
    assert!((off[1] - 0.25).abs() < 1e-4, "y offset {} should be 0.25", off[1]);
    assert!((scale[1] - 0.5).abs() < 1e-4, "y scale {} should be 0.5", scale[1]);
}

/// A stride divides the rect into whole texels, so the texture covers slightly
/// more than the rect when it does not divide evenly. The mapping must divide
/// by what the texture *covers*, not by the rect, or the picture drifts by up
/// to a texel — small, but it is a wrong answer that looks like a right one.
#[test]
fn the_mapping_divides_by_the_covered_span_not_the_rect() {
    let roi = Roi { x: 0, y: 0, w: 999, h: 999, stride: 4 };
    let (tw, _) = roi.texture_size();
    assert_eq!(tw, 250, "999 pixels at stride 4 needs 250 texels");
    assert_eq!(roi.covered_span().0, 1000, "those texels cover 1000 pixels, not 999");

    // The pixel at the far edge of the covered span maps to exactly 1.0.
    let (off, _) = roi.map_uv(1000, 1000, [1.0, 0.0], [0.0, 0.0]);
    assert!((off[0] - 1.0).abs() < 1e-6, "{off:?}");
}

/// Property sweep: whatever the view and whatever the device, the window fits,
/// covers the visible region, and stays inside the frame.
#[test]
fn every_view_yields_a_window_that_fits_and_covers() {
    let frames = [(64u32, 64u32), (4096, 4096), (40_000, 12_788), (200_000, 3), (3, 200_000)];
    let limits = [1u32, 512, 4096, 16_384, 32_768];
    let zooms = [1.0f32, 0.5, 0.1, 0.01, 0.001];
    let positions = [0.0f32, 0.13, 0.5, 0.87, 0.999];

    for &(w, h) in &frames {
        for &max_axis in &limits {
            for &z in &zooms {
                for &p in &positions {
                    let uv_off = [p * (1.0 - z), p * (1.0 - z)];
                    let uv_scale = [z, z];
                    let roi = resident(w, h, uv_off, uv_scale, max_axis, RGB8);
                    let case = format!("{w}x{h} limit {max_axis} zoom {z} at {p} -> {roi:?}");

                    // Fits the device. `plan` floors the limit at 2 — a GPU that
                    // cannot hold a 2x2 texture cannot run the renderer at all —
                    // so that floor is what a limit of 1 is held to. The value is
                    // still swept, to check it does not panic or divide by zero.
                    let effective = max_axis.max(2);
                    let (tw, th) = roi.texture_size();
                    assert!(tw <= effective && th <= effective, "does not fit: {case}");
                    assert!(tw >= 1 && th >= 1, "empty texture: {case}");

                    // Inside the frame.
                    assert!(roi.x + roi.w <= w.max(1), "runs off the right: {case}");
                    assert!(roi.y + roi.h <= h.max(1), "runs off the bottom: {case}");

                    // Covers what is on screen. Computed the same way `plan`
                    // does, so this checks the window against the request
                    // rather than restating the arithmetic.
                    let vx = (uv_off[0] * w as f32).clamp(0.0, w as f32).floor() as u32;
                    let vy = (uv_off[1] * h as f32).clamp(0.0, h as f32).floor() as u32;
                    let vx1 = ((uv_off[0] + uv_scale[0]) * w as f32).clamp(0.0, w as f32).ceil() as u32;
                    let vy1 = ((uv_off[1] + uv_scale[1]) * h as f32).clamp(0.0, h as f32).ceil() as u32;
                    assert!(roi.x <= vx.min(w - 1), "left edge uncovered: {case}");
                    assert!(roi.y <= vy.min(h - 1), "top edge uncovered: {case}");
                    assert!(roi.x + roi.w >= vx1.min(w), "right edge uncovered: {case}");
                    assert!(roi.y + roi.h >= vy1.min(h), "bottom edge uncovered: {case}");

                    // Aligned to the sampling grid.
                    assert_eq!(roi.x % roi.stride, 0, "x unaligned: {case}");
                    assert_eq!(roi.y % roi.stride, 0, "y unaligned: {case}");
                }
            }
        }
    }
}

/// Property sweep for the mapping: the visible region must land within the
/// texture, for every window the planner produces.
#[test]
fn the_mapped_uvs_always_land_inside_the_resident_texture() {
    let (w, h) = ANDROMEDA;
    for &z in &[1.0f32, 0.25, 0.02] {
        for i in 0..20 {
            let p = i as f32 / 20.0;
            let uv_off = [p * (1.0 - z), p * (1.0 - z)];
            let uv_scale = [z, z];
            let roi = resident(w, h, uv_off, uv_scale, 32_768, RGB8);
            let (off, scale) = roi.map_uv(w, h, uv_off, uv_scale);
            let case = format!("zoom {z} at {p} -> {roi:?} maps to {off:?} + {scale:?}");
            // A texel of slack: the rect is rounded outward from the request.
            let slack = 1.5 / roi.texture_size().0 as f32;
            assert!(off[0] >= -slack && off[1] >= -slack, "samples before the texture: {case}");
            assert!(off[0] + scale[0] <= 1.0 + slack, "samples past the right edge: {case}");
            assert!(off[1] + scale[1] <= 1.0 + slack, "samples past the bottom edge: {case}");
        }
    }
}

// ---------------------------------------------------------------------------
// Cutting the window out of a decoded plane
// ---------------------------------------------------------------------------

/// The plain case: a window at the origin, every pixel.
#[test]
fn extract_at_full_resolution_is_the_rows_it_names() {
    // 4x4 counting up; take the top-left 2x2.
    let src: Vec<u8> = (0..16).collect();
    let roi = Roi { x: 0, y: 0, w: 2, h: 2, stride: 1 };
    assert_eq!(extract(&src, 4, 4, &roi), vec![0, 1, 4, 5]);
}

/// A window away from the origin: the offset has to apply per row, which is
/// where a naive flat copy goes wrong.
#[test]
fn extract_takes_the_window_from_the_right_place() {
    let src: Vec<u8> = (0..16).collect();
    let roi = Roi { x: 2, y: 1, w: 2, h: 2, stride: 1 };
    assert_eq!(extract(&src, 4, 4, &roi), vec![6, 7, 10, 11]);
}

#[test]
fn extract_takes_every_nth_sample_when_strided() {
    let src: Vec<u16> = (0..16).collect();
    let roi = Roi { x: 0, y: 0, w: 4, h: 4, stride: 2 };
    assert_eq!(extract(&src, 4, 4, &roi), vec![0, 2, 8, 10]);
}

/// Stride and offset together, on a rect the stride does not divide.
#[test]
fn extract_handles_an_offset_window_that_the_stride_does_not_divide() {
    // 5 wide; take x from 1, 3 columns at stride 2 -> columns 1 and 3.
    let src: Vec<u8> = (0..15).collect();
    let roi = Roi { x: 1, y: 0, w: 3, h: 3, stride: 2 };
    assert_eq!(roi.texture_size(), (2, 2));
    assert_eq!(extract(&src, 5, 3, &roi), vec![1, 3, 11, 13]);
}

/// The output is always exactly the texture size. A short buffer is a GPU
/// validation failure, which is fatal — the failure mode this whole module
/// exists to prevent — so a plane that came back short is padded, not trusted.
#[test]
fn extract_always_fills_the_texture_even_from_a_truncated_plane() {
    let roi = Roi { x: 0, y: 0, w: 4, h: 4, stride: 1 };
    let (tw, th) = roi.texture_size();
    for len in [0usize, 1, 7, 15, 16] {
        let src: Vec<u8> = (0..len as u8).collect();
        let out = extract(&src, 4, 4, &roi);
        assert_eq!(out.len(), tw as usize * th as usize, "wrong length from a {len}-sample plane");
    }
}

/// Missing data reads as empty rather than as a smear of whatever happened to
/// be at the truncation point.
#[test]
fn extract_pads_missing_data_with_zero() {
    let src: Vec<u8> = vec![9; 4];
    let roi = Roi { x: 0, y: 0, w: 4, h: 4, stride: 1 };
    let out = extract(&src, 4, 4, &roi);
    assert_eq!(&out[..4], &[9, 9, 9, 9], "the row that is there survives");
    assert!(out[4..].iter().all(|&v| v == 0), "the rest should be empty: {out:?}");
}

/// Every window the planner produces must cut to exactly its texture size, for
/// any plane length. This is the pairing that matters: the plan sizes the
/// texture and the cut fills it, and if the two ever disagree the upload is
/// rejected.
#[test]
fn every_planned_window_cuts_to_exactly_its_texture() {
    let (w, h) = (600u32, 400u32);
    let src: Vec<u8> = (0..(w * h) as usize).map(|i| i as u8).collect();
    for &z in &[1.0f32, 0.5, 0.2, 0.05] {
        for i in 0..8 {
            let p = i as f32 / 8.0;
            let roi = resident(w, h, [p * (1.0 - z), p * (1.0 - z)], [z, z], 64, RGB8);
            let (tw, th) = roi.texture_size();
            let out = extract(&src, w, h, &roi);
            assert_eq!(
                out.len(),
                tw as usize * th as usize,
                "zoom {z} at {p}: {roi:?} wanted {tw}x{th} but cut {} samples",
                out.len()
            );
        }
    }
}

/// The guard that keeps all of this from costing anything on ordinary images.
///
/// A frame the device can hold takes the original path — decoded whole,
/// uploaded whole, no window planned and no copy made. Only a frame past the
/// texture limit pays. Worth pinning rather than leaving implicit, because the
/// cost of getting it wrong is a slower viewer for everyone in exchange for a
/// feature almost nobody needs.
#[test]
fn only_frames_past_the_texture_limit_need_a_window() {
    use fast_tiff_viewer::roi::needs_window;
    for (w, h) in [(1u32, 1u32), (512, 512), (4096, 4096), (16_384, 16_384)] {
        assert!(!needs_window(w, h, 16_384), "{w}x{h} fits a 16k device and must not be windowed");
    }
    assert!(needs_window(16_385, 16_384, 16_384), "one texel over on either axis is over");
    assert!(needs_window(16_384, 16_385, 16_384));
    assert!(needs_window(40_000, 12_788, 32_768), "the mosaic this exists for");
    assert!(!needs_window(40_000, 12_788, 65_536), "...but not on a device that could hold it");
}

// ---------------------------------------------------------------------------
// Sampling no finer than the display can show
// ---------------------------------------------------------------------------

/// A typical panel, in physical pixels.
const PANEL: [f32; 2] = [1900.0, 1000.0];

/// The whole point: showing a 40000-pixel mosaic on a 1900-pixel panel should
/// not build a texture with twenty thousand columns. Detail the display cannot
/// draw is invisible, and paid for on every zoom step.
#[test]
fn a_zoomed_out_view_is_sampled_for_the_screen_not_the_file() {
    let (w, h) = ANDROMEDA;
    let r = plan(w, h, WHOLE.0, WHOLE.1, PANEL, 32_768, RGB8).resident;
    let (tw, th) = r.texture_size();

    assert!(r.stride >= 8, "40000 pixels on a 1900-pixel panel needs heavy subsampling, got {r:?}");
    assert!(tw <= 4_000, "texture is {tw} wide for a {} pixel panel", PANEL[0]);
    let mb = (tw as f64 * th as f64 * RGB8 as f64) / 1e6;
    assert!(mb < 32.0, "{mb:.0} MB of texture for an overview is far too much");
}

/// ...but never *coarser* than the screen: the texture has to keep at least one
/// sample per pixel drawn, or the image looks soft.
#[test]
fn the_texture_always_has_at_least_a_sample_per_screen_pixel() {
    let (w, h) = ANDROMEDA;
    for &zoom in &[0.05f32, 0.1, 0.25, 0.5, 1.0, 2.0] {
        let vw = (PANEL[0] / zoom).min(w as f32);
        let vh = (PANEL[1] / zoom).min(h as f32);
        let uv_scale = [vw / w as f32, vh / h as f32];
        let uv_off = [(0.5 - uv_scale[0] / 2.0).max(0.0), (0.5 - uv_scale[1] / 2.0).max(0.0)];
        let r = plan(w, h, uv_off, uv_scale, PANEL, 32_768, RGB8).resident;

        // Texels covering the visible span, against the pixels drawing it.
        //
        // Past 100% zoom the file itself runs out: magnifying 950 source pixels
        // across 1900 screen pixels cannot yield 1900 texels, and stride 1 is
        // already everything there is. So the bar is the lesser of what the
        // screen can draw and what the image actually holds.
        let visible_texels = vw / r.stride as f32;
        let want = vw.min(PANEL[0]);
        assert!(
            visible_texels >= want * 0.9,
            "zoom {zoom}: {visible_texels:.0} texels where {want:.0} were available — too soft ({r:?})"
        );
    }
}

/// Zooming must cross only a handful of sampling levels. A stride free to take
/// any value changes on nearly every step, and each change re-decodes the
/// window — which is the stall this exists to prevent.
#[test]
fn zooming_crosses_only_a_few_sampling_levels() {
    let (w, h) = ANDROMEDA;
    let mut strides = Vec::new();
    for i in 0..40 {
        let zoom = 0.05 * 1.12f32.powi(i); // a smooth sweep from fit to ~4x
        let vw = (PANEL[0] / zoom).min(w as f32);
        let vh = (PANEL[1] / zoom).min(h as f32);
        let uv_scale = [vw / w as f32, vh / h as f32];
        let uv_off = [(0.5 - uv_scale[0] / 2.0).max(0.0), (0.5 - uv_scale[1] / 2.0).max(0.0)];
        strides.push(plan(w, h, uv_off, uv_scale, PANEL, 32_768, RGB8).resident.stride);
    }
    strides.dedup();
    assert!(
        strides.len() <= 8,
        "40 zoom steps produced {} distinct sampling levels ({strides:?}) — each one re-decodes",
        strides.len()
    );
    assert!(strides.iter().all(|s| s.is_power_of_two()), "levels should be powers of two: {strides:?}");
    // And they only ever get finer as you zoom in.
    assert!(strides.windows(2).all(|w| w[0] >= w[1]), "zooming in should not coarsen: {strides:?}");
}

/// No texture across the whole zoom range may be large enough to stall on.
#[test]
fn no_zoom_level_builds_a_huge_texture() {
    let (w, h) = ANDROMEDA;
    for i in 0..40 {
        let zoom = 0.05 * 1.12f32.powi(i);
        let vw = (PANEL[0] / zoom).min(w as f32);
        let vh = (PANEL[1] / zoom).min(h as f32);
        let uv_scale = [vw / w as f32, vh / h as f32];
        let uv_off = [(0.5 - uv_scale[0] / 2.0).max(0.0), (0.5 - uv_scale[1] / 2.0).max(0.0)];
        let r = plan(w, h, uv_off, uv_scale, PANEL, 32_768, RGB8).resident;
        let (tw, th) = r.texture_size();
        let mb = (tw as f64 * th as f64 * RGB8 as f64) / 1e6;
        assert!(mb < 96.0, "zoom {zoom:.3} builds {mb:.0} MB ({tw}x{th}, {r:?})");
    }
}

/// A frontend that has not said how big it is still gets a correct picture —
/// just a wastefully detailed one, bounded by the texture budget as before.
#[test]
fn an_unknown_viewport_falls_back_to_the_budget() {
    let (w, h) = ANDROMEDA;
    // The last is the interesting one: a sub-pixel width is not "unknown" but
    // is still nonsense, and taken at face value it asks for a stride of tens
    // of thousands — a 1x1 texture that would flash the moment the viewport
    // became real. Nonsense in, ordinary picture out.
    for panel in [[0.0f32, 0.0], [f32::NAN, 10.0], [-5.0, 5.0], [0.5, 1000.0]] {
        let r = plan(w, h, WHOLE.0, WHOLE.1, panel, 32_768, RGB8).resident;
        assert!(r.stride >= 1);
        let (tw, th) = r.texture_size();
        assert!(tw <= 32_768 && th <= 32_768, "still has to fit the device: {r:?}");
        assert!(
            tw > 16 && th > 16,
            "a nonsense viewport {panel:?} collapsed the texture to {tw}x{th}"
        );
    }
}

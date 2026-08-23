//! Which part of an over-large frame is resident on the GPU.
//!
//! A GPU caps a 2D texture at some per-axis size — 16384 and 32768 are the
//! common ones. Mosaics run past it (Hubble publishes Andromeda at
//! 40000 x 12788), and such a frame cannot become a texture at all.
//!
//! Subsampling the whole frame to fit is one answer, and the one this viewer
//! used first, but it throws away the detail that is the entire reason to open
//! a gigapixel image. What a slide viewer does instead is keep only the part
//! being *looked at* resident, at whatever resolution the current zoom asks
//! for. That is what this module plans: a rectangle of source pixels plus a
//! stride, sized to what the device can hold.
//!
//! The payoff is that zoom decides quality rather than file size. Zoomed out,
//! the window is the whole frame at a coarse stride — an overview. Zoomed in,
//! it is a few thousand pixels at stride 1, which is the file's own resolution,
//! however large the file happens to be.
//!
//! One resident window rather than a grid of tiles. A grid buys the ability to
//! keep distant tiles around for a later pan; with the decoded frame already in
//! memory, re-extracting a window costs a strided copy and no decode, so the
//! grid would add bookkeeping and an eviction policy to buy back something that
//! is already cheap. The margin below is what keeps ordinary panning from
//! touching the GPU at all.
//!
//! Everything here is computed in **texels**, not pixels. The rect has to be
//! aligned to the sampling grid, has to cover the visible region, and has to
//! fit a texel count — three constraints that interact through rounding, and
//! reasoning about them in pixels means chasing off-by-one errors that show up
//! as a texture the device rejects (fatal) or a seam at the edge of the view.
//! In texel space each is exact.

/// Bytes of texture the resident window may occupy, summed across channels.
///
/// A budget rather than an axis count because what matters is VRAM, and a
/// channel's cost depends on its format: three 8-bit channels and three float
/// ones differ fourfold for the same picture. Sized to be generous on any GPU
/// that can open a file this large in the first place while leaving room for
/// the rest of the app.
pub const MAX_ROI_BYTES: usize = 512 << 20;

/// How an over-large frame is kept on screen.
///
/// Both schemes show a window of the frame — the frame cannot become a texture
/// whole either way. They differ in *which* windows they are willing to build,
/// and that is the whole trade.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LargeImageMode {
    /// Two levels and no others: one coarse copy of the whole frame, built once
    /// and kept, plus a full-resolution crop of whatever is being looked at.
    ///
    /// Zooming therefore crosses a single boundary rather than a dozen, and the
    /// coarse level is decoded once for the life of the file instead of being
    /// rebuilt at every intermediate resolution. Two things follow. Panning and
    /// zooming out never decode at all — the retained level already spans the
    /// frame, so it serves any view. And the coarse level is the *finest* one
    /// that fits, usually half scale, where point-sampling a regular structure
    /// aliases far less than the eighth or sixteenth scale a continuous scheme
    /// picks for the same view.
    ///
    /// Paid for in RAM: that level is held for as long as the file is open, up
    /// to [`MAX_PRELOAD_BYTES`].
    Preload,
    /// Resolution follows the zoom continuously: every view gets a window at
    /// whatever sampling the display can actually resolve, and no more.
    ///
    /// Holds far less — only the window being looked at, plus a small retained
    /// overview — and never spends detail the screen cannot show. The cost is
    /// that a zoom gesture crosses many sampling levels and each one is a fresh
    /// decode of the region, so zooming is where the CPU goes; and the coarse
    /// levels it picks when zoomed out alias more than a half-scale copy would.
    ///
    /// The default, because it is the one that cannot fail on the images this
    /// path exists for. Its memory follows the viewport rather than the file,
    /// so a frame of any size opens; `Preload` holds a reduced copy of the
    /// whole frame, and on a large enough file the finest level that fits is
    /// still hundreds of megabytes to hold for as long as the file is open.
    /// Smoothness is worth choosing, but not by default and not silently.
    #[default]
    Tiled,
}

/// What the device can hold, and how the user wants it spent.
///
/// The three travel together because no one of them decides anything alone:
/// the texture limit and the per-texel cost jointly bound a window, and the
/// mode decides which windows are allowed to be considered in the first place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// The device's per-axis 2D texture limit.
    pub max_axis: u32,
    /// Summed per-texel cost of the resident channels, in bytes. A channel's
    /// cost depends on its format, so three 8-bit channels and three float ones
    /// differ fourfold for the same picture.
    pub bytes_per_texel: usize,
    pub mode: LargeImageMode,
}

impl Budget {
    pub fn new(max_axis: u32, bytes_per_texel: usize, mode: LargeImageMode) -> Self {
        Budget { max_axis, bytes_per_texel, mode }
    }
}

/// Bytes the preloaded coarse level may occupy, in RAM and again in VRAM.
///
/// This is [`LargeImageMode::Preload`]'s side of the bargain, so it is
/// deliberately generous where [`crate::window::OVERVIEW_MAX_BYTES`] — which
/// bounds the overview that `Tiled` retains incidentally — is deliberately not.
/// It also has to agree with what [`overview_stride`] plans: a level the
/// planner chooses but the retainer then refuses would be rebuilt on every
/// frame, which is worse than either mode.
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_PRELOAD_BYTES: usize = 512 << 20;
/// Smaller in a browser, where the whole file is already resident in a 32-bit
/// address space and a half-gigabyte plane on top of it would abort the module.
#[cfg(target_arch = "wasm32")]
pub const MAX_PRELOAD_BYTES: usize = 96 << 20;

/// The sampling of the preloaded coarse level: the *finest* power-of-two
/// sampling at which the whole frame fits both the device's texture limit and
/// [`MAX_PRELOAD_BYTES`].
///
/// Finest rather than coarsest because aliasing is the thing being avoided.
/// Point-sampling every second pixel of a regular structure — a sensor grid, a
/// stitched mosaic's seams — is visibly moiré-free next to sampling every
/// sixteenth, and a level twice as fine costs four times the memory, which is
/// what the budget is for.
///
/// Powers of two only, so the level is a clean 1/2, 1/4, 1/8 of the original.
pub fn overview_stride(
    width: u32,
    height: u32,
    max_axis: u32,
    bytes_per_texel: usize,
) -> u32 {
    let max_axis = max_axis.max(2);
    let bytes_per_texel = bytes_per_texel.max(1);
    let mut stride = 2;
    loop {
        let (tw, th) = (width.div_ceil(stride).max(1), height.div_ceil(stride).max(1));
        let fits = tw <= max_axis
            && th <= max_axis
            && (tw as usize) * (th as usize) * bytes_per_texel <= MAX_PRELOAD_BYTES;
        // The guard is for a frame so large that no sampling fits; there is
        // nothing coarser left to try, and doubling for ever would hang.
        if fits || stride >= 1 << 16 {
            return stride;
        }
        stride *= 2;
    }
}

/// How much bigger than the visible region the resident window is made, when
/// there is texture budget spare.
///
/// Entirely about not re-uploading during a drag: at 1 every pan would leave
/// the resident window immediately and the upload would run every frame. At 2
/// the window is twice the visible extent on each axis, so panning stays inside
/// it for most of a screen's movement in any direction.
const MARGIN: u32 = 2;

/// Whether a frame needs any of this at all.
///
/// The answer is no for very nearly every image, and that is the point. A frame
/// the device can hold takes the path it always did — decoded whole, uploaded
/// whole, no planning, no cropping, not one extra copy — so the machinery for
/// gigapixel mosaics costs nothing on the images that do not need it. Only a
/// frame past the per-axis texture limit, which cannot become a texture at all,
/// is worth paying for a window.
pub fn needs_window(width: u32, height: u32, max_axis: u32) -> bool {
    width > max_axis || height > max_axis
}

/// A window of source pixels resident on the GPU: the rect `(x, y, w, h)` in
/// image pixels, sampled every `stride`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// Subsampling within the rect. `1` is the file's own resolution.
    pub stride: u32,
}

/// What a view needs, and what to keep resident to serve it.
///
/// Two rects rather than one because they answer different questions, and
/// conflating them is a bug in both directions: comparing against the margined
/// rect re-uploads on every small pan (the margin then buys nothing), while
/// uploading only the required rect leaves no margin at all. They are returned
/// together, from one stride, so the two can never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency {
    /// The window to upload: the visible region plus margin.
    pub resident: Roi,
    /// The least that would serve this view. A window already on the GPU that
    /// [`serves`](Roi::serves) this one needs no upload.
    pub required: Roi,
}

/// What to draw now, and what to be preparing.
///
/// `show` is on the GPU (or can be put there at once); `want` is what the view
/// actually calls for. They differ only while a finer window is being built,
/// and that gap is the entire point — it is what lets the picture stay up
/// instead of the program stopping until the new window is ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Serving {
    /// The window to draw from.
    pub show: Roi,
    /// The window the view really wants. Equal to `show` when there is nothing
    /// better to prepare.
    pub want: Roi,
    /// Whether `show` can be produced now. `false` means it is already on the
    /// GPU and `want` should be built in the background.
    pub ready: bool,
}

/// Decide what to draw, given what is resident, what coarse overview is held,
/// and what the view now calls for.
///
/// Four cases, in order of how little work they need:
///
/// 1. What is resident [`serves`](Roi::serves) the view outright — the common
///    result of a small pan. Nothing is decoded and the GPU is not touched.
/// 2. It covers the same ground at the wrong resolution — a zoom in. It is kept
///    on screen, scaled, while the sharper window is built off-thread. Soft for
///    a moment beats stopped for a moment, and stopped is what this costs if
///    the two cases are merged: a zoom would then blank and wait like case 4.
/// 3. It does not cover the view at all, but the retained overview does — a
///    zoom *out*, or a jump clear across the image at full resolution. A fine
///    window covers too little ground to be stretched over a wider view, so
///    without this case there is nothing to draw and the rebuild has to be
///    waited for. The overview is already decoded, so showing it costs an
///    upload rather than a decode.
/// 4. Nothing usable at all — the first view of a file. There is no picture to
///    keep showing, so this one is waited for.
///
/// `overview` must already have been checked against the current frame, channel
/// set and formats by the caller. A stale one reaching here is worse than none:
/// it would steer `show` at a window nobody holds the pixels for.
///
/// The `show != plan.resident` guard on case 3 is what keeps the fit-to-window
/// view out of it. Zoomed fully out the planned window *is* the overview, and
/// there is nothing better to prepare; returning `ready: false` there would
/// leave `want == show`, so the caller would queue no build and request no
/// repaint, and the picture would stay coarse for ever.
///
/// Pure, and separate from the GPU for that reason: which case applies decides
/// whether zooming a gigapixel image stalls, and that should be checkable
/// without a display.
pub fn serve(current: Option<Roi>, overview: Option<Roi>, plan: &Residency) -> Serving {
    if let Some(show) = current {
        if show.serves(&plan.required) {
            return Serving { show, want: show, ready: true };
        }
        if show.covers_area_of(&plan.required) {
            return Serving { show, want: plan.resident, ready: false };
        }
    }
    if let Some(show) = overview {
        if show != plan.resident && show.covers_area_of(&plan.required) {
            return Serving { show, want: plan.resident, ready: false };
        }
    }
    Serving { show: plan.resident, want: plan.resident, ready: true }
}

impl Roi {
    /// The whole frame at full resolution — what every frame small enough to
    /// upload outright uses.
    pub fn full(width: u32, height: u32) -> Self {
        Roi { x: 0, y: 0, w: width.max(1), h: height.max(1), stride: 1 }
    }

    /// Texture dimensions this window uploads to.
    pub fn texture_size(&self) -> (u32, u32) {
        (self.w.div_ceil(self.stride).max(1), self.h.div_ceil(self.stride).max(1))
    }

    /// Image-pixel span the texture covers, which is the rect rounded *up* to
    /// whole texels. Slightly larger than `w`/`h` when the stride does not
    /// divide them, and it is this span — not the rect — that the UV mapping
    /// has to divide by, or the picture drifts by up to one texel.
    pub fn covered_span(&self) -> (u32, u32) {
        let (tw, th) = self.texture_size();
        (tw * self.stride, th * self.stride)
    }

    /// Whether this window can serve a view wanting `other`: same resolution,
    /// and covering all of it.
    pub fn serves(&self, other: &Roi) -> bool {
        self.stride == other.stride
            && self.x <= other.x
            && self.y <= other.y
            && self.x + self.w >= other.x + other.w
            && self.y + self.h >= other.y + other.h
    }

    /// Whether this window covers all of `other`'s area, at whatever
    /// resolution.
    ///
    /// Weaker than [`serves`](Self::serves), which also demands the right
    /// resolution. This is the question "can I keep showing what I have while a
    /// better one is prepared" — a coarser window covering the same ground
    /// looks soft but is otherwise right, where one that does not cover it
    /// shows the edge of the data.
    pub fn covers_area_of(&self, other: &Roi) -> bool {
        self.x <= other.x
            && self.y <= other.y
            && self.x + self.w >= other.x + other.w
            && self.y + self.h >= other.y + other.h
    }

    /// Map an image-normalised UV window — what a frontend computes from its
    /// own pan and zoom, in units of the whole frame — onto this window's
    /// texture.
    ///
    /// Keeping the frontend in image coordinates is deliberate: it already
    /// thinks in "which part of the picture am I showing", and a second
    /// frontend would have to learn about residency to do anything else. The
    /// translation belongs here, where residency is decided.
    pub fn map_uv(
        &self,
        width: u32,
        height: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) -> ([f32; 2], [f32; 2]) {
        let (span_x, span_y) = self.covered_span();
        let (sx, sy) = (span_x.max(1) as f32, span_y.max(1) as f32);
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        (
            [(uv_offset[0] * w - self.x as f32) / sx, (uv_offset[1] * h - self.y as f32) / sy],
            [uv_scale[0] * w / sx, uv_scale[1] * h / sy],
        )
    }
}

/// Plan residency for a view showing the image-normalised range
/// `uv_offset .. uv_offset + uv_scale` of a `width` x `height` frame.
///
/// The result always fits everything [`Budget`] bounds and always covers the
/// visible region.
pub fn plan(
    width: u32,
    height: u32,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
    viewport: [f32; 2],
    budget: Budget,
) -> Residency {
    let Budget { max_axis, bytes_per_texel, mode } = budget;
    let (width, height) = (width.max(1), height.max(1));
    // A device that cannot hold a 2x2 texture cannot run this renderer at all;
    // the floor keeps the arithmetic below from having to model that case.
    let max_axis = max_axis.max(2);
    let bytes_per_texel = bytes_per_texel.max(1);
    let (vx, vy, vw, vh) = visible_rect(width, height, uv_offset, uv_scale);

    if mode == LargeImageMode::Preload {
        return plan_preloaded(
            width,
            height,
            (vx, vy, vw, vh),
            viewport,
            max_axis,
            bytes_per_texel,
        );
    }

    // Resolution follows what is on screen, not the size of the file: a view of
    // 2000 pixels gets stride 1 whether the frame is 4000 wide or 400000.
    //
    // The floor is what the *display* can resolve. Keeping more detail than the
    // monitor has pixels for is invisible and not free: showing a 40000-pixel
    // mosaic on a 1900-pixel panel at stride 1 would mean decoding, subsampling
    // and uploading a hundred times what is drawn, and paying it again on every
    // zoom step. Below that floor the texture is bounded by the viewport rather
    // than by the file, which is what keeps a gigapixel image as cheap to look
    // at as any other.
    //
    // Start from the coarsest bound anything implies and walk up. Starting at 1
    // would be correct but can take a very long walk on a frame needing a
    // stride in the thousands, and the bounds are exact only to within a texel
    // or two — grid alignment can push the requirement past what pure division
    // suggests, which is why this still has to check.
    let mut stride = display_stride(vw, vh, viewport)
        .max(vw.div_ceil(max_axis))
        .max(vh.div_ceil(max_axis))
        .max(1);
    loop {
        let fits_device =
            needed_texels(vx, vw, stride) <= max_axis && needed_texels(vy, vh, stride) <= max_axis;
        if fits_device {
            let rx = axis_window(vx, vw, width, stride, max_axis, MARGIN);
            let ry = axis_window(vy, vh, height, stride, max_axis, MARGIN);
            let bytes = (rx.1 as usize) * (ry.1 as usize) * bytes_per_texel;
            // The escape is for a frame so large that even one texel per axis is
            // over budget; there is nothing coarser to fall back to.
            if bytes <= MAX_ROI_BYTES || (rx.1 <= 1 && ry.1 <= 1) {
                return Residency {
                    resident: roi(rx, ry, stride, width, height),
                    required: roi(
                        axis_window(vx, vw, width, stride, max_axis, 1),
                        axis_window(vy, vh, height, stride, max_axis, 1),
                        stride,
                        width,
                        height,
                    ),
                };
            }
        }
        // Both texel counts fall as the stride grows, so this terminates.
        stride += 1;
    }
}

/// [`LargeImageMode::Preload`]'s planner: the coarse whole-frame level, or a
/// full-resolution crop, and nothing in between.
///
/// The choice is simply whether the display can resolve the file's own pixels.
/// If it can — the view is zoomed in far enough that a source pixel is worth at
/// least half a screen pixel — the crop is worth building. If it cannot, the
/// retained level already spans the frame and serves the view with no decode at
/// all, which is the entire point of the mode.
///
/// Falls back to the coarse level whenever the crop will not fit, rather than
/// walking the stride up as `Tiled` does. Walking is what introduces the
/// intermediate levels this mode exists to avoid.
fn plan_preloaded(
    width: u32,
    height: u32,
    visible: (u32, u32, u32, u32),
    viewport: [f32; 2],
    max_axis: u32,
    bytes_per_texel: usize,
) -> Residency {
    let (vx, vy, vw, vh) = visible;
    let coarse = {
        let stride = overview_stride(width, height, max_axis, bytes_per_texel);
        // Deliberately the whole frame, not a window of it: a level that spans
        // everything serves every view, so panning and zooming out never
        // re-cut it and it is decoded exactly once.
        let r = Roi { x: 0, y: 0, w: width, h: height, stride };
        Residency { resident: r, required: r }
    };

    if display_stride(vw, vh, viewport) > 1 {
        return coarse;
    }

    // Zoomed in far enough to want the file's own resolution. Try it with the
    // pan margin, then without it, then give up and stay coarse.
    let fits_axes =
        needed_texels(vx, vw, 1) <= max_axis && needed_texels(vy, vh, 1) <= max_axis;
    if fits_axes {
        let required = roi(
            axis_window(vx, vw, width, 1, max_axis, 1),
            axis_window(vy, vh, height, 1, max_axis, 1),
            1,
            width,
            height,
        );
        for margin in [MARGIN, 1] {
            let rx = axis_window(vx, vw, width, 1, max_axis, margin);
            let ry = axis_window(vy, vh, height, 1, max_axis, margin);
            if (rx.1 as usize) * (ry.1 as usize) * bytes_per_texel <= MAX_ROI_BYTES {
                return Residency { resident: roi(rx, ry, 1, width, height), required };
            }
        }
    }
    coarse
}

/// Texels needed to cover `v .. v+len` at `stride`, on the aligned grid.
///
/// Deliberately unclamped: this is the question "would this stride fit the
/// device", and a clamped answer always says yes — which is exactly the bug
/// that let a 64-pixel frame plan a 2-pixel window and call it covered.
fn needed_texels(v: u32, len: u32, stride: u32) -> u32 {
    let t0 = v / stride;
    let t1 = (v + len).div_ceil(stride);
    (t1 - t0).max(1)
}

/// The coarsest sampling that still gives the display at least one source
/// sample per pixel it can draw.
///
/// Rounded **down** to a power of two, for two reasons. Down, so the texture
/// keeps slightly more detail than the screen strictly needs and never looks
/// soft. A power of two, so zooming crosses only a handful of levels — a stride
/// free to take any value changes on nearly every zoom step, and each change
/// re-decodes the window, which is precisely the stall this is meant to avoid.
///
/// `viewport` of zero means the frontend has not said how large it is; the
/// result is then 1 and residency falls back to being bounded by the texture
/// budget alone, which is correct if wasteful.
fn display_stride(vw: u32, vh: u32, viewport: [f32; 2]) -> u32 {
    let (pw, ph) = (viewport[0], viewport[1]);
    if !(pw.is_finite() && ph.is_finite()) || pw < 1.0 || ph < 1.0 {
        return 1;
    }
    // Source pixels per screen pixel, on the axis that needs the most.
    let ratio = (vw as f32 / pw).max(vh as f32 / ph);
    if !ratio.is_finite() || ratio < 2.0 {
        return 1;
    }
    // Largest power of two not exceeding the ratio.
    let n = ratio as u32;
    1u32 << (u32::BITS - 1 - n.leading_zeros())
}

/// Build a rect from two per-axis `(start texel, texel count)` results.
///
/// The span is clamped to the frame because the last texel of an axis is only
/// partly covered whenever the stride does not divide the frame: `count *
/// stride` would then claim pixels that are not there. Clamping does not change
/// the texel count — a partly covered texel still rounds up to one — so the
/// texture stays the size the plan sized it.
fn roi(x: (u32, u32), y: (u32, u32), stride: u32, width: u32, height: u32) -> Roi {
    let (px, py) = (x.0 * stride, y.0 * stride);
    Roi {
        x: px,
        y: py,
        w: (x.1 * stride).min(width.saturating_sub(px)).max(1),
        h: (y.1 * stride).min(height.saturating_sub(py)).max(1),
        stride,
    }
}

/// The visible region in image pixels: rounded outward so a partly covered edge
/// pixel is still inside, and clamped, since a frontend mid-gesture can hand
/// over a window that runs off the frame.
fn visible_rect(
    width: u32,
    height: u32,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
) -> (u32, u32, u32, u32) {
    let px = |v: f32, n: u32| {
        if v.is_finite() {
            (v * n as f32).clamp(0.0, n as f32)
        } else {
            0.0
        }
    };
    let x0 = (px(uv_offset[0], width).floor() as u32).min(width - 1);
    let y0 = (px(uv_offset[1], height).floor() as u32).min(height - 1);
    let x1 = px(uv_offset[0] + uv_scale[0], width).ceil() as u32;
    let y1 = px(uv_offset[1] + uv_scale[1], height).ceil() as u32;
    (x0, y0, x1.saturating_sub(x0).clamp(1, width - x0), y1.saturating_sub(y0).clamp(1, height - y0))
}

/// One axis, entirely in texels: the texel range covering `v .. v+len`, grown
/// by `margin` as far as the device limit and the frame allow.
///
/// Returns `(start texel, texel count)`. Working in texels is what makes the
/// grid alignment free — a texel index *is* an aligned position — and lets the
/// device limit be checked against the number it actually constrains.
fn axis_window(v: u32, len: u32, n: u32, stride: u32, max_texels: u32, margin: u32) -> (u32, u32) {
    // Texels covering the visible run. `t0` floors, so the window starts at or
    // before `v`; `t1` ceils, so it ends at or after `v + len`.
    let t0 = v / stride;
    let t1 = (v + len).div_ceil(stride);
    let need = (t1 - t0).max(1);
    // Texels the whole frame occupies at this stride — the window cannot be
    // bigger, and neither can the margin.
    let total = n.div_ceil(stride).max(1);
    let cap = max_texels.min(total);
    if need >= cap {
        // The visible run alone fills (or overflows) what is available. Give it
        // everything on offer, starting as far back as still covers `v`.
        let count = need.min(cap);
        let start = t0.min(total.saturating_sub(count));
        return (start, count);
    }
    let want = need.saturating_mul(margin).clamp(need, cap);
    // Centre the margin on the visible run, then slide back inside the frame
    // rather than clipping: a window pushed against an edge keeps its size and
    // gives up the margin on that side only.
    let half = (want - need) / 2;
    let start = t0.saturating_sub(half).min(total - want);
    (start, want)
}

/// Cut a window out of a decoded plane, taking every `stride`-th sample.
///
/// The output is always exactly the texture size, padded with `T::default()` if
/// the plane is shorter than its declared frame. Handing the GPU a buffer that
/// does not match the texture is a validation failure, and validation failures
/// here are fatal — the very thing this module exists to prevent.
pub fn extract<T: Copy + Default>(src: &[T], width: u32, height: u32, roi: &Roi) -> Vec<T> {
    let (tw, th) = roi.texture_size();
    extract_sampled(
        src,
        width,
        height,
        &Sampling {
            x: roi.x,
            y: roi.y,
            cols: tw,
            rows: th,
            x_step: roi.stride,
            y_step: roi.stride,
        },
    )
}

/// A rectangle of a buffer whose two axes are sampled at **different** rates.
///
/// [`Roi`] cannot express this: its one `stride` is both the column spacing and
/// the row spacing, which is right for an ordinary frame buffer and wrong for
/// one whose rows came from a *stepped* decode. There, whole strips between the
/// rows that were kept were never decompressed, so the surviving rows sit one
/// strip-height apart in the buffer — while their columns were not sampled at
/// all and are still every pixel.
///
/// Naming the two separately is the point. Conflating them is not a crash or an
/// error but a picture built from the right pixels in the wrong columns, which
/// reads as plausible image data with banding, and that is exactly the bug this
/// type exists to make impossible to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampling {
    /// First column of `src` to read.
    pub x: u32,
    /// First row of `src` to read.
    pub y: u32,
    /// Columns to produce.
    pub cols: u32,
    /// Rows to produce.
    pub rows: u32,
    /// Source columns between one output column and the next.
    pub x_step: u32,
    /// Source rows between one output row and the next.
    pub y_step: u32,
}

/// Read `cols` x `rows` samples out of `src`, advancing `x_step` per column and
/// `y_step` per row.
///
/// Anything the source runs short of comes back as `T::default()`, so the result
/// is always exactly `cols * rows` and every row starts at the same column — a
/// short source leaves a black edge rather than a sheared picture.
pub fn extract_sampled<T: Copy + Default>(
    src: &[T],
    width: u32,
    height: u32,
    s: &Sampling,
) -> Vec<T> {
    let (cols, rows) = (s.cols as usize, s.rows as usize);
    let want = cols * rows;
    let (w, h) = (width as usize, height as usize);
    let (xs, ys) = (s.x_step.max(1) as usize, s.y_step.max(1) as usize);
    let mut out: Vec<T> = Vec::with_capacity(want);

    for ty in 0..rows {
        let sy = s.y as usize + ty * ys;
        if sy >= h {
            break;
        }
        let start = sy * w + s.x as usize;
        // The last sample this row contributes, so the slice covers exactly the
        // texels wanted and `step_by` does the rest.
        let end = (start + cols.saturating_sub(1) * xs + 1).min(src.len());
        let Some(row) = src.get(start..end) else { break };
        let before = out.len();
        out.extend(row.iter().step_by(xs).copied());
        // A row that ran short leaves a ragged edge; square it off so every row
        // starts at the same column.
        out.resize(before + cols, T::default());
    }
    out.resize(want, T::default());
    out
}

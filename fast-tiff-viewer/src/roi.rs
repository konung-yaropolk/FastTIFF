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
/// `max_axis` is the device's per-axis texture limit and `bytes_per_texel` the
/// summed per-texel cost of the resident channels. The result always fits both
/// limits and always covers the visible region.
pub fn plan(
    width: u32,
    height: u32,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
    max_axis: u32,
    bytes_per_texel: usize,
) -> Residency {
    let (width, height) = (width.max(1), height.max(1));
    // A device that cannot hold a 2x2 texture cannot run this renderer at all;
    // the floor keeps the arithmetic below from having to model that case.
    let max_axis = max_axis.max(2);
    let bytes_per_texel = bytes_per_texel.max(1);
    let (vx, vy, vw, vh) = visible_rect(width, height, uv_offset, uv_scale);

    // Resolution follows what is on screen, not the size of the file: a view of
    // 2000 pixels gets stride 1 whether the frame is 4000 wide or 400000.
    //
    // Start from the coarsest bound either axis implies and walk up. Starting
    // at 1 would be correct but can take a very long walk on a frame that needs
    // a stride in the thousands, and the bound is exact to within a texel or
    // two — the grid alignment can push the requirement one texel past what
    // pure division suggests, which is why this still has to check.
    let mut stride = vw.div_ceil(max_axis).max(vh.div_ceil(max_axis)).max(1);
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
    let want = tw as usize * th as usize;
    let (w, h) = (width as usize, height as usize);
    let stride = roi.stride.max(1) as usize;
    let mut out: Vec<T> = Vec::with_capacity(want);

    for ty in 0..th as usize {
        let sy = roi.y as usize + ty * stride;
        if sy >= h {
            break;
        }
        let start = sy * w + roi.x as usize;
        // The last sample this row contributes, so the slice covers exactly the
        // texels wanted and `step_by` does the rest.
        let end = (start + (tw as usize - 1) * stride + 1).min(src.len());
        let Some(row) = src.get(start..end) else { break };
        let before = out.len();
        out.extend(row.iter().step_by(stride).copied());
        // A row that ran short leaves a ragged edge; square it off so every row
        // starts at the same column.
        out.resize(before + tw as usize, T::default());
    }
    out.resize(want, T::default());
    out
}

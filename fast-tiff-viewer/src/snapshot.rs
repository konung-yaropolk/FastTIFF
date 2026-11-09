//! The 2-D view as an RGBA image, composed on the CPU.
//!
//! What the window shows is composed on the GPU — raw samples go up as
//! textures and `composite.wgsl` does window/level, the LUT lookup and the
//! additive blend per pixel. That is what makes scrubbing a huge stack fast,
//! and it is also why there is no image anywhere on the CPU to hand to the
//! clipboard. This builds one.
//!
//! # Why not read the pixels back off the GPU
//!
//! Because the thing on screen is the wrong picture to copy. It is at whatever
//! zoom and pan the user left it, clipped to the window, and on a stack too
//! large to upload whole it is a *window* of the frame rather than the frame
//! (see [`crate::roi`]). A copy of that is a screenshot of a viewport; what is
//! wanted is the image — every pixel of the current frame, at its own size, as
//! the display is currently interpreting it.
//!
//! Composing here also means the result does not depend on having a working
//! GPU, a current swapchain, or a frame in flight, and it can be tested without
//! any of them. The 3-D view is the opposite case and is handled the opposite
//! way: there the view *is* the picture, so the frontend screenshots it.
//!
//! # Staying identical to the shader
//!
//! The arithmetic below is `composite.wgsl` transcribed, and it has to stay
//! that way or a copy would not match what the user was looking at. Two things
//! make that tractable:
//!
//! * The pixels come from [`build_jobs`] and the decoder the upload path uses
//!   ([`decode_jobs_rows`] is that decode, a band of rows at a time), so the
//!   sample values are byte-identical to the ones the textures hold —
//!   including the widening and the RGB/CMYK plane addressing, which are easy
//!   to get subtly wrong a second time.
//! * At the original size one output pixel is exactly one texel, so the
//!   shader's minification box filter collapses to `n = 1` — a plain nearest
//!   read. There is no filtering to reproduce, which is the other reason the
//!   copy is defined at the image's own size rather than at the zoom on screen.
//!
//! The parts that *are* reproduced are the window/level with its one-unit floor
//! on the span, the LUT lookup through a linear sampler, and the clamped
//! additive blend. Each has its own test.
//!
//! # A band at a time, and a limit
//!
//! The frame is decoded and composed a band of rows at a time, straight into
//! the image: all that is held besides the result is one band of each channel
//! and, for more than one channel, one band of the blend's `f32` accumulator.
//! A single channel needs no accumulator at all, since its sum is itself.
//! Composing the whole frame at once held every channel decoded plus 12 bytes
//! a pixel of accumulator, which on a 40000 x 12788 mosaic is gigabytes.
//!
//! The two bands are not the same size. A decoded band is at least
//! [`PARALLEL_MIN_PIXELS`] (a million pixels), the least fast-tiff-lib splits
//! across cores, so a large copy still decodes its strips in parallel when the
//! view's decodes do; the accumulator's band is a few megabytes, and a decoded
//! band is composed in as many of those as it takes. And each IFD is banded on
//! its own strips: the channels of a frame are IFDs of their own, which a file
//! may lay out in strips of different heights, so a band of one need not end
//! where a band of another does. Rows are composed once every channel has
//! them, and an IFD's next band is decoded only when the rows reach the end of
//! its last, so no strip is decoded twice.
//!
//! A frame whose image would be past 256 MiB (8192 x 8192 pixels) is refused,
//! before anything is decoded or allocated for it, with a message saying so.
//! It is not copied from the window on screen instead: that is a viewport at
//! whatever zoom and sampling the view is at (see above), so it would quietly
//! be a different picture from the one a copy promises. Refusing says what
//! happened, and cropping the frame first gives a copy that is the image.

use crate::prefetch::{build_jobs, decode_jobs_rows, ChannelJob, DecodedRef, Samples};
use crate::stack::Stack;
use anyhow::{Context, Result};
use fast_tiff_lib::PARALLEL_MIN_PIXELS;
use scivis_render::Lut;
use std::ops::Range;

/// A composed image, ready for the clipboard: 8-bit RGBA, row-major, no padding.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Snapshot {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, fully opaque.
    pub rgba: Vec<u8>,
}

impl Snapshot {
    /// The pixel at `(x, y)`, for tests and for callers inspecting a result.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        Some([
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ])
    }
}

/// The largest image a copy may be, in RGBA bytes: 256 MiB, an 8192 x 8192
/// frame.
///
/// 8192 is the 2D texture size every WebGPU device has to support (and wgpu's
/// default limit), so a frame this viewer can show whole on any such GPU can
/// be copied. The image is not the only cost either: the frontend converts it
/// for the clipboard and the system keeps a copy of its own, so a frame at the
/// limit already has about a gigabyte between them, and a gigapixel mosaic
/// would ask for several times the memory there is.
pub(crate) const MAX_SNAPSHOT_BYTES: u64 = 256 << 20;

/// About how much one band's accumulator may take: the rows composed at a time
/// are as many as fit in it. The rows decoded at a time are more (see the
/// [module docs](self)).
const BAND_BYTES: usize = 4 << 20;

/// A frame whose image would be past [`MAX_SNAPSHOT_BYTES`], refused before
/// anything was decoded or allocated for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TooLarge {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl std::fmt::Display for TooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = rgba_bytes(self.width, self.height);
        write!(
            f,
            "the frame is {} x {} pixels, {} MB as an image, and a copy can be at most {} MB \
             (8192 x 8192 pixels); crop it to copy part of it",
            self.width,
            self.height,
            bytes.div_ceil(1 << 20),
            MAX_SNAPSHOT_BYTES >> 20,
        )
    }
}

impl std::error::Error for TooLarge {}

/// Bytes a `width` x `height` RGBA image takes. In `u64`, because on a 32-bit
/// target the product of two `u32`s past 65536 does not fit a `usize`, and
/// saturating, because four times two of them past 2^31 does not fit a `u64`
/// either: a frame that large is past any limit, not wrapped back under it.
fn rgba_bytes(width: u32, height: u32) -> u64 {
    (width as u64 * height as u64).saturating_mul(4)
}

/// `value` mapped into 0..1 by a channel's contrast window.
///
/// The `max(span, 1.0)` is not a guard against division by zero, or not only
/// that: it is what the shader does, so it is what this does. A window whose
/// handles have been dragged together spans less than one sample unit, and the
/// floor makes such a window behave as a one-unit window rather than as an
/// infinitely steep one. Copying a picture that was thresholded differently
/// from the one on screen would be a quiet, plausible-looking wrong answer.
pub fn windowed(value: f32, min: f32, max: f32) -> f32 {
    let span = (max - min).max(1.0);
    ((value - min) / span).clamp(0.0, 1.0)
}

/// A LUT entry at fractional position `t`, as the GPU's sampler would read it.
///
/// The LUT is a 256-texel `Rgba8Unorm` row sampled with `Linear` filtering and
/// `ClampToEdge`, so this interpolates between neighbours rather than picking
/// the nearest — a smooth ramp read with nearest lookup would band visibly
/// against the same ramp on screen.
///
/// Texel centres sit at `(i + 0.5) / 256`, which is where the half-texel shift
/// comes from. At the ends the clamp takes over and both neighbours are the
/// same entry, so `t = 0` is exactly `lut[0]` and `t = 1` exactly `lut[255]`.
/// `Rgba8Unorm` rather than its sRGB twin, so the bytes scale linearly by 255
/// with no transfer function in the way.
pub fn lut_sample(lut: &Lut, t: f32) -> [f32; 3] {
    let x = t.clamp(0.0, 1.0) * lut.len() as f32 - 0.5;
    let floor = x.floor();
    let frac = x - floor;
    let lo = (floor as i32).clamp(0, lut.len() as i32 - 1) as usize;
    let hi = (floor as i32 + 1).clamp(0, lut.len() as i32 - 1) as usize;
    let (a, b) = (&lut[lo], &lut[hi]);
    [0, 1, 2].map(|k| {
        let v = a[k] as f32 * (1.0 - frac) + b[k] as f32 * frac;
        v / 255.0
    })
}

/// One decoded sample, as an `f32` in the units the channel's texture holds.
///
/// The shader reads its integer textures with `f32(textureLoad(...).r)` — the
/// stored integer, widened, not normalised — and compares it with the window
/// [`SampleUnits::to_texture`](crate::units::SampleUnits::to_texture) gave it.
/// Taking the samples from the decoder the upload uses, and the window from
/// the same conversion, is what makes this compose the picture the shader
/// draws rather than one that merely resembles it.
fn sample(data: Samples<'_>, i: usize) -> Option<f32> {
    match data {
        Samples::U8(v) => v.get(i).map(|&s| s as f32),
        Samples::U16(v) => v.get(i).map(|&s| s as f32),
        Samples::F32(v) => v.get(i).copied(),
    }
}

/// One blended colour channel as the byte it is written as: clamped, then
/// 255.0 and round, so a LUT entry of 255 comes back as 255 rather than as 254
/// from a truncating cast.
fn to_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// One channel as it is composed: where its samples are among the decoded
/// bands, its window in its texture's units, and its LUT.
struct Layer<'a> {
    /// Its IFD's band, in [`compose`]'s sources.
    source: usize,
    /// Its plane among that band's.
    plane: usize,
    min: f32,
    max: f32,
    lut: &'a Lut,
}

impl Layer<'_> {
    /// This channel's decoded samples, and where `row` starts in them.
    fn samples<'s>(
        &self,
        sources: &'s [Source<'_>],
        row: u32,
        width: usize,
    ) -> (Samples<'s>, usize) {
        let source = &sources[self.source];
        let first = (row - source.rows.start) as usize * width;
        (source.planes[self.plane].view(), first)
    }
}

/// The channels read from one IFD and the band of it decoded last.
///
/// One per IFD rather than one for the frame, because IFDs need not share a
/// strip layout (see the [module docs](self)). The channels of an RGB or CMYK
/// frame are planes of one IFD, so they are one source and still share a
/// single decompression pass.
struct Source<'a> {
    jobs: Vec<ChannelJob>,
    planes: Vec<DecodedRef<'a>>,
    /// The rows `planes` hold, whole strips of this IFD: empty before the
    /// first band.
    rows: Range<u32>,
}

/// Compose the frame `stack` is showing into an RGBA image at its own size.
///
/// Every enabled channel, each through its own contrast window and LUT, blended
/// the way the display blends them. Disabled channels are not decoded at all —
/// the shader would multiply them by zero, and not reading them is both faster
/// and the same picture.
///
/// Errors if the frame cannot be decoded, which is the one failure worth
/// reporting: the caller asked for a copy and there is nothing to give it. And
/// refuses, before decoding or allocating anything for it, a frame whose image
/// would be past 256 MiB (8192 x 8192 pixels); see the [module docs](self).
pub fn of_frame(stack: &Stack) -> Result<Snapshot> {
    let (width, height) = stack
        .dimensions()
        .context("nothing is open, so there is no view to copy")?;
    if rgba_bytes(width, height) > MAX_SNAPSHOT_BYTES {
        return Err(TooLarge { width, height }.into());
    }
    let width_px = (width as usize).max(1);
    let row_bytes = width_px.saturating_mul(std::mem::size_of::<[f32; 3]>());
    let compose_rows = (BAND_BYTES / row_bytes).max(1);
    let decode_rows = PARALLEL_MIN_PIXELS.div_ceil(width_px).max(compose_rows);
    let rows = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    compose(stack, width, height, rows(decode_rows), rows(compose_rows))
}

/// [`of_frame`] past its size check: decoded `decode_rows` rows at a time (or
/// the whole strips around them), composed `compose_rows` at a time.
fn compose(
    stack: &Stack,
    width: u32,
    height: u32,
    decode_rows: u32,
    compose_rows: u32,
) -> Result<Snapshot> {
    let settings = stack.display.settings();
    let masks = stack.display.masks();
    // Only the channels that contribute: on, and with settings and a LUT. One
    // missing either was skipped by the blend anyway. Each IFD read is one
    // source, in the order its first channel comes.
    let mut sources: Vec<Source<'_>> = Vec::new();
    let mut layers: Vec<Layer<'_>> = Vec::new();
    for job in build_jobs(stack, stack.frame_index, masks.enabled()) {
        let (Some(s), Some(lut)) = (settings.get(job.channel), stack.display.lut(job.channel))
        else {
            continue;
        };
        // The window as the *samples* are scaled, not as the slider holds it:
        // an Int16 channel's texture is on the decoder's 16-bit scale while its
        // window is in the file's units, and pairing the two directly composes
        // an 8-bit-widened image far too bright, a signed one shifted.
        let (min, max) = s.units.to_texture(s.window);
        let source = match sources
            .iter()
            .position(|src| src.jobs[0].ifd_idx == job.ifd_idx)
        {
            Some(i) => i,
            None => {
                sources.push(Source {
                    jobs: Vec::new(),
                    planes: Vec::new(),
                    rows: 0..0,
                });
                sources.len() - 1
            }
        };
        layers.push(Layer {
            source,
            plane: sources[source].jobs.len(),
            min,
            max,
            lut,
        });
        sources[source].jobs.push(job);
    }

    let width_px = width as usize;
    let pixels = width_px * height as usize;
    // Black and opaque: what a pixel no channel reaches is, as on screen.
    let mut rgba: Vec<u8> = [0, 0, 0, 255].repeat(pixels);
    if layers.is_empty() {
        return Ok(Snapshot {
            width,
            height,
            rgba,
        });
    }

    let compose_rows = compose_rows.clamp(1, height.max(1));
    let decode_rows = decode_rows.max(1);
    // Accumulate in f32 and convert once at the end: the blend is additive and
    // clamped only at the end, exactly as the shader does it. Rounding each
    // channel into 8 bits before adding would darken every overlap. One band
    // of it, reused for every band; a single channel's sum is itself and
    // needs none.
    let mut acc = if layers.len() > 1 {
        vec![[0.0f32; 3]; compose_rows as usize * width_px]
    } else {
        Vec::new()
    };

    let mut y = 0;
    while y < height {
        // Each IFD whose band ends at `y` decodes its next. The decode snaps
        // outward to whole strips (or tile rows), so a band can start before
        // `y`, but never before the end of the last: no strip is decoded twice.
        for source in &mut sources {
            if source.rows.end > y {
                continue;
            }
            // The last band goes first, so only one of each is ever held.
            source.planes.clear();
            let want = y..y.saturating_add(decode_rows).min(height);
            let (planes, rows) = decode_jobs_rows(
                &stack.tiff.data,
                &stack.tiff.frames,
                stack.tiff.byte_order,
                &source.jobs,
                want,
            )
            .context("could not decode the frame on screen")?;
            anyhow::ensure!(
                rows.start <= y && rows.end > y,
                "rows {rows:?} were decoded for row {y}"
            );
            source.planes = planes;
            source.rows = rows;
        }
        // Every channel has the rows from `y` to where the first band ends.
        let end = sources
            .iter()
            .map(|source| source.rows.end)
            .min()
            .unwrap_or(height)
            .min(height);
        let mut row = y;
        while row < end {
            let rows = (end - row).min(compose_rows);
            let n = rows as usize * width_px;
            let at = row as usize * width_px * 4;
            let out = &mut rgba[at..at + n * 4];
            match &layers[..] {
                [layer] => {
                    let (data, first) = layer.samples(&sources, row, width_px);
                    compose_one(layer, data, first, out);
                }
                _ => compose_many(&layers, &sources, row, width_px, &mut acc[..n], out),
            }
            row += rows;
        }
        y = end;
    }

    Ok(Snapshot {
        width,
        height,
        rgba,
    })
}

/// Write one channel's samples from `first` on straight into `out`, one RGBA
/// pixel each: the sum of one channel is itself, so nothing is accumulated.
fn compose_one(layer: &Layer<'_>, data: Samples<'_>, first: usize, out: &mut [u8]) {
    for (i, px) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        // A plane shorter than the frame is a damaged file, not a reason to
        // panic or to refuse the copy: the pixels that exist are composed and
        // the rest stay black, which is what the GPU's clamped texture read
        // would also give.
        let Some(value) = sample(data, first + i) else {
            break;
        };
        let rgb = lut_sample(layer.lut, windowed(value, layer.min, layer.max));
        for (o, v) in px.iter_mut().zip(rgb) {
            *o = to_byte(v);
        }
    }
}

/// Blend every channel's samples of the rows from `row` on into `acc`, in
/// channel order as the shader adds them, then write the sums into `out`.
fn compose_many(
    layers: &[Layer<'_>],
    sources: &[Source<'_>],
    row: u32,
    width: usize,
    acc: &mut [[f32; 3]],
    out: &mut [u8],
) {
    acc.fill([0.0; 3]);
    for layer in layers {
        let (data, first) = layer.samples(sources, row, width);
        for (i, sum) in acc.iter_mut().enumerate() {
            // A short plane: see `compose_one`.
            let Some(value) = sample(data, first + i) else {
                break;
            };
            let rgb = lut_sample(layer.lut, windowed(value, layer.min, layer.max));
            for (o, v) in sum.iter_mut().zip(rgb) {
                *o += v;
            }
        }
    }
    for (px, sum) in out.as_chunks_mut::<4>().0.iter_mut().zip(acc.iter()) {
        for (o, &v) in px.iter_mut().zip(sum) {
            *o = to_byte(v);
        }
    }
}

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod tests;

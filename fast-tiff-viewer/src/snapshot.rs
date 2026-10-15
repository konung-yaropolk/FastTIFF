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
//! * The pixels come from [`build_jobs`] + [`decode_jobs`], the same pair the
//!   upload path and the histogram use, so the sample values are byte-identical
//!   to the ones the textures hold — including the widening and the RGB/CMYK
//!   plane addressing, which are easy to get subtly wrong a second time.
//! * At the original size one output pixel is exactly one texel, so the
//!   shader's minification box filter collapses to `n = 1` — a plain nearest
//!   read. There is no filtering to reproduce, which is the other reason the
//!   copy is defined at the image's own size rather than at the zoom on screen.
//!
//! The parts that *are* reproduced are the window/level with its one-unit floor
//! on the span, the LUT lookup through a linear sampler, and the clamped
//! additive blend. Each has its own test.

use crate::prefetch::{build_jobs, decode_jobs, Decoded};
use crate::stack::Stack;
use anyhow::{Context, Result};
use scivis_render::{ChannelKind, Lut};

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

/// One decoded sample, as an `f32` in the units the contrast window is in.
///
/// The shader reads its integer textures with `f32(textureLoad(...).r)` — the
/// stored integer, widened, not normalised — so an 8-bit channel's window is in
/// 0..255 and a 16-bit channel's in 0..65535. Matching that is the whole point
/// of taking the samples from [`decode_jobs`] rather than re-reading the file
/// in its own units, where an 8-bit plane would arrive 257x off.
fn sample(data: &Decoded, i: usize) -> Option<f32> {
    match data {
        Decoded::U8(v) => v.get(i).map(|&s| s as f32),
        Decoded::U16(v) => v.get(i).map(|&s| s as f32),
        Decoded::F32(v) => v.get(i).copied(),
    }
}

/// Compose the frame `stack` is showing into an RGBA image at its own size.
///
/// Every enabled channel, each through its own contrast window and LUT, blended
/// the way the display blends them. Disabled channels are not decoded at all —
/// the shader would multiply them by zero, and not reading them is both faster
/// and the same picture.
///
/// Errors if the frame cannot be decoded, which is the one failure worth
/// reporting: the caller asked for a copy and there is nothing to give it.
pub fn of_frame(stack: &Stack) -> Result<Snapshot> {
    let (width, height) = stack
        .dimensions()
        .context("nothing is open, so there is no view to copy")?;
    let settings = &stack.display.settings;
    let enabled: Vec<bool> = settings.iter().map(|s| s.enabled).collect();
    let kinds: Vec<ChannelKind> = settings.iter().map(|s| s.kind).collect();

    let jobs = build_jobs(stack, stack.frame_index, &enabled, &kinds);
    let decoded = decode_jobs(
        &stack.tiff.data,
        &stack.tiff.frames,
        stack.tiff.byte_order,
        &jobs,
    )
    .context("could not decode the frame on screen")?;

    let pixels = width as usize * height as usize;
    // Accumulate in f32 and convert once at the end: the blend is additive and
    // clamped only at the end, exactly as the shader does it. Rounding each
    // channel into 8 bits before adding would darken every overlap.
    let mut acc = vec![[0.0f32; 3]; pixels];
    for (job, data) in jobs.iter().zip(&decoded) {
        let Some(s) = settings.get(job.channel) else {
            continue;
        };
        let Some(lut) = stack.display.luts.get(job.channel) else {
            continue;
        };
        // The window as the *samples* are scaled, not as the slider holds it.
        // An 8-bit channel's window is kept in 0..65535 by the interface while
        // its samples are raw 0..255; pairing the two directly composes the
        // image at 1/257 of its contrast, which is black.
        let (min, max) = crate::channels::sample_window(s.kind, s.min, s.max);
        for (i, out) in acc.iter_mut().enumerate() {
            // A plane shorter than the frame is a damaged file, not a reason to
            // panic or to refuse the copy: the pixels that exist are composed
            // and the rest stay at zero, which is what the GPU's clamped
            // texture read would also give.
            let Some(value) = sample(data, i) else {
                break;
            };
            let rgb = lut_sample(lut, windowed(value, min, max));
            for (o, v) in out.iter_mut().zip(rgb) {
                *o += v;
            }
        }
    }

    let mut rgba = Vec::with_capacity(pixels * 4);
    for channel in &acc {
        for v in channel {
            // 255.0 then round, so a LUT entry of 255 comes back as 255 rather
            // than as 254 from a truncating cast.
            rgba.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        rgba.push(255);
    }

    Ok(Snapshot {
        width,
        height,
        rgba,
    })
}

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod tests;

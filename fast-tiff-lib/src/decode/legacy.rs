//! The readers [`Frame`](crate::Frame) replaces, each now a wrapper over the
//! decode core with a scratch of its own.
//!
//! Kept so every caller compiles and every test runs unchanged while the
//! callers move to [`Frame::read`](crate::Frame::read) and
//! [`Frame::read_into`](crate::Frame::read_into); then they go. They keep what
//! each always did that the core does not: a plane index past the last is
//! clamped to the last rather than refused, and each refuses what it always
//! refused (an 8-bit reader a 16-bit frame, the CMYK readers anything but a
//! CMYK frame) in its own words.

use super::cmyk::require_cmyk;
use super::core::{self, refuse_4bit_for_display, require_f32_samples, Src};
use super::pieces::bytes_for_bits;
use crate::display::DisplayMap;
use crate::error::{bail, err, ErrorKind, OptionContext, Result};
use crate::frame::FrameBytes;
use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo, SampleFormat, TiffStack};
use crate::DecodeScratch;
// Only the preloads use it, and only where there are threads to fork onto.
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::borrow::Cow;

/// The last plane, for an index past it: what every per-plane reader did.
fn clamp_plane(frame: &FrameInfo, plane: usize) -> usize {
    plane.min((frame.samples_per_pixel as usize).max(1) - 1)
}

/// Size `out` to exactly `n` reusing its allocation; only newly-grown
/// elements pay initialization.
fn ensure_len<T: Copy + Default>(out: &mut Vec<T>, n: usize) {
    if out.len() != n {
        out.resize(n, T::default());
    }
}

/// Vet `src` (the size guards), then size `out` to the plane `len` says —
/// the order the old readers sized their output in, so a hostile header is
/// refused before anything is allocated from it.
fn sized_out<T: Copy + Default>(
    src: Src<'_, '_>,
    out: &mut Vec<T>,
    len: impl FnOnce() -> Result<usize>,
) -> Result<()> {
    core::vet(src)?;
    ensure_len(out, len()?);
    Ok(())
}

/// The `n * sample_bytes` bytes at the frame's one strip when it is a single
/// uncompressed, predictor-free strip: the zero-copy read the single-sample
/// readers always offered, which trusts the frame's size over the strip's byte
/// count. `Err` when the file does not hold that many bytes there.
fn single_strip<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    n: usize,
    sample_bytes: usize,
) -> Result<Option<&'a [u8]>> {
    if frame.is_tiled()
        || frame.compression != Compression::None
        || frame.predictor != 1
        || frame.strip_offsets.len() != 1
    {
        return Ok(None);
    }
    let len = n
        .checked_mul(sample_bytes)
        .with_context(ErrorKind::TooLarge, || "frame size overflows".to_string())?;
    let slice = FrameBytes::whole(mmap)
        .piece(frame.strip_offsets[0], len as u64)
        .map_err(|_| err!(Malformed, "strip data out of file bounds"))?;
    Ok(Some(slice))
}

/// Every destination of `out`, as slices.
fn slices<T>(out: &mut [Vec<T>]) -> Vec<&mut [T]> {
    out.iter_mut().map(|v| &mut v[..]).collect()
}

/// Decoded pixel data for one plane, always exposed as 16-bit samples:
/// - 8-bit sources are upcast (`v -> (v << 8) | v`, mapping 0..255 evenly
///   onto 0..65535) so the GPU window/level math stays uniform.
/// - 32-bit float sources are linearly rescaled into 0..65535 using
///   `float_range` (the channel's display range, in the float data's own
///   units — matching how ImageJ treats float images: contrast is defined
///   over the data's actual value range, not assumed to already be
///   16-bit-integer-shaped). Pass `None` to auto-range from this frame's
///   own min/max (e.g. for an initial probe before a stable per-channel
///   range has been established — see `frame_float_minmax`). Ignored for
///   non-float sources.
///
/// Borrowed straight from `mmap` when plane 0 is an uncompressed unsigned
/// 16-bit run in the host's byte order; see [`Frame::read`](crate::Frame::read).
pub fn read_frame_u16<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
) -> Result<Cow<'a, [u16]>> {
    let n_samples = frame.sample_count()?;
    // A single uncompressed strip is borrowed for the frame's size whatever
    // its byte count says, as this reader always has: the playback path of a
    // viewer reads files whose StripByteCounts undersell the strip.
    if frame.bits_per_sample == 16
        && frame.samples_per_pixel == 1
        && frame.sample_format != SampleFormat::SignedInt
        && file_order == ByteOrder::host()
    {
        if let Some(slice) = single_strip(mmap, frame, n_samples, 2)? {
            if let Ok(samples) = bytemuck::try_cast_slice::<u8, u16>(slice) {
                return Ok(Cow::Borrowed(samples));
            }
        }
    }
    if let Some(borrowed) = core::borrow_display(Src::new(mmap, frame, file_order), 0) {
        return Ok(Cow::Borrowed(borrowed));
    }
    Ok(Cow::Owned(read_plane_u16(
        mmap,
        frame,
        file_order,
        float_range,
        0,
    )?))
}

/// [`read_frame_u16`] into a caller-provided buffer, reusing its allocation.
///
/// Returns the [`DisplayMap`] it applied, which for a wide frame read without
/// a `float_range` is the only record of the range it chose.
pub fn read_frame_u16_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    out: &mut Vec<u16>,
) -> Result<DisplayMap> {
    read_plane_u16_into(mmap, frame, file_order, float_range, 0, out)
}

/// Decode an **unsigned 8-bit, single-sample** frame's raw bytes, *without* the
/// `0..255 -> 0..65535` widening `read_frame_u16` does for 8-bit. Zero-copy
/// (a borrow over the memory map) for an uncompressed, predictor-free frame
/// stored as one run; otherwise an owned `Vec<u8>` (decompressed, predictor
/// undone). For any other depth it is the frame's stored bytes in the file's
/// byte order, as [`read_frame_bytes_into`] gives them.
pub fn read_frame_u8<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Cow<'a, [u8]>> {
    if frame.bits_per_sample != 4 {
        if let Some(borrowed) =
            core::borrow_stored_frame(Src::new(mmap, frame, file_order), file_order)
        {
            return Ok(Cow::Borrowed(borrowed));
        }
    }
    let mut out = Vec::new();
    read_frame_u8_into(mmap, frame, file_order, &mut out)?;
    Ok(Cow::Owned(out))
}

/// [`read_frame_u8`] into a caller-provided buffer, reusing its allocation.
pub fn read_frame_u8_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    out: &mut Vec<u8>,
) -> Result<()> {
    read_frame_bytes_into(mmap, frame, file_order, file_order, out)
}

/// A frame's samples exactly as the file stores them: decompressed and with the
/// predictor undone, but with no widening, offset or rescale, as bytes in byte
/// order `want`, laid out as the file lays them out (chunky, or one plane after
/// another for a planar frame). 4-bit palette indices come back one per byte.
///
/// For writing a frame back out without changing a value, and for reading it
/// as `&[i16]`, `&[u32]`, ... through bytemuck (pass [`ByteOrder::host`]).
pub fn read_frame_bytes_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    want: ByteOrder,
    out: &mut Vec<u8>,
) -> Result<()> {
    let n = core::stored_frame_len(frame)?;
    let src = Src::new(mmap, frame, file_order);
    sized_out(src, out, || Ok(n))?;
    core::stored_frame_into(src, want, out, &mut DecodeScratch::new())
}

/// Decode a single sample plane (`plane`, `< samples_per_pixel`) of a frame
/// into `width * height` u16 values, deinterleaving chunky multi-sample data
/// such as RGB. For single-sample frames `plane` is 0 and this returns the
/// whole image. The values are in display space, not the file's units; the
/// [`DisplayMap`] for the frame says how:
/// - 8-bit is upcast (`v -> (v << 8) | v`) onto 0..65535.
/// - signed integers are offset into ImageJ's unsigned display space (sign-bit
///   flip), so signed and unsigned+calibration files render the same.
/// - 32-bit samples (int *or* float) are linearly rescaled into 0..65535 using
///   `float_range` (or this plane's own min/max when `None`).
///
/// For the file's own values use [`read_plane_f32`].
pub fn read_plane_u16(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    plane: usize,
) -> Result<Vec<u16>> {
    let mut out = Vec::new();
    read_plane_u16_into(mmap, frame, file_order, float_range, plane, &mut out)?;
    Ok(out)
}

/// [`read_plane_u16`] into a caller-provided buffer, reusing its allocation.
///
/// Returns the [`DisplayMap`] it applied (see [`read_frame_u16_into`]).
pub fn read_plane_u16_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    plane: usize,
    out: &mut Vec<u16>,
) -> Result<DisplayMap> {
    refuse_4bit_for_display(frame, "read_plane_u16")?;
    let src = Src::new(mmap, frame, file_order);
    sized_out(src, out, || core::display_plane_len(frame))?;
    let plane = clamp_plane(frame, plane);
    let maps = core::display(
        src,
        float_range,
        plane,
        &mut [&mut out[..]],
        &mut DecodeScratch::new(),
    )?;
    Ok(maps.plane(0))
}

/// Decode **all** sample planes to u16 with a single decompression pass — for
/// chunky RGB this is ~3x cheaper than three [`read_plane_u16`] calls on
/// compressed data (each of which decompresses the whole frame again). Returns
/// one plane per sample, in sample order; single-sample frames return one
/// plane. With `float_range = None`, each 32-bit plane auto-ranges to its own
/// min/max, exactly like the per-plane call.
pub fn read_planes_u16(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
) -> Result<Vec<Vec<u16>>> {
    let mut out = Vec::new();
    read_planes_u16_into(mmap, frame, file_order, float_range, &mut out)?;
    Ok(out)
}

/// [`read_planes_u16`] into caller-provided buffers (outer Vec sized to the
/// sample count, inner Vecs reused).
pub fn read_planes_u16_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    out: &mut Vec<Vec<u16>>,
) -> Result<()> {
    refuse_4bit_for_display(frame, "read_planes_u16")?;
    let spp = (frame.samples_per_pixel as usize).max(1);
    let src = Src::new(mmap, frame, file_order);
    core::vet(src)?;
    let n = core::display_plane_len(frame)?;
    out.resize_with(spp, Vec::new);
    for plane in out.iter_mut() {
        ensure_len(plane, n);
    }
    core::display(
        src,
        float_range,
        0,
        &mut slices(out),
        &mut DecodeScratch::new(),
    )?;
    Ok(())
}

/// Decode a single sample plane (`plane`, `< samples_per_pixel`) of a frame as
/// **raw unsigned 8-bit bytes**, deinterleaving chunky multi-sample data such as
/// 8-bit RGB. The `u8` sibling of [`read_plane_u16`]: same strided gather, but it
/// keeps the bytes at native 8-bit width (no widening to 0..65535) for callers
/// that upload to an `R8Uint` texture and scale to 0..255 themselves — mirroring
/// [`read_frame_u8`]. Only valid for 8-bit frames (and 4-bit, whose indices
/// come one per byte); anything else is [`ErrorKind::InvalidInput`].
///
/// [`ErrorKind::InvalidInput`]: crate::ErrorKind::InvalidInput
pub fn read_plane_u8(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    read_plane_u8_into(mmap, frame, file_order, plane, &mut out)?;
    Ok(out)
}

/// The refusal the 8-bit readers give any other depth.
fn require_8bit(frame: &FrameInfo) -> Result<()> {
    if frame.bits_per_sample != 8 {
        bail!(
            InvalidInput,
            "read_plane_u8 requires 8-bit samples, got {}",
            frame.bits_per_sample
        );
    }
    Ok(())
}

/// [`read_plane_u8`] into a caller-provided buffer, reusing its allocation.
pub fn read_plane_u8_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let src = Src::new(mmap, frame, file_order);
    // 4-bit is single-sample (plane 0 is the whole image).
    let plane = if frame.bits_per_sample == 4 {
        0
    } else {
        bytes_for_bits(frame.bits_per_sample)?;
        clamp_plane(frame, plane)
    };
    sized_out(src, out, || {
        if frame.bits_per_sample != 4 {
            require_8bit(frame)?;
        }
        core::stored_plane_len(frame)
    })?;
    core::stored(
        src,
        file_order,
        plane,
        &mut [&mut out[..]],
        &mut DecodeScratch::new(),
    )
}

/// Decode **all** sample planes as raw 8-bit bytes with a single
/// decompression pass — the u8 sibling of [`read_planes_u16`], ~3x cheaper
/// than three [`read_plane_u8`] calls on compressed chunky RGB. Same validity
/// rules as the per-plane call (8-bit frames).
pub fn read_planes_u8(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    read_planes_u8_into(mmap, frame, file_order, &mut out)?;
    Ok(out)
}

/// [`read_planes_u8`] into caller-provided buffers.
pub fn read_planes_u8_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    frame.pixel_count()?;
    bytes_for_bits(frame.bits_per_sample)?;
    let src = Src::new(mmap, frame, file_order);
    core::vet(src)?;
    require_8bit(frame)?;
    let n = core::stored_plane_len(frame)?;
    out.resize_with(spp, Vec::new);
    for plane in out.iter_mut() {
        ensure_len(plane, n);
    }
    core::stored(
        src,
        file_order,
        0,
        &mut slices(out),
        &mut DecodeScratch::new(),
    )
}

/// Decode a frame's samples as `f32` **in the file's own units**: an 8-bit
/// 200 is 200.0, a signed 16-bit -1 is -1.0, a float is itself. No widening,
/// no offset, no rescale — none of what the display-space `*_u16` readers do
/// (see [`DisplayMap`]). Every depth this crate decodes: 4-bit palette indices,
/// 8- and 16-bit integers, signed or not, 32- and 64-bit integers and floats
/// (64-bit narrowed to `f32`). 16-bit floats are refused.
///
/// For a float (R32F) GPU texture, where window/level happens on the GPU, and
/// for anything that measures pixels. Zero-copy (`Cow::Borrowed` over the
/// memory map) when plane 0 is an uncompressed 32-bit float run in the host's
/// byte order; otherwise an owned `Vec<f32>` decoded in one pass from the
/// file's bytes.
pub fn read_frame_f32<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Cow<'a, [f32]>> {
    let n_pixels = frame.pixel_count()?;
    // The same single-strip borrow as `read_frame_u16`'s.
    if frame.bits_per_sample == 32
        && frame.sample_format == SampleFormat::Float
        && frame.samples_per_pixel == 1
        && file_order == ByteOrder::host()
    {
        if let Some(slice) = single_strip(mmap, frame, n_pixels, 4)? {
            if let Ok(samples) = bytemuck::try_cast_slice::<u8, f32>(slice) {
                return Ok(Cow::Borrowed(samples));
            }
        }
    }
    if let Some(borrowed) = core::borrow_file(Src::new(mmap, frame, file_order), 0) {
        return Ok(Cow::Borrowed(borrowed));
    }
    Ok(Cow::Owned(read_plane_f32(mmap, frame, file_order, 0)?))
}

/// [`read_frame_f32`] into a caller-provided buffer, reusing its allocation.
pub fn read_frame_f32_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    out: &mut Vec<f32>,
) -> Result<()> {
    read_plane_f32_into(mmap, frame, file_order, 0, out)
}

/// Decode a single sample plane of a frame as `f32` in the file's own units
/// (see [`read_frame_f32`]), deinterleaving chunky multi-sample data. One pass
/// from the decoded bytes, parallelized across cores for large planes.
pub fn read_plane_f32(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    read_plane_f32_into(mmap, frame, file_order, plane, &mut out)?;
    Ok(out)
}

/// [`read_plane_f32`] into a caller-provided buffer, reusing its allocation.
pub fn read_plane_f32_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    let src = Src::new(mmap, frame, file_order);
    let plane = if frame.bits_per_sample == 4 {
        0
    } else {
        require_f32_samples(frame, "read_plane_f32")?;
        clamp_plane(frame, plane)
    };
    sized_out(src, out, || core::file_plane_len(frame))?;
    core::file(src, plane, &mut [&mut out[..]], &mut DecodeScratch::new())
}

/// Decode **all** sample planes as `f32` in the file's own units with a single
/// decompression pass — the file-units sibling of [`read_planes_u16`]. Same
/// depths as the per-plane call.
pub fn read_planes_f32(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::new();
    read_planes_f32_into(mmap, frame, file_order, &mut out)?;
    Ok(out)
}

/// [`read_planes_f32`] into caller-provided buffers.
pub fn read_planes_f32_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    out: &mut Vec<Vec<f32>>,
) -> Result<()> {
    if frame.bits_per_sample == 4 {
        out.resize_with(1, Vec::new);
        return read_plane_f32_into(mmap, frame, file_order, 0, &mut out[0]);
    }
    require_f32_samples(frame, "read_planes_f32")?;
    let spp = (frame.samples_per_pixel as usize).max(1);
    let src = Src::new(mmap, frame, file_order);
    core::vet(src)?;
    let n = core::file_plane_len(frame)?;
    out.resize_with(spp, Vec::new);
    for plane in out.iter_mut() {
        ensure_len(plane, n);
    }
    core::file(src, 0, &mut slices(out), &mut DecodeScratch::new())
}

/// Eagerly decode **every** frame to 16-bit samples, in parallel *across*
/// frames (rayon). This is the load-everything-at-once counterpart to the lazy
/// per-frame [`read_frame_u16`]: it returns one owned `Vec<u16>` per frame, all
/// resident in memory. Useful when a consumer genuinely needs the whole stack in
/// RAM (e.g. batch processing) rather than scrubbing it frame by frame.
///
/// Each frame decodes its plane 0 (the whole image for single-sample frames);
/// `float_range` is applied to every frame the same way `read_frame_u16` uses
/// it. Parallelism here is across frames and is independent of the
/// [`set_parallel_decode`](super::set_parallel_decode) hint (which controls
/// *within*-frame threading) — for a large stack the across-frame split is
/// what scales, so leaving the hint off avoids redundant nested parallelism.
///
/// Note the memory cost: this holds the entire decoded stack at once
/// (`frames * width * height * 2` bytes).
pub fn preload_frames_u16(
    stack: &TiffStack,
    float_range: Option<(f32, f32)>,
) -> Result<Vec<Vec<u16>>> {
    // Parallel across frames on native; serial on wasm, which has no threads.
    #[cfg(not(target_arch = "wasm32"))]
    let frames = stack.frames.par_iter();
    #[cfg(target_arch = "wasm32")]
    let frames = stack.frames.iter();
    frames
        .map(|frame| {
            Ok(read_frame_u16(&stack.data, frame, stack.byte_order, float_range)?.into_owned())
        })
        .collect()
}

/// Eagerly decode every frame to raw `f32`, in parallel across frames — the
/// float counterpart to [`preload_frames_u16`] (for 32-bit-float stacks, no
/// rescaling). See that function for the parallelism/memory notes.
pub fn preload_frames_f32(stack: &TiffStack) -> Result<Vec<Vec<f32>>> {
    // Parallel across frames on native; serial on wasm, which has no threads.
    #[cfg(not(target_arch = "wasm32"))]
    let frames = stack.frames.par_iter();
    #[cfg(target_arch = "wasm32")]
    let frames = stack.frames.iter();
    frames
        .map(|frame| Ok(read_frame_f32(&stack.data, frame, stack.byte_order)?.into_owned()))
        .collect()
}

/// Eagerly decode every frame to raw 8-bit bytes, in parallel across frames —
/// the byte counterpart to [`preload_frames_u16`], at half the memory (1 byte
/// per pixel, no widening). See that function for the parallelism/memory notes.
///
/// Only meaningful for **unsigned single-sample 8-bit** stacks (the same data
/// [`read_frame_u8`] handles); gate on `bits_per_sample == 8` before calling.
/// For other formats it returns the frame's raw native bytes as-is, which won't
/// be the display-ready samples — use [`preload_frames_u16`] instead.
pub fn preload_frames_u8(stack: &TiffStack) -> Result<Vec<Vec<u8>>> {
    // Parallel across frames on native; serial on wasm, which has no threads.
    #[cfg(not(target_arch = "wasm32"))]
    let frames = stack.frames.par_iter();
    #[cfg(target_arch = "wasm32")]
    let frames = stack.frames.iter();
    frames
        .map(|frame| Ok(read_frame_u8(&stack.data, frame, stack.byte_order)?.into_owned()))
        .collect()
}

/// The actual min/max of a 32-bit frame's raw values (int or float) — used to
/// establish a channel's initial display range, the same way ImageJ
/// auto-ranges a 32-bit image to its own data instead of assuming a fixed
/// scale. `None` for non-32-bit frames (8/16-bit use their native integer
/// min/max instead).
pub fn frame_float_minmax(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Option<(f32, f32)>> {
    core::value_range(Src::new(mmap, frame, file_order), &mut DecodeScratch::new())
}

/// Decode a CMYK frame's **three RGB planes** (red, green, blue), in one
/// decompression pass.
///
/// Requires [`FrameInfo::is_cmyk`]; anything else is an error rather than a
/// silent misreading. Samples past the fourth — an alpha channel, or spot
/// colours in a file that also carries the four process inks — take no part in
/// the composite, matching what a viewer can meaningfully show.
///
/// Returns 3 planes for a 4-plus-sample frame, so the usual "one plane per
/// sample" expectation deliberately does *not* hold here.
pub fn read_planes_rgb_u8(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Vec<Vec<u8>>> {
    rgb_u8(mmap, frame, file_order, 0, 3)
}

/// [`read_planes_rgb_u8`]'s 16-bit sibling: the same three RGB planes, in the
/// 0..=65535 space the rest of the crate's `u16` readers use.
///
/// An 8-bit CMYK frame is widened the same way [`read_planes_u16`] widens 8-bit
/// samples, so a caller on the 16-bit path gets consistent units either way.
pub fn read_planes_rgb_u16(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Vec<Vec<u16>>> {
    rgb_u16(mmap, frame, file_order, 0, 3)
}

/// One converted RGB plane of a CMYK frame — `0` red, `1` green, `2` blue.
///
/// All four inks have to be read to produce any one of them, so prefer
/// [`read_planes_rgb_u8`] when more than one is wanted. Out-of-range indices
/// clamp to blue, matching the per-sample readers' behaviour rather than
/// panicking.
pub fn read_plane_rgb_u8(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
) -> Result<Vec<u8>> {
    Ok(rgb_u8(mmap, frame, file_order, plane.min(2), 1)?.remove(0))
}

/// [`read_plane_rgb_u8`] in 16-bit units.
pub fn read_plane_rgb_u16(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
) -> Result<Vec<u16>> {
    Ok(rgb_u16(mmap, frame, file_order, plane.min(2), 1)?.remove(0))
}

/// `count` RGB planes of a CMYK frame from `first`, 8-bit.
fn rgb_u8(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    first: usize,
    count: usize,
) -> Result<Vec<Vec<u8>>> {
    require_cmyk(frame, "read_planes_rgb_u8")?;
    // What the ink reader beneath these always said to a 16-bit frame.
    require_8bit(frame)?;
    let src = Src::new(mmap, frame, file_order);
    core::vet(src)?;
    let n = frame.pixel_count()?;
    let mut rgb = vec![vec![0u8; n]; count];
    super::cmyk::rgb8(src, first, &mut slices(&mut rgb), &mut DecodeScratch::new())?;
    Ok(rgb)
}

/// `count` RGB planes of a CMYK frame from `first`, 16-bit.
fn rgb_u16(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    first: usize,
    count: usize,
) -> Result<Vec<Vec<u16>>> {
    require_cmyk(frame, "read_planes_rgb_u16")?;
    let src = Src::new(mmap, frame, file_order);
    core::vet(src)?;
    let n = frame.pixel_count()?;
    let mut rgb = vec![vec![0u16; n]; count];
    super::cmyk::rgb16(src, first, &mut slices(&mut rgb), &mut DecodeScratch::new())?;
    Ok(rgb)
}

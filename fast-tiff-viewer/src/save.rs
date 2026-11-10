//! Writing the open stack back out as a TIFF.
//!
//! The promise is "what is open, in a file" — every plane, and the metadata
//! needed to open it again the same way: the c/z/t interpretation, the display
//! mode and per-channel LUTs, the calibration, and whatever the source said
//! about itself in tag 270.
//!
//! # It is a re-encode, not a copy
//!
//! The planes are decoded and written again rather than the original bytes
//! being copied through. That is the point rather than a cost: what is open may
//! never have been a file — a plugin's result, or an OIR the importer built in
//! memory — and even when it was, the axes may have been reinterpreted since.
//! Writing what the viewer holds is what makes the saved file agree with the
//! window it came from.
//!
//! What it does *not* do is re-interpret the samples. A frame goes out in the
//! type it came in as, its samples copied as the file's own bytes rather than
//! through any of the decoders' display transforms.
//!
//! # Compressed: Deflate, level 1, no predictor
//!
//! Lossless, and read by everything that reads TIFF — ImageJ and Fiji,
//! Bio-Formats, tifffile, MATLAB, libtiff.
//!
//! The level and predictor were chosen on real two-photon recordings rather
//! than taken from convention. On a 1,463-frame, 512x512, 16-bit acquisition
//! (732 MB uncompressed, compressed in parallel):
//!
//! ```text
//!   Deflate               write    size    decode per frame
//!   level 1               0.93 s   66.5%   1.82 ms
//!   level 4               3.71 s   68.9%   2.44 ms
//!   level 6               6.39 s   68.0%   2.35 ms
//!   level 6 + predictor   4.26 s   71.9%   2.66 ms
//! ```
//!
//! Level 1 is the smallest *and* the fastest both ways, and the predictor makes
//! these files larger: shot noise is what a two-photon frame is mostly made
//! of, and differencing noise from its neighbour only makes more of it. On
//! smooth images — averaged, widefield, electron microscopy — a predictor and a
//! higher level can do better, so this is a choice for the data this viewer is
//! mostly pointed at, not a universal one.
//!
//! # Except with an overlay: uncompressed
//!
//! A stack carrying an ImageJ overlay ([`Stack::overlay`]: the points the file
//! came with, or the point counter's) is written uncompressed instead, and the
//! overlay with it. ImageJ reads the overlay of a multi-image TIFF only when it
//! opens the file as one contiguous stack, and it does that only for an
//! uncompressed one: its `TiffDecoder` takes `images=` from the description
//! only when the compression is none, and its other route through a
//! multi-image file never looks at the overlay. Compressed, the points would be
//! in the file and ImageJ would not show them. The writer's uncompressed layout
//! is the contiguous one it needs — see [`write_all`].
//!
//! Unless uncompressed is too big for a classic TIFF. Past 4 GiB the writer
//! turns to BigTIFF, which ImageJ 1.x does not open at all, so such a stack is
//! compressed as any other: the overlay is still written, for this viewer to
//! read back, and ImageJ — when the compressed file is small enough for it to
//! open — shows the image without it.
//!
//! # Two destinations, one encoder
//!
//! [`save_source`] writes a file; [`save_to_bytes`] fills a `Vec<u8>`. They are
//! the same encode — [`write_all`] takes any `Write + Seek` — and differ only
//! in where the bytes land and in what can be promised about the result.
//!
//! The file path is the careful one: it writes beside the target and renames,
//! so nothing is destroyed until the write has succeeded (see [`save_source`]).
//! The byte path cannot destroy anything, having nothing to overwrite, and so
//! needs none of that. It exists for the browser, where a download *is* a
//! buffer handed to the user agent and there is no filesystem to be careful
//! with — and the whole file is therefore in memory at once, which is the one
//! real difference between the two. A stack larger than the tab can hold fails
//! here where the desktop would have streamed it.

use crate::display::Dims;
use crate::stack::Stack;
use anyhow::{bail, Context, Result};
use fast_tiff_lib::metadata::{self, MetadataFormat};
use fast_tiff_lib::{
    read_frame_bytes_into, ByteOrder, Compression, FrameInfo, SampleFormat, SampleType,
    StackMetaWrite, TiffWriter, WriterOptions,
};

/// How files are compressed. See the module docs for why these.
const COMPRESSION: Compression = Compression::Deflate;
const COMPRESSION_LEVEL: i32 = 1;

/// How many bytes of decoded frames are gathered before they are compressed
/// together. Compression is per strip, and a frame under a megapixel is one or
/// two strips, so frames handed over singly compress on one core; a batch gives
/// every core a strip. This bounds the memory that costs.
const BATCH_BYTES: usize = 64 << 20;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// How a frame's samples travel from the file to the file.
///
/// The frames are copied as the file's own bytes, so this only names the type
/// they are written as — and refuses the ones the writer cannot reproduce
/// without changing the samples.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Samples {
    U8,
    I8,
    U16,
    I16,
    F32,
}

impl Samples {
    fn of(frame: &FrameInfo) -> Result<Samples> {
        Ok(match (frame.bits_per_sample, frame.sample_format) {
            (8, SampleFormat::UnsignedInt) => Samples::U8,
            (8, SampleFormat::SignedInt) => Samples::I8,
            // 4-bit palette indices are read one per byte, so that is how they
            // are written: the values survive, the packing does not.
            (4, SampleFormat::UnsignedInt) => Samples::U8,
            (16, SampleFormat::UnsignedInt) => Samples::U16,
            (16, SampleFormat::SignedInt) => Samples::I16,
            (32, SampleFormat::Float) => Samples::F32,
            (bits, format) => bail!(
                "this stack is {bits}-bit {format:?}, which this writer cannot reproduce \
                 without changing the samples"
            ),
        })
    }

    fn sample_type(self) -> SampleType {
        match self {
            Samples::U8 => SampleType::U8,
            Samples::I8 => SampleType::I8,
            Samples::U16 => SampleType::U16,
            Samples::I16 => SampleType::I16,
            Samples::F32 => SampleType::F32,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Samples::U8 | Samples::I8 => 1,
            Samples::U16 | Samples::I16 => 2,
            Samples::F32 => 4,
        }
    }
}

/// Everything a save reads, owned.
///
/// A save is a decode-and-encode of every frame, which on a big stack is
/// seconds — long enough that it belongs on a worker thread. This is what goes
/// there. It is not a copy of the image: `tiff` is a second handle on the same
/// indexed file, and `display` is the axis/contrast model, which is a few
/// vectors of settings. Taking it costs a refcount bump and one small clone.
///
/// Owned rather than borrowed because the window carries on being used while
/// the write runs, and a `&Stack` held across that would stop it.
pub struct SaveSource {
    tiff: std::sync::Arc<fast_tiff_lib::TiffStack>,
    display: crate::display::Display,
    /// [`Stack::overlay`], as it was when the snapshot was taken.
    overlay: Vec<fast_tiff_lib::PointRoi>,
}

impl SaveSource {
    /// Snapshot the stack as it is being looked at *now*.
    ///
    /// The metadata written comes from the display model, so changing the
    /// contrast or the axes after pressing Save must not change the file that
    /// is already being written — hence a snapshot rather than a live borrow.
    pub fn of(stack: &Stack) -> Self {
        SaveSource {
            tiff: std::sync::Arc::clone(&stack.tiff),
            display: stack.display.clone(),
            overlay: stack.overlay.clone(),
        }
    }

    /// How many frames the write will step through, for a progress readout.
    pub fn frames(&self) -> usize {
        self.tiff.frames.len()
    }
}

/// Write `stack` to `path` as a TIFF.
///
/// The blocking form, for callers with nothing to report progress to — tests,
/// and any future headless use. [`save_source`] is what the app calls.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_stack(stack: &Stack, path: &Path) -> Result<()> {
    save_source(&SaveSource::of(stack), path, &mut |_| true)
}

/// Write a snapshot to `path`, reporting progress and stopping when asked.
///
/// Streams: frames are decoded and written a batch at a time, so saving a stack
/// costs a few tens of megabytes of memory rather than a second copy of the
/// whole thing. [`save_to_bytes`] is the same encode without that property, for
/// a host that has no file to stream into.
///
/// `on_progress` is called once per frame with the fraction completed and
/// returns `false` to cancel.
///
/// # Nothing is destroyed until the write has succeeded
///
/// The pixels go to a temporary file beside the target and are renamed over it
/// at the end. Two things make that worth the extra step rather than writing
/// straight to `path`:
///
/// * A half-written TIFF has a valid header and a short IFD chain, so it
///   *opens* — as a file with fewer frames than it should have. Nothing
///   downstream can tell it is incomplete. Cancelling a save, or having one
///   fail, must not produce one.
/// * Saving over an existing file truncates it the moment the writer is
///   created. A save that then fails half way would have destroyed the file it
///   was supposed to replace — and "save over the file I am looking at" is a
///   thing people do. Writing beside it means the original survives every
///   failure, including the one where the rename itself is refused because
///   another window has that file memory-mapped (Windows os error 1224).
#[cfg(not(target_arch = "wasm32"))]
pub fn save_source(
    source: &SaveSource,
    path: &Path,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Result<()> {
    let temp = partial_path(path);
    match write_to_path(source, &temp, on_progress, BATCH_BYTES) {
        Ok(()) => std::fs::rename(&temp, path).with_context(|| {
            // Best effort: leaving the part file behind after a failed rename
            // would be a mystery file next to the one the user asked for.
            let _ = std::fs::remove_file(&temp);
            format!("could not put the finished file at {}", path.display())
        }),
        Err(e) => {
            // Best effort: if the part file cannot be removed the error being
            // reported is still the more useful one.
            let _ = std::fs::remove_file(&temp);
            Err(e)
        }
    }
}

/// Encode a snapshot into memory, reporting progress and stopping when asked.
///
/// The same TIFF [`save_source`] would have written — same compression, same
/// metadata, same bytes — built in a buffer instead of a file, for a host with
/// nowhere to put one. The browser build's only route, and the counterpart to
/// [`plugins::to_tiff_bytes`](crate::plugins::to_tiff_bytes), which does this
/// for a plugin's result rather than for what is open.
///
/// Not a replacement for [`save_source`] where a path exists. This holds the
/// finished file in memory, so it peaks at roughly its own size on top of
/// everything else; the path version streams and peaks at a batch.
///
/// Cancelling yields `Err`, as it does there, and the buffer is dropped. There
/// is no half-written file to clean up, which is the one way this is the
/// simpler of the two.
pub fn save_to_bytes(
    source: &SaveSource,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Result<Vec<u8>> {
    let sink = write_all(
        source,
        std::io::Cursor::new(Vec::new()),
        on_progress,
        BATCH_BYTES,
    )?;
    Ok(sink.into_inner())
}

/// Create `path` and encode into it, buffered.
///
/// Split from [`write_all`] so that the encoder does not know what a file is:
/// everything above this line is shared with the browser, and everything below
/// it is the part that needs a filesystem.
#[cfg(not(target_arch = "wasm32"))]
fn write_to_path(
    source: &SaveSource,
    path: &Path,
    on_progress: &mut dyn FnMut(f32) -> bool,
    batch_bytes: usize,
) -> Result<()> {
    use std::io::Write;
    let file = std::fs::File::create(path)
        .with_context(|| format!("could not create {}", path.display()))?;
    let mut sink = write_all(
        source,
        std::io::BufWriter::new(file),
        on_progress,
        batch_bytes,
    )?;
    // Explicitly, rather than on drop: a `BufWriter` dropped with bytes still
    // in it flushes and discards the error, which for the last strip of a save
    // would mean a truncated file reported as a success.
    sink.flush()
        .with_context(|| format!("finishing {}", path.display()))?;
    Ok(())
}

/// Where the pixels go until the write has succeeded.
///
/// In the same directory as the target, because a rename across filesystems is
/// a copy — and the temp directory is routinely on a different volume from the
/// data drive a microscopy stack is being saved to.
#[cfg(not(target_arch = "wasm32"))]
fn partial_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".fasttiff-part");
    path.with_file_name(name)
}

/// The encode itself, into any sink, returning it.
///
/// Split out twice over: so [`save_source`] can clean up after it without an
/// early `return` skipping the cleanup, and so that the same code serves a file
/// and a `Vec<u8>`. Nothing in here knows which it has.
///
/// `batch_bytes` is [`BATCH_BYTES`] outside of tests, which need several
/// batches from a stack small enough to build in one.
fn write_all<W: std::io::Write + std::io::Seek>(
    source: &SaveSource,
    sink: W,
    on_progress: &mut dyn FnMut(f32) -> bool,
    batch_bytes: usize,
) -> Result<W> {
    // A save is something the user waits for, on a thread of its own: its
    // decodes and its compression use every core, whatever the view is doing.
    let _parallel = fast_tiff_lib::parallel_decode_scope(true);
    let stack = source;
    let first = stack
        .tiff
        .frames
        .first()
        .context("this stack has no frames to save")?;
    let samples = Samples::of(first)?;
    let spp = first.samples_per_pixel.max(1);

    // One TIFF, one shape. The reader will happily index a file whose frames
    // differ — some scanners write a thumbnail as the first IFD — but a stack
    // of mixed shapes is not something this can write back out, and saying so
    // beats writing a file whose later frames are the wrong size.
    for (i, f) in stack.tiff.frames.iter().enumerate() {
        if (f.width, f.height, f.samples_per_pixel.max(1)) != (first.width, first.height, spp) {
            bail!(
                "frame {i} is {}x{}x{} where the first is {}x{}x{spp}; a stack of mixed \
                 frame shapes cannot be saved as one TIFF",
                f.width,
                f.height,
                f.samples_per_pixel.max(1),
                first.width,
                first.height
            );
        }
        if Samples::of(f)? != samples {
            bail!("frame {i} does not have the same sample type as the first");
        }
    }

    let bytes_per_frame = frame_bytes(first, samples);

    // The frames are copied as the file lays them out, so a planar source is
    // written planar.
    let options = WriterOptions::new(first.width, first.height, samples.sample_type())
        .samples_per_pixel(spp)
        .planar(spp > 1 && first.is_planar())
        .metadata(metadata_of(stack));
    // An overlay is only seen by ImageJ in an uncompressed stack (module docs).
    // Uncompressed, the writer puts each frame in one strip (no
    // `rows_per_strip` is set here), every strip straight after the last from
    // offset 16, and the IFDs after all of them at `finish` — so frame `i`
    // starts at `16 + i * frame_bytes`, which is the contiguous layout ImageJ
    // reads `images=N` frames from.
    //
    // Not when that is past what a classic TIFF holds (module docs).
    let uncompressed = !stack.overlay.is_empty()
        && fits_classic_uncompressed(stack.tiff.frames.len(), bytes_per_frame);
    let options = if !uncompressed {
        options
            .compression(COMPRESSION)
            .compression_level(COMPRESSION_LEVEL)
    } else {
        options.compression(Compression::None)
    };
    let mut writer = TiffWriter::new(sink, options).context("starting the TIFF")?;

    let data = &stack.tiff.data;
    let order = stack.tiff.byte_order;
    let per_batch = (batch_bytes / bytes_per_frame).max(1);
    // Buffers reused from batch to batch; `filled` of them hold this batch.
    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(per_batch);
    let mut filled = 0usize;
    let mut first_in_batch = 0usize;

    let total = stack.tiff.frames.len().max(1);
    for (i, frame) in stack.tiff.frames.iter().enumerate() {
        // Before the frame, not after: a stack of one frame should still show
        // that the write started, and the `finish` below is the tail this can
        // never account for.
        if !on_progress(i as f32 / total as f32) {
            bail!("cancelled");
        }
        if filled == batch.len() {
            batch.push(Vec::with_capacity(bytes_per_frame));
        }
        // The file's own samples, in the writer's byte order, decoded straight
        // into the batch: no display transform to undo, and no copy between.
        read_frame_bytes_into(data, frame, order, ByteOrder::Little, &mut batch[filled])
            .with_context(|| format!("writing frame {i}"))?;
        filled += 1;
        if filled == per_batch {
            write_batch(&mut writer, &batch[..filled])
                .with_context(|| format!("writing frames {first_in_batch}..={i}"))?;
            filled = 0;
            first_in_batch = i + 1;
        }
    }
    if filled > 0 {
        write_batch(&mut writer, &batch[..filled])
            .with_context(|| format!("writing frames {first_in_batch} onwards"))?;
    }
    writer.finish().context("finishing the TIFF")
}

/// Hand a batch of frames to the writer, which compresses them together.
fn write_batch(
    writer: &mut TiffWriter<impl std::io::Write + std::io::Seek>,
    frames: &[Vec<u8>],
) -> Result<()> {
    let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
    Ok(writer.write_frames_bytes(&refs)?)
}

/// Whether `frames` frames of `bytes_per_frame` each fit a classic TIFF
/// uncompressed — under 4 GiB, with room to spare for the IFDs (a kilobyte a
/// frame is several times what one takes), the description and the overlay.
/// Past it the writer would make a BigTIFF, which ImageJ 1.x cannot open.
fn fits_classic_uncompressed(frames: usize, bytes_per_frame: usize) -> bool {
    let size = (frames as u64)
        .saturating_mul((bytes_per_frame as u64).saturating_add(1024))
        .saturating_add(64 << 20);
    size <= u64::from(u32::MAX)
}

/// Whether ImageJ opens a stack of these axes, as a save writes them, as a
/// hyperstack: one with more than one channel (a composite), or with slices
/// and frames both. An RGB stack's channels are its samples, not an axis.
pub fn opens_as_imagej_hyperstack(dims: Dims, rgb: bool) -> bool {
    let channels = if rgb { 1 } else { dims.channels };
    channels > 1 || (dims.slices > 1 && dims.frames > 1)
}

/// The hyperstack position — channel, slice, frame, each from 1 — of flat
/// stack position `position` (the IFD index plus one) in a stack of `dims`,
/// whose IFDs are in the channel-fastest order [`plane_index`](crate::plane_index)
/// gives. In RGB the channels are samples of one IFD, and ImageJ has a single
/// channel: 0 for it, which is "all".
pub fn flat_to_hyperstack(position: u32, dims: Dims, rgb: bool) -> [u32; 3] {
    let ifd = position.saturating_sub(1) as usize;
    let (channels, slices) = (dims.channels.max(1), dims.slices.max(1));
    let (c, rest) = if rgb {
        (0, ifd)
    } else {
        (ifd % channels + 1, ifd / channels)
    };
    let (z, t) = (rest % slices + 1, rest / slices + 1);
    let n = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    [n(c), n(z), n(t)]
}

/// The flat stack position (the IFD index plus one) of hyperstack position
/// `[c, z, t]` in a stack of `dims`: [`flat_to_hyperstack`] the other way. An
/// axis at 0 — all of it — is taken at its first index, or at its last with
/// `last`, so the two give the span such a position covers. `None` for an
/// index past the axes.
pub fn hyperstack_to_flat([c, z, t]: [u32; 3], dims: Dims, rgb: bool, last: bool) -> Option<u32> {
    let channels = if rgb { 1 } else { dims.channels.max(1) };
    let (slices, frames) = (dims.slices.max(1), dims.frames.max(1));
    let index = |v: u32, n: usize| -> Option<usize> {
        match v as usize {
            0 if last => Some(n - 1),
            0 => Some(0),
            v if v <= n => Some(v - 1),
            _ => None,
        }
    };
    let (c, z, t) = (index(c, channels)?, index(z, slices)?, index(t, frames)?);
    let ifd = (t * slices + z) * channels + c;
    u32::try_from(ifd + 1).ok()
}

/// `roi` placed so that ImageJ draws it where its points are, on a stack
/// saved with axes `dims`; `read_in` are the axes its hyperstack position, if
/// it has one, is in — the file's own, for a ROI read from it.
///
/// On a hyperstack ImageJ reads a point ROI's one shared position as a slice
/// or frame number rather than the flat stack index it is, and draws the ROI
/// on no plane or the wrong one (see
/// [`PointRoi::hyperstack_position`](fast_tiff_lib::PointRoi::hyperstack_position)).
/// So:
///
/// * a ROI whose points share one position gets the hyperstack position of
///   that plane, worked out afresh for `dims` — the flat position is what
///   stays put when the axes are reassigned, where a hyperstack position
///   worked out for other axes would point at a plane that is not there;
/// * a ROI tied to a plane by its hyperstack position alone, or whose points
///   are spread over planes, keeps it, moved from `read_in` to `dims` when
///   they differ — an axis that was "all" stays all;
/// * on a stack that is not a hyperstack, ImageJ goes by the ROI's position,
///   and a ROI that has one carries no hyperstack position to mislead it.
fn placed_for_imagej(
    mut roi: fast_tiff_lib::PointRoi,
    dims: Dims,
    rgb: bool,
    read_in: Dims,
) -> fast_tiff_lib::PointRoi {
    let shared = roi
        .points
        .first()
        .map(|p| p.position)
        .filter(|&first| first != 0 && roi.points.iter().all(|p| p.position == first));
    if !opens_as_imagej_hyperstack(dims, rgb) {
        if shared.is_some() {
            roi.hyperstack_position = [0; 3];
        }
        return roi;
    }
    if let Some(position) = shared {
        roi.hyperstack_position = flat_to_hyperstack(position, dims, rgb);
    } else if roi.hyperstack_position != [0; 3] && read_in != dims {
        let old = roi.hyperstack_position;
        if let Some(flat) = hyperstack_to_flat(old, read_in, rgb, false) {
            let mut new = flat_to_hyperstack(flat, dims, rgb);
            for (axis, was) in new.iter_mut().zip(old) {
                if was == 0 {
                    *axis = 0;
                }
            }
            roi.hyperstack_position = new;
        }
    }
    roi
}

/// Whether ImageJ will read the overlay back from a save of `stack`: whether
/// the save is uncompressed, which it is only while that fits a classic TIFF
/// (see the module docs). Past that the overlay is still in the file, for this
/// viewer, but ImageJ shows the image without it.
pub fn overlay_readable_by_imagej(stack: &Stack) -> bool {
    let Some(first) = stack.tiff.frames.first() else {
        return true;
    };
    let Ok(samples) = Samples::of(first) else {
        // Not a stack this writer saves at all.
        return true;
    };
    fits_classic_uncompressed(stack.tiff.frames.len(), frame_bytes(first, samples))
}

/// One frame's uncompressed size as the writer writes it.
fn frame_bytes(first: &FrameInfo, samples: Samples) -> usize {
    (first.width as usize * first.height as usize * first.samples_per_pixel.max(1) as usize)
        .saturating_mul(samples.bytes())
        .max(1)
}

/// The metadata to write beside the pixels.
///
/// Taken from the *display* rather than from the file's own header wherever the
/// two can differ. A stack whose axes were reinterpreted, or whose channels were
/// recoloured, is saved as it is being looked at — which is the only reading of
/// "save" that does not surprise someone who just changed something.
fn metadata_of(stack: &SaveSource) -> StackMetaWrite {
    let dims = stack.display.dims();
    let meta = &stack.tiff.meta;
    let mut out = StackMetaWrite::new(dims.channels, dims.slices).mode(stack.display.mode);

    if let Some(unit) = &meta.unit {
        out = out.unit(unit.clone());
    }
    if let Some(fps) = meta.fps {
        out = out.fps(fps);
    }
    if let Some(interval) = meta.frame_interval_s {
        out = out.frame_interval_s(interval);
    }
    if let Some(spacing) = meta.spacing {
        out = out.spacing(spacing);
    }
    if let Some(looped) = meta.loop_playback {
        out = out.loop_playback(looped);
    }
    if let Some((c0, c1)) = meta.calibration {
        out = out.calibration(c0, c1);
    }
    if let (Some(w), Some(h)) = (meta.pixel_width, meta.pixel_height) {
        out = out.pixel_size(w, h);
    }
    // The window on screen. The format carries one for the stack, so the first
    // channel's speaks for it — the alternative is to write none and have the
    // file open on auto-contrast, which is a visible change to something the
    // user did not touch. In ImageJ's units for the samples, which are the
    // file's own except for signed 16-bit (see `SampleUnits::to_imagej`), and
    // the same definition opening a file reads the range back with.
    if let Some(first) = stack.display.settings().first() {
        let (lo, hi) = first.units.to_imagej(first.window);
        out = out.range(lo, hi);
    }
    // Every channel's LUT, always — including plain grayscale ones. They are
    // 768 bytes each, and writing them is what makes a saved file open in the
    // colours it was saved in rather than in whichever ones a reader defaults
    // to for that channel count.
    for lut in stack.display.luts() {
        out = out.channel_lut(*lut);
    }
    if let Some(text) = carried_description(stack) {
        out = out.trailing(text);
    }
    // The overlay — the file's own points, or the point counter's — as ImageJ
    // overlay ROIs, each where ImageJ will draw it on these axes.
    if !stack.overlay.is_empty() {
        let rgb = stack.display.rgb();
        // The axes the file was written with, which a hyperstack position read
        // from it is in.
        let read_in = Dims {
            channels: meta.channels,
            slices: meta.slices,
            frames: meta.frames,
        };
        out = out.overlay(
            stack
                .overlay
                .iter()
                .map(|roi| placed_for_imagej(roi.clone(), dims, rgb, read_in)),
        );
    }
    out
}

/// Whatever the source's `ImageDescription` said that this writer is not about
/// to say again.
///
/// A vendor record — the FluoView export an OIR import leaves there, an OME
/// document, free text from another tool — is the part of a file that cannot be
/// reconstructed, so it travels. The ImageJ `key=value` block in front of it
/// does not: every key in it is being regenerated from the values above, and
/// carrying the old copy through would leave two of each in the file.
fn carried_description(stack: &SaveSource) -> Option<String> {
    let text = stack.tiff.description.as_deref()?;
    let kept = match metadata::detect(Some(text)) {
        MetadataFormat::ImageJ => text
            .lines()
            .skip_while(|l| is_key_value(l))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => text.to_string(),
    };
    (!kept.trim().is_empty()).then_some(kept)
}

/// Whether a line is one of ImageJ's own `key=value` entries.
///
/// Deliberately narrow: a key is a bare identifier. The `"key"\t"value"` lines
/// of an instrument's export are not one, and neither is anything with a space
/// or a quote in it, which is what keeps this from eating the record it is
/// meant to preserve.
fn is_key_value(line: &str) -> bool {
    match line.split_once('=') {
        Some((key, _)) => {
            !key.is_empty()
                && key
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        }
        None => false,
    }
}

#[cfg(test)]
#[path = "save_tests.rs"]
mod tests;

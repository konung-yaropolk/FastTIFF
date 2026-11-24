//! One frame of a stack, ready to read: [`Frame`], and what a read takes and
//! gives back.
//!
//! Every read goes through two calls, generic over the [units](crate::units)
//! the samples come back in:
//!
//! - [`Frame::read`] fills a `Vec` the caller keeps, or hands back the source's
//!   own bytes when they already are the answer (an uncompressed plane in the
//!   right byte order). That is the call for a view, which wants zero copies
//!   when the file allows it.
//! - [`Frame::read_into`] writes into exact slices the caller owns — one per
//!   plane, so an RGB frame lands in three buffers from one decompression pass.
//!   That is the call for a plugin's buffer, a GPU staging slice or a batch.
//!
//! Both take a [`DecodeScratch`], the caller's, which holds whatever a decode
//! needs between the file and the destination (a strip's worth of decompressed
//! bytes, or for the few reads that need it the whole frame). A caller that
//! keeps one across frames decodes with no frame-sized allocation and no
//! zero-fill once it is warm — serially with no allocation at all; a read
//! split across cores still makes a few pointer-sized lists for the split —
//! and nothing frame-sized is kept in a thread-local behind its back.
//!
//! The contract every read keeps:
//!
//! - `Ok` means every element of every destination was written.
//! - `Err` leaves the destinations unspecified. They are never read before
//!   they are written, so a recycled buffer, or one a plugin owns, needs no
//!   zero-fill first.
//! - [`read`](Frame::read) resizes its `Vec` only when the length differs.
//! - A plane index at or past [`planes`](Frame::planes) is
//!   [`ErrorKind::InvalidInput`](crate::ErrorKind::InvalidInput), as is a
//!   destination of the wrong length.

use crate::decode::core::{self, Src};
use crate::display::DisplayMap;
use crate::error::{bail, err, Error, Result};
use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo, TiffStack};
use crate::units::{sealed, Units};
use std::borrow::Cow;
use std::ops::Range;

/// Where a frame's bytes are: the file, or a slice of it that starts at a
/// known file offset.
///
/// The one place a file offset becomes an index into memory is
/// [`piece`](Self::piece), in checked `u64` arithmetic — so a strip offset past
/// 4 GiB means the same on wasm32, where `usize` is 32 bits, as on a 64-bit
/// host, rather than wrapping into the wrong bytes.
#[derive(Clone, Copy)]
pub(crate) enum FrameBytes<'a> {
    /// `bytes[0]` is the file's byte `base`.
    Slice { base: u64, bytes: &'a [u8] },
}

/// The span, not the bytes: a frame's bytes are the whole file, and
/// formatting a mapping of gigabytes is a stall, not a diagnostic.
impl std::fmt::Debug for FrameBytes<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let FrameBytes::Slice { base, bytes } = *self;
        f.debug_struct("Slice")
            .field("base", &base)
            .field("len", &bytes.len())
            .finish()
    }
}

impl<'a> FrameBytes<'a> {
    /// A whole file.
    pub(crate) fn whole(bytes: &'a [u8]) -> Self {
        FrameBytes::Slice { base: 0, bytes }
    }

    /// A slice whose first byte is the file's byte `base`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn relative(base: u64, bytes: &'a [u8]) -> Self {
        FrameBytes::Slice { base, bytes }
    }

    /// The `len` bytes at file offset `off`, or [`ErrorKind::Malformed`] when
    /// any of them is outside what this holds.
    pub(crate) fn piece(&self, off: u64, len: u64) -> Result<&'a [u8]> {
        let FrameBytes::Slice { base, bytes } = *self;
        let range = off
            .checked_sub(base)
            .and_then(|rel| Some((rel, rel.checked_add(len)?)))
            .and_then(|(a, b)| Some(usize::try_from(a).ok()?..usize::try_from(b).ok()?));
        range.and_then(|r| bytes.get(r)).ok_or_else(|| {
            err!(
                Malformed,
                "strip at offset {off} (len {len}) out of file bounds"
            )
        })
    }

    /// [`piece`](Self::piece), cut short at the end of the bytes rather than
    /// refused — what the 4-bit reader has always done with a strip that runs
    /// off the end. Only a start past the end is an error.
    pub(crate) fn piece_clamped(&self, off: u64, len: u64) -> Result<&'a [u8]> {
        let FrameBytes::Slice { base, bytes } = *self;
        let end = base.saturating_add(bytes.len() as u64);
        if off < base || off > end {
            bail!(Malformed, "strip at offset {off} out of file bounds");
        }
        self.piece(off, len.min(end - off))
    }
}

/// One frame of a stack, ready to read: its [`FrameInfo`], the file's byte
/// order, and the bytes it lives in.
///
/// Cheap to make and to clone — it borrows the stack — and `Send + Sync`, so
/// several threads can read one at once, each with its own [`DecodeScratch`].
#[derive(Clone, Debug)]
pub struct Frame<'a> {
    info: Cow<'a, FrameInfo>,
    order: ByteOrder,
    bytes: FrameBytes<'a>,
}

impl TiffStack {
    /// Frame `index`, ready to read.
    /// [`ErrorKind::InvalidInput`](crate::ErrorKind::InvalidInput) past the
    /// end.
    pub fn frame(&self, index: usize) -> Result<Frame<'_>> {
        let info = self.frames.get(index).ok_or_else(|| {
            err!(
                InvalidInput,
                "frame {index} is past the end of a {}-frame stack",
                self.frames.len()
            )
        })?;
        Ok(Frame::new(&self.data, info, self.byte_order))
    }

    /// How many frames the stack holds.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

impl<'a> Frame<'a> {
    /// A frame of a whole file held in memory, described by `info`: for fuzzing,
    /// tests and embedders that index files themselves. A frame of a stack
    /// comes from [`TiffStack::frame`].
    pub fn new(file: &'a [u8], info: &'a FrameInfo, order: ByteOrder) -> Self {
        Frame {
            info: Cow::Borrowed(info),
            order,
            bytes: FrameBytes::whole(file),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_bytes(bytes: FrameBytes<'a>, info: &'a FrameInfo, order: ByteOrder) -> Self {
        Frame {
            info: Cow::Borrowed(info),
            order,
            bytes,
        }
    }

    /// What the frame is: its geometry, samples and strip table.
    pub fn info(&self) -> &FrameInfo {
        &self.info
    }

    /// The byte order the file stores its samples in.
    pub fn byte_order(&self) -> ByteOrder {
        self.order
    }

    pub(crate) fn src(&self) -> Src<'a, '_> {
        Src {
            bytes: self.bytes,
            info: &self.info,
            order: self.order,
        }
    }

    /// The pieces covering part of this frame, as a frame of its own, and what
    /// it covers. Arithmetic on the strip table only: nothing is read.
    ///
    /// Strips and tiles are the unit of compression, so the result is snapped
    /// *outward* to them, and the [`Coverage`] says which rows and columns of
    /// this frame it really holds — what a caller indexes against. A stripped
    /// frame keeps its full width whatever [`Crop::Region`] asks for, because a
    /// strip cannot be narrowed.
    pub fn crop(&self, crop: Crop) -> Result<(Frame<'a>, Coverage)> {
        let info = &*self.info;
        let per_piece = if info.is_tiled() {
            info.tile_size.map_or(1, |(_, h)| h.max(1))
        } else {
            info.rows_per_strip.max(1)
        };
        let (cropped, coverage) = match crop {
            Crop::Rows(rows) => {
                let band = info.crop_rows(rows)?;
                let coverage = Coverage {
                    cols: 0..info.width,
                    first_row: band.rows.start,
                    rows_per_piece: per_piece,
                    pieces: band.len().div_ceil(per_piece).max(1),
                    step: 1,
                    rows: band.rows,
                };
                (band.frame, coverage)
            }
            Crop::Region { cols, rows } => {
                let region = info.crop(cols, rows)?;
                let len = region.rows.end.saturating_sub(region.rows.start);
                let coverage = Coverage {
                    first_row: region.rows.start,
                    rows_per_piece: per_piece,
                    pieces: len.div_ceil(per_piece).max(1),
                    step: 1,
                    rows: region.rows,
                    cols: region.cols,
                };
                (region.frame, coverage)
            }
            Crop::Sampled { rows, step } => {
                let band = info.crop_rows_step(rows, step)?;
                // The source rows the taken pieces span, from the first row of
                // the first to the last row of the last.
                let span = band
                    .pieces
                    .saturating_sub(1)
                    .saturating_mul(band.step)
                    .saturating_mul(band.rows_per_piece);
                let last_rows = band.frame.height.saturating_sub(
                    band.pieces
                        .saturating_sub(1)
                        .saturating_mul(band.rows_per_piece),
                );
                let end = band
                    .first_row
                    .saturating_add(span)
                    .saturating_add(last_rows)
                    .min(info.height);
                let coverage = Coverage {
                    cols: 0..info.width,
                    rows: band.first_row..end,
                    first_row: band.first_row,
                    rows_per_piece: band.rows_per_piece,
                    pieces: band.pieces,
                    step: band.step,
                };
                (band.frame, coverage)
            }
        };
        Ok((
            Frame {
                info: Cow::Owned(cropped),
                order: self.order,
                bytes: self.bytes,
            },
            coverage,
        ))
    }

    /// How many planes a read in `units` has: the frame's samples per pixel,
    /// or three (red, green, blue) for the CMYK conversions.
    pub fn planes<U: Units>(&self, units: U) -> usize {
        units.planes(self, sealed::Private)
    }

    /// The length one plane's destination must have: `U::Sample` elements.
    ///
    /// The frame's size guards run first, so a header claiming more than its
    /// strips can hold is an error here, before a caller sizes anything from
    /// it — a few hundred hostile bytes must not be able to demand gigabytes.
    pub fn plane_len<U: Units>(&self, units: U) -> Result<usize> {
        let n = units.plane_len(&self.info, sealed::Private)?;
        core::vet(self.src())?;
        Ok(n)
    }

    /// Whether [`read`](Self::read) of `plane` in `units` hands back the
    /// source's own bytes, copying nothing: the frame's samples are already
    /// what was asked for, in one contiguous run (an uncompressed,
    /// predictor-free plane, in the byte order and representation the units
    /// want, suitably aligned).
    ///
    /// The one zero-copy predicate. A viewer can ask it to choose between
    /// touching a frame's pages ahead and decoding it ahead.
    pub fn borrows<U: Units>(&self, units: U, plane: usize) -> bool {
        self.borrowed(units, plane).is_some()
    }

    /// The source's own samples of `plane` in `units`, when they are the
    /// answer ([`borrows`](Self::borrows)): what [`read`](Self::read) would
    /// hand back without decoding anything. `None` when a read would decode,
    /// or would fail (a plane past the last).
    ///
    /// A [`read`](Self::read)'s samples live only as long as its borrow of
    /// `out`, which they may be in. These live as long as the frame's source,
    /// so a caller reading several planes or frames at once — one destination
    /// each, some borrowed and some decoded — can keep the borrowed ones
    /// beside its buffers rather than inside them.
    pub fn borrowed<U: Units>(&self, units: U, plane: usize) -> Option<Pixels<'a, U::Sample>> {
        units
            .borrow(self, plane, sealed::Private)
            .map(Pixels::Borrowed)
    }

    /// Plane `plane` in `units`: the source's own bytes when they are the
    /// answer ([`Pixels::Borrowed`], see [`borrows`](Self::borrows)), otherwise
    /// decoded into `out`, which is resized only if its length is not the
    /// plane's ([`Pixels::Decoded`]).
    ///
    /// The report is what the units say about the decode: for
    /// [`Display`](crate::units::Display), the [`DisplayMap`] applied.
    pub fn read<'s, U: Units>(
        &self,
        units: U,
        plane: usize,
        out: &'s mut Vec<U::Sample>,
        scratch: &mut DecodeScratch,
    ) -> Result<(Pixels<'s, U::Sample>, U::Report)>
    where
        'a: 's,
    {
        if let Some(borrowed) = units.borrow(self, plane, sealed::Private) {
            let report = units.borrowed_report(self, sealed::Private);
            return Ok((Pixels::Borrowed(borrowed), report));
        }
        let planes = units.planes(self, sealed::Private);
        if plane >= planes {
            return Err(plane_past_end(plane, planes));
        }
        // Vetted before `out` is sized from the header: a hostile frame must
        // not be able to demand gigabytes before it is found out.
        fit(out, self.plane_len(units)?)?;
        let report = units.read_into(self, plane, &mut [&mut out[..]], scratch, sealed::Private)?;
        Ok((Pixels::Decoded(&out[..]), report))
    }

    /// Planes `first..first + out.len()` in `units`, one destination each,
    /// from one pass over the frame's pieces.
    ///
    /// Every destination must be exactly [`plane_len`](Self::plane_len) long,
    /// and `first + out.len()` at most [`planes`](Self::planes); otherwise
    /// [`ErrorKind::InvalidInput`](crate::ErrorKind::InvalidInput) with
    /// nothing written.
    pub fn read_into<U: Units>(
        &self,
        units: U,
        first: usize,
        out: &mut [&mut [U::Sample]],
        scratch: &mut DecodeScratch,
    ) -> Result<U::Report> {
        let planes = units.planes(self, sealed::Private);
        let end = first.saturating_add(out.len());
        if out.is_empty() || end > planes {
            if out.is_empty() {
                bail!(InvalidInput, "a read needs at least one destination");
            }
            return Err(plane_past_end(end - 1, planes));
        }
        let n = units.plane_len(self.info(), sealed::Private)?;
        if let Some(bad) = out.iter().position(|o| o.len() != n) {
            bail!(
                InvalidInput,
                "destination {bad} holds {} samples, but a plane of this frame is {n}",
                out[bad].len()
            );
        }
        units.read_into(self, first, out, scratch, sealed::Private)
    }

    /// Bytes [`stored_frame_into`](Self::stored_frame_into) writes: the whole
    /// frame, every sample, at its stored width (4-bit indices one per byte).
    ///
    /// Vetted like [`plane_len`](Self::plane_len).
    pub fn stored_frame_len(&self) -> Result<usize> {
        let n = core::stored_frame_len(self.info())?;
        core::vet(self.src())?;
        Ok(n)
    }

    /// The whole frame's samples exactly as the file stores them —
    /// decompressed, the predictor undone, nothing widened, offset or rescaled
    /// — laid out as the file lays them out (chunky stays interleaved, planar
    /// is one plane after another), with each sample's bytes in `order`. 4-bit
    /// palette indices come one per byte.
    ///
    /// For writing a frame back out unchanged, and for reading it as `&[i16]`,
    /// `&[u32]`, ... through bytemuck (pass [`ByteOrder::host`]). `out` must be
    /// [`stored_frame_len`](Self::stored_frame_len) bytes.
    pub fn stored_frame_into(
        &self,
        order: ByteOrder,
        out: &mut [u8],
        scratch: &mut DecodeScratch,
    ) -> Result<()> {
        let n = self.stored_frame_len()?;
        if out.len() != n {
            bail!(
                InvalidInput,
                "destination holds {} bytes, but this frame stores {n}",
                out.len()
            );
        }
        core::stored_frame_into(self.src(), order, out, scratch)
    }

    /// The whole frame as [`stored_frame_into`](Self::stored_frame_into)
    /// gives it, borrowed from the source when the source already is it — an
    /// uncompressed, predictor-free frame stored as one run, every plane of it,
    /// in `order` (8-bit samples are in every order) — and otherwise decoded
    /// into `out`, which is resized only if its length is not the frame's.
    ///
    /// What [`read`](Self::read) is to one plane, for the whole frame: the
    /// call for writing a frame back out, which wants no copy when the file
    /// allows it, whatever its samples per pixel.
    pub fn stored_frame<'s>(
        &self,
        order: ByteOrder,
        out: &'s mut Vec<u8>,
        scratch: &mut DecodeScratch,
    ) -> Result<Pixels<'s, u8>>
    where
        'a: 's,
    {
        if let Some(borrowed) = core::borrow_stored_frame(self.src(), order) {
            return Ok(Pixels::Borrowed(borrowed));
        }
        fit(out, self.stored_frame_len()?)?;
        core::stored_frame_into(self.src(), order, out, scratch)?;
        Ok(Pixels::Decoded(&out[..]))
    }

    /// The smallest and largest finite values of a 32- or 64-bit frame,
    /// integer or float, over every sample — the range ImageJ auto-contrasts a
    /// 32-bit image to. `(0, 1)` when there is no usable range (no finite
    /// value, or a constant frame); `None` for 8- and 16-bit frames, whose
    /// range is their type's.
    pub fn value_range(&self, scratch: &mut DecodeScratch) -> Result<Option<(f32, f32)>> {
        core::value_range(self.src(), scratch)
    }

    /// Touch every page of the frame's pieces, so a decode that follows does
    /// not stall on page faults. First access to a mapped page faults, which is
    /// cheap on Linux and costs real time on Windows; doing it on a worker for
    /// the next frame keeps those faults off the latency-critical path. Purely a
    /// performance hint: safe to skip, safe to repeat.
    pub fn touch(&self) {
        const PAGE: usize = 4096;
        let info = &*self.info;
        for (&off, &len) in info.strip_offsets.iter().zip(info.strip_byte_counts.iter()) {
            let Ok(strip) = self.bytes.piece_clamped(off, len) else {
                continue;
            };
            let mut i = 0;
            while i < strip.len() {
                std::hint::black_box(strip[i]);
                i += PAGE;
            }
            if let Some(&last) = strip.last() {
                std::hint::black_box(last);
            }
        }
    }
}

/// Size `out` to `n`, fallibly, when it is not that already; what it held is
/// left for the decode to overwrite.
fn fit<T: Default + Clone>(out: &mut Vec<T>, n: usize) -> Result<()> {
    if out.len() != n {
        out.try_reserve_exact(n.saturating_sub(out.len()))
            .map_err(|_| err!(TooLarge, "cannot allocate {n} samples for a read"))?;
        out.resize(n, T::default());
    }
    Ok(())
}

pub(crate) fn plane_past_end(plane: usize, planes: usize) -> Error {
    err!(
        InvalidInput,
        "plane {plane} is past the last of this read's {planes} plane(s)"
    )
}

/// What part of a frame to read: the argument of [`Frame::crop`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Crop {
    /// The strips (or tile rows) covering these rows, full width.
    Rows(Range<u32>),
    /// The tiles covering these columns and rows. A stripped frame cannot be
    /// narrowed, so it keeps its full width.
    Region { cols: Range<u32>, rows: Range<u32> },
    /// Every `step`-th strip from the one covering `rows.start`, enough of them
    /// to reach `rows.end` — for sampling a frame coarsely, at a cost
    /// proportional to what is sampled rather than to what is spanned. Stripped
    /// frames only; `step` 1 is [`Crop::Rows`].
    Sampled { rows: Range<u32>, step: u32 },
}

/// What a cropped frame holds of the frame it was cut from.
///
/// Row `k` of the cropped frame is source row `first_row + (k /
/// rows_per_piece) * step * rows_per_piece + (k % rows_per_piece)`; for every
/// crop but [`Crop::Sampled`], `step` is 1 and that is simply `first_row + k`.
/// [`decoded_row_of`](Self::decoded_row_of) answers the other way round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coverage {
    /// Which columns of the source it holds.
    pub cols: Range<u32>,
    /// Which source rows it spans, snapped outward to whole pieces. For a
    /// sampled crop the rows of the pieces stepped over are spanned but not
    /// held.
    pub rows: Range<u32>,
    /// Source row that decoded row 0 came from.
    pub first_row: u32,
    /// Rows in one piece: decoded rows arrive in runs of this many.
    pub rows_per_piece: u32,
    /// Pieces taken (per plane).
    pub pieces: u32,
    /// Pieces skipped between the ones taken, plus one: 1 means none were.
    pub step: u32,
}

impl Coverage {
    /// Where `source_row` landed in the cropped frame, if it is there.
    ///
    /// Being in a piece that was taken is necessary but not sufficient: the
    /// last piece of a frame is usually short, and if that is the piece taken
    /// then the rows past its end were never in the file. The answer is checked
    /// against the rows the crop really holds, so it never sends a caller off
    /// the end of the decoded plane.
    pub fn decoded_row_of(&self, source_row: u32) -> Option<u32> {
        let per = self.rows_per_piece.max(1);
        let step = self.step.max(1);
        let offset = source_row.checked_sub(self.first_row)?;
        if source_row >= self.rows.end {
            return None;
        }
        let piece = offset / per;
        if !piece.is_multiple_of(step) {
            return None; // fell in a piece that was stepped over
        }
        let taken = piece / step;
        if taken >= self.pieces {
            return None;
        }
        // Checked: `rows_per_piece` is the file's own RowsPerStrip, so a hostile
        // one can make this overflow, and a saturated index is still an index
        // the result does not have.
        taken.checked_mul(per)?.checked_add(offset % per)
    }
}

/// The samples a [`Frame::read`] produced.
///
/// Derefs to the slice either way; which variant says where it lives.
#[derive(Debug)]
pub enum Pixels<'s, T> {
    /// The source's own bytes — the file's, through the stack's mapping or
    /// buffer. Nothing was copied, and `out` was left as it was.
    Borrowed(&'s [T]),
    /// Decoded into the caller's `out`.
    Decoded(&'s [T]),
}

impl<T> Pixels<'_, T> {
    /// Whether these are the source's own bytes rather than `out`'s.
    pub fn is_source(&self) -> bool {
        matches!(self, Pixels::Borrowed(_))
    }
}

impl<T> std::ops::Deref for Pixels<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Pixels::Borrowed(s) | Pixels::Decoded(s) => s,
        }
    }
}

/// The display maps a [`Display`](crate::units::Display) read applied.
#[derive(Clone, Debug, PartialEq)]
pub enum DisplayMaps {
    /// One map for every plane read.
    One(DisplayMap),
    /// One map per plane read, in destination order: a read of several planes
    /// of a wide (32- or 64-bit) frame with no range, where each plane is
    /// ranged to its own values.
    PerPlane(Vec<DisplayMap>),
}

impl DisplayMaps {
    /// The map of destination `i` of the read that returned this.
    pub fn plane(&self, i: usize) -> DisplayMap {
        match self {
            DisplayMaps::One(map) => *map,
            DisplayMaps::PerPlane(maps) => maps[i],
        }
    }
}

/// What a decode needs between the file and the destination, kept by the
/// caller so that a warm one costs no frame-sized allocation and no
/// zero-fill.
///
/// A read holds at most one decompressed piece (a strip or tile) per thread
/// it runs on. A few need a whole frame (or plane) assembled first: a wide
/// frame shown as `u16` with no range, whose values have to be known before
/// any is converted; [`value_range`](Frame::value_range) of a tiled frame, or
/// of one whose strips decompress on several cores; and a planar CMYK frame,
/// whose four inks are needed at once. A buffer here grows when a frame needs
/// more than it has, and is zero-filled only then;
/// [`release`](Self::release) gives it all back (on a document change, say,
/// or after a one-off windowed read).
#[derive(Default)]
pub struct DecodeScratch {
    /// A whole frame's (or plane's) decoded bytes, for the reads that need
    /// them all before converting any.
    pub(crate) native: Vec<u8>,
    /// Piece buffers, one per thread a parallel decode ran on.
    pub(crate) pieces: Vec<Vec<u8>>,
}

impl DecodeScratch {
    /// An empty scratch: it grows on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes it holds now.
    pub fn held_bytes(&self) -> usize {
        self.native.capacity() + self.pieces.iter().map(Vec::capacity).sum::<usize>()
    }

    /// Give back everything it holds.
    pub fn release(&mut self) {
        *self = Self::default();
    }
}

/// What it holds, not the bytes: they are whatever the last decode left.
impl std::fmt::Debug for DecodeScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeScratch")
            .field("held_bytes", &self.held_bytes())
            .field("pieces", &self.pieces.len())
            .finish()
    }
}

impl FrameInfo {
    /// File byte range of piece-plane `plane` when it is stored as one
    /// contiguous run of samples a read could borrow as it is: uncompressed,
    /// predictor-free, stripped, every strip at least as long as its rows and
    /// each starting where the one before it ends. A piece-plane is a sample
    /// plane of a planar frame, and the whole frame otherwise.
    ///
    /// `None` when any of that does not hold, or the geometry is not one the
    /// decoders read (4-bit, an unknown depth, an overflowing size).
    pub(crate) fn stored_run(&self, plane: usize) -> Option<Range<u64>> {
        self.stored_run_planes(plane..plane + 1)
    }

    /// [`stored_run`](Self::stored_run) of several piece-planes at once: one
    /// run when they are stored back to back.
    pub(crate) fn stored_run_planes(&self, planes: Range<usize>) -> Option<Range<u64>> {
        if self.is_tiled() || self.compression != Compression::None || self.predictor != 1 {
            return None;
        }
        if planes.is_empty() || planes.end > self.piece_planes() {
            return None;
        }
        if let Some(offset) = self.whole_strip() {
            // One strip holding the whole frame, the planes one after another.
            // Checked like every other offset sum (`whole_strip` refuses a
            // strip whose end would overflow, but a `FrameInfo` is anyone's to
            // build): a wrapped offset would be someone else's bytes.
            let plane_bytes = (self.decoded_len().ok()? / self.piece_planes()) as u64;
            let at = |p: usize| offset.checked_add((p as u64).checked_mul(plane_bytes)?);
            return Some(at(planes.start)?..at(planes.end)?);
        }
        let sb = self.sample_bytes().ok()?;
        self.decoded_len().ok()?;
        let per_px = if self.is_planar() {
            1
        } else {
            (self.samples_per_pixel as usize).max(1)
        };
        let row_bytes = (self.width as u64) * per_px as u64 * sb as u64;
        let rps = self.rows_per_strip.max(1) as u64;
        let height = self.height as u64;
        let per_plane = height.div_ceil(rps) as usize;
        let available = self.strip_offsets.len().min(self.strip_byte_counts.len());
        let first = planes.start.checked_mul(per_plane)?;
        let last = planes.end.checked_mul(per_plane)?;
        if last > available {
            return None;
        }
        let start = self.strip_offsets[first];
        let mut cursor = start;
        for j in first..last {
            let k = (j % per_plane) as u64;
            let rows = rps.min(height - k * rps);
            let need = rows * row_bytes;
            if self.strip_offsets[j] != cursor || self.strip_byte_counts[j] < need {
                return None;
            }
            cursor = cursor.checked_add(need)?;
        }
        Some(start..cursor)
    }
}

/// Cast a borrowed run to its samples, when it is aligned for them.
pub(crate) fn cast_run<T: bytemuck::Pod>(bytes: &[u8]) -> Option<&[T]> {
    bytemuck::try_cast_slice(bytes).ok()
}

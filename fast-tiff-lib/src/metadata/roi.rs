//! ImageJ's binary ROI format, for the one shape the viewer makes: a multi-point
//! selection.
//!
//! Two containers carry these bytes, and this module writes the bytes for both:
//!
//! * a **RoiSet `.zip`** ([`roi_set_zip`]) — what ImageJ's ROI Manager saves and
//!   opens, one `.roi` file per entry;
//! * a TIFF's **overlay** — one `over` entry per ROI in the `IJMetadata` block,
//!   which ImageJ draws over the image when it opens the file (see
//!   [`StackMetaWrite::overlay`](super::StackMetaWrite::overlay)).
//!
//! It reads them back too, from either writer: [`point_rois_from_imagej`] is
//! the encoder's inverse, which is how a file's overlay reaches
//! [`StackMeta::overlay`](super::StackMeta::overlay).
//!
//! The layout is `ij/io/RoiEncoder.java` and `ij/io/RoiDecoder.java` (ImageJ1
//! master, checked at commit `a376e43`): a 64-byte header, the coordinates, a
//! second 64-byte header, then the name and the per-point counters. Every
//! number in it is **big-endian**, whatever byte order the TIFF around it uses.
//!
//! # Coordinates: ImageJ puts a point on a pixel's centre
//!
//! A [`RoiPoint`] is in the coordinates the viewer draws in, where pixel
//! `(i, j)` covers `[i, i+1) x [j, j+1)` — a click in the middle of pixel
//! `(10, 20)` is `(10.5, 20.5)`. ImageJ does not count that way for points and
//! lines. They follow its "line sub-pixel convention" (`Roi.useLineSubpixelConvention`):
//! the integer coordinate *is* the pixel centre, so the Point Tool stores that
//! same click as `(10.0, 20.0)` (`Roi.offScreenXD` subtracts the half pixel,
//! and `Roi.screenXD` adds it back to draw). Writing `10.5` would put the mark
//! on the corner between four pixels, half a pixel down and to the right of
//! where it was placed. So the encoder subtracts `0.5` from both coordinates,
//! and a point clicked mid-pixel reads back in ImageJ on that pixel's centre.
//!
//! # One ROI per group
//!
//! ImageJ's own multi-point tool keeps every group ("counter") in one ROI and
//! tells them apart by a counter index per point. It then colours them from a
//! fixed table of its own and has nowhere to keep their names. One ROI per
//! group instead keeps both — the ROI's name is the group's name and its stroke
//! colour is the group's colour — at the cost of ImageJ seeing several
//! selections rather than one counted one. Every point's counter index is
//! therefore 0, and the counter word carries only its slice. Reading, a ROI
//! that does use several counters is split into one [`PointRoi`] per counter,
//! which is the same shape.

use std::collections::{BTreeMap, HashMap, HashSet};

/// `"Iout"`: the first four bytes of every ImageJ `.roi`.
const MAGIC: &[u8; 4] = b"Iout";
/// The format version written. The decoder reads the per-point counters (and
/// with them each point's slice) from 227 on, and the point type and size from
/// 226; ImageJ itself now writes 229, whose one addition — a 16-bit group
/// number — a point ROI with no group does not use.
const VERSION: u16 = 228;
/// ROI type 10: a point, or a multi-point selection.
const TYPE_POINT: u8 = 10;

/// The fixed header, and the second one after the coordinates.
const HEADER_SIZE: usize = 64;
const HEADER2_SIZE: usize = 64;

// Header offsets (`RoiDecoder`'s constants of the same names).
const VERSION_OFFSET: usize = 4;
const TYPE: usize = 6;
const TOP: usize = 8;
const LEFT: usize = 10;
const BOTTOM: usize = 12;
const RIGHT: usize = 14;
const N_COORDINATES: usize = 16;
/// Where the point count goes, as an i32, when it is too big for the u16 above.
const SIZE: usize = 18;
/// For a point ROI this is the point *size* index, not a stroke width.
const STROKE_WIDTH: usize = 34;
const STROKE_COLOR: usize = 40;
const OPTIONS: usize = 50;
const POINT_TYPE: usize = 52;
const POSITION: usize = 56;
const HEADER2_OFFSET: usize = 60;
const COORDINATES: usize = 64;

// Header2 offsets, relative to its start.
/// The hyperstack position: channel, slice and frame, each from 1, 0 for all.
const C_POSITION: usize = 4;
const Z_POSITION: usize = 8;
const T_POSITION: usize = 12;
const NAME_OFFSET: usize = 16;
const NAME_LENGTH: usize = 20;
const COUNTERS_OFFSET: usize = 48;

// Option bits.
/// The float coordinates are present and authoritative.
const SUB_PIXEL_RESOLUTION: u16 = 128;
/// Draw each point's number beside it.
const SHOW_LABELS: u16 = 1024;

/// Point size index 1 of ImageJ's Tiny, Small, Medium, … — Small, 3 px.
const POINT_SIZE_SMALL: i16 = 1;
/// Point type 1, a cross (`PointRoi.CROSS`).
const POINT_TYPE_CROSS: u8 = 1;
/// The ROI-wide position meaning "each point says its own"
/// (`PointRoi.POINTWISE_POSITION`).
const POINTWISE_POSITION: i32 = -2;
/// The largest slice a counter word can carry: it is the upper 24 bits of an
/// i32 that the decoder shifts down arithmetically.
const MAX_COUNTER_POSITION: u32 = 0x7F_FFFF;

/// One point of a [`PointRoi`].
///
/// `x`/`y` are image coordinates in which pixel `(i, j)` covers
/// `[i, i+1) x [j, j+1)`, so a click in the middle of pixel `(10, 20)` is
/// `(10.5, 20.5)`. (The encoder converts to ImageJ's pixel-centre convention;
/// see the module docs.) `position` is the 1-based ImageJ stack position the
/// point belongs to — the flat IFD index plus one — or 0 for "every slice".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoiPoint {
    pub x: f32,
    pub y: f32,
    pub position: u32,
}

/// A multi-point ROI as ImageJ writes one: a name, a colour, the points, and
/// where in a hyperstack it is.
#[derive(Clone, Debug, PartialEq)]
pub struct PointRoi {
    pub name: String,
    /// Stroke colour, RGB. Written opaque.
    pub color: [u8; 3],
    pub points: Vec<RoiPoint>,
    /// The ROI's hyperstack position — channel, slice, frame, each from 1, and
    /// 0 for "all" — as ImageJ keeps it in the second header, beside the
    /// points' own positions; `[0, 0, 0]` for none.
    ///
    /// ImageJ needs it on a hyperstack. There it reads a ROI's one shared
    /// position as a slice (or frame) number rather than as the flat stack
    /// index [`RoiPoint::position`] is, so a ROI whose points are all on one
    /// plane other than the first, and has no hyperstack position, is drawn
    /// on no plane at all (ImageJ 1.54p `ImageCanvas.drawOverlay`).
    pub hyperstack_position: [u32; 3],
}

impl PointRoi {
    /// ImageJ's `.roi` bytes for this ROI, as `RoiEncoder` would write them.
    ///
    /// A cross of size Small in [`color`](Self::color), with its points
    /// numbered, sub-pixel, and one counter word per point giving its slice.
    /// The ROI-wide position is the points' common one when they all share it
    /// (0 included), else ImageJ's "pointwise" -2. The length is always
    /// `128 + 16·n + 2·(name length in UTF-16 units)`.
    ///
    /// **Must not be called with no points.** ImageJ has no empty point
    /// selection: the bytes would be a well-formed header that its decoder
    /// turns into nothing. Debug builds assert; a release build returns that
    /// header. [`roi_set_zip`] and
    /// [`StackMetaWrite::overlay`](super::StackMetaWrite::overlay) skip such
    /// ROIs for you.
    pub fn to_imagej_bytes(&self) -> Vec<u8> {
        debug_assert!(
            !self.points.is_empty(),
            "an ImageJ point ROI needs at least one point"
        );
        let n = self.points.len();
        let name: Vec<u16> = self.name.encode_utf16().collect();

        let header2 = HEADER_SIZE + 4 * n + 8 * n; // i16 x, i16 y, f32 x, f32 y
        let name_at = header2 + HEADER2_SIZE;
        let counters_at = name_at + 2 * name.len();
        let mut out = vec![0u8; counters_at + 4 * n];

        // ImageJ's coordinates: the pixel centre is the integer (module docs).
        let xs: Vec<f32> = self.points.iter().map(|p| p.x - 0.5).collect();
        let ys: Vec<f32> = self.points.iter().map(|p| p.y - 0.5).collect();

        // The integer bounds `Roi.setIntBounds` gives a point ROI built from
        // floats: the extreme points' pixels, both included. In the viewer's
        // coordinates that is `floor(min)` to `floor(max) + 1`.
        let (min_x, max_x) = min_max(&xs);
        let (min_y, max_y) = min_max(&ys);
        let left = to_i16((min_x + 0.5).floor());
        let top = to_i16((min_y + 0.5).floor());
        let right = to_i16((max_x + 1.5).floor());
        let bottom = to_i16((max_y + 1.5).floor());

        out[..4].copy_from_slice(MAGIC);
        put_u16(&mut out, VERSION_OFFSET, VERSION);
        out[TYPE] = TYPE_POINT;
        put_i16(&mut out, TOP, top);
        put_i16(&mut out, LEFT, left);
        put_i16(&mut out, BOTTOM, bottom);
        put_i16(&mut out, RIGHT, right);
        // The count is a u16; a bigger one is written as 0 there and as an i32
        // in the slot after it (`RoiEncoder`, `type==point && n>65535`).
        match u16::try_from(n) {
            Ok(n) => put_u16(&mut out, N_COORDINATES, n),
            Err(_) => put_i32(&mut out, SIZE, i32::try_from(n).unwrap_or(i32::MAX)),
        }
        put_i16(&mut out, STROKE_WIDTH, POINT_SIZE_SMALL);
        let [r, g, b] = self.color;
        put_u32(&mut out, STROKE_COLOR, u32::from_be_bytes([0xFF, r, g, b]));
        put_u16(&mut out, OPTIONS, SUB_PIXEL_RESOLUTION | SHOW_LABELS);
        out[POINT_TYPE] = POINT_TYPE_CROSS;
        put_i32(&mut out, POSITION, self.common_position());
        put_i32(&mut out, HEADER2_OFFSET, header2 as i32);

        // Integer coordinates, relative to the bounds' corner, then the float
        // ones, absolute. The decoder only reads the integers when the
        // sub-pixel flag is missing, which it never is here; they are written
        // as the nearest pixel so that a reader which does use them is right.
        let x_rel = COORDINATES;
        let y_rel = x_rel + 2 * n;
        let x_abs = y_rel + 2 * n;
        let y_abs = x_abs + 4 * n;
        for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
            put_i16(&mut out, x_rel + 2 * i, relative(x, left));
            put_i16(&mut out, y_rel + 2 * i, relative(y, top));
            put_u32(&mut out, x_abs + 4 * i, x.to_bits());
            put_u32(&mut out, y_abs + 4 * i, y.to_bits());
        }

        // Header2: the hyperstack position, 0 on every axis for none, then the
        // name. No name means no name offset, as ImageJ writes it.
        for (at, v) in [C_POSITION, Z_POSITION, T_POSITION]
            .into_iter()
            .zip(self.hyperstack_position)
        {
            put_i32(&mut out, header2 + at, i32::try_from(v).unwrap_or(i32::MAX));
        }
        if !name.is_empty() {
            put_i32(&mut out, header2 + NAME_OFFSET, name_at as i32);
            put_i32(&mut out, header2 + NAME_LENGTH, name.len() as i32);
        }
        put_i32(&mut out, header2 + COUNTERS_OFFSET, counters_at as i32);
        for (i, unit) in name.iter().enumerate() {
            put_u16(&mut out, name_at + 2 * i, *unit);
        }
        // Counter word: counter index in the low byte (always 0 — one group per
        // ROI), the point's slice above it (`PointRoi.getCounters`).
        for (i, p) in self.points.iter().enumerate() {
            let word = p.position.min(MAX_COUNTER_POSITION) << 8;
            put_u32(&mut out, counters_at + 4 * i, word);
        }
        out
    }

    /// The position every point shares, or -2 when they differ.
    fn common_position(&self) -> i32 {
        let first = self.points.first().map_or(0, |p| p.position);
        if self.points.iter().all(|p| p.position == first) {
            i32::try_from(first).unwrap_or(i32::MAX)
        } else {
            POINTWISE_POSITION
        }
    }
}

/// A RoiSet `.zip` that ImageJ's ROI Manager opens — and that File > Open does
/// too, since it hands any zip whose first entry ends in `.roi` to the ROI
/// Manager.
///
/// One `<name>.roi` entry per ROI that has points (an empty one is skipped:
/// see [`PointRoi::to_imagej_bytes`]). A name used twice is made unique the
/// way `RoiManager.getUniqueName` does it, appending `-1`, `-2`, … to the
/// later one; the ROI's own name inside the entry is left as it was. Slashes in
/// a name become `_` in the entry name only, so that unpacking the zip cannot
/// create directories.
///
/// The entries are STORED (method 0), with the CRC-32 and both sizes in the
/// local header and no data descriptor, then the central directory and its end
/// record — the shape Java's `ZipInputStream` reads without help, which a
/// stored entry with a data descriptor is not. Timestamps are fixed at
/// 1980-01-01 00:00, the earliest a zip can say, so that the same ROIs give the
/// same bytes.
///
/// Not Zip64: more than 65,535 ROIs, or more than 4 GiB of them, does not fit,
/// and is [`RoiSetTooLarge`] rather than a broken archive. A file's overlay can
/// hold that many, one ROI per point.
pub fn roi_set_zip(rois: &[PointRoi]) -> Result<Vec<u8>, RoiSetTooLarge> {
    let kept: Vec<&PointRoi> = rois.iter().filter(|r| !r.points.is_empty()).collect();
    if kept.len() > usize::from(u16::MAX) {
        return Err(RoiSetTooLarge);
    }
    let names = unique_entry_names(kept.iter().map(|r| entry_stem(&r.name)).collect());
    let entries: Vec<(String, Vec<u8>)> = kept
        .iter()
        .zip(names)
        .map(|(roi, mut entry)| {
            entry.push_str(".roi");
            (entry, roi.to_imagej_bytes())
        })
        .collect();
    // The archive's whole size: per entry its local header and the central
    // record, each with the name, and the bytes once; then the end record.
    let size: u64 = entries
        .iter()
        .map(|(name, bytes)| (30 + 46 + 2 * name.len() + bytes.len()) as u64)
        .sum::<u64>()
        + 22;
    if size > u64::from(u32::MAX) {
        return Err(RoiSetTooLarge);
    }

    let mut zip = ZipWriter::default();
    for (entry, bytes) in &entries {
        zip.stored(entry, bytes);
    }
    Ok(zip.finish())
}

/// What [`roi_set_zip`] says when the ROIs do not fit a zip without Zip64:
/// more than 65,535 of them, or more than 4 GiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoiSetTooLarge;

impl std::fmt::Display for RoiSetTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "too many ROIs for an ImageJ ROI set: it holds at most 65,535, and 4 GiB in all",
        )
    }
}

impl std::error::Error for RoiSetTooLarge {}

/// The part of an entry name a ROI's name supplies: path separators replaced,
/// so the archive cannot describe a directory, a `.roi` it ends in left off —
/// every entry gets one, after the names are made unique, so that "A.roi" and
/// "A" cannot both become `A.roi` — and capped at 240 bytes (on a character
/// boundary) so that it can still be unpacked onto a filesystem with a
/// 255-byte name limit once `-N.roi` is added.
fn entry_stem(name: &str) -> String {
    const MAX: usize = 240;
    let name = name.strip_suffix(".roi").unwrap_or(name);
    let mut stem: String = name
        .chars()
        .map(|c| if c == '/' || c == '\\' { '_' } else { c })
        .collect();
    if stem.len() > MAX {
        let mut end = MAX;
        while !stem.is_char_boundary(end) {
            end -= 1;
        }
        stem.truncate(end);
    }
    stem
}

/// Make the names unique, in order, as ImageJ's ROI Manager does when it saves.
///
/// `RoiManager.saveMultiple` writes each label as it comes and only renames
/// one when the zip refuses it as a duplicate of an entry already written. The
/// rename (`getUniqueName`) then tries `name-1`, `name-2`, … against *every*
/// other label in the list, later ones included, and drops a short `-suffix`
/// (up to three characters after the last dash) before appending the next
/// number — which is what makes the third `A` become `A-2` rather than
/// `A-1-2`, and also what turns a second `WT-1` into `WT-2`.
///
/// In time linear in the names, where trying each candidate against the whole
/// list would be quadratic — cubic when one name repeats, which an overlay of
/// one ROI per point with a name in common is. Each name's count says whether
/// a candidate clashes; each stem remembers the number its last rename
/// reached, and the next starts from there, since every number below it
/// clashed. (That differs from ImageJ only when a later entry gives up a name
/// an earlier rename skipped, which leaves the names unique all the same.)
fn unique_entry_names(mut names: Vec<String>) -> Vec<String> {
    // How many entries hold each name, renames included.
    let mut held: HashMap<String, usize> = HashMap::new();
    for name in &names {
        *held.entry(name.clone()).or_default() += 1;
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut next: HashMap<String, usize> = HashMap::new();
    for entry in &mut names {
        if seen.insert(entry.clone()) {
            continue;
        }
        let original = std::mem::take(entry);
        // Another entry than this one holds `name`.
        let clashes = |held: &HashMap<String, usize>, name: &str| {
            held.get(name).copied().unwrap_or(0) > usize::from(name == original)
        };
        // Java's `name.length() - lastDash < 5`, in UTF-16 units.
        let stem = |name: &str| match name.rfind('-') {
            Some(dash) if name[dash..].encode_utf16().count() < 5 => name[..dash].to_string(),
            _ => name.to_string(),
        };
        let base = stem(&original);
        let mut n = next.get(&base).copied().unwrap_or(1);
        let mut name = format!("{base}-{n}");
        while clashes(&held, &name) {
            n += 1;
            name = format!("{base}-{n}");
        }
        next.insert(base, n + 1);
        if let Some(count) = held.get_mut(&original) {
            *count -= 1;
        }
        *held.entry(name.clone()).or_default() += 1;
        seen.insert(name.clone());
        *entry = name;
    }
    names
}

/// Smallest and largest value, in `f64` as ImageJ computes the bounds; `(0, 0)`
/// for none.
fn min_max(values: &[f32]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v as f64), hi.max(v as f64))
        })
}

/// A bound, clamped into the i16 the header holds. (NaN, which no click
/// produces, becomes 0.)
fn to_i16(v: f64) -> i16 {
    v.clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

/// A point's nearest pixel, relative to the bounds' corner `base`. The decoder
/// clamps a negative one to 0, so it is never written negative.
fn relative(v: f32, base: i16) -> i16 {
    let pixel = (v as f64 + 0.5).floor();
    to_i16((pixel - base as f64).max(0.0))
}

fn put_u16(out: &mut [u8], at: usize, v: u16) {
    out[at..at + 2].copy_from_slice(&v.to_be_bytes());
}

fn put_i16(out: &mut [u8], at: usize, v: i16) {
    out[at..at + 2].copy_from_slice(&v.to_be_bytes());
}

fn put_u32(out: &mut [u8], at: usize, v: u32) {
    out[at..at + 4].copy_from_slice(&v.to_be_bytes());
}

fn put_i32(out: &mut [u8], at: usize, v: i32) {
    out[at..at + 4].copy_from_slice(&v.to_be_bytes());
}

// ---- reading ----

/// The colour ImageJ draws a selection in when it has none of its own
/// (`Roi.ROIColor`): yellow, unless a user changed it in their preferences,
/// which a file cannot know about.
const DEFAULT_COLOR: [u8; 3] = [255, 255, 0];

/// The colours ImageJ draws counters 0 to 9 of a multi-point selection in when
/// it has more than one (`PointRoi.getColor`): Java's `yellow`, `magenta`,
/// `cyan`, `orange`, `green`, `blue`, `white`, `darkGray`, `pink` and
/// `lightGray`. Past 9 ImageJ makes up a random colour each session; reading,
/// the table repeats instead, so that one file always gives the same colours.
const COUNTER_COLORS: [[u8; 3]; 10] = [
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 200, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 255],
    [64, 64, 64],
    [255, 175, 175],
    [192, 192, 192],
];

/// `PointRoi.MAX_COUNTERS`. A counter index from here up is not a counter to
/// ImageJ, which counts such a point under counter 0 (`updateCounts`).
const MAX_COUNTERS: u8 = 100;

/// The first version whose options can say the float coordinates are there,
/// whose stroke colour is read, and whose second header has the counters.
const VERSION_SUB_PIXEL: i32 = 222;
const VERSION_STROKE_COLOR: i32 = 218;
/// The second header's C/Z/T, read from the same version as the stroke colour.
const VERSION_HYPERSTACK: i32 = 218;
const VERSION_COUNTERS: i32 = 227;

/// The point ROIs in an ImageJ `.roi`: the inverse of
/// [`PointRoi::to_imagej_bytes`], and a reader of what ImageJ's own
/// `RoiEncoder` writes, following `RoiDecoder.getRoi` for a point.
///
/// Usually that is one ROI. ImageJ's multi-point tool can keep several
/// counters in one selection, telling them apart by a counter index per
/// point; such a ROI comes out as one `PointRoi` per counter, in increasing
/// counter order, which is the shape this module writes. With a single counter
/// the ROI keeps its name and stroke colour. With several, each is named
/// `"<name> <k>"` — or `"Counter <k>"` when the ROI has no name — after
/// ImageJ's own numbering of counters from 0; counter 0 keeps the stroke colour
/// and counter `k > 0` takes the colour ImageJ draws it in. A ROI with no
/// stroke colour is ImageJ's default yellow.
///
/// Coordinates come back in this module's convention ([`RoiPoint`]): the float
/// ones when the sub-pixel option is set, else the integer ones from the
/// bounds' corner, with half a pixel added to both — the inverse of the
/// encoder's. A point's position is the slice in its counter word when there
/// are counters and that slice is not 0; else the ROI-wide position when it is
/// a slice (`> 0`: ImageJ's "pointwise" -2 is not one); else 0.
///
/// The hyperstack position of the second header (its C/Z/T) is read into
/// [`PointRoi::hyperstack_position`] as it is, and written back unchanged, so
/// a ROI that ImageJ tied to one plane of a hyperstack stays on it. It is not
/// folded into the points' positions: 0 on an axis means every index on it,
/// and which axes count as one viewer's frames is up to that viewer. A ROI
/// tied to it and to no slice therefore has points at position 0.
///
/// Empty for anything else: another shape, bytes that are not a `.roi`, and a
/// `.roi` that is cut short or points outside itself. Any part it declares —
/// the coordinates, the second header, the name, the counters — that does not
/// lie inside `bytes` drops the whole ROI, where ImageJ would fall back to a
/// file name or ignore the counters: a truncated entry read in part would put
/// its points on the wrong slices without a word. So does a coordinate that is
/// not a finite number. Every length and offset is bounded by `bytes`, so it
/// never panics and never allocates more than `bytes` could describe.
pub fn point_rois_from_imagej(bytes: &[u8]) -> Vec<PointRoi> {
    decode_point_roi(bytes).unwrap_or_default()
}

/// [`point_rois_from_imagej`], with `None` for "not a point ROI we can read".
fn decode_point_roi(d: &[u8]) -> Option<Vec<PointRoi>> {
    // `RoiDecoder` checks only the first two bytes of the magic.
    if d.len() < HEADER_SIZE || d[..2] != MAGIC[..2] || d[TYPE] != TYPE_POINT {
        return None;
    }
    let version = get_short(d, VERSION_OFFSET)?;
    let top = get_short(d, TOP)?;
    let left = get_short(d, LEFT)?;
    // The u16 count, or when that is 0, the i32 after it (over 65,535 points).
    let n = match get_ushort(d, N_COORDINATES)? {
        0 => usize::try_from(get_int(d, SIZE)?).ok()?,
        n => n as usize,
    };
    if n == 0 {
        return None;
    }
    let options = get_short(d, OPTIONS)?;
    let sub_pixel = options & i32::from(SUB_PIXEL_RESOLUTION) != 0 && version >= VERSION_SUB_PIXEL;

    // The coordinates: 4n bytes of integers, then 8n of floats if they are
    // there. Checked once, so that the offsets below cannot overflow.
    let floats = if sub_pixel { n.checked_mul(8)? } else { 0 };
    let end = COORDINATES
        .checked_add(n.checked_mul(4)?)?
        .checked_add(floats)?;
    if end > d.len() {
        return None;
    }
    let mut coordinates = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = if sub_pixel {
            let x = get_float(d, COORDINATES + 4 * n + 4 * i)?;
            let y = get_float(d, COORDINATES + 8 * n + 4 * i)?;
            (x, y)
        } else {
            // Relative to the bounds' corner; a negative one is 0, as there.
            let x = left + get_short(d, COORDINATES + 2 * i)?.max(0);
            let y = top + get_short(d, COORDINATES + 2 * n + 2 * i)?.max(0);
            (x as f32, y as f32)
        };
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        // From ImageJ's pixel centres (module docs).
        coordinates.push((x + 0.5, y + 0.5));
    }

    // The ROI-wide position: a slice, or not (0, and -2 for "pointwise").
    let roi_position = u32::try_from(get_int(d, POSITION)?).unwrap_or(0);
    let header2 = match get_int(d, HEADER2_OFFSET)? {
        0 => None,
        at => {
            let at = usize::try_from(at).ok()?;
            if at.checked_add(HEADER2_SIZE)? > d.len() {
                return None;
            }
            Some(at)
        }
    };
    let name = match header2 {
        Some(h) => read_name(d, h)?,
        None => String::new(),
    };
    // Inside the second header, which is inside `d`; a negative one is none.
    let hyperstack_position = match header2 {
        Some(h) if version >= VERSION_HYPERSTACK => {
            [C_POSITION, Z_POSITION, T_POSITION].map(|at| {
                get_int(d, h + at)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0)
            })
        }
        _ => [0; 3],
    };
    let counters = match header2 {
        Some(h) if version >= VERSION_COUNTERS => read_counters(d, h, n)?,
        _ => None,
    };
    let stroke = if version >= VERSION_STROKE_COLOR {
        get_int(d, STROKE_COLOR)? as u32
    } else {
        0
    };
    let color = match stroke {
        0 => DEFAULT_COLOR,
        argb => {
            let [_, r, g, b] = argb.to_be_bytes();
            [r, g, b]
        }
    };

    // Each point under its counter, in order.
    let mut by_counter: BTreeMap<u8, Vec<RoiPoint>> = BTreeMap::new();
    for (i, (x, y)) in coordinates.into_iter().enumerate() {
        let word = counters.as_ref().map_or(0, |c| c[i]);
        let counter = match (word & 0xff) as u8 {
            c if c < MAX_COUNTERS => c,
            _ => 0,
        };
        // An arithmetic shift, as `PointRoi.setCounters` does; a word with its
        // top bit set gives a negative slice, which is none.
        let position = u32::try_from(word >> 8)
            .ok()
            .filter(|&p| p > 0)
            .unwrap_or(roi_position);
        by_counter
            .entry(counter)
            .or_default()
            .push(RoiPoint { x, y, position });
    }
    if by_counter.len() == 1 {
        let points = by_counter.into_values().next()?;
        return Some(vec![PointRoi {
            name,
            color,
            points,
            hyperstack_position,
        }]);
    }
    let rois = by_counter
        .into_iter()
        .map(|(k, points)| PointRoi {
            name: if name.is_empty() {
                format!("Counter {k}")
            } else {
                format!("{name} {k}")
            },
            color: if k == 0 {
                color
            } else {
                COUNTER_COLORS[k as usize % COUNTER_COLORS.len()]
            },
            points,
            hyperstack_position,
        })
        .collect();
    Some(rois)
}

/// The ROI's name, from its second header at `h`: empty when it has none,
/// `None` when the one it declares is not inside `d`. UTF-16BE, as Java's
/// `char`s; an unpaired surrogate becomes U+FFFD.
fn read_name(d: &[u8], h: usize) -> Option<String> {
    let at = get_int(d, h + NAME_OFFSET)?;
    let len = get_int(d, h + NAME_LENGTH)?;
    if at == 0 || len == 0 {
        return Some(String::new());
    }
    let at = usize::try_from(at).ok()?;
    let len = usize::try_from(len).ok()?;
    let bytes = d.get(at..at.checked_add(len.checked_mul(2)?)?)?;
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    Some(String::from_utf16_lossy(&units))
}

/// The `n` counter words, from the second header at `h`: `Some(None)` when
/// the ROI has none, `None` when the ones it declares are not inside `d`.
fn read_counters(d: &[u8], h: usize, n: usize) -> Option<Option<Vec<i32>>> {
    let at = get_int(d, h + COUNTERS_OFFSET)?;
    if at == 0 {
        return Some(None);
    }
    let at = usize::try_from(at).ok()?;
    let bytes = d.get(at..at.checked_add(n.checked_mul(4)?)?)?;
    let words = bytes
        .chunks_exact(4)
        .map(|w| i32::from_be_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    Some(Some(words))
}

/// `N` bytes at `at`, or `None` past the end.
fn take<const N: usize>(d: &[u8], at: usize) -> Option<[u8; N]> {
    d.get(at..at.checked_add(N)?)?.try_into().ok()
}

/// `RoiDecoder.getShort`: signed, except that a value below -5000 is taken to
/// be an unsigned one past 32,767 — its guess for a bound that outgrew an i16.
fn get_short(d: &[u8], at: usize) -> Option<i32> {
    let b = take::<2>(d, at)?;
    let v = i32::from(i16::from_be_bytes(b));
    Some(if v < -5000 {
        i32::from(u16::from_be_bytes(b))
    } else {
        v
    })
}

fn get_ushort(d: &[u8], at: usize) -> Option<u16> {
    take(d, at).map(u16::from_be_bytes)
}

fn get_int(d: &[u8], at: usize) -> Option<i32> {
    take(d, at).map(i32::from_be_bytes)
}

fn get_float(d: &[u8], at: usize) -> Option<f32> {
    take(d, at).map(f32::from_be_bytes)
}

// ---- the zip container ----

/// Local file header, central directory header and end-of-central-directory
/// signatures.
const LOCAL_HEADER: u32 = 0x0403_4b50;
const CENTRAL_HEADER: u32 = 0x0201_4b50;
const END_OF_CENTRAL_DIRECTORY: u32 = 0x0605_4b50;
/// Version needed to extract a stored entry: 1.0.
const VERSION_NEEDED: u16 = 10;
/// Version made by: 2.0, MS-DOS attributes (upper byte 0).
const VERSION_MADE_BY: u16 = 20;
/// General purpose bit 11: the entry name is UTF-8.
const FLAG_UTF8: u16 = 1 << 11;
/// MS-DOS date for 1980-01-01 (year 0 since 1980, month 1, day 1); the time is 0.
const DOS_DATE: u16 = (1 << 5) | 1;

/// One entry's central directory record, kept until the end.
struct CentralEntry {
    name: Vec<u8>,
    flags: u16,
    crc: u32,
    size: u32,
    offset: u32,
}

/// The few lines of PKZIP a stored-only archive needs.
#[derive(Default)]
struct ZipWriter {
    out: Vec<u8>,
    entries: Vec<CentralEntry>,
}

impl ZipWriter {
    /// Append one STORED entry: local header, then the bytes verbatim.
    fn stored(&mut self, name: &str, data: &[u8]) {
        let mut crc = flate2::Crc::new();
        crc.update(data);
        let entry = CentralEntry {
            name: name.as_bytes().to_vec(),
            flags: if name.is_ascii() { 0 } else { FLAG_UTF8 },
            crc: crc.sum(),
            size: u32::try_from(data.len()).expect("a ROI set entry over 4 GiB needs Zip64"),
            offset: u32::try_from(self.out.len()).expect("a ROI set over 4 GiB needs Zip64"),
        };
        let name_len =
            u16::try_from(entry.name.len()).expect("entry names are capped well below 64 KiB");
        let o = &mut self.out;
        o.extend_from_slice(&LOCAL_HEADER.to_le_bytes());
        o.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
        o.extend_from_slice(&entry.flags.to_le_bytes());
        o.extend_from_slice(&0u16.to_le_bytes()); // method 0: stored
        o.extend_from_slice(&0u16.to_le_bytes()); // time 00:00:00
        o.extend_from_slice(&DOS_DATE.to_le_bytes());
        o.extend_from_slice(&entry.crc.to_le_bytes());
        o.extend_from_slice(&entry.size.to_le_bytes()); // compressed size
        o.extend_from_slice(&entry.size.to_le_bytes()); // uncompressed size
        o.extend_from_slice(&name_len.to_le_bytes());
        o.extend_from_slice(&0u16.to_le_bytes()); // no extra field
        o.extend_from_slice(&entry.name);
        o.extend_from_slice(data);
        self.entries.push(entry);
    }

    /// The central directory and its end record; the archive is complete.
    fn finish(mut self) -> Vec<u8> {
        let count = u16::try_from(self.entries.len()).expect("more than 65,535 ROIs needs Zip64");
        let start = u32::try_from(self.out.len()).expect("a ROI set over 4 GiB needs Zip64");
        let o = &mut self.out;
        for e in &self.entries {
            o.extend_from_slice(&CENTRAL_HEADER.to_le_bytes());
            o.extend_from_slice(&VERSION_MADE_BY.to_le_bytes());
            o.extend_from_slice(&VERSION_NEEDED.to_le_bytes());
            o.extend_from_slice(&e.flags.to_le_bytes());
            o.extend_from_slice(&0u16.to_le_bytes()); // method 0: stored
            o.extend_from_slice(&0u16.to_le_bytes()); // time
            o.extend_from_slice(&DOS_DATE.to_le_bytes());
            o.extend_from_slice(&e.crc.to_le_bytes());
            o.extend_from_slice(&e.size.to_le_bytes());
            o.extend_from_slice(&e.size.to_le_bytes());
            o.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
            o.extend_from_slice(&0u16.to_le_bytes()); // extra field length
            o.extend_from_slice(&0u16.to_le_bytes()); // comment length
            o.extend_from_slice(&0u16.to_le_bytes()); // disk number start
            o.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
            o.extend_from_slice(&0u32.to_le_bytes()); // external attributes
            o.extend_from_slice(&e.offset.to_le_bytes());
            o.extend_from_slice(&e.name);
        }
        let size = u32::try_from(o.len()).expect("a ROI set over 4 GiB needs Zip64") - start;
        o.extend_from_slice(&END_OF_CENTRAL_DIRECTORY.to_le_bytes());
        o.extend_from_slice(&0u16.to_le_bytes()); // this disk
        o.extend_from_slice(&0u16.to_le_bytes()); // disk with the directory
        o.extend_from_slice(&count.to_le_bytes()); // entries on this disk
        o.extend_from_slice(&count.to_le_bytes()); // entries in all
        o.extend_from_slice(&size.to_le_bytes());
        o.extend_from_slice(&start.to_le_bytes());
        o.extend_from_slice(&0u16.to_le_bytes()); // no comment
        self.out
    }
}

#[cfg(test)]
#[path = "roi_tests.rs"]
mod tests;

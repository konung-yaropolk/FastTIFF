//! Walking a DICOM dataset: elements in, the handful we need out.
//!
//! DICOM is a stream of tagged elements, and the encoding of each one depends
//! on a transfer syntax named in a *different* part of the file. So the walk
//! happens twice: the File Meta group first, always in Explicit VR Little
//! Endian whatever the rest is, to find out how to read the rest.
//!
//! # What this reads, and what it steps over
//!
//! Everything this importer needs is a top-level element, so nothing here
//! descends into a sequence. Sequences are **skipped**, and skipping them
//! correctly is the one part of a DICOM reader that cannot be left out: an
//! undefined-length sequence has no length to advance by, and a reader that
//! treats `0xFFFFFFFF` as a byte count or tries to resynchronise by searching
//! for the next plausible tag will read the rest of the file out of step. The
//! symptom is not an error — it is a `Rows` value picked out of the middle of
//! somebody's private data, and then an image.
//!
//! So [`Walk::skip_undefined`] tracks nesting properly. It is more code than
//! the rest of the parser put together and it is load-bearing.
//!
//! # Deliberately not a DICOM library
//!
//! No dictionary, no value-representation conversion, no character sets, no
//! writing. A dataset is walked once and a dozen known tags are copied out as
//! raw bytes, which the caller interprets because it knows what it asked for.
//! That is enough to open a scan and small enough to be read in full by whoever
//! has to trust it — the trade the OIR reader next door makes for the same
//! reason.

use fasttiff_plugin_api::PluginError;
use std::collections::HashMap;

/// A DICOM tag: group and element.
pub(super) type Tag = (u16, u16);

/// What the pixels are encoded with, which is a separate question from how the
/// elements are.
///
/// A transfer syntax UID names both at once, and conflating them is the mistake
/// to avoid: RLE and every JPEG variant put their *elements* in plain Explicit
/// VR Little Endian and only their Pixel Data in something else. A reader that
/// treated "compressed" as a property of the dataset would be unable to read
/// the geometry of a file it could in fact decode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Codec {
    /// Pixel Data is one contiguous value, as stored.
    Native,
    /// `1.2.840.10008.1.2.5` — PS3.5 Annex G: segmented byte planes, each
    /// PackBits-compressed.
    Rle,
    /// `1.2.840.10008.1.2.4.50` — baseline sequential DCT, 8-bit.
    JpegBaseline,
    /// `1.2.840.10008.1.2.4.57` / `.70` — lossless predictive, process 14.
    /// Nothing like baseline: no DCT at all.
    JpegLossless,
}

/// How a dataset's elements are laid out on the wire.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Syntax {
    /// `1.2.840.10008.1.2` — no VR on the wire, every length 4 bytes.
    ImplicitLe,
    /// `1.2.840.10008.1.2.1` — the usual one.
    ExplicitLe,
    /// `1.2.840.10008.1.2.2` — retired, but files exist.
    ExplicitBe,
}

impl Syntax {
    fn big_endian(self) -> bool {
        self == Syntax::ExplicitBe
    }
    fn explicit(self) -> bool {
        self != Syntax::ImplicitLe
    }
}

/// Value representations whose explicit-VR form carries a 4-byte length after
/// two reserved bytes, rather than a 2-byte length.
///
/// The classic six are `OB OW OF SQ UT UN`. The rest — `OD OL OV SV UC UR UV`
/// — came in with DICOM's 64-bit and long-string types and are included
/// because a file using one is read correctly either way *only* if this list
/// has it: guessing the short form on a long-form VR reads the first two bytes
/// of a 4-byte length as the length itself, which for any value longer than
/// 65535 bytes gives a wildly wrong one and desynchronises the walk.
///
/// A VR not in this list and not a known short one is treated as short, which
/// is the right default — every VR added to the standard so far that is *not*
/// here is a short-form one.
const LONG_FORM_VRS: [&[u8; 2]; 13] = [
    b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV",
];

/// The length that means "no length": read until the matching delimiter.
const UNDEFINED: u32 = 0xFFFF_FFFF;

/// Item, Item Delimitation and Sequence Delimitation.
///
/// These three carry a 4-byte length and **no VR**, in explicit-VR datasets as
/// well as implicit ones. Reading a VR for them is the other classic way to
/// lose sync, and it is an easy mistake because every other element in an
/// explicit dataset has one.
const ITEM: Tag = (0xFFFE, 0xE000);
const ITEM_END: Tag = (0xFFFE, 0xE00D);
const SEQ_END: Tag = (0xFFFE, 0xE0DD);

/// Pixel Data, which is the one element whose undefined length means something
/// other than a sequence.
pub(super) const PIXEL_DATA: Tag = (0x7FE0, 0x0010);

/// The raw values of the tags a caller asked for.
#[derive(Default, Debug)]
pub(super) struct Found {
    values: HashMap<Tag, Vec<u8>>,
    /// Where the encapsulated Pixel Data's item stream begins, when the
    /// element arrived with an undefined length.
    ///
    /// An offset rather than a flag, because the fragments are what a codec
    /// needs and they are found by walking from here — see [`fragments`].
    pub(super) encapsulated_at: Option<usize>,
}

impl Found {
    /// The raw bytes of a tag, if the file had it.
    pub(super) fn raw(&self, tag: Tag) -> Option<&[u8]> {
        self.values.get(&tag).map(Vec::as_slice)
    }

    /// A tag's value as text, trimmed of the padding DICOM pads with.
    ///
    /// Values are space-padded to an even length, and the UI-type ones are
    /// NUL-padded instead, so both have to come off — a `SeriesInstanceUID`
    /// with a trailing NUL compares unequal to the same UID without one, which
    /// would split a series in half.
    pub(super) fn text(&self, tag: Tag) -> Option<String> {
        let raw = self.raw(tag)?;
        let s = String::from_utf8_lossy(raw);
        let t = s.trim_matches(|c: char| c == '\0' || c.is_whitespace());
        (!t.is_empty()).then(|| t.to_string())
    }

    /// A tag's value as an unsigned short (`US`), which is how DICOM states
    /// every image dimension.
    pub(super) fn us(&self, tag: Tag, big_endian: bool) -> Option<u16> {
        let b = self.raw(tag)?.first_chunk::<2>().copied()?;
        Some(if big_endian {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    }

    /// A tag's value as a number, from whichever representation it is in.
    ///
    /// The numeric VRs that matter here are `DS` (a decimal *string*) and `IS`
    /// (an integer string) — DICOM writes `RescaleSlope` and `PixelSpacing` as
    /// text, which is why these cannot simply be read as binary. Multi-valued
    /// elements are backslash-separated and this takes the first.
    pub(super) fn num(&self, tag: Tag) -> Option<f64> {
        self.nums(tag).into_iter().next()
    }

    /// Every value of a multi-valued numeric element, in order.
    ///
    /// `PixelSpacing` is two numbers and `ImagePositionPatient` is three, both
    /// as one backslash-separated string. Splitting is not optional: the whole
    /// of `"0.9375\\0.9375"` parsed as one number is nothing at all, and a
    /// reader that took it as a single value would silently lose the spacing.
    pub(super) fn nums(&self, tag: Tag) -> Vec<f64> {
        let Some(text) = self.text(tag) else {
            return Vec::new();
        };
        text.split('\\')
            .filter_map(|p| p.trim().parse::<f64>().ok())
            .collect()
    }
}

/// A cursor over a dataset.
struct Walk<'a> {
    b: &'a [u8],
    at: usize,
    syntax: Syntax,
}

/// Reaching the end of the data in the middle of something.
fn truncated(what: &str) -> PluginError {
    PluginError::failed(format!("the DICOM data ends in the middle of {what}"))
}

impl<'a> Walk<'a> {
    fn new(b: &'a [u8], at: usize, syntax: Syntax) -> Self {
        Walk { b, at, syntax }
    }

    fn u16(&mut self) -> Result<u16, PluginError> {
        let c = self
            .b
            .get(self.at..self.at + 2)
            .and_then(|s| s.first_chunk::<2>().copied())
            .ok_or_else(|| truncated("a 16-bit field"))?;
        self.at += 2;
        Ok(if self.syntax.big_endian() {
            u16::from_be_bytes(c)
        } else {
            u16::from_le_bytes(c)
        })
    }

    fn u32(&mut self) -> Result<u32, PluginError> {
        let c = self
            .b
            .get(self.at..self.at + 4)
            .and_then(|s| s.first_chunk::<4>().copied())
            .ok_or_else(|| truncated("a 32-bit field"))?;
        self.at += 4;
        Ok(if self.syntax.big_endian() {
            u32::from_be_bytes(c)
        } else {
            u32::from_le_bytes(c)
        })
    }

    fn tag(&mut self) -> Result<Tag, PluginError> {
        Ok((self.u16()?, self.u16()?))
    }

    fn skip(&mut self, n: u32) -> Result<(), PluginError> {
        let to = self
            .at
            .checked_add(n as usize)
            .filter(|&to| to <= self.b.len())
            .ok_or_else(|| truncated("an element's value"))?;
        self.at = to;
        Ok(())
    }

    /// Read one element's header: its tag, its VR if the syntax carries one,
    /// and its length.
    ///
    /// Returns `(tag, vr, length)`. The item tags are the exception the
    /// standard makes and this makes with it: they have no VR even here.
    fn header(&mut self) -> Result<(Tag, [u8; 2], u32), PluginError> {
        let tag = self.tag()?;
        if matches!(tag, ITEM | ITEM_END | SEQ_END) {
            return Ok((tag, *b"  ", self.u32()?));
        }
        if !self.syntax.explicit() {
            return Ok((tag, *b"  ", self.u32()?));
        }
        let vr = self
            .b
            .get(self.at..self.at + 2)
            .and_then(|s| s.first_chunk::<2>().copied())
            .ok_or_else(|| truncated("a value representation"))?;
        self.at += 2;
        let len = if LONG_FORM_VRS.contains(&&vr) {
            // Two reserved bytes, then a 4-byte length.
            self.skip(2)?;
            self.u32()?
        } else {
            self.u16()? as u32
        };
        Ok((tag, vr, len))
    }

    /// Step over an undefined-length sequence, from just after its length to
    /// just after its Sequence Delimitation Item.
    ///
    /// Tracks nesting, because a sequence's items hold elements and those
    /// elements may be sequences of their own. The counter is what makes this
    /// correct on a real scanner's file, where an MR Image Storage object can
    /// easily be three levels deep before Pixel Data.
    fn skip_undefined(&mut self) -> Result<(), PluginError> {
        let mut depth = 1usize;
        while depth > 0 {
            let (tag, _vr, len) = self.header()?;
            match tag {
                SEQ_END => depth -= 1,
                // An item delimiter closes an undefined-length *item*, which
                // this does not count: an item cannot end a sequence, so the
                // depth belongs to sequences alone.
                ITEM_END => {}
                ITEM => {
                    // A defined-length item is stepped over whole. An
                    // undefined-length one is walked into — its contents are
                    // elements, and they are what the next turns of this loop
                    // read.
                    if len != UNDEFINED {
                        self.skip(len)?;
                    }
                }
                _ => {
                    if len == UNDEFINED {
                        depth += 1;
                    } else {
                        self.skip(len)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Walk `b` from `at`, copying out the values of `wanted`.
///
/// Stops at Pixel Data, whose value is handed back as a byte range rather than
/// copied — it is the whole image and there is no reason to have two of it.
/// Returns `(found, pixel data range)`.
pub(super) fn scan(
    b: &[u8],
    at: usize,
    syntax: Syntax,
    wanted: &[Tag],
) -> Result<(Found, Option<std::ops::Range<usize>>), PluginError> {
    let mut w = Walk::new(b, at, syntax);
    let mut found = Found::default();
    let mut pixels = None;

    while w.at < b.len() {
        // A dataset that ends cleanly is the normal case and not an error, so a
        // truncated *header* at the very end stops the walk rather than failing
        // it. A truncation anywhere inside an element is still an error.
        let Ok((tag, _vr, len)) = w.header() else {
            break;
        };
        if tag == PIXEL_DATA {
            if len == UNDEFINED {
                // Encapsulated: the value is a basic offset table and then one
                // or more compressed fragments per frame. Where it starts is
                // recorded and the walk stops; what the fragments *are* is
                // [`fragments`]'s job, and only the caller knows which codec
                // wants them.
                found.encapsulated_at = Some(w.at);
                break;
            }
            let end =
                w.at.checked_add(len as usize)
                    .filter(|&e| e <= b.len())
                    .ok_or_else(|| {
                        PluginError::failed(format!(
                            "the DICOM file declares {len} bytes of pixel data and holds {}",
                            b.len().saturating_sub(w.at)
                        ))
                    })?;
            pixels = Some(w.at..end);
            break;
        }
        if len == UNDEFINED {
            w.skip_undefined()?;
            continue;
        }
        if wanted.contains(&tag) {
            let from = w.at;
            w.skip(len)?;
            found.values.insert(tag, b[from..w.at].to_vec());
        } else {
            w.skip(len)?;
        }
    }
    Ok((found, pixels))
}

/// The Part 10 preamble is 128 bytes and then `DICM`.
pub(super) const MAGIC_AT: usize = 128;

/// Whether `head` looks like a Part 10 DICOM file.
///
/// The preamble's own 128 bytes are *not* checked: the standard allows them to
/// hold anything at all (they exist so a file can also be a TIFF, which some
/// vendors did), so the magic is the only evidence.
pub(super) fn has_magic(head: &[u8]) -> bool {
    head.get(MAGIC_AT..MAGIC_AT + 4) == Some(b"DICM")
}

/// The transfer syntax the dataset is in, and where the dataset starts.
///
/// The File Meta group is always Explicit VR Little Endian — that is fixed by
/// the standard and does not depend on anything in the file — so it can be
/// walked before anything is known. Its `(0002,0010)` names the syntax for
/// everything after it.
pub(super) fn meta(b: &[u8]) -> Result<(Syntax, Codec, usize), PluginError> {
    const TRANSFER_SYNTAX: Tag = (0x0002, 0x0010);

    if !has_magic(b) {
        return Err(PluginError::unsupported(
            "not a DICOM Part 10 file: the `DICM` marker is missing from \
             offset 128. A raw dataset without the preamble is not read here",
        ));
    }
    let start = MAGIC_AT + 4;
    // Where the meta group ends, which has to be settled *before* it is
    // walked. The group is Explicit VR Little Endian and the dataset after it
    // may not be, so a walk that ran past the boundary would be reading the
    // dataset in the wrong encoding — and would then either fail on a nonsense
    // length or, worse, find a plausible value for a tag it wanted. Bounding
    // the walk is what makes the two halves independent.
    let end = meta_group_end(b, start)?;

    let (found, _) = scan(&b[..end], start, Syntax::ExplicitLe, &[TRANSFER_SYNTAX])?;
    let uid = found.text(TRANSFER_SYNTAX).ok_or_else(|| {
        PluginError::failed(
            "the DICOM file's meta group does not say which transfer syntax its \
             data is in, so there is no way to read it",
        )
    })?;
    let (syntax, codec) = transfer(&uid)?;
    Ok((syntax, codec, end))
}

/// Where the File Meta group stops and the dataset begins.
///
/// `(0002,0000)` is a group length and the standard requires it to be the first
/// element of the group, so it is read at a known offset rather than searched
/// for. Its header is twelve bytes — tag, `UL`, a 2-byte length and four bytes
/// of value — and the count it holds is of everything after it.
///
/// Files that omit it exist, in defiance of the standard, so there is a
/// fallback: walk the group and stop at the first element whose group is not
/// `0002`. That is correct but weaker, because it trusts the elements to be
/// well-formed all the way to the boundary where the length would not have to.
fn meta_group_end(b: &[u8], start: usize) -> Result<usize, PluginError> {
    // Walked, not computed from `(0002,0000)`.
    //
    // That element is a group length and the standard requires it, so trusting
    // it looks like the better option — it gives the answer in one read instead
    // of a walk. It is the worse option, because it is the *only* number in the
    // file that nothing else checks: a writer that emitted a stale one leaves a
    // file every other tool reads happily, and a reader that believed it would
    // start the dataset inside an element and then desynchronise. There is no
    // symptom for that short of a wrong image.
    //
    // Walking has no such failure. The meta group is by definition the run of
    // elements whose group is 0x0002, so the first element that is not one *is*
    // the boundary, and finding it validates every length on the way. It also
    // does not care whether the group length is present, which files in the
    // wild sometimes are not.
    let mut w = Walk::new(b, start, Syntax::ExplicitLe);
    loop {
        let before = w.at;
        // A file that ends inside its own meta group has no dataset to find,
        // and the end of the data is the only honest answer — `meta` then fails
        // on the missing transfer syntax, which says more than a length error
        // here would.
        let Ok((tag, _, len)) = w.header() else {
            return Ok(b.len());
        };
        if tag.0 != 0x0002 {
            return Ok(before);
        }
        // An undefined length inside the meta group would be malformed: the
        // group holds no sequences, and nothing else uses one. A file that has
        // one is a file to stop reading rather than to resynchronise, because
        // the next thing read would be a guess.
        if len == UNDEFINED || w.skip(len).is_err() {
            return Err(PluginError::failed(
                "the DICOM file's meta group is malformed, so where its data                  begins cannot be worked out",
            ));
        }
    }
}

/// Turn a transfer syntax UID into how to read the elements and how to decode
/// the pixels, or into a refusal that names what the file actually is.
///
/// The refused syntaxes are listed individually rather than lumped together
/// because the name is the useful part of the message: "JPEG 2000" tells
/// someone what to convert *from*, where "an unsupported transfer syntax" tells
/// them to go and look it up.
///
/// Note that every compressed entry pairs its codec with **Explicit VR Little
/// Endian**. That is not a simplification — PS3.5 requires it: a compressed
/// transfer syntax compresses the pixels and nothing else.
pub(super) fn transfer(uid: &str) -> Result<(Syntax, Codec), PluginError> {
    let named = |what: &str| {
        PluginError::unsupported(format!(
            "this DICOM file's pixel data is {what} ({uid}), which is not \
             decoded here. Convert it to an uncompressed transfer syntax — \
             `dcmdjpeg`, `gdcmconv --raw`, or re-export from the scanner — or \
             convert the series to NIfTI with `dcm2niix`, which this also reads"
        ))
    };
    Ok(match uid {
        "1.2.840.10008.1.2" => (Syntax::ImplicitLe, Codec::Native),
        "1.2.840.10008.1.2.1" => (Syntax::ExplicitLe, Codec::Native),
        "1.2.840.10008.1.2.2" => (Syntax::ExplicitBe, Codec::Native),
        "1.2.840.10008.1.2.5" => (Syntax::ExplicitLe, Codec::Rle),
        "1.2.840.10008.1.2.4.50" => (Syntax::ExplicitLe, Codec::JpegBaseline),
        // Process 14 with first-order prediction, and its selection-value-1
        // restriction. The same decoder reads both: SV1 is a constraint on what
        // an encoder may emit, not a different bitstream.
        "1.2.840.10008.1.2.4.57" | "1.2.840.10008.1.2.4.70" => {
            (Syntax::ExplicitLe, Codec::JpegLossless)
        }
        // Not compressed at all, and refused anyway: the samples are native but
        // Pixel Data is an item stream with a basic offset table, one fragment
        // per frame. Reading it would mean traversing that stream, which is a
        // different code path from the single contiguous value every other
        // uncompressed syntax has. Named explicitly rather than left to fall
        // through, because "this is uncompressed and still refused" needs
        // saying — otherwise it looks like the reader did not recognise it.
        "1.2.840.10008.1.2.1.98" => {
            return Err(PluginError::unsupported(format!(
                "this DICOM file is encapsulated-uncompressed ({uid}): its \
                 pixels are not compressed, but they are split into fragments \
                 this reader does not reassemble. `gdcmconv --raw` converts it, \
                 and so does `dcm2niix`"
            )))
        }
        "1.2.840.10008.1.2.1.99" => return Err(named("deflated")),
        // Extended JPEG is 12-bit DCT with a different Huffman setup; the
        // baseline decoder cannot read it, and pretending otherwise would give
        // a picture built from misread coefficients.
        "1.2.840.10008.1.2.4.51" => return Err(named("extended (12-bit) JPEG")),
        "1.2.840.10008.1.2.4.80" | "1.2.840.10008.1.2.4.81" => return Err(named("JPEG-LS")),
        "1.2.840.10008.1.2.4.90" | "1.2.840.10008.1.2.4.91" => return Err(named("JPEG 2000")),
        "1.2.840.10008.1.2.4.92" | "1.2.840.10008.1.2.4.93" => {
            return Err(named("JPEG 2000 multi-component"))
        }
        other => {
            return Err(PluginError::unsupported(format!(
                "this DICOM file declares transfer syntax {other}, which is not \
                 one this reader knows. The uncompressed ones it reads are \
                 1.2.840.10008.1.2 (implicit VR), 1.2.840.10008.1.2.1 \
                 (explicit VR) and 1.2.840.10008.1.2.2 (explicit VR, big endian)"
            )))
        }
    })
}

#[cfg(test)]
#[path = "parse_tests.rs"]
pub(crate) mod tests;

/// The compressed fragments of an encapsulated Pixel Data, grouped by frame.
///
/// An encapsulated value is an item stream, not a single blob (PS3.5 A.4):
///
/// ```text
///   Item (FFFE,E000) len L0   the Basic Offset Table, possibly empty
///   Item (FFFE,E000) len L1   a fragment
///   Item (FFFE,E000) len L2   another fragment
///   ...
///   (FFFE,E0DD)   len 0       Sequence Delimitation
/// ```
///
/// A frame is one *or more* consecutive fragments — an encoder may split a
/// large frame to keep each item under some size — so grouping is needed and
/// the Basic Offset Table is what says how. Each of its `u32`s is the byte
/// offset of a frame's first fragment, counted from the first byte *after* the
/// table's own item.
///
/// # When the table is empty
///
/// Which it very often is, being optional. Then the only available reading is
/// one fragment per frame, and that is what is used. It is right for everything
/// a scanner writes; the case it gets wrong — an empty table *and* a split
/// frame — is unreadable without the table, so there is nothing better to do
/// than produce the fragments as found and let the frame count disagree
/// visibly.
pub(super) fn fragments(b: &[u8], at: usize) -> Result<Vec<Vec<u8>>, PluginError> {
    // The item stream is always little-endian, whatever the dataset was: a
    // compressed transfer syntax is Explicit VR Little Endian by definition.
    let mut w = Walk::new(b, at, Syntax::ExplicitLe);
    let mut table: Vec<u32> = Vec::new();
    let mut raw: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut first = true;
    let mut base = 0usize;

    loop {
        let (tag, _vr, len) = w.header()?;
        match tag {
            SEQ_END => break,
            ITEM => {
                if len == UNDEFINED {
                    return Err(PluginError::failed(
                        "this DICOM file's compressed pixel data has an item of \
                         undefined length, which the encapsulation rules do not \
                         allow",
                    ));
                }
                let from = w.at;
                w.skip(len)?;
                if first {
                    // The Basic Offset Table: `len / 4` little-endian offsets,
                    // and `len == 0` for the common empty one.
                    table = b[from..w.at]
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().expect("exact chunks")))
                        .collect();
                    base = w.at;
                    first = false;
                } else {
                    raw.push((from - base, b[from..w.at].to_vec()));
                }
            }
            other => {
                return Err(PluginError::failed(format!(
                    "this DICOM file's compressed pixel data holds a \
                     ({:04X},{:04X}) where only items are allowed",
                    other.0, other.1
                )))
            }
        }
        if w.at >= b.len() {
            break;
        }
    }
    if raw.is_empty() {
        return Err(PluginError::failed(
            "this DICOM file's pixel data is encapsulated but holds no fragments",
        ));
    }

    // One fragment per frame unless the table says otherwise.
    if table.len() < 2 {
        return Ok(raw.into_iter().map(|(_, f)| f).collect());
    }
    // Grouped by which table offset each fragment falls at or after. Walking
    // the table rather than the fragments, so a table naming an offset no
    // fragment starts at produces an empty frame and a clear size error later,
    // rather than a silent misalignment.
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(table.len());
    for (i, &start) in table.iter().enumerate() {
        let end = table.get(i + 1).copied().unwrap_or(u32::MAX);
        let mut frame = Vec::new();
        for (off, data) in &raw {
            if *off >= start as usize && (*off as u64) < end as u64 {
                frame.extend_from_slice(data);
            }
        }
        out.push(frame);
    }
    Ok(out)
}

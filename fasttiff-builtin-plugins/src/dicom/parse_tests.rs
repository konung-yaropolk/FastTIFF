//! Walking a dataset, and the ways a walk loses its place.
//!
//! Every file here is built byte by byte. DICOM has no small valid example to
//! borrow and a real one cannot be committed — it is a person's scan — so the
//! builder below is the fixture, and being able to write a file with a
//! deliberately awkward structure is most of why these tests are worth having.
//!
//! The failure these are mostly about is **desynchronisation**: a reader that
//! miscounts one element's length does not stop, it carries on reading tags out
//! of the middle of values. What comes out is not an error but a `Rows` of
//! 18245 or, worse, a plausible one — and then an image. So the tests assert
//! that specific values arrive from specific places, past structures designed
//! to throw the walk off.

use super::*;

/// Explicit VR Little Endian — the common one.
pub(crate) const EXPLICIT_LE: &str = "1.2.840.10008.1.2.1";
/// Implicit VR Little Endian — no VR on the wire.
pub(crate) const IMPLICIT_LE: &str = "1.2.840.10008.1.2";
/// Explicit VR Big Endian — retired, and files exist.
pub(crate) const EXPLICIT_BE: &str = "1.2.840.10008.1.2.2";

/// A DICOM Part 10 file under construction.
pub(crate) struct Dcm {
    syntax: &'static str,
    data: Vec<u8>,
}

impl Dcm {
    pub(crate) fn new(syntax: &'static str) -> Self {
        Dcm {
            syntax,
            data: Vec::new(),
        }
    }

    fn be(&self) -> bool {
        self.syntax == EXPLICIT_BE
    }
    fn explicit(&self) -> bool {
        self.syntax != IMPLICIT_LE
    }

    fn push_u16(&mut self, v: u16) {
        if self.be() {
            self.data.extend_from_slice(&v.to_be_bytes());
        } else {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn push_u32(&mut self, v: u32) {
        if self.be() {
            self.data.extend_from_slice(&v.to_be_bytes());
        } else {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// One element, with its value. Lengths are made even, as DICOM requires.
    pub(crate) fn el(&mut self, tag: Tag, vr: &[u8; 2], value: &[u8]) -> &mut Self {
        let mut v = value.to_vec();
        if v.len() % 2 == 1 {
            // Text pads with a space, binary with a NUL. Both are what the
            // standard says and neither changes the value once trimmed.
            v.push(if vr == b"UI" { 0 } else { b' ' });
        }
        self.push_u16(tag.0);
        self.push_u16(tag.1);
        if self.explicit() {
            self.data.extend_from_slice(vr);
            if LONG_FORM_VRS.contains(&vr) {
                self.data.extend_from_slice(&[0, 0]);
                self.push_u32(v.len() as u32);
            } else {
                self.push_u16(v.len() as u16);
            }
        } else {
            self.push_u32(v.len() as u32);
        }
        self.data.extend_from_slice(&v);
        self
    }

    /// An unsigned short element, which is how every dimension is stated.
    pub(crate) fn us(&mut self, tag: Tag, v: u16) -> &mut Self {
        let b = if self.be() {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        };
        self.el(tag, b"US", &b)
    }

    /// A text element — `DS`, `IS`, `UI`, `CS`, `LO` all travel as text.
    pub(crate) fn text(&mut self, tag: Tag, vr: &[u8; 2], v: &str) -> &mut Self {
        self.el(tag, vr, v.as_bytes())
    }

    /// A sequence with an explicit length, holding one item of `inner`.
    pub(crate) fn seq(&mut self, tag: Tag, inner: &[u8]) -> &mut Self {
        let mut item = Vec::new();
        item.extend_from_slice(&0xFFFEu16.to_le_bytes());
        item.extend_from_slice(&0xE000u16.to_le_bytes());
        item.extend_from_slice(&(inner.len() as u32).to_le_bytes());
        item.extend_from_slice(inner);
        self.el(tag, b"SQ", &item)
    }

    /// A sequence with **no** length, terminated by a Sequence Delimitation
    /// Item — the structure a reader has to track nesting through.
    ///
    /// `items` is the raw element bytes of each item's contents.
    pub(crate) fn undefined_seq(&mut self, tag: Tag, items: &[Vec<u8>]) -> &mut Self {
        self.push_u16(tag.0);
        self.push_u16(tag.1);
        if self.explicit() {
            self.data.extend_from_slice(b"SQ");
            self.data.extend_from_slice(&[0, 0]);
        }
        self.push_u32(0xFFFF_FFFF);
        for inner in items {
            // Item, with a length. An undefined-length item is exercised by
            // `undefined_item` below.
            self.push_u16(0xFFFE);
            self.push_u16(0xE000);
            self.push_u32(inner.len() as u32);
            self.data.extend_from_slice(inner);
        }
        // Sequence Delimitation Item: a 4-byte zero length and no VR.
        self.push_u16(0xFFFE);
        self.push_u16(0xE0DD);
        self.push_u32(0);
        self
    }

    /// An undefined-length sequence holding one undefined-length item, which
    /// is the shape that needs both delimiters handled.
    pub(crate) fn undefined_item_seq(&mut self, tag: Tag, inner: &[u8]) -> &mut Self {
        self.push_u16(tag.0);
        self.push_u16(tag.1);
        if self.explicit() {
            self.data.extend_from_slice(b"SQ");
            self.data.extend_from_slice(&[0, 0]);
        }
        self.push_u32(0xFFFF_FFFF);
        self.push_u16(0xFFFE);
        self.push_u16(0xE000);
        self.push_u32(0xFFFF_FFFF);
        self.data.extend_from_slice(inner);
        // Item Delimitation, then Sequence Delimitation.
        self.push_u16(0xFFFE);
        self.push_u16(0xE00D);
        self.push_u32(0);
        self.push_u16(0xFFFE);
        self.push_u16(0xE0DD);
        self.push_u32(0);
        self
    }

    /// Pixel Data with an undefined length, as an encapsulated (compressed)
    /// file has it.
    pub(crate) fn encapsulated_pixels(&mut self) -> &mut Self {
        self.push_u16(PIXEL_DATA.0);
        self.push_u16(PIXEL_DATA.1);
        if self.explicit() {
            self.data.extend_from_slice(b"OB");
            self.data.extend_from_slice(&[0, 0]);
        }
        self.push_u32(0xFFFF_FFFF);
        self
    }

    /// Pixel Data, with its bytes.
    pub(crate) fn pixels(&mut self, data: &[u8]) -> &mut Self {
        self.el(PIXEL_DATA, b"OW", data)
    }

    /// The whole file: preamble, magic, meta group, dataset.
    pub(crate) fn build(&self) -> Vec<u8> {
        self.build_with(true)
    }

    /// The whole file, optionally without the `(0002,0000)` group length — the
    /// non-conformant shape a reader still has to cope with.
    pub(crate) fn build_with(&self, group_length: bool) -> Vec<u8> {
        // The meta group is Explicit VR Little Endian regardless of what the
        // dataset is in, which is the whole reason it can be read first.
        let mut meta = Vec::new();
        let uid = {
            let mut v = self.syntax.as_bytes().to_vec();
            if v.len() % 2 == 1 {
                v.push(0);
            }
            v
        };
        meta.extend_from_slice(&0x0002u16.to_le_bytes());
        meta.extend_from_slice(&0x0010u16.to_le_bytes());
        meta.extend_from_slice(b"UI");
        meta.extend_from_slice(&(uid.len() as u16).to_le_bytes());
        meta.extend_from_slice(&uid);

        let mut out = vec![0u8; 128];
        // The preamble is 128 bytes of anything at all — some vendors put a
        // TIFF header there — so something non-zero goes in it to prove the
        // reader does not look.
        out[0..4].copy_from_slice(b"\x89PNG");
        out.extend_from_slice(b"DICM");
        if group_length {
            out.extend_from_slice(&0x0002u16.to_le_bytes());
            out.extend_from_slice(&0x0000u16.to_le_bytes());
            out.extend_from_slice(b"UL");
            out.extend_from_slice(&4u16.to_le_bytes());
            out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        }
        out.extend_from_slice(&meta);
        out.extend_from_slice(&self.data);
        out
    }
}

/// Raw element bytes, for stuffing inside a sequence item.
pub(crate) fn raw_el(tag: Tag, vr: &[u8; 2], value: &[u8]) -> Vec<u8> {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.el(tag, vr, value);
    d.data
}

const ROWS: Tag = (0x0028, 0x0010);
const COLUMNS: Tag = (0x0028, 0x0011);
const SPACING: Tag = (0x0028, 0x0030);

fn read(file: &[u8], wanted: &[Tag]) -> (Found, Option<std::ops::Range<usize>>) {
    let (syntax, _codec, at) = meta(file).expect("a readable meta group");
    scan(file, at, syntax, wanted).expect("a readable dataset")
}

#[test]
fn the_magic_is_at_offset_128_and_the_preamble_is_not_read() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(ROWS, 64);
    let file = d.build();
    assert!(has_magic(&file));
    // The builder puts a PNG signature in the preamble on purpose: the
    // standard allows anything there, so a reader that validated it would
    // reject conformant files.
    assert_eq!(&file[0..4], b"\x89PNG");
    let (found, _) = read(&file, &[ROWS]);
    assert_eq!(found.us(ROWS, false), Some(64));
}

#[test]
fn a_file_without_the_magic_is_refused() {
    let err = meta(&vec![0u8; 400]).expect_err("no magic, no DICOM");
    assert!(format!("{err}").contains("DICM"), "{err}");
    // And one too short to reach offset 128 at all.
    assert!(meta(&[0u8; 10]).is_err());
}

/// The three uncompressed syntaxes, each actually decoding its own encoding.
/// Implicit VR has no VR on the wire and Big Endian has its numbers the other
/// way round, so a reader that ignored the declared syntax would get a
/// different answer for each.
#[test]
fn each_uncompressed_transfer_syntax_is_read_in_its_own_encoding() {
    for (uid, be) in [
        (EXPLICIT_LE, false),
        (IMPLICIT_LE, false),
        (EXPLICIT_BE, true),
    ] {
        let mut d = Dcm::new(uid);
        d.us(ROWS, 256).us(COLUMNS, 192);
        let file = d.build();
        let (found, _) = read(&file, &[ROWS, COLUMNS]);
        assert_eq!(found.us(ROWS, be), Some(256), "{uid}");
        assert_eq!(found.us(COLUMNS, be), Some(192), "{uid}");
    }
}

/// Every compressed syntax refused, and refused *by name* — the name is the
/// actionable part, since it says what to convert from.
/// What is decoded, and with what. The pairing is the point: every compressed
/// syntax puts its *elements* in plain Explicit VR Little Endian and only its
/// pixels in something else, so a reader that treated "compressed" as a
/// property of the dataset could not read the geometry of a file it can in fact
/// decode.
#[test]
fn each_decodable_transfer_syntax_names_its_element_encoding_and_its_codec() {
    for (uid, syntax, codec) in [
        ("1.2.840.10008.1.2", Syntax::ImplicitLe, Codec::Native),
        ("1.2.840.10008.1.2.1", Syntax::ExplicitLe, Codec::Native),
        ("1.2.840.10008.1.2.2", Syntax::ExplicitBe, Codec::Native),
        ("1.2.840.10008.1.2.5", Syntax::ExplicitLe, Codec::Rle),
        (
            "1.2.840.10008.1.2.4.50",
            Syntax::ExplicitLe,
            Codec::JpegBaseline,
        ),
        (
            "1.2.840.10008.1.2.4.57",
            Syntax::ExplicitLe,
            Codec::JpegLossless,
        ),
        (
            "1.2.840.10008.1.2.4.70",
            Syntax::ExplicitLe,
            Codec::JpegLossless,
        ),
    ] {
        assert_eq!(transfer(uid).expect(uid), (syntax, codec), "{uid}");
    }
}

/// What is not, refused *by name* — the name is the actionable part, since it
/// says what to convert from.
#[test]
fn every_undecodable_transfer_syntax_is_refused_by_name() {
    for (uid, word) in [
        ("1.2.840.10008.1.2.4.51", "extended"),
        ("1.2.840.10008.1.2.4.80", "JPEG-LS"),
        ("1.2.840.10008.1.2.4.81", "JPEG-LS"),
        ("1.2.840.10008.1.2.4.90", "JPEG 2000"),
        ("1.2.840.10008.1.2.4.91", "JPEG 2000"),
        ("1.2.840.10008.1.2.1.99", "deflated"),
    ] {
        let err = transfer(uid).expect_err(uid);
        let msg = format!("{err}");
        assert!(msg.contains(word), "{uid} should be named {word:?}: {msg}");
        // And every refusal says what to do about it.
        assert!(
            msg.contains("dcm2niix") || msg.contains("gdcmconv"),
            "{uid} should suggest a conversion: {msg}"
        );
    }
    // An unknown UID is refused too, and lists what is read.
    let err = transfer("1.2.840.10008.1.2.9.9.9").expect_err("unknown");
    assert!(format!("{err}").contains("1.2.840.10008.1.2.1"), "{err}");
}

/// A long-form VR carries two reserved bytes and a 4-byte length. Reading it as
/// the short form takes the first two bytes of that length as the length
/// itself, which desynchronises the walk — so the value *after* it is what this
/// asserts on.
#[test]
fn a_long_form_vr_does_not_desynchronise_the_walk() {
    let mut d = Dcm::new(EXPLICIT_LE);
    // `OB` is long-form. Then a value that must still arrive intact.
    d.el((0x0029, 0x1010), b"OB", &vec![0xAB; 600])
        .us(ROWS, 321);
    let (found, _) = read(&d.build(), &[ROWS]);
    assert_eq!(
        found.us(ROWS, false),
        Some(321),
        "the element after a long-form VR was not found where it is"
    );
}

/// The catastrophic case. An undefined-length sequence has no byte count, so a
/// reader must find its Sequence Delimitation Item — and if it instead treats
/// 0xFFFFFFFF as a length, or tries to resynchronise, every tag after it comes
/// from the wrong offset.
#[test]
fn an_undefined_length_sequence_is_stepped_over_whole() {
    let inner = [
        raw_el((0x0008, 0x0100), b"SH", b"CODE"),
        raw_el((0x0008, 0x0104), b"LO", b"a coded meaning"),
    ]
    .concat();
    let mut d = Dcm::new(EXPLICIT_LE);
    d.undefined_seq((0x0008, 0x1032), &[inner]) // Procedure Code Sequence
        .us(ROWS, 111)
        .us(COLUMNS, 222);
    let (found, _) = read(&d.build(), &[ROWS, COLUMNS]);
    assert_eq!(found.us(ROWS, false), Some(111));
    assert_eq!(found.us(COLUMNS, false), Some(222));
    // And nothing from inside the sequence leaked out as a top-level value.
    assert!(found.raw((0x0008, 0x0100)).is_none());
}

/// Both delimiters. An undefined-length *item* inside an undefined-length
/// sequence needs the Item Delimitation Item handled as well, and counting it
/// as a sequence delimiter would end the skip one level too early — leaving the
/// walk inside the sequence.
#[test]
fn an_undefined_length_item_is_stepped_over_too() {
    let inner = raw_el((0x0008, 0x0100), b"SH", b"CODE");
    let mut d = Dcm::new(EXPLICIT_LE);
    d.undefined_item_seq((0x0008, 0x1032), &inner).us(ROWS, 77);
    let (found, _) = read(&d.build(), &[ROWS]);
    assert_eq!(found.us(ROWS, false), Some(77));
}

/// Nesting. A sequence inside a sequence, both undefined, is what a real MR
/// object has — and a depth counter that did not count would stop at the inner
/// delimiter and read the outer sequence's remainder as top-level elements.
#[test]
fn a_nested_undefined_length_sequence_is_stepped_over() {
    // Build the inner undefined-length sequence as raw bytes.
    let mut inner_builder = Dcm::new(EXPLICIT_LE);
    inner_builder.undefined_seq(
        (0x0040, 0xA730),
        &[raw_el((0x0008, 0x0100), b"SH", b"DEEP")],
    );
    let inner = inner_builder.data.clone();

    let mut d = Dcm::new(EXPLICIT_LE);
    d.undefined_seq((0x0008, 0x1032), &[inner]).us(ROWS, 999);
    let (found, _) = read(&d.build(), &[ROWS]);
    assert_eq!(
        found.us(ROWS, false),
        Some(999),
        "a nested sequence was not stepped over as one"
    );
}

/// A defined-length sequence is simply skipped by its length, and its contents
/// must not surface as top-level values.
#[test]
fn a_defined_length_sequence_is_skipped_by_its_length() {
    let inner = raw_el((0x0028, 0x0010), b"US", &4000u16.to_le_bytes());
    let mut d = Dcm::new(EXPLICIT_LE);
    d.seq((0x0008, 0x1032), &inner).us(ROWS, 64);
    let (found, _) = read(&d.build(), &[ROWS]);
    assert_eq!(
        found.us(ROWS, false),
        Some(64),
        "a Rows inside a sequence must not be mistaken for the image's"
    );
}

/// Implicit VR has no VR to tell a sequence from anything else, so an
/// undefined length is the only signal — and it is enough, because only
/// sequences and Pixel Data use one.
#[test]
fn an_undefined_length_sequence_is_handled_without_a_vr_to_identify_it() {
    let inner = raw_el((0x0008, 0x0100), b"SH", b"CODE");
    let mut d = Dcm::new(IMPLICIT_LE);
    d.undefined_seq((0x0008, 0x1032), &[inner]).us(ROWS, 42);
    let (found, _) = read(&d.build(), &[ROWS]);
    assert_eq!(found.us(ROWS, false), Some(42));
}

/// Encapsulated Pixel Data is how a compressed file carries its image. Spotted
/// so the caller can say the two halves of the file disagree, rather than
/// reading a basic offset table as pixels.
#[test]
fn encapsulated_pixel_data_is_recognised_as_such() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(ROWS, 64).encapsulated_pixels();
    let (found, pixels) = read(&d.build(), &[ROWS]);
    assert!(
        found.encapsulated_at.is_some(),
        "the item stream's offset is what a codec needs"
    );
    assert!(
        pixels.is_none(),
        "there is no single value to read as pixels"
    );
}

#[test]
fn pixel_data_is_handed_back_as_a_range_rather_than_copied() {
    let bytes: Vec<u8> = (0..64u8).collect();
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(ROWS, 8).us(COLUMNS, 8).pixels(&bytes);
    let file = d.build();
    let (_, pixels) = read(&file, &[ROWS]);
    let range = pixels.expect("pixel data");
    assert_eq!(&file[range], &bytes[..]);
}

/// `DS` and `IS` are *text*, not binary — DICOM writes every spacing and
/// rescale as digits. Reading them as numbers directly is the mistake that
/// gives a spacing of 13.1e9.
#[test]
fn decimal_strings_are_parsed_as_text_and_split_on_backslashes() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(SPACING, b"DS", "0.9375\\0.46875")
        .text((0x0020, 0x0013), b"IS", "17")
        .text((0x0020, 0x0032), b"DS", "-83.5\\-119.2\\61.0");
    let (found, _) = read(&d.build(), &[SPACING, (0x0020, 0x0013), (0x0020, 0x0032)]);
    assert_eq!(found.nums(SPACING), vec![0.9375, 0.46875]);
    assert_eq!(found.num((0x0020, 0x0013)), Some(17.0));
    assert_eq!(found.nums((0x0020, 0x0032)), vec![-83.5, -119.2, 61.0]);
    // A single number is the first of one.
    assert_eq!(found.num(SPACING), Some(0.9375));
}

/// A UID padded to an even length with a NUL must compare equal to the same
/// UID without one, or a series would split in half on the padding.
#[test]
fn a_padded_uid_is_trimmed_to_its_value() {
    let odd = "1.2.840.113619.2.1"; // 18 characters, even — pad an odd one
    let odd = format!("{odd}.5"); // 20... make it genuinely odd
    let odd = format!("{odd}9"); // 21 characters
    assert_eq!(odd.len() % 2, 1, "the test needs an odd-length UID");
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text((0x0020, 0x000E), b"UI", &odd);
    let (found, _) = read(&d.build(), &[(0x0020, 0x000E)]);
    assert_eq!(found.text((0x0020, 0x000E)).as_deref(), Some(odd.as_str()));
}

/// Files without `(0002,0000)` exist, in defiance of the standard. The group
/// boundary is then where the group number changes.
#[test]
fn a_meta_group_without_a_group_length_still_locates_the_dataset() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(ROWS, 300).us(COLUMNS, 400);
    let file = d.build_with(false);
    let (syntax, _codec, at) = meta(&file).expect("still readable");
    assert_eq!(syntax, Syntax::ExplicitLe);
    let (found, _) = scan(&file, at, syntax, &[ROWS, COLUMNS]).expect("dataset");
    assert_eq!(found.us(ROWS, false), Some(300));
    assert_eq!(found.us(COLUMNS, false), Some(400));
}

/// A dataset that simply stops is the ordinary end of a file, not an error —
/// but a truncation inside an element's value is, because the value is then
/// partly missing and using it would be using the bytes of whatever follows.
#[test]
fn a_truncated_element_is_an_error_and_a_clean_end_is_not() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(ROWS, 64).el((0x0029, 0x1010), b"OB", &[7u8; 100]);
    let whole = d.build();
    assert!(meta(&whole).is_ok());

    // Cut inside that last value.
    let cut = &whole[..whole.len() - 40];
    let (syntax, _codec, at) = meta(cut).expect("the meta group is intact");
    assert!(
        scan(cut, at, syntax, &[ROWS]).is_err(),
        "a value cut in half must be an error"
    );
}

/// The two syntaxes that are *not* compressed and are still refused, each
/// saying so — because "uncompressed and refused" reads as a reader that failed
/// to recognise the file unless it says otherwise.
#[test]
fn the_uncompressed_syntaxes_that_are_still_refused_say_why() {
    let err = transfer("1.2.840.10008.1.2.1.98").expect_err("encapsulated");
    let msg = format!("{err}");
    assert!(msg.contains("not compressed"), "{msg}");
    assert!(
        msg.contains("fragments"),
        "should say what the problem is: {msg}"
    );

    let err = transfer("1.2.840.10008.1.2.1.99").expect_err("deflated");
    assert!(format!("{err}").contains("deflated"), "{err}");
}

/// A meta group whose `(0002,0000)` is *wrong* must not move the dataset.
///
/// This is why the boundary is walked rather than computed: a stale group
/// length is the only number in the file nothing else checks, so believing it
/// would start the dataset inside an element and desynchronise everything after
/// — with no symptom short of a wrong image.
#[test]
fn a_wrong_meta_group_length_does_not_move_the_dataset() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(ROWS, 640).us(COLUMNS, 480);
    let mut file = d.build();

    // The group length's value sits at 128 + 4 ("DICM") + 8 (its own header).
    let at = 128 + 4 + 8;
    let honest = u32::from_le_bytes(file[at..at + 4].try_into().expect("four bytes"));
    for lie in [honest + 8, honest.saturating_sub(4), 0, 2] {
        file[at..at + 4].copy_from_slice(&lie.to_le_bytes());
        let (syntax, _codec, data_at) = meta(&file).expect("still readable");
        assert_eq!(syntax, Syntax::ExplicitLe, "group length {lie}");
        let (found, _) = scan(&file, data_at, syntax, &[ROWS, COLUMNS]).expect("dataset");
        assert_eq!(
            (found.us(ROWS, false), found.us(COLUMNS, false)),
            (Some(640), Some(480)),
            "a group length of {lie} must not move the dataset"
        );
    }
}

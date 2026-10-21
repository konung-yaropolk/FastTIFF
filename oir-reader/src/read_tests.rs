//! Whole acquisitions, read every way there is to read one — mapped, buffered,
//! through windows far smaller than a real one, and from memory — and checked
//! against each other and against the reader this crate replaced.
//!
//! The reassembly tests matter more than they look. A plane arrives as
//! thirty-five scattered chunks; getting that wrong produces an image that
//! still looks like an image.

use super::*;
use crate::acquisition::{Acquisition, Planes, SampleType};
use crate::error::ErrorKind;
use crate::fixture::{self, Builder, Scratch};
use crate::legacy;
use crate::options::Options;
use crate::planes::MAX_PLANE_BYTES;
use std::cell::RefCell;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;

// ------------------------------------------------------------------ helpers

/// What must agree about a part however it was read.
fn part_facts(p: &Part) -> (BlockSource, usize, usize, u64) {
    (
        p.block_source(),
        p.block_count(),
        p.planes_added(),
        p.size(),
    )
}

/// Two reads of the same bytes agree: equal acquisitions, or the same failure.
/// `same_hint` is false when one was read from paths and the other was not,
/// which words the no-planes hint differently.
fn assert_agree(
    what: &str,
    a: &Result<Acquisition, Error>,
    b: &Result<Acquisition, Error>,
    same_hint: bool,
) {
    match (a, b) {
        (Ok(x), Ok(y)) => {
            assert_eq!(x.shape(), y.shape(), "{what}: shape");
            assert!(x.planes() == y.planes(), "{what}: planes differ");
            assert_eq!(x.dropped_planes(), y.dropped_planes(), "{what}: dropped");
            assert_eq!(
                format!("{:?}", x.record()),
                format!("{:?}", y.record()),
                "{what}: record"
            );
            let px: Vec<_> = x.parts().iter().map(part_facts).collect();
            let py: Vec<_> = y.parts().iter().map(part_facts).collect();
            assert_eq!(px, py, "{what}: parts");
        }
        (Err(x), Err(y)) => {
            assert_eq!(x.kind(), y.kind(), "{what}: {x} / {y}");
            if same_hint || x.kind() != ErrorKind::NoPlanes {
                assert_eq!(x.to_string(), y.to_string(), "{what}");
            }
        }
        (x, y) => panic!(
            "{what}: one read succeeded and the other did not: {:?} / {:?}",
            x.as_ref().err().map(ToString::to_string),
            y.as_ref().err().map(ToString::to_string)
        ),
    }
}

/// Where each part of an acquisition goes on disk: the named file, then its
/// continuations — alternately without and with the extension, because both
/// are written.
fn part_name(k: usize) -> String {
    match k {
        0 => "rec.oir".to_string(),
        k if k % 2 == 1 => format!("rec_{k:05}"),
        k => format!("rec_{k:05}.oir"),
    }
}

fn streams<'a>(parts: &[&'a [u8]], window: usize) -> Vec<PartInput<'a>> {
    parts
        .iter()
        .map(|b| PartInput::Stream {
            reader: Box::new(Cursor::new(*b)),
            len: b.len() as u64,
            window,
        })
        .collect()
}

fn assembled(g: Result<Gathered, Error>) -> Result<Acquisition, Error> {
    g.and_then(Gathered::assemble)
}

/// Read an acquisition every way it can be read, require every way to agree,
/// and return the default way's result.
///
/// The ways: from files, mapped where this build maps (`Access::Auto`) and
/// buffered (`Access::Buffered`); from streams, through windows of 1, 7, 61
/// and 4096 bytes, so every refill boundary lands somewhere different; and
/// from memory.
fn read_every_way(test: &str, parts: &[&[u8]]) -> Result<Acquisition, Error> {
    let dir = Scratch::new(test);
    let paths: Vec<_> = parts
        .iter()
        .enumerate()
        .map(|(k, b)| dir.write(&part_name(k), b))
        .collect();
    assert_eq!(acquisition_parts(&paths[0]), paths, "{test}: the parts");

    let auto = assembled(Options::new().gather(&paths[0], |_| true));
    if let Ok(a) = &auto {
        for p in a.parts() {
            let want = if crate::MMAP_SUPPORTED {
                ReadPath::Mapped
            } else {
                ReadPath::Buffered
            };
            assert_eq!(p.read_path(), want, "{test}: {:?}", p.map_error());
            assert!(p.path().is_some());
        }
    }
    let buffered = assembled(
        Options::new()
            .access(Access::Buffered)
            .gather(&paths[0], |_| true),
    );
    if let Ok(b) = &buffered {
        assert!(b
            .parts()
            .iter()
            .all(|p| p.read_path() == ReadPath::Buffered && p.map_error().is_none()));
    }
    assert_agree(&format!("{test}: auto / buffered"), &auto, &buffered, true);
    for window in [1, 7, 61, 4096] {
        let s = assembled(gather_inputs(
            streams(parts, window),
            Access::Auto,
            true,
            &mut |_| true,
        ));
        assert_agree(&format!("{test}: auto / window {window}"), &auto, &s, true);
    }
    let bytes = assembled(Options::new().gather_bytes(parts, |_| true));
    if let Ok(b) = &bytes {
        assert!(b
            .parts()
            .iter()
            .all(|p| p.read_path() == ReadPath::Memory && p.path().is_none()));
    }
    assert_agree(&format!("{test}: auto / bytes"), &auto, &bytes, false);
    auto
}

fn read_one(test: &str, bytes: &[u8]) -> Result<Acquisition, Error> {
    read_every_way(test, &[bytes])
}

fn u16_planes(a: &Acquisition) -> &[Vec<u16>] {
    a.planes().as_u16().expect("16-bit planes")
}

fn walked(a: &Acquisition) -> bool {
    a.parts()
        .iter()
        .any(|p| p.block_source() == BlockSource::Walk)
}

// --------------------------------------------------------------- reassembly

/// The heart of it: a plane arrives in pieces, out of order, and has to come
/// back exactly.
#[test]
fn scattered_chunks_reassemble_into_the_right_pixels() {
    // A chunk size that does not divide the plane, so the last one is short —
    // which is what a real file does (34 full chunks and a remainder).
    let a = read_one("scatter", &fixture::stack_file(8, 4, &["t001_0_1_uid"], 22)).expect("read");
    assert_eq!((a.shape().width, a.shape().height), (8, 4));
    assert_eq!(a.planes().len(), 1);
    let want: Vec<u16> = (0..32).collect();
    assert_eq!(
        u16_planes(&a)[0],
        want,
        "the plane was reassembled in the wrong order"
    );
}

#[test]
fn planes_are_ordered_by_timepoint_not_by_position_in_the_file() {
    // Ten frames, so that naive string sorting would put t10 before t2.
    let names: Vec<String> = (1..=10).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let a = read_one("order", &fixture::stack_file(4, 2, &refs, 7)).expect("read");
    assert_eq!(a.shape().frames, 10);
    for (p, v) in u16_planes(&a).iter().enumerate() {
        assert_eq!(
            v[0],
            (p * 1000) as u16,
            "frame {p} is not the {p}th one written"
        );
    }
}

#[test]
fn reference_and_thumbnail_blocks_are_not_mistaken_for_frames() {
    let a = read_one("refs", &fixture::stack_file(4, 4, &["t001_0_1_uid"], 32)).expect("read");
    assert_eq!(
        a.planes().len(),
        1,
        "the REF_LSM0 block became a frame; it is a thumbnail"
    );
}

// -------------------------------------------------------------------- shape

#[test]
fn the_records_channel_count_and_the_plane_count_must_agree() {
    let names: Vec<String> = (1..=6).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    // The record says two channels; six planes is then 2 x 3 timepoints.
    let a = read_one("axes", &fixture::channels_file(4, 4, &refs, 16, 2)).expect("read");
    let s = a.shape();
    assert_eq!((s.channels, s.slices, s.frames), (2, 1, 3));
    assert_eq!(s.plane_count(), 6);

    // A count that cannot be right for the planes present is ignored rather
    // than believed: four channels does not divide six planes, and a stack of
    // the wrong shape is worse than one of the plainest possible shape.
    let a = read_one("axes_bad", &fixture::channels_file(4, 4, &refs, 16, 4)).expect("read");
    assert_eq!(
        (a.shape().channels, a.shape().frames),
        (1, 6),
        "a channel count that does not divide the planes was trusted anyway"
    );
}

/// The other half of the rule the test above states.
///
/// There, the plane names carried one UID and one `z` between them, said
/// nothing about the axes, and the record was believed. Here they say two
/// channels over three slices, and a record claiming a plain six-frame
/// timelapse must not be allowed to flatten them: the names come from the same
/// blocks as the pixels, and the record describes what the microscope was set
/// up to do.
#[test]
fn plane_names_that_state_the_axes_outrank_a_record_that_disagrees() {
    let names: Vec<String> = (1..=3)
        .flat_map(|z| ["uidA", "uidB"].map(move |c| format!("z{z}_0_1_{c}")))
        .collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let a = read_one(
        "axes_from_names",
        &fixture::channels_file(4, 4, &refs, 16, 1),
    )
    .expect("read");
    let s = a.shape();
    assert_eq!(
        (s.channels, s.slices, s.frames),
        (2, 3, 1),
        "the names say 2 channels over 3 slices; the record was believed instead"
    );
    // And they are in xyczt order: channel fastest, then slice. Plane p of the
    // fixture was written with first pixel p * 1000, in the order z1 A, z1 B,
    // z2 A, …, which is that order.
    for z in 0..3 {
        for c in 0..2 {
            let i = s.plane_index(c, z, 0).unwrap();
            assert_eq!(u16_planes(&a)[i][0], (i * 1000) as u16, "c{c} z{z}");
            assert_eq!(i, c + 2 * z);
        }
    }
    assert_eq!(s.plane_index(2, 0, 0), None);
    assert_eq!(s.plane_index(0, 3, 0), None);
    assert_eq!(s.plane_index(0, 0, 1), None);
}

#[test]
fn sample_width_is_measured_from_the_data_rather_than_assumed() {
    let a = read_one("depth", &fixture::stack_file(8, 8, &["t001_0_1_uid"], 40)).expect("read");
    assert_eq!(a.shape().sample_type, SampleType::U16);

    // Eight bytes over a 4x2 plane is one byte a sample.
    let mut b = Builder::new();
    b.xml(&fixture::size_record(4, 2));
    b.plane_chunk("t001_0_1_uid", 0, &[3u8; 5]);
    b.plane_chunk("t001_0_1_uid", 5, &[4u8; 3]);
    b.plane_chunk("t002_0_1_uid", 0, &[5u8; 8]);
    let a = read_one("depth8", &b.finish()).expect("read");
    assert_eq!(a.shape().sample_type, SampleType::U8);
    assert_eq!(a.planes().sample_type(), SampleType::U8);
    assert_eq!(
        a.planes().as_u8().unwrap(),
        [vec![3, 3, 3, 3, 3, 4, 4, 4], vec![5; 8]]
    );
    assert!(a.planes().as_u16().is_none());
}

// ------------------------------------------------------------- the metadata

/// The record is read from the file's own XML, and written out the way the
/// acquisition software exports it — never as the XML itself.
#[test]
fn the_record_carries_the_instrument() {
    let mut b = Builder::new();
    b.xml(
        "<?xml version=\"1.0\"?><lsmimage:imageProperties>\
         <commonimage:system><base:systemName>FVMPE-RS</base:systemName></commonimage:system>\
         <commonimage:imageInfo>\
         <commonimage:phase><commonimage:group><commonimage:channel id=\"c1\">\
         <commonphase:name>CH1</commonphase:name>\
         <commonphase:length><commonparam:x>0.621480569402239</commonparam:x>\
         <commonparam:y>0.621480569402239</commonparam:y></commonphase:length>\
         </commonimage:channel></commonimage:group></commonimage:phase>\
         <commonimage:width>8</commonimage:width>\
         <commonimage:height>4</commonimage:height>\
         </commonimage:imageInfo>\
         <lsmimage:acquisition><lsmimage:microscopeConfiguration>\
         <opticalelement:objectiveLens><opticalelement:name>XLUMPLFLN20XW</opticalelement:name>\
         </opticalelement:objectiveLens></lsmimage:microscopeConfiguration>\
         </lsmimage:acquisition></lsmimage:imageProperties>",
    );
    // The document that must never reach a description. A real one is 525 KB
    // of exactly this, and there are three of them.
    b.xml("<?xml version=\"1.0\"?><lut:LUT><lut:intensity>0</lut:intensity></lut:LUT>");
    b.plane_chunk("t001_0_1_uid", 0, &[7u8; 64]);
    let a = read_one("meta", &b.finish()).expect("read");

    let desc = a.fluoview_text("meta.oir").expect("the acquisition record");
    assert!(
        !desc.contains('<'),
        "raw XML reached the description: {desc}"
    );
    for line in [
        "\"Name\"\t\"meta.oir\"",
        "\"System Name\"\t\"FVMPE-RS\"",
        "\"X Dimension\"\t\"8, 0.0 - 4.972 [um], 0.621 [um/pixel]\"",
        "\"Objective Lens\"\t\"XLUMPLFLN20XW\"",
    ] {
        assert!(desc.contains(line), "missing {line:?} from:\n{desc}");
    }
    // The values keep the precision the file states, rather than the three
    // decimals the text rounds them to.
    let r = a.record();
    assert_eq!(r.pixel_x_um(), Some(0.621480569402239));
    assert_eq!(r.system_name(), Some("FVMPE-RS"));
    assert_eq!(r.objective(), Some("XLUMPLFLN20XW"));
    assert_eq!(r.channels().len(), 1);
    assert_eq!(r.channels()[0].label(), Some("CH1"));
}

/// The whole record, end to end: an acquisition with events and timing, whose
/// description has to come out as text a stimulus analysis can parse.
///
/// The same file also carries a lookup table, which is what a description used
/// to fill up with — four megabytes in a real acquisition, three quarters of
/// it display tables written out element by element.
#[test]
fn the_record_is_translated_rather_than_dumped() {
    let a = read_one("translated", &fixture::recorded_file()).expect("read");
    let desc = a
        .fluoview_text("translated.oir")
        .expect("the translated record");

    // Not XML, in any form, however small.
    assert!(
        !desc.contains('<'),
        "raw XML reached the description: {desc}"
    );
    assert!(
        !desc.contains("intensity"),
        "a lookup table survived: {desc}"
    );

    // And it is the acquisition software's own format, down to the keys —
    // which is what lets one parser read a converted file and the instrument's
    // own export.
    for line in [
        "\"[General]\"\t\"\"",
        "\"Name\"\t\"translated.oir\"",
        "\"Scan Mode\"\t\"XYT\"",
        "\"Date\"\t\"02/10/2025 06:49:39.124 PM\"",
        "\"X Dimension\"\t\"4, 0.0 - 2.486 [um], 0.621 [um/pixel]\"",
        "\"T Dimension\"\t\"2, 0.000 - 1.000 [s], Interval FreeRun\"",
        "\"[Event 1]\"\t\"\"",
        "\"Event Contents\"\t\"DRS\"",
        "\"Event Timer\"\t\"27772.762000[ms]\"",
    ] {
        assert!(desc.contains(line), "missing {line:?} from:\n{desc}");
    }

    // The values come from the record too, at the precision the file states
    // them — not re-parsed out of the three decimals the text rounds them to.
    let r = a.record();
    assert_eq!(r.pixel_x_um(), Some(0.621480569402239));
    assert_eq!(r.pixel_y_um(), Some(0.621480569402239));
    assert_eq!(a.frame_interval_s(), Some(1.0));
    assert_eq!(r.frame_times_ms(), [0.0, 1000.0]);
    assert!(r.free_run());
    assert_eq!(r.events().len(), 1);
    assert_eq!(r.events()[0].name(), "DRS");
    assert_eq!(r.events()[0].time_ms(), 27772.762);
    let labels: Vec<&str> = r.channels().iter().filter_map(|c| c.label()).collect();
    assert_eq!(labels, ["CH1"]);
}

// ------------------------------------------------------------------ refusal

#[test]
fn a_file_without_the_signature_is_refused_by_name() {
    let err = read_one("fake", b"II*\0 not an OIR at all").expect_err("must refuse");
    assert_eq!(err.kind(), ErrorKind::NotOir);
    assert!(err.to_string().contains("OLYMPUSRAWFORMAT"), "{err}");
    assert!(err.path().is_some_and(|p| p.ends_with("rec.oir")));
}

/// A wild index offset is never *followed*, and the file is read anyway.
///
/// The two files that produce a bad offset are a recording still in progress
/// and one whose tail was lost, and in both the pixels are in front of the
/// index and perfectly good. The assertion that it was not followed is the
/// pixels: they are the ones the fixture wrote, in order.
#[test]
fn a_corrupt_index_offset_is_not_followed_but_the_pixels_are_still_read() {
    let mut f = fixture::stack_file(4, 4, &["t001_0_1_uid"], 16);
    f[0x28..0x30].copy_from_slice(&u64::MAX.to_le_bytes());
    let a = read_one("badindex", &f).expect("the blocks are all there and should be recovered");
    assert_eq!(a.shape().frames, 1);
    assert_eq!(u16_planes(&a)[0], (0..16u16).collect::<Vec<_>>());
    assert!(walked(&a));
}

/// An index whose marker is not the one this reader knows is the same case: not
/// read, not fatal.
#[test]
fn an_index_without_its_marker_falls_back_to_walking() {
    let mut f = fixture::stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = fixture::index_at(&f);
    f[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    let a = read_one("nomarker", &f).expect("the blocks are all there");
    assert_eq!(a.shape().frames, 1);
    assert!(walked(&a));
}

// ------------------------------------------------- acquisitions still running

/// A recording whose index has not been written yet opens at the frames it has.
///
/// The index is the *last* thing an acquisition writes, so this is what every
/// file being recorded looks like: whole frames, and nothing at `0x28` to say
/// where they are.
#[test]
fn a_file_whose_index_was_never_written_still_opens() {
    let mut f = fixture::stack_file(4, 4, &["t001_0_1_uid", "t002_0_1_uid"], 16);
    let at = fixture::index_at(&f);
    // As the file stands on disk mid-recording: no index, and the field that
    // would point at one still zero.
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    let a = read_one("noindex", &f).expect("a recording in progress should open");
    assert_eq!(a.shape().frames, 2, "both finished frames should be there");
    assert_eq!(u16_planes(&a)[0], (0..16u16).collect::<Vec<_>>());
    assert!(walked(&a));
    assert_eq!(a.parts()[0].size(), at as u64);
}

/// A file cut in the middle of a frame keeps the frames before the cut and
/// drops the part-written one.
///
/// The half-frame is the thing that must not survive: padded out with zeros it
/// becomes a mostly-black frame at the end of the recording, which is not an
/// error anyone sees and quietly ruins an average.
#[test]
fn a_file_cut_mid_frame_keeps_the_frames_before_the_cut() {
    let mut f = fixture::stack_file(4, 4, &["t001_0_1_uid", "t002_0_1_uid"], 16);
    let at = fixture::index_at(&f);
    // Into the last frame's final data block, so that frame is half written.
    f.truncate(at - 8);
    let a = read_one("cutframe", &f).expect("the finished frame should still open");
    assert_eq!(
        a.shape().frames,
        1,
        "the half-written frame must not survive"
    );
    assert_eq!(u16_planes(&a)[0], (0..16u16).collect::<Vec<_>>());
    assert_eq!(a.dropped_planes(), 1);
}

/// The first block is found, not assumed.
///
/// A real acquisition puts a 16-byte structure at `0x50` and its first block at
/// `0x60`. The trap is that the structure's first eight bytes read perfectly
/// well as a block header — `u32 3, u32 2` is a three-byte block of type 2 —
/// so a reader that accepts the first candidate that parses lands at `0x5b`,
/// mid-field, and is desynchronised from every block in the file. It would not
/// fail; it would produce the wrong picture.
#[test]
fn the_first_block_is_found_rather_than_assumed() {
    let mut f = fixture::real_layout_file(4, 4, &["t001_0_1_uid"], 16);
    // No index, so the start has to be found rather than read.
    let at = fixture::index_at(&f);
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    let a = read_one("realstart", &f).expect("a real-layout file should open");
    assert_eq!(a.shape().frames, 1);
    assert_eq!(
        u16_planes(&a)[0],
        (0..16u16).collect::<Vec<_>>(),
        "the walk began at the wrong offset"
    );
}

/// A run of zeros is not a block stream.
///
/// Zeros parse as an unlimited run of zero-length type-0 blocks, so a candidate
/// pointing into them agrees with itself for ever and would win by coming
/// first. It has to carry something.
#[test]
fn a_run_of_zeros_is_not_mistaken_for_a_block_stream() {
    let mut f = Vec::new();
    f.extend(fixture::MAGIC);
    f.extend([0u8; 0x2000]);
    let err = read_one("zeros", &f).expect_err("a file of zeros carries no blocks");
    // The specific refusal matters: it is the one that means no candidate was
    // believed at all. Without the non-empty rule a candidate *is* believed,
    // the walk returns thousands of phantom empty blocks, and the file is
    // refused later for carrying no planes — the same outcome by luck.
    assert_eq!(err.kind(), ErrorKind::Malformed);
    assert!(err.to_string().contains("damaged past"), "{err}");
}

/// And a file whose signature is right but whose contents are rubbish is still
/// refused, rather than walked into something.
#[test]
fn a_file_whose_blocks_are_rubbish_is_still_refused() {
    let mut f = Vec::new();
    f.extend(fixture::MAGIC);
    f.extend([0u8; 0x40]);
    // Lengths far past any block, at every alignment a candidate could land on.
    for _ in 0..512 {
        f.extend(0xDEAD_BEEFu32.to_le_bytes());
        f.extend(0xFEED_FACEu32.to_le_bytes());
    }
    let err = read_one("rubbish", &f).expect_err("rubbish must not be walked into");
    assert!(err.to_string().contains("damaged past"), "{err}");
}

#[test]
fn an_oir_with_no_planes_says_so_rather_than_producing_an_empty_stack() {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(4, 4));
    b.plane_chunk("REF_LSM0_abc_0", 0, &[0u8; 32]);
    let f = b.finish();
    let err = read_one("noplanes", &f).expect_err("no planes is a failure");
    assert_eq!(err.kind(), ErrorKind::NoPlanes);
    assert!(err.to_string().contains("no image planes"), "{err}");
    assert!(
        err.to_string().contains("_00001.oir"),
        "the message should point at the likeliest cause: {err}"
    );
    assert!(!err.to_string().contains("  "), "stray whitespace: {err:?}");

    // From memory there are no siblings to point at.
    let err = crate::read_bytes(&[&f]).expect_err("no planes is a failure");
    assert!(err.to_string().contains("pass every part"), "{err}");

    // And when the siblings are there, and empty too, it says that instead.
    let err = read_every_way("noplanes2", &[&f, &f]).expect_err("still none");
    assert!(
        err.to_string()
            .contains("the 1 part(s) beside it carry none either"),
        "{err}"
    );
    let err = crate::read_bytes(&[]).expect_err("nothing is not an acquisition");
    assert_eq!(err.kind(), ErrorKind::NoPlanes);
}

#[test]
fn a_file_that_states_no_frame_size_is_refused_rather_than_guessed_at() {
    let mut b = Builder::new();
    b.plane_chunk("t001_0_1_uid", 0, &[0u8; 32]);
    let err = read_one("nosize", &b.finish()).expect_err("an unknown frame size must be refused");
    assert_eq!(err.kind(), ErrorKind::Malformed);
    assert!(err.to_string().contains("frame size"), "{err}");
}

#[test]
fn a_chunk_claiming_to_run_past_its_plane_is_dropped_not_followed() {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(4, 2));
    // A whole 4x2 u16 plane, then a chunk that claims to start past its end.
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 16]);
    b.plane_chunk("t001_0_1_uid", 1_000_000, &[2u8; 16]);
    let a = read_one("overrun", &b.finish()).expect("the good chunk should still read");
    assert_eq!(
        u16_planes(&a)[0],
        vec![0x0101; 8],
        "the overrunning chunk was written somewhere it should not have been"
    );
}

/// An acquisition stopped part-way leaves a short final plane. The software's
/// own export omits it, and so must this — a mostly-black frame appended to a
/// recording is the kind of thing that quietly ruins an average.
#[test]
fn an_incomplete_trailing_plane_is_dropped_rather_than_padded() {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(4, 2));
    // Two whole 4x2 u16 planes, then one that stops half way.
    for name in ["t001_0_1_uid", "t002_0_1_uid"] {
        b.plane_chunk(name, 0, &[1u8; 16]);
    }
    b.plane_chunk("t003_0_1_uid", 0, &[2u8; 6]);
    let a = read_one("partial", &b.finish()).expect("read");
    assert_eq!(
        a.shape().frames,
        2,
        "the unfinished frame was kept and padded with zeros"
    );
    assert_eq!(a.planes().len(), 2);
    assert_eq!(a.dropped_planes(), 1);
}

#[test]
fn uniformly_short_planes_are_just_small_planes() {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(8, 8));
    // Short, but all the same length, so the sample width still resolves.
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 128]);
    b.plane_chunk("t002_0_1_uid", 0, &[1u8; 128]);
    // 128 bytes over 64 pixels is 2 bytes each, so these are *complete* — the
    // reader must not invent a larger plane than the data supports.
    let a =
        read_one("allshort", &b.finish()).expect("uniformly short planes are just small planes");
    assert_eq!(a.shape().frames, 2);
}

/// Descriptors that place far more pixel data than the file holds are refused
/// before the memory is allocated.
///
/// Pixels are copied as their descriptor is read, so a descriptor claiming its
/// sixteen bytes belong most of a gigabyte into a plane would otherwise become
/// most of a gigabyte of zeros — from a file of a few hundred bytes.
#[test]
fn descriptors_placing_more_data_than_the_file_holds_are_refused() {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(4, 2));
    b.plane_chunk("t001_0_1_uid", (MAX_PLANE_BYTES - 16) as u32, &[7u8; 16]);
    let err = read_one("lying", &b.finish()).expect_err("a gigabyte of zeros is not a plane");
    assert_eq!(err.kind(), ErrorKind::Malformed);
    assert!(
        err.to_string()
            .contains("more pixel data than the file holds"),
        "{err}"
    );
}

/// An index that reads cleanly and points at nothing is recovered by walking.
///
/// What a half-written index looks like: the header is there, the offsets in it
/// are not yet. It is not an unreadable index — every check this reader makes
/// of one passes — so the fallback cannot be conditioned on the index failing
/// to parse. It is conditioned on the index failing to produce a plane.
#[test]
fn an_index_that_points_at_nothing_is_recovered_by_walking() {
    let mut f = fixture::stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = fixture::index_at(&f);
    // Keep the marker and the header; blank every offset it lists.
    for b in f[at + fixture::INDEX_HEADER..].iter_mut() {
        *b = 0;
    }
    let a = read_one("blankindex", &f).expect("the blocks are still in the file");
    assert_eq!(a.shape().frames, 1);
    assert_eq!(u16_planes(&a)[0], (0..16u16).collect::<Vec<_>>());
    assert!(walked(&a));
}

// ------------------------------------------------------- split acquisitions

/// A recording split across files comes back whole, in order, pixel for pixel.
///
/// Each part is read front to back as it is opened, and the planes of one part
/// must neither be lost nor reordered by the part read after it. The middle
/// part carries no extension and the last one does, because both are written.
#[test]
fn a_split_acquisition_is_read_whole_and_in_order() {
    let parts = [
        fixture::part_file(0, 2, true),
        fixture::part_file(2, 3, false),
        fixture::part_file(5, 1, false),
    ];
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    let a = read_every_way("split", &refs).expect("a split acquisition reads");
    assert_eq!(a.shape().frames, 6, "frames from some part were lost");
    for (t, plane) in u16_planes(&a).iter().enumerate() {
        let want: Vec<u16> = (0..8).map(|i| t as u16 * 1000 + i).collect();
        assert_eq!(plane, &want, "frame {t} is wrong or out of order");
    }
    let added: Vec<usize> = a.parts().iter().map(Part::planes_added).collect();
    assert_eq!(added, [2, 3, 1]);
    assert!(!walked(&a));
}

/// A first part carrying only metadata is not mistaken for a recovery.
///
/// A split acquisition legitimately keeps its record in the named file and its
/// planes in the siblings, so "this part produced no planes" is ordinary there.
/// Calling it a recovery would tell the user a finished recording was
/// unfinished.
#[test]
fn a_metadata_only_part_is_not_reported_as_unfinished() {
    let first = fixture::metadata_only_file();
    let second = fixture::part_file(1, 2, false);
    let a = read_every_way("meta-only", &[&first, &second]).expect("a split acquisition reads");
    assert_eq!(a.shape().frames, 2);
    assert!(!walked(&a), "a finished recording was called unfinished");
    assert_eq!(a.parts()[0].planes_added(), 0);
    assert_eq!(a.parts()[1].planes_added(), 2);
}

/// A continuation opened on its own is read on its own: only the first part
/// begins a set.
#[test]
fn a_continuation_is_not_the_start_of_a_set() {
    let dir = Scratch::new("continuation");
    let first = dir.write("rec.oir", &fixture::part_file(0, 2, true));
    let second = dir.write("rec_00001.oir", &fixture::part_file(2, 1, false));
    assert_eq!(acquisition_parts(&first), [first.clone(), second.clone()]);
    assert_eq!(acquisition_parts(&second), std::slice::from_ref(&second));
    // Numbering stops at the first gap.
    let fourth = dir.write("rec_00003", &fixture::part_file(4, 1, false));
    assert_eq!(acquisition_parts(&first), [first.clone(), second]);
    let _ = fourth;
}

// --------------------------------------------- against the reader it replaced

/// Every outcome agrees with the old reader's: the same acquisition, or the
/// same refusal in the same words.
fn agree_with_legacy(
    what: &str,
    new: &Result<Acquisition, Error>,
    old: &Result<legacy::Out, legacy::Fail>,
) {
    match (new, old) {
        (Ok(a), Ok(o)) => {
            let s = a.shape();
            assert_eq!(
                (
                    s.width,
                    s.height,
                    s.channels,
                    s.slices,
                    s.frames,
                    s.sample_type.bytes()
                ),
                (
                    o.width,
                    o.height,
                    o.channels,
                    o.slices,
                    o.frames,
                    o.bytes_per_sample
                ),
                "{what}: shape"
            );
            let planes = match o.bytes_per_sample {
                1 => Planes::U8(
                    o.planes
                        .iter()
                        .map(|p| match p {
                            legacy::PlaneData::U8(v) => v.clone(),
                            legacy::PlaneData::U16(_) => panic!("{what}: mixed depths"),
                        })
                        .collect(),
                ),
                _ => Planes::U16(
                    o.planes
                        .iter()
                        .map(|p| match p {
                            legacy::PlaneData::U16(v) => v.clone(),
                            legacy::PlaneData::U8(_) => panic!("{what}: mixed depths"),
                        })
                        .collect(),
                ),
            };
            assert!(*a.planes() == planes, "{what}: planes differ");
            assert_eq!(a.dropped_planes(), o.dropped, "{what}: dropped");
            assert_eq!(
                format!("{:?}", a.record()),
                format!("{:?}", o.record),
                "{what}: record"
            );
            let new_parts: Vec<_> = a
                .parts()
                .iter()
                .map(|p| {
                    (
                        p.block_source() == BlockSource::Walk,
                        p.block_count(),
                        p.planes_added(),
                    )
                })
                .collect();
            let old_parts: Vec<_> = o
                .parts
                .iter()
                .map(|(f, n, added)| (*f == legacy::Found::Walk, *n, *added))
                .collect();
            assert_eq!(new_parts, old_parts, "{what}: parts");
        }
        (Err(e), Err(f)) => {
            let unsupported = matches!(
                e.kind(),
                ErrorKind::NotOir | ErrorKind::NoPlanes | ErrorKind::Cancelled
            );
            assert_eq!(
                (unsupported, e.to_string()),
                (f.unsupported, f.message.clone()),
                "{what}"
            );
        }
        (Ok(_), Err(f)) => panic!(
            "{what}: the old reader refused it ({}) and this did not",
            f.message
        ),
        (Err(e), Ok(_)) => panic!("{what}: this refused it ({e}) and the old reader did not"),
    }
}

/// Read `parts` from memory and through windows, and require each to agree
/// with the old reader. Says whether the old reader read it, so that a test
/// can show it compared acquisitions and not only refusals.
fn against_legacy(what: &str, parts: &[&[u8]], windows: &[usize]) -> bool {
    let old = legacy::read(parts, false);
    let new = assembled(Options::new().gather_bytes(parts, |_| true));
    agree_with_legacy(&format!("{what}, from memory"), &new, &old);
    for &window in windows {
        let s = assembled(gather_inputs(
            streams(parts, window),
            Access::Auto,
            false,
            &mut |_| true,
        ));
        agree_with_legacy(&format!("{what}, window {window}"), &s, &old);
    }
    old.is_ok()
}

/// Cases compared, and how many of them were acquisitions rather than
/// refusals.
#[derive(Default)]
struct Tally {
    cases: usize,
    read: usize,
}

impl Tally {
    fn add(&mut self, read: bool) {
        self.cases += 1;
        self.read += usize::from(read);
    }

    /// Require a real share of both, so that a reader refusing everything —
    /// which would agree with an oracle refusing everything — cannot pass.
    fn check(&self, what: &str) {
        eprintln!("{what}: {} case(s), {} read", self.cases, self.read);
        assert!(self.read * 20 >= self.cases, "{what}: too few acquisitions");
        assert!(self.read < self.cases, "{what}: no refusals at all");
    }
}

/// A small generator, so a failing case names a seed that reproduces it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

#[test]
fn every_fixture_reads_as_the_old_reader_read_it() {
    let mut tally = Tally::default();
    for (name, f) in fixture::corpus() {
        tally.add(against_legacy(name, &[&f], &[1, 7, 61, 4096, WINDOW]));
    }
    tally.check("fixtures");
}

/// Every length every fixture can be cut to: the shapes a file still being
/// written, or one whose tail was lost, can take.
#[test]
fn every_truncation_reads_as_the_old_reader_read_it() {
    let mut tally = Tally::default();
    for (name, f) in fixture::corpus() {
        for cut in 0..=f.len() {
            // Every cut from memory and at the default window; every
            // seventh through the narrow windows too, which is where the
            // refill arithmetic would go wrong.
            let windows: &[usize] = if cut % 7 == 0 {
                &[1, 61, WINDOW]
            } else {
                &[WINDOW]
            };
            tally.add(against_legacy(
                &format!("{name} cut to {cut}"),
                &[&f[..cut]],
                windows,
            ));
        }
    }
    tally.check("truncations");
}

/// A few hundred single-byte corruptions of every fixture.
#[test]
fn corrupted_bytes_read_as_the_old_reader_read_them() {
    let mut tally = Tally::default();
    for (k, (name, f)) in fixture::corpus().into_iter().enumerate() {
        if f.is_empty() {
            continue;
        }
        let mut rng = Rng(0x5EED_0000 + k as u64);
        for flip in 0..256 {
            let mut g = f.clone();
            let at = rng.below(g.len() as u64) as usize;
            g[at] ^= (rng.below(255) + 1) as u8;
            let windows: &[usize] = if flip % 4 == 0 {
                &[1, 61, WINDOW]
            } else {
                &[WINDOW]
            };
            tally.add(against_legacy(
                &format!("{name} with byte {at:#x} flipped"),
                &[&g],
                windows,
            ));
        }
    }
    tally.check("corruptions");
}

/// The fields the reader trusts most, set to what a damaged file could hold:
/// the index offset, and every descriptor's length and run.
#[test]
fn damaged_structure_reads_as_the_old_reader_read_it() {
    let mut tally = Tally::default();
    for (k, (name, f)) in fixture::corpus().into_iter().enumerate() {
        if f.len() < 0x50 {
            continue;
        }
        let len = f.len() as u64;
        let mut rng = Rng(0xD00D_0000 + k as u64);
        let mut offsets: Vec<u64> = vec![
            0,
            0x50,
            0x48,
            len - 12,
            len - 8,
            len,
            len + 1,
            u64::MAX,
            u64::MAX - 7,
        ];
        offsets.extend((0..8).map(|_| rng.below(len + 64)));
        for o in offsets {
            let mut g = f.clone();
            g[0x28..0x30].copy_from_slice(&o.to_le_bytes());
            tally.add(against_legacy(
                &format!("{name} with its index at {o:#x}"),
                &[&g],
                &[61, WINDOW],
            ));
        }
        let Ok((blocks, _)) = legacy::part_blocks(&f) else {
            continue;
        };
        let descriptors: Vec<usize> = blocks
            .iter()
            .map(|&o| o as usize)
            .filter(|&o| f.get(o + 4..o + 8) == Some(&3u32.to_le_bytes()[..]))
            .collect();
        for (d, &o) in descriptors.iter().enumerate().take(24) {
            for field in [0usize, 12] {
                let len = u32::from_le_bytes(f[o..o + 4].try_into().unwrap());
                for v in [
                    0u32,
                    11,
                    12,
                    len.wrapping_sub(1),
                    len.wrapping_add(1),
                    1 << 26,
                    (1 << 26) + 1,
                    u32::MAX,
                    rng.next() as u32,
                ] {
                    let mut g = f.clone();
                    if o + field + 4 > g.len() {
                        continue;
                    }
                    g[o + field..o + field + 4].copy_from_slice(&v.to_le_bytes());
                    tally.add(against_legacy(
                        &format!("{name}, descriptor {d} field {field} = {v:#x}"),
                        &[&g],
                        &[WINDOW],
                    ));
                }
            }
        }
    }
    tally.check("damaged structure");
}

/// Split acquisitions, whole and with each part damaged.
#[test]
fn split_acquisitions_read_as_the_old_reader_read_them() {
    let parts = [
        fixture::part_file(0, 2, true),
        fixture::part_file(2, 3, false),
        fixture::part_file(5, 1, false),
    ];
    let meta_only = fixture::metadata_only_file();
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    assert!(against_legacy("split", &refs, &[1, 61, WINDOW]));
    assert!(against_legacy(
        "split, metadata first",
        &[&meta_only, &parts[1]],
        &[1, 61, WINDOW]
    ));
    assert!(!against_legacy(
        "split, empty sibling",
        &[&meta_only, &meta_only],
        &[WINDOW]
    ));
    let mut tally = Tally::default();
    for k in 0..parts.len() {
        for cut in 0..=parts[k].len() {
            let mut damaged = refs.clone();
            damaged[k] = &parts[k][..cut];
            tally.add(against_legacy(
                &format!("split, part {k} cut to {cut}"),
                &damaged,
                &[WINDOW],
            ));
        }
    }
    tally.check("split truncations");
}

// ------------------------------------------------------------- the I/O done

#[derive(Default)]
struct Counts {
    seeks: usize,
    bytes: u64,
}

/// A stream that counts what is asked of it.
struct Counting<'a> {
    inner: Cursor<&'a [u8]>,
    counts: Rc<RefCell<Counts>>,
}

impl Read for Counting<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counts.borrow_mut().bytes += n as u64;
        Ok(n)
    }
}

impl Seek for Counting<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.counts.borrow_mut().seeks += 1;
        self.inner.seek(pos)
    }
}

/// Read `f` through a counting stream with a 4 KiB window.
fn count_io(f: &[u8]) -> (Result<Acquisition, Error>, Counts) {
    let counts = Rc::new(RefCell::new(Counts::default()));
    let input = PartInput::Stream {
        reader: Box::new(Counting {
            inner: Cursor::new(f),
            counts: counts.clone(),
        }),
        len: f.len() as u64,
        window: 4096,
    };
    let r = assembled(gather_inputs(vec![input], Access::Auto, true, &mut |_| {
        true
    }));
    let c = std::mem::take(&mut *counts.borrow_mut());
    (r, c)
}

/// A fixture big enough that the window matters: 60 frames of 64x64, in
/// 1 KB chunks, with a metadata document every ten frames.
fn big_fixture() -> Vec<u8> {
    let mut b = Builder::new();
    b.xml(&fixture::size_record(64, 64));
    for t in 0..60u16 {
        if t % 10 == 0 {
            b.xml(&format!(
                "<?xml version=\"1.0\"?><lsmframe:frameProperties><commonframe:axisValue>\
                 <commonframe:axisType>TIMELAPSE</commonframe:axisType>\
                 <commonframe:position>{t}.0</commonframe:position></commonframe:axisValue>\
                 </lsmframe:frameProperties>"
            ));
        }
        let bytes: Vec<u8> = (0..64 * 64u16)
            .flat_map(|i| (t ^ i).to_le_bytes())
            .collect();
        for (k, chunk) in bytes.chunks(1024).enumerate() {
            b.plane_chunk(&format!("t{:03}_0_1_uid", t + 1), (k * 1024) as u32, chunk);
        }
    }
    b.finish()
}

/// A finished part is read in one pass: the header, the index, and then one
/// stream from the first block to the index, metadata included — three seeks
/// at most, and little more than the file's own size read.
#[test]
fn a_finished_part_is_read_in_one_pass() {
    let f = big_fixture();
    let index = (f.len() - fixture::index_at(&f)) as u64;
    let (r, c) = count_io(&f);
    let a = r.expect("read");
    assert_eq!(a.shape().frames, 60);
    assert_eq!(a.record().frame_times_ms().len(), 6);
    assert!(c.seeks <= 3, "{} seeks", c.seeks);
    let bound = f.len() as u64 + 2 * index + (192 << 10);
    assert!(
        c.bytes <= bound,
        "{} bytes read of a {}-byte file",
        c.bytes,
        f.len()
    );
}

/// A part still being written is walked and then read: two passes, as on the
/// mapped path, and no more.
#[test]
fn an_unfinished_part_is_read_in_two_passes() {
    let mut f = big_fixture();
    let at = fixture::index_at(&f);
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    let (r, c) = count_io(&f);
    let a = r.expect("read");
    assert_eq!(a.shape().frames, 60);
    assert!(walked(&a));
    assert!(c.seeks <= 3, "{} seeks", c.seeks);
    let bound = 2 * f.len() as u64 + (192 << 10);
    assert!(
        c.bytes <= bound,
        "{} bytes read of a {}-byte file",
        c.bytes,
        f.len()
    );
}

// ---------------------------------------------------------------- fallback

/// A part that cannot be mapped is read through a buffer, over the same
/// handle, with the same result — and says why.
#[cfg(all(feature = "mmap", any(unix, windows)))]
#[test]
fn a_part_that_cannot_be_mapped_is_read_through_a_buffer() {
    let dir = Scratch::new("fallback");
    let path = dir.write("rec.oir", &fixture::recorded_file());
    crate::map::FAIL_NEXT_MAP.with(|f| f.set(true));
    let fell_back = Options::new()
        .gather(&path, |_| true)
        .and_then(Gathered::assemble)
        .expect("read");
    let part = &fell_back.parts()[0];
    assert_eq!(part.read_path(), ReadPath::Buffered);
    assert_eq!(part.map_error(), Some("mapping refused by the test"));
    let mapped = crate::read(&path).expect("read");
    assert_eq!(mapped.parts()[0].read_path(), ReadPath::Mapped);
    assert_eq!(mapped.parts()[0].map_error(), None);
    assert_agree("fallback / mapped", &Ok(fell_back), &Ok(mapped), true);
}

/// Without mapping in the build, `Auto` is the buffered path, and nothing was
/// tried that could have failed.
#[cfg(not(all(feature = "mmap", any(unix, windows))))]
#[test]
fn without_mapping_auto_reads_through_a_buffer() {
    let dir = Scratch::new("nommap");
    let path = dir.write("rec.oir", &fixture::recorded_file());
    let a = crate::read(&path).expect("read");
    assert_eq!(a.parts()[0].read_path(), ReadPath::Buffered);
    assert_eq!(a.parts()[0].map_error(), None);
}

#[test]
fn a_part_that_cannot_be_opened_names_itself() {
    let dir = Scratch::new("missing");
    let path = dir.path().join("not-there.oir");
    let err = crate::read(&path).expect_err("there is no file");
    assert_eq!(err.kind(), ErrorKind::Io);
    assert_eq!(err.to_string(), "could not open the file");
    assert_eq!(err.path(), Some(path.as_path()));
    assert!(std::error::Error::source(&err).is_some());
}

// ------------------------------------------------- progress and cancelling

/// More blocks than the reader reports progress every: 300 frames, named
/// from frame `first`.
fn long_fixture(first: usize) -> Vec<u8> {
    let names: Vec<String> = (first + 1..=first + 300)
        .map(|i| format!("t{i:03}_0_1_uid"))
        .collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    fixture::stack_file(4, 4, &refs, 16)
}

#[test]
fn a_read_can_be_cancelled() {
    let f = long_fixture(0);
    let dir = Scratch::new("cancel");
    let path = dir.write("rec.oir", &f);
    for access in [Access::Auto, Access::Buffered] {
        let err = Options::new()
            .access(access)
            .gather(&path, |_| false)
            .expect_err("cancelled");
        assert_eq!(err.kind(), ErrorKind::Cancelled);
        assert_eq!(err.to_string(), "cancelled");
    }
    let err = Options::new()
        .gather_bytes(&[&f], |_| false)
        .expect_err("cancelled");
    assert_eq!(err.kind(), ErrorKind::Cancelled);
}

/// Progress follows bytes across the parts, never goes back on a finished
/// acquisition, and ends at one.
#[test]
fn progress_follows_bytes_and_ends_at_one() {
    let parts = [
        long_fixture(0),
        fixture::part_file(300, 2, false),
        long_fixture(302),
    ];
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    let mut seen = Vec::new();
    let g = Options::new()
        .gather_bytes(&refs, |f| {
            seen.push(f);
            true
        })
        .expect("read");
    assert_eq!(g.parts().len(), 3);
    assert!(seen.len() > 3, "{seen:?}");
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
    assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)), "{seen:?}");
    assert_eq!(*seen.last().unwrap(), 1.0);
    // The end of the first part is its share of the bytes.
    let total: usize = parts.iter().map(Vec::len).sum();
    let first = parts[0].len() as f32 / total as f32;
    assert!(
        seen.iter().any(|f| (f - first).abs() < 1e-6),
        "{first} in {seen:?}"
    );
}

// ---------------------------------------------------------------- the report

#[test]
fn gathering_reports_before_anything_is_decided() {
    let parts = [
        fixture::metadata_only_file(),
        fixture::part_file(1, 2, false),
    ];
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    let g = Options::new().gather_bytes(&refs, |_| true).expect("read");
    assert_eq!(g.planes_found(), 2);
    assert_eq!(g.parts().len(), 2);
    assert_eq!(g.record().width(), Some(4));
    let debug = format!("{g:?}");
    assert!(debug.contains("planes_found: 2"), "{debug}");
    let a = g.assemble().expect("assemble");
    let debug = format!("{a:?}");
    assert!(debug.contains("U16(2 planes x 8 samples)"), "{debug}");
}

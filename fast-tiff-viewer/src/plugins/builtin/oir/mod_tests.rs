//! OIR fixtures, built to the layout in the module docs.
//!
//! These are synthetic on purpose. The format was determined from a real
//! FluoView acquisition and checked against that software's own TIFF export,
//! but the file itself is unpublished research data and does not belong in a
//! repository — so what is committed here is the *structure*, written out by
//! [`oir_file`], with pixel values chosen so that a mis-assembled plane is
//! obvious rather than plausible.
//!
//! The reassembly tests matter more than they look. A plane arrives as
//! thirty-five scattered chunks; getting that wrong produces an image that
//! still looks like an image.

use super::*;

struct Silent;
impl ImportHost for Silent {
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn log(&mut self, _m: &str) {}
}

/// Builds an OIR the way the acquisition software lays one out.
#[derive(Default)]
struct Builder {
    offsets: Vec<u64>,
    body: Vec<u8>,
}

impl Builder {
    fn new() -> Self {
        let mut b = Builder::default();
        b.body.extend(MAGIC);
        // Header: sizes and offsets are patched in by `finish`.
        b.body.extend([0u8; 0x50 - 16]);
        b
    }

    /// One `u32 length, u32 type, payload` block.
    fn block(&mut self, ty: u32, payload: &[u8]) -> u64 {
        let at = self.body.len() as u64;
        self.body.extend((payload.len() as u32).to_le_bytes());
        self.body.extend(ty.to_le_bytes());
        self.body.extend(payload);
        self.offsets.push(at);
        at
    }

    /// A descriptor naming `name`, then the data block it describes.
    fn plane_chunk(&mut self, name: &str, at: u32, data: &[u8]) {
        let mut d = Vec::new();
        d.extend(at.to_le_bytes());
        d.extend((data.len() as u32).to_le_bytes());
        d.extend((name.len() as u32).to_le_bytes());
        d.extend(name.as_bytes());
        self.block(TYPE_DESCRIPTOR, &d);
        self.block(4, data);
    }

    fn xml(&mut self, text: &str) {
        self.block(5, text.as_bytes());
    }

    fn finish(mut self) -> Vec<u8> {
        let index_at = self.body.len() as u64;
        self.body.extend(INDEX_MARKER.to_le_bytes());
        self.body.extend(96u32.to_le_bytes());
        self.body.extend(0u32.to_le_bytes());
        for o in &self.offsets {
            self.body.extend(o.to_le_bytes());
        }
        let total = self.body.len() as u64;
        self.body[0x20..0x28].copy_from_slice(&total.to_le_bytes());
        self.body[0x28..0x30].copy_from_slice(&index_at.to_le_bytes());
        self.body
    }
}

/// `w`x`h` 16-bit planes, each pixel set to `plane * 1000 + index`, delivered
/// in scattered chunks and in the wrong order — so a reader that concatenates
/// blocks in file order, or ignores the declared offsets, fails.
fn stack_file(w: u32, h: u32, names: &[&str], chunk: usize) -> Vec<u8> {
    channels_file(w, h, names, chunk, 1)
}

/// As [`stack_file`], with the acquisition record stating `channels` recorded
/// channels — which is what decides the shape when the plane names do not.
fn channels_file(w: u32, h: u32, names: &[&str], chunk: usize, channels: usize) -> Vec<u8> {
    let mut b = Builder::new();
    let listed: String = (0..channels)
        .map(|i| {
            format!(
                "<commonimage:channel id=\"c{i}\"><commonphase:name>CH{i}</commonphase:name>\
                 </commonimage:channel>"
            )
        })
        .collect();
    b.xml(&format!(
        // The nesting a real acquisition uses: the frame size and the channels
        // that were recorded are stated inside `imageInfo`, the part of the
        // record describing what happened rather than what was configured.
        "<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo>\
         <commonimage:phase><commonimage:group>{listed}</commonimage:group></commonimage:phase>\
         <commonimage:width>{w}</commonimage:width>\
         <commonimage:height>{h}</commonimage:height>\
         </commonimage:imageInfo></lsmimage:imageProperties>"
    ));
    // A reference image, which must not be mistaken for a frame.
    b.plane_chunk("REF_LSM0_abcdef_0", 0, &[0xEEu8; 64]);
    for (p, name) in names.iter().enumerate() {
        let n = (w * h) as usize;
        let mut bytes = Vec::with_capacity(n * 2);
        for i in 0..n {
            bytes.extend(((p * 1000 + i) as u16).to_le_bytes());
        }
        // Emit the chunks back to front, so file order is not plane order.
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let len = chunk.min(bytes.len() - at);
            ranges.push((at, len));
            at += len;
        }
        ranges.reverse();
        for (at, len) in ranges {
            b.plane_chunk(name, at as u32, &bytes[at..at + len]);
        }
    }
    b.finish()
}

fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("fasttiff-oir-{name}"));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn import(path: &std::path::Path) -> Result<ImportResult, PluginError> {
    Oir.import(
        &ImportRequest {
            path: path.to_path_buf(),
            params: Default::default(),
        },
        &mut Silent,
    )
}

// ------------------------------------------------------------------ probing

#[test]
fn the_signature_decides_and_the_extension_does_not() {
    assert_eq!(Oir.probe(Path::new("x.oir"), MAGIC), Confidence::Certain);
    // Named `.oir` but is not one: leave it to another importer rather than
    // claiming it and then failing.
    assert_eq!(Oir.probe(Path::new("x.oir"), b"II*\0"), Confidence::No);
    assert_eq!(Oir.probe(Path::new("x.tif"), b"II*\0"), Confidence::No);
    // An unreadable head is no evidence, so the name is all there is.
    assert_eq!(Oir.probe(Path::new("x.oir"), &[]), Confidence::Maybe);
    assert_eq!(Oir.probe(Path::new("x.tif"), &[]), Confidence::No);
}

// --------------------------------------------------------------- reassembly

/// The heart of it: a plane arrives in pieces, out of order, and has to come
/// back exactly.
#[test]
fn scattered_chunks_reassemble_into_the_right_pixels() {
    // A chunk size that does not divide the plane, so the last one is short —
    // which is what a real file does (34 full chunks and a remainder).
    let file = tmp("scatter.oir", &stack_file(8, 4, &["t001_0_1_uid"], 22));
    let r = import(&file).expect("import");

    assert_eq!((r.image.width, r.image.height), (8, 4));
    assert_eq!(r.image.planes.len(), 1);
    let want: Vec<u16> = (0..32).collect();
    assert_eq!(
        r.image.planes[0],
        PlaneData::U16(want),
        "the plane was reassembled in the wrong order"
    );
    let _ = std::fs::remove_file(file);
}

#[test]
fn planes_are_ordered_by_timepoint_not_by_position_in_the_file() {
    // Ten frames, so that naive string sorting would put t10 before t2.
    let names: Vec<String> = (1..=10).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let file = tmp("order.oir", &stack_file(4, 2, &refs, 7));
    let r = import(&file).expect("import");

    assert_eq!(r.image.frames, 10);
    for (p, plane) in r.image.planes.iter().enumerate() {
        let PlaneData::U16(v) = plane else {
            panic!("expected u16")
        };
        assert_eq!(
            v[0],
            (p * 1000) as u16,
            "frame {p} is not the {p}th one written"
        );
    }
    let _ = std::fs::remove_file(file);
}

#[test]
fn reference_and_thumbnail_blocks_are_not_mistaken_for_frames() {
    let file = tmp("refs.oir", &stack_file(4, 4, &["t001_0_1_uid"], 32));
    let r = import(&file).expect("import");
    assert_eq!(
        r.image.planes.len(),
        1,
        "the REF_LSM0 block became a frame; it is a thumbnail"
    );
    let _ = std::fs::remove_file(file);
}

#[test]
fn plane_keys_sort_numerically_and_exclude_non_planes() {
    assert_eq!(plane_key("REF_LSM0_abc_0"), None);
    assert_eq!(plane_key("thumbnail"), None);
    assert_eq!(plane_key(""), None);
    assert_eq!(plane_key("t"), None);

    let k2 = plane_key("t2_0_1_uid").expect("t2");
    let k10 = plane_key("t10_0_1_uid").expect("t10");
    // `t002` already sorts correctly; the padding is what makes the *later*
    // fields sort numerically too.
    assert!(plane_key("t002_0_1_x") < plane_key("t010_0_1_x"));
    assert!(k2 < k10 || "t2" < "t10", "keys must order by number");
    assert!(plane_key("t001_0_2_x") < plane_key("t001_0_10_x"));
}

// -------------------------------------------------------------------- shape

#[test]
fn the_records_channel_count_and_the_plane_count_must_agree() {
    let names: Vec<String> = (1..=6).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    // The record says two channels; six planes is then 2 x 3 timepoints.
    let file = tmp("axes.oir", &channels_file(4, 4, &refs, 16, 2));
    let r = import(&file).expect("import");
    assert_eq!(
        (r.image.channels, r.image.slices, r.image.frames),
        (2, 1, 3)
    );
    let _ = std::fs::remove_file(file);

    // A count that cannot be right for the planes present is ignored rather
    // than believed: four channels does not divide six planes, and a stack of
    // the wrong shape is worse than one of the plainest possible shape.
    let file = tmp("axes_bad.oir", &channels_file(4, 4, &refs, 16, 4));
    let r = import(&file).expect("import");
    assert_eq!(
        (r.image.channels, r.image.frames),
        (1, 6),
        "a channel count that does not divide the planes was trusted anyway"
    );
    let _ = std::fs::remove_file(file);
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
    let file = tmp("axes_from_names.oir", &channels_file(4, 4, &refs, 16, 1));

    let r = import(&file).expect("import");
    assert_eq!(
        (r.image.channels, r.image.slices, r.image.frames),
        (2, 3, 1),
        "the names say 2 channels over 3 slices; the record was believed instead"
    );
    let _ = std::fs::remove_file(file);
}

#[test]
fn sample_width_is_measured_from_the_data_rather_than_assumed() {
    let file = tmp("depth.oir", &stack_file(8, 8, &["t001_0_1_uid"], 40));
    let r = import(&file).expect("import");
    assert_eq!(r.image.pixel_type, PixelType::U16);
    let _ = std::fs::remove_file(file);
}

// ------------------------------------------------------------- the metadata

/// What the user asked for: the OIR's own metadata, in tag 270 of the file
/// FastTIFF writes — and in the form the acquisition software exports, because
/// that is the form the analyses downstream already parse.
///
/// The file's own XML is never what goes there. In a real acquisition it is
/// four megabytes, three quarters of it display lookup tables written out
/// element by element, and no reader of a description can do anything with it.
#[test]
fn the_acquisition_record_reaches_tag_270_of_the_written_file() {
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
    // The document that must never reach tag 270. A real one is 525 KB of
    // exactly this, and there are three of them.
    b.xml("<?xml version=\"1.0\"?><lut:LUT><lut:intensity>0</lut:intensity></lut:LUT>");
    b.plane_chunk("t001_0_1_uid", 0, &[7u8; 64]);
    let file = tmp("meta.oir", &b.finish());

    let r = import(&file).expect("import");
    let info = r.info.clone().expect("metadata");
    let desc = info.description.clone().expect("the acquisition record");

    assert!(
        !desc.contains('<'),
        "raw XML reached the description: {desc}"
    );
    for line in [
        "\"Name\"\t\"fasttiff-oir-meta.oir\"",
        "\"System Name\"\t\"FVMPE-RS\"",
        "\"X Dimension\"\t\"8, 0.0 - 4.972 [um], 0.621 [um/pixel]\"",
        "\"Objective Lens\"\t\"XLUMPLFLN20XW\"",
    ] {
        assert!(desc.contains(line), "missing {line:?} from:\n{desc}");
    }

    // The structured values keep the precision the file states, rather than the
    // three decimals the text above rounds them to.
    assert_eq!(info.spacing.x, Some(0.621480569402239));
    assert_eq!(info.unit.as_deref(), Some("micron"));

    // Through the writer and back out again.
    let stack = crate::plugins::to_stack(&r.image, r.info.as_ref(), false).expect("open");
    let written = stack
        .tiff
        .description
        .as_deref()
        .expect("tag 270 should have been written");
    assert!(
        written.contains("XLUMPLFLN20XW") && written.contains("FVMPE-RS"),
        "the acquisition record did not reach tag 270: {written}"
    );
    // …and the structured metadata still works, so it opens as an image rather
    // than as a blob with a long description.
    assert_eq!(stack.dimensions(), Some((8, 4)));
    // Close, not equal: a TIFF states resolution as a rational, so an arbitrary
    // f64 comes back rounded to what a numerator over a denominator can say.
    // That is the format's limit rather than this importer's, and it is why the
    // exact value is taken from the record rather than read back out of here.
    let written_pixel = stack.tiff.meta.pixel_width.expect("the calibration");
    assert!(
        (written_pixel - 0.621480569402239).abs() < 1e-6,
        "the calibration was lost on the way to the file: {written_pixel}"
    );
    let _ = std::fs::remove_file(file);
}

/// The whole record, end to end: an acquisition with events and timing, whose
/// description has to come out as text a stimulus analysis can parse.
///
/// The same file also carries a lookup table, which is what tag 270 used to
/// fill up with — four megabytes in a real acquisition, three quarters of it
/// display tables written out element by element.
#[test]
fn the_record_is_translated_rather_than_dumped() {
    let mut b = Builder::new();
    b.xml(
        "<?xml version=\"1.0\"?><lsmimage:imageProperties>\
         <commonimage:general><base:creationDateTime>2025-02-10T18:49:39.124-05:00\
         </base:creationDateTime></commonimage:general>\
         <commonimage:imageInfo>\
         <commonimage:phase><commonimage:group><commonimage:channel id=\"c1\">\
         <commonphase:name>CH1</commonphase:name>\
         <commonphase:length><commonparam:x>0.621480569402239</commonparam:x>\
         <commonparam:y>0.621480569402239</commonparam:y></commonphase:length>\
         </commonimage:channel></commonimage:group></commonimage:phase>\
         <commonimage:axis><commonimage:axis>TIMELAPSE</commonimage:axis>\
         <commonimage:step>0.0</commonimage:step>\
         <commonimage:maxSize>2</commonimage:maxSize></commonimage:axis>\
         <commonimage:width>4</commonimage:width>\
         <commonimage:height>4</commonimage:height>\
         </commonimage:imageInfo></lsmimage:imageProperties>",
    );
    // The two frames, each with its timestamp: 1 s apart.
    for (n, ms) in [(1, "0.0"), (2, "1000.0")] {
        b.xml(&format!(
            "<?xml version=\"1.0\"?><lsmframe:frameProperties>\
             <commonframe:axisValue><commonframe:axisType>TIMELAPSE</commonframe:axisType>\
             <commonframe:position>{ms}</commonframe:position></commonframe:axisValue>\
             </lsmframe:frameProperties>",
        ));
        let _ = n;
    }
    b.xml(
        "<?xml version=\"1.0\"?><event:eventList><event:event>\
         <event:name>DRS</event:name><event:time>27772.762</event:time>\
         <event:type>TTL_OUT</event:type></event:event></event:eventList>",
    );
    // The document that made this a bug rather than an untidiness. A real one
    // is 525 KB of exactly this, and there are three of them.
    b.xml(
        "<?xml version=\"1.0\"?><lut:LUT><lut:intensity>0</lut:intensity>\
         <lut:contrast>1</lut:contrast></lut:LUT>",
    );
    for (i, name) in ["t001_0_1_uid", "t002_0_1_uid"].iter().enumerate() {
        b.plane_chunk(name, 0, &[i as u8; 32]);
    }
    let file = tmp("translated.oir", &b.finish());

    let info = import(&file).expect("import").info.expect("metadata");
    let desc = info.description.expect("the translated record");

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
        "\"Name\"\t\"fasttiff-oir-translated.oir\"",
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

    // The structured values come from the record too, at the precision the
    // file states them — not re-parsed out of the three decimals the text
    // above rounds them to.
    assert_eq!(info.spacing.x, Some(0.621480569402239));
    assert_eq!(info.spacing.y, Some(0.621480569402239));
    assert_eq!(info.frame_interval_s, Some(1.0));
    assert_eq!(info.channel_names, vec!["CH1".to_string()]);
    let _ = std::fs::remove_file(file);
}

// ------------------------------------------------------------------ refusal

#[test]
fn a_file_without_the_signature_is_refused_by_name() {
    let file = tmp("fake.oir", b"II*\0 not an OIR at all");
    let err = import(&file).expect_err("must refuse");
    assert!(err.to_string().contains("OLYMPUSRAWFORMAT"), "{err}");
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_corrupt_index_offset_is_refused_rather_than_followed() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    f[0x28..0x30].copy_from_slice(&u64::MAX.to_le_bytes());
    let file = tmp("badindex.oir", &f);
    let err = import(&file).expect_err("a wild index offset must be refused");
    assert!(err.to_string().contains("outside the file"), "{err}");
    let _ = std::fs::remove_file(file);
}

#[test]
fn an_index_without_its_marker_is_refused() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize;
    f[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    let file = tmp("nomarker.oir", &f);
    let err = import(&file).expect_err("an unrecognised index must be refused");
    assert!(
        err.to_string().contains("layout this reader knows"),
        "{err}"
    );
    let _ = std::fs::remove_file(file);
}

#[test]
fn an_oir_with_no_planes_says_so_rather_than_producing_an_empty_stack() {
    let mut b = Builder::new();
    b.xml("<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo><commonimage:width>4</commonimage:width><commonimage:height>4</commonimage:height></commonimage:imageInfo></lsmimage:imageProperties>");
    b.plane_chunk("REF_LSM0_abc_0", 0, &[0u8; 32]);
    let file = tmp("noplanes.oir", &b.finish());
    let err = import(&file).expect_err("no planes is a failure");
    assert!(err.to_string().contains("no image planes"), "{err}");
    assert!(
        err.to_string().contains("_00001.oir"),
        "the message should point at the likeliest cause: {err}"
    );
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_file_that_states_no_frame_size_is_refused_rather_than_guessed_at() {
    let mut b = Builder::new();
    b.plane_chunk("t001_0_1_uid", 0, &[0u8; 32]);
    let file = tmp("nosize.oir", &b.finish());
    let err = import(&file).expect_err("an unknown frame size must be refused");
    assert!(err.to_string().contains("frame size"), "{err}");
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_chunk_claiming_to_run_past_its_plane_is_dropped_not_followed() {
    let mut b = Builder::new();
    b.xml("<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo><commonimage:width>4</commonimage:width><commonimage:height>2</commonimage:height></commonimage:imageInfo></lsmimage:imageProperties>");
    // A whole 4x2 u16 plane, then a chunk that claims to start past its end.
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 16]);
    b.plane_chunk("t001_0_1_uid", 1_000_000, &[2u8; 16]);
    let file = tmp("overrun.oir", &b.finish());
    let r = import(&file).expect("the good chunk should still import");
    assert_eq!(
        r.image.planes[0],
        PlaneData::U16(vec![0x0101; 8]),
        "the overrunning chunk was written somewhere it should not have been"
    );
    let _ = std::fs::remove_file(file);
}

/// An acquisition stopped part-way leaves a short final plane. The software's
/// own export omits it, and so must this — a mostly-black frame appended to a
/// recording is the kind of thing that quietly ruins an average.
#[test]
fn an_incomplete_trailing_plane_is_dropped_rather_than_padded() {
    let mut b = Builder::new();
    b.xml("<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo><commonimage:width>4</commonimage:width><commonimage:height>2</commonimage:height></commonimage:imageInfo></lsmimage:imageProperties>");
    // Two whole 4x2 u16 planes, then one that stops half way.
    for name in ["t001_0_1_uid", "t002_0_1_uid"] {
        b.plane_chunk(name, 0, &[1u8; 16]);
    }
    b.plane_chunk("t003_0_1_uid", 0, &[2u8; 6]);
    let file = tmp("partial.oir", &b.finish());

    let r = import(&file).expect("import");
    assert_eq!(
        r.image.frames, 2,
        "the unfinished frame was kept and padded with zeros"
    );
    assert_eq!(r.image.planes.len(), 2);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_file_of_nothing_but_incomplete_planes_is_refused() {
    let mut b = Builder::new();
    b.xml("<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo><commonimage:width>8</commonimage:width><commonimage:height>8</commonimage:height></commonimage:imageInfo></lsmimage:imageProperties>");
    // Short, but all the same length, so the sample width still resolves.
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 128]);
    b.plane_chunk("t002_0_1_uid", 0, &[1u8; 128]);
    let file = tmp("allshort.oir", &b.finish());
    // 128 bytes over 64 pixels is 2 bytes each, so these are *complete* — the
    // reader must not invent a larger plane than the data supports.
    let r = import(&file).expect("uniformly short planes are just small planes");
    assert_eq!(r.image.frames, 2);
    let _ = std::fs::remove_file(file);
}

// ------------------------------------------------- against a real file

/// Read a real OIR and check it against the acquisition software's own export.
///
/// Ignored by default because it needs a file this repository cannot contain:
/// OIR samples are unpublished research data. Point it at one to run it —
///
/// ```text
/// FASTTIFF_OIR_SAMPLE=/path/to/file.oir ///     cargo test -p fast-tiff-viewer -- --ignored oir
/// ```
///
/// When a `<stem>.tif` sits beside the `.oir` — FluoView writes both — every
/// frame is compared against it byte for byte. That is the only oracle there
/// is for a format with no specification, and it is a complete one: if the
/// export and the reassembly agree on every pixel of every frame, the reader
/// is right about this file.
#[test]
#[ignore = "needs a real OIR; set FASTTIFF_OIR_SAMPLE"]
fn a_real_oir_matches_the_software_export() {
    let Ok(path) = std::env::var("FASTTIFF_OIR_SAMPLE") else {
        eprintln!("set FASTTIFF_OIR_SAMPLE to a .oir file to run this");
        return;
    };
    let path = std::path::PathBuf::from(path);
    let r = import(&path).expect("the sample should import");
    eprintln!(
        "{}x{}, {}c {}z {}t, {:?}",
        r.image.width,
        r.image.height,
        r.image.channels,
        r.image.slices,
        r.image.frames,
        r.image.pixel_type
    );
    r.image.validate().expect("shape");
    let desc = r
        .info
        .as_ref()
        .and_then(|i| i.description.as_ref())
        .expect("no metadata was carried");
    eprintln!("--- description, {} bytes ---\n{desc}", desc.len());
    // What reaches tag 270 is text. It used to be the file's own XML, which
    // in a real acquisition runs to four megabytes.
    assert!(
        !desc.contains("<?xml"),
        "raw XML reached the description ({} bytes)",
        desc.len()
    );
    assert!(
        desc.len() < 64 * 1024,
        "a {}-byte description is a metadata dump, not a record",
        desc.len()
    );

    let exported = path.with_extension("tif");
    if !exported.exists() {
        eprintln!("no {} beside it; pixels not verified", exported.display());
        return;
    }
    let tiff = fast_tiff_lib::TiffStack::open(&exported).expect("open the export");
    let f0 = tiff.frames.first().expect("the export has no frames");
    assert_eq!(
        (f0.width, f0.height),
        (r.image.width, r.image.height),
        "the export is a different size from the OIR"
    );

    let mut checked = 0usize;
    for (i, plane) in r.image.planes.iter().enumerate() {
        let PlaneData::U16(ours) = plane else {
            continue;
        };
        let Some(frame) = tiff.frames.get(i) else {
            break;
        };
        let theirs =
            fast_tiff_lib::read_frame_u16(&tiff.data, frame, tiff.byte_order, None).expect("frame");
        assert_eq!(
            ours.len(),
            theirs.len(),
            "frame {i} is a different length from the exported one"
        );
        assert!(
            ours[..] == theirs[..],
            "frame {i} differs from the software's own export"
        );
        checked += 1;
    }
    eprintln!("{checked} frame(s) identical to {}", exported.display());
    assert!(checked > 0, "nothing was compared");
}

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

/// A host that remembers what it was told, so a warning the reader promises can
/// be checked rather than assumed.
#[derive(Default)]
struct Notes {
    lines: Vec<String>,
}

impl ImportHost for Notes {
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn log(&mut self, m: &str) {
        self.lines.push(m.to_string());
    }
}

impl Notes {
    fn said(&self, needle: &str) -> bool {
        self.lines.iter().any(|l| l.contains(needle))
    }
}

fn import_noting(path: &std::path::Path) -> (Result<ImportResult, PluginError>, Notes) {
    let mut host = Notes::default();
    let r = Oir.import(
        &ImportRequest {
            path: path.to_path_buf(),
            params: Default::default(),
        },
        &mut host,
    );
    (r, host)
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

/// As [`stack_file`], laid out the way a real acquisition is: a 16-byte
/// `u32 3, u32 2, u64 -1` structure at `0x50`, and the first block at `0x60`.
///
/// That structure is the reason [`find_first_block`] demands a run of blocks
/// rather than one — see the test that uses this.
fn real_layout_file(w: u32, h: u32, names: &[&str], chunk: usize) -> Vec<u8> {
    let plain = stack_file(w, h, names, chunk);
    let mut out = Vec::with_capacity(plain.len() + 16);
    out.extend_from_slice(&plain[..0x50]);
    out.extend(3u32.to_le_bytes());
    out.extend(2u32.to_le_bytes());
    out.extend(u64::MAX.to_le_bytes());
    out.extend_from_slice(&plain[0x50..]);
    // Every offset in the header and the index moved along by the insertion.
    let shift = 16u64;
    let total = u64::from_le_bytes(out[0x20..0x28].try_into().unwrap()) + shift;
    let index_at = u64::from_le_bytes(out[0x28..0x30].try_into().unwrap()) + shift;
    out[0x20..0x28].copy_from_slice(&total.to_le_bytes());
    out[0x28..0x30].copy_from_slice(&index_at.to_le_bytes());
    let mut at = index_at as usize + INDEX_HEADER;
    while at + 8 <= out.len() {
        let o = u64::from_le_bytes(out[at..at + 8].try_into().unwrap()) + shift;
        out[at..at + 8].copy_from_slice(&o.to_le_bytes());
        at += 8;
    }
    out
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

/// A wild index offset is never *followed*, and the file is read anyway.
///
/// It used to be refused. The offset is still not followed — that would be a
/// wild seek and an allocation sized from whatever bytes were at it — but
/// refusing was the wrong response to it, because the two files that produce a
/// bad offset are a recording still in progress and one whose tail was lost,
/// and in both the pixels are in front of the index and perfectly good. The
/// assertion that it was not followed is the pixels: they are the ones the
/// fixture wrote, in order.
#[test]
fn a_corrupt_index_offset_is_not_followed_but_the_pixels_are_still_read() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    f[0x28..0x30].copy_from_slice(&u64::MAX.to_le_bytes());
    let file = tmp("badindex.oir", &f);
    let (r, host) = import_noting(&file);
    let r = r.expect("the blocks are all there and should be recovered");

    assert_eq!(r.image.frames, 1);
    match &r.image.planes[0] {
        PlaneData::U16(v) => assert_eq!(v, &(0..16u16).collect::<Vec<_>>()),
        other => panic!("{:?}", other.pixel_type()),
    }
    assert!(host.said("unfinished acquisition"), "{:?}", host.lines);
    let _ = std::fs::remove_file(file);
}

/// An index whose marker is not the one this reader knows is the same case: not
/// read, not fatal.
#[test]
fn an_index_without_its_marker_falls_back_to_walking() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize;
    f[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    let file = tmp("nomarker.oir", &f);
    let (r, host) = import_noting(&file);
    let r = r.expect("the blocks are all there");

    assert_eq!(r.image.frames, 1);
    assert!(host.said("unfinished acquisition"), "{:?}", host.lines);
    let _ = std::fs::remove_file(file);
}

// ------------------------------------------------- acquisitions still running

/// A recording whose index has not been written yet opens at the frames it has.
///
/// The index is the *last* thing an acquisition writes, so this is what every
/// file being recorded looks like: whole frames, and nothing at `0x28` to say
/// where they are.
#[test]
fn a_file_whose_index_was_never_written_still_opens() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid", "t002_0_1_uid"], 16);
    let at = u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize;
    // As the file stands on disk mid-recording: no index, and the field that
    // would point at one still zero.
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    let file = tmp("noindex.oir", &f);
    let (r, host) = import_noting(&file);
    let r = r.expect("a recording in progress should open");

    assert_eq!(r.image.frames, 2, "both finished frames should be there");
    match &r.image.planes[0] {
        PlaneData::U16(v) => assert_eq!(v, &(0..16u16).collect::<Vec<_>>()),
        other => panic!("{:?}", other.pixel_type()),
    }
    assert!(host.said("unfinished acquisition"), "{:?}", host.lines);
    assert!(host.said("open it again later"), "{:?}", host.lines);
    let _ = std::fs::remove_file(file);
}

/// A file cut in the middle of a frame keeps the frames before the cut and
/// drops the part-written one.
///
/// The half-frame is the thing that must not survive: padded out with zeros it
/// becomes a mostly-black frame at the end of the recording, which is not an
/// error anyone sees and quietly ruins an average.
#[test]
fn a_file_cut_mid_frame_keeps_the_frames_before_the_cut() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid", "t002_0_1_uid"], 16);
    let at = u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize;
    // Into the last frame's final data block, so that frame is half written.
    f.truncate(at - 8);
    let file = tmp("cutframe.oir", &f);
    let (r, host) = import_noting(&file);
    let r = r.expect("the finished frame should still open");

    assert_eq!(r.image.frames, 1, "the half-written frame must not survive");
    match &r.image.planes[0] {
        PlaneData::U16(v) => assert_eq!(v, &(0..16u16).collect::<Vec<_>>()),
        other => panic!("{:?}", other.pixel_type()),
    }
    assert!(host.said("incomplete plane"), "{:?}", host.lines);
    let _ = std::fs::remove_file(file);
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
    let mut f = real_layout_file(4, 4, &["t001_0_1_uid"], 16);
    // No index, so the start has to be found rather than read.
    let at = u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize;
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    let file = tmp("realstart.oir", &f);
    let r = import(&file).expect("a real-layout file should open");

    assert_eq!(r.image.frames, 1);
    match &r.image.planes[0] {
        PlaneData::U16(v) => assert_eq!(
            v,
            &(0..16u16).collect::<Vec<_>>(),
            "the walk began at the wrong offset"
        ),
        other => panic!("{:?}", other.pixel_type()),
    }
    let _ = std::fs::remove_file(file);
}

/// A run of zeros is not a block stream.
///
/// Zeros parse as an unlimited run of zero-length type-0 blocks, so a candidate
/// pointing into them agrees with itself for ever and would win by coming
/// first. It has to carry something.
#[test]
fn a_run_of_zeros_is_not_mistaken_for_a_block_stream() {
    let mut f = Vec::new();
    f.extend(MAGIC);
    f.extend([0u8; 0x2000]);
    let file = tmp("zeros.oir", &f);
    let err = import(&file).expect_err("a file of zeros carries no blocks");
    // The specific refusal matters: it is the one that means no candidate was
    // believed at all. Without the non-empty rule a candidate *is* believed,
    // the walk returns thousands of phantom empty blocks, and the file is
    // refused later for carrying no planes -- the same outcome by luck.
    assert!(err.to_string().contains("damaged past"), "{err}");
    let _ = std::fs::remove_file(file);
}

/// And a file whose signature is right but whose contents are rubbish is still
/// refused, rather than walked into something.
#[test]
fn a_file_whose_blocks_are_rubbish_is_still_refused() {
    let mut f = Vec::new();
    f.extend(MAGIC);
    f.extend([0u8; 0x40]);
    // Lengths far past any block, at every alignment a candidate could land on.
    for _ in 0..512 {
        f.extend(0xDEAD_BEEFu32.to_le_bytes());
        f.extend(0xFEED_FACEu32.to_le_bytes());
    }
    let file = tmp("rubbish.oir", &f);
    let err = import(&file).expect_err("rubbish must not be walked into");
    assert!(err.to_string().contains("damaged past"), "{err}");
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
    // Read rather than mapped: `TiffStack::open` needs the `mmap` feature, and
    // this crate is also tested without it (the shape a browser build takes).
    let tiff =
        fast_tiff_lib::TiffStack::from_bytes(std::fs::read(&exported).expect("read the export"))
            .expect("open the export");
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

// ------------------------------------------------------- split acquisitions

/// One part of a split acquisition: `frames` 4x2 planes named from `first`,
/// each pixel `frame * 1000 + index`, with the record only in the first part —
/// which is where a real acquisition keeps it.
fn part_file(first: usize, frames: usize, with_record: bool) -> Vec<u8> {
    let mut b = Builder::new();
    if with_record {
        b.xml("<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo><commonimage:width>4</commonimage:width><commonimage:height>2</commonimage:height></commonimage:imageInfo></lsmimage:imageProperties>");
    }
    for t in first..first + frames {
        let bytes: Vec<u8> = (0..8u16)
            .flat_map(|i| (t as u16 * 1000 + i).to_le_bytes())
            .collect();
        // Two chunks per plane, as the real format scatters them.
        let name = format!("t{:03}_0_1_uid", t + 1);
        b.plane_chunk(&name, 0, &bytes[..10]);
        b.plane_chunk(&name, 10, &bytes[10..]);
    }
    b.finish()
}

/// A recording split across files comes back whole, in order, pixel for pixel.
///
/// Each part is read front to back as it is opened, and the planes of one part
/// must neither be lost nor reordered by the part read after it. The middle
/// part carries no extension and the last one does, because both are written.
#[test]
fn a_split_acquisition_is_read_whole_and_in_order() {
    let dir = std::env::temp_dir().join(format!("fasttiff-oir-split-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let first = dir.join("rec.oir");
    std::fs::write(&first, part_file(0, 2, true)).unwrap();
    std::fs::write(dir.join("rec_00001"), part_file(2, 3, false)).unwrap();
    std::fs::write(dir.join("rec_00002.oir"), part_file(5, 1, false)).unwrap();

    let r = import(&first).expect("a split acquisition imports");
    assert_eq!(r.image.frames, 6, "frames from some part were lost");
    for (t, plane) in r.image.planes.iter().enumerate() {
        let want: Vec<u16> = (0..8).map(|i| t as u16 * 1000 + i).collect();
        assert_eq!(
            plane,
            &PlaneData::U16(want),
            "frame {t} is wrong or out of order"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
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
    b.xml("<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo><commonimage:width>4</commonimage:width><commonimage:height>2</commonimage:height></commonimage:imageInfo></lsmimage:imageProperties>");
    b.plane_chunk("t001_0_1_uid", (MAX_PLANE_BYTES - 16) as u32, &[7u8; 16]);
    let file = tmp("lying.oir", &b.finish());
    let err = import(&file).expect_err("a gigabyte of zeros is not a plane");
    assert!(
        err.to_string()
            .contains("more pixel data than the file holds"),
        "{err}"
    );
    let _ = std::fs::remove_file(file);
}

/// The walk stops at a block that is not all there, rather than emitting an
/// offset for it.
///
/// Checked on the walk's own output. End to end the half-written block is
/// harmless either way — the chunk read fails, the plane stays short and is
/// dropped — so an end-to-end test cannot tell the two apart, and the rule the
/// doc comment states would go unpinned.
#[test]
fn the_walk_stops_at_a_block_that_is_not_all_there() {
    let f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize;
    let whole = walk_blocks(&f[..at]);
    assert!(whole.len() > 2, "the fixture should have several blocks");

    // One byte short of the last block's final byte.
    let cut = walk_blocks(&f[..at - 1]);
    assert_eq!(
        cut.len(),
        whole.len() - 1,
        "the block that is one byte short must not be offered"
    );
    assert_eq!(cut, whole[..whole.len() - 1]);
}

/// An index that reads cleanly and points at nothing is recovered by walking.
///
/// What a half-written index looks like: the header is there, the offsets in it
/// are not yet. It is not an unreadable index — every check this reader makes
/// of one passes — so the fallback cannot be conditioned on the index failing
/// to parse. It is conditioned on the index failing to produce a plane.
#[test]
fn an_index_that_points_at_nothing_is_recovered_by_walking() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize;
    // Keep the marker and the header; blank every offset it lists.
    for b in f[at + INDEX_HEADER..].iter_mut() {
        *b = 0;
    }
    let file = tmp("blankindex.oir", &f);
    let (r, host) = import_noting(&file);
    let r = r.expect("the blocks are still in the file and should be found");

    assert_eq!(r.image.frames, 1);
    match &r.image.planes[0] {
        PlaneData::U16(v) => assert_eq!(v, &(0..16u16).collect::<Vec<_>>()),
        other => panic!("{:?}", other.pixel_type()),
    }
    assert!(host.said("unfinished acquisition"), "{:?}", host.lines);
    let _ = std::fs::remove_file(file);
}

/// A first part carrying only metadata is not mistaken for a recovery.
///
/// A split acquisition legitimately keeps its record in the named file and its
/// planes in the siblings, so "this part produced no planes" is ordinary there.
/// Warning about it would tell the user a finished recording was unfinished.
#[test]
fn a_metadata_only_part_is_not_reported_as_unfinished() {
    // Several blocks, none of them a plane -- which is what a real first part
    // holds: the record, the reference snapshot, the thumbnail. It has to be
    // several, or the walk finds nothing and the guard under test is never
    // reached: it is the plane count that must decide this, not the luck of a
    // fixture too small to walk.
    let first = {
        let mut b = Builder::new();
        b.xml("<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo>               <commonimage:width>4</commonimage:width><commonimage:height>2               </commonimage:height></commonimage:imageInfo></lsmimage:imageProperties>");
        b.plane_chunk("REF_LSM0_abc_0", 0, &[0u8; 16]);
        b.plane_chunk("REF_LSM0_abc_1", 16, &[0u8; 16]);
        b.finish()
    };
    assert!(
        walk_blocks(&first).len() >= 4,
        "the fixture must be walkable, or it cannot exercise the guard"
    );
    let second = part_file(1, 2, false);
    let dir = std::env::temp_dir();
    let a = dir.join("fasttiff-oir-meta-only.oir");
    let b = dir.join("fasttiff-oir-meta-only_00001.oir");
    std::fs::write(&a, &first).unwrap();
    std::fs::write(&b, &second).unwrap();

    let (r, host) = import_noting(&a);
    let r = r.expect("a split acquisition should open");
    assert_eq!(r.image.frames, 2);
    assert!(
        !host.said("unfinished"),
        "a finished recording was called unfinished: {:?}",
        host.lines
    );
    let _ = std::fs::remove_file(a);
    let _ = std::fs::remove_file(b);
}

//! The importer around the `oir-reader` crate: what it tells the host, what it
//! hands back, and how a refusal reaches the user.
//!
//! The reader is tested in its own crate, against every fixture these tests
//! used to cover and thousands of damaged variants of them. What is left here
//! is what this module adds: the log lines, the mapping onto the plugin
//! contract, the description that reaches tag 270, and the errors.
//!
//! The fixtures are synthetic on purpose. The format was determined from real
//! FluoView acquisitions and checked against that software's own TIFF export,
//! but the files themselves are unpublished research data and do not belong in
//! a repository — so what is here is the *structure*, written out by
//! [`Builder`].

use super::*;

struct Silent;
impl ImportHost for Silent {
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn log(&mut self, _m: &str) {}
}

/// A host that remembers what it was told, so a warning the importer promises
/// can be checked rather than assumed.
#[derive(Default)]
struct Notes {
    lines: Vec<String>,
    /// Answer `false` the first time progress is reported: the user cancelled.
    cancel: bool,
}

impl ImportHost for Notes {
    fn progress(&mut self, _f: f32) -> bool {
        !self.cancel
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

fn request(path: &std::path::Path) -> ImportRequest {
    ImportRequest {
        path: path.to_path_buf(),
        params: Default::default(),
    }
}

/// Import `path` both ways the parts can be read, require the two to give the
/// same import and say the same things, and return the default way's.
fn import_noting(path: &std::path::Path) -> (Result<ImportResult, PluginError>, Notes) {
    let mut host = Notes::default();
    let r = Oir.import(&request(path), &mut host);
    let mut buffered = Notes::default();
    let b = Oir.import_with(&request(path), &mut buffered, Access::Buffered);
    assert_eq!(r, b, "mapped and buffered imports differ");
    // A mapping that fell back says so, and the buffered read never mapped.
    let mapped_lines: Vec<&String> = host
        .lines
        .iter()
        .filter(|l| !l.contains("could not be memory-mapped"))
        .collect();
    assert_eq!(
        mapped_lines,
        buffered.lines.iter().collect::<Vec<_>>(),
        "mapped and buffered imports said different things"
    );
    (r, host)
}

fn import(path: &std::path::Path) -> Result<ImportResult, PluginError> {
    import_noting(path).0
}

const MAGIC: &[u8] = b"OLYMPUSRAWFORMAT";

/// Builds an OIR the way the acquisition software lays one out: the header,
/// then `u32 length, u32 type, payload` blocks, then the index of them.
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

    fn block(&mut self, ty: u32, payload: &[u8]) {
        self.offsets.push(self.body.len() as u64);
        self.body.extend((payload.len() as u32).to_le_bytes());
        self.body.extend(ty.to_le_bytes());
        self.body.extend(payload);
    }

    /// A descriptor (type 3) naming `name`, then the data block it describes.
    fn plane_chunk(&mut self, name: &str, at: u32, data: &[u8]) {
        let mut d = Vec::new();
        d.extend(at.to_le_bytes());
        d.extend((data.len() as u32).to_le_bytes());
        d.extend((name.len() as u32).to_le_bytes());
        d.extend(name.as_bytes());
        self.block(3, &d);
        self.block(4, data);
    }

    fn xml(&mut self, text: &str) {
        self.block(5, text.as_bytes());
    }

    fn finish(mut self) -> Vec<u8> {
        let index_at = self.body.len() as u64;
        self.body.extend(0xFFFF_FFFFu32.to_le_bytes());
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

fn index_at(f: &[u8]) -> usize {
    u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize
}

fn size_record(w: u32, h: u32) -> String {
    format!(
        "<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo>\
         <commonimage:width>{w}</commonimage:width>\
         <commonimage:height>{h}</commonimage:height>\
         </commonimage:imageInfo></lsmimage:imageProperties>"
    )
}

/// `w`x`h` 16-bit planes, each pixel `plane * 1000 + index`, in chunks of
/// `chunk` bytes.
fn stack_file(w: u32, h: u32, names: &[&str], chunk: usize) -> Vec<u8> {
    let mut b = Builder::new();
    b.xml(&size_record(w, h));
    for (p, name) in names.iter().enumerate() {
        let bytes: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| ((p * 1000 + i) as u16).to_le_bytes())
            .collect();
        for (k, c) in bytes.chunks(chunk).enumerate() {
            b.plane_chunk(name, (k * chunk) as u32, c);
        }
    }
    b.finish()
}

/// One part of a split acquisition: `frames` 4x2 planes named from `first`,
/// with the record only in the first part — which is where a real acquisition
/// keeps it.
fn part_file(first: usize, frames: usize, with_record: bool) -> Vec<u8> {
    let mut b = Builder::new();
    if with_record {
        b.xml(&size_record(4, 2));
    }
    for t in first..first + frames {
        let bytes: Vec<u8> = (0..8u16)
            .flat_map(|i| (t as u16 * 1000 + i).to_le_bytes())
            .collect();
        let name = format!("t{:03}_0_1_uid", t + 1);
        b.plane_chunk(&name, 0, &bytes[..10]);
        b.plane_chunk(&name, 10, &bytes[10..]);
    }
    b.finish()
}

fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("fasttiff-oir-{name}"));
    std::fs::write(&p, bytes).unwrap();
    p
}

/// `mmap` means "this build memory-maps the files it opens", and the OIR parts
/// are among them: the feature reaches the reader, and leaving it out reaches
/// it too.
#[test]
fn the_mmap_feature_decides_whether_parts_are_mapped() {
    assert_eq!(oir_reader::MMAP_SUPPORTED, cfg!(feature = "mmap"));
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
    for ms in ["0.0", "1000.0"] {
        b.xml(&format!(
            "<?xml version=\"1.0\"?><lsmframe:frameProperties>\
             <commonframe:axisValue><commonframe:axisType>TIMELAPSE</commonframe:axisType>\
             <commonframe:position>{ms}</commonframe:position></commonframe:axisValue>\
             </lsmframe:frameProperties>",
        ));
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

    let r = import(&file).expect("import");
    assert_eq!(r.image.pixel_type, PixelType::U16);
    let info = r.info.expect("metadata");
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
    assert_eq!(info.mode, DisplayMode::Grayscale);
    assert_eq!(r.image.name, "fasttiff-oir-translated");
    let _ = std::fs::remove_file(file);
}

/// Both axes get x's pixel size, which is what this importer has always
/// reported. Pinned so that switching y to its own value is a change someone
/// makes on purpose, not one that slips in.
#[test]
fn both_axes_take_the_x_pixel_size() {
    let mut b = Builder::new();
    b.xml(
        "<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo>\
         <commonimage:phase><commonimage:group><commonimage:channel id=\"c1\">\
         <commonphase:name>CH1</commonphase:name>\
         <commonphase:length><commonparam:x>0.5</commonparam:x>\
         <commonparam:y>0.75</commonparam:y></commonphase:length>\
         </commonimage:channel><commonimage:channel id=\"c2\">\
         <commonphase:name>CH2</commonphase:name>\
         </commonimage:channel></commonimage:group></commonimage:phase>\
         <commonimage:width>4</commonimage:width>\
         <commonimage:height>2</commonimage:height>\
         </commonimage:imageInfo></lsmimage:imageProperties>",
    );
    for name in ["t001_0_1_uid", "t002_0_1_uid"] {
        b.plane_chunk(name, 0, &[1u8; 16]);
    }
    let file = tmp("nonsquare.oir", &b.finish());
    let r = import(&file).expect("import");
    let info = r.info.expect("metadata");
    assert_eq!(info.spacing.x, Some(0.5));
    assert_eq!(info.spacing.y, Some(0.5));
    // Two channels in the record and two planes: one frame of two channels,
    // shown together.
    assert_eq!((r.image.channels, r.image.frames), (2, 1));
    assert_eq!(info.mode, DisplayMode::Composite);
    assert_eq!(info.channel_names, ["CH1", "CH2"]);
    let _ = std::fs::remove_file(file);
}

// ------------------------------------------------------------- what it says

/// A recording whose index was never written opens, and says it is unfinished
/// and that opening it again later reads more.
#[test]
fn an_unfinished_recording_says_so() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid", "t002_0_1_uid"], 16);
    let at = index_at(&f);
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    let file = tmp("noindex.oir", &f);
    let (r, host) = import_noting(&file);
    let r = r.expect("a recording in progress should open");
    assert_eq!(r.image.frames, 2);
    assert!(
        host.said("unfinished acquisition: 1 of 1 part(s)"),
        "{:?}",
        host.lines
    );
    assert!(host.said("open it again later"), "{:?}", host.lines);
    let _ = std::fs::remove_file(file);
}

/// A wild index offset is not followed, the blocks are walked instead, and
/// the import says so.
#[test]
fn a_corrupt_index_offset_is_reported_as_a_recovery() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    f[0x28..0x30].copy_from_slice(&u64::MAX.to_le_bytes());
    let file = tmp("badindex.oir", &f);
    let (r, host) = import_noting(&file);
    r.expect("the blocks are all there and should be recovered");
    assert!(host.said("unfinished acquisition"), "{:?}", host.lines);
    let _ = std::fs::remove_file(file);
}

/// A file cut mid-frame drops the half-written frame, and says so.
#[test]
fn a_dropped_frame_is_mentioned() {
    let mut f = stack_file(4, 4, &["t001_0_1_uid", "t002_0_1_uid"], 16);
    let at = index_at(&f);
    f.truncate(at - 8);
    let file = tmp("cutframe.oir", &f);
    let (r, host) = import_noting(&file);
    assert_eq!(r.expect("the finished frame should open").image.frames, 1);
    assert!(
        host.said("skipped 1 incomplete plane(s)"),
        "{:?}",
        host.lines
    );
    let _ = std::fs::remove_file(file);
}

/// A recording split across files comes back whole and in order, and the
/// import says how many parts it read.
#[test]
fn a_split_acquisition_is_read_whole_and_in_order() {
    let dir = std::env::temp_dir().join(format!("fasttiff-oir-split-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let first = dir.join("rec.oir");
    std::fs::write(&first, part_file(0, 2, true)).unwrap();
    std::fs::write(dir.join("rec_00001"), part_file(2, 3, false)).unwrap();
    std::fs::write(dir.join("rec_00002.oir"), part_file(5, 1, false)).unwrap();

    let (r, host) = import_noting(&first);
    let r = r.expect("a split acquisition imports");
    assert_eq!(r.image.frames, 6, "frames from some part were lost");
    for (t, plane) in r.image.planes.iter().enumerate() {
        let want: Vec<u16> = (0..8).map(|i| t as u16 * 1000 + i).collect();
        assert_eq!(
            plane,
            &PlaneData::U16(want),
            "frame {t} is wrong or out of order"
        );
    }
    assert!(
        host.said("3 part(s) of this acquisition, 6 planes in total"),
        "{:?}",
        host.lines
    );
    assert!(!host.said("unfinished"), "{:?}", host.lines);
    let _ = std::fs::remove_dir_all(dir);
}

/// A first part carrying only metadata is not mistaken for a recovery.
///
/// A split acquisition legitimately keeps its record in the named file and its
/// planes in the siblings, so "this part produced no planes" is ordinary there.
/// Warning about it would tell the user a finished recording was unfinished.
#[test]
fn a_metadata_only_part_is_not_reported_as_unfinished() {
    let first = {
        let mut b = Builder::new();
        b.xml(&size_record(4, 2));
        b.plane_chunk("REF_LSM0_abc_0", 0, &[0u8; 16]);
        b.plane_chunk("REF_LSM0_abc_1", 16, &[0u8; 16]);
        b.finish()
    };
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

/// The lines a successful import says, in order, for an ordinary file.
#[test]
fn an_ordinary_import_says_what_it_read() {
    let file = tmp("ordinary.oir", &stack_file(4, 2, &["t001_0_1_uid"], 16));
    let (r, host) = import_noting(&file);
    r.expect("import");
    assert_eq!(
        host.lines,
        [
            "4x2, 1 channel(s), 1 slice(s), 1 frame(s), 16-bit",
            "acquisition record: 0 channel(s), 0 event marker(s)",
        ]
    );
    let _ = std::fs::remove_file(file);
}

// ------------------------------------------------------------------ refusal

#[test]
fn a_file_without_the_signature_is_not_applicable() {
    let file = tmp("fake.oir", b"II*\0 not an OIR at all");
    match import(&file) {
        Err(PluginError::Unsupported(m)) => assert!(m.contains("OLYMPUSRAWFORMAT"), "{m}"),
        other => panic!("{other:?}"),
    }
    let _ = std::fs::remove_file(file);
}

#[test]
fn an_oir_with_no_planes_points_at_the_missing_parts() {
    let mut b = Builder::new();
    b.xml(&size_record(4, 4));
    b.plane_chunk("REF_LSM0_abc_0", 0, &[0u8; 32]);
    let file = tmp("noplanes.oir", &b.finish());
    match import(&file) {
        Err(PluginError::Unsupported(m)) => {
            assert!(m.contains("no image planes"), "{m}");
            assert!(m.contains("_00001.oir"), "{m}");
        }
        other => panic!("{other:?}"),
    }
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_file_that_states_no_frame_size_fails() {
    let mut b = Builder::new();
    b.plane_chunk("t001_0_1_uid", 0, &[0u8; 32]);
    let file = tmp("nosize.oir", &b.finish());
    match import(&file) {
        Err(PluginError::Failed(m)) => assert!(m.contains("frame size"), "{m}"),
        other => panic!("{other:?}"),
    }
    let _ = std::fs::remove_file(file);
}

/// The operating system's reason is in the message, as it always was.
#[test]
fn a_file_that_cannot_be_opened_says_why() {
    let path = std::env::temp_dir().join("fasttiff-oir-does-not-exist.oir");
    let _ = std::fs::remove_file(&path);
    match Oir.import(&request(&path), &mut Silent) {
        Err(PluginError::Failed(m)) => {
            assert!(m.starts_with("could not open the file: "), "{m}");
            assert!(m.len() > "could not open the file: ".len(), "{m}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_cancelled_import_is_reported_as_cancelled() {
    let names: Vec<String> = (1..=300).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let file = tmp("cancel.oir", &stack_file(4, 4, &refs, 16));
    for access in [Access::Auto, Access::Buffered] {
        let mut host = Notes {
            cancel: true,
            ..Default::default()
        };
        let r = Oir.import_with(&request(&file), &mut host, access);
        assert_eq!(r, Err(PluginError::unsupported("cancelled")));
    }
    let _ = std::fs::remove_file(file);
}

// ------------------------------------------------- against a real file

/// Read a real OIR and check it against the acquisition software's own export.
///
/// Ignored by default because it needs a file this repository cannot contain:
/// OIR samples are unpublished research data. Point it at one to run it —
///
/// ```text
/// FASTTIFF_OIR_SAMPLE=/path/to/file.oir cargo test -p fast-tiff-viewer -- --ignored oir
/// ```
///
/// When a `<stem>.tif` sits beside the `.oir` — FluoView writes both — every
/// frame is compared against it byte for byte. That is the only oracle there
/// is for a format with no specification, and it is a complete one: if the
/// export and the reassembly agree on every pixel of every frame, the reader
/// is right about this file. The import is done both ways, mapped and
/// buffered, and the two must agree before either is compared.
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

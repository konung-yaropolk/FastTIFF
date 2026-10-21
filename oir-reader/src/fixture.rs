//! OIR fixtures, built to the layout in the crate docs.
//!
//! These are synthetic on purpose. The format was determined from real FluoView
//! acquisitions and checked against that software's own TIFF export, but the
//! files themselves are unpublished research data and do not belong in a
//! repository — so what is here is the *structure*, written out by [`Builder`],
//! with pixel values chosen so that a mis-assembled plane is obvious rather than
//! plausible.
//!
//! Shared by the unit tests and, through `#[path]`, the integration tests, so
//! nothing here names a private item of the crate: the format's constants are
//! written out as the literals they are.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub(crate) const MAGIC: &[u8] = b"OLYMPUSRAWFORMAT";
pub(crate) const INDEX_MARKER: u32 = 0xFFFF_FFFF;
pub(crate) const INDEX_HEADER: usize = 12;
pub(crate) const TYPE_DESCRIPTOR: u32 = 3;

/// Builds an OIR the way the acquisition software lays one out.
#[derive(Default)]
pub(crate) struct Builder {
    pub(crate) offsets: Vec<u64>,
    pub(crate) body: Vec<u8>,
}

impl Builder {
    pub(crate) fn new() -> Self {
        let mut b = Builder::default();
        b.body.extend(MAGIC);
        // Header: sizes and offsets are patched in by `finish`.
        b.body.extend([0u8; 0x50 - 16]);
        b
    }

    /// One `u32 length, u32 type, payload` block.
    pub(crate) fn block(&mut self, ty: u32, payload: &[u8]) -> u64 {
        let at = self.body.len() as u64;
        self.body.extend((payload.len() as u32).to_le_bytes());
        self.body.extend(ty.to_le_bytes());
        self.body.extend(payload);
        self.offsets.push(at);
        at
    }

    /// A descriptor naming `name`, then the data block it describes.
    pub(crate) fn plane_chunk(&mut self, name: &str, at: u32, data: &[u8]) {
        let mut d = Vec::new();
        d.extend(at.to_le_bytes());
        d.extend((data.len() as u32).to_le_bytes());
        d.extend((name.len() as u32).to_le_bytes());
        d.extend(name.as_bytes());
        self.block(TYPE_DESCRIPTOR, &d);
        self.block(4, data);
    }

    pub(crate) fn xml(&mut self, text: &str) {
        self.block(5, text.as_bytes());
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
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

/// Where a finished file's index begins, from its header.
pub(crate) fn index_at(f: &[u8]) -> usize {
    u64::from_le_bytes(f[0x28..0x30].try_into().unwrap()) as usize
}

/// The smallest acquisition record that states a frame size.
pub(crate) fn size_record(w: u32, h: u32) -> String {
    format!(
        "<?xml version=\"1.0\"?><lsmimage:imageProperties><commonimage:imageInfo>\
         <commonimage:width>{w}</commonimage:width>\
         <commonimage:height>{h}</commonimage:height>\
         </commonimage:imageInfo></lsmimage:imageProperties>"
    )
}

/// `w`x`h` 16-bit planes, each pixel set to `plane * 1000 + index`, delivered
/// in scattered chunks and in the wrong order — so a reader that concatenates
/// blocks in file order, or ignores the declared offsets, fails.
pub(crate) fn stack_file(w: u32, h: u32, names: &[&str], chunk: usize) -> Vec<u8> {
    channels_file(w, h, names, chunk, 1)
}

/// As [`stack_file`], with the acquisition record stating `channels` recorded
/// channels — which is what decides the shape when the plane names do not.
pub(crate) fn channels_file(
    w: u32,
    h: u32,
    names: &[&str],
    chunk: usize,
    channels: usize,
) -> Vec<u8> {
    channels_file_with(w, h, names, chunk, channels, "")
}

/// As [`channels_file`], with `extra` inside the record's `imageInfo` — an
/// axis, say.
pub(crate) fn channels_file_with(
    w: u32,
    h: u32,
    names: &[&str],
    chunk: usize,
    channels: usize,
    extra: &str,
) -> Vec<u8> {
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
         {extra}\
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
/// That structure is the reason the walk demands a run of blocks rather than
/// one before it believes where the stream starts — see the test that uses
/// this.
pub(crate) fn real_layout_file(w: u32, h: u32, names: &[&str], chunk: usize) -> Vec<u8> {
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

/// One part of a split acquisition: `frames` 4x2 planes named from `first`,
/// each pixel `frame * 1000 + index`, with the record only in the first part —
/// which is where a real acquisition keeps it.
pub(crate) fn part_file(first: usize, frames: usize, with_record: bool) -> Vec<u8> {
    let mut b = Builder::new();
    if with_record {
        b.xml(&size_record(4, 2));
    }
    for t in first..first + frames {
        let bytes: Vec<u8> = (0..8u16)
            .flat_map(|i| (t as u16).wrapping_mul(1000).wrapping_add(i).to_le_bytes())
            .collect();
        // Two chunks per plane, as the real format scatters them.
        let name = format!("t{:03}_0_1_uid", t + 1);
        b.plane_chunk(&name, 0, &bytes[..10]);
        b.plane_chunk(&name, 10, &bytes[10..]);
    }
    b.finish()
}

/// A first part carrying only metadata, as a split acquisition's named file
/// does: the record, the reference snapshot, the thumbnail. Several blocks, so
/// that it can be walked — and the walk has to find no plane in it.
pub(crate) fn metadata_only_file() -> Vec<u8> {
    let mut b = Builder::new();
    b.xml(&size_record(4, 2));
    b.plane_chunk("REF_LSM0_abc_0", 0, &[0u8; 16]);
    b.plane_chunk("REF_LSM0_abc_1", 16, &[0u8; 16]);
    b.finish()
}

/// The record a real two-channel timelapse carries, abridged: everything the
/// translation reads, so that a reader that drops or garbles any of it shows.
pub(crate) fn full_record() -> String {
    "<?xml version=\"1.0\"?><lsmimage:imageProperties>\
     <commonimage:general><base:creationDateTime>2025-02-10T18:49:39.124-05:00\
     </base:creationDateTime></commonimage:general>\
     <commonimage:system><base:systemName>FVMPE-RS</base:systemName></commonimage:system>\
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
     </commonimage:imageInfo>\
     <lsmimage:acquisition><lsmimage:microscopeConfiguration>\
     <opticalelement:objectiveLens><opticalelement:name>XLUMPLFLN20XW</opticalelement:name>\
     </opticalelement:objectiveLens></lsmimage:microscopeConfiguration>\
     </lsmimage:acquisition></lsmimage:imageProperties>"
        .to_string()
}

/// A timelapse with everything a record can carry — frame timestamps, an
/// event, a lookup table that must be dropped — and two 4x4 frames.
pub(crate) fn recorded_file() -> Vec<u8> {
    let mut b = Builder::new();
    b.xml(&full_record());
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
    // The document that must never reach a description. A real one is 525 KB
    // of exactly this, and there are three of them.
    b.xml(
        "<?xml version=\"1.0\"?><lut:LUT><lut:intensity>0</lut:intensity>\
         <lut:contrast>1</lut:contrast></lut:LUT>",
    );
    for (i, name) in ["t001_0_1_uid", "t002_0_1_uid"].iter().enumerate() {
        b.plane_chunk(name, 0, &[i as u8; 32]);
    }
    b.finish()
}

/// A directory of its own under the system's temporary directory, removed
/// when this is dropped. Named for the process and the test, so tests running
/// at once never share one.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(test: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("oir-reader-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    /// Write `bytes` as `name` in here, and say where.
    pub(crate) fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Every shape of file the tests build, by name: whole files, files damaged in
/// each of the ways a real one is, and files that must be refused.
///
/// What the differential tests run through both readers, along with every
/// length each can be cut to and a few hundred single-byte corruptions of
/// each.
pub(crate) fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let ten: Vec<String> = (1..=10).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let ten: Vec<&str> = ten.iter().map(|s| s.as_str()).collect();
    let six: Vec<String> = (1..=6).map(|i| format!("t{i:03}_0_1_uid")).collect();
    let six: Vec<&str> = six.iter().map(|s| s.as_str()).collect();
    let zc: Vec<String> = (1..=3)
        .flat_map(|z| ["uidA", "uidB"].map(move |c| format!("z{z}_0_1_{c}")))
        .collect();
    let zc: Vec<&str> = zc.iter().map(|s| s.as_str()).collect();
    let two = ["t001_0_1_uid", "t002_0_1_uid"];

    let mut out: Vec<(&'static str, Vec<u8>)> = vec![
        ("scatter", stack_file(8, 4, &["t001_0_1_uid"], 22)),
        ("order", stack_file(4, 2, &ten, 7)),
        ("axes", channels_file(4, 4, &six, 16, 2)),
        ("axes_bad", channels_file(4, 4, &six, 16, 4)),
        ("axes_from_names", channels_file(4, 4, &zc, 16, 1)),
        ("real_layout", real_layout_file(4, 4, &["t001_0_1_uid"], 16)),
        ("recorded", recorded_file()),
        ("metadata_only", metadata_only_file()),
        ("part0", part_file(0, 2, true)),
        ("part1", part_file(2, 3, false)),
    ];

    let mut f = stack_file(4, 4, &two, 16);
    let at = index_at(&f);
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    out.push(("no_index", f));

    let mut f = stack_file(4, 4, &two, 16);
    let at = index_at(&f);
    f.truncate(at - 8);
    out.push(("cut_mid_frame", f));

    let mut f = real_layout_file(4, 4, &["t001_0_1_uid"], 16);
    let at = index_at(&f);
    f.truncate(at);
    f[0x28..0x30].copy_from_slice(&0u64.to_le_bytes());
    out.push(("real_layout_no_index", f));

    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    f[0x28..0x30].copy_from_slice(&u64::MAX.to_le_bytes());
    out.push(("wild_index_offset", f));

    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = index_at(&f);
    f[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    out.push(("index_without_marker", f));

    let mut f = stack_file(4, 4, &["t001_0_1_uid"], 16);
    let at = index_at(&f);
    for b in f[at + INDEX_HEADER..].iter_mut() {
        *b = 0;
    }
    out.push(("blank_index", f));

    let mut b = Builder::new();
    b.xml(&size_record(4, 4));
    b.plane_chunk("REF_LSM0_abc_0", 0, &[0u8; 32]);
    out.push(("no_planes", b.finish()));

    let mut b = Builder::new();
    b.plane_chunk("t001_0_1_uid", 0, &[0u8; 32]);
    out.push(("no_frame_size", b.finish()));

    let mut b = Builder::new();
    b.xml(&size_record(4, 2));
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 16]);
    b.plane_chunk("t001_0_1_uid", 1_000_000, &[2u8; 16]);
    out.push(("chunk_past_plane", b.finish()));

    let mut b = Builder::new();
    b.xml(&size_record(4, 2));
    for name in two {
        b.plane_chunk(name, 0, &[1u8; 16]);
    }
    b.plane_chunk("t003_0_1_uid", 0, &[2u8; 6]);
    out.push(("incomplete_trailing_plane", b.finish()));

    let mut b = Builder::new();
    b.xml(&size_record(8, 8));
    b.plane_chunk("t001_0_1_uid", 0, &[1u8; 128]);
    b.plane_chunk("t002_0_1_uid", 0, &[1u8; 128]);
    out.push(("uniformly_short", b.finish()));

    let mut b = Builder::new();
    b.xml(&size_record(4, 2));
    b.plane_chunk("t001_0_1_uid", 0, &[3u8; 5]);
    b.plane_chunk("t001_0_1_uid", 5, &[4u8; 3]);
    b.plane_chunk("t002_0_1_uid", 0, &[5u8; 8]);
    out.push(("eight_bit", b.finish()));

    let mut b = Builder::new();
    b.xml(&size_record(4, 2));
    b.plane_chunk("t001_0_1_uid", ((1usize << 30) - 16) as u32, &[7u8; 16]);
    out.push(("lying_descriptor", b.finish()));

    let mut f = Vec::new();
    f.extend(MAGIC);
    f.extend([0u8; 0x2000]);
    out.push(("zeros", f));

    let mut f = Vec::new();
    f.extend(MAGIC);
    f.extend([0u8; 0x40]);
    for _ in 0..512 {
        f.extend(0xDEAD_BEEFu32.to_le_bytes());
        f.extend(0xFEED_FACEu32.to_le_bytes());
    }
    out.push(("rubbish", f));

    out.push(("not_oir", b"II*\0 not an OIR at all".to_vec()));
    out.push(("empty", Vec::new()));
    out
}

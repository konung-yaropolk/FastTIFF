//! The `.roi` bytes are checked twice over: offset by offset against what
//! `RoiEncoder` writes, and by decoding them back with a reader that follows
//! `RoiDecoder` step for step — the only check that the offsets agree with one
//! another, which is what ImageJ actually depends on.

use super::*;

/// What `RoiDecoder.getRoi` would build from a point ROI's bytes, reduced to
/// the fields a point ROI has.
#[derive(Debug)]
struct Decoded {
    version: i32,
    kind: i32,
    top: i32,
    left: i32,
    bottom: i32,
    right: i32,
    options: i32,
    position: i32,
    size: i32,
    point_type: i32,
    stroke_color: u32,
    show_labels: bool,
    sub_pixel: bool,
    name: Option<String>,
    /// `left + x`, `top + y` from the integer arrays.
    int_points: Vec<(i32, i32)>,
    /// The float arrays, when the sub-pixel flag says to use them.
    float_points: Option<Vec<(f32, f32)>>,
    /// `(counter, position)` per point, from the counters array.
    counters: Option<Vec<(i32, i32)>>,
    c_z_t: (i32, i32, i32),
}

/// `RoiDecoder.getShort`: signed, except that anything below -5000 is taken to
/// be an unsigned value past 32767.
fn get_short(d: &[u8], at: usize) -> i32 {
    let v = i16::from_be_bytes([d[at], d[at + 1]]) as i32;
    if v < -5000 {
        u16::from_be_bytes([d[at], d[at + 1]]) as i32
    } else {
        v
    }
}

fn get_ushort(d: &[u8], at: usize) -> i32 {
    u16::from_be_bytes([d[at], d[at + 1]]) as i32
}

fn get_int(d: &[u8], at: usize) -> i32 {
    i32::from_be_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}

fn get_float(d: &[u8], at: usize) -> f32 {
    f32::from_bits(get_int(d, at) as u32)
}

/// `RoiDecoder.getRoi`, the point branch, in its order.
fn decode(d: &[u8]) -> Decoded {
    assert_eq!(&d[..2], b"Io", "not an ImageJ ROI");
    let size = d.len();
    let version = get_short(d, 4);
    let kind = d[6] as i32;
    let (top, left, bottom, right) = (
        get_short(d, 8),
        get_short(d, 10),
        get_short(d, 12),
        get_short(d, 14),
    );
    let mut n = get_ushort(d, 16);
    if n == 0 {
        n = get_int(d, 18);
    }
    let n = n as usize;
    let options = get_short(d, 50);
    let position = get_int(d, 56);
    let hdr2 = get_int(d, 60) as usize;
    let mut c_z_t = (0, 0, 0);
    if hdr2 > 0 && hdr2 + 32 + 4 <= size {
        c_z_t = (
            get_int(d, hdr2 + 4),
            get_int(d, hdr2 + 8),
            get_int(d, hdr2 + 12),
        );
    }
    let sub_pixel = options & 128 != 0 && version >= 222;

    let mut int_points = Vec::with_capacity(n);
    for i in 0..n {
        let x = get_short(d, 64 + 2 * i).max(0);
        let y = get_short(d, 64 + 2 * n + 2 * i).max(0);
        int_points.push((left + x, top + y));
    }
    let float_points = sub_pixel.then(|| {
        (0..n)
            .map(|i| {
                (
                    get_float(d, 64 + 4 * n + 4 * i),
                    get_float(d, 64 + 8 * n + 4 * i),
                )
            })
            .collect()
    });
    let (point_type, point_size) = if version >= 226 {
        (d[52] as i32, get_short(d, 34))
    } else {
        (0, 0)
    };

    // `getRoiName`
    let name = (hdr2 != 0)
        .then(|| {
            (
                get_int(d, hdr2 + 16) as usize,
                get_int(d, hdr2 + 20) as usize,
            )
        })
        .filter(|&(offset, len)| offset != 0 && len != 0 && offset + 2 * len <= size)
        .map(|(offset, len)| {
            let units: Vec<u16> = (0..len)
                .map(|i| get_ushort(d, offset + 2 * i) as u16)
                .collect();
            String::from_utf16(&units).expect("a name is valid UTF-16")
        });

    // `getPointCounters`, then `PointRoi.setCounters`.
    let counters = (version >= 227 && hdr2 != 0)
        .then(|| get_int(d, hdr2 + 48) as usize)
        .filter(|&offset| offset != 0 && offset + 4 * n <= size)
        .map(|offset| {
            (0..n)
                .map(|i| {
                    let word = get_int(d, offset + 4 * i);
                    (word & 0xff, word >> 8)
                })
                .collect()
        });

    Decoded {
        version,
        kind,
        top,
        left,
        bottom,
        right,
        options,
        position,
        size: point_size,
        point_type,
        stroke_color: get_int(d, 40) as u32,
        show_labels: options & 1024 != 0,
        sub_pixel,
        name,
        int_points,
        float_points,
        counters,
        c_z_t,
    }
}

fn pt(x: f32, y: f32, position: u32) -> RoiPoint {
    RoiPoint { x, y, position }
}

fn roi(name: &str, color: [u8; 3], points: Vec<RoiPoint>) -> PointRoi {
    PointRoi {
        hyperstack_position: [0; 3],
        name: name.to_string(),
        color,
        points,
    }
}

// ------------------------------------------------------------ the .roi bytes

/// Every field of a three-point ROI, at its offset, with the value ImageJ's own
/// encoder would put there.
#[test]
fn every_header_field_is_at_its_offset() {
    let points = vec![pt(10.5, 20.5, 4), pt(2.25, 3.75, 4), pt(7.9, 1.1, 4)];
    let r = roi("Cells", [0xe4, 0x1a, 0x1c], points);
    let b = r.to_imagej_bytes();
    let n = 3;

    assert_eq!(&b[0..4], b"Iout");
    assert_eq!(&b[4..6], &228u16.to_be_bytes(), "version");
    assert_eq!(b[6], 10, "type: point");
    assert_eq!(b[7], 0);
    // Bounds: x from pixel 2 to pixel 10 inclusive, y from 1 to 20 inclusive.
    assert_eq!(&b[8..10], &1i16.to_be_bytes(), "top");
    assert_eq!(&b[10..12], &2i16.to_be_bytes(), "left");
    assert_eq!(&b[12..14], &21i16.to_be_bytes(), "bottom");
    assert_eq!(&b[14..16], &11i16.to_be_bytes(), "right");
    assert_eq!(&b[16..18], &3u16.to_be_bytes(), "point count");
    assert!(b[18..34].iter().all(|&v| v == 0), "18..34 unused by points");
    assert_eq!(&b[34..36], &1i16.to_be_bytes(), "point size index 1: Small");
    assert_eq!(&b[36..40], &[0; 4], "not a composite ROI");
    assert_eq!(
        &b[40..44],
        &[0xFF, 0xe4, 0x1a, 0x1c],
        "stroke colour, opaque ARGB"
    );
    assert_eq!(&b[44..48], &[0; 4], "no fill colour");
    assert_eq!(&b[48..50], &[0; 2], "subtype");
    assert_eq!(
        &b[50..52],
        &(128u16 | 1024).to_be_bytes(),
        "sub-pixel + labels"
    );
    assert_eq!(b[52], 1, "point type: cross");
    assert_eq!(&b[53..56], &[0; 3]);
    assert_eq!(
        &b[56..60],
        &4i32.to_be_bytes(),
        "the points' common position"
    );
    let header2 = 64 + 4 * n + 8 * n;
    assert_eq!(
        &b[60..64],
        &(header2 as i32).to_be_bytes(),
        "header2 offset"
    );

    // Integer coordinates, relative to (left, top) = (2, 1): each point's
    // pixel. The third point's y of 1.1 is pixel 1, which is `top` itself.
    let shorts = |at: usize| -> Vec<i16> {
        (0..n)
            .map(|i| i16::from_be_bytes([b[at + 2 * i], b[at + 2 * i + 1]]))
            .collect()
    };
    assert_eq!(shorts(64), vec![10 - 2, 2 - 2, 7 - 2], "x, relative");
    assert_eq!(
        shorts(64 + 2 * n),
        vec![20 - 1, 3 - 1, 1 - 1],
        "y, relative"
    );

    // Float coordinates, absolute, on ImageJ's pixel-centre convention.
    let floats = |at: usize| -> Vec<f32> {
        (0..n)
            .map(|i| f32::from_be_bytes(b[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
            .collect()
    };
    assert_eq!(floats(64 + 4 * n), vec![10.0, 1.75, 7.4]);
    assert_eq!(floats(64 + 8 * n), vec![20.0, 3.25, 0.6]);

    // Header2.
    let h = header2;
    assert_eq!(&b[h..h + 4], &[0; 4], "unused");
    assert_eq!(&b[h + 4..h + 16], &[0; 12], "C, Z, T");
    let name_at = h + 64;
    assert_eq!(
        &b[h + 16..h + 20],
        &(name_at as i32).to_be_bytes(),
        "name offset"
    );
    assert_eq!(
        &b[h + 20..h + 24],
        &5i32.to_be_bytes(),
        "name length, chars"
    );
    assert_eq!(
        &b[h + 24..h + 48],
        &[0; 24],
        "label colour, font, group, stroke…"
    );
    let counters_at = name_at + 2 * 5;
    assert_eq!(
        &b[h + 48..h + 52],
        &(counters_at as i32).to_be_bytes(),
        "counters offset"
    );
    assert_eq!(&b[h + 52..h + 64], &[0; 12]);
    assert_eq!(
        &b[name_at..counters_at],
        &[0, b'C', 0, b'e', 0, b'l', 0, b'l', 0, b's'],
        "name, UTF-16BE"
    );
    for i in 0..n {
        let at = counters_at + 4 * i;
        assert_eq!(&b[at..at + 4], &(4i32 << 8).to_be_bytes(), "counter {i}");
    }
    assert_eq!(b.len(), counters_at + 4 * n, "nothing after the counters");
}

/// `128 + 16n + 2·nameLen`, for several sizes and names — the name counted in
/// UTF-16 units, as Java's `String.length()` counts it.
#[test]
fn the_length_is_the_formula() {
    for (name, units) in [("", 0), ("Group 1", 7), ("Zellen ü", 8), ("🦠 x", 4)] {
        for n in [1usize, 2, 7, 300] {
            let points = (0..n).map(|i| pt(i as f32 + 0.5, 2.5, 0)).collect();
            let b = roi(name, [1, 2, 3], points).to_imagej_bytes();
            assert_eq!(b.len(), 128 + 16 * n + 2 * units, "{name:?} x {n}");
            assert_eq!(
                decode(&b).name.as_deref(),
                (!name.is_empty()).then_some(name)
            );
        }
    }
}

/// A click in the middle of a pixel is that pixel's centre in ImageJ.
#[test]
fn a_mid_pixel_point_lands_on_imagejs_pixel_centre() {
    let b = roi("p", [0, 0, 0], vec![pt(10.5, 20.5, 0)]).to_imagej_bytes();
    let d = decode(&b);
    assert_eq!(d.float_points, Some(vec![(10.0, 20.0)]));
    assert_eq!(d.int_points, vec![(10, 20)]);
    // A single point's bounds are its own pixel, one wide and one high, as
    // `Roi.setIntBounds` makes them for a point.
    assert_eq!((d.left, d.top, d.right, d.bottom), (10, 20, 11, 21));
}

/// Everything ImageJ reads back is what was written.
#[test]
fn the_bytes_decode_back_the_way_roidecoder_reads_them() {
    let points = vec![pt(5.5, 6.5, 3), pt(100.25, 40.75, 9), pt(0.5, 0.5, 0)];
    let r = roi("Group 2", [0x37, 0x7e, 0xb8], points.clone());
    let d = decode(&r.to_imagej_bytes());

    assert_eq!(d.version, 228);
    assert_eq!(d.kind, 10);
    assert!(d.sub_pixel, "float coordinates must be used");
    assert!(d.show_labels, "points must be numbered");
    assert_eq!(d.options, 128 | 1024);
    assert_eq!(d.point_type, 1, "cross");
    assert_eq!(d.size, 1, "small");
    assert_eq!(d.stroke_color, 0xFF37_7EB8);
    assert_eq!(d.name.as_deref(), Some("Group 2"));
    assert_eq!(d.c_z_t, (0, 0, 0));
    assert_eq!(d.position, -2, "the points are on different slices");

    let floats = d.float_points.expect("sub-pixel");
    for (p, (x, y)) in points.iter().zip(&floats) {
        assert_eq!((*x, *y), (p.x - 0.5, p.y - 0.5));
    }
    // The integer arrays agree with the floats to the pixel.
    for ((ix, iy), (x, y)) in d.int_points.iter().zip(&floats) {
        assert_eq!(
            (*ix, *iy),
            ((x + 0.5).floor() as i32, (y + 0.5).floor() as i32)
        );
    }
    // Every point keeps its own slice, under counter 0.
    assert_eq!(d.counters, Some(vec![(0, 3), (0, 9), (0, 0)]));
}

/// The ROI-wide position: the common one, 0 included, or -2 when they differ.
#[test]
fn the_position_is_common_or_pointwise() {
    let at = |positions: &[u32]| {
        let points = positions
            .iter()
            .enumerate()
            .map(|(i, &p)| pt(i as f32, 1.0, p))
            .collect();
        decode(&roi("x", [0; 3], points).to_imagej_bytes()).position
    };
    assert_eq!(at(&[0, 0, 0]), 0, "every slice");
    assert_eq!(at(&[7, 7]), 7, "all on slice 7");
    assert_eq!(at(&[12]), 12, "one point");
    assert_eq!(at(&[1, 2]), -2);
    assert_eq!(at(&[0, 5]), -2, "some on every slice, one on slice 5");
}

/// No name: no name offset or length, and the counters straight after header2.
#[test]
fn a_nameless_roi_has_no_name_offset() {
    let b = roi("", [9, 9, 9], vec![pt(1.5, 1.5, 2), pt(3.5, 3.5, 2)]).to_imagej_bytes();
    let h = 64 + 12 * 2;
    assert_eq!(&b[h + 16..h + 24], &[0; 8]);
    assert_eq!(&b[h + 48..h + 52], &((h + 64) as i32).to_be_bytes());
    assert_eq!(decode(&b).name, None);
    assert_eq!(decode(&b).counters, Some(vec![(0, 2), (0, 2)]));
}

/// More points than a u16 counts: 0 at offset 16, the real count at 18.
#[test]
fn more_than_65535_points_moves_the_count() {
    let n = 65_536 + 5;
    let points = (0..n)
        .map(|i| pt((i % 512) as f32, (i / 512) as f32, 0))
        .collect();
    let b = roi("many", [0; 3], points).to_imagej_bytes();
    assert_eq!(&b[16..18], &[0, 0]);
    assert_eq!(&b[18..22], &(n as i32).to_be_bytes());
    assert_eq!(b.len(), 128 + 16 * n + 2 * 4);
    let d = decode(&b);
    assert_eq!(d.float_points.map(|p| p.len()), Some(n));
    assert_eq!(d.name.as_deref(), Some("many"));
}

/// Bounds that do not fit an i16 are clamped; the float coordinates, which
/// ImageJ uses, are not.
#[test]
fn out_of_range_bounds_are_clamped() {
    let b = roi("far", [0; 3], vec![pt(1.5, 2.5, 0), pt(40_000.5, 2.5, 0)]).to_imagej_bytes();
    assert_eq!(&b[14..16], &i16::MAX.to_be_bytes(), "right");
    assert_eq!(decode(&b).float_points.unwrap()[1], (40_000.0, 2.0));
}

/// Slices past what a counter word can carry are clamped rather than wrapped
/// into a negative one.
#[test]
fn a_huge_position_is_clamped_in_the_counters() {
    let b = roi("x", [0; 3], vec![pt(1.0, 1.0, u32::MAX)]).to_imagej_bytes();
    assert_eq!(decode(&b).counters, Some(vec![(0, 0x7F_FFFF)]));
}

// --------------------------------------------------------------- the RoiSet

/// One entry from a test-side reading of the zip.
struct Entry {
    name: String,
    data: Vec<u8>,
}

/// CRC-32 (IEEE), bitwise — independent of the writer's.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn le16(d: &[u8], at: usize) -> usize {
    u16::from_le_bytes([d[at], d[at + 1]]) as usize
}

fn le32(d: &[u8], at: usize) -> usize {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as usize
}

/// Read the archive both ways a reader can — forwards through the local
/// headers, as Java's `ZipInputStream` does, and through the central
/// directory — and require the two to agree.
fn read_zip(z: &[u8]) -> Vec<Entry> {
    // End of central directory: the last 22 bytes, there being no comment.
    let eocd = z.len() - 22;
    assert_eq!(le32(z, eocd), 0x0605_4b50, "end record");
    assert_eq!(le16(z, eocd + 4), 0, "disk");
    assert_eq!(le16(z, eocd + 6), 0, "directory disk");
    let count = le16(z, eocd + 8);
    assert_eq!(le16(z, eocd + 10), count);
    let cd_size = le32(z, eocd + 12);
    let cd_start = le32(z, eocd + 16);
    assert_eq!(
        cd_start + cd_size,
        eocd,
        "the directory ends at the end record"
    );
    assert_eq!(le16(z, eocd + 20), 0, "comment");

    // Forwards, local headers only.
    let mut entries = Vec::new();
    let mut offsets = Vec::new();
    let mut at = 0;
    while at < cd_start {
        assert_eq!(le32(z, at), 0x0403_4b50, "local header at {at}");
        assert_eq!(le16(z, at + 4), 10, "version needed");
        let flags = le16(z, at + 6);
        assert_eq!(flags & 0x8, 0, "no data descriptor");
        assert_eq!(le16(z, at + 8), 0, "stored");
        let crc = le32(z, at + 14) as u32;
        let compressed = le32(z, at + 18);
        let size = le32(z, at + 22);
        assert_eq!(compressed, size, "stored: both sizes alike");
        let name_len = le16(z, at + 26);
        assert_eq!(le16(z, at + 28), 0, "extra field");
        let name = &z[at + 30..at + 30 + name_len];
        if !name.is_ascii() {
            assert_ne!(flags & 0x800, 0, "a UTF-8 name says so");
        }
        let data = z[at + 30 + name_len..at + 30 + name_len + size].to_vec();
        assert_eq!(
            crc32(&data),
            crc,
            "CRC-32 of {:?}",
            String::from_utf8_lossy(name)
        );
        offsets.push(at);
        entries.push(Entry {
            name: String::from_utf8(name.to_vec()).unwrap(),
            data,
        });
        at += 30 + name_len + size;
    }
    assert_eq!(at, cd_start, "the entries end where the directory starts");
    assert_eq!(entries.len(), count);

    // The directory: one record per entry, pointing back at it.
    let mut at = cd_start;
    for (entry, offset) in entries.iter().zip(&offsets) {
        assert_eq!(le32(z, at), 0x0201_4b50, "central header");
        assert_eq!(le16(z, at + 6), 10, "version needed");
        assert_eq!(le16(z, at + 10), 0, "stored");
        assert_eq!(le32(z, at + 16) as u32, crc32(&entry.data));
        assert_eq!(le32(z, at + 20), entry.data.len());
        assert_eq!(le32(z, at + 24), entry.data.len());
        let name_len = le16(z, at + 28);
        let (extra, comment) = (le16(z, at + 30), le16(z, at + 32));
        assert_eq!(le32(z, at + 42), *offset, "local header offset");
        assert_eq!(&z[at + 46..at + 46 + name_len], entry.name.as_bytes());
        at += 46 + name_len + extra + comment;
    }
    assert_eq!(at, eocd, "the directory holds exactly the entries");
    entries
}

/// Each ROI is one entry holding exactly its `.roi` bytes, in order.
#[test]
fn the_zip_holds_one_stored_entry_per_roi() {
    let rois = vec![
        roi("Group 1", SET1_RED, vec![pt(1.5, 1.5, 1), pt(2.5, 2.5, 2)]),
        roi("Group 2", [0x37, 0x7e, 0xb8], vec![pt(9.5, 9.5, 1)]),
    ];
    let entries = read_zip(&roi_set_zip(&rois).unwrap());
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["Group 1.roi", "Group 2.roi"]);
    for (entry, r) in entries.iter().zip(&rois) {
        assert_eq!(entry.data, r.to_imagej_bytes());
    }
}

const SET1_RED: [u8; 3] = [0xe4, 0x1a, 0x1c];

/// Names made unique as RoiManager makes them.
#[test]
fn clashing_names_are_made_unique_like_the_roi_manager() {
    let names = |list: &[&str]| -> Vec<String> {
        let rois: Vec<PointRoi> = list
            .iter()
            .map(|n| roi(n, [0; 3], vec![pt(1.0, 1.0, 0)]))
            .collect();
        read_zip(&roi_set_zip(&rois).unwrap())
            .into_iter()
            .map(|e| e.name)
            .collect()
    };
    assert_eq!(names(&["A", "B"]), ["A.roi", "B.roi"]);
    assert_eq!(names(&["A", "A", "A"]), ["A.roi", "A-1.roi", "A-2.roi"]);
    // A later label already called `A-1` is avoided, not duplicated.
    assert_eq!(names(&["A", "A", "A-1"]), ["A.roi", "A-2.roi", "A-1.roi"]);
    assert_eq!(names(&["A", "A-1", "A"]), ["A.roi", "A-1.roi", "A-2.roi"]);
    // RoiManager drops a short dash suffix before numbering.
    assert_eq!(names(&["WT-1", "WT-1"]), ["WT-1.roi", "WT-2.roi"]);
    // A name already ending in `.roi` is not given a second one.
    assert_eq!(names(&["x.roi"]), ["x.roi"]);
    // Separators would make a directory on unpacking.
    assert_eq!(names(&["CA1/CA3", "a\\b"]), ["CA1_CA3.roi", "a_b.roi"]);
}

/// The ROI's own name is untouched by the entry's renaming.
#[test]
fn a_renamed_entry_keeps_the_rois_own_name() {
    let rois = vec![
        roi("Same", [0; 3], vec![pt(1.0, 1.0, 0)]),
        roi("Same", [0; 3], vec![pt(2.0, 2.0, 0)]),
    ];
    let entries = read_zip(&roi_set_zip(&rois).unwrap());
    assert_eq!(entries[1].name, "Same-1.roi");
    assert_eq!(decode(&entries[1].data).name.as_deref(), Some("Same"));
}

/// A ROI with no points has no `.roi`, so it has no entry.
#[test]
fn a_roi_without_points_is_left_out() {
    let rois = vec![
        roi("Empty", [0; 3], vec![]),
        roi("Kept", [0; 3], vec![pt(1.0, 1.0, 0)]),
    ];
    let entries = read_zip(&roi_set_zip(&rois).unwrap());
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "Kept.roi");
    // And none at all is an empty archive: the end record alone.
    let empty = roi_set_zip(&[]).unwrap();
    assert_eq!(empty.len(), 22);
    assert!(read_zip(&empty).is_empty());
}

/// A name outside ASCII is written as UTF-8 and flagged as such.
#[test]
fn a_non_ascii_name_is_flagged_utf8() {
    let z = roi_set_zip(&[roi("Zellen ü", [0; 3], vec![pt(1.0, 1.0, 0)])]).unwrap();
    assert_eq!(le16(&z, 6) & 0x800, 0x800);
    assert_eq!(read_zip(&z)[0].name, "Zellen ü.roi");
}

/// The same ROIs give the same bytes: no clock in the archive.
#[test]
fn the_zip_is_deterministic() {
    let rois = vec![roi("G", [0; 3], vec![pt(1.0, 1.0, 0)])];
    assert_eq!(roi_set_zip(&rois).unwrap(), roi_set_zip(&rois).unwrap());
    assert_eq!(le16(&roi_set_zip(&rois).unwrap(), 12), 0x21, "1980-01-01");
}

// --------------------------------------------------------- reading them back

/// Whether two point lists are the same to f32 rounding: the encoder takes
/// half a pixel off and the decoder puts it back, which is exact for the
/// halves and quarters most tests use and within an ulp or two otherwise.
fn assert_same_points(got: &[RoiPoint], want: &[RoiPoint]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.position, w.position, "point {i}: {got:?} vs {want:?}");
        let near = |a: f32, b: f32| (a - b).abs() <= 1e-5 * b.abs().max(1.0);
        assert!(
            near(g.x, w.x) && near(g.y, w.y),
            "point {i}: {got:?} vs {want:?}"
        );
    }
}

/// Every shape the encoder writes reads back as the ROI it was: one common
/// position, points on different slices ("pointwise"), every slice; with a
/// name, without one, outside ASCII; one point and several.
#[test]
fn every_encoded_shape_reads_back() {
    let shapes = vec![
        roi(
            "Cells",
            [0xe4, 0x1a, 0x1c],
            vec![pt(10.5, 20.5, 4), pt(2.25, 3.75, 4), pt(7.9, 1.1, 4)],
        ),
        roi(
            "Group 2",
            [0x37, 0x7e, 0xb8],
            vec![pt(5.5, 6.5, 3), pt(100.25, 40.75, 9), pt(0.5, 0.5, 0)],
        ),
        roi(
            "all",
            [0x4d, 0xaf, 0x4a],
            vec![pt(1.5, 1.5, 0), pt(3.5, 2.5, 0)],
        ),
        roi("", [9, 9, 9], vec![pt(1.5, 1.5, 2), pt(3.5, 3.5, 2)]),
        roi("Zellen ü 🦠", [1, 2, 3], vec![pt(0.0, 0.0, 7)]),
        // Black is a colour, not "unset": the encoder writes it opaque.
        roi("black", [0, 0, 0], vec![pt(12.5, 0.5, 1)]),
        // Far out: bounds clamped to an i16, floats kept.
        roi(
            "far",
            [5, 6, 7],
            vec![pt(1.5, 2.5, 0), pt(40_000.5, 2.5, 0)],
        ),
    ];
    for shape in &shapes {
        let back = point_rois_from_imagej(&shape.to_imagej_bytes());
        assert_eq!(back.len(), 1, "{shape:?}");
        assert_eq!(back[0].name, shape.name);
        assert_eq!(back[0].color, shape.color);
        assert_same_points(&back[0].points, &shape.points);
    }
}

/// Mid-pixel points are exact: `x.5 - 0.5 + 0.5` loses nothing.
#[test]
fn a_mid_pixel_point_reads_back_exactly() {
    let r = roi("p", [1, 2, 3], vec![pt(10.5, 20.5, 3), pt(0.5, 99.5, 0)]);
    assert_eq!(point_rois_from_imagej(&r.to_imagej_bytes()), vec![r]);
}

/// The count past a u16 is read from where the encoder moved it.
#[test]
fn more_than_65535_points_read_back() {
    let n = 65_536 + 5;
    let points: Vec<RoiPoint> = (0..n)
        .map(|i| {
            pt(
                (i % 512) as f32 + 0.5,
                (i / 512) as f32 + 0.5,
                1 + i as u32 % 3,
            )
        })
        .collect();
    let r = roi("many", [0; 3], points);
    assert_eq!(point_rois_from_imagej(&r.to_imagej_bytes()), vec![r]);
}

/// A `.roi` laid out the way `RoiEncoder.write` lays one out: header, integer
/// coordinates relative to the bounds, float ones when given, the second
/// header, the name, the counters. Anything a test leaves at its default is
/// what ImageJ writes for a point ROI that does not use it.
struct Raw<'a> {
    version: u16,
    kind: u8,
    options: u16,
    position: i32,
    stroke: u32,
    /// Absolute pixel coordinates; the bounds are made from them.
    ints: &'a [(i16, i16)],
    floats: Option<&'a [(f32, f32)]>,
    name: &'a str,
    counters: Option<&'a [i32]>,
    c_z_t: (i32, i32, i32),
    /// No second header at all, as before ImageJ 1.43i.
    no_header2: bool,
}

impl Default for Raw<'_> {
    fn default() -> Self {
        Raw {
            version: 228,
            kind: 10,
            options: 0,
            position: 0,
            stroke: 0,
            ints: &[],
            floats: None,
            name: "",
            counters: None,
            c_z_t: (0, 0, 0),
            no_header2: false,
        }
    }
}

fn raw(r: &Raw) -> Vec<u8> {
    fn put16(d: &mut [u8], at: usize, v: u16) {
        d[at..at + 2].copy_from_slice(&v.to_be_bytes());
    }
    fn put32(d: &mut [u8], at: usize, v: u32) {
        d[at..at + 4].copy_from_slice(&v.to_be_bytes());
    }
    let n = r.ints.len();
    let float_size = if r.floats.is_some() { 8 * n } else { 0 };
    let name: Vec<u16> = r.name.encode_utf16().collect();
    let h = 64 + 4 * n + float_size;
    let header2_size = if r.no_header2 { 0 } else { 64 };
    let name_at = h + header2_size;
    let counters_at = name_at + 2 * name.len();
    let counters_size = r.counters.map_or(0, |c| 4 * c.len());
    let mut d = vec![0u8; counters_at + counters_size];

    d[..4].copy_from_slice(b"Iout");
    put16(&mut d, 4, r.version);
    d[6] = r.kind;
    let left = r.ints.iter().map(|p| p.0).min().unwrap_or(0);
    let top = r.ints.iter().map(|p| p.1).min().unwrap_or(0);
    let right = r.ints.iter().map(|p| p.0).max().unwrap_or(0) + 1;
    let bottom = r.ints.iter().map(|p| p.1).max().unwrap_or(0) + 1;
    put16(&mut d, 8, top as u16);
    put16(&mut d, 10, left as u16);
    put16(&mut d, 12, bottom as u16);
    put16(&mut d, 14, right as u16);
    put16(&mut d, 16, n as u16);
    put16(&mut d, 34, 1); // point size: Small
    put32(&mut d, 40, r.stroke);
    put16(&mut d, 50, r.options);
    // Byte 52, the point type, stays 0: ImageJ's default, Hybrid.
    put32(&mut d, 56, r.position as u32);
    if !r.no_header2 {
        put32(&mut d, 60, h as u32);
    }
    for (i, &(x, y)) in r.ints.iter().enumerate() {
        put16(&mut d, 64 + 2 * i, (x - left) as u16);
        put16(&mut d, 64 + 2 * n + 2 * i, (y - top) as u16);
    }
    if let Some(floats) = r.floats {
        for (i, &(x, y)) in floats.iter().enumerate() {
            put32(&mut d, 64 + 4 * n + 4 * i, x.to_bits());
            put32(&mut d, 64 + 8 * n + 4 * i, y.to_bits());
        }
    }
    if !r.no_header2 {
        let (c, z, t) = r.c_z_t;
        put32(&mut d, h + 4, c as u32);
        put32(&mut d, h + 8, z as u32);
        put32(&mut d, h + 12, t as u32);
        if !name.is_empty() {
            put32(&mut d, h + 16, name_at as u32);
            put32(&mut d, h + 20, name.len() as u32);
        }
        if r.counters.is_some() {
            put32(&mut d, h + 48, counters_at as u32);
        }
    }
    for (i, unit) in name.iter().enumerate() {
        put16(&mut d, name_at + 2 * i, *unit);
    }
    for (i, word) in r.counters.unwrap_or_default().iter().enumerate() {
        put32(&mut d, counters_at + 4 * i, *word as u32);
    }
    d
}

/// A plain integer multi-point ROI — no sub-pixel coordinates, no counters,
/// no stroke colour — as an older ImageJ, or another writer, leaves one: the
/// integer coordinate is ImageJ's pixel centre, so it reads as mid-pixel; the
/// ROI-wide position goes to every point; no colour is ImageJ's yellow.
#[test]
fn a_plain_integer_roi_reads_with_its_slice_and_imagejs_yellow() {
    for version in [227, 228] {
        let d = raw(&Raw {
            version,
            position: 5,
            ints: &[(10, 20), (3, 4), (12, 7)],
            name: "Nuclei",
            ..Default::default()
        });
        let back = point_rois_from_imagej(&d);
        assert_eq!(
            back,
            vec![roi(
                "Nuclei",
                [255, 255, 0],
                vec![pt(10.5, 20.5, 5), pt(3.5, 4.5, 5), pt(12.5, 7.5, 5)]
            )],
            "version {version}"
        );
    }
    // Every slice, no name, no second header at all: still the points.
    let d = raw(&Raw {
        version: 217,
        ints: &[(1, 2), (3, 4)],
        no_header2: true,
        ..Default::default()
    });
    assert_eq!(
        point_rois_from_imagej(&d),
        vec![roi(
            "",
            [255, 255, 0],
            vec![pt(1.5, 2.5, 0), pt(3.5, 4.5, 0)]
        )]
    );
}

/// The integer coordinates are only a fallback: with the sub-pixel option
/// set (and a version that knows it), the floats are what count. Before 222
/// the option means nothing, and the integers are read.
#[test]
fn the_float_coordinates_win_when_the_option_says_so() {
    let ints = [(4i16, 5i16)];
    let floats = [(4.25f32, 5.75f32)];
    let read = |version: u16, options: u16| {
        point_rois_from_imagej(&raw(&Raw {
            version,
            options,
            ints: &ints,
            floats: Some(&floats),
            ..Default::default()
        }))[0]
            .points[0]
    };
    assert_eq!(read(228, 128), pt(4.75, 6.25, 0));
    assert_eq!(read(228, 0), pt(4.5, 5.5, 0), "no option: the integers");
    assert_eq!(read(221, 128), pt(4.5, 5.5, 0), "too old for the option");
}

/// ImageJ's multi-point tool with three counters in one ROI, as it saves it:
/// one `PointRoi` per counter in counter order, named after the ROI and the
/// counter, counter 0 in the stroke colour and the others in ImageJ's.
#[test]
fn a_multi_counter_roi_is_one_roi_per_counter() {
    let ints = [(1i16, 1i16), (2, 2), (3, 3), (4, 4), (5, 5)];
    let floats = [
        (1.0f32, 1.0f32),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 4.0),
        (5.0, 5.0),
    ];
    // (counter | slice << 8): counter 2 first, so that the order is the
    // counters', not the first point's.
    let counters = [2 | (3 << 8), 1 << 8, 1 | (2 << 8), 2 << 8, 2 | (3 << 8)];
    let multi = |name: &'static str, stroke: u32| {
        raw(&Raw {
            options: 128 | 1024,
            position: -2,
            stroke,
            ints: &ints,
            floats: Some(&floats),
            name,
            counters: Some(&counters),
            ..Default::default()
        })
    };
    let back = point_rois_from_imagej(&multi("Cells", 0xFF_12_34_56));
    assert_eq!(
        back,
        vec![
            roi(
                "Cells 0",
                [0x12, 0x34, 0x56],
                vec![pt(2.5, 2.5, 1), pt(4.5, 4.5, 2)]
            ),
            roi("Cells 1", [255, 0, 255], vec![pt(3.5, 3.5, 2)]),
            roi(
                "Cells 2",
                [0, 255, 255],
                vec![pt(1.5, 1.5, 3), pt(5.5, 5.5, 3)]
            ),
        ]
    );
    // No name and no colour: "Counter k", and yellow for counter 0.
    let back = point_rois_from_imagej(&multi("", 0));
    let names: Vec<&str> = back.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, ["Counter 0", "Counter 1", "Counter 2"]);
    assert_eq!(back[0].color, [255, 255, 0]);
}

/// ImageJ's counter colours, and what is done past its fixed ten.
#[test]
fn counters_past_the_tenth_repeat_imagejs_colours() {
    let ints = [(1i16, 1i16), (2, 2), (3, 3), (4, 4)];
    let counters = [0, 3, 9, 13];
    let back = point_rois_from_imagej(&raw(&Raw {
        ints: &ints,
        counters: Some(&counters),
        stroke: 0xFF_00_00_00,
        ..Default::default()
    }));
    let colours: Vec<[u8; 3]> = back.iter().map(|r| r.color).collect();
    assert_eq!(
        colours,
        [[0, 0, 0], [255, 200, 0], [192, 192, 192], [255, 200, 0]],
        "black (set, so kept), orange, light grey, then 13 as 3"
    );
}

/// One counter that is not 0 is still one ROI: its own name and colour.
/// A counter index ImageJ does not count (100 and up) is counter 0, which
/// can make two counters one.
#[test]
fn a_single_counter_keeps_the_rois_name_and_colour() {
    let ints = [(1i16, 1i16), (2, 2)];
    let read = |counters: &[i32]| {
        point_rois_from_imagej(&raw(&Raw {
            ints: &ints,
            counters: Some(counters),
            name: "Only",
            stroke: 0xFF_01_02_03,
            ..Default::default()
        }))
    };
    let back = read(&[4 | (7 << 8), 4]);
    assert_eq!(
        back,
        vec![roi(
            "Only",
            [1, 2, 3],
            vec![pt(1.5, 1.5, 7), pt(2.5, 2.5, 0)]
        )]
    );
    let back = read(&[0, 200 | (2 << 8)]);
    assert_eq!(back.len(), 1, "200 is counted as counter 0: {back:?}");
    assert_eq!(back[0].points[1].position, 2);
}

/// A point's slice: its counter word's when that is not 0, else the ROI's
/// when that is a slice, else every slice. A hyperstack position alone is
/// not converted (see `point_rois_from_imagej`).
#[test]
fn a_points_position_comes_from_its_counter_then_the_roi() {
    let ints = [(1i16, 1i16), (2, 2), (3, 3)];
    let positions = |r: &Raw| -> Vec<u32> {
        point_rois_from_imagej(&raw(r))[0]
            .points
            .iter()
            .map(|p| p.position)
            .collect()
    };
    let read = |position: i32, counters: Option<&[i32]>, c_z_t| {
        positions(&Raw {
            position,
            ints: &ints,
            counters,
            c_z_t,
            ..Default::default()
        })
    };
    let words = [4 << 8, 0, 9 << 8];
    assert_eq!(read(0, Some(&words), (0, 0, 0)), [4, 0, 9]);
    assert_eq!(read(6, Some(&words), (0, 0, 0)), [4, 6, 9]);
    assert_eq!(read(-2, Some(&words), (0, 0, 0)), [4, 0, 9]);
    assert_eq!(read(6, None, (0, 0, 0)), [6, 6, 6]);
    assert_eq!(
        read(-2, None, (0, 0, 0)),
        [0, 0, 0],
        "pointwise, no counters"
    );
    // A word with its top bit set shifts to a negative slice: none.
    assert_eq!(read(3, Some(&[-256, 1 << 8, 0]), (0, 0, 0)), [3, 1, 3]);
    assert_eq!(read(0, None, (2, 3, 4)), [0, 0, 0], "C/Z/T alone");
    // Before 227 there are no counters to read.
    let old = Raw {
        version: 226,
        position: 2,
        ints: &ints,
        counters: Some(&words),
        ..Default::default()
    };
    assert_eq!(positions(&old), [2, 2, 2]);
}

/// Any other shape is not a point ROI, and reads as none.
#[test]
fn a_roi_of_another_shape_is_no_points() {
    for kind in [0u8, 1, 2, 3, 4, 5, 7, 8, 9, 11, 255] {
        let d = raw(&Raw {
            kind,
            ints: &[(1, 1), (5, 1), (5, 5)],
            name: "shape",
            ..Default::default()
        });
        assert!(point_rois_from_imagej(&d).is_empty(), "type {kind}");
    }
    assert!(point_rois_from_imagej(b"").is_empty());
    assert!(point_rois_from_imagej(&[0u8; 200]).is_empty());
    // No points at all: nothing to read, as `RoiDecoder` makes nothing of it.
    let empty = raw(&Raw::default());
    assert!(point_rois_from_imagej(&empty).is_empty());
}

/// Cut short anywhere — header, coordinates, second header, name, counters —
/// a ROI reads as nothing rather than as a part of itself.
#[test]
fn a_truncated_roi_reads_as_nothing_at_every_length() {
    let ints = [(10i16, 20i16), (3, 4)];
    let floats = [(10.0f32, 20.0f32), (3.25, 4.75)];
    let counters = [1 << 8, 1 | (2 << 8)];
    let samples = [
        roi(
            "Cells",
            [1, 2, 3],
            vec![pt(10.5, 20.5, 4), pt(2.25, 3.75, 9)],
        )
        .to_imagej_bytes(),
        roi("", [1, 2, 3], vec![pt(1.5, 1.5, 0)]).to_imagej_bytes(),
        raw(&Raw {
            version: 227,
            ints: &ints,
            name: "Nuclei",
            ..Default::default()
        }),
        raw(&Raw {
            ints: &ints,
            ..Default::default()
        }),
        raw(&Raw {
            version: 217,
            ints: &ints,
            no_header2: true,
            ..Default::default()
        }),
        raw(&Raw {
            options: 128,
            ints: &ints,
            floats: Some(&floats),
            name: "multi",
            counters: Some(&counters),
            ..Default::default()
        }),
    ];
    for (s, bytes) in samples.iter().enumerate() {
        assert!(
            !point_rois_from_imagej(bytes).is_empty(),
            "sample {s} whole"
        );
        for len in 0..bytes.len() {
            assert!(
                point_rois_from_imagej(&bytes[..len]).is_empty(),
                "sample {s} cut to {len} of {}",
                bytes.len()
            );
        }
    }
}

/// Untrusted bytes: every byte of a real ROI set to values that make lengths
/// and offsets huge, negative or zero. Whatever comes back, nothing panics,
/// and nothing holds more points than the bytes could describe.
#[test]
fn a_damaged_roi_never_panics() {
    let ints = [(10i16, 20i16), (3, 4)];
    let floats = [(10.0f32, 20.0f32), (3.25, 4.75)];
    let counters = [1 << 8, 1 | (2 << 8)];
    let samples = [
        roi(
            "Cells",
            [1, 2, 3],
            vec![pt(10.5, 20.5, 4), pt(2.25, 3.75, 9)],
        )
        .to_imagej_bytes(),
        raw(&Raw {
            options: 128,
            ints: &ints,
            floats: Some(&floats),
            name: "multi",
            counters: Some(&counters),
            ..Default::default()
        }),
    ];
    for bytes in &samples {
        for at in 0..bytes.len() {
            for v in [0x00, 0x01, 0x7F, 0x80, 0xFF] {
                let mut d = bytes.clone();
                d[at] = v;
                for r in point_rois_from_imagej(&d) {
                    assert!(r.points.len() <= d.len() / 4);
                    assert!(r.points.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
                }
            }
        }
    }
    // A NaN coordinate is not a place.
    let nan = [(f32::NAN, 1.0f32)];
    let d = raw(&Raw {
        options: 128,
        ints: &[(0, 1)],
        floats: Some(&nan),
        ..Default::default()
    });
    assert!(point_rois_from_imagej(&d).is_empty());
}

/// The hyperstack position is the second header's C/Z/T, and comes back as it
/// went, on every ROI split from it by counter.
#[test]
fn the_hyperstack_position_is_in_the_second_header() {
    let mut r = roi("H", [1, 2, 3], vec![pt(1.5, 1.5, 3), pt(2.5, 2.5, 3)]);
    r.hyperstack_position = [0, 1, 2];
    let d = r.to_imagej_bytes();
    let h2 = get_int(&d, 60) as usize;
    assert_eq!(
        [
            get_int(&d, h2 + 4),
            get_int(&d, h2 + 8),
            get_int(&d, h2 + 12)
        ],
        [0, 1, 2]
    );
    assert_eq!(point_rois_from_imagej(&d), vec![r.clone()]);
    // None at all is zeros, as before.
    let plain = roi("P", [0; 3], vec![pt(1.5, 1.5, 0)]).to_imagej_bytes();
    let h2 = get_int(&plain, 60) as usize;
    assert_eq!(&plain[h2 + 4..h2 + 16], &[0; 12]);
}

/// More ROIs than a zip without Zip64 holds is an error to report, not a
/// panic.
#[test]
fn too_many_rois_for_a_roi_set_is_an_error() {
    let one = roi("A", [0; 3], vec![pt(1.0, 1.0, 0)]);
    let rois = vec![one; 65_536];
    assert_eq!(roi_set_zip(&rois), Err(RoiSetTooLarge));
    assert!(roi_set_zip(&rois[..65_535]).is_ok());
}

/// Many ROIs of one name are named apart in linear time, every name once.
#[test]
fn many_clashing_names_are_made_unique_quickly() {
    let names = vec!["Cell".to_string(); 30_000];
    let started = std::time::Instant::now();
    let unique = unique_entry_names(names);
    assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
    assert_eq!(unique[0], "Cell");
    assert_eq!(unique[1], "Cell-1");
    assert_eq!(unique[29_999], "Cell-29999");
    let set: std::collections::HashSet<&String> = unique.iter().collect();
    assert_eq!(set.len(), unique.len());
}

/// A name that ends in `.roi` and one that is the same without it are two
/// entries, not one written twice.
#[test]
fn a_name_ending_in_roi_does_not_clash_into_one_entry() {
    let rois = vec![
        roi("A.roi", [0; 3], vec![pt(1.0, 1.0, 0)]),
        roi("A", [0; 3], vec![pt(1.0, 1.0, 0)]),
    ];
    let names: Vec<String> = read_zip(&roi_set_zip(&rois).unwrap())
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(names, ["A.roi", "A-1.roi"]);
}

//! The public API, used the way a caller outside the crate uses it.

#[path = "../src/fixture.rs"]
mod fixture;

use fixture::Scratch;
use oir_reader::{
    Access, Acquisition, BlockSource, Error, ErrorKind, Gathered, Options, Planes, ReadPath,
    SampleType,
};

#[test]
fn results_and_errors_can_cross_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Acquisition>();
    send_sync::<Gathered>();
    send_sync::<Error>();
    send_sync::<Options>();
}

#[test]
fn the_signature_is_what_makes_an_oir() {
    assert!(oir_reader::is_oir(b"OLYMPUSRAWFORMAT\0\0\0"));
    assert!(!oir_reader::is_oir(b"OLYMPUSRAWFORMA"));
    assert!(!oir_reader::is_oir(b"II*\0"));
    assert!(!oir_reader::is_oir(b""));
}

#[test]
fn mmap_supported_says_what_the_build_does() {
    assert_eq!(
        oir_reader::MMAP_SUPPORTED,
        cfg!(all(feature = "mmap", any(unix, windows)))
    );
}

/// Planes come in xyczt order, and `plane_index` finds them there.
#[test]
fn plane_index_is_channel_fastest_then_slice_then_time() {
    let names: Vec<String> = (1..=3)
        .flat_map(|z| ["uidA", "uidB"].map(move |c| format!("z{z}_0_1_{c}")))
        .collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let f = fixture::channels_file(4, 4, &refs, 16, 2);
    let a = oir_reader::read_bytes(&[&f]).expect("read");
    let s = a.shape();
    assert_eq!((s.channels, s.slices, s.frames), (2, 3, 1));
    assert_eq!(s.plane_len(), 16);
    assert_eq!(s.plane_count(), 6);
    assert_eq!(s.sample_type, SampleType::U16);
    assert_eq!(s.sample_type.bytes(), 2);
    let planes = a.planes().as_u16().unwrap();
    for z in 0..3 {
        for c in 0..2 {
            let i = s.plane_index(c, z, 0).unwrap();
            // The fixture writes plane p with first pixel p * 1000, in the
            // order z1 A, z1 B, z2 A, …
            assert_eq!(planes[i][0] as usize, (2 * z + c) * 1000, "c{c} z{z}");
        }
    }
    assert_eq!(s.plane_index(0, 0, 1), None);
}

#[test]
fn the_parts_of_a_split_acquisition_are_found_beside_the_first() {
    let dir = Scratch::new("api-parts");
    let first = dir.write("rec.oir", &fixture::part_file(0, 2, true));
    let second = dir.write("rec_00001", &fixture::part_file(2, 2, false));
    let third = dir.write("rec_00002.oir", &fixture::part_file(4, 1, false));
    // A gap: this one is not part of the set.
    dir.write("rec_00004.oir", &fixture::part_file(5, 1, false));
    let parts = oir_reader::acquisition_parts(&first);
    assert_eq!(parts, [first.clone(), second.clone(), third.clone()]);
    // A continuation stands alone.
    assert_eq!(
        oir_reader::acquisition_parts(&second),
        std::slice::from_ref(&second)
    );
    // And a file that is not there is still the first of its own set.
    let missing = dir.path().join("missing.oir");
    assert_eq!(
        oir_reader::acquisition_parts(&missing),
        std::slice::from_ref(&missing)
    );

    // Read from the files and from memory, the same acquisition.
    let from_files = oir_reader::read(&first).expect("read");
    let bytes: Vec<Vec<u8>> = parts.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let from_memory = oir_reader::read_bytes(&refs).expect("read");
    assert_eq!(from_files.shape(), from_memory.shape());
    assert_eq!(from_files.planes(), from_memory.planes());
    assert_eq!(from_files.record(), from_memory.record());
    assert_eq!(from_files.shape().frames, 5);

    let want = if oir_reader::MMAP_SUPPORTED {
        ReadPath::Mapped
    } else {
        ReadPath::Buffered
    };
    for (part, path) in from_files.parts().iter().zip(&parts) {
        assert_eq!(part.read_path(), want);
        assert_eq!(part.path(), Some(path.as_path()));
        assert_eq!(part.block_source(), BlockSource::Index);
    }
    assert!(from_memory
        .parts()
        .iter()
        .all(|p| p.read_path() == ReadPath::Memory && p.path().is_none()));
    let buffered = Options::new()
        .access(Access::Buffered)
        .gather(&first, |_| true)
        .and_then(Gathered::assemble)
        .expect("read");
    assert!(buffered
        .parts()
        .iter()
        .all(|p| p.read_path() == ReadPath::Buffered && p.map_error().is_none()));
    assert_eq!(buffered.planes(), from_files.planes());
}

#[test]
fn a_missing_file_is_an_io_error_that_names_it() {
    let dir = Scratch::new("api-missing");
    let path = dir.path().join("nothing.oir");
    let err = oir_reader::read(&path).expect_err("no file");
    assert_eq!(err.kind(), ErrorKind::Io);
    assert_eq!(err.path(), Some(path.as_path()));
    let cause = std::error::Error::source(&err).expect("the operating system's reason");
    assert!(!cause.to_string().is_empty());
    // The message is the reader's; the reason is the source's.
    assert_eq!(err.to_string(), "could not open the file");
}

#[test]
fn a_file_that_is_not_an_oir_says_so() {
    let err = oir_reader::read_bytes(&[b"II*\0 a TIFF"]).expect_err("not an OIR");
    assert_eq!(err.kind(), ErrorKind::NotOir);
    assert!(err.path().is_none());
    assert!(std::error::Error::source(&err).is_none());
}

/// The enums say they may grow, so a caller's match needs a wildcard — and one
/// with a wildcard compiles.
#[test]
fn every_open_ended_enum_can_be_matched() {
    let a = oir_reader::read_bytes(&[&fixture::recorded_file()]).expect("read");
    let depth = match a.planes() {
        Planes::U8(_) => 8,
        Planes::U16(_) => 16,
        _ => 0,
    };
    assert_eq!(depth, 16);
    let sample = match a.shape().sample_type {
        SampleType::U8 => "u8",
        SampleType::U16 => "u16",
        _ => "?",
    };
    let read = match a.parts()[0].read_path() {
        ReadPath::Mapped => "mapped",
        ReadPath::Buffered => "buffered",
        ReadPath::Memory => "memory",
        _ => "?",
    };
    let blocks = match a.parts()[0].block_source() {
        BlockSource::Index => "index",
        BlockSource::Walk => "walk",
        _ => "?",
    };
    let access = match Access::default() {
        Access::Auto => "auto",
        Access::Buffered => "buffered",
        _ => "?",
    };
    let err = oir_reader::read_bytes(&[&b""[..]]).expect_err("empty");
    let kind = match err.kind() {
        ErrorKind::Io => "io",
        ErrorKind::NotOir => "not an OIR",
        ErrorKind::Malformed => "malformed",
        ErrorKind::NoPlanes => "no planes",
        ErrorKind::Cancelled => "cancelled",
        _ => "?",
    };
    assert_eq!(
        [sample, read, blocks, access, kind],
        ["u16", "memory", "index", "auto", "not an OIR"]
    );
}

#[test]
fn the_planes_can_be_taken_without_a_copy() {
    let a = oir_reader::read_bytes(&[&fixture::recorded_file()]).expect("read");
    let before = a.planes().as_u16().unwrap()[0].as_ptr();
    let Planes::U16(planes) = a.into_planes() else {
        panic!("16-bit");
    };
    assert_eq!(planes[0].as_ptr(), before);
}

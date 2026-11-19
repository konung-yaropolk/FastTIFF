//! What kind of failure each kind of bad input is, through the public API.
//!
//! A caller decides on [`ErrorKind`], never on the words, so each kind is
//! pinned here on a failure that is unmistakably it. The words are pinned
//! too, where callers have been known to match on them, and the cause chain
//! is checked as `anyhow`'s `{:#}` would print it: written out from
//! `source()`, so that this crate needs no dev-dependency on anyhow.

use fast_tiff_lib::{
    read_frame_f32, read_frame_u16, read_plane_u8, Compression, Error, ErrorKind, FrameInfo,
    SampleFormat, SampleType, StackMetaWrite, TiffStack, TiffWriter, WriterOptions,
};
use std::io::{self, Cursor, Seek, SeekFrom, Write};

/// The chain as `anyhow`'s `{:#}` prints it: each link's `{}`, joined by ": ".
fn chain_text(e: &(dyn std::error::Error + 'static)) -> String {
    let mut parts = vec![e.to_string()];
    let mut next = e.source();
    while let Some(link) = next {
        parts.push(link.to_string());
        next = link.source();
    }
    parts.join(": ")
}

/// A small valid stack, written in memory.
fn stack_bytes(compression: Compression) -> Vec<u8> {
    let opts = WriterOptions::new(6, 4, SampleType::U16).compression(compression);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..2u16 {
        w.write_frame_u16(&[f; 24]).unwrap();
    }
    w.finish().unwrap().into_inner()
}

/// The stack with its Compression entry (tag 259, SHORT, one value, 1) set to
/// `code`, which the writer itself would refuse.
fn with_compression(code: u16) -> Vec<u8> {
    let mut bytes = stack_bytes(Compression::None);
    let none = [0x03u8, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00];
    let mut at = 0;
    while let Some(i) = bytes[at..].windows(none.len()).position(|w| w == none) {
        let i = at + i;
        bytes[i + 8..i + 10].copy_from_slice(&code.to_le_bytes());
        at = i + none.len();
    }
    assert!(at > 0, "no Compression entry found");
    bytes
}

fn open_err(bytes: Vec<u8>) -> Error {
    match TiffStack::from_bytes(bytes) {
        Ok(_) => panic!("opened"),
        Err(e) => e,
    }
}

#[test]
fn bytes_of_another_format_are_not_a_tiff() {
    let e = open_err(b"%PDF-1.7 this is not a TIFF at all".to_vec());
    assert_eq!(e.kind(), ErrorKind::NotTiff);
    assert_eq!(e.to_string(), "not a TIFF");

    // The byte order reads, the magic does not.
    let e = open_err(b"II\x29\x00\x08\x00\x00\x00 and then some".to_vec());
    assert_eq!(e.kind(), ErrorKind::NotTiff);
    assert_eq!(e.to_string(), "not a TIFF (magic 41; expected 42 or 43)");
}

#[test]
fn a_truncated_file_is_malformed() {
    let good = stack_bytes(Compression::None);
    for len in [0, 5, 20, good.len() / 2, good.len() - 3] {
        let e = open_err(good[..len].to_vec());
        assert_eq!(e.kind(), ErrorKind::Malformed, "{len} bytes: {e:#}");
    }
    // A broken BigTIFF header is a broken TIFF, not another format.
    let e = open_err(b"II\x2b\x00\x04\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec());
    assert_eq!(e.kind(), ErrorKind::Malformed);
    assert_eq!(
        e.to_string(),
        "malformed BigTIFF header (offset size 4, reserved 0)"
    );

    // A directory offset past the end is damage, and the same damage on every
    // target: past `usize` on a 32-bit one, where it cannot be converted, is
    // still past the end of bytes whose length is a `usize`.
    let mut far = b"II\x2b\x00\x08\x00\x00\x00".to_vec();
    far.extend_from_slice(&(1u64 << 40).to_le_bytes());
    far.resize(64, 0);
    let e = open_err(far);
    assert_eq!(e.kind(), ErrorKind::Malformed);
    assert_eq!(e.to_string(), "IFD offset 1099511627776 out of bounds");
}

#[test]
fn a_compression_this_library_has_no_code_for_is_unsupported() {
    // 7 is JPEG: a TIFF anyone may write, and one this library does not read.
    let stack = TiffStack::from_bytes(with_compression(7)).expect("it indexes");
    let f = &stack.frames[0];
    assert_eq!(f.compression, Compression::Other(7));
    let e = read_frame_u16(&stack.data, f, stack.byte_order, None).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Unsupported);
    assert_eq!(e.to_string(), "unsupported TIFF compression scheme: 7");
}

#[test]
fn a_sample_format_this_library_does_not_read_is_unsupported() {
    let frame = FrameInfo {
        width: 2,
        height: 2,
        bits_per_sample: 16,
        samples_per_pixel: 1,
        sample_format: SampleFormat::Float,
        compression: Compression::None,
        predictor: 1,
        photometric: 1,
        planar_config: 1,
        tile_size: None,
        ink_set: 1,
        strip_offsets: vec![0].into(),
        strip_byte_counts: vec![8].into(),
        rows_per_strip: 2,
    };
    let data = [0u8; 8];
    let e = read_frame_f32(&data, &frame, fast_tiff_lib::ByteOrder::Little).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Unsupported);
    assert_eq!(
        e.to_string(),
        "read_plane_f32 does not read 16-bit floating-point samples"
    );
    let twelve = FrameInfo {
        bits_per_sample: 12,
        sample_format: SampleFormat::UnsignedInt,
        ..frame
    };
    let e = read_frame_u16(&data, &twelve, fast_tiff_lib::ByteOrder::Little, None).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Unsupported);
    assert_eq!(e.to_string(), "unsupported bits_per_sample: 12");
}

#[test]
fn zstd_is_unavailable_only_without_its_feature() {
    let stack = TiffStack::from_bytes(with_compression(50000)).expect("it indexes");
    let got = read_frame_u16(&stack.data, &stack.frames[0], stack.byte_order, None);
    let write = TiffWriter::new(
        Cursor::new(Vec::new()),
        WriterOptions::new(2, 2, SampleType::U8).compression(Compression::Zstd),
    );
    if cfg!(feature = "codec-zstd") {
        // Uncompressed bytes are not a ZSTD stream: the build has the codec,
        // so it is the data that is wrong.
        assert_eq!(got.unwrap_err().kind(), ErrorKind::Malformed);
        assert!(write.is_ok());
    } else {
        let e = got.unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Unavailable);
        assert!(e.to_string().contains("`codec-zstd`"), "{e}");
        // Refused when the writer is made, before a frame is written.
        let e = write.err().expect("refused");
        assert_eq!(e.kind(), ErrorKind::Unavailable);
        assert!(e.to_string().contains("`codec-zstd`"), "{e}");
    }
}

/// The operating system's failure is `Io`, says what was being done, and
/// keeps the `io::Error` as its source — `{:#}` reading exactly as the one
/// message it used to be.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[test]
fn a_missing_file_is_io_with_the_os_error_as_its_source() {
    let dir = std::env::temp_dir().join(format!("FastTIFF_no_such_dir_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("x.tif");

    let e = match TiffWriter::create(&path, WriterOptions::new(2, 2, SampleType::U8)) {
        Ok(_) => panic!("created a file in a directory that does not exist"),
        Err(e) => e,
    };
    let why = std::fs::File::create(&path).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Io);
    assert_eq!(
        e.to_string(),
        format!("could not create {}", path.display())
    );
    let old = format!("could not create {}: {why}", path.display());
    assert_eq!(chain_text(&e), old);
    assert_eq!(format!("{e:#}"), old);
    let io = std::error::Error::source(&e)
        .and_then(|s| s.downcast_ref::<std::io::Error>())
        .expect("the io::Error is the source");
    assert_eq!(io.kind(), why.kind());

    #[cfg(feature = "mmap")]
    {
        let e = TiffStack::open(&path).err().expect("opened a missing file");
        let why = std::fs::File::open(&path).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Io);
        let old = format!("could not open {}: {why}", path.display());
        assert_eq!(chain_text(&e), old);
        assert_eq!(format!("{e:#}"), old);
    }
}

/// A sink with room for `room` bytes, which then fails as a full disk does.
struct Full {
    room: usize,
    bytes: Cursor<Vec<u8>>,
}

impl Full {
    fn new(room: usize) -> Self {
        Full {
            room,
            bytes: Cursor::new(Vec::new()),
        }
    }
}

impl Write for Full {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.room == 0 {
            return Err(io::Error::new(io::ErrorKind::StorageFull, "disk full"));
        }
        let n = buf.len().min(self.room);
        self.room -= n;
        self.bytes.write(&buf[..n])
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for Full {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        self.bytes.seek(to)
    }
}

/// A write the operating system fails part-way through is `Io`, reads as
/// what the system said, and hands back the `io::Error` itself — so a
/// program can tell a full disk from a refused permission — wherever the
/// writer meets it: the header, a frame, or the directory at the end.
#[test]
fn a_failed_write_keeps_its_io_error() {
    let full = |e: Error| {
        assert_eq!(e.kind(), ErrorKind::Io, "{e:#}");
        assert_eq!(format!("{e:#}"), "disk full");
        let io = e.io_error().expect("the io::Error");
        assert_eq!(io.kind(), io::ErrorKind::StorageFull);
    };
    let opts = || WriterOptions::new(6, 4, SampleType::U16);

    // No room for the header.
    full(
        TiffWriter::new(Full::new(0), opts())
            .err()
            .expect("refused"),
    );

    // Room for the header, not the frame.
    let mut w = TiffWriter::new(Full::new(16), opts()).unwrap();
    full(w.write_frame_u16(&[1; 24]).unwrap_err());

    // Room for the frame, not the directory.
    let mut w = TiffWriter::new(Full::new(16 + 48), opts()).unwrap();
    w.write_frame_u16(&[1; 24]).unwrap();
    full(w.finish().err().expect("refused"));
}

#[test]
fn misusing_the_writer_is_invalid_input() {
    let new = |o: WriterOptions| TiffWriter::new(Cursor::new(Vec::new()), o).err();
    let kind = |e: Option<Error>| e.expect("refused").kind();

    assert_eq!(
        kind(new(WriterOptions::new(0, 4, SampleType::U16))),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        kind(new(
            WriterOptions::new(4, 4, SampleType::U16).description("a\0b")
        )),
        ErrorKind::InvalidInput
    );
    let e =
        new(WriterOptions::new(4, 4, SampleType::U16)
            .metadata(StackMetaWrite::new(1, 1).unit("u\0m")))
        .expect("refused");
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        e.to_string(),
        "unit must not contain NUL bytes (TIFF ASCII fields are NUL-terminated)"
    );

    let mut w = TiffWriter::new(
        Cursor::new(Vec::new()),
        WriterOptions::new(4, 4, SampleType::U16),
    )
    .unwrap();
    assert_eq!(
        w.write_frame_u8(&[0; 16]).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        w.write_frame_u16(&[0; 15]).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let e = w.finish().expect_err("an empty stack is refused");
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        e.to_string(),
        "no frames were written — a TIFF needs at least one image"
    );

    // A reader for a depth the frame does not have.
    let stack = TiffStack::from_bytes(stack_bytes(Compression::None)).unwrap();
    let e = read_plane_u8(&stack.data, &stack.frames[0], stack.byte_order, 0).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);

    // 4-bit samples, which the 8-bit and float readers give as they are and
    // the 16-bit display reader does not widen: the wrong reader, too.
    let four = FrameInfo {
        width: 4,
        height: 2,
        bits_per_sample: 4,
        samples_per_pixel: 1,
        sample_format: SampleFormat::UnsignedInt,
        compression: Compression::None,
        predictor: 1,
        photometric: 3,
        planar_config: 1,
        tile_size: None,
        ink_set: 1,
        strip_offsets: vec![0].into(),
        strip_byte_counts: vec![4].into(),
        rows_per_strip: 2,
    };
    let data = [0x01u8, 0x23, 0x45, 0x67];
    let order = fast_tiff_lib::ByteOrder::Little;
    assert_eq!(
        &read_plane_u8(&data, &four, order, 0).unwrap()[..],
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );
    let e = read_frame_u16(&data, &four, order, None).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        e.to_string(),
        "read_plane_u16 does not read 4-bit samples; read_plane_u8 and read_plane_f32 give \
         them as they are"
    );
}

#[test]
fn a_size_past_the_address_space_is_too_large() {
    // 2147483648 x 2147483648 x 4 samples is one past usize::MAX on 64-bit.
    let frame = FrameInfo {
        width: 2_147_483_648,
        height: 2_147_483_648,
        bits_per_sample: 16,
        samples_per_pixel: 4,
        sample_format: SampleFormat::UnsignedInt,
        compression: Compression::None,
        predictor: 1,
        photometric: 1,
        planar_config: 1,
        tile_size: None,
        ink_set: 1,
        strip_offsets: vec![8].into(),
        strip_byte_counts: vec![16].into(),
        rows_per_strip: 1,
    };
    let e = frame.sample_count().unwrap_err();
    assert_eq!(e.kind(), ErrorKind::TooLarge);
    assert_eq!(
        e.to_string(),
        "frame geometry overflows address space: 2147483648x2147483648 x 4 sample(s)/px"
    );
    let e = read_frame_u16(&[0u8; 64], &frame, fast_tiff_lib::ByteOrder::Little, None).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::TooLarge);

    // The writer asked for such a frame says the same.
    let e = TiffWriter::new(
        Cursor::new(Vec::new()),
        WriterOptions::new(u32::MAX, u32::MAX, SampleType::F64).samples_per_pixel(u16::MAX),
    )
    .err()
    .expect("refused");
    assert_eq!(e.kind(), ErrorKind::TooLarge);
    assert!(e.to_string().contains("overflows address space"), "{e}");
}

/// The error goes into the usual containers by `?`, and the kind survives a
/// trip through `Box<dyn Error>`, which is how an application finds it again.
#[test]
fn the_kind_is_found_again_behind_a_box() {
    fn open(bytes: Vec<u8>) -> Result<TiffStack, Box<dyn std::error::Error + Send + Sync>> {
        Ok(TiffStack::from_bytes(bytes)?)
    }
    let boxed = open(b"GIF89a and the rest of a picture".to_vec())
        .err()
        .expect("refused");
    let found = boxed
        .downcast_ref::<Error>()
        .expect("the crate's Error, as it was");
    assert_eq!(found.kind(), ErrorKind::NotTiff);
}

//! The C interface against the Rust one it wraps: the same answers on the
//! crate's own fixtures and on stacks written through either, and every
//! refusal a status with a reason rather than a crash.

use super::*;
use crate::{read_frame_u8, StackMetaWrite, TiffWriter, WriterOptions};
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};

fn last_error() -> String {
    unsafe { CStr::from_ptr(ftl_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// Assert `status` is `want`, with a reason on record.
#[track_caller]
fn refused(status: c_int, want: c_int) {
    assert_eq!(
        status,
        want,
        "status {status}, wanted {want}: {}",
        last_error()
    );
    assert!(!last_error().is_empty(), "a refusal says why");
}

fn fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("tests/fixtures")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "tif"))
        .collect();
    paths.sort();
    assert!(paths.len() > 20, "the fixture matrix is there");
    paths
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn c_path(p: &Path) -> CString {
    CString::new(p.to_str().unwrap()).unwrap()
}

/// A scratch path no other test uses.
fn temp(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ftl_capi_{}_{nanos}_{name}", std::process::id()))
}

/// Open through C: mapped where this build can, from bytes where it cannot.
fn open_c(path: &Path) -> *mut FtlStack {
    let mut s = null_mut();
    #[cfg(feature = "mmap")]
    let status = unsafe { ftl_open(c_path(path).as_ptr(), &mut s) };
    #[cfg(not(feature = "mmap"))]
    let status = {
        let bytes = std::fs::read(path).unwrap();
        unsafe { ftl_open_bytes(bytes.as_ptr().cast(), bytes.len(), &mut s) }
    };
    assert_eq!(status, FTL_OK, "{}: {}", path.display(), last_error());
    assert!(!s.is_null());
    s
}

fn open_c_bytes(bytes: &[u8]) -> *mut FtlStack {
    let mut s = null_mut();
    let status = unsafe { ftl_open_bytes(bytes.as_ptr().cast(), bytes.len(), &mut s) };
    assert_eq!(status, FTL_OK, "{}", last_error());
    s
}

fn open_rust(path: &Path) -> TiffStack {
    TiffStack::from_bytes(std::fs::read(path).unwrap()).unwrap()
}

fn flat<T: Clone>(planes: Vec<Vec<T>>) -> Vec<T> {
    planes.concat()
}

fn description(s: *const FtlStack) -> String {
    let n = unsafe { ftl_description(s, null_mut(), 0) };
    let mut buf = vec![0u8; n + 1];
    assert_eq!(
        unsafe { ftl_description(s, buf.as_mut_ptr().cast(), buf.len()) },
        n
    );
    assert_eq!(buf[n], 0);
    String::from_utf8(buf[..n].to_vec()).unwrap()
}

fn unit(s: *const FtlStack) -> String {
    let mut buf = [0u8; 64];
    let n = unsafe { ftl_metadata_unit(s, buf.as_mut_ptr().cast(), buf.len()) };
    String::from_utf8(buf[..n].to_vec()).unwrap()
}

fn metadata(s: *const FtlStack) -> FtlMetadata {
    let mut m = std::mem::MaybeUninit::uninit();
    assert_eq!(unsafe { ftl_metadata(s, m.as_mut_ptr()) }, FTL_OK);
    unsafe { m.assume_init() }
}

/// Two `FtlMetadata` are the same, NaN for NaN.
#[track_caller]
fn same_metadata(a: &FtlMetadata, b: &FtlMetadata) {
    let bits = |m: &FtlMetadata| {
        [
            m.pixel_width,
            m.pixel_height,
            m.spacing,
            m.calibration[0],
            m.calibration[1],
            m.fps,
            m.frame_interval_s,
        ]
        .map(f64::to_bits)
    };
    assert_eq!(
        (
            a.channels,
            a.slices,
            a.frames,
            a.format,
            a.mode,
            a.loop_playback,
            bits(a)
        ),
        (
            b.channels,
            b.slices,
            b.frames,
            b.format,
            b.mode,
            b.loop_playback,
            bits(b)
        )
    );
}

/// What `read` writes for frame `i`, or its status.
fn read_c<T: Copy + Default>(
    read: impl Fn(*mut T, usize) -> c_int,
    len: usize,
) -> Result<Vec<T>, c_int> {
    let mut out = vec![T::default(); len];
    match read(out.as_mut_ptr(), out.len()) {
        FTL_OK => Ok(out),
        status => Err(status),
    }
}

/// Every reader on every frame of every fixture: what C hands back is what
/// the Rust readers do, and where Rust refuses, so does C, with a reason.
#[test]
fn every_fixture_reads_through_c_as_through_rust() {
    for path in fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let rust = open_rust(&path);
        let bytes = std::fs::read(&path).unwrap();
        for s in [open_c(&path), open_c_bytes(&bytes)] {
            assert_eq!(unsafe { ftl_frame_count(s) }, rust.frames.len(), "{name}");
            assert_eq!(
                unsafe { ftl_stack_is_bigtiff(s) },
                rust.flavor == TiffFlavor::Big
            );
            assert_eq!(
                unsafe { ftl_stack_is_big_endian(s) },
                rust.byte_order == ByteOrder::Big
            );
            assert_eq!(description(s), rust.description.clone().unwrap_or_default());
            same_metadata(&metadata(s), &FtlMetadata::from(&rust.meta));
            assert_eq!(unit(s), rust.meta.unit.clone().unwrap_or_default());

            for (i, f) in rust.frames.iter().enumerate() {
                let what = format!("{name} frame {i}");
                let mut info = std::mem::MaybeUninit::uninit();
                assert_eq!(unsafe { ftl_frame_info(s, i, info.as_mut_ptr()) }, FTL_OK);
                let info = unsafe { info.assume_init() };
                assert_eq!(info, FtlFrameInfo::from(f));
                assert_eq!(
                    (
                        info.width,
                        info.height,
                        info.bits_per_sample,
                        info.samples_per_pixel
                    ),
                    (f.width, f.height, f.bits_per_sample, f.samples_per_pixel)
                );
                assert_eq!(
                    (info.tiled, info.cmyk, info.rgb),
                    (f.is_tiled(), f.is_cmyk(), f.is_rgb())
                );
                let mut n = 0;
                assert_eq!(unsafe { ftl_sample_count(s, i, &mut n) }, FTL_OK);
                assert_eq!(n, f.sample_count().unwrap());
                let (data, order) = (&rust.data[..], rust.byte_order);

                // u16, from each plane's own range.
                let got = read_c(|o, l| unsafe { ftl_read_u16(s, i, null(), o, l) }, n);
                match read_planes_u16(data, f, order, None) {
                    Ok(want) => assert_eq!(got.as_deref(), Ok(&flat(want)[..]), "{what}"),
                    Err(_) => refused(got.clone().unwrap_err(), FTL_FORMAT),
                }
                if f.samples_per_pixel == 1 {
                    if let Ok(want) = read_frame_u16(data, f, order, None) {
                        assert_eq!(got.as_deref(), Ok(&want[..]), "{what}");
                    }
                }

                // u8: 8-bit (and 4-bit) only.
                let got = read_c(|o, l| unsafe { ftl_read_u8(s, i, o, l) }, n);
                match f.bits_per_sample {
                    8 => assert_eq!(
                        got.as_deref(),
                        Ok(&flat(read_planes_u8(data, f, order).unwrap())[..]),
                        "{what}"
                    ),
                    4 => assert_eq!(
                        got.as_deref(),
                        Ok(&read_frame_u8(data, f, order).unwrap()[..]),
                        "{what}"
                    ),
                    _ => refused(got.unwrap_err(), FTL_FORMAT),
                }

                // f32: 32- and 64-bit only. Compared bit for bit, NaN and all.
                let got = read_c(|o, l| unsafe { ftl_read_f32(s, i, o, l) }, n);
                match read_planes_f32(data, f, order) {
                    Ok(want) => {
                        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(&got.unwrap()), bits(&flat(want)), "{what}");
                    }
                    Err(_) => refused(got.unwrap_err(), FTL_FORMAT),
                }

                // CMYK to RGB: CMYK only.
                let three = 3 * f.pixel_count().unwrap();
                let got8 = read_c(|o, l| unsafe { ftl_read_rgb_u8(s, i, o, l) }, three);
                let got16 = read_c(|o, l| unsafe { ftl_read_rgb_u16(s, i, o, l) }, three);
                match read_planes_rgb_u8(data, f, order) {
                    Ok(want) => assert_eq!(got8.as_deref(), Ok(&flat(want)[..]), "{what}"),
                    Err(_) => refused(got8.clone().unwrap_err(), FTL_FORMAT),
                }
                match read_planes_rgb_u16(data, f, order) {
                    Ok(want) => assert_eq!(got16.as_deref(), Ok(&flat(want)[..]), "{what}"),
                    Err(_) => refused(got16.clone().unwrap_err(), FTL_FORMAT),
                }
                if f.is_cmyk() {
                    assert!(got16.is_ok(), "{what}: CMYK converts");
                } else {
                    assert!(
                        got8.is_err() && got16.is_err(),
                        "{what}: only CMYK converts"
                    );
                }

                let (mut lo, mut hi) = (0.0f32, 0.0f32);
                assert_eq!(unsafe { ftl_float_minmax(s, i, &mut lo, &mut hi) }, FTL_OK);
                match frame_float_minmax(data, f, order).unwrap() {
                    Some(want) => assert_eq!((lo, hi), want, "{what}"),
                    None => assert!(lo.is_nan() && hi.is_nan(), "{what}"),
                }
            }
            unsafe { ftl_stack_free(s) };
        }
    }
}

/// A fixed range scales a wide frame the way the Rust reader does.
#[test]
fn a_given_range_scales_as_in_rust() {
    let path = fixture("tff_f32_spp1_p2_none-le.tif");
    let rust = open_rust(&path);
    let s = open_c(&path);
    let range = [-1.0f32, 3.5];
    for (i, f) in rust.frames.iter().enumerate() {
        let want =
            read_frame_u16(&rust.data, f, rust.byte_order, Some((range[0], range[1]))).unwrap();
        let got = read_c(
            |o, l| unsafe { ftl_read_u16(s, i, range.as_ptr(), o, l) },
            want.len(),
        );
        assert_eq!(got.as_deref(), Ok(&want[..]));
    }
    unsafe { ftl_stack_free(s) };
}

/// The hyperstack fixture's shape, its resolution, and the plane each
/// position lands on.
#[test]
fn a_hyperstack_maps_positions_to_planes() {
    let path = fixture("ij_u16_spp1_p6_hyperstack.tif");
    let rust = open_rust(&path);
    let s = open_c(&path);
    let m = metadata(s);
    assert_eq!(m.format, FTL_METADATA_IMAGEJ);
    assert_eq!(
        (m.channels, m.slices, m.frames),
        (rust.meta.channels, rust.meta.slices, rust.meta.frames)
    );
    assert_eq!(m.channels * m.slices * m.frames, rust.frames.len());

    let mut seen = vec![false; rust.frames.len()];
    for t in 0..m.frames {
        for z in 0..m.slices {
            for c in 0..m.channels {
                let mut i = usize::MAX;
                assert_eq!(
                    unsafe { ftl_plane_index(s, null(), c, z, t, &mut i) },
                    FTL_OK
                );
                assert_eq!(i, t * m.slices * m.channels + z * m.channels + c);
                seen[i] = true;
            }
        }
    }
    assert!(seen.iter().all(|&b| b), "every plane has a position");

    let mut d = std::mem::MaybeUninit::uninit();
    assert_eq!(
        unsafe { ftl_resolve_dimensions(m.channels, m.slices, m.frames, d.as_mut_ptr()) },
        FTL_OK
    );
    let d = unsafe { d.assume_init() };
    assert_eq!(d, resolve_dimensions(m.channels, m.slices, m.frames).into());
    let mut i = 0;
    assert_eq!(
        unsafe { ftl_plane_index(s, &d, d.channels - 1, d.slices - 1, d.frames - 1, &mut i) },
        FTL_OK
    );
    assert_eq!(i, rust.frames.len() - 1);

    // A label the file got wrong is re-read, never re-counted.
    for (c, z, f) in [(100, 1, 3), (3, 4, 5), (2, 1, 1), (1, 7, 1)] {
        let mut d = std::mem::MaybeUninit::uninit();
        assert_eq!(
            unsafe { ftl_resolve_dimensions(c, z, f, d.as_mut_ptr()) },
            FTL_OK
        );
        assert_eq!(
            unsafe { d.assume_init() },
            resolve_dimensions(c, z, f).into()
        );
    }

    refused(
        unsafe { ftl_plane_index(s, null(), m.channels, 0, 0, &mut i) },
        FTL_INVALID_ARGUMENT,
    );
    let too_many = FtlDims {
        channels: m.channels,
        slices: m.slices,
        frames: m.frames + 1,
        triple_axis_warning: false,
    };
    refused(
        unsafe { ftl_plane_index(s, &too_many, 0, 0, m.frames, &mut i) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_plane_index(s, null(), 0, 0, 0, null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    unsafe { ftl_stack_free(s) };
}

fn options() -> FtlWriterOptions {
    let mut o = std::mem::MaybeUninit::uninit();
    assert_eq!(
        unsafe { ftl_writer_options_default(o.as_mut_ptr()) },
        FTL_OK
    );
    unsafe { o.assume_init() }
}

/// Write `frames` through C, finishing; the file's bytes.
fn write_c<T>(
    path: &Path,
    o: &FtlWriterOptions,
    frames: &[Vec<T>],
    write: unsafe extern "C" fn(*mut FtlWriter, *const T, usize) -> c_int,
) -> Vec<u8> {
    let mut w = null_mut();
    let status = unsafe { ftl_writer_create(c_path(path).as_ptr(), o, &mut w) };
    assert_eq!(status, FTL_OK, "{}", last_error());
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(
            unsafe { write(w, f.as_ptr(), f.len()) },
            FTL_OK,
            "{}",
            last_error()
        );
        assert_eq!(unsafe { ftl_writer_frames_written(w) }, i + 1);
    }
    assert_eq!(unsafe { ftl_writer_finish(w) }, FTL_OK, "{}", last_error());
    std::fs::read(path).unwrap()
}

/// The same frames through the Rust writer, as raw bytes; the file's bytes.
fn write_rust(path: &Path, o: WriterOptions, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut w = TiffWriter::create(path, o).unwrap();
    for f in frames {
        w.write_frame_bytes(f).unwrap();
    }
    w.finish().unwrap();
    std::fs::read(path).unwrap()
}

/// `file` with its OME `UUID` blanked: the one thing that differs between two
/// writes of the same stack.
fn without_uuid(mut file: Vec<u8>) -> Vec<u8> {
    let key = b"urn:uuid:";
    if let Some(at) = file.windows(key.len()).position(|w| w == key) {
        for b in &mut file[at + key.len()..at + key.len() + 36] {
            *b = b'0';
        }
    }
    file
}

/// [`ftl_writer_write_bytes`] in the shape [`write_c`] takes.
unsafe extern "C" fn write_bytes(w: *mut FtlWriter, data: *const u8, len: usize) -> c_int {
    ftl_writer_write_bytes(w, data.cast(), len)
}

fn le_bytes<T: Copy, const N: usize>(v: &[T], to: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|&x| to(x)).collect()
}

/// A 2-channel, 3-slice, 2-timepoint 16-bit hyperstack with every metadata
/// field set, written through C: byte for byte what the Rust writer makes of
/// the same options, and it reads back through C as through Rust.
#[test]
fn an_imagej_stack_written_through_c_is_the_rust_writers() {
    let (w, h, c, z, t) = (23u32, 17u32, 2usize, 3usize, 2usize);
    let frames: Vec<Vec<u16>> = (0..c * z * t)
        .map(|p| {
            (0..w * h)
                .map(|i| ((i * 37 + p as u32 * 1000) % 65536) as u16)
                .collect()
        })
        .collect();
    let unit_c = CString::new("um").unwrap();
    let o = FtlWriterOptions {
        width: w,
        height: h,
        sample_type: FTL_TYPE_U16,
        compression: FTL_COMPRESSION_DEFLATE,
        compression_level: 9,
        predictor: true,
        rows_per_strip: 5,
        metadata_format: FTL_METADATA_IMAGEJ,
        mode: FTL_MODE_COMPOSITE,
        channels: c,
        slices: z,
        unit: unit_c.as_ptr(),
        pixel_width: 0.25,
        pixel_height: 0.5,
        spacing: 2.0,
        calibration: [-3.0, 0.5],
        range: [100.0, 60000.0],
        fps: 7.5,
        frame_interval_s: 0.2,
        loop_playback: 1,
        ..options()
    };
    let (pc, pr) = (temp("ij_c.tif"), temp("ij_rust.tif"));
    let from_c = write_c(&pc, &o, &frames, ftl_writer_write_u16);
    let rust_opts = WriterOptions::new(w, h, SampleType::U16)
        .compression(Compression::Deflate)
        .compression_level(9)
        .predictor(true)
        .rows_per_strip(5)
        .metadata(
            StackMetaWrite::new(c, z)
                .mode(DisplayMode::Composite)
                .unit("um")
                .pixel_size(0.25, 0.5)
                .spacing(2.0)
                .calibration(-3.0, 0.5)
                .range(100.0, 60000.0)
                .fps(7.5)
                .frame_interval_s(0.2)
                .loop_playback(true),
        );
    let as_bytes: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| le_bytes(f, u16::to_le_bytes))
        .collect();
    assert!(
        from_c == write_rust(&pr, rust_opts, &as_bytes),
        "the same file"
    );

    let s = open_c(&pc);
    let rust = open_rust(&pc);
    assert_eq!(unsafe { ftl_frame_count(s) }, c * z * t);
    for (i, want) in frames.iter().enumerate() {
        let got = read_c(
            |o, l| unsafe { ftl_read_u16(s, i, null(), o, l) },
            want.len(),
        );
        assert_eq!(got.as_deref(), Ok(&want[..]), "frame {i}");
    }
    let m = metadata(s);
    same_metadata(&m, &FtlMetadata::from(&rust.meta));
    assert_eq!((m.channels, m.slices, m.frames), (c, z, t));
    assert_eq!(
        (m.format, m.mode),
        (FTL_METADATA_IMAGEJ, FTL_MODE_COMPOSITE)
    );
    assert!((m.pixel_width - 0.25).abs() < 1e-9 && (m.pixel_height - 0.5).abs() < 1e-9);
    assert_eq!((m.spacing, m.calibration, m.fps), (2.0, [-3.0, 0.5], 7.5));
    assert_eq!((m.frame_interval_s, m.loop_playback), (0.2, 1));
    assert_eq!(unit(s), rust.meta.unit.clone().unwrap());
    let mut info = std::mem::MaybeUninit::uninit();
    assert_eq!(unsafe { ftl_frame_info(s, 0, info.as_mut_ptr()) }, FTL_OK);
    let info = unsafe { info.assume_init() };
    assert_eq!(
        (info.compression, info.predictor, info.rows_per_strip),
        (FTL_COMPRESSION_DEFLATE, 2, 5)
    );
    unsafe { ftl_stack_free(s) };
    let _ = std::fs::remove_file(pc);
    let _ = std::fs::remove_file(pr);
}

/// OME, float, named channels: the same file as the Rust writer's, and the
/// names land in the OME-XML.
#[test]
fn an_ome_stack_with_named_channels_matches_rust() {
    let (w, h) = (16u32, 9u32);
    let frames: Vec<Vec<f32>> = (0..4)
        .map(|p| {
            (0..w * h)
                .map(|i| i as f32 * 0.5 - p as f32 * 3.25)
                .collect()
        })
        .collect();
    let names = [CString::new("DAPI").unwrap(), CString::new("GFP").unwrap()];
    let name_ptrs = [names[0].as_ptr(), names[1].as_ptr()];
    let colors = [0u8, 0, 255, 0, 255, 0];
    let o = FtlWriterOptions {
        width: w,
        height: h,
        sample_type: FTL_TYPE_F32,
        compression: FTL_COMPRESSION_LZW,
        predictor: true,
        metadata_format: FTL_METADATA_OME,
        mode: FTL_MODE_COMPOSITE,
        channels: 2,
        pixel_width: 0.1,
        pixel_height: 0.1,
        channel_names: name_ptrs.as_ptr(),
        channel_colors: colors.as_ptr(),
        named_channels: 2,
        ..options()
    };
    let (pc, pr) = (temp("ome_c.tif"), temp("ome_rust.tif"));
    let from_c = write_c(&pc, &o, &frames, ftl_writer_write_f32);
    let rust_opts = WriterOptions::new(w, h, SampleType::F32)
        .compression(Compression::Lzw)
        .predictor(true)
        .metadata_format(MetadataFormat::Ome)
        .metadata(
            StackMetaWrite::new(2, 1)
                .mode(DisplayMode::Composite)
                .pixel_size(0.1, 0.1)
                .channel("DAPI", [0, 0, 255])
                .channel("GFP", [0, 255, 0]),
        );
    let as_bytes: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| le_bytes(f, f32::to_le_bytes))
        .collect();
    let from_rust = write_rust(&pr, rust_opts, &as_bytes);
    assert!(
        without_uuid(from_c) == without_uuid(from_rust),
        "the same file but its UUID"
    );

    let s = open_c(&pc);
    let m = metadata(s);
    assert_eq!((m.format, m.channels, m.frames), (FTL_METADATA_OME, 2, 2));
    let d = description(s);
    assert!(d.contains("DAPI") && d.contains("GFP"), "{d}");
    for (i, want) in frames.iter().enumerate() {
        let got = read_c(|o, l| unsafe { ftl_read_f32(s, i, o, l) }, want.len());
        assert_eq!(got.as_deref(), Ok(&want[..]), "frame {i}");
    }
    unsafe { ftl_stack_free(s) };
    let _ = std::fs::remove_file(pc);
    let _ = std::fs::remove_file(pr);
}

/// The other layouts and types: planar RGB, CMYK, signed 16-bit through raw
/// bytes, f64, forced BigTIFF, a verbatim description — each the Rust
/// writer's file, each read back through C.
#[test]
fn every_layout_and_type_matches_rust() {
    let (w, h) = (11u32, 6u32);
    let n = (w * h) as usize;
    let rgb: Vec<Vec<u8>> = (0..2)
        .map(|p| (0..3 * n).map(|i| (i * 7 + p * 50) as u8).collect())
        .collect();
    let cmyk: Vec<Vec<u8>> = vec![(0..4 * n).map(|i| (i * 13) as u8).collect()];
    let signed: Vec<Vec<u8>> = vec![le_bytes(
        &(0..n).map(|i| i as i16 * 97 - 3000).collect::<Vec<_>>(),
        i16::to_le_bytes,
    )];
    let doubles: Vec<Vec<f64>> = vec![(0..n).map(|i| i as f64 / 7.0 - 1.0).collect()];
    let text = CString::new("a verbatim description").unwrap();

    let p = |tag: &str| {
        (
            temp(&format!("{tag}_c.tif")),
            temp(&format!("{tag}_rust.tif")),
        )
    };

    // Planar RGB, PackBits.
    let (pc, pr) = p("rgb");
    let o = FtlWriterOptions {
        width: w,
        height: h,
        sample_type: FTL_TYPE_U8,
        samples_per_pixel: 3,
        planar: true,
        compression: FTL_COMPRESSION_PACKBITS,
        ..options()
    };
    let from_c = write_c(&pc, &o, &rgb, ftl_writer_write_u8);
    let ro = WriterOptions::new(w, h, SampleType::U8)
        .samples_per_pixel(3)
        .planar(true)
        .compression(Compression::PackBits);
    assert!(from_c == write_rust(&pr, ro, &rgb));
    let s = open_c(&pc);
    let mut info = std::mem::MaybeUninit::uninit();
    assert_eq!(unsafe { ftl_frame_info(s, 0, info.as_mut_ptr()) }, FTL_OK);
    let info = unsafe { info.assume_init() };
    assert!(info.rgb && info.planar && !info.cmyk);
    for (i, want) in rgb.iter().enumerate() {
        // Planar in, planes out: the same order.
        let got = read_c(|o, l| unsafe { ftl_read_u8(s, i, o, l) }, want.len());
        assert_eq!(got.as_deref(), Ok(&want[..]));
    }
    unsafe { ftl_stack_free(s) };

    // CMYK, interleaved: planes out, and RGB through the conversion.
    let (pc, pr) = p("cmyk");
    let o = FtlWriterOptions {
        width: w,
        height: h,
        sample_type: FTL_TYPE_U8,
        samples_per_pixel: 4,
        cmyk: true,
        ..options()
    };
    let from_c = write_c(&pc, &o, &cmyk, ftl_writer_write_u8);
    let ro = WriterOptions::new(w, h, SampleType::U8)
        .samples_per_pixel(4)
        .cmyk(true);
    assert!(from_c == write_rust(&pr, ro, &cmyk));
    let s = open_c(&pc);
    let rust = open_rust(&pc);
    let got = read_c(|o, l| unsafe { ftl_read_u8(s, 0, o, l) }, 4 * n).unwrap();
    let planar: Vec<u8> = (0..4)
        .flat_map(|k| cmyk[0].iter().skip(k).step_by(4).copied())
        .collect();
    assert_eq!(got, planar, "interleaved in, planes out");
    let got = read_c(|o, l| unsafe { ftl_read_rgb_u8(s, 0, o, l) }, 3 * n).unwrap();
    assert_eq!(
        got,
        flat(read_planes_rgb_u8(&rust.data, &rust.frames[0], rust.byte_order).unwrap())
    );
    unsafe { ftl_stack_free(s) };

    // Signed 16-bit, through raw bytes, ZSTD where this build has it.
    let (pc, pr) = p("i16");
    let zstd = cfg!(feature = "codec-zstd");
    let o = FtlWriterOptions {
        width: w,
        height: h,
        sample_type: FTL_TYPE_I16,
        compression: if zstd {
            FTL_COMPRESSION_ZSTD
        } else {
            FTL_COMPRESSION_DEFLATE
        },
        predictor: true,
        ..options()
    };
    let from_c = write_c(&pc, &o, &signed, write_bytes);
    let ro = WriterOptions::new(w, h, SampleType::I16)
        .compression(if zstd {
            Compression::Zstd
        } else {
            Compression::Deflate
        })
        .predictor(true);
    assert!(from_c == write_rust(&pr, ro, &signed));
    let s = open_c(&pc);
    let rust = open_rust(&pc);
    let got = read_c(|o, l| unsafe { ftl_read_u16(s, 0, null(), o, l) }, n);
    let want = read_frame_u16(&rust.data, &rust.frames[0], rust.byte_order, None).unwrap();
    assert_eq!(got.as_deref(), Ok(&want[..]));
    unsafe { ftl_stack_free(s) };

    // f64 in a forced BigTIFF with a verbatim description.
    let (pc, pr) = p("f64");
    let o = FtlWriterOptions {
        width: w,
        height: h,
        sample_type: FTL_TYPE_F64,
        bigtiff: true,
        description: text.as_ptr(),
        ..options()
    };
    let from_c = write_c(&pc, &o, &doubles, ftl_writer_write_f64);
    let ro = WriterOptions::new(w, h, SampleType::F64)
        .bigtiff(true)
        .description("a verbatim description");
    let as_bytes: Vec<Vec<u8>> = doubles
        .iter()
        .map(|f| le_bytes(f, f64::to_le_bytes))
        .collect();
    assert!(from_c == write_rust(&pr, ro, &as_bytes));
    let s = open_c(&pc);
    assert!(unsafe { ftl_stack_is_bigtiff(s) });
    assert_eq!(description(s), "a verbatim description");
    let got = read_c(|o, l| unsafe { ftl_read_f32(s, 0, o, l) }, n);
    let want: Vec<f32> = doubles[0].iter().map(|&v| v as f32).collect();
    assert_eq!(got.as_deref(), Ok(&want[..]));
    unsafe { ftl_stack_free(s) };

    for tag in ["rgb", "cmyk", "i16", "f64"] {
        let (pc, pr) = p(tag);
        let _ = (std::fs::remove_file(pc), std::fs::remove_file(pr));
    }
}

/// The snprintf convention, at every edge.
#[test]
fn strings_follow_the_snprintf_convention() {
    let s = open_c(&fixture("ij_u16_spp1_p6_hyperstack.tif"));
    let full = description(s);
    assert!(full.len() > 8);
    assert_eq!(unsafe { ftl_description(s, null_mut(), 0) }, full.len());
    let mut buf = [0x55u8; 6];
    assert_eq!(
        unsafe { ftl_description(s, buf.as_mut_ptr().cast(), 0) },
        full.len()
    );
    assert_eq!(buf, [0x55; 6], "cap 0 writes nothing");
    assert_eq!(
        unsafe { ftl_description(s, buf.as_mut_ptr().cast(), 5) },
        full.len()
    );
    assert_eq!(&buf[..4], &full.as_bytes()[..4]);
    assert_eq!(buf[4], 0, "cut short, still terminated");
    assert_eq!(buf[5], 0x55, "nothing past cap");
    unsafe { ftl_stack_free(s) };

    // None at all: empty, and length 0.
    let s = open_c(&fixture("tff_u16_spp1_p2_none-le.tif"));
    let mut buf = [0x55u8; 4];
    assert_eq!(
        unsafe { ftl_metadata_unit(s, buf.as_mut_ptr().cast(), buf.len()) },
        0
    );
    assert_eq!(buf[0], 0);
    unsafe { ftl_stack_free(s) };
    assert_eq!(
        unsafe { ftl_description(null(), buf.as_mut_ptr().cast(), buf.len()) },
        0
    );
    assert_eq!(unsafe { ftl_metadata_unit(null(), null_mut(), 0) }, 0);
}

/// What C can say and Rust cannot, and what neither accepts, comes back as a
/// status and a reason — on the reading side.
#[test]
fn bad_reads_are_refused_with_a_reason() {
    let mut s = null_mut();
    refused(
        unsafe { ftl_open_bytes(b"II".as_ptr().cast(), 2, null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_open_bytes(null(), 10, &mut s) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_open_bytes(b"x".as_ptr().cast(), usize::MAX, &mut s) },
        FTL_INVALID_ARGUMENT,
    );
    let not_tiff = b"%PDF-1.7 this is not a TIFF";
    refused(
        unsafe { ftl_open_bytes(not_tiff.as_ptr().cast(), not_tiff.len(), &mut s) },
        FTL_NOT_TIFF,
    );
    assert!(s.is_null());
    let bytes = std::fs::read(fixture("tff_u16_spp1_p2_none-le.tif")).unwrap();
    refused(
        unsafe { ftl_open_bytes(bytes.as_ptr().cast(), 20, &mut s) },
        FTL_FORMAT,
    );

    refused(unsafe { ftl_open(null(), &mut s) }, FTL_INVALID_ARGUMENT);
    let missing = c_path(&temp("does_not_exist.tif"));
    #[cfg(feature = "mmap")]
    {
        let bad_utf8 = [0xffu8, 0xfe, 0];
        refused(unsafe { ftl_open(missing.as_ptr(), &mut s) }, FTL_IO);
        refused(
            unsafe { ftl_open(bad_utf8.as_ptr().cast(), &mut s) },
            FTL_INVALID_ARGUMENT,
        );
        let pdf = temp("not_a.tif");
        std::fs::write(&pdf, not_tiff).unwrap();
        refused(
            unsafe { ftl_open(c_path(&pdf).as_ptr(), &mut s) },
            FTL_NOT_TIFF,
        );
        let _ = std::fs::remove_file(pdf);
    }
    #[cfg(not(feature = "mmap"))]
    refused(
        unsafe { ftl_open(missing.as_ptr(), &mut s) },
        FTL_UNAVAILABLE,
    );
    assert!(s.is_null());

    // NULL handles and outputs.
    let mut info = std::mem::MaybeUninit::uninit();
    let mut n = 0;
    assert_eq!(unsafe { ftl_frame_count(null()) }, 0);
    assert!(
        !unsafe { ftl_stack_is_bigtiff(null()) } && !unsafe { ftl_stack_is_big_endian(null()) }
    );
    refused(
        unsafe { ftl_frame_info(null(), 0, info.as_mut_ptr()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_metadata(null(), null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_sample_count(null(), 0, &mut n) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_read_u16(null(), 0, null(), null_mut(), 0) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_resolve_dimensions(1, 1, 1, null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_float_minmax(null(), 0, null_mut(), null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    unsafe { ftl_stack_free(null_mut()) };

    let s = open_c_bytes(&bytes);
    let count = unsafe { ftl_frame_count(s) };
    refused(
        unsafe { ftl_frame_info(s, 0, null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(unsafe { ftl_metadata(s, null_mut()) }, FTL_INVALID_ARGUMENT);
    refused(
        unsafe { ftl_sample_count(s, 0, null_mut()) },
        FTL_INVALID_ARGUMENT,
    );

    // An index past the end, everywhere one is taken.
    refused(
        unsafe { ftl_frame_info(s, count, info.as_mut_ptr()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_sample_count(s, count, &mut n) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_float_minmax(s, count, null_mut(), null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    assert_eq!(unsafe { ftl_sample_count(s, 0, &mut n) }, FTL_OK);
    let mut out = vec![0u16; n];
    refused(
        unsafe { ftl_read_u16(s, count, null(), out.as_mut_ptr(), n) },
        FTL_INVALID_ARGUMENT,
    );

    // A buffer too small, a NULL one, a range that is not one.
    refused(
        unsafe { ftl_read_u16(s, 0, null(), out.as_mut_ptr(), n - 1) },
        FTL_INVALID_ARGUMENT,
    );
    assert!(last_error().contains(&n.to_string()), "{}", last_error());
    refused(
        unsafe { ftl_read_u16(s, 0, null(), null_mut(), n) },
        FTL_INVALID_ARGUMENT,
    );
    let nan = [f32::NAN, 1.0];
    refused(
        unsafe { ftl_read_u16(s, 0, nan.as_ptr(), out.as_mut_ptr(), n) },
        FTL_INVALID_ARGUMENT,
    );
    // A bigger buffer is fine; only the frame's share is written.
    let mut roomy = vec![7u16; n + 3];
    assert_eq!(
        unsafe { ftl_read_u16(s, 0, null(), roomy.as_mut_ptr(), n + 3) },
        FTL_OK
    );
    assert_eq!(&roomy[n..], &[7, 7, 7]);

    // A 16-bit frame through the 8-bit and float readers.
    let mut bytes8 = vec![0u8; 3 * n];
    let mut floats = vec![0f32; n];
    refused(
        unsafe { ftl_read_u8(s, 0, bytes8.as_mut_ptr(), n) },
        FTL_FORMAT,
    );
    refused(
        unsafe { ftl_read_f32(s, 0, floats.as_mut_ptr(), n) },
        FTL_FORMAT,
    );
    refused(
        unsafe { ftl_read_rgb_u8(s, 0, bytes8.as_mut_ptr(), 3 * n) },
        FTL_FORMAT,
    );
    unsafe { ftl_stack_free(s) };
}

/// The same, on the writing side; and a refused writer leaves no file.
#[test]
fn bad_writes_are_refused_with_a_reason() {
    let path = temp("refused.tif");
    let p = c_path(&path);
    let mut w = null_mut();
    let base = FtlWriterOptions {
        width: 4,
        height: 3,
        ..options()
    };
    let empty = CString::new("").unwrap();
    let bad_utf8 = [0xffu8, 0];
    let one_name = [empty.as_ptr()];

    refused(
        unsafe { ftl_writer_options_default(null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_create(p.as_ptr(), &base, null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_create(null(), &base, &mut w) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_create(p.as_ptr(), null(), &mut w) },
        FTL_INVALID_ARGUMENT,
    );
    let ij = FtlWriterOptions {
        metadata_format: FTL_METADATA_IMAGEJ,
        ..base
    };
    for broken in [
        FtlWriterOptions {
            sample_type: 42,
            ..base
        },
        FtlWriterOptions {
            compression: 7,
            ..base
        },
        FtlWriterOptions {
            metadata_format: 9,
            ..base
        },
        FtlWriterOptions { mode: 9, ..ij },
        FtlWriterOptions {
            loop_playback: 2,
            ..ij
        },
        FtlWriterOptions { channels: 0, ..ij },
        FtlWriterOptions {
            calibration: [1.0, f64::NAN],
            ..ij
        },
        FtlWriterOptions {
            pixel_width: 0.5,
            ..ij
        },
        FtlWriterOptions {
            fps: f64::INFINITY,
            ..ij
        },
        FtlWriterOptions {
            unit: bad_utf8.as_ptr().cast(),
            ..ij
        },
        FtlWriterOptions {
            named_channels: 1,
            ..ij
        },
        FtlWriterOptions {
            named_channels: 1,
            channel_names: one_name.as_ptr(),
            ..ij
        },
        // What Rust refuses too.
        FtlWriterOptions { width: 0, ..base },
        FtlWriterOptions {
            samples_per_pixel: 3,
            cmyk: true,
            ..base
        },
        FtlWriterOptions {
            description: empty.as_ptr(),
            ..ij
        },
    ] {
        refused(
            unsafe { ftl_writer_create(p.as_ptr(), &broken, &mut w) },
            FTL_INVALID_ARGUMENT,
        );
        assert!(w.is_null());
        assert!(!path.exists(), "a refused writer leaves no file");
    }
    let nowhere = c_path(&temp("no_such_dir").join("x.tif"));
    refused(
        unsafe { ftl_writer_create(nowhere.as_ptr(), &base, &mut w) },
        FTL_IO,
    );

    // Frames of the wrong type or size.
    assert_eq!(
        unsafe { ftl_writer_create(p.as_ptr(), &base, &mut w) },
        FTL_OK
    );
    let frame = [1u16; 12];
    refused(
        unsafe { ftl_writer_write_u8(w, [0u8; 12].as_ptr(), 12) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_write_f32(w, [0f32; 12].as_ptr(), 12) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_write_f64(w, [0f64; 12].as_ptr(), 12) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_write_u16(w, frame.as_ptr(), 11) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_write_u16(w, frame.as_ptr(), usize::MAX) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_write_u16(w, null(), 12) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_write_bytes(w, frame.as_ptr().cast(), 23) },
        FTL_INVALID_ARGUMENT,
    );
    refused(
        unsafe { ftl_writer_write_u16(null_mut(), frame.as_ptr(), 12) },
        FTL_INVALID_ARGUMENT,
    );
    assert_eq!(unsafe { ftl_writer_frames_written(w) }, 0);
    // Nothing written: finish refuses, and the handle is gone either way.
    refused(unsafe { ftl_writer_finish(w) }, FTL_INVALID_ARGUMENT);
    refused(
        unsafe { ftl_writer_finish(null_mut()) },
        FTL_INVALID_ARGUMENT,
    );
    assert_eq!(unsafe { ftl_writer_frames_written(null()) }, 0);

    // Three planes do not divide into two channels.
    let two = FtlWriterOptions { channels: 2, ..ij };
    assert_eq!(
        unsafe { ftl_writer_create(p.as_ptr(), &two, &mut w) },
        FTL_OK
    );
    for _ in 0..3 {
        assert_eq!(
            unsafe { ftl_writer_write_u16(w, frame.as_ptr(), 12) },
            FTL_OK
        );
    }
    refused(unsafe { ftl_writer_finish(w) }, FTL_INVALID_ARGUMENT);

    // Abandoned: freed, and NULL is ignored.
    assert_eq!(
        unsafe { ftl_writer_create(p.as_ptr(), &base, &mut w) },
        FTL_OK
    );
    assert_eq!(
        unsafe { ftl_writer_write_u16(w, frame.as_ptr(), 12) },
        FTL_OK
    );
    unsafe { ftl_writer_free(w) };
    unsafe { ftl_writer_free(null_mut()) };
    let _ = std::fs::remove_file(&path);
}

/// The defaults are `WriterOptions::new`'s: the same file either way.
#[test]
fn the_default_options_are_the_writers() {
    let o = FtlWriterOptions {
        width: 5,
        height: 4,
        ..options()
    };
    assert_eq!(o.sample_type, FTL_TYPE_U16);
    let frames = vec![(0..20u16).collect::<Vec<_>>()];
    let (pc, pr) = (temp("default_c.tif"), temp("default_rust.tif"));
    let from_c = write_c(&pc, &o, &frames, ftl_writer_write_u16);
    let as_bytes: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| le_bytes(f, u16::to_le_bytes))
        .collect();
    assert!(from_c == write_rust(&pr, WriterOptions::new(5, 4, SampleType::U16), &as_bytes));
    let _ = (std::fs::remove_file(pc), std::fs::remove_file(pr));
}

/// Parallel decoding is a hint: the pixels do not change.
#[test]
fn parallel_decode_changes_nothing_but_speed() {
    let path = fixture("tff_u16_spp1_p2_zip-rps4.tif");
    let s = open_c(&path);
    let mut n = 0;
    assert_eq!(unsafe { ftl_sample_count(s, 0, &mut n) }, FTL_OK);
    let serial = read_c(|o, l| unsafe { ftl_read_u16(s, 0, null(), o, l) }, n);
    ftl_set_parallel_decode(true);
    let parallel = read_c(|o, l| unsafe { ftl_read_u16(s, 0, null(), o, l) }, n);
    ftl_set_parallel_decode(false);
    assert_eq!(serial, parallel);
    unsafe { ftl_stack_free(s) };
}

#[test]
fn the_version_is_the_crates() {
    let v = unsafe { CStr::from_ptr(ftl_version()) };
    assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

/// Review findings, each a refusal that used to be an answer.
#[test]
fn the_reviewed_edges_are_refused() {
    // A writer geometry whose frame overflows: refused, and no file left.
    let path = temp("overflow.tif");
    let p = c_path(&path);
    let o = FtlWriterOptions {
        width: 968_973_220,
        height: 384_773,
        samples_per_pixel: 49_477,
        sample_type: FTL_TYPE_U8,
        ..options()
    };
    let mut w = null_mut();
    let status = unsafe { ftl_writer_create(p.as_ptr(), &o, &mut w) };
    assert_ne!(status, FTL_OK, "an overflowing frame was accepted");
    assert!(!last_error().is_empty());
    assert!(w.is_null());
    assert!(!path.exists(), "a refused writer left a file behind");

    // Dimensions with no product to keep.
    let mut d = std::mem::MaybeUninit::<FtlDims>::uninit();
    for (c, z, t) in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (usize::MAX, 2, 1)] {
        refused(
            unsafe { ftl_resolve_dimensions(c, z, t, d.as_mut_ptr()) },
            FTL_INVALID_ARGUMENT,
        );
    }
    assert_eq!(unsafe { ftl_resolve_dimensions(2, 3, 4, d.as_mut_ptr()) }, FTL_OK);
}

/// Reading ZSTD in a build without the codec is the build's limit, not the
/// file's fault: the same FTL_UNAVAILABLE that writing it gets.
#[cfg(not(feature = "codec-zstd"))]
#[test]
fn zstd_without_the_codec_is_unavailable_to_read() {
    use crate::encode::SampleType;
    let mut w = TiffWriter::new(
        std::io::Cursor::new(Vec::new()),
        WriterOptions::new(4, 2, SampleType::U8),
    )
    .unwrap();
    w.write_frame_u8(&[7u8; 8]).unwrap();
    let mut bytes = w.finish().unwrap().into_inner();
    // The Compression entry, little-endian classic TIFF: tag 259, SHORT, one
    // value, 1 (none). Rewritten to 50000, ZSTD.
    let none = [0x03u8, 0x01, 0x03, 0x00, 0x01, 0, 0, 0, 0x01, 0x00];
    let at = bytes
        .windows(none.len())
        .position(|w| w == none)
        .expect("the Compression entry");
    bytes[at + 8..at + 10].copy_from_slice(&50000u16.to_le_bytes());
    let mut s = null_mut();
    assert_eq!(
        unsafe { ftl_open_bytes(bytes.as_ptr().cast(), bytes.len(), &mut s) },
        FTL_OK,
        "{}",
        last_error()
    );
    let mut out = [0u8; 8];
    refused(
        unsafe { ftl_read_u8(s, 0, out.as_mut_ptr(), out.len()) },
        FTL_UNAVAILABLE,
    );
    assert!(last_error().contains("codec-zstd"), "{}", last_error());
    unsafe { ftl_stack_free(s) };
}

//! The C readers checked against what they gave before they moved onto
//! [`Frame`](crate::Frame) (item 25d).
//!
//! [`DIGESTS`] were taken from the C interface as it was before the move,
//! 25c's tree, where each reader decoded a frame into planes of its own
//! through the library's old readers and then copied them into the caller's
//! buffer. For every file here — every fixture, fuzz regression and fuzz
//! seed, and a synthetic matrix of sample types, sample counts, interleavings,
//! CMYK, strip layouts, codecs and predictors, in small frames and in frames
//! large enough to decode across cores — and every frame of it (the first
//! [`FRAMES`]), a digest of what each reader gives: `ftl_read_u8`,
//! `ftl_read_u16` with no range and with one, `ftl_read_f32`,
//! `ftl_read_rgb_u8` and `ftl_read_rgb_u16` (the values, or the status), and
//! `ftl_float_minmax` and `ftl_sample_count`.
//!
//! Each read is made into a fresh buffer of exactly the frame's values and
//! into a dirty one longer than that, which must agree, and the values past
//! the frame must stay as they were; and into a buffer one value short, which
//! is refused with nothing written. Every file is read with parallel decoding
//! off and on, and a file on disk through `ftl_open` (mapped) as well as
//! `ftl_open_bytes`: all must agree. The synthetic files of one shape hold the
//! same samples whatever their codec, predictor and strips, so they must all
//! read the same, and one digest stands for the shape.
//!
//! A file that does not open and a read that fails are digested by status, so
//! an error where there was none, or none where there was one, shows as
//! plainly as a changed sample. Only the status: what `ftl_last_error` says is
//! free to change.
//!
//! While the library's old readers still exist, each read is also compared
//! live with the C readers as they were ([`old`]). They go in item 25e, and
//! [`old`] with them; the digests stay, and the case list must not change
//! after that — a digest is only as good as the code it was taken from.

use super::*;
use crate::parallel_decode_scope;
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};

/// Frames read of each file.
const FRAMES: usize = 8;

/// A buffer larger than this is not made: a hostile header can claim
/// gigabytes. Such a read is only asked of a buffer too small for it.
const MAX_BUFFER: usize = 256 << 20;

/// The C readers as they were before 25d, over the library's old readers:
/// the live half of the comparison, until they go. Each decodes the frame
/// into planes of its own and copies them, in order, into the caller's
/// buffer — given here as the slice it was, and returning the status.
mod old {
    use super::*;
    use crate::error::Context;
    use crate::{
        frame_float_minmax, read_frame_f32, read_frame_u16, read_frame_u8, read_planes_f32,
        read_planes_rgb_u16, read_planes_rgb_u8, read_planes_u16, read_planes_u8,
    };
    use std::borrow::Cow;

    pub(super) type Decode<T> =
        for<'a> fn(&'a TiffStack, &'a FrameInfo, Option<(f32, f32)>) -> Result<Vec<Cow<'a, [T]>>>;

    pub(super) type Count = fn(&FrameInfo) -> Result<usize>;

    pub(super) fn all_samples(f: &FrameInfo) -> Result<usize> {
        f.sample_count()
    }

    pub(super) fn three_planes(f: &FrameInfo) -> Result<usize> {
        f.pixel_count()?.checked_mul(3).context(
            ErrorKind::TooLarge,
            "frame geometry overflows address space",
        )
    }

    fn owned<T: Clone>(planes: Vec<Vec<T>>) -> Vec<Cow<'static, [T]>> {
        planes.into_iter().map(Cow::Owned).collect()
    }

    pub(super) fn decode_u8<'a>(
        st: &'a TiffStack,
        f: &'a FrameInfo,
        _: Option<(f32, f32)>,
    ) -> Result<Vec<Cow<'a, [u8]>>> {
        if !matches!(f.bits_per_sample, 4 | 8) {
            crate::error::bail!(InvalidInput, "not 4- or 8-bit");
        }
        if f.samples_per_pixel <= 1 {
            Ok(vec![read_frame_u8(&st.data, f, st.byte_order)?])
        } else {
            Ok(owned(read_planes_u8(&st.data, f, st.byte_order)?))
        }
    }

    pub(super) fn decode_u16<'a>(
        st: &'a TiffStack,
        f: &'a FrameInfo,
        range: Option<(f32, f32)>,
    ) -> Result<Vec<Cow<'a, [u16]>>> {
        if f.bits_per_sample == 4 && f.samples_per_pixel <= 1 {
            crate::error::bail!(InvalidInput, "4-bit");
        }
        if f.samples_per_pixel <= 1 {
            Ok(vec![read_frame_u16(&st.data, f, st.byte_order, range)?])
        } else {
            Ok(owned(read_planes_u16(&st.data, f, st.byte_order, range)?))
        }
    }

    pub(super) fn decode_f32<'a>(
        st: &'a TiffStack,
        f: &'a FrameInfo,
        _: Option<(f32, f32)>,
    ) -> Result<Vec<Cow<'a, [f32]>>> {
        if f.samples_per_pixel <= 1 {
            Ok(vec![read_frame_f32(&st.data, f, st.byte_order)?])
        } else {
            Ok(owned(read_planes_f32(&st.data, f, st.byte_order)?))
        }
    }

    pub(super) fn decode_rgb_u8<'a>(
        st: &'a TiffStack,
        f: &'a FrameInfo,
        _: Option<(f32, f32)>,
    ) -> Result<Vec<Cow<'a, [u8]>>> {
        Ok(owned(read_planes_rgb_u8(&st.data, f, st.byte_order)?))
    }

    pub(super) fn decode_rgb_u16<'a>(
        st: &'a TiffStack,
        f: &'a FrameInfo,
        _: Option<(f32, f32)>,
    ) -> Result<Vec<Cow<'a, [u16]>>> {
        Ok(owned(read_planes_rgb_u16(&st.data, f, st.byte_order)?))
    }

    /// The old `read_into`, after its pointer checks.
    pub(super) fn read_into<T: Copy>(
        st: &TiffStack,
        index: usize,
        range: Option<(f32, f32)>,
        out: &mut [T],
        count: Count,
        decode: Decode<T>,
    ) -> c_int {
        let Some(f) = st.frames.get(index) else {
            return FTL_INVALID_ARGUMENT;
        };
        let needed = match count(f) {
            Ok(n) => n,
            Err(e) => return status_of(&e),
        };
        if out.len() < needed {
            return FTL_INVALID_ARGUMENT;
        }
        let planes = match decode(st, f, range) {
            Ok(p) => p,
            Err(e) => return status_of(&e),
        };
        let total: usize = planes.iter().map(|p| p.len()).sum();
        if total != needed {
            return FTL_FAILED;
        }
        let mut at = 0;
        for p in &planes {
            out[at..at + p.len()].copy_from_slice(p);
            at += p.len();
        }
        FTL_OK
    }

    /// The old `ftl_float_minmax`: the range, or the status.
    pub(super) fn minmax(st: &TiffStack, index: usize) -> std::result::Result<(f32, f32), c_int> {
        match frame_float_minmax(&st.data, &st.frames[index], st.byte_order) {
            Ok(r) => Ok(r.unwrap_or((f32::NAN, f32::NAN))),
            Err(e) => Err(status_of(&e)),
        }
    }
}

// ------------------------------------------------------------------ digests

/// FNV-1a's step over 64-bit words: a fingerprint, not a defence.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }
    fn u64(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(0x0100_0000_01b3);
    }
    fn status(&mut self, s: c_int) {
        self.u64(s as u32 as u64);
    }
    fn bytes(&mut self, bs: &[u8]) {
        self.u64(bs.len() as u64);
        let (words, rest) = bs.as_chunks::<8>();
        for w in words {
            self.u64(u64::from_le_bytes(*w));
        }
        for &b in rest {
            self.u64(b as u64);
        }
    }
}

fn bytes_of<T: bytemuck::Pod>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// One reader on one frame, `needed` values long, digested: into a fresh
/// buffer, into a dirty one three longer, through the old reader, and into
/// one a value short.
fn read_case<T: bytemuck::Pod>(
    label: &str,
    needed: Option<usize>,
    dirt: T,
    c: impl Fn(*mut T, usize) -> c_int,
    old: impl Fn(&mut [T]) -> c_int,
    full: bool,
    h: &mut Digest,
) {
    let fits = needed.filter(|&n| n.saturating_mul(std::mem::size_of::<T>()) <= MAX_BUFFER);
    let Some(n) = fits else {
        // Too many values to make room for, or more than a size_t counts:
        // what a buffer of none gets, which is never a decode.
        let mut none: [T; 1] = [dirt];
        let status = c(none.as_mut_ptr(), 0);
        assert_eq!(status, old(&mut none[..0]), "{label}: the old status");
        assert!(bytes_of(&none) == bytes_of(&[dirt]), "{label}: written");
        h.u64(0xee);
        h.status(status);
        return;
    };
    let mut fresh = vec![T::zeroed(); n];
    let status = c(fresh.as_mut_ptr(), n);
    h.status(status);
    if status == FTL_OK {
        h.bytes(bytes_of(&fresh));
    }
    let mut dirty = vec![dirt; n + 3];
    assert_eq!(c(dirty.as_mut_ptr(), n + 3), status, "{label}: a longer buffer");
    if status == FTL_OK {
        assert!(
            bytes_of(&dirty[..n]) == bytes_of(&fresh),
            "{label}: a dirty buffer reads differently"
        );
    }
    assert!(
        bytes_of(&dirty[n..]) == bytes_of(&[dirt; 3]),
        "{label}: written past the frame's values"
    );
    if !full {
        return;
    }
    let mut was = vec![T::zeroed(); n];
    assert_eq!(old(&mut was), status, "{label}: the old reader's status");
    if status == FTL_OK {
        assert!(
            bytes_of(&was) == bytes_of(&fresh),
            "{label}: not what the old reader gave"
        );
    }
    if n > 0 {
        let mut short = vec![dirt; n - 1];
        let refused = c(short.as_mut_ptr(), n - 1);
        assert_eq!(refused, FTL_INVALID_ARGUMENT, "{label}: a buffer one short");
        assert!(
            short.iter().all(|v| bytes_of(&[*v]) == bytes_of(&[dirt])),
            "{label}: a buffer one short was written"
        );
    }
}

/// `ftl_float_minmax` of frame `i`, digested, with either output NULL as
/// well, and against the old one.
fn minmax_case(s: *const FtlStack, i: usize, full: bool, h: &mut Digest) {
    let (mut lo, mut hi) = (7.0f32, 7.0f32);
    let status = unsafe { ftl_float_minmax(s, i, &mut lo, &mut hi) };
    h.status(status);
    if status == FTL_OK {
        h.u64(lo.to_bits() as u64);
        h.u64(hi.to_bits() as u64);
    } else {
        assert!(lo == 7.0 && hi == 7.0, "frame {i}: a failed range wrote");
    }
    if !full {
        return;
    }
    assert_eq!(
        unsafe { ftl_float_minmax(s, i, null_mut(), null_mut()) },
        status
    );
    let mut only = 7.0f32;
    assert_eq!(
        unsafe { ftl_float_minmax(s, i, null_mut(), &mut only) },
        status
    );
    if status == FTL_OK {
        assert_eq!(only.to_bits(), hi.to_bits());
    }
    let st = unsafe { &(*s).inner };
    match old::minmax(st, i) {
        Ok((a, b)) => {
            assert_eq!(status, FTL_OK, "frame {i}: the old range");
            assert_eq!((a.to_bits(), b.to_bits()), (lo.to_bits(), hi.to_bits()));
        }
        Err(was) => assert_eq!(status, was, "frame {i}: the old range's status"),
    }
}

/// The range the ranged `u16` cases read with.
const RANGE: [f32; 2] = [-3.0, 300.0];

/// Every case of frame `i`; how many.
fn frame_cases(s: *const FtlStack, i: usize, full: bool, h: &mut Digest) -> usize {
    let st = unsafe { &(*s).inner };
    let mut n = 0usize;
    let counted = unsafe { ftl_sample_count(s, i, &mut n) };
    h.status(counted);
    let samples = (counted == FTL_OK).then_some(n);
    if let Some(n) = samples {
        h.u64(n as u64);
    }
    let mut info = std::mem::MaybeUninit::uninit();
    assert_eq!(unsafe { ftl_frame_info(s, i, info.as_mut_ptr()) }, FTL_OK);
    let info = unsafe { info.assume_init() };
    let three = (info.width as usize)
        .checked_mul(info.height as usize)
        .and_then(|p| p.checked_mul(3));
    let range = Some((RANGE[0], RANGE[1]));

    read_case(
        &format!("frame {i} u8"),
        samples,
        0xa5u8,
        |o, l| unsafe { ftl_read_u8(s, i, o, l) },
        |out| old::read_into(st, i, None, out, old::all_samples, old::decode_u8),
        full,
        h,
    );
    read_case(
        &format!("frame {i} u16"),
        samples,
        0xa5a5u16,
        |o, l| unsafe { ftl_read_u16(s, i, null(), o, l) },
        |out| old::read_into(st, i, None, out, old::all_samples, old::decode_u16),
        full,
        h,
    );
    read_case(
        &format!("frame {i} u16 ranged"),
        samples,
        0xa5a5u16,
        |o, l| unsafe { ftl_read_u16(s, i, RANGE.as_ptr(), o, l) },
        |out| old::read_into(st, i, range, out, old::all_samples, old::decode_u16),
        full,
        h,
    );
    read_case(
        &format!("frame {i} f32"),
        samples,
        f32::from_bits(0x7fc0_a5a5),
        |o, l| unsafe { ftl_read_f32(s, i, o, l) },
        |out| old::read_into(st, i, None, out, old::all_samples, old::decode_f32),
        full,
        h,
    );
    read_case(
        &format!("frame {i} rgb u8"),
        three,
        0xa5u8,
        |o, l| unsafe { ftl_read_rgb_u8(s, i, o, l) },
        |out| old::read_into(st, i, None, out, old::three_planes, old::decode_rgb_u8),
        full,
        h,
    );
    read_case(
        &format!("frame {i} rgb u16"),
        three,
        0xa5a5u16,
        |o, l| unsafe { ftl_read_rgb_u16(s, i, o, l) },
        |out| old::read_into(st, i, None, out, old::three_planes, old::decode_rgb_u16),
        full,
        h,
    );
    minmax_case(s, i, full, h);
    8
}

/// Every case of an open stack, digested; and how many.
fn stack_cases(s: *const FtlStack, full: bool) -> (u64, usize) {
    let mut h = Digest::new();
    let frames = unsafe { ftl_frame_count(s) };
    h.u64(frames as u64);
    let mut cases = 0;
    for i in 0..frames.min(FRAMES) {
        cases += frame_cases(s, i, full, &mut h);
    }
    (h.0, cases)
}

/// Every case of one file, read every way: the digest, and how many cases.
fn check_file(name: &str, bytes: &[u8], path: Option<&Path>) -> (u64, usize) {
    let mut s = null_mut();
    let opened = unsafe { ftl_open_bytes(bytes.as_ptr().cast(), bytes.len(), &mut s) };
    if opened != FTL_OK {
        let mut h = Digest::new();
        h.u64(0xff);
        h.status(opened);
        #[cfg(feature = "mmap")]
        if let Some(path) = path {
            let mut m = null_mut();
            let c = CString::new(path.to_str().unwrap()).unwrap();
            assert_eq!(unsafe { ftl_open(c.as_ptr(), &mut m) }, opened, "{name}");
        }
        let _ = path;
        return (h.0, 1);
    }
    let (digest, cases) = {
        let _serial = parallel_decode_scope(false);
        stack_cases(s, true)
    };
    {
        let _parallel = parallel_decode_scope(true);
        assert!(
            stack_cases(s, false).0 == digest,
            "{name}: decoded across cores, it reads differently"
        );
    }
    unsafe { ftl_stack_free(s) };
    #[cfg(feature = "mmap")]
    if let Some(path) = path {
        let mut m = null_mut();
        let c = CString::new(path.to_str().unwrap()).unwrap();
        assert_eq!(unsafe { ftl_open(c.as_ptr(), &mut m) }, FTL_OK, "{name}");
        let _serial = parallel_decode_scope(false);
        assert!(
            stack_cases(m, false).0 == digest,
            "{name}: mapped, it reads differently"
        );
        unsafe { ftl_stack_free(m) };
    }
    let _ = path;
    (digest, cases)
}

// ------------------------------------------------------------------ files

/// Every TIFF in `dir`, by name, with its path.
fn files_in(sub: &str) -> Vec<(String, PathBuf)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(sub);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tif" || x == "tiff"))
        .collect();
    paths.sort();
    let label = sub.rsplit('/').next().unwrap();
    paths
        .into_iter()
        .map(|p| {
            let name = format!("{label}/{}", p.file_name().unwrap().to_string_lossy());
            (name, p)
        })
        .collect()
}

/// One synthetic file's layout.
#[derive(Clone, Copy)]
struct Shape {
    st: SampleType,
    spp: u16,
    planar: bool,
    cmyk: bool,
    width: u32,
    height: u32,
    frames: usize,
}

/// Frame `f` of `shape` as the bytes the writer takes. Every byte varies, so
/// a sample read from the wrong place, plane or frame shows; floats are kept
/// finite and modest, but for a NaN and an infinity in frame 1. The last
/// frame of three is constant.
fn frame_bytes(shape: Shape, f: usize) -> Vec<u8> {
    let sb = shape.st.bytes();
    let n = shape.width as usize * shape.height as usize * shape.spp as usize * sb;
    let float = matches!(shape.st, SampleType::F32 | SampleType::F64);
    if shape.frames == 3 && f == 2 {
        return vec![0x11; n];
    }
    let mut bytes: Vec<u8> = (0..n)
        .map(|i| {
            (i as u8)
                .wrapping_mul(37)
                .wrapping_add((f as u8).wrapping_mul(101))
                ^ (i >> 8) as u8
        })
        .map(|b| if float { b & 0x3f } else { b })
        .collect();
    if float && f == 1 && n >= 2 * sb {
        let (nan, inf) = if sb == 4 {
            (f32::NAN.to_le_bytes().to_vec(), f32::INFINITY.to_le_bytes().to_vec())
        } else {
            (f64::NAN.to_le_bytes().to_vec(), f64::INFINITY.to_le_bytes().to_vec())
        };
        bytes[..sb].copy_from_slice(&nan);
        bytes[sb..2 * sb].copy_from_slice(&inf);
    }
    bytes
}

/// `shape` written by this crate's writer with a codec, a predictor and a
/// strip height; `None` where the writer does not write that.
fn written(
    shape: Shape,
    compression: Compression,
    predictor: bool,
    rows_per_strip: Option<u32>,
) -> Option<Vec<u8>> {
    let mut opts = WriterOptions::new(shape.width, shape.height, shape.st)
        .samples_per_pixel(shape.spp)
        .planar(shape.planar)
        .cmyk(shape.cmyk)
        .compression(compression)
        .predictor(predictor);
    if let Some(rows) = rows_per_strip {
        opts = opts.rows_per_strip(rows);
    }
    let mut wr = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).ok()?;
    for f in 0..shape.frames {
        wr.write_frame_bytes(&frame_bytes(shape, f)).ok()?;
    }
    Some(wr.finish().ok()?.into_inner())
}

/// The codecs a synthetic shape is written with, each with and without the
/// predictor where it matters.
fn codecs() -> Vec<(Compression, bool)> {
    let mut c = vec![
        (Compression::None, false),
        (Compression::Lzw, false),
        (Compression::Lzw, true),
        (Compression::Deflate, false),
        (Compression::Deflate, true),
        (Compression::PackBits, false),
    ];
    if cfg!(feature = "codec-zstd") {
        c.push((Compression::Zstd, false));
        c.push((Compression::Zstd, true));
    }
    c
}

/// A synthetic shape: a name, and every file of it, which hold the same
/// samples.
type Group = (String, Vec<(String, Vec<u8>)>);

/// The synthetic matrix: every sample type in grayscale, chunky and planar
/// RGB, RGBA, and chunky and planar CMYK, small, in every codec, one strip a
/// frame or three rows a strip; and a few shapes of 1024 x 1024, a million
/// pixels, which decode across cores when parallel decoding is on, in
/// several strips.
fn synthetic() -> Vec<Group> {
    let types = [
        SampleType::U8,
        SampleType::I8,
        SampleType::U16,
        SampleType::I16,
        SampleType::U32,
        SampleType::I32,
        SampleType::F32,
        SampleType::U64,
        SampleType::I64,
        SampleType::F64,
    ];
    // Samples per pixel, planar, CMYK, and a label.
    let layouts: [(u16, bool, bool, &str); 6] = [
        (1, false, false, "gray"),
        (3, false, false, "rgb"),
        (3, true, false, "rgb planar"),
        (4, false, false, "rgba"),
        (4, false, true, "cmyk"),
        (4, true, true, "cmyk planar"),
    ];
    let mut groups = Vec::new();
    for st in types {
        for (spp, planar, cmyk, label) in layouts {
            let shape = Shape {
                st,
                spp,
                planar,
                cmyk,
                width: 13,
                height: 10,
                frames: 3,
            };
            let mut files = Vec::new();
            for (compression, predictor) in codecs() {
                for rps in [None, Some(3)] {
                    let variant = format!(
                        "{compression:?}{} {}",
                        if predictor { "+pred" } else { "" },
                        rps.map_or("one strip".to_string(), |r| format!("rps{r}"))
                    );
                    if let Some(bytes) = written(shape, compression, predictor, rps) {
                        files.push((variant, bytes));
                    }
                }
            }
            if !files.is_empty() {
                groups.push((format!("synthetic/{st:?} {label}"), files));
            }
        }
    }

    // Large: the codec, predictor and strip height of each variant.
    type Large = (SampleType, u16, bool, bool, &'static str, Vec<(Compression, bool, Option<u32>)>);
    let mut large: Vec<Large> = vec![
        (
            SampleType::U16,
            1,
            false,
            false,
            "gray",
            vec![
                (Compression::None, false, Some(64)),
                (Compression::Lzw, false, None),
                (Compression::Deflate, true, None),
            ],
        ),
        (
            SampleType::U8,
            3,
            false,
            false,
            "rgb",
            vec![
                (Compression::Lzw, true, None),
                (Compression::Deflate, false, Some(32)),
            ],
        ),
        (
            SampleType::F32,
            1,
            false,
            false,
            "gray",
            vec![
                (Compression::None, false, None),
                (Compression::Deflate, true, None),
            ],
        ),
        (
            SampleType::U32,
            1,
            false,
            false,
            "gray",
            vec![
                (Compression::None, false, Some(100)),
                (Compression::Deflate, false, None),
            ],
        ),
        (
            SampleType::U8,
            4,
            true,
            true,
            "cmyk planar",
            vec![(Compression::Lzw, false, None)],
        ),
    ];
    if cfg!(feature = "codec-zstd") {
        large[0].5.push((Compression::Zstd, true, None));
        large[2].5.push((Compression::Zstd, true, None));
    }
    for (st, spp, planar, cmyk, label, variants) in large {
        let shape = Shape {
            st,
            spp,
            planar,
            cmyk,
            width: 1024,
            height: 1024,
            frames: 1,
        };
        let files: Vec<(String, Vec<u8>)> = variants
            .into_iter()
            .map(|(compression, predictor, rps)| {
                let bytes = written(shape, compression, predictor, rps)
                    .expect("the writer writes the large shapes");
                (format!("{compression:?} {predictor} {rps:?}"), bytes)
            })
            .collect();
        groups.push((format!("large/{st:?} {label}"), files));
    }
    groups
}

/// One unit of work: a file on disk or a synthetic shape.
enum Work {
    Disk(String, PathBuf),
    Shape(Group),
}

impl Work {
    fn name(&self) -> &str {
        match self {
            Work::Disk(name, _) | Work::Shape((name, _)) => name,
        }
    }

    /// Its digest and how many cases it had. A shape's variants must all
    /// read alike.
    fn check(&self) -> (u64, usize) {
        match self {
            Work::Disk(name, path) => {
                let bytes = std::fs::read(path).unwrap();
                check_file(name, &bytes, Some(path))
            }
            Work::Shape((name, files)) => {
                let mut first: Option<(u64, &str)> = None;
                let mut cases = 0;
                for (variant, bytes) in files {
                    let (digest, n) = check_file(&format!("{name} {variant}"), bytes, None);
                    cases += n;
                    match first {
                        None => first = Some((digest, variant)),
                        Some((d, v)) => assert!(
                            d == digest,
                            "{name}: {variant} reads differently from {v}, the same samples"
                        ),
                    }
                }
                (first.expect("a shape has files").0, cases)
            }
        }
    }
}

#[test]
fn the_c_readers_equal_what_they_gave_before() {
    let mut work: Vec<Work> = Vec::new();
    for sub in ["tests/fixtures", "tests/fuzz-regressions", "fuzz/seeds"] {
        work.extend(files_in(sub).into_iter().map(|(n, p)| Work::Disk(n, p)));
    }
    work.extend(synthetic().into_iter().map(Work::Shape));
    assert!(work.len() > 120, "only {} files and shapes", work.len());

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(16));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<(String, u64, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(w) = work.get(i) else { break };
                        let (digest, cases) = w.check();
                        done.push((i, digest, cases));
                    }
                    done
                })
            })
            .collect();
        let mut all: Vec<(usize, u64, usize)> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect();
        all.sort_by_key(|r| r.0);
        all.into_iter()
            .map(|(i, d, c)| (work[i].name().to_string(), d, c))
            .collect()
    });

    if std::env::var_os("FASTTIFF_PRINT_DIGESTS").is_some() {
        let rows: Vec<String> = results
            .iter()
            .map(|(name, digest, _)| format!("    ({name:?}, {digest:#018x}),"))
            .collect();
        println!(
            "const DIGESTS: &[(&str, u64)] = &[\n{}\n];",
            rows.join("\n")
        );
    }
    for (name, digest, _) in &results {
        let want = DIGESTS.iter().find(|(n, _)| n == name).map(|&(_, d)| d);
        assert_eq!(want, Some(*digest), "{name}: not what it gave before");
    }
    let unchecked: Vec<&str> = DIGESTS
        .iter()
        .map(|&(name, _)| name)
        .filter(|name| !results.iter().any(|(n, _, _)| n == name))
        .collect();
    assert!(
        unchecked.is_empty(),
        "digests with no file or shape to check: {unchecked:?}"
    );
    let cases: usize = results.iter().map(|r| r.2).sum();
    assert!(cases > 10_000, "only {cases} cases");
}

/// Printed by `FASTTIFF_PRINT_DIGESTS=1 cargo test -p fast-tiff-lib
/// --features capi --lib capi::oracle_tests -- --nocapture`, at 25c's tree,
/// before the C readers moved.
const DIGESTS: &[(&str, u64)] = &[];

//! The shared pieces: a host to run plugins against, and the PSF file reader.

use super::*;
use fasttiff_plugin_api::{
    HostContext, ImageInfo, ParamDecl, ParamValue, Params, PixelType, Plane, Plugin, PluginError,
    ViewParams, VolumeMode, VolumeView,
};
use std::io::Cursor;

// ------------------------------------------------------------- a host to run in

/// A stack held in memory, addressable as `(c, z, t)`.
///
/// Shared with `psf_tests` and `run_tests`, which both need to drive a whole
/// plugin and check what came back. Deliberately not a mock that records
/// calls: what these plugins must get right is arithmetic over pixels, so the
/// host has to be able to hand out real ones at a real shape.
pub(crate) struct TestHost {
    pub image: ImageInfo,
    pub view: ViewParams,
    pub info: StackInfo,
    /// Every plane, `xyczt` order, concatenated.
    pub pixels: Vec<f32>,
    /// What the dialog is holding, for a `params` call. See [`settle`].
    pub pending: Params,
    /// Stop after this many `progress` calls, to exercise cancellation.
    pub cancel_after: Option<usize>,
    pub progress_calls: usize,
    pub logged: Vec<String>,
}

impl TestHost {
    pub(crate) fn new(
        width: u32,
        height: u32,
        channels: usize,
        slices: usize,
        frames: usize,
    ) -> Self {
        let n = width as usize * height as usize * channels * slices * frames;
        TestHost {
            image: ImageInfo {
                width,
                height,
                channels,
                slices,
                frames,
                samples_per_pixel: 1,
                pixel_type: PixelType::F32,
            },
            view: ViewParams {
                frame_index: 0,
                volume_view: false,
                channels: Vec::new(),
                luts: Vec::new(),
                volume: VolumeView {
                    mode: VolumeMode::Mip,
                    density: 1.0,
                    iso: 0.5,
                    eye: [0.0; 3],
                    forward: [0.0, 0.0, 1.0],
                    up: [0.0, 1.0, 0.0],
                    right: [1.0, 0.0, 0.0],
                },
            },
            info: StackInfo {
                name: "test".into(),
                ..Default::default()
            },
            pixels: vec![0.0; n],
            pending: Params::new(),
            cancel_after: None,
            progress_calls: 0,
            logged: Vec::new(),
        }
    }

    /// The offset of a plane in `pixels`.
    fn at(&self, p: Plane) -> usize {
        let plane = self.image.plane_len();
        let (c, z, t) = (self.image.channels, self.image.slices, self.image.frames);
        debug_assert!(p.t < t);
        (p.t * (z * c) + p.z * c + p.c) * plane
    }

    /// Overwrite one plane.
    pub(crate) fn set_plane(&mut self, p: Plane, data: &[f32]) {
        let at = self.at(p);
        self.pixels[at..at + data.len()].copy_from_slice(data);
    }

    /// Whether `log` said something containing `needle`.
    pub(crate) fn said(&self, needle: &str) -> bool {
        self.logged.iter().any(|m| m.contains(needle))
    }
}

impl HostContext for TestHost {
    fn image(&self) -> ImageInfo {
        self.image
    }
    fn view(&self) -> &ViewParams {
        &self.view
    }
    fn stack_info(&self) -> &StackInfo {
        &self.info
    }
    fn pending_params(&self) -> &Params {
        &self.pending
    }
    fn read_plane_u16(&mut self, p: Plane, out: &mut Vec<u16>) -> Result<(), PluginError> {
        let mut f = Vec::new();
        self.read_plane_f32(p, &mut f)?;
        *out = f.iter().map(|&v| v.clamp(0.0, 65535.0) as u16).collect();
        Ok(())
    }
    fn read_plane_f32(&mut self, p: Plane, out: &mut Vec<f32>) -> Result<(), PluginError> {
        if !self.image.contains(p.c, p.z, p.t) {
            return Err(PluginError::OutOfRange(format!("{p:?}")));
        }
        let at = self.at(p);
        out.clear();
        out.extend_from_slice(&self.pixels[at..at + self.image.plane_len()]);
        Ok(())
    }
    fn progress(&mut self, _f: f32) -> bool {
        self.progress_calls += 1;
        match self.cancel_after {
            Some(n) => self.progress_calls <= n,
            None => true,
        }
    }
    fn log(&mut self, m: &str) {
        self.logged.push(m.to_string());
    }
}

// -------------------------------------------------------------- the dialog

/// Open a plugin dialog, set `overrides`, and let it settle.
///
/// This is the host loop, which is the only way to test a dialog that
/// changes as it is answered. The app asks for the declarations, draws them,
/// and asks again whenever a value changes — so a test that asked once would
/// be testing a dialog no user ever sees. Here that loop is run to a fixed
/// point and the settled `(declarations, values)` come back.
///
/// `overrides` are re-applied after each round because they stand for things
/// the user set, and a control can only be set after the round that declared
/// it: choosing the Gaussian PSF source is what makes a sigma field exist.
///
/// Returns the values already clamped, exactly as `run` would receive them —
/// so anything the settled dialog does not declare is *gone*, which is the
/// behaviour the plugins have to be right about.
pub(crate) fn settle<P: Plugin>(
    plugin: &P,
    host: &mut TestHost,
    overrides: &[(&str, ParamValue)],
) -> (Vec<ParamDecl>, Params) {
    let apply = |values: &mut Params| {
        for (k, v) in overrides {
            values.set(*k, v.clone());
        }
    };
    host.pending = Params::new();
    let mut decls = plugin.params(host);
    let mut values = Params::defaults(&decls);
    apply(&mut values);

    // Eight rounds is far more than any dialog here needs — two choices deep
    // is the most either of them goes — and terminating rather than looping
    // is what matters if a plugin ever declares a dialog that oscillates.
    for _ in 0..8 {
        host.pending = values.clone();
        let next = plugin.params(host);
        if next == decls {
            break;
        }
        for (k, v) in Params::defaults(&next).iter() {
            if values.get(k).is_none() {
                values.set(k, v.clone());
            }
        }
        apply(&mut values);
        decls = next;
    }
    host.pending = Params::new();
    let clamped = values.clamp_to(&decls);
    (decls, clamped)
}

/// The keys a settled dialog declares.
pub(crate) fn keys(decls: &[ParamDecl]) -> Vec<String> {
    decls.iter().map(|d| d.key.clone()).collect()
}

// ------------------------------------------------------------------- fixtures

/// Where this module's temporary TIFFs go. Keyed by process id so two test
/// binaries running at once cannot collide.
pub(crate) fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fasttiff-deconv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir.join(name)
}

/// Write a stack of 32-bit planes as a TIFF and return the path.
pub(crate) fn write_f32_tiff(
    name: &str,
    w: u32,
    h: u32,
    planes: &[Vec<f32>],
) -> std::path::PathBuf {
    use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(w, h, SampleType::F32)
        .metadata(StackMetaWrite::new(1, planes.len().max(1)));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    for p in planes {
        writer.write_frame_f32(p).expect("frame");
    }
    let path = temp(name);
    std::fs::write(&path, writer.finish().expect("finish").into_inner()).expect("write");
    path
}

// ------------------------------------------------------------- the PSF reader

#[test]
fn an_empty_path_says_what_to_do_about_it() {
    let err = load_psf("").expect_err("no path is not a PSF");
    assert!(matches!(err, PluginError::Unsupported(_)), "{err:?}");
    // The message has to name the way out, because the dialog offers two and
    // "no PSF file chosen" alone leaves the user looking for a file they may
    // not have.
    let text = err.to_string();
    assert!(text.contains("Generate PSF"), "{text}");
}

#[test]
fn a_missing_file_is_reported_with_its_name() {
    let path = temp("definitely-not-here.tif");
    let _ = std::fs::remove_file(&path);
    let err = load_psf(&path.display().to_string()).expect_err("missing");
    assert!(err.to_string().contains("definitely-not-here"), "{err}");
}

#[test]
fn a_file_that_is_not_a_tiff_is_refused() {
    let path = temp("not-a-tiff.tif");
    std::fs::write(&path, b"this is not a TIFF at all").expect("write");
    let err = load_psf(&path.display().to_string()).expect_err("not a TIFF");
    assert!(err.to_string().contains("not a TIFF"), "{err}");
}

/// Every IFD is a Z slice, whatever the file called them.
#[test]
fn a_tiff_reads_back_as_a_volume() {
    let planes: Vec<Vec<f32>> = (0..4)
        .map(|z| (0..3 * 2).map(|i| (z * 10 + i) as f32).collect())
        .collect();
    let path = write_f32_tiff("volume.tif", 3, 2, &planes);

    let psf = load_psf(&path.display().to_string()).expect("read");
    assert_eq!((psf.dims.x, psf.dims.y, psf.dims.z), (3, 2, 4));
    assert_eq!(psf.data.len(), 24);
    // Not merely the right length: the right samples, in the right order.
    assert_eq!(psf.data[0], 0.0);
    assert_eq!(psf.data[6], 10.0, "slice 1 should start at 10");
    assert_eq!(psf.data[23], 35.0);
}

/// 16-bit is the type a measured PSF is usually saved as, so it cannot be the
/// path that only works by accident.
#[test]
fn a_16_bit_tiff_reads_back_at_its_own_scale() {
    use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(2, 2, SampleType::U16).metadata(StackMetaWrite::new(1, 1));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    writer.write_frame_u16(&[0, 1, 1000, 65535]).expect("frame");
    let path = temp("sixteen.tif");
    std::fs::write(&path, writer.finish().expect("finish").into_inner()).expect("write");

    let psf = load_psf(&path.display().to_string()).expect("read");
    // The sample values themselves, not a display rescale of them. Reading
    // through the viewer's 16-bit path would have given 0..65535 stretched to
    // the window, which for a PSF is a different kernel.
    assert_eq!(psf.data, vec![0.0, 1.0, 1000.0, 65535.0]);
}

#[test]
fn an_8_bit_tiff_is_not_widened_to_the_16_bit_range() {
    use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(2, 2, SampleType::U8).metadata(StackMetaWrite::new(1, 1));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    writer.write_frame_u8(&[0, 1, 128, 255]).expect("frame");
    let path = temp("eight.tif");
    std::fs::write(&path, writer.finish().expect("finish").into_inner()).expect("write");

    let psf = load_psf(&path.display().to_string()).expect("read");
    // `read_plane_u16` would have given `(v << 8) | v` — 255 becoming 65535.
    // Harmless after normalisation and wrong before it, which is exactly the
    // kind of thing that is never noticed.
    assert_eq!(psf.data, vec![0.0, 1.0, 128.0, 255.0]);
}

/// A little-endian TIFF of `slices` IFDs, each claiming `side` x `side` 8-bit
/// pixels in a single strip that holds one byte. Built by hand because no
/// writer will make one: only its header is large.
fn claiming_tiff(side: u32, slices: u32) -> Vec<u8> {
    const ENTRIES: u16 = 9;
    let ifd_len = 2 + 12 * ENTRIES as u32 + 4;
    let pixel = 8u32;
    let mut out = b"II*\0".to_vec();
    out.extend_from_slice(&12u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    // The one byte every strip points at.
    out[pixel as usize] = 7;
    for s in 0..slices {
        let at = 12 + s * ifd_len;
        let next = if s + 1 < slices { at + ifd_len } else { 0 };
        out.extend_from_slice(&ENTRIES.to_le_bytes());
        let entry = |out: &mut Vec<u8>, tag: u16, kind: u16, value: u32| {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        };
        let (short, long) = (3, 4);
        entry(&mut out, 256, long, side); // ImageWidth
        entry(&mut out, 257, long, side); // ImageLength
        entry(&mut out, 258, short, 8); // BitsPerSample
        entry(&mut out, 259, short, 1); // Compression: none
        entry(&mut out, 262, short, 1); // Photometric: black is zero
        entry(&mut out, 273, long, pixel); // StripOffsets
        entry(&mut out, 277, short, 1); // SamplesPerPixel
        entry(&mut out, 278, long, side); // RowsPerStrip
        entry(&mut out, 279, long, 1); // StripByteCounts
        out.extend_from_slice(&next.to_le_bytes());
    }
    out
}

/// A PSF whose voxel count overflows a `usize` is refused, not read as an
/// empty one. Four slices of 2^31 x 2^31 are 2^64 voxels, which wraps to
/// none: the size check passed it, the volume was allocated at that wrapped
/// length, and the read loop over it found no slice to read — so nothing
/// checked a slice against its strips, and a file of a few hundred bytes came
/// back as a PSF whose data did not match its own shape, which the 2-D run
/// then sliced past. Sixteen slices of 2^30 x 2^30 wrap the same way.
#[test]
fn a_psf_claiming_more_voxels_than_a_usize_holds_is_refused() {
    for (side, slices) in [(1u32 << 31, 4u32), (1 << 30, 16)] {
        let path = temp(&format!("claims-{side}x{side}x{slices}.tif"));
        std::fs::write(&path, claiming_tiff(side, slices)).expect("write");
        let path = path.display().to_string();
        // The file itself opens: only its claim is out of reach.
        let opened = fast_tiff_lib::TiffStack::from_bytes(std::fs::read(&path).unwrap());
        assert_eq!(opened.map(|t| t.frames.len()).ok(), Some(slices as usize));
        match load_psf(&path) {
            Err(PluginError::Unsupported(msg)) => {
                assert!(msg.contains("more than this can hold"), "{msg}")
            }
            Ok(psf) => panic!(
                "{side}x{side}x{slices}: read as a PSF of {}x{}x{} holding {} samples",
                psf.dims.x,
                psf.dims.y,
                psf.dims.z,
                psf.data.len()
            ),
            Err(e) => panic!("{side}x{side}x{slices}: refused for the wrong reason: {e:?}"),
        }
    }
}

// ------------------------------------------------------- the PSF reader, oracle

/// The PSF reader as it was before it decoded each slice straight into the
/// volume: one slice at a time into a buffer of its own, appended. The live
/// half of [`the_psf_reader_reads_what_it_read_before`], kept while the
/// library's old reader it calls exists; both go in item 25e, and
/// [`PSF_DIGESTS`] stays.
mod old {
    use super::super::{Dims, Psf, MAX_VOXELS};
    use fasttiff_plugin_api::PluginError;

    pub(super) fn load_psf(path: &str) -> Result<Psf, PluginError> {
        use fast_tiff_lib::TiffStack;
        if path.trim().is_empty() {
            return Err(PluginError::unsupported("no PSF file chosen"));
        }
        let bytes = std::fs::read(path).map_err(|e| PluginError::failed(format!("{e}")))?;
        let tiff =
            TiffStack::from_bytes(bytes).map_err(|e| PluginError::failed(format!("{e:#}")))?;
        let first = tiff
            .frames
            .first()
            .ok_or_else(|| PluginError::failed("no images"))?;
        let dims = Dims::new(
            first.width as usize,
            first.height as usize,
            tiff.frames.len(),
        );
        if dims.x == 0 || dims.y == 0 {
            return Err(PluginError::failed("a zero dimension"));
        }
        if tiff
            .frames
            .iter()
            .any(|f| f.width != first.width || f.height != first.height)
        {
            return Err(PluginError::failed("slices of different sizes"));
        }
        if dims.len() > MAX_VOXELS {
            return Err(PluginError::unsupported("too many voxels"));
        }
        let mut data = Vec::with_capacity(dims.len());
        let mut slice = Vec::new();
        for frame in tiff.frames.iter() {
            fast_tiff_lib::read_plane_f32_into(&tiff.data, frame, tiff.byte_order, 0, &mut slice)
                .map_err(|e| PluginError::failed(format!("{e:#}")))?;
            data.extend_from_slice(&slice);
        }
        if data.len() != dims.len() {
            return Err(PluginError::failed("decoded to the wrong length"));
        }
        Ok(Psf { data, dims })
    }
}

fn lib_dir(sub: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fast-tiff-lib")
        .join(sub)
}

/// Every TIFF in the library's fixtures, fuzz regressions and fuzz seeds, by
/// name and path; and a synthetic matrix written beside them: every depth a
/// PSF is saved as, one sample or three, one strip a slice or several,
/// uncompressed, LZW and LZW with the predictor.
fn psf_files() -> Vec<(String, std::path::PathBuf)> {
    use fast_tiff_lib::{Compression, SampleType, TiffWriter, WriterOptions};
    let mut out = Vec::new();
    for sub in ["tests/fixtures", "tests/fuzz-regressions", "fuzz/seeds"] {
        let Ok(entries) = std::fs::read_dir(lib_dir(sub)) else {
            continue;
        };
        let mut paths: Vec<std::path::PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "tif" || x == "tiff"))
            .collect();
        paths.sort();
        for p in paths {
            let name = format!("{sub}/{}", p.file_name().unwrap().to_string_lossy());
            out.push((name, p));
        }
    }
    let types = [
        SampleType::U8,
        SampleType::I8,
        SampleType::U16,
        SampleType::I16,
        SampleType::U32,
        SampleType::F32,
    ];
    let codecs = [
        (Compression::None, false),
        (Compression::Lzw, false),
        (Compression::Lzw, true),
    ];
    for st in types {
        for spp in [1u16, 3] {
            for (compression, predictor) in codecs {
                for rps in [None, Some(4)] {
                    let (w, h) = (11u32, 7u32);
                    let mut opts = WriterOptions::new(w, h, st)
                        .samples_per_pixel(spp)
                        .compression(compression)
                        .predictor(predictor);
                    if let Some(rows) = rps {
                        opts = opts.rows_per_strip(rows);
                    }
                    let mut wr = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
                    let n = (w * h) as usize * spp as usize * st.bytes();
                    for z in 0..5u8 {
                        let bytes: Vec<u8> = (0..n)
                            .map(|i| (i as u8).wrapping_mul(29).wrapping_add(z.wrapping_mul(71)))
                            .map(|b| if st == SampleType::F32 { b & 0x3f } else { b })
                            .collect();
                        wr.write_frame_bytes(&bytes).unwrap();
                    }
                    let name = format!(
                        "synthetic/{st:?} spp{spp} {compression:?}{} {}",
                        if predictor { "+pred" } else { "" },
                        rps.map_or("one strip".to_string(), |r| format!("rps{r}"))
                    );
                    let file = format!("oracle-{}.tif", name.replace(['/', ' ', '+'], "_"));
                    let path = temp(&file);
                    std::fs::write(&path, wr.finish().unwrap().into_inner()).unwrap();
                    out.push((name, path));
                }
            }
        }
    }
    out
}

/// What a read came to, digested (64-bit FNV-1a): the shape and every
/// sample's bits, or the kind of error.
fn psf_digest(got: &Result<Psf, PluginError>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut put = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    };
    match got {
        Ok(psf) => {
            put(&[1]);
            for v in [psf.dims.x, psf.dims.y, psf.dims.z, psf.data.len()] {
                put(&(v as u64).to_le_bytes());
            }
            for v in &psf.data {
                put(&v.to_bits().to_le_bytes());
            }
        }
        Err(PluginError::Unsupported(_)) => put(&[2]),
        Err(PluginError::OutOfRange(_)) => put(&[3]),
        Err(PluginError::Failed(_)) => put(&[4]),
    }
    h
}

/// Printed by `FASTTIFF_PRINT_DIGESTS=1 cargo test -p fasttiff-builtin-plugins
/// --lib psf_reader_reads -- --nocapture`, from the reader before it decoded
/// straight into the volume.
const PSF_DIGESTS: &[(&str, u64)] = &[
    ("tests/fixtures/ij_u16_spp1_p6_hyperstack.tif", 0x12fc2dc074be8617),
    ("tests/fixtures/pil_u16_spp1_p1_lzw.tif", 0xa3cdb1fdd1a002b8),
    ("tests/fixtures/pil_u8_spp1_p1_lzw.tif", 0xffd1996e00c2ef42),
    ("tests/fixtures/pil_u8_spp1_p1_pb.tif", 0xffd1996e00c2ef42),
    ("tests/fixtures/pil_u8_spp3_p1_zip.tif", 0x552d874222a8d359),
    ("tests/fixtures/pil_u8_spp4_p1_cmyk-lzw.tif", 0xd83f0ed01ce2cb10),
    ("tests/fixtures/tff_f32_spp1_p1_tiled-zip-pred3.tif", 0x689792952411694b),
    ("tests/fixtures/tff_f32_spp1_p2_lzw-pred3.tif", 0xa655f9e237a77625),
    ("tests/fixtures/tff_f32_spp1_p2_none-le.tif", 0xa655f9e237a77625),
    ("tests/fixtures/tff_f32_spp1_p2_zip-pred3-be.tif", 0xa655f9e237a77625),
    ("tests/fixtures/tff_f32_spp1_p2_zip-pred3.tif", 0xa655f9e237a77625),
    ("tests/fixtures/tff_i16_spp1_p2_lzw-pred2.tif", 0xaa2b276fac163801),
    ("tests/fixtures/tff_i16_spp1_p2_none-be.tif", 0xaa2b276fac163801),
    ("tests/fixtures/tff_i16_spp1_p2_none-le.tif", 0xaa2b276fac163801),
    ("tests/fixtures/tff_i32_spp1_p2_none-le.tif", 0x5182e690f3b93133),
    ("tests/fixtures/tff_i8_spp1_p2_none-le.tif", 0xb687b4dbf04dea89),
    ("tests/fixtures/tff_u16_spp1_p1_tiled.tif", 0xa3cdb1fdd1a002b8),
    ("tests/fixtures/tff_u16_spp1_p2_lzw-pred2-be.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp1_p2_lzw-pred2-rps4.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp1_p2_lzw-rps4.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp1_p2_none-be.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp1_p2_none-le.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp1_p2_pb-rps4.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp1_p2_zip-pred2.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp1_p2_zip-rps4.tif", 0x45df94f074f05f0b),
    ("tests/fixtures/tff_u16_spp3_p2_zip-pred2.tif", 0x98aae04f83572170),
    ("tests/fixtures/tff_u16_spp4_p1_cmyk-zip-pred2.tif", 0x0f590964b72437ed),
    ("tests/fixtures/tff_u16_spp4_p1_cmyk.tif", 0x0f590964b72437ed),
    ("tests/fixtures/tff_u32_spp1_p2_none-le.tif", 0xa5044ec239db1e8d),
    ("tests/fixtures/tff_u32_spp1_p2_zip-pred2.tif", 0xa5044ec239db1e8d),
    ("tests/fixtures/tff_u8_spp1_p2_none-le.tif", 0x935797a52975eced),
    ("tests/fixtures/tff_u8_spp1_p2_zip-pred2.tif", 0x935797a52975eced),
    ("tests/fixtures/tff_u8_spp3_p1_tiled-lzw-pred2.tif", 0x552d874222a8d359),
    ("tests/fixtures/tff_u8_spp3_p2_lzw-pred2.tif", 0x0132df076ea81db8),
    ("tests/fixtures/tff_u8_spp3_p2_none.tif", 0x0132df076ea81db8),
    ("tests/fixtures/tff_u8_spp4_p1_cmyk-lzw.tif", 0xd83f0ed01ce2cb10),
    ("tests/fixtures/tff_u8_spp4_p1_cmyk.tif", 0xd83f0ed01ce2cb10),
    ("tests/fixtures/tff_u8_spp4_p2_cmyk-planar.tif", 0xb45692a6bb665190),
    ("tests/fixtures/tld_u16_spp1_p1_grid-lzw.tif", 0xb411ed2a3f6092ae),
    ("tests/fixtures/tld_u16_spp1_p1_grid.tif", 0xb411ed2a3f6092ae),
    ("tests/fixtures/tld_u8_spp3_p1_grid-planar.tif", 0x76ce7e67fd2c52a4),
    ("tests/fixtures/tld_u8_spp3_p1_grid-pred2.tif", 0x0873e70ce464a3e9),
    ("tests/fuzz-regressions/overflow-crop-rows-height-u32max.tif", 0xaf63bf4c8601bb45),
    ("tests/fuzz-regressions/overflow-strip-band-h4g-rps2g.tif", 0xaf63bf4c8601bb45),
    ("tests/fuzz-regressions/overflow-tiled-crop-h4g-tile2g.tif", 0xaf63bf4c8601bb45),
    ("tests/fuzz-regressions/overflow-w2g-h2g-spp4.tif", 0xaf63bf4c8601bb45),
    ("tests/fuzz-regressions/slow-tall-tile.tif", 0xaf63b94c8601b113),
    ("tests/fuzz-regressions/slow-tile-grid.tif", 0xaf63b94c8601b113),
    ("fuzz/seeds/bigtiff.tif", 0x67692e6fee849655),
    ("fuzz/seeds/cmyk16_lzw.tif", 0x117f380a21db0c99),
    ("fuzz/seeds/cmyk_chunky.tif", 0x8b653f3d9a96ef63),
    ("fuzz/seeds/cmyk_planar.tif", 0x39f5ffced4dea4ba),
    ("fuzz/seeds/float32.tif", 0x55b8cdd670f2f50d),
    ("fuzz/seeds/float32_zlib_pred.tif", 0x55b8cdd670f2f50d),
    ("fuzz/seeds/gray16.tif", 0x67692e6fee849655),
    ("fuzz/seeds/gray16_be.tif", 0x67692e6fee849655),
    ("fuzz/seeds/gray16_lzw.tif", 0x67692e6fee849655),
    ("fuzz/seeds/gray16_packbits.tif", 0x67692e6fee849655),
    ("fuzz/seeds/gray16_strips.tif", 0x4cb3c26ed914ecc9),
    ("fuzz/seeds/gray16_zlib.tif", 0x67692e6fee849655),
    ("fuzz/seeds/gray16_zlib_pred.tif", 0x67692e6fee849655),
    ("fuzz/seeds/gray8.tif", 0x42d127a231aac830),
    ("fuzz/seeds/imagej_hyperstack.tif", 0x1677e1b41c728dbf),
    ("fuzz/seeds/imagej_lut.tif", 0x67692e6fee849655),
    ("fuzz/seeds/ome.tif", 0x1677e1b41c728dbf),
    ("fuzz/seeds/palette4.tif", 0xb721659b720a4e2d),
    ("fuzz/seeds/palette8.tif", 0x42d127a231aac830),
    ("fuzz/seeds/regression-oom-huge-width.tif", 0xaf63bf4c8601bb45),
    ("fuzz/seeds/rgb_chunky.tif", 0xcdca10de2fd25263),
    ("fuzz/seeds/rgb_planar.tif", 0xcdca10de2fd25263),
    ("fuzz/seeds/stack3.tif", 0xe769ed9ca6e8358e),
    ("fuzz/seeds/tiled16.tif", 0x1c607d470b22ecb2),
    ("fuzz/seeds/tiled16_lzw.tif", 0x1c607d470b22ecb2),
    ("fuzz/seeds/tiled_planar.tif", 0x830bc07f852e88c9),
    ("fuzz/seeds/tiled_rgb_pred.tif", 0x3f96e0e123b5d1d8),
    ("synthetic/U8 spp1 None one strip", 0x64d17721c21d0d14),
    ("synthetic/U8 spp1 None rps4", 0x64d17721c21d0d14),
    ("synthetic/U8 spp1 Lzw one strip", 0x64d17721c21d0d14),
    ("synthetic/U8 spp1 Lzw rps4", 0x64d17721c21d0d14),
    ("synthetic/U8 spp1 Lzw+pred one strip", 0x64d17721c21d0d14),
    ("synthetic/U8 spp1 Lzw+pred rps4", 0x64d17721c21d0d14),
    ("synthetic/U8 spp3 None one strip", 0x64302f1586fb122c),
    ("synthetic/U8 spp3 None rps4", 0x64302f1586fb122c),
    ("synthetic/U8 spp3 Lzw one strip", 0x64302f1586fb122c),
    ("synthetic/U8 spp3 Lzw rps4", 0x64302f1586fb122c),
    ("synthetic/U8 spp3 Lzw+pred one strip", 0x64302f1586fb122c),
    ("synthetic/U8 spp3 Lzw+pred rps4", 0x64302f1586fb122c),
    ("synthetic/I8 spp1 None one strip", 0xac3831941a0710d7),
    ("synthetic/I8 spp1 None rps4", 0xac3831941a0710d7),
    ("synthetic/I8 spp1 Lzw one strip", 0xac3831941a0710d7),
    ("synthetic/I8 spp1 Lzw rps4", 0xac3831941a0710d7),
    ("synthetic/I8 spp1 Lzw+pred one strip", 0xac3831941a0710d7),
    ("synthetic/I8 spp1 Lzw+pred rps4", 0xac3831941a0710d7),
    ("synthetic/I8 spp3 None one strip", 0x92614ff48bbcf916),
    ("synthetic/I8 spp3 None rps4", 0x92614ff48bbcf916),
    ("synthetic/I8 spp3 Lzw one strip", 0x92614ff48bbcf916),
    ("synthetic/I8 spp3 Lzw rps4", 0x92614ff48bbcf916),
    ("synthetic/I8 spp3 Lzw+pred one strip", 0x92614ff48bbcf916),
    ("synthetic/I8 spp3 Lzw+pred rps4", 0x92614ff48bbcf916),
    ("synthetic/U16 spp1 None one strip", 0x026745c54a520e8c),
    ("synthetic/U16 spp1 None rps4", 0x026745c54a520e8c),
    ("synthetic/U16 spp1 Lzw one strip", 0x026745c54a520e8c),
    ("synthetic/U16 spp1 Lzw rps4", 0x026745c54a520e8c),
    ("synthetic/U16 spp1 Lzw+pred one strip", 0x026745c54a520e8c),
    ("synthetic/U16 spp1 Lzw+pred rps4", 0x026745c54a520e8c),
    ("synthetic/U16 spp3 None one strip", 0x614fd2ebe0cfbda1),
    ("synthetic/U16 spp3 None rps4", 0x614fd2ebe0cfbda1),
    ("synthetic/U16 spp3 Lzw one strip", 0x614fd2ebe0cfbda1),
    ("synthetic/U16 spp3 Lzw rps4", 0x614fd2ebe0cfbda1),
    ("synthetic/U16 spp3 Lzw+pred one strip", 0x614fd2ebe0cfbda1),
    ("synthetic/U16 spp3 Lzw+pred rps4", 0x614fd2ebe0cfbda1),
    ("synthetic/I16 spp1 None one strip", 0xf4856d0c3af763fb),
    ("synthetic/I16 spp1 None rps4", 0xf4856d0c3af763fb),
    ("synthetic/I16 spp1 Lzw one strip", 0xf4856d0c3af763fb),
    ("synthetic/I16 spp1 Lzw rps4", 0xf4856d0c3af763fb),
    ("synthetic/I16 spp1 Lzw+pred one strip", 0xf4856d0c3af763fb),
    ("synthetic/I16 spp1 Lzw+pred rps4", 0xf4856d0c3af763fb),
    ("synthetic/I16 spp3 None one strip", 0x477c9b27a627c1fc),
    ("synthetic/I16 spp3 None rps4", 0x477c9b27a627c1fc),
    ("synthetic/I16 spp3 Lzw one strip", 0x477c9b27a627c1fc),
    ("synthetic/I16 spp3 Lzw rps4", 0x477c9b27a627c1fc),
    ("synthetic/I16 spp3 Lzw+pred one strip", 0x477c9b27a627c1fc),
    ("synthetic/I16 spp3 Lzw+pred rps4", 0x477c9b27a627c1fc),
    ("synthetic/U32 spp1 None one strip", 0xf4acb50a444946e5),
    ("synthetic/U32 spp1 None rps4", 0xf4acb50a444946e5),
    ("synthetic/U32 spp1 Lzw one strip", 0xf4acb50a444946e5),
    ("synthetic/U32 spp1 Lzw rps4", 0xf4acb50a444946e5),
    ("synthetic/U32 spp1 Lzw+pred one strip", 0xf4acb50a444946e5),
    ("synthetic/U32 spp1 Lzw+pred rps4", 0xf4acb50a444946e5),
    ("synthetic/U32 spp3 None one strip", 0x5e84f45d7a612b60),
    ("synthetic/U32 spp3 None rps4", 0x5e84f45d7a612b60),
    ("synthetic/U32 spp3 Lzw one strip", 0x5e84f45d7a612b60),
    ("synthetic/U32 spp3 Lzw rps4", 0x5e84f45d7a612b60),
    ("synthetic/U32 spp3 Lzw+pred one strip", 0x5e84f45d7a612b60),
    ("synthetic/U32 spp3 Lzw+pred rps4", 0x5e84f45d7a612b60),
    ("synthetic/F32 spp1 None one strip", 0x607e8a951950b5b3),
    ("synthetic/F32 spp1 None rps4", 0x607e8a951950b5b3),
    ("synthetic/F32 spp1 Lzw one strip", 0x607e8a951950b5b3),
    ("synthetic/F32 spp1 Lzw rps4", 0x607e8a951950b5b3),
    ("synthetic/F32 spp1 Lzw+pred one strip", 0x607e8a951950b5b3),
    ("synthetic/F32 spp1 Lzw+pred rps4", 0x607e8a951950b5b3),
    ("synthetic/F32 spp3 None one strip", 0x8fc9666d0453e313),
    ("synthetic/F32 spp3 None rps4", 0x8fc9666d0453e313),
    ("synthetic/F32 spp3 Lzw one strip", 0x8fc9666d0453e313),
    ("synthetic/F32 spp3 Lzw rps4", 0x8fc9666d0453e313),
    ("synthetic/F32 spp3 Lzw+pred one strip", 0x8fc9666d0453e313),
    ("synthetic/F32 spp3 Lzw+pred rps4", 0x8fc9666d0453e313),
];

/// Every file read as a PSF gives the volume it gave before, sample for
/// sample, or fails where it failed: against the old reader, live, and
/// against the digests of what it gave.
#[test]
fn the_psf_reader_reads_what_it_read_before() {
    let files = psf_files();
    assert!(
        files.len() > 100,
        "only {} files: the fixtures are missing",
        files.len()
    );
    let print = std::env::var_os("FASTTIFF_PRINT_DIGESTS").is_some();
    let mut rows = Vec::new();
    for (name, path) in &files {
        let path = path.display().to_string();
        let digest = psf_digest(&load_psf(&path));
        assert_eq!(
            digest,
            psf_digest(&old::load_psf(&path)),
            "{name}: not what the old reader gave"
        );
        rows.push(format!("    ({name:?}, {digest:#018x}),"));
        let want = PSF_DIGESTS.iter().find(|(n, _)| n == name).map(|&(_, d)| d);
        if !print {
            assert_eq!(want, Some(digest), "{name}: not what it read before");
        }
    }
    if print {
        println!(
            "const PSF_DIGESTS: &[(&str, u64)] = &[\n{}\n];",
            rows.join("\n")
        );
    }
    let unchecked: Vec<&str> = PSF_DIGESTS
        .iter()
        .map(|&(n, _)| n)
        .filter(|n| !files.iter().any(|(f, _)| f == n))
        .collect();
    assert!(
        unchecked.is_empty(),
        "stored answers no file was checked against: {unchecked:?}"
    );
}

/// Counts the allocations of at least a given size made on the threads that
/// ask it to — this crate's unit tests run side by side, and only the one
/// measuring should be counted.
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        /// Allocations of at least this many bytes are counted, on this
        /// thread; `usize::MAX` counts none.
        static AT_LEAST: Cell<usize> = const { Cell::new(usize::MAX) };
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }

    struct Counting;

    fn note(size: usize) {
        // `try_with`: an allocation made while this thread's locals are torn
        // down is simply not counted.
        let _ = AT_LEAST.try_with(|at_least| {
            if size >= at_least.get() {
                let _ = COUNT.try_with(|n| n.set(n.get() + 1));
            }
        });
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            note(layout.size());
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            note(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            note(new_size);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Counting = Counting;

    /// How many allocations of at least `at_least` bytes `f` made on this
    /// thread.
    pub(super) fn allocations(at_least: usize, f: impl FnOnce()) -> usize {
        COUNT.with(|n| n.set(0));
        AT_LEAST.with(|a| a.set(at_least));
        f();
        AT_LEAST.with(|a| a.set(usize::MAX));
        COUNT.with(|n| n.get())
    }
}

/// A PSF stored as LZW in two strips a slice decodes each slice straight into
/// the volume, through one scratch for them all: what it allocates of half a
/// strip or more — the file, the volume, one strip's worth of decoded samples
/// — is the same for sixteen slices as for two. It was a strip's decode
/// buffer per slice, besides a buffer each slice was decoded into first.
#[test]
fn a_compressed_psf_allocates_nothing_per_slice() {
    use fast_tiff_lib::{Compression, SampleType, TiffWriter, WriterOptions};
    // Two strips of 32 rows, 128 KiB each decompressed: counted from half of
    // that, which is more than the 32 KiB of tables the LZW decoder makes
    // for every strip it reads (a codec's cost, not a frame's).
    let (w, h) = (2048u32, 64u32);
    let strip = w as usize * 32 * 2;
    let write = |slices: usize| {
        let opts = WriterOptions::new(w, h, SampleType::U16)
            .compression(Compression::Lzw)
            .rows_per_strip(32);
        let mut wr = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
        for z in 0..slices {
            let px: Vec<u16> = (0..w * h).map(|i| (i % 509 + z as u32) as u16).collect();
            wr.write_frame_u16(&px).unwrap();
        }
        let path = temp(&format!("lzw-psf-{slices}.tif"));
        std::fs::write(&path, wr.finish().unwrap().into_inner()).unwrap();
        path.display().to_string()
    };
    let (two, sixteen) = (write(2), write(16));
    let mut psf = None;
    let few = counting::allocations(strip / 2, || psf = Some(load_psf(&two)));
    let psf = psf.unwrap().expect("two slices read");
    assert_eq!(psf.dims.len(), (w * h) as usize * 2);
    assert_eq!(psf.data[(w * h) as usize + 5], 6.0, "slice 1, sample 5");
    let mut psf = None;
    let many = counting::allocations(strip / 2, || psf = Some(load_psf(&sixteen)));
    psf.unwrap().expect("sixteen slices read");
    assert_eq!(
        many, few,
        "allocations of a strip or more: {few} for two slices, {many} for sixteen"
    );
}

// ------------------------------------------------------------------ metadata

/// A generated image describes itself and nothing else.
#[test]
fn fresh_metadata_carries_no_second_opinion() {
    let info = fresh_info("psf".into(), 0.1, 0.25);
    assert_eq!(info.spacing.x, Some(0.1));
    assert_eq!(info.spacing.z, Some(0.25));
    assert_eq!(info.unit.as_deref(), Some("micron"));
    // The one that matters: a carried `description` is a second ImageJ block
    // in the written file, and the dialect keeps the first occurrence of each
    // key — so a stale `slices=` in it silently redescribes the result's
    // shape. See `fast_tiff_lib::metadata::imagej::OWNED_KEYS`.
    assert_eq!(info.description, None);
    assert_eq!(info.path, None);
}

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

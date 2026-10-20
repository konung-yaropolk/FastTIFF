//! Every built-in plugin gives the same answer through the C boundary as it
//! does when called directly.
//!
//! The registry holds nothing but plugins that came through the C ABI: a
//! library through its query function, a built-in through the same
//! registration run in this process. So the app never calls a built-in's trait
//! directly — and this is what proves that costs nothing. Each built-in is run
//! both ways on one stack, with one set of values, and the two answers have to
//! be the same: the same dialog, the same pixels to the bit, the same
//! metadata, the same plot, the same error.
//!
//! That is a test of the boundary as much as of the plugins. A field the
//! marshalling forgets, a sample type it mangles, an error kind it folds into
//! another, all show up here as a built-in whose two answers differ, which is
//! how the two gaps minor 4 closed — the channel names a result carries, and
//! the colour tables a plugin is shown — would have been caught.
//!
//! What is *not* compared is what the host deliberately does not take from a
//! plugin: the name and path inside a result's metadata, which the host fills
//! in itself (it knows which file it asked about), and never writes.

#![cfg(feature = "builtin-plugins")]

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::plugins::{builtin, describe_view, Origin, Registry, StackHost};
use fast_tiff_viewer::Stack;
use fasttiff_plugin_api::{
    ExportRequest, Exporter, ImageResult, ImportRequest, Importer, Outcome, ParamDecl, ParamKind,
    ParamValue, Params, Plugin, PluginError, StackInfo, VolumeMode, VolumeView,
};
use std::io::Cursor;

const W: u32 = 24;
const H: u32 = 20;
const C: usize = 2;
const Z: usize = 3;
const T: usize = 4;

/// A stack with something in it for every plugin to work on: two channels in
/// colour, three slices, four frames, a calibration, and values that vary in
/// space and time so that a projection, a registration or a deconvolution has
/// a picture to act on rather than a constant.
///
/// The channels are given names and colours. The names do not come back —
/// the host never reports an open stack's channel names to a plugin, so there
/// is nothing for a filter to keep — but the colours do, as the tables the
/// PNG exporter draws through.
fn stack() -> Stack {
    let meta = StackMetaWrite::new(C, Z)
        .unit("micron")
        .pixel_size(0.25, 0.25)
        .spacing(1.5)
        .frame_interval_s(2.0)
        .channel("DAPI", [0, 0, 255])
        .channel("GFP", [0, 255, 0])
        // The tables a reader actually shows the channels in, which is what
        // reaches a plugin as `view().luts`.
        .mode(fast_tiff_lib::DisplayMode::Composite)
        .channel_lut(fast_tiff_lib::color_ramp_lut([255, 0, 255]))
        .channel_lut(fast_tiff_lib::color_ramp_lut([0, 255, 0]));
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(meta);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for t in 0..T {
        for z in 0..Z {
            for c in 0..C {
                let px: Vec<u16> = (0..W * H)
                    .map(|i| {
                        let (x, y) = ((i % W) as f32, (i / W) as f32);
                        let blob =
                            (-((x - 9.0 - t as f32).powi(2) + (y - 8.0).powi(2)) / 18.0).exp();
                        (200.0
                            + 3000.0 * blob * (1.0 + c as f32)
                            + 40.0 * (x * 0.7 + z as f32).sin()
                            + 25.0 * (y * 0.3).cos()) as u16
                    })
                    .collect();
                let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.write_frame_bytes(&bytes).unwrap();
            }
        }
    }
    let bytes = w.finish().unwrap().into_inner();
    Stack::from_bytes(bytes, "parity.tif".into(), false).expect("open")
}

fn host(s: &Stack) -> StackHost {
    StackHost::new(
        s,
        describe_view(
            s,
            1,
            false,
            VolumeView {
                mode: VolumeMode::Mip,
                density: 1.0,
                iso: 0.5,
                eye: [0.0; 3],
                forward: [0.0, 0.0, 1.0],
                up: [0.0, 1.0, 0.0],
                right: [1.0, 0.0, 0.0],
            },
        ),
    )
}

/// The parts of a result's metadata the host fills in itself, cleared.
fn without_host_fields(info: &mut Option<StackInfo>) {
    if let Some(i) = info {
        i.name.clear();
        i.path = None;
    }
}

fn normalised_image(img: &mut ImageResult) {
    without_host_fields(&mut img.metadata);
}

/// An outcome as text, with what the host owns cleared. `Debug` is exact for
/// this: an `f32` prints as the shortest decimal that reads back as the same
/// bits, so two planes print alike only if they hold the same values — and,
/// unlike `==`, a NaN compares equal to the NaN it came from.
fn shown(r: Result<Outcome, PluginError>) -> String {
    let r = r.map(|o| match o {
        Outcome::NewDocument(mut img) => {
            normalised_image(&mut img);
            Outcome::NewDocument(img)
        }
        Outcome::ReplaceDocument(mut img) => {
            normalised_image(&mut img);
            Outcome::ReplaceDocument(img)
        }
        Outcome::SaveToFile { mut image, path } => {
            normalised_image(&mut image);
            Outcome::SaveToFile { image, path }
        }
        other => other,
    });
    format!("{r:?}")
}

/// Run `T` directly and through the registry, and require the same answer.
///
/// Returns its id, so the caller can check that every built-in was covered.
fn same_both_ways<P: Plugin + Default + 'static>(s: &Stack) -> String {
    same_both_ways_with::<P>(s, |_| {})
}

/// [`same_both_ways`], with the dialog's defaults adjusted by `set` first.
fn same_both_ways_with<P: Plugin + Default + 'static>(
    s: &Stack,
    set: impl FnOnce(&mut Params),
) -> String {
    let direct = P::default();
    let id = direct.info().id;

    let mut reg = Registry::empty();
    assert!(reg.add::<P>(Origin::BuiltIn), "{:?}", reg.problems);
    let entry = reg.get_mut(0).expect("installed");
    assert_eq!(entry.info, direct.info(), "{id}: the description changed");

    // The dialog, both ways.
    let decls: Vec<ParamDecl> = direct.params(&host(s));
    let crossed = entry.plugin.dialog(&host(s)).expect("the dialog crossed");
    assert_eq!(crossed, decls, "{id}: the dialog changed on the way across");

    // And the dialog again with values moved off their defaults, as the user
    // would move them: a dialog that follows its own choices has to be shown
    // those choices, through the boundary as directly. Every choice to its
    // last option, every switch flipped.
    let mut moved = Params::defaults(&decls);
    for d in &decls {
        match &d.kind {
            ParamKind::Choice { options, .. } if !options.is_empty() => {
                moved.set(d.key.clone(), ParamValue::Choice(options.len() - 1));
            }
            ParamKind::Bool { default } => moved.set(d.key.clone(), ParamValue::Bool(!default)),
            _ => {}
        }
    }
    let pending = || host(s).with_pending_params(moved.clone());
    let followed = direct.params(&pending());
    assert_eq!(
        entry.plugin.dialog(&pending()).expect("the dialog crossed"),
        followed,
        "{id}: the dialog did not follow its choices through the boundary"
    );

    // A run with the dialog's own defaults, both ways.
    let mut values = Params::defaults(&decls).clamp_to(&decls);
    set(&mut values);
    let by_hand = shown(P::default().run(&mut host(s), &values));
    let through = shown(entry.plugin.run(&mut host(s), &values));
    assert_eq!(through, by_hand, "{id}: the run changed on the way across");
    id
}

/// Every filter the build carries, both ways — and the list has to be every
/// one of them, or a plugin added later would be installed through the
/// boundary without anything checking that it survives the trip.
#[test]
// The literal clippy asks for cannot be written when every element is
// conditional on a feature.
#[allow(clippy::vec_init_then_push)]
fn every_built_in_filter_answers_the_same_through_the_boundary() {
    let s = stack();
    #[allow(unused_mut)]
    let mut covered: Vec<String> = Vec::new();
    #[cfg(feature = "plugin-invert")]
    covered.push(same_both_ways::<builtin::Invert>(&s));
    #[cfg(feature = "plugin-zproject")]
    covered.push(same_both_ways::<builtin::ZProject>(&s));
    #[cfg(feature = "plugin-plot-axis")]
    covered.push(same_both_ways::<builtin::PlotAxis>(&s));
    #[cfg(feature = "plugin-stabilize")]
    covered.push(same_both_ways::<builtin::Stabilize>(&s));
    #[cfg(feature = "plugin-turboreg")]
    covered.push(same_both_ways::<builtin::TurboReg>(&s));
    #[cfg(feature = "plugin-stack-tools")]
    {
        covered.push(same_both_ways::<builtin::SliceKeeper>(&s));
        covered.push(same_both_ways::<builtin::SliceRemover>(&s));
        covered.push(same_both_ways::<builtin::SliceOrderInvert>(&s));
    }
    #[cfg(feature = "plugin-deconvolve")]
    {
        covered.push(same_both_ways::<builtin::GeneratePsf>(&s));
        covered.push(same_both_ways::<builtin::Deconvolve>(&s));
    }

    let mut installed: Vec<String> = Registry::new()
        .entries()
        .iter()
        .map(|e| e.info.id.clone())
        .collect();
    installed.sort();
    covered.sort();
    assert_eq!(
        covered, installed,
        "every installed built-in filter must be checked here"
    );
}

/// The PNG exporter draws the displayed frame through the channels' colour
/// tables, so it is the plugin that shows whether those crossed: before minor
/// 4, the same exporter in a library wrote a grey picture of a composite.
#[cfg(feature = "plugin-png")]
#[test]
fn the_png_exporter_writes_the_same_file_through_the_boundary() {
    let s = stack();
    let dir = std::env::temp_dir().join(format!("ft-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (a, b) = (dir.join("direct.png"), dir.join("through.png"));

    // The fixture's own premise: its channels are drawn in colour, or a
    // picture that lost its tables would look the same as one that kept them.
    let h = host(&s);
    let luts = &fasttiff_plugin_api::HostContext::view(&h).luts;
    assert_eq!(luts.len(), C, "one table per channel");
    assert_eq!(
        luts[0][255],
        [255, 0, 255],
        "the first channel should be magenta"
    );

    let decls = builtin::Png.params(&host(&s));
    let values = Params::defaults(&decls).clamp_to(&decls);
    builtin::Png
        .export(
            &ExportRequest {
                path: a.clone(),
                params: values.clone(),
            },
            &mut host(&s),
        )
        .expect("direct export");

    let mut reg = Registry::empty();
    reg.add_exporter::<builtin::Png>(Origin::BuiltIn);
    let entry = reg.exporter_mut(0).unwrap();
    assert_eq!(entry.exporter.dialog(&host(&s)).unwrap(), decls);
    entry
        .exporter
        .export(
            &ExportRequest {
                path: b.clone(),
                params: values,
            },
            &mut host(&s),
        )
        .expect("export through the boundary");

    let (da, db) = (std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!da.is_empty());
    assert!(da == db, "the two PNGs differ");
}

/// And the PNG importer reads that file back the same both ways — what the
/// importer says about the file included, apart from the name and path the
/// host gives an import itself.
#[cfg(feature = "plugin-png")]
#[test]
fn the_png_importer_reads_the_same_through_the_boundary() {
    let s = stack();
    let dir = std::env::temp_dir().join(format!("ft-parity-in-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("frame.png");
    builtin::Png
        .export(
            &ExportRequest {
                path: file.clone(),
                params: Params::new(),
            },
            &mut host(&s),
        )
        .expect("export");

    struct Quiet;
    impl fasttiff_plugin_api::ImportHost for Quiet {
        fn progress(&mut self, _f: f32) -> bool {
            true
        }
        fn log(&mut self, _m: &str) {}
    }
    let request = ImportRequest {
        path: file.clone(),
        params: Params::new(),
    };
    let head = Registry::read_head(&file);
    let mut direct = builtin::PngImport;
    let by_hand = direct.import(&request, &mut Quiet).expect("direct import");

    let mut reg = Registry::empty();
    reg.add_importer::<builtin::PngImport>(Origin::BuiltIn);
    let entry = reg.importer_mut(0).unwrap();
    assert_eq!(
        entry.importer.probe(&file, &head),
        direct.probe(&file, &head)
    );
    let through = entry
        .importer
        .import(&request, &mut Quiet)
        .expect("import through the boundary");
    let _ = std::fs::remove_dir_all(&dir);

    let (mut a, mut b) = (by_hand, through);
    for r in [&mut a, &mut b] {
        r.image.name.clear();
        normalised_image(&mut r.image);
        without_host_fields(&mut r.info);
    }
    assert_eq!(format!("{:?}", b.image), format!("{:?}", a.image));
    // An importer that says nothing about the file still gets an empty record
    // from the host, which has a name and path for it either way.
    assert_eq!(
        b.info.unwrap_or_default(),
        a.info.unwrap_or_default(),
        "what the importer said about the file changed on the way across"
    );
}

/// Deconvolution with a real PSF, both ways — the one built-in whose default
/// run stops at "no PSF file chosen", so the comparison above only checks its
/// refusal. Here it reads a file of its own choosing and does the arithmetic.
#[cfg(feature = "plugin-deconvolve")]
#[test]
fn deconvolution_with_a_real_psf_answers_the_same_through_the_boundary() {
    let s = stack();
    // A PSF from the built-in that makes them, saved where Deconvolve can
    // open it.
    // Smaller than the 24x20 frame, which Deconvolve requires of a PSF.
    let decls = builtin::GeneratePsf.params(&host(&s));
    let mut values = Params::defaults(&decls);
    values.set("width", fasttiff_plugin_api::ParamValue::Int(9));
    values.set("height", fasttiff_plugin_api::ParamValue::Int(9));
    values.set("slices", fasttiff_plugin_api::ParamValue::Int(3));
    let values = values.clamp_to(&decls);
    let Ok(Outcome::NewDocument(psf)) = builtin::GeneratePsf.run(&mut host(&s), &values) else {
        panic!("Generate PSF should produce a document");
    };
    let dir = std::env::temp_dir().join(format!("ft-parity-psf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("psf.tif");
    let bytes = fast_tiff_viewer::plugins::to_tiff_bytes(&psf, None).expect("encode the PSF");
    std::fs::write(&file, bytes).unwrap();

    let path = file.display().to_string();
    same_both_ways_with::<builtin::Deconvolve>(&s, |v| {
        v.set(
            "psf_path",
            fasttiff_plugin_api::ParamValue::Path(path.clone()),
        );
    });
    let _ = std::fs::remove_dir_all(&dir);
}

use super::*;
use fasttiff_plugin_api::PlaneData;

fn result(channels: usize, slices: usize, frames: usize, w: u32, h: u32) -> ImageResult {
    let n = (w * h) as usize;
    let mut planes = Vec::new();
    // xyczt order, each plane tagged so a re-ordering is visible.
    for t in 0..frames.max(1) {
        for z in 0..slices.max(1) {
            for c in 0..channels.max(1) {
                let tag = (c * 100 + z * 10 + t) as f32;
                planes.push(PlaneData::F32((0..n).map(|_| tag).collect()));
            }
        }
    }
    ImageResult {
        width: w,
        height: h,
        channels,
        slices,
        frames,
        pixel_type: PixelType::F32,
        planes,
        channel_colors: Vec::new(),
        metadata: None,
        name: "result".into(),
    }
}

/// The round trip must preserve the shape and every pixel, in order — this is
/// what makes a plugin result a real document rather than an approximation.
#[test]
fn a_result_round_trips_through_the_ordinary_reader() {
    // All three axes above 1, so the resolver keeps the Z axis.
    let img = result(2, 3, 2, 5, 4);
    let stack = to_stack(&img, None, false).expect("the result should open");

    assert_eq!(stack.dimensions(), Some((5, 4)));
    assert_eq!(
        (
            stack.display.dims.channels,
            stack.display.dims.slices,
            stack.display.dims.frames
        ),
        (2, 3, 2),
        "the declared axes must survive the round trip"
    );

    // And the pixels landed where they were declared to be.
    let view = crate::plugins::describe_view(
        &stack,
        0,
        false,
        fasttiff_plugin_api::VolumeView {
            mode: fasttiff_plugin_api::VolumeMode::Mip,
            density: 1.0,
            iso: 0.5,
            eye: [0.0; 3],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
        },
    );
    let mut host = crate::plugins::StackHost::new(&stack, view);
    use fasttiff_plugin_api::{HostContext, Plane};
    let mut buf = Vec::new();
    for t in 0..2 {
        for z in 0..3 {
            for c in 0..2 {
                host.read_plane_f32(Plane::new(c, z, t), &mut buf).unwrap();
                assert_eq!(
                    buf[0],
                    (c * 100 + z * 10 + t) as f32,
                    "plane (c{c}, z{z}, t{t}) came back as something else"
                );
            }
        }
    }
}

#[test]
fn every_pixel_type_survives_the_round_trip() {
    let cases: Vec<(PixelType, PlaneData)> = vec![
        (PixelType::U8, PlaneData::U8(vec![7u8; 4])),
        (PixelType::U16, PlaneData::U16(vec![4242u16; 4])),
        (PixelType::F32, PlaneData::F32(vec![1.5f32; 4])),
    ];
    for (ty, plane) in cases {
        let img = ImageResult {
            width: 2,
            height: 2,
            channels: 1,
            slices: 1,
            frames: 1,
            pixel_type: ty,
            planes: vec![plane],
            channel_colors: Vec::new(),
            metadata: None,
            name: "t".into(),
        };
        let stack = to_stack(&img, None, false).unwrap_or_else(|e| panic!("{ty:?}: {e:#}"));
        assert_eq!(stack.dimensions(), Some((2, 2)), "{ty:?}");
    }
}

/// A malformed result must be refused here, before it becomes a corrupt file.
#[test]
fn a_malformed_result_is_refused_before_encoding() {
    let mut img = result(1, 1, 1, 2, 2);
    img.planes = vec![PlaneData::F32(vec![0.0; 3])]; // one sample short
    let err = to_tiff_bytes(&img, None).unwrap_err().to_string();
    assert!(err.contains("unusable"), "unexpected error: {err}");
}

/// A multi-channel result with no metadata should composite, because that is
/// what a plugin producing several channels almost always means.
#[test]
fn a_multichannel_result_defaults_to_composite() {
    let img = result(2, 1, 2, 3, 3);
    let stack = to_stack(&img, None, false).unwrap();
    assert_eq!(stack.tiff.meta.mode, fast_tiff_lib::DisplayMode::Composite);
}

/// Metadata an importer supplies must reach the opened document.
#[test]
fn supplied_metadata_reaches_the_document() {
    let img = result(1, 1, 1, 2, 2);
    let info = StackInfo {
        name: "from-instrument".into(),
        unit: Some("micron".into()),
        frame_interval_s: Some(0.25),
        mode: DisplayMode::Grayscale,
        ..Default::default()
    };
    let stack = to_stack(&img, Some(&info), false).unwrap();
    assert_eq!(stack.tiff.meta.unit.as_deref(), Some("micron"));
    assert_eq!(stack.tiff.meta.frame_interval_s, Some(0.25));
    assert!(
        stack.path.to_string_lossy().contains("from-instrument"),
        "the supplied name should title the document"
    );
}

// --------------------------------------------------------------- source LUTs

/// A LUT that is not a ramp of anything, so a table rebuilt from one colour
/// cannot pass for it.
fn odd_lut(seed: u8) -> fasttiff_plugin_api::Lut {
    let mut lut = [[0u8; 3]; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let i = i as u8;
        *entry = [
            i.wrapping_mul(7).wrapping_add(seed),
            255 - i,
            (i / 2).wrapping_add(seed),
        ];
    }
    lut
}

/// The tables the source is shown with reach the written file, entry for entry.
///
/// A stabilised copy of a recording should open looking like the recording. The
/// whole table travels, not a colour: a file's own LUT need not be a ramp, and
/// one rebuilt from a single colour would be a different picture.
#[test]
fn the_sources_luts_are_carried_into_the_result() {
    let img = result(2, 1, 2, 5, 4);
    let luts = [odd_lut(3), odd_lut(200)];
    let bytes = to_tiff_bytes_reporting(&img, None, &luts, &mut |_| true)
        .expect("encode")
        .expect("not cancelled");
    let back = fast_tiff_lib::TiffStack::from_bytes(bytes).expect("the result should open");

    let written: Vec<_> = back.meta.channel_display.iter().map(|d| d.lut).collect();
    assert_eq!(
        written.len(),
        2,
        "one LUT per channel should have been written"
    );
    assert_eq!(written[0], luts[0], "channel 0's table changed");
    assert_eq!(written[1], luts[1], "channel 1's table changed");
}

/// A plugin that states its own colours outranks the source's tables: it has an
/// opinion about what its result means, and the source only about how it looked.
#[test]
fn a_plugins_own_colours_win_over_the_sources_luts() {
    let mut img = result(1, 1, 2, 5, 4);
    img.channel_colors = vec![[0, 0, 255]];
    let bytes = to_tiff_bytes_reporting(&img, None, &[odd_lut(3)], &mut |_| true)
        .expect("encode")
        .expect("not cancelled");
    let back = fast_tiff_lib::TiffStack::from_bytes(bytes).expect("open");
    assert_eq!(
        back.meta.channel_display[0].lut,
        fast_tiff_lib::color_ramp_lut([0, 0, 255]),
        "the plugin's colour was overwritten by the source's table"
    );
}

/// LUTs for a different number of channels are not applied at all.
///
/// A plugin that changed the channel count — a projection down to one, a split
/// into several — produced something whose channels do not mean what the
/// source's did, and colouring them by position would be inventing a
/// correspondence.
#[test]
fn luts_are_ignored_when_the_channel_count_changed() {
    let img = result(1, 1, 2, 5, 4);
    let bytes = to_tiff_bytes_reporting(&img, None, &[odd_lut(3), odd_lut(9)], &mut |_| true)
        .expect("encode")
        .expect("not cancelled");
    let back = fast_tiff_lib::TiffStack::from_bytes(bytes).expect("open");
    assert_ne!(
        back.meta.channel_display[0].lut,
        odd_lut(3),
        "a two-channel source's tables were put on a one-channel result"
    );
}

/// LUTs survive next to channel *names*.
///
/// Names and LUTs are appended to the same per-channel list inside
/// `StackMetaWrite`, and the ImageJ writer drops the whole LUT block when its
/// count does not match the channel count. A result carrying both — which any
/// stabilised recording with named channels does — has to come out with both.
#[test]
fn luts_and_channel_names_both_survive() {
    let img = result(2, 1, 2, 5, 4);
    let info = StackInfo {
        name: "source".into(),
        channel_names: vec!["GCaMP".into(), "tdTomato".into()],
        ..Default::default()
    };
    let luts = [odd_lut(11), odd_lut(77)];
    let bytes = to_tiff_bytes_reporting(&img, Some(&info), &luts, &mut |_| true)
        .expect("encode")
        .expect("not cancelled");
    let back = fast_tiff_lib::TiffStack::from_bytes(bytes).expect("open");

    let written: Vec<_> = back.meta.channel_display.iter().map(|d| d.lut).collect();
    assert_eq!(
        written.len(),
        2,
        "the LUT block was dropped when names were written too"
    );
    assert_eq!(written[0], luts[0]);
    assert_eq!(written[1], luts[1]);
}

/// Taking the result writes the same file as borrowing it: dropping each plane
/// once it is written changes the memory the encode needs, not what it makes.
/// A malformed result is refused the same way, and a stop is still a stop.
#[test]
fn taking_the_result_encodes_the_same_file() {
    let img = result(2, 3, 2, 7, 5);
    let luts = [odd_lut(3), odd_lut(9)];
    let borrowed = to_tiff_bytes_reporting(&img, None, &luts, &mut |_| true)
        .unwrap()
        .unwrap();
    let taken = into_tiff_bytes_reporting(img.clone(), None, &luts, &mut |_| true)
        .unwrap()
        .unwrap();
    assert_eq!(borrowed, taken);
    assert_eq!(
        to_tiff_bytes(&img, None).unwrap(),
        into_tiff_bytes(img.clone(), None).unwrap()
    );

    let mut bad = img.clone();
    bad.planes.pop();
    assert!(into_tiff_bytes(bad, None).is_err());
    let mut calls = 0;
    let stopped = into_tiff_bytes_reporting(img, None, &[], &mut |_| {
        calls += 1;
        calls < 3
    })
    .unwrap();
    assert!(stopped.is_none(), "stopped on the third plane");
}

// ------------------------------------------------------------- streaming

/// FNV-1a, to pin a file's bytes in a line.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Results covering what the header carries — calibration, names, a vendor
/// description, the source's tables, a plugin's own colour, signed and 8-bit
/// samples, tables for the wrong number of channels — each with the length and
/// hash of the file the encoder wrote for it before results were streamed.
fn golden_cases() -> Vec<(ImageResult, Vec<fasttiff_plugin_api::Lut>, usize, u64)> {
    let mut cases = Vec::new();
    // 1: u16, two channels, three frames, a full set of metadata and LUTs.
    let mut planes = Vec::new();
    for t in 0..3 {
        for c in 0..2 {
            planes.push(PlaneData::U16(
                (0..30).map(|i| (i * 977 + c * 31 + t * 7) as u16).collect(),
            ));
        }
    }
    cases.push((
        ImageResult {
            width: 6,
            height: 5,
            channels: 2,
            slices: 1,
            frames: 3,
            pixel_type: PixelType::U16,
            planes,
            channel_colors: Vec::new(),
            metadata: Some(StackInfo {
                name: "src.tif".into(),
                path: None,
                mode: DisplayMode::Composite,
                unit: Some("micron".into()),
                spacing: fasttiff_plugin_api::Spacing {
                    x: Some(0.25),
                    y: Some(0.25),
                    z: Some(1.5),
                },
                frame_interval_s: Some(2.0),
                channel_names: vec!["GCaMP".into(), "tdTomato".into()],
                calibration: Some((-3.0, 0.5)),
                description: Some("vendor=thing\nsetting=1".into()),
            }),
            name: "src-stabilized".into(),
        },
        vec![odd_lut(3), odd_lut(200)],
        3192,
        0xaf06_895e_8b81_8d72,
    ));
    // 2: f32, one channel, three slices, two frames, a colour of its own.
    let mut planes = Vec::new();
    for t in 0..2 {
        for z in 0..3 {
            planes.push(PlaneData::F32(
                (0..12)
                    .map(|i| (i as f32 / 12.0) * (1.0 + z as f32) - t as f32 * 0.5)
                    .collect(),
            ));
        }
    }
    cases.push((
        ImageResult {
            width: 4,
            height: 3,
            channels: 1,
            slices: 3,
            frames: 2,
            pixel_type: PixelType::F32,
            planes,
            channel_colors: vec![[255, 0, 255]],
            metadata: None,
            name: "projection".into(),
        },
        Vec::new(),
        1956,
        0x6c85_2d1a_d58d_0906,
    ));
    // 3: i16 in the u16 lane, no metadata.
    let planes = (0..4)
        .map(|t| PlaneData::U16((0..9).map(|i| ((i as i16 - 4) * 1000 + t) as u16).collect()))
        .collect();
    cases.push((
        ImageResult {
            width: 3,
            height: 3,
            channels: 1,
            slices: 1,
            frames: 4,
            pixel_type: PixelType::I16,
            planes,
            channel_colors: Vec::new(),
            metadata: None,
            name: "signed".into(),
        },
        Vec::new(),
        636,
        0x653b_90af_0102_a254,
    ));
    // 4: u8, three channels, LUTs for a different channel count (ignored).
    let planes = (0..3)
        .map(|c| PlaneData::U8((0..16).map(|i| (i * 13 + c * 50) as u8).collect()))
        .collect();
    cases.push((
        ImageResult {
            width: 4,
            height: 4,
            channels: 3,
            slices: 1,
            frames: 1,
            pixel_type: PixelType::U8,
            planes,
            channel_colors: Vec::new(),
            metadata: None,
            name: "rgbish".into(),
        },
        vec![odd_lut(1), odd_lut(2)],
        504,
        0x172c_5962_bcb5_a381,
    ));
    cases
}

/// A directory of its own per test and process, so the "nothing left behind"
/// checks see only their own files.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ft-result-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn files_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(entries) => entries.map(|e| e.unwrap().path()).collect(),
        Err(_) => Vec::new(),
    }
}

/// The encoder writes, byte for byte, the files it wrote before results were
/// streamed — the hashes were taken from that encoder — whether a result
/// arrives whole or a plane at a time, in memory or in a file.
#[test]
fn a_streamed_result_is_the_file_the_whole_one_was() {
    let dir = scratch("same-bytes");
    for (i, (img, luts, len, hash)) in golden_cases().into_iter().enumerate() {
        let whole = to_tiff_bytes_reporting(&img, None, &luts, &mut |_| true)
            .unwrap()
            .unwrap();
        assert_eq!(
            (whole.len(), fnv1a(&whole)),
            (len, hash),
            "case {i}: the whole-result encode changed"
        );

        for dest in [ResultDest::Memory, ResultDest::Dir(dir.clone())] {
            let mut w = ResultWriter::begin(img.spec(), &luts, &dest).unwrap();
            for p in &img.planes {
                w.push(p.as_plane_ref()).unwrap();
            }
            let streamed = w.finish().unwrap();
            assert_eq!(streamed.spec(), &img.spec());
            let back = streamed.to_image().unwrap();
            assert_eq!(back, img, "case {i} via {dest:?}: read back differently");
            let bytes = streamed.into_bytes().unwrap();
            assert!(bytes == whole, "case {i} via {dest:?}: the file differs");
        }
    }
    assert!(
        files_in(&dir).is_empty(),
        "results nobody kept were left behind: {:?}",
        files_in(&dir)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Written as it arrives: after each plane the file already holds it, so the
/// writer is holding none of them.
#[test]
fn a_streamed_result_is_on_disk_plane_by_plane() {
    let dir = scratch("on-disk");
    let img = result(1, 1, 6, 64, 32);
    let plane_bytes = 64 * 32 * 4;
    let mut w = ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
    let path = w.path().expect("written to a file").to_path_buf();
    for (k, p) in img.planes.iter().enumerate() {
        w.push(p.as_plane_ref()).unwrap();
        let on_disk = std::fs::metadata(&path).unwrap().len() as usize;
        assert!(
            on_disk >= (k + 1) * plane_bytes,
            "after plane {k} the file holds {on_disk} bytes, short of {}",
            (k + 1) * plane_bytes
        );
    }
    let done = w.finish().unwrap();
    assert_eq!(done.path(), Some(path.as_path()));
    let kept = done.into_kept_path().expect("a file");
    assert!(kept.exists(), "a kept result must stay");
    std::fs::remove_file(&kept).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A result dropped before it was finished — the run was cancelled, failed or
/// panicked — leaves no file; so does one that came up short.
#[test]
fn an_unfinished_result_leaves_no_file() {
    let dir = scratch("unfinished");
    let img = result(1, 1, 4, 8, 8);

    let mut w = ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
    w.push(img.planes[0].as_plane_ref()).unwrap();
    assert_eq!(files_in(&dir).len(), 1, "the file is written as it goes");
    drop(w);
    assert!(files_in(&dir).is_empty(), "a dropped writer left its file");

    let mut w = ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
    w.push(img.planes[0].as_plane_ref()).unwrap();
    let e = w.finish().unwrap_err().to_string();
    assert!(e.contains("4 planes but carries 1"), "{e}");
    assert!(files_in(&dir).is_empty(), "a short result left its file");

    // And a stop part-way through a whole result.
    let mut calls = 0;
    let stopped = write_result(img, None, &[], &ResultDest::Dir(dir.clone()), &mut |_| {
        calls += 1;
        calls < 3
    })
    .unwrap();
    assert!(stopped.is_none());
    assert!(files_in(&dir).is_empty(), "a stopped encode left its file");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Each plane is checked against the declaration as it arrives, and a second
/// result of the same name goes in a directory of its own rather than in the
/// first one's place — keeping the clean name a window is titled with.
#[test]
fn planes_are_checked_and_names_do_not_collide() {
    let dir = scratch("names");
    let img = result(1, 1, 2, 4, 4);
    let mut first = ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
    let e = first
        .push(PlaneRef::F32(&[0.0; 3]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("plane 0 has 3 samples"), "{e}");
    let e = first.push(PlaneRef::U16(&[0; 16])).unwrap_err().to_string();
    assert!(e.contains("plane 0 is U16"), "{e}");

    let second = ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
    let names: Vec<String> = [&first, &second]
        .iter()
        .map(|w| {
            w.path()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into()
        })
        .collect();
    assert_eq!(names, ["result.tif", "result.tif"]);
    let dirs: Vec<_> = [&first, &second]
        .iter()
        .map(|w| w.path().unwrap().parent().unwrap().to_path_buf())
        .collect();
    assert_ne!(dirs[0], dirs[1], "two results shared a directory");
    for d in &dirs {
        assert_eq!(d.parent(), Some(dir.as_path()));
    }
    drop((first, second));
    assert!(
        files_in(&dir).is_empty(),
        "a dropped writer left its directory: {:?}",
        files_in(&dir)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// There is no limit on how many results there can be at once: the hundred
/// and first is written like the first. (A cap on names was how a busy
/// session ended up holding results in memory.)
#[test]
fn there_is_no_limit_on_results_kept_at_once() {
    let dir = scratch("many");
    let img = result(1, 1, 1, 2, 2);
    let kept: Vec<std::path::PathBuf> = (0..150)
        .map(|_| {
            let mut w =
                ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
            w.push(img.planes[0].as_plane_ref()).unwrap();
            let done = w.finish().unwrap();
            assert!(done.path().is_some(), "held in memory instead of a file");
            done.into_kept_path().unwrap()
        })
        .collect();
    assert_eq!(files_in(&dir).len(), 150);
    assert!(kept.iter().all(|p| p.ends_with("result.tif")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The sweep removes old run directories and loose files, and leaves alone
/// this process's own, recent ones, and anything it did not make.
#[test]
fn the_sweep_removes_only_old_results_it_made() {
    let dir = scratch("sweep");
    std::fs::create_dir_all(&dir).unwrap();
    let img = result(1, 1, 1, 2, 2);
    let keep = |dest: &std::path::Path| {
        let mut w = ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(dest.into())).unwrap();
        w.push(img.planes[0].as_plane_ref()).unwrap();
        w.finish().unwrap().into_kept_path().unwrap()
    };
    // Another process's result, and a loose file of the old flat layout.
    let other = dir.join("run-1-0-0");
    std::fs::create_dir(&other).unwrap();
    std::fs::write(other.join("old.tif"), b"x").unwrap();
    std::fs::write(dir.join("flat.tif"), b"x").unwrap();
    // Something else entirely.
    let stranger = dir.join("not-ours");
    std::fs::create_dir(&stranger).unwrap();
    // And this process's own.
    let mine = keep(&dir);

    // Nothing is a day old yet.
    assert_eq!(sweep(&dir, std::time::Duration::from_secs(86_400)), 0);
    assert!(other.exists() && dir.join("flat.tif").exists());

    // Everything is older than nothing: the other process's goes, the loose
    // file goes, this process's own and the stranger stay.
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(sweep(&dir, std::time::Duration::ZERO), 1);
    assert!(!other.exists(), "an old result was left");
    assert!(
        !dir.join("flat.tif").exists(),
        "an old loose result was left"
    );
    assert!(mine.exists(), "this process's own result was swept");
    assert!(stranger.exists(), "a directory it did not make was swept");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Where a result's file goes, and the bug the rule comes from: a window
/// showing a result **memory-maps that file for as long as it is open**.
/// Writing the next result over it fails outright on Windows, and the last
/// resort — show it in this window instead — then replaced the document the
/// user was working from. So a second result never lands on the first one's
/// file, and the first is never so much as opened for writing again — which
/// is what makes the rule hold where overwriting a mapped file quietly works.
#[test]
fn an_earlier_result_is_left_exactly_as_it_was() {
    let dir = scratch("untouched");
    let small = result(1, 1, 1, 2, 2);
    let mut w = ResultWriter::begin(small.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
    w.push(small.planes[0].as_plane_ref()).unwrap();
    let first = w.finish().unwrap().into_kept_path().unwrap();
    let bytes = std::fs::read(&first).unwrap();
    let before = std::fs::metadata(&first).unwrap().modified().ok();

    let big = result(1, 1, 3, 9, 9);
    let mut paths = vec![first.clone()];
    for _ in 0..2 {
        let mut w = ResultWriter::begin(big.spec(), &[], &ResultDest::Dir(dir.clone())).unwrap();
        for p in &big.planes {
            w.push(p.as_plane_ref()).unwrap();
        }
        let kept = w.finish().unwrap().into_kept_path().unwrap();
        assert!(kept.ends_with("result.tif"), "{}", kept.display());
        paths.push(kept);
    }
    paths.dedup();
    assert_eq!(paths.len(), 3, "a later result took an earlier one's file");
    assert_eq!(
        std::fs::read(&first).unwrap(),
        bytes,
        "the first result changed"
    );
    assert_eq!(
        std::fs::metadata(&first).unwrap().modified().ok(),
        before,
        "the first result was reopened"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Somewhere that cannot hold a file — here, a path whose parent is a file —
/// is not the end of the run: the result is held in memory, as every result
/// was before, and still comes out right.
#[test]
fn nowhere_to_write_holds_the_result_in_memory() {
    let dir = scratch("nowhere");
    std::fs::create_dir_all(&dir).unwrap();
    let blocker = dir.join("a-file");
    std::fs::write(&blocker, b"x").unwrap();
    let img = result(1, 1, 2, 3, 3);
    let mut w =
        ResultWriter::begin(img.spec(), &[], &ResultDest::Dir(blocker.join("inside"))).unwrap();
    assert!(w.path().is_none(), "there was nowhere to put a file");
    for p in &img.planes {
        w.push(p.as_plane_ref()).unwrap();
    }
    let done = w.finish().unwrap();
    assert!(done.path().is_none() && done.bytes().is_some());
    assert_eq!(done.to_image().unwrap(), img);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_result_name_becomes_a_file_name() {
    assert_eq!(file_stem("cells-stabilized"), "cells-stabilized");
    assert_eq!(file_stem("a/b:c*?"), "a_b_c__");
    assert_eq!(file_stem("  ..  "), "result");
    assert_eq!(file_stem(""), "result");
}

/// Poll for `path` to be gone, for as long as a closing document's workers
/// might take to let go of it.
fn gone_soon(path: &std::path::Path) -> bool {
    for _ in 0..100 {
        if !path.exists() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    !path.exists()
}

/// A result opened in this window is the document's: named as the result, it
/// says it was made from the document it came from — which is where "Save
/// as…" starts and what a plugin is told — and not from its scratch file,
/// and closing it deletes that file and its directory.
#[test]
fn a_result_opened_here_belongs_to_its_document() {
    let dir = scratch("owned");
    let source_dir = scratch("owned-source");
    std::fs::create_dir_all(&source_dir).unwrap();
    let source = source_dir.join("recording.tif");
    std::fs::write(&source, b"stands for the document it was made from").unwrap();

    let mut img = result(1, 1, 2, 4, 4);
    img.name = "recording-stabilized".into();
    let written = write_result(img, None, &[], &ResultDest::Dir(dir.clone()), &mut |_| true)
        .unwrap()
        .unwrap();
    let file = written.path().unwrap().to_path_buf();
    let run_dir = file.parent().unwrap().to_path_buf();

    let stack = written
        .into_document(Some(source.clone()))
        .unwrap()
        .load(false, &mut |_| {})
        .unwrap();
    assert_eq!(
        stack.path.file_name().unwrap(),
        "recording-stabilized.tif",
        "titled as the result"
    );
    assert_eq!(
        stack.source_path(),
        None,
        "the scratch file is not a source"
    );
    assert_eq!(crate::plugins::describe_stack(&stack).path, None);
    assert_eq!(stack.original().as_deref(), Some(source.as_path()));
    assert_eq!(stack.save_dir().as_deref(), Some(source_dir.as_path()));
    let derived = stack.derived.as_ref().expect("a derived document");
    assert_eq!(derived.made_from(), Some(source.as_path()));

    #[cfg(feature = "mmap")]
    {
        assert_eq!(derived.scratch(), Some(file.as_path()), "mapped, and owned");
        assert!(file.exists(), "deleted while it was on show");
    }
    #[cfg(not(feature = "mmap"))]
    assert!(!file.exists(), "read into memory, and the file let go");

    drop(stack);
    assert!(
        gone_soon(&file),
        "closing the document left its result file"
    );
    assert!(
        gone_soon(&run_dir),
        "closing the document left its directory"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&source_dir);
}

/// The memory a result falls back to is asked for, not taken: one too big for
/// the machine is an error saying so, rather than an abort.
#[test]
fn a_result_too_big_for_memory_is_an_error() {
    let spec = ImageSpec {
        width: 1 << 16,
        height: 1 << 16,
        channels: 1,
        slices: 1,
        frames: 1 << 20,
        pixel_type: PixelType::F32,
        channel_colors: Vec::new(),
        metadata: None,
        name: "huge".into(),
    };
    let e = ResultWriter::begin(spec, &[], &ResultDest::Memory)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("cannot be held in memory") || e.contains("too large"),
        "{e}"
    );
}

/// A colour set after the planes is in the file: the description is written
/// when the result is finished, so a plugin may say it at any point until then.
#[test]
fn a_description_changed_after_the_planes_is_what_the_file_says() {
    let dir = scratch("late");
    let mut img = result(2, 1, 2, 4, 4);
    img.metadata = Some(StackInfo {
        unit: Some("micron".into()),
        ..Default::default()
    });
    for dest in [ResultDest::Memory, ResultDest::Dir(dir.clone())] {
        let mut w = ResultWriter::begin(img.spec(), &[], &dest).unwrap();
        for p in &img.planes {
            w.push(p.as_plane_ref()).unwrap();
        }
        let mut later = img.spec();
        later.channel_colors = vec![[255, 0, 255], [0, 255, 0]];
        later.metadata.as_mut().unwrap().channel_names = vec!["a".into(), "b".into()];
        w.redescribe(later.clone()).unwrap();

        // The shape and the name are the declaration's.
        let mut wider = later.clone();
        wider.width += 1;
        assert!(w.redescribe(wider).is_err());
        let mut renamed = later.clone();
        renamed.name = "other".into();
        assert!(w.redescribe(renamed).is_err());

        let done = w.finish().unwrap();
        assert_eq!(done.spec(), &later, "the spec kept up");
        let back = fast_tiff_lib::TiffStack::from_bytes(done.into_bytes().unwrap()).unwrap();
        assert!(
            back.meta.has_explicit_luts,
            "{dest:?}: the late colours are missing"
        );
        assert_eq!(back.meta.channel_display[0].lut[255], [255, 0, 255]);
        assert_eq!(back.meta.channel_display[1].lut[255], [0, 255, 0]);
        assert_eq!(back.meta.unit.as_deref(), Some("micron"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

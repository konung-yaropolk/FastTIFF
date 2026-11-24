//! The batch readers — the 3D volume build, the save, and a plugin's plane
//! reads — checked against what they gave before they moved onto
//! fast-tiff-lib's [`Frame`](fast_tiff_lib::Frame) (item 25c).
//!
//! [`DIGESTS`] were taken from the code as it was before the move, 25b's
//! tree, where every one of these read a frame through the library's old
//! readers. For every file here — every fixture, fuzz seed and fuzz
//! regression, and a synthetic matrix of depths, signs, channel counts,
//! interleavings, 4D shapes, strip layouts and codecs — a digest of:
//!
//! - every volume the 3D view builds of it: at full size, reduced to a
//!   budget (each voxel the average of a box of them), in half floats, and at
//!   the last timepoint of a 4D stack;
//! - the file a save writes of it, which must also be the same whether the
//!   frames reach the writer all at once, one at a time or a few at a time;
//! - every plane a plugin reads of it, as `u16` through the view's window and
//!   as `f32` in the file's units, into a fresh buffer and into one left dirty
//!   by the reads before — which must agree.
//!
//! A file that does not open, a volume that is not built and a read that
//! fails are digested as such, so an error where there was none, or none
//! where there was one, shows as plainly as a changed sample.
//!
//! While the library's old readers still exist, a plugin's reads are also
//! compared live with them ([`old`]), plane by plane. They go in item 25e,
//! and [`old`] with them; the digests stay, and the case list must not change
//! after that — a digest is only as good as the code it was taken from.

use crate::document::Document;
use crate::plugins::{describe_view, StackHost};
use crate::volume::{build_volume, DEFAULT_VOLUME_BUDGET};
use crate::worker::CancelToken;
use crate::{VolumeInputs, VolumeSettings};
use fast_tiff_lib::{Compression, SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fasttiff_plugin_api::{HostContext, Plane, VolumeMode, VolumeView};
use scivis_render::pack::{self, PackedVolume};
use scivis_render::VolumeCaps;
use std::path::{Path, PathBuf};

/// The library's old readers, as a plugin's reads called them before: the
/// live half of the comparison, until they go.
mod old {
    use crate::document::Document;
    use fasttiff_plugin_api::Plane;

    /// The IFD and sample plane `plane` reads, as the host resolves it.
    fn located(doc: &Document, p: Plane) -> (usize, usize) {
        crate::dimensions::plane_index(
            doc.display.dims(),
            doc.display.layout().shown_as_rgb(),
            p.c,
            p.z,
            p.t,
        )
    }

    pub(super) fn plane_u16(
        doc: &Document,
        p: Plane,
        range: Option<(f32, f32)>,
    ) -> Option<Vec<u16>> {
        let (ifd, sample) = located(doc, p);
        let tiff = &doc.tiff;
        let mut out = Vec::new();
        fast_tiff_lib::read_plane_u16_into(
            &tiff.data,
            tiff.frames.get(ifd)?,
            tiff.byte_order,
            range,
            sample,
            &mut out,
        )
        .ok()?;
        Some(out)
    }

    pub(super) fn plane_f32(doc: &Document, p: Plane) -> Option<Vec<f32>> {
        let (ifd, sample) = located(doc, p);
        let tiff = &doc.tiff;
        let mut out = Vec::new();
        fast_tiff_lib::read_plane_f32_into(
            &tiff.data,
            tiff.frames.get(ifd)?,
            tiff.byte_order,
            sample,
            &mut out,
        )
        .ok()?;
        Some(out)
    }
}

// ------------------------------------------------------------------ files

fn lib_dir(sub: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fast-tiff-lib")
        .join(sub)
}

/// Every TIFF in `dir`, by name, whether or not it opens.
fn files_in(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tif" || x == "tiff"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let name = format!(
                "{}/{}",
                dir.file_name()?.to_string_lossy(),
                p.file_name()?.to_string_lossy()
            );
            Some((name, std::fs::read(&p).ok()?))
        })
        .collect()
}

/// A stack of `planes` planes written by this crate's writer, with ImageJ's
/// `channels` and `slices`, so it opens with that shape.
#[allow(clippy::too_many_arguments)]
fn written(
    st: SampleType,
    spp: u16,
    planar: bool,
    (channels, slices, planes): (usize, usize, usize),
    compression: Compression,
    predictor: bool,
    rows_per_strip: Option<u32>,
) -> Vec<u8> {
    let (w, h) = (13u32, 10u32);
    let mut opts = WriterOptions::new(w, h, st)
        .samples_per_pixel(spp)
        .planar(planar)
        .compression(compression)
        .predictor(predictor)
        .metadata(StackMetaWrite::new(channels, slices));
    if let Some(rows) = rows_per_strip {
        opts = opts.rows_per_strip(rows);
    }
    let mut wr = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    let n = (w * h) as usize * spp as usize * st.bytes();
    for f in 0..planes {
        // Every byte varies, so a sample read from the wrong place, plane or
        // frame shows; floats are kept finite and modest.
        let bytes: Vec<u8> = (0..n)
            .map(|i| {
                (i as u8)
                    .wrapping_mul(37)
                    .wrapping_add((f as u8).wrapping_mul(101))
                    ^ (i >> 8) as u8
            })
            .map(|b| if st == SampleType::F32 { b & 0x3f } else { b })
            .collect();
        wr.write_frame_bytes(&bytes).unwrap();
    }
    wr.finish().unwrap().into_inner()
}

/// The synthetic matrix: depths and signs; one channel, two, a 4D stack, and
/// RGB chunky and planar; one strip a frame or several; uncompressed, LZW,
/// and LZW with the predictor.
fn synthetic() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let types = [
        SampleType::U8,
        SampleType::I8,
        SampleType::U16,
        SampleType::I16,
        SampleType::U32,
        SampleType::F32,
    ];
    // (samples per pixel, planar, (channels, slices, planes), label)
    let shapes: [(u16, bool, (usize, usize, usize), &str); 5] = [
        (1, false, (1, 1, 5), "movie"),
        (1, false, (2, 1, 6), "2ch"),
        (1, false, (2, 2, 8), "2ch 4D"),
        (3, false, (1, 1, 4), "rgb"),
        (3, true, (1, 1, 4), "rgb planar"),
    ];
    let codecs = [
        (Compression::None, false),
        (Compression::Lzw, false),
        (Compression::Lzw, true),
    ];
    for st in types {
        for (spp, planar, shape, label) in shapes {
            for (compression, predictor) in codecs {
                for rps in [None, Some(3)] {
                    let name = format!(
                        "synthetic/{st:?} {label} {compression:?}{} {}",
                        if predictor { "+pred" } else { "" },
                        rps.map_or("one strip".to_string(), |r| format!("rps{r}"))
                    );
                    let bytes = written(st, spp, planar, shape, compression, predictor, rps);
                    out.push((name, bytes));
                }
            }
        }
    }
    out
}

fn every_file() -> Vec<(String, Vec<u8>)> {
    let mut all = files_in(&lib_dir("tests/fixtures"));
    all.extend(files_in(&lib_dir("tests/fuzz-regressions")));
    all.extend(files_in(&lib_dir("fuzz/seeds")));
    all.extend(synthetic());
    all
}

// ------------------------------------------------------------------ digests

/// 64-bit FNV-1a.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn byte(&mut self, b: u8) {
        self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    fn bytes(&mut self, bs: &[u8]) {
        for &b in bs {
            self.byte(b);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// A volume, every texel of every level of every texture and every brick
/// maximum; or that there was none.
fn digest_volume(built: &Option<PackedVolume>, h: &mut Fnv) {
    let Some(p) = built else {
        h.byte(0);
        return;
    };
    h.byte(1);
    let (x, y, z) = p.layout.dims;
    for v in [x, y, z, p.layout.levels] {
        h.u64(v as u64);
    }
    h.bytes(format!("{:?}", p.layout.format).as_bytes());
    for texture in &p.levels {
        h.u64(texture.len() as u64);
        for level in texture {
            h.u64(level.len() as u64);
            h.bytes(level);
        }
    }
    let (bx, by, bz) = p.bricks.dims;
    for v in [bx, by, bz] {
        h.u64(v as u64);
    }
    for channel in &p.bricks.max {
        h.u64(channel.len() as u64);
        for m in channel {
            h.bytes(&m.to_bits().to_le_bytes());
        }
    }
}

// ------------------------------------------------------------------ cases

const VIEW: VolumeView = VolumeView {
    mode: VolumeMode::Mip,
    density: 1.0,
    iso: 0.5,
    eye: [0.0; 3],
    forward: [0.0, 0.0, 1.0],
    up: [0.0, 1.0, 0.0],
    right: [1.0, 0.0, 0.0],
};

/// Every volume the 3D view builds of `doc`: at full size in the formats a
/// GPU with 16-bit and float textures stores, in half floats where it has
/// neither, and reduced to a fifth of the full size's bytes; at the first
/// timepoint, and at the last of a 4D stack.
fn volume_cases(doc: &Document, h: &mut Fnv) -> usize {
    let full = VolumeCaps {
        max_3d_size: 2048,
        unorm16: true,
        float32: true,
    };
    let halves = VolumeCaps {
        max_3d_size: 2048,
        unorm16: false,
        float32: false,
    };
    let view = VolumeSettings::default();
    let inputs = VolumeInputs::default();
    let dims = doc.display.dims();
    let mut times = vec![0];
    if dims.slices > 1 && dims.frames > 1 {
        times.push(dims.frames - 1);
    }
    let mut cases = 0;
    for time in times {
        for (caps, reduce) in [(full, true), (halves, false)] {
            let plan =
                crate::sync::plan_volume(doc, caps, time, DEFAULT_VOLUME_BUDGET, &view, &inputs);
            let built =
                build_volume(&doc.tiff, &plan, &CancelToken::never()).expect("not cancelled");
            digest_volume(&built, h);
            cases += 1;
            if reduce {
                // A fifth of what the full size takes: every axis it can give
                // is reduced by a box average.
                let Some(p) = &built else { continue };
                let (x, y, z) = p.layout.dims;
                let bytes = x as f64
                    * y as f64
                    * z as f64
                    * pack::bytes_per_voxel(
                        plan.units.len().min(scivis_render::MAX_CHANNELS),
                        p.layout.format,
                    );
                let small = crate::sync::plan_volume(
                    doc,
                    caps,
                    time,
                    ((bytes / 5.0) as usize).max(1),
                    &view,
                    &inputs,
                );
                let reduced =
                    build_volume(&doc.tiff, &small, &CancelToken::never()).expect("not cancelled");
                digest_volume(&reduced, h);
                cases += 1;
            }
        }
    }
    cases
}

/// The file a save writes of `doc`, with every frame in one batch; the same
/// file, byte for byte, with one frame a batch and with two.
///
/// Not of a stack whose first frame claims more samples than its strips can
/// hold: the save before the move sized its batch from that claim, and a
/// hostile one aborted the process rather than failing. That it is an error
/// now is [`a_save_of_a_frame_claiming_more_than_its_strips_hold_fails`].
fn save_cases(doc: &Document, h: &mut Fnv) -> usize {
    if doc
        .tiff
        .frame(0)
        .and_then(|f| f.stored_frame_len())
        .is_err()
    {
        h.byte(2);
        return 1;
    }
    let source = crate::save::SaveSource::of(doc);
    let save = |batch_bytes: usize| {
        crate::save::write_all(
            &source,
            std::io::Cursor::new(Vec::new()),
            &mut |_| true,
            batch_bytes,
        )
        .map(std::io::Cursor::into_inner)
        .ok()
    };
    // The save's own batch, which holds every frame of every file here.
    let whole = save(64 << 20);
    match &whole {
        None => h.byte(0),
        Some(bytes) => {
            h.byte(1);
            h.u64(bytes.len() as u64);
            h.bytes(bytes);
        }
    }
    let frame_bytes = doc.tiff.frames.first().map_or(1, |f| {
        f.width as usize * f.height as usize * f.samples_per_pixel.max(1) as usize
    });
    for batch_bytes in [1, frame_bytes * 2 + 1] {
        assert!(
            save(batch_bytes) == whole,
            "saved in batches of {batch_bytes} bytes, the file differs"
        );
    }
    3
}

/// One read's answer, digested: the samples' bytes, or that it failed.
fn digest_read<T: bytemuck::Pod>(got: &Option<Vec<T>>, h: &mut Fnv) {
    match got {
        None => h.byte(0),
        Some(v) => {
            h.byte(1);
            h.u64(v.len() as u64);
            h.bytes(bytemuck::cast_slice(v));
        }
    }
}

/// Every plane a plugin can read of `doc` (the first 64), as `u16` through
/// the view's window and as `f32`: into a fresh buffer, into one the reads
/// before left dirty, and — while they exist — through the old readers.
fn plugin_read_cases(doc: &Document, h: &mut Fnv) -> usize {
    let view = describe_view(doc, 0, false, VIEW);
    let mut host = StackHost::new(doc, view.clone());
    let image = host.image();
    let mut dirty16: Vec<u16> = vec![0xa5a5; 7];
    let mut dirty32: Vec<f32> = vec![f32::NAN; 3];
    let mut cases = 0;
    'planes: for t in 0..image.frames {
        for z in 0..image.slices {
            for c in 0..image.channels {
                if cases >= 64 {
                    break 'planes;
                }
                let p = Plane { c, z, t };
                let mut fresh16 = Vec::new();
                let got16 = host.read_plane_u16(p, &mut fresh16).ok().map(|()| fresh16);
                let again16 = host
                    .read_plane_u16(p, &mut dirty16)
                    .ok()
                    .map(|()| dirty16.clone());
                assert!(
                    got16 == again16,
                    "{p:?} as u16: a dirty buffer reads differently"
                );
                let range = view.channels.get(c).map(|ch| (ch.min, ch.max));
                assert!(
                    got16 == old::plane_u16(doc, p, range),
                    "{p:?} as u16: not what the old reader gave"
                );
                digest_read(&got16, h);

                let mut fresh32 = Vec::new();
                let got32 = host.read_plane_f32(p, &mut fresh32).ok().map(|()| fresh32);
                let again32 = host
                    .read_plane_f32(p, &mut dirty32)
                    .ok()
                    .map(|()| dirty32.clone());
                let bits = |v: &Option<Vec<f32>>| {
                    v.as_ref()
                        .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                };
                assert!(
                    bits(&got32) == bits(&again32),
                    "{p:?} as f32: a dirty buffer reads differently"
                );
                assert!(
                    bits(&got32) == bits(&old::plane_f32(doc, p)),
                    "{p:?} as f32: not what the old reader gave"
                );
                digest_read(&got32, h);
                cases += 2;
            }
        }
    }
    cases
}

/// Every case of one file, digested; and how many there were.
fn check_file(name: &str, bytes: &[u8]) -> (u64, usize) {
    let mut h = Fnv::new();
    let Ok(doc) = Document::from_bytes(bytes.to_vec(), PathBuf::from(name), false) else {
        h.byte(0xff);
        return (h.0, 1);
    };
    let mut cases = volume_cases(&doc, &mut h);
    cases += save_cases(&doc, &mut h);
    cases += plugin_read_cases(&doc, &mut h);
    (h.0, cases)
}

/// Printed by `FASTTIFF_PRINT_DIGESTS=1 cargo test -p fast-tiff-viewer
/// --lib batch_oracle -- --nocapture`, at 25b's tree, before the batch
/// readers moved.
const DIGESTS: &[(&str, u64)] = &[
    ("fixtures/ij_u16_spp1_p6_hyperstack.tif", 0x17d46802343cbf42),
    ("fixtures/pil_u16_spp1_p1_lzw.tif", 0x2f5aa36c504d994a),
    ("fixtures/pil_u8_spp1_p1_lzw.tif", 0x29417b4dc85c819e),
    ("fixtures/pil_u8_spp1_p1_pb.tif", 0x29417b4dc85c819e),
    ("fixtures/pil_u8_spp3_p1_zip.tif", 0x8fa88e834b642a0d),
    ("fixtures/pil_u8_spp4_p1_cmyk-lzw.tif", 0x39385107a126111a),
    (
        "fixtures/tff_f32_spp1_p1_tiled-zip-pred3.tif",
        0xb5cf0792a82bc30a,
    ),
    ("fixtures/tff_f32_spp1_p2_lzw-pred3.tif", 0x55a1cbfeb3ff068b),
    ("fixtures/tff_f32_spp1_p2_none-le.tif", 0x55a1cbfeb3ff068b),
    (
        "fixtures/tff_f32_spp1_p2_zip-pred3-be.tif",
        0x55a1cbfeb3ff068b,
    ),
    ("fixtures/tff_f32_spp1_p2_zip-pred3.tif", 0x55a1cbfeb3ff068b),
    ("fixtures/tff_i16_spp1_p2_lzw-pred2.tif", 0x0677201a289845e9),
    ("fixtures/tff_i16_spp1_p2_none-be.tif", 0x0677201a289845e9),
    ("fixtures/tff_i16_spp1_p2_none-le.tif", 0x0677201a289845e9),
    ("fixtures/tff_i32_spp1_p2_none-le.tif", 0x48335970ba77dc8c),
    ("fixtures/tff_i8_spp1_p2_none-le.tif", 0xfbb7d77341c6bc1f),
    ("fixtures/tff_u16_spp1_p1_tiled.tif", 0x88ef1ec0eb1a3e27),
    (
        "fixtures/tff_u16_spp1_p2_lzw-pred2-be.tif",
        0x6b930ee52cb9bf91,
    ),
    (
        "fixtures/tff_u16_spp1_p2_lzw-pred2-rps4.tif",
        0x6b930ee52cb9bf91,
    ),
    ("fixtures/tff_u16_spp1_p2_lzw-rps4.tif", 0x6b930ee52cb9bf91),
    ("fixtures/tff_u16_spp1_p2_none-be.tif", 0x6b930ee52cb9bf91),
    ("fixtures/tff_u16_spp1_p2_none-le.tif", 0x6b930ee52cb9bf91),
    ("fixtures/tff_u16_spp1_p2_pb-rps4.tif", 0x6b930ee52cb9bf91),
    ("fixtures/tff_u16_spp1_p2_zip-pred2.tif", 0x6b930ee52cb9bf91),
    ("fixtures/tff_u16_spp1_p2_zip-rps4.tif", 0x6b930ee52cb9bf91),
    ("fixtures/tff_u16_spp3_p2_zip-pred2.tif", 0x7dc735aeaa4eed48),
    (
        "fixtures/tff_u16_spp4_p1_cmyk-zip-pred2.tif",
        0xfd7ab0dd64994d91,
    ),
    ("fixtures/tff_u16_spp4_p1_cmyk.tif", 0xfd7ab0dd64994d91),
    ("fixtures/tff_u32_spp1_p2_none-le.tif", 0x784e1f7a7b6abe16),
    ("fixtures/tff_u32_spp1_p2_zip-pred2.tif", 0x784e1f7a7b6abe16),
    ("fixtures/tff_u8_spp1_p2_none-le.tif", 0x1126e19a4524d3e5),
    ("fixtures/tff_u8_spp1_p2_zip-pred2.tif", 0x1126e19a4524d3e5),
    (
        "fixtures/tff_u8_spp3_p1_tiled-lzw-pred2.tif",
        0x2fb2dace315909f5,
    ),
    ("fixtures/tff_u8_spp3_p2_lzw-pred2.tif", 0x7c2439a7a15c2467),
    ("fixtures/tff_u8_spp3_p2_none.tif", 0x7c2439a7a15c2467),
    ("fixtures/tff_u8_spp4_p1_cmyk-lzw.tif", 0xae3d22024e0cb2f0),
    ("fixtures/tff_u8_spp4_p1_cmyk.tif", 0xae3d22024e0cb2f0),
    (
        "fixtures/tff_u8_spp4_p2_cmyk-planar.tif",
        0x8179c26d156c3632,
    ),
    ("fixtures/tld_u16_spp1_p1_grid-lzw.tif", 0xa8b62fa5b804636d),
    ("fixtures/tld_u16_spp1_p1_grid.tif", 0xa8b62fa5b804636d),
    (
        "fixtures/tld_u8_spp3_p1_grid-planar.tif",
        0x448dec995351669c,
    ),
    ("fixtures/tld_u8_spp3_p1_grid-pred2.tif", 0xa24ab12c9fc07678),
    (
        "fuzz-regressions/overflow-crop-rows-height-u32max.tif",
        0xf60f45d92efbe6a5,
    ),
    (
        "fuzz-regressions/overflow-strip-band-h4g-rps2g.tif",
        0xf60f45d92efbe6a5,
    ),
    (
        "fuzz-regressions/overflow-tiled-crop-h4g-tile2g.tif",
        0xf60f45d92efbe6a5,
    ),
    (
        "fuzz-regressions/overflow-w2g-h2g-spp4.tif",
        0xef5a77972ccef8bd,
    ),
    ("fuzz-regressions/slow-tall-tile.tif", 0xe604823a249029bf),
    ("fuzz-regressions/slow-tile-grid.tif", 0x70a9794ec837cf75),
    ("seeds/bigtiff.tif", 0x58ffb34595506ea3),
    ("seeds/cmyk16_lzw.tif", 0xcc5481329bed6f3e),
    ("seeds/cmyk_chunky.tif", 0x2a3f334ed4fd0b45),
    ("seeds/cmyk_planar.tif", 0xe788d70b2682e43d),
    ("seeds/float32.tif", 0xf58531bcf4058877),
    ("seeds/float32_zlib_pred.tif", 0xf58531bcf4058877),
    ("seeds/gray16.tif", 0x58ffb34595506ea3),
    ("seeds/gray16_be.tif", 0x58ffb34595506ea3),
    ("seeds/gray16_lzw.tif", 0x58ffb34595506ea3),
    ("seeds/gray16_packbits.tif", 0x58ffb34595506ea3),
    ("seeds/gray16_strips.tif", 0xd3ba20c7306e5c55),
    ("seeds/gray16_zlib.tif", 0x58ffb34595506ea3),
    ("seeds/gray16_zlib_pred.tif", 0x58ffb34595506ea3),
    ("seeds/gray8.tif", 0x6fb81acd9dfa77d5),
    ("seeds/imagej_hyperstack.tif", 0xa08d4907e3f077f1),
    ("seeds/imagej_lut.tif", 0x732b15569d35f6f0),
    ("seeds/ome.tif", 0x5ce761f234bd195e),
    ("seeds/palette4.tif", 0x2dd21c10e88556be),
    ("seeds/palette8.tif", 0xc26e2851a824c442),
    ("seeds/regression-oom-huge-width.tif", 0xf60f45d92efbe6a5),
    ("seeds/rgb_chunky.tif", 0x46076dfb870e8c1a),
    ("seeds/rgb_planar.tif", 0x1e1fb658dcc1a445),
    ("seeds/stack3.tif", 0x88cf28c1ba57ab87),
    ("seeds/tiled16.tif", 0x87932b3337fdf9f1),
    ("seeds/tiled16_lzw.tif", 0x87932b3337fdf9f1),
    ("seeds/tiled_planar.tif", 0x59dad7e6d6fc8ae6),
    ("seeds/tiled_rgb_pred.tif", 0x5bdd0473755f7a71),
    ("synthetic/U8 movie None one strip", 0xdf984b9dc6bfd23f),
    ("synthetic/U8 movie None rps3", 0xdf984b9dc6bfd23f),
    ("synthetic/U8 movie Lzw one strip", 0xdf984b9dc6bfd23f),
    ("synthetic/U8 movie Lzw rps3", 0xdf984b9dc6bfd23f),
    ("synthetic/U8 movie Lzw+pred one strip", 0xdf984b9dc6bfd23f),
    ("synthetic/U8 movie Lzw+pred rps3", 0xdf984b9dc6bfd23f),
    ("synthetic/U8 2ch None one strip", 0x93af6f11c59950e1),
    ("synthetic/U8 2ch None rps3", 0x93af6f11c59950e1),
    ("synthetic/U8 2ch Lzw one strip", 0x93af6f11c59950e1),
    ("synthetic/U8 2ch Lzw rps3", 0x93af6f11c59950e1),
    ("synthetic/U8 2ch Lzw+pred one strip", 0x93af6f11c59950e1),
    ("synthetic/U8 2ch Lzw+pred rps3", 0x93af6f11c59950e1),
    ("synthetic/U8 2ch 4D None one strip", 0x34959950a44aa484),
    ("synthetic/U8 2ch 4D None rps3", 0x34959950a44aa484),
    ("synthetic/U8 2ch 4D Lzw one strip", 0x34959950a44aa484),
    ("synthetic/U8 2ch 4D Lzw rps3", 0x34959950a44aa484),
    ("synthetic/U8 2ch 4D Lzw+pred one strip", 0x34959950a44aa484),
    ("synthetic/U8 2ch 4D Lzw+pred rps3", 0x34959950a44aa484),
    ("synthetic/U8 rgb None one strip", 0x3dd15791603dc33e),
    ("synthetic/U8 rgb None rps3", 0x3dd15791603dc33e),
    ("synthetic/U8 rgb Lzw one strip", 0x3dd15791603dc33e),
    ("synthetic/U8 rgb Lzw rps3", 0x3dd15791603dc33e),
    ("synthetic/U8 rgb Lzw+pred one strip", 0x3dd15791603dc33e),
    ("synthetic/U8 rgb Lzw+pred rps3", 0x3dd15791603dc33e),
    ("synthetic/U8 rgb planar None one strip", 0x580248069338493a),
    ("synthetic/U8 rgb planar None rps3", 0x580248069338493a),
    ("synthetic/U8 rgb planar Lzw one strip", 0x580248069338493a),
    ("synthetic/U8 rgb planar Lzw rps3", 0x580248069338493a),
    (
        "synthetic/U8 rgb planar Lzw+pred one strip",
        0x580248069338493a,
    ),
    ("synthetic/U8 rgb planar Lzw+pred rps3", 0x580248069338493a),
    ("synthetic/I8 movie None one strip", 0x6665b016b3b6c585),
    ("synthetic/I8 movie None rps3", 0x6665b016b3b6c585),
    ("synthetic/I8 movie Lzw one strip", 0x6665b016b3b6c585),
    ("synthetic/I8 movie Lzw rps3", 0x6665b016b3b6c585),
    ("synthetic/I8 movie Lzw+pred one strip", 0x6665b016b3b6c585),
    ("synthetic/I8 movie Lzw+pred rps3", 0x6665b016b3b6c585),
    ("synthetic/I8 2ch None one strip", 0x09aaf775aee33bed),
    ("synthetic/I8 2ch None rps3", 0x09aaf775aee33bed),
    ("synthetic/I8 2ch Lzw one strip", 0x09aaf775aee33bed),
    ("synthetic/I8 2ch Lzw rps3", 0x09aaf775aee33bed),
    ("synthetic/I8 2ch Lzw+pred one strip", 0x09aaf775aee33bed),
    ("synthetic/I8 2ch Lzw+pred rps3", 0x09aaf775aee33bed),
    ("synthetic/I8 2ch 4D None one strip", 0x9cfb949e346aea4e),
    ("synthetic/I8 2ch 4D None rps3", 0x9cfb949e346aea4e),
    ("synthetic/I8 2ch 4D Lzw one strip", 0x9cfb949e346aea4e),
    ("synthetic/I8 2ch 4D Lzw rps3", 0x9cfb949e346aea4e),
    ("synthetic/I8 2ch 4D Lzw+pred one strip", 0x9cfb949e346aea4e),
    ("synthetic/I8 2ch 4D Lzw+pred rps3", 0x9cfb949e346aea4e),
    ("synthetic/I8 rgb None one strip", 0xedfd562460c733b1),
    ("synthetic/I8 rgb None rps3", 0xedfd562460c733b1),
    ("synthetic/I8 rgb Lzw one strip", 0xedfd562460c733b1),
    ("synthetic/I8 rgb Lzw rps3", 0xedfd562460c733b1),
    ("synthetic/I8 rgb Lzw+pred one strip", 0xedfd562460c733b1),
    ("synthetic/I8 rgb Lzw+pred rps3", 0xedfd562460c733b1),
    ("synthetic/I8 rgb planar None one strip", 0xdf89ca7d55452d98),
    ("synthetic/I8 rgb planar None rps3", 0xdf89ca7d55452d98),
    ("synthetic/I8 rgb planar Lzw one strip", 0xdf89ca7d55452d98),
    ("synthetic/I8 rgb planar Lzw rps3", 0xdf89ca7d55452d98),
    (
        "synthetic/I8 rgb planar Lzw+pred one strip",
        0xdf89ca7d55452d98,
    ),
    ("synthetic/I8 rgb planar Lzw+pred rps3", 0xdf89ca7d55452d98),
    ("synthetic/U16 movie None one strip", 0x6ef9b5194f4b7130),
    ("synthetic/U16 movie None rps3", 0x6ef9b5194f4b7130),
    ("synthetic/U16 movie Lzw one strip", 0x6ef9b5194f4b7130),
    ("synthetic/U16 movie Lzw rps3", 0x6ef9b5194f4b7130),
    ("synthetic/U16 movie Lzw+pred one strip", 0x6ef9b5194f4b7130),
    ("synthetic/U16 movie Lzw+pred rps3", 0x6ef9b5194f4b7130),
    ("synthetic/U16 2ch None one strip", 0xb789c9b2ae4dae96),
    ("synthetic/U16 2ch None rps3", 0xb789c9b2ae4dae96),
    ("synthetic/U16 2ch Lzw one strip", 0xb789c9b2ae4dae96),
    ("synthetic/U16 2ch Lzw rps3", 0xb789c9b2ae4dae96),
    ("synthetic/U16 2ch Lzw+pred one strip", 0xb789c9b2ae4dae96),
    ("synthetic/U16 2ch Lzw+pred rps3", 0xb789c9b2ae4dae96),
    ("synthetic/U16 2ch 4D None one strip", 0xb8e4662d79c0b49b),
    ("synthetic/U16 2ch 4D None rps3", 0xb8e4662d79c0b49b),
    ("synthetic/U16 2ch 4D Lzw one strip", 0xb8e4662d79c0b49b),
    ("synthetic/U16 2ch 4D Lzw rps3", 0xb8e4662d79c0b49b),
    (
        "synthetic/U16 2ch 4D Lzw+pred one strip",
        0xb8e4662d79c0b49b,
    ),
    ("synthetic/U16 2ch 4D Lzw+pred rps3", 0xb8e4662d79c0b49b),
    ("synthetic/U16 rgb None one strip", 0x3d3dd35b70b191bb),
    ("synthetic/U16 rgb None rps3", 0x3d3dd35b70b191bb),
    ("synthetic/U16 rgb Lzw one strip", 0x3d3dd35b70b191bb),
    ("synthetic/U16 rgb Lzw rps3", 0x3d3dd35b70b191bb),
    ("synthetic/U16 rgb Lzw+pred one strip", 0x3d3dd35b70b191bb),
    ("synthetic/U16 rgb Lzw+pred rps3", 0x3d3dd35b70b191bb),
    (
        "synthetic/U16 rgb planar None one strip",
        0x330d41d9f15dac39,
    ),
    ("synthetic/U16 rgb planar None rps3", 0x330d41d9f15dac39),
    ("synthetic/U16 rgb planar Lzw one strip", 0x330d41d9f15dac39),
    ("synthetic/U16 rgb planar Lzw rps3", 0x330d41d9f15dac39),
    (
        "synthetic/U16 rgb planar Lzw+pred one strip",
        0x330d41d9f15dac39,
    ),
    ("synthetic/U16 rgb planar Lzw+pred rps3", 0x330d41d9f15dac39),
    ("synthetic/I16 movie None one strip", 0xf75c6e3dd572ea62),
    ("synthetic/I16 movie None rps3", 0xf75c6e3dd572ea62),
    ("synthetic/I16 movie Lzw one strip", 0xf75c6e3dd572ea62),
    ("synthetic/I16 movie Lzw rps3", 0xf75c6e3dd572ea62),
    ("synthetic/I16 movie Lzw+pred one strip", 0xf75c6e3dd572ea62),
    ("synthetic/I16 movie Lzw+pred rps3", 0xf75c6e3dd572ea62),
    ("synthetic/I16 2ch None one strip", 0x12843d8c29130407),
    ("synthetic/I16 2ch None rps3", 0x12843d8c29130407),
    ("synthetic/I16 2ch Lzw one strip", 0x12843d8c29130407),
    ("synthetic/I16 2ch Lzw rps3", 0x12843d8c29130407),
    ("synthetic/I16 2ch Lzw+pred one strip", 0x12843d8c29130407),
    ("synthetic/I16 2ch Lzw+pred rps3", 0x12843d8c29130407),
    ("synthetic/I16 2ch 4D None one strip", 0xa769056e1cc8a8ca),
    ("synthetic/I16 2ch 4D None rps3", 0xa769056e1cc8a8ca),
    ("synthetic/I16 2ch 4D Lzw one strip", 0xa769056e1cc8a8ca),
    ("synthetic/I16 2ch 4D Lzw rps3", 0xa769056e1cc8a8ca),
    (
        "synthetic/I16 2ch 4D Lzw+pred one strip",
        0xa769056e1cc8a8ca,
    ),
    ("synthetic/I16 2ch 4D Lzw+pred rps3", 0xa769056e1cc8a8ca),
    ("synthetic/I16 rgb None one strip", 0x270832c2e239e0ff),
    ("synthetic/I16 rgb None rps3", 0x270832c2e239e0ff),
    ("synthetic/I16 rgb Lzw one strip", 0x270832c2e239e0ff),
    ("synthetic/I16 rgb Lzw rps3", 0x270832c2e239e0ff),
    ("synthetic/I16 rgb Lzw+pred one strip", 0x270832c2e239e0ff),
    ("synthetic/I16 rgb Lzw+pred rps3", 0x270832c2e239e0ff),
    (
        "synthetic/I16 rgb planar None one strip",
        0xc79d7e45ac5de0d5,
    ),
    ("synthetic/I16 rgb planar None rps3", 0xc79d7e45ac5de0d5),
    ("synthetic/I16 rgb planar Lzw one strip", 0xc79d7e45ac5de0d5),
    ("synthetic/I16 rgb planar Lzw rps3", 0xc79d7e45ac5de0d5),
    (
        "synthetic/I16 rgb planar Lzw+pred one strip",
        0xc79d7e45ac5de0d5,
    ),
    ("synthetic/I16 rgb planar Lzw+pred rps3", 0xc79d7e45ac5de0d5),
    ("synthetic/U32 movie None one strip", 0x5ef794cfeb3b2856),
    ("synthetic/U32 movie None rps3", 0x5ef794cfeb3b2856),
    ("synthetic/U32 movie Lzw one strip", 0x5ef794cfeb3b2856),
    ("synthetic/U32 movie Lzw rps3", 0x5ef794cfeb3b2856),
    ("synthetic/U32 movie Lzw+pred one strip", 0x5ef794cfeb3b2856),
    ("synthetic/U32 movie Lzw+pred rps3", 0x5ef794cfeb3b2856),
    ("synthetic/U32 2ch None one strip", 0xa470141c29b78170),
    ("synthetic/U32 2ch None rps3", 0xa470141c29b78170),
    ("synthetic/U32 2ch Lzw one strip", 0xa470141c29b78170),
    ("synthetic/U32 2ch Lzw rps3", 0xa470141c29b78170),
    ("synthetic/U32 2ch Lzw+pred one strip", 0xa470141c29b78170),
    ("synthetic/U32 2ch Lzw+pred rps3", 0xa470141c29b78170),
    ("synthetic/U32 2ch 4D None one strip", 0xdc4813a1fe5158b4),
    ("synthetic/U32 2ch 4D None rps3", 0xdc4813a1fe5158b4),
    ("synthetic/U32 2ch 4D Lzw one strip", 0xdc4813a1fe5158b4),
    ("synthetic/U32 2ch 4D Lzw rps3", 0xdc4813a1fe5158b4),
    (
        "synthetic/U32 2ch 4D Lzw+pred one strip",
        0xdc4813a1fe5158b4,
    ),
    ("synthetic/U32 2ch 4D Lzw+pred rps3", 0xdc4813a1fe5158b4),
    ("synthetic/U32 rgb None one strip", 0x66b7bfd2f80b35cc),
    ("synthetic/U32 rgb None rps3", 0x66b7bfd2f80b35cc),
    ("synthetic/U32 rgb Lzw one strip", 0x66b7bfd2f80b35cc),
    ("synthetic/U32 rgb Lzw rps3", 0x66b7bfd2f80b35cc),
    ("synthetic/U32 rgb Lzw+pred one strip", 0x66b7bfd2f80b35cc),
    ("synthetic/U32 rgb Lzw+pred rps3", 0x66b7bfd2f80b35cc),
    (
        "synthetic/U32 rgb planar None one strip",
        0x06d66e5a310321ca,
    ),
    ("synthetic/U32 rgb planar None rps3", 0x06d66e5a310321ca),
    ("synthetic/U32 rgb planar Lzw one strip", 0x06d66e5a310321ca),
    ("synthetic/U32 rgb planar Lzw rps3", 0x06d66e5a310321ca),
    (
        "synthetic/U32 rgb planar Lzw+pred one strip",
        0x06d66e5a310321ca,
    ),
    ("synthetic/U32 rgb planar Lzw+pred rps3", 0x06d66e5a310321ca),
    ("synthetic/F32 movie None one strip", 0xdb2606f28853548c),
    ("synthetic/F32 movie None rps3", 0xdb2606f28853548c),
    ("synthetic/F32 movie Lzw one strip", 0xdb2606f28853548c),
    ("synthetic/F32 movie Lzw rps3", 0xdb2606f28853548c),
    ("synthetic/F32 movie Lzw+pred one strip", 0xdb2606f28853548c),
    ("synthetic/F32 movie Lzw+pred rps3", 0xdb2606f28853548c),
    ("synthetic/F32 2ch None one strip", 0x34a27f7c80f0e0a4),
    ("synthetic/F32 2ch None rps3", 0x34a27f7c80f0e0a4),
    ("synthetic/F32 2ch Lzw one strip", 0x34a27f7c80f0e0a4),
    ("synthetic/F32 2ch Lzw rps3", 0x34a27f7c80f0e0a4),
    ("synthetic/F32 2ch Lzw+pred one strip", 0x34a27f7c80f0e0a4),
    ("synthetic/F32 2ch Lzw+pred rps3", 0x34a27f7c80f0e0a4),
    ("synthetic/F32 2ch 4D None one strip", 0x05831aed2bf54556),
    ("synthetic/F32 2ch 4D None rps3", 0x05831aed2bf54556),
    ("synthetic/F32 2ch 4D Lzw one strip", 0x05831aed2bf54556),
    ("synthetic/F32 2ch 4D Lzw rps3", 0x05831aed2bf54556),
    (
        "synthetic/F32 2ch 4D Lzw+pred one strip",
        0x05831aed2bf54556,
    ),
    ("synthetic/F32 2ch 4D Lzw+pred rps3", 0x05831aed2bf54556),
    ("synthetic/F32 rgb None one strip", 0xf4f0426966c49a67),
    ("synthetic/F32 rgb None rps3", 0xf4f0426966c49a67),
    ("synthetic/F32 rgb Lzw one strip", 0xf4f0426966c49a67),
    ("synthetic/F32 rgb Lzw rps3", 0xf4f0426966c49a67),
    ("synthetic/F32 rgb Lzw+pred one strip", 0xf4f0426966c49a67),
    ("synthetic/F32 rgb Lzw+pred rps3", 0xf4f0426966c49a67),
    (
        "synthetic/F32 rgb planar None one strip",
        0x37d9723431a07e04,
    ),
    ("synthetic/F32 rgb planar None rps3", 0x37d9723431a07e04),
    ("synthetic/F32 rgb planar Lzw one strip", 0x37d9723431a07e04),
    ("synthetic/F32 rgb planar Lzw rps3", 0x37d9723431a07e04),
    (
        "synthetic/F32 rgb planar Lzw+pred one strip",
        0x37d9723431a07e04,
    ),
    ("synthetic/F32 rgb planar Lzw+pred rps3", 0x37d9723431a07e04),
];

#[test]
fn the_batch_readers_equal_what_they_gave_before() {
    let files = every_file();
    assert!(
        files.len() > 200,
        "only {} files: the fixtures are missing",
        files.len()
    );
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let chunk = files.len().div_ceil(threads);
    let results: Vec<(String, u64, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = files
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|(name, bytes)| {
                            let (digest, cases) = check_file(name, bytes);
                            (name.clone(), digest, cases)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
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
        "stored answers no file was checked against: {unchecked:?}"
    );
    let cases: usize = results.iter().map(|r| r.2).sum();
    assert!(cases > 2_000, "only {cases} cases");
}

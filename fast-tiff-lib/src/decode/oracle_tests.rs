//! The decode core against the decoder it replaced, bit for bit.
//!
//! [`old`] is the decoder as it was at 3492148, kept verbatim (see its header)
//! as scaffolding until the old readers go. Every frame here — every fixture,
//! every fuzz regression and seed, and a synthetic matrix of depths, signs,
//! byte orders, sample counts, interleavings, strip layouts, tiles,
//! predictors and codecs — is read through every old reader and through its
//! wrapper over the core, and through the [`Frame`] API in every unit, plane
//! selection and crop, into fresh, dirty and exact destinations, with parallel
//! decoding off and on. The answers must be the same: equal samples, equal
//! [`DisplayMap`]s, and an error of the same kind where the old one failed.
//!
//! One kind of difference is allowed, and only where the old decoder
//! disagreed with itself. Its single-sample fast paths (`read_frame_u16_into`,
//! `read_frame_f32_into`, `read_frame_u8_into`, `read_frame_bytes_into`, and the
//! single-strip borrows) walked a tiled frame's tiles as if they were strips,
//! and refused a whole-frame strip whose RowsPerStrip undersold it, where its
//! general path read both correctly. There the core must give what the old
//! general path gave.

#[path = "oracle_old.rs"]
mod old;

use crate::decode::{parallel_decode_scope, PARALLEL_MIN_PIXELS};
use crate::display::DisplayMap;
use crate::frame::FrameBytes;
use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo, SampleFormat, TiffStack};
use crate::units::{CmykRgb16, CmykRgb8, Display, File, Stored};
use crate::{Crop, DecodeScratch, DisplayMaps, ErrorKind, Frame, Pixels, Result};
use std::fmt::Debug;
use std::path::Path;

// ------------------------------------------------------------- comparison

/// What a result is, for comparing: the value, or the kind of error.
fn outcome<T: Clone>(r: &Result<T>) -> std::result::Result<T, ErrorKind> {
    match r {
        Ok(v) => Ok(v.clone()),
        Err(e) => Err(e.kind()),
    }
}

fn same<T: PartialEq + Debug + Clone>(label: &str, old: &Result<T>, new: &Result<T>) {
    let (o, n) = (outcome(old), outcome(new));
    assert!(o == n, "{label}: old {:?} / new {:?}", short(&o), short(&n),);
}

/// Equal to the old reader, or — where the old fast path disagreed with the
/// old general path — equal to the general path.
fn same_or_fixed<T: PartialEq + Debug + Clone>(
    label: &str,
    old: &Result<T>,
    general: &Result<T>,
    new: &Result<T>,
) {
    let (o, g, n) = (outcome(old), outcome(general), outcome(new));
    if o == n {
        return;
    }
    assert!(
        o != g && g == n,
        "{label}: old {:?} / old general {:?} / new {:?}",
        short(&o),
        short(&g),
        short(&n)
    );
}

fn short<T: Debug>(v: &T) -> String {
    let s = format!("{v:?}");
    if s.len() > 200 {
        format!("{}…", &s[..200])
    } else {
        s
    }
}

/// `f32`s by their bits: `NaN == NaN`, and `-0.0 != 0.0`.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn bits_r(r: Result<Vec<f32>>) -> Result<Vec<u32>> {
    r.map(|v| bits(&v))
}

fn map_bits(m: DisplayMap) -> (u8, u32, u32) {
    match m {
        DisplayMap::Identity => (0, 0, 0),
        DisplayMap::Widen8 => (1, 0, 0),
        DisplayMap::SignedWiden8 => (2, 0, 0),
        DisplayMap::Signed16 => (3, 0, 0),
        DisplayMap::Linear { lo, hi } => (4, lo.to_bits(), hi.to_bits()),
    }
}

fn range_bits(r: Option<(f32, f32)>) -> Option<(u32, u32)> {
    r.map(|(a, b)| (a.to_bits(), b.to_bits()))
}

/// A buffer that already holds something, of the wrong length.
fn dirty<T: Copy>(fill: T, len: usize) -> Vec<T> {
    vec![fill; len]
}

fn spp_of(f: &FrameInfo) -> usize {
    (f.samples_per_pixel as usize).max(1)
}

/// Plane `p` gathered out of a whole stored frame (the old
/// `read_frame_bytes_into`'s layout): what a `Stored` plane read must give.
fn stored_plane(frame: &[u8], f: &FrameInfo, p: usize) -> Option<Vec<u8>> {
    let sb = f.sample_bytes().ok()?;
    let n = f.pixel_count().ok()?;
    let spp = spp_of(f);
    if f.is_planar() {
        return frame.get(p * n * sb..(p + 1) * n * sb).map(<[u8]>::to_vec);
    }
    let mut out = Vec::with_capacity(n * sb);
    for i in 0..n {
        let at = (i * spp + p) * sb;
        out.extend_from_slice(frame.get(at..at + sb)?);
    }
    Some(out)
}

fn swap_all(mut v: Vec<u8>, sb: usize) -> Vec<u8> {
    if sb > 1 {
        for c in v.chunks_exact_mut(sb) {
            c.reverse();
        }
    }
    v
}

// -------------------------------------------------- one frame, every read

/// Every read of `info` over `file`, old against new.
fn compare_frame(file: &[u8], info: &FrameInfo, order: ByteOrder, label: &str) {
    let spp = spp_of(info);
    let planes: Vec<usize> = [0usize, 1, 2, 3, 7]
        .into_iter()
        .filter(|&p| p <= spp)
        .collect();
    let other = match order {
        ByteOrder::Little => ByteOrder::Big,
        ByteOrder::Big => ByteOrder::Little,
    };
    let ranges = [None, Some((-1000.0f32, 5000.0f32)), Some((3.0, 3.0))];

    // --- the old readers against their wrappers --------------------------
    for range in ranges {
        let l = format!("{label} range {range:?}");
        let old_general = old::read_plane_u16(file, info, order, range, 0);
        same_or_fixed(
            &format!("{l} read_frame_u16"),
            &old::read_frame_u16(file, info, order, range).map(|c| c.into_owned()),
            &old_general,
            &super::read_frame_u16(file, info, order, range).map(|c| c.into_owned()),
        );
        let into = |f: &dyn Fn(&mut Vec<u16>) -> Result<DisplayMap>, fill: Vec<u16>| {
            let mut v = fill;
            f(&mut v).map(|m| (map_bits(m), v))
        };
        let old_into = into(
            &|v| old::read_frame_u16_into(file, info, order, range, v),
            Vec::new(),
        );
        let old_general_into = into(
            &|v| old::read_plane_u16_into(file, info, order, range, 0, v),
            Vec::new(),
        );
        for fill in [Vec::new(), dirty(0xA5A5u16, 7)] {
            same_or_fixed(
                &format!("{l} read_frame_u16_into"),
                &old_into,
                &old_general_into,
                &into(
                    &|v| super::read_frame_u16_into(file, info, order, range, v),
                    fill,
                ),
            );
        }
        for &p in &planes {
            same(
                &format!("{l} read_plane_u16 {p}"),
                &old::read_plane_u16(file, info, order, range, p),
                &super::read_plane_u16(file, info, order, range, p),
            );
            same(
                &format!("{l} read_plane_u16_into {p}"),
                &into(
                    &|v| old::read_plane_u16_into(file, info, order, range, p, v),
                    Vec::new(),
                ),
                &into(
                    &|v| super::read_plane_u16_into(file, info, order, range, p, v),
                    dirty(0x5A5A, 3),
                ),
            );
        }
        same(
            &format!("{l} read_planes_u16"),
            &old::read_planes_u16(file, info, order, range),
            &super::read_planes_u16(file, info, order, range),
        );
        let mut reused = vec![dirty(1u16, 5); 2];
        same(
            &format!("{l} read_planes_u16_into"),
            &old::read_planes_u16(file, info, order, range),
            &super::read_planes_u16_into(file, info, order, range, &mut reused).map(|()| reused),
        );
    }

    let old_u8 = old::read_frame_u8(file, info, order).map(|c| c.into_owned());
    same(
        &format!("{label} read_frame_u8"),
        &old_u8,
        &super::read_frame_u8(file, info, order).map(|c| c.into_owned()),
    );
    let mut v = Vec::new();
    let old_u8_into = old::read_frame_u8_into(file, info, order, &mut v).map(|()| v);
    let mut v = dirty(0xA5u8, 11);
    same_or_fixed(
        &format!("{label} read_frame_u8_into"),
        &old_u8_into,
        &old_u8,
        &super::read_frame_u8_into(file, info, order, &mut v).map(|()| v),
    );
    for want in [order, other] {
        let mut v = Vec::new();
        let old_bytes = old::read_frame_bytes_into(file, info, order, want, &mut v).map(|()| v);
        let general = old_u8.as_ref().map(|v| {
            if info.bits_per_sample == 4 || want == order {
                v.clone()
            } else {
                swap_all(v.clone(), info.sample_bytes().unwrap_or(1))
            }
        });
        let general = general.map_err(|e| crate::Error::new(e.kind(), e.to_string()));
        let mut v = dirty(0x11u8, 3);
        let new = super::read_frame_bytes_into(file, info, order, want, &mut v).map(|()| v);
        same_or_fixed(
            &format!("{label} read_frame_bytes_into {want:?}"),
            &old_bytes,
            &general,
            &new,
        );
        // And the Frame API's own whole-frame read.
        let frame = Frame::new(file, info, order);
        let new = frame.stored_frame_len().and_then(|n| {
            let mut out = vec![0xC3u8; n];
            frame
                .stored_frame_into(want, &mut out, &mut DecodeScratch::new())
                .map(|()| out)
        });
        same_or_fixed(
            &format!("{label} stored_frame_into {want:?}"),
            &old_bytes,
            &general,
            &new,
        );
    }
    for &p in &planes {
        same(
            &format!("{label} read_plane_u8 {p}"),
            &old::read_plane_u8(file, info, order, p),
            &super::read_plane_u8(file, info, order, p),
        );
        let mut v = dirty(9u8, 2);
        same(
            &format!("{label} read_plane_u8_into {p}"),
            &old::read_plane_u8(file, info, order, p),
            &super::read_plane_u8_into(file, info, order, p, &mut v).map(|()| v),
        );
    }
    same(
        &format!("{label} read_planes_u8"),
        &old::read_planes_u8(file, info, order),
        &super::read_planes_u8(file, info, order),
    );
    let mut reused = vec![dirty(3u8, 1); 5];
    same(
        &format!("{label} read_planes_u8_into"),
        &old::read_planes_u8(file, info, order),
        &super::read_planes_u8_into(file, info, order, &mut reused).map(|()| reused),
    );

    let old_f32_general = bits_r(old::read_plane_f32(file, info, order, 0));
    same_or_fixed(
        &format!("{label} read_frame_f32"),
        &bits_r(old::read_frame_f32(file, info, order).map(|c| c.into_owned())),
        &old_f32_general,
        &bits_r(super::read_frame_f32(file, info, order).map(|c| c.into_owned())),
    );
    let mut v = Vec::new();
    let old_f32_into = bits_r(old::read_frame_f32_into(file, info, order, &mut v).map(|()| v));
    let mut v = dirty(f32::NAN, 4);
    same_or_fixed(
        &format!("{label} read_frame_f32_into"),
        &old_f32_into,
        &old_f32_general,
        &bits_r(super::read_frame_f32_into(file, info, order, &mut v).map(|()| v)),
    );
    for &p in &planes {
        same(
            &format!("{label} read_plane_f32 {p}"),
            &bits_r(old::read_plane_f32(file, info, order, p)),
            &bits_r(super::read_plane_f32(file, info, order, p)),
        );
        let mut v = dirty(-7.5f32, 9);
        same(
            &format!("{label} read_plane_f32_into {p}"),
            &bits_r(old::read_plane_f32(file, info, order, p)),
            &bits_r(super::read_plane_f32_into(file, info, order, p, &mut v).map(|()| v)),
        );
    }
    let planes_bits =
        |r: Result<Vec<Vec<f32>>>| r.map(|v| v.iter().map(|p| bits(p)).collect::<Vec<_>>());
    same(
        &format!("{label} read_planes_f32"),
        &planes_bits(old::read_planes_f32(file, info, order)),
        &planes_bits(super::read_planes_f32(file, info, order)),
    );
    let mut reused = vec![dirty(1.0f32, 3); 4];
    same(
        &format!("{label} read_planes_f32_into"),
        &planes_bits(old::read_planes_f32(file, info, order)),
        &planes_bits(super::read_planes_f32_into(file, info, order, &mut reused).map(|()| reused)),
    );
    same(
        &format!("{label} frame_float_minmax"),
        &old::frame_float_minmax(file, info, order).map(range_bits),
        &super::frame_float_minmax(file, info, order).map(range_bits),
    );
    same(
        &format!("{label} read_planes_rgb_u8"),
        &old::read_planes_rgb_u8(file, info, order),
        &super::read_planes_rgb_u8(file, info, order),
    );
    same(
        &format!("{label} read_planes_rgb_u16"),
        &old::read_planes_rgb_u16(file, info, order),
        &super::read_planes_rgb_u16(file, info, order),
    );
    for p in [0usize, 2, 5] {
        same(
            &format!("{label} read_plane_rgb_u8 {p}"),
            &old::read_plane_rgb_u8(file, info, order, p),
            &super::read_plane_rgb_u8(file, info, order, p),
        );
        same(
            &format!("{label} read_plane_rgb_u16 {p}"),
            &old::read_plane_rgb_u16(file, info, order, p),
            &super::read_plane_rgb_u16(file, info, order, p),
        );
    }

    // --- the Frame API against the old readers ----------------------------
    compare_frame_api(&Frame::new(file, info, order), file, info, order, label);
}

/// The [`Frame`] reads, every unit and plane, against the old readers of the
/// same thing.
fn compare_frame_api(
    frame: &Frame<'_>,
    file: &[u8],
    info: &FrameInfo,
    order: ByteOrder,
    label: &str,
) {
    let spp = spp_of(info);
    let mut scratch = DecodeScratch::new();
    // One scratch for every read of the frame, dirtied by each: what a warm
    // scratch holds must never show through.
    let mut warm = DecodeScratch::new();
    warm.pieces = vec![vec![0xEE; 3], vec![0x77; 100]];
    warm.native = vec![0x99; 17];

    for range in [None, Some((-1000.0f32, 5000.0f32))] {
        let units = Display { range };
        let l = format!("{label} Display {range:?}");
        for p in 0..=spp {
            let mut v = Vec::new();
            let old = old::read_plane_u16_into(file, info, order, range, p, &mut v)
                .map(|m| (map_bits(m), v));
            let mut out = dirty(0xA5A5u16, 5);
            let new = frame.read(units, p, &mut out, &mut warm).map(|(px, maps)| {
                if px.is_source() {
                    assert_borrowed_from(file, &px, &l);
                }
                (map_bits(maps.plane(0)), px.to_vec())
            });
            if p == spp {
                assert_eq!(
                    new.map(drop).unwrap_err().kind(),
                    ErrorKind::InvalidInput,
                    "{l} {p}"
                );
                continue;
            }
            same(&format!("{l} read {p}"), &old, &new);
            assert_eq!(
                frame.borrows(units, p),
                new.is_ok() && {
                    let mut o = Vec::new();
                    frame
                        .read(units, p, &mut o, &mut scratch)
                        .map(|(px, _)| px.is_source())
                        .unwrap_or(false)
                },
                "{l} borrows {p}"
            );
        }
        // Every plane at once, into exact dirty slices.
        let old_all = old::read_planes_u16(file, info, order, range).map(|planes| {
            let maps: Vec<_> = (0..planes.len())
                .map(|p| {
                    let mut v = Vec::new();
                    map_bits(old::read_plane_u16_into(file, info, order, range, p, &mut v).unwrap())
                })
                .collect();
            (maps, planes)
        });
        let new_all = frame.plane_len(units).and_then(|n| {
            let mut planes = vec![vec![0xA5A5u16; n]; spp];
            let mut slices: Vec<&mut [u16]> = planes.iter_mut().map(|v| &mut v[..]).collect();
            let maps = frame.read_into(units, 0, &mut slices, &mut warm)?;
            let maps = match maps {
                DisplayMaps::One(m) => vec![map_bits(m); spp],
                DisplayMaps::PerPlane(v) => {
                    assert_eq!(v.len(), spp);
                    v.into_iter().map(map_bits).collect()
                }
            };
            Ok((maps, planes))
        });
        same(&format!("{l} read_into all"), &old_all, &new_all);
        if spp >= 3 {
            let old_two = (|| -> Result<Vec<Vec<u16>>> {
                Ok(vec![
                    old::read_plane_u16(file, info, order, range, 1)?,
                    old::read_plane_u16(file, info, order, range, 2)?,
                ])
            })();
            let new_two = frame.plane_len(units).and_then(|n| {
                let (mut a, mut b) = (vec![1u16; n], vec![2u16; n]);
                frame.read_into(units, 1, &mut [&mut a[..], &mut b[..]], &mut scratch)?;
                Ok(vec![a, b])
            });
            same(&format!("{l} read_into 1..3"), &old_two, &new_two);
        }
    }

    // File units.
    for p in 0..spp {
        let old = bits_r(old::read_plane_f32(file, info, order, p));
        let mut out = dirty(f32::NAN, 2);
        let new = frame
            .read(File, p, &mut out, &mut warm)
            .map(|(px, ())| bits(&px));
        same(&format!("{label} File read {p}"), &old, &new);
    }
    let old_all = old::read_planes_f32(file, info, order)
        .map(|v| v.iter().map(|p| bits(p)).collect::<Vec<_>>());
    let new_all = frame.plane_len(File).and_then(|n| {
        let mut planes = vec![vec![-0.0f32; n]; spp];
        let mut slices: Vec<&mut [f32]> = planes.iter_mut().map(|v| &mut v[..]).collect();
        frame.read_into(File, 0, &mut slices, &mut warm)?;
        Ok(planes.iter().map(|p| bits(p)).collect::<Vec<_>>())
    });
    // The old all-planes f32 reader gave a 4-bit frame one plane; so does this.
    same(&format!("{label} File read_into all"), &old_all, &new_all);

    // Stored units, both byte orders, against planes of the old whole frame.
    let other = match order {
        ByteOrder::Little => ByteOrder::Big,
        ByteOrder::Big => ByteOrder::Little,
    };
    for want in [order, other] {
        let mut v = Vec::new();
        let old_frame = old::read_frame_u8(file, info, order)
            .map(|c| c.into_owned())
            .map(|f| {
                if info.bits_per_sample == 4 || want == order {
                    f
                } else {
                    swap_all(f, info.sample_bytes().unwrap_or(1))
                }
            });
        let _ = old::read_frame_bytes_into(file, info, order, want, &mut v);
        for p in 0..spp {
            let old = old_frame.as_ref().map_err(|e| e.kind()).map(|f| {
                if info.bits_per_sample == 4 {
                    f.clone()
                } else {
                    stored_plane(f, info, p).expect("a decoded frame holds its planes")
                }
            });
            let mut out = dirty(0x3Cu8, 1);
            let new = frame
                .read(Stored(want), p, &mut out, &mut warm)
                .map(|(px, ())| {
                    if px.is_source() {
                        assert_borrowed_from(file, &px, label);
                    }
                    px.to_vec()
                })
                .map_err(|e| e.kind());
            assert!(
                old == new,
                "{label} Stored({want:?}) read {p}: old {:?} / new {:?}",
                short(&old),
                short(&new)
            );
        }
        let new_all = frame.plane_len(Stored(want)).and_then(|n| {
            let mut planes = vec![vec![0xF0u8; n]; spp];
            let mut slices: Vec<&mut [u8]> = planes.iter_mut().map(|v| &mut v[..]).collect();
            frame.read_into(Stored(want), 0, &mut slices, &mut warm)?;
            Ok(planes)
        });
        if let (Ok(f), Ok(new_all)) = (&old_frame, &new_all) {
            if info.bits_per_sample != 4 {
                for (p, plane) in new_all.iter().enumerate() {
                    assert_eq!(
                        Some(plane),
                        stored_plane(f, info, p).as_ref(),
                        "{label} Stored({want:?}) read_into plane {p}"
                    );
                }
            }
        } else if info.bits_per_sample != 4 || spp == 1 {
            assert_eq!(
                old_frame.is_ok(),
                new_all.is_ok(),
                "{label} Stored({want:?}) read_into all: old {:?} / new {:?}",
                old_frame.as_ref().map(drop).map_err(|e| e.kind()),
                new_all.as_ref().map(drop).map_err(|e| e.kind())
            );
        }
    }

    // CMYK conversions.
    let old8 = old::read_planes_rgb_u8(file, info, order);
    let new8 = frame.plane_len(CmykRgb8).and_then(|n| {
        let mut planes = vec![vec![0x42u8; n]; 3];
        let mut slices: Vec<&mut [u8]> = planes.iter_mut().map(|v| &mut v[..]).collect();
        frame.read_into(CmykRgb8, 0, &mut slices, &mut warm)?;
        Ok(planes)
    });
    same(&format!("{label} CmykRgb8"), &old8, &new8);
    let old16 = old::read_planes_rgb_u16(file, info, order);
    let new16 = frame.plane_len(CmykRgb16).and_then(|n| {
        let mut planes = vec![vec![0x4242u16; n]; 3];
        let mut slices: Vec<&mut [u16]> = planes.iter_mut().map(|v| &mut v[..]).collect();
        frame.read_into(CmykRgb16, 0, &mut slices, &mut warm)?;
        Ok(planes)
    });
    same(&format!("{label} CmykRgb16"), &old16, &new16);
    if let Ok(old16) = &old16 {
        let mut out = Vec::new();
        let (px, ()) = frame.read(CmykRgb16, 2, &mut out, &mut scratch).unwrap();
        assert_eq!(&*px, &old16[2][..], "{label} CmykRgb16 blue alone");
    }

    same(
        &format!("{label} value_range"),
        &old::frame_float_minmax(file, info, order).map(range_bits),
        &frame.value_range(&mut warm).map(range_bits),
    );

    // A destination of the wrong length is refused, with nothing decoded.
    if let Ok(n) = frame.plane_len(File) {
        let mut short = vec![0f32; n + 1];
        let e = frame
            .read_into(File, 0, &mut [&mut short[..]], &mut scratch)
            .unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidInput, "{label} wrong length");
    }
}

fn assert_borrowed_from<T>(file: &[u8], px: &Pixels<'_, T>, label: &str) {
    let range = file.as_ptr_range();
    let at = px.as_ptr() as *const u8;
    assert!(
        range.start <= at && at <= range.end,
        "{label}: a borrowed read must point into the source"
    );
}

/// [`compare_frame`] for the frame and every crop the fuzz target makes of it.
fn compare_with_crops(file: &[u8], info: &FrameInfo, order: ByteOrder, label: &str) {
    compare_frame(file, info, order, label);
    let frame = Frame::new(file, info, order);
    let max = u32::MAX;
    #[allow(clippy::reversed_empty_ranges)]
    let rows: [std::ops::Range<u32>; 6] = [0..1, 0..max, 3..9, max - 1..max, 7..2, max..0];
    for r in rows.clone() {
        let old = info.crop_rows(r.clone());
        let new = frame.crop(Crop::Rows(r.clone()));
        match (old, new) {
            (Ok(band), Ok((cropped, cov))) => {
                assert_eq!(cov.rows, band.rows, "{label} rows {r:?}");
                assert_eq!(cov.cols, 0..info.width, "{label} rows {r:?}");
                compare_frame(file, &band.frame, order, &format!("{label} rows {r:?}"));
                compare_frame_api(
                    &cropped,
                    file,
                    &band.frame,
                    order,
                    &format!("{label} Rows {r:?}"),
                );
            }
            (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind(), "{label} rows {r:?}"),
            (a, b) => panic!("{label} rows {r:?}: {:?} / {:?}", a.is_ok(), b.is_ok()),
        }
    }
    #[allow(clippy::reversed_empty_ranges)]
    let regions: [(std::ops::Range<u32>, std::ops::Range<u32>); 4] =
        [(0..1, 0..1), (0..max, 0..max), (2..9, 3..7), (9..2, 7..3)];
    for (c, r) in regions.clone() {
        let old = info.crop(c.clone(), r.clone());
        let new = frame.crop(Crop::Region {
            cols: c.clone(),
            rows: r.clone(),
        });
        match (old, new) {
            (Ok(region), Ok((cropped, cov))) => {
                assert_eq!(
                    (cov.cols, cov.rows),
                    (region.cols, region.rows),
                    "{label} region"
                );
                compare_frame_api(
                    &cropped,
                    file,
                    &region.frame,
                    order,
                    &format!("{label} Region {c:?} {r:?}"),
                );
            }
            (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind(), "{label} region {c:?} {r:?}"),
            (a, b) => panic!(
                "{label} region {c:?} {r:?}: {:?} / {:?}",
                a.is_ok(),
                b.is_ok()
            ),
        }
    }
    #[allow(clippy::reversed_empty_ranges)]
    let sampled: [std::ops::Range<u32>; 4] = [0..1, 0..max, 3..9, 6..1];
    for r in sampled.clone() {
        for step in [0u32, 1, 2, 7, max] {
            let old = info.crop_rows_step(r.clone(), step);
            let new = frame.crop(Crop::Sampled {
                rows: r.clone(),
                step,
            });
            match (old, new) {
                (Ok(band), Ok((cropped, cov))) => {
                    assert_eq!(
                        (cov.first_row, cov.rows_per_piece, cov.pieces, cov.step),
                        (band.first_row, band.rows_per_piece, band.pieces, band.step),
                        "{label} sampled {r:?} {step}"
                    );
                    for probe in 0..info.height.min(200) {
                        assert_eq!(
                            cov.decoded_row_of(probe),
                            band.decoded_row_of(probe),
                            "{label} sampled {r:?} {step}: row {probe}"
                        );
                    }
                    for probe in [u32::MAX, band.first_row.wrapping_add(1)] {
                        assert_eq!(cov.decoded_row_of(probe), band.decoded_row_of(probe));
                    }
                    compare_frame_api(
                        &cropped,
                        file,
                        &band.frame,
                        order,
                        &format!("{label} Sampled {r:?} {step}"),
                    );
                }
                (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind(), "{label} sampled"),
                (a, b) => panic!(
                    "{label} sampled {r:?} {step}: {:?} / {:?}",
                    a.is_ok(),
                    b.is_ok()
                ),
            }
        }
    }
}

// --------------------------------------------------------------- fixtures

fn tifs_in(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "tif"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// Regressions the old decoder is too slow on to sweep: a tile claiming
/// 393232 rows over a 36-row image, which it decompressed, zero-filled and
/// un-predicted in full for every read.
const SLOW_FOR_THE_OLD_DECODER: &[&str] = &["slow-tall-tile.tif"];

/// Every checked-in TIFF in `dir`: each frame (the first two) read every way
/// and through every crop, and the whole-stack readers. Parallel decoding is
/// tried both ways on a frame big enough for it to matter.
fn compare_fixtures(dir: &str) -> usize {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut seen = 0;
    for path in tifs_in(&root.join(dir)) {
        let bytes = std::fs::read(&path).unwrap();
        let Ok(stack) = TiffStack::from_bytes(bytes) else {
            continue;
        };
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if SLOW_FOR_THE_OLD_DECODER.contains(&name.as_str()) {
            // The old decoder takes seconds a read here in a debug build, so
            // only the readers that reach its tiles, once each.
            let (b, f, o) = (&stack.data[..], &stack.frames[0], stack.byte_order);
            same(
                &format!("{name} read_planes_u8"),
                &old::read_planes_u8(b, f, o),
                &super::read_planes_u8(b, f, o),
            );
            same(
                &format!("{name} read_planes_u16"),
                &old::read_planes_u16(b, f, o, None),
                &super::read_planes_u16(b, f, o, None),
            );
            seen += 1;
            continue;
        }
        for (i, info) in stack.frames.iter().enumerate().take(2) {
            let big = info.pixel_count().is_ok_and(|n| n >= PARALLEL_MIN_PIXELS);
            for parallel in [false, true] {
                if parallel && !big {
                    continue;
                }
                let _p = parallel_decode_scope(parallel);
                compare_with_crops(
                    &stack.data,
                    info,
                    stack.byte_order,
                    &format!("{name} frame {i} parallel {parallel}"),
                );
            }
        }
        same(
            &format!("{name} preload u16"),
            &old::preload_frames_u16(&stack, None),
            &super::preload_frames_u16(&stack, None),
        );
        same(
            &format!("{name} preload u8"),
            &old::preload_frames_u8(&stack),
            &super::preload_frames_u8(&stack),
        );
        let f = |r: Result<Vec<Vec<f32>>>| r.map(|v| v.iter().map(|p| bits(p)).collect::<Vec<_>>());
        same(
            &format!("{name} preload f32"),
            &f(old::preload_frames_f32(&stack)),
            &f(super::preload_frames_f32(&stack)),
        );
        seen += 1;
    }
    seen
}

#[test]
fn every_fixture_reads_as_the_old_decoder_read_it() {
    assert!(compare_fixtures("tests/fixtures") >= 40);
}

#[test]
fn every_fuzz_regression_reads_as_the_old_decoder_read_it() {
    assert!(compare_fixtures("tests/fuzz-regressions") >= 4);
}

#[test]
fn every_fuzz_seed_reads_as_the_old_decoder_read_it() {
    assert!(compare_fixtures("fuzz/seeds") >= 25);
}

// ------------------------------------------------------- synthetic frames

/// How a synthetic frame's pieces are laid out in its file.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lay {
    /// One strip.
    One,
    /// Strips of this many rows, back to back.
    Strips(u32),
    /// Strips with a few bytes of something else between them.
    Gapped(u32),
    /// Strips whose byte counts claim more than their rows: padding.
    Padded(u32),
    /// One strip holding the frame, under a RowsPerStrip that undersells it.
    Whole(u32),
    /// Tiles of this size.
    Tiled(u32, u32),
    /// One strip, an odd number of bytes into the file.
    Misaligned,
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    w: u32,
    h: u32,
    bits: u16,
    format: SampleFormat,
    spp: u16,
    planar: bool,
    order: ByteOrder,
    lay: Lay,
    predictor: u16,
    compression: Compression,
    photometric: u16,
}

/// Deterministic samples with structure (so codecs compress and predictors
/// matter) and the awkward values: extremes, negatives, NaN, infinities and
/// both zeros.
fn sample_bytes_of(spec: &Spec, i: usize) -> Vec<u8> {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    let smooth = (i % 97) as u64 * 3 + (i / 97) as u64;
    let v = if i.is_multiple_of(5) {
        x
    } else {
        smooth.wrapping_mul(0x0101_0101_0101)
    };
    let le: Vec<u8> = match (spec.bits, spec.format) {
        (32, SampleFormat::Float) => {
            let f = match i % 23 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => -0.0,
                3 => 0.0,
                4 => f32::NEG_INFINITY,
                _ => ((v % 20000) as f32 - 7000.0) * 0.37,
            };
            f.to_le_bytes().to_vec()
        }
        (64, SampleFormat::Float) => {
            let f = match i % 19 {
                0 => f64::NAN,
                1 => -0.0,
                2 => 1e300,
                _ => ((v % 50000) as f64 - 9000.0) * 1.25,
            };
            f.to_le_bytes().to_vec()
        }
        (bits, _) => v.to_le_bytes()[..bits as usize / 8].to_vec(),
    };
    match spec.order {
        ByteOrder::Little => le,
        ByteOrder::Big => le.into_iter().rev().collect(),
    }
}

/// Horizontal differencing, per row of `row_samples` samples `stride` apart,
/// in the file's byte order (the inverse of the decoder's undo).
fn difference(data: &mut [u8], row_samples: usize, stride: usize, sb: usize, order: ByteOrder) {
    let read = |b: &[u8]| -> u64 {
        let mut a = [0u8; 8];
        match order {
            ByteOrder::Little => a[..sb].copy_from_slice(b),
            ByteOrder::Big => {
                for (k, &v) in b.iter().rev().enumerate() {
                    a[k] = v;
                }
            }
        }
        u64::from_le_bytes(a)
    };
    let write = |b: &mut [u8], v: u64| {
        let a = v.to_le_bytes();
        match order {
            ByteOrder::Little => b.copy_from_slice(&a[..sb]),
            ByteOrder::Big => {
                for (k, o) in b.iter_mut().rev().enumerate() {
                    *o = a[k];
                }
            }
        }
    };
    for row in data.chunks_exact_mut(row_samples * sb) {
        for i in (stride..row_samples).rev() {
            let cur = read(&row[i * sb..(i + 1) * sb]);
            let prev = read(&row[(i - stride) * sb..(i - stride + 1) * sb]);
            write(&mut row[i * sb..(i + 1) * sb], cur.wrapping_sub(prev));
        }
    }
}

/// TechNote 3 floating-point differencing: each row's values split into
/// byte-significance planes, most significant first, then differenced.
fn float_difference(
    data: &mut [u8],
    row_samples: usize,
    stride: usize,
    sb: usize,
    order: ByteOrder,
) {
    let row_bytes = row_samples * sb;
    let mut planes = vec![0u8; row_bytes];
    for row in data.chunks_exact_mut(row_bytes) {
        for v in 0..row_samples {
            for p in 0..sb {
                let byte = match order {
                    ByteOrder::Little => row[v * sb + (sb - 1 - p)],
                    ByteOrder::Big => row[v * sb + p],
                };
                planes[p * row_samples + v] = byte;
            }
        }
        for i in (stride..row_bytes).rev() {
            planes[i] = planes[i].wrapping_sub(planes[i - stride]);
        }
        row.copy_from_slice(&planes);
    }
}

fn compress(data: &[u8], compression: Compression, row_bytes: usize) -> Vec<u8> {
    use std::io::Write;
    match compression {
        Compression::None => data.to_vec(),
        Compression::Lzw => {
            let mut out = Vec::new();
            weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                .into_stream(&mut out)
                .encode_all(data)
                .status
                .unwrap();
            out
        }
        Compression::Deflate => {
            let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
            e.write_all(data).unwrap();
            e.finish().unwrap()
        }
        Compression::PackBits => {
            // Per row: a literal run, then the rest as repeats where they are.
            let mut out = Vec::new();
            for row in data.chunks(row_bytes.max(1)) {
                let mut i = 0;
                while i < row.len() {
                    let mut run = 1;
                    while run < 128 && i + run < row.len() && row[i + run] == row[i] {
                        run += 1;
                    }
                    if run >= 2 {
                        out.push((1i32 - run as i32) as i8 as u8);
                        out.push(row[i]);
                        i += run;
                    } else {
                        let n = (row.len() - i).min(5);
                        out.push((n - 1) as u8);
                        out.extend_from_slice(&row[i..i + n]);
                        i += n;
                    }
                }
            }
            out
        }
        #[cfg(feature = "codec-zstd")]
        Compression::Zstd => zstd::stream::encode_all(data, 3).unwrap(),
        other => panic!("no encoder for {other:?}"),
    }
}

/// A frame laid out as `spec` says, and the file holding it.
fn build(spec: &Spec) -> (Vec<u8>, FrameInfo) {
    let (w, h) = (spec.w as usize, spec.h as usize);
    let spp = spec.spp as usize;
    let four = spec.bits == 4;
    let sb = if four { 1 } else { spec.bits as usize / 8 };
    let per_px = if spec.planar { 1 } else { spp };
    let n_planes = if spec.planar { spp } else { 1 };
    // The samples of each piece-plane, in file order: rows of `w * per_px`.
    let row_bytes = if four { w.div_ceil(2) } else { w * per_px * sb };
    let plane_data: Vec<Vec<u8>> = (0..n_planes)
        .map(|p| {
            let mut v = Vec::with_capacity(row_bytes * h);
            if four {
                for i in 0..row_bytes * h {
                    v.push(((i * 37 + i / 7) % 256) as u8);
                }
            } else {
                for i in 0..w * h * per_px {
                    v.extend(sample_bytes_of(spec, p * w * h * per_px + i));
                }
            }
            v
        })
        .collect();
    let predict = |piece: &mut [u8], width: usize| {
        let stride = per_px;
        match spec.predictor {
            2 => difference(piece, width * per_px, stride, sb, spec.order),
            3 => float_difference(piece, width * per_px, stride, sb, spec.order),
            _ => {}
        }
    };

    let mut file = vec![0u8; 16];
    let (mut offsets, mut counts) = (Vec::new(), Vec::new());
    let mut push = |file: &mut Vec<u8>, bytes: &[u8], extra: usize, gap: usize| {
        file.extend(std::iter::repeat_n(0xCDu8, gap));
        offsets.push(file.len() as u64);
        counts.push((bytes.len() + extra) as u64);
        file.extend_from_slice(bytes);
        file.extend(std::iter::repeat_n(0xEEu8, extra));
    };
    let rows_per_strip;
    let mut tile_size = None;
    match spec.lay {
        Lay::Tiled(tw, th) => {
            tile_size = Some((tw, th));
            rows_per_strip = th;
            let (tw, th) = (tw as usize, th as usize);
            let tile_row = tw * per_px * sb;
            for data in &plane_data {
                for ty in 0..h.div_ceil(th) {
                    for tx in 0..w.div_ceil(tw) {
                        // Edge tiles are stored full size, the padding holding
                        // something that must never show.
                        let mut tile = vec![0x6Bu8; tile_row * th];
                        for r in 0..th {
                            let y = ty * th + r;
                            if y >= h {
                                continue;
                            }
                            let x0 = tx * tw;
                            let keep = tw.min(w - x0) * per_px * sb;
                            let src = &data[y * row_bytes + x0 * per_px * sb..][..keep];
                            tile[r * tile_row..r * tile_row + keep].copy_from_slice(src);
                        }
                        predict(&mut tile, tw);
                        let packed = compress(&tile, spec.compression, tile_row);
                        // Align so borrows can be tested; tiles are never borrowed.
                        while !file.len().is_multiple_of(8) {
                            file.push(0);
                        }
                        push(&mut file, &packed, 0, 0);
                    }
                }
            }
        }
        lay => {
            let (rps, gap, extra, whole) = match lay {
                Lay::One | Lay::Misaligned => (h, 0, 0, false),
                Lay::Strips(r) => (r as usize, 0, 0, false),
                Lay::Gapped(r) => (r as usize, 5, 0, false),
                Lay::Padded(r) => (r as usize, 0, 7, false),
                Lay::Whole(r) => (r as usize, 0, 0, true),
                Lay::Tiled(..) => unreachable!(),
            };
            rows_per_strip = rps as u32;
            if lay == Lay::Misaligned {
                file.push(0);
            }
            if whole {
                // Every plane in the one strip.
                let mut all: Vec<u8> = plane_data.concat();
                for part in all.chunks_mut(row_bytes * h) {
                    predict(part, w);
                }
                push(
                    &mut file,
                    &compress(&all, spec.compression, row_bytes),
                    0,
                    0,
                );
            } else {
                for data in &plane_data {
                    for (k, rows) in data.chunks(rps * row_bytes).enumerate() {
                        let mut strip = rows.to_vec();
                        predict(&mut strip, w);
                        let packed = compress(&strip, spec.compression, row_bytes);
                        let gap = if k > 0 { gap } else { 0 };
                        if lay != Lay::Misaligned {
                            while !file.len().is_multiple_of(8) && gap == 0 && extra == 0 {
                                file.push(0);
                            }
                        }
                        push(&mut file, &packed, extra, gap);
                    }
                }
            }
        }
    }
    let info = FrameInfo {
        width: spec.w,
        height: spec.h,
        bits_per_sample: spec.bits,
        samples_per_pixel: spec.spp,
        sample_format: spec.format,
        compression: spec.compression,
        predictor: spec.predictor,
        photometric: spec.photometric,
        planar_config: if spec.planar { 2 } else { 1 },
        tile_size,
        ink_set: 1,
        strip_offsets: offsets.into(),
        strip_byte_counts: counts.into(),
        rows_per_strip,
    };
    (file, info)
}

fn codecs() -> Vec<Compression> {
    let mut c = vec![
        Compression::None,
        Compression::Lzw,
        Compression::Deflate,
        Compression::PackBits,
    ];
    if cfg!(feature = "codec-zstd") {
        c.push(Compression::Zstd);
    }
    c
}

/// The matrix: every depth, sign and byte order, at 1, 3 and 4 samples
/// chunky and planar, each through a rotation of layouts, predictors and
/// codecs so that every layout meets every codec somewhere.
fn matrix() -> Vec<Spec> {
    use SampleFormat::*;
    let formats = [
        (8, UnsignedInt),
        (8, SignedInt),
        (16, UnsignedInt),
        (16, SignedInt),
        (16, Float),
        (32, UnsignedInt),
        (32, SignedInt),
        (32, Float),
        (64, UnsignedInt),
        (64, SignedInt),
        (64, Float),
    ];
    let lays = [
        Lay::One,
        Lay::Strips(3),
        Lay::Gapped(4),
        Lay::Padded(2),
        Lay::Whole(3),
        Lay::Tiled(16, 16),
        Lay::Tiled(32, 16),
        Lay::Misaligned,
    ];
    let codecs = codecs();
    let mut specs = Vec::new();
    let mut turn = 0usize;
    for (bits, format) in formats {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            for (spp, planar) in [(1u16, false), (3, false), (3, true), (4, false), (4, true)] {
                let photometric = match spp {
                    1 => 1,
                    3 => 2,
                    // CMYK where the depth allows it, so the conversions run.
                    _ if format == UnsignedInt && bits <= 16 => 5,
                    _ => 2,
                };
                let predictors: &[u16] = match (bits, format) {
                    (32 | 64, Float) => &[1, 3],
                    (16, Float) => &[1],
                    _ => &[1, 2],
                };
                for (li, &lay) in lays.iter().enumerate() {
                    for &predictor in predictors {
                        turn += 1;
                        let compression = codecs[(turn + li) % codecs.len()];
                        let compression = match lay {
                            // A gap or padding only matters uncompressed half
                            // the time; a whole-frame strip is uncompressed.
                            Lay::Whole(_) | Lay::Misaligned => Compression::None,
                            _ => compression,
                        };
                        specs.push(Spec {
                            w: 37,
                            h: 21,
                            bits,
                            format,
                            spp,
                            planar,
                            order,
                            lay,
                            predictor,
                            compression,
                            photometric,
                        });
                    }
                }
            }
        }
    }
    // 4-bit palette frames, an odd width so a row ends mid-byte.
    for order in [ByteOrder::Little, ByteOrder::Big] {
        for lay in [Lay::One, Lay::Strips(4), Lay::Gapped(5)] {
            for compression in codecs.iter().copied() {
                specs.push(Spec {
                    w: 33,
                    h: 17,
                    bits: 4,
                    format: UnsignedInt,
                    spp: 1,
                    planar: false,
                    order,
                    lay,
                    predictor: 1,
                    compression,
                    photometric: 3,
                });
            }
        }
    }
    specs
}

/// The part of the matrix of one depth, format and byte order, every read;
/// then every seventh of those through every crop as well. Parallel decoding
/// stays off: these frames are below the size that would split them anyway.
fn compare_matrix(bits: u16, format: SampleFormat, order: ByteOrder) {
    let specs: Vec<Spec> = matrix()
        .into_iter()
        .filter(|s| s.bits == bits && s.format == format && s.order == order)
        .collect();
    assert!(!specs.is_empty());
    for (i, spec) in specs.iter().enumerate() {
        let (file, info) = build(spec);
        // Every frame of the matrix is well formed: one the old decoder could
        // not read would compare nothing but two errors.
        assert!(
            old::read_frame_u8(&file, &info, spec.order).is_ok(),
            "{spec:?} does not decode"
        );
        if i % 7 == 0 {
            compare_with_crops(&file, &info, spec.order, &format!("{spec:?}"));
        } else {
            compare_frame(&file, &info, spec.order, &format!("{spec:?}"));
        }
    }
}

/// One test per depth, format and byte order, so the harness runs them side
/// by side.
macro_rules! matrix_tests {
    ($($name:ident: $bits:expr, $format:ident, $order:ident;)*) => {$(
        #[test]
        fn $name() {
            compare_matrix($bits, SampleFormat::$format, ByteOrder::$order);
        }
    )*};
}

matrix_tests! {
    synthetic_u4_le_reads_as_before: 4, UnsignedInt, Little;
    synthetic_u4_be_reads_as_before: 4, UnsignedInt, Big;
    synthetic_u8_le_reads_as_before: 8, UnsignedInt, Little;
    synthetic_u8_be_reads_as_before: 8, UnsignedInt, Big;
    synthetic_i8_le_reads_as_before: 8, SignedInt, Little;
    synthetic_i8_be_reads_as_before: 8, SignedInt, Big;
    synthetic_u16_le_reads_as_before: 16, UnsignedInt, Little;
    synthetic_u16_be_reads_as_before: 16, UnsignedInt, Big;
    synthetic_i16_le_reads_as_before: 16, SignedInt, Little;
    synthetic_i16_be_reads_as_before: 16, SignedInt, Big;
    synthetic_f16_le_reads_as_before: 16, Float, Little;
    synthetic_f16_be_reads_as_before: 16, Float, Big;
    synthetic_u32_le_reads_as_before: 32, UnsignedInt, Little;
    synthetic_u32_be_reads_as_before: 32, UnsignedInt, Big;
    synthetic_i32_le_reads_as_before: 32, SignedInt, Little;
    synthetic_i32_be_reads_as_before: 32, SignedInt, Big;
    synthetic_f32_le_reads_as_before: 32, Float, Little;
    synthetic_f32_be_reads_as_before: 32, Float, Big;
    synthetic_u64_le_reads_as_before: 64, UnsignedInt, Little;
    synthetic_u64_be_reads_as_before: 64, UnsignedInt, Big;
    synthetic_i64_le_reads_as_before: 64, SignedInt, Little;
    synthetic_i64_be_reads_as_before: 64, SignedInt, Big;
    synthetic_f64_le_reads_as_before: 64, Float, Little;
    synthetic_f64_be_reads_as_before: 64, Float, Big;
}

#[test]
fn the_matrix_covers_what_it_claims() {
    let specs = matrix();
    assert!(specs.len() > 1000, "{} specs", specs.len());
    for lay in [
        Lay::Gapped(4),
        Lay::Tiled(16, 16),
        Lay::Whole(3),
        Lay::Misaligned,
    ] {
        assert!(specs
            .iter()
            .any(|s| s.lay == lay && s.order == ByteOrder::Big));
    }
    for c in codecs() {
        assert!(specs
            .iter()
            .any(|s| s.compression == c && matches!(s.lay, Lay::Tiled(..))));
        assert!(specs.iter().any(|s| s.compression == c && s.predictor == 3));
        assert!(specs
            .iter()
            .any(|s| s.compression == c && s.planar && s.predictor == 2));
    }
}

/// Frames big enough to decode on several cores, with parallel decoding off
/// and on: the strip groups, the band tasks and the chunked conversions.
fn compare_large(which: usize) {
    use SampleFormat::*;
    let side = (PARALLEL_MIN_PIXELS as f64).sqrt() as u32 + 1;
    let zstd_or_lzw = if cfg!(feature = "codec-zstd") {
        Compression::Zstd
    } else {
        Compression::Lzw
    };
    let specs = [
        (
            16,
            UnsignedInt,
            1,
            false,
            Lay::Strips(64),
            2,
            zstd_or_lzw,
            1,
        ),
        (
            16,
            SignedInt,
            1,
            false,
            Lay::Strips(40),
            1,
            Compression::Lzw,
            1,
        ),
        (
            32,
            Float,
            1,
            false,
            Lay::Strips(64),
            3,
            Compression::Deflate,
            1,
        ),
        (32, UnsignedInt, 1, false, Lay::One, 1, Compression::None, 1),
        (
            8,
            UnsignedInt,
            3,
            false,
            Lay::Strips(32),
            2,
            Compression::Lzw,
            2,
        ),
        (
            16,
            UnsignedInt,
            3,
            true,
            Lay::Strips(50),
            1,
            Compression::Deflate,
            2,
        ),
        (
            16,
            UnsignedInt,
            1,
            false,
            Lay::Tiled(64, 64),
            2,
            Compression::Deflate,
            1,
        ),
        (
            8,
            UnsignedInt,
            4,
            false,
            Lay::Strips(64),
            1,
            Compression::Lzw,
            5,
        ),
        (
            64,
            Float,
            1,
            false,
            Lay::Strips(100),
            1,
            Compression::PackBits,
            1,
        ),
    ];
    for (depth, format, spp, planar, lay, predictor, compression, photometric) in
        specs.into_iter().skip(which).take(1)
    {
        let spec = Spec {
            w: side,
            h: side,
            bits: depth,
            format,
            spp,
            planar,
            order: ByteOrder::Little,
            lay,
            predictor,
            compression,
            photometric,
        };
        let (file, info) = build(&spec);
        for parallel in [false, true] {
            let _p = parallel_decode_scope(parallel);
            let frame = Frame::new(&file, &info, spec.order);
            let label = format!("{spec:?} parallel {parallel}");
            // The heavy reads only: each is a full decode of a megapixel.
            for range in [None, Some((0.0, 40000.0))] {
                let old = old::read_planes_u16(&file, &info, spec.order, range);
                let new = frame.plane_len(Display { range }).and_then(|n| {
                    let mut planes = vec![vec![0xA5A5u16; n]; spp as usize];
                    let mut s: Vec<&mut [u16]> = planes.iter_mut().map(|v| &mut v[..]).collect();
                    frame.read_into(Display { range }, 0, &mut s, &mut DecodeScratch::new())?;
                    Ok(planes)
                });
                same(&format!("{label} Display {range:?}"), &old, &new);
            }
            let old = old::read_planes_f32(&file, &info, spec.order)
                .map(|v| v.iter().map(|p| bits(p)).collect::<Vec<_>>());
            let new = frame.plane_len(File).and_then(|n| {
                let mut planes = vec![vec![f32::NAN; n]; spp as usize];
                let mut s: Vec<&mut [f32]> = planes.iter_mut().map(|v| &mut v[..]).collect();
                frame.read_into(File, 0, &mut s, &mut DecodeScratch::new())?;
                Ok(planes.iter().map(|p| bits(p)).collect::<Vec<_>>())
            });
            same(&format!("{label} File"), &old, &new);
            let mut v = Vec::new();
            let old = old::read_frame_bytes_into(&file, &info, spec.order, ByteOrder::Big, &mut v)
                .map(|()| v);
            let mut v = vec![1u8; 3];
            let new =
                super::read_frame_bytes_into(&file, &info, spec.order, ByteOrder::Big, &mut v)
                    .map(|()| v);
            same(&format!("{label} stored frame"), &old, &new);
            same(
                &format!("{label} value_range"),
                &old::frame_float_minmax(&file, &info, spec.order).map(range_bits),
                &frame.value_range(&mut DecodeScratch::new()).map(range_bits),
            );
            if info.is_cmyk() {
                same(
                    &format!("{label} cmyk"),
                    &old::read_planes_rgb_u8(&file, &info, spec.order),
                    &super::read_planes_rgb_u8(&file, &info, spec.order),
                );
            }
        }
    }
}

macro_rules! large_tests {
    ($($name:ident: $which:expr;)*) => {$(
        #[test]
        fn $name() {
            compare_large($which);
        }
    )*};
}

large_tests! {
    large_u16_zstd_pred_reads_as_before: 0;
    large_i16_lzw_reads_as_before: 1;
    large_f32_deflate_pred3_reads_as_before: 2;
    large_u32_auto_range_reads_as_before: 3;
    large_rgb8_lzw_pred2_reads_as_before: 4;
    large_planar_rgb16_reads_as_before: 5;
    large_tiled_u16_reads_as_before: 6;
    large_cmyk8_reads_as_before: 7;
    large_f64_packbits_reads_as_before: 8;
}

/// A frame at file offset 2^33, read through a slice that starts there: the
/// `u64` piece arithmetic, which on wasm32 is where an `as usize` would wrap.
#[test]
fn a_frame_far_into_the_file_decodes_as_one_at_its_start() {
    const BASE: u64 = 1 << 33;
    for spec in matrix().iter().step_by(13) {
        let (file, info) = build(spec);
        let mut far = info.clone();
        far.strip_offsets = info.strip_offsets.iter().map(|o| o + BASE).collect();
        let near = Frame::new(&file, &info, spec.order);
        let frame = Frame::with_bytes(FrameBytes::relative(BASE, &file), &far, spec.order);
        let read = |f: &Frame<'_>| -> Result<Vec<Vec<u16>>> {
            let n = f.plane_len(Display { range: None })?;
            let mut planes = vec![vec![0u16; n]; spp_of(f.info())];
            let mut s: Vec<&mut [u16]> = planes.iter_mut().map(|v| &mut v[..]).collect();
            f.read_into(
                Display { range: None },
                0,
                &mut s,
                &mut DecodeScratch::new(),
            )?;
            Ok(planes)
        };
        same(&format!("{spec:?} at 2^33"), &read(&near), &read(&frame));
        let stored = |f: &Frame<'_>| -> Result<Vec<u8>> {
            let mut out = vec![0u8; f.stored_frame_len()?];
            f.stored_frame_into(ByteOrder::Little, &mut out, &mut DecodeScratch::new())?;
            Ok(out)
        };
        same(
            &format!("{spec:?} stored at 2^33"),
            &stored(&near),
            &stored(&frame),
        );
        assert_eq!(
            near.borrows(Stored(spec.order), 0),
            frame.borrows(Stored(spec.order), 0)
        );
        // Below the slice's start is outside it, not a wrap to the end.
        let mut before = far.clone();
        before.strip_offsets = far.strip_offsets.iter().map(|o| o - BASE / 2).collect();
        let lost = Frame::with_bytes(FrameBytes::relative(BASE, &file), &before, spec.order);
        assert!(read(&lost).is_err(), "{spec:?}: an offset before the slice");
    }
}

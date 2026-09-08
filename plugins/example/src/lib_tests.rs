//! Tests for the plugin's own logic, called directly.
//!
//! Nothing here crosses the C boundary — that is `tests/boundary.rs`, next
//! door. These check the arithmetic, so a failure there is unambiguously the
//! boundary's fault rather than this plugin's.

use super::*;
use fasttiff_plugin::api::Params;

struct NoHost;
impl ImportHost for NoHost {
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn log(&mut self, _m: &str) {}
}

fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("fasttiff-raw-{name}"));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn request(path: &std::path::Path, pairs: &[(&str, i64)]) -> ImportRequest {
    let mut params = Params::new();
    for (k, v) in pairs {
        params.set(*k, fasttiff_plugin::api::ParamValue::Int(*v));
    }
    ImportRequest {
        path: path.to_path_buf(),
        params,
    }
}

#[test]
fn reads_a_u16_plane_in_the_declared_byte_order() {
    // 2x2, little-endian: 1, 2, 3, 4.
    let bytes: Vec<u8> = [1u16, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let p = tmp("le.raw", &bytes);
    let r = RawImport
        .import(
            &request(&p, &[("width", 2), ("height", 2), ("type", 1)]),
            &mut NoHost,
        )
        .expect("import");
    assert_eq!(r.image.planes, vec![PlaneData::U16(vec![1, 2, 3, 4])]);

    // The same bytes read big-endian must not accidentally agree.
    let mut params = Params::new();
    params.set("width", fasttiff_plugin::api::ParamValue::Int(2));
    params.set("height", fasttiff_plugin::api::ParamValue::Int(2));
    params.set("type", fasttiff_plugin::api::ParamValue::Int(1));
    params.set(
        "little_endian",
        fasttiff_plugin::api::ParamValue::Bool(false),
    );
    let r = RawImport
        .import(
            &ImportRequest {
                path: p.clone(),
                params,
            },
            &mut NoHost,
        )
        .expect("import");
    assert_eq!(
        r.image.planes,
        vec![PlaneData::U16(vec![256, 512, 768, 1024])]
    );
    let _ = std::fs::remove_file(p);
}

#[test]
fn signed_samples_are_offset_rather_than_cast() {
    let bytes: Vec<u8> = [-32768i16, -1, 0, 32767]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let p = tmp("i16.raw", &bytes);
    let r = RawImport
        .import(
            &request(&p, &[("width", 4), ("height", 1), ("type", 2)]),
            &mut NoHost,
        )
        .expect("import");
    // Monotonic: the whole point of offsetting rather than `as u16`.
    assert_eq!(
        r.image.planes,
        vec![PlaneData::U16(vec![0, 32767, 32768, 65535])]
    );
    let _ = std::fs::remove_file(p);
}

#[test]
fn a_header_offset_skips_exactly_that_many_bytes() {
    let mut bytes = vec![0xAAu8; 7];
    bytes.extend([9u16, 8].iter().flat_map(|v| v.to_le_bytes()));
    let p = tmp("offset.raw", &bytes);
    let r = RawImport
        .import(
            &request(
                &p,
                &[("width", 2), ("height", 1), ("type", 1), ("offset", 7)],
            ),
            &mut NoHost,
        )
        .expect("import");
    assert_eq!(r.image.planes, vec![PlaneData::U16(vec![9, 8])]);
    let _ = std::fs::remove_file(p);
}

#[test]
fn a_short_file_says_how_short_rather_than_reading_past_it() {
    let p = tmp("short.raw", &[0u8; 8]);
    let err = RawImport
        .import(
            &request(&p, &[("width", 64), ("height", 64), ("type", 1)]),
            &mut NoHost,
        )
        .expect_err("should refuse");
    let msg = err.to_string();
    // The numbers are what lets a user fix the dialog.
    assert!(msg.contains("8192"), "{msg}");
    assert!(msg.contains('8'), "{msg}");
    let _ = std::fs::remove_file(p);
}

#[test]
fn multiple_images_become_timepoints() {
    let bytes: Vec<u8> = (0u16..8).flat_map(|v| v.to_le_bytes()).collect();
    let p = tmp("frames.raw", &bytes);
    let r = RawImport
        .import(
            &request(
                &p,
                &[("width", 2), ("height", 2), ("type", 1), ("images", 2)],
            ),
            &mut NoHost,
        )
        .expect("import");
    assert_eq!(r.image.frames, 2);
    assert_eq!(
        r.image.planes,
        vec![
            PlaneData::U16(vec![0, 1, 2, 3]),
            PlaneData::U16(vec![4, 5, 6, 7])
        ]
    );
    r.image.validate().expect("shape");
    let _ = std::fs::remove_file(p);
}

#[test]
fn the_dialog_guesses_a_square_that_divides_the_file() {
    // 32x32 u16 = 2048 bytes.
    let p = tmp("guess.raw", &vec![0u8; 32 * 32 * 2]);
    let decls = RawImport.params(&p);
    let width = decls.iter().find(|d| d.key == "width").expect("width");
    assert!(
        matches!(width.kind, ParamKind::Int { default: 32, .. }),
        "{:?}",
        width.kind
    );
    let _ = std::fs::remove_file(p);

    // Nothing divides: fall back rather than offering a wrong-looking number.
    assert_eq!(square_guess(4097, 2), 512);
    assert_eq!(square_guess(0, 2), 512);
}

#[test]
fn probe_answers_only_for_its_own_extensions() {
    assert_eq!(
        RawImport.probe(Path::new("a.raw"), &[]),
        Confidence::Maybe,
        "a .raw file is this importer's business"
    );
    assert_eq!(RawImport.probe(Path::new("a.RAW"), &[]), Confidence::Maybe);
    assert_eq!(RawImport.probe(Path::new("a.tif"), &[]), Confidence::No);
}

use super::*;
use fasttiff_plugin_api::Params;

struct Silent;
impl ImportHost for Silent {
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn log(&mut self, _m: &str) {}
}

fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("fasttiff_netpbm_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn import(path: &std::path::Path) -> Result<ImportResult, PluginError> {
    Netpbm.import(
        &ImportRequest {
            path: path.to_path_buf(),
            params: Params::new(),
        },
        &mut Silent,
    )
}

#[test]
fn reads_an_8_bit_greyscale_pgm() {
    // 3x2, maxval 255.
    let px: Vec<u8> = vec![0, 10, 20, 30, 40, 250];
    let mut f = b"P5\n3 2\n255\n".to_vec();
    f.extend_from_slice(&px);
    let r = import(&write("gray8.pgm", &f)).expect("should import");

    assert_eq!((r.image.width, r.image.height), (3, 2));
    assert_eq!(r.image.channels, 1);
    assert_eq!(r.image.pixel_type, PixelType::U8);
    assert_eq!(r.image.planes.len(), 1);
    match &r.image.planes[0] {
        PlaneData::U8(v) => assert_eq!(v, &px),
        other => panic!("expected U8, got {:?}", other.pixel_type()),
    }
    r.image
        .validate()
        .expect("the result must describe itself correctly");
}

/// 16-bit Netpbm is big-endian; reading it as little-endian would give
/// plausible-looking nonsense rather than an error, so pin the byte order.
#[test]
fn reads_a_16_bit_pgm_as_big_endian() {
    let vals: [u16; 4] = [0, 1, 256, 65535];
    let mut f = b"P5\n2 2\n65535\n".to_vec();
    for v in vals {
        f.extend_from_slice(&v.to_be_bytes());
    }
    let r = import(&write("gray16.pgm", &f)).expect("should import");
    assert_eq!(r.image.pixel_type, PixelType::U16);
    match &r.image.planes[0] {
        PlaneData::U16(v) => assert_eq!(v.as_slice(), &vals),
        other => panic!("expected U16, got {:?}", other.pixel_type()),
    }
}

/// PPM is interleaved RGB; the importer must de-interleave into planes.
#[test]
fn splits_a_ppm_into_three_planes() {
    // Two pixels: (1,2,3) and (4,5,6).
    let mut f = b"P6\n2 1\n255\n".to_vec();
    f.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
    let r = import(&write("rgb.ppm", &f)).expect("should import");

    assert_eq!(r.image.channels, 3);
    let planes: Vec<&Vec<u8>> = r
        .image
        .planes
        .iter()
        .map(|p| match p {
            PlaneData::U8(v) => v,
            other => panic!("expected U8, got {:?}", other.pixel_type()),
        })
        .collect();
    assert_eq!(planes[0], &vec![1, 4], "red");
    assert_eq!(planes[1], &vec![2, 5], "green");
    assert_eq!(planes[2], &vec![3, 6], "blue");
    assert_eq!(
        r.info.as_ref().map(|i| i.mode),
        Some(fasttiff_plugin_api::DisplayMode::Composite),
        "a colour image should composite"
    );
}

/// Comments are legal anywhere in the header, including between the dimensions.
#[test]
fn tolerates_comments_and_extra_whitespace_in_the_header() {
    let mut f =
        b"P5\n# made by a microscope\n  3\n\n2   # width above, height left\n255\n".to_vec();
    f.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
    let r = import(&write("commented.pgm", &f)).expect("should import");
    assert_eq!((r.image.width, r.image.height), (3, 2));
}

/// The magic number decides, not the extension — both ways round.
#[test]
fn probe_trusts_the_signature_over_the_extension() {
    assert_eq!(
        Netpbm.probe(Path::new("a.pgm"), b"P5\n"),
        Confidence::Certain
    );
    assert_eq!(
        Netpbm.probe(Path::new("a.dat"), b"P6\n"),
        Confidence::Certain
    );
    assert_eq!(
        Netpbm.probe(Path::new("a.pgm"), b"II*\0"),
        Confidence::No,
        "a TIFF named .pgm must be declined, not mangled"
    );
    // With nothing to look at, the extension is all there is.
    assert_eq!(Netpbm.probe(Path::new("a.pgm"), b""), Confidence::Maybe);
    assert_eq!(Netpbm.probe(Path::new("a.tif"), b""), Confidence::No);
}

#[test]
fn a_truncated_file_is_an_error_not_a_panic() {
    let mut f = b"P5\n4 4\n255\n".to_vec();
    f.extend_from_slice(&[1, 2, 3]); // 16 bytes declared, 3 supplied
    let err = import(&write("short.pgm", &f)).unwrap_err();
    assert!(
        err.to_string().contains("pixel bytes"),
        "the error should say what was missing: {err}"
    );
}

#[test]
fn malformed_headers_are_refused() {
    for (name, bytes) in [
        ("notnetpbm.pgm", &b"NOPE\n1 1\n255\n\0"[..]),
        ("ascii.pgm", &b"P2\n1 1\n255\n0"[..]), // ASCII variants are not supported
        ("zerodim.pgm", &b"P5\n0 5\n255\n"[..]),
        ("badmax.pgm", &b"P5\n1 1\n70000\n\0"[..]),
        ("truncheader.pgm", &b"P5\n1 "[..]),
        // No whitespace byte between maxval and pixels: every pixel would shift.
        ("nogap.pgm", &b"P5\n1 1\n255"[..]),
    ] {
        assert!(
            import(&write(name, bytes)).is_err(),
            "{name} should have been refused"
        );
    }
}

/// The whole point: the imported result opens as an ordinary document.
#[test]
fn an_imported_image_becomes_a_real_stack() {
    let mut f = b"P6\n2 2\n255\n".to_vec();
    f.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let r = import(&write("doc.ppm", &f)).expect("should import");
    let stack = crate::plugins::to_stack(&r.image, r.info.as_ref(), false)
        .expect("an imported image must open like any other");
    assert_eq!(stack.dimensions(), Some((2, 2)));
    assert_eq!(stack.display.dims.channels, 3);
}

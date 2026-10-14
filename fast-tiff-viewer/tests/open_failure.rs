//! What the status line says when a file will not open.
//!
//! The wording matters more than it looks. This viewer reads TIFF *and*
//! whatever its importer plugins bring — OIR, NIfTI, DICOM, PNG, and anything
//! loaded from the plugin folder — so a failure that names TIFF describes the
//! program as it was two formats ago. But a *broken* TIFF is a different thing
//! from an *unsupported* file, and the user does different things about each:
//! one sends you looking for a better copy of the file, the other for a
//! different program. These tests hold that line in both directions.

use fast_tiff_viewer::Viewer;
use std::path::PathBuf;

/// Drive a failed open the way the browser entry point does, and return the
/// status line the user would read.
fn status_after_opening(bytes: Vec<u8>, name: &str) -> String {
    let mut viewer = Viewer::new();
    let err = viewer.load_bytes(bytes, PathBuf::from(name));
    assert!(err.is_err(), "{name} was supposed to fail to open");
    viewer.status.expect("a failed open should leave a status")
}

/// The case the wording was changed for: bytes that are not any format this
/// knows. Every importer has already declined by the time the TIFF reader sees
/// them, so naming TIFF would answer a narrower question than the user asked.
#[test]
fn an_unrecognised_file_is_unsupported_rather_than_not_a_tiff() {
    let status = status_after_opening(b"this is not any image at all".to_vec(), "mystery.dat");
    assert_eq!(status, "Failed to open file: file not supported");
    assert!(
        !status.contains("TIFF"),
        "the one format in the name is not the only one this reads: {status}"
    );
}

/// Same for a file whose byte order is TIFF's but whose magic is not — a
/// near-miss is still not a TIFF, and still not a format failure the user can
/// do anything about by finding a better copy.
#[test]
fn a_tiff_byte_order_with_the_wrong_magic_is_also_unsupported() {
    let mut bytes = b"II".to_vec();
    bytes.extend_from_slice(&7u16.to_le_bytes()); // neither 42 nor 43
    bytes.extend_from_slice(&8u32.to_le_bytes());
    assert_eq!(
        status_after_opening(bytes, "almost.tif"),
        "Failed to open file: file not supported"
    );
}

/// The other direction, and the reason this is not a blanket replacement: a
/// file that *is* a TIFF and is damaged keeps its own message. Flattening this
/// into "file not supported" would be a lie — the format is supported, this
/// copy of it is broken — and would throw away the only part of the message
/// that tells the user which of the two problems they have.
#[test]
fn a_damaged_tiff_keeps_its_specific_message() {
    // BigTIFF magic, then an offset size that is not 8.
    let mut bytes = b"II".to_vec();
    bytes.extend_from_slice(&43u16.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes()); // must be 8
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&16u64.to_le_bytes());

    let status = status_after_opening(bytes, "damaged.tif");
    assert!(
        status.contains("malformed BigTIFF header"),
        "a broken TIFF should say so: {status}"
    );
    assert!(
        !status.contains("file not supported"),
        "BigTIFF is supported; this copy is damaged: {status}"
    );
}

/// A file too short to hold a header is damaged rather than unsupported too —
/// there are not enough bytes to tell what it was meant to be, and claiming to
/// know would be guessing.
#[test]
fn a_truncated_file_is_not_reported_as_unsupported() {
    let status = status_after_opening(b"II".to_vec(), "truncated.tif");
    assert!(
        !status.contains("file not supported"),
        "too short to classify, so it should not be classified: {status}"
    );
    assert!(status.contains("too small"), "{status}");
}

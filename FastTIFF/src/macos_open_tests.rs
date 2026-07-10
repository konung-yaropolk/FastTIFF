//! Unit tests for `macos_open`'s pure URL-to-path parsing (the FFI half
//! needs a live AppKit and is exercised manually / in CI on macOS).

use super::*;

#[test]
fn plain_path() {
    assert_eq!(file_url_to_path(b"file:///Users/me/scan.tif"), Some(PathBuf::from("/Users/me/scan.tif")));
}

#[test]
fn percent_encoded_spaces_and_unicode() {
    // "/Users/me/My Scan é.tif"
    let url = b"file:///Users/me/My%20Scan%20%C3%A9.tif";
    assert_eq!(file_url_to_path(url), Some(PathBuf::from("/Users/me/My Scan é.tif")));
}

#[test]
fn localhost_authority_is_stripped() {
    assert_eq!(file_url_to_path(b"file://localhost/tmp/a.tiff"), Some(PathBuf::from("/tmp/a.tiff")));
}

#[test]
fn non_file_url_rejected() {
    assert_eq!(file_url_to_path(b"http://example.com/a.tif"), None);
}

#[test]
fn malformed_escape_is_literal() {
    assert_eq!(percent_decode(b"a%2z%"), b"a%2z%");
}

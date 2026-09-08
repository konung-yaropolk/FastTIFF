//! Where a plugin's result gets written.
//!
//! One rule, and the bug it comes from: a window showing a result **memory-maps
//! that file for as long as it is open**. Writing the next result over it fails
//! outright on Windows, and the caller's last resort — show it in this window
//! instead — then replaced the document the user was working from. So running a
//! plugin twice without closing the first result cost you the file you ran it
//! on.

use super::write_result;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fasttiff-result-tests-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// A second result never lands on the first one's file.
#[test]
fn a_second_result_gets_a_file_of_its_own() {
    let dir = scratch("second");

    let first = write_result(&dir, "stack-derivatives", b"one").expect("the first result");
    assert_eq!(first.file_name().unwrap(), "stack-derivatives.tif");

    let second = write_result(&dir, "stack-derivatives", b"two").expect("the second result");
    assert_eq!(second.file_name().unwrap(), "stack-derivatives-2.tif");
    let third = write_result(&dir, "stack-derivatives", b"three").expect("the third result");
    assert_eq!(third.file_name().unwrap(), "stack-derivatives-3.tif");

    // And each one still holds what it was given: a name that is merely unique
    // is no use if the bytes went somewhere else.
    for (path, want) in [(&first, "one"), (&second, "two"), (&third, "three")] {
        assert_eq!(std::fs::read(path).expect("read it back"), want.as_bytes());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The first result's file is never opened for writing at all, which is what
/// makes the rule hold on platforms where overwriting a mapped file quietly
/// succeeds rather than failing.
#[test]
fn an_existing_result_is_left_exactly_as_it_was() {
    let dir = scratch("untouched");
    let first = write_result(&dir, "r", b"original").expect("the first result");
    let before = std::fs::metadata(&first).expect("metadata").modified().ok();

    write_result(&dir, "r", b"a much longer second result").expect("the second result");

    assert_eq!(std::fs::read(&first).expect("read"), b"original");
    assert_eq!(
        std::fs::metadata(&first).expect("metadata").modified().ok(),
        before,
        "the first result was reopened"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Somewhere unwritable is reported rather than retried into silence: the
/// caller shows the result in the current window instead, and needs to say why.
#[test]
fn nowhere_to_write_comes_back_as_an_error() {
    let dir = std::env::temp_dir().join("fasttiff-result-tests-missing/not/here");
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("fasttiff-result-tests-missing"));
    assert!(write_result(&dir, "r", b"x").is_err());
}

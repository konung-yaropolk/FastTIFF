//! What the layout pins cannot check.
//!
//! The sizes, offsets and wire numbers are asserted in a `const` block in
//! `lib.rs`, not here: a layout is a compile-time property, so a compile-time
//! assertion checks it for every target the crate is *built* for — including
//! the 32-bit ones nobody runs a test suite on. These tests used to do that job
//! and did it badly, returning early on any target that was not 64-bit, which
//! meant they passed green on i686 while asserting nothing at all.
//!
//! What is left here is the part that genuinely needs to run: the behaviour of
//! [`FtStr::as_str`], which is the one place a hostile plugin's bytes become a
//! Rust `&str`.

use super::*;

/// The symbol name is a string, not a layout, so it is checked here — and the
/// *plugin* side's copy of it is checked at compile time by `export_plugin!`,
/// through [`is_query_symbol`].
#[test]
fn the_entry_symbol_carries_the_major_version() {
    // A v2 host must not find a v1 plugin's entry point, so the mismatch is a
    // missing symbol rather than a layout disagreement mid-call.
    assert_eq!(QUERY_SYMBOL, b"ft_plugin_v1_query\0");
    assert!(QUERY_SYMBOL.ends_with(&[0]), "dlsym needs a NUL terminator");
    let name = core::str::from_utf8(&QUERY_SYMBOL[..QUERY_SYMBOL.len() - 1]).unwrap();
    assert!(
        name.contains(&format!("v{ABI_MAJOR}")),
        "the symbol name must contain the major version, or version negotiation cannot work"
    );
    assert!(is_query_symbol(QUERY_SYMBOL));
    assert!(!is_query_symbol(b"ft_plugin_v2_query\0"));
    assert!(
        !is_query_symbol(b"ft_plugin_v1_query"),
        "the NUL is part of it"
    );
}

/// `as_str` is the one place a hostile plugin's bytes become a Rust `&str`, so
/// it has to refuse everything that is not one.
#[test]
fn ftstr_refuses_what_it_should() {
    unsafe {
        assert_eq!(FtStr::EMPTY.as_str(), Some(""));

        // Null with a non-zero length is a corrupt struct.
        let bad = FtStr {
            ptr: core::ptr::null(),
            len: 5,
        };
        assert_eq!(bad.as_str(), None);

        // Invalid UTF-8 must not become a &str.
        let raw = [0xffu8, 0xfe, 0xfd];
        let s = FtStr {
            ptr: raw.as_ptr(),
            len: 3,
        };
        assert_eq!(s.as_str(), None);

        // An absurd length is refused before a slice that size is constructed.
        let huge = FtStr {
            ptr: raw.as_ptr(),
            len: u64::MAX,
        };
        assert_eq!(huge.as_str(), None);

        // And a real string still works.
        let ok = FtStr::from_str("Invert");
        assert_eq!(ok.as_str(), Some("Invert"));
    }
}

/// A zero-length string with a dangling-but-nonnull pointer is legal and must
/// not be dereferenced.
#[test]
fn a_zero_length_string_is_never_dereferenced() {
    unsafe {
        let s = FtStr {
            ptr: core::ptr::dangling::<u8>(),
            len: 0,
        };
        assert_eq!(s.as_str(), Some(""));
    }
}

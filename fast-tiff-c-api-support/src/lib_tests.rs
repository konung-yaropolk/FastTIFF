// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use super::*;
use std::ptr::{null, null_mut};

thread_local! {
    static SLOT: RefCell<CString> = RefCell::new(CString::default());
    static OTHER: RefCell<CString> = RefCell::new(CString::default());
}
static ERRORS: Errors = Errors::new(&SLOT, status::INVALID_ARGUMENT, status::FAILED, "the test");
static OTHER_ERRORS: Errors = Errors::new(
    &OTHER,
    status::INVALID_ARGUMENT,
    status::FAILED,
    "the other",
);

fn last(e: &Errors) -> String {
    unsafe { CStr::from_ptr(e.last_error()) }
        .to_str()
        .unwrap()
        .to_string()
}

#[test]
fn a_failure_is_recorded_in_its_own_librarys_slot() {
    assert_eq!(
        ERRORS.fail(status::UNAVAILABLE, "not here"),
        status::UNAVAILABLE
    );
    assert_eq!(last(&ERRORS), "not here");
    // Another library on the same thread keeps its own.
    assert_eq!(last(&OTHER_ERRORS), "");
    OTHER_ERRORS.fail(status::FAILED, "elsewhere");
    assert_eq!(last(&ERRORS), "not here");
    // A NUL in the reason cannot truncate it.
    ERRORS.fail(status::FAILED, "a\0b");
    assert_eq!(last(&ERRORS), "a b");
    assert_eq!(ERRORS.fail_len(status::INVALID_ARGUMENT, "x"), -2);
}

#[test]
fn a_panic_becomes_the_failed_status_with_its_message() {
    let got = ERRORS.guarded(|| panic!("boom"));
    assert_eq!(got, status::FAILED);
    assert_eq!(last(&ERRORS), "the test panicked: boom");
    let got = ERRORS.guarded_len(|| panic!("{}", String::from("formatted")));
    assert_eq!(got, -(status::FAILED as isize));
    assert_eq!(last(&ERRORS), "the test panicked: formatted");
    assert_eq!(ERRORS.guarded(|| status::OK), status::OK);
}

#[test]
fn text_arguments_are_checked() {
    assert_eq!(unsafe { ERRORS.text_arg(c"hi".as_ptr(), "name") }, Ok("hi"));
    assert_eq!(
        unsafe { ERRORS.text_arg(null(), "path") },
        Err(status::INVALID_ARGUMENT)
    );
    assert_eq!(last(&ERRORS), "path is NULL");
    let bad = [0xffu8, 0];
    assert_eq!(
        unsafe { ERRORS.text_arg(bad.as_ptr().cast(), "path") },
        Err(status::INVALID_ARGUMENT)
    );
    assert_eq!(last(&ERRORS), "path is not UTF-8");
}

#[test]
fn text_goes_out_the_snprintf_way_cut_between_characters() {
    let s = "ab\u{b5}m";
    assert_eq!(
        unsafe { ERRORS.put_text(s, null_mut(), 0) },
        5,
        "asks the length"
    );
    assert_eq!(
        unsafe { ERRORS.put_text(s, null_mut(), 3) },
        -(status::INVALID_ARGUMENT as isize),
        "NULL with room is refused"
    );
    for (cap, want) in [
        (1, ""),
        (3, "ab"),
        (4, "ab"),
        (5, "ab\u{b5}"),
        (9, "ab\u{b5}m"),
    ] {
        let mut buf = vec![0x55u8; cap];
        assert_eq!(
            unsafe { ERRORS.put_text(s, buf.as_mut_ptr().cast(), cap) },
            5
        );
        let got = CStr::from_bytes_until_nul(&buf).unwrap().to_str();
        assert_eq!(got, Ok(want), "cap {cap}");
    }
    let mut untouched = [0x55u8; 2];
    unsafe { ERRORS.put_text(s, untouched.as_mut_ptr().cast(), 0) };
    assert_eq!(untouched, [0x55; 2], "cap 0 writes nothing");
}

#[test]
fn an_index_past_the_end_is_an_invalid_argument() {
    let list = [10, 20];
    assert_eq!(ERRORS.index_in(&list, 1, "channel"), Ok(&20));
    assert_eq!(
        ERRORS.index_in(&list, 2, "channel"),
        Err(status::INVALID_ARGUMENT)
    );
    assert_eq!(last(&ERRORS), "channel 2 is past the end; there are 2");
    assert_eq!(len_of(Ok(7)), 7);
    assert_eq!(len_of(Err(status::FAILED)), -4);
}

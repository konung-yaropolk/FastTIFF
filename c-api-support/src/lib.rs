// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The scaffolding of this repository's C APIs, written once.
//!
//! `fast-tiff-lib`, `oir-reader`, `suite2p-registration` and
//! `turboreg-registration` each expose a C interface, and each used to carry
//! its own copy of the same machinery around it: a thread-local "why did the
//! last call fail", a guard that turns a panic into a status instead of an
//! unwind into C, the `snprintf` convention for handing strings out, and the
//! checks on a string handed in. The copies drifted — one learned to cut a
//! string between characters, the others went on cutting through them — so
//! they are kept here instead.
//!
//! What stays with each library is what is its own: its status names (built
//! on [`status`], so the first five mean the same everywhere), its
//! thread-local slot (a library's `*_last_error` must report that library's
//! failures, not another's), and how its own error type maps onto statuses.
//!
//! ```
//! use c_api_support::{status, Errors};
//! use std::cell::RefCell;
//! use std::ffi::{c_int, CString};
//!
//! pub const MY_INVALID_ARGUMENT: c_int = status::INVALID_ARGUMENT;
//! pub const MY_FAILED: c_int = status::FAILED;
//!
//! thread_local! {
//!     static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
//! }
//! static ERRORS: Errors = Errors::new(&LAST_ERROR, MY_INVALID_ARGUMENT, MY_FAILED, "the library");
//!
//! pub extern "C" fn my_do_it() -> c_int {
//!     ERRORS.guarded(|| {
//!         // ...
//!         status::OK
//!     })
//! }
//! ```

use std::any::Any;
use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::thread::LocalKey;

/// The statuses every C API here shares, by number.
///
/// A host that opens a file with one of these libraries and hands it to
/// another handles these five once. Each library names them with its own
/// prefix (`FTL_OK`, `OIR_OK`, ...) and numbers its own statuses from
/// [`FIRST_OWN`].
pub mod status {
    use std::ffi::c_int;

    /// The call did what it was asked.
    pub const OK: c_int = 0;
    /// A progress callback asked to stop. Reserved in a library with no call
    /// that can be stopped.
    pub const CANCELLED: c_int = 1;
    /// A pointer, size, index, buffer or option the call cannot take.
    pub const INVALID_ARGUMENT: c_int = 2;
    /// Not in this build or on this target.
    pub const UNAVAILABLE: c_int = 3;
    /// Something inside the library went wrong: a panic, or a state it
    /// should not have reached.
    pub const FAILED: c_int = 4;
    /// The first number a library may give a status of its own.
    pub const FIRST_OWN: c_int = 5;
}

/// The thread-local slot a library keeps its last failure in.
pub type Slot = LocalKey<RefCell<CString>>;

/// One C API's failure reporting: its own last-error slot, and the statuses
/// it returns for a bad argument and for a panic.
///
/// Built once per library as a `static`, around that library's own
/// `thread_local!` (see the crate docs).
pub struct Errors {
    slot: &'static Slot,
    invalid_argument: c_int,
    failed: c_int,
    /// What panicked, for the message: "the reader", "the registration".
    who: &'static str,
}

impl Errors {
    pub const fn new(
        slot: &'static Slot,
        invalid_argument: c_int,
        failed: c_int,
        who: &'static str,
    ) -> Self {
        Errors {
            slot,
            invalid_argument,
            failed,
            who,
        }
    }

    /// Record why a call failed, for the library's `*_last_error`, and return
    /// its status.
    pub fn fail(&self, status: c_int, why: impl Into<String>) -> c_int {
        let why = CString::new(why.into().replace('\0', " ")).unwrap_or_default();
        // `try_with`: a call from a thread being torn down (a C++
        // `thread_local` destructor) finds the slot gone, and `with` would
        // panic there, which past `extern "C"` is an abort. The status still
        // comes back.
        let _ = self.slot.try_with(|e| *e.borrow_mut() = why);
        status
    }

    /// [`fail`](Self::fail), for a call that returns a length: the status,
    /// negated.
    pub fn fail_len(&self, status: c_int, why: impl Into<String>) -> isize {
        -(self.fail(status, why) as isize)
    }

    /// The reason recorded by the last [`fail`](Self::fail) on this thread,
    /// NUL-terminated; empty before any. Valid until the next failure on the
    /// same thread.
    pub fn last_error(&self) -> *const c_char {
        self.slot
            .try_with(|e| e.borrow().as_ptr())
            .unwrap_or(c"".as_ptr())
    }

    /// Run `work`, turning a panic into the library's failed status rather
    /// than letting it unwind into a caller that has no idea what one is.
    pub fn guarded(&self, work: impl FnOnce() -> c_int) -> c_int {
        catch_unwind(AssertUnwindSafe(work))
            .unwrap_or_else(|p| self.fail(self.failed, self.panicked(p)))
    }

    /// [`guarded`](Self::guarded), for a call that returns a length.
    pub fn guarded_len(&self, work: impl FnOnce() -> isize) -> isize {
        catch_unwind(AssertUnwindSafe(work))
            .unwrap_or_else(|p| self.fail_len(self.failed, self.panicked(p)))
    }

    /// A NUL-terminated UTF-8 argument, or the status to return for it.
    ///
    /// # Safety
    ///
    /// `s` is null or points to a NUL-terminated string that outlives `'a`.
    pub unsafe fn text_arg<'a>(&self, s: *const c_char, name: &str) -> Result<&'a str, c_int> {
        if s.is_null() {
            return Err(self.fail(self.invalid_argument, format!("{name} is NULL")));
        }
        CStr::from_ptr(s)
            .to_str()
            .map_err(|_| self.fail(self.invalid_argument, format!("{name} is not UTF-8")))
    }

    /// Hand `s` to C the way `snprintf` would: as much as fits in `cap` bytes
    /// with a NUL after it — cut at a character boundary, never inside one, so
    /// C is never handed invalid UTF-8 — and the length of the whole, without
    /// the NUL, as the answer. `buf` NULL with `cap` 0 asks for the length;
    /// NULL with any other `cap` is refused, as the negated invalid-argument
    /// status.
    ///
    /// # Safety
    ///
    /// `buf` is null or points to `cap` writable bytes.
    pub unsafe fn put_text(&self, s: &str, buf: *mut c_char, cap: usize) -> isize {
        if buf.is_null() {
            if cap != 0 {
                return self.fail_len(
                    self.invalid_argument,
                    format!("buf is NULL but cap is {cap}; pass 0 to ask for the length"),
                );
            }
        } else if cap != 0 {
            let mut n = s.len().min(cap - 1);
            while !s.is_char_boundary(n) {
                n -= 1;
            }
            std::ptr::copy_nonoverlapping(s.as_ptr(), buf.cast::<u8>(), n);
            buf.add(n).write(0);
        }
        // No string is longer than isize::MAX bytes.
        s.len() as isize
    }

    /// Element `i` of `list`, or the status to return for one past the end.
    pub fn index_in<'a, T>(&self, list: &'a [T], i: usize, what: &str) -> Result<&'a T, c_int> {
        list.get(i).ok_or_else(|| {
            self.fail(
                self.invalid_argument,
                format!("{what} {i} is past the end; there are {}", list.len()),
            )
        })
    }

    /// The message for a caught panic: what panicked, and its own words.
    fn panicked(&self, panic: Box<dyn Any + Send>) -> String {
        let why = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no message".to_string());
        format!("{} panicked: {why}", self.who)
    }
}

/// The status of a `Result<_, c_int>` as a length: the length, or the status
/// negated.
pub fn len_of(r: Result<isize, c_int>) -> isize {
    r.unwrap_or_else(|status| -(status as isize))
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

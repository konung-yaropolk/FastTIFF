//! Minimal, hand-written FFI for the two C readers we compare against.
//!
//! Only the handful of symbols the benchmark actually calls are declared.
//! Keeping this tiny (rather than pulling in bindgen) means no extra network
//! deps and a binding surface that is easy to audit. A few extra symbols are
//! declared for completeness/documentation and may be unused.
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// libtiff (system, -ltiff) — only under `--features libtiff`
// ---------------------------------------------------------------------------
#[cfg(feature = "libtiff")]
pub mod libtiff {
    use super::*;

    #[repr(C)]
    pub struct TIFF {
        _private: [u8; 0],
    }

    // TIFF tag numbers we read.
    pub const TIFFTAG_IMAGEWIDTH: u32 = 256;
    pub const TIFFTAG_IMAGELENGTH: u32 = 257;
    pub const TIFFTAG_BITSPERSAMPLE: u32 = 258;
    pub const TIFFTAG_SAMPLESPERPIXEL: u32 = 277;
    pub const TIFFTAG_ROWSPERSTRIP: u32 = 278;
    // Tags used only on the writer side.
    pub const TIFFTAG_PHOTOMETRIC: u32 = 262;
    pub const TIFFTAG_PLANARCONFIG: u32 = 284;
    pub const TIFFTAG_SAMPLEFORMAT: u32 = 339;
    pub const TIFFTAG_COMPRESSION: u32 = 259;
    pub const TIFFTAG_ORIENTATION: u32 = 274;

    pub const PHOTOMETRIC_MINISBLACK: u32 = 1;
    pub const PLANARCONFIG_CONTIG: u32 = 1;
    pub const SAMPLEFORMAT_UINT: u32 = 1;
    pub const COMPRESSION_NONE: u32 = 1;
    pub const ORIENTATION_TOPLEFT: u32 = 1;

    extern "C" {
        pub fn TIFFOpen(filename: *const c_char, mode: *const c_char) -> *mut TIFF;
        pub fn TIFFClose(tif: *mut TIFF);

        // TIFFGetField/TIFFSetField are variadic in C. We declare the exact
        // single-u32-argument forms we use; passing one u32 (by value for set,
        // by out-pointer for get) is ABI-compatible with the variadic call on
        // every platform libtiff supports.
        pub fn TIFFGetField(tif: *mut TIFF, tag: u32, out: *mut u32) -> c_int;
        pub fn TIFFSetField(tif: *mut TIFF, tag: u32, value: u32) -> c_int;

        pub fn TIFFReadEncodedStrip(
            tif: *mut TIFF,
            strip: u32,
            buf: *mut c_void,
            size: isize,
        ) -> isize;
        pub fn TIFFWriteEncodedStrip(
            tif: *mut TIFF,
            strip: u32,
            buf: *mut c_void,
            size: isize,
        ) -> isize;
        pub fn TIFFNumberOfStrips(tif: *mut TIFF) -> u32;
        pub fn TIFFStripSize(tif: *mut TIFF) -> isize;
        pub fn TIFFDefaultStripSize(tif: *mut TIFF, request: u32) -> u32;

        pub fn TIFFReadDirectory(tif: *mut TIFF) -> c_int;
        pub fn TIFFWriteDirectory(tif: *mut TIFF) -> c_int;
        pub fn TIFFGetVersion() -> *const c_char;
    }
}

// ---------------------------------------------------------------------------
// TinyTIFF (vendored, compiled by build.rs)
// ---------------------------------------------------------------------------
pub mod tinytiff {
    use super::*;

    #[repr(C)]
    pub struct TinyTIFFReaderFile {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn TinyTIFFReader_open(filename: *const c_char) -> *mut TinyTIFFReaderFile;
        pub fn TinyTIFFReader_close(tiff: *mut TinyTIFFReaderFile);
        pub fn TinyTIFFReader_getWidth(tiff: *mut TinyTIFFReaderFile) -> u32;
        pub fn TinyTIFFReader_getHeight(tiff: *mut TinyTIFFReaderFile) -> u32;
        pub fn TinyTIFFReader_getBitsPerSample(tiff: *mut TinyTIFFReaderFile, sample: c_int) -> u16;
        pub fn TinyTIFFReader_getSampleData(
            tiff: *mut TinyTIFFReaderFile,
            buffer: *mut c_void,
            sample: u16,
        ) -> c_int;
        pub fn TinyTIFFReader_readNext(tiff: *mut TinyTIFFReaderFile) -> c_int;
        pub fn TinyTIFFReader_hasNext(tiff: *mut TinyTIFFReaderFile) -> c_int;
        pub fn TinyTIFFReader_countFrames(tiff: *mut TinyTIFFReaderFile) -> u32;
        pub fn TinyTIFFReader_getVersion() -> *const c_char;
        pub fn TinyTIFFReader_wasError(tiff: *mut TinyTIFFReaderFile) -> c_int;
    }
}

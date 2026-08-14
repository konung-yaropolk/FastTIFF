//! Fuzz the *indexing* half: header parse + IFD-chain walk + metadata parse.
//!
//! `TiffStack::from_bytes` takes a completely untrusted byte string, so the only
//! acceptable outcomes are `Ok(stack)` or `Err(_)`. A panic, an abort (which is
//! what an oversized `vec![]` produces — allocation failure is *not* catchable),
//! or an OOM is a bug. libFuzzer's `-rss_limit_mb` catches the last one.
//!
//! Run with:  cargo +nightly fuzz run tiff_open
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The fuzzer owns `data`; `from_bytes` wants an owned Vec.
    if let Ok(stack) = fast_tiff_lib::TiffStack::from_bytes(data.to_vec()) {
        // Touch the parsed surface so metadata work is not optimized away.
        let _ = stack.frames.len();
        let _ = stack.meta.channels;
        let _ = stack.meta.slices;
        let _ = stack.meta.frames;
        let _ = stack.meta.voxel_scale();
        let _ = stack.description.as_deref().map(str::len);
        if let Some(cd) = stack.meta.channel_display.first() {
            let _ = cd.lut[255];
        }
    }
});

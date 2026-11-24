//! What a warm decode allocates: nothing frame-sized.
//!
//! A test binary of its own because it replaces the global allocator with one
//! that counts, which would count every other suite's allocations too. The
//! tests take a lock so that only one of them is decoding at a time, and they
//! count what every thread allocates — a parallel decode's workers included.
//!
//! The old readers allocated and zero-filled a frame-sized buffer for every
//! compressed frame (and a frame of `f32` for every wide frame shown as
//! `u16`), whatever the caller held. Through [`Frame::read_into`] with a
//! [`DecodeScratch`] kept across frames, a steady stream of frames allocates
//! none of that.

use fast_tiff_lib::units::{Display, File, Stored};
use fast_tiff_lib::{
    parallel_decode_scope, Compression, DecodeScratch, Frame, SampleType, TiffStack, TiffWriter,
    WriterOptions,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Counts allocations of at least [`AT_LEAST`] bytes.
struct Counting;

static AT_LEAST: AtomicUsize = AtomicUsize::new(usize::MAX);
static COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= AT_LEAST.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= AT_LEAST.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= AT_LEAST.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// How many allocations of at least `bytes` `f` makes.
fn allocations_of_at_least(bytes: usize, f: impl FnOnce()) -> usize {
    COUNT.store(0, Ordering::SeqCst);
    AT_LEAST.store(bytes, Ordering::SeqCst);
    f();
    AT_LEAST.store(usize::MAX, Ordering::SeqCst);
    COUNT.load(Ordering::SeqCst)
}

fn stack(
    st: SampleType,
    compression: Compression,
    predictor: bool,
    rows_per_strip: u32,
) -> TiffStack {
    let (w, h) = (2048u32, 2048u32);
    let opts = WriterOptions::new(w, h, st)
        .compression(compression)
        .predictor(predictor)
        .rows_per_strip(rows_per_strip);
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let n = (w * h) as usize;
    let bytes: Vec<u8> = (0..n * st.bytes())
        .map(|i| ((i / st.bytes()) % 4093 + i % st.bytes() * 31) as u8)
        .collect();
    writer.write_frame_bytes(&bytes).unwrap();
    TiffStack::from_bytes(writer.finish().unwrap().into_inner()).unwrap()
}

fn codecs() -> Vec<Compression> {
    let mut c = vec![Compression::Deflate, Compression::Lzw];
    if cfg!(feature = "codec-zstd") {
        c.push(Compression::Zstd);
    }
    c
}

/// The second read of a compressed 2048x2048 u16 frame into a warm
/// destination and scratch allocates nothing of 64 KiB or more: the strips
/// decompress straight into the destination. The old reader allocated and
/// zeroed an 8 MiB frame for every one.
#[test]
fn a_warm_compressed_u16_read_allocates_nothing_big() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for compression in codecs() {
        let stack = stack(SampleType::U16, compression, true, 64);
        let frame = stack.frame(0).unwrap();
        let units = Display { range: None };
        for parallel in [false, true] {
            let _p = parallel_decode_scope(parallel);
            let mut out = vec![0u16; frame.plane_len(units).unwrap()];
            let mut scratch = DecodeScratch::new();
            frame
                .read_into(units, 0, &mut [&mut out[..]], &mut scratch)
                .unwrap();
            let big = allocations_of_at_least(64 << 10, || {
                frame
                    .read_into(units, 0, &mut [&mut out[..]], &mut scratch)
                    .unwrap();
            });
            assert_eq!(big, 0, "{compression:?}, parallel {parallel}");
        }
    }
}

/// Allocations of 64 KiB or more over `reads` warm reads, after two to warm
/// up: none when serial. In parallel a piece buffer is one per worker that
/// ran a group, and which workers do is the scheduler's to say, so a warm
/// scratch can still meet a worker it has not seen; it can do that once per
/// worker at most, where losing its buffers would cost a few every read.
fn big_allocations_warm(parallel: bool, mut read: impl FnMut()) -> (usize, usize) {
    let _p = parallel_decode_scope(parallel);
    read();
    read();
    let reads = if parallel {
        2 * rayon::current_num_threads()
    } else {
        3
    };
    let big = allocations_of_at_least(64 << 10, || (0..reads).for_each(|_| read()));
    let allowed = if parallel {
        rayon::current_num_threads()
    } else {
        0
    };
    (big, allowed)
}

/// A conversion a piece at a time needs piece buffers, not a frame: once the
/// scratch holds them, `u16` to `f32` and chunky RGB gathers allocate nothing
/// big either.
#[test]
fn a_warm_converting_read_allocates_nothing_big() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lzw = stack(SampleType::U16, Compression::Lzw, false, 64);
    let frame = lzw.frame(0).unwrap();
    for parallel in [false, true] {
        let mut out = vec![0f32; frame.plane_len(File).unwrap()];
        let mut scratch = DecodeScratch::new();
        // The piece buffers are a strip each, 256 KiB: below what is counted
        // they would go unseen.
        let (big, allowed) = big_allocations_warm(parallel, || {
            frame
                .read_into(File, 0, &mut [&mut out[..]], &mut scratch)
                .unwrap();
        });
        assert!(
            big <= allowed,
            "u16 LZW to f32, parallel {parallel}: {big} allocations of 64 KiB or more"
        );
    }

    let opts = WriterOptions::new(1024, 1024, SampleType::U8)
        .samples_per_pixel(3)
        .compression(Compression::Deflate)
        .predictor(true)
        .rows_per_strip(32);
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let rgb: Vec<u8> = (0..1024 * 1024 * 3).map(|i| (i % 251) as u8).collect();
    writer.write_frame_u8(&rgb).unwrap();
    let stack = TiffStack::from_bytes(writer.finish().unwrap().into_inner()).unwrap();
    let frame = stack.frame(0).unwrap();
    let order = stack.byte_order;
    let n = frame.plane_len(Stored(order)).unwrap();
    let mut planes = vec![vec![0u8; n]; 3];
    let mut scratch = DecodeScratch::new();
    let read = |planes: &mut [Vec<u8>], scratch: &mut DecodeScratch| {
        let [r, g, b] = planes else { unreachable!() };
        frame
            .read_into(
                Stored(order),
                0,
                &mut [&mut r[..], &mut g[..], &mut b[..]],
                scratch,
            )
            .unwrap();
    };
    read(&mut planes, &mut scratch);
    let big = allocations_of_at_least(64 << 10, || read(&mut planes, &mut scratch));
    assert_eq!(big, 0, "chunky RGB8 Deflate+predictor into three planes");
}

/// Wide samples shown as `u16` with no range have to be ranged first. The old
/// reader converted the whole frame to a `Vec<f32>` to do it; the core folds
/// over the stored samples — the source's own bytes when uncompressed, so not
/// even a cold scratch allocates, and the scratch's assembled frame when
/// compressed.
#[test]
fn an_auto_ranged_u32_read_allocates_no_frame_of_f32() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let units = Display { range: None };
    let raw = stack(SampleType::U32, Compression::None, false, 2048);
    let frame = raw.frame(0).unwrap();
    let mut out = vec![0u16; frame.plane_len(units).unwrap()];
    let big = allocations_of_at_least(1 << 20, || {
        frame
            .read_into(units, 0, &mut [&mut out[..]], &mut DecodeScratch::new())
            .unwrap();
    });
    assert_eq!(big, 0, "uncompressed u32, cold scratch");

    let packed = stack(SampleType::U32, Compression::Deflate, false, 64);
    let frame: Frame<'_> = packed.frame(0).unwrap();
    let mut scratch = DecodeScratch::new();
    frame
        .read_into(units, 0, &mut [&mut out[..]], &mut scratch)
        .unwrap();
    let big = allocations_of_at_least(1 << 20, || {
        frame
            .read_into(units, 0, &mut [&mut out[..]], &mut scratch)
            .unwrap();
    });
    assert_eq!(big, 0, "Deflate u32, warm scratch");
    let ranged = allocations_of_at_least(1 << 20, || {
        frame.value_range(&mut scratch).unwrap();
    });
    assert_eq!(ranged, 0, "its value range, warm scratch");
}

/// A tiled ZSTD frame whose height is not a whole number of tiles: its bottom
/// tiles are decompressed only as far as the image goes, which ZSTD's one-shot
/// call refuses, so they stream. Through a context kept across reads that
/// allocates nothing big once warm; a fresh streaming decoder per edge tile
/// was a context and a 128 KiB input buffer for every one, every frame. (Only
/// the buffer is the counter's to see: zstd takes its contexts from C's
/// `malloc`.)
#[cfg(feature = "codec-zstd")]
#[test]
fn a_warm_read_of_partial_zstd_tiles_allocates_nothing_big() {
    use fast_tiff_lib::{ByteOrder, FrameInfo, SampleFormat};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let (w, h, t) = (1024usize, 1000usize, 256usize);
    let value = |x: usize, y: usize| ((x * 3 + y * 5) % 65536) as u16;
    let mut file = vec![0u8; 16];
    let (mut offsets, mut counts) = (Vec::new(), Vec::new());
    for ty in 0..h.div_ceil(t) {
        for tx in 0..w.div_ceil(t) {
            let tile: Vec<u8> = (0..t * t)
                .flat_map(|i| value(tx * t + i % t, ty * t + i / t).to_le_bytes())
                .collect();
            let packed = zstd::stream::encode_all(&tile[..], 3).unwrap();
            offsets.push(file.len() as u64);
            counts.push(packed.len() as u64);
            file.extend_from_slice(&packed);
        }
    }
    let info = FrameInfo {
        width: w as u32,
        height: h as u32,
        bits_per_sample: 16,
        samples_per_pixel: 1,
        sample_format: SampleFormat::UnsignedInt,
        compression: Compression::Zstd,
        predictor: 1,
        photometric: 1,
        planar_config: 1,
        tile_size: Some((t as u32, t as u32)),
        ink_set: 1,
        strip_offsets: offsets.into(),
        strip_byte_counts: counts.into(),
        rows_per_strip: t as u32,
    };
    let frame = Frame::new(&file, &info, ByteOrder::Little);
    let units = Display { range: None };
    for parallel in [false, true] {
        let mut out = vec![0u16; w * h];
        let mut scratch = DecodeScratch::new();
        let (big, allowed) = big_allocations_warm(parallel, || {
            frame
                .read_into(units, 0, &mut [&mut out[..]], &mut scratch)
                .unwrap();
        });
        assert!(
            big <= allowed,
            "parallel {parallel}: {big} allocations of 64 KiB or more"
        );
        assert!(
            (0..w * h).all(|i| out[i] == value(i % w, i / w)),
            "parallel {parallel}: the pixels"
        );
    }
}

/// The C readers decode through a scratch the stack keeps, straight into the
/// caller's buffer, so a warm read allocates nothing frame-sized: not the
/// output, which is the caller's, nor the strips' decompressed bytes. Before,
/// each call decoded into planes of its own and copied them into the buffer:
/// an 8 MiB plane for every compressed 2048x2048 u16 frame, two (the frame
/// as stored, then as floats) for its f32 read, and a plane and a frame for
/// a 32-bit frame shown as u16.
#[cfg(feature = "capi")]
#[test]
fn a_warm_c_read_allocates_nothing_frame_sized() {
    use fast_tiff_lib::capi::*;
    use std::ptr::{null, null_mut};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let open = |st: SampleType, compression: Compression| {
        let opts = WriterOptions::new(2048, 2048, st)
            .compression(compression)
            .predictor(true)
            .rows_per_strip(64);
        let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
        let n = 2048 * 2048;
        let bytes: Vec<u8> = (0..n * st.bytes())
            .map(|i| ((i / st.bytes()) % 4093 + i % st.bytes() * 31) as u8)
            .collect();
        writer.write_frame_bytes(&bytes).unwrap();
        let file = writer.finish().unwrap().into_inner();
        let mut s = null_mut();
        assert_eq!(
            unsafe { ftl_open_bytes(file.as_ptr().cast(), file.len(), &mut s) },
            FTL_OK
        );
        s
    };
    let n = 2048 * 2048;
    let mut words = vec![0u16; n];
    let mut floats = vec![0f32; n];
    for compression in codecs() {
        let s = open(SampleType::U16, compression);
        for parallel in [false, true] {
            let (big, allowed) = big_allocations_warm(parallel, || {
                let status = unsafe { ftl_read_u16(s, 0, null(), words.as_mut_ptr(), n) };
                assert_eq!(status, FTL_OK);
            });
            assert!(
                big <= allowed,
                "u16 {compression:?}, parallel {parallel}: {big} allocations of 64 KiB or more"
            );
            let (big, allowed) = big_allocations_warm(parallel, || {
                let status = unsafe { ftl_read_f32(s, 0, floats.as_mut_ptr(), n) };
                assert_eq!(status, FTL_OK);
            });
            assert!(
                big <= allowed,
                "u16 {compression:?} as f32, parallel {parallel}: {big} allocations of 64 KiB \
                 or more"
            );
        }
        unsafe { ftl_stack_free(s) };
    }
    // 32-bit samples shown as u16 with no range are assembled and ranged in
    // the scratch, which a warm one already holds; finding the range alone
    // folds over the strips as they come.
    let s = open(SampleType::U32, Compression::Deflate);
    for parallel in [false, true] {
        let (big, allowed) = big_allocations_warm(parallel, || {
            let status = unsafe { ftl_read_u16(s, 0, null(), words.as_mut_ptr(), n) };
            assert_eq!(status, FTL_OK);
        });
        assert!(
            big <= allowed,
            "u32 Deflate as u16, parallel {parallel}: {big} allocations of 64 KiB or more"
        );
        let (big, allowed) = big_allocations_warm(parallel, || {
            let status = unsafe { ftl_float_minmax(s, 0, null_mut(), null_mut()) };
            assert_eq!(status, FTL_OK);
        });
        assert!(
            big <= allowed,
            "u32 Deflate's range, parallel {parallel}: {big} allocations of 64 KiB or more"
        );
    }
    unsafe { ftl_stack_free(s) };
}

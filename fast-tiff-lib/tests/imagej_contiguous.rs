//! ImageJ's >4 GiB escape hatch: a classic TIFF with ONE IFD, `images=N` in
//! the description, and frames 2..N appended as raw contiguous pixel data.
//! The reader must expand it into N virtual frames (this is what "opens but
//! scrubbing does nothing" looked like before). Files are hand-built here —
//! our own writer (correctly) never produces this layout.

use fast_tiff_lib::{read_frame_u16, TiffStack};
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn unique_temp_path(name: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("fast_tiff_ijcontig_{}_{}_{}", std::process::id(), n, name))
}

struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Build a little-endian classic TIFF the way ImageJ writes its big
/// contiguous stacks: header, `frames_in_file` raw contiguous 2x2 u16 frames,
/// then a single IFD whose description claims `images=` `images_claimed`.
/// Frame i's pixels are `[i*100, i*100+1, i*100+2, i*100+3]`.
fn build_imagej_contiguous(images_claimed: usize, frames_in_file: usize) -> Vec<u8> {
    let (w, h) = (2u32, 2u32);
    let frame_bytes = (w * h * 2) as usize;

    let mut file = Vec::new();
    file.extend_from_slice(b"II");
    file.extend_from_slice(&42u16.to_le_bytes());
    let header_ifd_ptr = file.len(); // patched below
    file.extend_from_slice(&0u32.to_le_bytes());

    let data_start = file.len() as u32; // 8
    for i in 0..frames_in_file {
        for p in 0..4u16 {
            file.extend_from_slice(&((i as u16) * 100 + p).to_le_bytes());
        }
    }

    // External data for the description, NUL-terminated, padded to even.
    let desc = format!("ImageJ=1.54f\nimages={images_claimed}\n\0");
    let desc_offset = file.len() as u32;
    file.extend_from_slice(desc.as_bytes());
    if file.len() % 2 == 1 {
        file.push(0);
    }

    // The single IFD, entries in ascending tag order.
    let ifd_offset = file.len() as u32;
    let entry =
        |tag: u16, ftype: u16, count: u32, value: u32| -> Vec<u8> {
            let mut e = Vec::with_capacity(12);
            e.extend_from_slice(&tag.to_le_bytes());
            e.extend_from_slice(&ftype.to_le_bytes());
            e.extend_from_slice(&count.to_le_bytes());
            e.extend_from_slice(&value.to_le_bytes());
            e
        };
    let entries = [
        entry(256, 4, 1, w),                              // ImageWidth
        entry(257, 4, 1, h),                              // ImageLength
        entry(258, 3, 1, 16),                             // BitsPerSample
        entry(259, 3, 1, 1),                              // Compression: none
        entry(262, 3, 1, 1),                              // Photometric: BlackIsZero
        entry(270, 2, desc.len() as u32, desc_offset),    // ImageDescription
        entry(273, 4, 1, data_start),                     // StripOffsets
        entry(277, 3, 1, 1),                              // SamplesPerPixel
        entry(278, 4, 1, h),                              // RowsPerStrip
        entry(279, 4, 1, frame_bytes as u32),             // StripByteCounts (frame 0 only)
        entry(339, 3, 1, 1),                              // SampleFormat: unsigned
    ];
    file.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for e in &entries {
        file.extend_from_slice(e);
    }
    file.extend_from_slice(&0u32.to_le_bytes()); // next IFD: none

    file[header_ifd_ptr..header_ifd_ptr + 4].copy_from_slice(&ifd_offset.to_le_bytes());
    file
}

#[test]
fn single_ifd_contiguous_stack_expands_to_virtual_frames() {
    let path = unique_temp_path("expand.tif");
    let _cleanup = Cleanup(path.clone());
    std::fs::write(&path, build_imagej_contiguous(3, 3)).unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames.len(), 3, "images=3 must expand to 3 virtual frames");
    assert_eq!(stack.meta.frames, 3);
    for i in 0..3u16 {
        let got = read_frame_u16(&stack.mmap, &stack.frames[i as usize], stack.byte_order, None).unwrap();
        assert_eq!(got.as_ref(), &[i * 100, i * 100 + 1, i * 100 + 2, i * 100 + 3]);
        // Virtual frames are single uncompressed native strips: zero-copy.
        #[cfg(target_endian = "little")]
        assert!(matches!(got, Cow::Borrowed(_)), "virtual frame {i} should be zero-copy");
    }
}

#[test]
fn truncated_contiguous_stack_clamps_to_available_frames() {
    // The description claims 5 frames but only 3 fit in the file (truncated
    // writes exist in the wild) — clamp instead of reading out of bounds.
    let path = unique_temp_path("clamped.tif");
    let _cleanup = Cleanup(path.clone());
    std::fs::write(&path, build_imagej_contiguous(5, 3)).unwrap();

    let stack = TiffStack::open(&path).unwrap();
    // The IFD region sits after frame 3's data, so its bytes are counted as
    // "available" space — but never more frames than the description claims,
    // and every synthesized frame must decode within the file.
    assert!(stack.frames.len() >= 3 && stack.frames.len() <= 5);
    for f in &stack.frames {
        assert!(read_frame_u16(&stack.mmap, f, stack.byte_order, None).is_ok());
    }
    let got = read_frame_u16(&stack.mmap, &stack.frames[2], stack.byte_order, None).unwrap();
    assert_eq!(got.as_ref(), &[200, 201, 202, 203]);
}

#[test]
fn multi_ifd_files_are_not_touched_by_the_expansion() {
    // A normal 2-IFD file whose description happens to say images=2 must
    // keep its real IFDs (expansion only applies to the single-IFD layout).
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let path = unique_temp_path("normal.tif");
    let _cleanup = Cleanup(path.clone());
    let opts = WriterOptions::new(2, 2, SampleType::U16).description("ImageJ=1.54f\nimages=2\n");
    let mut w = TiffWriter::create(&path, opts).unwrap();
    w.write_frame_u16(&[1, 2, 3, 4]).unwrap();
    w.write_frame_u16(&[5, 6, 7, 8]).unwrap();
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames.len(), 2);
    let got = read_frame_u16(&stack.mmap, &stack.frames[1], stack.byte_order, None).unwrap();
    assert_eq!(got.as_ref(), &[5, 6, 7, 8]);
}

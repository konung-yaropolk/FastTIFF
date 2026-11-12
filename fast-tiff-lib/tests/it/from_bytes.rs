//! `TiffStack::from_bytes` — the filesystem-free entry point.
//!
//! Deliberately has no `#![cfg(feature = "mmap")]` gate: this is the path a
//! wasm host takes, so it must keep working under `--no-default-features`.
//! Everything here writes a stack in memory and reads it straight back, with no
//! temp file anywhere.

use fast_tiff_lib::{read_frame_u16, Bytes, Compression, SampleType, TiffStack, WriterOptions};
use std::io::Cursor;
use std::sync::Arc;

/// Write `frames` of 16-bit pixels into an in-memory TIFF.
fn write(width: u32, height: u32, frames: &[Vec<u16>], compression: Compression) -> Vec<u8> {
    let opts = WriterOptions::new(width, height, SampleType::U16).compression(compression);
    let mut w = TiffWriterAlias::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in frames {
        let bytes: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_frame_bytes(&bytes).unwrap();
    }
    w.finish().unwrap().into_inner()
}
use fast_tiff_lib::TiffWriter as TiffWriterAlias;

#[test]
fn round_trips_a_multi_frame_stack_with_no_filesystem() {
    let (w, h) = (7u32, 5u32);
    let frames: Vec<Vec<u16>> = (0..3)
        .map(|f| (0..w * h).map(|i| (i as u16 + f * 100) % 4096).collect())
        .collect();
    let bytes = write(w, h, &frames, Compression::None);

    let stack = TiffStack::from_bytes(bytes).expect("from_bytes should index the stack");
    assert_eq!(stack.frames.len(), 3);
    assert!(!stack.data.is_empty(), "the backing bytes are retained");

    for (i, want) in frames.iter().enumerate() {
        let got = read_frame_u16(&stack.data, &stack.frames[i], stack.byte_order, None).unwrap();
        assert_eq!(got.as_ref(), &want[..], "frame {i}");
    }
}

/// LZW and Deflate are pure Rust, so they must survive a build with every
/// optional codec off — which is exactly the wasm configuration.
#[test]
fn pure_rust_codecs_decode_from_bytes() {
    let (w, h) = (9u32, 4u32);
    let pixels: Vec<u16> = (0..w * h).map(|i| (i as u16 * 61) % 900).collect();
    for compression in [
        Compression::None,
        Compression::Lzw,
        Compression::Deflate,
        Compression::PackBits,
    ] {
        let bytes = write(w, h, std::slice::from_ref(&pixels), compression);
        let stack = TiffStack::from_bytes(bytes).unwrap();
        let got = read_frame_u16(&stack.data, &stack.frames[0], stack.byte_order, None).unwrap();
        assert_eq!(got.as_ref(), &pixels[..], "{compression:?}");
    }
}

/// A buffer handed over behind an `Arc` is held, not copied — the browser's
/// dropped file arrives that way, and opening it is then no whole-file memcpy
/// — and the stack it becomes reads exactly as one made from an owned `Vec` of
/// the same bytes.
#[test]
fn a_shared_buffer_is_held_rather_than_copied() {
    let (w, h) = (6u32, 4u32);
    let frames: Vec<Vec<u16>> = (0..3)
        .map(|f| {
            (0..w * h)
                .map(|i| (i as u16 * 37 + f * 11) % 2048)
                .collect()
        })
        .collect();
    let bytes = write(w, h, &frames, Compression::Lzw);

    let owned = TiffStack::from_bytes(bytes.clone()).unwrap();
    let shared: Arc<[u8]> = Arc::from(bytes);
    let stack = TiffStack::from_bytes(Arc::clone(&shared)).unwrap();

    match &stack.data {
        Bytes::Shared(held) => assert!(Arc::ptr_eq(held, &shared), "a different buffer"),
        other => panic!("expected the shared buffer itself, got {other:?}"),
    }
    assert_eq!(
        Arc::strong_count(&shared),
        2,
        "the stack holds one reference"
    );
    assert_eq!(
        stack.data.as_ptr(),
        shared.as_ptr(),
        "the bytes were copied"
    );

    assert_eq!(stack.frames.len(), owned.frames.len());
    for (i, want) in frames.iter().enumerate() {
        let from_shared =
            read_frame_u16(&stack.data, &stack.frames[i], stack.byte_order, None).unwrap();
        let from_owned =
            read_frame_u16(&owned.data, &owned.frames[i], owned.byte_order, None).unwrap();
        assert_eq!(from_shared.as_ref(), &want[..], "frame {i}");
        assert_eq!(from_shared.as_ref(), from_owned.as_ref(), "frame {i}");
    }

    // And the reference goes when the stack does.
    drop(stack);
    assert_eq!(Arc::strong_count(&shared), 1);
}

#[test]
fn rejects_garbage_without_panicking() {
    assert!(TiffStack::from_bytes(vec![0u8; 16]).is_err());
    assert!(TiffStack::from_bytes(Vec::new()).is_err());
}

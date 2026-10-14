//! JPEG inside DICOM, which is two unrelated formats under one name.
//!
//! The marker that starts the frame says which:
//!
//! * **`SOF0`/`SOF1`** — baseline and extended sequential DCT. The JPEG
//!   everyone means by JPEG, lossy, and what `1.2.840.10008.1.2.4.50` carries.
//!   Decoded by `zune-jpeg`.
//! * **`SOF3`** — *lossless*, and nothing like the others: no DCT, no
//!   quantisation, no blocks. Each sample is predicted from its neighbours and
//!   only the difference is Huffman-coded. This is what
//!   `1.2.840.10008.1.2.4.57` and `.70` carry, and it is the compression a
//!   hospital archive is actually full of — a CT series exported from a PACS is
//!   usually this. No crate on crates.io decodes it, so [`super::lossless`]
//!   does.
//!
//! Dispatching on the marker rather than on the transfer syntax UID is
//! deliberate. The UID says which the file is *supposed* to be, and the two
//! lossless UIDs in particular are routinely interchanged by converters that
//! rewrite one and not the other. The bitstream cannot be wrong about itself.

use super::pixels::Format;
use fasttiff_plugin_api::PluginError;

/// Decode one frame's JPEG into native interleaved little-endian samples.
pub(super) fn decode(fragment: &[u8], fmt: Format) -> Result<Vec<u8>, PluginError> {
    match start_of_frame(fragment)? {
        // Lossless, Huffman. The one that matters here.
        0xC3 => super::lossless::decode(fragment, fmt),
        // Baseline and extended sequential.
        0xC0 | 0xC1 => baseline(fragment, fmt),
        // The rest are named rather than attempted, because each is a different
        // algorithm and a decoder handed the wrong one does not fail cleanly —
        // it produces an image.
        0xC2 => Err(PluginError::unsupported(
            "this DICOM file's JPEG is progressive, which no DICOM transfer \
             syntax allows and nothing here decodes",
        )),
        0xC5..=0xC7 => Err(PluginError::unsupported(
            "this DICOM file's JPEG is hierarchical, which nothing here decodes",
        )),
        0xC9..=0xCB | 0xCD..=0xCF => Err(PluginError::unsupported(
            "this DICOM file's JPEG is arithmetic-coded rather than Huffman-\
             coded, which nothing here decodes. `gdcmconv --raw` converts it",
        )),
        other => Err(PluginError::unsupported(format!(
            "this DICOM file's JPEG starts its frame with marker FF{other:02X}, \
             which is not one this reads"
        ))),
    }
}

/// The `SOF` marker byte that starts this stream's frame.
///
/// Walked rather than assumed at a fixed offset: a stream may carry any number
/// of `APP`, `COM` and `DQT`/`DHT` segments before its `SOF`, and medical
/// encoders are fond of `APP` blocks.
fn start_of_frame(b: &[u8]) -> Result<u8, PluginError> {
    let mut i = 0usize;
    // Skip to SOI. A fragment should start with it, but an encoder that left a
    // stray byte in front is not worth failing over.
    while i + 1 < b.len() && !(b[i] == 0xFF && b[i + 1] == 0xD8) {
        i += 1;
    }
    if i + 1 >= b.len() {
        return Err(PluginError::failed(
            "this DICOM frame's compressed data has no JPEG start-of-image \
             marker, so it is not a JPEG stream",
        ));
    }
    i += 2;
    loop {
        // Markers are `FF xx`, and any number of `FF` fill bytes may precede
        // the `xx`.
        while i < b.len() && b[i] != 0xFF {
            i += 1;
        }
        while i < b.len() && b[i] == 0xFF {
            i += 1;
        }
        let Some(&marker) = b.get(i) else {
            return Err(PluginError::failed(
                "this DICOM frame's JPEG stream ends before it says what kind \
                 of JPEG it is",
            ));
        };
        i += 1;
        // Every `SOFn` is `FFC0..FFCF` except `FFC4` (a Huffman table),
        // `FFC8` (reserved) and `FFCC` (arithmetic conditioning).
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            return Ok(marker);
        }
        // Standalone markers carry no length; everything else does.
        if matches!(marker, 0x01 | 0xD0..=0xD9) {
            continue;
        }
        let len = b
            .get(i..i + 2)
            .map(|s| u16::from_be_bytes([s[0], s[1]]) as usize)
            .ok_or_else(|| PluginError::failed("this DICOM frame's JPEG stream is truncated"))?;
        // The length counts its own two bytes.
        i += len.max(2);
    }
}

/// Baseline and extended sequential DCT, through `zune-jpeg`.
fn baseline(fragment: &[u8], fmt: Format) -> Result<Vec<u8>, PluginError> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    // The output colourspace is *asked for*, not discovered. Left to itself the
    // decoder converts to RGB, which for the greyscale frames that most of
    // medical JPEG is means three identical channels where the DICOM header
    // says one sample — a disagreement the reader then has to either refuse or
    // paper over. Naming it removes the question, and the check below becomes a
    // confirmation rather than a negotiation.
    let want = if fmt.samples == 1 {
        ColorSpace::Luma
    } else {
        ColorSpace::RGB
    };
    // Through a `Cursor`: the decoder's reader trait wants `Seek`, which a bare
    // slice does not implement.
    let mut dec = zune_jpeg::JpegDecoder::new_with_options(
        std::io::Cursor::new(fragment),
        DecoderOptions::default().jpeg_set_out_colorspace(want),
    );
    let pixels = dec
        .decode()
        .map_err(|e| PluginError::failed(format!("decoding this frame's JPEG: {e}")))?;
    let info = dec
        .info()
        .ok_or_else(|| PluginError::failed("this frame's JPEG declared no dimensions"))?;

    // The decoder's own idea of the picture has to match the dataset's, or one
    // of the two is describing a different image — and the dataset is what the
    // slice geometry and the series assembly were built from.
    if u32::from(info.width) != fmt.columns || u32::from(info.height) != fmt.rows {
        return Err(PluginError::failed(format!(
            "this frame's JPEG is {}x{} where the DICOM header says {}x{}",
            info.width, info.height, fmt.columns, fmt.rows
        )));
    }
    let got = dec
        .output_colorspace()
        .map_or(0, |c: ColorSpace| c.num_components());
    if got != fmt.samples {
        return Err(PluginError::failed(format!(
            "this frame's JPEG decodes to {got} component(s) where the DICOM \
             header says {} sample(s) per pixel",
            fmt.samples
        )));
    }
    // Baseline JPEG is 8-bit, and a DICOM carrying it says so. A 16-bit header
    // over a baseline stream is a file disagreeing with itself.
    if fmt.bits_allocated != 8 {
        return Err(PluginError::failed(format!(
            "this frame is baseline JPEG, which is 8-bit, but its DICOM header \
             allocates {} bits per sample",
            fmt.bits_allocated
        )));
    }
    Ok(pixels)
}

#[cfg(test)]
#[path = "jpeg_tests.rs"]
mod tests;

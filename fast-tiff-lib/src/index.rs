//! Walks the full IFD chain of a TIFF file and builds a per-frame index.
//! Each IFD in the chain is treated as one "frame" (one plane: one channel
//! at one Z/T position), matching how ImageJ writes hyperstacks (one IFD
//! per plane, in `xyczt` order by default). This is a generic multi-page
//! TIFF walker, not an ImageJ-specific format — it doesn't assume anything
//! about how many IFDs there are or what writer produced them.

use crate::ifd::{self, ByteOrder, RawIfdEntry};
use crate::ij_metadata::{self, StackMeta};
use anyhow::{anyhow, bail, Result};
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

// `pub(crate)`: shared with the write side (`encode`), so reader and writer
// can never disagree on tag numbers.
pub(crate) const TAG_IMAGE_WIDTH: u16 = 256;
pub(crate) const TAG_IMAGE_LENGTH: u16 = 257;
pub(crate) const TAG_BITS_PER_SAMPLE: u16 = 258;
pub(crate) const TAG_COMPRESSION: u16 = 259;
pub(crate) const TAG_PHOTOMETRIC: u16 = 262;
pub(crate) const TAG_IMAGE_DESCRIPTION: u16 = 270;
pub(crate) const TAG_STRIP_OFFSETS: u16 = 273;
pub(crate) const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub(crate) const TAG_ROWS_PER_STRIP: u16 = 278;
pub(crate) const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub(crate) const TAG_PREDICTOR: u16 = 317;
pub(crate) const TAG_PLANAR_CONFIG: u16 = 284;
pub(crate) const TAG_SAMPLE_FORMAT: u16 = 339;
// Tags 50838/50839 (IJMetadataByteCounts / IJMetadata) carry ImageJ's binary
// per-channel LUT/range block. The format is undocumented and best-effort to
// parse, so it's used only as a supplementary fallback for display info the
// `ImageDescription` (tag 270) didn't provide — see `ij_metadata`.
const TAG_IJ_METADATA_BYTE_COUNTS: u16 = 50838;
const TAG_IJ_METADATA: u16 = 50839;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    UnsignedInt,
    SignedInt,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    Lzw,
    PackBits,
    Deflate,
    Other(u16),
}

/// Everything needed to locate and decode one plane (one IFD) in the file.
#[derive(Clone, Debug)]
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
    pub samples_per_pixel: u16,
    pub sample_format: SampleFormat,
    pub compression: Compression,
    pub predictor: u16, // 1 = none, 2 = horizontal differencing
    /// PhotometricInterpretation (tag 262): 2 = RGB, others treated as a
    /// single-plane grayscale/whatever. Used to decide whether a frame's
    /// multiple samples are color components to deinterleave.
    pub photometric: u16,
    /// PlanarConfiguration (tag 284): 1 = chunky (samples interleaved per
    /// pixel, the default and only layout we deinterleave), 2 = planar
    /// (separate sample planes — not supported for multi-sample data).
    pub planar_config: u16,
    pub strip_offsets: Vec<u64>,
    pub strip_byte_counts: Vec<u64>,
    pub rows_per_strip: u32,
}

impl FrameInfo {
    /// True for a chunky (interleaved) RGB frame whose 3+ samples are color
    /// components we can deinterleave into red/green/blue planes.
    pub fn is_rgb(&self) -> bool {
        self.photometric == 2 && self.samples_per_pixel >= 3 && self.planar_config == 1
    }
}

pub struct TiffStack {
    pub mmap: Mmap,
    pub byte_order: ByteOrder,
    pub frames: Vec<FrameInfo>,
    pub meta: StackMeta,
}

impl TiffStack {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| anyhow!("could not open {}: {e}", path.as_ref().display()))?;
        // SAFETY: standard caveat of memmap2 — the file must not be mutated
        // out from under us while mapped. We open read-only and treat the
        // mapping as immutable for the lifetime of the TiffStack.
        let mmap = unsafe { Mmap::map(&file)? };

        let (order, first_ifd) = ifd::read_header(&mmap)?;

        let mut frames = Vec::new();
        let mut description: Option<String> = None;
        let mut ij_metadata_bytes: Option<Vec<u8>> = None;
        let mut ij_metadata_counts: Option<Vec<u32>> = None;

        let mut offset = first_ifd as usize;
        let mut visited = HashSet::new();
        let mut first = true;

        while offset != 0 {
            if !visited.insert(offset) {
                bail!("malformed TIFF: IFD chain loops back to offset {offset}");
            }
            let parsed = ifd::read_ifd(&mmap, offset, order)?;
            let frame = frame_info_from_entries(&parsed.entries, &mmap, order)?;

            if first {
                for e in &parsed.entries {
                    match e.tag {
                        TAG_IMAGE_DESCRIPTION => {
                            description = e.as_ascii(&mmap, order).ok();
                        }
                        TAG_IJ_METADATA => {
                            ij_metadata_bytes = e.owned_bytes(&mmap, order).ok();
                        }
                        TAG_IJ_METADATA_BYTE_COUNTS => {
                            ij_metadata_counts = e.as_u32_array(&mmap, order).ok();
                        }
                        _ => {}
                    }
                }
                first = false;
            }

            frames.push(frame);
            offset = parsed.next_offset as usize;
        }

        if frames.is_empty() {
            bail!("TIFF has no image directories");
        }

        // Every frame in the stack must share the first frame's geometry and
        // pixel layout. The viewer uploads every frame into one fixed-size GPU
        // texture (sized to frame 0) and decodes with a single stride, so a
        // differently-shaped frame — e.g. the reduced-resolution levels of a
        // pyramidal TIFF, or an appended thumbnail page — would otherwise be
        // silently mis-rendered. Catch it here with a clear error instead.
        let f0 = &frames[0];
        let f0_shape = (f0.width, f0.height, f0.bits_per_sample, f0.samples_per_pixel);
        if let Some((i, f)) = frames.iter().enumerate().find(|(_, f)| {
            (f.width, f.height, f.bits_per_sample, f.samples_per_pixel) != f0_shape
        }) {
            bail!(
                "TIFF frames are not uniform: frame 0 is {}x{} ({}-bit, {} sample(s)/px) but \
                 frame {} is {}x{} ({}-bit, {} sample(s)/px). This looks like a pyramidal or \
                 mixed-size TIFF, which this stack viewer doesn't support.",
                f0.width,
                f0.height,
                f0.bits_per_sample,
                f0.samples_per_pixel,
                i,
                f.width,
                f.height,
                f.bits_per_sample,
                f.samples_per_pixel,
            );
        }

        let meta = ij_metadata::build_stack_meta(
            description.as_deref(),
            ij_metadata_bytes.as_deref(),
            ij_metadata_counts.as_deref(),
            frames.len(),
        );

        Ok(TiffStack {
            mmap,
            byte_order: order,
            frames,
            meta,
        })
    }
}

fn frame_info_from_entries(
    entries: &[RawIfdEntry],
    file: &[u8],
    order: ByteOrder,
) -> Result<FrameInfo> {
    let mut width = None;
    let mut height = None;
    let mut bits_per_sample = 16u16; // baseline TIFF default if tag absent
    let mut samples_per_pixel = 1u16; // default per spec
    let mut sample_format_raw = 1u16; // default: unsigned integer
    let mut compression_raw = 1u16; // default: no compression
    let mut predictor = 1u16;
    let mut photometric = 1u16; // default: BlackIsZero grayscale
    let mut planar_config = 1u16; // default: chunky / interleaved
    let mut rows_per_strip = u32::MAX; // default: whole image is one strip
    let mut strip_offsets = None;
    let mut strip_byte_counts = None;

    for e in entries {
        match e.tag {
            TAG_IMAGE_WIDTH => width = Some(e.as_u32(file, order)?),
            TAG_IMAGE_LENGTH => height = Some(e.as_u32(file, order)?),
            TAG_BITS_PER_SAMPLE => bits_per_sample = e.as_u32(file, order)? as u16,
            TAG_SAMPLES_PER_PIXEL => samples_per_pixel = e.as_u32(file, order)? as u16,
            TAG_SAMPLE_FORMAT => sample_format_raw = e.as_u32(file, order)? as u16,
            TAG_COMPRESSION => compression_raw = e.as_u32(file, order)? as u16,
            TAG_PREDICTOR => predictor = e.as_u32(file, order)? as u16,
            TAG_ROWS_PER_STRIP => rows_per_strip = e.as_u32(file, order)?,
            TAG_STRIP_OFFSETS => strip_offsets = Some(e.as_u32_array(file, order)?),
            TAG_STRIP_BYTE_COUNTS => strip_byte_counts = Some(e.as_u32_array(file, order)?),
            TAG_PHOTOMETRIC => photometric = e.as_u32(file, order)? as u16,
            TAG_PLANAR_CONFIG => planar_config = e.as_u32(file, order)? as u16,
            _ => {}
        }
    }

    let width = width.ok_or_else(|| anyhow!("IFD missing ImageWidth"))?;
    let height = height.ok_or_else(|| anyhow!("IFD missing ImageLength"))?;
    let strip_offsets =
        strip_offsets.ok_or_else(|| anyhow!("IFD missing StripOffsets (tiled TIFFs not supported)"))?;
    let strip_byte_counts =
        strip_byte_counts.ok_or_else(|| anyhow!("IFD missing StripByteCounts"))?;

    if rows_per_strip == u32::MAX {
        rows_per_strip = height;
    }

    let sample_format = match sample_format_raw {
        2 => SampleFormat::SignedInt,
        3 => SampleFormat::Float,
        _ => SampleFormat::UnsignedInt,
    };
    let compression = match compression_raw {
        1 => Compression::None,
        5 => Compression::Lzw,
        32773 => Compression::PackBits,
        8 | 32946 => Compression::Deflate,
        other => Compression::Other(other),
    };

    Ok(FrameInfo {
        width,
        height,
        bits_per_sample,
        samples_per_pixel,
        sample_format,
        compression,
        predictor,
        photometric,
        planar_config,
        strip_offsets: strip_offsets.into_iter().map(|v| v as u64).collect(),
        strip_byte_counts: strip_byte_counts.into_iter().map(|v| v as u64).collect(),
        rows_per_strip,
    })
}

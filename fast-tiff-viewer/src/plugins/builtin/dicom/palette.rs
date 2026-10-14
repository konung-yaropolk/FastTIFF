//! Palette-colour DICOM: pixels that are indices, and three lookup tables.
//!
//! An ultrasound cine loop or a colour-flow Doppler frame is usually stored
//! this way — one byte per pixel into a 256-entry table — and without the table
//! the pixels are not an image at all. They are not even a *greyscale* image:
//! the indices of a Doppler palette run through blue, red and green in whatever
//! order the vendor chose, so showing them as intensities produces a picture
//! made of arbitrary numbers.
//!
//! So the table is applied here, and the result is three channels of colour.
//! That is more work than refusing, and it is what makes such a file openable
//! rather than merely diagnosable.
//!
//! # Why the host cannot hold the palette itself
//!
//! It could — the viewer has a palette path of its own, with a window pinned to
//! an index-to-LUT identity, which is how it opens a palette TIFF. But an
//! [`ImageResult`](fasttiff_plugin_api::ImageResult) has nowhere to put a
//! colour table: the contract carries planes, a pixel type and channel
//! *colours*, not channel LUTs. Expanding to RGB is therefore the only thing a
//! plugin can do, and it is also what `gdcmconv --raw` does, so the result
//! matches what the conversion everyone already runs would have produced.

use super::parse::{Found, Tag};
use super::pixels::Format;
use fasttiff_plugin_api::{PixelType, PlaneData, PluginError};

const RED_DESCRIPTOR: Tag = (0x0028, 0x1101);
const GREEN_DESCRIPTOR: Tag = (0x0028, 0x1102);
const BLUE_DESCRIPTOR: Tag = (0x0028, 0x1103);
const RED_DATA: Tag = (0x0028, 0x1201);
const GREEN_DATA: Tag = (0x0028, 0x1202);
const BLUE_DATA: Tag = (0x0028, 0x1203);
/// The segmented form, which is run-length-coded rather than a plain table.
const RED_SEGMENTED: Tag = (0x0028, 0x1221);

/// Every tag a palette needs, for the reader's one shared want-list.
pub(super) const TAGS: [Tag; 7] = [
    RED_DESCRIPTOR,
    GREEN_DESCRIPTOR,
    BLUE_DESCRIPTOR,
    RED_DATA,
    GREEN_DATA,
    BLUE_DATA,
    RED_SEGMENTED,
];

/// Three lookup tables and the index they start at.
///
/// `Debug` so a test can assert on a refusal; the tables are at most three
/// times 65536 numbers, which is a lot to print and nothing to hide.
#[derive(Debug)]
pub(super) struct Palette {
    channels: [Vec<u16>; 3],
    /// The first index the tables describe. Indices below it take the first
    /// entry, above the last take the last — which is the standard's own
    /// clamping rule and not a convenience.
    first: i32,
    /// 8 or 16, from the descriptor. Decides the output's sample width.
    bits: u16,
}

impl Palette {
    /// What the host should call the expanded samples.
    pub(super) fn pixel_type(&self) -> PixelType {
        if self.bits > 8 {
            PixelType::U16
        } else {
            PixelType::U8
        }
    }

    /// Read the tables, or say why they cannot be read.
    pub(super) fn of(f: &Found, big_endian: bool) -> Result<Self, PluginError> {
        if f.raw(RED_SEGMENTED).is_some() {
            return Err(PluginError::unsupported(
                "this DICOM file's colour table is in the segmented form, which \
                 is run-length-coded rather than a plain table and is not \
                 expanded here. `gdcmconv --raw` converts the file",
            ));
        }
        let mut channels = [Vec::new(), Vec::new(), Vec::new()];
        let mut first = 0i32;
        let mut bits = 0u16;
        for (i, (desc, data)) in [
            (RED_DESCRIPTOR, RED_DATA),
            (GREEN_DESCRIPTOR, GREEN_DATA),
            (BLUE_DESCRIPTOR, BLUE_DATA),
        ]
        .into_iter()
        .enumerate()
        {
            let raw = f.raw(desc).ok_or_else(|| {
                PluginError::failed(
                    "this DICOM file is palette-colour but does not carry the \
                     descriptors of its colour table",
                )
            })?;
            let word = |n: usize| -> Option<u16> {
                let c: [u8; 2] = raw.get(n * 2..n * 2 + 2)?.try_into().ok()?;
                Some(if big_endian {
                    u16::from_be_bytes(c)
                } else {
                    u16::from_le_bytes(c)
                })
            };
            let (count, start, per) = (word(0), word(1), word(2));
            let (Some(count), Some(start), Some(per)) = (count, start, per) else {
                return Err(PluginError::failed(
                    "this DICOM file's colour table descriptor is not the three \
                     values the format requires",
                ));
            };
            // A count of zero means 65536 — the one place DICOM uses zero for a
            // full-range value, because 65536 does not fit the `US` it is
            // stored in.
            let count = if count == 0 {
                65536usize
            } else {
                count as usize
            };
            if !matches!(per, 8 | 16) {
                return Err(PluginError::failed(format!(
                    "this DICOM file's colour table declares {per} bits per \
                     entry, and the format allows 8 or 16"
                )));
            }
            let bytes = f.raw(data).ok_or_else(|| {
                PluginError::failed(
                    "this DICOM file is palette-colour but one of its colour \
                     tables has no data",
                )
            })?;
            // 16-bit entries are two bytes each. 8-bit entries are one — and
            // the length is what settles the ambiguity the standard leaves
            // here, because an 8-bit table may be declared as `OW` and arrive
            // padded. Trusting the declared width over the length it actually
            // has would read half a table and stretch it over the image.
            let entries: Vec<u16> = if bytes.len() >= count * 2 {
                bytes
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| {
                        let c = [c[0], c[1]];
                        if big_endian {
                            u16::from_be_bytes(c)
                        } else {
                            u16::from_le_bytes(c)
                        }
                    })
                    .collect()
            } else {
                bytes.iter().take(count).map(|&b| b as u16).collect()
            };
            if entries.is_empty() {
                return Err(PluginError::failed(
                    "this DICOM file's colour table is empty",
                ));
            }
            channels[i] = entries;
            first = start as i32;
            bits = per;
        }
        Ok(Palette {
            channels,
            first,
            bits,
        })
    }

    /// One index through the tables.
    fn look_up(&self, index: i32) -> [u16; 3] {
        let mut out = [0u16; 3];
        for (c, table) in self.channels.iter().enumerate() {
            // Clamped at both ends, per the standard: an index below the first
            // mapped value takes the first entry and one above the last takes
            // the last. Wrapping instead — which an unchecked cast would do —
            // paints the out-of-range pixels with whatever colour happens to
            // sit at the other end of the table.
            let at = (index - self.first).clamp(0, table.len() as i32 - 1) as usize;
            out[c] = table[at];
        }
        out
    }

    /// Expand one frame of indices into three planes of colour.
    ///
    /// `indices` is the frame's pixel data as stored — one sample per pixel,
    /// which for a palette image is the index.
    pub(super) fn expand(
        &self,
        indices: &[u8],
        fmt: Format,
        frame: usize,
    ) -> Result<Vec<PlaneData>, PluginError> {
        let n_px = (fmt.rows as usize) * (fmt.columns as usize);
        let word = fmt.sample_bytes();
        let at = n_px
            .checked_mul(word)
            .and_then(|stride| stride.checked_mul(frame))
            .ok_or_else(|| PluginError::failed("the DICOM frame offset overflows"))?;
        let raw = indices.get(at..at + n_px * word).ok_or_else(|| {
            PluginError::failed("the DICOM pixel data ends in the middle of a frame")
        })?;

        let index = |p: usize| -> i32 {
            match word {
                1 => raw[p] as i32,
                _ => u16::from_le_bytes([raw[2 * p], raw[2 * p + 1]]) as i32,
            }
        };
        let wide = self.bits > 8;
        let mut out = Vec::with_capacity(3);
        for c in 0..3 {
            out.push(if wide {
                PlaneData::U16((0..n_px).map(|p| self.look_up(index(p))[c]).collect())
            } else {
                PlaneData::U8((0..n_px).map(|p| self.look_up(index(p))[c] as u8).collect())
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "palette_tests.rs"]
mod tests;

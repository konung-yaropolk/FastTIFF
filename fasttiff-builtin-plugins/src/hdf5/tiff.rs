//! Use the same TIFF decoder as a directly opened file for embedded payloads.
use super::{failed, progress};
use fast_tiff_lib::{SampleFormat, TiffStack};
use fasttiff_plugin_api::*;

// Read only the header and IFD of ordinary TIFF byte vectors. The IFD can
// be at the far end of the dataset.
fn plain_frame(ds: &hdf5_reader::Dataset) -> Option<super::FrameShape> {
    use fast_tiff_lib::{ifd, TiffFlavor};
    use hdf5_reader::{SliceInfo, SliceInfoElem};
    let &[len] = ds.shape() else { return None };
    if len < 16 {
        return None;
    }
    let range = |start: u64, size: u64| {
        let end = start.checked_add(size)?;
        if end > len {
            return None;
        }
        ds.read_slice::<u8>(&SliceInfo {
            selections: vec![SliceInfoElem::Slice {
                start,
                end,
                step: 1,
            }],
        })
        .ok()
    };
    let header = range(0, 16)?;
    let (order, flavor, at) = ifd::read_header(header.as_slice()?).ok()?;
    let (prefix, entry, suffix) = match flavor {
        TiffFlavor::Classic => (2, 12, 4),
        TiffFlavor::Big => (8, 20, 8),
    };
    let count = range(at, prefix)?;
    let count = match flavor {
        TiffFlavor::Classic => u64::from(order.u16(count.as_slice()?)),
        TiffFlavor::Big => order.u64(count.as_slice()?),
    };
    let directory = range(at, count.checked_mul(entry)?.checked_add(prefix + suffix)?)?;
    let tags = ifd::read_ifd(directory.as_slice()?, 0, order, flavor).ok()?;
    // ImageJ can describe a contiguous stack in tag 270 with just one IFD.
    // Let the regular indexer resolve that and multi-page channel layouts.
    if tags.next_offset != 0 || tags.entries.iter().any(|e| e.tag == 270) {
        return None;
    }
    let scalar = |tag| {
        tags.entries
            .iter()
            .find(|e| e.tag == tag && e.count == 1)?
            .as_u32(&[], order)
            .ok()
    };
    Some((
        scalar(256)?,
        scalar(257)?,
        scalar(277).unwrap_or(1).max(1) as usize,
    ))
}

pub(super) fn single_frame(ds: &hdf5_reader::Dataset) -> Option<super::FrameShape> {
    plain_frame(ds).or_else(|| {
        // Opaque HDF5 scalars are only readable as a whole. Index them without
        // decoding planes, also the fallback for nontrivial TIFF directories.
        let stack = TiffStack::from_bytes(ds.read_raw_bytes().ok()?).ok()?;
        let first = stack.frames.first()?;
        let (channels, slices, frames) = dimensions(&stack);
        (slices == 1 && frames == 1).then_some((first.width, first.height, channels))
    })
}

fn dimensions(stack: &TiffStack) -> (usize, usize, usize) {
    let components = usize::from(stack.frames[0].samples_per_pixel.max(1));
    let meta = &stack.meta;
    if meta
        .channels
        .checked_mul(meta.slices)
        .and_then(|n| n.checked_mul(meta.frames))
        == Some(stack.frames.len())
    {
        (meta.channels * components, meta.slices, meta.frames)
    } else if meta.slices.checked_mul(meta.frames) == Some(stack.frames.len())
        && meta.channels == components
    {
        (components, meta.slices, meta.frames)
    } else {
        (components, 1, stack.frames.len())
    }
}

pub(super) fn read(
    bytes: Vec<u8>,
    name: &str,
    host: &mut dyn ImportHost,
) -> Result<ImportResult, PluginError> {
    let stack = TiffStack::from_bytes(bytes).map_err(failed)?;
    let first = stack
        .frames
        .first()
        .ok_or_else(|| failed("Empty embedded TIFF"))?;
    let components = usize::from(first.samples_per_pixel.max(1));
    let meta = &stack.meta;
    let (channels, slices, frames) = dimensions(&stack);
    let pixel_type = match (first.bits_per_sample, first.sample_format) {
        (4 | 8, SampleFormat::UnsignedInt) => PixelType::U8,
        (16, SampleFormat::UnsignedInt) => PixelType::U16,
        (8 | 16, SampleFormat::SignedInt) => PixelType::I16,
        (32 | 64, _) => PixelType::F32,
        _ => return Err(failed("Unsupported embedded TIFF sample format")),
    };
    let mut planes = Vec::new();
    for (i, frame) in stack.frames.iter().enumerate() {
        progress(host, i, stack.frames.len())?;
        if (
            frame.width,
            frame.height,
            frame.samples_per_pixel,
            frame.bits_per_sample,
            frame.sample_format,
        ) != (
            first.width,
            first.height,
            first.samples_per_pixel,
            first.bits_per_sample,
            first.sample_format,
        ) {
            return Err(failed(
                "Embedded TIFF contains pages with different dimensions or sample types",
            ));
        }
        for c in 0..components {
            let plane = match pixel_type {
                PixelType::U8 => PlaneData::U8(
                    fast_tiff_lib::read_plane_u8(&stack.data, frame, stack.byte_order, c)
                        .map_err(failed)?,
                ),
                PixelType::U16 => PlaneData::U16(
                    fast_tiff_lib::read_plane_u16(&stack.data, frame, stack.byte_order, None, c)
                        .map_err(failed)?,
                ),
                PixelType::I16 => PlaneData::U16(
                    fast_tiff_lib::read_plane_f32(&stack.data, frame, stack.byte_order, c)
                        .map_err(failed)?
                        .into_iter()
                        .map(|v| v as i16 as u16)
                        .collect(),
                ),
                PixelType::F32 => PlaneData::F32(
                    fast_tiff_lib::read_plane_f32(&stack.data, frame, stack.byte_order, c)
                        .map_err(failed)?,
                ),
            };
            planes.push(plane);
        }
    }
    Ok(ImportResult {
        image: ImageResult {
            width: first.width,
            height: first.height,
            channels,
            slices,
            frames,
            pixel_type,
            planes,
            channel_colors: if meta.has_explicit_luts {
                meta.channel_display.iter().map(|ch| ch.lut[255]).collect()
            } else {
                vec![]
            },
            metadata: None,
            name: name.into(),
        },
        info: Some(StackInfo {
            name: name.into(),
            mode: if channels > 1 {
                DisplayMode::Composite
            } else {
                DisplayMode::Grayscale
            },
            unit: meta.unit.clone(),
            spacing: Spacing {
                x: meta.pixel_width,
                y: meta.pixel_height,
                z: meta.spacing,
            },
            calibration: meta.calibration,
            frame_interval_s: meta.frame_interval_s,
            description: stack.description,
            ..Default::default()
        }),
    })
}

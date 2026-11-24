use super::{failed, progress};
use fasttiff_plugin_api::*;
use hdf5_reader::{Dataset, Datatype, H5Type, SliceInfo, SliceInfoElem};

pub(super) struct Raster {
    pub axes: String,
    pixel_type: PixelType,
}

impl Raster {
    pub fn of(ds: &Dataset) -> Option<Self> {
        use Datatype::{FixedPoint, FloatingPoint};
        let pixel_type = match ds.dtype() {
            FixedPoint {
                size: 1,
                signed: false,
                ..
            } => PixelType::U8,
            FixedPoint {
                size: 2,
                signed: false,
                ..
            } => PixelType::U16,
            FixedPoint {
                size: 1 | 2,
                signed: true,
                ..
            } => PixelType::I16,
            FixedPoint { size: 4 | 8, .. } | FloatingPoint { size: 4 | 8, .. } => PixelType::F32,
            _ => return None,
        };
        let shape = ds.shape();
        if shape.contains(&0) || !(2..=5).contains(&shape.len()) {
            return None;
        }
        let attr = |name| ds.attribute(name).ok()?.read_string().ok();
        // Some NeXus files use "axes" for coordinate dataset names. Only a
        // sequence of recognized image axes describes a storage order here.
        let declared = attr("axes")
            .map(|s| {
                s.trim_matches('\0')
                    .trim()
                    .to_ascii_uppercase()
                    .replace('S', "C")
            })
            .filter(|s| {
                s.len() == shape.len()
                    && s.contains('X')
                    && s.contains('Y')
                    && s.chars()
                        .all(|a| "TCZYX".contains(a) && s.matches(a).count() == 1)
            });
        let axes = if let Some(axes) = declared {
            axes
        } else {
            let last_is_channel = matches!(shape.last(), Some(1 | 3 | 4));
            match (shape.len(), last_is_channel) {
                (2, _) => "YX",
                (3, _)
                    if attr("INTERLACE_MODE")
                        .is_some_and(|s| s.trim_end_matches('\0') == "INTERLACE_PLANE") =>
                {
                    "CYX"
                }
                (3, true) => "YXC",
                (3, false) => "ZYX",
                (4, true) => "ZYXC",
                (4, false) => "TZYX",
                (5, true) => "TZYXC",
                (5, false) => "TCZYX",
                _ => return None,
            }
            .to_owned()
        };
        Some(Self { axes, pixel_type })
    }

    fn extent(&self, ds: &Dataset, axis: char) -> u64 {
        self.axes.find(axis).map_or(1, |i| ds.shape()[i])
    }

    pub(super) fn single_frame(&self, ds: &Dataset) -> Option<super::FrameShape> {
        if self.extent(ds, 'T') != 1 || self.extent(ds, 'Z') != 1 {
            return None;
        }
        Some((
            self.extent(ds, 'X').try_into().ok()?,
            self.extent(ds, 'Y').try_into().ok()?,
            self.extent(ds, 'C').try_into().ok()?,
        ))
    }

    pub fn read(
        &self,
        ds: &Dataset,
        name: &str,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let size = |a| self.extent(ds, a);
        let width = u32::try_from(size('X')).map_err(failed)?;
        let height = u32::try_from(size('Y')).map_err(failed)?;
        let channels = usize::try_from(size('C')).map_err(failed)?;
        let slices = usize::try_from(size('Z')).map_err(failed)?;
        let frames = usize::try_from(size('T')).map_err(failed)?;
        let count = frames
            .checked_mul(slices)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| failed("Too many image planes"))?;
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(count))
            .and_then(|n| n.checked_mul(self.pixel_type.bytes()));
        if pixels.is_none_or(|n| n > isize::MAX as usize) {
            return Err(failed("The selected image is too large to address"));
        }
        let transposed = (self.axes.find('X') < self.axes.find('Y'))
            .then_some((width as usize, height as usize));
        let mut planes = Vec::new();
        planes.try_reserve_exact(count).map_err(failed)?;
        let packed = self.axes.ends_with('C');
        for t in 0..frames {
            for z in 0..slices {
                for c in 0..if packed { 1 } else { channels } {
                    progress(host, planes.len(), count)?;
                    let selections = self
                        .axes
                        .chars()
                        .zip(ds.shape())
                        .map(|(axis, &len)| match axis {
                            'T' => SliceInfoElem::Index(t as u64),
                            'Z' => SliceInfoElem::Index(z as u64),
                            'C' if !packed => SliceInfoElem::Index(c as u64),
                            _ => SliceInfoElem::Slice {
                                start: 0,
                                end: len,
                                step: 1,
                            },
                        })
                        .collect();
                    let interleave = if packed { channels } else { 1 };
                    let slab = SliceInfo { selections };
                    // A common case is a single interleaved RGB image. Use the reader's
                    // contiguous full-array path and deinterleave it in memory once.
                    let whole = count == interleave;
                    planes.extend(decode(ds, &slab, whole, interleave, transposed)?);
                }
            }
        }
        let mut info = StackInfo {
            name: name.into(),
            mode: if channels > 1 {
                DisplayMode::Composite
            } else {
                DisplayMode::Grayscale
            },
            ..Default::default()
        };
        if let Ok(a) = ds.attribute("element_size_um") {
            let sizes = a
                .read_1d::<f32>()
                .map(|v| v.into_iter().map(f64::from).collect())
                .or_else(|_| a.read_1d::<f64>());
            if let Ok(sizes) = sizes {
                if let [z, y, x] = sizes[..] {
                    if sizes.iter().all(|v| v.is_finite() && *v > 0.) {
                        info.unit = Some("um".into());
                        info.spacing = Spacing {
                            x: Some(x),
                            y: Some(y),
                            z: Some(z),
                        };
                    }
                }
            }
        }
        Ok(ImportResult {
            image: ImageResult {
                width,
                height,
                channels,
                slices,
                frames,
                pixel_type: self.pixel_type,
                planes,
                channel_colors: vec![],
                metadata: None,
                name: name.into(),
            },
            info: Some(info),
        })
    }
}

fn decode(
    ds: &Dataset,
    slab: &SliceInfo,
    whole: bool,
    interleave: usize,
    transposed: Option<(usize, usize)>,
) -> Result<Vec<PlaneData>, PluginError> {
    use Datatype::{FixedPoint, FloatingPoint};
    macro_rules! read {
        ($ty:ty, $variant:ident, $convert:expr) => {
            read_planes::<$ty, _>(ds, slab, whole, interleave, transposed, $convert)?
                .into_iter()
                .map(PlaneData::$variant)
                .collect()
        };
    }
    Ok(match ds.dtype() {
        FixedPoint {
            size: 1,
            signed: false,
            ..
        } => read!(u8, U8, |x| x),
        FixedPoint {
            size: 1,
            signed: true,
            ..
        } => read!(i8, U16, |x| x as i16 as u16),
        FixedPoint {
            size: 2,
            signed: false,
            ..
        } => read!(u16, U16, |x| x),
        FixedPoint {
            size: 2,
            signed: true,
            ..
        } => read!(i16, U16, |x| x as u16),
        FixedPoint {
            size: 4,
            signed: false,
            ..
        } => read!(u32, F32, |x| x as f32),
        FixedPoint {
            size: 4,
            signed: true,
            ..
        } => read!(i32, F32, |x| x as f32),
        FixedPoint {
            size: 8,
            signed: false,
            ..
        } => read!(u64, F32, |x| x as f32),
        FixedPoint {
            size: 8,
            signed: true,
            ..
        } => read!(i64, F32, |x| x as f32),
        FloatingPoint { size: 4, .. } => read!(f32, F32, |x| x),
        FloatingPoint { size: 8, .. } => read!(f64, F32, |x| x as f32),
        _ => return Err(failed("Unsupported image data type")),
    })
}

fn read_planes<T: H5Type + Copy, U>(
    ds: &Dataset,
    slab: &SliceInfo,
    whole: bool,
    channels: usize,
    transposed: Option<(usize, usize)>,
    convert: impl Fn(T) -> U,
) -> Result<Vec<Vec<U>>, PluginError> {
    let array = if whole {
        ds.read_array::<T>()
    } else {
        ds.read_slice::<T>(slab)
    }
    .map_err(failed)?;
    if channels == 1 && transposed.is_none() && array.is_standard_layout() {
        let len = array.len();
        let (data, offset) = array.into_raw_vec_and_offset();
        if offset.unwrap_or(0) == 0 && data.len() == len {
            return Ok(vec![data.into_iter().map(convert).collect()]);
        }
        return Ok(vec![data
            .into_iter()
            .skip(offset.unwrap_or(0))
            .take(len)
            .map(convert)
            .collect()]);
    }
    let data = array
        .as_slice()
        .ok_or_else(|| failed("Non-contiguous HDF5 plane"))?;
    if let Some((w, h)) = transposed {
        return Ok((0..channels)
            .map(|c| {
                (0..w * h)
                    .map(|i| convert(data[((i % w) * h + i / w) * channels + c]))
                    .collect()
            })
            .collect());
    }
    Ok((0..channels)
        .map(|c| {
            data[c..]
                .iter()
                .step_by(channels)
                .copied()
                .map(&convert)
                .collect()
        })
        .collect())
}

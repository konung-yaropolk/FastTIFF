use super::{failed, Entry, FrameShape};
use fasttiff_plugin_api::{
    ImageResult, ImportHost, ImportResult, PixelType, PlaneData, PluginError, StackInfo,
};
use hdf5_reader::Hdf5File;
use std::cmp::Reverse;
use std::collections::BTreeMap;

pub(super) struct Group {
    shape: FrameShape,
    members: Vec<usize>,
}

pub(super) fn groups(entries: &[Entry]) -> Vec<Group> {
    let mut by_shape = BTreeMap::<FrameShape, Vec<usize>>::new();
    for (i, entry) in entries.iter().enumerate() {
        if let Some(shape) = entry.single_frame {
            by_shape.entry(shape).or_default().push(i);
        }
    }
    let mut groups: Vec<_> = by_shape
        .into_iter()
        .filter(|(_, members)| members.len() > 1)
        .map(|(shape, members)| Group { shape, members })
        .collect();
    groups.sort_by_key(|g| (Reverse(g.members.len()), g.shape));
    groups
}

impl Group {
    pub fn label(&self) -> String {
        let (w, h, c) = self.shape;
        let channels = if c == 1 {
            String::new()
        } else {
            format!(", {c} channels")
        };
        format!(
            "All {w}x{h} images ({} frames{channels})",
            self.members.len()
        )
    }

    pub fn read(
        &self,
        file: &Hdf5File,
        entries: &[Entry],
        filename: &str,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let mut stack: Option<ImportResult> = None;
        let mut paths = String::from("HDF5 datasets in frame order:");
        for (i, &member) in self.members.iter().enumerate() {
            let mut frame_host = FrameProgress {
                host,
                index: i,
                count: self.members.len(),
            };
            let entry = &entries[member];
            let mut frame = entry.read(file, filename, &mut frame_host)?;
            let image = &mut frame.image;
            if (image.width, image.height, image.channels) != self.shape
                || image.slices != 1
                || image.frames != 1
            {
                return Err(failed(format!(
                    "{} no longer matches the selected group",
                    entry.path
                )));
            }
            paths.push('\n');
            paths.push_str(&entry.path);
            if let Some(stack) = &mut stack {
                let out = &mut stack.image;
                let ty = common_type(out.pixel_type, image.pixel_type);
                promote(out, ty);
                promote(image, ty);
                out.planes.append(&mut image.planes);
                out.frames += 1;
                if out.channel_colors != image.channel_colors {
                    out.channel_colors.clear();
                }
                if let (Some(a), Some(b)) = (&mut stack.info, &frame.info) {
                    if a.unit != b.unit || a.spacing != b.spacing {
                        a.spacing = Default::default();
                        a.unit = None;
                    }
                    if a.calibration != b.calibration {
                        a.calibration = None;
                    }
                    if a.channel_names != b.channel_names {
                        a.channel_names.clear();
                    }
                }
            } else {
                stack = Some(frame);
            }
        }
        let mut stack = stack.ok_or_else(|| failed("The selected HDF5 group is empty"))?;
        stack.image.name = format!("{filename} - {}", self.label());
        let info = stack.info.get_or_insert_with(StackInfo::default);
        info.name = stack.image.name.clone();
        info.frame_interval_s = None;
        info.description = Some(paths);
        Ok(stack)
    }
}

// Every supported integer sample is exactly representable in f32. Only widen
// if needed, including when signed and unsigned 16-bit planes occur together.
fn common_type(a: PixelType, b: PixelType) -> PixelType {
    use PixelType::*;
    match (a, b) {
        (a, b) if a == b => a,
        (U8, other) | (other, U8) => other,
        _ => F32,
    }
}

fn promote(image: &mut ImageResult, ty: PixelType) {
    if image.pixel_type == ty {
        return;
    }
    for plane in &mut image.planes {
        *plane = match (plane.as_plane_ref(), ty) {
            (fasttiff_plugin_api::PlaneRef::U8(v), PixelType::U16 | PixelType::I16) => {
                PlaneData::U16(v.iter().copied().map(u16::from).collect())
            }
            (fasttiff_plugin_api::PlaneRef::U8(v), PixelType::F32) => {
                PlaneData::F32(v.iter().copied().map(f32::from).collect())
            }
            (fasttiff_plugin_api::PlaneRef::U16(v), PixelType::F32)
                if image.pixel_type == PixelType::I16 =>
            {
                PlaneData::F32(v.iter().map(|&n| f32::from(n as i16)).collect())
            }
            (fasttiff_plugin_api::PlaneRef::U16(v), PixelType::F32) => {
                PlaneData::F32(v.iter().copied().map(f32::from).collect())
            }
            _ => unreachable!("common_type only widens samples"),
        };
    }
    image.pixel_type = ty;
}

struct FrameProgress<'a> {
    host: &'a mut dyn ImportHost,
    index: usize,
    count: usize,
}
impl ImportHost for FrameProgress<'_> {
    fn progress(&mut self, fraction: f32) -> bool {
        self.host
            .progress((self.index as f32 + fraction) / self.count as f32)
    }
    fn log(&mut self, message: &str) {
        self.host.log(message);
    }
}

//! Desktop import of one image from an HDF5 container. The dialog lists dataset
//! paths and storage axes so ambiguous unlabelled arrays show their interpretation.

mod group;
mod raster;
mod tiff;

use fasttiff_plugin_api::*;
use hdf5_reader::{Dataset, Datatype, Hdf5File, SliceInfo, SliceInfoElem};
use std::collections::{hash_map::DefaultHasher, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;

#[derive(Default)]
pub struct Hdf5;

const MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const EXTENSIONS: &[&str] = &[
    "h5", "hdf5", "hdf", "he5", "hd5", "h5j", "h5part", "h5ebsd", "nxs", "nx5", "nexus", "cxi",
    "ims", "mat", "nc", "nc4", "ilp", "jld", "jld2", "h5ad", "h5mu",
];

fn failed(e: impl std::fmt::Display) -> PluginError {
    PluginError::failed(e.to_string())
}

fn progress(host: &mut dyn ImportHost, done: usize, total: usize) -> Result<(), PluginError> {
    if host.progress(done as f32 / total as f32) {
        Ok(())
    } else {
        Err(failed("Import cancelled"))
    }
}

enum Content {
    Pixels(raster::Raster),
    Tiff,
}

type FrameShape = (u32, u32, usize); // width, height, channels

struct Entry {
    single_frame: Option<FrameShape>,
    path: String,
    label: String,
    content: Content,
}

fn embedded(ds: &Dataset) -> bool {
    match (ds.shape(), ds.dtype()) {
        (
            [n],
            Datatype::FixedPoint {
                size: 1,
                signed: false,
                ..
            },
        ) if *n >= 4 => {
            let prefix = SliceInfo {
                selections: vec![SliceInfoElem::Slice {
                    start: 0,
                    end: 4,
                    step: 1,
                }],
            };
            ds.read_slice::<u8>(&prefix).ok().is_some_and(|a| {
                matches!(
                    a.as_slice(),
                    Some(b"II*\0" | b"MM\0*" | b"II+\0" | b"MM\0+")
                )
            })
        }
        ([] | [1], Datatype::Opaque { tag, .. }) => {
            let name = ds.name().to_ascii_lowercase();
            name.ends_with(".tif")
                || name.ends_with(".tiff")
                || tag.to_ascii_lowercase().contains("tiff")
        }
        _ => false,
    }
}

fn catalog(file: &Hdf5File) -> Result<Vec<Entry>, PluginError> {
    let mut groups = vec![(String::new(), file.root_group().map_err(failed)?)];
    let mut visited = HashSet::new();
    let mut entries = Vec::new();
    while let Some((parent, group)) = groups.pop() {
        if !visited.insert(group.address()) {
            continue;
        }
        let (children, datasets) = group.members().map_err(failed)?;
        for child in children {
            groups.push((format!("{parent}/{}", child.name()), child));
        }
        for ds in datasets {
            let path = format!("{parent}/{}", ds.name());
            let (label, single_frame, content) = if let Some(raster) = raster::Raster::of(&ds) {
                (
                    format!("{path}   {:?} ({})", ds.shape(), raster.axes),
                    raster.single_frame(&ds),
                    Content::Pixels(raster),
                )
            } else if embedded(&ds) {
                (format!("{path}   (embedded TIFF)"), None, Content::Tiff)
            } else {
                continue;
            };
            entries.push(Entry {
                path,
                label,
                content,
                single_frame,
            });
        }
    }
    if entries.is_empty() {
        return Err(failed(
            "No supported images, stacks or embedded TIFFs in this HDF5 file",
        ));
    }
    // A lone embedded TIFF need not be indexed until it is selected.
    if entries
        .iter()
        .filter(|e| e.single_frame.is_some() || matches!(e.content, Content::Tiff))
        .take(2)
        .count()
        == 2
    {
        for e in &mut entries {
            if matches!(e.content, Content::Tiff) {
                e.single_frame = tiff::single_frame(&file.dataset(&e.path).map_err(failed)?);
            }
        }
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
}

// The ABI returns a choice index. Bind it to the listing the user actually saw
// so a container edited while the dialog is open cannot select another image.
fn choice_key(entries: &[Entry]) -> String {
    let mut h = DefaultHasher::new();
    for e in entries {
        e.label.hash(&mut h);
        e.single_frame.hash(&mut h);
    }
    format!("dataset_{:016x}", h.finish())
}

impl Importer for Hdf5 {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.hdf5", "HDF5")
            .version(env!("CARGO_PKG_VERSION"))
            .description("Select an image, stack or embedded TIFF from an HDF5 container.")
    }
    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("HDF5 containers", EXTENSIONS)]
    }
    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        let offsets =
            std::iter::once(0).chain(std::iter::successors(Some(512usize), |n| n.checked_mul(2)));
        if offsets
            .take_while(|&at| at < head.len())
            .any(|at| head[at..].starts_with(MAGIC))
        {
            Confidence::Certain
        } else if !head.starts_with(b"\x0e\x03\x13\x01") && self.file_types()[0].matches(path) {
            Confidence::Maybe
        } else {
            Confidence::No
        }
    }
    fn params(&self, path: &Path) -> Vec<ParamDecl> {
        // With no catalog the import entry point returns the underlying read error.
        let Ok(entries) = Hdf5File::open(path)
            .map_err(failed)
            .and_then(|f| catalog(&f))
        else {
            return vec![];
        };
        let groups = group::groups(&entries);
        let key = choice_key(&entries);
        let options = groups
            .iter()
            .map(group::Group::label)
            .chain(entries.into_iter().map(|e| e.label))
            .collect();
        vec![ParamDecl::new(
            key,
            "Image or stack",
            ParamKind::Choice {
                default: 0,
                options,
            },
        )]
    }
    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        progress(host, 0, 1)?;
        let file = Hdf5File::open(&request.path).map_err(failed)?;
        let entries = catalog(&file)?;
        let index = request.params.choice(&choice_key(&entries), usize::MAX);
        let groups = group::groups(&entries);
        let filename = request
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        let mut result = if let Some(group) = groups.get(index) {
            group.read(&file, &entries, &filename, host)?
        } else {
            let entry = entries.get(index - groups.len()).ok_or_else(|| failed("Select an image from the current HDF5 dataset list. Reopen the container if it has changed."))?;
            entry.read(&file, &filename, host)?
        };
        result.info.get_or_insert_with(StackInfo::default).path =
            Some(request.path.display().to_string());
        progress(host, 1, 1)?;
        Ok(result)
    }
}

impl Entry {
    fn read(
        &self,
        file: &Hdf5File,
        filename: &str,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let ds = file.dataset(&self.path).map_err(failed)?;
        let name = format!(
            "{filename} - {}",
            self.path.trim_start_matches('/').replace('/', "_")
        );
        let mut result = match &self.content {
            Content::Pixels(raster) => raster.read(&ds, &name, host),
            Content::Tiff => tiff::read(ds.read_raw_bytes().map_err(failed)?, &name, host),
        }
        .map_err(|e| failed(format!("{}: {e}", self.path)))?;
        let info = result.info.get_or_insert_with(StackInfo::default);
        let description = info.description.get_or_insert_with(String::new);
        if !description.is_empty() {
            description.push('\n');
        }
        description.push_str(&format!("HDF5 dataset: {}", self.path));
        Ok(result)
    }
}

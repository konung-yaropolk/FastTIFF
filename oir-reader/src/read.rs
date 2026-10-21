//! Reading an acquisition's parts, one after another, into one set of planes
//! and one record.

use crate::acquisition::{BlockSource, Gathered, Part, ReadPath};
use crate::container::{block_offsets, read_part, walk_blocks};
use crate::error::Error;
use crate::meta::RecordBuilder;
use crate::options::Access;
use crate::planes::Reassembly;
use crate::source::{Buffered, Memory, Source, WINDOW};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Parts of one acquisition to look for. Far beyond any real recording, and a
/// bound on the directory scan rather than a limit anyone should reach.
const MAX_PARTS: usize = 9_999;

/// Every file of an acquisition, in order, starting with the one named.
///
/// A long recording is split at a size limit into `<stem>.oir`, `<stem>_00001`,
/// `<stem>_00002`, … Each part is a whole container — its own header, its own
/// block index — and holds whole planes, so the set is read by reading each and
/// merging: no plane straddles a boundary.
///
/// The continuations are matched **with and without** the `.oir` extension.
/// Both are written in practice — the acquisition that this was built against
/// names them with no extension at all — and a reader that expected only one
/// form would silently open a quarter of a timelapse and call it the whole
/// thing, which is the worst way to be wrong about a file.
///
/// Numbering stops at the first gap rather than scanning a range: a missing
/// part means the set is incomplete, and reading across the hole would join
/// timepoints that are not adjacent. A file that is itself a continuation —
/// `<stem>_00001` — is returned alone, because only the first part begins a
/// set.
///
/// Nothing is read but directory entries; `first` is returned whether or not
/// it exists.
pub fn acquisition_parts(first: impl AsRef<Path>) -> Vec<PathBuf> {
    let path = first.as_ref();
    let mut parts = vec![path.to_path_buf()];
    let (Some(dir), Some(stem)) = (path.parent(), path.file_stem()) else {
        return parts;
    };
    // A part is itself named `<stem>_00001`, so following on from one would
    // look for `<stem>_00001_00001`. Only the file the user named begins a set.
    let stem = stem.to_string_lossy();
    if let Some((_, tail)) = stem.rsplit_once('_') {
        if tail.len() == 5 && tail.bytes().all(|b| b.is_ascii_digit()) {
            return parts;
        }
    }
    for n in 1..=MAX_PARTS {
        let base = format!("{stem}_{n:05}");
        let next = [dir.join(&base), dir.join(format!("{base}.oir"))]
            .into_iter()
            .find(|p| p.is_file());
        match next {
            Some(p) => parts.push(p),
            None => break,
        }
    }
    parts
}

/// One part, as it is handed to [`gather_inputs`].
pub(crate) enum PartInput<'a> {
    Path(PathBuf),
    Bytes(&'a [u8]),
    /// Any reader, through a window of the given size: how the tests drive the
    /// buffered path over windows far smaller than a real one.
    #[cfg(test)]
    Stream {
        reader: Box<dyn ReadSeek + 'a>,
        len: u64,
        window: usize,
    },
}

#[cfg(test)]
pub(crate) trait ReadSeek: std::io::Read + std::io::Seek {}
#[cfg(test)]
impl<T: std::io::Read + std::io::Seek> ReadSeek for T {}

impl PartInput<'_> {
    /// What the progress bar counts this part as.
    ///
    /// The bar follows bytes, not parts: the parts of a split recording are
    /// not the same size, and the last is usually short. A file whose size
    /// cannot be had counts as nothing — it fails properly when it is opened.
    fn weight(&self) -> u64 {
        match self {
            PartInput::Path(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            PartInput::Bytes(b) => b.len() as u64,
            #[cfg(test)]
            PartInput::Stream { len, .. } => *len,
        }
    }
}

/// What reading one part produced, besides its planes.
struct PartRead {
    block_source: BlockSource,
    block_count: usize,
    /// The metadata documents of the block list that was kept.
    docs: Vec<String>,
}

/// Read every part, in order, into one set of planes and one record.
pub(crate) fn gather_inputs(
    inputs: Vec<PartInput<'_>>,
    access: Access,
    from_paths: bool,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<Gathered, Error> {
    let sizes: Vec<u64> = inputs.iter().map(PartInput::weight).collect();
    let total = sizes.iter().sum::<u64>().max(1) as f32;
    let mut before = 0u64;
    let mut planes = Reassembly::default();
    // Every part's metadata, not only the first's: each part carries the frame
    // timestamps of the frames *it* holds — a four-part recording states 7,213
    // frames and puts about a quarter of the timestamps in each file — so the
    // recording's timing is in none of them alone.
    let mut record = RecordBuilder::default();
    let mut parts = Vec::with_capacity(inputs.len());
    for (input, &weight) in inputs.into_iter().zip(&sizes) {
        let (start, size) = (before as f32, weight as f32);
        let mut tick = |f: f32| progress((start + f * size) / total);
        let had = planes.map.len();
        let (path, size, read_path, map_error, read) = match input {
            PartInput::Bytes(b) => {
                let read = run_part(&mut Memory(b), &mut planes, &mut tick)?;
                (None, b.len() as u64, ReadPath::Memory, None, read)
            }
            PartInput::Path(p) => {
                let file = File::open(&p)
                    .map_err(|e| Error::io("could not open the file", e).at_path(&p))?;
                let (size, read_path, map_error, read) =
                    read_file(file, access, &mut planes, &mut tick).map_err(|e| e.at_path(&p))?;
                (Some(p), size, read_path, map_error, read)
            }
            #[cfg(test)]
            PartInput::Stream {
                reader,
                len,
                window,
            } => {
                let mut src = Buffered::new(reader, len, window);
                let read = run_part(&mut src, &mut planes, &mut tick)?;
                (None, len, ReadPath::Buffered, None, read)
            }
        };
        record.absorb(&read.docs);
        parts.push(Part {
            path,
            size,
            read_path,
            map_error,
            block_source: read.block_source,
            block_count: read.block_count,
            planes_added: planes.map.len() - had,
        });
        before += weight;
    }
    Ok(Gathered {
        planes: planes.map,
        record: record.finish(),
        parts,
        from_paths,
    })
}

/// One part from an open file: mapped when `access` and the build allow it and
/// the mapping succeeds, through a buffer over the same handle otherwise.
///
/// The mapping is dropped when the part has been read, so an acquisition of
/// many parts holds one at a time. Nothing read from it outlives it: every
/// pixel is copied into its plane as its descriptor is read.
fn read_file(
    file: File,
    access: Access,
    planes: &mut Reassembly,
    tick: &mut dyn FnMut(f32) -> bool,
) -> Result<(u64, ReadPath, Option<String>, PartRead), Error> {
    #[cfg(all(feature = "mmap", any(unix, windows)))]
    let map_error = match access {
        Access::Auto => match crate::map::map(&file) {
            Ok(map) => {
                let read = run_part(&mut Memory(&map), planes, tick)?;
                return Ok((map.len() as u64, ReadPath::Mapped, None, read));
            }
            // Not the end of the part: the same handle is read through a
            // buffer instead, and the reason is kept for whoever asks.
            Err(e) => Some(e.to_string()),
        },
        Access::Buffered => None,
    };
    #[cfg(not(all(feature = "mmap", any(unix, windows))))]
    let map_error = {
        let _ = access;
        None
    };
    // The length now, from the handle being read, is the snapshot: what a
    // mapping made at this moment would have covered.
    let len = file
        .metadata()
        .map_err(|e| Error::io("could not read the file", e))?
        .len();
    let read = run_part(&mut Buffered::new(file, len, WINDOW), planes, tick)?;
    Ok((len, ReadPath::Buffered, map_error, read))
}

/// One part, from its block list or, failing that, by walking it.
fn run_part<S: Source>(
    src: &mut S,
    planes: &mut Reassembly,
    tick: &mut dyn FnMut(f32) -> bool,
) -> Result<PartRead, Error> {
    let (offsets, mut block_source) = block_offsets(src)?;
    let had = planes.map.len();
    let mut docs = Vec::new();
    read_part(src, &offsets, planes, &mut docs, tick)?;
    let mut block_count = offsets.len();
    // An index that read cleanly and produced nothing is what a half-written
    // one looks like: the header is there, the offsets in it are not yet.
    // Walking the file is the same recovery as having no index at all, so try
    // it before giving up on the part.
    //
    // Conditioned on planes rather than on blocks because the first part of a
    // split acquisition legitimately carries only metadata, and re-reading that
    // one must not be mistaken for a recovery — so the walk's blocks, and the
    // metadata found in them, are kept only when it found planes.
    if block_source == BlockSource::Index && planes.map.len() == had {
        let walked = walk_blocks(src)?;
        if !walked.is_empty() {
            let mut walked_docs = Vec::new();
            read_part(src, &walked, planes, &mut walked_docs, tick)?;
            if planes.map.len() > had {
                block_source = BlockSource::Walk;
                block_count = walked.len();
                docs = walked_docs;
            }
        }
    }
    Ok(PartRead {
        block_source,
        block_count,
        docs,
    })
}

#[cfg(test)]
#[path = "read_tests.rs"]
mod tests;

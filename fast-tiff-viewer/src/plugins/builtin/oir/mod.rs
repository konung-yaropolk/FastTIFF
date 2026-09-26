//! Olympus/Evident OIR.
//!
//! OIR is proprietary and has no published specification. What follows was
//! determined from a real FluoView file and **checked against an oracle**: the
//! same acquisition exported to TIFF by the acquisition software itself. Frame
//! zero reassembled from the container is byte-identical to the exported one,
//! which is the only kind of evidence worth having about a format nobody
//! documents.
//!
//! # The container
//!
//! ```text
//!   0x00  "OLYMPUSRAWFORMAT"
//!   0x20  u64  total file size
//!   0x28  u64  offset of the block index
//!   0x48  "FLUOVIEW"
//!   ...   blocks
//!   index: u32 0xFFFFFFFF, u32, u32, then one u64 file offset per block
//! ```
//!
//! Every block is `u32 length, u32 type, length bytes`. They come in pairs: a
//! *descriptor* (type 3) naming a plane and the byte range of it that follows,
//! then the block carrying those bytes. A 512x512 16-bit plane arrives as 34
//! chunks of 15,360 bytes and one of 2,048 — so a plane is scattered, and
//! reassembling it is the whole job.
//!
//! Descriptor payload: `u32 offset-within-plane, u32 length, u32 name length,
//! name`.
//!
//! # Axes
//!
//! Names look like `t001_0_1_<uid>_<chunk>` in a timelapse and
//! `z001_0_1_<uid>_<chunk>` in a z-stack, and the UID is the *channel* — two
//! channels of one slice differ only there. See [`plane_key`], which is where
//! all three of those facts have to be got right at once.
//!
//! # Multi-file acquisitions
//!
//! A long recording is split into `name.oir`, `name_00001`, `name_00002`, …
//! (with or without the extension). Each part is a complete container holding
//! whole planes, so opening the first reads them all — see
//! [`acquisition_parts`].
//!
//! # Files that are not finished
//!
//! The block index is the **last** thing an acquisition writes. A recording
//! still in progress, or one whose tail was lost, has no index — and the
//! offset at `0x28` that should say where it lives is either zero or points
//! past the end of what was written. Reading from the index is therefore the
//! one thing that cannot be done to such a file, and it is where this reader
//! used to stop.
//!
//! It no longer has to. Blocks are self-delimiting, so when the index is
//! missing or unusable the stream is **walked** from the first block instead —
//! see [`walk_blocks`]. The walk stops at the first block that does not fit in
//! what is there, which is exactly what a half-written tail looks like, and
//! everything in front of it is whole. [`Shape::derive`] then drops the
//! part-written plane at the end as it already did for an acquisition stopped
//! mid-frame, and the import says what it recovered.
//!
//! This is a snapshot, not a subscription: what was on disk when the file was
//! opened is what is read. Reopening it later reads more.
//!
//! # Metadata
//!
//! Everything a converted file says about the acquisition comes out of the OIR
//! itself — see [`meta`], which reads the file's XML and writes it as the
//! `"key"\t"value"` record the acquisition software exports beside an OIR as a
//! `.txt`.
//!
//! That format is kept, and the `.txt` is not read. Two separate points. The
//! format is kept because things downstream parse it: a converted file's
//! description is the only place an analysis can learn when the stimulus fired,
//! and it should not have to learn two dialects to do it. The file is not read
//! because it is a second, optional, detachable copy of what the OIR already
//! contains — it goes missing, it gets renamed, it can be left behind when the
//! `.oir` is moved, and (since the exported name carries a series number) it
//! can perfectly well belong to a different acquisition in the same folder.
//! Reading the container means the record cannot disagree with the pixels
//! beside it.
//!
//! # Reading
//!
//! Each part is **memory-mapped**, and everything below works on `&[u8]`. That
//! is not an optimisation applied to a working reader so much as the shape the
//! format asks for: a plane arrives as thirty-five scattered chunks, so
//! reassembling one recording meant a quarter of a million `seek`+`read` pairs,
//! each a syscall, to copy bytes the page cache was already holding. Mapped,
//! the same work is a `copy_from_slice` per chunk and the kernel faults in what
//! is actually touched — which also means the scan for metadata blocks costs
//! only the pages it looks at rather than a read of every block in the file.
//!
//! And each part is read **once, front to back**. A descriptor and the pixels
//! it describes sit next to each other, so the pixels are copied out the moment
//! their descriptor has been read. Reading every descriptor first and the
//! pixels afterwards — which this did — walks a file that is not yet in memory
//! twice: once touching a page every fifteen kilobytes to find the descriptors,
//! and once more for the pages in between. On the spinning disk a lab keeps its
//! recordings on, that second walk was the difference between six seconds and
//! eight and a half for a 750 MB acquisition.
//!
//! # What it does not do
//!
//! Every plane is held in memory, because that is what the importer contract
//! asks for — fine for the gigabyte-scale files this was built against, not for
//! a forty-gigabyte timelapse. Mapping does not change that: the planes are
//! reassembled into owned buffers because the contract hands back owned
//! buffers.

/// Translating the acquisition record the file carries in its own XML.
mod meta;

use fasttiff_plugin_api::{
    Confidence, FileType, ImageResult, ImportHost, ImportRequest, ImportResult, Importer,
    PixelType, PlaneData, PluginError, PluginInfo, Spacing, StackInfo,
};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

/// The signature every OIR file starts with.
const MAGIC: &[u8] = b"OLYMPUSRAWFORMAT";
/// Where the header states the block index begins.
const INDEX_OFFSET_AT: u64 = 0x28;
/// The index's own leading marker, checked before anything is read from it.
const INDEX_MARKER: u32 = 0xFFFF_FFFF;
/// Bytes of index header before the offsets begin.
const INDEX_HEADER: usize = 12;
/// A descriptor block, naming the data block that follows it.
const TYPE_DESCRIPTOR: u32 = 3;

/// Refusals for structures that cannot be real, so a corrupt or unfamiliar file
/// fails on a bound rather than on an allocation.
const MAX_BLOCKS: u64 = 8_000_000;
const MAX_BLOCK_BYTES: u32 = 64 << 20;
const MAX_PLANE_BYTES: usize = 1 << 30;
const MAX_PLANES: usize = 200_000;
/// Parts of one acquisition to look for. Far beyond any real recording, and a
/// bound on the directory scan rather than a limit anyone should reach.
const MAX_PARTS: usize = 9_999;

/// Where the block stream is looked for when there is no index to say.
///
/// The first block's position is **found, not assumed**: a real acquisition
/// starts at `0x60`, behind a 16-byte structure this reader has no other use
/// for, while the synthetic files in this module's tests start at `0x50`.
const FIRST_BLOCK_FROM: u64 = 0x50;
/// Candidates are eight-byte aligned, because every field in the header is.
const FIRST_BLOCK_STEP: u64 = 8;
/// How far past the header to look — room for a header structure an order of
/// magnitude larger than any seen, and a bound so a file of rubbish ends rather
/// than hangs.
const FIRST_BLOCK_CANDIDATES: usize = 64;
/// Consecutive plausible blocks a candidate must show before it is believed.
///
/// One is not enough, and this is the trap the whole of [`find_first_block`]
/// exists to avoid: `0x50` in a real acquisition holds `u32 3, u32 2`, which
/// reads perfectly well as a three-byte block of type 2. A reader satisfied
/// with one block takes it, lands at `0x5b`, reads the bytes there as a length
/// and is desynchronised from every block in the file — producing not an error
/// but a wrong picture. The second block never parses from there, so a run is
/// what rejects it.
const FIRST_BLOCK_PROBE: usize = 4;
/// Block types in real files run 0 to 5. This is the bound
/// [`find_first_block`] uses to tell a real block header from two halves of
/// something else; the walk itself never checks the type, because an unknown
/// type is simply stepped over.
const MAX_BLOCK_TYPE: u32 = 15;
/// Consecutive zero-length blocks tolerated before the walk gives up.
///
/// A zero-length block is legitimate — one type-5 marker ends every timepoint —
/// but a region of zeros parses as an endless run of them, and stepping eight
/// bytes at a time through a gigabyte of zeros is a hang, not an error.
const MAX_EMPTY_RUN: usize = 4_096;
/// How much of a block to read before deciding whether it holds XML. See
/// [`embedded_xml`].
const XML_PEEK: usize = 512;

pub struct Oir;

impl Importer for Oir {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.oir", "Olympus OIR")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Read an Olympus/Evident FluoView OIR acquisition.")
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("Olympus OIR", &["oir"])]
    }

    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        // The signature is the whole answer when it is there. A file named
        // `.oir` without it is not one, whatever it is called, so it goes to
        // another importer rather than failing in this one.
        if head.starts_with(MAGIC) {
            return Confidence::Certain;
        }
        if head.is_empty() && self.file_types().iter().any(|t| t.matches(path)) {
            // An unreadable head — a file on a slow share — is no evidence
            // either way, so the name is all there is.
            return Confidence::Maybe;
        }
        Confidence::No
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        // Every part of the acquisition, the named one first. Each is a
        // complete container with its own index, so they are read the same way
        // and their planes merged into one map.
        let parts = acquisition_parts(&request.path);
        // One mapping per part, held for the whole import: the metadata is read
        // from them after the planes are.
        let mut files: Vec<Mmap> = Vec::with_capacity(parts.len());
        let mut planes = Planes::default();
        // Every part's block index, kept rather than only the first: each part
        // carries the frame timestamps of the frames *it* holds, and the
        // recording's timing is not in any one of them.
        let mut indexes: Vec<Vec<u64>> = Vec::with_capacity(parts.len());
        // Parts that had no usable index and had to be walked, which is what a
        // recording still in progress looks like.
        let mut walked = 0usize;
        // The bar follows bytes, not parts: the parts of a split recording are
        // not the same size, and the last is usually short.
        let sizes: Vec<u64> = parts
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .collect();
        let total = sizes.iter().sum::<u64>().max(1) as f32;
        let mut before = 0u64;
        for (part, path) in parts.iter().enumerate() {
            let file = File::open(path)
                .map_err(|e| PluginError::failed(format!("could not open the file: {e}")))?;
            // SAFETY: the standard caveat of a mapping — the bytes under it
            // must not change. They do not: a mapping's length is fixed when
            // it is made, and an acquisition only ever *appends*, so a file
            // still being written is read as the snapshot it was at this
            // moment and every byte in that snapshot is one already written
            // and never rewritten. That is what lets this reader open an
            // unfinished recording at all.
            //
            // The bargain would be broken by the file being *truncated* while
            // mapped, which is not something an acquisition does. It is the
            // same bargain the TIFF reader in `fast-tiff-lib` makes.
            let map = unsafe { Mmap::map(&file) }
                .map_err(|e| PluginError::failed(format!("could not map the file: {e}")))?;
            let (mut part_offsets, mut found) = block_offsets(&map)?;
            let size = sizes.get(part).copied().unwrap_or(0) as f32;
            let start = before as f32;
            let had = planes.map.len();
            read_part(&map, &part_offsets, &mut planes, &mut |f| {
                host.progress(0.9 * (start + f * size) / total)
            })?;
            // An index that read cleanly and produced nothing is what a
            // half-written one looks like: the header is there, the offsets in
            // it are not yet. Walking the file is the same recovery as having
            // no index at all, so try it before giving up on the part.
            //
            // Conditioned on planes rather than on blocks because the first
            // part of a split acquisition legitimately carries only metadata,
            // and re-reading that one must not be mistaken for a recovery.
            if found == Found::Index && planes.map.len() == had {
                let walked_offsets = walk_blocks(&map);
                if !walked_offsets.is_empty() {
                    read_part(&map, &walked_offsets, &mut planes, &mut |f| {
                        host.progress(0.9 * (start + f * size) / total)
                    })?;
                    if planes.map.len() > had {
                        part_offsets = walked_offsets;
                        found = Found::Walk;
                    }
                }
            }
            if found == Found::Walk {
                walked += 1;
            }
            before += sizes.get(part).copied().unwrap_or(0);
            indexes.push(part_offsets);
            files.push(map);
        }
        let mut planes = planes.map;
        if walked > 0 {
            // Said plainly, because the frames that are missing are the ones
            // the file does not have yet — not ones this reader declined to
            // read. Someone who expected the whole recording should know to
            // open it again when the acquisition has finished.
            host.log(&format!(
                "unfinished acquisition: {walked} of {} part(s) have no block index, so \
                 their blocks were recovered by walking the file. This is what was on \
                 disk when it was opened; open it again later for the rest.",
                parts.len()
            ));
        }
        if parts.len() > 1 {
            host.log(&format!(
                "{} part(s) of this acquisition, {} planes in total",
                parts.len(),
                planes.len()
            ));
        }
        if planes.is_empty() {
            // A long acquisition keeps its metadata in the named file and its
            // planes in siblings, so an empty map usually means the siblings
            // are not there to be read. Say which case this is: whether they
            // were never found, or were found and were empty too.
            let hint = if parts.len() > 1 {
                format!(
                    "the {} part(s) beside it carry none either",
                    parts.len() - 1
                )
            } else {
                "a split acquisition keeps them in siblings named                  `<name>_00001.oir`, `_00002.oir` and so on, and none were                  found beside this file: if they were moved or renamed, put                  them back and open this file again"
                    .to_string()
            };
            return Err(PluginError::unsupported(format!(
                "this OIR carries no image planes this reader recognises: {hint}"
            )));
        }
        // The acquisition's own account of itself, read once here and used
        // twice below: for the values this import needs, and for the record it
        // carries into the converted file. Every part is read, a part at a
        // time — a four-part recording states 7,213 frames and puts about a
        // quarter of the frame timestamps in each file, so the first part
        // alone cannot say how long the recording was.
        let mut reader = meta::Reader::default();
        for (file, offsets) in files.iter().zip(&indexes) {
            reader.absorb(&embedded_xml(file, offsets));
        }
        let record = reader.finish();

        let summary = Summary::from_record(&record);

        let (width, height) = dimensions(&record, &planes)?;
        let (shape, dropped) = Shape::derive(&mut planes, &summary, width, height)?;
        host.log(&format!(
            "{width}x{height}, {} channel(s), {} slice(s), {} frame(s), {}-bit",
            shape.channels,
            shape.slices,
            shape.frames,
            shape.bytes_per_sample * 8
        ));
        if dropped > 0 {
            // Worth saying rather than silently shortening the stack: the user
            // may have expected the frame that the acquisition did not finish.
            host.log(&format!(
                "skipped {dropped} incomplete plane(s) — the acquisition was stopped part-way"
            ));
        }

        host.progress(0.95);
        let data = finish_planes(planes, &shape);

        let name = request
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "oir".into());

        // What tag 270 of the converted file will carry: the acquisition
        // record, written the way the acquisition software exports it. Never
        // the file's own XML — that is four megabytes in one real acquisition,
        // 3.1 MB of it three 65,536-entry lookup tables, and nothing can read
        // it as a description.
        let file_name = request
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("{name}.oir"));
        let description = record.to_text(&file_name, shape.frames);
        host.log(&match description {
            Some(_) => format!(
                "acquisition record: {} channel(s), {} event marker(s)",
                record.channels.len(),
                record.events.len()
            ),
            None => "this OIR carries nothing this reader recognises as an acquisition record"
                .to_string(),
        });

        Ok(ImportResult {
            image: ImageResult {
                width,
                height,
                channels: shape.channels,
                slices: shape.slices,
                frames: shape.frames,
                pixel_type: if shape.bytes_per_sample == 1 {
                    PixelType::U8
                } else {
                    PixelType::U16
                },
                planes: data,
                channel_colors: Vec::new(),
                metadata: None,
                name: name.clone(),
            },
            info: Some(StackInfo {
                name,
                mode: if shape.channels > 1 {
                    fasttiff_plugin_api::DisplayMode::Composite
                } else {
                    fasttiff_plugin_api::DisplayMode::Grayscale
                },
                unit: summary.pixel_size.map(|_| "micron".to_string()),
                spacing: Spacing {
                    x: summary.pixel_size,
                    y: summary.pixel_size,
                    z: summary.z_step,
                },
                // From the file's own frame timestamps, over the frames
                // actually imported.
                frame_interval_s: record.frame_interval_s(shape.frames),
                channel_names: summary.channel_names.clone(),
                description,
                ..Default::default()
            }),
        })
    }
}

// ---------------------------------------------------------------- container

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
/// timepoints that are not adjacent.
fn acquisition_parts(path: &Path) -> Vec<std::path::PathBuf> {
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

/// The `len` bytes at `at`, or an error saying the file stops short of them.
///
/// Every read in this module goes through here, so a truncated or lying file
/// produces a refusal naming the offset rather than a panic on a slice index.
fn read_at(bytes: &[u8], at: u64, len: usize) -> Result<&[u8], PluginError> {
    usize::try_from(at)
        .ok()
        .and_then(|start| bytes.get(start..start.checked_add(len)?))
        .ok_or_else(|| {
            PluginError::failed(format!(
                "this OIR ends before the {len} byte(s) it points to at {at:#x}"
            ))
        })
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    let s = b.get(at..at + 8)?;
    Some(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

/// Refuse anything that is not an OIR at all.
fn check_signature(bytes: &[u8]) -> Result<(), PluginError> {
    let head = read_at(bytes, 0, 0x50.min(bytes.len()))?;
    if !head.starts_with(MAGIC) {
        return Err(PluginError::unsupported(
            "not an OIR file: the OLYMPUSRAWFORMAT signature is missing",
        ));
    }
    Ok(())
}

/// Where the header says the block index begins.
///
/// Read, not vetted: [`read_index`] bounds-checks the offset before it reads a
/// byte from it, so a second check here would only be a second copy of the same
/// rule. What changed is what happens when it is wrong — a bad offset is still
/// never *followed*, but it is no longer the end of the read either, because
/// the files that produce one are a recording still in progress and a recording
/// whose tail was lost, and in both the pixels sit in front of the index and
/// are unaffected by it.
fn index_offset(bytes: &[u8]) -> Option<u64> {
    u64_at(bytes.get(..0x50)?, INDEX_OFFSET_AT as usize)
}

/// How a part's blocks were found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Found {
    /// From the file's own index, which is what a finished file has.
    Index,
    /// By walking the block stream, because there was no usable index — which
    /// is what a file still being written looks like.
    Walk,
}

/// Every block in a part, and how they were found.
fn block_offsets(bytes: &[u8]) -> Result<(Vec<u64>, Found), PluginError> {
    check_signature(bytes)?;
    let indexed = index_offset(bytes)
        .and_then(|at| read_index(bytes, at).ok())
        .filter(|o| !o.is_empty());
    if let Some(offsets) = indexed {
        return Ok((offsets, Found::Index));
    }
    let walked = walk_blocks(bytes);
    if walked.is_empty() {
        return Err(PluginError::failed(
            "this OIR has no usable block index, and no run of blocks could be found in \
             it either: the file is damaged past what this reader can recover",
        ));
    }
    Ok((walked, Found::Walk))
}

/// Walk at most `want` blocks from `from`.
///
/// Reports how many parsed, whether any of them carried a payload, and whether
/// the walk ended at the end of the file rather than at something implausible.
/// The three are what [`find_first_block`] needs to tell a real start from a
/// coincidence.
fn probe_blocks(bytes: &[u8], from: u64, want: usize) -> (usize, bool, bool) {
    let len = bytes.len() as u64;
    let mut at = from;
    let mut seen = 0usize;
    let mut substantial = false;
    while seen < want {
        let Some(head) = at
            .checked_add(8)
            .filter(|e| *e <= len)
            .and_then(|_| bytes.get(at as usize..at as usize + 8))
        else {
            return (seen, substantial, true);
        };
        let (Some(blen), Some(ty)) = (u32_at(head, 0), u32_at(head, 4)) else {
            return (seen, substantial, true);
        };
        if blen > MAX_BLOCK_BYTES || ty > MAX_BLOCK_TYPE {
            return (seen, substantial, false);
        }
        if at.saturating_add(8).saturating_add(blen as u64) > len {
            return (seen, substantial, true);
        }
        substantial |= blen > 0;
        at += 8 + blen as u64;
        seen += 1;
    }
    (seen, substantial, true)
}

/// Where the block stream starts.
///
/// Each eight-byte-aligned candidate past the header is tried and the first
/// from which [`FIRST_BLOCK_PROBE`] blocks parse is kept.
///
/// Two further rules earn their keep. A candidate is rejected the moment a
/// header is implausible, but merely **running out of file** is not a
/// rejection: a file with only one block in it has nothing wrong with it, and
/// the best such candidate is the answer when none can show a full run. And a
/// run must contain at least one **non-empty** block, because a stretch of
/// zeros parses as an unlimited run of zero-length type-0 blocks and would
/// otherwise win simply by coming first.
fn find_first_block(bytes: &[u8]) -> Option<u64> {
    let len = bytes.len() as u64;
    let mut partial: Option<(usize, u64)> = None;
    let mut at = FIRST_BLOCK_FROM;
    for _ in 0..FIRST_BLOCK_CANDIDATES {
        if at.saturating_add(8) > len {
            break;
        }
        let (count, substantial, ran_out) = probe_blocks(bytes, at, FIRST_BLOCK_PROBE);
        if count >= FIRST_BLOCK_PROBE && substantial {
            return Some(at);
        }
        if ran_out && substantial && partial.is_none_or(|(c, _)| count > c) {
            partial = Some((count, at));
        }
        at += FIRST_BLOCK_STEP;
    }
    partial.map(|(_, at)| at)
}

/// Every block, found by walking the stream rather than by reading the index.
///
/// Blocks are `u32 length, u32 type, length bytes`, so each says where the next
/// begins and no index is needed to visit them all. The walk stops at the first
/// block that does not fit in what is there — which is what the tail of a file
/// still being written looks like, and what makes everything in front of it
/// whole.
///
/// It also stops cleanly on a file that *does* have an index: an index begins
/// with `0xFFFFFFFF`, which as a block length is past [`MAX_BLOCK_BYTES`].
fn walk_blocks(bytes: &[u8]) -> Vec<u64> {
    let Some(mut at) = find_first_block(bytes) else {
        return Vec::new();
    };
    let len = bytes.len() as u64;
    let mut offsets = Vec::new();
    let mut empties = 0usize;
    while (offsets.len() as u64) < MAX_BLOCKS {
        if at.saturating_add(8) > len {
            break;
        }
        let Some(head) = bytes.get(at as usize..at as usize + 8) else {
            break;
        };
        let Some(blen) = u32_at(head, 0) else { break };
        if blen > MAX_BLOCK_BYTES || at.saturating_add(8).saturating_add(blen as u64) > len {
            break;
        }
        empties = if blen == 0 { empties + 1 } else { 0 };
        if empties > MAX_EMPTY_RUN {
            break;
        }
        offsets.push(at);
        at += 8 + blen as u64;
    }
    offsets
}

/// The block index: one file offset per block.
fn read_index(bytes: &[u8], at: u64) -> Result<Vec<u64>, PluginError> {
    let len = bytes.len() as u64;
    let head = read_at(bytes, at, INDEX_HEADER)?;
    if u32_at(head, 0) != Some(INDEX_MARKER) {
        return Err(PluginError::unsupported(
            "this OIR's block index is not in a layout this reader knows",
        ));
    }
    // The index is not counted anywhere: it runs from its header to the end of
    // the file, one u64 per block.
    let count = (len - at - INDEX_HEADER as u64) / 8;
    if count == 0 || count > MAX_BLOCKS {
        return Err(PluginError::failed(format!(
            "the OIR block index claims {count} blocks, which cannot be right"
        )));
    }
    let raw = read_at(bytes, at + INDEX_HEADER as u64, (count * 8) as usize)?;
    let mut offsets = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let o = u64_at(raw, i * 8).unwrap_or(u64::MAX);
        // Blocks past the end are the shape a truncated or foreign file takes.
        if o + 8 > at {
            continue;
        }
        offsets.push(o);
    }
    Ok(offsets)
}

/// One run of bytes written into a plane.
#[derive(Clone, Copy)]
struct Run {
    /// Where in the reassembled plane these bytes went.
    at: usize,
    len: usize,
}

/// A plane as it is reassembled.
#[derive(Default)]
struct Plane {
    /// The plane's bytes, held in 16-bit units. Nearly every plane is 16-bit,
    /// and storing it this way means a finished plane *is* its sample buffer:
    /// no zeroed byte buffer copied into and then converted, which was three
    /// passes over every plane where one will do. An 8-bit plane is converted
    /// once, at the end.
    samples: Vec<u16>,
    /// Every run written, in the order written — what [`Shape::derive`] checks
    /// a plane's coverage against, and what decides the sample width.
    runs: Vec<Run>,
}

impl Plane {
    /// Copy a run of the file's bytes into place.
    fn write(&mut self, at: usize, src: &[u8]) {
        let end = at + src.len();
        let need = end.div_ceil(2);
        if self.samples.len() < need {
            self.samples.resize(need, 0);
        }
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut self.samples);
        bytes[at..end].copy_from_slice(src);
        self.runs.push(Run { at, len: src.len() });
    }
}

/// Every plane in the file, keyed by its name.
///
/// A `BTreeMap` because the key ordering *is* the plane ordering: names are
/// `t001_0_1`, `t002_0_1`, … so sorting them sorts by timepoint and then by the
/// remaining axes, which is the order the stack wants.
type PlaneMap = BTreeMap<String, Plane>;

/// The planes of an acquisition as its parts are read, and what they have cost.
#[derive(Default)]
struct Planes {
    map: PlaneMap,
    /// Bytes the reassembled planes hold between them.
    held: usize,
    /// Bytes they may hold: the size of every part read so far, plus slack.
    ///
    /// A file cannot carry more pixel data than it has bytes, so planes that
    /// grow past this are being grown by descriptors that lie — an offset of a
    /// gigabyte into a plane is a gigabyte of zeros, not a gigabyte of pixels.
    /// Copying as the descriptors are read means an allocation happens before
    /// anything could check a descriptor against the frame size, so this is the
    /// check instead.
    budget: usize,
    /// The largest plane so far, in samples. Every plane of an acquisition is
    /// the same size, so each new one is given this much room up front rather
    /// than growing a chunk at a time and copying itself on every doubling.
    typical: usize,
}

/// Slack in [`Planes::budget`] beyond the file's own size. Covers the zeroed
/// tail of a plane whose last chunk is short, many times over.
const BUDGET_SLACK: usize = 64 << 20;

/// Read one part: every descriptor, and the pixels it describes, in file order.
///
/// `on_progress` gets the fraction of this part's blocks done, and returns
/// `false` to stop.
fn read_part(
    bytes: &[u8],
    offsets: &[u64],
    planes: &mut Planes,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Result<(), PluginError> {
    planes.budget = planes.budget.max(BUDGET_SLACK).saturating_add(bytes.len());
    let mut i = 0usize;
    let mut reported = 0usize;
    while i + 1 < offsets.len() {
        if i >= reported + 1024 {
            reported = i;
            if !on_progress(i as f32 / offsets.len() as f32) {
                return Err(PluginError::unsupported("cancelled"));
            }
        }
        let head = match read_at(bytes, offsets[i], 16) {
            Ok(h) => h,
            Err(_) => {
                i += 1;
                continue;
            }
        };
        let (Some(len), Some(ty)) = (u32_at(head, 0), u32_at(head, 4)) else {
            i += 1;
            continue;
        };
        if ty != TYPE_DESCRIPTOR || !(12..=MAX_BLOCK_BYTES).contains(&len) {
            i += 1;
            continue;
        }
        // Descriptor payload: offset-within-plane, length, name length, name.
        let body = read_at(bytes, offsets[i] + 8, len as usize)?;
        let (Some(at), Some(run), Some(nlen)) = (u32_at(body, 0), u32_at(body, 4), u32_at(body, 8))
        else {
            i += 1;
            continue;
        };
        let Some(raw_name) = body.get(12..12 + nlen as usize) else {
            i += 1;
            continue;
        };
        let Ok(name) = std::str::from_utf8(raw_name) else {
            i += 1;
            continue;
        };

        // The block that follows carries the bytes this one describes.
        let data_head = read_at(bytes, offsets[i + 1], 8)?;
        let data_len = u32_at(data_head, 0).unwrap_or(0);
        if data_len as usize != run as usize || data_len > MAX_BLOCK_BYTES {
            i += 1;
            continue;
        }

        if let Some(key) = plane_key(name) {
            let at = at as usize;
            let end = at.saturating_add(run as usize);
            if end <= MAX_PLANE_BYTES {
                // A run the part does not actually hold is a corrupt
                // descriptor; it is skipped and that stretch of the plane left
                // zeroed, as a run past the end of the plane is.
                if let Ok(src) = read_at(bytes, offsets[i + 1] + 8, run as usize) {
                    let typical = planes.typical;
                    let plane = planes.map.entry(key).or_insert_with(|| Plane {
                        samples: Vec::with_capacity(typical),
                        runs: Vec::new(),
                    });
                    let was = plane.samples.len();
                    let grows = end.div_ceil(2).saturating_sub(was) * 2;
                    if planes.held.saturating_add(grows) > planes.budget {
                        return Err(PluginError::failed(
                            "this OIR's descriptors place more pixel data than the file holds",
                        ));
                    }
                    plane.write(at, src);
                    planes.held += (plane.samples.len() - was) * 2;
                    planes.typical = planes.typical.max(plane.samples.len());
                }
            }
        }
        i += 2;
    }
    if planes.map.len() > MAX_PLANES {
        return Err(PluginError::failed(format!(
            "this OIR declares {} planes, which cannot be right",
            planes.map.len()
        )));
    }
    on_progress(1.0);
    Ok(())
}

/// A sort key identifying the plane a block belongs to, or `None` for anything
/// that is not an image plane — the reference and thumbnail blocks are named
/// `REF_LSM0_…` and must not become frames.
///
/// Block names are `<axis><n>_<a>_<b>_<uid>_<chunk>`, and every part of that
/// matters:
///
/// * **The leading axis is `t` in a timelapse and `z` in a z-stack.** Reading
///   only `t` is why every z-stack failed: not one plane was recognised, and
///   the file was reported as carrying no images at all.
/// * **The UID identifies the channel.** Two channels of the same slice are
///   `z001_0_1_<uidA>` and `z001_0_1_<uidB>` — identical but for the UID, with
///   the numeric fields both fixed. Dropping it merges the channels into one
///   plane whose chunks overlap, which is not a decode error and so does not
///   announce itself; it just produces the wrong picture.
/// * **The trailing chunk index is not part of the key.** It is what makes a
///   plane several blocks, so it is exactly what must be grouped over.
///
/// The key is built so that sorting it gives ImageJ's `xyczt` plane order —
/// channel fastest, then z, then time — because the map's ordering *is* the
/// stack's ordering. Numbers are zero-padded so `10` sorts after `9`.
fn plane_key(name: &str) -> Option<String> {
    let mut fields = name.split('_');
    let lead = fields.next()?;
    let digits = lead
        .get(1..)
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))?;
    let n: u64 = digits.parse().ok()?;
    let (t, z) = match lead.get(..1)? {
        "t" => (n, 0),
        "z" => (0, n),
        _ => return None,
    };
    // Both axes always present, so a file using one sorts alongside a file
    // using the other and neither needs a special case downstream.
    let mut key = format!("t{t:06}_z{z:06}");
    for field in fields {
        match field.parse::<u64>() {
            Ok(v) => key.push_str(&format!("_{v:06}")),
            // The first non-numeric field is the UID. It goes in — it is the
            // channel — and everything after it is the chunk index, which does
            // not.
            Err(_) => {
                key.push('_');
                key.push_str(field);
                break;
            }
        }
    }
    Some(key)
}

// ------------------------------------------------------------------- shape

struct Shape {
    channels: usize,
    slices: usize,
    frames: usize,
    bytes_per_sample: usize,
    plane_bytes: usize,
}

impl Shape {
    /// Also drops incomplete planes, and says how many.
    ///
    /// An acquisition stopped mid-frame leaves a short last plane — the sample
    /// this was built against ends with 92,160 bytes of a 524,288-byte frame,
    /// and the acquisition software's own export omits it. Keeping it would add
    /// a mostly-black frame to the end of every interrupted recording, which is
    /// the kind of thing that quietly ruins an average.
    fn derive(
        planes: &mut PlaneMap,
        summary: &Summary,
        width: u32,
        height: u32,
    ) -> Result<(Shape, usize), PluginError> {
        let px = (width as usize)
            .checked_mul(height as usize)
            .filter(|v| *v > 0)
            .ok_or_else(|| PluginError::failed("this OIR states an impossible frame size"))?;

        // Sample width is measured, not assumed: the chunks of a plane add up
        // to its byte count, and that over the pixel count is the answer.
        // A chunk cannot begin beyond the largest plane these dimensions
        // allow, so one that claims to is excluded here rather than being left
        // to decide the sample width for the whole file.
        let ceiling = px.saturating_mul(2);
        let plane_bytes = planes
            .values()
            .map(|p| {
                p.runs
                    .iter()
                    .filter(|r| r.at < ceiling)
                    .map(|r| r.at + r.len)
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let bytes_per_sample = match plane_bytes.checked_div(px) {
            Some(1) => 1,
            Some(2) => 2,
            _ => {
                return Err(PluginError::failed(format!(
                    "a {width}x{height} plane of {plane_bytes} bytes is neither 8- nor 16-bit"
                )))
            }
        };

        let full = px * bytes_per_sample;
        let before = planes.len();
        planes.retain(|_, p| {
            let covered: usize = p
                .runs
                .iter()
                .filter(|r| r.at < ceiling)
                .map(|r| r.len)
                .sum();
            covered >= full
        });
        let dropped = before - planes.len();
        let n = planes.len();
        if n == 0 {
            return Err(PluginError::failed(format!(
                "every plane in this OIR is incomplete ({before} found)"
            )));
        }

        // The axes, preferably from the plane names themselves: distinct UIDs
        // are the channels, distinct `z` fields the slices. The names are what
        // the container says about its own contents, which is as close to the
        // pixels as this gets — the acquisition record beside them describes
        // what was *configured*, and the two can differ.
        //
        // But only when the names actually distinguish something. Planes named
        // `t001_0_1`, `t002_0_1`, … have one UID and one `z` field between
        // them, and reading that as "one channel, one slice" would be taking
        // silence for evidence. So a single UID and a single `z` defers to the
        // record, which also covers a file carrying an axis in some form these
        // names do not spell out.
        let counted = axis_counts(planes);
        let informative = counted.0 > 1 || counted.1 > 1;
        let (c, z) = match counted {
            (c, z) if informative && c * z > 0 && n.is_multiple_of(c * z) => (c, z),
            _ => match (summary.channels, summary.slices) {
                (Some(c), Some(z)) if c * z > 0 && n.is_multiple_of(c * z) => (c, z),
                (Some(c), None) if c > 0 && n.is_multiple_of(c) => (c, 1),
                (None, Some(z)) if z > 0 && n.is_multiple_of(z) => (1, z),
                _ => (1, 1),
            },
        };
        Ok((
            Shape {
                channels: c,
                slices: z,
                frames: n / (c * z),
                bytes_per_sample,
                plane_bytes: full,
            },
            dropped,
        ))
    }
}

/// Channels and slices, counted from the plane keys.
///
/// A key is `t<t>_z<z>_<a>_<b>_<uid>` (see [`plane_key`]), so the distinct UIDs
/// are the channels and the distinct `z` fields the slices. Both come back as
/// at least 1, and a key that does not parse simply contributes nothing rather
/// than derailing the count.
fn axis_counts(planes: &PlaneMap) -> (usize, usize) {
    let mut uids = std::collections::BTreeSet::new();
    let mut zs = std::collections::BTreeSet::new();
    for key in planes.keys() {
        let mut fields = key.split('_');
        if let Some(z) = fields.next().and_then(|_t| fields.next()) {
            zs.insert(z.to_string());
        }
        // The UID is last; a key with no non-numeric tail has one channel.
        if let Some(uid) = key.rsplit('_').next() {
            uids.insert(uid.to_string());
        }
    }
    (uids.len().max(1), zs.len().max(1))
}

/// The frame size the acquisition record states, checked before it is used.
///
/// Validated rather than trusted: everything downstream multiplies these
/// together — to size a plane buffer, to divide the bytes of a plane into a
/// sample width — so a number out of a malformed file becomes an allocation
/// and an arithmetic overflow rather than a refusal.
///
/// The record looks in three places for it ([`meta::Record::parse`]), so a file
/// that reaches here without one is one whose metadata this reader does not
/// recognise at all.
fn dimensions(record: &meta::Record, planes: &PlaneMap) -> Result<(u32, u32), PluginError> {
    const LIMIT: u32 = 1_000_000;
    match (record.width, record.height) {
        (Some(w), Some(h)) if (1..=LIMIT).contains(&w) && (1..=LIMIT).contains(&h) => Ok((w, h)),
        _ => Err(PluginError::failed(format!(
            "could not determine the frame size: this OIR's own metadata does not state it \
             (found {} plane(s))",
            planes.len()
        ))),
    }
}

/// Turn reassembled planes into the stack's planes, in plane order.
///
/// Nothing is read here: every byte was copied as its descriptor was read. A
/// plane is cut to exactly the frame's size — a run that claimed to go past the
/// end of its plane wrote only beyond it, and that is dropped with the tail.
fn finish_planes(planes: PlaneMap, shape: &Shape) -> Vec<PlaneData> {
    let samples = shape.plane_bytes / shape.bytes_per_sample;
    planes
        .into_values()
        .map(|plane| {
            let mut buf = plane.samples;
            match shape.bytes_per_sample {
                1 => {
                    let bytes: &[u8] = bytemuck::cast_slice(&buf);
                    let mut out = bytes[..bytes.len().min(samples)].to_vec();
                    out.resize(samples, 0);
                    PlaneData::U8(out)
                }
                _ => {
                    // Overlapping runs can cover a plane's byte count without
                    // reaching its end; that stretch stays zero, as it always
                    // did.
                    buf.resize(samples, 0);
                    buf.truncate(samples);
                    buf.shrink_to_fit();
                    // The file is little-endian and the bytes went in as they
                    // are. Only a big-endian host has anything to do.
                    #[cfg(target_endian = "big")]
                    for v in buf.iter_mut() {
                        *v = u16::from_le(*v);
                    }
                    PlaneData::U16(buf)
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------- metadata

/// The values from the acquisition record that this import *acts on*, as
/// opposed to the ones it merely carries into the description.
///
/// A separate type from [`meta::Record`] because they answer different
/// questions. The record is everything the file says, at the precision it says
/// it, on its way to tag 270. This is the handful of numbers that decide what
/// the stack *is* — how many channels, how many slices, how big a pixel — and
/// it exists so the code that decides that cannot accidentally reach for a
/// laser wavelength.
#[derive(Default)]
struct Summary {
    channels: Option<usize>,
    slices: Option<usize>,
    /// Microns per pixel.
    pixel_size: Option<f64>,
    /// Microns between slices.
    z_step: Option<f64>,
    channel_names: Vec<String>,
}

impl Summary {
    /// Taken from the record, at the precision the file states.
    ///
    /// Read from the [`Record`](meta::Record) rather than parsed back out of
    /// the text it renders, and the difference matters: that text states a
    /// pixel size to three decimals, because that is how the acquisition
    /// software's own export states it. These numbers calibrate every
    /// measurement made on the stack, and 0.621 where the file says
    /// 0.621480569402239 is an error of 0.08% in every distance — small, and no
    /// reason to introduce it when the exact value is right there.
    fn from_record(r: &meta::Record) -> Summary {
        Summary {
            channels: (!r.channels.is_empty()).then_some(r.channels.len()),
            slices: r.slices,
            pixel_size: r.pixel_x,
            z_step: r.z_step,
            channel_names: r.channels.iter().filter_map(|c| c.label()).collect(),
        }
    }
}

/// Every XML document in the file's metadata blocks that this reader
/// understands.
///
/// Two things are going on beyond reading blocks. A block holds one or more
/// *complete* documents laid end to end with binary padding between them — one
/// real acquisition puts ten in a single block — so returning the block would
/// return the padding along with them, and splitting is what makes each one
/// parseable. And a document whose root element this reader has no use for is
/// dropped here rather than carried: the display lookup tables alone are
/// 3.1 MB of the 4 MB an ordinary acquisition holds, and nothing downstream
/// wants them in memory, let alone in a TIFF tag.
///
/// Only the blocks the index points at are examined, so this never scans the
/// pixel data — which in a real acquisition is 95% of the file.
fn embedded_xml(bytes: &[u8], offsets: &[u64]) -> Vec<String> {
    let mut out = Vec::new();
    for &o in offsets {
        let Ok(head) = read_at(bytes, o, 8) else {
            continue;
        };
        let Some(len) = u32_at(head, 0) else {
            continue;
        };
        // Metadata blocks are XML documents of a few tens of kilobytes; pixel
        // blocks are a fixed small size and never start with `<?xml`.
        if !(16..=(4 << 20)).contains(&len) {
            continue;
        }
        // Look at the head of the block before reading the whole of it. Nearly
        // every block in an acquisition is pixel data — 12,186 of the 12,190 in
        // one part of the recording this was built against — so reading each
        // one in full to find out that it is not XML means reading the entire
        // file, gigabytes of it, twice: once here and once for the pixels. In
        // every file seen the declaration sits at offset 40 of the block, well
        // inside this window.
        let peek = (len as usize).min(XML_PEEK);
        let Ok(head) = read_at(bytes, o + 8, peek) else {
            continue;
        };
        if !head.windows(5).any(|w| w == b"<?xml") {
            continue;
        }
        let Ok(body) = read_at(bytes, o + 8, len as usize) else {
            continue;
        };
        let Some(start) = body.windows(5).position(|w| w == b"<?xml") else {
            continue;
        };
        // Lossy, not strict. One of these documents in a real acquisition is
        // not valid UTF-8 — an operator's name, a unit symbol, something typed
        // on a machine with a code page — and discarding the whole document for
        // it discarded the one that states the frame size, which then failed
        // the import with "this OIR's own metadata does not state it". Nothing
        // here reads prose: these documents are searched for numeric
        // tags, and a replacement character in a field nobody looks at is not a
        // reason to throw the file's dimensions away.
        let text = String::from_utf8_lossy(&body[start..]);
        out.extend(
            split_documents(&text)
                .into_iter()
                .filter(|d| meta::is_known_document(d)),
        );
    }
    out
}

/// The complete XML documents in `text`, each cut free of whatever follows it.
///
/// Documents are separated by the binary padding of the block they share, so
/// each one runs from its declaration to its own last `>`; anything after that
/// belongs to the block, not to the document.
fn split_documents(text: &str) -> Vec<String> {
    let starts: Vec<usize> = text.match_indices("<?xml").map(|(i, _)| i).collect();
    let mut out = Vec::with_capacity(starts.len());
    for (i, &from) in starts.iter().enumerate() {
        let to = starts.get(i + 1).copied().unwrap_or(text.len());
        let doc = &text[from..to];
        let Some(close) = doc.rfind('>') else {
            continue;
        };
        let doc = doc[..=close].trim();
        if !doc.is_empty() {
            out.push(doc.to_string());
        }
    }
    out
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

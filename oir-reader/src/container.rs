//! The container: the signature, the block index, walking the block stream when
//! there is no index to read, and the one pass over a part that reassembles its
//! planes and collects its metadata.

use crate::acquisition::BlockSource;
use crate::error::{truncated, Error, ErrorKind};
use crate::meta;
use crate::planes::{plane_key, PlaneBuf, Reassembly, BUDGET_SLACK, MAX_PLANES, MAX_PLANE_BYTES};
use crate::source::Source;
use std::io;

/// The signature every OIR file starts with.
pub(crate) const MAGIC: &[u8] = b"OLYMPUSRAWFORMAT";
/// Where the header states the block index begins.
const INDEX_OFFSET_AT: u64 = 0x28;
/// The index's own leading marker, checked before anything is read from it.
const INDEX_MARKER: u32 = 0xFFFF_FFFF;
/// Bytes of index header before the offsets begin.
pub(crate) const INDEX_HEADER: usize = 12;
/// A descriptor block, naming the data block that follows it.
pub(crate) const TYPE_DESCRIPTOR: u32 = 3;
/// The header: signature, sizes, offsets, and the format's own name.
const HEADER_LEN: u64 = 0x50;

/// Refusals for structures that cannot be real, so a corrupt or unfamiliar file
/// fails on a bound rather than on an allocation.
const MAX_BLOCKS: u64 = 8_000_000;
const MAX_BLOCK_BYTES: u32 = 64 << 20;

/// Where the block stream is looked for when there is no index to say.
///
/// The first block's position is **found, not assumed**: a real acquisition
/// starts at `0x60`, behind a 16-byte structure this reader has no other use
/// for, while the synthetic files in this crate's tests start at `0x50`.
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
/// [`xml_in_block`].
const XML_PEEK: usize = 512;

pub(crate) fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    let s = b.get(at..at + 8)?;
    Some(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

/// The `n` bytes at `at`, or an error saying the file stops short of them.
fn need<S: Source>(src: &mut S, at: u64, n: usize) -> Result<&[u8], Error> {
    match src.get(at, n)? {
        Some(b) => Ok(b),
        None => Err(truncated(at, n)),
    }
}

/// Refuse anything that is not an OIR at all.
fn check_signature<S: Source>(src: &mut S) -> Result<(), Error> {
    let n = HEADER_LEN.min(src.len()) as usize;
    let head = need(src, 0, n)?;
    if !head.starts_with(MAGIC) {
        return Err(Error::new(
            ErrorKind::NotOir,
            "not an OIR file: the OLYMPUSRAWFORMAT signature is missing",
        ));
    }
    Ok(())
}

/// Where the header says the block index begins.
///
/// Read, not vetted: [`read_index`] bounds-checks the offset before it reads a
/// byte from it. A bad offset is never *followed*, but it is not the end of the
/// read either, because the files that produce one are a recording still in
/// progress and a recording whose tail was lost, and in both the pixels sit in
/// front of the index and are unaffected by it.
fn index_offset<S: Source>(src: &mut S) -> io::Result<Option<u64>> {
    if src.len() < HEADER_LEN {
        return Ok(None);
    }
    Ok(src.get(INDEX_OFFSET_AT, 8)?.and_then(|b| u64_at(b, 0)))
}

/// Every block in a part, and how they were found.
pub(crate) fn block_offsets<S: Source>(src: &mut S) -> Result<(Vec<u64>, BlockSource), Error> {
    check_signature(src)?;
    let indexed = match index_offset(src)? {
        Some(at) => read_index(src, at)?,
        None => None,
    }
    .filter(|o| !o.is_empty());
    if let Some(offsets) = indexed {
        return Ok((offsets, BlockSource::Index));
    }
    let walked = walk_blocks(src)?;
    if walked.is_empty() {
        return Err(Error::malformed(
            "this OIR has no usable block index, and no run of blocks could be found in \
             it either: the file is damaged past what this reader can recover",
        ));
    }
    Ok((walked, BlockSource::Walk))
}

/// The block index: one file offset per block, or `None` when there is no
/// index here that this reader can use.
fn read_index<S: Source>(src: &mut S, at: u64) -> io::Result<Option<Vec<u64>>> {
    let len = src.len();
    let Some(head) = src.get(at, INDEX_HEADER)? else {
        return Ok(None);
    };
    if u32_at(head, 0) != Some(INDEX_MARKER) {
        return Ok(None);
    }
    // The index is not counted anywhere: it runs from its header to the end of
    // the file, one u64 per block.
    let count = (len - at - INDEX_HEADER as u64) / 8;
    if count == 0 || count > MAX_BLOCKS {
        return Ok(None);
    }
    let Some(raw) = src.get(at + INDEX_HEADER as u64, (count * 8) as usize)? else {
        return Ok(None);
    };
    let mut offsets = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let o = u64_at(raw, i * 8).unwrap_or(u64::MAX);
        // Blocks past the end are the shape a truncated or foreign file takes.
        //
        // Wrapping, deliberately: an entry within eight of `u64::MAX` wraps
        // past zero and is kept, to be found later not to be in the file. That
        // is what this reader has always done as a release build; an unchecked
        // add would make a debug build panic on such a file instead.
        if o.wrapping_add(8) > at {
            continue;
        }
        offsets.push(o);
    }
    Ok(Some(offsets))
}

/// Walk at most `want` blocks from `from`.
///
/// Reports how many parsed, whether any of them carried a payload, and whether
/// the walk ended at the end of the file rather than at something implausible.
/// The three are what [`find_first_block`] needs to tell a real start from a
/// coincidence.
fn probe_blocks<S: Source>(src: &mut S, from: u64, want: usize) -> io::Result<(usize, bool, bool)> {
    let len = src.len();
    let mut at = from;
    let mut seen = 0usize;
    let mut substantial = false;
    while seen < want {
        let Some(head) = src.get(at, 8)? else {
            return Ok((seen, substantial, true));
        };
        let (Some(blen), Some(ty)) = (u32_at(head, 0), u32_at(head, 4)) else {
            return Ok((seen, substantial, true));
        };
        if blen > MAX_BLOCK_BYTES || ty > MAX_BLOCK_TYPE {
            return Ok((seen, substantial, false));
        }
        if at.saturating_add(8).saturating_add(blen as u64) > len {
            return Ok((seen, substantial, true));
        }
        substantial |= blen > 0;
        at += 8 + blen as u64;
        seen += 1;
    }
    Ok((seen, substantial, true))
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
fn find_first_block<S: Source>(src: &mut S) -> io::Result<Option<u64>> {
    let len = src.len();
    let mut partial: Option<(usize, u64)> = None;
    let mut at = FIRST_BLOCK_FROM;
    for _ in 0..FIRST_BLOCK_CANDIDATES {
        if at.saturating_add(8) > len {
            break;
        }
        let (count, substantial, ran_out) = probe_blocks(src, at, FIRST_BLOCK_PROBE)?;
        if count >= FIRST_BLOCK_PROBE && substantial {
            return Ok(Some(at));
        }
        if ran_out && substantial && partial.is_none_or(|(c, _)| count > c) {
            partial = Some((count, at));
        }
        at += FIRST_BLOCK_STEP;
    }
    Ok(partial.map(|(_, at)| at))
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
pub(crate) fn walk_blocks<S: Source>(src: &mut S) -> io::Result<Vec<u64>> {
    let Some(mut at) = find_first_block(src)? else {
        return Ok(Vec::new());
    };
    let len = src.len();
    let mut offsets = Vec::new();
    let mut empties = 0usize;
    while (offsets.len() as u64) < MAX_BLOCKS {
        if at.saturating_add(8) > len {
            break;
        }
        let Some(head) = src.get(at, 8)? else {
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
    Ok(offsets)
}

/// Read one part: every descriptor and the pixels it describes, and every
/// metadata document, in one pass and in file order.
///
/// `tick` gets the fraction of this part's blocks done, and returns `false` to
/// stop.
///
/// # One pass
///
/// A descriptor and the pixels it describes sit next to each other, so the
/// pixels are copied out the moment their descriptor has been read. The
/// metadata documents are collected in the same pass: every block is looked at
/// for XML exactly once, as the pass reaches it — descriptors, the data blocks
/// they describe, and the last block of the part, which the descriptor loop
/// never visits. That used to be a second sweep over every block, after every
/// plane of every part had been read, and on a recording larger than memory it
/// read most of the file from disk a second time.
///
/// The order of requests within a block never goes backwards — the block's
/// head, its XML peek, its body, the next block's head, its peek, its payload —
/// so a buffered reader serves the whole part as one stream.
pub(crate) fn read_part<S: Source>(
    src: &mut S,
    offsets: &[u64],
    planes: &mut Reassembly,
    docs: &mut Vec<String>,
    tick: &mut dyn FnMut(f32) -> bool,
) -> Result<(), Error> {
    let part_len = usize::try_from(src.len()).unwrap_or(usize::MAX);
    planes.budget = planes.budget.max(BUDGET_SLACK).saturating_add(part_len);
    let n = offsets.len();
    let mut i = 0usize;
    let mut reported = 0usize;
    // `offsets[..seen]` have been looked at for XML.
    let mut seen = 0usize;
    while i + 1 < n {
        if i >= reported + 1024 {
            reported = i;
            if !tick(i as f32 / n as f32) {
                return Err(Error::cancelled());
            }
        }
        let head = src
            .get(offsets[i], 16)?
            .map(|h| (u32_at(h, 0), u32_at(h, 4)));
        xml_upto(src, offsets, &mut seen, i + 1, docs)?;
        let Some((Some(len), Some(ty))) = head else {
            i += 1;
            continue;
        };
        if ty != TYPE_DESCRIPTOR || !(12..=MAX_BLOCK_BYTES).contains(&len) {
            i += 1;
            continue;
        }
        // Descriptor payload: offset-within-plane, length, name length, name.
        let body = need(src, offsets[i] + 8, len as usize)?;
        let (Some(at), Some(run), Some(nlen)) = (u32_at(body, 0), u32_at(body, 4), u32_at(body, 8))
        else {
            i += 1;
            continue;
        };
        let Some(raw_name) = 12usize
            .checked_add(nlen as usize)
            .and_then(|end| body.get(12..end))
        else {
            i += 1;
            continue;
        };
        let Ok(name) = std::str::from_utf8(raw_name) else {
            i += 1;
            continue;
        };
        // Taken now, while the name is still borrowed from the descriptor:
        // the next read replaces what the borrow points into.
        let key = plane_key(name);

        // The block that follows carries the bytes this one describes.
        let data_head = need(src, offsets[i + 1], 8)?;
        let data_len = u32_at(data_head, 0).unwrap_or(0);
        xml_upto(src, offsets, &mut seen, i + 2, docs)?;
        if data_len as usize != run as usize || data_len > MAX_BLOCK_BYTES {
            i += 1;
            continue;
        }

        if let Some(key) = key {
            let at = at as usize;
            let end = at.saturating_add(run as usize);
            if end <= MAX_PLANE_BYTES {
                // A run the part does not actually hold is a corrupt
                // descriptor; it is skipped and that stretch of the plane left
                // zeroed, as a run past the end of the plane is.
                if let Some(bytes) = src.get(offsets[i + 1] + 8, run as usize)? {
                    let typical = planes.typical;
                    let plane = planes.map.entry(key).or_insert_with(|| PlaneBuf {
                        samples: Vec::with_capacity(typical),
                        runs: Vec::new(),
                    });
                    let was = plane.samples.len();
                    let grows = end.div_ceil(2).saturating_sub(was) * 2;
                    if planes.held.saturating_add(grows) > planes.budget {
                        return Err(Error::malformed(
                            "this OIR's descriptors place more pixel data than the file holds",
                        ));
                    }
                    plane.write(at, bytes);
                    planes.held += (plane.samples.len() - was) * 2;
                    planes.typical = planes.typical.max(plane.samples.len());
                }
            }
        }
        i += 2;
    }
    // The blocks the descriptor loop does not reach: the last one always, and
    // everything after a part ends in non-descriptors.
    xml_upto(src, offsets, &mut seen, n, docs)?;
    if planes.map.len() > MAX_PLANES {
        return Err(Error::malformed(format!(
            "this OIR declares {} planes, which cannot be right",
            planes.map.len()
        )));
    }
    tick(1.0);
    Ok(())
}

/// Look at `offsets[*seen..upto]` for XML, and note them as seen.
fn xml_upto<S: Source>(
    src: &mut S,
    offsets: &[u64],
    seen: &mut usize,
    upto: usize,
    docs: &mut Vec<String>,
) -> io::Result<()> {
    while *seen < upto.min(offsets.len()) {
        xml_in_block(src, offsets[*seen], docs)?;
        *seen += 1;
    }
    Ok(())
}

/// Every XML document in one block that this reader understands.
///
/// Two things are going on beyond reading the block. It holds one or more
/// *complete* documents laid end to end with binary padding between them — one
/// real acquisition puts ten in a single block — so returning the block would
/// return the padding along with them, and splitting is what makes each one
/// parseable. And a document whose root element this reader has no use for is
/// dropped here rather than carried: the display lookup tables alone are
/// 3.1 MB of the 4 MB an ordinary acquisition holds, and nothing wants them in
/// memory.
fn xml_in_block<S: Source>(src: &mut S, o: u64, docs: &mut Vec<String>) -> io::Result<()> {
    let Some(head) = src.get(o, 8)? else {
        return Ok(());
    };
    let Some(len) = u32_at(head, 0) else {
        return Ok(());
    };
    // Metadata blocks are XML documents of a few tens of kilobytes; pixel
    // blocks are a fixed small size and never start with `<?xml`.
    if !(16..=(4 << 20)).contains(&len) {
        return Ok(());
    }
    // Look at the head of the block before reading the whole of it. Nearly
    // every block in an acquisition is pixel data — 12,186 of the 12,190 in one
    // part of the recording this was built against — and in every file seen the
    // declaration sits at offset 40 of the block, well inside this window.
    let peek = (len as usize).min(XML_PEEK);
    let Some(head) = src.get(o + 8, peek)? else {
        return Ok(());
    };
    if !head.windows(5).any(|w| w == b"<?xml") {
        return Ok(());
    }
    let Some(body) = src.get(o + 8, len as usize)? else {
        return Ok(());
    };
    let Some(start) = body.windows(5).position(|w| w == b"<?xml") else {
        return Ok(());
    };
    // Lossy, not strict. One of these documents in a real acquisition is not
    // valid UTF-8 — an operator's name, a unit symbol, something typed on a
    // machine with a code page — and discarding the whole document for it
    // discarded the one that states the frame size. Nothing here reads prose:
    // these documents are searched for numeric tags, and a replacement
    // character in a field nobody looks at is not a reason to throw the file's
    // dimensions away.
    let text = String::from_utf8_lossy(&body[start..]);
    docs.extend(
        split_documents(&text)
            .into_iter()
            .filter(|d| meta::is_known_document(d)),
    );
    Ok(())
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
#[path = "container_tests.rs"]
mod tests;

//! Reassembling planes from the chunks they arrive in.

use std::collections::BTreeMap;

pub(crate) const MAX_PLANE_BYTES: usize = 1 << 30;
pub(crate) const MAX_PLANES: usize = 200_000;

/// Slack in [`Reassembly::budget`] beyond the files' own size. Covers the zeroed
/// tail of a plane whose last chunk is short, many times over.
pub(crate) const BUDGET_SLACK: usize = 64 << 20;

/// One run of bytes written into a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    /// Where in the reassembled plane these bytes went.
    pub(crate) at: usize,
    pub(crate) len: usize,
}

/// A plane as it is reassembled.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct PlaneBuf {
    /// The plane's bytes, held in 16-bit units. Nearly every plane is 16-bit,
    /// and storing it this way means a finished plane *is* its sample buffer:
    /// no zeroed byte buffer copied into and then converted, which was three
    /// passes over every plane where one will do. An 8-bit plane is converted
    /// once, at the end.
    pub(crate) samples: Vec<u16>,
    /// Every run written, in the order written — what the shape decision checks
    /// a plane's coverage against, and what decides the sample width.
    pub(crate) runs: Vec<Run>,
}

impl PlaneBuf {
    /// Copy a run of the file's bytes into place.
    pub(crate) fn write(&mut self, at: usize, src: &[u8]) {
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

/// Every plane in the acquisition, keyed by its name.
///
/// A `BTreeMap` because the key ordering *is* the plane ordering: names are
/// `t001_0_1`, `t002_0_1`, … so sorting them sorts by timepoint and then by the
/// remaining axes, which is the order the stack wants.
pub(crate) type PlaneMap = BTreeMap<String, PlaneBuf>;

/// The planes of an acquisition as its parts are read, and what they have cost.
#[derive(Default)]
pub(crate) struct Reassembly {
    pub(crate) map: PlaneMap,
    /// Bytes the reassembled planes hold between them.
    pub(crate) held: usize,
    /// Bytes they may hold: the size of every part read so far, plus slack.
    ///
    /// A file cannot carry more pixel data than it has bytes, so planes that
    /// grow past this are being grown by descriptors that lie — an offset of a
    /// gigabyte into a plane is a gigabyte of zeros, not a gigabyte of pixels.
    /// Copying as the descriptors are read means an allocation happens before
    /// anything could check a descriptor against the frame size, so this is the
    /// check instead.
    pub(crate) budget: usize,
    /// The largest plane so far, in samples. Every plane of an acquisition is
    /// the same size, so each new one is given this much room up front rather
    /// than growing a chunk at a time and copying itself on every doubling.
    pub(crate) typical: usize,
}

/// A sort key identifying the plane a block belongs to, or `None` for anything
/// that is not an image plane — the reference and thumbnail blocks are named
/// `REF_LSM0_…` and must not become frames.
///
/// Block names are `<axis><n>_<a>_<b>_<uid>_<chunk>`, and every part of that
/// matters:
///
/// * **The leading axis is `t` in a timelapse and `z` in a z-stack.** Reading
///   only `t` is why every z-stack once failed: not one plane was recognised,
///   and the file was reported as carrying no images at all.
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
pub(crate) fn plane_key(name: &str) -> Option<String> {
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

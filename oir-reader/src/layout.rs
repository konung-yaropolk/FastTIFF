//! Deciding what the reassembled planes are: their sample width, which of them
//! are whole, and how they divide into channels, slices and frames.

use crate::acquisition::{Planes, SampleType};
use crate::error::Error;
use crate::meta::Record;
use crate::planes::PlaneMap;

/// The stack the planes make up.
pub(crate) struct Layout {
    pub(crate) channels: usize,
    pub(crate) slices: usize,
    pub(crate) frames: usize,
    pub(crate) bytes_per_sample: usize,
    pub(crate) plane_bytes: usize,
}

impl Layout {
    /// The sample type the layout's width implies.
    pub(crate) fn sample_type(&self) -> SampleType {
        if self.bytes_per_sample == 1 {
            SampleType::U8
        } else {
            SampleType::U16
        }
    }

    /// Also drops incomplete planes, and says how many.
    ///
    /// An acquisition stopped mid-frame leaves a short last plane — the sample
    /// this was built against ends with 92,160 bytes of a 524,288-byte frame,
    /// and the acquisition software's own export omits it. Keeping it would add
    /// a mostly-black frame to the end of every interrupted recording, which is
    /// the kind of thing that quietly ruins an average.
    pub(crate) fn derive(
        planes: &mut PlaneMap,
        summary: &Summary,
        width: u32,
        height: u32,
    ) -> Result<(Layout, usize), Error> {
        let px = (width as usize)
            .checked_mul(height as usize)
            .filter(|v| *v > 0)
            .ok_or_else(|| Error::malformed("this OIR states an impossible frame size"))?;

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
                return Err(Error::malformed(format!(
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
            return Err(Error::malformed(format!(
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
        //
        // The products wrap, as they always have in a release build: a record
        // stating an absurd slice count multiplies past `usize::MAX`, and the
        // wrapped product then fails to divide the plane count, so the record
        // is ignored — rather than a debug build panicking on the file.
        let counted = axis_counts(planes);
        let informative = counted.0 > 1 || counted.1 > 1;
        let (c, z) = match counted {
            (c, z) if informative && c * z > 0 && n.is_multiple_of(c * z) => (c, z),
            _ => match (summary.channels, summary.slices) {
                (Some(c), Some(z))
                    if c.wrapping_mul(z) > 0 && n.is_multiple_of(c.wrapping_mul(z)) =>
                {
                    (c, z)
                }
                (Some(c), None) if c > 0 && n.is_multiple_of(c) => (c, 1),
                (None, Some(z)) if z > 0 && n.is_multiple_of(z) => (1, z),
                _ => (1, 1),
            },
        };
        Ok((
            Layout {
                channels: c,
                slices: z,
                frames: n / c.wrapping_mul(z),
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
///
/// [`plane_key`]: crate::planes::plane_key
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
/// The record looks in three places for it, so a file that reaches here
/// without one is one whose metadata this reader does not recognise at all.
pub(crate) fn dimensions(record: &Record, planes: &PlaneMap) -> Result<(u32, u32), Error> {
    const LIMIT: u32 = 1_000_000;
    match (record.width(), record.height()) {
        (Some(w), Some(h)) if (1..=LIMIT).contains(&w) && (1..=LIMIT).contains(&h) => Ok((w, h)),
        _ => Err(Error::malformed(format!(
            "could not determine the frame size: this OIR's own metadata does not state it \
             (found {} plane(s))",
            planes.len()
        ))),
    }
}

/// The values from the acquisition record that decide the shape.
///
/// A separate type from [`Record`] because they answer different questions:
/// the record is everything the file says, and this is the two numbers the
/// layout is allowed to consult, so the code deciding it cannot accidentally
/// reach for anything else.
pub(crate) struct Summary {
    pub(crate) channels: Option<usize>,
    pub(crate) slices: Option<usize>,
}

impl Summary {
    pub(crate) fn from_record(r: &Record) -> Summary {
        Summary {
            channels: (!r.channels().is_empty()).then_some(r.channels().len()),
            slices: r.slices(),
        }
    }
}

/// Turn reassembled planes into the stack's planes, in plane order.
///
/// Nothing is read here: every byte was copied as its descriptor was read. A
/// plane is cut to exactly the frame's size — a run that claimed to go past the
/// end of its plane wrote only beyond it, and that is dropped with the tail.
pub(crate) fn finish_planes(planes: PlaneMap, layout: &Layout) -> Planes {
    let samples = layout.plane_bytes / layout.bytes_per_sample;
    match layout.bytes_per_sample {
        1 => Planes::U8(
            planes
                .into_values()
                .map(|plane| {
                    let bytes: &[u8] = bytemuck::cast_slice(&plane.samples);
                    let mut out = bytes[..bytes.len().min(samples)].to_vec();
                    out.resize(samples, 0);
                    out
                })
                .collect(),
        ),
        _ => Planes::U16(
            planes
                .into_values()
                .map(|plane| {
                    let mut buf = plane.samples;
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
                    buf
                })
                .collect(),
        ),
    }
}

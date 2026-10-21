//! What a read produces.

use crate::error::{Error, ErrorKind};
use crate::layout::{dimensions, finish_planes, Layout, Summary};
use crate::meta::Record;
use crate::planes::PlaneMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Everything an acquisition's files hold, read and reassembled — before the
/// stack's shape is decided.
///
/// The first of two phases. All of the reading happens here; [`assemble`] then
/// decides what the planes are and puts them in order, which reads nothing.
/// Splitting the two lets a caller report on the files — how many parts, which
/// were still being written, which could not be mapped — before, and whether or
/// not, the planes make a stack.
///
/// [`assemble`]: Gathered::assemble
pub struct Gathered {
    pub(crate) planes: PlaneMap,
    pub(crate) record: Record,
    pub(crate) parts: Vec<Part>,
    pub(crate) from_paths: bool,
}

impl Gathered {
    /// The acquisition record, from the XML of every part.
    pub fn record(&self) -> &Record {
        &self.record
    }

    /// What happened to each part, in order.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Distinct image planes found, whole or not.
    pub fn planes_found(&self) -> usize {
        self.planes.len()
    }

    /// The second phase: decide the stack and put its planes in order.
    ///
    /// Takes the frame size from the record, measures the sample width from
    /// the data, leaves out planes the acquisition did not finish, and decides
    /// channels, slices and frames — from the plane names when they distinguish
    /// those, from the record when they do not.
    ///
    /// Fails with [`ErrorKind::NoPlanes`] when there is not one image plane,
    /// and [`ErrorKind::Malformed`] when the frame size or sample width cannot
    /// be determined or every plane is incomplete.
    pub fn assemble(self) -> Result<Acquisition, Error> {
        let Gathered {
            mut planes,
            record,
            parts,
            from_paths,
        } = self;
        if planes.is_empty() {
            return Err(no_planes(parts.len(), from_paths));
        }
        let summary = Summary::from_record(&record);
        let (width, height) = dimensions(&record, &planes)?;
        let (layout, dropped) = Layout::derive(&mut planes, &summary, width, height)?;
        let shape = Shape {
            width,
            height,
            channels: layout.channels,
            slices: layout.slices,
            frames: layout.frames,
            sample_type: layout.sample_type(),
        };
        Ok(Acquisition {
            shape,
            planes: finish_planes(planes, &layout),
            record,
            parts,
            dropped,
        })
    }
}

/// An acquisition with no image planes. Usually the parts that hold them are
/// not there: a long recording keeps its metadata in the named file and its
/// planes in siblings, so the message says which case this is.
fn no_planes(parts: usize, from_paths: bool) -> Error {
    let hint = match (parts, from_paths) {
        (0, _) => "no part of it was given".to_string(),
        (2.., true) => format!("the {} part(s) beside it carry none either", parts - 1),
        (2.., false) => format!("the other {} part(s) carry none either", parts - 1),
        (_, true) => "a split acquisition keeps them in siblings named \
             `<name>_00001.oir`, `_00002.oir` and so on, and none were found beside \
             this file: if they were moved or renamed, put them back and open this file \
             again"
            .to_string(),
        (_, false) => "a split acquisition keeps them in further parts: pass every \
             part of it, in order"
            .to_string(),
    };
    Error::new(
        ErrorKind::NoPlanes,
        format!("this OIR carries no image planes this reader recognises: {hint}"),
    )
}

impl fmt::Debug for Gathered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gathered")
            .field("planes_found", &self.planes.len())
            .field("parts", &self.parts)
            .field("record", &self.record)
            .finish()
    }
}

/// An acquisition, read: its planes in order, its shape, and its record.
#[derive(Clone, PartialEq)]
pub struct Acquisition {
    shape: Shape,
    planes: Planes,
    record: Record,
    parts: Vec<Part>,
    dropped: usize,
}

impl Acquisition {
    /// The stack the planes make up.
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Every plane, in `xyczt` order — see [`Planes`].
    pub fn planes(&self) -> &Planes {
        &self.planes
    }

    /// The planes, moved out without a copy.
    pub fn into_planes(self) -> Planes {
        self.planes
    }

    /// The acquisition record, from the XML of every part.
    pub fn record(&self) -> &Record {
        &self.record
    }

    /// What happened to each part, in order.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Planes left out because the acquisition did not finish them — a
    /// recording stopped mid-frame leaves a short last plane, which would be a
    /// mostly-black frame if it were kept.
    pub fn dropped_planes(&self) -> usize {
        self.dropped
    }

    /// The record as FluoView exports it beside an OIR — see
    /// [`Record::to_fluoview_text`] — for a stack of exactly this many frames,
    /// rather than the count the record states.
    pub fn fluoview_text(&self, file_name: &str) -> Option<String> {
        self.record.to_fluoview_text(file_name, self.shape.frames)
    }

    /// Seconds between frames, from the file's own frame timestamps, over the
    /// frames this stack holds. See [`Record::frame_interval_s`].
    pub fn frame_interval_s(&self) -> Option<f64> {
        self.record.frame_interval_s(self.shape.frames)
    }
}

impl fmt::Debug for Acquisition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Acquisition")
            .field("shape", &self.shape)
            .field("planes", &self.planes)
            .field("dropped_planes", &self.dropped)
            .field("parts", &self.parts)
            .field("record", &self.record)
            .finish()
    }
}

/// The stack an acquisition's planes make up.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Pixels per row.
    pub width: u32,
    /// Rows per plane.
    pub height: u32,
    /// Channels recorded.
    pub channels: usize,
    /// Z slices.
    pub slices: usize,
    /// Time points.
    pub frames: usize,
    /// What one sample is stored as.
    pub sample_type: SampleType,
}

impl Shape {
    /// Samples in one plane: `width * height`.
    pub fn plane_len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Planes in the stack: `channels * slices * frames`.
    pub fn plane_count(&self) -> usize {
        self.channels * self.slices * self.frames
    }

    /// Where channel `c` of slice `z` at time `t` is in [`Planes`] — channel
    /// fastest, then slice, then time — or `None` when that is outside the
    /// stack.
    pub fn plane_index(&self, c: usize, z: usize, t: usize) -> Option<usize> {
        (c < self.channels && z < self.slices && t < self.frames)
            .then(|| c + self.channels * (z + self.slices * t))
    }
}

/// What one sample is stored as.
///
/// An OIR's samples are 8- or 16-bit containers for fewer significant bits —
/// 10 or 12 is common, so a 16-bit sample may saturate at 1023 or 4095. See
/// [`Channel::bits`](crate::Channel::bits).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleType {
    /// One byte per sample.
    U8,
    /// Two bytes per sample.
    U16,
}

impl SampleType {
    /// Bytes per sample.
    pub fn bytes(self) -> usize {
        match self {
            SampleType::U8 => 1,
            SampleType::U16 => 2,
        }
    }
}

/// Every plane of an acquisition, in `xyczt` order — channel fastest, then
/// slice, then time, which is ImageJ's — each exactly `width * height` samples,
/// native-endian.
///
/// One enum over all of them, rather than an enum per plane, because every
/// plane of an acquisition has the same depth.
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq)]
pub enum Planes {
    /// 8-bit planes.
    U8(Vec<Vec<u8>>),
    /// 16-bit planes.
    U16(Vec<Vec<u16>>),
}

impl Planes {
    /// How many planes.
    pub fn len(&self) -> usize {
        match self {
            Planes::U8(p) => p.len(),
            Planes::U16(p) => p.len(),
        }
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What the samples are stored as.
    pub fn sample_type(&self) -> SampleType {
        match self {
            Planes::U8(_) => SampleType::U8,
            Planes::U16(_) => SampleType::U16,
        }
    }

    /// The planes, when they are 8-bit.
    pub fn as_u8(&self) -> Option<&[Vec<u8>]> {
        match self {
            Planes::U8(p) => Some(p),
            _ => None,
        }
    }

    /// The planes, when they are 16-bit.
    pub fn as_u16(&self) -> Option<&[Vec<u16>]> {
        match self {
            Planes::U16(p) => Some(p),
            _ => None,
        }
    }
}

impl fmt::Debug for Planes {
    /// A count and a size, never the samples: an acquisition is gigabytes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, len, each) = match self {
            Planes::U8(p) => ("U8", p.len(), p.first().map_or(0, Vec::len)),
            Planes::U16(p) => ("U16", p.len(), p.first().map_or(0, Vec::len)),
        };
        write!(f, "{kind}({len} planes x {each} samples)")
    }
}

/// What happened to one file of an acquisition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    pub(crate) path: Option<PathBuf>,
    pub(crate) size: u64,
    pub(crate) read_path: ReadPath,
    pub(crate) map_error: Option<String>,
    pub(crate) block_source: BlockSource,
    pub(crate) block_count: usize,
    pub(crate) planes_added: usize,
}

impl Part {
    /// The file, when the acquisition was read from files.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Bytes read: the file's length when it was opened. A recording still
    /// being written is read as it stood then.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// How its bytes were read.
    pub fn read_path(&self) -> ReadPath {
        self.read_path
    }

    /// Why it could not be memory-mapped, when mapping was tried and failed —
    /// in which case it was read through a buffer instead. `None` when no
    /// mapping was attempted, or when it succeeded.
    pub fn map_error(&self) -> Option<&str> {
        self.map_error.as_deref()
    }

    /// How its blocks were found.
    pub fn block_source(&self) -> BlockSource {
        self.block_source
    }

    /// How many blocks it holds.
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// Image planes first seen in this part. Zero is ordinary for the first
    /// part of a split recording, which carries the metadata.
    pub fn planes_added(&self) -> usize {
        self.planes_added
    }
}

/// How a part's bytes were read.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReadPath {
    /// Memory-mapped: parsed in place, pixels copied once into their planes.
    Mapped,
    /// Through a bounded buffer, front to back.
    Buffered,
    /// From bytes the caller handed over, in place.
    Memory,
}

/// How a part's blocks were found.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockSource {
    /// From the part's own block index, which is the last thing an
    /// acquisition writes: a finished file.
    Index,
    /// By walking the block stream, because there was no usable index — or an
    /// index that produced no plane. This is what a recording still being
    /// written looks like: what is read is what was on disk when it was opened.
    Walk,
}

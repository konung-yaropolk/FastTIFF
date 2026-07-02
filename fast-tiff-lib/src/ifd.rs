//! Low-level TIFF primitives: byte order, IFD entries, and the handful of
//! field types we actually need to read — for both classic TIFF (magic 42,
//! 32-bit offsets) and BigTIFF (magic 43, 64-bit offsets). This module knows
//! nothing about ImageJ — it's a minimal, correct TIFF directory reader.

use anyhow::{anyhow, bail, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

/// Classic TIFF (magic 42: u16 entry counts, 12-byte entries, u32 offsets) or
/// BigTIFF (magic 43: u64 entry counts, 20-byte entries, u64 offsets). Decided
/// by the header; both flavors share everything above the directory layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiffFlavor {
    Classic,
    Big,
}

impl TiffFlavor {
    /// Bytes available for an entry's inline value (4 classic, 8 BigTIFF).
    #[inline]
    fn inline_capacity(self) -> usize {
        match self {
            TiffFlavor::Classic => 4,
            TiffFlavor::Big => 8,
        }
    }
}

impl ByteOrder {
    #[inline]
    pub fn host() -> Self {
        if cfg!(target_endian = "little") {
            ByteOrder::Little
        } else {
            ByteOrder::Big
        }
    }

    #[inline]
    pub fn u16(self, b: &[u8]) -> u16 {
        match self {
            ByteOrder::Little => u16::from_le_bytes([b[0], b[1]]),
            ByteOrder::Big => u16::from_be_bytes([b[0], b[1]]),
        }
    }

    #[inline]
    pub fn u32(self, b: &[u8]) -> u32 {
        match self {
            ByteOrder::Little => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            ByteOrder::Big => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        }
    }

    #[inline]
    pub fn u64(self, b: &[u8]) -> u64 {
        let arr: [u8; 8] = b[..8].try_into().unwrap();
        match self {
            ByteOrder::Little => u64::from_le_bytes(arr),
            ByteOrder::Big => u64::from_be_bytes(arr),
        }
    }

    #[inline]
    pub fn f64_from(self, b: &[u8]) -> f64 {
        let arr: [u8; 8] = b.try_into().unwrap();
        match self {
            ByteOrder::Little => f64::from_le_bytes(arr),
            ByteOrder::Big => f64::from_be_bytes(arr),
        }
    }
}

/// TIFF field type codes we handle (baseline TIFF6 §2 + the BigTIFF trio).
fn type_size(field_type: u16) -> Option<usize> {
    match field_type {
        1 => Some(1),  // BYTE
        2 => Some(1),  // ASCII
        3 => Some(2),  // SHORT
        4 => Some(4),  // LONG
        5 => Some(8),  // RATIONAL (2x u32)
        6 => Some(1),  // SBYTE
        7 => Some(1),  // UNDEFINED
        8 => Some(2),  // SSHORT
        9 => Some(4),  // SLONG
        10 => Some(8), // SRATIONAL
        11 => Some(4), // FLOAT
        12 => Some(8), // DOUBLE
        16 => Some(8), // LONG8  (BigTIFF)
        17 => Some(8), // SLONG8 (BigTIFF)
        18 => Some(8), // IFD8   (BigTIFF)
        _ => None,
    }
}

/// A single raw IFD entry. Classic entries are 12 bytes on disk with a 4-byte
/// value field; BigTIFF entries are 20 bytes with an 8-byte value field — both
/// are normalized into this struct (`flavor` records which layout it came
/// from, i.e. how many of `value_or_offset`'s bytes are meaningful inline).
#[derive(Clone, Copy, Debug)]
pub struct RawIfdEntry {
    pub tag: u16,
    pub field_type: u16,
    pub count: u64,
    /// The value/offset field, verbatim, in file byte order. Classic entries
    /// fill only the first 4 bytes.
    pub value_or_offset: [u8; 8],
    pub flavor: TiffFlavor,
}

impl RawIfdEntry {
    /// Total byte length of this entry's data.
    fn data_len(&self) -> Option<usize> {
        let sz = type_size(self.field_type)? as u64;
        usize::try_from(sz.checked_mul(self.count)?).ok()
    }

    /// The value field interpreted as a data offset (u32 classic, u64 BigTIFF).
    fn offset(&self, order: ByteOrder) -> u64 {
        match self.flavor {
            TiffFlavor::Classic => order.u32(&self.value_or_offset[..4]) as u64,
            TiffFlavor::Big => order.u64(&self.value_or_offset),
        }
    }

    /// Resolve this entry's raw bytes, whether inline or via offset into `file`.
    pub fn raw_bytes<'a>(&self, file: &'a [u8], order: ByteOrder) -> Result<&'a [u8]> {
        let len = self
            .data_len()
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        if len <= self.flavor.inline_capacity() {
            // Data lives inline in value_or_offset; slice-returning callers
            // must use `owned_bytes` for that case.
            bail!("internal: use owned_bytes() for inline fields");
        }
        let offset = usize::try_from(self.offset(order))
            .map_err(|_| anyhow!("IFD entry tag {} offset exceeds address space", self.tag))?;
        file.get(offset..offset + len)
            .ok_or_else(|| anyhow!("IFD entry tag {} points outside file bounds", self.tag))
    }

    /// Resolve bytes regardless of inline-vs-offset storage, always owned.
    pub fn owned_bytes(&self, file: &[u8], order: ByteOrder) -> Result<Vec<u8>> {
        let len = self
            .data_len()
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        if len <= self.flavor.inline_capacity() {
            Ok(self.value_or_offset[..len].to_vec())
        } else {
            Ok(self.raw_bytes(file, order)?.to_vec())
        }
    }

    /// Interpret as an array of u64 (widens BYTE/SHORT/LONG; reads LONG8).
    pub fn as_u64_array(&self, file: &[u8], order: ByteOrder) -> Result<Vec<u64>> {
        let bytes = self.owned_bytes(file, order)?;
        let sz = type_size(self.field_type)
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        let mut out = Vec::with_capacity(self.count as usize);
        for chunk in bytes.chunks_exact(sz) {
            let v = match sz {
                1 => chunk[0] as u64,
                2 => order.u16(chunk) as u64,
                4 => order.u32(chunk) as u64,
                8 => order.u64(chunk),
                _ => bail!("tag {} has non-integer field type {}", self.tag, self.field_type),
            };
            out.push(v);
        }
        Ok(out)
    }

    /// Interpret as an array of u32 (fails if any value exceeds u32).
    pub fn as_u32_array(&self, file: &[u8], order: ByteOrder) -> Result<Vec<u32>> {
        self.as_u64_array(file, order)?
            .into_iter()
            .map(|v| {
                u32::try_from(v)
                    .map_err(|_| anyhow!("tag {} value {v} does not fit in 32 bits", self.tag))
            })
            .collect()
    }

    /// Interpret as a single u32 (first element; common for scalar tags).
    pub fn as_u32(&self, file: &[u8], order: ByteOrder) -> Result<u32> {
        let sz = type_size(self.field_type)
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        // Fast path for the overwhelmingly common scalar case (ImageWidth,
        // BitsPerSample, …): the value is stored inline in the value field, so
        // read it directly without the per-call Vec allocation of
        // `as_u64_array` — this is called ~once per tag per IFD, i.e. tens of
        // thousands of times when opening a large multi-frame stack.
        if self.count == 1 && sz <= self.flavor.inline_capacity() {
            let v = match sz {
                1 => self.value_or_offset[0] as u64,
                2 => order.u16(&self.value_or_offset[0..2]) as u64,
                4 => order.u32(&self.value_or_offset[0..4]) as u64,
                8 => order.u64(&self.value_or_offset),
                _ => bail!("tag {} has non-integer field type {}", self.tag, self.field_type),
            };
            return u32::try_from(v)
                .map_err(|_| anyhow!("tag {} value {v} does not fit in 32 bits", self.tag));
        }
        self.as_u32_array(file, order)?
            .first()
            .copied()
            .ok_or_else(|| anyhow!("tag {} has no data", self.tag))
    }

    /// Interpret as ASCII text (null-terminator and trailing nulls stripped).
    pub fn as_ascii(&self, file: &[u8], order: ByteOrder) -> Result<String> {
        let bytes = self.owned_bytes(file, order)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }
}

/// One parsed IFD: its entries plus the file offset of the *next* IFD (0 = none).
pub struct ParsedIfd {
    pub entries: Vec<RawIfdEntry>,
    pub next_offset: u64,
}

/// Read the IFD at `offset` (header-relative, i.e. absolute file offset),
/// in the layout `flavor` dictates.
pub fn read_ifd(file: &[u8], offset: usize, order: ByteOrder, flavor: TiffFlavor) -> Result<ParsedIfd> {
    let (count_len, entry_len, next_len) = match flavor {
        TiffFlavor::Classic => (2usize, 12usize, 4usize),
        TiffFlavor::Big => (8, 20, 8),
    };
    let count_bytes = file
        .get(offset..offset + count_len)
        .ok_or_else(|| anyhow!("IFD offset {} out of bounds", offset))?;
    let entry_count = match flavor {
        TiffFlavor::Classic => order.u16(count_bytes) as u64,
        TiffFlavor::Big => order.u64(count_bytes),
    };
    let entry_count = usize::try_from(entry_count)
        .map_err(|_| anyhow!("IFD at {} declares an absurd entry count", offset))?;

    let entries_start = offset + count_len;
    let entries_len = entry_count
        .checked_mul(entry_len)
        .ok_or_else(|| anyhow!("IFD at {} declares an absurd entry count", offset))?;
    let entries_bytes = file
        .get(entries_start..entries_start + entries_len)
        .ok_or_else(|| anyhow!("IFD at {} truncated (entry table out of bounds)", offset))?;

    let mut entries = Vec::with_capacity(entry_count);
    for chunk in entries_bytes.chunks_exact(entry_len) {
        let tag = order.u16(&chunk[0..2]);
        let field_type = order.u16(&chunk[2..4]);
        let mut value_or_offset = [0u8; 8];
        let count = match flavor {
            TiffFlavor::Classic => {
                value_or_offset[..4].copy_from_slice(&chunk[8..12]);
                order.u32(&chunk[4..8]) as u64
            }
            TiffFlavor::Big => {
                value_or_offset.copy_from_slice(&chunk[12..20]);
                order.u64(&chunk[4..12])
            }
        };
        entries.push(RawIfdEntry {
            tag,
            field_type,
            count,
            value_or_offset,
            flavor,
        });
    }

    let next_offset_pos = entries_start + entries_len;
    let next_bytes = file
        .get(next_offset_pos..next_offset_pos + next_len)
        .ok_or_else(|| anyhow!("IFD at {} truncated (missing next-IFD offset)", offset))?;
    let next_offset = match flavor {
        TiffFlavor::Classic => order.u32(next_bytes) as u64,
        TiffFlavor::Big => order.u64(next_bytes),
    };

    Ok(ParsedIfd {
        entries,
        next_offset,
    })
}

/// Parse the TIFF header: byte order, flavor (classic magic 42 / BigTIFF
/// magic 43), and the first IFD's absolute offset.
pub fn read_header(file: &[u8]) -> Result<(ByteOrder, TiffFlavor, u64)> {
    if file.len() < 8 {
        bail!("file too small to be a TIFF (need at least 8 bytes, got {})", file.len());
    }
    let order = match &file[0..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        _other => bail!("not a TIFF"),
    };
    let magic = order.u16(&file[2..4]);
    match magic {
        42 => Ok((order, TiffFlavor::Classic, order.u32(&file[4..8]) as u64)),
        43 => {
            // BigTIFF: u16 offset size (always 8), u16 reserved (always 0),
            // then the u64 first-IFD offset.
            if file.len() < 16 {
                bail!("file too small to be a BigTIFF (need at least 16 bytes, got {})", file.len());
            }
            let offset_size = order.u16(&file[4..6]);
            let reserved = order.u16(&file[6..8]);
            if offset_size != 8 || reserved != 0 {
                bail!("malformed BigTIFF header (offset size {offset_size}, reserved {reserved})");
            }
            Ok((order, TiffFlavor::Big, order.u64(&file[8..16])))
        }
        other => bail!("not a TIFF (magic {other}; expected 42 or 43)"),
    }
}

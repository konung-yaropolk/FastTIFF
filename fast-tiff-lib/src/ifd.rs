//! Low-level TIFF6 baseline primitives: byte order, IFD entries, and the
//! handful of field types we actually need to read. This module knows
//! nothing about ImageJ — it's a minimal, correct TIFF directory reader.

use anyhow::{anyhow, bail, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
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
    pub fn f64_from(self, b: &[u8]) -> f64 {
        let arr: [u8; 8] = b.try_into().unwrap();
        match self {
            ByteOrder::Little => f64::from_le_bytes(arr),
            ByteOrder::Big => f64::from_be_bytes(arr),
        }
    }
}

/// TIFF field type codes we handle (baseline TIFF6 §2).
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
        _ => None,
    }
}

/// A single raw 12-byte IFD entry, as stored in the file.
#[derive(Clone, Copy, Debug)]
pub struct RawIfdEntry {
    pub tag: u16,
    pub field_type: u16,
    pub count: u32,
    /// The 4-byte value/offset field, verbatim, in file byte order.
    pub value_or_offset: [u8; 4],
}

impl RawIfdEntry {
    /// Total byte length of this entry's data.
    fn data_len(&self) -> Option<usize> {
        type_size(self.field_type).map(|sz| sz * self.count as usize)
    }

    /// Resolve this entry's raw bytes, whether inline or via offset into `file`.
    pub fn raw_bytes<'a>(&self, file: &'a [u8], order: ByteOrder) -> Result<&'a [u8]> {
        let len = self
            .data_len()
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        if len <= 4 {
            // Data lives inline in value_or_offset. We can't return a slice
            // into `self` with lifetime `'a`, so callers that need this path
            // should use `small_bytes()` instead. This branch is only valid
            // when len > 4 in practice for slice-returning use; for len<=4
            // see `small_bytes`.
            bail!("internal: use small_bytes() for inline (<=4 byte) fields");
        }
        let offset = order.u32(&self.value_or_offset) as usize;
        file.get(offset..offset + len)
            .ok_or_else(|| anyhow!("IFD entry tag {} points outside file bounds", self.tag))
    }

    /// Resolve bytes regardless of inline-vs-offset storage, always owned.
    pub fn owned_bytes(&self, file: &[u8], order: ByteOrder) -> Result<Vec<u8>> {
        let len = self
            .data_len()
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        if len <= 4 {
            Ok(self.value_or_offset[..len].to_vec())
        } else {
            let offset = order.u32(&self.value_or_offset) as usize;
            let slice = file
                .get(offset..offset + len)
                .ok_or_else(|| anyhow!("IFD entry tag {} points outside file bounds", self.tag))?;
            Ok(slice.to_vec())
        }
    }

    /// Interpret as an array of u32 (works for SHORT, LONG; widens SHORT).
    pub fn as_u32_array(&self, file: &[u8], order: ByteOrder) -> Result<Vec<u32>> {
        let bytes = self.owned_bytes(file, order)?;
        let sz = type_size(self.field_type)
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        let mut out = Vec::with_capacity(self.count as usize);
        for chunk in bytes.chunks_exact(sz) {
            let v = match sz {
                1 => chunk[0] as u32,
                2 => order.u16(chunk) as u32,
                4 => order.u32(chunk),
                _ => bail!("tag {} has non-integer field type {}", self.tag, self.field_type),
            };
            out.push(v);
        }
        Ok(out)
    }

    /// Interpret as a single u32 (first element; common for scalar tags).
    pub fn as_u32(&self, file: &[u8], order: ByteOrder) -> Result<u32> {
        let sz = type_size(self.field_type)
            .ok_or_else(|| anyhow!("unsupported TIFF field type {}", self.field_type))?;
        // Fast path for the overwhelmingly common scalar case (ImageWidth,
        // BitsPerSample, …): the value is stored inline in the 4-byte field, so
        // read it directly without the per-call Vec allocation of
        // `as_u32_array` — this is called ~once per tag per IFD, i.e. tens of
        // thousands of times when opening a large multi-frame stack.
        if self.count == 1 && sz <= 4 {
            return Ok(match sz {
                1 => self.value_or_offset[0] as u32,
                2 => order.u16(&self.value_or_offset[0..2]) as u32,
                4 => order.u32(&self.value_or_offset),
                _ => bail!("tag {} has non-integer field type {}", self.tag, self.field_type),
            });
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
    pub next_offset: u32,
}

/// Read the IFD at `offset` (header-relative, i.e. absolute file offset).
pub fn read_ifd(file: &[u8], offset: usize, order: ByteOrder) -> Result<ParsedIfd> {
    let count_bytes = file
        .get(offset..offset + 2)
        .ok_or_else(|| anyhow!("IFD offset {} out of bounds", offset))?;
    let entry_count = order.u16(count_bytes) as usize;

    let entries_start = offset + 2;
    let entries_len = entry_count * 12;
    let entries_bytes = file
        .get(entries_start..entries_start + entries_len)
        .ok_or_else(|| anyhow!("IFD at {} truncated (entry table out of bounds)", offset))?;

    let mut entries = Vec::with_capacity(entry_count);
    for chunk in entries_bytes.chunks_exact(12) {
        let tag = order.u16(&chunk[0..2]);
        let field_type = order.u16(&chunk[2..4]);
        let count = order.u32(&chunk[4..8]);
        let mut value_or_offset = [0u8; 4];
        value_or_offset.copy_from_slice(&chunk[8..12]);
        entries.push(RawIfdEntry {
            tag,
            field_type,
            count,
            value_or_offset,
        });
    }

    let next_offset_pos = entries_start + entries_len;
    let next_bytes = file
        .get(next_offset_pos..next_offset_pos + 4)
        .ok_or_else(|| anyhow!("IFD at {} truncated (missing next-IFD offset)", offset))?;
    let next_offset = order.u32(next_bytes);

    Ok(ParsedIfd {
        entries,
        next_offset,
    })
}

/// Parse the 8-byte TIFF header and return (byte_order, first_ifd_offset).
pub fn read_header(file: &[u8]) -> Result<(ByteOrder, u32)> {
    if file.len() < 8 {
        bail!("file too small to be a TIFF (need at least 8 bytes, got {})", file.len());
    }
    let order = match &file[0..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        _other => bail!("not a TIFF"),
    };
    let magic = order.u16(&file[2..4]);
    if magic != 42 {
        bail!("not a classic TIFF. BigTIFF is not supported.");
    }
    let first_ifd = order.u32(&file[4..8]);
    Ok((order, first_ifd))
}

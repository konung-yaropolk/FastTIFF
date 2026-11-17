//! Lossless JPEG — ITU-T T.81 process 14, which is the compression a hospital
//! archive is actually full of and which no Rust crate decodes.
//!
//! It shares a container with ordinary JPEG and nothing else. There is no DCT,
//! no quantisation table, no 8x8 block, no colour transform. Each sample is
//! **predicted** from its already-decoded neighbours and only the difference
//! from that prediction is Huffman-coded:
//!
//! ```text
//!     c b          Ra = a (left), Rb = b (above), Rc = c (above-left)
//!     a x          x = prediction + difference
//! ```
//!
//! The `SOS` header picks which of seven predictors is in use. Selection value
//! 1 — `Ra`, the sample to the left — is what the `.70` transfer syntax pins
//! and what essentially every encoder emits, but all seven are here because
//! they cost a line each and a file using another one is otherwise silently
//! wrong rather than refused.
//!
//! # Why hand-rolled
//!
//! Because the alternative is nothing. `zune-jpeg` says in as many words that
//! it cannot decode lossless Huffman, `jpeg-decoder` is baseline and
//! progressive, and the C libraries that can (CharLS, libjpeg's lossless fork,
//! DCMTK) are C. The whole decoder is a Huffman table, a bit reader and a
//! prediction — some two hundred lines — against a transfer syntax that is a
//! third of the DICOM in the world.
//!
//! # What it does not do
//!
//! Arithmetic coding (process 15), hierarchical progression, and more than one
//! component interleaved per scan. The first two are refused by marker in
//! [`super::jpeg`]; the third is refused here, because a colour lossless JPEG
//! is rare enough that an untested path for it would be worse than a message.

use super::pixels::Format;
use fasttiff_plugin_api::PluginError;

/// A Huffman table as the decoder wants it: canonical codes, flattened.
///
/// Built from the `BITS`/`HUFFVAL` arrays a `DHT` segment carries, by the
/// procedure in T.81 Annex C. Stored as parallel arrays of (length, code,
/// value) sorted by length, which is enough for the one-bit-at-a-time read
/// below — a lookup table would be faster and is not the bottleneck here,
/// since a lossless frame is one pass over the samples with no inverse
/// transform behind it.
#[derive(Default, Clone)]
struct Huffman {
    /// `mincode[l]`, `maxcode[l]` and `valptr[l]` for code length `l + 1`,
    /// exactly as T.81 Figure F.15 defines them. `maxcode` of `-1` marks a
    /// length with no codes.
    mincode: [i32; 16],
    maxcode: [i32; 16],
    valptr: [usize; 16],
    values: Vec<u8>,
}

impl Huffman {
    /// Build from a `DHT`'s sixteen length counts and its value list.
    fn new(counts: &[u8; 16], values: Vec<u8>) -> Self {
        let mut t = Huffman {
            maxcode: [-1; 16],
            values,
            ..Default::default()
        };
        let mut code = 0i32;
        let mut k = 0usize;
        for (l, &count) in counts.iter().enumerate() {
            let n = count as usize;
            if n == 0 {
                // Left as -1, which is what makes the "no code of this length"
                // test below a comparison rather than a special case.
                code <<= 1;
                continue;
            }
            t.valptr[l] = k;
            t.mincode[l] = code;
            code += n as i32;
            k += n;
            t.maxcode[l] = code - 1;
            code <<= 1;
        }
        t
    }

    /// Decode one value, reading bit by bit until a code matches.
    fn decode(&self, bits: &mut Bits) -> Result<u8, PluginError> {
        let mut code = 0i32;
        for l in 0..16 {
            code = (code << 1) | bits.bit()? as i32;
            if self.maxcode[l] >= code && code >= self.mincode[l] {
                let at = self.valptr[l] + (code - self.mincode[l]) as usize;
                return self.values.get(at).copied().ok_or_else(|| {
                    PluginError::failed(
                        "this frame's lossless JPEG names a Huffman value its \
                         table does not have",
                    )
                });
            }
        }
        Err(PluginError::failed(
            "this frame's lossless JPEG has a Huffman code longer than the \
             sixteen bits the format allows",
        ))
    }
}

/// A most-significant-bit-first bit reader over entropy-coded data, with JPEG's
/// byte stuffing undone.
struct Bits<'a> {
    b: &'a [u8],
    at: usize,
    cur: u8,
    left: u8,
}

impl<'a> Bits<'a> {
    fn new(b: &'a [u8]) -> Self {
        Bits {
            b,
            at: 0,
            cur: 0,
            left: 0,
        }
    }

    /// One bit, MSB first.
    ///
    /// `FF 00` is a stuffed `FF` and yields the `FF`; the `00` is the stuffing
    /// and is skipped. Any other `FF xx` is a marker and ends the data — which
    /// is how a scan finds its own end without being told a length.
    fn bit(&mut self) -> Result<u8, PluginError> {
        if self.left == 0 {
            let Some(&byte) = self.b.get(self.at) else {
                // Running out mid-sample is a truncated frame. Returning zeros
                // would finish the image with a plausible gradient instead.
                return Err(PluginError::failed(
                    "this frame's lossless JPEG data ends in the middle of a \
                     sample",
                ));
            };
            self.at += 1;
            if byte == 0xFF {
                match self.b.get(self.at) {
                    Some(0x00) => self.at += 1,
                    _ => {
                        return Err(PluginError::failed(
                            "this frame's lossless JPEG reaches a marker before \
                             it has decoded every sample",
                        ))
                    }
                }
            }
            self.cur = byte;
            self.left = 8;
        }
        self.left -= 1;
        Ok((self.cur >> self.left) & 1)
    }

    /// `n` bits as an unsigned integer, MSB first.
    fn bits(&mut self, n: u8) -> Result<i32, PluginError> {
        let mut v = 0i32;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as i32;
        }
        Ok(v)
    }

    /// Abandon the rest of the current byte, as a restart marker requires.
    fn align(&mut self) {
        self.left = 0;
    }
}

/// What the `SOF3` and `SOS` headers between them say.
struct Scan {
    width: usize,
    height: usize,
    /// Sample precision in bits, `P`.
    precision: u8,
    components: usize,
    /// `Ss`: which of the seven predictors, 1..=7.
    predictor: u8,
    /// `Al`: the point transform, a left shift applied to every sample.
    point_transform: u8,
    /// Which Huffman table each component uses.
    table_of: Vec<usize>,
    /// Where the entropy-coded data starts.
    data_at: usize,
    /// `DRI`: samples between restart markers, 0 for none.
    restart_interval: usize,
}

/// Decode one frame's lossless JPEG into native little-endian samples.
pub(super) fn decode(b: &[u8], fmt: Format) -> Result<Vec<u8>, PluginError> {
    let (scan, tables) = headers(b)?;

    if scan.components != 1 {
        return Err(PluginError::unsupported(format!(
            "this frame is a {}-component lossless JPEG. Only single-component \
             (greyscale) lossless frames are decoded here; `gdcmconv --raw` \
             converts the rest",
            scan.components
        )));
    }
    if scan.width != fmt.columns as usize || scan.height != fmt.rows as usize {
        return Err(PluginError::failed(format!(
            "this frame's JPEG is {}x{} where the DICOM header says {}x{}",
            scan.width, scan.height, fmt.columns, fmt.rows
        )));
    }
    if !(2..=16).contains(&scan.precision) {
        return Err(PluginError::failed(format!(
            "this frame's lossless JPEG declares {} bits of precision",
            scan.precision
        )));
    }

    let table = tables
        .get(*scan.table_of.first().unwrap_or(&0))
        .cloned()
        .flatten()
        .ok_or_else(|| {
            PluginError::failed(
                "this frame's lossless JPEG names a Huffman table it does not \
                 carry",
            )
        })?;

    let (w, h) = (scan.width, scan.height);
    let mut out: Vec<i32> = vec![0; w * h];
    let mut bits = Bits::new(&b[scan.data_at..]);
    // The first sample of the image has no neighbours, so T.81 H.2.1 gives it a
    // fixed prediction of half the range.
    let origin = 1i32 << (scan.precision - 1 - scan.point_transform.min(scan.precision - 1));
    let mut since_restart = 0usize;
    let mut fresh = true;

    for y in 0..h {
        for x in 0..w {
            // Restart markers reset the predictor *and* the bit alignment, and
            // treating one as data is how a decoder turns the bottom two thirds
            // of an image into noise.
            if scan.restart_interval > 0 && since_restart == scan.restart_interval {
                bits.align();
                skip_restart(&mut bits)?;
                since_restart = 0;
                fresh = true;
            }
            since_restart += 1;

            let at = y * w + x;
            let ra = if x > 0 { out[at - 1] } else { 0 };
            let rb = if y > 0 { out[at - w] } else { 0 };
            let rc = if x > 0 && y > 0 { out[at - w - 1] } else { 0 };

            // The prediction rules, in the order T.81 gives them. The edges are
            // not special cases of the seven predictors — they are their own
            // rules, and using predictor 1 on the first column would reference
            // the previous row's last sample.
            let predicted = if fresh {
                // The first sample after a restart, and of the image.
                fresh = false;
                origin
            } else if y == 0 {
                ra
            } else if x == 0 {
                rb
            } else {
                match scan.predictor {
                    1 => ra,
                    2 => rb,
                    3 => rc,
                    4 => ra + rb - rc,
                    5 => ra + ((rb - rc) >> 1),
                    6 => rb + ((ra - rc) >> 1),
                    7 => (ra + rb) / 2,
                    other => {
                        return Err(PluginError::failed(format!(
                            "this frame's lossless JPEG asks for predictor \
                             {other}, and the format defines 1 to 7"
                        )))
                    }
                }
            };

            let diff = difference(&table, &mut bits)?;
            // Wrapping at the precision, which is the format's own arithmetic:
            // T.81 H.2.2 reduces every reconstruction modulo 2^P.
            let modulus = 1i32 << scan.precision;
            out[at] = (predicted + diff) & (modulus - 1);
        }
    }

    // Out as the native layout: little-endian, one or two bytes per sample,
    // with the point transform undone. The transform is a right shift the
    // encoder applied to throw away low bits; putting it back is a left shift,
    // and skipping it would leave every sample too small by a power of two.
    let bytes = fmt.sample_bytes();
    let shift = scan.point_transform;
    let mut native = vec![0u8; out.len() * bytes];
    for (i, &v) in out.iter().enumerate() {
        let v = (v << shift) as u32;
        native[i * bytes] = (v & 0xFF) as u8;
        if bytes > 1 {
            native[i * bytes + 1] = ((v >> 8) & 0xFF) as u8;
        }
    }
    Ok(native)
}

/// One sample's difference: a Huffman-coded magnitude category, then that many
/// bits of the value.
fn difference(table: &Huffman, bits: &mut Bits) -> Result<i32, PluginError> {
    let ssss = table.decode(bits)?;
    match ssss {
        0 => Ok(0),
        // 16 is the one special category: the difference is exactly 32768 and
        // no extra bits follow. A decoder that read sixteen bits here would be
        // one sample out for the rest of the scan.
        16 => Ok(32768),
        n if n < 16 => {
            let raw = bits.bits(n)?;
            // T.81's sign extension: the top bit set means positive, clear
            // means negative and the value is biased.
            Ok(if raw < (1 << (n - 1)) {
                raw - (1 << n) + 1
            } else {
                raw
            })
        }
        n => Err(PluginError::failed(format!(
            "this frame's lossless JPEG names magnitude category {n}, and the \
             format defines 0 to 16"
        ))),
    }
}

/// Step over an `RSTn` marker.
fn skip_restart(bits: &mut Bits) -> Result<(), PluginError> {
    // The reader stopped at the `FF`; the marker is the byte after it.
    while bits.b.get(bits.at) == Some(&0xFF) {
        bits.at += 1;
    }
    match bits.b.get(bits.at) {
        Some(0xD0..=0xD7) => {
            bits.at += 1;
            Ok(())
        }
        _ => Err(PluginError::failed(
            "this frame's lossless JPEG is missing a restart marker its own \
             restart interval promised",
        )),
    }
}

/// Walk the markers, collecting the Huffman tables and the scan parameters.
fn headers(b: &[u8]) -> Result<(Scan, Vec<Option<Huffman>>), PluginError> {
    let mut tables: Vec<Option<Huffman>> = vec![None; 16];
    let mut frame: Option<(usize, usize, u8, Vec<u8>)> = None;
    let mut restart_interval = 0usize;
    let mut i = 0usize;

    let short = || PluginError::failed("this frame's lossless JPEG header is truncated");
    while i + 1 < b.len() && !(b[i] == 0xFF && b[i + 1] == 0xD8) {
        i += 1;
    }
    i += 2;
    loop {
        while b.get(i) == Some(&0xFF) {
            i += 1;
        }
        let marker = *b.get(i).ok_or_else(short)?;
        i += 1;
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        let len = b
            .get(i..i + 2)
            .map(|s| u16::from_be_bytes([s[0], s[1]]) as usize)
            .ok_or_else(short)?;
        let body = b.get(i + 2..i + len).ok_or_else(short)?;
        match marker {
            // SOF3: lossless, Huffman.
            0xC3 => {
                let p = *body.first().ok_or_else(short)?;
                let y = u16::from_be_bytes([body[1], body[2]]) as usize;
                let x = u16::from_be_bytes([body[3], body[4]]) as usize;
                let nf = *body.get(5).ok_or_else(short)? as usize;
                let ids = (0..nf)
                    .map(|k| body.get(6 + 3 * k).copied().ok_or_else(short))
                    .collect::<Result<Vec<u8>, _>>()?;
                frame = Some((x, y, p, ids));
            }
            // DHT: one or more tables, back to back.
            0xC4 => {
                let mut at = 0usize;
                while at < body.len() {
                    let tc_th = *body.get(at).ok_or_else(short)?;
                    // Only the DC/lossless class exists here (Tc == 0); an AC
                    // table in a lossless stream would be an encoder bug.
                    let th = (tc_th & 0x0F) as usize;
                    let counts: [u8; 16] = body
                        .get(at + 1..at + 17)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(short)?;
                    let total: usize = counts.iter().map(|&c| c as usize).sum();
                    let values = body
                        .get(at + 17..at + 17 + total)
                        .ok_or_else(short)?
                        .to_vec();
                    if th < tables.len() {
                        tables[th] = Some(Huffman::new(&counts, values));
                    }
                    at += 17 + total;
                }
            }
            // DRI: the restart interval, in samples for a lossless scan.
            0xDD => {
                restart_interval =
                    u16::from_be_bytes([*body.first().ok_or_else(short)?, body[1]]) as usize;
            }
            // SOS: the scan, and then the data.
            0xDA => {
                let ns = *body.first().ok_or_else(short)? as usize;
                let mut table_of = Vec::with_capacity(ns);
                for k in 0..ns {
                    // Each entry is (Cs, Td<<4 | Ta); the DC table is the high
                    // nibble, and a lossless scan uses only that one.
                    let td_ta = *body.get(2 + 2 * k).ok_or_else(short)?;
                    table_of.push((td_ta >> 4) as usize);
                }
                let ss = *body.get(1 + 2 * ns).ok_or_else(short)?;
                let al = *body.get(3 + 2 * ns).ok_or_else(short)? & 0x0F;
                let (width, height, precision, ids) = frame.ok_or_else(|| {
                    PluginError::failed(
                        "this frame's JPEG starts a scan before saying what \
                             the frame is",
                    )
                })?;
                return Ok((
                    Scan {
                        width,
                        height,
                        precision,
                        components: ids.len(),
                        predictor: ss,
                        point_transform: al,
                        table_of,
                        data_at: i + len,
                        restart_interval,
                    },
                    tables,
                ));
            }
            0xD9 => {
                return Err(PluginError::failed(
                    "this frame's lossless JPEG ends without a scan",
                ))
            }
            _ => {}
        }
        i += len.max(2);
    }
}

#[cfg(test)]
#[path = "lossless_tests.rs"]
mod tests;

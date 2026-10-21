//! Real acquisitions, read both ways: mapped and buffered must agree on every
//! byte.
//!
//! Ignored by default, because it needs files this repository cannot contain:
//! OIR samples are unpublished research data. Point it at some to run it —
//!
//! ```text
//! FASTTIFF_OIR_SAMPLE="a.oir;b.oir" cargo test -p oir-reader --release --test real_file -- --ignored --nocapture
//! ```
//!
//! Each acquisition is read whole, twice, and only a hash of each is kept: the
//! largest real ones hold more pixels than fit in memory twice over.

use oir_reader::{Access, Options, Planes};
use std::time::Instant;

/// FNV-1a, over everything a read produces that a caller sees.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn eat(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

fn fingerprint(path: &str, access: Access) -> u64 {
    let started = Instant::now();
    let a = Options::new()
        .access(access)
        .gather(path, |_| true)
        .and_then(oir_reader::Gathered::assemble)
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    let took = started.elapsed();
    let mut h = Fnv::new();
    h.eat(format!("{:?}", a.shape()).as_bytes());
    h.eat(format!("{:?}", a.record()).as_bytes());
    h.eat(a.fluoview_text("sample.oir").unwrap_or_default().as_bytes());
    h.eat(&a.dropped_planes().to_le_bytes());
    match a.planes() {
        Planes::U8(p) => p.iter().for_each(|p| h.eat(p)),
        Planes::U16(p) => p.iter().for_each(|p| {
            for v in p {
                h.eat(&v.to_le_bytes());
            }
        }),
        _ => panic!("a sample type this test does not know"),
    }
    for (n, part) in a.parts().iter().enumerate() {
        eprintln!(
            "  {access:?} part {}: {:?}, {:?}, {} blocks{}",
            n + 1,
            part.read_path(),
            part.block_source(),
            part.block_count(),
            part.map_error()
                .map(|e| format!(", not mapped: {e}"))
                .unwrap_or_default()
        );
    }
    let s = a.shape();
    eprintln!(
        "  {access:?}: {}x{} c{} z{} t{} in {took:.2?}, hash {:016x}",
        s.width, s.height, s.channels, s.slices, s.frames, h.0
    );
    h.0
}

#[test]
#[ignore = "needs real OIR files; set FASTTIFF_OIR_SAMPLE"]
fn mapped_and_buffered_reads_of_a_real_file_agree() {
    let Ok(list) = std::env::var("FASTTIFF_OIR_SAMPLE") else {
        eprintln!("set FASTTIFF_OIR_SAMPLE to one or more .oir files, separated by ';'");
        return;
    };
    for path in list.split(';').filter(|p| !p.is_empty()) {
        eprintln!("{path}");
        let mapped = fingerprint(path, Access::Auto);
        let buffered = fingerprint(path, Access::Buffered);
        assert_eq!(mapped, buffered, "{path}: the two reads differ");
    }
}

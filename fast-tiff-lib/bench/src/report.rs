//! Turning measurements into something a person can read: the per-run tables,
//! the closing summary, and the one CSV everything downstream is drawn from.
//!
//! There is a single results file, with a row per (family, frame count,
//! reader). Both readings the benchmark exists to give come out of it by
//! grouping differently:
//!
//! - group by **run** — how the readers compare on one kind of file;
//! - group by **family** and read along `frames` — how each reader scales.
//!
//! That is why the sweep is no longer a separate mode with a separate schema:
//! it was the second grouping, re-measured.

use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;

use crate::matrix::Run;
use crate::measure::{best_tiff_mean, geomean, Measured, Outcome};
use crate::reader::Reader;

/// One reader on one run, flattened for the summary and the CSV.
pub struct Row {
    pub family: String,
    pub run: String,
    pub format: &'static str,
    pub compression: &'static str,
    pub predictor: bool,
    pub strips: usize,
    pub bigtiff: bool,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub reader: Reader,
    pub ok: bool,
    pub reason: String,
    pub open_us: f64,
    pub mean_us: f64,
    pub min_us: f64,
    pub total_read_ms: f64,
    pub mb_s: f64,
    /// Trimmed mean over the fastest TIFF reader's. 1.0 = fastest; the floor
    /// is usually below 1 because it does no decoding.
    pub rel: f64,
    pub write_mb_s: f64,
}

/// Fold one run's outcomes into rows.
pub fn rows_for(run: &Run, outcomes: &[Outcome], strips: usize, write_mb_s: f64) -> Vec<Row> {
    let best = best_tiff_mean(outcomes);
    let f = &run.family;
    outcomes
        .iter()
        .map(|o| {
            let (ok, reason, open_us, mean_us, min_us, total_ms, mb_s, rel) = match o {
                Outcome::Measured(m) => (
                    true,
                    String::new(),
                    m.open_us,
                    m.mean_us(),
                    m.min_us(),
                    m.total_read_ms(),
                    m.throughput_mb_s(),
                    m.mean_us() / best,
                ),
                Outcome::Unsupported { reason, .. } => {
                    (false, reason.clone(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                }
            };
            Row {
                family: f.label(),
                run: run.label(),
                format: f.format.label(),
                compression: f.compression_label(),
                predictor: f.predictor,
                strips,
                bigtiff: f.bigtiff,
                width: f.width,
                height: f.height,
                frames: run.frames,
                reader: o.reader(),
                ok,
                reason,
                open_us,
                mean_us,
                min_us,
                total_read_ms: total_ms,
                mb_s,
                rel,
                write_mb_s,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------

const RULE: &str = "------------------------------------------------------------------------";

pub fn print_run(run: &Run, outcomes: &[Outcome], strips: usize, write_mb_s: f64) {
    let mb = run.pixel_bytes() as f64 / (1024.0 * 1024.0);
    println!("\n========================================================================");
    println!("  {}", run.label());
    println!("  {}", run.family.asks);
    println!("  {mb:.1} MB decoded | {strips} strip(s)/frame | written at {write_mb_s:.0} MB/s");
    println!("{RULE}");
    println!("  {:<24} {:>12} {:>12} {:>13}", "reader", "mean us/fr", "min us/fr", "MB/s");
    println!("{RULE}");

    let mut done: Vec<&Measured> = outcomes.iter().filter_map(Outcome::measured).collect();
    done.sort_by(|a, b| a.mean_us().partial_cmp(&b.mean_us()).expect("no NaN"));
    let best = best_tiff_mean(outcomes);

    for m in &done {
        let rel = m.mean_us() / best;
        let note = if m.reader.is_floor() {
            format!("  {rel:.2}x (no-decode floor)")
        } else if (rel - 1.0).abs() < 1e-9 {
            "  <- fastest reader".to_string()
        } else {
            format!("  {rel:.2}x")
        };
        println!(
            "  {:<24} {:>12.2} {:>12.2} {:>13.1}{note}",
            m.reader.label(),
            m.mean_us(),
            m.min_us(),
            m.throughput_mb_s(),
        );
    }
    for o in outcomes {
        if let Outcome::Unsupported { reader, reason } = o {
            println!("  {:<24} {:>12} ({reason})", reader.label(), "n/s");
        }
    }

    print!("  checksums:");
    for m in &done {
        print!("  {}={:#010x}", m.reader.short(), m.checksum as u32);
    }
    println!();
}

/// Everything the run found, in the order a reader wants it: who won overall,
/// how each reader scales, and what the writer managed.
pub fn print_summary(rows: &[Row]) {
    let runs: BTreeSet<&str> = rows.iter().map(|r| r.run.as_str()).collect();
    let families: BTreeSet<&str> = rows.iter().map(|r| r.family.as_str()).collect();

    println!("\n########################################################################");
    println!(
        "  SUMMARY   {} runs = {} families x frame counts",
        runs.len(),
        families.len()
    );
    println!("########################################################################");

    print_relative_speed(rows);
    print_scaling(rows);
    print_write_throughput(rows);

    println!("\n  Per-run data: bench_results.csv");
    println!("  Figures:      python plot.py");
}

/// How the readers compare across everything, as a table and a bar.
fn print_relative_speed(rows: &[Row]) {
    println!("\n  RELATIVE SPEED  (1.00x = fastest TIFF reader in each run; lower is better)");
    println!(
        "  {:<24} {:>5} {:>6} {:>9} {:>9} {:>9} {:>11}",
        "reader", "runs", "wins", "geomean", "median", "worst", "mean MB/s"
    );
    println!("  {RULE}");

    struct Agg {
        reader: Reader,
        geo: f64,
        runs: usize,
    }
    let mut aggs = Vec::new();

    for reader in Reader::ALL {
        let mine: Vec<&Row> = rows.iter().filter(|r| r.reader == reader && r.ok).collect();
        if mine.is_empty() {
            continue;
        }
        let mut rels: Vec<f64> = mine.iter().map(|r| r.rel).collect();
        rels.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let wins = rels.iter().filter(|&&r| (r - 1.0).abs() < 1e-9).count();
        let geo = geomean(&rels);
        let mean_mbs = mine.iter().map(|r| r.mb_s).sum::<f64>() / mine.len() as f64;
        println!(
            "  {:<24} {:>5} {:>6} {:>8.2}x {:>8.2}x {:>8.2}x {:>11.0}",
            reader.label(),
            rels.len(),
            wins,
            geo,
            rels[rels.len() / 2],
            rels.last().copied().unwrap_or(0.0),
            mean_mbs
        );
        aggs.push(Agg { reader, geo, runs: rels.len() });
    }

    aggs.sort_by(|a, b| a.geo.partial_cmp(&b.geo).expect("no NaN"));
    println!();
    let cap = 8.0;
    let widest = aggs.iter().map(|a| a.geo.min(cap)).fold(1.0f64, f64::max);
    for a in &aggs {
        let len = ((a.geo.min(cap) / widest) * 46.0).round().max(1.0) as usize;
        let tail = if a.reader.is_floor() { "  (no-decode floor)" } else { "" };
        println!(
            "  {:<24} {} {:.2}x  ({} runs){tail}",
            a.reader.label(),
            "#".repeat(len),
            a.geo,
            a.runs
        );
    }
}

/// The scaling reading: what a frame costs as a stack gets longer, on the
/// family built to show it. This is what the old separate `sweep` mode
/// re-measured; it is the same rows, grouped the other way.
fn print_scaling(rows: &[Row]) {
    let Some(family) = rows.first().map(|r| r.family.clone()) else { return };
    let mut counts: Vec<usize> =
        rows.iter().filter(|r| r.family == family).map(|r| r.frames).collect();
    counts.sort_unstable();
    counts.dedup();
    if counts.len() < 2 {
        return;
    }

    println!("\n  SCALING  on {family}");
    println!("  open+index cost vs per-frame read, as the stack gets longer");
    print!("  {:<24}", "reader");
    for n in &counts {
        print!("{:>13}", fmt_count(*n));
    }
    println!();
    println!("  {RULE}");

    for reader in Reader::ALL {
        let mut cells = Vec::new();
        for n in &counts {
            let cell = rows
                .iter()
                .find(|r| r.family == family && r.frames == *n && r.reader == reader && r.ok)
                .map(|r| format!("{:.2}", r.mean_us))
                .unwrap_or_else(|| "-".into());
            cells.push(cell);
        }
        if cells.iter().all(|c| c == "-") {
            continue;
        }
        print!("  {:<24}", format!("{} us/fr", reader.short()));
        for c in &cells {
            print!("{c:>13}");
        }
        println!();
    }

    // The open cost is the other half of the trade, and the one a per-frame
    // number hides entirely.
    for reader in [Reader::FastTiff, Reader::TiffRs] {
        let mut any = false;
        let mut line = format!("  {:<24}", format!("{} open ms", reader.short()));
        for n in &counts {
            match rows
                .iter()
                .find(|r| r.family == family && r.frames == *n && r.reader == reader && r.ok)
            {
                Some(r) => {
                    any = true;
                    line.push_str(&format!("{:>13.2}", r.open_us / 1000.0));
                }
                _ => line.push_str(&format!("{:>13}", "-")),
            }
        }
        if any {
            println!("{line}");
        }
    }
}

fn print_write_throughput(rows: &[Row]) {
    println!("\n  WRITE THROUGHPUT  (fast-tiff-lib's writer produced every stack)");
    let mut codecs: Vec<&str> = rows.iter().map(|r| r.compression).collect();
    codecs.sort_unstable();
    codecs.dedup();
    for codec in codecs {
        let v: Vec<f64> = rows
            .iter()
            .filter(|r| r.compression == codec && r.reader == Reader::FastTiff)
            .map(|r| r.write_mb_s)
            .collect();
        if !v.is_empty() {
            println!("    {:<10} {:>8.0} MB/s", codec, v.iter().sum::<f64>() / v.len() as f64);
        }
    }
}

/// 1000 -> "1k", 1_000_000 -> "1M". Frame counts are powers of ten here, so
/// this only ever has to be readable, not general.
pub fn fmt_count(n: usize) -> String {
    match n {
        n if n >= 1_000_000 => format!("{}M", n / 1_000_000),
        n if n >= 1_000 => format!("{}k", n / 1_000),
        n => n.to_string(),
    }
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

pub const CSV_HEADER: &str = "family,run,format,compression,predictor,strips,bigtiff,width,height,\
                              frames,reader,ok,reason,open_us,mean_us,min_us,total_read_ms,mb_s,\
                              rel,write_mb_s";

pub fn write_csv(path: &Path, rows: &[Row], env: &[String]) -> Result<()> {
    let mut csv = String::new();
    for line in env {
        csv.push_str(&format!("# {line}\n"));
    }
    csv.push_str(CSV_HEADER);
    csv.push('\n');
    for r in rows {
        csv.push_str(&format!(
            "\"{}\",\"{}\",{},{},{},{},{},{},{},{},{},{},\"{}\",{:.3},{:.4},{:.4},{:.4},{:.3},{:.4},{:.1}\n",
            r.family,
            r.run,
            r.format,
            r.compression,
            r.predictor,
            r.strips,
            r.bigtiff,
            r.width,
            r.height,
            r.frames,
            r.reader.id(),
            r.ok,
            r.reason.replace('"', "'"),
            r.open_us,
            r.mean_us,
            r.min_us,
            r.total_read_ms,
            r.mb_s,
            r.rel,
            r.write_mb_s,
        ));
    }
    std::fs::write(path, csv)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header and every written row must have the same number of fields,
    /// or the CSV is unreadable in the obvious way — silently, with columns
    /// shifted by one.
    #[test]
    fn the_csv_row_matches_its_header() {
        let row = Row {
            family: "16x16 u16 none".into(),
            run: "16x16 u16 none / 10 frames".into(),
            format: "u16",
            compression: "none",
            predictor: false,
            strips: 1,
            bigtiff: false,
            width: 16,
            height: 16,
            frames: 10,
            reader: Reader::FastTiff,
            ok: true,
            reason: String::new(),
            open_us: 1.0,
            mean_us: 2.0,
            min_us: 3.0,
            total_read_ms: 4.0,
            mb_s: 5.0,
            rel: 1.0,
            write_mb_s: 7.0,
        };
        let dir = std::env::temp_dir().join("fast_tiff_bench_csv_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("rows.csv");
        write_csv(&path, std::slice::from_ref(&row), &["env line".into()]).expect("write");
        let text = std::fs::read_to_string(&path).expect("read back");

        let header_fields = CSV_HEADER.split(',').count();
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.starts_with("family")) {
            // Fields are simple or fully quoted, and no quoted field here
            // contains a comma, so a plain split is enough to count them.
            assert_eq!(
                line.split(',').count(),
                header_fields,
                "row has the wrong field count:\n{line}"
            );
        }
        assert!(text.starts_with("# env line"), "the environment header is missing");
        assert!(text.contains("fast-tiff"), "the reader id is missing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn frame_counts_read_as_magnitudes() {
        assert_eq!(fmt_count(1), "1");
        assert_eq!(fmt_count(100), "100");
        assert_eq!(fmt_count(1_000), "1k");
        assert_eq!(fmt_count(10_000), "10k");
        assert_eq!(fmt_count(1_000_000), "1M");
    }
}

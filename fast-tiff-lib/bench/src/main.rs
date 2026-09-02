//! TIFF reader/writer speed benchmark.
//!
//! Compares per-frame read speed across every reader that can open the same
//! stacks, over the whole feature envelope of `fast-tiff-lib` — every sample
//! format, codec, predictor, strip layout, RGB, BigTIFF — each written out at
//! several frame counts. Method follows jkriege2/TinyTIFF's
//! `tinytiffwriter_speedtest`; see [`measure`] for what that means in detail.
//!
//! **One run, two readings.** Every family is crossed with every frame count
//! that fits the size budget, so the same measurements answer both questions
//! the benchmark exists for:
//!
//! - *per run* — on this kind of file, which reader is fastest;
//! - *swept* — as the stack gets longer, how does each reader scale, and how
//!   much of its cost is paid once at open rather than per frame.
//!
//! There used to be two modes writing two CSVs in two schemas, plotted by two
//! scripts, with the second re-measuring one family the first had already
//! covered. Now there is one command, one `bench_results.csv`, and one
//! `plot.py`; the swept reading is a grouping of the rows, not a second run.
//!
//! ```text
//! cargo run --release                    # the matrix
//! cargo run --release -- --quick         # smoke run, two frame counts
//! cargo run --release --features libtiff # include system libtiff
//! python plot.py                         # figures from bench_results.csv
//! ```
//!
//! Module map:
//!
//! | module        | what lives there                                  |
//! |---------------|---------------------------------------------------|
//! | [`reader`]    | who is measured, and their one set of names       |
//! | [`matrix`]    | what is measured: families x frame counts         |
//! | [`measure`]   | how a measurement is taken; timings, checksums    |
//! | [`readers`]   | one function per contender                        |
//! | [`report`]    | tables, summary, the single CSV                   |
//! | [`environment`] | the machine, for the header and the CSV         |
//! | [`ffi`]       | hand-written bindings for the C readers           |

mod environment;
mod ffi;
mod matrix;
mod measure;
mod reader;
mod readers;
mod report;

use anyhow::Result;
use fast_tiff_lib::TiffStack;

use report::Row;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    let quick = args.iter().any(|a| a == "--quick");
    if let Some(unknown) = args.iter().find(|a| a.as_str() != "--quick") {
        eprintln!("unknown argument '{unknown}'\n");
        print_help();
        std::process::exit(2);
    }

    let env = environment::describe();
    println!("TIFF reader/writer speed benchmark{}", if quick { "   [--quick]" } else { "" });
    println!("========================================================================");
    for line in &env {
        println!("  {line}");
    }
    println!("\n  readers:");
    for r in reader::Reader::ALL {
        println!("    {:<24} {}", r.label(), r.how());
    }
    println!("========================================================================");

    // Steady-state single-frame latency, which is the viewer's scrubbing
    // regime. The batch reader turns this on for itself inside its own call.
    fast_tiff_lib::set_parallel_decode(false);

    let scratch = measure::scratch_dir();
    std::fs::create_dir_all(&scratch)?;
    let runs = matrix::runs(quick);
    println!("\n  {} runs; scratch dir {}", runs.len(), scratch.display());

    let mut rows: Vec<Row> = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        println!("\n[{}/{}]", i + 1, runs.len());
        let stacks = measure::write_stacks(&scratch, run)?;
        let write_mb_s = (run.pixel_bytes() as f64 / (1024.0 * 1024.0)) / stacks.write_secs;

        measure::warm_cache(&stacks.tiff)?;
        measure::warm_cache(&stacks.raw)?;

        let strips = TiffStack::open(&stacks.tiff)?.frames[0].strip_offsets.len();
        let outcomes = readers::run_all(&stacks.tiff, &stacks.raw, run)?;

        report::print_run(run, &outcomes, strips, write_mb_s);
        rows.extend(report::rows_for(run, &outcomes, strips, write_mb_s));

        // Removed as we go: the matrix peaks around 7.5 GB on disk for one run,
        // and keeping them all would need far more than any of them does.
        stacks.remove();
    }

    report::print_summary(&rows);
    let csv = std::path::Path::new("bench_results.csv");
    report::write_csv(csv, &rows, &env)?;
    println!("\nwrote {}", csv.display());
    Ok(())
}

fn print_help() {
    println!(
        "TIFF reader/writer speed benchmark

USAGE
    cargo run --release [-- --quick]

OPTIONS
    --quick     Two frame counts instead of seven; a smoke run, still enough
                for every chart to have a line rather than a point.
    -h, --help  This text.

FEATURES
    --features libtiff    Also measure the system libtiff (needs headers).

ENVIRONMENT
    TIFF_BENCH_DIR        Where generated stacks go. The biggest runs peak
                          around 7.5 GB; point this at a roomier volume if the
                          system drive is tight.

OUTPUT
    bench_results.csv     One row per (family, frame count, reader).
    python plot.py        Renders the figures from that file."
    );
}

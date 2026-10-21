//! Read an OIR acquisition and say what is in it, and how it was read.
//!
//! ```text
//! cargo run --release --example oir_info -- [--buffered] [--text] <file.oir>
//! ```
//!
//! `--buffered` reads every part through the buffer instead of mapping it, and
//! `--text` prints the record as FluoView would have exported it.
//!
//! Also the way to time the two paths against each other. Read the same file
//! twice and the second read comes from the page cache, so alternate which
//! path goes first, or drop the cache between runs.

use oir_reader::{Access, Options};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let mut access = Access::Auto;
    let mut text = false;
    let mut path = None;
    for arg in std::env::args_os().skip(1) {
        match arg.to_str() {
            Some("--buffered") => access = Access::Buffered,
            Some("--text") => text = true,
            _ => path = Some(arg),
        }
    }
    let Some(path) = path else {
        eprintln!("usage: oir_info [--buffered] [--text] <file.oir>");
        return ExitCode::from(2);
    };

    let started = Instant::now();
    let gathered = match Options::new().access(access).gather(&path, |_| true) {
        Ok(g) => g,
        Err(e) => return fail(&e),
    };
    let read = started.elapsed();
    for (n, part) in gathered.parts().iter().enumerate() {
        println!(
            "part {}: {} — {} bytes, {:?}, {} blocks from the {:?}, {} new plane(s)",
            n + 1,
            part.path().map_or("?".into(), |p| p.display().to_string()),
            part.size(),
            part.read_path(),
            part.block_count(),
            part.block_source(),
            part.planes_added()
        );
        if let Some(why) = part.map_error() {
            println!("        could not be mapped: {why}");
        }
    }
    let acquisition = match gathered.assemble() {
        Ok(a) => a,
        Err(e) => return fail(&e),
    };
    let total = started.elapsed();

    let s = acquisition.shape();
    println!(
        "{}x{}, {} channel(s), {} slice(s), {} frame(s), {:?}; {} plane(s) dropped as incomplete",
        s.width,
        s.height,
        s.channels,
        s.slices,
        s.frames,
        s.sample_type,
        acquisition.dropped_planes()
    );
    let r = acquisition.record();
    println!(
        "{} um/pixel, {} um/slice, {} s/frame",
        show(r.pixel_x_um()),
        show(r.z_step_um()),
        show(acquisition.frame_interval_s())
    );
    for c in r.channels() {
        println!(
            "channel {}: {} bit(s) significant",
            c.label().unwrap_or("?"),
            c.bits().unwrap_or("?")
        );
    }
    for e in r.events() {
        println!("event {} at {} ms", e.name(), e.time_ms());
    }
    let bytes = s.plane_len() * s.plane_count() * s.sample_type.bytes();
    println!(
        "read in {:.2?}, assembled in {:.2?}: {:.0} MB of pixels, {:.0} MB/s",
        read,
        total - read,
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 / total.as_secs_f64()
    );
    if text {
        let name = std::path::Path::new(&path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        match acquisition.fluoview_text(&name) {
            Some(t) => print!("{t}"),
            None => println!("(no acquisition record this reader recognises)"),
        }
    }
    ExitCode::SUCCESS
}

fn show(v: Option<f64>) -> String {
    v.map_or("?".into(), |v| v.to_string())
}

fn fail(e: &oir_reader::Error) -> ExitCode {
    match std::error::Error::source(e) {
        Some(cause) => eprintln!("{e}: {cause}"),
        None => eprintln!("{e}"),
    }
    if let Some(p) = e.path() {
        eprintln!("  in {}", p.display());
    }
    ExitCode::FAILURE
}

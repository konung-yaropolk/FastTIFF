//! What machine produced these numbers.
//!
//! Printed above every run and embedded as `#` comments at the top of the CSV,
//! so a results file is self-describing: a benchmark figure without the machine
//! it came from is not a measurement, it is an anecdote.

/// One line each: OS, CPU, RAM, toolchain, and the version of every library in
/// the comparison.
pub fn describe() -> Vec<String> {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_all();
    let cpu = sys.cpus().first();

    // `mut` is used only when the libtiff feature appends its version.
    #[allow(unused_mut)]
    let mut lines = vec![
        format!(
            "OS:        {} {} ({})",
            System::name().unwrap_or_default(),
            System::os_version().unwrap_or_default(),
            System::cpu_arch().unwrap_or_default()
        ),
        format!(
            "CPU:       {} ({} physical / {} logical cores, {} MHz)",
            cpu.map(|c| c.brand().trim().to_string()).unwrap_or_default(),
            sys.physical_core_count().unwrap_or(0),
            sys.cpus().len(),
            cpu.map(|c| c.frequency()).unwrap_or(0)
        ),
        format!("RAM:       {:.1} GiB", sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)),
        format!("toolchain: {}", env!("BENCH_RUSTC_VERSION")),
        format!("fast-tiff-lib: {} (path dependency)", env!("FAST_TIFF_LIB_VERSION")),
        format!(
            "tiff (rust): 0.11 | TinyTIFF: vendored | libtiff: {}",
            if cfg!(libtiff) { "linked" } else { "not found on this machine" }
        ),
    ];

    #[cfg(libtiff)]
    unsafe {
        let v = crate::ffi::libtiff::TIFFGetVersion();
        if !v.is_null() {
            let s = std::ffi::CStr::from_ptr(v).to_string_lossy();
            lines.push(format!("libtiff:   {}", s.lines().next().unwrap_or("").trim()));
        }
    }
    lines
}

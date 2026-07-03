# TIFF reader/writer speed benchmark

Compares per-frame **read speed** (and, as a side effect, fast-tiff-lib's
**write speed**) across the whole feature envelope of `fast-tiff-lib`, on
identical multi-frame TIFF stacks:

| Reader                | Source                            | How it reads a frame                         |
|-----------------------|-----------------------------------|----------------------------------------------|
| `fast-tiff-lib`       | this repo (path dep), mmap        | `read_frame_u8/u16/f32`, `read_planes_*` (RGB) |
| `fast-tiff preload`   | same, batch mode                  | `preload_frames_*` (rayon, all frames at once) |
| `tiff-rs`             | crates.io `tiff` 0.11, pure Rust  | `Decoder::read_image` per IFD                |
| `TinyTIFF (C)`        | jkriege2/TinyTIFF, vendored, FFI  | `readNext` + `getSampleData` (uncompressed only) |
| `libtiff (C)`         | system libtiff, FFI, **optional** | `TIFFReadEncodedStrip` per strip             |
| `RAW fread`           | plain `std::fs` reads             | sequential read of decoded-size bytes (floor) |

Methodology follows jkriege2/TinyTIFF's speedtest: per-frame `Instant` timing,
the slowest 10% trimmed before averaging, page cache pre-warmed, an FNV-1a
checksum accumulated so the optimizer can't elide decodes. Every reader
decodes into an **owned host buffer** (mmap borrows are forced to
`.into_owned()`), so zero-copy paths can't "win" by doing no work.

## What the matrix covers

Every stack is **written by fast-tiff-lib's own `TiffWriter`** (which the
crate's test suite cross-validates against libtiff/tifffile-generated
fixtures), so the write timing per configuration is reported for free.

Axes, crossed with **several frame counts each** (40 / 160 / 640 at 256²,
8 / 24 at 2048²):

- **Sample formats**: u8, u16, f32, chunky RGB8, chunky RGB16
- **Codecs**: none, LZW, Deflate, PackBits, ZSTD
- **Predictor**: horizontal (2) on integers, floating-point (3) on f32
- **Strip layouts**: single-strip (the zero-copy fast path) and multi-strip
- **BigTIFF** (magic 43)
- **Frame sizes**: 256×256 (per-frame overhead visible) and 2048×2048
  (pixel-throughput bound)

Readers that can't handle a configuration are reported as `n/s` with the
reason (e.g. TinyTIFF on compressed input, tiff-rs on ZSTD) rather than
silently skipped.

## Running

```bash
cargo run --release                      # matrix mode (formats x codecs x frames)
cargo run --release -- sweep             # frame-count sweep (tiny frames, 10..1M)
cargo run --release -- matrix --quick    # fast smoke run
cargo run --release --features libtiff   # also bench system libtiff (needs libtiff-dev)
```

The header prints the **machine that ran the bench**: OS, CPU model,
physical/logical cores, frequency, RAM, the exact rustc, and every library
version. The same lines are embedded as `#` comments in the CSVs, so results
files are self-describing.

## Output

- **Per-run tables** — mean/min µs per frame, MB/s, relative speed. Relative
  numbers are vs the **fastest TIFF reader** in that run; `RAW fread` is
  labeled as the no-decode floor (its relative speed is usually < 1).
- **Overall summary** — per reader: runs supported, wins, geometric-mean /
  median / worst relative speed, mean throughput; an ASCII bar infographic;
  fast-tiff-lib write throughput by codec.
- **`bench_results.csv`** — every (config × frames × reader) data point.
- **Figures** — one command renders all of them:

```bash
python plot_results.py               # defaults: bench_results.csv -> PNGs
# or: python plot_results.py bench_results.csv bench_summary.png graphs/
```

  - `bench_summary.png` — the overall four-panel infographic (relative speed,
    throughput by codec, frame-count scaling, environment/writer summary).
  - `graphs/all_tests.png` — a **compilation**: one mini bar chart per test,
    all in a single grid, so every configuration's reader ranking is visible
    at a glance.
  - `graphs/tests/NN_<config>.png` — a **separate, labeled bar chart for each
    individual test** (mean µs/frame per reader, with the relative multiplier
    and any `n/s` readers noted).

The sweep has its own plots:

```bash
cargo run --release -- sweep
python plot_sweep.py sweep_results.csv graphs/
```

## Reading the results honestly

- **Relative-to-fastest is per run.** RAW fread does no decode work, so on
  compressed configs everything is 10–100× "slower" than it — that gap is the
  decompression itself, which is why RAW is a floor, not a competitor.
- **Checksum domains differ by design.** RAW/TinyTIFF/libtiff checksum raw
  file bytes; fast-tiff-lib and tiff-rs checksum decoded samples (u16/f32
  values), and fast-tiff's u16 path applies its display transforms (signed
  offset etc.). Readers within the same domain must agree — and do.
- **`fast-tiff preload` is throughput, not latency**: one rayon-parallel call
  decodes the whole stack; its "per-frame" number is total/frames. It shines
  on compressed stacks (parallel across frames) and small-frame overheads
  disappear into batch cost on tiny runs.
- **Windows note:** forcing owned buffers penalizes the mmap design here —
  every first touch of a mapped page soft-faults, which on Windows costs
  noticeably more than a buffered `read` into a reused buffer. The viewer's
  real path uploads straight from the borrow (no copy, no second pass), so
  these numbers are a *lower bound* on fast-tiff-lib's real-world advantage
  for uncompressed scrubbing. On Linux the fault cost is far smaller and the
  single-strip rows historically tie or beat libtiff.
- The **sweep** isolates the open-vs-read tradeoff: fast-tiff-lib indexes the
  whole IFD chain up front (open cost grows with frame count) and then reads
  frames ~10–20× faster than the lazy readers. Open once + read many — the
  viewer's workload — amortizes the indexing; a single sequential pass over a
  huge stack may not.

## Prerequisites

- Rust toolchain + a C compiler (MSVC, gcc, or clang — the vendored TinyTIFF
  is built by `build.rs` via the `cc` crate).
- `--features libtiff` additionally needs a system libtiff with headers
  (`sudo apt-get install libtiff-dev`; linked via pkg-config, falling back to
  `-ltiff`). Off by default so the bench builds out of the box on Windows.
- Plotting: `pip install matplotlib`.

## Layout

```
bench/
├── Cargo.toml            # standalone package ([workspace] detaches it)
├── build.rs              # compiles vendored TinyTIFF; links libtiff (feature)
├── src/
│   ├── main.rs           # writer, timers, readers, matrix + sweep, reporting
│   └── ffi.rs            # hand-written FFI for TinyTIFF (+ libtiff, feature)
├── plot_results.py       # bench_results.csv -> single-figure infographic
├── plot_sweep.py         # sweep_results.csv -> sweep graphs
└── vendor/tinytiff/      # TinyTIFF reader C sources (LGPL-3.0)
```

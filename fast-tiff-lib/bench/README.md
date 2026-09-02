# TIFF read/write speed benchmark

How fast `fast-tiff-lib` reads a frame, against every other reader that can
open the same file — over the whole feature envelope of the library, at frame
counts from one to a million.

```bash
cargo run --release        # the whole matrix
python plot.py             # the figures
```

That is the entire workflow. One command measures, one draws.

## The two questions, from one run

Every configuration is written out at several frame counts, so the same
measurements answer both things a reader wants to know:

- **Per run** — on *this kind of file*, which reader is fastest?
- **Swept** — as the stack gets longer, how does each reader *scale*, and how
  much of its cost is paid once at open rather than per frame?

The second used to be a separate `sweep` mode that re-measured one
configuration and wrote its own CSV in its own schema, plotted by its own
script. It measured nothing the matrix was not already measuring. There is now
one run, one `bench_results.csv`, and one `plot.py`; the swept reading is a
*grouping* of the rows.

## Who is measured

| reader                    | source                             | how it reads a frame                          |
|---------------------------|------------------------------------|-----------------------------------------------|
| `fast-tiff-lib`           | this repo (path dep), mmap         | `read_frame_*_into`, `read_planes_*_into`     |
| `fast-tiff-lib (preload)` | **the same crate**, batch API      | `preload_frames_*`, rayon, whole stack at once |
| `tiff-rs`                 | crates.io `tiff` 0.11, pure Rust   | `Decoder::read_image` per IFD                 |
| `TinyTIFF (C)`            | jkriege2/TinyTIFF, vendored, FFI   | `readNext` + `getSampleData` (uncompressed)   |
| `libtiff (C)`             | system libtiff, FFI, auto-detected | `TIFFReadEncodedStrip` per strip              |
| `RAW fread`               | plain `std::fs`                    | sequential read, no decode — the floor        |

The first two are **one library in two modes**, not two projects. Their names
share a stem so a chart cannot be misread as a five-way comparison between
libraries when it is four libraries and two APIs.

Those names live in exactly one place, `src/reader.rs`, as an enum with a
label, a short form and a machine id. They used to be free-text strings
repeated at four sites each, and they had drifted: the CSV wrote
`fast-tiff-preload` while the plotting script looked for
`fast-tiff-lib (preload)`, so that series silently lost its colour and its
place in the reader order.

## Method

Follows jkriege2/TinyTIFF's `tinytiffwriter_speedtest`: per-frame `Instant`
timing, slowest 10% trimmed before averaging, page cache pre-warmed, an FNV-1a
checksum accumulated so the optimiser cannot elide the decode.

Two rules keep it honest, and both are enforced in `src/measure.rs`:

- **Every reader decodes into an owned host buffer.** A zero-copy mmap borrow
  would otherwise "win" by not doing the work the others do.
- **Open is timed separately from reads.** A reader that indexes the whole IFD
  chain up front pays at open and reads quickly after; one that walks lazily
  pays per frame. Summing them hides the trade, and the trade is the point.

Decode parallelism is off for the per-frame readers — steady-state single-frame
latency, which is the viewer's scrubbing regime. The preload reader turns it on
inside its own call, which is what that API is for.

## What is covered

Every stack is written by fast-tiff-lib's own `TiffWriter` (cross-validated
against libtiff/tifffile fixtures in the crate's test suite), so write
throughput per codec is reported for free.

- **Formats** — u8, u16, f32, chunky RGB8, chunky RGB16
- **Codecs** — none, LZW, Deflate, PackBits, Zstd
- **Predictors** — horizontal (2) on integers, floating-point (3) on f32
- **Layouts** — single-strip (the zero-copy path) and multi-strip
- **BigTIFF**
- **Frame sizes** — 16×16 (per-frame overhead with pixels out of the way),
  256×256, 2048×2048 (throughput-bound)

Crossed with **1 / 10 / 100 / 1k / 10k / 100k / 1M** frames, capped by a 4 GiB
per-stack budget: the 256² families reach 10k, 2048² reaches 100, and only the
16×16 family covers all seven decades.

A reader that cannot handle a configuration is reported as `n/s` **with its
reason**, never dropped silently — a gap in a chart should always be
explainable from the CSV.

## Output

`bench_results.csv` — one row per (family, frame count, reader), with the
machine embedded as `#` comments so a results file is self-describing.

`python plot.py` renders three things:

- **`bench_summary.png`** — the headline. Overall read speed, throughput by
  codec, per-frame cost as the stack grows, cost paid once at open, the
  machine, and what the writer managed.
- **`graphs/scaling.png`** — one panel per configuration, µs/frame against
  frame count. Every configuration's scaling on a single sheet.
- **`graphs/runs/NN_*.png`** — one labelled bar chart per run, with the
  unsupported readers and their reasons noted underneath.

## Reading the results honestly

- **Relative speed is per run, against the fastest *TIFF reader*.** RAW does no
  decode, so on compressed configurations everything is many times "slower"
  than it — that gap is the decompression, which is why RAW is a floor and not
  a competitor.
- **Checksum domains differ by design.** RAW/TinyTIFF/libtiff checksum raw file
  bytes; fast-tiff-lib and tiff-rs checksum decoded samples. Readers within a
  domain must agree, and do.
- **`preload` is throughput, not latency.** One call decodes the whole stack;
  its "per-frame" figure is total ÷ frames. It shines on compressed stacks and
  looks poor on tiny ones, where batch setup dominates.
- **Windows note.** Forcing owned buffers penalises the mmap design: every
  first touch of a mapped page soft-faults, which costs more on Windows than a
  buffered read into a reused buffer. The viewer's real path uploads straight
  from the borrow, so these are a *lower bound* on the real-world advantage for
  uncompressed scrubbing. On Linux the fault cost is far smaller.
- **The open/read trade is the interesting part of the sweep.** fast-tiff-lib
  indexes the whole IFD chain at open and then reads frames far faster; a lazy
  reader pays as it goes. Open once and read many — the viewer's workload —
  amortises the indexing. A single sequential pass over a huge stack may not.

## Options

```bash
cargo run --release -- --quick          # two frame counts; a smoke run
cargo run --release --features libtiff  # also measure system libtiff
TIFF_BENCH_DIR=/big/disk cargo run --release
```

`--quick` still gives every chart a line rather than a point.

`TIFF_BENCH_DIR` moves the generated stacks off the system drive; the biggest
runs peak around 7.5 GB (a 4 GiB stack plus its raw sibling). Stacks are
deleted as the run proceeds, so only one pair exists at a time.

## Prerequisites

- Rust, plus a C compiler for the vendored TinyTIFF (`cc` builds it in
  `build.rs`).
- **libtiff is used automatically wherever the machine has one.** No feature
  flag, no configuration. `build.rs` searches, in order: `LIBTIFF_DIR` /
  `LIBTIFF_LIB_DIR`, `pkg-config`, vcpkg, then the prefixes package managers
  actually install into — MSYS2 (`clang64`, `ucrt64`, `mingw64`, ...), Homebrew
  on both architectures, MacPorts, conda, and `/usr`. The run header and the
  CSV both record whether it was found, so a result file is never ambiguous
  about what it measured.

  Only the library is needed, never the headers — the bindings in `src/ffi.rs`
  are hand-written.

  | to get one | |
  |---|---|
  | Linux | `apt install libtiff-dev` |
  | macOS | `brew install libtiff` |
  | Windows, MSYS2 | `pacman -S mingw-w64-clang-x86_64-libtiff` |
  | Windows, vcpkg | `vcpkg install tiff:x64-windows-static-md` |

  On Windows an MSYS2 libtiff is a DLL import library, so its `bin` directory
  has to be on `PATH` at run time — `build.rs` prints exactly which one. Copying
  the single DLL next to the exe is not enough: Windows resolves libtiff's own
  imports from the *exe's* directory, and that pulls in a dozen more. The vcpkg
  `-static-md` triplet avoids the question entirely.

  `--features libtiff` does not enable anything; it makes *absence* a build
  error, for CI that requires the comparison to be complete.

### Why libtiff is not vendored, when TinyTIFF is

The obvious question, since `vendor/tinytiff` sits right there. Three reasons,
in increasing order of how much they matter:

- **Size.** TinyTIFF is 8 files and 70 KB, two `.c` files you can read in an
  afternoon. libtiff stripped to its dependency-free core is 43 files and
  1.5 MB — `tif_dirread.c` alone is four times all of TinyTIFF.
- **Configuration.** libtiff generates `tif_config.h` and `tiffconf.h` from
  CMake or autotools. Vendoring means hand-writing both, per platform, plus
  choosing between `tif_win32.c` and `tif_unix.c`.
- **It would not be libtiff.** Every optional codec is `#ifdef`-guarded and
  wires to `_notConfigured()` when its library is missing. With no zlib and no
  libzstd, a vendored build reads none, LZW and PackBits — but **not Deflate or
  Zstd**, which is a third of this matrix. The comparison would be against a
  stripped libtiff that nobody actually runs, which is worse than no comparison.

TinyTIFF has no codecs at all, so a vendored copy *is* TinyTIFF. libtiff's
codecs are most of what it is.

## Layout

```
bench/
├── Cargo.toml           # standalone package ([workspace] detaches it)
├── build.rs             # compiles vendored TinyTIFF; links libtiff (feature)
├── plot.py              # bench_results.csv -> every figure
├── src/
│   ├── main.rs          # CLI and the run loop
│   ├── reader.rs        # who is measured, and their one set of names
│   ├── matrix.rs        # what is measured: families x frame counts
│   ├── measure.rs       # how: timings, trimming, checksums, stack writing
│   ├── readers.rs       # one function per contender
│   ├── report.rs        # tables, summary, the single CSV
│   ├── environment.rs   # the machine, for the header and the CSV
│   └── ffi.rs           # hand-written bindings for the C readers
└── vendor/tinytiff/     # TinyTIFF reader C sources (LGPL-3.0)
```

`cargo test` covers the parts that are logic rather than measurement: the name
table, the matrix's budget and coverage, the trimmed mean, the geometric mean,
the checksums, and the CSV's shape.

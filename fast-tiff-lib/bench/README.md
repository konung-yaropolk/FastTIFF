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
- **Every reader's indexing is timed somewhere.** For the lazy readers the walk
  to the next directory is timed *with the frame that needs it*. It used to sit
  outside both timers, which put a real and large cost in no column at all: on a
  10 000-frame stack libtiff spends **13.6 us per frame** in
  `TIFFReadDirectory`, against the **0.20 us per frame** fast-tiff-lib reports
  at open for the same chain. Unmeasured, that made the reader that indexes
  honestly look like the slow one on every chart that mentions indexing.

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
  buffered read into a reused buffer. On an uncompressed 256x256 u16 frame that
  is 29 of the 37 us — read the same frames twice and the second pass takes
  7.4. The viewer's real path uploads straight from the borrow and can absorb
  the faults on a background thread (`TiffStack::prefetch_frame`), so these are
  a *lower bound* on the real-world advantage for uncompressed scrubbing. On
  Linux the fault cost is far smaller.
- **The open/read trade is the interesting part of the sweep.** fast-tiff-lib
  indexes the whole IFD chain at open and then reads frames far faster; a lazy
  reader pays as it goes, now visibly, in its per-frame column. Open once and
  read many — the viewer's workload — amortises the indexing, and random access
  is where it pays off most: seeking to an arbitrary frame of a 10 000-frame
  stack costs 1.1 us against libtiff's 14.1 us, because `TIFFSetDirectory`
  walks the chain and an index does not.

## Options

```bash
cargo run --release -- --quick          # two frame counts; a smoke run
TIFF_BENCH_DIR=/big/disk cargo run --release
```

`--quick` still gives every chart a line rather than a point.

`TIFF_BENCH_DIR` moves the generated stacks off the system drive; the biggest
runs peak around 7.5 GB (a 4 GiB stack plus its raw sibling). Stacks are
deleted as the run proceeds, so only one pair exists at a time.

## Prerequisites

Rust and a C compiler. Nothing else — no libtiff to install, no headers, no
package manager, no `PATH` to arrange before running.

Both C readers are vendored and built by `build.rs`: TinyTIFF from
`vendor/tinytiff`, libtiff from `vendor/libtiff`. The two codec libraries
libtiff needs come from source as well, through the `libz-sys` and `zstd-sys`
crates, so the vendored build supports every codec this matrix exercises rather
than silently degrading on the compressed half of it.

Only the libraries are built, never their headers — the bindings in
`src/ffi.rs` are hand-written, and the libtiff version reported in the run
header is read out of `vendor/libtiff/libtiff.map` at build time.

`--features libtiff` is accepted and does nothing. It used to mean "fail the
build if no system libtiff was found"; there is no longer anything to find.

### Why libtiff is vendored, and what that took

It was not, for a long time, and the reason is worth keeping: **a vendored
libtiff is only worth having if it is a complete one.** Every optional codec in
libtiff is `#ifdef`-guarded and wires to `_notConfigured()` when its library is
missing — it does not fail to build, it fails at run time, on the files that
need it. Without zlib and libzstd a vendored build reads none, LZW and PackBits
but **not Deflate or Zstd**, which is a third of this matrix. Measuring that
would have been worse than measuring nothing, because the number would have
looked like a libtiff result.

What changed is that both libraries are now built from source too, as ordinary
cargo dependencies (`libz-sys` with its bundled zlib, `zstd-sys` with its
bundled libzstd). That removes the objection, and the two that remained were
only ever costs rather than blockers:

- **Size.** TinyTIFF is 8 files and 70 KB; libtiff is 45 sources, of which this
  build compiles 36 — the rest need JPEG, JBIG, LERC, LZMA or WebP, none of
  which this matrix uses.
- **Configuration.** libtiff generates `tif_config.h`, `tiffconf.h` and
  `tiffvers.h` from CMake or autotools. `build.rs` writes all three directly;
  the probing they do only matters for platforms this benchmark does not target.

What it buys is that every machine measures *the same* libtiff, at a version the
run header states exactly, with no setup step that can be skipped or done
differently.

Two details worth knowing if you touch this. `zstd-sys` is named as a direct
dependency with `default-features = false`: as a direct dependency it turns on
its `bindgen` feature by default, which needs libclang installed — precisely the
kind of prerequisite this is trying to avoid — and it has to be direct anyway,
because cargo passes a `links` crate's `DEP_*` variables only to the build
scripts of crates that depend on it directly. And `src/ffi.rs` names both codec
crates in `extern crate ... as _;` bindings: rustc drops a dependency nothing
references, and drops its `#[link]` with it, which surfaces as
`unresolved external symbol deflate` from libtiff's own objects at the end of
the build rather than as anything about a missing crate.

## Layout

```
bench/
├── Cargo.toml           # standalone package ([workspace] detaches it)
├── build.rs             # compiles vendored TinyTIFF and libtiff
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
└── vendor/
    ├── tinytiff/        # TinyTIFF reader C sources (LGPL-3.0)
    └── libtiff/         # libtiff C sources (libtiff license)
```

`cargo test` covers the parts that are logic rather than measurement: the name
table, the matrix's budget and coverage, the trimmed mean, the geometric mean,
the checksums, and the CSV's shape.

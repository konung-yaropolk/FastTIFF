# Fuzzing `fast-tiff-lib`

`TiffStack::from_bytes` and the `read_*` decoders take **completely untrusted
bytes**. For any input at all, the only acceptable outcomes are `Ok` or `Err`.
Three things are not acceptable:

| Outcome | Why it matters |
| --- | --- |
| **panic** | Unwinds through a library boundary — a DoS for any embedder. |
| **abort** | An oversized `vec![0u8; n]` calls `abort()`. It is **not** catchable, so it kills the whole host process. |
| **unbounded memory** | A few hundred bytes must not be able to demand gigabytes. |

These targets explore for such inputs. Their always-on counterpart is
[`tests/malformed_robustness.rs`](../tests/malformed_robustness.rs), which pins
the cases already found and runs in ordinary CI on every platform with no
nightly toolchain.

## Targets

- **`tiff_open`** — indexing only: header parse, IFD-chain walk, metadata
  (ImageJ `key=value` + binary block, OME-XML). Fast, so it explores the parser
  deeply.
- **`tiff_decode`** — everything `tiff_open` does, then pulls pixels through
  *every* public reader. This is the one that exercises the size arithmetic
  (`width × height × samples_per_pixel × bytes_per_sample` from file-declared
  values), the per-codec strip paths, the predictor undo, and the chunky/planar
  plane gathers.

## Running

Needs nightly (libFuzzer):

```sh
cargo install cargo-fuzz --locked
```

Then, from the `fast-tiff-lib/` directory:

```sh
mkdir -p fuzz/corpus/tiff_decode   # libFuzzer requires it to already exist
cargo +nightly fuzz run tiff_decode fuzz/corpus/tiff_decode fuzz/seeds -- -rss_limit_mb=2048 -malloc_limit_mb=512
```

**Pass the corpus directory first.** libFuzzer *writes* newly discovered inputs
into the first directory it is given and treats the rest as read-only. Naming
`fuzz/seeds` alone makes it the corpus and buries the 20 curated seeds under
thousands of generated files; `fuzz/corpus/` is gitignored precisely so it can
absorb that instead.

`-rss_limit_mb` / `-malloc_limit_mb` turn runaway allocation into a *reported*
crash instead of letting the OS OOM-killer take the process with no artifact —
which matters here, because unbounded allocation is one of the bugs being hunted.

Crashing inputs land in `fuzz/artifacts/<target>/`. To replay and shrink one:

```sh
cargo +nightly fuzz run  tiff_decode fuzz/artifacts/tiff_decode/<file>
cargo +nightly fuzz tmin tiff_decode fuzz/artifacts/tiff_decode/<file>
```

### On Windows

The linked binary needs the ASan runtime on `PATH`, or it exits immediately with
`STATUS_DLL_NOT_FOUND`:

```powershell
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64;$env:PATH"
```

(`-s none` is *not* a workaround — dropping the sanitizer breaks sancov linkage
on MSVC with `unresolved external symbol __start___sancov_pcs`.)

## Seeds

`fuzz/seeds/` holds ~19 small, valid TIFFs covering every branch the reader has
a dedicated path for: both byte orders, BigTIFF, each codec (LZW / Deflate /
PackBits), both predictors, multi-strip, multi-page, chunky and planar RGB,
8-bit and 4-bit palettes, and both metadata dialects. Good seeds are the
difference between a fuzzer that explores the parser and one that spends its
life failing the magic-number check — start every campaign from them.

The `zstd` codec is deliberately excluded (the fuzz crate builds
`fast-tiff-lib` with `default-features = false`): it is a C dependency with its
own upstream fuzzing, and leaving it out keeps this harness pure Rust.

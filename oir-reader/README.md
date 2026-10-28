# oir-reader

[![Crates.io](https://img.shields.io/crates/v/oir-reader?color=green)](https://crates.io/crates/oir-reader)
[![Downloads](https://img.shields.io/crates/d/oir-reader)](https://crates.io/crates/oir-reader)
[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/oir-reader/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/docsrs/oir-reader?label=docs.rs)](https://docs.rs/oir-reader)

A reader for Olympus/Evident FluoView **OIR** microscopy files, in pure Rust. It
reassembles the scattered image planes of an acquisition — one file, or a long
recording split across several — into a stack in ImageJ's `xyczt` order. It
recovers recordings that are still being written, and translates the
acquisition's own XML into the record FluoView exports beside an OIR. Each
part is memory-mapped where the platform allows and read through a bounded
buffer everywhere else, and the crate builds for `wasm32-unknown-unknown`.

It is the reader behind [FastTIFF](https://github.com/konung-yaropolk/FastTIFF)'s
OIR importer, split out so that it can be used on its own.

The API is 0.x and may change between minor versions.

```toml
[dependencies]
oir-reader = "0.25"
```

## Quick start

```rust,no_run
let acquisition = oir_reader::read("recording.oir")?;
let shape = acquisition.shape();
println!(
    "{}x{}, {} channel(s), {} slice(s), {} frame(s), {:?}",
    shape.width, shape.height, shape.channels, shape.slices, shape.frames, shape.sample_type
);

// Every plane, in xyczt order. Channel 1 of slice 0 at the tenth time point:
if let (Some(planes), Some(i)) = (acquisition.planes().as_u16(), shape.plane_index(1, 0, 9)) {
    let plane: &[u16] = &planes[i];
    println!("{} samples, first {}", plane.len(), plane[0]);
}

// What the file says about itself.
let record = acquisition.record();
println!("{:?} um/pixel, objective {:?}", record.pixel_x_um(), record.objective());
for event in record.events() {
    println!("{} at {} ms", event.name(), event.time_ms());
}
println!("{:?} s between frames", acquisition.frame_interval_s());

// The record as FluoView would have exported it, for this stack.
if let Some(text) = acquisition.fluoview_text("recording.oir") {
    std::fs::write("recording.txt", text)?;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`examples/oir_info.rs` prints all of that for a file of your own:

```text
cargo run --release --example oir_info -- path/to/recording.oir
```

## The planes

`Acquisition::planes` is every plane of the stack, each exactly `width * height`
samples, in ImageJ's `xyczt` order: channel fastest, then slice, then time.
`Shape::plane_index(c, z, t)` says where any one of them is.

The **shape** is decided from the plane names first, and from the acquisition
record only where the names are silent. The names come from the same blocks as
the pixels. The record describes what the microscope was set up to do, and the
two can differ. A count from the record that does not divide the planes evenly
is ignored rather than believed.

A plane the acquisition **did not finish** — a recording stopped mid-frame
leaves a short last one — is left out, and `Acquisition::dropped_planes` says
how many. Kept, it would be a mostly-black frame at the end of the stack.

**Sample values** come in 8- or 16-bit containers holding fewer significant
bits, commonly 10 or 12. A 16-bit plane from a 12-bit detector saturates at
4095, not 65535. `Channel::bits` gives the depth the file states.

## Split acquisitions

A long recording is written as `name.oir`, `name_00001`, `name_00002`, … The
continuations are written both with and without the `.oir` extension, and both
are found. `read` takes the first file and reads every part. Numbering stops at
the first missing part, because reading across a gap would join time points
that are not adjacent. `acquisition_parts` lists the files without reading
them, and a continuation passed on its own is read on its own.

The first part carries the metadata, and may carry no planes of its own,
with all of them in the siblings. If they are not found, the error says so, and says where they were
looked for.

## Recordings still being written

The block index is the last thing an acquisition writes, so a file still being
recorded — or one whose tail was lost — has none. Those are read anyway: the
block stream is walked from the start, up to the first block that is not all
there, and `Part::block_source` is `BlockSource::Walk`. What is read is what
was on disk when the file was opened, so reading it again later reads more.

## Metadata

`Record` is translated from the XML inside the OIR, never from the `.txt`
FluoView may have exported beside it. The `.txt` is a detachable second copy:
it goes missing, gets renamed or left behind, and it can belong to another
acquisition in the same folder. Read from the container, the record cannot
disagree with the pixels next to it. Display lookup tables, overlays and
per-frame duplicates are dropped, which is most of an OIR's XML by size.

`Acquisition::fluoview_text` writes the record in FluoView's own
`"key"\t"value"` export format. It was checked against the vendor's own export
of the same acquisition. Of that export's 53 lines it reproduces 46 byte for
byte and in the same order, and contradicts none. The other seven hold values
the XML does not carry, so they are left out rather than guessed.

Pass the stack's own frame count when timing matters: `fluoview_text` and
`frame_interval_s` on `Acquisition` already do. A recording stopped early
states more frames in its record than it holds.

## Memory and I/O

Everything is read into memory: an acquisition becomes owned planes, and the
largest this was built against is about thirteen gigabytes of pixels. Each
part is read once, front to back. A descriptor and the pixels it describes sit
next to each other, so the pixels are copied out as soon as their descriptor
has been read, and the metadata is picked up in the same pass.

How the bytes are read is `Options::access`:

| `Access` | how a part is read |
|---|---|
| `Auto` (default) | memory-mapped where `MMAP_SUPPORTED` holds, through a bounded buffer otherwise; a part whose mapping fails is read through the buffer instead, and `Part::map_error` says why |
| `Buffered` | always through the buffer |

`read_bytes` and `Options::gather_bytes` read parts already in memory, in
place. That is the way in on a target with no filesystem.

What **zero-copy** means here: headers, descriptors and XML are parsed in
place, in the mapping or the caller's bytes, and each pixel chunk is copied
exactly once, into its plane. A plane arrives as thirty-odd chunks scattered
among the headers, so it cannot be borrowed from the file. No mapping outlives
the part it maps.

A mapping relies on one thing the reader cannot check: that the file is not
*truncated* while it is mapped. An acquisition only ever appends, so a
recording in progress is safe to map. A file another process may shorten, or
one on a network share that may disconnect, is not: under a mapping, an I/O
failure there is a crash rather than an error. Read those with
`Access::Buffered`.

```rust,no_run
use oir_reader::{Access, Options};

let acquisition = Options::new()
    .access(Access::Buffered)
    .gather(r"\\server\share\recording.oir", |done| {
        eprint!("\r{:.0}%", done * 100.0);
        true // false cancels
    })?
    .assemble()?;
# Ok::<(), oir_reader::Error>(())
```

`gather` does all of the reading and `assemble` decides the stack, so a
caller can report on the parts — which were walked, which could not be mapped
— before, and whether or not, the planes make a stack.

## Not supported

Writing OIR; the older OIB and OIF formats; reading single planes on demand
rather than the whole acquisition; axes other than time (`t`) and z (`z`).

## Feature flags

| feature | default | what it does |
|---|---|---|
| `mmap` | on | memory-maps parts on unix and Windows. It changes how parts are read, never the API. Elsewhere it does nothing, so a wasm build needs no `--no-default-features`. |
| `capi` | off | the C interface, `include/oir_reader.h`: see "From C and C++". Adds no dependency. |

## From C and C++

With the `capi` feature the crate builds as a C library, shared or static, and
[`include/oir_reader.h`](include/oir_reader.h) declares it. The crate type is chosen on
the command line rather than fixed in `Cargo.toml`, so a Rust program that
depends on this crate never builds a C library it has no use for:

```sh
cargo rustc -p oir-reader --release --features capi --crate-type cdylib,staticlib
```

`-p oir-reader` is for this repository's workspace; in an unpacked crate,
leave it out. The libraries land in `target/release/`:

| platform | shared library | link against | static library |
|---|---|---|---|
| Windows (MSVC) | `oir_reader.dll` | `oir_reader.dll.lib` | `oir_reader.lib` |
| Linux | `liboir_reader.so` | the same | `liboir_reader.a` |
| macOS | `liboir_reader.dylib` | the same | `liboir_reader.a` |

A static library leaves the system libraries it needs to the final link; add
`-- --print native-static-libs` to the command above for the list on your
platform (on Windows it is `kernel32 ntdll userenv ws2_32 dbghelp`). Build the
program against the C runtime that list names: `/defaultlib:libcmt` is the
static one, so `cl /MT` — which is what you get from this repository, whose
`.cargo/config.toml` builds Windows targets with `+crt-static` — and `msvcrt`
is the DLL one, `cl /MD`.

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "oir_reader.h"

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    OirAcquisition *a = NULL;
    int status = oir_read(argv[1], NULL, NULL, NULL, &a); /* every part of a split recording */
    if (status != OIR_OK) {
        fprintf(stderr, "%s\n", oir_last_error());
        return 1;
    }
    OirShape s;
    oir_shape(a, &s);
    printf("%ux%u, %zu channel(s), %zu slice(s), %zu frame(s)\n",
           s.width, s.height, s.channels, s.slices, s.frames);

    /* Channel 0 of slice 0 at the last time point, copied out. */
    size_t i;
    oir_plane_index(a, 0, 0, s.frames - 1, &i);
    size_t bytes = s.plane_len * s.bytes_per_sample;
    void *plane = malloc(bytes);
    if (plane && oir_plane_copy(a, i, plane, bytes) == OIR_OK && s.sample_type == OIR_SAMPLE_U16)
        printf("first pixel %u\n", (unsigned)((const uint16_t *)plane)[0]);
    free(plane);

    /* Text comes back the way snprintf writes it. */
    char objective[64];
    oir_record_string(a, OIR_FIELD_OBJECTIVE, objective, sizeof objective);
    OirRecordInfo r;
    oir_record_info(a, &r);
    printf("%g um/pixel, objective %s, %g s between frames\n",
           r.pixel_x_um, objective, r.frame_interval_s);

    oir_acquisition_free(a);
    return 0;
}
```

```sh
# Linux and macOS, from the workspace root (in an unpacked crate: -I include)
cc main.c -I oir-reader/include -L target/release -loir_reader
# Windows, MSVC, likewise
cl main.c /I oir-reader\include target\release\oir_reader.dll.lib
```

The shared library has to be findable when the program runs: beside the `.exe`
on Windows, on `LD_LIBRARY_PATH` (or an rpath) on Linux, on
`DYLD_LIBRARY_PATH` on macOS.

What a C caller should know:

- **Statuses, not crashes.** Every call that can fail returns `OIR_OK` or the
  reason it did not — `OIR_IO_ERROR`, `OIR_NOT_OIR`, `OIR_MALFORMED` and
  `OIR_NO_PLANES` are the reader's own errors — and `oir_last_error()` says it
  in words, with the file it was reading. A panic inside the reader comes back
  as `OIR_FAILED` instead of unwinding into C, as long as the library is built
  with `panic = "unwind"`, the default.
- **Text is `snprintf`'s convention.** A function that returns text takes a
  buffer and its size, writes what fits with a NUL after it, and returns the
  length of the whole; `NULL, 0` asks for the length. A negative return is
  minus a status. A value the file does not state is the empty string, and a
  number it does not state is NaN (or 0, for a count).
- **Everything is in memory**, as with the Rust API: `oir_read` reads the whole
  acquisition. `oir_plane_data` hands out a pointer to a plane where it
  already is, valid until `oir_acquisition_free`; `oir_plane_copy` copies one
  out.
- **Progress and cancelling.** `oir_read` and `oir_read_bytes` take an
  optional callback, called with the fraction of the bytes read; returning 0
  stops the read with `OIR_CANCELLED`. It is asked every thousand or so
  blocks, so a small file is read before it can be stopped.
- **No filesystem, no paths.** In a build for `wasm32-unknown-unknown`,
  `oir_read` and the part listing return `OIR_UNAVAILABLE`; `oir_read_bytes`
  works everywhere. The header is the same on every target.
- **Threads.** An acquisition is never changed once read, so one can be used
  from any number of threads at once. `oir_last_error` is per thread.
- **Unsafe code.** The crate is `#![deny(unsafe_code)]`. The C interface,
  `src/capi.rs`, is allowed it, to take C's pointers; the only other `unsafe`
  is the one call that maps a file. The reader behind the interface is the
  same safe code with or without the feature.
- **One header per version.** The header describes the library it shipped
  with; check `oir_version()` if they may have come apart.
- **Licence.** MPL-2.0 is per file: a C or C++ program may link the library,
  statically or dynamically, under any licence of its own. Changes to the
  library's own files stay MPL-2.0 and have to be made available.

## License

[MPL-2.0](LICENSE).

The format was worked out from real acquisitions and checked against the
vendor's own TIFF export. No vendor SDK, no published specification (there is
none) and no code from another reader went into it.

Olympus, Evident and FluoView are trademarks of their respective owners. This
crate is not affiliated with or endorsed by them.

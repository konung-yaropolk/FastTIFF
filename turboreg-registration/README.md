# turboreg-registration

Image registration by the TurboReg/StackReg pyramid method, in Rust: cubic
B-spline interpolation, a least-squares spline pyramid, and Thevenaz's inverse
Marquardt-Levenberg optimizer. Five transformation models, from translation to
bilinear.

The API mirrors [pystackreg](https://github.com/glichtner/pystackreg) so results
can be compared against the reference implementation.

## Features

| feature | default | what it adds |
|---|---|---|
| `threads` | on | `Backend::MultiThread`, through rayon: every core within each frame, and separate frames side by side |
| `gpu` | off | `Backend::Gpu`, through wgpu: the per-pixel sums of the large pyramid levels on the graphics card |
| `capi` | off | a C interface, for building the crate as a shared or static library; see [From C and C++](#from-c-and-c) |

The two CPU backends give the same answer bit for bit. The GPU backend computes
in `f32`, so it agrees with them to a tolerance (within a fiftieth of a pixel in
the tests) and is reproducible run to run on the same card. A backend that
cannot run — a feature left out, no usable adapter, a browser — is refused by
`Backend::unavailable_reason` rather than quietly swapped for another.

A host that draws with wgpu should hand the crate its own device before the
first GPU run — `gpu::use_device(device, queue, &adapter.get_info())` — rather
than let it open one. A second device opened beside a renderer's, on a card
whose memory the two have overrun, can deadlock both, and the system then
resets the GPU and every device in the process is lost. On the one device,
everything the registration allocates or submits runs inside error scopes: a
card with no room left for its buffers fails the batch instead of panicking,
and a registration in which the device failed frames fails as
`Error::Device`, saying why (`gpu::take_trouble` has the reason for a pair
registered directly). Registrations on several threads take turns on the
device a batch at a time, so each gets its own batches' reasons (start them
from ordinary threads, not from inside rayon tasks, which could deadlock on
that turn). wgpu still treats running out of memory for its own staging, a
pipeline or a submission as losing the device — the host's, when it is shared
— so uploads go a slab at a time to keep that staging small.

## A stack that is not in memory

`register_stack` takes a stack held in memory. A recording too large for that,
or read from a file plane by plane, goes through a `Session`, which asks for
the frames it needs a batch at a time and keeps only what a batch needs:

```rust
use turboreg_registration::{Batch, Session, Settings};

let mut session = Session::new(ly, lx, frames, &Settings::default())?;
let mut batch: Vec<Vec<f32>> = Vec::new();
while let Some(wanted) = session.wanted() {
    batch.clear();
    batch.extend(wanted.frames.map(|t| read_frame(t))); // from a file, say
    session.feed(Batch::Split(&batch))?;
}
let registered = session.finish()?;
```

The batch size is the session's (`Session::batch_len`), decided once for every
caller: as many frames as the GPU takes at once, or on the CPU a few per core
within 256 MiB of frames. Under `Reference::First` and `Reference::Mean` the
frames that make the target are asked for first, in a pass of their own, and
then the stack, from the first frame fitted or the first its window reaches
back to; `Session::plan` says which frames each pass reads, for dividing a
progress bar. What both passes need is read twice: nothing under the
first-frame default with no smoothing, the target's frames and their windows
with a longer target or a moving average, and every frame under the mean. So
a source that can be read only once, such as a camera, can feed
`Reference::Previous` or that default and nothing else. A moving average
reaches across batch edges through a ring of the `2h` frames before each
batch (for a window of `2h + 1`; a window wider than the stack is the whole
stack), so the answers are the ones `register_stack` gives, bit for bit,
whatever the batches. A batch is lent as one buffer (`Batch::Packed`), a
buffer per frame (`Batch::Split`) or a borrowed slice per frame
(`Batch::Slices`), and never copied whole. The viewer's
TurboReg plugin registers a file this way, and `register_stack` and
`tr_register` are the same session fed from memory.

## From C and C++

With the `capi` feature the crate builds as a C library, shared or static, and
[`include/turboreg_registration.h`](include/turboreg_registration.h) declares it. The crate type is chosen on
the command line rather than fixed in `Cargo.toml`, so a Rust program that
depends on this crate never builds a C library it has no use for:

```sh
cargo rustc -p turboreg-registration --release --features capi --crate-type cdylib,staticlib
```

`-p turboreg-registration` is for this repository's workspace; in an unpacked crate,
leave it out. Add `gpu` to the features (`--features capi,gpu`) for `TR_BACKEND_GPU`. The libraries land in `target/release/`:

| platform | shared library | link against | static library |
|---|---|---|---|
| Windows (MSVC) | `turboreg_registration.dll` | `turboreg_registration.dll.lib` | `turboreg_registration.lib` |
| Linux | `libturboreg_registration.so` | the same | `libturboreg_registration.a` |
| macOS | `libturboreg_registration.dylib` | the same | `libturboreg_registration.a` |

A static library leaves the system libraries it needs to the final link; add
`-- --print native-static-libs` to the command above for the list on your
platform (on Windows it is `kernel32 ntdll userenv ws2_32 dbghelp`). Build the
program against the C runtime that list names: `/defaultlib:libcmt` is the
static one, so `cl /MT` — which is what you get from this repository, whose
`.cargo/config.toml` builds Windows targets with `+crt-static` — and `msvcrt`
is the DLL one, `cl /MD`.

```c
#include <stdio.h>
#include "turboreg_registration.h"

/* `stack`: n_frames frames of ly x lx floats, one after another, row-major. */
int stabilise(float *stack, size_t n_frames, size_t ly, size_t lx) {
    TrSettings s;
    tr_settings_default(&s); /* always start here: the struct can grow */
    s.transformation = TR_TRANSFORMATION_AFFINE;
    s.border = TR_BORDER_EDGE;
    TrRegistered *r = NULL;
    int status = tr_register(stack, n_frames, ly, lx, &s, NULL, NULL, &r);
    if (status != TR_OK) {
        fprintf(stderr, "registration failed: %s\n", tr_last_error());
        return status;
    }
    for (size_t t = 0; t < n_frames; t++) {
        float *frame = stack + t * ly * lx;
        tr_registered_apply(r, frame, t, frame); /* in place is allowed */
    }
    tr_registered_free(r);
    return TR_OK;
}
```

```sh
# Linux and macOS, from the workspace root (in an unpacked crate: -I include)
cc main.c -I turboreg-registration/include -L target/release -lturboreg_registration
# Windows, MSVC, likewise
cl main.c /I turboreg-registration\include target\release\turboreg_registration.dll.lib
```

The shared library has to be findable when the program runs: beside the `.exe`
on Windows, on `LD_LIBRARY_PATH` (or an rpath) on Linux, on
`DYLD_LIBRARY_PATH` on macOS.

What a C caller should know:

- **Statuses, not crashes.** Every call returns `TR_OK` or the reason it
  did not, and `tr_last_error()` says it in words. A panic inside the
  registration comes back as `TR_FAILED` instead of unwinding into C — as
  long as the library is built with `panic = "unwind"`, the default.
- **Measure once, apply to every channel.** `tr_registered_apply` takes any
  frame of timepoint `t`, which is how a second channel follows the first.
- **The frames are read in place** by `tr_register`, a batch at a time, and
  not copied: beside the caller's stack the registration holds one batch's
  working memory (a few frames per core, at most 256 MiB of them, or what the
  GPU takes at once, and the fits' own), plus, with a moving average of
  `2h + 1` frames, a copy of the `2h` frames before each batch. That is not a
  second stack unless the window is as wide as the stack. Keep the buffer
  unchanged until the call returns; it is the caller's again afterwards.
- **Threads.** Separate results can be used on separate threads; one result
  can be applied from several at once, but not while
  `tr_registered_set_border` changes it. `tr_last_error` is per thread.
  GPU runs started from several threads take turns on the one device, a
  batch at a time.
- **One header per version.** The header describes the library it shipped
  with; check `tr_version()` if they may have come apart.
- **Licence.** The library is GPL-3.0-only, and so is a program distributed with it linked in, statically or dynamically; see Licensing below.

## Licensing

GPL-3.0-only. The algorithm is Thevenaz, Ruttimann & Unser's, shipped by EPFL as
TurboReg/StackReg, which EPFL relicensed to GPL-3-or-later in April 2023.

This crate is **not** a port of pystackreg's code. pystackreg carries the
pre-2023 terms ("you should not redistribute it without our consent"), which are
not GPL-compatible; it lends this crate its interface and nothing else.

Please cite:

> P. Thevenaz, U.E. Ruttimann, M. Unser, "A Pyramid Approach to Subpixel
> Registration Based on Intensity", IEEE Transactions on Image Processing,
> vol. 7, no. 1, pp. 27-41, January 1998.

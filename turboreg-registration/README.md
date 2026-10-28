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
card with no room left for its buffers fails the batch (its frames are
reported as not fitted, and `gpu::take_trouble` says why) instead of
panicking. wgpu still treats running out of memory for its own staging, a
pipeline or a submission as losing the device — the host's, when it is shared
— so uploads go a slab at a time to keep that staging small.

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
- **The frames are copied** by `tr_register`, and the registration keeps
  working copies of its own: while it runs it needs about three times the stack's
  size in memory, counting the caller's. The caller's buffer is free once it
  returns.
- **Threads.** Separate results can be used on separate threads; one result
  can be applied from several at once, but not while
  `tr_registered_set_border` changes it. `tr_last_error` is per thread.
  GPU runs started from several threads take turns on the one device.
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

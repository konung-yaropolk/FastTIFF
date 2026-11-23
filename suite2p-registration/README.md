# suite2p-registration

Motion correction for two-photon timelapses, in Rust. A port of
[suite2p](https://github.com/MouseLand/suite2p)'s `registration` module —
parameter for parameter, default for default.

A recording of living tissue moves. Breathing, heartbeat, the animal shifting:
over a few minutes a field drifts by tens of pixels, and anything measured
per-pixel afterwards is measuring a different piece of tissue at each timepoint.
This puts the frames back on top of each other first.

```toml
[dependencies]
suite2p-registration = "0.19"
```

## What it does

- **Rigid registration** — one whole-frame shift per frame, by phase
  correlation against a reference built from the recording itself.
- **Non-rigid registration** — a shift per block, interpolated to a shift per
  pixel, for tissue that deforms rather than merely sliding. Measured on top of
  the rigid correction, not instead of it.
- **Bidirectional phase** — the comb artefact a resonant scanner leaves on
  alternate lines.
- **Bad-frame detection** — frames whose shift is an outlier, or that barely
  correlated at all. Reported, never discarded: which frames to throw away is
  the experimenter's decision.
- **Three backends** — one thread, every core, or the graphics card. They give
  the same answer; only how many frames are in flight differs.

## Using it

For a multi-channel recording, pass one `Frames` per channel to
`register_channels(&channels, &settings, on_progress)`. By default the reference
and shifts are measured on the pixelwise arithmetic mean of the raw channels.
Set `Settings::measure_on` to `Some(2)` to measure on the third channel.
Apply the returned correction to each original plane to keep the channels aligned.

For a single movie or an already prepared measurement channel, use `register`:

```rust
use suite2p_registration::{register, Frames, Settings};

# fn main() {
# let (ly, lx) = (64, 64);
# let movie: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0; ly * lx]).collect();
// `movie` is one `Vec<f32>` per frame, row-major, `ly * lx` samples each.
let frames = Frames { ly, lx, frames: &movie };
let settings = Settings {
    nonrigid: false,
    spatial_taper: 5.0,
    ..Settings::default()
};

let out = register(&frames, &settings, &mut |_fraction| true).expect("not cancelled");

for (t, shift) in out.shifts.iter().enumerate() {
    println!("frame {t}: dy {}, dx {} (peak {:.3})", shift.dy, shift.dx, shift.corr);
}
println!("{} frame(s) flagged", out.bad_frames.iter().filter(|b| **b).count());

// Put a frame back where it belongs: the frame as recorded, any channel of
// timepoint 1. One call, so the comb is undone once, the rigid shift cannot be
// applied twice and the deformation cannot be forgotten. It can fail only
// under `Border::Crop`, when the frames share no area to crop to.
let corrected: Vec<f32> = out.apply(&movie[1], ly, lx, 1).expect("no crop asked for");
# let _ = corrected;
# }
```

The closure is progress: it is called with a fraction and returns `false` to
stop, which makes `register` return `None`.

### A recording that does not fit in memory

`register` wants every frame at once. A host that reads a recording off disk
feeds a `Session` instead, which names the frames it wants and does the rest:

```rust
use suite2p_registration::{Border, Session, Settings};

# fn main() {
# let (ly, lx) = (32, 32);
# let n_frames = 6;
# let read = |_t: usize| vec![1.0f32; ly * lx];
let settings = Settings { nonrigid: false, spatial_taper: 5.0, ..Settings::default() };
let mut session = Session::new(ly, lx, n_frames, &settings);
while session.needs_reference() {            // once, or twice for two-step
    let sample = session.reference_frames().iter().map(|&t| read(t)).collect();
    session.build_reference(sample, &mut |_fraction| true).expect("not cancelled");
    while let Some(batch) = session.next_batch() {
        let mut frames: Vec<Vec<f32>> = batch.clone().map(read).collect();
        // Ours, so corrected where they are and handed back as recorded.
        let measured = session.measure_in_place(&mut frames);
        if measured.is_final() && settings.border != Border::Crop {
            let at: Vec<usize> = batch.collect();
            measured.apply(&mut frames, &at);  // or any channel of these timepoints
            // ... and now they can be written out
        }
    }
}
let registered = session.finish();             // shifts, bad frames, block field
# let _ = registered;
# }
```

Under `Border::Crop` the planes wait for `finish`: the area to cut to depends on
every frame. Read them again then and hand them to `Registered::apply_batch`,
which moves and cuts them in one resampling — what this repository's viewer
does.

Frames go in as recorded, everywhere: the session undoes the scanner's comb
itself, on the sample, the batches and every plane it moves. It picks the
reference sample as suite2p does (`nimg_init` frames, `np.linspace` over the
recording), measures the comb on that sample, chooses the batches — they are
the window `smooth_sigma_time` smooths across, so where they fall is part of
the answer, and a batch is bounded to `pipeline::BATCH_BUDGET` (256 MiB) of
frames, which `shortened_window` reports — runs the second pass of
`two_step_registration`, measures the block field and flags the bad frames.
`register` is a feeder of one too, and so is this repository's viewer, which is
what keeps them giving the same answer.

What a session holds is bounded whatever the length of the recording: the
reference sample while the reference is built, then a batch, plus a shift per
frame.

- `measure` borrows a batch and never writes it — for frames the caller cannot
  lend mutably, like `register`'s movie. Frames with a comb to undo, or a first
  pass's shift to apply, are copied into one batch-sized buffer kept from batch
  to batch. A feeder whose batch is its own hands it to `measure_in_place`
  instead, which corrects the frames where they are and hands them back bit for
  bit, so the batch is held once.
- A non-rigid run's block fields are held a batch at a time — the ones that
  batch's planes are moved by — unless the feeder asks to keep every frame's
  (`keep_fields`) to move frames after `finish`. `register` asks, and a crop
  keeps them regardless; a field per frame grows with the recording.
- Everything a `Measured` says, the session says too (`shifts`, `field`,
  `apply`), for a caller that cannot hold a borrow from one call to the next.

### Settings

[`Settings`] carries suite2p's own option names and defaults — `maxregshift`,
`smooth_sigma`, `spatial_taper`, `nimg_init`, `batch_size`, `nonrigid`,
`block_size`, `snr_thresh`, `subpixel`, `th_badframes` and the rest — so a
value copied from a lab's `ops.npy` means the same thing here.

Two worth knowing before a first run:

- **`maxregshift`** (default 0.1) is the largest shift allowed, as a fraction of
  the smaller frame dimension. It has to cover the motion *plus* wherever the
  reference landed; a shift beyond it is clipped, which reads as nearly working.
- **`spatial_taper`** (default 50.0) fades the frame border before correlating,
  because an FFT wraps. Keep it well above `3 * smooth_sigma`. On small test
  frames the default fades everything — hence the `5.0` in the example above.

### Choosing where it runs

```rust
use suite2p_registration::{Backend, Settings};

let settings = Settings { backend: Backend::MultiThread, ..Settings::default() };
if let Some(why) = settings.backend.unavailable_reason(512, 512) {
    eprintln!("not this backend: {why}");
}
```

`Backend::Gpu` needs the `gpu` feature and takes power-of-two frames up to 1024
a side, its FFT being radix-2. It needs a real graphics adapter (a software
renderer does not count), and it does not run in a browser, though the feature
still compiles for `wasm32-unknown-unknown`, so one build can enable it
everywhere. In every one of those cases it is **refused rather than silently
downgraded** — a run that said GPU and used the processor is indistinguishable
from a slow one — so ask `unavailable_reason` first.

A host that draws with wgpu should hand the crate its own device before the
first GPU run, `gpu::use_device(device, queue, &adapter.get_info())`, rather
than let it open one: a second device opened beside a renderer's, on a card
whose memory the two have overrun, can deadlock both. On the one device,
everything the registration allocates or submits runs inside error scopes, so
a card with no room left for the registration's buffers sends the batch to the
processor (or, part-way through, back as flat maps) rather than panicking, and
`gpu::take_trouble` says what happened. wgpu still treats running out of
memory for its own staging, a pipeline or a submission as losing the device —
the host's, when it is shared — so uploads go a slab at a time to keep that
staging small.

```toml
suite2p-registration = { version = "0.19", features = ["gpu"] }
```

Measured on 300 frames of 512×512, on a machine with 16 CPU threads and a
Quadro P620 (a low-end card, and the memory-bandwidth limit of this workload):

| backend | correlating 300 frames |
|---|---|
| single thread | 1480 ms |
| every core | 329 ms |
| GPU | 374 ms |

A card with more memory bandwidth should pull ahead; on that one, the cores win
by a nose. Both are ten times what they were before batching, and the answers
are identical either way — `tests/gpu_agrees.rs` checks the device against the
processor for square frames, non-square frames, clipping and batches spanning
several submissions.

## Lower-level pieces

`register` and `Session` are the whole pipeline, but every stage of it is
public, for timing a stage or testing one against another. Sequencing them into
a registration is the session's job; a host that did it itself would be a
second copy of the pipeline to keep in step.

| module | what is in it |
|---|---|
| [`session`] | `Session`, `Measured`, `reference_sample` |
| [`rigid`] | `correlation_map`, `peak_of`, `phase_correlate`, `shift_frame` |
| [`nonrigid`] | the block grid, `measure_blocks`, `warp` |
| [`masks`] | the taper and the whitened reference spectrum (`RefFilters`) |
| [`pipeline`] | `measure_batch`, `apply_batch`, `batch_len`, `smooth_in_time`, `bad_frames` |
| [`bidiphase`] | `compute` and `shift` for the scanner comb |
| [`fft`] | the cached 2D transform, `fftshift`, `ifftshift` |
| [`settings`] | `Settings`, `Backend`, `Border` |
| [`work`] | relative stage costs, for dividing a progress bar |

## From C and C++

With the `capi` feature the crate builds as a C library, shared or static, and
[`include/suite2p_registration.h`](include/suite2p_registration.h) declares it. The crate type is chosen on
the command line rather than fixed in `Cargo.toml`, so a Rust program that
depends on this crate never builds a C library it has no use for:

```sh
cargo rustc -p suite2p-registration --release --features capi --crate-type cdylib,staticlib
```

`-p suite2p-registration` is for this repository's workspace; in an unpacked crate,
leave it out. Add `gpu` to the features (`--features capi,gpu`) for `S2P_BACKEND_GPU`. The libraries land in `target/release/`:

| platform | shared library | link against | static library |
|---|---|---|---|
| Windows (MSVC) | `suite2p_registration.dll` | `suite2p_registration.dll.lib` | `suite2p_registration.lib` |
| Linux | `libsuite2p_registration.so` | the same | `libsuite2p_registration.a` |
| macOS | `libsuite2p_registration.dylib` | the same | `libsuite2p_registration.a` |

A static library leaves the system libraries it needs to the final link; add
`-- --print native-static-libs` to the command above for the list on your
platform (on Windows it is `kernel32 ntdll userenv ws2_32 dbghelp`). Build the
program against the C runtime that list names: `/defaultlib:libcmt` is the
static one, so `cl /MT` — which is what you get from this repository, whose
`.cargo/config.toml` builds Windows targets with `+crt-static` — and `msvcrt`
is the DLL one, `cl /MD`.

```c
#include <stdio.h>
#include "suite2p_registration.h"

/* `stack`: n_frames frames of ly x lx floats, one after another, row-major. */
int stabilise(float *stack, size_t n_frames, size_t ly, size_t lx) {
    S2pSettings s;
    s2p_settings_default(&s); /* always start here: the struct can grow */
    s.nonrigid = false;
    s.border = S2P_BORDER_EDGE;
    S2pRegistered *r = NULL;
    int status = s2p_register(stack, n_frames, ly, lx, &s, NULL, NULL, &r);
    if (status != S2P_OK) {
        fprintf(stderr, "registration failed: %s\n", s2p_last_error());
        return status;
    }
    for (size_t t = 0; t < n_frames; t++) {
        float *frame = stack + t * ly * lx;
        s2p_registered_apply(r, frame, t, frame); /* in place is allowed */
    }
    s2p_registered_free(r);
    return S2P_OK;
}
```

```sh
# Linux and macOS, from the workspace root (in an unpacked crate: -I include)
cc main.c -I suite2p-registration/include -L target/release -lsuite2p_registration
# Windows, MSVC, likewise
cl main.c /I suite2p-registration\include target\release\suite2p_registration.dll.lib
```

The shared library has to be findable when the program runs: beside the `.exe`
on Windows, on `LD_LIBRARY_PATH` (or an rpath) on Linux, on
`DYLD_LIBRARY_PATH` on macOS.

What a C caller should know:

- **Statuses, not crashes.** Every call returns `S2P_OK` or the reason it
  did not, and `s2p_last_error()` says it in words. A panic inside the
  registration comes back as `S2P_FAILED` instead of unwinding into C — as
  long as the library is built with `panic = "unwind"`, the default.
- **Measure once, apply to every channel.** `s2p_registered_apply` takes any
  frame of timepoint `t`, which is how a second channel follows the first.
- **The frames are copied** by `s2p_register`, once: while it runs it needs
  about twice the stack's size in memory, counting the caller's, plus twice
  the `nimg_init` frames the reference is built from (the sample, and the copy
  the initial pick makes of it), and one batch more when a comb is corrected or
  a second pass run. The caller's buffer is free once it returns.
- **Frames go in as recorded.** `s2p_registered_apply` undoes the scanner's
  comb itself before it moves a frame.
- **Threads.** Separate results can be used on separate threads; one result
  can be applied from several at once, but not while
  `s2p_registered_set_border` changes it. `s2p_last_error` is per thread.
  GPU runs started from several threads take turns on the one device.
- **One header per version.** The header describes the library it shipped
  with; check `s2p_version()` if they may have come apart.
- **Licence.** The library is GPL-3.0-only, and so is a program distributed with it linked in, statically or dynamically.

## Licence

GPL-3.0-only. This is a derivative work of suite2p (Copyright © 2023 Howard
Hughes Medical Institute, authored by Carsen Stringer and Marius Pachitariu),
which is GPL-3, and the licence travels with the algorithm whoever retypes it.

If you use this for published work, cite suite2p:

> Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L. F.,
> Dalgleish, H., Carandini, M., & Harris, K. D. (2017). *Suite2p: beyond
> 10,000 neurons with standard two-photon microscopy.* bioRxiv 061507.

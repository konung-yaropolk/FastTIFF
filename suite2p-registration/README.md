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

Measure the shifts, then apply them. They are separate because a two-colour
recording is registered *once* — measured on one channel and applied to both,
or the channels drift apart.

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

// Put a frame back where it belongs. One call, so the rigid shift cannot be
// applied twice or the deformation forgotten.
let corrected: Vec<f32> = out.apply(&movie[1], ly, lx, 1);
# let _ = corrected;
# }
```

The closure is progress: it is called with a fraction and returns `false` to
stop, which makes `register` return `None`.

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
a side, its FFT being radix-2. It is **refused rather than silently downgraded**
— a run that said GPU and used the processor is indistinguishable from a slow
one — so ask `unavailable_reason` first.

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

`register` is the whole pipeline, but every stage of it is public, because a
host that streams a recording off disk cannot hand over one `Vec` of every
frame:

| module | what is in it |
|---|---|
| [`rigid`] | `correlation_map`, `peak_of`, `phase_correlate`, `shift_frame` |
| [`nonrigid`] | the block grid, `measure_blocks`, `warp` |
| [`masks`] | the taper and the whitened reference spectrum (`RefFilters`) |
| [`pipeline`] | `measure_batch`, `apply_batch`, `smooth_in_time`, `bad_frames` |
| [`bidiphase`] | `compute` and `shift` for the scanner comb |
| [`fft`] | the cached 2D transform, `fftshift`, `ifftshift` |
| [`settings`] | `Settings`, `Backend` |
| [`work`] | relative stage costs, for dividing a progress bar |

## Licence

GPL-3.0-only. This is a derivative work of suite2p (Copyright © 2023 Howard
Hughes Medical Institute, authored by Carsen Stringer and Marius Pachitariu),
which is GPL-3, and the licence travels with the algorithm whoever retypes it.

If you use this for published work, cite suite2p:

> Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L. F.,
> Dalgleish, H., Carandini, M., & Harris, K. D. (2017). *Suite2p: beyond
> 10,000 neurons with standard two-photon microscopy.* bioRxiv 061507.

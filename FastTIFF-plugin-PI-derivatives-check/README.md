# PI Derivatives Check

A [FastTIFF](https://github.com/konung-yaropolk/FastTIFF) plugin that turns a
two-photon stimulation recording into a map of where the tissue responded, one
image per stimulus condition, shown as a single magenta/green composite.

It exists for one protocol — presynaptic inhibition measured by alternating two
stimulators — which is why it is a plugin and not part of the viewer.

## What it computes

The recording is stimulated on a repeating pattern: each *epoch* is N *steps* of
equal length, and each step either fires a given stimulator or does not. For
every step that fires, the plugin finds the response window after it in every
epoch, takes the **positive part of the temporal derivative** across that
window, sums it, and averages over epochs. The result is one image per firing
step: a map of where the signal was rising while that stimulus was on.

Two maps are then arranged as one RGB image — the first into red *and* blue, the
second into green. Red and blue together read as magenta, so what responded to
both conditions appears white and what responded to only one keeps its colour.
Magenta and green rather than red and green because they stay distinguishable
under the common forms of colour blindness, and their overlap is white rather
than a muddy yellow that is hard to tell from either parent.

## Where the timing comes from

Not from the dialog. The trigger time, the frame count and the recording's
duration are read out of `ImageDescription` (tag 270) — the acquisition's own
record, which FastTIFF's OIR importer writes there in the form the instrument
exports. Only lines matching the patterns in [`src/meta.rs`](src/meta.rs) are
read; everything else in that description is ignored, which matters because it
is a mixture: ImageJ's own `key=value` block, the instrument's export, and
whatever a previous tool left behind.

Nothing here has to be told when the stimulus started, and nothing can disagree
with the file about it.

The epoch count is not a parameter either: it is the largest number of whole
epochs that fits between the trigger and the end of the recording, response
window included. A recording stopped early therefore yields fewer epochs rather
than an error or a set of averages quietly containing a truncated one.

## Parameters

| Name | Default | What it is |
| --- | --- | --- |
| Stimulation pattern | `10,11` | One row per stimulator, comma-separated; `1` fires, `0` does not. |
| Step duration (s) | 10.0 | How long one step of an epoch lasts. |
| Response window (s) | 0.8 | Must be long enough to contain the response peak. |
| Trigger event | 1 | Which event marker in the file's metadata starts the sequence. |
| First epoch | 1 | Epochs before this are ignored. |
| Frame lag | −1 | Shifts the response window, to line the derivative up with the stimulus. |
| Clock correction | −0.003 | Fractional correction to the sampling interval, for stimulator/scanner drift. |
| Gaussian sigma | 1.5 | Smoothing for the derivative, in pixels and frames. |
| Source channel | 1 | Counted from 1, as the channels are named everywhere else. |

## Building

```sh
cargo build --release
```

That produces a shared library — `fasttiff_plugin_derivatives.dll` on Windows,
`.so` on Linux, `.dylib` on macOS. Copy it into FastTIFF's plugin folder, which
the app will open for you from **Plugins ▸ Open plugin folder…**. Restart the
app and it appears under **Plugins ▸ Analysis**.

## Developing against a local FastTIFF

While this directory still sits inside a FastTIFF checkout, `Cargo.toml` points
at the host crates by path and everything works as it stands.

Once it is its own repository, comment those path lines out and uncomment the
`git` lines beside them. To work against a local FastTIFF checkout after that,
add a patch section rather than editing the dependencies back:

```toml
[patch.'https://github.com/konung-yaropolk/FastTIFF']
fasttiff-plugin = { path = "../FastTIFF/fasttiff-plugin" }
fast-tiff-viewer = { path = "../FastTIFF/fast-tiff-viewer" }
fasttiff-plugin-api = { path = "../FastTIFF/fasttiff-plugin-api" }
```

## Tests

```sh
cargo test
```

`tests/overlay.rs` is the one worth knowing about. It builds a synthetic
recording where the left half of the frame brightens after one stimulus and the
right half after the other, runs the plugin **through FastTIFF's real plugin
loader**, and checks that red and blue are bright on the left and green on the
right. Getting the epoch arithmetic wrong by a single step swaps them, which is
the failure this is here to catch: a map of the wrong moment looks exactly as
convincing as a map of the right one.

That is also why the host is a dev-dependency. A test that could only reach the
plugin from inside its own crate would prove nothing about the boundary it
actually crosses in use.

## Licence

This directory carries a copy of the licence file from the FastTIFF repository
it came out of. Note that the repository's `LICENSE` is GPL-3.0 while the
crate manifests declare `MPL-2.0`; that disagreement is inherited, not
introduced here, and is worth settling before this is published on its own.

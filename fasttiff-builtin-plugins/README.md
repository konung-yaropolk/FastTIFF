# fasttiff-builtin-plugins

[![License](https://img.shields.io/badge/License-GPL--3.0--only-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/fasttiff-builtin-plugins/LICENSE)

The plugins [FastTIFF](https://github.com/konung-yaropolk/FastTIFF) compiles
in: Crop, Invert, Z Project, Plot third axis, the three Slice tools, 3D
animation, Generate PSF and Deconvolve, suite2p and TurboReg stabilization, the
PNG importer and exporter, and the Olympus OIR, HDF5, NIfTI and DICOM importers.

```text
  fasttiff-plugin-abi / -api / fasttiff-plugin   the plugin SDK
        │
  fasttiff-builtin-plugins  ← this crate: the plugins, written against the
        │                     SDK and nothing of the host's
  FastTIFF, FastTIFF-web    hand them to the viewer's registry at start-up
```

## Written against the SDK alone

Every plugin here is written against what a third-party plugin is written
against — the traits and types of `fasttiff-plugin-api`, and the `Registrar`
of `fasttiff-plugin` — and against the algorithm and format crates it needs
(`suite2p-registration`, `turboreg-registration`, `rustfft`, `png`,
`oir-reader`, …). Not against `fast-tiff-viewer`, the host: that is a
dev-dependency, for the tests, and nothing more. So a plugin here cannot
reach into the viewer, and the rule that built-ins and libraries are one kind
of plugin is checked by the compiler rather than kept by convention.

An application installs them with one call:

```rust,ignore
use fast_tiff_viewer::plugins::Registry;

// The desktop: the built-ins first, then the plugin libraries on disk.
let registry = Registry::with_installed(fasttiff_builtin_plugins::register);
// The browser, which can load no library.
let registry = Registry::new(fasttiff_builtin_plugins::register);
```

`register` hands each plugin type to the registry, which installs it through
the same C ABI a `.dll` is installed through, in the application's own process,
and marks it built-in. From then on a built-in is called, checked and contained
exactly as a library is.

On the desktop, an application that draws with wgpu also hands the
registrations its GPU device, so that they run on the renderer's device rather
than opening a second one:

```rust,ignore
fasttiff_builtin_plugins::share_gpu_device(&device, &queue, &adapter_info);
```

## Features

| Feature | Default | Effect |
|---|---|---|
| `builtin-plugins` | on | Every plugin that ships. |
| `plugin-image-tools` | via `builtin-plugins` | Invert, Crop, Slice Keeper, Slice Remover, Slice Order Invert, Split channels and Split Stack. |
| `plugin-<name>` | via `builtin-plugins` | One plugin, with whatever it alone depends on: `plugin-zproject`, `plugin-plot-axis`, `plugin-animation-3d`, `plugin-suite2p-stabilize`, `plugin-turboreg-stabilize`, `plugin-deconvolve`, `plugin-png`, `plugin-oir`, `plugin-hdf5`, `plugin-nifti`, `plugin-dicom`. A plugin left out has no module, no menu entry and no dependency of its own in the build. |
| `gpu-registration` | off | The registrations' GPU paths, and `share_gpu_device`. Pulls in wgpu. |
| `threads` | on | Deconvolution across threads; off for a single-threaded host such as `wasm32-unknown-unknown`. |
| `mmap` | on | The OIR importer memory-maps the files it reads; off, it reads them through a bounded buffer. |

The OIR, HDF5, NIfTI and DICOM importers are compiled for the desktop only, whatever
the features say: an importer is handed a path, and a browser has none. The PNG
importer is compiled for every target, in the module it shares with the PNG
exporter, but a browser host opens TIFF alone and never calls it.

The point counter, also in FastTIFF's Plugins menu, is not here. It is a tool
on the picture rather than a plugin — a session of clicks, with marks drawn
over the image while its window is open — and the plugin contract has no kind
of plugin for that yet, so it is written against the viewer's document and
lives in `fast-tiff-viewer`. It moves here when the contract gains an
interactive tool.

## License

GPL-3.0-only, where `fast-tiff-viewer` and the plugin SDK are MPL-2.0. Two of
the plugins link GPL-3.0-only crates — `plugin-suite2p-stabilize` links
`suite2p-registration`, a port of suite2p's registration, and
`plugin-turboreg-stabilize` links `turboreg-registration`, both with the
`image-registration-support` they share — and the copyleft travels with them
into whatever links them. Keeping every plugin in this one crate is what lets
the viewer core stay MPL; an application built with this crate is GPL-3.0, as
`FastTIFF` and `FastTIFF-web` are.

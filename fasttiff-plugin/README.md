# fasttiff-plugin

Write a FastTIFF plugin as an ordinary Rust type. This crate generates the
`extern "C"` boundary so you never have to look at it.

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
fasttiff-plugin = "0.18"
```

```rust
use fasttiff_plugin::api::*;

#[derive(Default)]
struct Invert;

impl Plugin for Invert {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("com.example.invert", "Invert").menu_path("Filters")
    }

    fn run(&mut self, host: &mut dyn HostContext, _p: &Params)
        -> Result<Outcome, PluginError>
    {
        let info = host.image();
        let mut buf = Vec::new();
        host.read_plane_f32(Plane::new(0, 0, host.view().frame_index), &mut buf)?;

        let hi = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out: Vec<f32> = buf.iter().map(|&v| hi - v).collect();

        Ok(Outcome::NewDocument(Box::new(ImageResult {
            width: info.width,
            height: info.height,
            channels: 1, slices: 1, frames: 1,
            pixel_type: PixelType::F32,
            planes: vec![PlaneData::F32(out)],
            name: format!("{}-inverted", host.stack_info().name),
        })))
    }
}

fasttiff_plugin::export_plugin! { plugins: [Invert] }
```

`cargo build --release`, then drop the resulting `.dll` / `.so` / `.dylib` into
the folder that **Plugins ▸ Open plugin folder…** opens. Restart FastTIFF and it
is in the menu.

A worked example with a filter, an importer and a dialog lives in
[`fasttiff-plugin-example`](../fasttiff-plugin-example/src/lib.rs).

## The two kinds of plugin

**`Plugin`** runs against the stack that is open. It gets a `HostContext` —
the image's shape in file coordinates, the viewer's current display state, and
a pull-based plane reader — and returns an `Outcome`: a new document, a file to
write, a message, or nothing.

**`Importer`** reads a file format FastTIFF does not know. It declares the
extensions it handles, which the host adds to the Open dialog and to
drag-and-drop *before your code has ever run*; when such a file is opened, your
importer is called and what it returns becomes an ordinary FastTIFF document.
It is the one plugin type that runs with nothing open, which is why it takes a
path rather than a `HostContext`.

Both can declare a dialog by returning `ParamDecl`s. You describe the controls;
the host draws them, clamps the values to the ranges you gave, and hands back a
`Params`. There is no UI toolkit in your dependency tree and no way for a plugin
to draw something the host did not expect.

## Reading pixels

```rust
let info = host.image();          // width, height, channels, slices, frames
for z in 0..info.slices {
    host.read_plane_f32(Plane::new(c, z, t), &mut buf)?;   // fills to plane_len()
    if !host.progress(z as f32 / info.slices as f32) {
        return Ok(Outcome::Cancelled);                     // the user pressed stop
    }
}
```

One plane at a time, into a buffer you own. A 4 GB stack cannot be handed across
a plugin boundary, and lending you a slice of the host's memory map would stop
being sound the moment your library was compiled by a different toolchain.

`read_plane_f32` gives the file's own values, untouched — that is the one to
process with. `read_plane_u16` gives what the viewer displays: 8-bit widened,
signed offset into unsigned, float rescaled through the contrast window.

Plane coordinates are **file** coordinates. `info.channels` is what the file
has, not what the viewer is compositing — the renderer only has six texture
slots, so a 12-channel stack shows six, and a plugin that could only reach those
six would be useless on exactly the data that most needs processing.

## Reading the file's scale

```rust
let info = host.stack_info();
let microns_per_px = info.spacing.x;          // Option<f64> — None means the
let seconds_per_frame = info.frame_interval_s; //   file did not say
let value = match info.calibration {
    Some((c0, c1)) => c0 + c1 * raw as f64,   // the file's own units
    None => raw as f64,
};
```

Every one of these is an `Option`, and that is load-bearing: a plugin that
cannot tell "the file states 0" from "the file states nothing" will silently
report measurements in the wrong units. `unit` names what the spacing is in
(`"micron"`, usually); `channel_names` and `description` carry what the file
said about itself.

## What the host guarantees

For the whole of one run:

* `image()` and `view()` do not change. The user can drag a contrast slider
  while you work; you will not see it, so a long run cannot produce a result
  computed from two different states.
* A plane that was in range stays in range.
* Every `read_*` either fills your buffer to exactly `plane_len()` or returns
  `Err` — never a short read.

## What you must not assume

* **Nothing you allocate crosses the boundary.** The host copies every string,
  descriptor and plane during the call that supplies it. Your allocator and the
  host's never meet, so linking against a different `malloc` is fine.
* **Your panics are yours to contain.** `export_plugin!` wraps every entry point
  in `catch_unwind` for you — this is not optional politeness. Since Rust 1.81
  an `extern "C"` function aborts the process on unwind, so a panic escaping
  your entry point would kill FastTIFF from inside your library, before any host
  code could react. The generated guard is what turns it into an error message
  with your plugin's name on it.
* **The vtable is stateless.** An instance of your type is built per call and
  dropped inside it, which is why `Default` is required. Keep state in your own
  statics if you need it, where your own allocator owns it.
* **The library is never unloaded.** Reinstalling a plugin needs a restart.

## Versions

`fasttiff-plugin-abi` is the frozen binary contract; this crate is the
convenience layer that generates it. Only the ABI crate's layout is permanent —
this crate compiles *into* your plugin and may change freely between releases.

The ABI's major version is part of the exported symbol name
(`ft_plugin_v1_query`), so a host of a different major version simply does not
find your plugin and says so, rather than the two sides disagreeing about a
struct layout halfway through a call. Within a major version, every struct
carries its own size and fields are only ever appended, so an older plugin runs
on a newer host and a newer plugin runs on an older one — each side reads only
the fields both agree exist.

## Where plugins live

There is no single folder — `plugins/` beside the executable is the Windows
habit and the wrong answer on macOS (writing into a signed bundle breaks the
signature) and Linux (a packaged binary lands in `/usr/bin`). FastTIFF searches,
in order:

| | for |
|---|---|
| `$FASTTIFF_PLUGIN_PATH` | development, CI, a lab pointing every workstation at one shared folder |
| the user's data directory | the only place a normal user can always write, and it survives reinstalling |
| beside the executable | portable installs — the unzipped folder on a USB stick that instrument PCs run from |
| the system directory | what a `.deb`/`.rpm` writes to, shared by every user |

Earlier entries win, so a user can override an administrator-installed plugin
without needing an administrator. **Plugins ▸ Open plugin folder…** opens the
first writable one.

Native only: there is no `dlopen` in a browser, so the wasm build has no plugin
interface at all rather than a disabled one.

## Licence

MPL-2.0, deliberately — the same as `fasttiff-plugin-api` and
`fasttiff-plugin-abi`. The FastTIFF application is GPL-3.0-only; an SDK under
that licence would set the terms of every plugin written against it, which is
not a decision this project gets to make for you.

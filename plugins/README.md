# Plugins that ship as separate libraries

This directory holds **plugin implementations** built as their own crates —
`cdylib`s that FastTIFF loads at runtime with `dlopen`.

It is deliberately not where the plugin *interface* lives. That is three crates
at the workspace root, and the split matters because they change at completely
different rates:

| | what it is | changes |
|---|---|---|
| `fasttiff-plugin-abi/` | the frozen `#[repr(C)]` wire format | never |
| `fasttiff-plugin-api/` | the Rust traits a plugin implements | freely |
| `fasttiff-plugin/` | the macro that generates the boundary | freely |
| `plugins/` | plugins themselves | independently of all of the above |

The built-in plugins have the same split on the other side of the fence:
`fast-tiff-viewer/src/plugins/` is the host interface, and
`fast-tiff-viewer/src/plugins/builtin/` is the plugins. A file in either plugin
directory may use `fasttiff-plugin-api` and nothing else, so if one of them ever
needs something the API does not offer, that is a gap in the contract rather
than a reason to reach sideways into the viewer.

The fence is the same one, too. The app registers its built-ins through the C
ABI — the same `fasttiff-plugin` registration a `.dll` runs, done in its own
process — so a built-in and a library are one kind of plugin, called the same
way and held to the same rules. `fast-tiff-viewer/tests/plugin_parity.rs` runs
every built-in both directly and through the boundary and requires the same
answer to the bit, which is what keeps that true.

## Not the runtime folder

This is source, not an install location. At runtime FastTIFF looks for compiled
plugins somewhere else entirely — the user's data directory, beside the
executable, or the system plugin directory. **Plugins ▸ Open plugin folder…**
opens the right one; `fasttiff-plugin/README.md` lists them all.

## What is here

- **`example/`** — a filter, a metadata reporter, a raw-binary importer with a
  dialog, and a plugin that panics on purpose. It is also the test oracle for
  the whole C boundary: its `Invert` is a copy of the viewer's built-in one, so
  running both over the same stack must produce byte-identical output.

## Adding one

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
fasttiff-plugin = { path = "../../fasttiff-plugin" }
```

Implement `Plugin` or `Importer`, call `export_plugin!`, and add the directory
to the workspace `members` list. `fasttiff-plugin/README.md` is the guide.

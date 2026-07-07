#!/usr/bin/env python3
"""Bake the perceptual colormaps used by the grayscale color selector into a
Rust source module (`FastTIFF/src/colormap.rs`).

Each colormap is emitted as a full 256-entry RGB lookup table (index = display
intensity 0..=255), matching the layout of the app's other channel LUTs. Baking
them means the app carries no runtime or build-time dependency on matplotlib.

Run from anywhere; writes `../src/colormap.rs` relative to this script.
Requires matplotlib (>= 3.3 for `turbo`).

    python FastTIFF/scripts/gen_colormaps.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# (Rust name, matplotlib colormap name), in the order shown in the UI.
COLORMAPS = [
    ("MAGMA", "magma"),
    ("PLASMA", "plasma"),
    ("INFERNO", "inferno"),
    ("VIRIDIS", "viridis"),
    ("TURBO", "turbo"),
]


def table(mpl_name: str) -> list[tuple[int, int, int]]:
    cmap = plt.get_cmap(mpl_name)
    rows = []
    for i in range(256):
        r, g, b, _ = cmap(i / 255.0)
        rows.append((round(r * 255), round(g * 255), round(b * 255)))
    return rows


def emit_const(rust_name: str, rows: list[tuple[int, int, int]]) -> str:
    lines = [f"const {rust_name}: [[u8; 3]; 256] = ["]
    # Six entries per line keeps the table compact but still scannable.
    for start in range(0, 256, 6):
        chunk = rows[start : start + 6]
        cells = ", ".join(f"[{r}, {g}, {b}]" for r, g, b in chunk)
        lines.append(f"    {cells},")
    lines.append("];")
    return "\n".join(lines)


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "src" / "colormap.rs"

    names = ", ".join(f'"{rust.capitalize()}"' for rust, _ in COLORMAPS)
    consts = ", ".join(rust for rust, _ in COLORMAPS)

    header = f'''//! Perceptual colormaps for displaying a single grayscale channel through a
//! color LUT: the matplotlib "viridis family" (magma, plasma, inferno,
//! viridis) plus Google's turbo. Each is a full 256-entry RGB table indexed by
//! display intensity (0 = `lut[0]`, 255 = `lut[255]`) — the same layout as the
//! app's other channel LUTs (see `fast_tiff_lib::grayscale_lut`).
//!
//! The tables are baked in, so the app has no runtime or build-time dependency
//! on matplotlib. Regenerate with `scripts/gen_colormaps.py` (matplotlib
//! {matplotlib.__version__} was used here).

/// Colormap display names, in UI order (parallel to [`LUTS`]).
pub const NAMES: [&str; {len(COLORMAPS)}] = [{names}];

/// The baked 256-entry RGB lookup tables, in the same order as [`NAMES`].
pub const LUTS: [[[u8; 3]; 256]; {len(COLORMAPS)}] = [{consts}];'''

    blocks = [header]
    for rust_name, mpl_name in COLORMAPS:
        blocks.append(emit_const(rust_name, table(mpl_name)))

    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the frame-count sweep CSV into PNG comparison graphs.

Usage:
    python plot_sweep.py [sweep_results.csv] [out_dir]

Both arguments are optional (defaults: sweep_results.csv, graphs/). The CSV is
produced by `cargo run --release -- sweep`; leading `# ...` comment lines
(machine info) are skipped.

Produces, in out_dir/:
    sweep_open_time.png        open + index time vs frame count
    sweep_per_frame_read.png   mean per-frame read time vs frame count
    sweep_total_time.png       total wall time (open + all reads) vs frame count
    sweep_read_throughput.png  read throughput (MB/s) vs frame count
    sweep_combined.png         all four panels in one figure

Requires matplotlib (`pip install matplotlib`).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: never depends on a GUI backend
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")

# Stable per-reader styling (color + marker), so every chart is consistent.
# Readers not listed here still plot, with an automatic fallback style.
STYLE = {
    "RAW":               dict(color="#9aa0a6", marker="o", label="RAW fread (floor)"),
    "libtiff":           dict(color="#1a73e8", marker="s", label="libtiff (C)"),
    "TinyTIFF":          dict(color="#e8710a", marker="^", label="TinyTIFF (C)"),
    "fast-tiff-lib":     dict(color="#137333", marker="D", label="fast-tiff-lib (Rust)"),
    "fast-tiff-lib (preload)": dict(color="#90ee90", marker="P", label="fast-tiff-lib preload (Rust)"),
    "tiff-rs":           dict(color="#a142f4", marker="v", label="tiff crate (Rust)"),
}
ORDER = ["fast-tiff-lib", "fast-tiff-lib (preload)", "libtiff", "TinyTIFF", "tiff-rs", "RAW"]


def style_for(reader):
    return STYLE.get(reader, dict(color="#90ee90", marker="x", label=reader))


def load(csv_path: Path):
    """reader -> list of numeric row dicts, sorted by frame count."""
    rows = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        data_lines = [l for l in f if not l.lstrip().startswith("#") and l.strip()]
    for r in csv.DictReader(data_lines):
        try:
            open_us = float(r["open_us"])
            total_read_ms = float(r["total_read_ms"])
            rows[r["reader"]].append({
                "frames": int(r["frames"]),
                "open_us": open_us,
                "mean_read_us": float(r["mean_read_us"]),
                "total_read_ms": total_read_ms,
                # What the user actually waits for: open/index + every read.
                "wall_ms": open_us / 1000.0 + total_read_ms,
                "read_throughput_mb_s": float(r["read_throughput_mb_s"]),
            })
        except (KeyError, ValueError) as e:
            sys.exit(f"{csv_path}: malformed row {r}: {e}")
    for k in rows:
        rows[k].sort(key=lambda d: d["frames"])
    return rows


def readers_in(rows):
    known = [r for r in ORDER if r in rows]
    return known + sorted(set(rows) - set(known))


def frame_ticks(rows):
    return sorted({d["frames"] for series in rows.values() for d in series})


def draw_panel(ax, rows, ykey, ylabel, title, yscale=1.0, logy=True):
    ticks = frame_ticks(rows)
    for reader in readers_in(rows):
        xs = [d["frames"] for d in rows[reader]]
        ys = [max(d[ykey] * yscale, 1e-9) for d in rows[reader]]
        st = style_for(reader)
        ax.plot(xs, ys, linestyle="-", marker=st["marker"], color=st["color"], label=st["label"])
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}".replace(",", " ") for t in ticks], fontsize=8)
        ax.minorticks_off()
    ax.set_xlabel("frames in stack")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)


PANELS = [
    # (filename, ykey, yscale, logy, ylabel, title)
    ("sweep_open_time.png", "open_us", 1e-3, True, "open + index time (ms)",
     "Open + IFD-index cost vs frame count"),
    ("sweep_per_frame_read.png", "mean_read_us", 1.0, True, "mean per-frame read (us)",
     "Steady-state per-frame read time"),
    ("sweep_total_time.png", "wall_ms", 1.0, True, "open + all reads (ms)",
     "Total wall time for one full pass"),
    ("sweep_read_throughput.png", "read_throughput_mb_s", 1.0, False, "read throughput (MB/s)",
     "Read throughput"),
]


def render(csv_path: Path, out_dir: Path):
    """Render every sweep chart from `csv_path` into `out_dir`. Callable from
    other scripts (plot_results.py delegates here when a sweep CSV exists)."""
    rows = load(csv_path)
    if not rows:
        sys.exit(f"{csv_path} contains no data rows — re-run the benchmark.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual panels.
    for fname, ykey, yscale, logy, ylabel, title in PANELS:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        draw_panel(ax, rows, ykey, ylabel, title, yscale=yscale, logy=logy)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")

    # Combined 2x2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (_, ykey, yscale, logy, ylabel, title) in zip(axes.flat, PANELS):
        draw_panel(ax, rows, ykey, ylabel, title, yscale=yscale, logy=logy)
    fig.suptitle("TIFF reader frame-count sweep  (16x16, 16-bit, single-strip)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "sweep_combined.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sweep_results.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("graphs")
    if not csv_path.exists():
        sys.exit(
            f"{csv_path} not found.\n"
            "Generate it first with the sweep benchmark:\n"
            "    cargo run --release -- sweep\n"
            "then re-run:  python plot_sweep.py"
        )
    render(csv_path, out_dir)


if __name__ == "__main__":
    main()

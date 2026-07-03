#!/usr/bin/env python3
"""
Render the frame-count sweep CSV (produced by `tiff_read_bench sweep`) into
PNG comparison graphs.

Usage:
    python3 plot_sweep.py sweep_results.csv out_dir/

Produces, in out_dir/:
    sweep_open_time.png        open + index time vs frame count
    sweep_per_frame_read.png   mean per-frame read time vs frame count
    sweep_total_time.png       total time (open + all reads) vs frame count
    sweep_read_throughput.png  read throughput (MB/s) vs frame count
    sweep_combined.png         all four panels in one figure
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# Stable per-reader styling (color + marker), so every chart is consistent.
STYLE = {
    "RAW":               dict(color="#9aa0a6", marker="o", label="RAW fread (floor)"),
    "libtiff":           dict(color="#1a73e8", marker="s", label="libtiff (C)"),
    "TinyTIFF":          dict(color="#e8710a", marker="^", label="TinyTIFF (C)"),
    "fast-tiff-lib":     dict(color="#137333", marker="D", label="fast-tiff-lib (Rust)"),
    "fast-tiff-preload": dict(color="#81c995", marker="P", label="fast-tiff-lib preload (batch)"),
    "tiff-rs":           dict(color="#a142f4", marker="v", label="tiff crate (Rust)"),
}
ORDER = ["fast-tiff-lib", "fast-tiff-preload", "libtiff", "TinyTIFF", "tiff-rs", "RAW"]


def load(csv_path):
    """reader -> sorted list of row dicts (numeric)."""
    rows = defaultdict(list)
    with open(csv_path, newline="") as f:
        # The bench embeds system info as leading `# ...` comment lines.
        data_lines = [l for l in f if not l.startswith("#")]
        for r in csv.DictReader(data_lines):
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
    for k in rows:
        rows[k].sort(key=lambda d: d["frames"])
    return rows


def _series(rows, reader, ykey, yscale=1.0):
    xs = [d["frames"] for d in rows[reader]]
    ys = [d[ykey] * yscale for d in rows[reader]]
    return xs, ys


def _tidy_logx(ax, xs):
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs], rotation=45, ha="right")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", ls="--", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.15)


def _plot_on(ax, rows, ykey, yscale, ylabel, title, logy=True):
    any_reader = next(iter(rows))
    xs_all = [d["frames"] for d in rows[any_reader]]
    for reader in ORDER:
        if reader not in rows:
            continue
        xs, ys = _series(rows, reader, ykey, yscale)
        st = STYLE[reader]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                label=st["label"], lw=2, ms=6, markeredgecolor="white",
                markeredgewidth=0.6)
    _tidy_logx(ax, xs_all)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlabel("number of frames in stack")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=11)


def save_single(rows, ykey, yscale, ylabel, title, out, logy=True, note=None):
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=140)
    _plot_on(ax, rows, ykey, yscale, ylabel, title, logy=logy)
    ax.legend(frameon=True, fontsize=9, loc="best")
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def save_combined(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
    _plot_on(axes[0, 0], rows, "open_us", 1e-3,
             "open + index time (ms, log)",
             "1. Open / index cost vs frame count")
    _plot_on(axes[0, 1], rows, "mean_read_us", 1.0,
             "mean per-frame read (µs, log)",
             "2. Per-frame read latency vs frame count")
    _plot_on(axes[1, 0], rows, "total_read_ms", 1.0,
             "total read time, all frames (ms, log)",
             "3. Total read time vs frame count")
    _plot_on(axes[1, 1], rows, "read_throughput_mb_s", 1.0,
             "read throughput (MB/s)",
             "4. Read throughput vs frame count", logy=False)
    # One shared legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("TIFF reader frame-count sweep  (16×16, 16-bit, single-strip)",
                 fontweight="bold", fontsize=14, y=1.06)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(csv_path)

    eager_note = ("Eager indexers (fast-tiff-lib, TinyTIFF) pay open cost that "
                  "scales with frame count; lazy ones (libtiff, tiff crate) do not.")

    save_single(rows, "open_us", 1e-3, "open + index time (ms, log scale)",
                "Open / index cost vs frame count",
                out_dir / "sweep_open_time.png", note=eager_note)
    save_single(rows, "mean_read_us", 1.0, "mean per-frame read time (µs, log scale)",
                "Per-frame read latency vs frame count",
                out_dir / "sweep_per_frame_read.png")
    save_single(rows, "total_read_ms", 1.0, "total read time for all frames (ms, log scale)",
                "Total read time (all frames) vs frame count",
                out_dir / "sweep_total_time.png")
    save_single(rows, "read_throughput_mb_s", 1.0, "read throughput (MB/s)",
                "Read throughput vs frame count",
                out_dir / "sweep_read_throughput.png", logy=False)
    save_combined(rows, out_dir / "sweep_combined.png")


if __name__ == "__main__":
    main()

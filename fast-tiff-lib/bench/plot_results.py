#!/usr/bin/env python3
"""One-figure infographic from the matrix benchmark's bench_results.csv.

Usage:
    python plot_results.py [bench_results.csv] [bench_summary.png]

Both arguments are optional (defaults shown). The CSV is produced by
`cargo run --release` (matrix mode); its leading `# ...` comment lines carry
the machine/toolchain info and are rendered into panel D.

Panels:
  A. Relative read speed per reader (geometric mean; 1.0 = fastest TIFF reader)
  B. Mean read throughput by compression codec, per reader
  C. Per-frame read time vs frame count (256x256 u16, uncompressed, 1 strip)
  D. Machine + toolchain info and fast-tiff-lib write throughput by codec

Requires matplotlib (`pip install matplotlib`).
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: never depends on a GUI backend
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")

READER_ORDER = ["RAW", "fast-tiff-lib", "fast-tiff-preload", "tiff-rs", "TinyTIFF", "libtiff"]
COLORS = {
    "RAW": "#9e9e9e",
    "fast-tiff-lib": "#d62728",
    "fast-tiff-preload": "#ff9896",
    "tiff-rs": "#1f77b4",
    "TinyTIFF": "#2ca02c",
    "libtiff": "#9467bd",
}
FALLBACK_COLOR = "#555555"
NUMERIC = ("open_us", "mean_us", "min_us", "mb_s", "rel", "write_mb_s")


def geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load(csv_path: Path):
    """Returns (sysinfo_lines, ok_rows). Tolerates comment lines anywhere."""
    sysinfo, rows = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        data_lines = []
        for line in f:
            if line.lstrip().startswith("#"):
                sysinfo.append(line.lstrip()[1:].strip())
            elif line.strip():
                data_lines.append(line)
        for r in csv.DictReader(data_lines):
            if r.get("ok") != "true":
                continue
            try:
                for k in NUMERIC:
                    r[k] = float(r[k])
                r["frames"] = int(r["frames"])
            except (KeyError, ValueError) as e:
                sys.exit(f"{csv_path}: malformed row {r}: {e}")
            rows.append(r)
    return sysinfo, rows


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench_results.csv")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bench_summary.png")
    if not csv_path.exists():
        sys.exit(
            f"{csv_path} not found.\n"
            "Generate it first with the matrix benchmark:\n"
            "    cargo run --release\n"
            "then re-run:  python plot_results.py"
        )

    sysinfo, ok = load(csv_path)
    if not ok:
        sys.exit(f"{csv_path} contains no successful runs — re-run the benchmark.")

    readers = [n for n in READER_ORDER if any(r["reader"] == n for r in ok)]
    readers += sorted({r["reader"] for r in ok} - set(readers))  # future-proof
    color = lambda name: COLORS.get(name, FALLBACK_COLOR)  # noqa: E731

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("TIFF reader/writer benchmark — overall summary", fontsize=15, fontweight="bold")

    # --- A: geomean relative speed --------------------------------------
    ax = axes[0][0]
    names, geos = [], []
    for name in readers:
        rels = [r["rel"] for r in ok if r["reader"] == name]
        if rels:
            names.append(f"{name}\n({len(rels)} runs)")
            geos.append(geomean(rels))
    bars = ax.barh(names, geos, color=[color(n.split("\n")[0]) for n in names])
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    for b, g in zip(bars, geos):
        ax.text(b.get_width() + max(geos) * 0.01, b.get_y() + b.get_height() / 2, f"{g:.2f}x", va="center", fontsize=9)
    ax.set_xlim(0, max(geos) * 1.15)
    ax.set_xlabel("relative per-frame read time (geometric mean; 1.0 = fastest TIFF reader)")
    ax.set_title("A. Overall relative read speed (lower is better; RAW = no-decode floor)")
    ax.invert_yaxis()

    # --- B: throughput by codec ------------------------------------------
    ax = axes[0][1]
    codecs = sorted({r["compression"] for r in ok})
    present = [n for n in readers if any(r["reader"] == n for r in ok)]
    width = 0.8 / max(len(present), 1)
    for i, name in enumerate(present):
        xs, ys = [], []
        for j, codec in enumerate(codecs):
            v = [r["mb_s"] for r in ok if r["reader"] == name and r["compression"] == codec]
            if v:
                xs.append(j + i * width)
                ys.append(sum(v) / len(v))
        if xs:
            ax.bar(xs, ys, width=width, label=name, color=color(name))
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(codecs))])
    ax.set_xticklabels(codecs)
    ax.set_ylabel("mean read throughput (MB/s, decoded)")
    ax.set_title("B. Read throughput by compression codec")
    ax.legend(fontsize=8)

    # --- C: frame-count scaling -------------------------------------------
    ax = axes[1][0]
    family = [
        r
        for r in ok
        if r["format"] == "u16"
        and r["compression"] == "none"
        and r["strips"] == "1"
        and r["bigtiff"] == "false"
        and r["width"] == "256"
    ]
    counts = sorted({r["frames"] for r in family})
    plotted = False
    for name in readers:
        pts = sorted((r["frames"], r["mean_us"]) for r in family if r["reader"] == name)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name, color=color(name))
            plotted = True
    if plotted and len(counts) > 1:
        ax.set_xscale("log")
        ax.set_xticks(counts)
        ax.set_xticklabels([str(c) for c in counts])
        ax.minorticks_off()
    if not plotted:
        ax.text(0.5, 0.5, "no 256x256 u16 uncompressed runs in the CSV", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("frames in stack")
    ax.set_ylabel("mean per-frame read (us)")
    ax.set_title("C. Frame-count scaling (256x256 u16, uncompressed, 1 strip)")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- D: system info + write speeds --------------------------------------
    ax = axes[1][1]
    ax.axis("off")
    write_by_codec = defaultdict(list)
    for r in ok:
        if r["reader"] == "fast-tiff-lib":
            write_by_codec[r["compression"]].append(r["write_mb_s"])
    lines = ["Benchmark machine:"] + [f"  {s}" for s in sysinfo]
    lines.append("")
    lines.append("fast-tiff-lib WRITE throughput by codec (all stacks written by it):")
    for codec in sorted(write_by_codec):
        v = write_by_codec[codec]
        lines.append(f"  {codec:<9} {sum(v) / len(v):>7.0f} MB/s")
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace", transform=ax.transAxes)
    ax.set_title("D. Environment & writer throughput")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

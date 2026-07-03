#!/usr/bin/env python3
"""One-figure infographic from the matrix benchmark's bench_results.csv.

Usage:  python plot_results.py [bench_results.csv] [bench_summary.png]

Panels:
  A. Relative read speed per reader (geometric mean; 1.0 = fastest TIFF reader)
  B. Mean read throughput by compression codec, per reader
  C. Per-frame read time vs frame count (u16 / uncompressed / single-strip)
  D. Machine + toolchain info and fast-tiff-lib write throughput by codec

Requires matplotlib (`pip install matplotlib`).
"""

import csv
import math
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "bench_results.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "bench_summary.png"

READER_ORDER = ["RAW", "fast-tiff-lib", "fast-tiff-preload", "tiff-rs", "TinyTIFF", "libtiff"]
COLORS = {
    "RAW": "#9e9e9e",
    "fast-tiff-lib": "#d62728",
    "fast-tiff-preload": "#ff9896",
    "tiff-rs": "#1f77b4",
    "TinyTIFF": "#2ca02c",
    "libtiff": "#9467bd",
}


def geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ----------------------------- load ----------------------------------------
sysinfo, rows = [], []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    header = None
    for line in f:
        if line.startswith("#"):
            sysinfo.append(line[1:].strip())
            continue
        if header is None:
            header = next(csv.reader([line]))
            continue
        rows.append(dict(zip(header, next(csv.reader([line])))))

ok = [r for r in rows if r["ok"] == "true"]
for r in ok:
    for k in ("open_us", "mean_us", "min_us", "mb_s", "rel", "write_mb_s"):
        r[k] = float(r[k])
    r["frames"] = int(r["frames"])

readers = [n for n in READER_ORDER if any(r["reader"] == n for r in ok)]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("TIFF reader/writer benchmark — overall summary", fontsize=15, fontweight="bold")

# --- A: geomean relative speed ---------------------------------------------
ax = axes[0][0]
names, geos = [], []
for name in readers:
    rels = [r["rel"] for r in ok if r["reader"] == name]
    if rels:
        names.append(f"{name}\n({len(rels)} runs)")
        geos.append(geomean(rels))
bars = ax.barh(names, geos, color=[COLORS.get(n.split("\n")[0], "#333") for n in names])
ax.axvline(1.0, color="black", lw=0.8, ls="--")
for b, g in zip(bars, geos):
    ax.text(b.get_width() * 1.02, b.get_y() + b.get_height() / 2, f"{g:.2f}x", va="center", fontsize=9)
ax.set_xlabel("relative per-frame read time (geometric mean; 1.0 = fastest TIFF reader)")
ax.set_title("A. Overall relative read speed (lower is better; RAW = no-decode floor)")
ax.invert_yaxis()

# --- B: throughput by codec --------------------------------------------------
ax = axes[0][1]
codecs = sorted({r["compression"] for r in ok})
width = 0.8 / max(len(readers), 1)
for i, name in enumerate(readers):
    xs, ys = [], []
    for j, codec in enumerate(codecs):
        v = [r["mb_s"] for r in ok if r["reader"] == name and r["compression"] == codec]
        if v:
            xs.append(j + i * width)
            ys.append(sum(v) / len(v))
    ax.bar(xs, ys, width=width, label=name, color=COLORS.get(name, "#333"))
ax.set_xticks([j + 0.4 for j in range(len(codecs))])
ax.set_xticklabels(codecs)
ax.set_ylabel("mean read throughput (MB/s, decoded)")
ax.set_title("B. Read throughput by compression codec")
ax.legend(fontsize=8)

# --- C: frame-count scaling ---------------------------------------------------
ax = axes[1][0]
family = [
    r
    for r in ok
    if r["format"] == "u16" and r["compression"] == "none" and r["strips"] == "1"
    and r["bigtiff"] == "false" and r["width"] == "256"
]
for name in readers:
    pts = sorted((r["frames"], r["mean_us"]) for r in family if r["reader"] == name)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name, color=COLORS.get(name, "#333"))
ax.set_xscale("log")
ax.set_xlabel("frames in stack")
ax.set_ylabel("mean per-frame read (us)")
ax.set_title("C. Frame-count scaling (256x256 u16, uncompressed, 1 strip)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- D: system info + write speeds -------------------------------------------
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
fig.savefig(OUT_PATH, dpi=140)
print(f"wrote {OUT_PATH}")

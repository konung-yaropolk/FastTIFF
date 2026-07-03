#!/usr/bin/env python3
"""Figures from the matrix benchmark's bench_results.csv.

Usage:
    python plot_results.py [bench_results.csv] [bench_summary.png] [graphs_dir]

All arguments optional (defaults: bench_results.csv, bench_summary.png,
graphs/). The CSV is produced by `cargo run --release` (matrix mode); its
leading `# ...` comment lines carry machine/toolchain info.

Produces:
  bench_summary.png            overall 4-panel infographic
  <graphs>/all_tests.png       compilation: one mini bar chart per test
  <graphs>/tests/NN_<slug>.png one bar chart per individual test

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
    from matplotlib.patches import Patch
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
SHORT = {
    "RAW": "RAW",
    "fast-tiff-lib": "fast",
    "fast-tiff-preload": "preload",
    "tiff-rs": "tiff-rs",
    "TinyTIFF": "TinyTIFF",
    "libtiff": "libtiff",
}
FALLBACK_COLOR = "#555555"
NUMERIC = ("open_us", "mean_us", "min_us", "mb_s", "rel", "write_mb_s")


def color(name):
    return COLORS.get(name, FALLBACK_COLOR)


def short(name):
    return SHORT.get(name, name)


def geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def slugify(config):
    s = "".join(c if c.isalnum() else "_" for c in config)
    return "_".join(t for t in s.split("_") if t)


def load(csv_path: Path):
    """Returns (sysinfo, ok_rows, skipped_by_config).

    `skipped_by_config[config]` is a set of reader names that couldn't run
    that configuration (so per-test charts can note them)."""
    sysinfo, ok, skipped = [], [], defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as f:
        data_lines = []
        for line in f:
            if line.lstrip().startswith("#"):
                sysinfo.append(line.lstrip()[1:].strip())
            elif line.strip():
                data_lines.append(line)
        for r in csv.DictReader(data_lines):
            if r.get("ok") != "true":
                skipped[r.get("config", "")].add(r.get("reader", "?"))
                continue
            try:
                for k in NUMERIC:
                    r[k] = float(r[k])
                r["frames"] = int(r["frames"])
            except (KeyError, ValueError) as e:
                sys.exit(f"{csv_path}: malformed row {r}: {e}")
            ok.append(r)
    return sysinfo, ok, skipped


# --------------------------- per-test bar chart ----------------------------


def draw_test(ax, rows, skipped, compact=False):
    """One test = one config: horizontal bars of each reader's mean us/frame,
    fastest at the top. `rel` is vs the fastest *TIFF reader* (RAW excluded,
    it's the no-decode floor)."""
    rows = sorted(rows, key=lambda r: r["mean_us"])
    means = [r["mean_us"] for r in rows]
    non_raw = [r["mean_us"] for r in rows if r["reader"] != "RAW"]
    best = min(non_raw) if non_raw else (min(means) if means else 1.0)

    ax.barh(range(len(rows)), means, color=[color(r["reader"]) for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short(r["reader"]) for r in rows], fontsize=6 if compact else 9)
    ax.invert_yaxis()

    label_fs = 6 if compact else 8
    for i, r in enumerate(rows):
        rel = r["mean_us"] / best if best else 1.0
        txt = f"{r['mean_us']:.0f}" if compact else f"{r['mean_us']:.1f} us  ({rel:.2f}x)"
        ax.text(r["mean_us"], i, " " + txt, va="center", ha="left", fontsize=label_fs)

    ax.set_xlim(0, (max(means) if means else 1.0) * (1.4 if compact else 1.6))
    ax.tick_params(axis="x", labelsize=6 if compact else 8)
    ax.margins(y=0.02)
    if not compact:
        ax.set_xlabel("mean us/frame  (lower is better)")
        if skipped:
            ax.text(
                0.99, 0.02, "n/s: " + ", ".join(sorted(short(s) for s in skipped)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#888",
            )


def per_test_figures(order, grouped, skipped, tests_dir: Path):
    tests_dir.mkdir(parents=True, exist_ok=True)
    for i, cfg in enumerate(order, 1):
        fig, ax = plt.subplots(figsize=(7, 2.6))
        draw_test(ax, grouped[cfg], skipped.get(cfg, set()), compact=False)
        ax.set_title(cfg, fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(tests_dir / f"{i:02d}_{slugify(cfg)}.png", dpi=130)
        plt.close(fig)
    print(f"wrote {len(order)} per-test charts to {tests_dir}/")


def compilation_figure(order, grouped, skipped, readers_present, out_path: Path):
    n = len(order)
    cols = 4 if n <= 12 else (5 if n <= 30 else 6)
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.5, nrows * 2.15), squeeze=False)
    for idx, cfg in enumerate(order):
        ax = axes[idx // cols][idx % cols]
        draw_test(ax, grouped[cfg], skipped.get(cfg, set()), compact=True)
        ax.set_title(cfg, fontsize=8)
    for idx in range(n, nrows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles = [Patch(color=color(r), label=r) for r in readers_present]
    top = 1.0 - 0.5 / (nrows * 2.15)  # scale headroom to the figure height
    fig.suptitle(
        "Per-test comparison — mean us/frame per reader (lower is better)",
        fontsize=15, fontweight="bold", y=0.997,
    )
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=10, frameon=False, bbox_to_anchor=(0.5, top))
    fig.tight_layout(rect=[0, 0, 1, top - 0.01])
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------------------- overall summary -----------------------------


def summary_figure(sysinfo, ok, readers, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("TIFF reader/writer benchmark — overall summary", fontsize=15, fontweight="bold")

    # A: geomean relative speed
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

    # B: throughput by codec
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
        if xs:
            ax.bar(xs, ys, width=width, label=name, color=color(name))
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(codecs))])
    ax.set_xticklabels(codecs)
    ax.set_ylabel("mean read throughput (MB/s, decoded)")
    ax.set_title("B. Read throughput by compression codec")
    ax.legend(fontsize=8)

    # C: frame-count scaling
    ax = axes[1][0]
    family = [
        r for r in ok
        if r["format"] == "u16" and r["compression"] == "none" and r["strips"] == "1"
        and r["bigtiff"] == "false" and r["width"] == "256"
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

    # D: system info + write speeds
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
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench_results.csv")
    summary_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bench_summary.png")
    graphs_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("graphs")
    if not csv_path.exists():
        sys.exit(
            f"{csv_path} not found.\n"
            "Generate it first with the matrix benchmark:\n"
            "    cargo run --release\n"
            "then re-run:  python plot_results.py"
        )

    sysinfo, ok, skipped = load(csv_path)
    if not ok:
        sys.exit(f"{csv_path} contains no successful runs — re-run the benchmark.")

    readers = [n for n in READER_ORDER if any(r["reader"] == n for r in ok)]
    readers += sorted({r["reader"] for r in ok} - set(readers))  # future-proof

    grouped = defaultdict(list)
    for r in ok:
        grouped[r["config"]].append(r)
    order = list(grouped)  # first-seen (= benchmark emission) order

    summary_figure(sysinfo, ok, readers, summary_path)
    compilation_figure(order, grouped, skipped, readers, graphs_dir / "all_tests.png")
    per_test_figures(order, grouped, skipped, graphs_dir / "tests")


if __name__ == "__main__":
    main()

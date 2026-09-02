#!/usr/bin/env python3
"""Figures for the TIFF benchmark, from the single `bench_results.csv`.

One input, one command, three kinds of output:

  bench_summary.png      the headline: who is faster, at what, and how it scales
  graphs/scaling.png     one panel per family, us/frame against frame count
  graphs/runs/NN_*.png   one bar chart per individual run

There used to be two scripts reading two different CSVs, and they disagreed
about what the readers were called: this one looked for `fast-tiff-lib
(preload)` while the benchmark wrote `fast-tiff-preload`, so that series lost
its colour and its position in the order without anyone noticing. Reader
identity now comes from the CSV's `reader` column, which holds the ids defined
once in `src/reader.rs`; READERS below mirrors those ids and nothing else keys
off a display name.

    python plot.py [bench_results.csv] [outdir]
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

# --------------------------------------------------------------------------
# Identity and style
# --------------------------------------------------------------------------

# Keyed by the ids in src/reader.rs. Order is the order everything is drawn in,
# so a reader keeps the same colour and the same position in every figure — the
# eye should not have to re-learn the legend between panels.
READERS = [
    # id                    label                      colour
    ("raw",               "RAW fread",                "#9aa0a6"),
    ("fast-tiff",         "fast-tiff-lib",            "#137333"),
    ("fast-tiff-preload", "fast-tiff-lib (preload)",  "#5bb974"),
    ("tiff-rs",           "tiff-rs",                  "#a142f4"),
    ("tinytiff",          "TinyTIFF (C)",             "#e8710a"),
    ("libtiff",           "libtiff (C)",              "#1a73e8"),
]
ORDER = [r[0] for r in READERS]
LABEL = {r[0]: r[1] for r in READERS}
COLOR = {r[0]: r[2] for r in READERS}
FLOOR = "raw"  # does no decoding; a reference line, not a competitor

INK = "#202124"
MUTED = "#5f6368"
GRID = "#e3e5e8"
PAPER = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11.5,
    "legend.frameon": False,
    "font.size": 10,
    "figure.dpi": 110,
})


def tidy(ax, xgrid=True, ygrid=True):
    """Drop the box, keep only the gridlines the panel actually reads along."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", visible=xgrid)
    ax.grid(axis="y", visible=ygrid)
    # `which="both"`: minor ticks default to being drawn, and on a log
    # axis they read as a column of stray stubs beside the real labels.
    ax.tick_params(which="both", length=0)


# The library this benchmark is about. Its two entries are picked out wherever
# they are named, so a reader scanning a wall of charts can find them without
# reading every label.
HIGHLIGHT_PREFIX = "fast-tiff-lib"


def emphasise_subject(fig):
    r"""Bold and underline every label naming the subject of the benchmark.

    Matplotlib has no underline: mathtext's `\underline` would work but it
    renders hyphens as minus signs and eats the spaces, so "fast-tiff-lib
    (preload)" would come out subtly wrong. Instead the text is measured after
    it is bolded and a rule is drawn under it in figure coordinates, which
    survives the tight bounding box `savefig` applies.

    Called once per figure, just before saving, so no panel has to remember.
    """
    targets = []
    for ax in fig.get_axes():
        targets += [t for t in ax.get_yticklabels() + ax.get_xticklabels()]
        legend = ax.get_legend()
        if legend:
            targets += legend.get_texts()
    if fig.legends:
        for legend in fig.legends:
            targets += legend.get_texts()

    named = [t for t in targets if t.get_text().startswith(HIGHLIGHT_PREFIX)]
    if not named:
        return
    for t in named:
        t.set_fontweight("bold")

    # Bolding changes the extents, so measure only after it is applied.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for t in named:
        bb = t.get_window_extent()
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, _) = inv.transform((bb.x1, bb.y0))
        drop = 0.16 * (inv.transform((0, bb.height))[1] - inv.transform((0, 0))[1])
        fig.add_artist(
            Line2D([x0, x1], [y0 - drop, y0 - drop],
                   color=t.get_color(), lw=0.9, zorder=5)
        )

    plt.style.use('seaborn-v0_8')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

NUMERIC = ("open_us", "mean_us", "min_us", "total_read_ms", "mb_s", "rel", "write_mb_s")
INTS = ("strips", "width", "height", "frames")


def load(path: Path):
    """Return (environment lines, rows). `#` comments carry the machine."""
    env, body = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        (env if line.startswith("#") else body).append(line.lstrip("# ") if line.startswith("#") else line)

    rows = []
    for r in csv.DictReader(body):
        for k in NUMERIC:
            r[k] = float(r[k] or 0.0)
        for k in INTS:
            r[k] = int(r[k] or 0)
        r["ok"] = r["ok"] == "true"
        rows.append(r)
    return env, rows


def present(rows):
    """Reader ids that actually appear, in the canonical order."""
    seen = {r["reader"] for r in rows}
    return [r for r in ORDER if r in seen]


def geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    from math import exp, log
    return exp(sum(log(v) for v in vals) / len(vals))


def fmt_count(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def fmt_si(v, _=None):
    """Axis tick that stays readable across the six decades these span."""
    if v <= 0:
        return ""
    if v >= 1000:
        return f"{v / 1000:g}k"
    if v >= 1:
        return f"{v:g}"
    return f"{v:g}"


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def panel_relative(ax, rows, readers):
    """Who is faster overall: geometric mean of per-run relative speed."""
    stats = []
    for rid in readers:
        rels = [r["rel"] for r in rows if r["reader"] == rid and r["ok"] and r["rel"] > 0]
        if rels:
            wins = sum(1 for v in rels if abs(v - 1.0) < 1e-9)
            stats.append((rid, geomean(rels), len(rels), wins))
    stats.sort(key=lambda s: s[1])

    y = range(len(stats))
    ax.barh(list(y), [s[1] for s in stats],
            color=[COLOR[s[0]] for s in stats], height=0.62, zorder=3)
    ax.set_yticks(list(y), [LABEL[s[0]] for s in stats], fontsize=9.5)
    ax.invert_yaxis()
    ax.axvline(1.0, color=INK, lw=1.1, ls="--", zorder=4)
    ax.set_xlabel("relative time per frame  (1.0 = fastest TIFF reader in each run; lower is better)")
    ax.set_title("Overall read speed")
    tidy(ax, xgrid=True, ygrid=False)

    span = max((s[1] for s in stats), default=1.0)
    for i, (rid, geo, n, wins) in enumerate(stats):
        note = f"{geo:.2f}x   {n} runs, {wins} wins"
        if rid == FLOOR:
            note += "   (no decode)"
        ax.text(geo + span * 0.02, i, note, va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, span * 1.42)


def panel_codecs(ax, rows, readers):
    """Where the differences come from: throughput per codec."""
    codecs = sorted({r["compression"] for r in rows if r["ok"]})
    readers = [r for r in readers if r != FLOOR]
    width = 0.8 / max(len(readers), 1)

    for i, rid in enumerate(readers):
        vals = []
        for c in codecs:
            v = [r["mb_s"] for r in rows if r["ok"] and r["reader"] == rid and r["compression"] == c]
            vals.append(sum(v) / len(v) if v else 0.0)
        xs = [j - 0.4 + width * (i + 0.5) for j in range(len(codecs))]
        ax.bar(xs, vals, width=width * 0.92, color=COLOR[rid], label=LABEL[rid], zorder=3)

    ax.set_xticks(range(len(codecs)), codecs)
    ax.set_ylabel("MB/s decoded (mean)")
    ax.set_title("Read throughput by codec")
    tidy(ax, xgrid=False, ygrid=True)
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")


def scaling_series(rows, family, rid, key="mean_us"):
    pts = sorted(
        (r["frames"], r[key])
        for r in rows
        if r["ok"] and r["family"] == family and r["reader"] == rid
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def panel_scaling(ax, rows, readers, family, key, ylabel, title):
    """The swept reading: how a number moves as the stack gets longer."""
    drawn = False
    for rid in readers:
        xs, ys = scaling_series(rows, family, rid, key)
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, "-o", color=COLOR[rid], label=LABEL[rid],
                lw=1.9, ms=4.2, zorder=3)
        drawn = True
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frames in the stack")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_count(int(v))))
    # A log axis over one or two decades auto-labels its minor ticks, which
    # renders as a column of stray stubs next to the real ones. Decades only.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_si))
    tidy(ax)
    if not drawn:
        ax.text(0.5, 0.5, "not enough frame counts\n(run without --quick)",
                ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=9)
    return drawn


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def summary_figure(env, rows, out: Path):
    readers = present(rows)
    ok = [r for r in rows if r["ok"]]
    families = sorted({r["family"] for r in rows})
    sweep_family = families[0] if families else ""
    # The family with the most frame counts is the one worth sweeping; ties go
    # to the first, which is the tiny-frame family the matrix leads with.
    best_n = -1
    for fam in families:
        n = len({r["frames"] for r in rows if r["family"] == fam})
        if n > best_n:
            best_n, sweep_family = n, fam

    fig = plt.figure(figsize=(15.5, 11.6))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.02, 0.95, 0.62], hspace=0.42, wspace=0.22,
        left=0.075, right=0.975, top=0.885, bottom=0.055,
    )

    fig.suptitle("fast-tiff-lib — TIFF read benchmark", x=0.075, y=0.968,
                 ha="left", fontsize=19, fontweight="bold", color=INK)
    runs = len({r["run"] for r in rows})
    fig.text(0.075, 0.933,
             f"{runs} runs — {len(families)} configurations x frame counts — "
             f"{len(readers)} readers, all decoding into owned buffers",
             ha="left", fontsize=10.5, color=MUTED)

    panel_relative(fig.add_subplot(grid[0, 0]), rows, readers)
    panel_codecs(fig.add_subplot(grid[0, 1]), rows, readers)
    panel_scaling(fig.add_subplot(grid[1, 0]), rows, readers, sweep_family,
                  "mean_us", "microseconds per frame",
                  f"Per-frame cost as the stack grows\n{sweep_family}")
    panel_scaling(fig.add_subplot(grid[1, 1]), rows, readers, sweep_family,
                  "open_us", "microseconds to open + index",
                  f"Cost paid once, at open\n{sweep_family}")

    # Footer: the machine on the left, what the writer managed on the right.
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    ax.text(0.0, 1.0, "Machine", fontsize=10.5, fontweight="bold", va="top")
    ax.text(0.0, 0.84, "\n".join(env[:6]), fontsize=8.6, family="monospace",
            va="top", color=MUTED, linespacing=1.5)

    codecs = sorted({r["compression"] for r in rows})
    writes = []
    for c in codecs:
        v = [r["write_mb_s"] for r in rows if r["compression"] == c and r["reader"] == "fast-tiff"]
        if v:
            writes.append((c, sum(v) / len(v)))
    ax.text(0.56, 1.0, "fast-tiff-lib write throughput  (it produced every stack)",
            fontsize=10.5, fontweight="bold", va="top")
    if writes:
        # The bar track stops well short of the value column: at full width the
        # fastest codec's bar ran underneath its own number.
        top = max(w for _, w in writes)
        bar_left, bar_span = 0.645, 0.21
        for i, (c, w) in enumerate(writes):
            y = 0.80 - i * 0.145
            ax.text(0.56, y, f"{c:<9}", fontsize=9, family="monospace", va="center", color=MUTED)
            ax.barh([y], [bar_span * w / top], left=bar_left, height=0.085,
                    color="#137333", alpha=0.85)
            ax.text(1.0, y, f"{w:,.0f} MB/s", fontsize=9, va="center",
                    ha="right", color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

     

    emphasise_subject(fig)
    fig.savefig(out, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out


def scaling_figure(rows, out: Path):
    """One panel per family: every configuration's scaling in a single sheet."""
    readers = present(rows)
    families = sorted({r["family"] for r in rows})
    families = [f for f in families
                if len({r["frames"] for r in rows if r["family"] == f and r["ok"]}) >= 2]
    if not families:
        return None

    cols = 3
    rowsn = (len(families) + cols - 1) // cols
    fig, axes = plt.subplots(rowsn, cols, figsize=(5.1 * cols, 3.5 * rowsn), squeeze=False)
    fig.suptitle("Per-frame read cost against stack length, by configuration",
                 x=0.5, y=0.995, fontsize=15, fontweight="bold")

    for i, fam in enumerate(families):
        ax = axes[i // cols][i % cols]
        panel_scaling(ax, rows, readers, fam, "mean_us", "us / frame", fam)
        ax.set_title(fam, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("us / frame" if i % cols == 0 else "")
    for j in range(len(families), rowsn * cols):
        axes[j // cols][j % cols].axis("off")

    handles = [Line2D([], [], color=COLOR[r], lw=2.2, marker="o", ms=4.5, label=LABEL[r])
               for r in readers]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9.5, bbox_to_anchor=(0.5, -0.006))

     
    
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    emphasise_subject(fig)

    fig.savefig(out, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out


def run_figures(rows, out_dir: Path):
    """A labelled bar chart per run, including who could not read it."""
    # Clear first: a run with different frame counts produces differently
    # named charts, and the leftovers from the previous one are stale results
    # sitting in the same folder looking current.
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    runs = sorted({r["run"] for r in rows}, key=lambda s: (len(s), s))
    made = []
    for i, run in enumerate(runs, 1):
        mine = [r for r in rows if r["run"] == run]
        done = sorted((r for r in mine if r["ok"]), key=lambda r: r["mean_us"])
        missing = [r for r in mine if not r["ok"]]
        if not done:
            continue

        fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(done) + 2.5))
        y = range(len(done))
        ax.barh(list(y), [r["mean_us"] for r in done],
                color=[COLOR.get(r["reader"], MUTED) for r in done], height=0.6, zorder=3)
        ax.set_yticks(list(y), [LABEL.get(r["reader"], r["reader"]) for r in done], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel("microseconds per frame (trimmed mean; lower is better)")
        ax.set_title(run, fontsize=11)
        tidy(ax, xgrid=True, ygrid=False)

        span = max(r["mean_us"] for r in done) or 1.0
        for i2, r in enumerate(done):
            tag = f"{r['mean_us']:.2f} us   {r['mb_s']:,.0f} MB/s   {r['rel']:.2f}x"
            if r["reader"] == FLOOR:
                tag += "  (no decode)"
            ax.text(r["mean_us"] + span * 0.02, i2, tag, va="center", fontsize=8.4, color=MUTED)
        ax.set_xlim(0, span * 1.5)

        if missing:
            note = "  |  ".join(f"{LABEL.get(m['reader'], m['reader'])}: {m['reason']}"
                                for m in missing)
            # Below the axis label, not on top of it; `bbox_inches="tight"`
            # picks up the negative offset when the figure is saved.
            fig.text(0.5, -0.04, "not supported — " + note, ha="center",
                     fontsize=8, color=MUTED, wrap=True)

        path = out_dir / f"{i:02d}_{slug(run)}.png"

         

        emphasise_subject(fig)
        fig.savefig(path, bbox_inches="tight", facecolor=PAPER)
        plt.close(fig)
        made.append(path)
    return made


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench_results.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found — run the benchmark first (cargo run --release)")

    env, rows = load(csv_path)
    if not rows:
        sys.exit(f"{csv_path} has no data rows")

    unknown = {r["reader"] for r in rows} - set(ORDER)
    if unknown:
        # Loud, because this is exactly the drift that made the old scripts
        # silently mis-colour a series.
        sys.exit(f"unknown reader id(s) in {csv_path}: {sorted(unknown)}\n"
                 f"add them to READERS in this script (ids come from src/reader.rs)")

    graphs = out_dir / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)

    made = [summary_figure(env, rows, out_dir / "bench_summary.png")]
    s = scaling_figure(rows, graphs / "scaling.png")
    if s:
        made.append(s)
    made += run_figures(rows, graphs / "runs")

    print(f"{len(made)} figures from {len(rows)} rows:")
    for p in made[:3]:
        print(f"  {p}")
    if len(made) > 3:
        print(f"  ... and {len(made) - 3} per-run charts in {graphs / 'runs'}")


if __name__ == "__main__":
    main()

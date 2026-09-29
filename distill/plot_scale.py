"""Paper figure: training curves of the in-training LRSD runs (summary table: analyze_scale.py).

Top row: generalization EM without patching (unique questions, 1toN gold) after every epoch; bottom row
(flatter): memorization 1toN, evaluated every 5 epochs.  One column per model with finished runs; line = mean over
the seeds (848/1/2), band = mean ± 1 sd over the seeds.  Baseline (lam 0) in neutral gray, lam 0.1 and lam 1 in the
first two categorical slots of the dataviz reference palette.  Grey column in the top row: epochs 30-50 (the
averaging window of the summary table).  Dashed line through both rows of a column: the first evaluated epoch at
which the mean memorization of every arm is >= SAT = 0.90 (memorization saturated).  No figure title; the task
name and layer pairs are left to the caption.

Sized for a full-width figure: 12 x 4 in, so at \\linewidth (5.5 in) the fonts print at ~7-8.7 pt.
Times-compatible serif (Liberation Serif) with TrueType embedding in the PDF.

Usage (from the repo root): python distill/plot_scale.py --out results/lrsd/figs/scale  -> scale_chaining.{png,pdf}
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from analyze_scale import MODELS, PAIR, RUNS, SEEDS, load, run_dir  # noqa: E402

NAME = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B", "llama3.2-1b": "LLaMA-3.2-1B",
        "llama3.2-3b": "LLaMA-3.2-3B"}
ARMS = [("B", "Baseline ($\\lambda=0$)", "#8a8984"), ("L01", "LRSD $\\lambda=0.1$", "#2a78d6"),
        ("L1", "LRSD $\\lambda=1$", "#eb6834")]
INK, INK2, GRID, WINDOW = "#0b0b0b", "#52514e", "#e6e5e0", "#f1f0ec"
SAT = 0.90
MEM_EPOCHS = np.arange(0, 51, 5)


def curves(arm, model, task, runs=RUNS, seeds=SEEDS):
    """(EM, memorization) arrays over the finished runs of one arm (seeds x epochs), or None if there are none."""
    name = "B" if arm == "B" else f"P{PAIR[model]}{arm}"
    em, mem = [], []
    for s in seeds:
        d = run_dir(name, model, task, s, runs)
        if load(d) is None:
            continue
        ev = {r["epoch"]: r for r in map(json.loads, open(os.path.join(d, "log.jsonl"))) if r.get("epoch", -1) >= 0}
        em.append([ev[e]["chain_uq_em"] for e in range(51)])
        mem.append([ev[e]["mem_em"] for e in MEM_EPOCHS])
    return (np.array(em), np.array(mem)) if em else None


def band(ax, x, Y, color, z):
    m, sd = Y.mean(0), (Y.std(0, ddof=1) if len(Y) > 1 else np.zeros(Y.shape[1]))
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.2, lw=0, zorder=z)
    ax.plot(x, m, color=color, lw=2.6, zorder=z + 0.5, solid_capstyle="round")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output prefix: <out>_<task>.{png,pdf}")
    ap.add_argument("--runs_dir", default=RUNS)
    ap.add_argument("--tasks", default="chaining")
    args = ap.parse_args()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 16, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.1,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 1.1, "ytick.major.width": 1.1, "text.color": INK})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for task in args.tasks.split(","):
        data = {m: {arm: curves(arm, m, task, args.runs_dir) for arm, _, _ in ARMS} for m in MODELS}
        models = [m for m in MODELS if any(c is not None for c in data[m].values())]
        if not models:
            print(f"{task}: no finished runs in {args.runs_dir}, skipped")
            continue
        # fixed layout (top to bottom): legend, column titles, 2 x 4 axes, tick labels, x label
        # 12 x 4 in (aspect 3:1).  A rotated y label per row does not fit this height (the memorization row is
        # ~0.5 in tall), so the rows are named inside the first column and the shared y axis says "Accuracy".
        fig, axs = plt.subplots(2, len(models), figsize=(12, 4.0), sharex=True, sharey="row", squeeze=False,
                                gridspec_kw=dict(height_ratios=[3, 1], left=0.07, right=0.975, top=0.83,
                                                 bottom=0.145, wspace=0.10, hspace=0.12))
        sat_epochs = {}
        for j, model in enumerate(models):
            top, bot = axs[0, j], axs[1, j]
            top.axvspan(30, 50, color=WINDOW, lw=0, zorder=0)
            mems = []
            for arm, _, color in ARMS:
                if data[model][arm] is None:
                    continue
                em, mem = data[model][arm]
                z = 4 if arm == "B" else 2   # baseline on top: it coincides with lam 0.1 in memorization
                band(top, np.arange(51), em, color, z)
                band(bot, MEM_EPOCHS, mem, color, z)
                mems.append(mem.mean(0))
            ok = np.all(np.array(mems) >= SAT, axis=0)
            top.set_title(NAME[model], color=INK, pad=6)
            for ax in (top, bot):
                ax.grid(axis="y", color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                ax.set_xlim(0, 50)
                ax.set_xticks([0, 10, 20, 30, 40, 50])
            top.yaxis.set_major_locator(MaxNLocator(4))
            if not ok.any():
                continue
            sat = int(MEM_EPOCHS[np.argmax(ok)])
            sat_epochs[model] = sat
            # one dashed line through both rows, from the top of the upper axis to the bottom of the lower one
            fig.add_artist(ConnectionPatch(xyA=(sat, 1), coordsA=top.get_xaxis_transform(),
                                           xyB=(sat, 0), coordsB=bot.get_xaxis_transform(),
                                           color=INK, lw=1.6, ls=(0, (4, 3)), zorder=10))
        axs[0, 0].set_ylim(bottom=0)
        axs[1, 0].set_ylim(0, 1.05)
        axs[1, 0].set_yticks([0, 1])
        for ax, row in ((axs[0, 0], "Generalization"), (axs[1, 0], "Memorization")):
            # lower right of the first column: empty in every panel (curves sit higher there)
            ax.text(0.99, 0.07 if row == "Generalization" else 0.14, row, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=16, color=INK, zorder=12,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        fig.text(0.008, (0.83 + 0.145) / 2, "Accuracy", rotation=90, ha="left", va="center", fontsize=18)
        fig.text(0.5225, 0.012, "Training epoch", ha="center", va="bottom", fontsize=18)
        handles = [Line2D([], [], color=c, lw=2.6) for _, _, c in ARMS]
        labels = [lab for _, lab, _ in ARMS]
        handles += [Line2D([], [], color=INK, lw=1.6, ls=(0, (4, 3))), Patch(color=WINDOW)]
        labels += [f"Memorization $\\geq$ {SAT:.0%}", "Epochs 30–50"]
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.95), ncol=5, frameon=False, handlelength=1.6,
                   columnspacing=1.0, handletextpad=0.5)
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.out}_{task}.{ext}", dpi=200)
        plt.close(fig)
        print("saved", f"{args.out}_{task}.{{png,pdf}}", "| memorization-saturation epoch per model:", sat_epochs)


if __name__ == "__main__":
    main()

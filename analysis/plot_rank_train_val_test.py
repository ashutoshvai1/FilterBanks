"""Train / validation / test MSE vs. RRR rank (paper Figure 1a).

Generates both stacked panels (ETTh1 and Weather, H=96) with identical styling.

Data: latest output/rrr_analysis_v2/{ETTh1,weather}_720_96_*/metrics/rank_analysis.csv
from rrr_rank_analysis.py (RRR with intercept, ranks 1..H, i.e. up to full rank).

Usage:
    python analysis/plot_rank_train_val_test.py
    python analysis/plot_rank_train_val_test.py --dataset weather --csv path/to/rank_analysis.csv
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "etth1": {"title": "ETTh1", "run": "ETTh1_720_96_*",
              "out": "etth1_horizon96_train_val_test.pdf"},
    "weather": {"title": "Weather", "run": "weather_720_96_*",
                "out": "weather_horizon96_train_val_test.pdf"},
}

SERIES = [  # column, label, color, linestyle, marker
    ("rrr_train_mse", "Train", "#1f77b4", "-", "o"),
    ("rrr_val_mse", "Validation", "#ff7f0e", "--", "s"),
    ("rrr_test_mse", "Test", "#2ca02c", "-.", "^"),
]


def setup_style():
    for d in (os.path.join(HERE, "..", "nimbus-sans"),
              os.path.expanduser("~/.fonts/nimbus-sans")):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith("NimbusSanL") and f.endswith((".ttf", ".otf")):
                    fm.fontManager.addfont(os.path.join(d, f))
    names = {f.name for f in fm.fontManager.ttflist}
    family = next((n for n in ("Nimbus Sans", "Nimbus Sans L") if n in names), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family,
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,  # embed TrueType: editable, accepted by NeurIPS checker
        "ps.fonttype": 42,
    })


def plot(df, title, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ranks = df["rank"].to_numpy()
    # sparse markers (~13 per curve) so they do not clutter
    every = 8 if ranks.max() > 60 else 5
    mark_idx = [i for i, r in enumerate(ranks) if r == 1 or r % every == 0]
    for col, label, color, ls, marker in SERIES:
        ax.plot(ranks, df[col], label=label, color=color, linestyle=ls, linewidth=1.8,
                marker=marker, markersize=6, markevery=mark_idx,
                markeredgecolor="white", markeredgewidth=0.7, clip_on=False, zorder=3)

    ax.set_xlim(0, ranks.max() + 1)
    step = 20 if ranks.max() <= 60 else (24 if ranks.max() <= 96 else None)
    if step:
        ax.set_xticks(list(range(0, int(ranks.max()) + 1, step)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # same label width in both panels
    ax.set_xlabel("Rank")
    ax.set_ylabel("MSE")
    ax.grid(True, axis="y", color="0.88", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    # Legend in one row above the axes (never overlaps data); dataset name sits in the
    # empty upper-right region (curves are flat and low there in both panels).
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=2.2, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.3)
    ax.text(0.97, 0.95, title, transform=ax.transAxes, ha="right", va="top",
            fontsize=14, fontweight="bold")

    # Fixed axes box (no tight bbox) so stacked panels align exactly
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.19, top=0.86)
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), help="default: all")
    p.add_argument("--csv", help="override input rank_analysis.csv (requires --dataset)")
    p.add_argument("--runs_dir", default="output/rrr_analysis_v2")
    p.add_argument("--out_dir", default=HERE)
    args = p.parse_args()
    setup_style()
    os.makedirs(args.out_dir, exist_ok=True)
    for key in [args.dataset] if args.dataset else DATASETS:
        cfg = DATASETS[key]
        csv = args.csv
        if csv is None:
            runs = sorted(glob.glob(os.path.join(args.runs_dir, cfg["run"], "metrics", "rank_analysis.csv")))
            if not runs:
                raise FileNotFoundError(f"No run matching {cfg['run']} in {args.runs_dir}")
            csv = runs[-1]
        print(f"{key}: {csv}")
        plot(pd.read_csv(csv), cfg["title"], os.path.join(args.out_dir, cfg["out"]))


if __name__ == "__main__":
    main()

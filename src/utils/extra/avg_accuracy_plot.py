#!/usr/bin/env python3
"""
Generate one accuracy bar-plot per dataset.

For every <dataset> in --datasets and every <run> in --run_names the script
expects:

    outputs/averages/<dataset>/data/<run>.jsonl

and writes the figure to:

    outputs/averages/<dataset>/image/<output_name>.png
"""
import argparse
import json
import os
import random
import sys
from typing import List

import matplotlib.pyplot as plt


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-dataset grouped accuracy bar plots from JSONL files"
    )
    p.add_argument("--datasets",   required=True,
                   help="Comma-sep list of datasets (e.g. csqa,obqa,qasc)")
    p.add_argument("--run_names",  required=True,
                   help="Comma-sep list of run names / files without extension")
    p.add_argument("--grouping",   default="",
                   help="Comma-sep grouping sizes (must sum to n bars)")
    p.add_argument("--colors",     default="",
                   help="Comma-sep list like 'pink:1,blue:2'")
    p.add_argument("--column_names", default="",
                   help="Comma-sep custom x-axis labels (same length as run list)")
    p.add_argument("--output_name",  default="plot",
                   help="Base name of output image (no extension)")
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_grouped_positions(grouping, bar_width=0.5, intra_gap=0.1,
                              inter_group_gap=0.6):
    pos, cur = [], 0.0
    for g in grouping:
        for _ in range(g):
            pos.append(cur)
            cur += bar_width + intra_gap
        cur += inter_group_gap - intra_gap
    return pos


# full 10-shade palette per colour
COLOR_SHADES = {
    "red":        ["#FFE5E5", "#FFCCCC", "#FFB2B2", "#FF9999", "#FF8080",
                   "#FF6666", "#FF4D4D", "#FF3333", "#FF1A1A", "#FF0000"],
    "orange":     ["#FFF1E0", "#FFE3C2", "#FFD6A3", "#FFC885", "#FFBB66",
                   "#FFAD47", "#FFA02A", "#FF920D", "#FF8500", "#E67800"],
    "yellow":     ["#FFFBE5", "#FFF7CC", "#FFF3B2", "#FFEF99", "#FFEB80",
                   "#FFE766", "#FFE34D", "#FFDF33", "#FFDB1A", "#FFD700"],
    "green":      ["#E6F4EA", "#CCE9D6", "#B3DFC3", "#99D4AF", "#80C99B",
                   "#66BF87", "#4DB473", "#33AA5F", "#1A9F4B", "#009533"],
    "blue":       ["#E5F0FA", "#CCE0F5", "#B2D1F0", "#99C2EB", "#80B3E6",
                   "#66A3E0", "#4D94DB", "#3385D6", "#1A75D1", "#0066CC"],
    "purple":     ["#F3E6FA", "#E6CCF5", "#D9B3F0", "#CC99EB", "#BF80E6",
                   "#B266E0", "#A64DDB", "#9933D6", "#8C1AD1", "#8000CC"],
    "pink":       ["#FAE6EC", "#F5CCD9", "#F0B3C6", "#EB99B3", "#E680A0",
                   "#E0668D", "#DB4D7A", "#D63366", "#D11A53", "#CC0040"],
    "turquoise":  ["#E0F7F9", "#C2EEF3", "#A3E6ED", "#85DDE7", "#66D5E1",
                   "#47CCD9", "#2AC4D3", "#0DBBCE", "#00B3C8", "#009EB2"],
}


def build_bar_colors(n, colors_to_use):
    if not colors_to_use:
        flat = [s for shades in COLOR_SHADES.values() for s in shades]
        random.shuffle(flat)
        return [flat[i % len(flat)] for i in range(n)]

    out = []
    for i in range(n):
        if i < len(colors_to_use):
            cname, shade = colors_to_use[i]
            shade = min(shade, len(COLOR_SHADES[cname]) - 1)
            out.append(COLOR_SHADES[cname][shade])
        else:
            flat = [s for shades in COLOR_SHADES.values() for s in shades]
            out.append(random.choice(flat))
    return out


def plot_one_file(
    records: List[dict],
    labels: List[str],
    output_path: str,
    grouping,
    colors_to_use
):
    if not records:
        print("❌  No data; skipping.")
        return

    # keep order
    accuracies = [r["xfinder_average_accuracy"] for r in records]

    bar_colors = build_bar_colors(len(labels), colors_to_use)

    bar_width = 0.5
    if grouping:
        if sum(grouping) != len(labels):
            sys.exit("Grouping sizes don't match number of bars.")
        positions = compute_grouped_positions(grouping, bar_width=bar_width)
    else:
        positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(positions, accuracies, width=bar_width, color=bar_colors)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    ytick_spacing = 0.02
    ax.set_ylabel("Accuracy", labelpad=10, fontsize=12)
    ax.set_ylim(max(0, round( ( (min(accuracies)*0.95) / ytick_spacing) )*ytick_spacing),
                min(1.0, max(accuracies)*1.05))
    ax.tick_params(axis='y', which='both', length=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, ha="center", fontsize=10)
    ax.tick_params(axis='x', which='both', length=0)

    
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='-', linewidth=1, alpha=0.3, clip_on=False)
    ax.set_axisbelow(True)  
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, transparent=False)
    plt.close()
    print(f"✅  Saved → {output_path}")


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    runs     = [r.strip() for r in args.run_names.split(",") if r.strip()]

    grouping = list(map(int, args.grouping.split(","))) if args.grouping else []
    colors_to_use = []
    if args.colors:
        for itm in args.colors.split(","):
            cname, shade = itm.split(":")
            colors_to_use.append((cname, int(shade)))

    # column labels
    supplied_labels = [l.strip() for l in args.column_names.split(",")
                       if l.strip()]
    if supplied_labels and len(supplied_labels) != len(runs):
        sys.exit("Number of --column_names must equal number of run names")

    for ds in datasets:
        records, labels = [], []
        for idx, run in enumerate(runs):
            f = f"outputs/averages/{ds}/data/{run}.jsonl"
            if not os.path.isfile(f):
                print(f"⚠️  Missing file: {f}")
                continue
            recs = load_jsonl(f)
            if not recs:
                print(f"⚠️  Empty file: {f}")
                continue
            records.append(recs[0])       # first (and usually only) line
            label = supplied_labels[idx] if supplied_labels else run
            label = label.replace("\\n", "\n")

            labels.append(label)

        if not records:
            print(f"⏭️  No data for dataset '{ds}'")
            continue

        out_png = f"outputs/averages/{ds}/image/{args.output_name}.png"
        plot_one_file(records, labels, out_png,
                      grouping=grouping, colors_to_use=colors_to_use)

    print("\n🎉  All done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figures and LaTeX tables for the paper."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

ORDER = ["simple_sg", "simple_pl", "negation", "pronoun", "distractor_sg", "distractor_pl"]
LABELS = {"simple_sg": "Simple (sg)", "simple_pl": "Simple (pl)", "negation": "Negation",
          "pronoun": "Pronoun", "distractor_sg": "Attraction (sg)",
          "distractor_pl": "Attraction (pl)"}
BLUE, ORANGE = "#4C72B0", "#DD8452"

AFRICA_CIRCUIT = [(11, 6), (0, 4), (11, 4), (0, 8), (11, 7), (2, 6),
                  (1, 0), (2, 1), (1, 1), (6, 0), (10, 0), (9, 4)]


def load(results_dir):
    table = pd.read_csv(f"{results_dir}/accuracy_by_setting.csv").set_index("setting")
    summary = json.load(open(f"{results_dir}/summary.json"))
    return table.reindex(ORDER), summary


def heatmap(results_dir, label, out_path):
    effects = np.load(f"{results_dir}/head_effects.npy")
    _, summary = load(results_dir)
    limit = np.abs(effects).max()

    fig, ax = plt.subplots(figsize=(0.42 * effects.shape[1] + 1.6,
                                    2.2 + 0.16 * effects.shape[0]))
    im = ax.imshow(effects, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")

    for layer, head in summary["circuit"]:
        ax.add_patch(plt.Rectangle((head - 0.5, layer - 0.5), 1, 1,
                                   fill=False, edgecolor="black", linewidth=1.1))

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{label}: patching effect per head\n(boxed = selected circuit)")
    fig.colorbar(im, ax=ax, label="Logit-difference recovery", shrink=0.8)
    fig.savefig(out_path)
    plt.close(fig)


def accuracy(results_dirs, labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=True)

    for ax, results_dir, label in zip(axes, results_dirs, labels):
        table, _ = load(results_dir)
        x = np.arange(len(table))
        ax.axvspan(3.5, 5.5, color="grey", alpha=0.08)
        for offset, column, name, colour in [(-0.2, "full_model", "Full model", BLUE),
                                             (0.2, "circuit_mean", "Circuit only", ORANGE)]:
            bars = ax.bar(x + offset, table[column], 0.4, label=name, color=colour)
            ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in table.index], rotation=30, ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_title(label)

    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def ablation(results_dirs, labels, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(ORDER))
    shades = {"mean": [BLUE, "#8FA9D4"], "zero": [ORANGE, "#EFBC9B"]}

    ax.axvspan(3.5, 5.5, color="grey", alpha=0.08)
    for i, (results_dir, label) in enumerate(zip(results_dirs, labels)):
        table, _ = load(results_dir)
        for j, method in enumerate(["mean", "zero"]):
            ax.bar(x + (i * 2 + j - 1.5) * 0.2, table[f"circuit_{method}"], 0.2,
                   label=f"{label} ({method})", color=shades[method][i])

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=30, ha="right")
    ax.set_ylabel("Circuit accuracy")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.set_title("Circuit accuracy under mean vs zero ablation")
    fig.savefig(out_path)
    plt.close(fig)


def logit_diffs(results_dirs, labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2), sharey=True)

    for ax, results_dir, label in zip(axes, results_dirs, labels):
        items = pd.read_csv(f"{results_dir}/circuit_accuracy_mean.csv")
        groups = [items[items["setting"] == s]["logit_diff"].values for s in ORDER]
        parts = ax.violinplot(groups, showmeans=True, widths=0.8)
        for body, setting in zip(parts["bodies"], ORDER):
            body.set_facecolor(ORANGE if setting.startswith("distractor") else BLUE)
            body.set_alpha(0.65)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(1, len(ORDER) + 1))
        ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=30, ha="right")
        ax.set_title(label)

    axes[0].set_ylabel("Logit difference (correct - incorrect)")
    fig.savefig(out_path)
    plt.close(fig)


def depth(comparison_dir, out_path):
    profile = pd.read_csv(f"{comparison_dir}/depth_profile.csv")
    models = [c for c in profile.columns if c != "depth"]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    x = np.arange(len(profile))
    for i, model in enumerate(models):
        ax.bar(x + (i - 0.5) * 0.35, profile[model], 0.35, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(profile["depth"])
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Share of total effect")
    ax.legend(frameon=False)
    ax.set_title("Where the agreement signal concentrates")
    fig.savefig(out_path)
    plt.close(fig)


def overlap(results_dir, out_path):
    _, summary = load(results_dir)
    ours = {tuple(h) for h in summary["circuit"]}
    theirs = set(AFRICA_CIRCUIT)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.set_xlim(-0.5, summary["n_heads"] - 0.5)
    ax.set_ylim(summary["n_layers"] - 0.5, -0.5)
    ax.set_xticks(range(summary["n_heads"]))
    ax.set_yticks(range(summary["n_layers"]))
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.grid(color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)

    for heads, colour, label in [(theirs - ours, BLUE, "Africa (2025)"),
                                 (ours - theirs, ORANGE, "This work"),
                                 (ours & theirs, "#2B2B2B", "Both")]:
        ax.scatter([h for _, h in heads], [l for l, _ in heads], s=110, c=colour,
                   label=label, zorder=3, edgecolors="white", linewidths=0.8)

    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.set_title("Two circuits for the same task in GPT-2 Small")
    fig.savefig(out_path)
    plt.close(fig)


def tables(results_dirs, labels, out_path):
    rows = []
    for results_dir, label in zip(results_dirs, labels):
        table, summary = load(results_dir)
        rows.append(r"\begin{table}[t]\centering")
        rows.append(r"\begin{tabular}{lrrrrr}\toprule")
        rows.append(r"Setting & Full & Circuit$_{\text{mean}}$ & $\Delta$ "
                    r"& Circuit$_{\text{zero}}$ & $\Delta$ \\ \midrule")
        for setting, row in table.iterrows():
            rows.append(f"{LABELS[setting]} & {row['full_model']:.3f} & "
                        f"{row['circuit_mean']:.3f} & {row['drop_mean']:+.3f} & "
                        f"{row['circuit_zero']:.3f} & {row['drop_zero']:+.3f} \\\\")
        rows.append(r"\bottomrule\end{tabular}")
        rows.append(rf"\caption{{Accuracy by condition, {label}.}}")
        rows.append(r"\end{table}")
        rows.append("% circuit: " + ", ".join(f"L{l}H{h}" for l, h in summary["circuit"]))
        rows.append("")

    open(out_path, "w").write("\n".join(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dirs = [f"{args.results}/gpt2", f"{args.results}/gpt2-medium"]
    labels = ["GPT-2 Small", "GPT-2 Medium"]
    out = lambda name: os.path.join(args.out_dir, name)

    heatmap(dirs[0], labels[0], out("heatmap_gpt2.pdf"))
    heatmap(dirs[1], labels[1], out("heatmap_gpt2-medium.pdf"))
    accuracy(dirs, labels, out("accuracy_by_setting.pdf"))
    ablation(dirs, labels, out("ablation_comparison.pdf"))
    logit_diffs(dirs, labels, out("logit_diff_distribution.pdf"))
    overlap(dirs[0], out("circuit_overlap.pdf"))
    depth(f"{args.results}/comparison", out("depth_profile.pdf"))
    tables(dirs, labels, out("tables.tex"))

    print(f"wrote figures and tables to {args.out_dir}/")


if __name__ == "__main__":
    main()

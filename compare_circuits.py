import json
import os
import sys

import numpy as np
import pandas as pd

LABELS = ["GPT-2 Small", "GPT-2 Medium"]


def depth_profile(effects, bins=3):
    per_layer = np.abs(effects).sum(axis=1)
    edges = np.linspace(0, len(per_layer), bins + 1).astype(int)
    profile = np.array([per_layer[edges[i]:edges[i + 1]].sum() for i in range(bins)])
    return profile / profile.sum()


def concentration(effects, threshold=0.8):
    cumulative = np.cumsum(np.sort(np.abs(effects).flatten())[::-1])
    return int(np.searchsorted(cumulative, threshold * cumulative[-1]) + 1)


def main():
    dirs = sys.argv[1:3] or ["results/gpt2", "results/gpt2-medium"]
    out_dir = "results/comparison"
    os.makedirs(out_dir, exist_ok=True)

    effects = [np.load(f"{d}/head_effects.npy") for d in dirs]

    profile = pd.DataFrame({"depth": ["early", "mid", "late"],
                            LABELS[0]: depth_profile(effects[0]),
                            LABELS[1]: depth_profile(effects[1])})
    profile.to_csv(f"{out_dir}/depth_profile.csv", index=False)
    print(profile.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    counts = [concentration(e) for e in effects]
    print("\nheads for 80% of effect: " + ", ".join(
        f"{label} {n}/{e.size} ({n / e.size:.1%})"
        for label, n, e in zip(LABELS, counts, effects)))

    tables = [pd.read_csv(f"{d}/accuracy_by_setting.csv") for d in dirs]
    merged = tables[0].merge(tables[1], on="setting",
                             suffixes=(f" ({LABELS[0]})", f" ({LABELS[1]})"))
    merged.to_csv(f"{out_dir}/accuracy_comparison.csv", index=False)
    print("\n" + merged.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    json.dump({label: {"heads": int(e.size), "concentration_80": n,
                       "depth_profile": depth_profile(e).tolist()}
               for label, n, e in zip(LABELS, counts, effects)},
              open(f"{out_dir}/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()

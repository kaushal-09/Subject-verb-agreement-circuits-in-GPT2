#!/usr/bin/env python3
"""Find the subject-verb agreement circuit in a model and test what it can do.

    python run_experiment.py --model gpt2
    python run_experiment.py --model gpt2-medium --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import build_dataset, build_patching_pairs
from circuit_analysis import (evaluate, patch_heads_averaged, top_heads,
                              mean_head_activations, evaluate_circuit)

CIRCUIT_FRACTION = 12 / 144


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "results",
                                           args.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    model = HookedTransformer.from_pretrained(args.model, device=device)
    if args.dtype != "float32":
        model = model.to({"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype])
    model.eval()

    dataset = build_dataset()
    print(f"{args.model} on {device}, {len(dataset)} items")

    full = pd.DataFrame(evaluate(model, dataset))
    full.to_csv(os.path.join(out_dir, "full_model_accuracy.csv"), index=False)

    pairs = build_patching_pairs(dataset)
    effects, n_pairs = patch_heads_averaged(model, pairs)
    np.save(os.path.join(out_dir, "head_effects.npy"), effects)
    print(f"patched {n_pairs} pairs")

    n_heads_total = model.cfg.n_layers * model.cfg.n_heads
    circuit = top_heads(effects, max(1, round(CIRCUIT_FRACTION * n_heads_total)))
    print(f"circuit: {len(circuit)} of {n_heads_total} heads")

    mean_acts = mean_head_activations(model, dataset)

    table = full.groupby("setting")["correct_pred"].mean().to_frame("full_model")
    for ablation in ["mean", "zero"]:
        circ = pd.DataFrame(evaluate_circuit(model, dataset, circuit, ablation, mean_acts))
        circ.to_csv(os.path.join(out_dir, f"circuit_accuracy_{ablation}.csv"), index=False)
        table[f"circuit_{ablation}"] = circ.groupby("setting")["correct_pred"].mean()
        table[f"drop_{ablation}"] = table[f"circuit_{ablation}"] - table["full_model"]

    table.to_csv(os.path.join(out_dir, "accuracy_by_setting.csv"))
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "n_layers": model.cfg.n_layers,
                   "n_heads": model.cfg.n_heads, "n_pairs": n_pairs,
                   "circuit": circuit}, f, indent=2)


if __name__ == "__main__":
    main()

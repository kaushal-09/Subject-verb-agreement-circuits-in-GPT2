from typing import Dict, List, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer


def logit_diff(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    return (logits[0, -1, correct_id] - logits[0, -1, incorrect_id]).item()


def evaluate(model: HookedTransformer, dataset: List[Dict[str, str]]) -> List[Dict]:
    results = []
    for item in dataset:
        try:
            correct_id = model.to_single_token(item["correct"])
            incorrect_id = model.to_single_token(item["incorrect"])
        except AssertionError:
            continue
        logits = model(model.to_tokens(item["prompt"]))
        d = logit_diff(logits, correct_id, incorrect_id)
        results.append({**item, "logit_diff": d, "correct_pred": d > 0})
    return results


def patch_heads(model: HookedTransformer, pair: Dict[str, str]) -> np.ndarray:
    clean_tokens = model.to_tokens(pair["clean"])
    corrupted_tokens = model.to_tokens(pair["corrupted"])
    correct_id = model.to_single_token(pair["correct"])
    incorrect_id = model.to_single_token(pair["incorrect"])

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

    clean_diff = logit_diff(clean_logits, correct_id, incorrect_id)
    corrupted_diff = logit_diff(corrupted_logits, correct_id, incorrect_id)
    denom = clean_diff - corrupted_diff

    effects = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        for head in range(model.cfg.n_heads):
            def hook(value, hook, head=head, name=hook_name):
                value[:, -1, head, :] = clean_cache[name][:, -1, head, :]
                return value

            patched = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, hook)])
            patched_diff = logit_diff(patched, correct_id, incorrect_id)
            effects[layer, head] = (patched_diff - corrupted_diff) / denom if abs(denom) > 1e-6 else 0.0
    return effects


def patch_heads_averaged(model: HookedTransformer,
                         pairs: List[Dict[str, str]]) -> Tuple[np.ndarray, int]:
    total = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    n = 0
    for pair in pairs:
        try:
            total += patch_heads(model, pair)
            n += 1
        except AssertionError:
            continue
    if n == 0:
        raise RuntimeError("no usable pairs")
    return total / n, n


def top_heads(effects: np.ndarray, k: int) -> List[Tuple[int, int]]:
    idx = np.argsort(-effects, axis=None)[:k]
    layers, heads = np.unravel_index(idx, effects.shape)
    return [(int(l), int(h)) for l, h in zip(layers, heads)]


def mean_head_activations(model: HookedTransformer, dataset: List[Dict[str, str]],
                          n_samples: int = 60, seed: int = 0) -> torch.Tensor:
    import random
    sums = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head)
    counts = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    for item in random.Random(seed).sample(dataset, min(n_samples, len(dataset))):
        _, cache = model.run_with_cache(model.to_tokens(item["prompt"]))
        for layer in range(model.cfg.n_layers):
            z = cache[f"blocks.{layer}.attn.hook_z"][0].detach().cpu()
            sums[layer] += z.sum(dim=0)
            counts[layer] += z.shape[0]
    return sums / counts.unsqueeze(-1).clamp(min=1)


def evaluate_circuit(model: HookedTransformer, dataset: List[Dict[str, str]],
                     circuit: List[Tuple[int, int]], ablation: str = "mean",
                     mean_acts: torch.Tensor = None) -> List[Dict]:
    if ablation == "mean" and mean_acts is None:
        raise ValueError("mean ablation needs mean_acts")

    circuit_set = set(circuit)
    hooks = []
    for layer in range(model.cfg.n_layers):
        ablated = [h for h in range(model.cfg.n_heads) if (layer, h) not in circuit_set]
        if not ablated:
            continue

        def hook(value, hook, layer=layer, ablated=ablated):
            for h in ablated:
                if ablation == "zero":
                    value[:, :, h, :] = 0.0
                else:
                    value[:, :, h, :] = mean_acts[layer, h].to(value.device, value.dtype)
            return value

        hooks.append((f"blocks.{layer}.attn.hook_z", hook))

    results = []
    for item in dataset:
        try:
            correct_id = model.to_single_token(item["correct"])
            incorrect_id = model.to_single_token(item["incorrect"])
        except AssertionError:
            continue
        logits = model.run_with_hooks(model.to_tokens(item["prompt"]), fwd_hooks=hooks)
        d = logit_diff(logits, correct_id, incorrect_id)
        results.append({**item, "logit_diff": d, "correct_pred": d > 0})
    return results

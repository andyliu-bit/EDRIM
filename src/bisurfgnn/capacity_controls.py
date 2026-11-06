"""Response-free width selection for a separate capacity sensitivity track."""
from __future__ import annotations

from copy import deepcopy
import torch

from .models.factorial_bisurf import FactorialBiSurfGNN, VARIANTS


def make_model(atom_dim, cell_dim, model_config, variant, width, seed=41):
    exchange, fusion = variant.split("__")
    config = deepcopy(model_config)
    config["hidden_dim"] = width
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FactorialBiSurfGNN(atom_dim, cell_dim, exchange=exchange, fusion=fusion, **config)


def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))


def select_widths(atom_dim, cell_dim, config, widths, tolerance):
    if not widths or len(set(widths)) != len(widths):
        raise ValueError("Width grid must be nonempty and unique")
    if any(not isinstance(w, int) or w < 1 or w % config["attention_heads"] for w in widths):
        raise ValueError("Widths must be positive multiples of the head count")
    if not 0 < tolerance < 1:
        raise ValueError("Invalid capacity tolerance")
    target = counts(make_model(atom_dim, cell_dim, config, "bixt__gate", 136))["trainable"]
    candidates, selected = [], []
    for variant in VARIANTS:
        rows = []
        for width in sorted(widths):
            count = counts(make_model(atom_dim, cell_dim, config, variant, width))
            row = dict(variant=variant, width=width, **count,
                       relative_gap=abs(count["trainable"]-target)/target)
            rows.append(row)
        best = min(rows, key=lambda row: (row["relative_gap"], row["width"]))
        if best["relative_gap"] > tolerance:
            raise ValueError(f"No in-band width for {variant}")
        selected.append(best)
        candidates.extend(rows)
    return dict(target_trainable=target, target_variant="bixt__gate", target_width=136,
                tolerance=tolerance, selection_rule="minimum_relative_gap_then_smaller_width",
                candidates=candidates, selected=selected)

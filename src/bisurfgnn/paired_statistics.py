"""Paired, fold-stratified block bootstrap of seed-averaged surface RMSE.

This is conditional on fitted checkpoints and the observed seed average. It
does not correct selection optimism, training-set reuse, or cross-block
dependence. In chemical OOD, clusters rather than individual surfaces are the
primary sampling blocks; group resampling is a separate sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PairedStatisticsError(ValueError):
    pass


KEYS = ["fold", "seed", "surface_id", "model"]
METADATA = ["group_id", "cluster_id", "assay_block_id", "n_points"]


def paired_block_bootstrap(
    ledger: pd.DataFrame, *, unit: str = "cluster_id", reference: str = "hsa",
    iterations: int = 2000, seed: int = 41,
) -> dict:
    """Average per-surface RMSE across seeds, then bootstrap whole blocks.

    The estimator is equal-fold mean of surface-macro RMSE, not equal-cluster
    mean or RMSE of a seed ensemble. Unequal block sizes are retained by summing
    selected block errors and dividing by their total selected surface count.
    """
    if unit not in {"cluster_id", "group_id"}:
        raise PairedStatisticsError("Resampling unit must be cluster_id or group_id")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 10:
        raise PairedStatisticsError("iterations must be an integer >= 10")
    required = [*KEYS, *METADATA, "rmse"]
    if ledger.empty or set(required) - set(ledger.columns):
        raise PairedStatisticsError("Empty or incomplete paired error ledger")
    if ledger[required].isna().any().any() or ledger.duplicated(KEYS).any():
        raise PairedStatisticsError("Missing values or duplicate paired keys")
    # Canonical summation order makes replay independent of input file row order.
    ledger = ledger.sort_values(KEYS, kind="stable")
    if not np.isfinite(ledger.rmse.to_numpy(dtype=float)).all() or (ledger.rmse < 0).any():
        raise PairedStatisticsError("RMSE must be finite and non-negative")
    if (ledger.n_points < 1).any() or (ledger.n_points % 1 != 0).any():
        raise PairedStatisticsError("n_points must be positive integers")
    surfaces = ["fold", "surface_id"]
    if (ledger.groupby(surfaces)[METADATA].nunique() != 1).any().any():
        raise PairedStatisticsError("Surface metadata differs across models or seeds")
    for label in ("surface_id", "group_id", "cluster_id"):
        if (ledger.groupby(label).fold.nunique() > 1).any():
            raise PairedStatisticsError("Cross-fold repeated units require a different joint resampling design")
    if (ledger.groupby("group_id").cluster_id.nunique() > 1).any():
        raise PairedStatisticsError("One prediction group maps to multiple held-out clusters")
    models = sorted(ledger.model.unique().tolist())
    if reference not in models or len(models) < 2:
        raise PairedStatisticsError("Reference and at least one model are required")
    inventory = None
    for _, part in ledger.groupby(["model", "seed"]):
        actual = set(map(tuple, part[surfaces].to_numpy()))
        if inventory is None:
            inventory = actual
        elif actual != inventory:
            raise PairedStatisticsError("Model/seed surface inventories are not identical")
    seed_sets = ledger.groupby("model").seed.apply(lambda x: frozenset(x))
    if len(set(seed_sets)) != 1:
        raise PairedStatisticsError("Models do not share the same seed inventory")

    averaged = ledger.groupby([*surfaces, "model"], sort=True).rmse.mean().unstack("model")
    metadata = ledger.drop_duplicates(surfaces).set_index(surfaces)[METADATA]
    paired = metadata.join(averaged, how="inner", validate="one_to_one")
    folds = sorted(ledger.fold.unique().tolist())
    blocks, counts, observed_folds = {}, {}, []
    for fold in folds:
        part = paired.xs(fold, level="fold")
        grouped = part.groupby(unit, sort=True)
        blocks[fold] = grouped[models].sum().to_numpy(float)
        counts[fold] = grouped.size().to_numpy(float)
        observed_folds.append(part[models].mean().to_numpy(float))
    observed = np.mean(observed_folds, axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, len(models)), dtype=float)
    for draw in range(iterations):
        fold_values = []
        for fold in folds:
            indices = rng.integers(0, len(counts[fold]), size=len(counts[fold]))
            fold_values.append(blocks[fold][indices].sum(axis=0) / counts[fold][indices].sum())
        draws[draw] = np.mean(fold_values, axis=0)
    anchor_index = models.index(reference)
    comparisons = {}
    for index, model in enumerate(models):
        if model == reference:
            continue
        differences = draws[:, index] - draws[:, anchor_index]
        interval = np.quantile(differences, [0.025, 0.975]).tolist()
        relative = (observed[anchor_index] - observed[index]) / observed[anchor_index] if observed[anchor_index] > 0 else None
        if (draws[:, anchor_index] > 0).all():
            relatives = 1 - draws[:, index] / draws[:, anchor_index]
            relative_ci = np.quantile(relatives, [0.025, 0.975]).tolist()
        else:
            relative_ci = None
        comparisons[model] = {
            "model_minus_reference_rmse": float(observed[index] - observed[anchor_index]),
            "difference_ci95": interval,
            "relative_improvement": None if relative is None else float(relative),
            "relative_ci95": relative_ci,
        }
    return {
        "version": "paired-block-bootstrap-v1", "unit": unit, "iterations": iterations,
        "seed": seed, "reference": reference,
        "estimator": "mean_folds_mean_surfaces_mean_seed_surface_RMSE",
        "observed": dict(zip(models, observed.tolist())), "comparisons": comparisons,
        "folds": [{"fold": int(f), "blocks": len(counts[f]), "surfaces": int(counts[f].sum())} for f in folds],
        "seed_count": len(next(iter(seed_sets))),
        "claim_scope": "conditional_development_uncertainty_not_confirmatory_generalization",
        "limitations": [
            "Does not correct validation checkpoint/model/hyperparameter selection optimism",
            "Conditional on the observed seed average, not a bootstrap of training randomness",
            "Overlapping training sets and residual dependence between blocks are not modelled",
            "Intervals are marginal, not multiplicity-corrected",
            "Assay block IDs are retained for audit, not an independent crossed-effects bootstrap",
        ],
    }

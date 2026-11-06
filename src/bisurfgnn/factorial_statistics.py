"""Shared cluster draws and three prespecified Bonferroni percentile intervals."""
import numpy as np

from .paired_statistics import paired_block_bootstrap, METADATA

CORE = ("pooled__concat", "pooled__gate", "bixt__concat", "bixt__gate")
CONTRASTS = np.array([[-.5, -.5, .5, .5], [-.5, .5, -.5, .5], [1, -1, -1, 1]])
NAMES = ("bixt_main_effect", "gate_main_effect", "interaction")


def factorial_bootstrap(ledger, *, iterations=10000, seed=41, unit="cluster_id"):
    if type(iterations) is not int or iterations < 100:
        raise ValueError("At least 100 bootstrap draws required")
    part = ledger[ledger.model.isin(CORE)].sort_values(["fold","surface_id","seed","model"],kind="stable").copy()
    if set(part.model) != set(CORE):
        raise ValueError("All four factorial arms required")
    # Reuse the existing strict inventory/dependence checks, not its intervals.
    paired_block_bootstrap(part, reference=CORE[0], iterations=10, seed=seed, unit=unit)
    averaged = part.groupby(["fold", "surface_id", "model"], sort=True).rmse.mean().unstack("model")[list(CORE)]
    meta = part.drop_duplicates(["fold", "surface_id"]).set_index(["fold", "surface_id"])[METADATA]
    joined = meta.join(averaged, validate="one_to_one")
    rng = np.random.default_rng(seed)
    draws = np.zeros((iterations, 4))
    estimates, blocks = [], []
    folds = sorted(part.fold.unique())
    for fold in folds:
        frame = joined.xs(fold, level="fold")
        grouped = frame.groupby(unit, sort=True)
        sums, sizes = grouped[list(CORE)].sum().to_numpy(), grouped.size().to_numpy()
        if len(sizes) < 2:
            raise ValueError("Each fold needs at least two resampling blocks")
        indices = rng.integers(0, len(sizes), (iterations, len(sizes)))
        draws += sums[indices].sum(1)/sizes[indices].sum(1)[:, None]/len(folds)
        estimates.append(frame[list(CORE)].mean().to_numpy())
        blocks.append(dict(fold=int(fold), blocks=len(sizes)))
    observed = CONTRASTS@np.mean(estimates, axis=0)
    contrasts = draws@CONTRASTS.T
    family_alpha = .05
    tail = family_alpha/(2*len(NAMES))
    return dict(version="factorial_shared_draws_v1", unit=unit, iterations=iterations, seed=seed,
        contrasts={name: dict(estimate=float(observed[i]),
            marginal_ci95=np.quantile(contrasts[:, i], [.025, .975]).tolist(),
            bonferroni_family_ci95=np.quantile(contrasts[:, i], [tail, 1-tail]).tolist())
            for i, name in enumerate(NAMES)}, folds=blocks,
        interval_method="percentile_bootstrap_with_Bonferroni_tail_adjustment_three_core_contrasts",
        claim_scope="approximate_conditional_development_intervals_not_selection_corrected",
        limitations=["Not finite-sample coverage guaranteed", "Does not remove selection optimism or shared-training dependence",
                     "Conditional on observed seed mean", "Secondary comparisons outside this three-contrast family"])

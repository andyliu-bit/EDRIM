"""Assumption-dependent replicate-noise sensitivity on matched surface subsets.

Individual-observation variability is not the sampling uncertainty of a paired
model comparison. These estimates are not proven attainable-error bounds, even
after matching the metric and cohort. Model comparisons need their own analysis.

Selection isolation: this module reads only the ``train`` and ``validation``
shards of a fold. Opening that fold's test shard is refused; rows can still be
validation/train here and test in a different fold of the development protocol.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .learned_benchmark import (
    _runtime_environment,
    _write_json_atomic,
    surface_macro_rmse,
)

REPLICATE_NOISE_VERSION = "replicate-noise-sensitivity-v3-matched-pairs"
STRATA = ("all", "unclipped", "clipped_high_both")
QUANTITIES = ("response", "residual")

READABLE_ROLES = ("train", "validation")
FORBIDDEN_ROLES = ("test",)
PAIR_KEY_COLUMNS = ("drug_a_id", "drug_b_id", "dose_a_model", "dose_b_model")
DOSE_ROUND_DECIMALS = 9


class ReplicateNoiseError(RuntimeError):
    """Raised when the replicate-noise estimate cannot be produced safely."""


@dataclass(frozen=True)
class ReplicatePair:
    """One technical replicate pair of a single prediction group.

    Replicates of the same group always sit on different assay blocks, so the
    raw response difference can carry between-plate variation. ``*_reference``
    stores each replicate's own same-block HSA value. Subtracting it defines the
    residual contrast; it does not prove that shared block effects are removed.
    """

    prediction_group_id: str
    role: str
    first_surface_id: str
    second_surface_id: str
    first_response: np.ndarray
    second_response: np.ndarray
    first_reference: np.ndarray
    second_reference: np.ndarray
    clipped_high_any: np.ndarray
    clipped_high_both: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.first_response.size)

    def quantity(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the aligned pair of series for ``response`` or ``residual``."""

        if name == "response":
            return self.first_response, self.second_response
        if name == "residual":
            return (
                self.first_response - self.first_reference,
                self.second_response - self.second_reference,
            )
        raise ReplicateNoiseError(f"Unknown quantity {name!r}")


def _shard_path(shard_dir: Path, fold: int, role: str) -> Path:
    if role in FORBIDDEN_ROLES:
        raise ReplicateNoiseError(
            f"Role {role!r} is sealed for noise estimation; only {READABLE_ROLES} may be read"
        )
    if role not in READABLE_ROLES:
        raise ReplicateNoiseError(f"Unknown role {role!r}")
    path = shard_dir / f"fold_{fold}" / f"{role}.parquet"
    if not path.is_file():
        raise ReplicateNoiseError(f"Missing shard: {path}")
    return path


def _hsa_reference(frame: pd.DataFrame) -> np.ndarray:
    """Within-block observed HSA reference, matching ``_observed_reference``."""

    mono_a = frame["mono_response_a"].to_numpy(dtype=np.float64)
    mono_b = frame["mono_response_b"].to_numpy(dtype=np.float64)
    return np.minimum(mono_a, mono_b)


def _monotherapy_complete(frame: pd.DataFrame) -> bool:
    return bool(
        frame["mono_mask_a"].to_numpy(dtype=bool).all()
        and frame["mono_mask_b"].to_numpy(dtype=bool).all()
    )


def _grid_key(frame: pd.DataFrame) -> pd.Series:
    dose_a = frame["dose_a_model"].round(DOSE_ROUND_DECIMALS)
    dose_b = frame["dose_b_model"].round(DOSE_ROUND_DECIMALS)
    return (
        frame["drug_a_id"].astype(str)
        + "|"
        + frame["drug_b_id"].astype(str)
        + "|"
        + dose_a.astype(str)
        + "|"
        + dose_b.astype(str)
    )


def collect_replicate_pairs(
    shard_dir: Path | str,
    fold: int,
    roles: Sequence[str] = READABLE_ROLES,
) -> list[ReplicatePair]:
    """Return every exactly-aligned replicate pair inside the readable shards."""

    shard_dir = Path(shard_dir)
    columns = [
        "prediction_group_id",
        "surface_replicate_id",
        "response",
        "response_clipped_high",
        "mono_response_a",
        "mono_response_b",
        "mono_mask_a",
        "mono_mask_b",
        *PAIR_KEY_COLUMNS,
    ]
    pairs: list[ReplicatePair] = []
    for role in roles:
        frame = pd.read_parquet(_shard_path(shard_dir, fold, role), columns=columns)
        frame = frame.assign(grid_key=_grid_key(frame))
        surfaces_by_group: dict[str, list[str]] = defaultdict(list)
        for group_id, surface_id in (
            frame[["prediction_group_id", "surface_replicate_id"]]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            surfaces_by_group[group_id].append(surface_id)
        multi = {
            group_id: sorted(surface_ids)
            for group_id, surface_ids in surfaces_by_group.items()
            if len(surface_ids) >= 2
        }
        if not multi:
            continue
        subset = frame[frame["prediction_group_id"].isin(multi)]
        by_surface = {
            surface_id: part.set_index("grid_key").sort_index()
            for surface_id, part in subset.groupby("surface_replicate_id", sort=False)
        }
        for group_id, surface_ids in sorted(multi.items()):
            for index in range(len(surface_ids) - 1):
                first = by_surface[surface_ids[index]]
                second = by_surface[surface_ids[index + 1]]
                shared = first.index.intersection(second.index)
                if shared.empty:
                    continue
                first_aligned = first.loc[shared]
                second_aligned = second.loc[shared]
                if not (
                    _monotherapy_complete(first_aligned)
                    and _monotherapy_complete(second_aligned)
                ):
                    continue
                pairs.append(
                    ReplicatePair(
                        prediction_group_id=group_id,
                        role=role,
                        first_surface_id=surface_ids[index],
                        second_surface_id=surface_ids[index + 1],
                        first_response=first_aligned["response"].to_numpy(dtype=np.float64),
                        second_response=second_aligned["response"].to_numpy(dtype=np.float64),
                        first_reference=_hsa_reference(first_aligned),
                        second_reference=_hsa_reference(second_aligned),
                        clipped_high_any=(
                            first_aligned["response_clipped_high"].to_numpy(dtype=bool)
                            | second_aligned["response_clipped_high"].to_numpy(dtype=bool)
                        ),
                        clipped_high_both=(
                            first_aligned["response_clipped_high"].to_numpy(dtype=bool)
                            & second_aligned["response_clipped_high"].to_numpy(dtype=bool)
                        ),
                    )
                )
    return pairs


def _stratum_mask(pair: ReplicatePair, stratum: str) -> np.ndarray:
    if stratum == "all":
        return np.ones(pair.point_count, dtype=bool)
    if stratum == "unclipped":
        return ~pair.clipped_high_any
    if stratum == "clipped_high_both":
        return pair.clipped_high_both
    raise ReplicateNoiseError(f"Unknown stratum {stratum!r}")


def paired_surface_macro_rmse(
    pairs: Iterable[ReplicatePair],
    stratum: str = "all",
    quantity: str = "response",
) -> tuple[float, float, np.ndarray, int]:
    """Return macro RMSE, micro RMSE, per-pair RMSE values, and the point count.

    The macro value reuses :func:`surface_macro_rmse` so that it matches the
    learned benchmark's selection metric exactly. ``quantity`` selects whether
    the comparison runs on raw responses or on within-block HSA residuals.
    """

    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    surface_labels: list[np.ndarray] = []
    for index, pair in enumerate(pairs):
        mask = _stratum_mask(pair, stratum)
        if not mask.any():
            continue
        first, second = pair.quantity(quantity)
        targets.append(first[mask])
        predictions.append(second[mask])
        surface_labels.append(np.full(int(mask.sum()), index, dtype=np.int64))
    if not targets:
        raise ReplicateNoiseError(f"Stratum {stratum!r} retained no replicate points")
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    surfaces = np.concatenate(surface_labels)
    macro = surface_macro_rmse(target, prediction, surfaces)
    micro = float(np.sqrt(np.mean(np.square(prediction - target))))
    per_pair = np.array(
        [
            float(np.sqrt(np.mean(np.square(p - t))))
            for t, p in zip(targets, predictions, strict=True)
        ],
        dtype=np.float64,
    )
    return macro, micro, per_pair, int(target.size)


def anchor_self_error(
    pairs: Iterable[ReplicatePair],
    stratum: str = "all",
    min_points: int = 1,
) -> tuple[float, int]:
    """Surface-macro RMSE of each replicate's own within-block HSA reference.

    This is the comparison baseline for the residual noise floor: it is what the
    HSA anchor achieves on exactly the surfaces used for the noise estimate, so
    the two numbers are measured on the same cohort and the same metric.
    """

    if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 1:
        raise ReplicateNoiseError("anchor min_points must be a positive integer")
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    surface_labels: list[np.ndarray] = []
    index = 0
    for pair in pairs:
        mask = _stratum_mask(pair, stratum)
        if int(mask.sum()) < min_points:
            continue
        for response, reference in (
            (pair.first_response, pair.first_reference),
            (pair.second_response, pair.second_reference),
        ):
            targets.append(response[mask])
            predictions.append(reference[mask])
            surface_labels.append(np.full(int(mask.sum()), index, dtype=np.int64))
            index += 1
    if not targets:
        raise ReplicateNoiseError(f"Stratum {stratum!r} retained no anchor points")
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    surfaces = np.concatenate(surface_labels)
    return surface_macro_rmse(target, prediction, surfaces), int(target.size)


def decompose_pair_variance(
    pairs: Iterable[ReplicatePair],
    stratum: str = "all",
    quantity: str = "residual",
    min_points: int = 2,
) -> dict[str, object]:
    """Split replicate differences into between-block offset and within-block noise.

    The floor reported by :func:`paired_surface_macro_rmse` assumes the two
    replicate series differ only by independent point noise. If a replicate pair
    also differs by a constant block-level offset, that offset inflates the
    estimate and the floor is too pessimistic. For a pair with ``n`` shared
    points and differences ``d_j``:

    ``mean(d_j^2) = mean_d^2 + mean((d_j - mean_d)^2)``

    The second term estimates ``(n - 1) / n`` of the within-block difference
    variance, so the unbiased within-block component is
    ``n / (n - 1) * mean((d_j - mean_d)^2)``. Subtracting ``within / n`` from
    ``mean_d^2`` leaves the between-block offset variance. Each difference
    variance is halved to reach a single-observation variance.

    These are sensitivity estimates under an additive, independent-noise model,
    not a guaranteed bracket. Shared monotherapy errors can correlate dose
    points, and residual predictability conditional on model inputs is unknown.
    A degrees-of-freedom correction can even put within above total.
    """

    if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 2:
        raise ReplicateNoiseError("decomposition min_points must be an integer >= 2")
    retained_pairs: list[ReplicatePair] = []
    total_terms: list[float] = []
    within_terms: list[float] = []
    between_terms: list[float] = []
    total_macro_terms: list[float] = []
    within_macro_terms: list[float] = []
    used = 0
    skipped_small = 0
    for pair in pairs:
        mask = _stratum_mask(pair, stratum)
        count = int(mask.sum())
        if count < min_points:
            if count > 0:
                skipped_small += 1
            continue
        first, second = pair.quantity(quantity)
        retained_pairs.append(pair)
        difference = first[mask] - second[mask]
        offset = float(np.mean(difference))
        centred = difference - offset
        within_biased = float(np.mean(np.square(centred)))
        within_unbiased = within_biased * count / (count - 1)
        pair_total = float(np.mean(np.square(difference)))
        total_terms.append(pair_total)
        within_terms.append(within_unbiased)
        between_terms.append(offset**2 - within_unbiased / count)
        total_macro_terms.append(math.sqrt(pair_total / 2.0))
        within_macro_terms.append(math.sqrt(max(within_unbiased, 0.0) / 2.0))
        used += 1
    if used == 0:
        raise ReplicateNoiseError(
            f"Stratum {stratum!r} has no pair with at least {min_points} shared points"
        )
    mean_total = float(np.mean(total_terms))
    mean_within = float(np.mean(within_terms))
    mean_between = float(np.mean(between_terms))
    between_share = mean_between / mean_total if mean_total > 0 else float("nan")
    matched_anchor, matched_anchor_points = anchor_self_error(
        retained_pairs, stratum, min_points=min_points
    )
    return {
        "minimum_shared_points": min_points,
        "matched_anchor_surface_macro_rmse": matched_anchor,
        "matched_anchor_points": matched_anchor_points,
        "matched_anchor_surfaces": 2 * used,
        "matched_pair_ids": [
            [pair.prediction_group_id, pair.first_surface_id, pair.second_surface_id]
            for pair in retained_pairs
        ],
        "interpretation": "assumption_dependent_sensitivity_not_proven_error_bounds",
        "pairs_used": used,
        "pairs_skipped_too_small": skipped_small,
        "mean_squared_difference_total": mean_total,
        "mean_squared_difference_within_block": mean_within,
        "mean_squared_difference_between_block": mean_between,
        "between_block_variance_share": between_share,
        "floor_total_rms": float(math.sqrt(mean_total / 2.0)),
        "floor_within_block_rms": float(math.sqrt(max(mean_within, 0.0) / 2.0)),
        "between_block_sd_rms": float(math.sqrt(max(mean_between, 0.0) / 2.0)),
        "floor_total_macro": float(np.mean(total_macro_terms)),
        "floor_within_block_macro": float(np.mean(within_macro_terms)),
        "scale_note": (
            "The *_rms values average squared differences before taking the root; "
            "the *_macro values take the root per surface first and match the "
            "surface_macro_rmse convention used by the benchmark and by "
            "anchor_self_error. Only *_macro may be compared with anchor numbers."
        ),
    }


def heteroscedasticity_by_response(
    pairs: Iterable[ReplicatePair],
    stratum: str = "all",
    quantity: str = "residual",
    edges: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.95, 1.01, 1.25, 1.5, 2.0, 3.0),
) -> list[dict[str, object]]:
    """Report the replicate floor inside bins of mean observed response.

    A single floor is only meaningful if the noise is roughly homoscedastic.
    Responses pinned near the upper bound have almost no replicate spread, so a
    pooled number can hide the level actually faced in the informative range.
    """

    level_values: list[np.ndarray] = []
    difference_values: list[np.ndarray] = []
    for pair in pairs:
        mask = _stratum_mask(pair, stratum)
        if not mask.any():
            continue
        first, second = pair.quantity(quantity)
        level_values.append(
            (pair.first_response[mask] + pair.second_response[mask]) / 2.0
        )
        difference_values.append(first[mask] - second[mask])
    if not level_values:
        raise ReplicateNoiseError(f"Stratum {stratum!r} retained no points")
    level = np.concatenate(level_values)
    difference = np.concatenate(difference_values)
    bins: list[dict[str, object]] = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        selected = (level >= low) & (level < high)
        count = int(selected.sum())
        entry: dict[str, object] = {
            "response_low": float(low),
            "response_high": float(high),
            "points": count,
        }
        if count > 0:
            paired = float(np.sqrt(np.mean(np.square(difference[selected]))))
            entry["paired_rmse"] = paired
            entry["floor"] = independent_observation_floor(paired)
        bins.append(entry)
    return bins


def bootstrap_interval(
    values: np.ndarray,
    seed: int,
    resamples: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval over independent replicate pairs."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ReplicateNoiseError("Bootstrap requires at least one replicate pair")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = generator.integers(0, values.size, size=values.size)
        means[index] = float(np.mean(values[sample]))
    lower = float(np.quantile(means, (1.0 - level) / 2.0))
    upper = float(np.quantile(means, 1.0 - (1.0 - level) / 2.0))
    return lower, upper


def independent_observation_floor(paired_rmse: float) -> float:
    """Convert a replicate-versus-replicate RMSE into a single-observation floor.

    With ``y1 = mu + e1`` and ``y2 = mu + e2`` sharing variance ``sigma^2``, the
    expected squared replicate difference is ``2 * sigma^2`` if errors are
    independent. A model predicting
    ``mu`` exactly is still scored against one noisy observation, so its
    attainable RMSE is ``sigma``, i.e. the paired RMSE divided by ``sqrt(2)``.
    The variance identity does not alone justify the same conversion for an
    expected macro RMSE: that needs further distributional assumptions. We
    report the conventional conversion as a sensitivity estimate, not a bound.
    """

    return float(paired_rmse / math.sqrt(2.0))


def _matched_gate_comparison(
    anchor: float,
    total_floor: float,
    within_floor: float,
    relative_gate: float,
) -> dict[str, object]:
    """Compare one matched replicate subset with a relative-improvement target.

    This comparison is deliberately local to one cohort.  A floor estimated on
    replicate surfaces cannot be combined with an anchor measured on the full
    validation cohort to infer attainable relative improvement.
    """

    values = (anchor, total_floor, within_floor, relative_gate)
    if not all(math.isfinite(value) for value in values):
        raise ReplicateNoiseError("Matched gate comparison requires finite values")
    if anchor < 0:
        raise ReplicateNoiseError("Matched gate comparison requires a non-negative anchor")
    if total_floor < 0 or within_floor < 0:
        raise ReplicateNoiseError("Noise floors cannot be negative")
    if not 0 < relative_gate < 1:
        raise ReplicateNoiseError("relative_gate must lie strictly between zero and one")

    target = anchor * (1.0 - relative_gate)
    total_above = total_floor >= target
    within_above = within_floor >= target
    if anchor == 0:
        status = "relative_target_undefined_zero_anchor"
    elif total_above and within_above:
        status = "sensitivity_estimates_above_target"
    elif not total_above and not within_above:
        status = "sensitivity_estimates_below_target"
    else:
        status = "sensitivity_estimates_straddle_target"
    return {
        "anchor_surface_macro_rmse": anchor,
        "relative_improvement_target": relative_gate,
        "target_surface_macro_rmse": target,
        "total_floor_surface_macro_rmse": total_floor,
        "within_block_floor_surface_macro_rmse": within_floor,
        "total_floor_above_target": total_above,
        "within_block_floor_above_target": within_above,
        "status": status,
        "scope": "matched_replicate_surfaces_only",
        "gate_reachability": "not_identified_without_noise_model_assumptions",
    }


def representativeness(
    shard_dir: Path | str,
    fold: int,
    roles: Sequence[str] = READABLE_ROLES,
) -> dict[str, object]:
    """Compare replicated and single-measurement groups on response statistics.

    Replicated groups are a small, non-randomly selected subset, so the noise
    estimate transfers to the full cohort only if the two populations look
    alike. This records the comparison rather than assuming it.
    """

    shard_dir = Path(shard_dir)
    columns = [
        "prediction_group_id",
        "surface_replicate_id",
        "response",
        "response_clipped_high",
    ]
    frames = [
        loaded
        for loaded in (
            pd.read_parquet(_shard_path(shard_dir, fold, role), columns=columns)
            for role in roles
        )
        if not loaded.empty
    ]
    if not frames:
        raise ReplicateNoiseError(f"Fold {fold} exposes no readable response rows")
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby("prediction_group_id")["surface_replicate_id"].nunique()
    replicated = set(counts[counts >= 2].index)
    is_replicated = frame["prediction_group_id"].isin(replicated)
    summary: dict[str, object] = {}
    for label, part in (
        ("replicated_groups", frame[is_replicated]),
        ("single_groups", frame[~is_replicated]),
    ):
        response = part["response"].to_numpy(dtype=np.float64)
        clipped = part["response_clipped_high"].to_numpy(dtype=bool)
        summary[label] = {
            "groups": int(part["prediction_group_id"].nunique()),
            "points": int(response.size),
            "response_mean": float(np.mean(response)) if response.size else float("nan"),
            "response_sd": float(np.std(response, ddof=1)) if response.size > 1 else float("nan"),
            "clipped_high_fraction": (
                float(np.mean(clipped)) if clipped.size else float("nan")
            ),
        }
    return summary


def estimate_fold(
    shard_dir: Path | str,
    fold: int,
    seed: int,
    resamples: int = 2000,
) -> dict[str, object]:
    """Estimate the replicate-noise floor for one fold's readable shards."""

    pairs = collect_replicate_pairs(shard_dir, fold)
    if not pairs:
        raise ReplicateNoiseError(f"Fold {fold} exposes no replicate pair in {READABLE_ROLES}")
    strata: dict[str, object] = {}
    for stratum in STRATA:
        entry: dict[str, object] = {}
        for quantity in QUANTITIES:
            try:
                macro, micro, per_pair, points = paired_surface_macro_rmse(
                    pairs, stratum, quantity
                )
            except ReplicateNoiseError:
                entry[quantity] = {"status": "empty"}
                continue
            lower, upper = bootstrap_interval(
                per_pair, seed=seed + fold, resamples=resamples
            )
            entry[quantity] = {
                "status": "estimated",
                "replicate_pairs": int(per_pair.size),
                "points": points,
                "paired_surface_macro_rmse": macro,
                "paired_micro_rmse": micro,
                "paired_macro_ci95": [lower, upper],
                "noise_floor_surface_macro_rmse": independent_observation_floor(macro),
                "noise_floor_ci95": [
                    independent_observation_floor(lower),
                    independent_observation_floor(upper),
                ],
                "noise_floor_micro_rmse": independent_observation_floor(micro),
            }
        try:
            anchor_rmse, anchor_points = anchor_self_error(pairs, stratum)
            entry["anchor_self_error"] = {
                "status": "estimated",
                "surface_macro_rmse": anchor_rmse,
                "points": anchor_points,
            }
        except ReplicateNoiseError:
            entry["anchor_self_error"] = {"status": "empty"}
        residual = entry.get("residual")
        anchor = entry.get("anchor_self_error")
        if (
            isinstance(residual, dict)
            and isinstance(anchor, dict)
            and residual.get("status") == "estimated"
            and anchor.get("status") == "estimated"
        ):
            floor = float(residual["noise_floor_surface_macro_rmse"])
            observed = float(anchor["surface_macro_rmse"])
            entry["residual_headroom"] = {
                "anchor_surface_macro_rmse": observed,
                "residual_noise_floor": floor,
                "absolute_headroom": observed - floor,
                "relative_headroom": (observed - floor) / observed if observed > 0 else None,
                "relative_headroom_status": "defined" if observed > 0 else "undefined_zero_anchor",
                "floor_exceeds_anchor": floor >= observed,
            }
        try:
            entry["variance_decomposition"] = decompose_pair_variance(pairs, stratum)
        except ReplicateNoiseError:
            entry["variance_decomposition"] = {"status": "empty"}
        decomposition = entry.get("variance_decomposition")
        if (
            isinstance(decomposition, dict)
            and isinstance(anchor, dict)
            and "floor_within_block_macro" in decomposition
            and anchor.get("status") == "estimated"
        ):
            within = float(decomposition["floor_within_block_macro"])
            observed = float(decomposition["matched_anchor_surface_macro_rmse"])
            entry["within_block_headroom"] = {
                "scope": "decomposition_retained_pairs_and_points_only",
                "minimum_shared_points": decomposition["minimum_shared_points"],
                "anchor_points": decomposition["matched_anchor_points"],
                "anchor_surface_macro_rmse": observed,
                "within_block_floor": within,
                "absolute_headroom": observed - within,
                "relative_headroom": (observed - within) / observed if observed > 0 else None,
                "relative_headroom_status": "defined" if observed > 0 else "undefined_zero_anchor",
                "floor_exceeds_anchor": within >= observed,
            }
            total = float(decomposition["floor_total_macro"])
            entry["decomposition_total_headroom"] = {
                "scope": "decomposition_retained_pairs_and_points_only",
                "anchor_surface_macro_rmse": observed,
                "total_floor": total,
                "absolute_headroom": observed - total,
                "relative_headroom": (observed - total) / observed if observed > 0 else None,
                "relative_headroom_status": "defined" if observed > 0 else "undefined_zero_anchor",
            }
        strata[stratum] = entry
    role_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        role_counts[pair.role] += 1
    return {
        "fold": fold,
        "replicate_pairs": len(pairs),
        "replicate_pairs_by_role": dict(sorted(role_counts.items())),
        "points_per_pair": sorted({pair.point_count for pair in pairs}),
        "strata": strata,
        "heteroscedasticity_residual": heteroscedasticity_by_response(pairs, "all"),
        "representativeness": representativeness(shard_dir, fold),
    }


def estimate_all_folds(
    shard_dir: Path | str,
    output_path: Path | str,
    folds: Sequence[int] = (0, 1, 2, 3, 4),
    seed: int = 41,
    resamples: int = 2000,
    anchor_reference: float | None = None,
    relative_gate: float = 0.05,
    endpoint: str = "not_declared",
) -> dict[str, object]:
    """Run the estimate across folds and write an auditable manifest."""

    if not math.isfinite(relative_gate) or not 0 < relative_gate < 1:
        raise ReplicateNoiseError("relative_gate must lie strictly between zero and one")
    shard_dir = Path(shard_dir)
    fold_results = [
        estimate_fold(shard_dir, fold, seed=seed, resamples=resamples) for fold in folds
    ]
    all_strata = [result["strata"]["all"] for result in fold_results]  # type: ignore[index]
    fold_floors = [
        float(stratum["variance_decomposition"]["floor_total_macro"])
        for stratum in all_strata
    ]
    fold_anchor = [
        float(stratum["variance_decomposition"]["matched_anchor_surface_macro_rmse"])
        for stratum in all_strata
    ]
    fold_within = [
        float(stratum["variance_decomposition"]["floor_within_block_macro"])
        for stratum in all_strata
    ]
    fold_between_share = [
        float(stratum["variance_decomposition"]["between_block_variance_share"])
        for stratum in all_strata
    ]
    fold_gate_comparisons = []
    for result in fold_results:
        stratum = result["strata"]["all"]  # type: ignore[index]
        fold_gate_comparisons.append(
            {
                "fold": int(result["fold"]),
                **_matched_gate_comparison(
                    float(stratum["variance_decomposition"]["matched_anchor_surface_macro_rmse"]),
                    float(
                        stratum["variance_decomposition"]["floor_total_macro"]
                    ),
                    float(
                        stratum["variance_decomposition"][
                            "floor_within_block_macro"
                        ]
                    ),
                    relative_gate,
                ),
            }
        )
    fold_mean_anchor = float(np.mean(fold_anchor))
    fold_mean_total = float(np.mean(fold_floors))
    fold_mean_within = float(np.mean(fold_within))
    manifest: dict[str, object] = {
        "version": REPLICATE_NOISE_VERSION,
        "endpoint": endpoint,
        "strata_definition": {
            "all": "all aligned points under the declared endpoint",
            "unclipped": "exclude points flagged response_clipped_high in either replicate; not an endpoint conversion",
            "clipped_high_both": "both replicates carry response_clipped_high",
        },
        "protocol": "OneNewScaffoldAnchored-v1",
        "metric": "surface_macro.rmse",
        "roles_read": list(READABLE_ROLES),
        "roles_refused": list(FORBIDDEN_ROLES),
        "test_shard_opened": False,
        "status": "validation_and_train_only_no_test_shard_opened",
        "claim_scope": (
            "replicate_subset_noise_sensitivity_no_proven_bounds_or_gate_reachability"
        ),
        "shard_dir": str(shard_dir),
        "seed": seed,
        "bootstrap_resamples": resamples,
        "quantities": list(QUANTITIES),
        "primary_quantity": "residual",
        "primary_quantity_rationale": (
            "Subtract each replicate's own HSA reference to examine an anchored "
            "residual contrast. This does not establish that shared block effects "
            "cancel or that the residual variance is unpredictable from model inputs."
        ),
        "folds": fold_results,
        "matched_replicate_subset": {
            "scope": "same replicate surfaces for floor and HSA anchor",
            "fold_mean_total_floor_surface_macro_rmse": fold_mean_total,
            "fold_sd_total_floor_surface_macro_rmse": (
                float(np.std(fold_floors, ddof=1))
                if len(fold_floors) > 1
                else None
            ),
            "fold_mean_anchor_surface_macro_rmse": fold_mean_anchor,
            "fold_mean_within_block_floor_surface_macro_rmse": fold_mean_within,
            "fold_mean_between_block_variance_share": float(
                np.mean(fold_between_share)
            ),
            "fold_mean_gate_comparison": _matched_gate_comparison(
                fold_mean_anchor,
                fold_mean_total,
                fold_mean_within,
                relative_gate,
            ),
            "per_fold_gate_comparisons": fold_gate_comparisons,
        },
        "sensitivity_interpretation": (
            "Total and centred within-block contrasts are assumption-dependent "
            "noise sensitivities, not proven lower/upper prediction-error bounds. "
            "Shared anchor errors, cross-dose covariance, and conditional "
            "predictability remain unidentified."
        ),
        "cohort_transfer_limitation": (
            "Absolute noise can vary with response composition, and the HSA anchor "
            "varies with it too. Replicate-subset floors therefore cannot establish "
            "relative headroom or gate reachability on the full validation cohort."
        ),
        "runtime_environment": _runtime_environment(),
    }
    if anchor_reference is not None:
        if not math.isfinite(anchor_reference) or anchor_reference <= 0:
            raise ReplicateNoiseError("anchor_reference must be finite and positive")
        manifest["external_anchor_context"] = {
            "anchor_surface_macro_rmse": anchor_reference,
            "relative_improvement_target": relative_gate,
            "target_surface_macro_rmse": anchor_reference * (1.0 - relative_gate),
            "scope": "external_or_cohort_wide_not_matched_to_replicate_subset",
            "gate_reachability_from_this_manifest": "not_identified",
            "reason": (
                "The anchor and replicate floors are measured on different cohort "
                "compositions, so cross-cohort headroom is not reported."
            ),
        }
    _write_json_atomic(Path(output_path), manifest)
    return manifest

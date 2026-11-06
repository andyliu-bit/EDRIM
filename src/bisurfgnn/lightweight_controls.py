"""Train-only feature controls and surface-weighted linear-kernel ridge.

Prediction is clipped HSA + boundary_gate * linear residual, not the neural
logit decoder. These are competing baselines, not decoder-matched ablations.
"""
from __future__ import annotations

import numpy as np
import torch

from .models.surface_head import DoseFeatures

VARIANTS = ("mono_dose_ridge", "mono_dose_cell_ridge", "morgan_cell_ridge",
            "entity_onehot_ridge", "morgan_shuffle_ridge", "cell_shuffle_ridge")


def _require_development(table):
    if table.role not in ("train", "validation"):
        raise ValueError("Only train/validation tables are allowed")


def fit_feature_state(train, banks, *, seed=41):
    _require_development(train)
    if train.role != "train":
        raise ValueError("Feature state must be fitted on train only")
    drug_ids = torch.unique(torch.cat([train.drug_a_index, train.drug_b_index])).long()
    cell_ids = torch.unique(train.cell_index).long()
    drug_map, cell_map = torch.arange(len(banks.drug_ids)), torch.arange(len(banks.cell_ids))
    # Permutations are among TRAIN entities only; unseen entities retain their
    # original covariates. The same mapping is applied in both roles.
    generator = torch.Generator().manual_seed(seed)
    drug_map[drug_ids] = drug_ids[torch.randperm(len(drug_ids), generator=generator)]
    cell_map[cell_ids] = cell_ids[torch.randperm(len(cell_ids), generator=generator)]
    return dict(train_drug_indices=drug_ids, train_cell_indices=cell_ids,
                drug_permutation=drug_map, cell_permutation=cell_map,
                drug_ids=list(banks.drug_ids), cell_ids=list(banks.cell_ids), seed=seed)


def _onehot(indices, vocabulary):
    # Unseen validation entities map to all zeros, not a fitted validation ID.
    return (indices.long()[:, None] == vocabulary[None, :]).double()


def feature_matrix(table, banks, state, variant):
    _require_development(table)
    if variant not in VARIANTS:
        raise ValueError("Unknown lightweight control")
    if list(banks.drug_ids) != state["drug_ids"] or list(banks.cell_ids) != state["cell_ids"]:
        raise ValueError("Feature bank entity order changed")
    a, b, cell = table.drug_a_index.long(), table.drug_b_index.long(), table.cell_index.long()
    dose = DoseFeatures(4)
    arm_a = torch.cat([dose(table.doses[:, 0]), table.mono_a[:, None]], 1).double()
    arm_b = torch.cat([dose(table.doses[:, 1]), table.mono_b[:, None]], 1).double()
    columns = [arm_a + arm_b, (arm_a - arm_b).abs(), arm_a * arm_b]
    if variant in ("morgan_cell_ridge", "morgan_shuffle_ridge", "entity_onehot_ridge"):
        if variant == "entity_onehot_ridge":
            fa, fb = _onehot(a, state["train_drug_indices"]), _onehot(b, state["train_drug_indices"])
        else:
            mapping = state["drug_permutation"] if variant == "morgan_shuffle_ridge" else torch.arange(len(banks.drug_ids))
            fa, fb = banks.fingerprints[mapping[a]].double(), banks.fingerprints[mapping[b]].double()
        # Drug identity/covariates stay attached to the corresponding dose and
        # monotherapy without arbitrary A/B sorting.
        columns += [fa + fb, fa * table.doses[:, 0, None] + fb * table.doses[:, 1, None],
                    fa * table.mono_a[:, None] + fb * table.mono_b[:, None]]
    if variant == "entity_onehot_ridge":
        columns.append(_onehot(cell, state["train_cell_indices"]))
    elif variant != "mono_dose_ridge":
        lookup = state["cell_permutation"][cell] if variant == "cell_shuffle_ridge" else cell
        columns.append(banks.hallmarks[lookup].double())
    features = torch.cat(columns, 1)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Non-finite features")
    return features


def reference_and_gate(table):
    _require_development(table)
    reference = torch.minimum(table.mono_a, table.mono_b).double()
    doses = table.doses.double()
    gate = (1 - torch.exp(-2 * doses[:, 0])) * (1 - torch.exp(-2 * doses[:, 1]))
    return reference, gate


def surface_weights(surface_ids):
    _, inverse, counts = np.unique(surface_ids, return_inverse=True, return_counts=True)
    return torch.from_numpy(1.0 / (len(counts) * counts[inverse])).double()


def ridge_fit(features, table, *, alpha=0.01, solver="auto"):
    _require_development(table)
    if table.role != "train":
        raise ValueError("Ridge fitting is train-only")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if solver not in ("auto", "primal", "dual"):
        raise ValueError("Invalid ridge solver")
    x = features.double()
    if x.ndim != 2 or len(x) != len(table) or not bool(torch.isfinite(x).all()):
        raise ValueError("Invalid training features")
    weight = surface_weights(table.surface_ids)
    mean = (x * weight[:, None]).sum(0)
    variance = ((x - mean).square() * weight[:, None]).sum(0)
    scale = torch.where(variance > 1e-12, variance.sqrt(), torch.ones_like(variance))
    reference, gate = reference_and_gate(table)
    design = torch.cat([torch.ones(len(x), 1, dtype=torch.float64), (x - mean) / scale], 1) * gate[:, None]
    residual = table.target.double() - reference
    weighted = design * weight.sqrt()[:, None]
    target = residual * weight.sqrt()
    resolved = ("dual" if len(x) < design.shape[1] else "primal") if solver == "auto" else solver
    if resolved == "dual":
        coefficient = weighted.T @ torch.linalg.solve(
            weighted @ weighted.T + alpha * torch.eye(len(x), dtype=torch.float64), target)
    else:
        coefficient = torch.linalg.solve(weighted.T @ weighted + alpha * torch.eye(
            design.shape[1], dtype=torch.float64), weighted.T @ target)
    normal_residual = weighted.T @ (weighted @ coefficient - target) + alpha * coefficient
    relative_error = float(normal_residual.norm() / (1 + (weighted.T @ target).norm()))
    if not bool(torch.isfinite(coefficient).all()) or relative_error > 1e-7:
        raise RuntimeError("Ridge normal equation check failed")
    objective = float((weighted @ coefficient - target).square().sum() + alpha * coefficient.square().sum())
    anchor_objective = float(target.square().sum())
    if objective > anchor_objective + 1e-9:
        raise RuntimeError("Ridge objective exceeds its zero-coefficient candidate")
    return dict(coefficient=coefficient, mean=mean, scale=scale, alpha=float(alpha),
                solver=resolved, normal_equation_relative_error=relative_error,
                training_objective=objective, zero_candidate_objective=anchor_objective,
                upper_bound=float(table.monotherapy_upper_bound))


def ridge_predict(features, table, state, *, use_anchor=False):
    reference, gate = reference_and_gate(table)
    if float(table.monotherapy_upper_bound) != state["upper_bound"]:
        raise ValueError("Response bound differs from fitted state")
    if features.shape != (len(table), len(state["mean"])) or not bool(torch.isfinite(features).all()):
        raise ValueError("Invalid prediction features")
    if use_anchor:
        return reference.numpy()
    standardized = (features.double() - state["mean"]) / state["scale"]
    delta = state["coefficient"][0] + standardized @ state["coefficient"][1:]
    prediction = (reference + gate * delta).clamp(0, state["upper_bound"])
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Non-finite ridge prediction")
    return prediction.numpy()

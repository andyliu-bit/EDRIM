"""Exact primal sufficient statistics with bounded feature batches, no N x N kernel."""
from dataclasses import fields, replace

import numpy as np
import torch

from . import lightweight_controls as light


def slice_table(table, start, stop):
    updates = {}
    for field in fields(table):
        value = getattr(table, field.name)
        if isinstance(value, (torch.Tensor, np.ndarray)):
            updates[field.name] = value[start:stop]
    return replace(table, **updates)


def feature_blocks(table, banks, feature_state, variant, batch_points):
    if type(batch_points) is not int or batch_points < 1:
        raise ValueError("batch_points must be positive")
    for start in range(0, len(table), batch_points):
        part = slice_table(table, start, min(start+batch_points, len(table)))
        yield start, part, light.feature_matrix(part, banks, feature_state, variant)


def fit_streaming(table, banks, feature_state, variant, *, alpha=.01, batch_points=512,
                  device="cpu", max_working_bytes=2_000_000_000):
    if table.role != "train" or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Train-only fitting with positive alpha required")
    weights = light.surface_weights(table.surface_ids)
    device = torch.device(device)
    mean = None
    for start, part, x in feature_blocks(table, banks, feature_state, variant, batch_points):
        x = x.to(device)
        if mean is None:
            dimension = x.shape[1]+1
            estimate = 8*(4*dimension**2 + 6*batch_points*dimension)
            if estimate > max_working_bytes:
                raise ValueError("Declared sufficient-statistic memory budget exceeded")
            mean = torch.zeros(x.shape[1], dtype=torch.float64, device=device)
        mean += (x*weights[start:start+len(part)].to(device)[:, None]).sum(0)
    variance = torch.zeros_like(mean)
    for start, part, x in feature_blocks(table, banks, feature_state, variant, batch_points):
        variance += ((x.to(device)-mean).square()*weights[start:start+len(part)].to(device)[:, None]).sum(0)
    scale = torch.where(variance > 1e-12, variance.sqrt(), torch.ones_like(variance))
    gram = torch.zeros((dimension, dimension), dtype=torch.float64, device=device)
    rhs = torch.zeros(dimension, dtype=torch.float64, device=device)
    zero_objective = 0.0
    for start, part, x in feature_blocks(table, banks, feature_state, variant, batch_points):
        reference, gate = light.reference_and_gate(part)
        design = torch.cat([torch.ones(len(part), 1, dtype=torch.float64, device=device),
                            (x.to(device)-mean)/scale], 1)*gate.to(device)[:, None]
        root_weight = weights[start:start+len(part)].sqrt().to(device)
        design *= root_weight[:, None]
        y = (part.target.double()-reference).to(device)*root_weight
        gram.addmm_(design.T, design)
        rhs += design.T@y
        zero_objective += float(y.square().sum())
    gram.diagonal().add_(alpha)
    coefficient = torch.linalg.solve(gram, rhs)
    normal_error = float((gram@coefficient-rhs).norm()/(1+rhs.norm()))
    objective = float(coefficient@(gram@coefficient)-2*coefficient@rhs)+zero_objective
    if not torch.isfinite(coefficient).all() or normal_error > 1e-7 or objective > zero_objective+1e-8:
        raise RuntimeError("Streaming ridge numerical verification failed")
    return dict(coefficient=coefficient.cpu(), mean=mean.cpu(), scale=scale.cpu(), alpha=float(alpha),
        upper_bound=float(table.monotherapy_upper_bound), solver="streaming_primal",
        normal_equation_relative_error=normal_error, training_objective=objective,
        zero_candidate_objective=zero_objective, estimated_working_bytes=estimate, batch_points=batch_points)


def predict_streaming(table, banks, feature_state, variant, state, *, use_anchor=False, batch_points=512):
    return np.concatenate([light.ridge_predict(x, part, state, use_anchor=use_anchor)
        for _, part, x in feature_blocks(table, banks, feature_state, variant, batch_points)])

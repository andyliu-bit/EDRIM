"""JointFeature-v1: all full-XGBoost features modelled jointly.

Feature layout (identical to ``run_algorithm_search.xgb_features``, 55 columns):
  [m_a, m_b, d_a, d_b, min(m), m_a*m_b, |m_a-m_b|, drugPCA_a(16), drugPCA_b(16), cellPCA(16)]
Exchange (drug A <-> drug B) swaps m, d and drug PCA blocks together; the cell
block and the three pair scalars are unchanged.

* ``JointTabM`` (candidate A): official TabM over the 55 features, predicting the
  response residual over HSA; members trained with their own losses, averaged at
  inference; full function exchange-averaged.
  encoding="ple_all": rtdl PLE on all 55 columns.
  encoding="ple_mono_linear_pca": PLE on the 7 monotherapy/dose columns,
  training-standardised linear input for the 48 PCA columns.
* ``StateConditionedCorrector`` (candidate B): small arm-shared MLP over
  [m, d, drugPCA] per arm + pair scalars + cell PCA, added on top of a frozen
  numeric TabM; exactly symmetric by construction.
All fitted transforms (PCA, standardisation, bins) come from the training role.
"""
from __future__ import annotations

from dataclasses import dataclass

import rtdl_num_embeddings as rne
import torch
from torch import Tensor, nn

from .numeric_surface import boundary_gate, pair_scalars


@dataclass
class JointStats:
    drug_codes: Tensor      # [n_drugs, 16]
    cell_codes: Tensor      # [n_cells, 16]
    pca_mean: Tensor        # [48] standardisation of PCA columns (training rows)
    pca_sd: Tensor          # [48]
    drug_mean: Tensor       # [16] pooled over both drug arms (keeps the corrector exchange-symmetric)
    drug_sd: Tensor         # [16]
    bins_all: list          # 55 bin tensors (training rows)
    bins_mono: list         # 7 bin tensors
    residual_scale: float


def raw_features(mono_a, mono_b, doses, a, b, c, drug_codes, cell_codes, swap=False):
    if swap:
        mono_a, mono_b, doses, a, b = mono_b, mono_a, doses.flip(1), b, a
    return torch.cat([torch.stack([mono_a, mono_b, doses[:, 0], doses[:, 1]], 1), pair_scalars(mono_a, mono_b),
                      drug_codes[a.long()], drug_codes[b.long()], cell_codes[c.long()]], 1)


def fit_joint_stats(mono_a, mono_b, doses, a, b, c, target, drug_codes, cell_codes, n_bins=16) -> JointStats:
    x = raw_features(mono_a, mono_b, doses, a, b, c, drug_codes, cell_codes).double()
    pca = x[:, 7:]
    arms = torch.cat([x[:, 7:23], x[:, 23:39]])
    return JointStats(drug_codes=drug_codes.float(), cell_codes=cell_codes.float(),
                      pca_mean=pca.mean(0).float(), pca_sd=pca.std(0).clamp_min(1e-6).float(),
                      drug_mean=arms.mean(0).float(), drug_sd=arms.std(0).clamp_min(1e-6).float(),
                      bins_all=[t.float() for t in rne.compute_bins(x, n_bins=n_bins)],
                      bins_mono=[t.float() for t in rne.compute_bins(x[:, :7], n_bins=n_bins)],
                      residual_scale=float((target - torch.minimum(mono_a, mono_b)).double().std()))


class JointTabM(nn.Module):
    def __init__(self, stats: JointStats, encoding: str = "ple_all", d_embedding: int = 16, k: int = 32,
                 width: int = 256, blocks: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        import tabm
        if encoding not in ("ple_all", "ple_mono_linear_pca"):
            raise ValueError(encoding)
        self.encoding, self.scale = encoding, stats.residual_scale
        for name in ("drug_codes", "cell_codes", "pca_mean", "pca_sd"):
            self.register_buffer(name, getattr(stats, name).clone(), persistent=True)
        if encoding == "ple_all":
            self.net = tabm.TabM.make(n_num_features=55, d_out=1, k=k, d_block=width, n_blocks=blocks, dropout=dropout,
                                      num_embeddings=rne.PiecewiseLinearEmbeddings(stats.bins_all, d_embedding,
                                                                                   activation=False, version="B"))
        else:
            self.mono_embedding = rne.PiecewiseLinearEmbeddings(stats.bins_mono, d_embedding, activation=False, version="B")
            self.net = tabm.TabM.make(n_num_features=7 * d_embedding + 48, d_out=1, k=k, d_block=width,
                                      n_blocks=blocks, dropout=dropout)
        for p in self.net.output.parameters():
            nn.init.zeros_(p)

    def _one(self, x):
        if self.encoding == "ple_all":
            return self.net(x).squeeze(-1)
        z = torch.cat([self.mono_embedding(x[:, :7]).flatten(1), (x[:, 7:] - self.pca_mean) / self.pca_sd], 1)
        return self.net(z).squeeze(-1)

    def members(self, mono_a, mono_b, doses, a, b, c) -> Tensor:
        f = lambda s: self._one(raw_features(mono_a, mono_b, doses, a, b, c, self.drug_codes, self.cell_codes, swap=s))
        return 0.5 * self.scale * (f(False) + f(True))  # [B, k]

    def forward(self, mono_a, mono_b, doses, a, b, c) -> Tensor:
        return self.members(mono_a, mono_b, doses, a, b, c).mean(1)


class StateConditionedCorrector(nn.Module):
    def __init__(self, stats: JointStats, arm_width: int = 64, width: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.scale = stats.residual_scale
        for name in ("drug_codes", "cell_codes", "pca_mean", "pca_sd", "drug_mean", "drug_sd"):
            self.register_buffer(name, getattr(stats, name).clone(), persistent=True)
        self.arm = nn.Sequential(nn.LayerNorm(18), nn.Linear(18, arm_width), nn.GELU(), nn.Linear(arm_width, arm_width))
        self.head = nn.Sequential(nn.LayerNorm(3 * arm_width + 3 + 16), nn.Linear(3 * arm_width + 3 + 16, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, mono_a, mono_b, doses, a, b, c) -> Tensor:
        ms, ss = self.pca_mean, self.pca_sd
        za = (self.drug_codes[a.long()] - self.drug_mean) / self.drug_sd  # same transform for both arms
        zb = (self.drug_codes[b.long()] - self.drug_mean) / self.drug_sd
        zc = (self.cell_codes[c.long()] - ms[32:]) / ss[32:]
        ua = self.arm(torch.cat([mono_a[:, None], doses[:, :1], za], 1))
        ub = self.arm(torch.cat([mono_b[:, None], doses[:, 1:], zb], 1))
        h = torch.cat([ua + ub, (ua - ub).abs(), ua * ub, pair_scalars(mono_a, mono_b), zc], 1)
        return self.scale * self.head(h).squeeze(-1)


class JointModel(nn.Module):
    """HSA + gate * (base + joint). A: base=None, joint=JointTabM. B: base=TabMNumeric, joint=corrector."""

    def __init__(self, joint: nn.Module, base: nn.Module | None = None) -> None:
        super().__init__()
        self.joint, self.base = joint, base

    def residual_members(self, mono_a, mono_b, doses, a, b, c, members: bool):
        def part(m, numeric):
            if members and hasattr(m, "members") and m.training:
                return m.members(mono_a, mono_b, doses) if numeric else m.members(mono_a, mono_b, doses, a, b, c)
            return m(mono_a, mono_b, doses) if numeric else m(mono_a, mono_b, doses, a, b, c)
        r = part(self.joint, False)
        if self.base is not None:
            rb = part(self.base, True)
            r = (rb + r[:, None]) if rb.ndim == 2 else ((rb[:, None] + r) if r.ndim == 2 else rb + r)
        return r

    def predict(self, mono_a, mono_b, doses, a, b, c, members=False) -> Tensor:
        r = self.residual_members(mono_a, mono_b, doses, a, b, c, members)
        anchor, gate = torch.minimum(mono_a, mono_b), boundary_gate(doses)
        return anchor[:, None] + gate[:, None] * r if r.ndim == 2 else anchor + gate * r


__all__ = ["JointStats", "JointTabM", "StateConditionedCorrector", "JointModel", "fit_joint_stats", "raw_features"]

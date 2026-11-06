"""SparseMono-v1 models.

Stage 1 (monotherapy reconstruction):
  * ``MonoPriorNet``: drug encoder (pooled GNN over the molecular graph, or an MLP over
    Morgan-512 bits) + pooled cell pathway-graph encoder + standardised dose -> monotherapy
    response. The k=0 prediction.
  * ``CorrectionMLP``: small shared network predicting the residual y(d) - p(d) from
    [y_s - p(d_s), d - d_s, p(d), p(d_s)]; with k=1 the reconstruction is p(d) + correction.
Stage 2 (combination): ``FeatureTabM`` — official TabM over a precomputed feature matrix,
  exchange-averaged through an explicit column permutation, predicting the residual over the
  ESTIMATED HSA = min(partner monotherapy, reconstructed target monotherapy).
"""
from __future__ import annotations

import rtdl_num_embeddings as rne
import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch
from .graph_encoder import GraphEncoder
from .numeric_surface import boundary_gate


class MonoPriorNet(nn.Module):
    def __init__(self, drug_input: str, atom_feature_dim: int, cell_feature_dim: int, morgan_dim: int = 512,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1, dose_mean: float = 0.0,
                 dose_sd: float = 1.0, mono_mean: float = 0.8) -> None:
        super().__init__()
        if drug_input not in ("gnn", "morgan"):
            raise ValueError(drug_input)
        self.drug_input = drug_input
        if drug_input == "gnn":
            self.drug_encoder = GraphEncoder(atom_feature_dim, hidden, layers, dropout)
        else:
            self.drug_encoder = nn.Sequential(nn.Linear(morgan_dim, 2 * hidden), nn.GELU(), nn.Dropout(dropout),
                                              nn.Linear(2 * hidden, hidden))
        self.cell_encoder = GraphEncoder(cell_feature_dim, hidden, layers, dropout)
        self.register_buffer("dose_stats", torch.tensor([dose_mean, max(dose_sd, 1e-6)]), persistent=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden + 1), nn.Linear(2 * hidden + 1, 2 * hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.head[-1].bias, mono_mean)

    def encode(self, drug_input, cell_graphs: GraphBatch) -> tuple[Tensor, Tensor]:
        drugs = self.drug_encoder(drug_input).pooled if self.drug_input == "gnn" else self.drug_encoder(drug_input)
        return drugs, self.cell_encoder(cell_graphs).pooled

    def predict(self, drugs: Tensor, cells: Tensor, d_idx: Tensor, c_idx: Tensor, dose: Tensor) -> Tensor:
        z = (dose - self.dose_stats[0]) / self.dose_stats[1]
        return self.head(torch.cat([drugs[d_idx.long()], cells[c_idx.long()], z[:, None]], 1)).squeeze(-1)


class CorrectionMLP(nn.Module):
    def __init__(self, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def features(prior_q, prior_s, support_value, dose_q, dose_s):
        return torch.stack([support_value - prior_s, dose_q - dose_s, prior_q, prior_s], 1)

    def forward(self, prior_q, prior_s, support_value, dose_q, dose_s) -> Tensor:
        return self.net(self.features(prior_q, prior_s, support_value, dose_q, dose_s)).squeeze(-1)


class FeatureTabM(nn.Module):
    """TabM over a fixed feature matrix; ``swap`` is the column permutation for drug exchange.

    Column 0/1 = arm-a/arm-b monotherapy actually supplied (observed or reconstructed),
    so the estimated HSA is min(x[:,0], x[:,1]); columns ``dose_cols`` give the doses for the gate."""

    def __init__(self, bins: list, swap: list[int], residual_scale: float, dose_cols=(4, 5), d_embedding: int = 16,
                 k: int = 32, width: int = 256, blocks: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        import tabm
        self.register_buffer("swap", torch.as_tensor(swap, dtype=torch.long), persistent=True)
        if sorted(swap) != list(range(len(bins))) or [swap[i] for i in swap] != list(range(len(bins))):
            raise ValueError("swap must be an involutive permutation of all columns")
        self.scale, self.dose_cols = float(residual_scale), tuple(dose_cols)
        self.net = tabm.TabM.make(n_num_features=len(bins), d_out=1, k=k, d_block=width, n_blocks=blocks, dropout=dropout,
                                  num_embeddings=rne.PiecewiseLinearEmbeddings(bins, d_embedding, activation=False, version="B"))
        for p in self.net.output.parameters():
            nn.init.zeros_(p)

    def members(self, x: Tensor) -> Tensor:
        return 0.5 * self.scale * (self.net(x).squeeze(-1) + self.net(x[:, self.swap]).squeeze(-1))

    def predict(self, x: Tensor, members: bool = False) -> Tensor:
        r = self.members(x) if members else self.members(x).mean(1)
        anchor = torch.minimum(x[:, 0], x[:, 1])
        gate = boundary_gate(x[:, list(self.dose_cols)])
        return anchor[:, None] + gate[:, None] * r if r.ndim == 2 else anchor + gate * r


__all__ = ["MonoPriorNet", "CorrectionMLP", "FeatureTabM"]

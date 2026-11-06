"""NumericSurface-v1: strong numeric predictor + small graph-conditioned surface corrector.

    prediction = HSA + g(dose) * [ r_num(m_a, d_a, m_b, d_b) + r_graph(A, B, cell, d_a, d_b) ]

* ``HSA = min(m_a, m_b)`` is the observed-monotherapy anchor; ``g`` is the existing
  boundary gate (zero when either dose is zero), so the zero-dose boundary returns
  the anchor exactly.
* ``r_num`` sees only the information of the low-dimensional XGBoost reference:
  per-arm observed monotherapy and dose, plus the symmetric pair scalars
  min/product/|difference| of the monotherapy values.
* ``r_graph = sum_ij C_ij(A, B, cell) phi_i(d_a) phi_j(d_b)``: a small pooled-GNN
  pair/cell representation produces 3x3 coefficients of a tensor-product
  piecewise-linear dose basis whose knots are fixed from training-arm doses.
  All dose points of one (pair, cell) share the coefficients.

Borrowed components (not original to this project):
  piecewise-linear numeric embeddings, Gorishniy et al. 2022 (arXiv 2203.05556),
  package ``rtdl_num_embeddings`` 0.0.12 (Apache-2.0); TabM, Gorishniy et al. 2024,
  package ``tabm`` 0.0.3 (Apache-2.0); graph encoder reused from this repository.

Exchange symmetry: arm-shared encoders with symmetric pair aggregation for the PLE
MLP; explicit exchange averaging of the complete function for TabM and for the
graph correction (coefficients and dose axes swap together).
"""
from __future__ import annotations

from dataclasses import dataclass

import rtdl_num_embeddings as rne
import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch
from .graph_encoder import GraphEncoder


@dataclass(frozen=True)
class NumericStats:
    """Fitted on the training role only."""
    arm_bins: list          # [m bins, d bins]
    pair_bins: list         # [min, product, absdiff] bins
    flat_bins: list         # [m_a, m_b, d_a, d_b, min, product, absdiff] bins (TabM)
    residual_scale: float   # SD of (response - HSA) on training points
    dose_knots: tuple       # 3 knots for the graph dose basis

    @staticmethod
    def fit(mono_a: Tensor, mono_b: Tensor, dose_a: Tensor, dose_b: Tensor, target: Tensor,
            n_bins: int = 16) -> "NumericStats":
        arm = torch.stack([torch.cat([mono_a, mono_b]), torch.cat([dose_a, dose_b])], dim=1).double()
        pair = pair_scalars(mono_a, mono_b).double()
        flat = torch.cat([torch.stack([mono_a, mono_b, dose_a, dose_b], 1), pair_scalars(mono_a, mono_b)], 1).double()
        knots = torch.quantile(torch.cat([dose_a, dose_b]).double(), torch.tensor([.1, .5, .9], dtype=torch.float64))
        hsa = torch.minimum(mono_a, mono_b)
        return NumericStats(
            arm_bins=[b.float() for b in rne.compute_bins(arm, n_bins=n_bins)],
            pair_bins=[b.float() for b in rne.compute_bins(pair, n_bins=n_bins)],
            flat_bins=[b.float() for b in rne.compute_bins(flat, n_bins=n_bins)],
            residual_scale=float((target - hsa).double().std()),
            dose_knots=tuple(float(k) for k in knots))


def pair_scalars(mono_a: Tensor, mono_b: Tensor) -> Tensor:
    return torch.stack([torch.minimum(mono_a, mono_b), mono_a * mono_b, (mono_a - mono_b).abs()], dim=-1)


def boundary_gate(doses: Tensor) -> Tensor:
    return (1.0 - torch.exp(-2.0 * doses[:, 0])) * (1.0 - torch.exp(-2.0 * doses[:, 1]))


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class PLEResidualMLP(nn.Module):
    """Candidate A: piecewise-linear embeddings + arm-shared encoder + residual MLP."""

    def __init__(self, stats: NumericStats, d_embedding: int = 16, arm_width: int = 128,
                 width: int = 256, blocks: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = stats.residual_scale
        self.arm_embedding = rne.PiecewiseLinearEmbeddings(stats.arm_bins, d_embedding, activation=False, version="B")
        self.pair_embedding = rne.PiecewiseLinearEmbeddings(stats.pair_bins, d_embedding, activation=False, version="B")
        self.arm = nn.Sequential(nn.Linear(2 * d_embedding, arm_width), nn.ReLU(), nn.Linear(arm_width, arm_width))
        self.inp = nn.Linear(3 * arm_width + 3 * d_embedding, width)
        self.blocks = nn.Sequential(*[_ResidualBlock(width, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, mono_a: Tensor, mono_b: Tensor, doses: Tensor) -> Tensor:
        ua = self.arm(self.arm_embedding(torch.stack([mono_a, doses[:, 0]], 1)).flatten(1))
        ub = self.arm(self.arm_embedding(torch.stack([mono_b, doses[:, 1]], 1)).flatten(1))
        pair = self.pair_embedding(pair_scalars(mono_a, mono_b)).flatten(1)
        x = self.inp(torch.cat([ua + ub, (ua - ub).abs(), ua * ub, pair], 1))
        return self.scale * self.head(self.blocks(x)).squeeze(-1)


class TabMNumeric(nn.Module):
    """Candidate B: official TabM (k members, BatchEnsemble) + PLE, exchange-averaged."""

    def __init__(self, stats: NumericStats, d_embedding: int = 16, k: int = 32, width: int = 256,
                 blocks: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        import tabm
        self.scale = stats.residual_scale
        self.net = tabm.TabM.make(n_num_features=7, d_out=1, k=k, d_block=width, n_blocks=blocks, dropout=dropout,
                                  num_embeddings=rne.PiecewiseLinearEmbeddings(stats.flat_bins, d_embedding,
                                                                               activation=False, version="B"))
        out = self.net.output  # zero-start: the untrained model returns the anchor
        for p in out.parameters():
            nn.init.zeros_(p)

    def members(self, mono_a: Tensor, mono_b: Tensor, doses: Tensor) -> Tensor:
        """[B, k] member predictions, averaged over both drug orders."""
        def one(ma, mb, da, db):
            x = torch.cat([torch.stack([ma, mb, da, db], 1), pair_scalars(ma, mb)], 1)
            return self.net(x).squeeze(-1)
        return 0.5 * self.scale * (one(mono_a, mono_b, doses[:, 0], doses[:, 1])
                                   + one(mono_b, mono_a, doses[:, 1], doses[:, 0]))

    def forward(self, mono_a: Tensor, mono_b: Tensor, doses: Tensor) -> Tensor:
        return self.members(mono_a, mono_b, doses).mean(dim=1)


class DoseBasis(nn.Module):
    """Partition-of-unity piecewise-linear basis on 3 fixed knots.

    Beyond the outer knots the basis is CONSTANT ([1,0,0] below the first knot,
    [0,0,1] above the last), because ``clamp(0, 1)`` is applied; it does not
    extrapolate linearly. This matches the frozen NumericSurface-v1 protocol;
    an earlier version of this docstring wrongly said "linear ramps"
    (corrected 2026-10-10, behaviour unchanged). Contrast: the official
    ``rtdl_num_embeddings`` PLE used by the numeric branch DOES extrapolate
    linearly in its first and last components.
    """

    def __init__(self, knots: tuple) -> None:
        super().__init__()
        if len(knots) != 3 or not (knots[0] < knots[1] < knots[2]):
            raise ValueError("Three increasing knots are required")
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32), persistent=True)

    def forward(self, dose: Tensor) -> Tensor:
        k0, k1, k2 = self.knots
        left = ((k1 - dose) / (k1 - k0)).clamp(0, 1)
        right = ((dose - k1) / (k2 - k1)).clamp(0, 1)
        return torch.stack([left, 1 - left - right, right], dim=-1)  # [B, 3], rows sum to 1


class GraphSurfaceCorrector(nn.Module):
    """Pooled drug/cell GNN -> 3x3 dose-basis coefficients per (drug A, drug B, cell)."""

    def __init__(self, atom_feature_dim: int, cell_feature_dim: int, stats: NumericStats,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = stats.residual_scale
        self.drug_encoder = GraphEncoder(atom_feature_dim, hidden, layers, dropout)
        self.cell_encoder = GraphEncoder(cell_feature_dim, hidden, layers, dropout)
        self.basis = DoseBasis(stats.dose_knots)
        self.coefficients = nn.Sequential(nn.LayerNorm(4 * hidden), nn.Linear(4 * hidden, 128), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(128, 9))
        nn.init.zeros_(self.coefficients[-1].weight); nn.init.zeros_(self.coefficients[-1].bias)

    def encode(self, drug_graphs: GraphBatch, cell_graphs: GraphBatch) -> tuple[Tensor, Tensor]:
        return self.drug_encoder(drug_graphs).pooled, self.cell_encoder(cell_graphs).pooled

    def triple_coefficients(self, drugs: Tensor, cells: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """Ordered 3x3 coefficients, computed once per distinct (A, B, cell) and gathered."""
        a, b, c = a.long(), b.long(), c.long()
        nd, nc = drugs.shape[0], cells.shape[0]
        key = (a * nd + b) * nc + c
        unique, inverse = torch.unique(key, return_inverse=True)
        uc = unique % nc; ab = torch.div(unique, nc, rounding_mode="floor")
        ua, ub = torch.div(ab, nd, rounding_mode="floor"), ab % nd
        ha, hb, hc = drugs[ua], drugs[ub], cells[uc]
        coef = self.coefficients(torch.cat([ha, hb, ha * hb, hc], -1)).view(-1, 3, 3)
        return coef[inverse]

    def forward_banks(self, drugs: Tensor, cells: Tensor, a: Tensor, b: Tensor, c: Tensor, doses: Tensor) -> Tensor:
        pa, pb = self.basis(doses[:, 0]), self.basis(doses[:, 1])
        c_ab = self.triple_coefficients(drugs, cells, a, b, c)
        c_ba = self.triple_coefficients(drugs, cells, b, a, c)
        # exchange averaging of the complete function: C_ba is applied with the axes swapped
        f_ab = torch.einsum("ni,nij,nj->n", pa, c_ab, pb)
        f_ba = torch.einsum("ni,nij,nj->n", pb, c_ba, pa)
        return 0.5 * self.scale * (f_ab + f_ba)


class LowDimSurfaceCorrector(GraphSurfaceCorrector):
    """C4: same dose basis and coefficient head, but conditioned on fixed low-dimensional codes.

    ``drug_codes`` [n_drugs, p] and ``cell_codes`` [n_cells, q] are precomputed (PCA fitted on
    training drugs/cells only) and indexed by the global bank indices. No graph encoder.
    """

    def __init__(self, drug_codes: Tensor, cell_codes: Tensor, stats: NumericStats, dropout: float = 0.1) -> None:
        nn.Module.__init__(self)
        self.scale = stats.residual_scale
        self.register_buffer("drug_codes", drug_codes.float(), persistent=True)
        self.register_buffer("cell_codes", cell_codes.float(), persistent=True)
        self.basis = DoseBasis(stats.dose_knots)
        width = 3 * drug_codes.shape[1] + cell_codes.shape[1]
        self.coefficients = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(128, 9))
        nn.init.zeros_(self.coefficients[-1].weight); nn.init.zeros_(self.coefficients[-1].bias)

    def encode(self, drug_graphs=None, cell_graphs=None) -> tuple[Tensor, Tensor]:
        return self.drug_codes, self.cell_codes


class NumericSurfaceModel(nn.Module):
    """HSA + gate * (numeric + graph). ``use_graph=False`` is the numeric-only candidate."""

    def __init__(self, numeric: nn.Module, corrector: GraphSurfaceCorrector | None = None) -> None:
        super().__init__()
        self.numeric = numeric
        self.corrector = corrector

    def residuals(self, mono_a, mono_b, doses, drugs=None, cells=None, a=None, b=None, c=None):
        r_num = self.numeric(mono_a, mono_b, doses)
        r_graph = torch.zeros_like(r_num) if self.corrector is None else \
            self.corrector.forward_banks(drugs, cells, a, b, c, doses)
        return r_num, r_graph

    def predict(self, mono_a, mono_b, doses, drugs=None, cells=None, a=None, b=None, c=None) -> Tensor:
        r_num, r_graph = self.residuals(mono_a, mono_b, doses, drugs, cells, a, b, c)
        return torch.minimum(mono_a, mono_b) + boundary_gate(doses) * (r_num + r_graph)


__all__ = ["NumericStats", "PLEResidualMLP", "TabMNumeric", "DoseBasis", "GraphSurfaceCorrector",
           "NumericSurfaceModel", "boundary_gate", "pair_scalars"]

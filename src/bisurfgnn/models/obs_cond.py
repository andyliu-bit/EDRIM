"""ObsCond-v1: observation-conditioned dual-drug response model.

Protocol: docs/OBS_COND_V1_PROTOCOL.md. Unlike SparseMono-v1 (graph -> monotherapy scalar ->
generic numeric correction -> fixed-feature combination head), the drug representation here is
updated by the permitted observations and the UPDATED representations (pooled and atom tokens)
enter the symmetric combination interaction directly; the combination loss reaches the encoder.

Per arm (both arms share every module):
  A  z = Z([drug pooled, cell pooled])                  drug-in-cell representation
  B  s = masked mean_j E([z, d_j, y_j, p_j, y_j - p_j])  support-set encoding (order invariant)
     z' = z * (1 + g) + b,  [g, b] = has_support * U([z, s])           (FiLM, zero-init)
     tokens' = tokens * (1 + g_c + g_t) + (b_c + b_t)                   shared per-drug modulation
       [g_c, b_c] = C(cell), [g_t, b_t] = has_support * T([z, s])      (no atom-index dependence)
  D  mono_hat(d) = p(d) + M([z', cell, d, s_q, p(d)]),  s_q = masked mean_j Q([d - d_j, y_j, p_j, r_j, p(d)])
  C  SymmetricBiXT(tokens'_a, tokens'_b) -> pooled_a, pooled_b, pair
     arm_x = R([pooled_x, z'_x, cell, d_x, m_x, flag_x]),  m_x = observed if flag_x else clamp(mono_hat_x)
     y = min(m_a, m_b) + gate(d_a, d_b) * scale * H([arm_a + arm_b, |arm_a - arm_b|, arm_a * arm_b, pair, cell])
``p`` are frozen structural-prior predictions supplied by the caller (out-of-fold for training targets).
With an empty support set the update is exactly zero (structural prior only).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch
from .graph_encoder import GraphEncoder, GraphEncoding, masked_max, masked_mean
from .numeric_surface import boundary_gate
from .symmetric_bixt import SymmetricBiXT


@dataclass(frozen=True)
class BitTokens:
    """Active Morgan bits per drug, padded: ``bits`` [D, M] long, ``mask`` [D, M] bool."""
    bits: Tensor
    mask: Tensor

    def to(self, device) -> "BitTokens":
        return BitTokens(self.bits.to(device), self.mask.to(device))

    @staticmethod
    def from_fingerprints(fp: Tensor, pad_to: int | None = None) -> "BitTokens":
        on = [torch.nonzero(row > 0.5).flatten() for row in fp]
        if any(len(o) == 0 for o in on):
            raise ValueError("Every drug needs at least one active bit")
        m = pad_to or max(len(o) for o in on)
        bits = torch.zeros(len(on), m, dtype=torch.long); mask = torch.zeros(len(on), m, dtype=torch.bool)
        for i, o in enumerate(on):
            bits[i, :len(o)] = o; mask[i, :len(o)] = True
        return BitTokens(bits, mask)


class MorganTokenEncoder(nn.Module):
    """Fixed substructure bits as tokens (learned bit embeddings, no message passing).

    Same output contract as ``GraphEncoder`` (tokens + mean/max pooled projection) so the
    update, interaction and heads are identical between the graph and Morgan versions."""

    def __init__(self, n_bits: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_bits, hidden_dim)
        self.ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_projection = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: BitTokens) -> GraphEncoding:
        t = self.embedding(x.bits)
        t = self.output_norm(t + self.ffn(t)) * x.mask.unsqueeze(-1).to(t.dtype)
        return GraphEncoding(tokens=t, pooled=self.pool_projection(torch.cat([masked_mean(t, x.mask), masked_max(t, x.mask)], -1)))


def drug_mask(drug_input) -> Tensor:
    return drug_input.node_mask if isinstance(drug_input, GraphBatch) else drug_input.mask


def make_drug_encoder(kind: str, atom_feature_dim: int, n_bits: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    if kind == "gnn":
        return GraphEncoder(atom_feature_dim, hidden, layers, dropout)
    if kind == "morgan":
        return MorganTokenEncoder(n_bits, hidden, dropout)
    raise ValueError(kind)


class StructPrior(nn.Module):
    """Stage-1 structural monotherapy prior: encoder pooled drug + pooled cell + standardised dose.
    For ``gnn`` this is the SparseMono-v1 ``MonoPriorNet`` architecture; its encoders initialise ObsCond."""

    def __init__(self, drug_encoder: nn.Module, cell_feature_dim: int, hidden: int, layers: int, dropout: float,
                 dose_mean: float, dose_sd: float, mono_mean: float) -> None:
        super().__init__()
        self.drug_encoder = drug_encoder
        self.cell_encoder = GraphEncoder(cell_feature_dim, hidden, layers, dropout)
        self.register_buffer("dose_stats", torch.tensor([dose_mean, max(dose_sd, 1e-6)]), persistent=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden + 1), nn.Linear(2 * hidden + 1, 2 * hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.head[-1].bias, mono_mean)

    def forward(self, drug_input, cell_graphs: GraphBatch, d_idx: Tensor, c_idx: Tensor, dose: Tensor) -> Tensor:
        drugs, cells = self.drug_encoder(drug_input).pooled, self.cell_encoder(cell_graphs).pooled
        z = (dose - self.dose_stats[0]) / self.dose_stats[1]
        return self.head(torch.cat([drugs[d_idx.long()], cells[c_idx.long()], z[:, None]], 1)).squeeze(-1)


@dataclass
class ArmInputs:
    """One arm of a batch of combination points. Supports: [B, J]."""
    drug: Tensor            # long [B]
    dose: Tensor            # [B]
    mono: Tensor            # [B] observed value (0 where not observed)
    flag: Tensor            # [B] 1.0 where the arm's monotherapy at this dose is observed
    prior: Tensor           # [B] structural prior p(dose)
    sup_dose: Tensor        # [B, J]
    sup_value: Tensor       # [B, J]
    sup_prior: Tensor       # [B, J]
    sup_mask: Tensor        # bool [B, J]

    def index(self, i) -> "ArmInputs":
        return ArmInputs(*(getattr(self, f)[i] for f in self.__dataclass_fields__))

    def to(self, device) -> "ArmInputs":
        return ArmInputs(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


@dataclass(frozen=True)
class ArmState:
    z: Tensor
    z_updated: Tensor
    tokens: Tensor
    mask: Tensor
    mono_hat: Tensor


@dataclass(frozen=True)
class ObsCondOutput:
    prediction: Tensor
    reference: Tensor
    mono_hat_a: Tensor
    mono_hat_b: Tensor
    arm_a: ArmState
    arm_b: ArmState


def _mlp(i: int, h: int, o: int, dropout: float, zero_out: bool = False) -> nn.Sequential:
    net = nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))
    if zero_out:
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    return net


class ObsCondBiSurf(nn.Module):
    def __init__(self, drug_encoder: nn.Module, cell_encoder: GraphEncoder, hidden: int, heads: int, dropout: float,
                 dose_mean: float, dose_sd: float, residual_scale: float, mono_max: float = 3.0) -> None:
        super().__init__()
        H = hidden
        self.drug_encoder, self.cell_encoder = drug_encoder, cell_encoder
        self.register_buffer("dose_stats", torch.tensor([dose_mean, max(dose_sd, 1e-6)]), persistent=True)
        self.scale, self.mono_max = float(residual_scale), float(mono_max)
        self.drug_cell = _mlp(2 * H, 2 * H, H, dropout)
        self.support_encoder = _mlp(H + 4, 2 * H, H, dropout)
        self.update = nn.Linear(2 * H, 2 * H)
        self.token_update = nn.Linear(2 * H, 2 * H)
        self.cell_film = nn.Linear(H, 2 * H)
        for lin in (self.update, self.token_update, self.cell_film):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.query_support = _mlp(5, H, H, dropout)
        self.mono_head = _mlp(3 * H + 2, 2 * H, 1, dropout, zero_out=True)
        self.bixt = SymmetricBiXT(H, heads, dropout)
        self.arm_net = _mlp(3 * H + 3, 2 * H, H, dropout)
        self.interaction = nn.Sequential(nn.LayerNorm(5 * H), nn.Linear(5 * H, 2 * H), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, 1))
        nn.init.zeros_(self.interaction[-1].weight); nn.init.zeros_(self.interaction[-1].bias)

    # -- banks (support independent; the only cached quantities, keyed by drug / cell index)
    def encode_banks(self, drug_input, cell_graphs: GraphBatch) -> tuple[GraphEncoding, Tensor, Tensor]:
        enc = self.drug_encoder(drug_input)
        return enc, drug_mask(drug_input), self.cell_encoder(cell_graphs).pooled

    def _std(self, d: Tensor) -> Tensor:
        return (d - self.dose_stats[0]) / self.dose_stats[1]

    def arm_state(self, banks, cell_idx: Tensor, x: ArmInputs) -> tuple[ArmState, Tensor]:
        enc, dmask, cells = banks
        c = cells[cell_idx]
        z = self.drug_cell(torch.cat([enc.pooled[x.drug], c], -1))
        m = x.sup_mask
        has = m.any(1, keepdim=True).to(z.dtype)
        ds, ys, ps = self._std(x.sup_dose), x.sup_value, x.sup_prior
        J = m.shape[1]
        e = self.support_encoder(torch.cat([z[:, None].expand(-1, J, -1), torch.stack([ds, ys, ps, ys - ps], -1)], -1))
        s = masked_mean(e, m)
        g, b = (has * self.update(torch.cat([z, s], -1))).chunk(2, -1)
        z_up = z * (1 + g) + b
        gc, bc = self.cell_film(c).chunk(2, -1)
        gt, bt = (has * self.token_update(torch.cat([z, s], -1))).chunk(2, -1)
        tmask = dmask[x.drug]
        tokens = (enc.tokens[x.drug] * (1 + gc + gt)[:, None] + (bc + bt)[:, None]) * tmask.unsqueeze(-1).to(z.dtype)
        dq = self._std(x.dose)
        q = self.query_support(torch.stack([dq[:, None].expand(-1, J) - ds, ys, ps, ys - ps, x.prior[:, None].expand(-1, J)], -1))
        sq = masked_mean(q, m)
        mono_hat = x.prior + self.mono_head(torch.cat([z_up, c, sq, dq[:, None], x.prior[:, None]], -1)).squeeze(-1)
        return ArmState(z=z, z_updated=z_up, tokens=tokens, mask=tmask, mono_hat=mono_hat), c

    def forward(self, banks, cell_idx: Tensor, a: ArmInputs, b: ArmInputs) -> ObsCondOutput:
        sa, c = self.arm_state(banks, cell_idx, a)
        sb, _ = self.arm_state(banks, cell_idx, b)
        ma = torch.where(a.flag > 0.5, a.mono, sa.mono_hat.clamp(0.0, self.mono_max))
        mb = torch.where(b.flag > 0.5, b.mono, sb.mono_hat.clamp(0.0, self.mono_max))
        ex = self.bixt(sa.tokens, sb.tokens, sa.mask, sb.mask)
        ra = self.arm_net(torch.cat([ex.pooled_a, sa.z_updated, c, self._std(a.dose)[:, None], ma[:, None], a.flag[:, None]], -1))
        rb = self.arm_net(torch.cat([ex.pooled_b, sb.z_updated, c, self._std(b.dose)[:, None], mb[:, None], b.flag[:, None]], -1))
        h = self.interaction(torch.cat([ra + rb, (ra - rb).abs(), ra * rb, ex.pair, c], -1)).squeeze(-1)
        ref = torch.minimum(ma, mb)
        pred = ref + boundary_gate(torch.stack([a.dose, b.dose], 1)) * self.scale * h
        return ObsCondOutput(prediction=pred, reference=ref, mono_hat_a=sa.mono_hat, mono_hat_b=sb.mono_hat, arm_a=sa, arm_b=sb)


__all__ = ["BitTokens", "MorganTokenEncoder", "StructPrior", "ArmInputs", "ObsCondBiSurf", "make_drug_encoder", "drug_mask"]

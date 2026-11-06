"""AlgorithmSearch-v1 candidates built on the mono-input v2 BiSurfGNN.

Three mechanisms, each switchable, all reducing exactly to v2 at initialisation
because every new output projection is zero-initialised:

* ``pathway_readout`` (K1): a drug queries the 50 pathway tokens of the cell
  encoder with shared multi-head attention, giving a drug-specific cell readout
  r(d, c) that is added to that drug's decoder arm. The query is the drug's own
  representation (atom-token mean, no partner information), so r depends only on
  (drug, cell) and is computed once per distinct (drug, cell) key.
* ``drug_cell_drug`` (K3): atom tokens cross-attend to pathway tokens before the
  symmetric BiXT exchange, which then runs per distinct (drug A, drug B, cell)
  triple instead of per drug pair.
* ``monotherapy_auxiliary`` (K2): a training-only head predicts the observed
  monotherapy response of one drug from its own representation, a cell vector and
  the dose. Its target never enters it as an input; the runner only supervises it
  with training-role monotherapy values.

Drug exchange symmetry holds because every new module is shared by the two arms
and the decoder combines arms only through symmetric functions.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch
from .baselines import _anchored_output, _validate_observed_monotherapy
from .compact_bisurf import CompactBiSurfGNN
from .graph_encoder import masked_mean
from .surface_head import DoseFeatures


class _TokenCrossAttention(nn.Module):
    """Queries attend to masked key tokens; zero-initialised output projection."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads, self.head_dim = heads, hidden_dim // heads
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, queries: Tensor, keys: Tensor, key_mask: Tensor) -> Tensor:
        # queries [B, Q, H], keys [B, K, H], key_mask [B, K]
        b, q, h = queries.shape
        k = keys.shape[1]
        split = lambda x, n: x.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        qh = split(self.query(self.query_norm(queries)), q)
        kn = self.key_norm(keys)
        kh, vh = split(self.key(kn), k), split(self.value(kn), k)
        logits = torch.matmul(qh, kh.transpose(-2, -1)) * self.head_dim ** -0.5
        logits = logits.masked_fill(~key_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attention = self.dropout(torch.softmax(logits, dim=-1))
        merged = torch.matmul(attention, vh).transpose(1, 2).reshape(b, q, h)
        return self.output(merged)


class MonotherapyAuxiliaryHead(nn.Module):
    def __init__(self, hidden_dim: int, dose_frequencies: int, dropout: float,
                 upper_bound: float) -> None:
        super().__init__()
        self.dose_features = DoseFeatures(dose_frequencies)
        width = 2 * hidden_dim + self.dose_features.output_dim
        self.network = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1))
        self.upper_bound = float(upper_bound)

    def forward(self, drug: Tensor, cell: Tensor, dose: Tensor) -> Tensor:
        features = torch.cat([drug, cell, self.dose_features(dose)], dim=-1)
        return self.upper_bound * torch.sigmoid(self.network(features).squeeze(-1))


class MonotherapyDoseBranch(nn.Module):
    """Explicit, symmetric pair features from the observed monotherapy and dose.

    In v2 each arm's monotherapy is one scalar inside a ~280-wide LayerNorm input,
    and permuting it barely changes predictions. This branch gives the residual
    logit a direct, separately scaled route from (m_a, m_b, dose_a, dose_b).
    Arms share one encoder and are combined with symmetric operations only.
    """

    def __init__(self, hidden_dim: int, dose_frequencies: int, dropout: float,
                 upper_bound: float) -> None:
        super().__init__()
        self.dose_features = DoseFeatures(dose_frequencies)
        self.upper_bound = float(upper_bound)
        arm_width = 2 + self.dose_features.output_dim
        self.arm = nn.Sequential(nn.Linear(arm_width, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.pair = nn.Sequential(nn.Linear(3 * hidden_dim + 3, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
                                  nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _logit(self, mono: Tensor) -> Tensor:
        scaled = (mono / self.upper_bound).clamp(1e-4, 1 - 1e-4)
        return torch.log(scaled) - torch.log1p(-scaled)

    def forward(self, mono_a: Tensor, mono_b: Tensor, doses: Tensor) -> Tensor:
        ua = self.arm(torch.cat([mono_a[:, None], self._logit(mono_a)[:, None],
                                 self.dose_features(doses[:, 0])], dim=-1))
        ub = self.arm(torch.cat([mono_b[:, None], self._logit(mono_b)[:, None],
                                 self.dose_features(doses[:, 1])], dim=-1))
        pair_scalars = torch.stack([torch.minimum(mono_a, mono_b), mono_a * mono_b,
                                    torch.abs(mono_a - mono_b)], dim=-1)
        hidden = self.pair(torch.cat([ua + ub, torch.abs(ua - ub), ua * ub, pair_scalars], dim=-1))
        return self.head(hidden).squeeze(-1)


class SearchBiSurfGNN(CompactBiSurfGNN):
    def __init__(self, atom_feature_dim: int, cell_feature_dim: int, hidden_dim: int, *,
                 pathway_readout: bool = False, drug_cell_drug: bool = False,
                 monotherapy_auxiliary: bool = False, monotherapy_dose_branch: bool = False,
                 **kwargs) -> None:
        kwargs.setdefault("residual_monotherapy_input", True)
        super().__init__(atom_feature_dim, cell_feature_dim, hidden_dim, **kwargs)
        heads = self.bixt.num_heads
        dropout = self.bixt.attention_dropout.p
        self.use_pathway_readout = bool(pathway_readout)
        self.use_drug_cell_drug = bool(drug_cell_drug)
        self.use_monotherapy_auxiliary = bool(monotherapy_auxiliary)
        # Base modules are built by the parent first, so for the same seed every
        # shared v2 parameter starts from the same values as the v2 control.
        if self.use_pathway_readout or self.use_monotherapy_auxiliary:
            self.drug_self_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        if self.use_pathway_readout:
            self.pathway_readout = _TokenCrossAttention(hidden_dim, heads, dropout)
        if self.use_drug_cell_drug:
            self.atom_pathway = _TokenCrossAttention(hidden_dim, heads, dropout)
        self.use_monotherapy_dose_branch = bool(monotherapy_dose_branch)
        if self.use_monotherapy_dose_branch:
            self.monotherapy_dose_branch = MonotherapyDoseBranch(
                hidden_dim, self.interaction_decoder.dose_features.frequencies.numel(),
                dropout, self.monotherapy_upper_bound)
        if self.use_monotherapy_auxiliary:
            self.monotherapy_head = MonotherapyAuxiliaryHead(
                hidden_dim, self.interaction_decoder.dose_features.frequencies.numel(),
                dropout, self.monotherapy_upper_bound)

    # ----- banks ---------------------------------------------------------
    def encode_context(self, cell_graphs: GraphBatch):
        encoded = self.context_encoder(cell_graphs)
        return encoded.pooled, encoded.tokens, cell_graphs.node_mask

    def _drug_self(self, drug_tokens: Tensor, drug_mask: Tensor, index: Tensor) -> Tensor:
        return self.drug_self_projection(masked_mean(drug_tokens[index], drug_mask[index]))

    def _readout(self, drug_tokens, drug_mask, context_bank, drug_index, cell_index):
        """r(d, c), computed once per distinct (drug, cell) key and gathered to points."""
        pooled, tokens, mask = context_bank
        key = drug_index.to(torch.int64) * pooled.shape[0] + cell_index.to(torch.int64)
        unique, inverse = torch.unique(key, return_inverse=True)
        drugs = torch.div(unique, pooled.shape[0], rounding_mode="floor")
        cells = unique % pooled.shape[0]
        query = self._drug_self(drug_tokens, drug_mask, drugs).unsqueeze(1)
        readout = self.pathway_readout(query, tokens[cells], mask[cells]).squeeze(1)
        return readout[inverse]

    def _exchange_triples(self, drug_tokens, drug_mask, context_bank, a, b, cell):
        """K3: cell-conditioned atom tokens, then BiXT per distinct (A, B, cell)."""
        pooled, tokens, mask = context_bank
        n_drug, n_cell = drug_tokens.shape[0], pooled.shape[0]
        a, b, cell = a.to(torch.int64), b.to(torch.int64), cell.to(torch.int64)
        triple = (a * n_drug + b) * n_cell + cell
        unique, inverse = torch.unique(triple, return_inverse=True)
        uc = unique % n_cell
        ab = torch.div(unique, n_cell, rounding_mode="floor")
        ua, ub = torch.div(ab, n_drug, rounding_mode="floor"), ab % n_drug

        def conditioned(drugs: Tensor) -> tuple[Tensor, Tensor]:
            atoms, atom_mask = drug_tokens[drugs], drug_mask[drugs]
            update = self.atom_pathway(atoms, tokens[uc], mask[uc])
            return (atoms + update) * atom_mask.unsqueeze(-1).to(atoms.dtype), atom_mask

        ta, ma = conditioned(ua)
        tb, mb = conditioned(ub)
        exchange = self.bixt(ta, tb, ma, mb)
        return exchange.pair[inverse], exchange.pooled_a[inverse], exchange.pooled_b[inverse]

    # ----- prediction ----------------------------------------------------
    def forward_pooled(self, drug_bank, context_bank, drug_a_index, drug_b_index, cell_index,
                       doses, mono_a, mono_b, mono_mask_a, mono_mask_b):
        return self._forward_parts(drug_bank, context_bank, drug_a_index, drug_b_index,
                                   cell_index, doses, mono_a, mono_b, mono_mask_a,
                                   mono_mask_b)[0]

    def _forward_parts(self, drug_bank, context_bank, drug_a_index, drug_b_index, cell_index,
                       doses, mono_a, mono_b, mono_mask_a, mono_mask_b):
        _validate_observed_monotherapy(doses, mono_a, mono_b, mono_mask_a, mono_mask_b,
                                       self.monotherapy_upper_bound)
        drug_tokens, drug_mask = drug_bank
        pooled = context_bank[0]
        if drug_tokens.ndim != 3 or drug_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Drug token bank must have shape [drugs, atoms, hidden]")
        if pooled.ndim != 2 or pooled.shape[-1] != self.hidden_dim:
            raise ValueError("Context bank must start with a [cells, hidden] pooled tensor")
        if self.use_drug_cell_drug:
            pair, ex_a, ex_b = self._exchange_triples(drug_tokens, drug_mask, context_bank,
                                                      drug_a_index, drug_b_index, cell_index)
        else:
            pair, ex_a, ex_b = self.exchange_unique_pairs(drug_tokens, drug_mask,
                                                          drug_a_index, drug_b_index)
        context = pooled[cell_index.to(torch.int64)]
        cell_a = cell_b = context
        readouts = None
        if self.use_pathway_readout:
            readouts = (self._readout(drug_tokens, drug_mask, context_bank, drug_a_index, cell_index),
                        self._readout(drug_tokens, drug_mask, context_bank, drug_b_index, cell_index))
            cell_a, cell_b = context + readouts[0], context + readouts[1]
        conditioned_pair = self.context_fusion(pair, context)
        decoder = self.interaction_decoder
        arm_a = decoder._arm(ex_a, cell_a, doses[:, 0], mono_a)
        arm_b = decoder._arm(ex_b, cell_b, doses[:, 1], mono_b)
        hidden = decoder.interaction_network(torch.cat(
            [arm_a + arm_b, torch.abs(arm_a - arm_b), arm_a * arm_b, conditioned_pair, context],
            dim=-1))
        interaction_logit = decoder.residual_head(hidden).squeeze(-1)
        if self.use_monotherapy_dose_branch:
            interaction_logit = interaction_logit + self.monotherapy_dose_branch(mono_a, mono_b, doses)
        output = _anchored_output(doses, mono_a, mono_b, mono_mask_a, mono_mask_b,
                                  interaction_logit, self.reference_mode,
                                  self.monotherapy_upper_bound)
        return output, readouts

    def forward(self, drug_graphs, cell_graphs, drug_a_index, drug_b_index, cell_index,
                doses, mono_a, mono_b, mono_mask_a, mono_mask_b):
        return self.forward_pooled(self.encode_drugs(drug_graphs), self.encode_context(cell_graphs),
                                   drug_a_index, drug_b_index, cell_index, doses, mono_a,
                                   mono_b, mono_mask_a, mono_mask_b)

    def forward_with_auxiliary(self, drug_graphs, cell_graphs, drug_a_index, drug_b_index,
                               cell_index, doses, mono_a, mono_b, mono_mask_a, mono_mask_b):
        """Training path: combination output plus per-arm monotherapy predictions.

        The auxiliary head sees the drug's own representation, a cell vector and
        the dose. It never receives ``mono_a``/``mono_b`` (its targets).
        """
        drug_bank = self.encode_drugs(drug_graphs)
        context_bank = self.encode_context(cell_graphs)
        output, readouts = self._forward_parts(drug_bank, context_bank, drug_a_index, drug_b_index,
                                               cell_index, doses, mono_a, mono_b,
                                               mono_mask_a, mono_mask_b)
        if not self.use_monotherapy_auxiliary:
            return output, None
        tokens, mask = drug_bank
        context = context_bank[0][cell_index.to(torch.int64)]
        predictions = []
        for arm, drugs in enumerate((drug_a_index, drug_b_index)):
            cell = context if readouts is None else context + readouts[arm]
            predictions.append(self.monotherapy_head(
                self._drug_self(tokens, mask, drugs.to(torch.int64)), cell, doses[:, arm]))
        return output, tuple(predictions)


__all__ = ["SearchBiSurfGNN"]

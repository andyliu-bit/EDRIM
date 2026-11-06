"""D12 (card B): component ablations of BiSurfGNN v2, VALIDATION split only (split 3 is never opened).

Each arm changes one thing in the final D7 configuration ('both' input parity + D6 60-epoch cosine schedule, data,
selection rule and seeds unchanged); run_long_both.train_long_both trains it. Replacement modules are created inside a
forked RNG (seed + 7919) after the parity builder has checked the shared initialisation, so every shared parameter
and the training RNG stream equal the v2 run of the same seed.

  B1  BiXT -> pooled exchange (no cross messages; factorial_bisurf.ControlledExchange 'pooled')
  B2  BiXT -> ordinary bidirectional cross attention (ControlledExchange 'cross_attention')
  B3  gated context fusion -> concatenation MLP (factorial_bisurf.ConcatContextFusion)
  B4  pathway graph without edges: adjacency = identity (self loops only)
  B5  pathway graph, degree-preserving random rewiring (edge swaps, RNG 20261016; weights stay with edge slots)
  B6  cell encoder -> MLP on the flattened 50 x 53 Hallmark node features (no message passing)
  B7  MoLFormer only (arm 'molformer')       B8  976 genes only (arm 'genes976')
  B9  random vectors in place of both inputs (random_control.run_random.random_features)
  B10 molecular graphs without bonds: every drug adjacency = identity (atoms keep their features; MoLFormer
      injection and BiXT unchanged). Added 10-19 (DEVIATIONS.md D12 supplement), mirrors B4 for the drug encoder.

  Block-level arms (card F, docs/COARSE_ABLATION_CARD_20261022.md); same fork-RNG rule:
  B11 A     no HSA anchor: y = b + delta (b learnable scalar, init = pooled mean of the train targets); r = min(m_A, m_B)
            is an extra decoder input (first interaction layer one column wider); the dose gate g is unused
  B12 I-    BiXT -> pooled exchange (as B1) and the context-fusion output (p and its cell gating) zeroed (ZeroPair)
  B13 C-    c = 0: input arm 'molformer' (as B7, no gene projection) and pathway encoder -> ZeroContext
  B14 I-C-  B12 and B13 together

  LITE (card I, docs/OURS_LITE_CARD_20261025.md, D18): D7 with only the structure swapped. Drug graph encoder, pathway
       encoder, BiXT and cell gating are removed; each drug and each cell gets one learnable 136-d embedding, plus the
       D7 zero-initialised MoLFormer / gene projections; p = D7's pair readout of (h_A+h_B, |h_A-h_B|, h_A*h_B) and
       p_c = p. Anchor, decoder, dose arm, loss, schedule and selection are D7's; same fork-RNG rule. The graph banks
       become single-node graphs carrying the bank index (modified_banks), so the training/prediction paths are
       unchanged. load_lite() rebuilds a LITE model from its model.pt for the T batch.

  .venv/Scripts/python.exe -I -B experiments/model_improvement/ablation/run_ablation.py --arm B1 --seed 41 [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "input_parity"))
sys.path.insert(0, str(HERE.parent / "long_schedule"))
sys.path.insert(0, str(HERE.parent / "random_control"))
import run_parity as rp  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch is imported)
import run_long_both as rlb  # noqa: E402

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from torch import nn  # noqa: E402

from bisurfgnn.data.batch import GraphItem  # noqa: E402
from bisurfgnn.models.factorial_bisurf import ConcatContextFusion, ControlledExchange  # noqa: E402
from bisurfgnn.models.graph_encoder import GraphEncoding  # noqa: E402
from bisurfgnn.models.symmetric_bixt import BiXTOutput  # noqa: E402

gate, run_gate, learned = rp.gate, rlb.run_gate, rp.learned
OUT = rp.V2 / "ablation"
ARMS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "LITE")
REWIRE_SEED = 20261016


class FlatContextEncoder(nn.Module):
    """B6: MLP over the flattened node features; returns the pooled embedding only (all the model uses)."""

    def __init__(self, nodes: int, features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(nodes * features), nn.Linear(nodes * features, 2 * hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))

    def forward(self, graph):
        x = graph.node_features * graph.node_mask.unsqueeze(-1).to(graph.node_features.dtype)
        return GraphEncoding(tokens=None, pooled=self.net(x.flatten(1)))


class ZeroPair(nn.Module):
    """B12/B14: replaces the context fusion, so p and its cell gating are zero; the decoder keeps sum/|diff|/product."""

    def forward(self, pair, context):
        return torch.zeros_like(pair)


class ZeroContext(nn.Module):
    """B13/B14: c = 0 for every cell (the pathway encoder is gone; the input arm 'molformer' has no gene projection)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, graph):
        x = graph.node_features
        return GraphEncoding(tokens=None, pooled=x.new_zeros(x.shape[0], self.hidden))


# D7 initial mean norms measured by the D18 verifier: drug vectors after BiXT 10.7, cell context 3.2
LITE_DRUG_NORM, LITE_CELL_NORM = 10.7, 3.2


class BankEmbedding(nn.Module):
    """LITE: one learnable vector per bank entry, looked up by the index modified_banks put in node feature 0."""

    def __init__(self, entries: int, hidden: int, target_norm: float):
        super().__init__()
        self.table = nn.Embedding(entries, hidden)
        # D18 (registered before any LITE result): initial mean vector norm matched to the D7 representation it replaces
        nn.init.normal_(self.table.weight, std=target_norm / hidden ** 0.5)

    def forward(self, graph):
        vector = self.table(graph.node_features[:, 0, 0].long())
        return GraphEncoding(tokens=vector.unsqueeze(1), pooled=vector)


class LitePair(nn.Module):
    """LITE: no exchange; h_A, h_B are the drug vectors, p = D7's pair readout of (h_A+h_B, |h_A-h_B|, h_A*h_B)."""

    def __init__(self, pair_readout: nn.Module):
        super().__init__()
        self.pair_readout = pair_readout

    def forward(self, tokens_a, tokens_b, mask_a, mask_b):
        if tokens_a.shape[1] != 1 or tokens_b.shape[1] != 1:
            raise ValueError("LITE expects one token per drug")
        a, b = tokens_a[:, 0], tokens_b[:, 0]
        pair = self.pair_readout(torch.cat([a + b, torch.abs(a - b), a * b], dim=-1))
        return BiXTOutput(tokens_a=tokens_a, tokens_b=tokens_b, pooled_a=a, pooled_b=b, pair=pair)


class PassPair(nn.Module):
    """LITE: no cell gating, p_c = p."""

    def forward(self, pair, context):
        return pair


def decoder_with_hsa(self, drug_a, drug_b, context, conditioned_pair, doses, mono_a=None, mono_b=None):
    """B11: ContinuousInteractionDecoder.forward with r = min(m_A, m_B) appended to the interaction input."""
    arm_a = self._arm(drug_a, context, doses[:, 0], mono_a)
    arm_b = self._arm(drug_b, context, doses[:, 1], mono_b)
    hsa = torch.minimum(mono_a, mono_b).unsqueeze(-1)
    hidden = self.interaction_network(torch.cat([arm_a + arm_b, torch.abs(arm_a - arm_b), arm_a * arm_b,
                                                 conditioned_pair, context, hsa], dim=-1))
    return self.residual_head(hidden).squeeze(-1), self.log_variance_head(hidden).squeeze(-1).clamp(-8.0, 2.0)


def remove_anchor(model, bias: float):
    """B11: y = b + delta. The decoder's first interaction layer gets one extra input column (r); its other weights are
    the model's own, the new column keeps nn.Linear's default init. The residual head stays zero, so y starts at b."""
    net = model.interaction_decoder.interaction_network
    old = net[1]
    wide = nn.Linear(old.in_features + 1, old.out_features)
    with torch.no_grad():
        wide.weight[:, :-1].copy_(old.weight)
        wide.bias.copy_(old.bias)
    net[0], net[1] = nn.LayerNorm(old.in_features + 1), wide
    model.interaction_decoder.forward = types.MethodType(decoder_with_hsa, model.interaction_decoder)
    model.output_bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
    anchored = model.forward_pooled                         # the parent's pass; its r + g*delta is discarded

    def forward_pooled(self, *args, **kwargs):
        out = anchored(*args, **kwargs)
        return dataclasses.replace(out, prediction=self.output_bias + out.interaction_logit)
    model.forward_pooled = types.MethodType(forward_pooled, model)


def rewire(adjacency: torch.Tensor, seed: int) -> tuple[torch.Tensor, dict]:
    a = adjacency.clone()
    n = a.shape[0]
    iu = [(i, j) for i in range(n) for j in range(i + 1, n) if a[i, j] > 0]
    weights = [float(a[i, j]) for i, j in iu]
    edges, present = list(iu), set(iu)
    rng = np.random.default_rng(seed)
    swaps = 0
    for _ in range(100 * len(edges)):
        p, q = rng.choice(len(edges), 2, replace=False)
        (u, v), (x, y) = edges[p], edges[q]
        if rng.random() < 0.5:
            x, y = y, x
        e1, e2 = tuple(sorted((u, y))), tuple(sorted((x, v)))
        if u == y or x == v or e1 in present or e2 in present or e1 == e2:
            continue
        present -= {edges[p], edges[q]}
        present |= {e1, e2}
        edges[p], edges[q] = e1, e2
        swaps += 1
    out = torch.zeros_like(a)
    out.fill_diagonal_(1.0)
    for (i, j), w in zip(edges, weights):
        out[i, j] = out[j, i] = w
    deg_before = (a > 0).sum(1) - 1
    deg_after = (out > 0).sum(1) - 1
    if not torch.equal(deg_before, deg_after):
        raise RuntimeError("Rewiring changed node degrees")
    return out, dict(edges=len(edges), swaps=swaps, kept=len(set(edges) & set(iu)), seed=seed)


def single_node(graphs) -> tuple:
    """LITE: graph i -> one node whose features are zero except feature 0 = i; widths stay, so the D7 builder, its
    initialisation checks and RNG draws are unchanged. _unique_graph_batch then carries each entry's identity."""
    out = []
    for i, g in enumerate(graphs):
        x = torch.zeros(1, g.node_features.shape[1], dtype=g.node_features.dtype)
        x[0, 0] = i
        out.append(GraphItem(node_features=x, adjacency=torch.ones(1, 1, dtype=g.adjacency.dtype)))
    return tuple(out)


def modified_banks(banks, arm: str):
    if arm == "LITE":
        return (dataclasses.replace(banks, drug_graphs=single_node(banks.drug_graphs),
                                    cell_graphs=single_node(banks.cell_graphs)),
                dict(lite="single-node banks carrying the bank index", drugs=len(banks.drug_graphs),
                     cells=len(banks.cell_graphs)))
    if arm == "B10":
        graphs = tuple(GraphItem(node_features=g.node_features,
                                 adjacency=torch.eye(g.adjacency.shape[0], dtype=g.adjacency.dtype))
                       for g in banks.drug_graphs)
        bonds = sum(int((g.adjacency > 0).sum()) - int((g.adjacency.diagonal() > 0).sum()) for g in banks.drug_graphs)
        return dataclasses.replace(banks, drug_graphs=graphs), dict(drug_bond_entries_removed=bonds)
    if arm not in ("B4", "B5"):
        return banks, {}
    base = banks.cell_graphs[0].adjacency
    if not all(torch.equal(g.adjacency, base) for g in banks.cell_graphs):
        raise RuntimeError("Expected one shared pathway topology")
    if arm == "B4":
        new, info = torch.eye(base.shape[0], dtype=base.dtype), dict(edges=0)
    else:
        new, info = rewire(base, REWIRE_SEED)
    graphs = tuple(GraphItem(node_features=g.node_features, adjacency=new.clone()) for g in banks.cell_graphs)
    replaced = dataclasses.replace(banks, cell_graphs=graphs)
    return replaced, info


def patch_builder(arm: str, banks, tables=None):
    original = rp.build_parity_model

    def build(banks_, arm_, seed, device):
        model = original(banks_, arm_, seed, device)
        if arm in ("B1", "B2", "B3", "B6"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 7919)
                if arm in ("B1", "B2"):
                    model.bixt = ControlledExchange(model.bixt, "pooled" if arm == "B1" else "cross_attention")
                elif arm == "B3":
                    model.context_fusion = ConcatContextFusion(model.context_fusion, model.hidden_dim, 0.1)
                else:
                    nodes, feats = banks.cell_graphs[0].node_features.shape
                    model.context_encoder = FlatContextEncoder(nodes, feats, model.hidden_dim, 0.1)
            model = model.to(torch.device(device))
        elif arm in ("B11", "B12", "B13", "B14"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 7919)
                if arm == "B11":
                    remove_anchor(model, float(tables["train"].target.double().mean()))
                if arm in ("B12", "B14"):
                    model.bixt = ControlledExchange(model.bixt, "pooled")
                    model.context_fusion = ZeroPair()
                if arm in ("B13", "B14"):
                    model.context_encoder = ZeroContext(model.hidden_dim)
            model = model.to(torch.device(device))
        elif arm == "LITE":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 7919)
                model.drug_encoder = BankEmbedding(len(banks_.drug_graphs), model.hidden_dim, LITE_DRUG_NORM)
                model.context_encoder = BankEmbedding(len(banks_.cell_graphs), model.hidden_dim, LITE_CELL_NORM)
                model.bixt = LitePair(model.bixt.pair_readout)
                model.context_fusion = PassPair()
            model = model.to(torch.device(device))
        return model
    rp.build_parity_model = build


def load_lite(model_pt, seed: int, banks, device="cpu"):
    """T batch: rebuild the LITE model (input arm 'both') of `seed` from its model.pt, for inference only. `banks` are
    the unmodified run_gate.prepare banks; returns the eval-mode model and the LITE banks predict_ext must be given."""
    banks, _ = modified_banks(banks, "LITE")
    unpatched = rp.build_parity_model
    patch_builder("LITE", banks)
    try:
        model = rp.build_parity_model(banks, "both", seed, device)
    finally:
        rp.build_parity_model = unpatched
    model.load_state_dict(torch.load(model_pt, map_location=device, weights_only=True))
    return model.eval(), banks


def main():
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
        raise SystemExit("run with `python -I -B`")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, choices=(41, 42, 43, 44, 45), required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT / f"{'smoke_' if a.smoke else ''}{a.arm}_seed_{a.seed}"
    if out.exists():
        raise SystemExit(f"{out} exists")
    cfg = rp.effective_config()
    cfg["training"]["max_epochs"] = 2 if a.smoke else rlb.SCHEDULE["max_epochs"]
    cfg["schedule"] = dict(rlb.SCHEDULE, registered="DEVIATIONS.md D6 schedule; ablation D12")
    input_arm = {"B7": "molformer", "B8": "genes976", "B13": "molformer", "B14": "molformer"}.get(a.arm, "both")
    rlb.ARM = input_arm
    if a.arm == "B9":
        import run_random
        rp.load_deepsynba_features = run_random.random_features
    out.mkdir(parents=True)
    device = torch.device(cfg["device"])
    with threadpool_limits(limits=cfg["threads"]):
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frames, banks, tables = run_gate.prepare(cfg)
        features = rp.load_deepsynba_features(banks.drug_ids, banks.cell_ids, frames,
                                              feature_dir=rp.guard(rp.FEATURE_DIR), mapping_dir=rp.guard(gate.S1))
        banks, graph_info = modified_banks(banks, a.arm)
        if a.smoke:
            frames = run_gate.tiny_frames(frames, cfg["timing"]["subset_seed"] + 2)
        tables = gate.build_tables(frames, banks)
        patch_builder(a.arm, banks, tables)
        ext = rp.DeviceExternal(features, input_arm, device)
        va = tables["validation"]
        print(f"START ablation {a.arm} seed={a.seed} input_arm={input_arm} graph={graph_info}", flush=True)
        pred, detail = rlb.train_long_both(tables, banks, ext, cfg, a.seed, out, check=False,
                                           constant_initial=a.arm == "B11")
        metrics = {"arm": a.arm, "input_arm": input_arm, "seed": a.seed, "smoke": a.smoke, "graph_change": graph_info,
                   "validation_surface_macro_rmse": gate.surface_macro(va, pred),
                   "validation_pooled_rmse_percent": gate.pooled_rmse_percent(va.target.numpy(), pred),
                   **detail, "split3_rows_read": 0}
        pd.DataFrame(dict(source_row=va.source_rows, surface_id=va.surface_ids, target=va.target.numpy(),
                          prediction=pred)).to_csv(out / "validation_predictions.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
        print(f"DONE ablation {a.arm} seed={a.seed} val_pooled={metrics['validation_pooled_rmse_percent']:.4f} "
              f"best_epoch={detail['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()

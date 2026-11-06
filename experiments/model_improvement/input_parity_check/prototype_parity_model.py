"""Input-parity feasibility, step 2 (no training, no response data): prototype of the proposed injection subclass.

On synthetic inputs (CPU; real drug-graph sizes from the structure cache) this checks that the design composes with
the frozen CompactBiSurfGNN / AdditiveCompactBiSurfGNN forward path:
  * flags off -> state dict identical to AdditiveCompactBiSurfGNN for the same seed (baseline arm == gate model);
  * flags on  -> base parameters unchanged and the injected terms exactly zero at initialisation (zero-init last layer);
  * exchange symmetry (eval mode, all heads non-zero) and that permuting the external rows changes the output;
  * parameter overhead and CPU forward / forward+backward time ratios for a gate-sized batch (no optimiser step).
The subclass lives only in this script; nothing is written. src/ is imported read-only, so run with -B.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -I -B experiments/model_improvement/input_parity_check/prototype_parity_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if not (sys.flags.isolated and sys.flags.dont_write_bytecode):
    raise SystemExit("run with `python -I -B` (torch cache file; no bytecode in src/)")
ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent / "bisurfgnn"
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from bisurfgnn.data.batch import GraphItem, pad_graph_items  # noqa: E402
from bisurfgnn.models.compact_bisurf_additive import AdditiveCompactBiSurfGNN  # noqa: E402

KW = dict(graph_layers=2, attention_heads=4, dropout=0.1, monotherapy_upper_bound=3.0, residual_monotherapy_input=True)
ATOM_DIM, CELL_DIM, HIDDEN = 62, 53, 136             # structure_manifest atom_feature_dim; 50 pathways + 3
DRUG_EXT, CELL_EXT = 768, 976                         # MoLFormer width; landmark genes
STRUCTURES = MAIN / "data/processed/nci_almanac_unclipped/structures.pt"
SEED = 41


def ext_mlp(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                        nn.Linear(hidden, hidden))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ParityAdditiveCompactBiSurfGNN(AdditiveCompactBiSurfGNN):
    """Prototype: optional per-drug and per-cell external vectors added to the existing representations.

    drug: projected MoLFormer vector added to every valid atom token after the graph encoder (feeds BiXT, the
          pair embedding and both decoder arms); cell: projected expression vector added to the pooled pathway
          context (feeds the context gate, both arms and the interaction network). New modules are built after
          the parent's, so the parent's parameters and their RNG stream are unchanged.
    """

    def __init__(self, atom_feature_dim, cell_feature_dim, hidden_dim, *, drug_ext_dim=None, cell_ext_dim=None, **kw):
        super().__init__(atom_feature_dim, cell_feature_dim, hidden_dim, **kw)
        dropout = float(kw.get("dropout", 0.0))
        self.drug_ext_proj = ext_mlp(drug_ext_dim, hidden_dim, dropout) if drug_ext_dim else None
        self.cell_ext_proj = ext_mlp(cell_ext_dim, hidden_dim, dropout) if cell_ext_dim else None

    def encode_drugs(self, drug_graphs, drug_ext=None):
        tokens, mask = super().encode_drugs(drug_graphs)
        if self.drug_ext_proj is not None:
            if drug_ext is None or drug_ext.shape[0] != tokens.shape[0]:
                raise ValueError("drug_ext rows must align with the encoded drug graphs")
            tokens = tokens + self.drug_ext_proj(drug_ext).unsqueeze(1) * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, mask

    def encode_context(self, cell_graphs, cell_ext=None):
        context = super().encode_context(cell_graphs)
        if self.cell_ext_proj is not None:
            if cell_ext is None or cell_ext.shape[0] != context.shape[0]:
                raise ValueError("cell_ext rows must align with the encoded cell graphs")
            context = context + self.cell_ext_proj(cell_ext)
        return context

    def forward(self, drug_graphs, cell_graphs, drug_a_index, drug_b_index, cell_index, doses, mono_a, mono_b,
                mask_a, mask_b, drug_ext=None, cell_ext=None):
        return self.forward_pooled(self.encode_drugs(drug_graphs, drug_ext), self.encode_context(cell_graphs, cell_ext),
                                   drug_a_index, drug_b_index, cell_index, doses, mono_a, mono_b, mask_a, mask_b)


def build(seed, **flags):
    torch.manual_seed(seed)
    if flags:
        return ParityAdditiveCompactBiSurfGNN(ATOM_DIM, CELL_DIM, HIDDEN, **flags, **KW)
    return AdditiveCompactBiSurfGNN(ATOM_DIM, CELL_DIM, HIDDEN, **KW)


def same_state(a, b, keys=None):
    sa, sb = a.state_dict(), b.state_dict()
    keys = keys if keys is not None else sa.keys()
    return all(torch.equal(sa[k], sb[k]) for k in keys)


def synthetic_inputs(generator):
    cache = torch.load(STRUCTURES, map_location="cpu", weights_only=True)
    atoms = sorted(int(v["node_features"].shape[0]) for v in cache.values() if isinstance(v, dict))
    drugs = [GraphItem(node_features=torch.rand(n, ATOM_DIM, generator=generator),
                       adjacency=(torch.rand(n, n, generator=generator) < 0.1).float()) for n in atoms]
    adjacency = (torch.rand(50, 50, generator=generator) < 0.12).float()
    adjacency = ((adjacency + adjacency.T) > 0).float()
    cells = [GraphItem(node_features=torch.rand(50, CELL_DIM, generator=generator), adjacency=adjacency) for _ in range(59)]
    points = 128 * 9                                    # one training batch: 128 surfaces x 9 combination points
    pairs = torch.randint(0, len(drugs), (128, 2), generator=generator)
    a = pairs[:, 0].repeat_interleave(9)
    b = pairs[:, 1].repeat_interleave(9)
    c = torch.randint(0, 59, (128,), generator=generator).repeat_interleave(9)
    doses = torch.rand(points, 2, generator=generator) * 8.3
    mono = torch.rand(points, 2, generator=generator) * 1.3 + 0.01
    return dict(atoms=atoms, drugs=drugs, cells=cells, a=a, b=b, c=c, doses=doses, mono=mono,
                drug_ext=torch.randn(len(drugs), DRUG_EXT, generator=generator),
                cell_ext=torch.randn(59, CELL_EXT, generator=generator))


def run(model, x, *, swap=False, drug_perm=None, ext=True):
    """Unique-bank forward exactly as graph._model_forward / _unique_graph_batch, plus aligned external rows."""
    a, b = (x["b"], x["a"]) if swap else (x["a"], x["b"])
    doses = x["doses"].flip(1) if swap else x["doses"]
    ma, mb = (x["mono"][:, 1], x["mono"][:, 0]) if swap else (x["mono"][:, 0], x["mono"][:, 1])
    n = len(a)
    drug_unique, drug_inverse = torch.unique(torch.cat([a, b]), sorted=True, return_inverse=True)
    cell_unique, cell_inverse = torch.unique(x["c"], sorted=True, return_inverse=True)
    drug_graphs = pad_graph_items([x["drugs"][i] for i in drug_unique.tolist()])
    cell_graphs = pad_graph_items([x["cells"][i] for i in cell_unique.tolist()])
    masks = torch.ones(n, dtype=torch.bool)
    extra = {}
    if ext and isinstance(model, ParityAdditiveCompactBiSurfGNN):
        drug_rows = x["drug_ext"] if drug_perm is None else x["drug_ext"][drug_perm]
        extra = dict(drug_ext=drug_rows[drug_unique] if model.drug_ext_proj is not None else None,
                     cell_ext=x["cell_ext"][cell_unique] if model.cell_ext_proj is not None else None)
    return model(drug_graphs, cell_graphs, drug_inverse[:n], drug_inverse[n:], cell_inverse, doses, ma, mb,
                 masks, masks, **extra)


def one_round(model, x):
    """One timed forward (no grad) and one timed forward+backward (cost measurement only: no optimiser, no data)."""
    model.train()
    t0 = time.perf_counter()
    with torch.no_grad():
        run(model, x)
    forward = time.perf_counter() - t0
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    ((run(model, x).prediction - 0.5) ** 2).mean().backward()
    backward = time.perf_counter() - t0
    model.zero_grad(set_to_none=True)
    return forward, backward


def timing(models, x, rounds=5):
    """Warm every model up once, then interleave the rounds so allocator/thread drift hits all arms alike."""
    for m in models.values():
        one_round(m, x)
    times = {k: [] for k in models}
    for _ in range(rounds):
        for k, m in models.items():
            times[k].append(one_round(m, x))
    return {k: tuple(float(torch.tensor(v).median(dim=0).values[i]) for i in range(2)) for k, v in times.items()}


def main():
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(20261013)
    x = synthetic_inputs(g)
    base = build(SEED)
    torch.manual_seed(SEED)
    off = ParityAdditiveCompactBiSurfGNN(ATOM_DIM, CELL_DIM, HIDDEN, **KW)          # both flags off
    arms = {"drug": dict(drug_ext_dim=DRUG_EXT), "cell": dict(cell_ext_dim=CELL_EXT),
            "both": dict(drug_ext_dim=DRUG_EXT, cell_ext_dim=CELL_EXT)}
    models = {k: build(SEED, **v) for k, v in arms.items()}
    report = {"drug_graph_atoms": dict(n=len(x["atoms"]), min=x["atoms"][0], median=x["atoms"][len(x["atoms"]) // 2],
                                       max=x["atoms"][-1])}
    base_keys = list(base.state_dict().keys())
    report["flags_off_state_identical_to_gate_model"] = (list(off.state_dict().keys()) == base_keys
                                                         and same_state(base, off))
    report["base_parameters_unchanged_when_flags_on"] = {k: same_state(base, m, base_keys) for k, m in models.items()}
    params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    report["parameters"] = {"baseline": params(base), **{k: params(m) for k, m in models.items()}}

    base.eval(); [m.eval() for m in models.values()]
    with torch.no_grad():
        drug_graphs = pad_graph_items(x["drugs"]); cell_graphs = pad_graph_items(x["cells"])
        tb, _ = base.encode_drugs(drug_graphs); cb = base.encode_context(cell_graphs)
        tm, _ = models["both"].encode_drugs(drug_graphs, x["drug_ext"]); cm = models["both"].encode_context(cell_graphs, x["cell_ext"])
        report["injection_exactly_zero_at_init"] = dict(drug_tokens=bool(torch.equal(tb, tm)), context=bool(torch.equal(cb, cm)))
        report["prediction_equals_hsa_reference_at_init"] = bool(torch.equal(run(models["both"], x).prediction,
                                                                              run(base, x).reference))
    # make every head non-zero, then test symmetry and sensitivity
    torch.manual_seed(7)
    for m in models.values():
        for module in (m.interaction_decoder.residual_head, m.drug_ext_proj, m.cell_ext_proj):
            if module is None:
                continue
            last = module[-1] if isinstance(module, nn.Sequential) else module
            nn.init.normal_(last.weight, std=0.05); nn.init.normal_(last.bias, std=0.05)
    sym, sens = {}, {}
    with torch.no_grad():
        for k, m in models.items():
            p = run(m, x).prediction
            sym[k] = float((p - run(m, x, swap=True).prediction).abs().max())
            if m.drug_ext_proj is not None:
                perm = torch.randperm(len(x["drugs"]), generator=torch.Generator().manual_seed(3))
                sens[k] = float((p - run(m, x, drug_perm=perm).prediction).abs().mean())
    report["exchange_symmetry_max_abs_diff_eval"] = sym
    report["mean_abs_change_when_drug_rows_permuted"] = sens
    t = timing({"baseline": base, **models}, x)
    report["cpu_seconds_per_batch"] = {k: dict(forward=round(v[0], 4), forward_backward=round(v[1], 4)) for k, v in t.items()}
    report["cpu_time_ratio_vs_baseline"] = {k: dict(forward=round(v[0] / t["baseline"][0], 3),
                                                    forward_backward=round(v[1] / t["baseline"][1], 3))
                                            for k, v in t.items() if k != "baseline"}
    report["note"] = ("synthetic inputs; median of 5 interleaved rounds after warm-up; timing ratios only "
                      "(gate runs on GPU); no optimiser step, nothing written")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

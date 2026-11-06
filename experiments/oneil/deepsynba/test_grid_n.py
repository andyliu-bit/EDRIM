"""Unit test: with dose_grid n = 4 the modified copy (author_n/) is bit-identical to the untouched author code
(commit 43c16961, read-only in the sibling repo): same initial weights under the same seed, same state_dict, and the same
forward outputs (every output key) on the same random inputs, in eval mode and in train mode (same dropout RNG).
Also checks that n = 5 gives 5x5 outputs.

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/oneil/deepsynba/test_grid_n.py
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_deepsynba import AUTHOR, load_config  # noqa: E402  (installs the card E open guard on import)

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only author repo
import torch  # noqa: E402


def load_model_module(pkg_dir):
    """Import deepsynba.py (and its `blocks` package) from pkg_dir under fresh module names."""
    for name in [m for m in sys.modules if m in ("deepsynba", "blocks") or m.startswith("blocks.")]:
        del sys.modules[name]
    sys.path.insert(0, str(pkg_dir))
    try:
        return importlib.import_module("deepsynba")
    finally:
        sys.path.remove(str(pkg_dir))


def inputs(b, n, seed=7):
    g = torch.Generator().manual_seed(seed)
    dose = lambda: torch.cat([torch.zeros(b, 1), torch.rand(b, n - 1, generator=g) * 10], 1)  # noqa: E731
    return (torch.randn(b, 1, 768, generator=g), torch.randn(b, 1, 768, generator=g),
            torch.randn(b, 968, 1, generator=g), dose(), dose())


def main():
    cfg4 = load_config(4)
    author_mod = load_model_module(AUTHOR)
    mine_mod = load_model_module(HERE / "author_n")
    assert Path(author_mod.__file__).resolve().parent == AUTHOR.resolve()
    assert Path(mine_mod.__file__).resolve().parent == (HERE / "author_n").resolve()

    torch.manual_seed(0)
    author = author_mod.DeepSynBa(cfg4)
    torch.manual_seed(0)
    mine = mine_mod.DeepSynBa(cfg4)
    sa, sm = author.state_dict(), mine.state_dict()
    assert list(sa) == list(sm), "state_dict keys differ"
    assert all(torch.equal(sa[k], sm[k]) for k in sa), "initial weights differ under the same seed"

    # same weights (perturbed away from init so BatchNorm stats and biases are non-trivial), same inputs
    with torch.no_grad():
        for p in author.parameters():
            p.add_(torch.randn_like(p) * 0.01)
        author.drug_cell_agg[1].running_mean.normal_()
        author.drug_cell_agg[1].running_var.uniform_(0.5, 2.0)
    mine.load_state_dict(author.state_dict(), strict=True)
    x = inputs(16, 4)
    checked = []
    for mode in ("eval", "train"):
        author.train(mode == "train")
        mine.train(mode == "train")
        torch.manual_seed(123)
        out_a = author(*x)
        torch.manual_seed(123)
        out_m = mine(*x)
        assert out_a.keys() == out_m.keys()
        for k in out_a:
            assert out_a[k].dtype == out_m[k].dtype and torch.equal(out_a[k], out_m[k]), (mode, k)
        assert out_a["dose_response"].shape == (16, 4, 4)
        checked.append(f"{mode}: {len(out_a)} output keys bit-identical")
    # gradients identical too (train mode, same dropout masks)
    for m in (author, mine):
        m.zero_grad()
    torch.manual_seed(5)
    author(*x)["dose_response"].square().mean().backward()
    torch.manual_seed(5)
    mine(*x)["dose_response"].square().mean().backward()
    pa, pm = dict(author.named_parameters()), dict(mine.named_parameters())
    same = lambda a, b: (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))  # noqa: E731
    assert all(same(pa[k].grad, pm[k].grad) for k in pa), "gradients differ"  # sigma head: unused, grad None in both
    checked.append("train: all parameter gradients bit-identical")

    five = mine_mod.DeepSynBa(load_config(5)).eval()
    out5 = five(*inputs(8, 5))
    assert out5["dose_response"].shape == (8, 5, 5) and out5["bias"].shape == (8, 5, 5)
    assert five.bias_predictor1.prediction_head[3].out_features == 5
    checked.append("n=5: dose_response and bias are 5x5")
    print("PASS test_grid_n:", "; ".join(checked))


if __name__ == "__main__":
    main()

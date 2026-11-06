"""D21 (docs/COMBOFM_CARD_20261027.md): PyTorch port of TFFM (geffy/tffm 1.0.1, the PyPI release of comboFM's time).

Model, as TFFMCore.init_main_block with use_diag=False: y = b + x.w1 + sum_{k=2..order} sum_f t_k / k!, with
power sums p_m = (x^m) @ (w_k^m) and t_k = p_1^k - sum coef * prod p_in^out over tffm's powers_and_coefs(k).
Each order k has its own factor matrix w_k (n_features x rank); w_1 is n_features x 1. The coefficients below were
dumped from tffm 1.0.1 (numpy 1.18.5) in tensorflow/tensorflow:1.15.5-py3
(runs/model_improvement/combofm/tffm_check/tffm_1.0.1_powers_and_coefs.txt). For k = 2..4 they give the exact
ANOVA kernel (k! times the elementary symmetric polynomial); for k = 5 four of tffm's coefficients differ from it
(exact: +20 p1^2 p3, +15 p1 p2^2, -30 p1 p4, +24 p5; tffm: +10, +5, -15, +29), so comboFM's fifth-order term is
not the textbook ANOVA term. The port keeps tffm's values, because that is the model comboFM published.
Objective, as TFFMCore with reweight_reg=False: mean squared error + reg * sum_k mean((w_k / n_features)^2).
Init: w ~ U(-init_std, init_std), bias = init_std.
Training, as TFFMBaseModel._fit: np.random.seed(seed) once if seed, a fresh permutation every epoch, batches in
order with the last partial batch kept, Adam (TF1 defaults beta 0.9/0.999, eps 1e-8; TF1 applies eps after the
bias correction of the step size, PyTorch before it, which only matters while sqrt(v) is near 1e-8).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch import nn


# tffm 1.0.1 utils.powers_and_coefs(k): (in_pows, out_pows, coef)
TFFM_POWERS_AND_COEFS = {
    2: [((2,), (1,), 1.0)],
    3: [((1, 2), (1, 1), 3.0), ((3,), (1,), -2.0)],
    4: [((1, 2), (2, 1), 6.0), ((1, 3), (1, 1), -8.0), ((2,), (2,), -3.0), ((4,), (1,), 6.0)],
    5: [((1, 2), (3, 1), 10.0), ((1, 3), (2, 1), -10.0), ((1, 2), (1, 2), -5.0), ((1, 4), (1, 1), 15.0),
        ((2, 3), (1, 1), 20.0), ((5,), (1,), -29.0)],
}


def _order_term(k, p):
    """tffm's order-k contribution from power sums p[1..k], each (batch, rank)."""
    dot = p[1] ** k
    for in_pows, out_pows, coef in TFFM_POWERS_AND_COEFS[k]:
        prod = torch.ones_like(dot)
        for i, o in zip(in_pows, out_pows):
            prod = prod * p[i] ** o
        dot = dot - coef * prod
    return dot.sum(1) / float(math.factorial(k))


class TFFM(nn.Module):
    def __init__(self, n_features: int, order: int = 5, rank: int = 50, init_std: float = 0.01,
                 seed: int | None = None):
        super().__init__()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        def uniform(cols):
            return (torch.rand(n_features, cols, generator=gen) * 2 - 1) * init_std

        self.order = order
        self.w = nn.ParameterList([nn.Parameter(uniform(1))] + [nn.Parameter(uniform(rank)) for _ in range(order - 1)])
        self.b = nn.Parameter(torch.tensor(float(init_std)))

    def forward(self, x):
        out = self.b + (x @ self.w[0]).squeeze(1)
        x_pow = [None] + [x ** m for m in range(1, self.order + 1)]
        for k in range(2, self.order + 1):
            w = self.w[k - 1]
            p = [None] + [x_pow[m] @ w ** m for m in range(1, k + 1)]  # power sums, (batch, rank)
            out = out + _order_term(k, p)
        return out

    def forward_codes(self, onehot_idx, dense):
        """forward() for x = [one-hot block with ones at onehot_idx (batch, k), dense]; 1**m = 1, so the one-hot
        part of every power sum is a sum of gathered weight rows."""
        n_onehot = self.w[0].shape[0] - dense.shape[1]
        out = self.b + self.w[0][onehot_idx].sum((1, 2)) + (dense @ self.w[0][n_onehot:]).squeeze(1)
        d_pow = [None] + [dense ** m for m in range(1, self.order + 1)]
        for k in range(2, self.order + 1):
            w = self.w[k - 1]
            rows = w[onehot_idx]  # (batch, k_onehot, rank)
            p = [None] + [(rows ** m).sum(1) + d_pow[m] @ w[n_onehot:] ** m for m in range(1, k + 1)]
            out = out + _order_term(k, p)
        return out

    def penalty(self):
        n = self.w[0].shape[0]
        return sum(((w / n) ** 2).mean() for w in self.w)


def fit(model, make_batch, n_rows, n_epochs, batch_size=1024, lr=1e-3, reg=0.0, seed=None, on_epoch=None,
        forward=None):
    """make_batch(row_indices on device) -> (inputs tuple, y); forward(*inputs) defaults to model(...)."""
    forward = forward or model
    if seed:  # TFFM seeds numpy only when seed is truthy
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    device = next(model.parameters()).device
    for epoch in range(1, n_epochs + 1):
        perm = torch.from_numpy(np.random.permutation(n_rows)).to(device)  # one copy per epoch, sliced on device
        model.train()
        for i in range(0, n_rows, batch_size):
            inputs, y = make_batch(perm[i:i + batch_size])
            loss = ((y - forward(*inputs)) ** 2).mean() + reg * model.penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if on_epoch is not None:
            on_epoch(epoch)


@torch.no_grad()
def predict(model, make_batch, n_rows, batch_size=65536, forward=None):
    forward = forward or model
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([forward(*make_batch(torch.arange(i, min(i + batch_size, n_rows), device=device))[0])
                      for i in range(0, n_rows, batch_size)]).cpu().numpy()


def _self_check():
    """Orders 2-4 equal the brute-force ANOVA sum (tffm's order 5 does not; it is checked against tffm itself by
    tffm_check/check_vs_tffm.py); forward_codes equals forward."""
    torch.manual_seed(0)
    n, rank, order = 7, 3, 4
    model = TFFM(n, order=order, rank=rank, init_std=0.5, seed=1).double()
    x = torch.randn(4, n, dtype=torch.float64)
    brute = model.b + (x @ model.w[0]).squeeze(1)
    for k in range(2, order + 1):
        for idx in itertools.combinations(range(n), k):
            brute = brute + torch.stack([x[:, i:i + 1] * model.w[k - 1][i] for i in idx]).prod(0).sum(1)
    assert torch.allclose(model(x), brute, rtol=1e-10, atol=1e-10), (model(x), brute)
    idx = torch.tensor([[0, 3], [1, 2], [0, 2], [3, 1]])  # one-hot block = first 4 features, two ones per row
    dense = torch.randn(4, n - 4, dtype=torch.float64)
    x_full = torch.cat([torch.zeros(4, 4, dtype=torch.float64).scatter_(1, idx, 1.0), dense], 1)
    assert torch.allclose(model.forward_codes(idx, dense), model(x_full), rtol=1e-10, atol=1e-10)
    model5 = TFFM(n, order=5, rank=rank, init_std=0.5, seed=1).double()
    assert torch.allclose(model5.forward_codes(idx, dense), model5(x_full), rtol=1e-10, atol=1e-10)
    print("tffm_torch self-check passed")


if __name__ == "__main__":
    _self_check()

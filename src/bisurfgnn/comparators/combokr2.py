"""Small-data adapter around comboKR 2.0's MIT-licensed author implementation.

The dense MINRES backend replaces RLScore's compiled sampled-vec operator,
retaining its kernel equation, zero regularisation and iteration stopping.
It has quadratic memory and is deliberately capped; it is not a scalable
replacement for RLScore. Original author files are not modified.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import minres

from .ddprism import verify_source


class DenseMINRES:
    def __init__(self, *, K1, K2, Y, label_row_inds, label_col_inds,
                 regparam=0.0, maxiter=5):
        self.rows = np.asarray(label_row_inds).reshape(-1)
        self.cols = np.asarray(label_col_inds).reshape(-1)
        if len(self.rows) > 1000:
            raise ValueError("Dense pilot backend is limited to 1000 training surfaces")
        matrix = K1[np.ix_(self.rows, self.rows)] * K2[np.ix_(self.cols, self.cols)]
        matrix = matrix + np.eye(len(self.rows)) * regparam
        self.A, self.info = minres(matrix, np.asarray(Y).reshape(-1),
                                  maxiter=maxiter, rtol=1e-20)
        if not np.isfinite(self.A).all():
            raise RuntimeError("Non-finite dense MINRES weights")

    def predict(self, K1, K2, rows, cols):
        rows, cols = np.asarray(rows).reshape(-1), np.asarray(cols).reshape(-1)
        matrix = K1[np.ix_(rows, self.rows)] * K2[np.ix_(cols, self.cols)]
        return matrix @ self.A


def load_author_module(root: Path):
    for filename in ("comboKR2/__init__.py", "comboKR2/helpers.py",
                     "comboKR2/braid_surface_model_for_fit.py", "comboKR2/scalable_comboKR.py"):
        verify_source(root, "combokr2", filename)
    directory = root / "data/raw/model_comparators/combokr2/comboKR2"
    name = "bisurf_combokr2_author"
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                 submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    stub, learner = types.ModuleType("rlscore"), types.ModuleType("rlscore.learner")
    learner.CGKronRLS = DenseMINRES
    with patch.dict(sys.modules, {"rlscore": stub, "rlscore.learner": learner}):
        spec.loader.exec_module(package)
        module = importlib.import_module(name + ".scalable_comboKR")
    return module


def bind_small_kernel_backend(model, max_training_surfaces=1000):
    """Avoid materialising the full cell x drug Kronecker feature universe."""
    cache = {}
    def predict_raw(instance, cdd, difference=False):
        del difference
        lookup_c = {str(v): i for i, v in enumerate(instance.cell_ids)}
        lookup_d = {str(v): i for i, v in enumerate(instance.drug_ids)}

        def index(triplets):
            return tuple(np.array([lookup[str(row[col])] for row in triplets])
                         for col, lookup in enumerate((lookup_c, lookup_d, lookup_d)))

        tr, va = index(instance.cdd_tr), index(cdd)
        if len(tr[0]) > max_training_surfaces:
            raise ValueError("Dense backend cannot launch formal-scale training")
        def kernel(left, right):
            return (instance.Kc[np.ix_(left[0], right[0])]
                    * instance.Kd[np.ix_(left[1], right[1])]
                    * instance.Kd[np.ix_(left[2], right[2])])
        # Retraining replaces Phiy_tr; never reuse weights from a prior fit.
        identity = (id(instance.Phiy_tr), id(instance.cdd_tr), instance.maxiter)
        if cache.get("identity") != identity:
            train_k = kernel(tr, tr)
            weights, infos = [], []
            for column in instance.Phiy_tr.T:
                weight, info = minres(train_k, column, maxiter=instance.maxiter, rtol=1e-20)
                if info < 0 or not np.isfinite(weight).all():
                    raise RuntimeError("MINRES failed")
                weights.append(weight); infos.append(int(info))
            cache.update(identity=identity, weights=np.column_stack(weights))
            instance.dense_fit_diagnostics = {"training_surfaces": len(tr[0]),
                "maxiter": instance.maxiter, "solver_info": infos}
        result = kernel(va, tr) @ cache["weights"]
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite comboKR output features")
        return result

    model._predict_raw = types.MethodType(predict_raw, model)

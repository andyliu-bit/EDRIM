"""NCI60-Pretrain-v1 data utilities (docs/NCI60_PRETRAIN_V1_PROTOCOL.md).

* streaming aggregation of the DTP DOSERESP table (read through ``unzip -p``; the
  2.5 GB CSV is never written to disk);
* NCI-60 cell-label mapping with the FROZEN rules of ``data/mappings/nci60_cell_aliases.json``
  (lenient copy of ``bisurfgnn.data.cellminer.build_strict_cell_mapping``: an unmappable label is
  dropped and counted instead of raising, because the NCI-60 history contains cell lines that are
  not part of the frozen CellMiner/ALMANAC cohort);
* model dose scale u = log10(1 + D / 1e-12 M), identical to ``bisurfgnn.data.nci_almanac``;
* RDKit structure standardisation with the ALMANAC policy and the atom-v1 graph featuriser
  (``bisurfgnn.data.chemistry``), stored as a compact CSR graph bank.

Raw downloads are untrusted data: nothing from the raw directory is ever executed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

DOSERESP_COLUMNS = ["RELEASE_DATE", "EXPID", "PREFIX", "NSC", "CONCENTRATION_UNIT", "LOG_HI_CONCENTRATION",
                    "CONCENTRATION", "PANEL_NUMBER", "CELL_NUMBER", "PANEL_NAME", "CELL_NAME", "PANEL_CODE",
                    "COUNT_GIPRCNT", "AVERAGE_GIPRCNT", "STDDEV_GIPRCNT", "COUNT_PTC", "AVERAGE_PTC", "STDDEV_PTC"]
USECOLS = ["RELEASE_DATE", "EXPID", "PREFIX", "NSC", "CONCENTRATION_UNIT", "LOG_HI_CONCENTRATION", "CONCENTRATION",
           "PANEL_NUMBER", "CELL_NUMBER", "PANEL_NAME", "CELL_NAME", "COUNT_GIPRCNT", "AVERAGE_GIPRCNT",
           "COUNT_PTC", "AVERAGE_PTC"]
DOSE_REFERENCE_MOLAR = 1e-12          # bisurfgnn.data.nci_almanac DoseTransform.model_reference_molar
CONC_SCALE = 10_000                    # CONCENTRATION is reported with 4 decimals (log10 units)
CONC_OFFSET = 1 << 19
PTC_EDGES = np.arange(-200.0, 1000.0 + 1e-9, 5.0)
GI_EDGES = np.arange(-200.0, 400.0 + 1e-9, 5.0)


DATA_ROOT = Path("C:/LHZ/research/bisurfgnn")       # read-only main checkout (same as SparseMono data_root)
BENCHMARK_CONFIG = "configs/benchmark_bisurf_unclipped.yaml"
CELL_CONFIG = "configs/cell_context.yaml"
FORBIDDEN_PARQUET = {"test.parquet", "points.parquet"}


class forbid_test_responses:
    """Same guard as the SparseMono runners: any read of test.parquet / ALMANAC points.parquet raises."""

    def __init__(self) -> None:
        self.accesses: list[str] = []

    def __enter__(self):
        from unittest.mock import patch
        original = pd.read_parquet

        def guard(path, *a, **kw):
            resolved = Path(path).resolve()
            if resolved.name.lower() in FORBIDDEN_PARQUET:
                raise RuntimeError(f"Forbidden response source: {resolved}")
            self.accesses.append(str(resolved)); return original(path, *a, **kw)
        self._patch = patch.object(pd, "read_parquet", guard); self._patch.__enter__(); return self

    def __exit__(self, *exc):
        return self._patch.__exit__(*exc)


def load_fold0_cell_graphs(data_root: Path = DATA_ROOT):
    """Fold-0 pathway graphs keyed by CellMiner biological ID (the graphs SparseMono's banks use)."""
    from bisurfgnn.benchmark_config import load_benchmark_config
    from bisurfgnn.data.benchmark_context import load_benchmark_cell_context
    from bisurfgnn.data.cellminer import load_cellminer_config
    bench = load_benchmark_config(data_root / BENCHMARK_CONFIG)
    ctx = load_benchmark_cell_context(bench, load_cellminer_config(data_root / CELL_CONFIG))
    return dict(ctx.folds[0].graphs), ctx, bench


def append_log(path: str | Path, text: str) -> None:
    """Append a timestamped entry to the human-readable step log (never rewrites earlier entries)."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"\n({stamp}) {text.rstrip()}\n")


def sha256_file(path: str | Path, block: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def dose_model(log10_molar) -> np.ndarray:
    """u = log10(1 + D / 1e-12 M) with D = 10**log10_molar (same as nci_almanac ``dose_*_model``)."""
    return np.log10(1.0 + np.power(10.0, np.asarray(log10_molar, dtype=np.float64)) / DOSE_REFERENCE_MOLAR)


# ---------------------------------------------------------------- DOSERESP streaming aggregation
class _Codes:
    """Global string -> int code book filled chunk by chunk."""

    def __init__(self) -> None:
        self.index: dict[str, int] = {}

    def lut(self, categories) -> np.ndarray:
        return np.array([self.index.setdefault(str(c), len(self.index)) for c in categories] or [0], dtype=np.int64)

    def map(self, series: pd.Series) -> np.ndarray:
        s = series.astype("category")
        codes = s.cat.codes.to_numpy().astype(np.int64)
        lut = self.lut(s.cat.categories)
        out = np.where(codes >= 0, lut[np.clip(codes, 0, None)], self.index.setdefault("<NA>", len(self.index)))
        return out

    def names(self) -> list[str]:
        return [k for k, _ in sorted(self.index.items(), key=lambda kv: kv[1])]


def _pack(prefix, unit, cell, nsc, conc_int) -> np.ndarray:
    if len(prefix) and (prefix.max() >= 16 or unit.max() >= 16 or cell.max() >= 1024):
        raise RuntimeError("DOSERESP code book exceeds the packed-key layout")
    return ((((prefix.astype(np.int64) << 4 | unit.astype(np.int64)) << 10 | cell.astype(np.int64)) << 21
             | nsc.astype(np.int64)) << 20) | (conc_int.astype(np.int64) + CONC_OFFSET)


def _unpack(key: np.ndarray):
    conc_int = (key & ((1 << 20) - 1)) - CONC_OFFSET
    nsc = (key >> 20) & ((1 << 21) - 1)
    cell = (key >> 41) & ((1 << 10) - 1)
    unit = (key >> 51) & 0xF
    prefix = (key >> 55) & 0xF
    return prefix, unit, cell, nsc, conc_int


def stream_doseresp_aggregate(zip_path: Path, unzip_exe: str, member: str = "DOSERESP.csv",
                              chunksize: int = 1_000_000, cwd: Path | None = None, log=print,
                              max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One pass over ``unzip -p``; returns (aggregate, cell-label combos, stats).

    Aggregate key = (PREFIX, NSC, CONCENTRATION_UNIT, cell-label combo, round(CONCENTRATION, 4)); values are
    sums of COUNT_PTC, COUNT_PTC*AVERAGE_PTC, COUNT_PTC*AVERAGE_PTC**2 and the number of contributing rows
    (one row = one experiment ID), so that later re-aggregation after cell mapping is exact."""
    t0 = time.monotonic()
    proc = subprocess.Popen([unzip_exe, "-p", str(zip_path), member], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=str(cwd) if cwd else None)
    prefix_c, unit_c, pname_c, cname_c, exp_c = _Codes(), _Codes(), _Codes(), _Codes(), _Codes()
    combo_index: dict[tuple, int] = {}
    combo_rows: dict[int, int] = {}
    parts, curve_parts = [], []
    st: dict[str, Any] = dict(rows=0, chunks=0, nsc_non_numeric=0, conc_nan=0, conc_out_of_pack_range=0,
                              ptc_invalid_rows=0, ptc_count_zero_or_nan=0, ptc_value_nan=0, max_conc_rounding_error=0.0,
                              release_date_min=None, release_date_max=None, nsc_over_pack_range=0)
    by_unit: dict[str, dict] = {}
    ptc_hist = np.zeros(len(PTC_EDGES) + 1, dtype=np.int64)
    gi_hist = np.zeros(len(GI_EDGES) + 1, dtype=np.int64)
    dtypes = {"EXPID": "category", "PREFIX": "category", "NSC": "string", "CONCENTRATION_UNIT": "category",
              "PANEL_NAME": "category", "CELL_NAME": "category", "LOG_HI_CONCENTRATION": "float64",
              "CONCENTRATION": "float64", "PANEL_NUMBER": "float64", "CELL_NUMBER": "float64",
              "COUNT_GIPRCNT": "float64", "AVERAGE_GIPRCNT": "float64", "COUNT_PTC": "float64",
              "AVERAGE_PTC": "float64", "RELEASE_DATE": "float64"}
    reader = pd.read_csv(proc.stdout, chunksize=chunksize, usecols=USECOLS, dtype=dtypes, encoding="latin-1",
                         keep_default_na=True)
    header_checked = False
    for df in reader:
        if not header_checked:
            missing = sorted(set(USECOLS) - set(df.columns))
            if missing:
                raise RuntimeError(f"DOSERESP columns missing: {missing}")
            header_checked = True
        n = len(df); st["rows"] += n; st["chunks"] += 1
        rd = df.RELEASE_DATE.dropna()
        if len(rd):
            st["release_date_min"] = float(rd.min()) if st["release_date_min"] is None else min(st["release_date_min"], float(rd.min()))
            st["release_date_max"] = float(rd.max()) if st["release_date_max"] is None else max(st["release_date_max"], float(rd.max()))
        prefix = prefix_c.map(df.PREFIX); unit = unit_c.map(df.CONCENTRATION_UNIT)
        pn, cn = pname_c.map(df.PANEL_NAME), cname_c.map(df.CELL_NAME)
        pnum = df.PANEL_NUMBER.fillna(-1).to_numpy().astype(np.int64); cnum = df.CELL_NUMBER.fillna(-1).to_numpy().astype(np.int64)
        if cn.max() >= 4096 or pnum.max() >= 998 or cnum.max() >= 998:
            raise RuntimeError("Cell label code book exceeds the packed layout")
        combo_raw = ((pn * 4096 + cn) * 1000 + (pnum + 1)) * 1000 + (cnum + 1)
        uniq, inv = np.unique(combo_raw, return_inverse=True)
        lut = np.empty(len(uniq), dtype=np.int64)
        for i, v in enumerate(uniq):
            v = int(v); c1 = v % 1000 - 1; v //= 1000; p1 = v % 1000 - 1; v //= 1000
            key = (int(v // 4096), int(v % 4096), int(p1), int(c1))
            lut[i] = combo_index.setdefault(key, len(combo_index))
        cell = lut[inv]
        for code, cnt in zip(*np.unique(cell, return_counts=True)):
            combo_rows[int(code)] = combo_rows.get(int(code), 0) + int(cnt)
        nsc = pd.to_numeric(df.NSC, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        conc = df.CONCENTRATION.to_numpy(); hi = df.LOG_HI_CONCENTRATION.to_numpy()
        cnt_ptc = df.COUNT_PTC.to_numpy(); ptc = df.AVERAGE_PTC.to_numpy()
        gi = df.AVERAGE_GIPRCNT.to_numpy()
        bad_nsc = ~np.isfinite(nsc); st["nsc_non_numeric"] += int(bad_nsc.sum())
        over = np.isfinite(nsc) & ((nsc < 0) | (nsc >= (1 << 21))); st["nsc_over_pack_range"] += int(over.sum())
        cnan = ~np.isfinite(conc); st["conc_nan"] += int(cnan.sum())
        conc_int = np.where(cnan, 0, np.round(np.nan_to_num(conc) * CONC_SCALE)).astype(np.int64)
        oor = (conc_int < -CONC_OFFSET) | (conc_int >= CONC_OFFSET); st["conc_out_of_pack_range"] += int(oor.sum())
        if (~cnan).any():
            st["max_conc_rounding_error"] = max(st["max_conc_rounding_error"],
                                                float(np.max(np.abs(conc[~cnan] - conc_int[~cnan] / CONC_SCALE))))
        cz = ~(np.isfinite(cnt_ptc) & (cnt_ptc > 0)); vn = ~np.isfinite(ptc)
        st["ptc_count_zero_or_nan"] += int(cz.sum()); st["ptc_value_nan"] += int(vn.sum())
        valid_ptc = ~(cz | vn); st["ptc_invalid_rows"] += int((~valid_ptc).sum())
        ptc_hist += np.bincount(np.searchsorted(PTC_EDGES, ptc[valid_ptc], side="right"), minlength=len(PTC_EDGES) + 1)
        gv = np.isfinite(gi)
        gi_hist += np.bincount(np.searchsorted(GI_EDGES, gi[gv], side="right"), minlength=len(GI_EDGES) + 1)
        unit_names = unit_c.names()
        for u in np.unique(unit):
            sel = unit == u; name = unit_names[int(u)]
            d = by_unit.setdefault(name, dict(rows=0, conc_min=np.inf, conc_max=-np.inf, hi_min=np.inf, hi_max=-np.inf,
                                              conc_above_hi=0, valid_ptc_rows=0))
            d["rows"] += int(sel.sum()); d["valid_ptc_rows"] += int((sel & valid_ptc).sum())
            cs, hs = conc[sel & ~cnan], hi[sel & np.isfinite(hi)]
            if len(cs):
                d["conc_min"] = min(d["conc_min"], float(cs.min())); d["conc_max"] = max(d["conc_max"], float(cs.max()))
            if len(hs):
                d["hi_min"] = min(d["hi_min"], float(hs.min())); d["hi_max"] = max(d["hi_max"], float(hs.max()))
            both = sel & ~cnan & np.isfinite(hi)
            d["conc_above_hi"] += int((conc[both] > hi[both] + 1e-4).sum())
        keep = valid_ptc & ~bad_nsc & ~over & ~cnan & ~oor
        nk = nsc[keep].astype(np.int64)
        key = _pack(prefix[keep], unit[keep], cell[keep], nk, conc_int[keep])
        w = cnt_ptc[keep]; y = ptc[keep]
        uk, kinv = np.unique(key, return_inverse=True)
        parts.append((uk, np.bincount(kinv, weights=w), np.bincount(kinv, weights=w * y),
                      np.bincount(kinv, weights=w * y * y), np.bincount(kinv).astype(np.int64)))
        expc = exp_c.map(df.EXPID)
        ok = ~bad_nsc & ~over
        curve = (((expc[ok] << 4 | prefix[ok]) << 21 | nsc[ok].astype(np.int64)) << 10) | cell[ok]
        curve_parts.append(np.unique(curve))
        if st["chunks"] % 5 == 0:
            log(f"  chunk {st['chunks']} rows {st['rows']:,} elapsed {time.monotonic() - t0:.0f}s")
        if max_rows is not None and st["rows"] >= max_rows:   # smoke tests only
            proc.kill(); break
    rc = proc.wait(); err = proc.stderr.read().decode("latin-1", "replace")
    if rc != 0 and max_rows is None:
        raise RuntimeError(f"unzip failed rc={rc}: {err[:500]}")
    st["smoke_max_rows"] = max_rows
    st["unzip_stderr"] = err.strip()[:500]
    keys = np.concatenate([p[0] for p in parts])
    uk, inv = np.unique(keys, return_inverse=True)
    agg = {name: np.bincount(inv, weights=np.concatenate([p[i] for p in parts]))
           for i, name in ((1, "w_sum"), (2, "wy_sum"), (3, "wyy_sum"))}
    n_rows = np.bincount(inv, weights=np.concatenate([p[4] for p in parts])).astype(np.int64)
    prefix, unit, cell, nsc, conc_int = _unpack(uk)
    pnames, unames = prefix_c.names(), unit_c.names()
    out = pd.DataFrame({"prefix": pd.Categorical.from_codes(prefix, pnames), "unit": pd.Categorical.from_codes(unit, unames),
                        "nsc": nsc.astype(np.int32), "cell_code": cell.astype(np.int16),
                        "log10_conc": conc_int / CONC_SCALE, "w_sum": agg["w_sum"], "wy_sum": agg["wy_sum"],
                        "wyy_sum": agg["wyy_sum"], "n_rows": n_rows.astype(np.int32)})
    pn_names, cn_names = pname_c.names(), cname_c.names()
    combos = pd.DataFrame([dict(cell_code=code, panel_name=pn_names[k[0]], cell_name=cn_names[k[1]], panel_number=k[2],
                                cell_number=k[3], raw_rows=combo_rows.get(code, 0))
                           for k, code in sorted(combo_index.items(), key=lambda kv: kv[1])])
    curves = np.unique(np.concatenate(curve_parts))
    cprefix = (curves >> 31) & 0xF
    st["curves_expid_nsc_cell_all_prefixes"] = int(len(curves))
    st["curves_by_prefix"] = {pnames[int(p)]: int(c) for p, c in zip(*np.unique(cprefix, return_counts=True))}
    st["distinct_expid"] = len([k for k in exp_c.index if k != "<NA>"])
    st["prefix_codes"], st["unit_codes"] = pnames, unames
    st["by_unit"] = by_unit
    st["ptc_histogram"] = dict(edges=PTC_EDGES.tolist(), counts=ptc_hist.tolist())
    st["giprcnt_histogram"] = dict(edges=GI_EDGES.tolist(), counts=gi_hist.tolist())
    st["aggregate_keys"] = int(len(out)); st["seconds"] = time.monotonic() - t0
    return out, combos, st


# ---------------------------------------------------------------- cell mapping (frozen rules, lenient)
def map_cell_labels(combos: pd.DataFrame, alias_json: Path, graph_cells: set[str]) -> pd.DataFrame:
    """Apply the frozen NCI-60 label rules. Copy of the rule order in
    ``cellminer.build_strict_cell_mapping`` (exclusion -> known source alias group -> explicit alias ->
    panel prefix exact), but an unmapped label becomes ``status != mapped`` instead of raising.
    A mapped label additionally needs a fold-0 cell graph (``graph_cells``)."""
    from bisurfgnn.data.cellminer import load_cell_alias_rules, normalize_cell_label
    rules = load_cell_alias_rules(alias_json)
    prefixes = {normalize_cell_label(p): normalize_cell_label(v) for p, v in rules["panel_prefixes"].items()}
    aliases = {normalize_cell_label(k): normalize_cell_label(v) for k, v in rules["aliases"].items()}
    excluded = {normalize_cell_label(k): v for k, v in rules["excluded_almanac_cells"].items()}
    groups = {}
    for g in rules["known_source_alias_groups"]:
        names = {normalize_cell_label(x) for x in g["source_labels"]}
        if len(names) != 1:
            raise ValueError("Known alias group does not collapse")
        groups[next(iter(names))] = normalize_cell_label(g["biological_cell_id"])
    rows = []
    for r in combos.itertuples(index=False):
        try:
            name = normalize_cell_label(r.cell_name)
        except Exception:
            rows.append(dict(cell_code=r.cell_code, normalized_name="", biological_cell_id="", status="unmapped_empty_label",
                             method="")); continue
        panel = normalize_cell_label(r.panel_name) if isinstance(r.panel_name, str) and r.panel_name.strip() else ""
        if name in excluded:
            rows.append(dict(cell_code=r.cell_code, normalized_name=name, biological_cell_id=normalize_cell_label(
                excluded[name]["biological_cell_id"]), status="excluded_frozen", method="explicit_exclusion")); continue
        if panel not in prefixes:
            rows.append(dict(cell_code=r.cell_code, normalized_name=name, biological_cell_id="", status="unmapped_panel",
                             method="")); continue
        if name in groups:
            target, method = groups[name], "known_source_alias_group"
        elif name in aliases:
            target, method = aliases[name], "explicit_alias"
        else:
            target, method = f"{prefixes[panel]}:{name}", "panel_prefix_exact"
        target = normalize_cell_label(target)
        if target.partition(":")[0] != prefixes[panel]:
            status = "unmapped_panel_target_mismatch"
        elif target not in graph_cells:
            status = "unmapped_no_fold0_graph"
        else:
            status = "mapped"
        rows.append(dict(cell_code=r.cell_code, normalized_name=name, biological_cell_id=target,  # attempted target if unmapped
                         status=status, method=method))
    return combos.merge(pd.DataFrame(rows), on="cell_code", how="left", validate="one_to_one")


# ---------------------------------------------------------------- structures
ALMANAC_POLICY = dict(reject_ambiguous_multicomponent=False, canonicalize_tautomer=True, preserve_formal_charge=True,
                      max_heavy_atoms=256)   # data/processed/nci_almanac_unclipped/structure_manifest.json


def standardize_batch(items: list[tuple[int, str]]) -> list[dict]:
    """Worker: standardise + featurise (atom-v1) a batch of (nsc, smiles). Returns plain python/numpy records."""
    import warnings
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from bisurfgnn.data.chemistry import StandardizationPolicy, standardize_smiles, standardized_smiles_to_graph
    policy = StandardizationPolicy(**ALMANAC_POLICY)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for nsc, smi in items:
        rec: dict[str, Any] = dict(nsc=int(nsc), raw_smiles=smi)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = standardize_smiles(smi, policy)
                g = standardized_smiles_to_graph(s.canonical_smiles)
            mol = Chem.MolFromSmiles(s.canonical_smiles)
            fp = gen.GetFingerprint(mol)
            adj = g.adjacency.numpy()
            iu = np.argwhere(np.triu(adj, 1) > 0).astype(np.int16)
            # identity rule copied from nci_almanac.py (drug table): connectivity key, else SMILES-hash fallback
            identity = s.connectivity_key or ("SMILES-" + hashlib.sha256(s.canonical_smiles.encode()).hexdigest()[:14])
            rec.update(ok=True, canonical_smiles=s.canonical_smiles, identity_group=identity,
                       murcko_scaffold=s.murcko_scaffold, inchikey=s.inchikey, flags=json.dumps(list(s.flags)),
                       n_atoms=int(g.node_features.shape[0]), node_features=g.node_features.numpy().astype(np.float32),
                       edges=iu, fp=DataStructs.BitVectToBinaryText(fp),
                       diag_ok=bool(np.all(np.diag(adj) == 1.0)), sym_ok=bool(np.array_equal(adj, adj.T)),
                       binary_ok=bool(np.isin(adj, (0.0, 1.0)).all()))
        except Exception as error:  # dropped and counted
            rec.update(ok=False, error_type=type(error).__name__, error=str(error)[:200])
        out.append(rec)
    return out


class GraphBank:
    """CSR molecular graph bank: atom-feature vocabulary [V, F] + per-atom vocabulary index, undirected bond
    list (i < j, local atom indices) per molecule.

    ``batch(indices)`` reproduces ``bisurfgnn.data.batch.pad_graph_items`` of the atom-v1 graphs
    (adjacency = I + bonds, as ``chemistry.standardized_smiles_to_graph``) directly on the device."""

    def __init__(self, atom_vocab, atom_index, atom_ptr, edges, edge_ptr, device="cpu"):
        import torch
        self.vocab = torch.as_tensor(atom_vocab, dtype=torch.float32).to(device)
        self.atom_index = torch.as_tensor(atom_index).to(device=device, dtype=torch.int64)
        self.atom_ptr = torch.as_tensor(atom_ptr).to(device=device, dtype=torch.int64)
        self.edges = torch.as_tensor(edges).to(device=device, dtype=torch.int64)
        self.edge_ptr = torch.as_tensor(edge_ptr).to(device=device, dtype=torch.int64)
        self.n_atoms = self.atom_ptr[1:] - self.atom_ptr[:-1]
        self.n_edges = self.edge_ptr[1:] - self.edge_ptr[:-1]
        self.device = device

    def __len__(self) -> int:
        return int(len(self.atom_ptr) - 1)

    @classmethod
    def load(cls, path, device="cpu"):
        import torch
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(d["atom_vocab"], d["atom_index"], d["atom_ptr"], d["edges"], d["edge_ptr"], device=device), d

    def item(self, i: int):
        """Single GraphItem (CPU) for equality checks against ``pad_graph_items``."""
        from bisurfgnn.data.batch import GraphItem
        import torch
        a0, a1 = int(self.atom_ptr[i]), int(self.atom_ptr[i + 1]); e0, e1 = int(self.edge_ptr[i]), int(self.edge_ptr[i + 1])
        x = self.vocab[self.atom_index[a0:a1]].cpu(); n = a1 - a0
        adj = torch.eye(n); e = self.edges[e0:e1].cpu()
        if len(e):
            adj[e[:, 0], e[:, 1]] = 1.0; adj[e[:, 1], e[:, 0]] = 1.0
        return GraphItem(node_features=x, adjacency=adj)

    def batch(self, idx):
        import torch
        from bisurfgnn.data.batch import GraphBatch
        idx = torch.as_tensor(idx, dtype=torch.int64, device=self.device)
        b = len(idx); na = self.n_atoms[idx]; ne = self.n_edges[idx]
        nmax = int(na.max())
        bid = torch.repeat_interleave(torch.arange(b, device=self.device), na)
        start = torch.cumsum(na, 0) - na
        local = torch.arange(int(na.sum()), device=self.device) - torch.repeat_interleave(start, na)
        gat = self.atom_ptr[idx][bid] + local
        x = torch.zeros(b, nmax, self.vocab.shape[1], device=self.device)
        x[bid, local] = self.vocab[self.atom_index[gat]]
        mask = torch.zeros(b, nmax, dtype=torch.bool, device=self.device); mask[bid, local] = True
        adj = torch.zeros(b, nmax, nmax, device=self.device); adj[bid, local, local] = 1.0
        if int(ne.sum()) > 0:
            ebid = torch.repeat_interleave(torch.arange(b, device=self.device), ne)
            estart = torch.cumsum(ne, 0) - ne
            elocal = torch.arange(int(ne.sum()), device=self.device) - torch.repeat_interleave(estart, ne)
            e = self.edges[self.edge_ptr[idx][ebid] + elocal]
            adj[ebid, e[:, 0], e[:, 1]] = 1.0; adj[ebid, e[:, 1], e[:, 0]] = 1.0
        return GraphBatch(node_features=x, adjacency=adj, node_mask=mask)


__all__ = ["stream_doseresp_aggregate", "map_cell_labels", "dose_model", "standardize_batch", "GraphBank",
           "sha256_file", "ALMANAC_POLICY", "DOSE_REFERENCE_MOLAR", "forbid_test_responses", "load_fold0_cell_graphs",
           "append_log", "DATA_ROOT"]

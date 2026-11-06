"""PublicSplitGate-v1 step S1: development data preparation (docs/PUBLIC_SPLIT_GATE_V1_PROTOCOL.md section 5).

Reads DeepSynBa's released NCI-ALMANAC files (author commit 43c16961) and writes the development point tables:
  * machine-readable protocol.json (card SHA-256, input hashes) BEFORE any response value is loaded;
  * cell and drug mappings onto the frozen CellMiner mapping and structure table;
  * the split-1 single-agent lookup (same definition as external_deepsynba_lookup_v1, restricted to evaluable cells);
  * combination points of evaluable split-1 (train) and split-2 (validation) matrices, anchored on the lookup.
Split 3 (test) is only counted from the metadata: no split-3 response value is indexed or read.

Run with the project venv in isolated mode (the released pickle is only ever opened under python -I):
  .venv/Scripts/python.exe -I experiments/model_improvement/public_split_gate/s1_prepare.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT.parent / "bisurfgnn"
SRC = DATA / "data/raw/external_benchmarks/deepsynba/data"
META, RESP = SRC / "nci_synergy_metadata.csv", SRC / "nci_synergy_dose_response.pickle"
CELL_MAP = DATA / "data/processed/cellminer/cell_mapping.csv"
DRUGS = DATA / "data/processed/nci_almanac_unclipped/drugs.csv"
NAMES = DATA / "data/raw/nci_almanac/ComboCompoundNames_all.txt"
CARD = ROOT / "docs/PUBLIC_SPLIT_GATE_V1_PROTOCOL.md"
RUN = ROOT / "runs/model_improvement/public_split_gate_v1"
OUT = RUN / "s1"
EXPECTED = {"metadata": "42b00d74c376d35a5f38c6a63d9a2369f8888d45d37d5b500e334e72ee6ae5b1",
            "responses": "de56596840262950b1b390d40d610789d46af19c2f162d45fa41a7fc23ea5ad1"}
DOSE_REFERENCE_M = 1e-12          # bisurfgnn.data.nci_almanac: dose_model = log10(1 + D / 1e-12 M)
MONO_UPPER = 3.0                  # CompactBiSurfGNN monotherapy contract, ratio scale
SPLITS = {1: "train", 2: "validation", 3: "test"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, ensure_ascii=False, default=float) + "\n")


def dose_key(x) -> str:
    return f"{float(x):.6g}"      # identical to external_deepsynba_lookup_baseline.py


def map_cells(cells: list[str]) -> tuple[dict, list[dict]]:
    rows = list(csv.DictReader(CELL_MAP.open(encoding="utf-8")))
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["source_label"]].append(r)
    mapping, records = {}, []
    for cell in cells:
        hits = by_label.get(cell, [])
        status = hits[0]["status"] if len(hits) == 1 else ("missing" if not hits else "ambiguous")
        if status == "mapped":
            mapping[cell] = dict(cell_id=hits[0]["source_cell_id"], biological_cell_id=hits[0]["biological_cell_id"])
        records.append(dict(deepsynba_cell=cell, status=status, cell_id=hits[0]["source_cell_id"] if hits else "",
                            biological_cell_id=hits[0]["biological_cell_id"] if hits else "",
                            reason=hits[0]["reason"] if hits else ""))
    return mapping, records


def map_drugs(drugs: list[str]) -> tuple[dict, list[dict]]:
    table = {r["drug_id"]: r for r in csv.DictReader(DRUGS.open(encoding="utf-8"))}
    nsc_to_id = {r["nsc"]: r["drug_id"] for r in table.values()}
    names = defaultdict(set)
    for line in NAMES.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            names[parts[1].strip().lower()].add(parts[0])
    mapping, records = {}, []
    for drug in drugs:
        nscs = sorted(names.get(drug.strip().lower(), set()))
        ids = sorted({nsc_to_id[n] for n in nscs if n in nsc_to_id})
        canonical = sorted({table[i]["structure_alias_of"] or i for i in ids})
        ok = len(canonical) == 1 and bool(table[canonical[0]]["canonical_smiles"])
        if ok:
            mapping[drug] = canonical[0]
        records.append(dict(deepsynba_drug=drug, name_nscs=";".join(nscs), structure_ids=";".join(ids),
                            canonical_drug_id=";".join(canonical), mapped=ok,
                            alias_merged=len(ids) > 1 and len(canonical) == 1))
    return mapping, records


def main():
    if not CARD.exists() or not (RUN / "CARD_FREEZE.json").exists():
        raise RuntimeError("The launch card and its freeze record are required")
    freeze = json.loads((RUN / "CARD_FREEZE.json").read_text(encoding="utf-8"))
    card_hash = sha256(CARD)
    if card_hash != freeze["sha256"]:
        raise RuntimeError("Launch card changed after its freeze record")
    inputs = {"metadata": sha256(META), "responses": sha256(RESP), "cell_mapping": sha256(CELL_MAP),
              "drug_structures": sha256(DRUGS), "compound_names": sha256(NAMES), "s1_script": sha256(Path(__file__))}
    for k, v in EXPECTED.items():
        if inputs[k] != v:
            raise RuntimeError(f"Released input differs from the recorded hash: {k}")
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(RUN / "protocol.json", {
        "protocol": "PublicSplitGate-v1", "card": str(CARD.relative_to(ROOT)), "card_sha256": card_hash,
        "card_freeze_record": freeze, "inputs_sha256": inputs, "written_before_response_loading": True,
        "information_level": "pooled single-agent edges (all-plate means in the release), recovered from split 1",
        "evaluation_unit": "combination entries of evaluable split-3 matrices; pooled point RMSE in percent",
        "gates": {"G1": "BiSurfGNN combination RMSE <= DeepSynBa v2 (same set) if v2 passed [6.948, 8.492]; "
                        "otherwise <= 0.858 x lookup-HSA combination RMSE (same set)",
                  "G2": "BiSurfGNN vs XGBoost residual combination RMSE <= -5% and pair-bootstrap 95% CI upper < 0"},
        "seeds": [41, 42, 43, 44, 45], "recorded_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    meta = pd.read_csv(META)
    if list(meta.columns) != ["cell", "drug1", "drug2", "split"]:
        raise RuntimeError(f"Unexpected metadata columns: {list(meta.columns)}")
    cell_map, cell_records = map_cells(sorted(meta.cell.unique()))
    drug_map, drug_records = map_drugs(sorted(set(meta.drug1) | set(meta.drug2)))
    pd.DataFrame(cell_records).to_csv(OUT / "cell_mapping_used.csv", index=False)
    pd.DataFrame(drug_records).to_csv(OUT / "drug_mapping_used.csv", index=False)
    meta["evaluable"] = meta.cell.isin(cell_map) & meta.drug1.isin(drug_map) & meta.drug2.isin(drug_map)
    meta["key"] = meta.drug1 + "_" + meta.drug2 + "_" + meta.cell
    unmapped_cells = sorted(r["deepsynba_cell"] for r in cell_records if r["status"] != "mapped")
    counts = {SPLITS[s]: dict(matrices=int((meta.split == s).sum()), evaluable=int((meta.evaluable & (meta.split == s)).sum()),
                              excluded=int((~meta.evaluable & (meta.split == s)).sum()))
              for s in SPLITS}
    stop = []
    if unmapped_cells != ["MDA-MB-468"]:
        stop.append(f"unmapped cells other than MDA-MB-468: {unmapped_cells}")
    if len(drug_map) < 101 or len(drug_map) != len(drug_records):
        stop.append(f"drug coverage {len(drug_map)}/{len(drug_records)}")
    if counts["validation"]["evaluable"] < 0.95 * counts["validation"]["matrices"]:
        stop.append("evaluable validation combination entries below 95%")
    if stop:
        write_json(OUT / "STOP.json", {"stop": stop, "counts": counts})
        raise RuntimeError(f"S1 stop rule: {stop}")

    data = pickle.load(RESP.open("rb"))                     # python -I only; hash checked above
    dev = meta[meta.split.isin([1, 2])].reset_index(drop=True)
    missing = [k for k in dev.key if k not in data]
    if missing:
        raise RuntimeError(f"{len(missing)} development keys missing from the response file")
    test_keys_present = int(sum(k in data for k in meta.key[meta.split == 3]))   # key presence only
    entries = {k: data[k] for k in dev.key}                  # split 1/2 values only from here on
    del data
    anomalies = defaultdict(int)
    for k, e in entries.items():
        m, d1, d2 = np.asarray(e["matrix"], float), np.asarray(e["x1_dose"], float), np.asarray(e["x2_dose"], float)
        if m.shape != (4, 4) or d1.shape != (4,) or d2.shape != (4,):
            anomalies["bad_shape"] += 1
        if d1[0] != 0 or d2[0] != 0 or (d1[1:] <= 0).any() or (d2[1:] <= 0).any():
            anomalies["dose_zero_layout"] += 1
        if not np.isfinite(m).all():
            anomalies["non_finite_response"] += 1
    if anomalies.get("bad_shape") or anomalies.get("non_finite_response"):
        raise RuntimeError(f"Matrix format anomalies: {dict(anomalies)}")

    # split-1 single-agent lookup on evaluable matrices (MDA-MB-468 excluded from every method)
    exact, by_drug, control = defaultdict(list), defaultdict(list), []
    train_eval = dev[(dev.split == 1) & dev.evaluable]
    for a, b, c, k in zip(train_eval.drug1, train_eval.drug2, train_eval.cell, train_eval.key):
        e = entries[k]; m = np.asarray(e["matrix"], float)
        control.append(m[0, 0])
        for i in range(1, 4):
            exact[(c, a, dose_key(e["x1_dose"][i]))].append(m[i, 0]); by_drug[(a, dose_key(e["x1_dose"][i]))].append(m[i, 0])
            exact[(c, b, dose_key(e["x2_dose"][i]))].append(m[0, i]); by_drug[(b, dose_key(e["x2_dose"][i]))].append(m[0, i])
    exact_mean = {k: float(np.mean(v)) for k, v in exact.items()}
    by_drug_mean = {k: float(np.mean(v)) for k, v in by_drug.items()}
    global_mean = float(np.mean([v for vs in exact.values() for v in vs]))
    control_mean = float(np.mean(control))
    spread = np.array([max(v) - min(v) for v in exact.values()])
    pd.DataFrame([dict(cell=k[0], drug=k[1], dose_key=k[2], mean_percent=exact_mean[k], n=len(v), spread=max(v) - min(v))
                  for k, v in exact.items()]).to_csv(OUT / "lookup_split1_exact.csv", index=False)
    pd.DataFrame([dict(drug=k[0], dose_key=k[1], mean_percent=by_drug_mean[k], n=len(v)) for k, v in by_drug.items()]
                 ).to_csv(OUT / "lookup_split1_drug_dose.csv", index=False)

    def anchor(cell, drug, dose, tally):
        k = (cell, drug, dose_key(dose))
        if k in exact_mean:
            tally["exact"] += 1; return exact_mean[k]
        if k[1:] in by_drug_mean:
            tally["drug_dose"] += 1; return by_drug_mean[k[1:]]
        tally["global"] += 1; return global_mean

    report_roles, id_hashes = {}, {}
    for split, role in ((1, "train"), (2, "validation")):
        part = dev[(dev.split == split) & dev.evaluable].reset_index(drop=True)
        rows, tally, own_diff = [], defaultdict(int), []
        for surface, (a, b, c, k) in enumerate(zip(part.drug1, part.drug2, part.cell, part.key)):
            e = entries[k]; m = np.asarray(e["matrix"], float)
            d1, d2 = np.asarray(e["x1_dose"], float), np.asarray(e["x2_dose"], float)
            ma = [anchor(c, a, d1[i], tally) for i in range(1, 4)]
            mb = [anchor(c, b, d2[j], tally) for j in range(1, 4)]
            own_diff += [abs(ma[i - 1] - m[i, 0]) for i in range(1, 4)] + [abs(mb[j - 1] - m[0, j]) for j in range(1, 4)]
            pair = "|".join(sorted((drug_map[a], drug_map[b])))
            for i in range(1, 4):
                for j in range(1, 4):
                    rows.append((k, surface, pair, c, cell_map[c]["cell_id"], cell_map[c]["biological_cell_id"], a, b,
                                 drug_map[a], drug_map[b], i, j, d1[i], d2[j], ma[i - 1], mb[j - 1], m[i, j]))
        frame = pd.DataFrame(rows, columns=["matrix_key", "surface_index", "pair_id", "deepsynba_cell", "cell_id",
                                            "biological_cell_id", "deepsynba_drug_a", "deepsynba_drug_b", "drug_a_id",
                                            "drug_b_id", "row_i", "col_j", "dose_a_M", "dose_b_M", "mono_a_percent",
                                            "mono_b_percent", "response_percent"])
        frame["dose_a_model"] = np.log10(1.0 + frame.dose_a_M / DOSE_REFERENCE_M)
        frame["dose_b_model"] = np.log10(1.0 + frame.dose_b_M / DOSE_REFERENCE_M)
        for arm in ("a", "b"):
            frame[f"mono_{arm}_ratio_raw"] = frame[f"mono_{arm}_percent"] / 100.0
            frame[f"mono_response_{arm}"] = frame[f"mono_{arm}_ratio_raw"].clip(0.0, MONO_UPPER)
        frame["response"] = frame.response_percent / 100.0
        frame["hsa_lookup_percent"] = np.minimum(frame.mono_a_percent, frame.mono_b_percent)
        frame.to_parquet(OUT / f"points_{role}.parquet", index=False)
        frame[["matrix_key", "row_i", "col_j"]].to_csv(OUT / f"{role}_ids.csv", index=False)
        id_hashes[role] = sha256(OUT / f"{role}_ids.csv")
        y, hsa = frame.response_percent.to_numpy(), frame.hsa_lookup_percent.to_numpy()
        mono = np.concatenate([frame.mono_a_percent.to_numpy(), frame.mono_b_percent.to_numpy()])
        own = np.asarray(own_diff)
        report_roles[role] = dict(
            matrices=int(len(part)), pairs=int(part.drug1.combine(part.drug2, lambda x, z: "|".join(sorted((x, z)))).nunique()),
            cells=int(part.cell.nunique()), drugs=int(len(set(part.drug1) | set(part.drug2))), combination_points=int(len(frame)),
            anchor_fallback=dict(tally),
            anchor_vs_own_matrix_edge=dict(max_abs=float(own.max()), share_below_1e_4=float(np.mean(own < 1e-4))),
            response_percent=dict(min=float(y.min()), p01=float(np.percentile(y, 1)), median=float(np.median(y)),
                                  p99=float(np.percentile(y, 99)), max=float(y.max()), below_0=int((y < 0).sum()),
                                  above_300=int((y > 300).sum())),
            anchor_percent=dict(min=float(mono.min()), max=float(mono.max()),
                                clipped_into_0_3_ratio=int(((mono / 100 < 0) | (mono / 100 > MONO_UPPER)).sum())),
            dose_M=dict(min=float(min(frame.dose_a_M.min(), frame.dose_b_M.min())),
                        max=float(max(frame.dose_a_M.max(), frame.dose_b_M.max()))),
            lookup_hsa_combination_rmse_percent=float(np.sqrt(np.mean((hsa - y) ** 2))))
        print(role, json.dumps(report_roles[role], ensure_ascii=False), flush=True)

    report = dict(
        protocol="PublicSplitGate-v1", step="S1", card_sha256=card_hash, counts=counts,
        test=dict(matrices=counts["test"]["matrices"], evaluable_matrices=counts["test"]["evaluable"],
                  evaluable_combination_points=9 * counts["test"]["evaluable"], keys_present_in_response_file=test_keys_present,
                  response_values_read=0),
        cells=dict(total=len(cell_records), mapped=len(cell_map), unmapped=unmapped_cells),
        drugs=dict(total=len(drug_records), mapped=len(drug_map),
                   alias_merged=[r["deepsynba_drug"] for r in drug_records if r["alias_merged"]]),
        matrix_format_anomalies=dict(anomalies),
        lookup=dict(source="evaluable split-1 matrices", exact_keys=len(exact_mean), drug_dose_keys=len(by_drug_mean),
                    control_mean_percent=control_mean, global_mean_percent=global_mean,
                    exact_key_value_spread=dict(max=float(spread.max()), share_identical_within_1e_4=float(np.mean(spread < 1e-4)),
                                                keys_with_multiple_values=int(sum(len(v) > 1 for v in exact.values())))),
        roles=report_roles, id_sha256=id_hashes)
    write_json(OUT / "s1_report.json", report)
    write_json(OUT / "manifest.json", {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.name != "manifest.json"})
    print(json.dumps({k: v for k, v in report.items() if k != "roles"}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

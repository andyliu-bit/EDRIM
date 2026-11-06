"""SparseMono-v1 task construction: one monotherapy reading per new-scaffold drug x cell.

Information policy (docs/SPARSE_MONO_V1_PROTOCOL.md):
* target drug of a group = ``heldout_drug_id`` from the frozen OneNewScaffoldAnchored
  assignment (response-independent); the other drug is the partner;
* partner monotherapy: as in the original protocol (observed, same-block);
* target monotherapy: for each (target drug, cell) exactly ONE reading is visible —
  the reading at the median-rank distinct dose (lower median if even), taken from
  the referencing point with the smallest ``source_combodrugseq``. Every other
  reading of that drug x cell — other doses, other assay blocks at the same dose,
  every partner and both arms — is hidden;
* primary query points: target dose != support dose.

``RoleView`` is the only object the models receive; hidden readings live in
``RoleTruth`` and are used for training-role supervision and for evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

DOSE_DECIMALS = 6
VIEW_COLUMNS = ["source_combodrugseq", "surface_replicate_id", "prediction_group_id", "cell_id", "target_drug",
                "partner_drug", "target_arm", "target_dose", "partner_dose", "partner_mono", "visible_flag",
                "visible_value", "support_dose", "support_value", "primary_query", "response"]


@dataclass
class RoleView:
    points: pd.DataFrame        # VIEW_COLUMNS; visible_value is NaN where hidden
    supports: pd.DataFrame      # one row per (target drug, cell)

    def sha256(self) -> str:
        cols = [c for c in VIEW_COLUMNS if c != "response"]
        return hashlib.sha256(pd.util.hash_pandas_object(self.points[cols], index=False).values.tobytes()).hexdigest()


@dataclass
class RoleTruth:
    target_true_mono: np.ndarray  # aligned with RoleView.points; never passed to models


def dose_key(x) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float), DOSE_DECIMALS)


def with_target(frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    meta = assignments[["surface_replicate_id", "heldout_drug_id"]].drop_duplicates("surface_replicate_id")
    m = frame.merge(meta, on="surface_replicate_id", how="left", validate="many_to_one")
    if m.heldout_drug_id.isna().any():
        raise ValueError("Points without a designated target drug")
    is_a = m.heldout_drug_id.to_numpy() == m.drug_a_id.to_numpy()
    is_b = m.heldout_drug_id.to_numpy() == m.drug_b_id.to_numpy()
    if not (is_a ^ is_b).all():
        raise ValueError("Target drug must be exactly one arm (self pairs or missing target)")
    m["target_arm"] = np.where(is_a, "a", "b")
    m["target_drug"] = m.heldout_drug_id
    m["partner_drug"] = np.where(is_a, m.drug_b_id, m.drug_a_id)
    m["target_dose"] = dose_key(np.where(is_a, m.dose_a_model, m.dose_b_model))
    m["partner_dose"] = dose_key(np.where(is_a, m.dose_b_model, m.dose_a_model))
    m["partner_mono"] = np.where(is_a, m.mono_response_b, m.mono_response_a).astype(float)
    m["_target_mono"] = np.where(is_a, m.mono_response_a, m.mono_response_b).astype(float)
    return m


def choose_supports(m: pd.DataFrame) -> pd.DataFrame:
    """Median-rank distinct dose per (target drug, cell); single reading by smallest source row.

    Uses doses and row ids only — never the response or monotherapy values — to choose."""
    rows = []
    for (drug, cell), g in m.groupby(["target_drug", "cell_id"], sort=True):
        doses = np.sort(g.target_dose.unique())
        sdose = doses[(len(doses) - 1) // 2]
        at = g[g.target_dose == sdose]
        pick = at.loc[at.source_combodrugseq.idxmin()]
        rows.append(dict(target_drug=drug, cell_id=cell, n_distinct_doses=len(doses), support_dose=float(sdose),
                         support_source_row=int(pick.source_combodrugseq), support_value=float(pick._target_mono),
                         readings_at_support_dose=int(at._target_mono.nunique())))
    return pd.DataFrame(rows)


def build_role(frame: pd.DataFrame, assignments: pd.DataFrame, k: int = 1) -> tuple[RoleView, RoleTruth]:
    if k not in (0, 1):
        raise ValueError("Only k in {0, 1} is defined")
    m = with_target(frame, assignments)
    sup = choose_supports(m)
    m = m.merge(sup[["target_drug", "cell_id", "support_dose", "support_value"]], on=["target_drug", "cell_id"], how="left")
    at_support = m.target_dose.to_numpy() == m.support_dose.to_numpy()
    m["primary_query"] = ~at_support
    if k == 1:
        m["visible_flag"] = at_support
        m["visible_value"] = np.where(at_support, m.support_value, np.nan)
    else:  # k=0: no target reading at all; same query points
        m["visible_flag"] = False
        m["visible_value"] = np.nan
        m["support_dose"] = np.nan
        m["support_value"] = np.nan
        sup = sup.assign(support_value=np.nan, support_source_row=-1)
    truth = RoleTruth(target_true_mono=m["_target_mono"].to_numpy().copy())
    return RoleView(points=m[VIEW_COLUMNS].reset_index(drop=True), supports=sup), truth


def monotherapy_table(frame: pd.DataFrame) -> pd.DataFrame:
    """All monotherapy readings referenced by a role (both arms), deduplicated by observation identity
    (drug, cell, dose, reading). Only ever built from the TRAINING role."""
    parts = [pd.DataFrame({"drug": frame[f"drug_{a}_id"], "cell_id": frame.cell_id, "dose": dose_key(frame[f"dose_{a}_model"]),
                           "mono": frame[f"mono_response_{a}"].astype(float), "source_row": frame.source_combodrugseq})
             for a in ("a", "b")]
    t = pd.concat(parts)
    t = t.sort_values("source_row").drop_duplicates(["drug", "cell_id", "dose", "mono"], keep="first")
    return t.reset_index(drop=True)


def correction_examples(mono: pd.DataFrame) -> pd.DataFrame:
    """Simulated k=1 supports on training drug x cell pairs, same rule as deployment.

    Returns query readings (doses other than the support dose) with their support."""
    out = []
    for (drug, cell), g in mono.groupby(["drug", "cell_id"], sort=True):
        doses = np.sort(g.dose.unique())
        if len(doses) < 2:
            continue
        sdose = doses[(len(doses) - 1) // 2]
        at = g[g.dose == sdose]
        pick = at.loc[at.source_row.idxmin()]
        q = g[g.dose != sdose].assign(support_dose=float(sdose), support_value=float(pick.mono))
        out.append(q)
    return pd.concat(out).reset_index(drop=True)


def cluster_folds(assignments: pd.DataFrame, train_surfaces, k: int = 4) -> dict[str, int]:
    """Training drug -> fold, by hash of its chemical cluster id (response-independent)."""
    a = assignments[assignments.surface_replicate_id.isin(set(train_surfaces))]
    clusters = pd.concat([a[["heldout_drug_id", "heldout_cluster_id"]].set_axis(["drug", "cluster"], axis=1),
                          a[["partner_drug_id", "partner_cluster_id"]].set_axis(["drug", "cluster"], axis=1)]).drop_duplicates()
    if clusters.drug.duplicated().any():
        raise ValueError("Drug mapped to more than one cluster")
    return {d: int(hashlib.sha256(c.encode()).hexdigest(), 16) % k for d, c in zip(clusters.drug, clusters.cluster)}


__all__ = ["RoleView", "RoleTruth", "build_role", "choose_supports", "monotherapy_table", "correction_examples",
           "cluster_folds", "with_target", "dose_key"]

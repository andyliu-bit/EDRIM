"""Cross-screen (CS) monotherapy values for ALMANAC combination points.

Protocol: docs/CROSS_SCREEN_MONO_V1_PROTOCOL.md section 2.

* Source: ``data/interim/nci60_pretrain_v1/nci60_points.parquet`` (NCI DTP 2026-10 DOSERESP,
  AVERAGE_PTC/100 aggregated per (NSC, CellMiner cell, log10 molar dose); built by NCI60-Pretrain-v1 S2).
* Drug matching follows the NCI60-Pretrain-v1 NSC / identity rules:
  tier 0 = the ALMANAC drug's own NSC (plus frozen ``structure_alias_of`` partners),
  tier 1 = NCI-60 compounds in the same identity group (connectivity key).
  For every (drug, cell) the first tier that has a curve on that cell is used; inside a tier the curve
  with the most measured doses wins, then the smallest NSC (deterministic, response-independent).
* Cell matching: the frozen CellMiner mapping (ALMANAC source cell id -> biological cell id).
* Dose alignment: inside one (NSC, cell) curve, linear interpolation in log10 molar concentration between
  adjacent measured doses; outside the measured range the nearest end value (constant extrapolation),
  flagged per value.

Nothing in this module reads ALMANAC monotherapy or combination responses.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DOSE_REFERENCE_MOLAR = 1e-12        # bisurfgnn.data.nci_almanac DoseTransform.model_reference_molar
NODE_TOLERANCE_LOG10 = 5e-5         # NCI-60 log10 concentrations are stored with 4 decimals


def almanac_log10_molar(dose_model) -> np.ndarray:
    """Invert u = log10(1 + D / 1e-12 M) to log10 D (molar)."""
    u = np.asarray(dose_model, dtype=np.float64)
    molar = np.expm1(u * np.log(10.0)) * DOSE_REFERENCE_MOLAR
    if not np.isfinite(molar).all() or (molar <= 0).any():
        raise ValueError("ALMANAC doses must be finite and positive to align on a log10 scale")
    return np.log10(molar)


def candidate_compounds(drugs: pd.DataFrame, compounds: pd.DataFrame, drug_ids: Iterable[str]) -> pd.DataFrame:
    """ALMANAC drug -> NCI-60 NSC candidates with their matching tier.

    ``drugs``: ALMANAC drugs.csv (drug_id, nsc, identity_group, structure_alias_of).
    ``compounds``: nci60_compounds.parquet (nsc, identity_group) -- structured NCI-60 compounds only."""
    table = drugs.set_index("drug_id")
    nsc_of = {d: int(n) for d, n in zip(drugs.drug_id, drugs.nsc)}
    alias = {r.drug_id: r.structure_alias_of for r in drugs.itertuples() if r.structure_alias_of}
    available = set(int(n) for n in compounds.nsc)
    by_identity = compounds.groupby("identity_group").nsc.apply(lambda s: sorted(int(x) for x in s)).to_dict()
    rows = []
    for drug in sorted(set(drug_ids)):
        own = nsc_of[drug]
        partners = {a for a, b in alias.items() if b == drug} | ({alias[drug]} if drug in alias else set())
        tier0 = [(own, "same_nsc")] + [(nsc_of[p], "structure_alias_nsc") for p in sorted(partners)]
        seen = set()
        for nsc, method in tier0:
            if nsc not in seen:
                seen.add(nsc)
                rows.append(dict(drug_id=drug, nsc=nsc, tier=0, method=method, structured_in_nci60=nsc in available))
        for nsc in by_identity.get(table.loc[drug, "identity_group"], []):
            if nsc not in seen:
                seen.add(nsc)
                rows.append(dict(drug_id=drug, nsc=nsc, tier=1, method="identity_group", structured_in_nci60=True))
    return pd.DataFrame(rows, columns=["drug_id", "nsc", "tier", "method", "structured_in_nci60"])


class CurveBank:
    """(NSC, biological cell) -> sorted log10 molar doses and endpoint values."""

    def __init__(self, points: pd.DataFrame) -> None:
        frame = points[["nsc", "cell", "log10_conc", "y"]].copy()
        frame["cell"] = frame.cell.astype(str)
        frame["nsc"] = frame.nsc.astype(np.int64)
        if frame.duplicated(["nsc", "cell", "log10_conc"]).any():
            raise ValueError("NCI-60 points must be unique per (NSC, cell, dose)")
        if not np.isfinite(frame[["log10_conc", "y"]].to_numpy()).all():
            raise ValueError("NCI-60 points must be finite")
        frame = frame.sort_values(["nsc", "cell", "log10_conc"], kind="mergesort")
        self.curves: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {
            (int(nsc), str(cell)): (g.log10_conc.to_numpy(np.float64), g.y.to_numpy(np.float64))
            for (nsc, cell), g in frame.groupby(["nsc", "cell"], sort=True)}

    def n_doses(self, nsc: int, cell: str) -> int:
        curve = self.curves.get((int(nsc), str(cell)))
        return 0 if curve is None else len(curve[0])


def choose_curves(candidates: pd.DataFrame, curves: CurveBank, pairs: pd.DataFrame) -> pd.DataFrame:
    """One curve per requested (drug_id, cell_bio) pair, or none (uncovered)."""
    cand = {d: g.sort_values(["tier", "nsc"]) for d, g in candidates.groupby("drug_id")}
    rows = []
    for drug, cell in pairs[["drug_id", "cell_bio"]].drop_duplicates().itertuples(index=False):
        chosen = None
        group = cand.get(drug)
        if group is not None:
            for tier, part in group.groupby("tier", sort=True):
                # most measured doses first, then the smallest NSC
                options = sorted((-curves.n_doses(n, cell), int(n), m) for n, m in zip(part.nsc, part.method)
                                 if curves.n_doses(n, cell) > 0)
                if options:
                    negative_doses, nsc, method = options[0]
                    chosen = dict(nsc=nsc, tier=int(tier), method=method, n_curve_doses=-negative_doses,
                                  tied_candidates=sum(o[0] == negative_doses for o in options))
                    break
        if chosen is None:
            reason = ("no_structured_nci60_compound" if group is None or not group.structured_in_nci60.any()
                      else "no_curve_on_this_cell")
            rows.append(dict(drug_id=drug, cell_bio=cell, covered=False, nsc=-1, tier=-1, method="", n_curve_doses=0,
                             tied_candidates=0, curve_log10_lo=np.nan, curve_log10_hi=np.nan, uncovered_reason=reason))
        else:
            x, _ = curves.curves[(chosen["nsc"], cell)]
            rows.append(dict(drug_id=drug, cell_bio=cell, covered=True, **chosen, curve_log10_lo=float(x[0]),
                             curve_log10_hi=float(x[-1]), uncovered_reason=""))
    return pd.DataFrame(rows)


def cross_screen_values(drug_ids, cells_bio, dose_model, choice: pd.DataFrame, curves: CurveBank,
                        tolerance: float = NODE_TOLERANCE_LOG10) -> pd.DataFrame:
    """CS value per requested (drug, cell, dose) row, aligned with the inputs; NaN where uncovered."""
    req = pd.DataFrame({"drug_id": np.asarray(drug_ids, dtype=object), "cell_bio": np.asarray(cells_bio, dtype=object),
                        "dose_model": np.asarray(dose_model, dtype=np.float64)})
    req["log10_molar"] = almanac_log10_molar(req.dose_model)
    req = req.merge(choice, on=["drug_id", "cell_bio"], how="left", validate="many_to_one")
    if req.covered.isna().any():
        raise ValueError("Requested (drug, cell) pair without a curve decision")
    n = len(req)
    value = np.full(n, np.nan); extrap_low = np.zeros(n, bool); extrap_high = np.zeros(n, bool)
    at_node = np.zeros(n, bool); distance = np.zeros(n)
    covered = req.covered.to_numpy(bool)
    for (nsc, cell), idx in req[covered].groupby(["nsc", "cell_bio"]).groups.items():
        rows = req.index.get_indexer(idx)
        x, y = curves.curves[(int(nsc), str(cell))]
        q = req.log10_molar.to_numpy()[rows]
        value[rows] = np.interp(q, x, y)          # constant extrapolation outside [x[0], x[-1]]
        extrap_low[rows] = q < x[0] - tolerance
        extrap_high[rows] = q > x[-1] + tolerance
        distance[rows] = np.where(q < x[0], x[0] - q, np.where(q > x[-1], q - x[-1], 0.0))
        at_node[rows] = np.min(np.abs(q[:, None] - x[None, :]), axis=1) <= tolerance
    req["cs_mono"] = value
    req["extrapolated_low"] = extrap_low & covered
    req["extrapolated_high"] = extrap_high & covered
    req["extrapolated"] = req.extrapolated_low | req.extrapolated_high
    req["extrapolation_distance_log10"] = np.where(covered, distance, np.nan)
    req["at_measured_dose"] = at_node & covered
    return req


__all__ = ["almanac_log10_molar", "candidate_compounds", "CurveBank", "choose_curves", "cross_screen_values",
           "DOSE_REFERENCE_MOLAR", "NODE_TOLERANCE_LOG10"]

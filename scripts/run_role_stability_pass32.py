#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
LENS_KEY = {
    "surface_token": "surface_token_key",
    "conservative_morphology": "conservative_family_key",
    "broad_shape": "broad_shape_family_key",
    "stolfi_inspired_signature": "stolfi_signature_key",
}
CONTROL_COLS = ["meta_I", "meta_L", "meta_H", "line_marker_family"]


def load_data():
    df = pd.read_csv(OUT / "pass31_token_occurrences_with_morphology_context.csv")
    df = df.reset_index(drop=True).copy()
    df["occurrence_id"] = np.arange(len(df))
    inv = pd.read_csv(OUT / "pass31_role_unit_inventory.csv")
    cat31 = pd.read_csv(OUT / "pass31_maximal_reliable_role_catalog.csv")
    return df, inv, cat31


def unit_occurrence_ids(df: pd.DataFrame, lens: str, unit: str) -> set[int]:
    key = LENS_KEY[lens]
    return set(df.index[df[key].astype(str) == str(unit)].tolist())


def scoped_occurrence_ids(df: pd.DataFrame, lens: str, unit: str, scope_type: str, scope_value: str) -> set[int]:
    key = LENS_KEY[lens]
    return set(df.index[(df[key].astype(str) == str(unit)) & (df[scope_type].astype(str) == str(scope_value))].tolist())


def global_catalog(cat31: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g = cat31[cat31["catalog_level"].eq("global_core_induction_supported")].copy()
    for _, r in g.iterrows():
        ids = unit_occurrence_ids(df, r["lens"], r["role_unit_key"])
        rows.append({
            "tier": "reliable_global",
            "scope_type": "global",
            "scope_value": "global",
            "lens": r["lens"],
            "role_unit_key": str(r["role_unit_key"]),
            "dominant_role": r["dominant_role"],
            "occurrences": int(r["occurrences"]),
            "distinct_lines": int(r["distinct_lines"]),
            "distinct_token_types": int(r["distinct_token_types"]),
            "strata_total": int(r["strata_total"]),
            "section_count": int(r["section_count"]),
            "dominant_role_share": np.nan,
            "within_scope_purity": np.nan,
            "remaining_metadata_risk": np.nan,
            "induction_agree_count": r.get("induction_agree_count", np.nan),
            "induction_resolutions": r.get("induction_resolutions", np.nan),
            "new_occurrences_after_higher_tiers": np.nan,
            "top_tokens": r["top_tokens"],
            "occurrence_ids": " ".join(map(str, sorted(ids))),
        })
    return pd.DataFrame(rows)


def conditional_local_candidates(df: pd.DataFrame, cat_global: pd.DataFrame) -> pd.DataFrame:
    rows = []
    global_same = set(zip(cat_global["lens"].astype(str), cat_global["role_unit_key"].astype(str)))
    for scope_type in ["meta_I", "meta_L", "meta_H"]:
        remaining = [c for c in CONTROL_COLS if c != scope_type]
        for lens, key in LENS_KEY.items():
            for scope_value, sdf in df.groupby(scope_type, dropna=False):
                for unit, g in sdf.groupby(key, dropna=False):
                    n = int(len(g))
                    lines = int(g["loc"].nunique())
                    types = int(g["token"].nunique())
                    if n < 25 or lines < 20:
                        continue
                    if lens != "surface_token" and types < 2:
                        continue
                    if (lens, str(unit)) in global_same:
                        continue
                    vc = g["role"].astype(str).value_counts()
                    dom = str(vc.index[0])
                    share = float(vc.iloc[0] / vc.sum())
                    strata = g.groupby(remaining + ["role"]).size().rename("n").reset_index()
                    purity = float(strata.groupby(remaining)["n"].max().sum() / n)
                    remaining_risk = float(max(g[c].astype(str).value_counts(normalize=True).max() for c in remaining))
                    spreads = {c: int(g[c].astype(str).nunique()) for c in remaining}
                    spread_ok = (sum(v >= 2 for v in spreads.values()) >= 2 and spreads.get("line_marker_family", 2) >= 2)
                    if not (share >= 0.98 and purity >= 0.98 and remaining_risk <= 0.97 and spread_ok):
                        continue
                    ids = scoped_occurrence_ids(df, lens, unit, scope_type, scope_value)
                    rows.append({
                        "tier": "reliable_conditioned",
                        "scope_type": scope_type,
                        "scope_value": str(scope_value),
                        "lens": lens,
                        "role_unit_key": str(unit),
                        "dominant_role": dom,
                        "occurrences": n,
                        "distinct_lines": lines,
                        "distinct_token_types": types,
                        "strata_total": int(g["stratum_I_L_H_marker"].nunique()),
                        "section_count": int(g["meta_I"].astype(str).nunique()),
                        "dominant_role_share": round(share, 3),
                        "within_scope_purity": round(purity, 3),
                        "remaining_metadata_risk": round(remaining_risk, 3),
                        "induction_agree_count": np.nan,
                        "induction_resolutions": np.nan,
                        "new_occurrences_after_higher_tiers": np.nan,
                        "top_tokens": ", ".join(g["token"].astype(str).value_counts().head(8).index.tolist()),
                        "occurrence_ids": " ".join(map(str, sorted(ids))),
                    })
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    # Greedy compression keeps only conditioned units that add real coverage beyond global units and earlier conditioned units.
    covered = set()
    for ids in cat_global["occurrence_ids"]:
        covered.update(int(x) for x in str(ids).split() if x)
    selected = []
    used = set()
    while True:
        best_i = None
        best_new = 0
        for i, r in cand.iterrows():
            if i in used:
                continue
            ids = set(int(x) for x in str(r["occurrence_ids"]).split() if x)
            new = len(ids - covered)
            if new > best_new:
                best_new = new
                best_i = i
        if best_i is None or best_new < 10:
            break
        row = cand.loc[best_i].copy()
        row["new_occurrences_after_higher_tiers"] = int(best_new)
        selected.append(row)
        used.add(best_i)
        covered.update(int(x) for x in str(cand.loc[best_i, "occurrence_ids"]).split() if x)
    if not selected:
        return pd.DataFrame(columns=cand.columns)
    return pd.DataFrame(selected).sort_values(["new_occurrences_after_higher_tiers", "occurrences"], ascending=[False, False])


def promising_units(df: pd.DataFrame, inv: pd.DataFrame, promoted: pd.DataFrame) -> pd.DataFrame:
    promoted_pairs = set(zip(promoted["lens"].astype(str), promoted["role_unit_key"].astype(str)))
    promoted_ids = set()
    for ids in promoted["occurrence_ids"]:
        promoted_ids.update(int(x) for x in str(ids).split() if x)
    mask = (
        (inv["occurrences"] >= 15)
        & (inv["distinct_lines"] >= 10)
        & (inv["dominant_role_share"] >= 0.90)
        & (inv["stratified_role_purity"] >= 0.90)
        & (inv["stratum_majority_role_agreement"] >= 0.85)
        & ((inv["lens"].eq("surface_token")) | (inv["distinct_token_types"] >= 2))
    )
    rows = []
    for _, r in inv[mask].iterrows():
        pair = (str(r["lens"]), str(r["role_unit_key"]))
        if pair in promoted_pairs:
            continue
        ids = unit_occurrence_ids(df, r["lens"], r["role_unit_key"])
        new_occ = len(ids - promoted_ids)
        if new_occ < 5:
            continue
        rows.append({
            "tier": "promising_not_promoted",
            "scope_type": "global_candidate",
            "scope_value": "not_promoted",
            "lens": r["lens"],
            "role_unit_key": str(r["role_unit_key"]),
            "dominant_role": r["dominant_role"],
            "occurrences": int(r["occurrences"]),
            "distinct_lines": int(r["distinct_lines"]),
            "distinct_token_types": int(r["distinct_token_types"]),
            "strata_total": int(r["strata_total"]),
            "section_count": int(r["distinct_section_I"]),
            "dominant_role_share": round(float(r["dominant_role_share"]), 3),
            "within_scope_purity": round(float(r["stratified_role_purity"]), 3),
            "remaining_metadata_risk": round(float(r["metadata_concentration_risk_max_share"]), 3),
            "new_occurrences_after_higher_tiers": int(new_occ),
            "top_tokens": r["top_tokens"],
            "occurrence_ids": " ".join(map(str, sorted(ids))),
            "reason_not_promoted": reason_not_promoted(r),
        })
    if not rows:
        return pd.DataFrame()
    prom = pd.DataFrame(rows)
    return prom.sort_values(["new_occurrences_after_higher_tiers", "occurrences"], ascending=[False, False])


def reason_not_promoted(r: pd.Series) -> str:
    reasons = []
    if str(r.get("status_standard", "")) != "ROLE_STABLE_ACROSS_CONFOUNDS":
        reasons.append("fails global standard status")
    if bool(r.get("supported_standard", False)) and not bool(r.get("supported_strict", False)):
        reasons.append("threshold sensitive")
    if float(r.get("metadata_concentration_risk_max_share", 1.0)) >= 0.90:
        reasons.append("metadata concentration")
    if int(r.get("distinct_section_I", 0)) < 3:
        reasons.append("section spread too narrow")
    if not reasons:
        reasons.append("not independently promoted")
    return "; ".join(reasons)


def assign_occurrence_tiers(df: pd.DataFrame, global_cat: pd.DataFrame, local_cat: pd.DataFrame, prom: pd.DataFrame) -> pd.DataFrame:
    occ = df[["occurrence_id", "loc", "folio", "line_order", "token_position", "token", "role", "meta_I", "meta_L", "meta_H", "line_marker_family"]].copy()
    occ["assigned_tier"] = "low_reliability_or_unassigned"
    occ["assigned_lens"] = ""
    occ["assigned_unit"] = ""
    occ["assigned_scope"] = ""
    occ["assigned_role"] = ""
    # Priority order: global, conditioned, promising.
    for tier_df, tier_name in [(global_cat, "reliable_global"), (local_cat, "reliable_conditioned"), (prom, "promising_not_promoted")]:
        if tier_df is None or tier_df.empty:
            continue
        for _, r in tier_df.iterrows():
            ids = [int(x) for x in str(r["occurrence_ids"]).split() if x]
            mask = occ["occurrence_id"].isin(ids) & occ["assigned_tier"].eq("low_reliability_or_unassigned")
            occ.loc[mask, "assigned_tier"] = tier_name
            occ.loc[mask, "assigned_lens"] = r["lens"]
            occ.loc[mask, "assigned_unit"] = r["role_unit_key"]
            occ.loc[mask, "assigned_scope"] = f"{r['scope_type']}={r['scope_value']}"
            occ.loc[mask, "assigned_role"] = r["dominant_role"]
    return occ


def summarize_coverage(df: pd.DataFrame, occ_assign: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_occ = len(df)
    total_lines = df["loc"].nunique()
    rows = []
    priority = ["reliable_global", "reliable_conditioned", "promising_not_promoted", "low_reliability_or_unassigned"]
    cumulative_ids = set()
    for tier in priority:
        ids = set(occ_assign.index[occ_assign["assigned_tier"].eq(tier)].tolist())
        cumulative_ids |= ids
        rows.append({
            "tier": tier,
            "token_occurrences": len(ids),
            "token_coverage_pct": round(100 * len(ids) / total_occ, 2),
            "lines_touched": int(occ_assign.loc[list(ids), "loc"].nunique()) if ids else 0,
            "line_coverage_pct": round(100 * (occ_assign.loc[list(ids), "loc"].nunique() if ids else 0) / total_lines, 2),
            "cumulative_token_occurrences": len(cumulative_ids),
            "cumulative_token_coverage_pct": round(100 * len(cumulative_ids) / total_occ, 2),
            "cumulative_lines_touched": int(occ_assign.loc[list(cumulative_ids), "loc"].nunique()) if cumulative_ids else 0,
            "cumulative_line_coverage_pct": round(100 * (occ_assign.loc[list(cumulative_ids), "loc"].nunique() if cumulative_ids else 0) / total_lines, 2),
        })
    coverage = pd.DataFrame(rows)
    role_rows = []
    for tier in ["reliable_global", "reliable_conditioned", "promising_not_promoted"]:
        sub = occ_assign[occ_assign["assigned_tier"].eq(tier)]
        if sub.empty:
            continue
        for role, g in sub.groupby("assigned_role"):
            role_rows.append({
                "tier": tier,
                "role": role,
                "token_occurrences": len(g),
                "token_coverage_pct": round(100 * len(g) / total_occ, 2),
                "lines_touched": int(g["loc"].nunique()),
                "top_tokens": ", ".join(g["token"].astype(str).value_counts().head(8).index.tolist()),
            })
    rolecov = pd.DataFrame(role_rows).sort_values(["tier", "token_occurrences"], ascending=[True, False])
    return coverage, rolecov


def main():
    df, inv, cat31 = load_data()
    gcat = global_catalog(cat31, df)
    lcat = conditional_local_candidates(df, gcat)
    promoted = pd.concat([gcat, lcat], ignore_index=True)
    prom = promising_units(df, inv, promoted)
    occ_assign = assign_occurrence_tiers(df, gcat, lcat, prom)
    coverage, rolecov = summarize_coverage(df, occ_assign)

    gcat.drop(columns=["occurrence_ids"]).to_csv(OUT / "pass32_reliable_global_catalog.csv", index=False)
    lcat.drop(columns=["occurrence_ids"]).to_csv(OUT / "pass32_reliable_conditioned_catalog.csv", index=False)
    promoted.drop(columns=["occurrence_ids"]).to_csv(OUT / "pass32_promoted_reliable_catalog.csv", index=False)
    prom.drop(columns=["occurrence_ids"]).to_csv(OUT / "pass32_promising_not_promoted_units.csv", index=False)
    occ_assign.to_csv(OUT / "pass32_occurrence_reliability_assignments.csv", index=False)
    coverage.to_csv(OUT / "pass32_reliability_coverage_summary.csv", index=False)
    rolecov.to_csv(OUT / "pass32_role_coverage_summary.csv", index=False)
    # full unit classification without occurrence_ids for review
    promoted_noids = promoted.drop(columns=["occurrence_ids"]).copy()
    prom_noids = prom.drop(columns=["occurrence_ids"]).copy()
    low_rows = []
    promoted_pairs = set(zip(promoted["lens"].astype(str), promoted["role_unit_key"].astype(str)))
    prom_pairs = set(zip(prom["lens"].astype(str), prom["role_unit_key"].astype(str)))
    for _, r in inv.iterrows():
        pair = (str(r["lens"]), str(r["role_unit_key"]))
        if pair in promoted_pairs or pair in prom_pairs:
            continue
        low_rows.append({
            "tier": "low_reliability_or_unlikely",
            "lens": r["lens"],
            "role_unit_key": str(r["role_unit_key"]),
            "dominant_role": r["dominant_role"],
            "occurrences": int(r["occurrences"]),
            "distinct_lines": int(r["distinct_lines"]),
            "distinct_token_types": int(r["distinct_token_types"]),
            "status_standard": r["status_standard"],
            "dominant_role_share": round(float(r["dominant_role_share"]), 3),
            "stratified_role_purity": round(float(r["stratified_role_purity"]), 3),
            "metadata_concentration_risk_max_share": round(float(r["metadata_concentration_risk_max_share"]), 3),
            "top_tokens": r["top_tokens"],
        })
    low = pd.DataFrame(low_rows)
    low.to_csv(OUT / "pass32_low_reliability_or_unlikely_units.csv", index=False)
    print("Pass 32 complete")
    print(coverage.to_string(index=False))
    print("reliable global", len(gcat), "conditioned", len(lcat), "promising", len(prom), "low", len(low))

if __name__ == "__main__":
    main()

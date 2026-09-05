#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

LENS_KEY = {
    "surface_token": "surface_token_key",
    "conservative_morphology": "conservative_family_key",
    "broad_shape": "broad_shape_family_key",
    "stolfi_inspired_signature": "stolfi_signature_key",
}

# These features are used for independent context induction.  The old role labels
# are not included as input features.  Manuscript metadata controls are also
# omitted so that induction support does not simply restate section, Currier,
# hand, or line-marker metadata.
INDUCTION_FEATURES = [
    "pos_bin",
    "line_len_bin",
    "frame_prefix",
    "terminal_class",
    "shape_remainder",
    "gallows_signature",
    "has_gallows",
    "prev_terminal_class",
    "next_terminal_class",
    "prev_shape_remainder",
    "next_shape_remainder",
    "prev_frame_prefix",
    "next_frame_prefix",
    "prev_broad_shape_family_key",
    "next_broad_shape_family_key",
]

SCOPES = [
    ("global", []),
    ("section", ["meta_I"]),
    ("currier", ["meta_L"]),
    ("hand", ["meta_H"]),
    ("line_marker", ["line_marker_family"]),
    ("section_currier", ["meta_I", "meta_L"]),
    ("section_hand", ["meta_I", "meta_H"]),
    ("section_marker", ["meta_I", "line_marker_family"]),
    ("currier_hand", ["meta_L", "meta_H"]),
    ("currier_marker", ["meta_L", "line_marker_family"]),
]

K_VALUES = [16, 24, 32, 48, 64]


def load_occurrences() -> pd.DataFrame:
    df = pd.read_csv(OUT / "pass31_token_occurrences_with_morphology_context.csv")
    df = df.reset_index(drop=True).copy()
    df["occurrence_id"] = np.arange(len(df))
    return df


def ids_for_catalog_row(df: pd.DataFrame, row: pd.Series) -> set[int]:
    key = LENS_KEY[str(row["lens"])]
    unit = str(row["role_unit_key"])
    if str(row["scope_type"]) == "global":
        return set(df.index[df[key].astype(str).eq(unit)].tolist())
    scope_type = str(row["scope_type"])
    scope_value = str(row["scope_value"])
    if scope_type in df.columns:
        return set(df.index[df[key].astype(str).eq(unit) & df[scope_type].astype(str).eq(scope_value)].tolist())
    return set()


def build_induction_maps(df: pd.DataFrame):
    x = df[INDUCTION_FEATURES].astype(str).fillna("NA")
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
    x_enc = enc.fit_transform(x)
    maps = {}
    for k in K_VALUES:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=1337 + k,
            n_init=10,
            batch_size=1024,
            max_iter=500,
        )
        labels = model.fit_predict(x_enc)
        tmp = pd.DataFrame({"cluster": labels, "role": df["role"].astype(str)})
        counts = tmp.groupby(["cluster", "role"]).size().rename("n").reset_index()
        idx = counts.groupby("cluster")["n"].idxmax()
        dom = counts.loc[idx].set_index("cluster")
        totals = tmp.groupby("cluster").size()
        dom_role = dom["role"].to_dict()
        purity = {int(cl): float(dom.loc[cl, "n"] / totals.loc[cl]) for cl in dom.index}
        maps[k] = {"labels": labels, "dominant_role": dom_role, "purity": purity}
    return maps


def induction_support_for_ids(ids: list[int], role: str, maps) -> dict:
    votes = 0
    loose_votes = 0
    parts = []
    matches = []
    purities = []
    for k in K_VALUES:
        labels = maps[k]["labels"][ids]
        dom = maps[k]["dominant_role"]
        purity_map = maps[k]["purity"]
        match_rate = float(np.mean([dom[int(c)] == role for c in labels]))
        mean_purity = float(np.mean([purity_map[int(c)] for c in labels]))
        if match_rate >= 0.60 and mean_purity >= 0.45:
            votes += 1
        if match_rate >= 0.50 and mean_purity >= 0.40:
            loose_votes += 1
        matches.append(match_rate)
        purities.append(mean_purity)
        parts.append(f"{k}:{match_rate:.3f}/{mean_purity:.3f}")
    return {
        "induction_votes": votes,
        "loose_induction_votes": loose_votes,
        "mean_induction_match": round(float(np.mean(matches)), 3),
        "mean_induction_cluster_purity": round(float(np.mean(purities)), 3),
        "induction_detail": " ".join(parts),
    }


def candidate_rows(df: pd.DataFrame, maps) -> pd.DataFrame:
    rows = []
    for scope_name, scope_cols in SCOPES:
        for lens, key in LENS_KEY.items():
            group_cols = scope_cols + [key]
            for vals, g in df.groupby(group_cols, dropna=False):
                if not isinstance(vals, tuple):
                    vals = (vals,)
                unit = str(vals[-1])
                scope_value = "|".join(map(str, vals[:-1])) if scope_cols else "global"
                n = int(len(g))
                lines = int(g["loc"].nunique())
                types = int(g["token"].nunique())
                if n < 15 or lines < 10:
                    continue
                if lens != "surface_token" and types < 2:
                    continue
                counts = g["role"].astype(str).value_counts()
                role = str(counts.index[0])
                share = float(counts.iloc[0] / n)
                ids = list(map(int, g.index.tolist()))
                support = induction_support_for_ids(ids, role, maps)
                rows.append({
                    "scope_name": scope_name,
                    "scope_type": scope_cols[0] if len(scope_cols) == 1 else scope_name,
                    "scope_value": scope_value,
                    "lens": lens,
                    "role_unit_key": unit,
                    "dominant_role": role,
                    "occurrences": n,
                    "distinct_lines": lines,
                    "distinct_token_types": types,
                    "dominant_role_share": round(share, 3),
                    "section_count": int(g["meta_I"].astype(str).nunique()),
                    "currier_count": int(g["meta_L"].astype(str).nunique()),
                    "hand_count": int(g["meta_H"].astype(str).nunique()),
                    "marker_count": int(g["line_marker_family"].astype(str).nunique()),
                    "top_tokens": ", ".join(g["token"].astype(str).value_counts().head(8).index.tolist()),
                    "occurrence_ids": " ".join(map(str, sorted(ids))),
                    **support,
                })
    return pd.DataFrame(rows)


def select_environment_sensitive(df: pd.DataFrame, p32: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    covered = set()
    for _, row in p32.iterrows():
        covered |= ids_for_catalog_row(df, row)

    strict_pool = candidates[
        (candidates["occurrences"] >= 20)
        & (candidates["distinct_lines"] >= 15)
        & (candidates["dominant_role_share"] >= 0.95)
        & (candidates["induction_votes"] >= 2)
    ].copy()

    # Do not repeat exact Pass 32 promoted rows.
    old_keys = set(
        zip(
            p32["scope_type"].astype(str),
            p32["scope_value"].astype(str),
            p32["lens"].astype(str),
            p32["role_unit_key"].astype(str),
            p32["dominant_role"].astype(str),
        )
    )
    strict_pool = strict_pool[
        ~strict_pool.apply(
            lambda r: (
                str(r["scope_type"]),
                str(r["scope_value"]),
                str(r["lens"]),
                str(r["role_unit_key"]),
                str(r["dominant_role"]),
            )
            in old_keys,
            axis=1,
        )
    ].copy()

    scope_complexity = {name: len(cols) for name, cols in SCOPES}
    selected = []
    pool = strict_pool.copy()
    while True:
        best_i = None
        best_score = -10**9
        best_new = 0
        for i, row in pool.iterrows():
            ids = set(int(x) for x in str(row["occurrence_ids"]).split() if x)
            new = len(ids - covered)
            if new < 10:
                continue
            complexity = scope_complexity.get(str(row["scope_name"]), 2)
            score = (
                new
                - 5 * complexity
                + 4 * float(row["induction_votes"])
                + 25 * (float(row["dominant_role_share"]) - 0.95)
                + (2 if int(row["distinct_token_types"]) > 1 else 0)
            )
            if score > best_score:
                best_score = score
                best_i = i
                best_new = new
        if best_i is None:
            break
        row = pool.loc[best_i].copy()
        row["tier"] = "reliable_environment_sensitive"
        row["new_occurrences_after_higher_tiers"] = int(best_new)
        selected.append(row)
        covered |= set(int(x) for x in str(row["occurrence_ids"]).split() if x)
        pool = pool.drop(index=best_i)

    if not selected:
        return pd.DataFrame(columns=list(candidates.columns) + ["tier", "new_occurrences_after_higher_tiers"])
    out = pd.DataFrame(selected)
    return out.sort_values(["new_occurrences_after_higher_tiers", "occurrences"], ascending=[False, False])


def find_role_switches(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    local = candidates[
        (candidates["scope_name"].isin(["section", "currier", "hand", "line_marker"]))
        & (candidates["occurrences"] >= 15)
        & (candidates["distinct_lines"] >= 10)
        & (candidates["dominant_role_share"] >= 0.90)
        & (candidates["induction_votes"] >= 1)
    ].copy()
    for (lens, unit), g in local.groupby(["lens", "role_unit_key"]):
        roles = sorted(g["dominant_role"].astype(str).unique().tolist())
        if len(roles) <= 1:
            continue
        rows.append({
            "lens": lens,
            "role_unit_key": unit,
            "scope_count": int(len(g)),
            "distinct_roles": ", ".join(roles),
            "scopes": " | ".join(
                f"{r.scope_name}={r.scope_value}:{r.dominant_role} n={int(r.occurrences)}"
                for r in g.sort_values("occurrences", ascending=False).itertuples()
            ),
        })
    return pd.DataFrame(rows)


def assign_occurrences(df: pd.DataFrame, p32: pd.DataFrame, env: pd.DataFrame, good: pd.DataFrame) -> pd.DataFrame:
    occ = df[[
        "occurrence_id",
        "loc",
        "folio",
        "line_order",
        "token_position",
        "token",
        "role",
        "meta_I",
        "meta_L",
        "meta_H",
        "line_marker_family",
    ]].copy()
    occ["assigned_tier"] = "low_reliability_or_unassigned"
    occ["assigned_lens"] = ""
    occ["assigned_unit"] = ""
    occ["assigned_scope"] = ""
    occ["assigned_role"] = ""

    def apply_rows(rows: pd.DataFrame, tier_name: str):
        nonlocal occ
        if rows is None or rows.empty:
            return
        for _, r in rows.iterrows():
            if "occurrence_ids" in r:
                ids = [int(x) for x in str(r["occurrence_ids"]).split() if x]
            else:
                ids = list(ids_for_catalog_row(df, r))
            mask = occ["occurrence_id"].isin(ids) & occ["assigned_tier"].eq("low_reliability_or_unassigned")
            occ.loc[mask, "assigned_tier"] = tier_name
            occ.loc[mask, "assigned_lens"] = str(r["lens"])
            occ.loc[mask, "assigned_unit"] = str(r["role_unit_key"])
            occ.loc[mask, "assigned_scope"] = f"{r.get('scope_type','global')}={r.get('scope_value','global')}"
            occ.loc[mask, "assigned_role"] = str(r["dominant_role"])

    apply_rows(p32[p32["tier"].eq("reliable_global")], "reliable_global")
    apply_rows(p32[p32["tier"].eq("reliable_conditioned")], "reliable_conditioned")
    apply_rows(env, "reliable_environment_sensitive")
    apply_rows(good, "good_but_not_reliable_enough")
    return occ


def build_good_and_low(df: pd.DataFrame, candidates: pd.DataFrame, p32: pd.DataFrame, env: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    promoted_keys = set()
    for rows in [p32, env]:
        for _, r in rows.iterrows():
            promoted_keys.add((str(r.get("scope_type", "")), str(r.get("scope_value", "")), str(r["lens"]), str(r["role_unit_key"]), str(r["dominant_role"])))

    covered = set()
    for _, r in p32.iterrows():
        covered |= ids_for_catalog_row(df, r)
    for _, r in env.iterrows():
        covered |= set(int(x) for x in str(r["occurrence_ids"]).split() if x)

    good_pool = candidates[
        (candidates["occurrences"] >= 15)
        & (candidates["distinct_lines"] >= 10)
        & (candidates["dominant_role_share"] >= 0.90)
        & (
            (candidates["induction_votes"] >= 1)
            | (candidates["loose_induction_votes"] >= 2)
        )
    ].copy()
    good_rows = []
    for _, r in good_pool.iterrows():
        key = (str(r["scope_type"]), str(r["scope_value"]), str(r["lens"]), str(r["role_unit_key"]), str(r["dominant_role"]))
        if key in promoted_keys:
            continue
        ids = set(int(x) for x in str(r["occurrence_ids"]).split() if x)
        new = len(ids - covered)
        if new < 8:
            continue
        reasons = []
        if float(r["dominant_role_share"]) < 0.95:
            reasons.append("role purity below reliable threshold")
        if int(r["induction_votes"]) < 2:
            reasons.append("induction support below reliable threshold")
        if int(r["occurrences"]) < 20 or int(r["distinct_lines"]) < 15:
            reasons.append("evidence count below reliable threshold")
        if str(r["scope_name"]).count("_") >= 1:
            reasons.append("narrow compound scope")
        if not reasons:
            reasons.append("redundant after higher reliability tiers")
        rr = r.copy()
        rr["tier"] = "good_but_not_reliable_enough"
        rr["new_occurrences_after_higher_tiers"] = int(new)
        rr["reason_not_promoted"] = ", ".join(reasons)
        good_rows.append(rr)
    good = pd.DataFrame(good_rows)
    if not good.empty:
        good = good.sort_values(["new_occurrences_after_higher_tiers", "occurrences"], ascending=[False, False])

    low = candidates[
        ~candidates.index.isin(good.index if not good.empty else [])
    ].copy()
    low["tier"] = "low_reliability_or_unlikely"
    low["reason"] = "does not meet good candidate thresholds"
    return good, low


def coverage_summaries(df: pd.DataFrame, occ: pd.DataFrame):
    total = len(df)
    total_lines = int(df["loc"].nunique())
    order = [
        "reliable_global",
        "reliable_conditioned",
        "reliable_environment_sensitive",
        "good_but_not_reliable_enough",
        "low_reliability_or_unassigned",
    ]
    rows = []
    cumulative = set()
    for tier in order:
        ids = set(occ.loc[occ["assigned_tier"].eq(tier), "occurrence_id"].astype(int).tolist())
        cumulative |= ids
        lines = int(occ.loc[occ["occurrence_id"].isin(ids), "loc"].nunique()) if ids else 0
        clines = int(occ.loc[occ["occurrence_id"].isin(cumulative), "loc"].nunique()) if cumulative else 0
        rows.append({
            "tier": tier,
            "token_occurrences": len(ids),
            "token_coverage_pct": round(100 * len(ids) / total, 2),
            "lines_touched": lines,
            "line_coverage_pct": round(100 * lines / total_lines, 2),
            "cumulative_token_occurrences": len(cumulative),
            "cumulative_token_coverage_pct": round(100 * len(cumulative) / total, 2),
            "cumulative_lines_touched": clines,
            "cumulative_line_coverage_pct": round(100 * clines / total_lines, 2),
        })
    cov = pd.DataFrame(rows)

    role_rows = []
    for tier in order[:-1]:
        sub = occ[occ["assigned_tier"].eq(tier)]
        for role, g in sub.groupby("assigned_role"):
            role_rows.append({
                "tier": tier,
                "role": role,
                "token_occurrences": int(len(g)),
                "token_coverage_pct": round(100 * len(g) / total, 2),
                "lines_touched": int(g["loc"].nunique()),
                "top_tokens": ", ".join(g["token"].astype(str).value_counts().head(8).index.tolist()),
            })
    rolecov = pd.DataFrame(role_rows)
    return cov, rolecov


def main():
    df = load_occurrences()
    p32 = pd.read_csv(OUT / "pass32_promoted_reliable_catalog.csv")
    maps = build_induction_maps(df)
    candidates = candidate_rows(df, maps)
    env = select_environment_sensitive(df, p32, candidates)
    role_switches = find_role_switches(df, candidates)
    good, low = build_good_and_low(df, candidates, p32, env)
    occ = assign_occurrences(df, p32, env, good)
    coverage, rolecov = coverage_summaries(df, occ)

    candidate_export = candidates.drop(columns=["occurrence_ids"])
    env_export = env.drop(columns=["occurrence_ids"])
    good_export = good.drop(columns=["occurrence_ids"]) if not good.empty else good
    low_export = low.drop(columns=["occurrence_ids"]) if "occurrence_ids" in low.columns else low

    candidate_export.to_csv(OUT / "pass33_scope_conditioned_candidate_units.csv", index=False)
    env_export.to_csv(OUT / "pass33_environment_sensitive_reliable_catalog.csv", index=False)
    good_export.to_csv(OUT / "pass33_good_but_not_reliable_enough_units.csv", index=False)
    low_export.to_csv(OUT / "pass33_low_reliability_or_unlikely_units.csv", index=False)
    role_switches.to_csv(OUT / "pass33_role_switch_audit.csv", index=False)
    occ.to_csv(OUT / "pass33_occurrence_reliability_assignments.csv", index=False)
    coverage.to_csv(OUT / "pass33_reliability_coverage_summary.csv", index=False)
    rolecov.to_csv(OUT / "pass33_role_coverage_summary.csv", index=False)
    support_summary = candidates.groupby(["scope_name", "lens"]).agg(
        candidates=("role_unit_key", "count"),
        induction_supported=("induction_votes", lambda s: int((s >= 2).sum())),
        weak_induction_supported=("loose_induction_votes", lambda s: int((s >= 2).sum())),
    ).reset_index()
    support_summary.to_csv(OUT / "pass33_induction_support_summary.csv", index=False)

    print("Pass 33 complete")
    print(coverage.to_string(index=False))
    print("environment-sensitive reliable rows", len(env))
    print("good but not reliable rows", len(good))
    print("role switch rows", len(role_switches))


if __name__ == "__main__":
    main()

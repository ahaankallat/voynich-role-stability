#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_role_stability_pass30 as p30

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / "data" / "processed" / "voynich_pass26_core_token_occurrences_exploded.csv"
LENSES = p30.LENSES
INDUCTION_K = [24, 32, 40, 50]


def copy_base_outputs():
    for path in OUT.glob("pass30_*.csv"):
        target = OUT / path.name.replace("pass30_", "pass31_")
        if not target.exists():
            target.write_bytes(path.read_bytes())


def load_augmented_data():
    base = OUT / "pass30_token_occurrences_with_morphology.csv"
    if base.exists():
        df = pd.read_csv(base)
    else:
        df = p30.add_morphology(pd.read_csv(INPUT))
    df = df.sort_values(["loc", "token_position"]).reset_index(drop=True).copy()
    df["line_len"] = df.groupby("loc")["token_position"].transform("max")
    df["rel_pos"] = df["token_position"] / df["line_len"]
    def pos_bin(p, L):
        p = int(p)
        L = int(L)
        if L == 1:
            return "single"
        if p == 1:
            return "first"
        if p == 2:
            return "second"
        if p == L:
            return "last"
        if p == L - 1:
            return "penult"
        if p / L <= 0.33:
            return "early"
        if p / L <= 0.66:
            return "middle"
        return "late"
    df["pos_bin"] = [pos_bin(p, L) for p, L in zip(df["token_position"], df["line_len"])]
    df["line_len_bin"] = pd.cut(df["line_len"], [0, 3, 6, 10, 999], labels=["short", "medium", "long", "verylong"]).astype(str)
    for col in ["token", "terminal_class", "shape_remainder", "frame_prefix", "broad_shape_family_key", "stolfi_signature_key", "conservative_family_key"]:
        df[f"prev_{col}"] = df.groupby("loc")[col].shift(1).fillna("BOL")
        df[f"next_{col}"] = df.groupby("loc")[col].shift(-1).fillna("EOL")
    return df


def split_by_hash(series, salt, fraction=0.30):
    return series.astype(str).map(lambda x: int(hashlib.sha256((salt + str(x)).encode()).hexdigest()[:8], 16) / 16**8 < fraction)


def aggregate_induction_features(df, key_col, candidates):
    cats = [
        "frame_prefix", "terminal_class", "shape_body", "shape_remainder", "gallows_signature", "pos_bin", "line_len_bin",
        "prev_terminal_class", "next_terminal_class", "prev_shape_remainder", "next_shape_remainder", "prev_frame_prefix", "next_frame_prefix",
        "prev_broad_shape_family_key", "next_broad_shape_family_key",
    ]
    sub = df[df[key_col].astype(str).isin(candidates)].copy()
    units = []
    dicts = []
    for unit, g in sub.groupby(key_col, dropna=False):
        d = {
            "num_mean_rel_pos": float(g["rel_pos"].mean()),
            "num_first_share": float((g["token_position"] == 1).mean()),
            "num_last_share": float((g["token_position"] == g["line_len"]).mean()),
            "num_mean_line_len_scaled": float(g["line_len"].mean()) / 20.0,
        }
        for c in cats:
            for val, share in g[c].astype(str).value_counts(normalize=True).head(12).items():
                d[f"{c}={val}"] = float(share)
        for c in ["prev_token", "next_token"]:
            for val, share in g[c].astype(str).value_counts(normalize=True).head(10).items():
                if share >= 0.05 or val in ("BOL", "EOL"):
                    d[f"{c}={val}"] = float(share)
        units.append(str(unit))
        dicts.append(d)
    return units, dicts


def independent_induction(df, inv):
    rows = []
    details = []
    clusters = []
    for k in INDUCTION_K:
        for lens, key_col in LENSES:
            ginv = inv[(inv["lens"] == lens) & (inv["occurrences"] >= 10)].copy()
            if lens != "surface_token":
                ginv = ginv[ginv["distinct_token_types"] >= 2]
            candidates = set(ginv["role_unit_key"].astype(str))
            units, dicts = aggregate_induction_features(df, key_col, candidates)
            if len(units) < 4:
                continue
            n_clusters = min(k, max(2, len(units) // 2))
            vec = DictVectorizer(sparse=True)
            X = normalize(vec.fit_transform(dicts))
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=31000 + k, n_init=12, batch_size=256, max_iter=200)
            labels = km.fit_predict(X)
            assign = pd.DataFrame({"lens": lens, "role_unit_key": units, "induction_k": n_clusters, "induced_cluster": labels})
            odf = df[[key_col, "role"]].copy().rename(columns={key_col: "role_unit_key"})
            odf["role_unit_key"] = odf["role_unit_key"].astype(str)
            odf = odf[odf["role_unit_key"].isin(candidates)].merge(assign, on="role_unit_key")
            crows = []
            for cl, cg in odf.groupby("induced_cluster"):
                vc = cg["role"].astype(str).value_counts()
                crows.append({
                    "lens": lens,
                    "induction_k": n_clusters,
                    "induced_cluster": cl,
                    "cluster_occurrences": len(cg),
                    "cluster_units": int(assign[assign["induced_cluster"] == cl].shape[0]),
                    "cluster_dominant_role": vc.index[0],
                    "cluster_dominant_role_share": round(float(vc.iloc[0] / vc.sum()), 3),
                    "cluster_top_roles": ",".join(vc.head(3).index.tolist()),
                })
            cdf = pd.DataFrame(crows)
            clusters.append(cdf)
            assign = assign.merge(cdf[["induced_cluster", "cluster_dominant_role", "cluster_dominant_role_share"]], on="induced_cluster", how="left")
            assign = assign.merge(inv[inv["lens"] == lens][["role_unit_key", "dominant_role", "occurrences", "status_standard", "supported_standard", "supported_loose"]], on="role_unit_key", how="left")
            assign["induced_agrees_with_audited_role"] = assign["cluster_dominant_role"].astype(str) == assign["dominant_role"].astype(str)
            assign["induced_high_purity"] = assign["cluster_dominant_role_share"].fillna(0) >= 0.55
            details.append(assign)
            for subset_name, mask in [("standard", assign["supported_standard"].fillna(False)), ("loose", assign["supported_loose"].fillna(False))]:
                sg = assign[mask]
                rows.append({
                    "lens": lens,
                    "induction_k": n_clusters,
                    "subset": subset_name,
                    "units": len(sg),
                    "induced_role_agreement": int(sg["induced_agrees_with_audited_role"].sum()),
                    "induced_high_purity_agreement": int((sg["induced_agrees_with_audited_role"] & sg["induced_high_purity"]).sum()),
                    "mean_cluster_role_purity": round(float(sg["cluster_dominant_role_share"].mean()) if len(sg) else 0, 3),
                })
    detail = pd.concat(details, ignore_index=True)
    cluster_summary = pd.concat(clusters, ignore_index=True)
    summary = pd.DataFrame(rows)
    det = detail[detail["supported_standard"].fillna(False)].copy()
    votes = det.groupby(["lens", "role_unit_key"]).agg(
        induction_resolutions=("induction_k", "nunique"),
        induction_agree_count=("induced_agrees_with_audited_role", "sum"),
        mean_induced_cluster_purity=("cluster_dominant_role_share", "mean"),
    ).reset_index()
    hp = det[det["induced_agrees_with_audited_role"] & det["induced_high_purity"]].groupby(["lens", "role_unit_key"]).size().rename("induction_high_purity_agree_count").reset_index()
    votes = votes.merge(hp, on=["lens", "role_unit_key"], how="left")
    votes["induction_high_purity_agree_count"] = votes["induction_high_purity_agree_count"].fillna(0).astype(int)
    votes["induction_majority_agrees"] = votes["induction_agree_count"] >= np.ceil(votes["induction_resolutions"] / 2)
    votes["induction_all_agree"] = votes["induction_agree_count"] == votes["induction_resolutions"]
    votes["mean_induced_cluster_purity"] = votes["mean_induced_cluster_purity"].round(3)
    return summary, detail, cluster_summary, votes


def predictive_validation(df):
    feature_sets = {
        "majority_role_baseline": [],
        "position_only": ["pos_bin", "line_len_bin"],
        "morphology_only": ["frame_prefix", "terminal_class", "shape_body", "shape_remainder", "gallows_signature", "broad_shape_family_key", "stolfi_signature_key"],
        "morphology_plus_context": [
            "frame_prefix", "terminal_class", "shape_body", "shape_remainder", "gallows_signature", "broad_shape_family_key", "stolfi_signature_key",
            "pos_bin", "line_len_bin", "prev_terminal_class", "next_terminal_class", "prev_shape_remainder", "next_shape_remainder", "prev_broad_shape_family_key", "next_broad_shape_family_key",
        ],
    }
    rows = []
    for basis, key in [("folio", "folio"), ("line", "loc")]:
        hold = split_by_hash(df[key], f"predict_{basis}", fraction=0.30)
        train = df[~hold].copy()
        test = df[hold].copy()
        for name, cols in feature_sets.items():
            if not cols:
                pred = np.repeat(train["role"].astype(str).value_counts().idxmax(), len(test))
            else:
                train_dicts = [{c: str(row[c]) for c in cols} for _, row in train.iterrows()]
                test_dicts = [{c: str(row[c]) for c in cols} for _, row in test.iterrows()]
                vec = DictVectorizer()
                X_train = vec.fit_transform(train_dicts)
                X_test = vec.transform(test_dicts)
                clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=4242, max_iter=1200, tol=1e-4, class_weight="balanced")
                clf.fit(X_train, train["role"].astype(str))
                pred = clf.predict(X_test)
            rows.append({
                "basis": basis,
                "model": name,
                "train_occurrences": len(train),
                "test_occurrences": len(test),
                "accuracy": round(float(accuracy_score(test["role"].astype(str), pred)), 3),
                "balanced_accuracy": round(float(balanced_accuracy_score(test["role"].astype(str), pred)), 3),
                "macro_f1": round(float(f1_score(test["role"].astype(str), pred, average="macro", zero_division=0)), 3),
                "weighted_f1": round(float(f1_score(test["role"].astype(str), pred, average="weighted", zero_division=0)), 3),
            })
    return pd.DataFrame(rows)


def section_local_units(df, inv):
    rows = []
    for lens, key_col in LENSES:
        for sec, sdf in df.groupby("meta_I", dropna=False):
            for unit, g in sdf.groupby(key_col, dropna=False):
                n = len(g)
                lines = g["loc"].nunique()
                types = g["token"].nunique()
                if n < 12 or lines < 8:
                    continue
                if lens != "surface_token" and types < 2:
                    continue
                vc = g["role"].astype(str).value_counts()
                dom = vc.index[0]
                share = float(vc.iloc[0] / vc.sum())
                sr = g.groupby(["meta_L", "meta_H", "line_marker_family", "role"]).size().rename("n").reset_index()
                purity = float(sr.groupby(["meta_L", "meta_H", "line_marker_family"])["n"].max().sum() / n)
                risk = max(g["meta_L"].astype(str).value_counts(normalize=True).max(), g["meta_H"].astype(str).value_counts(normalize=True).max(), g["line_marker_family"].astype(str).value_counts(normalize=True).max())
                support = n >= 18 and lines >= 12 and share >= 0.85 and purity >= 0.85 and g["stratum_I_L_H_marker"].nunique() >= 3 and g["line_marker_family"].nunique() >= 2 and risk < 0.95
                probable = n >= 15 and lines >= 10 and share >= 0.80 and purity >= 0.80 and g["stratum_I_L_H_marker"].nunique() >= 2 and g["line_marker_family"].nunique() >= 2 and risk < 0.98
                rows.append({
                    "lens": lens,
                    "section_I": sec,
                    "role_unit_key": str(unit),
                    "dominant_role": dom,
                    "occurrences": n,
                    "distinct_lines": lines,
                    "distinct_token_types": types,
                    "strata_total": g["stratum_I_L_H_marker"].nunique(),
                    "marker_families": g["line_marker_family"].nunique(),
                    "dominant_role_share": round(share, 3),
                    "within_section_stratified_purity": round(purity, 3),
                    "within_section_meta_risk": round(float(risk), 3),
                    "section_local_supported": bool(support),
                    "section_local_probable": bool(probable),
                    "top_tokens": ", ".join(g["token"].astype(str).value_counts().head(6).index.tolist()),
                })
    loc = pd.DataFrame(rows)
    std = set((r.lens, str(r.role_unit_key)) for _, r in inv[inv["supported_standard"]].iterrows())
    loose = set((r.lens, str(r.role_unit_key)) for _, r in inv[inv["supported_loose"]].iterrows())
    loc["globally_standard_supported"] = [(r.lens, str(r.role_unit_key)) in std for _, r in loc.iterrows()]
    loc["globally_loose_supported"] = [(r.lens, str(r.role_unit_key)) in loose for _, r in loc.iterrows()]
    return loc.sort_values(["section_local_supported", "section_local_probable", "lens", "occurrences"], ascending=[False, False, True, False])


def reliable_catalog(inv, votes, section_local):
    m = inv.merge(votes, on=["lens", "role_unit_key"], how="left")
    for col in ["induction_resolutions", "induction_agree_count", "induction_high_purity_agree_count"]:
        m[col] = m[col].fillna(0).astype(int)
    m["induction_majority_agrees"] = m["induction_majority_agrees"].fillna(False)
    rows = []
    def add_row(level, scope, r):
        rows.append({
            "catalog_level": level,
            "lens": r["lens"],
            "scope": scope,
            "role_unit_key": r["role_unit_key"],
            "dominant_role": r["dominant_role"],
            "occurrences": int(r["occurrences"]),
            "distinct_lines": int(r["distinct_lines"]),
            "distinct_token_types": int(r["distinct_token_types"]),
            "strata_total": int(r["strata_total"]),
            "section_count": int(r.get("distinct_section_I", 1)),
            "induction_agree_count": r.get("induction_agree_count", "NA"),
            "induction_resolutions": r.get("induction_resolutions", "NA"),
            "top_tokens": r["top_tokens"],
        })
    for _, r in m[m["supported_standard"] & m["induction_majority_agrees"]].iterrows():
        add_row("global_core_induction_supported", "global", r)
    for _, r in m[m["supported_loose"] & ~m["supported_standard"] & m["induction_majority_agrees"]].iterrows():
        add_row("global_probable_loose_induction_supported", "global", r)
    sl = section_local[section_local["section_local_supported"] & ~section_local["globally_standard_supported"]].copy()
    for _, r in sl.iterrows():
        rows.append({
            "catalog_level": "section_local_supported",
            "lens": r["lens"],
            "scope": f"section_{r['section_I']}",
            "role_unit_key": r["role_unit_key"],
            "dominant_role": r["dominant_role"],
            "occurrences": int(r["occurrences"]),
            "distinct_lines": int(r["distinct_lines"]),
            "distinct_token_types": int(r["distinct_token_types"]),
            "strata_total": int(r["strata_total"]),
            "section_count": 1,
            "induction_agree_count": "NA",
            "induction_resolutions": "NA",
            "top_tokens": r["top_tokens"],
        })
    return pd.DataFrame(rows).sort_values(["catalog_level", "lens", "occurrences"], ascending=[True, True, False])


def main():
    copy_base_outputs()
    print("load")
    df = load_augmented_data()
    df.to_csv(OUT / "pass31_token_occurrences_with_morphology_context.csv", index=False)
    inv = pd.read_csv(OUT / "pass31_role_unit_inventory.csv")
    print("induction")
    induction_summary, induction_detail, induction_clusters, votes = independent_induction(df, inv)
    print("prediction")
    pred = predictive_validation(df)
    print("section local")
    sl = section_local_units(df, inv)
    catalog = reliable_catalog(inv, votes, sl)
    induction_summary.to_csv(OUT / "pass31_independent_induction_summary.csv", index=False)
    induction_detail.to_csv(OUT / "pass31_independent_induction_detail.csv", index=False)
    induction_clusters.to_csv(OUT / "pass31_independent_induction_cluster_summary.csv", index=False)
    votes.to_csv(OUT / "pass31_independent_induction_votes.csv", index=False)
    pred.to_csv(OUT / "pass31_predictive_role_validation.csv", index=False)
    sl.to_csv(OUT / "pass31_section_local_role_units.csv", index=False)
    catalog.to_csv(OUT / "pass31_maximal_reliable_role_catalog.csv", index=False)
    print("Pass 31 complete")
    print(induction_summary.to_string(index=False))
    print(votes.groupby("lens").agg(standard_units=("role_unit_key", "count"), majority_induction_agreement=("induction_majority_agrees", "sum"), all_resolution_agreement=("induction_all_agree", "sum")).reset_index().to_string(index=False))
    print(pred.to_string(index=False))
    print(sl.groupby(["section_local_supported", "section_local_probable", "lens"]).size().rename("count").reset_index().to_string(index=False))
    print(catalog.groupby(["catalog_level", "lens"]).size().rename("count").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()

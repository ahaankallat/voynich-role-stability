#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re, hashlib
import numpy as np
import pandas as pd
import gc

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "voynich_pass26_core_token_occurrences_exploded.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

FRAME_PREFIXES = ["qok", "qot", "qo", "q", "ok", "ot", "ol", "or", "o", "d", "ch", "sh", "s", "y"]
TERMINAL_CLASSES = ["aiiin", "aiin", "ain", "iin", "in", "eedy", "edy", "eey", "ey", "dy", "y", "ar", "al", "ol", "or", "ee", "e", "n", "l", "r", "m"]
GALLOWS = set("ktpf")
LENSES = [
    ("surface_token", "surface_token_key"),
    ("conservative_morphology", "conservative_family_key"),
    ("broad_shape", "broad_shape_family_key"),
    ("stolfi_inspired_signature", "stolfi_signature_key"),
]
THRESHOLDS = {
    "strict": {"min_occurrences": 35, "min_lines": 28, "min_dom_share": 0.86, "min_strat_purity": 0.86, "min_agreement": 0.82, "min_strata": 8, "min_sections": 4, "max_meta_risk": 0.88},
    "standard": {"min_occurrences": 25, "min_lines": 20, "min_dom_share": 0.80, "min_strat_purity": 0.80, "min_agreement": 0.75, "min_strata": 6, "min_sections": 3, "max_meta_risk": 0.92},
    "loose": {"min_occurrences": 18, "min_lines": 14, "min_dom_share": 0.74, "min_strat_purity": 0.74, "min_agreement": 0.68, "min_strata": 5, "min_sections": 3, "max_meta_risk": 0.95},
}

def split_frame(token: str):
    token = str(token)
    for prefix in sorted(FRAME_PREFIXES, key=len, reverse=True):
        if token.startswith(prefix) and len(token) > len(prefix) + 1:
            return prefix, token[len(prefix):]
    return "NONE", token

def split_terminal(token: str):
    token = str(token)
    for term in sorted(TERMINAL_CLASSES, key=len, reverse=True):
        if token.endswith(term) and len(token) > len(term):
            return token[:-len(term)], term
    return token, "NONE"

def eva_shape(token: str) -> str:
    token = str(token)
    out = []
    i = 0
    while i < len(token):
        if token.startswith("ch", i) or token.startswith("sh", i):
            out.append("B"); i += 2
        elif token.startswith("ee", i):
            out.append("E"); i += 2
        elif token.startswith("ii", i):
            out.append("I"); i += 2
        else:
            c = token[i]
            if c in "ktpf": out.append("G")
            elif c == "q": out.append("Q")
            elif c in "oa": out.append("O")
            elif c == "y": out.append("Y")
            elif c == "d": out.append("D")
            elif c in "lrnm": out.append("L")
            elif c in "ei": out.append("E")
            else: out.append("X")
            i += 1
    return re.sub(r"(.)\1+", r"\1", "".join(out))

def morphology_features(token: str):
    frame, rem = split_frame(token)
    body, terminal = split_terminal(rem)
    full_body, full_terminal = split_terminal(token)
    shape_body = eva_shape(body)
    shape_rem = eva_shape(rem)
    gallows = "".join(c for c in str(token) if c in GALLOWS)
    gallows_sig = gallows[:2] if gallows else "NONE"
    return {
        "frame_prefix": frame,
        "morph_body": body,
        "terminal_class": terminal,
        "shape_body": shape_body,
        "shape_remainder": shape_rem,
        "gallows_signature": gallows_sig,
        "has_gallows": int(bool(gallows)),
        "conservative_family_key": f"{body}|{terminal}|{shape_body}",
        "broad_shape_family_key": f"{shape_body}|{terminal}",
        "stolfi_signature_key": f"G_{gallows_sig}|S_{shape_rem}|T_{terminal}" if gallows else f"NG|S_{shape_rem}|T_{terminal}",
        "full_terminal_class": full_terminal,
        "full_body_shape": eva_shape(full_body),
    }

def add_morphology(df):
    feats = pd.DataFrame([morphology_features(t) for t in df["token"].astype(str)])
    out = pd.concat([df.reset_index(drop=True), feats], axis=1)
    out["surface_token_key"] = out["token"].astype(str)
    return out

def entropy_norm_from_counts(rc, group_col, value_col):
    total = rc.groupby(group_col)["count"].transform("sum")
    x = rc.copy()
    x["p"] = x["count"] / total
    x["plogp"] = x["p"] * np.log(x["p"])
    ent = -x.groupby(group_col)["plogp"].sum()
    n = x.groupby(group_col)[value_col].nunique()
    val = (ent / np.log(n.where(n > 1, np.e))).fillna(0).replace([np.inf, -np.inf], 0)
    val.loc[n <= 1] = 0.0
    return val

def summarize_units(df, key_col, lens_name):
    grouped = df.groupby(key_col, dropna=False)
    base = pd.DataFrame({
        "occurrences": grouped.size(),
        "distinct_lines": grouped["loc"].nunique(),
        "distinct_token_types": grouped["token"].nunique(),
        "strata_total": grouped["stratum_I_L_H_marker"].nunique(),
        "distinct_section_I": grouped["meta_I"].nunique(),
        "distinct_currier_L": grouped["meta_L"].nunique(),
        "distinct_hand_H": grouped["meta_H"].nunique(),
        "distinct_marker_family": grouped["line_marker_family"].nunique(),
    })
    rc = df.groupby([key_col, "role"], dropna=False).size().rename("count").reset_index()
    dom = rc.sort_values([key_col, "count"], ascending=[True, False]).drop_duplicates(key_col).set_index(key_col)
    base["dominant_role"] = dom["role"].astype(str)
    base["dominant_role_count"] = dom["count"]
    base["dominant_role_share"] = (base["dominant_role_count"] / base["occurrences"]).round(3)
    base["role_entropy_norm"] = entropy_norm_from_counts(rc, key_col, "role").round(3)

    for col, prefix in [("meta_I","section_I"),("meta_L","currier_L"),("meta_H","hand_H"),("line_marker_family","marker")]:
        cc = df.groupby([key_col, col], dropna=False).size().rename("count").reset_index()
        cdom = cc.sort_values([key_col, "count"], ascending=[True, False]).drop_duplicates(key_col).set_index(key_col)
        base[f"dominant_{prefix}"] = cdom[col].astype(str)
        base[f"dominant_{prefix}_share"] = (cdom["count"] / base["occurrences"]).round(3)
        base[f"{prefix}_entropy_norm"] = entropy_norm_from_counts(cc.rename(columns={col:"value"}), key_col, "value").round(3)

    sr = df.groupby([key_col, "stratum_I_L_H_marker", "role"], dropna=False).size().rename("n").reset_index()
    max_per = sr.groupby([key_col, "stratum_I_L_H_marker"], dropna=False)["n"].max().groupby(level=0).sum()
    base["stratified_role_purity"] = (max_per / base["occurrences"]).round(3)
    local = sr.sort_values("n", ascending=False).drop_duplicates([key_col, "stratum_I_L_H_marker"])
    stratum_sizes = df.groupby([key_col, "stratum_I_L_H_marker"], dropna=False).size().rename("size").reset_index()
    local = local.merge(stratum_sizes, on=[key_col, "stratum_I_L_H_marker"])
    local["global_role"] = local[key_col].map(base["dominant_role"])
    local["good"] = local["size"] * (local["role"].astype(str) == local["global_role"].astype(str))
    base["stratum_majority_role_agreement"] = (local.groupby(key_col)["good"].sum() / base["occurrences"]).round(3)
    risk_cols = ["dominant_section_I_share","dominant_currier_L_share","dominant_hand_H_share","dominant_marker_share"]
    base["metadata_concentration_risk_max_share"] = base[risk_cols].max(axis=1).round(3)
    obs = grouped.agg({
        "frame_prefix": lambda x: ",".join(sorted(set(map(str,x)))[:5]),
        "terminal_class": lambda x: ",".join(sorted(set(map(str,x)))[:5]),
        "token": lambda x: ", ".join(pd.Series(x).astype(str).value_counts().head(6).index.tolist()),
        "loc": lambda x: ", ".join(pd.Series(x).astype(str).drop_duplicates().head(8).tolist())
    })
    base["observed_frame_prefixes"] = obs["frame_prefix"]
    base["observed_terminal_classes"] = obs["terminal_class"]
    base["top_tokens"] = obs["token"]
    base["sample_loci"] = obs["loc"]
    out = base.reset_index().rename(columns={key_col: "role_unit_key"})
    out.insert(0, "lens", lens_name)
    return out.sort_values(["occurrences", "dominant_role_share"], ascending=[False, False])

def type_ok(row):
    return True if row.get("lens") == "surface_token" else row["distinct_token_types"] >= 2

def is_supported(row, th):
    return bool(row["occurrences"] >= th["min_occurrences"] and row["distinct_lines"] >= th["min_lines"] and type_ok(row) and row["dominant_role_share"] >= th["min_dom_share"] and row["stratified_role_purity"] >= th["min_strat_purity"] and row["stratum_majority_role_agreement"] >= th["min_agreement"] and row["strata_total"] >= th["min_strata"] and row["distinct_section_I"] >= th["min_sections"] and row["metadata_concentration_risk_max_share"] < th["max_meta_risk"])

def status_for(row, th):
    if is_supported(row, th): return "ROLE_STABLE_ACROSS_CONFOUNDS"
    if row["occurrences"] >= max(12, th["min_occurrences"]-6) and row["distinct_lines"] >= max(10, th["min_lines"]-5) and type_ok(row) and row["dominant_role_share"] >= th["min_dom_share"]-0.05 and row["stratified_role_purity"] >= th["min_strat_purity"]-0.05 and row["strata_total"] >= max(4, th["min_strata"]-1):
        return "STABLE_BUT_LOCAL_OR_THRESHOLD_SENSITIVE"
    if row["occurrences"] >= 10 and row["stratified_role_purity"] >= 0.70: return "CONTEXTUAL_ROLE_SIGNAL"
    return "LOW_POWER_OR_UNSTABLE"

def classify_all(inv):
    out = inv.copy()
    for name, th in THRESHOLDS.items():
        out[f"status_{name}"] = out.apply(lambda r: status_for(r, th), axis=1)
        out[f"supported_{name}"] = out[f"status_{name}"] == "ROLE_STABLE_ACROSS_CONFOUNDS"
    out["supported_all_thresholds"] = out[["supported_strict","supported_standard","supported_loose"]].all(axis=1)
    out["supported_standard_not_strict"] = out["supported_standard"] & ~out["supported_strict"]
    return out

def build_inventory(df):
    units = []
    occs = []
    for lens, col in LENSES:
        units.append(summarize_units(df, col, lens))
        keep = ["loc","folio","line_order","token_position","token","role","module","status","meta_I","meta_L","meta_H","line_marker_family","stratum_I_L_H_marker","pass20_tier","frame_prefix","morph_body","terminal_class","shape_body","shape_remainder","gallows_signature",col]
        o = df[keep].copy(); o.insert(0, "lens", lens); o = o.rename(columns={col:"role_unit_key"}); occs.append(o)
    return classify_all(pd.concat(units, ignore_index=True)), pd.concat(occs, ignore_index=True)

def status_counts(inv, status_col="status_standard"):
    order = ["ROLE_STABLE_ACROSS_CONFOUNDS","STABLE_BUT_LOCAL_OR_THRESHOLD_SENSITIVE","CONTEXTUAL_ROLE_SIGNAL","LOW_POWER_OR_UNSTABLE"]
    ct = inv.groupby(["lens", status_col]).size().rename("count").reset_index().rename(columns={status_col:"status"})
    rows = []
    for lens in [x[0] for x in LENSES]:
        for st in order:
            rows.append({"lens":lens,"status":st,"count":int(ct[(ct["lens"]==lens)&(ct["status"]==st)]["count"].sum())})
    return pd.DataFrame(rows)

def sensitivity_counts(inv):
    rows = []
    for lens,g in inv.groupby("lens"):
        rows.append({"lens":lens,"strict_supported":int(g["supported_strict"].sum()),"standard_supported":int(g["supported_standard"].sum()),"loose_supported":int(g["supported_loose"].sum()),"all_three_supported":int(g["supported_all_thresholds"].sum()),"standard_not_strict":int(g["supported_standard_not_strict"].sum())})
    return pd.DataFrame(rows)

def ablation_counts(inv):
    th = THRESHOLDS["standard"]
    def base_ok(r): return r["occurrences"] >= th["min_occurrences"] and r["distinct_lines"] >= th["min_lines"] and type_ok(r) and r["dominant_role_share"] >= th["min_dom_share"]
    def strat_ok(r): return base_ok(r) and r["stratified_role_purity"] >= th["min_strat_purity"] and r["stratum_majority_role_agreement"] >= th["min_agreement"]
    models = [
        ("role stability only", lambda r: base_ok(r)),
        ("plus stratum consistency", lambda r: strat_ok(r)),
        ("plus section spread", lambda r: strat_ok(r) and r["distinct_section_I"] >= th["min_sections"] and r["dominant_section_I_share"] < th["max_meta_risk"]),
        ("plus Currier spread", lambda r: strat_ok(r) and r["distinct_currier_L"] >= 2 and r["dominant_currier_L_share"] < 0.98),
        ("plus hand spread", lambda r: strat_ok(r) and r["distinct_hand_H"] >= 2 and r["dominant_hand_H_share"] < th["max_meta_risk"]),
        ("plus line marker spread", lambda r: strat_ok(r) and r["distinct_marker_family"] >= 2 and r["dominant_marker_share"] < th["max_meta_risk"]),
        ("full combined controls", lambda r: is_supported(r, th)),
    ]
    rows=[]
    for lens,g in inv.groupby("lens"):
        for model,fn in models:
            rows.append({"lens":lens,"model":model,"supported_count":int(g.apply(fn, axis=1).sum())})
    return pd.DataFrame(rows)

def random_baselines(df, n_iter=30, seed=53129):
    rng = np.random.default_rng(seed)
    roles = df["role"].astype(str).to_numpy()
    rows=[]
    for baseline in ["global_role_shuffle","within_stratum_role_shuffle"]:
        for i in range(n_iter):
            tmp = df.copy()
            if baseline == "global_role_shuffle":
                tmp["role"] = rng.permutation(roles)
            else:
                parts=[]
                for _,g in df.groupby("stratum_I_L_H_marker", sort=False):
                    arr = g["role"].astype(str).to_numpy().copy(); rng.shuffle(arr); parts.append(pd.Series(arr, index=g.index))
                tmp["role"] = pd.concat(parts).sort_index().to_numpy()
            inv,_ = build_inventory(tmp)
            for _,r in status_counts(inv).query("status == 'ROLE_STABLE_ACROSS_CONFOUNDS'").iterrows():
                rows.append({"baseline":baseline,"iteration":i,"lens":r["lens"],"supported_count":int(r["count"])})
    base=pd.DataFrame(rows)
    out=base.groupby(["baseline","lens"])["supported_count"].agg(["mean","median","max","std"]).reset_index()
    out[["mean","median","std"]] = out[["mean","median","std"]].round(2)
    return out

def split_by_hash(series, salt, fraction=0.30):
    def h(x):
        return int(hashlib.sha256((salt + str(x)).encode()).hexdigest()[:8], 16) / 16**8 < fraction
    return series.astype(str).map(h)

def heldout_validation(df, unit_basis="folio"):
    key = df["folio"] if unit_basis == "folio" else df["loc"]
    hold = split_by_hash(key, salt=f"holdout_{unit_basis}", fraction=0.30)
    dev = df[~hold].copy(); test = df[hold].copy()
    dev_inv,_ = build_inventory(dev); test_inv,_ = build_inventory(test)
    keys=["lens","role_unit_key"]
    sup=dev_inv[dev_inv["supported_standard"]][keys+["dominant_role","occurrences","distinct_lines"]]
    m=sup.merge(test_inv[keys+["dominant_role","occurrences","distinct_lines","dominant_role_share","stratified_role_purity","strata_total","distinct_section_I","metadata_concentration_risk_max_share","supported_loose","supported_standard"]], on=keys, how="left", suffixes=("_dev","_heldout"))
    m["basis"] = unit_basis
    m["seen_in_heldout"] = m["occurrences_heldout"].fillna(0) > 0
    m["same_dominant_role"] = m["dominant_role_dev"].astype(str) == m["dominant_role_heldout"].astype(str)
    m["heldout_retained_loose"] = m["seen_in_heldout"] & m["same_dominant_role"] & m["supported_loose"].fillna(False)
    rows=[]
    for lens,g in m.groupby("lens", dropna=False):
        rows.append({"basis":unit_basis,"lens":lens,"development_supported_units":len(g),"seen_in_heldout":int(g["seen_in_heldout"].sum()),"same_dominant_role":int((g["seen_in_heldout"] & g["same_dominant_role"]).sum()),"retained_under_loose_heldout":int(g["heldout_retained_loose"].sum())})
    return pd.DataFrame(rows), m.sort_values(["lens","occurrences_dev"], ascending=[True,False])

def leave_section_out(df):
    rows=[]; details=[]
    for sec in sorted(df["meta_I"].astype(str).unique()):
        dev=df[df["meta_I"].astype(str)!=sec].copy(); test=df[df["meta_I"].astype(str)==sec].copy()
        if len(dev)==0 or len(test)==0: continue
        dev_inv,_=build_inventory(dev); test_inv,_=build_inventory(test)
        keys=["lens","role_unit_key"]
        sup=dev_inv[dev_inv["supported_standard"]][keys+["dominant_role","occurrences"]]
        m=sup.merge(test_inv[keys+["dominant_role","occurrences","dominant_role_share"]], on=keys, how="left", suffixes=("_dev","_heldout"))
        m["heldout_section_I"] = sec
        m["seen"] = m["occurrences_heldout"].fillna(0) > 0
        m["same_role"] = m["dominant_role_dev"].astype(str) == m["dominant_role_heldout"].astype(str)
        details.append(m)
        for lens,g in m.groupby("lens", dropna=False):
            rows.append({"heldout_section_I":sec,"lens":lens,"dev_supported_units":len(g),"seen_in_section":int(g["seen"].sum()),"same_dominant_role":int((g["seen"]&g["same_role"]).sum())})
    detail=pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary=pd.DataFrame(rows)
    return summary, detail

def tier_robustness(df, full_inv):
    full_sup=full_inv[full_inv["supported_standard"]][["lens","role_unit_key","dominant_role","occurrences"]].copy()
    rows=[]; details=[]
    for tier,gdf in df.groupby("pass20_tier"):
        inv,_=build_inventory(gdf)
        m=full_sup.merge(inv[["lens","role_unit_key","dominant_role","occurrences","distinct_lines","dominant_role_share","supported_loose","supported_standard"]], on=["lens","role_unit_key"], how="left", suffixes=("_full","_tier"))
        m["tier"] = tier
        m["seen_in_tier"] = m["occurrences_tier"].fillna(0) > 0
        m["same_dominant_role"] = m["dominant_role_full"].astype(str) == m["dominant_role_tier"].astype(str)
        details.append(m)
        for lens,gg in m.groupby("lens", dropna=False):
            rows.append({"tier":tier,"lens":lens,"full_supported_units":len(gg),"seen_in_tier":int(gg["seen_in_tier"].sum()),"same_dominant_role":int((gg["seen_in_tier"]&gg["same_dominant_role"]).sum()),"supported_loose_in_tier":int(gg["supported_loose"].fillna(False).sum()),"supported_standard_in_tier":int(gg["supported_standard"].fillna(False).sum())})
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)

def section_profiles(df):
    rows=[]
    for sec,g in df.groupby("meta_I", dropna=False):
        top=g["role"].astype(str).value_counts(normalize=True).head(4)
        rows.append({"section_I":sec,"occurrences":len(g),"distinct_lines":g["loc"].nunique(),"distinct_tokens":g["token"].nunique(),"top_role_1":top.index[0] if len(top) else "NA","top_role_1_share":round(float(top.iloc[0]),3) if len(top) else 0,"top_role_2":top.index[1] if len(top)>1 else "NA","top_role_2_share":round(float(top.iloc[1]),3) if len(top)>1 else 0,"top_role_3":top.index[2] if len(top)>2 else "NA","top_role_3_share":round(float(top.iloc[2]),3) if len(top)>2 else 0})
    return pd.DataFrame(rows).sort_values("occurrences", ascending=False)


def random_baselines_fast(df, inv, n_iter=12, seed=53129):
    rng=np.random.default_rng(seed)
    rows=[]
    th=THRESHOLDS["standard"]
    nonrole_ok={}
    keycols={lens:col for lens,col in LENSES}
    for lens,col in LENSES:
        ginv=inv[inv["lens"]==lens].copy()
        mask=(ginv["occurrences"]>=th["min_occurrences"]) & (ginv["distinct_lines"]>=th["min_lines"]) & (ginv["strata_total"]>=th["min_strata"]) & (ginv["distinct_section_I"]>=th["min_sections"]) & (ginv["metadata_concentration_risk_max_share"]<th["max_meta_risk"])
        if lens != "surface_token":
            mask &= ginv["distinct_token_types"]>=2
        nonrole_ok[lens]=set(ginv.loc[mask,"role_unit_key"].astype(str))
    base_roles=df["role"].astype(str).to_numpy()
    for baseline in ["global_role_shuffle","within_stratum_role_shuffle"]:
        for it in range(n_iter):
            if baseline=="global_role_shuffle":
                shuffled=rng.permutation(base_roles)
            else:
                shuffled=np.empty(len(df), dtype=object)
                for _,idx in df.groupby("stratum_I_L_H_marker", sort=False).groups.items():
                    arr=df.loc[idx,"role"].astype(str).to_numpy().copy(); rng.shuffle(arr); shuffled[list(idx)] = arr
            tdf=df.copy()
            tdf["role_shuffled"]=shuffled
            for lens,col in LENSES:
                sub=tdf[[col,"stratum_I_L_H_marker","role_shuffled"]].copy().rename(columns={col:"unit", "role_shuffled":"role"})
                sub["unit"]=sub["unit"].astype(str)
                allowed=nonrole_ok[lens]
                sub=sub[sub["unit"].isin(allowed)]
                if sub.empty:
                    rows.append({"baseline":baseline,"iteration":it,"lens":lens,"supported_count":0})
                    continue
                occ=sub.groupby("unit").size()
                rc=sub.groupby(["unit","role"]).size().rename("n").reset_index()
                dom=rc.sort_values(["unit","n"], ascending=[True,False]).drop_duplicates("unit").set_index("unit")
                dom_share=dom["n"] / occ
                sr=sub.groupby(["unit","stratum_I_L_H_marker","role"]).size().rename("n").reset_index()
                purity=sr.groupby(["unit","stratum_I_L_H_marker"])["n"].max().groupby(level=0).sum()/occ
                local=sr.sort_values("n", ascending=False).drop_duplicates(["unit","stratum_I_L_H_marker"])
                sizes=sub.groupby(["unit","stratum_I_L_H_marker"]).size().rename("size").reset_index()
                local=local.merge(sizes,on=["unit","stratum_I_L_H_marker"])
                local["global_role"]=local["unit"].map(dom["role"])
                local["good"]=local["size"]*(local["role"].astype(str)==local["global_role"].astype(str))
                agree=local.groupby("unit")["good"].sum()/occ
                support=((dom_share>=th["min_dom_share"]) & (purity>=th["min_strat_purity"]) & (agree>=th["min_agreement"]))
                rows.append({"baseline":baseline,"iteration":it,"lens":lens,"supported_count":int(support.sum())})
    raw=pd.DataFrame(rows)
    out=raw.groupby(["baseline","lens"])["supported_count"].agg(["mean","median","max","std"]).reset_index()
    out[["mean","median","std"]]=out[["mean","median","std"]].fillna(0).round(2)
    return out

def main():
    print("load")
    df=add_morphology(pd.read_csv(INPUT))
    print("build inventory")
    inv,occ=build_inventory(df)
    print("counts")
    counts=status_counts(inv)
    sens=sensitivity_counts(inv)
    abl=ablation_counts(inv)
    print("heldout folio")
    hfol_sum,hfol_detail=heldout_validation(df, "folio")
    print("heldout line")
    hline_sum,hline_detail=heldout_validation(df, "line")
    print("tier")
    tier_sum,tier_detail=tier_robustness(df, inv)
    print("section profiles")
    sec=section_profiles(df)
    print("random baselines")
    base=random_baselines_fast(df, inv, n_iter=12)
    print("write")
    df.to_csv(OUT/"pass30_token_occurrences_with_morphology.csv", index=False)
    inv.to_csv(OUT/"pass30_role_unit_inventory.csv", index=False)
    occ.to_csv(OUT/"pass30_role_unit_occurrences.csv", index=False)
    counts.to_csv(OUT/"pass30_status_counts_by_lens.csv", index=False)
    sens.to_csv(OUT/"pass30_sensitivity_counts.csv", index=False)
    abl.to_csv(OUT/"pass30_ablation_counts.csv", index=False)
    base.to_csv(OUT/"pass30_random_baseline_summary.csv", index=False)
    pd.concat([hfol_sum,hline_sum], ignore_index=True).to_csv(OUT/"pass30_heldout_validation_summary.csv", index=False)
    pd.concat([hfol_detail,hline_detail], ignore_index=True).to_csv(OUT/"pass30_heldout_validation_detail.csv", index=False)
    hfol_sum.to_csv(OUT/"pass30_heldout_folio_summary.csv", index=False)
    hfol_detail.to_csv(OUT/"pass30_heldout_folio_detail.csv", index=False)
    hline_sum.to_csv(OUT/"pass30_heldout_line_summary.csv", index=False)
    hline_detail.to_csv(OUT/"pass30_heldout_line_detail.csv", index=False)
    tier_sum.to_csv(OUT/"pass30_transcription_tier_robustness_summary.csv", index=False)
    tier_detail.to_csv(OUT/"pass30_transcription_tier_robustness_detail.csv", index=False)
    sec.to_csv(OUT/"pass30_section_role_profiles.csv", index=False)
    print("Pass 30 complete")
    print(counts.to_string(index=False))
    print(sens.to_string(index=False))
    print(abl.to_string(index=False))
    print(base.to_string(index=False))
    print(pd.concat([hfol_sum,hline_sum], ignore_index=True).to_string(index=False))
    print(tier_sum.to_string(index=False))

if __name__ == "__main__": main()

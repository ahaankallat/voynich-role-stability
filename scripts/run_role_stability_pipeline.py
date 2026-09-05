#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

LENS_KEY = {
    "surface_token": "surface_token_key",
    "conservative_morphology": "conservative_family_key",
    "broad_shape": "broad_shape_family_key",
    "stolfi_inspired_signature": "stolfi_signature_key",
}
SCOPES = [
    ("global", []), ("section", ["meta_I"]), ("currier", ["meta_L"]), ("hand", ["meta_H"]),
    ("line_marker", ["line_marker_family"]), ("section_currier", ["meta_I", "meta_L"]),
    ("section_hand", ["meta_I", "meta_H"]), ("section_marker", ["meta_I", "line_marker_family"]),
    ("currier_hand", ["meta_L", "meta_H"]), ("currier_marker", ["meta_L", "line_marker_family"]),
]
TOKENIZATION_VARIANTS = [
    "surface_token_key", "conservative_family_key", "broad_shape_family_key", "stolfi_signature_key",
    "frame_terminal_key", "edge_signature_key",
    "left_boundary_merge_broad_key", "right_boundary_merge_broad_key",
    "left_boundary_merge_stolfi_key", "right_boundary_merge_stolfi_key",
]

def load_occurrences():
    df = pd.read_csv(OUT / "pass31_token_occurrences_with_morphology_context.csv").reset_index(drop=True)
    df["occurrence_id"] = np.arange(len(df))
    tok = df["token"].astype(str)
    df["token_start2"] = tok.str[:2]
    df["token_end2"] = tok.str[-2:]
    df["tok_len_bin"] = pd.cut(tok.str.len(), bins=[0,3,5,8,99], labels=["l1_3","l4_5","l6_8","l9p"]).astype(str)
    df["frame_terminal_key"] = df["frame_prefix"].astype(str) + "|" + df["terminal_class"].astype(str)
    df["edge_signature_key"] = df["token_start2"] + "|" + df["token_end2"] + "|" + df["tok_len_bin"]
    df["left_boundary_merge_broad_key"] = df["prev_broad_shape_family_key"].astype(str) + "+" + df["broad_shape_family_key"].astype(str)
    df["right_boundary_merge_broad_key"] = df["broad_shape_family_key"].astype(str) + "+" + df["next_broad_shape_family_key"].astype(str)
    df["left_boundary_merge_stolfi_key"] = df["prev_stolfi_signature_key"].astype(str) + "+" + df["stolfi_signature_key"].astype(str)
    df["right_boundary_merge_stolfi_key"] = df["stolfi_signature_key"].astype(str) + "+" + df["next_stolfi_signature_key"].astype(str)
    for c in TOKENIZATION_VARIANTS:
        df[c] = df[c].astype(str).fillna("NA")
    return df

def scope_mask(df, scope_name, scope_value):
    if scope_name == "global" or scope_value == "global": return pd.Series(True, index=df.index)
    cols = dict(SCOPES).get(scope_name, [])
    vals = str(scope_value).split("|")
    mask = pd.Series(True, index=df.index)
    for col, val in zip(cols, vals): mask &= df[col].astype(str).eq(str(val))
    return mask

def ids_for_row(df, row):
    key = LENS_KEY[str(row["lens"])]
    unit = str(row["role_unit_key"])
    sn = str(row.get("scope_name", "")) if "scope_name" in row.index else ""
    st = str(row.get("scope_type", "global"))
    sv = str(row.get("scope_value", "global"))
    mask = df[key].astype(str).eq(unit)
    if sn and sn != "nan":
        mask &= scope_mask(df, sn, sv)
    elif st == "global" or sv == "global":
        pass
    elif st in df.columns:
        mask &= df[st].astype(str).eq(sv)
    else:
        mask &= scope_mask(df, st, sv)
    return set(df.index[mask].tolist())

def add_ids_and_filter(df, cand):
    occ_ids = []
    for _, r in cand.iterrows(): occ_ids.append(" ".join(map(str, sorted(ids_for_row(df, r)))))
    cand = cand.copy(); cand["occurrence_ids"] = occ_ids
    cand = cand[cand["occurrence_ids"].str.len() > 0].copy()
    return cand

def precompute_support_sets(df, candidates):
    strict, loose = {}, {}
    for scope_name, scope_value in candidates[["scope_name", "scope_value"]].drop_duplicates().itertuples(index=False):
        sub = df[scope_mask(df, str(scope_name), str(scope_value))]
        if sub.empty: continue
        for variant in TOKENIZATION_VARIANTS:
            counts = sub.groupby([variant, "role"]).size().rename("n").reset_index()
            if counts.empty: continue
            idx = counts.groupby(variant)["n"].idxmax()
            dom = counts.loc[idx].copy().set_index(variant)
            totals = sub.groupby(variant).size().rename("total")
            lines = sub.groupby(variant)["loc"].nunique().rename("lines")
            dom["total"] = totals; dom["lines"] = lines; dom["share"] = dom["n"] / dom["total"]
            for role, dg in dom.groupby("role"):
                strict[(str(scope_name), str(scope_value), variant, str(role))] = set(dg[(dg["share"] >= 0.88) & (dg["total"] >= 12) & (dg["lines"] >= 8)].index.astype(str))
                loose[(str(scope_name), str(scope_value), variant, str(role))] = set(dg[(dg["share"] >= 0.80) & (dg["total"] >= 8) & (dg["lines"] >= 5)].index.astype(str))
    return strict, loose

def add_tokenization_support(df, cand):
    potential = cand[(cand["occurrences"] >= 12) & (cand["distinct_lines"] >= 8) & (cand["dominant_role_share"] >= 0.80) & ((cand["induction_votes"] >= 1) | (cand["loose_induction_votes"] >= 2))].copy()
    strict, loose = precompute_support_sets(df, potential)
    arrays = {v: df[v].astype(str).values for v in TOKENIZATION_VARIANTS}
    cand = cand.copy()
    cand["tokenization_votes"] = 0; cand["tokenization_loose_votes"] = 0; cand["supported_tokenization_variants"] = ""; cand["tokenization_support_detail"] = ""
    for i, r in potential.iterrows():
        ids = np.fromiter((int(x) for x in str(r["occurrence_ids"]).split() if x), dtype=int)
        role, sn, sv = str(r["dominant_role"]), str(r["scope_name"]), str(r["scope_value"])
        votes = loose_votes = 0; names=[]; details=[]
        for v in TOKENIZATION_VARIANTS:
            vals = arrays[v][ids]
            sset = strict.get((sn, sv, v, role), set())
            lset = loose.get((sn, sv, v, role), set())
            sf = float(np.mean(np.isin(vals, list(sset)))) if sset else 0.0
            lf = float(np.mean(np.isin(vals, list(lset)))) if lset else 0.0
            if sf >= 0.55: votes += 1; names.append(v)
            if lf >= 0.55: loose_votes += 1
            details.append(f"{v}:{sf:.2f}/{lf:.2f}")
        cand.loc[i, "tokenization_votes"] = votes
        cand.loc[i, "tokenization_loose_votes"] = loose_votes
        cand.loc[i, "supported_tokenization_variants"] = ", ".join(names)
        cand.loc[i, "tokenization_support_detail"] = " ".join(details)
    return cand

def promoted_keys(*catalogs):
    keys=set()
    for cat in catalogs:
        if cat is None or cat.empty: continue
        for _, r in cat.iterrows(): keys.add((str(r.get("scope_name", r.get("scope_type", "global"))), str(r.get("scope_value", "global")), str(r["lens"]), str(r["role_unit_key"]), str(r["dominant_role"])))
    return keys

def select_new(df, p32, env, cand):
    covered=set()
    for rows in [p32, env]:
        for _, r in rows.iterrows(): covered |= ids_for_row(df, r)
    old=promoted_keys(p32, env)
    pool=cand[(cand["occurrences"]>=15)&(cand["distinct_lines"]>=9)&(cand["dominant_role_share"]>=0.90)&(cand["induction_votes"]>=2)&(cand["tokenization_votes"]>=2)].copy()
    pool=pool[~pool.apply(lambda r:(str(r["scope_name"]),str(r["scope_value"]),str(r["lens"]),str(r["role_unit_key"]),str(r["dominant_role"])) in old, axis=1)]
    comp={name:len(cols) for name,cols in SCOPES}
    selected=[]
    while True:
        best_i=None; best_score=-1e9; best_new=0
        for i,r in pool.iterrows():
            ids=set(int(x) for x in str(r["occurrence_ids"]).split() if x); new=len(ids-covered)
            if new<6: continue
            score=new+6*float(r["induction_votes"])+5*float(r["tokenization_votes"])+35*(float(r["dominant_role_share"])-0.90)-5*comp.get(str(r["scope_name"]),2)+(3 if int(r["distinct_token_types"])>1 else 0)
            if score>best_score: best_i=i; best_score=score; best_new=new
        if best_i is None: break
        row=pool.loc[best_i].copy(); row["tier"]="reliable_tokenization_robust"; row["new_occurrences_after_higher_tiers"]=int(best_new)
        selected.append(row); covered |= set(int(x) for x in str(row["occurrence_ids"]).split() if x); pool=pool.drop(index=best_i)
        if len(selected)>=50: break
    return pd.DataFrame(selected)

def build_good_low(df, cand, p32, env, new):
    promoted=promoted_keys(p32, env, new); covered=set()
    for rows in [p32, env, new]:
        if rows is None or rows.empty: continue
        for _, r in rows.iterrows():
            if "occurrence_ids" in r and pd.notna(r.get("occurrence_ids", np.nan)): covered |= set(int(x) for x in str(r["occurrence_ids"]).split() if x)
            else: covered |= ids_for_row(df, r)
    pool=cand[(cand["occurrences"]>=12)&(cand["distinct_lines"]>=8)&(cand["dominant_role_share"]>=0.85)&((cand["induction_votes"]>=1)|(cand["loose_induction_votes"]>=2))&(cand["tokenization_loose_votes"]>=1)].copy()
    rows=[]
    for _,r in pool.iterrows():
        key=(str(r["scope_name"]),str(r["scope_value"]),str(r["lens"]),str(r["role_unit_key"]),str(r["dominant_role"]))
        if key in promoted: continue
        ids=set(int(x) for x in str(r["occurrence_ids"]).split() if x); newcount=len(ids-covered)
        if newcount<5: continue
        reasons=[]
        if float(r["dominant_role_share"])<0.90: reasons.append("role purity below promoted threshold")
        if int(r["induction_votes"])<2: reasons.append("induction support below promoted threshold")
        if int(r["tokenization_votes"])<2: reasons.append("tokenization robustness below promoted threshold")
        if int(r["occurrences"])<15 or int(r["distinct_lines"])<9: reasons.append("evidence count below promoted threshold")
        rr=r.copy(); rr["tier"]="good_but_not_reliable_enough"; rr["new_occurrences_after_higher_tiers"]=int(newcount); rr["reason_not_promoted"]=", ".join(reasons) if reasons else "redundant after higher tiers"
        rows.append(rr)
    good=pd.DataFrame(rows)
    if not good.empty: good=good.sort_values(["new_occurrences_after_higher_tiers","occurrences"], ascending=[False,False])
    low=cand.drop(index=good.index if not good.empty else []).copy(); low["tier"]="low_reliability_or_unlikely"; low["reason"]="does not meet good candidate thresholds"
    return good, low

def assign_occ(df, p32, env, new, good):
    occ=df[["occurrence_id","loc","folio","line_order","token_position","token","role","meta_I","meta_L","meta_H","line_marker_family"]].copy()
    occ["assigned_tier"]="low_reliability_or_unassigned"; occ["assigned_lens"]=""; occ["assigned_unit"]=""; occ["assigned_scope"]=""; occ["assigned_role"]=""
    def apply(rows,tier):
        if rows is None or rows.empty: return
        for _,r in rows.iterrows():
            ids=[int(x) for x in str(r["occurrence_ids"]).split() if x] if "occurrence_ids" in r and pd.notna(r.get("occurrence_ids",np.nan)) else list(ids_for_row(df,r))
            mask=occ["occurrence_id"].isin(ids)&occ["assigned_tier"].eq("low_reliability_or_unassigned")
            occ.loc[mask,"assigned_tier"]=tier; occ.loc[mask,"assigned_lens"]=str(r["lens"]); occ.loc[mask,"assigned_unit"]=str(r["role_unit_key"]); occ.loc[mask,"assigned_scope"]=f"{r.get('scope_name',r.get('scope_type','global'))}={r.get('scope_value','global')}"; occ.loc[mask,"assigned_role"]=str(r["dominant_role"])
    apply(p32[p32["tier"].eq("reliable_global")],"reliable_global"); apply(p32[p32["tier"].eq("reliable_conditioned")],"reliable_conditioned"); apply(env,"reliable_environment_sensitive"); apply(new,"reliable_tokenization_robust"); apply(good,"good_but_not_reliable_enough")
    return occ

def cov_summaries(df, occ):
    total=len(df); total_lines=int(df["loc"].nunique()); order=["reliable_global","reliable_conditioned","reliable_environment_sensitive","reliable_tokenization_robust","good_but_not_reliable_enough","low_reliability_or_unassigned"]
    rows=[]; cumulative=set()
    for tier in order:
        ids=set(occ.loc[occ["assigned_tier"].eq(tier),"occurrence_id"].astype(int).tolist()); cumulative|=ids
        lines=int(occ.loc[occ["occurrence_id"].isin(ids),"loc"].nunique()) if ids else 0; clines=int(occ.loc[occ["occurrence_id"].isin(cumulative),"loc"].nunique()) if cumulative else 0
        rows.append({"tier":tier,"token_occurrences":len(ids),"token_coverage_pct":round(100*len(ids)/total,2),"lines_touched":lines,"line_coverage_pct":round(100*lines/total_lines,2),"cumulative_token_occurrences":len(cumulative),"cumulative_token_coverage_pct":round(100*len(cumulative)/total,2),"cumulative_lines_touched":clines,"cumulative_line_coverage_pct":round(100*clines/total_lines,2)})
    role_rows=[]
    for tier in order[:-1]:
        sub=occ[occ["assigned_tier"].eq(tier)]
        for role,g in sub.groupby("assigned_role"):
            role_rows.append({"tier":tier,"role":role,"token_occurrences":int(len(g)),"token_coverage_pct":round(100*len(g)/total,2),"lines_touched":int(g["loc"].nunique()),"top_tokens":", ".join(g["token"].astype(str).value_counts().head(8).index.tolist())})
    return pd.DataFrame(rows), pd.DataFrame(role_rows)

def main():
    df=load_occurrences()
    cand=pd.read_csv(OUT/"pass33_scope_conditioned_candidate_units.csv")
    cand=add_ids_and_filter(df,cand)
    cand=cand[~((cand["scope_name"].astype(str)!="global") & (cand["scope_value"].astype(str).str.lower().isin(["nan", "none", ""])) )].copy()
    cand=add_tokenization_support(df,cand)
    p32=pd.read_csv(OUT/"pass32_promoted_reliable_catalog.csv")
    env=pd.read_csv(OUT/"pass33_environment_sensitive_reliable_catalog.csv")
    new=select_new(df,p32,env,cand)
    good,low=build_good_low(df,cand,p32,env,new)
    occ=assign_occ(df,p32,env,new,good)
    cov,rolecov=cov_summaries(df,occ)
    def dropids(x): return x[[c for c in x.columns if c!="occurrence_ids"]] if not x.empty else x
    dropids(cand).to_csv(OUT/"pass34_tokenization_robust_candidate_units.csv",index=False)
    dropids(new).to_csv(OUT/"pass34_tokenization_robust_reliable_catalog.csv",index=False)
    dropids(good).to_csv(OUT/"pass34_good_but_not_reliable_enough_units.csv",index=False)
    dropids(low).to_csv(OUT/"pass34_low_reliability_or_unlikely_units.csv",index=False)
    occ.to_csv(OUT/"pass34_occurrence_reliability_assignments.csv",index=False)
    cov.to_csv(OUT/"pass34_reliability_coverage_summary.csv",index=False); rolecov.to_csv(OUT/"pass34_role_coverage_summary.csv",index=False)
    rows=[]
    for (scope_name,lens),sub in cand.groupby(["scope_name","lens"]): rows.append({"scope_name":scope_name,"lens":lens,"candidates":int(len(sub)),"induction_supported":int((sub["induction_votes"]>=2).sum()),"tokenization_supported":int((sub["tokenization_votes"]>=2).sum()),"both_supported":int(((sub["induction_votes"]>=2)&(sub["tokenization_votes"]>=2)).sum())})
    pd.DataFrame(rows).to_csv(OUT/"pass34_tokenization_robustness_summary.csv",index=False)
    pd.DataFrame([{"tier":"reliable_global","units":int((p32["tier"]=="reliable_global").sum())},{"tier":"reliable_conditioned","units":int((p32["tier"]=="reliable_conditioned").sum())},{"tier":"reliable_environment_sensitive","units":int(len(env))},{"tier":"reliable_tokenization_robust","units":int(len(new))},{"tier":"good_but_not_reliable_enough","units":int(len(good))}]).to_csv(OUT/"pass34_promoted_unit_counts.csv",index=False)
    print("Pass 34 complete")
    print(cov.to_string(index=False)); print("new",len(new),"good",len(good));
    if not new.empty: print(new[["scope_name","scope_value","lens","role_unit_key","dominant_role","occurrences","distinct_lines","top_tokens","induction_votes","tokenization_votes","new_occurrences_after_higher_tiers"]].head(40).to_string(index=False))
if __name__=="__main__": main()

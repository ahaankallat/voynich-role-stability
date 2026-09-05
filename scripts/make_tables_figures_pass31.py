#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def esc(x):
    s = str(x)
    s = s.replace("\\", r"\textbackslash{}")
    for a,b in [('&',r'\&'),('%',r'\%'),('$',r'\$'),('#',r'\#'),('_',r'\_'),('{',r'\{'),('}',r'\}'),('~',r'\textasciitilde{}'),('^',r'\textasciicircum{}')]:
        s = s.replace(a,b)
    return s

def simple_table(path, headers, rows, align=None, size=None):
    align = align or "l" * len(headers)
    lines = []
    if size:
        lines.append("{" + size)
    lines.append(r"\begin{tabular}{" + align + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(esc(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(esc(v) for v in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if size:
        lines.append("}")
    path.write_text("\n".join(lines) + "\n")

def long_catalog(path, df):
    lines = []
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{p{0.19\linewidth}p{0.24\linewidth}p{0.07\linewidth}rrrrp{0.23\linewidth}}")
    lines.append(r"\toprule")
    lines.append(r"Lens & Unit & Role & Occ & Lines & Types & Ind & Top tokens \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Lens & Unit & Role & Occ & Lines & Types & Ind & Top tokens \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for _, r in df.iterrows():
        ind = "NA" if pd.isna(r.get("induction_agree_count")) else f"{int(float(r['induction_agree_count']))}/{int(float(r['induction_resolutions']))}"
        row = [r["lens"], r["role_unit_key"], r["dominant_role"], int(r["occurrences"]), int(r["distinct_lines"]), int(r["distinct_token_types"]), ind, r["top_tokens"]]
        lines.append(" & ".join(esc(v) for v in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n")

status = pd.read_csv(DATA / "pass31_status_counts_by_lens.csv")
status_piv = status.pivot(index="lens", columns="status", values="count").reset_index().fillna(0)
simple_table(TABLES / "status_counts_by_lens.tex", ["Lens", "Role stable", "Local", "Contextual", "Low power"], [[r["lens"], int(r.get("ROLE_STABLE_ACROSS_CONFOUNDS",0)), int(r.get("STABLE_BUT_LOCAL_OR_THRESHOLD_SENSITIVE",0)), int(r.get("CONTEXTUAL_ROLE_SIGNAL",0)), int(r.get("LOW_POWER_OR_UNSTABLE",0))] for _,r in status_piv.iterrows()], "lrrrr", r"\small")

sens = pd.read_csv(DATA / "pass31_sensitivity_counts.csv")
simple_table(TABLES / "sensitivity_counts.tex", ["Lens", "Strict", "Standard", "Loose", "All three"], [[r.lens, r.strict_supported, r.standard_supported, r.loose_supported, r.all_three_supported] for _,r in sens.iterrows()], "lrrrr", r"\small")

abl = pd.read_csv(DATA / "pass31_ablation_counts.csv")
# compact models only
keep_models = ["role stability only", "plus Currier spread", "plus line marker spread", "full combined controls"]
abl2 = abl[abl.model.isin(keep_models)].pivot(index="lens", columns="model", values="supported_count").reset_index().fillna(0)
simple_table(TABLES / "ablation_counts.tex", ["Lens", "Role only", "Currier", "Line marker", "Full"], [[r["lens"], int(r["role stability only"]), int(r["plus Currier spread"]), int(r["plus line marker spread"]), int(r["full combined controls"])] for _,r in abl2.iterrows()], "lrrrr", r"\small")

base = pd.read_csv(DATA / "pass31_random_baseline_summary.csv")
base2 = base.pivot(index="lens", columns="baseline", values="max").reset_index().fillna(0)
simple_table(TABLES / "baseline_summary.tex", ["Lens", "Global shuffle max", "Within stratum max"], [[r["lens"], int(r.get("global_role_shuffle",0)), int(r.get("within_stratum_role_shuffle",0))] for _,r in base2.iterrows()], "lrr", r"\small")

held = pd.read_csv(DATA / "pass31_heldout_validation_summary.csv")
simple_table(TABLES / "heldout_summary.tex", ["Basis", "Lens", "Dev", "Seen", "Same role", "Retained"], [[r.basis, r.lens, r.development_supported_units, r.seen_in_heldout, r.same_dominant_role, r.retained_under_loose_heldout] for _,r in held.iterrows()], "llrrrr", r"\small")

tier = pd.read_csv(DATA / "pass31_transcription_tier_robustness_summary.csv")
tier["tier_short"] = tier.tier.map(lambda x: "S1" if str(x).startswith("S1") else "S2")
simple_table(TABLES / "tier_robustness.tex", ["Tier", "Lens", "Full", "Seen", "Same role", "Loose"], [[r.tier_short, r.lens, r.full_supported_units, r.seen_in_tier, r.same_dominant_role, r.supported_loose_in_tier] for _,r in tier.iterrows()], "llrrrr", r"\small")

votes = pd.read_csv(DATA / "pass31_independent_induction_votes.csv")
vote_sum = votes.groupby("lens").agg(standard_units=("role_unit_key","count"), majority_agrees=("induction_majority_agrees","sum"), all_agree=("induction_all_agree","sum"), mean_cluster_purity=("mean_induced_cluster_purity","mean")).reset_index()
vote_sum["mean_cluster_purity"] = vote_sum["mean_cluster_purity"].round(3)
simple_table(TABLES / "induction_votes.tex", ["Lens", "Standard units", "Majority agreement", "All resolutions", "Mean purity"], [[r.lens, r.standard_units, r.majority_agrees, r.all_agree, r.mean_cluster_purity] for _,r in vote_sum.iterrows()], "lrrrr", r"\small")

pred = pd.read_csv(DATA / "pass31_predictive_role_validation.csv")
npred = pred[pred.model.isin(["majority_role_baseline","position_only","morphology_only","morphology_plus_context"])]
simple_table(TABLES / "predictive_validation.tex", ["Basis", "Model", "Acc", "Bal acc", "Macro F1", "Weighted F1"], [[r.basis, r.model.replace("_"," "), r.accuracy, r.balanced_accuracy, r.macro_f1, r.weighted_f1] for _,r in npred.iterrows()], "llrrrr", r"\small")

catalog = pd.read_csv(DATA / "pass31_maximal_reliable_role_catalog.csv")
long_catalog(TABLES / "maximal_reliable_role_catalog.tex", catalog)

section_local = pd.read_csv(DATA / "pass31_section_local_role_units.csv")
sl = section_local[(section_local.section_local_supported == True) & (section_local.globally_standard_supported == False)]
simple_table(TABLES / "section_local_supported.tex", ["Lens", "Section", "Unit", "Role", "Occ", "Lines", "Top tokens"], [[r.lens, r.section_I, r.role_unit_key, r.dominant_role, r.occurrences, r.distinct_lines, r.top_tokens] for _,r in sl.iterrows()], "lllrrrl", r"\small")

sec = pd.read_csv(DATA / "pass31_section_role_profiles.csv")
simple_table(TABLES / "section_profiles.tex", ["Section", "Occ", "Lines", "Top role", "Share", "Second", "Share"], [[r.section_I, r.occurrences, r.distinct_lines, r.top_role_1, r.top_role_1_share, r.top_role_2, r.top_role_2_share] for _,r in sec.iterrows()], "lrrrrrr", r"\small")

simple_table(TABLES / "layer_boundary.tex", ["Layer", "Status in this paper"], [
    ["Transcription", "Used as evidence assumption"],
    ["Morphology", "Operational foundation"],
    ["Token families", "Role test units"],
    ["Role stability", "Main claim"],
    ["Formal grammar", "Future work"],
    ["Semantics", "Future work"],
], "ll", r"\small")

simple_table(TABLES / "transcription_assumptions.tex", ["Item", "Treatment"], [
    ["S1 tier", "Strict multi transcription core"],
    ["S2 tier", "Strong but partial support"],
    ["Raw witnesses", "Documented as external sources"],
    ["Unit uncertainty", "Handled through morphology lenses and tier checks"],
], "ll", r"\small")

simple_table(TABLES / "morphology_lenses.tex", ["Lens", "Unit tested"], [
    ["surface token", "Exact token string"],
    ["conservative morphology", "Body, terminal class, and reduced body shape"],
    ["broad shape", "Reduced body shape and terminal class"],
    ["Stolfi inspired signature", "Gallows pattern, reduced remainder shape, and terminal class"],
], "ll", r"\small")

simple_table(TABLES / "prior_work_comparison.tex", ["Prior layer", "Contribution used here", "What this paper adds"], [
    ["Currier", "Textual variety is a major confound", "Controls role claims against variety"],
    ["IVTFF", "Multiple transcription witnesses matter", "Uses supported tiers rather than one stream"],
    ["Stolfi", "Word internal form is structured", "Builds role units from morphology aware lenses"],
    ["Entropy and topics", "Large scale structure is measurable", "Tests role stability at unit level"],
    ["Recent unit cautions", "Tokens and spaces are uncertain", "Avoids lexical and grammatical claims"],
], "lll", r"\scriptsize")

simple_table(TABLES / "hypotheses_future.tex", ["Hypothesis", "Required future test"], [
    ["Operator like roles", "Show ordered transitions and action like distribution"],
    ["Terminal like roles", "Show termination beyond line position effects"],
    ["Quantifier like roles", "Show count or measure behavior from external anchors"],
    ["Procedure like structure", "Show formal sequence rules across held out pages"],
], "ll", r"\small")

# figures
plt.figure(figsize=(7,4))
plot = status[status.status=="ROLE_STABLE_ACROSS_CONFOUNDS"].set_index("lens")["count"]
plot.plot(kind="bar")
plt.ylabel("stable units")
plt.tight_layout()
plt.savefig(FIGS / "stable_units_by_lens.png", dpi=200)
plt.close()

plt.figure(figsize=(7,4))
sens.set_index("lens")[["strict_supported","standard_supported","loose_supported"]].plot(kind="bar")
plt.ylabel("supported units")
plt.tight_layout()
plt.savefig(FIGS / "threshold_sensitivity.png", dpi=200)
plt.close()

plt.figure(figsize=(7,4))
abl2.set_index("lens")[["role stability only","plus line marker spread","full combined controls"]].plot(kind="bar")
plt.ylabel("supported units")
plt.tight_layout()
plt.savefig(FIGS / "ablation_counts.png", dpi=200)
plt.close()

plt.figure(figsize=(7,4))
vote_sum.set_index("lens")[["standard_units","majority_agrees","all_agree"]].plot(kind="bar")
plt.ylabel("units")
plt.tight_layout()
plt.savefig(FIGS / "induction_agreement.png", dpi=200)
plt.close()

plt.figure(figsize=(7,4))
context = pred[pred.model.isin(["majority_role_baseline","morphology_only","morphology_plus_context"])]
for basis, g in context.groupby("basis"):
    plt.plot(g["model"], g["weighted_f1"], marker="o", label=basis)
plt.xticks(rotation=20, ha="right")
plt.ylabel("weighted F1")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "predictive_validation.png", dpi=200)
plt.close()

plt.figure(figsize=(7,4))
sec2 = sec.set_index("section_I")["top_role_1_share"]
sec2.plot(kind="bar")
plt.ylabel("top role share")
plt.tight_layout()
plt.savefig(FIGS / "section_role_profiles.png", dpi=200)
plt.close()

print("Pass 31 tables and figures written")

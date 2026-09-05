#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
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
    if len(s) > 95:
        s = s[:92] + "..."
    repl = [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("|", r"$\mid$")]
    for a,b in repl:
        s = s.replace(a,b)
    return s

def tabular(df, cols, headers, path, align=None, max_rows=None):
    if max_rows:
        df = df.head(max_rows)
    if align is None:
        align = "l" * len(cols)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\midrule")
    for _,row in df.iterrows():
        lines.append(" & ".join(esc(row[c]) for c in cols) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines))

def longtabular(df, cols, headers, path, align=None, max_rows=None):
    if max_rows:
        df = df.head(max_rows)
    if align is None:
        align = "l" * len(cols)
    lines = [f"\\begin{{longtable}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    for _,row in df.iterrows():
        lines.append(" & ".join(esc(row[c]) for c in cols) + r" \\")
    lines += ["\\bottomrule", "\\end{longtable}"]
    path.write_text("\n".join(lines))

def main():
    inv = pd.read_csv(DATA / "pass30_role_unit_inventory.csv")
    counts = pd.read_csv(DATA / "pass30_status_counts_by_lens.csv")
    sens = pd.read_csv(DATA / "pass30_sensitivity_counts.csv")
    abl = pd.read_csv(DATA / "pass30_ablation_counts.csv")
    base = pd.read_csv(DATA / "pass30_random_baseline_summary.csv")
    held = pd.read_csv(DATA / "pass30_heldout_validation_summary.csv")
    tier = pd.read_csv(DATA / "pass30_transcription_tier_robustness_summary.csv")
    sec = pd.read_csv(DATA / "pass30_section_role_profiles.csv")

    counts_p = counts.pivot(index="lens", columns="status", values="count").fillna(0).astype(int).reset_index()
    wanted = ["lens", "ROLE_STABLE_ACROSS_CONFOUNDS", "STABLE_BUT_LOCAL_OR_THRESHOLD_SENSITIVE", "CONTEXTUAL_ROLE_SIGNAL", "LOW_POWER_OR_UNSTABLE"]
    counts_p = counts_p[[c for c in wanted if c in counts_p.columns]]
    tabular(counts_p, counts_p.columns.tolist(), ["Lens", "Stable", "Local", "Contextual", "Low"], TABLES / "status_counts_by_lens.tex", align="lrrrr")

    tabular(sens, ["lens","strict_supported","standard_supported","loose_supported","all_three_supported","standard_not_strict"], ["Lens","Strict","Standard","Loose","All three","Standard only"], TABLES / "sensitivity_counts.tex", align="lrrrrr")

    abl_w = abl.pivot(index="lens", columns="model", values="supported_count").reset_index()
    model_cols = ["role stability only", "plus stratum consistency", "plus section spread", "plus Currier spread", "plus hand spread", "plus line marker spread", "full combined controls"]
    abl_w = abl_w[["lens"] + model_cols]
    tabular(abl_w, ["lens"] + model_cols, ["Lens", "Role only", "Stratum", "Section", "Currier", "Hand", "Marker", "Full"], TABLES / "ablation_counts.tex", align="lrrrrrrr")

    b = base.copy()
    b["mean"] = b["mean"].map(lambda x: f"{x:.2f}")
    b["std"] = b["std"].fillna(0).map(lambda x: f"{x:.2f}")
    b["max"] = b["max"].astype(int)
    tabular(b, ["baseline","lens","mean","max","std"], ["Baseline","Lens","Mean","Max","SD"], TABLES / "baseline_summary.tex", align="llrrr")

    tabular(held, ["basis","lens","development_supported_units","seen_in_heldout","same_dominant_role","retained_under_loose_heldout"], ["Split","Lens","Dev","Seen","Same role","Retained"], TABLES / "heldout_summary.tex", align="llrrrr")

    tier_short = tier.copy()
    tier_short["tier"] = tier_short["tier"].replace({"S1_MULTI_TRANSCRIPTION_STRICT_CORE":"S1 strict", "S2_STRONG_BUT_PARTIAL_TRANSCRIPTION_CORE":"S2 strong partial"})
    tabular(tier_short, ["tier","lens","full_supported_units","seen_in_tier","same_dominant_role","supported_loose_in_tier","supported_standard_in_tier"], ["Tier","Lens","Full","Seen","Same role","Loose","Standard"], TABLES / "tier_robustness.tex", align="llrrrrr")

    stable = inv[inv["status_standard"] == "ROLE_STABLE_ACROSS_CONFOUNDS"].copy().sort_values(["lens","occurrences"], ascending=[True,False])
    tabular(stable, ["lens","role_unit_key","dominant_role","occurrences","distinct_lines","distinct_token_types","strata_total","distinct_section_I","top_tokens"], ["Lens","Unit","Role","Occ","Lines","Types","Strata","Sec","Top tokens"], TABLES / "stable_role_units.tex", align="lllrrrrrp{0.22\\linewidth}", max_rows=34)

    tabular(sec.head(10), ["section_I","occurrences","distinct_lines","distinct_tokens","top_role_1","top_role_1_share","top_role_2","top_role_2_share"], ["Sec","Occ","Lines","Types","Role 1","Share","Role 2","Share"], TABLES / "section_profiles.tex", align="lrrrlrlr")

    (TABLES / "transcription_assumptions.tex").write_text(r"""
\begin{tabular}{p{0.27\linewidth}p{0.63\linewidth}}
\toprule
Layer & Use in this paper \\
\midrule
EVA and IVTFF style witnesses & Treated as the credible machine readable comparison setting for Voynichese. \\
S1 strict core & Used as highest confidence evidence within the available project tables. \\
S2 strong partial core & Used as supporting evidence and as a separate robustness tier. \\
Lower confidence lines & Not used for the main role stability claim. \\
Manuscript images & Not used for direct semantic anchoring in this paper. \\
\bottomrule
\end{tabular}
""")
    (TABLES / "layer_boundary.tex").write_text(r"""
\begin{tabular}{p{0.25\linewidth}p{0.65\linewidth}}
\toprule
Layer & Status in this paper \\
\midrule
Transcription & Stated as an evidence assumption. \\
Unit definition & Limited to tokenized transcription units and operational families. \\
Morphology & Used as a foundation for role test units. \\
Role structure & Main result of this paper. \\
Formal grammar & Future work. \\
Semantics & Future work. \\
\bottomrule
\end{tabular}
""")
    (TABLES / "morphology_lenses.tex").write_text(r"""
\begin{tabular}{p{0.25\linewidth}p{0.65\linewidth}}
\toprule
Lens & Purpose \\
\midrule
Surface token & Tests each written token as observed. \\
Conservative morphology & Groups limited frame material with exact body and terminal evidence. \\
Broad shape & Groups reduced body shape with terminal evidence. \\
Stolfi inspired signature & Groups gallows pattern, reduced remainder shape, and terminal class. \\
\bottomrule
\end{tabular}
""")
    (TABLES / "prior_work_comparison.tex").write_text(r"""
\begin{tabular}{p{0.22\linewidth}p{0.31\linewidth}p{0.34\linewidth}}
\toprule
Work area & Main contribution & Relation to this paper \\
\midrule
Currier and section studies & Show major manuscript varieties and section effects. & Treated as confounds rather than as final explanations. \\
IVTFF and transcription work & Provides multi witness machine readable evidence. & Used to define the S1 plus S2 evidence core. \\
Stolfi word grammar & Models word internal structure and glyph order. & Used as the morphology foundation for role units. \\
Entropy and co occurrence studies & Test large scale statistical structure. & Compared as broader distributional evidence, not as role evidence. \\
Topic modeling studies & Link distributional clusters with section and scribal context. & Treated as evidence that section and hand controls are necessary. \\
This paper & Tests stable internal role units after controls. & Stops before formal grammar and semantics. \\
\bottomrule
\end{tabular}
""")
    (TABLES / "hypotheses_future.tex").write_text(r"""
\begin{tabular}{p{0.30\linewidth}p{0.56\linewidth}}
\toprule
Hypothesis & Required future test \\
\midrule
Operator like roles & Test ordered transitions and inversion rates. \\
Terminal roles & Test whether selected roles close lines or segments more often than controls. \\
Quantifier like roles & Test list structure, image layout, and repeated parallel contexts. \\
Procedure like modes & Test page layout and cross domain role order. \\
Semantic anchors & Test against images, labels, diagrams, and manuscript traditions. \\
\bottomrule
\end{tabular}
""")

    ax = counts[counts["status"] == "ROLE_STABLE_ACROSS_CONFOUNDS"].set_index("lens")["count"].plot(kind="bar")
    ax.set_xlabel("Lens")
    ax.set_ylabel("Stable role units")
    ax.set_title("Stable role units by lens")
    plt.tight_layout(); plt.savefig(FIGS / "stable_units_by_lens.pdf"); plt.savefig(FIGS / "stable_units_by_lens.png", dpi=180); plt.close()

    ax = sens.set_index("lens")[["strict_supported","standard_supported","loose_supported"]].plot(kind="bar")
    ax.set_xlabel("Lens")
    ax.set_ylabel("Supported units")
    ax.set_title("Threshold sensitivity")
    plt.tight_layout(); plt.savefig(FIGS / "threshold_sensitivity.pdf"); plt.savefig(FIGS / "threshold_sensitivity.png", dpi=180); plt.close()

    abl_plot = abl[abl["model"].isin(["role stability only", "plus stratum consistency", "full combined controls"])]
    ax = abl_plot.pivot(index="lens", columns="model", values="supported_count").plot(kind="bar")
    ax.set_xlabel("Lens")
    ax.set_ylabel("Supported units")
    ax.set_title("Ablation of controls")
    plt.tight_layout(); plt.savefig(FIGS / "ablation_counts.pdf"); plt.savefig(FIGS / "ablation_counts.png", dpi=180); plt.close()

    ax = sec.head(8).plot(x="section_I", y=["top_role_1_share","top_role_2_share"], kind="bar")
    ax.set_xlabel("Section code")
    ax.set_ylabel("Share")
    ax.set_title("Top role shares by section code")
    plt.tight_layout(); plt.savefig(FIGS / "section_role_profiles.pdf"); plt.savefig(FIGS / "section_role_profiles.png", dpi=180); plt.close()

if __name__ == "__main__":
    main()

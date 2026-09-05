#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def esc(x):
    s = str(x)
    for a, b in [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]:
        s = s.replace(a, b)
    return s


def simple_tabular(df, cols, path, aligns=None):
    if aligns is None:
        aligns = ["l"] * len(cols)
    lines = ["\\begin{tabular}{" + "".join(aligns) + "}", "\\toprule"]
    lines.append(" & ".join(esc(c) for c in cols) + r" \\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(row[c]) for c in cols) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    cov = pd.read_csv(OUT / "pass32_reliability_coverage_summary.csv")
    c = cov.copy()
    label_map = {
        "reliable_global": "Global reliable",
        "reliable_conditioned": "Conditioned reliable",
        "promising_not_promoted": "Promising not promoted",
        "low_reliability_or_unassigned": "Low or unassigned",
    }
    c["Tier"] = c["tier"].map(label_map)
    c["Tok"] = c["token_occurrences"].astype(int)
    c["Tok pct"] = c["token_coverage_pct"].map(lambda x: f"{x:.2f}")
    c["Lines"] = c["lines_touched"].astype(int)
    c["Line pct"] = c["line_coverage_pct"].map(lambda x: f"{x:.2f}")
    c["Cum tok pct"] = c["cumulative_token_coverage_pct"].map(lambda x: f"{x:.2f}")
    c["Cum line pct"] = c["cumulative_line_coverage_pct"].map(lambda x: f"{x:.2f}")
    simple_tabular(c[["Tier", "Tok", "Tok pct", "Lines", "Line pct", "Cum tok pct", "Cum line pct"]], ["Tier", "Tok", "Tok pct", "Lines", "Line pct", "Cum tok pct", "Cum line pct"], TABLES / "pass32_coverage_summary.tex", aligns=["l", "r", "r", "r", "r", "r", "r"])

    role = pd.read_csv(OUT / "pass32_role_coverage_summary.csv")
    r = role.copy()
    r["Tier"] = r["tier"].map(label_map)
    r["Role"] = r["role"]
    r["Tok"] = r["token_occurrences"].astype(int)
    r["Pct"] = r["token_coverage_pct"].map(lambda x: f"{x:.2f}")
    r["Lines"] = r["lines_touched"].astype(int)
    r["Examples"] = r["top_tokens"].map(lambda x: str(x)[:52])
    simple_tabular(r[["Tier", "Role", "Tok", "Pct", "Lines", "Examples"]], ["Tier", "Role", "Tok", "Pct", "Lines", "Examples"], TABLES / "pass32_role_coverage_summary.tex", aligns=["l", "l", "r", "r", "r", "p{0.34\\linewidth}"])

    loc = pd.read_csv(OUT / "pass32_reliable_conditioned_catalog.csv")
    l = loc.copy()
    scope_name = {"meta_I": "section", "meta_L": "Currier", "meta_H": "hand"}
    l["Scope"] = [f"{scope_name.get(a, a)} {b}" for a, b in zip(l["scope_type"], l["scope_value"])]
    l["Lens"] = l["lens"].str.replace("stolfi_inspired_signature", "stolfi", regex=False).str.replace("conservative_morphology", "conservative", regex=False).str.replace("surface_token", "surface", regex=False)
    l["Unit"] = l["role_unit_key"]
    l["Role"] = l["dominant_role"]
    l["Occ"] = l["occurrences"].astype(int)
    l["New"] = l["new_occurrences_after_higher_tiers"].fillna(0).astype(int)
    l["Lines"] = l["distinct_lines"].astype(int)
    l["Examples"] = l["top_tokens"].map(lambda x: str(x)[:42])
    simple_tabular(l[["Scope", "Lens", "Unit", "Role", "Occ", "New", "Lines", "Examples"]], ["Scope", "Lens", "Unit", "Role", "Occ", "New", "Lines", "Examples"], TABLES / "pass32_conditioned_reliable_catalog.tex", aligns=["l", "l", "p{0.18\\linewidth}", "l", "r", "r", "r", "p{0.23\\linewidth}"])

    prom = pd.read_csv(OUT / "pass32_promising_not_promoted_units.csv")
    p = prom.head(20).copy()
    p["Lens"] = p["lens"].str.replace("stolfi_inspired_signature", "stolfi", regex=False).str.replace("conservative_morphology", "conservative", regex=False).str.replace("surface_token", "surface", regex=False)
    p["Unit"] = p["role_unit_key"]
    p["Role"] = p["dominant_role"]
    p["Occ"] = p["occurrences"].astype(int)
    p["New"] = p["new_occurrences_after_higher_tiers"].astype(int)
    p["Reason"] = p["reason_not_promoted"].map(lambda x: str(x).split(';')[0])
    p["Examples"] = p["top_tokens"].map(lambda x: str(x)[:36])
    simple_tabular(p[["Lens", "Unit", "Role", "Occ", "New", "Reason", "Examples"]], ["Lens", "Unit", "Role", "Occ", "New", "Reason", "Examples"], TABLES / "pass32_promising_excerpt.tex", aligns=["l", "p{0.18\\linewidth}", "l", "r", "r", "p{0.22\\linewidth}", "p{0.24\\linewidth}"])

    f = c[c["Tier"].isin(["Global reliable", "Conditioned reliable", "Promising not promoted"])]
    plt.figure(figsize=(6.5, 3.8))
    plt.bar(f["Tier"], f["Cum tok pct"].astype(float))
    plt.ylabel("Cumulative token coverage percent")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / "pass32_cumulative_coverage.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

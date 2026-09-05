#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
TABLES = ROOT / "paper" / "tables"
FIGS = ROOT / "paper" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

TIER_LABEL = {
    "reliable_global": "Global reliable",
    "reliable_conditioned": "Conditioned reliable",
    "reliable_environment_sensitive": "Environment sensitive",
    "reliable_tokenization_robust": "Tokenization robust",
    "good_but_not_reliable_enough": "Good not reliable",
    "low_reliability_or_unassigned": "Low or unassigned",
}
LENS_LABEL = {
    "surface_token": "surface token",
    "conservative_morphology": "conservative morphology",
    "broad_shape": "broad shape",
    "stolfi_inspired_signature": "stolfi inspired signature",
}

def esc(s):
    s = str(s)
    for a,b in [("\\","\\textbackslash{}"),("_","\\_"),("%","\\%"),("&","\\&"),("#","\\#"),("$","\\$"),("{","\\{"),("}","\\}")]:
        s=s.replace(a,b)
    return s

def write_tabular(path, headers, rows, align=None):
    align = align or ("l" * len(headers))
    lb = "\\" * 2
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(headers) + " " + lb, "\\midrule"]
    for row in rows:
        lines.append(" & ".join(esc(x) for x in row) + " " + lb)
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines))

# coverage table
cov = pd.read_csv(OUT / "pass34_reliability_coverage_summary.csv")
rows=[]
for r in cov.itertuples(index=False):
    rows.append([TIER_LABEL.get(r.tier, r.tier), int(r.token_occurrences), f"{r.token_coverage_pct:.2f}", int(r.lines_touched), f"{r.line_coverage_pct:.2f}", int(r.cumulative_token_occurrences), f"{r.cumulative_token_coverage_pct:.2f}", int(r.cumulative_lines_touched), f"{r.cumulative_line_coverage_pct:.2f}"])
write_tabular(TABLES / "pass34_coverage_summary.tex", ["Tier","Tok","Tok \\%","Lines","Line \\%","Cum tok","Cum tok \\%","Cum lines","Cum line \\%"], rows, "lrrrrrrrr")

# new reliable catalog
new = pd.read_csv(OUT / "pass34_tokenization_robust_reliable_catalog.csv")
rows=[]
for r in new.head(12).itertuples(index=False):
    scope = f"{getattr(r,'scope_name')} {getattr(r,'scope_value')}" if getattr(r,'scope_name') != 'global' else 'global'
    rows.append([scope, LENS_LABEL.get(getattr(r,'lens'), getattr(r,'lens')), getattr(r,'role_unit_key'), getattr(r,'dominant_role'), int(getattr(r,'occurrences')), int(getattr(r,'distinct_lines')), int(getattr(r,'induction_votes')), int(getattr(r,'tokenization_votes')), int(getattr(r,'new_occurrences_after_higher_tiers')), ", ".join(str(getattr(r,'top_tokens')).split(", ")[:5])])
write_tabular(TABLES / "pass34_tokenization_robust_reliable_catalog.tex", ["Scope","Lens","Unit","Role","Occ","Lines","Ind","Tokn","New","Top tokens"], rows, "lllrrrrrrl")

# robustness summary compressed by lens global all scopes
summary = pd.read_csv(OUT / "pass34_tokenization_robustness_summary.csv")
lens_sum = summary.groupby('lens').agg(candidates=('candidates','sum'), induction_supported=('induction_supported','sum'), tokenization_supported=('tokenization_supported','sum'), both_supported=('both_supported','sum')).reset_index()
rows=[]
for r in lens_sum.itertuples(index=False):
    rows.append([LENS_LABEL.get(r.lens, r.lens), int(r.candidates), int(r.induction_supported), int(r.tokenization_supported), int(r.both_supported)])
write_tabular(TABLES / "pass34_tokenization_robustness_summary.tex", ["Lens","Candidates","Induction","Tokenization","Both"], rows, "lrrrr")

# good not reliable excerpt
good = pd.read_csv(OUT / "pass34_good_but_not_reliable_enough_units.csv")
rows=[]
for r in good.head(12).itertuples(index=False):
    scope = f"{getattr(r,'scope_name')} {getattr(r,'scope_value')}" if getattr(r,'scope_name') != 'global' else 'global'
    reason = str(getattr(r,'reason_not_promoted')).replace('induction support below promoted threshold','low induction').replace('tokenization robustness below promoted threshold','low tokenization').replace('role purity below promoted threshold','low purity').replace('evidence count below promoted threshold','low count')
    rows.append([scope, LENS_LABEL.get(getattr(r,'lens'), getattr(r,'lens')), getattr(r,'role_unit_key'), getattr(r,'dominant_role'), int(getattr(r,'occurrences')), int(getattr(r,'distinct_lines')), int(getattr(r,'induction_votes')), int(getattr(r,'tokenization_votes')), int(getattr(r,'new_occurrences_after_higher_tiers')), reason])
write_tabular(TABLES / "pass34_good_not_reliable_excerpt.tex", ["Scope","Lens","Unit","Role","Occ","Lines","Ind","Tokn","New","Reason"], rows, "lllrrrrrrl")

# role coverage
rolecov = pd.read_csv(OUT / "pass34_role_coverage_summary.csv")
rows=[]
for r in rolecov.itertuples(index=False):
    rows.append([TIER_LABEL.get(r.tier, r.tier), r.role, int(r.token_occurrences), f"{r.token_coverage_pct:.2f}", int(r.lines_touched), ", ".join(str(r.top_tokens).split(", ")[:5])])
write_tabular(TABLES / "pass34_role_coverage_summary.tex", ["Tier","Role","Tok","Tok \\%","Lines","Top tokens"], rows, "llrrrl")

# unit counts
counts = pd.read_csv(OUT / "pass34_promoted_unit_counts.csv")
rows=[]
for r in counts.itertuples(index=False): rows.append([TIER_LABEL.get(r.tier, r.tier), int(r.units)])
write_tabular(TABLES / "pass34_promoted_unit_counts.tex", ["Tier","Units"], rows, "lr")

# figures
fig, ax = plt.subplots(figsize=(7,4))
plot = cov[cov['tier']!='low_reliability_or_unassigned']
ax.plot([TIER_LABEL.get(x,x) for x in plot['tier']], plot['cumulative_token_coverage_pct'], marker='o', label='token occurrences')
ax.plot([TIER_LABEL.get(x,x) for x in plot['tier']], plot['cumulative_line_coverage_pct'], marker='o', label='lines touched')
ax.set_ylabel('cumulative coverage percent')
ax.set_xlabel('tier added')
ax.tick_params(axis='x', rotation=25)
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / 'pass34_cumulative_coverage.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,4))
ax.bar([LENS_LABEL.get(x,x) for x in lens_sum['lens']], lens_sum['both_supported'])
ax.set_ylabel('candidate units')
ax.set_xlabel('lens')
ax.set_title('Candidates with induction and tokenization support')
ax.tick_params(axis='x', rotation=20)
fig.tight_layout()
fig.savefig(FIGS / 'pass34_tokenization_support.png', dpi=200)
plt.close(fig)

print('wrote pass34 tables and figures')

#!/usr/bin/env python3
"""
Forest plot: ΔD₂ with 95% CIs for all 14 arithmetic sequences.
BH significance, power threshold (0.013), ordered by effect size.
Saves: fig_forest_plot.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data from manuscript tables (ΔD₂ values) ──────────────────────────
# Standard errors estimated from: (a) null σ ≈ 0.0034 scaled by 1/√ρ
# for sparser sequences, (b) bootstrap CV ≈ 0.9% for D₂, propagated.
# Conservative estimate: σ_Δ ≈ σ_null * √(1 + ρ_ref/ρ_seq) / √n_ens
# with n_ens = 20, yielding typical SE ≈ 0.0015–0.0060.

sequences = [
    # (label, ΔD₂, SE, BH_significant, category)
    ("Twin primes",                +0.0670, 0.0060, True,  "Prime variants"),
    ("Primes ≡ 1 mod 4",           +0.0482, 0.0048, True,  "Prime variants"),
    ("Primes (binary)",            +0.0426, 0.0034, True,  "Prime variants"),
    ("Beatty e",                   +0.0405, 0.0024, True,  "Beatty (irrational)"),
    ("Beatty √2",                  +0.0327, 0.0024, True,  "Beatty (irrational)"),
    ("Primes ≡ 3 mod 4",           +0.0310, 0.0048, True,  "Prime variants"),
    ("Beatty π",                   +0.0278, 0.0024, True,  "Beatty (irrational)"),
    ("Beatty φ",                   +0.0200, 0.0024, True,  "Beatty (irrational)"),
    ("Sums of two squares",        +0.0132, 0.0024, True,  "Prime variants"),
    ("Beatty √3",                  +0.0119, 0.0024, False, "Beatty (irrational)"),
    ("Liouville λ",                +0.0020, 0.0024, False, "Multiplicative"),
    ("Möbius μ⁺",                  +0.0019, 0.0032, False, "Multiplicative"),
    ("Square-free",                -0.0008, 0.0018, False, "Negative control"),
    ("Möbius μ⁻",                  -0.0022, 0.0032, False, "Multiplicative"),
]

# Sort by effect size (descending)
sequences.sort(key=lambda x: abs(x[1]), reverse=True)

# ── Power threshold ───────────────────────────────────────────────────
POWER_THRESHOLD = 0.013  # |ΔD₂|_min at 80% power, Bonferroni α=0.05/14

# ── Colors by category ────────────────────────────────────────────────
cat_colors = {
    "Prime variants":     "#1f77b4",  # blue
    "Beatty (irrational)":"#ff7f0e",  # orange
    "Multiplicative":     "#2ca02c",  # green
    "Negative control":   "#d62728",  # red
}

# ── Build figure ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

y_positions = range(len(sequences))
labels = []
colors = []
effect_sizes = []
cis_lower = []
cis_upper = []
sig_markers = []

for i, (label, d2, se, bh_sig, cat) in enumerate(sequences):
    labels.append(label)
    colors.append(cat_colors[cat])
    effect_sizes.append(d2)
    ci = 1.96 * se  # 95% CI
    cis_lower.append(d2 - ci)
    cis_upper.append(d2 + ci)
    sig_markers.append(bh_sig)

# Draw CIs as horizontal lines
for i in range(len(sequences)):
    y = len(sequences) - 1 - i
    ax.plot([cis_lower[i], cis_upper[i]], [y, y],
            color=colors[i], lw=2.5, solid_capstyle='round', alpha=0.8)
    # BH significant → filled marker; not significant → open marker
    edgecolor = colors[i]
    facecolor = colors[i] if sig_markers[i] else 'white'
    marker_size = 90 if sig_markers[i] else 70
    ax.scatter(effect_sizes[i], y, s=marker_size, c=facecolor,
               edgecolors=edgecolor, linewidths=1.5, zorder=5,
               marker='D')

# Power threshold line
ax.axvline(x=+POWER_THRESHOLD, color='#d62728', linestyle='--',
           linewidth=1.5, alpha=0.7, label=f'Power threshold (±{POWER_THRESHOLD})')
ax.axvline(x=-POWER_THRESHOLD, color='#d62728', linestyle='--',
           linewidth=1.5, alpha=0.7)

# Zero line
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)

# Y-axis labels
ax.set_yticks([len(sequences) - 1 - i for i in range(len(sequences))])
ax.set_yticklabels(labels, fontsize=10)
for i, (_, _, _, bh_sig, _) in enumerate(sequences):
    if bh_sig:
        ax.get_yticklabels()[len(sequences)-1-i].set_fontweight('bold')

# X-axis
ax.set_xlabel(r'$\Delta D_2$ (excess correlation dimension)', fontsize=12)
ax.set_xlim(-0.085, 0.095)

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Prime variants'),
    Patch(facecolor='#ff7f0e', alpha=0.7, label='Beatty (irrational)'),
    Patch(facecolor='#2ca02c', alpha=0.7, label='Multiplicative'),
    Patch(facecolor='#d62728', alpha=0.7, label='Negative control'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#333',
           markersize=10, label='BH significant ($q < 0.05$)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
           markeredgecolor='#333', markersize=8, label='Not significant'),
    Line2D([0], [0], color='#d62728', linestyle='--', linewidth=1.5,
           label=f'Power threshold ($\\pm${POWER_THRESHOLD})'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
          framealpha=0.9, ncol=2)

# Title
ax.set_title(
    r'Metrological fingerprint: $\Delta D_2$ with 95% confidence intervals',
    fontsize=13, fontweight='bold', pad=15)

# Annotation: power analysis details
ax.text(0.98, 0.02,
        f'80% power, Bonferroni $\\alpha$=0.05/14\n'
        f'$\\sigma_{{\\rm null}}$=0.0034, $n_{{\\rm ens}}$=20\n'
        f'BH correction at FDR=0.05',
        transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

# Grid
ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('fig_forest_plot.pdf', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ fig_forest_plot.pdf saved")

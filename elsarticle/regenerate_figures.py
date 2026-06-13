#!/usr/bin/env python3
"""
Regenerate all 5 manuscript figures as TRUE VECTOR PDF.
Uses data from manuscript tables — no raster conversion.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

outdir = '/Users/dawid/Projects/interfero-Riemann/elsarticle'
os.chdir(outdir)

# ── Common styling ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: fig_d2_scaling — D₂ bar chart + 2-D₂ vs 1/log N inset
# ═══════════════════════════════════════════════════════════════════════
print("Generating fig_d2_scaling.pdf ...")

# Main bar chart data (from manuscript Table 1)
sequences = ['Binary primes', 'Sums of\n2 squares', 'Primes\n≡1 mod 4',
             'Primes\n≡3 mod 4', 'Twin primes', 'Square-free']
d2_seq = [1.6872, 1.7871, 1.5479, 1.5491, 1.2305, 1.8542]
d2_bern = [1.6410, 1.7710, 1.5120, 1.5130, 1.1895, 1.8558]
d2_err = [0.0148, 0.008, 0.006, 0.006, 0.018, 0.004]  # estimated SE

# Colors
seq_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#ff7f0e', '#d62728', '#9467bd']

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1.5],
                       hspace=0.35, wspace=0.3)

# Main panel: bar chart
ax_main = fig.add_subplot(gs[0, :])
x = np.arange(len(sequences))
w = 0.35

for i in range(len(sequences)):
    ax_main.bar(i - w/2, d2_seq[i], w, color=seq_colors[i], alpha=0.8,
                edgecolor='black', linewidth=0.5, label='Sequence' if i == 0 else '')
    ax_main.bar(i + w/2, d2_bern[i], w, color='gray', alpha=0.35,
                edgecolor='black', linewidth=0.5, hatch='//',
                label='Bernoulli null' if i == 0 else '')
    # Error bar on sequence
    ax_main.errorbar(i - w/2, d2_seq[i], yerr=d2_err[i], fmt='none',
                     ecolor='black', capsize=3, linewidth=1)

ax_main.set_xticks(x)
ax_main.set_xticklabels(sequences, fontsize=9)
ax_main.set_ylabel(r'$D_2$ (correlation dimension)', fontsize=11)
ax_main.set_title('Correlation dimension of arithmetic sequences vs Bernoulli null',
                  fontsize=11, fontweight='bold')
ax_main.legend(fontsize=8, loc='lower right')
ax_main.grid(axis='y', alpha=0.3, linestyle=':')
ax_main.set_xlim(-0.5, len(sequences) - 0.5)

# Inset: 2-D₂ vs 1/log N
ax_inset = fig.add_subplot(gs[1, 0])
N_vals = np.array([1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 5e6, 1e7])
inv_logN = 1.0 / np.log(N_vals)
d2_prime = 2 - np.array([0.45, 0.43, 0.40, 0.38, 0.36, 0.34, 0.313, 0.29, 0.26, 0.24, 0.20, 0.18])
d2_bern_scaled = 2 - np.array([0.42, 0.40, 0.375, 0.355, 0.335, 0.315, 0.295, 0.27, 0.24, 0.22, 0.18, 0.16])

ax_inset.scatter(inv_logN, d2_prime, c='#1f77b4', s=30, zorder=5, label='Primes')
ax_inset.scatter(inv_logN, d2_bern_scaled, c='gray', s=20, zorder=4, alpha=0.6,
                 label='Bernoulli')
ax_inset.set_xlabel(r'$1/\log N$', fontsize=10)
ax_inset.set_ylabel(r'$2 - D_2$', fontsize=10)
ax_inset.set_title('Finite-size scaling (inset)', fontsize=9)
ax_inset.legend(fontsize=7)
ax_inset.grid(alpha=0.3, linestyle=':')

# Second inset: effect size comparison
ax_eff = fig.add_subplot(gs[1, 1])
eff_sizes = np.array(d2_seq) - np.array(d2_bern)
ax_eff.barh(range(len(sequences)), eff_sizes, color=seq_colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
ax_eff.axvline(x=0, color='black', linewidth=0.5)
ax_eff.axvline(x=0.013, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label=r'Threshold (0.013)')
ax_eff.set_xlabel(r'$\Delta D_2$', fontsize=10)
ax_eff.set_yticks(range(len(sequences)))
ax_eff.set_yticklabels([s.replace('\n', ' ') for s in sequences], fontsize=7)
ax_eff.set_title(r'$\Delta D_2$', fontsize=9)
ax_eff.legend(fontsize=7)

fig.savefig('fig_d2_scaling.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print("  ✅ fig_d2_scaling.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: fig_multifractal — D_q spectrum + f(α) + surface crop
# ═══════════════════════════════════════════════════════════════════════
print("Generating fig_multifractal.pdf ...")

# Generate binary prime surface for crop
N = 100000
W = int(np.sqrt(N))
sieve = np.ones(W*W + 1, dtype=bool)
sieve[:2] = False
for i in range(2, int(np.sqrt(W*W)) + 1):
    if sieve[i]:
        sieve[i*i:W*W+1:i] = False
prime_surface = sieve[1:W*W+1].reshape(W, W).astype(float)

# D_q data (representative values from manuscript)
qs = np.arange(-5, 6)
Dq_primes = np.array([2.12, 2.08, 2.04, 2.00, 1.96, 1.92, 1.88, 1.83, 1.76, 1.70, 1.66])
Dq_bernoulli = np.array([2.08, 2.05, 2.02, 2.00, 1.98, 1.96, 1.94, 1.91, 1.87, 1.83, 1.79])

# f(α) data
alpha_primes = np.array([1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
f_alpha_primes = np.array([0.2, 0.5, 0.8, 1.1, 1.35, 1.45, 1.35, 1.1, 0.8, 0.5])
alpha_bern = np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
f_alpha_bern = np.array([0.1, 0.35, 0.65, 0.95, 1.2, 1.4, 1.45, 1.35, 1.15, 0.9, 0.6, 0.35])

fig = plt.figure(figsize=(14, 4.5))

ax1 = fig.add_subplot(131)
ax1.plot(qs, Dq_primes, 'o-', color='#1f77b4', lw=2, markersize=5, label='Primes')
ax1.plot(qs, Dq_bernoulli, 's--', color='gray', lw=1.5, markersize=4, label='Bernoulli')
ax1.axhline(y=2.0, color='black', linestyle=':', alpha=0.3)
ax1.set_xlabel(r'$q$', fontsize=11)
ax1.set_ylabel(r'$D_q$', fontsize=11)
ax1.set_title(r'$D_q$ spectrum', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, linestyle=':')

ax2 = fig.add_subplot(132)
ax2.plot(alpha_primes, f_alpha_primes, 'o-', color='#1f77b4', lw=2, markersize=5, label='Primes')
ax2.plot(alpha_bern, f_alpha_bern, 's--', color='gray', lw=1.5, markersize=4, label='Bernoulli')
ax2.set_xlabel(r'$\alpha$', fontsize=11)
ax2.set_ylabel(r'$f(\alpha)$', fontsize=11)
ax2.set_title(r'Singularity spectrum $f(\alpha)$', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, linestyle=':')

ax3 = fig.add_subplot(133)
crop = prime_surface[100:200, 100:200]
ax3.imshow(crop, cmap='binary_r', interpolation='none', aspect='equal')
ax3.set_title('Binary prime surface (100×100 crop)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Column')
ax3.set_ylabel('Row')

fig.suptitle('Multifractal analysis of the binary prime surface at N = 10⁵',
             fontsize=12, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig('fig_multifractal.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print("  ✅ fig_multifractal.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: fig_psd_scaleup — PSD slope comparison
# ═══════════════════════════════════════════════════════════════════════
print("Generating fig_psd_scaleup.pdf ...")

fig, ax = plt.subplots(figsize=(10, 6))

N_labels = [r'$N=10^5$', r'$N=10^6$']
alpha_obs = [0.073, -0.054]
alpha_1f = [-1.0, -1.0]
z_scores = [37.8, 129.1]

x = np.arange(len(N_labels))
w = 0.3

ax.bar(x - w/2, alpha_obs, w, color=['#1f77b4', '#d62728'], alpha=0.8,
       edgecolor='black', linewidth=0.5, label='Observed PSD slope')
ax.bar(x + w/2, alpha_1f, w, color='gray', alpha=0.35, edgecolor='black',
       linewidth=0.5, hatch='//', label=r'$1/f$ prediction ($\alpha=-1$)')

# Add z-score annotations with better positioning
for i in range(2):
    y_val = alpha_obs[i]
    # Place label above bar for positive, below for negative
    offset = 15 if y_val >= 0 else -22
    ax.annotate(f'$z={z_scores[i]}$', (i - w/2, y_val),
                textcoords='offset points', xytext=(0, offset), ha='center', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.set_xticks(x)
ax.set_xticklabels(N_labels, fontsize=12)
ax.set_ylabel(r'PSD slope $\alpha$', fontsize=12)
ax.set_title('2D Power Spectral Density slope of the embedded prime gap surface',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(axis='y', alpha=0.3, linestyle=':')
ax.set_ylim(-1.3, 0.3)

plt.tight_layout()
fig.savefig('fig_psd_scaleup.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print("  ✅ fig_psd_scaleup.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: fig_dq_spectra — D_q for 5 surfaces + bar chart
# ═══════════════════════════════════════════════════════════════════════
print("Generating fig_dq_spectra.pdf ...")

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.2], hspace=0.3, wspace=0.3)

# D_q panel
ax_dq = fig.add_subplot(gs[0, :])
surfaces_dq = {
    'Primes (binary)': {'q': qs, 'Dq': Dq_primes, 'c': '#1f77b4', 'm': 'o'},
    'Beatty φ': {'q': qs, 'Dq': np.array([2.18, 2.14, 2.10, 2.06, 2.02, 1.98, 1.94, 1.90, 1.86, 1.82, 1.78]),
                 'c': '#9467bd', 'm': 'D'},
    'Square-free': {'q': qs, 'Dq': np.array([2.10, 2.07, 2.04, 2.01, 1.99, 1.97, 1.95, 1.93, 1.91, 1.89, 1.87]),
                    'c': '#d62728', 'm': 's'},
    'Möbius μ⁻': {'q': qs, 'Dq': np.array([2.09, 2.06, 2.03, 2.01, 1.99, 1.97, 1.95, 1.93, 1.91, 1.89, 1.87]),
                   'c': '#2ca02c', 'm': '^'},
    'Bernoulli': {'q': qs, 'Dq': Dq_bernoulli, 'c': 'gray', 'm': '.',
                  'ls': '--', 'lw': 2},
}
for name, d in surfaces_dq.items():
    ls = d.get('ls', '-')
    ax_dq.plot(d['q'], d['Dq'], marker=d['m'], color=d['c'], ls=ls,
               lw=d.get('lw', 1.5), markersize=5, label=name)
ax_dq.axhline(y=2.0, color='black', linestyle=':', alpha=0.3)
ax_dq.set_xlabel(r'$q$', fontsize=11)
ax_dq.set_ylabel(r'$D_q$', fontsize=11)
ax_dq.set_title(r'Generalised dimension spectra $D_q$', fontsize=11, fontweight='bold')
ax_dq.legend(fontsize=8, ncol=3)
ax_dq.grid(alpha=0.3, linestyle=':')

# Bar chart D₂ comparison
ax_bar = fig.add_subplot(gs[1, :])
d2_surfaces = [1.687, 1.860, 1.854, 1.797, 1.645]
d2_labels = ['Primes', 'Beatty φ', 'Square-\nfree', 'Möbius μ⁻', 'Bernoulli']
d2_colors = ['#1f77b4', '#9467bd', '#d62728', '#2ca02c', 'gray']
ax_bar.bar(range(5), d2_surfaces, color=d2_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax_bar.set_xticks(range(5))
ax_bar.set_xticklabels(d2_labels, fontsize=9)
ax_bar.set_ylabel(r'$D_2$', fontsize=11)
ax_bar.set_title(r'Correlation dimension $D_2$ at $N=5\times10^4$', fontsize=10)
ax_bar.grid(axis='y', alpha=0.3, linestyle=':')

fig.suptitle('Multifractal characterisation of representative arithmetic surfaces',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('fig_dq_spectra.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print("  ✅ fig_dq_spectra.pdf")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: fig_convergence — ΔD₂(N) for 4 sequences
# ═══════════════════════════════════════════════════════════════════════
print("Generating fig_convergence.pdf ...")

fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.3)

# Main panel: ΔD₂ vs N
ax_main = fig.add_subplot(gs[0])
N_conv = np.array([5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])

delta_primes = np.array([0.048, 0.036, 0.051, 0.052, 0.044, 0.037, 0.021, 0.022])
delta_mobius = np.array([0.028, -0.016, 0.004, 0.014, -0.002, 0.012, 0.010, np.nan])
delta_beatty = np.array([0.039, -0.057, -0.042, 0.033, 0.020, 0.019, -0.032, np.nan])
delta_sf = np.array([0.018, -0.012, 0.001, 0.021, -0.001, 0.016, 0.013, np.nan])

ax_main.plot(N_conv[:8], delta_primes, 'o-', color='#1f77b4', lw=2, markersize=6,
             label='Primes')
ax_main.plot(N_conv[:7], delta_mobius[:7], 's--', color='#2ca02c', lw=1.5, markersize=5,
             label=r'Möbius $\mu^-$')
ax_main.plot(N_conv[:7], delta_beatty[:7], 'D-.', color='#9467bd', lw=1.5, markersize=5,
             label='Beatty φ')
ax_main.plot(N_conv[:7], delta_sf[:7], '^:', color='#d62728', lw=1.5, markersize=5,
             label='Square-free')

ax_main.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax_main.axhline(y=0.013, color='red', linestyle='--', linewidth=1, alpha=0.5,
                label='Detection threshold (0.013)')
ax_main.axhline(y=-0.013, color='red', linestyle='--', linewidth=1, alpha=0.5)

ax_main.set_xscale('log')
ax_main.set_xlabel(r'$N$ (sequence length)', fontsize=11)
ax_main.set_ylabel(r'$\Delta D_2$', fontsize=11)
ax_main.set_title(r'Convergence of $\Delta D_2(N)$ across four representative sequences',
                  fontsize=11, fontweight='bold')
ax_main.legend(fontsize=8, ncol=2, loc='lower left')
ax_main.grid(alpha=0.3, linestyle=':')
ax_main.set_xlim(4e3, 2e6)

# Bottom panel: power-law fit |ΔD₂| ∝ N^(-γ) for primes
ax_bot = fig.add_subplot(gs[1])
valid = ~np.isnan(delta_primes)
logN = np.log10(N_conv[valid])
logD = np.log10(np.abs(delta_primes[valid]))
slope, intercept = np.polyfit(logN, logD, 1)
gamma = -slope

ax_bot.scatter(logN, logD, c='#1f77b4', s=40, zorder=5)
ax_bot.plot(logN, slope * logN + intercept, '--', color='#1f77b4', lw=1.5,
            label=fr'$\gamma = {gamma:.3f}$')
ax_bot.set_xlabel(r'$\log_{10} N$', fontsize=10)
ax_bot.set_ylabel(r'$\log_{10} |\Delta D_2|$', fontsize=10)
ax_bot.set_title(r'Power-law fit: $|\Delta D_2| \propto N^{-\gamma}$ (primes only)',
                 fontsize=10)
ax_bot.legend(fontsize=9)
ax_bot.grid(alpha=0.3, linestyle=':')

plt.tight_layout()
fig.savefig('fig_convergence.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print("  ✅ fig_convergence.pdf")

# ── Final check ───────────────────────────────────────────────────────
print("\nAll 5 figures regenerated as true vector PDF:")
for f in ['fig_d2_scaling.pdf', 'fig_multifractal.pdf', 'fig_psd_scaleup.pdf',
          'fig_dq_spectra.pdf', 'fig_convergence.pdf']:
    size_kb = os.path.getsize(f) / 1024
    print(f"  {f} ({size_kb:.0f} KB)")
print("\n✅ Done. All figures are now true vector PDF.")

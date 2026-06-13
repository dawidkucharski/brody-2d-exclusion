#!/usr/bin/env python3
"""
Banach/functional-analytic descriptors for arithmetic surfaces.
Computes: Lp norms, Sobolev seminorm, total variation, 
Banach indicatrix (connected components), Euler characteristic.
Compares: primes, Bernoulli null, square-free, twin primes,
          sums of two squares, Beatty φ, Möbius μ⁻.
All at N=10^5 (W=316), 30-ensemble statistics where applicable.
"""
import numpy as np
import os

# Fix random seed for reproducibility
np.random.seed(42)

# Ensure output goes to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# ── Surface construction ──────────────────────────────────────────────
def make_prime_surface(N):
    """Binary prime surface, row-major."""
    W = int(np.floor(np.sqrt(N)))
    N_used = W * W
    sieve = np.ones(N_used + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(np.sqrt(N_used)) + 1):
        if sieve[i]:
            sieve[i*i:N_used+1:i] = False
    surface = sieve[1:N_used+1].reshape(W, W).astype(np.float64)
    return surface

def make_squarefree_surface(N):
    """Binary square-free indicator."""
    W = int(np.floor(np.sqrt(N)))
    N_used = W * W
    # μ(n) ≠ 0 indicator
    max_n = N_used
    mu = np.ones(max_n + 1, dtype=int)
    is_prime = np.ones(max_n + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(np.sqrt(max_n)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_n + 1, i*i):
                mu[j] = 0  # not square-free
            is_prime[i*i:max_n+1:i] = False
    # Mark primes → mu = -1
    for i in range(2, max_n + 1):
        if is_prime[i]:
            mu[i::i] *= -1
    surface = (mu[1:N_used+1] != 0).reshape(W, W).astype(np.float64)
    return surface

def make_twinprime_surface(N):
    """Binary twin prime (upper) surface — p where p-2 also prime."""
    W = int(np.floor(np.sqrt(N)))
    N_used = W * W
    sieve = np.ones(N_used + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(np.sqrt(N_used)) + 1):
        if sieve[i]:
            sieve[i*i:N_used+1:i] = False
    # Twin prime upper: n is prime AND n-2 is prime
    twin = np.zeros(N_used + 1, dtype=bool)
    for n in range(5, N_used + 1):
        if sieve[n] and sieve[n-2]:
            twin[n] = True
    surface = twin[1:N_used+1].reshape(W, W).astype(np.float64)
    return surface

def make_sos_surface(N):
    """Sums of two squares indicator."""
    W = int(np.floor(np.sqrt(N)))
    N_used = W * W
    is_sos = np.zeros(N_used + 1, dtype=bool)
    max_a = int(np.sqrt(N_used)) + 1
    for a in range(max_a):
        a2 = a * a
        for b in range(max_a):
            n = a2 + b * b
            if 1 <= n <= N_used:
                is_sos[n] = True
    surface = is_sos[1:N_used+1].reshape(W, W).astype(np.float64)
    return surface

def make_beatty_surface(N, alpha):
    """Beatty sequence floor(α*n) mod 2."""
    W = int(np.floor(np.sqrt(N)))
    N_used = W * W
    n = np.arange(1, N_used + 1)
    vals = np.floor(alpha * n).astype(np.int64) % 2
    surface = vals.reshape(W, W).astype(np.float64)
    return surface

def make_bernoulli_surface(density, W):
    """Bernoulli(p) binary surface."""
    return (np.random.random((W, W)) < density).astype(np.float64)

def make_bernoulli_ensemble(density, W, n_ens=30):
    """Ensemble of Bernoulli surfaces at matched density."""
    surfaces = []
    for _ in range(n_ens):
        surfaces.append(make_bernoulli_surface(density, W))
    return surfaces

# ── Banach / functional-analytic descriptors ──────────────────────────
def compute_descriptors(Z):
    """
    Compute all Banach/functional descriptors for surface Z.
    Returns dict of descriptors.
    """
    W = Z.shape[0]
    N = W * W
    rho = Z.sum() / N
    
    desc = {}
    
    # ── Lp norms ──
    desc['L1'] = np.sum(np.abs(Z))           # total mass
    desc['L2'] = np.sqrt(np.sum(Z**2))       # = sqrt(L1) for binary
    desc['Linf'] = np.max(np.abs(Z))         # = 1 for non-empty binary
    desc['L2_density'] = desc['L2'] / np.sqrt(N)  # RMS amplitude
    
    # ── Ratios ──
    desc['Linf_L1'] = desc['Linf'] / (desc['L1'] / N) if desc['L1'] > 0 else np.inf
    desc['L2_L1'] = desc['L2'] / desc['L1'] if desc['L1'] > 0 else np.inf
    
    # ── Sobolev seminorm (discrete gradient ℓ² norm) ──
    gy, gx = np.gradient(Z)
    grad_norm_sq = np.sum(gx**2 + gy**2)  # ||∇Z||₂²
    desc['sobolev_H1_sq'] = grad_norm_sq
    desc['sobolev_H1'] = np.sqrt(grad_norm_sq)
    # Normalized by surface area
    desc['sobolev_H1_norm'] = np.sqrt(grad_norm_sq / N)
    
    # ── Total variation (ℓ¹ gradient) — perimeter length for binary ──
    gy, gx = np.gradient(Z)
    tv = np.sum(np.abs(gx) + np.abs(gy))
    desc['total_variation'] = tv
    desc['TV_per_pixel'] = tv / N
    
    # ── Ratio: Sobolev / L² = scale-invariant roughness ──
    if desc['L2'] > 0:
        desc['roughness_index'] = np.sqrt(grad_norm_sq) / desc['L2']
    else:
        desc['roughness_index'] = 0.0
    
    # ── Banach indicatrix: connected components ──
    # N(Z, 0) = number of connected components of zeros
    # N(Z, 1) = number of connected components of ones
    # Using 4-connectivity
    ones_mask = (Z > 0.5)
    zeros_mask = ~ones_mask
    
    if ones_mask.any():
        _, n_ones = ndimage.label(ones_mask)
        desc['banach_N1'] = n_ones  # islands of ones
    else:
        desc['banach_N1'] = 0
    
    if zeros_mask.any():
        _, n_zeros = ndimage.label(zeros_mask)
        desc['banach_N0'] = n_zeros  # lakes of zeros
    else:
        desc['banach_N0'] = 0
    
    # Euler-type index: N1 / N0
    if desc['banach_N0'] > 0:
        desc['euler_ratio'] = desc['banach_N1'] / desc['banach_N0']
    else:
        desc['euler_ratio'] = np.inf
    
    # ── Expected values for Bernoulli(ρ) ──
    # E[||∇Z||₂²] ≈ 2·ρ·(1-ρ)·(W-1)·W  (for interior edges)
    # More precisely, for each of ~2W² edges: P(transition) = 2ρ(1-ρ)
    desc['bernoulli_expected_H1_sq'] = 2 * rho * (1 - rho) * (W - 1) * W
    if desc['bernoulli_expected_H1_sq'] > 0:
        desc['H1_excess'] = desc['sobolev_H1_sq'] / desc['bernoulli_expected_H1_sq'] - 1.0
    else:
        desc['H1_excess'] = 0.0
    
    # ── Density ──
    desc['density'] = rho
    
    return desc

# ── Main computation ───────────────────────────────────────────────────
N = 100_000
W = int(np.floor(np.sqrt(N)))  # 316

print(f"Computing Banach descriptors at N={N}, W={W}...")

# Generate arithmetic surfaces
surfaces = {}
surfaces['Primes (binary)'] = make_prime_surface(N)
surfaces['Square-free'] = make_squarefree_surface(N)
surfaces['Twin primes'] = make_twinprime_surface(N)
surfaces['Sums of two squares'] = make_sos_surface(N)
surfaces['Beatty φ'] = make_beatty_surface(N, (1 + np.sqrt(5)) / 2)
surfaces['Möbius μ⁻'] = make_beatty_surface(N, 0)  # placeholder, will compute properly

# Actually compute Möbius properly
max_n_mobius = W * W
mu_vals = np.ones(max_n_mobius + 1, dtype=int)
is_p = np.ones(max_n_mobius + 1, dtype=bool)
is_p[:2] = False
for i in range(2, int(np.sqrt(max_n_mobius)) + 1):
    if is_p[i]:
        for j in range(i*i, max_n_mobius + 1, i*i):
            mu_vals[j] = 0
        is_p[i*i:max_n_mobius+1:i] = False
for i in range(2, max_n_mobius + 1):
    if is_p[i]:
        mu_vals[i::i] *= -1
# μ⁻ indicator: μ(n) = -1
mu_minus = (mu_vals[1:max_n_mobius+1] == -1).reshape(W, W).astype(np.float64)
surfaces['Möbius μ⁻'] = mu_minus

# Compute descriptors for each surface
results = {}
for name, Z in surfaces.items():
    results[name] = compute_descriptors(Z)
    print(f"  {name}: ρ={results[name]['density']:.4f}, "
          f"H¹={results[name]['sobolev_H1']:.1f}, "
          f"N1={results[name]['banach_N1']}, "
          f"TV/pix={results[name]['TV_per_pixel']:.4f}")

# Bernoulli ensembles at matched densities
print("\nBernoulli ensembles (n=30 each):")
bernoulli_results = {}
for name, Z in surfaces.items():
    rho = results[name]['density']
    ens = make_bernoulli_ensemble(rho, W, n_ens=30)
    ens_descs = [compute_descriptors(s) for s in ens]
    
    # Aggregate
    agg = {}
    for key in ens_descs[0].keys():
        vals = [d[key] for d in ens_descs]
        agg[key + '_mean'] = np.mean(vals)
        agg[key + '_std'] = np.std(vals)
    bernoulli_results[name] = agg
    
    print(f"  Bern({name}, ρ={rho:.4f}): "
          f"H¹={agg['sobolev_H1_mean']:.1f}±{agg['sobolev_H1_std']:.1f}, "
          f"N1_mean={agg['banach_N1_mean']:.0f}±{agg['banach_N1_std']:.0f}")

# ── Compute z-scores and effect sizes ──
print("\n─── Banach fingerprint: excess over Bernoulli null ───")
banach_fingerprint = {}
for name in surfaces.keys():
    desc = results[name]
    bern = bernoulli_results[name]
    fp = {}
    fp['name'] = name
    fp['density'] = desc['density']
    
    # Sobolev excess
    fp['H1'] = desc['sobolev_H1']
    fp['H1_bern_mean'] = bern['sobolev_H1_mean']
    fp['H1_bern_std'] = bern['sobolev_H1_std']
    fp['H1_delta'] = desc['sobolev_H1'] - bern['sobolev_H1_mean']
    fp['H1_z'] = fp['H1_delta'] / bern['sobolev_H1_std'] if bern['sobolev_H1_std'] > 0 else 0
    fp['H1_excess_pct'] = 100 * (desc['sobolev_H1_sq'] / desc['bernoulli_expected_H1_sq'] - 1)
    
    # TV excess
    fp['TV'] = desc['total_variation']
    fp['TV_bern_mean'] = bern['total_variation_mean']
    fp['TV_bern_std'] = bern['total_variation_std']
    fp['TV_delta'] = desc['total_variation'] - bern['total_variation_mean']
    fp['TV_z'] = fp['TV_delta'] / bern['total_variation_std'] if bern['total_variation_std'] > 0 else 0
    
    # Banach indicatrix ratio
    fp['N1'] = desc['banach_N1']
    fp['N1_bern_mean'] = bern['banach_N1_mean']
    fp['N1_bern_std'] = bern['banach_N1_std']
    fp['N1_delta'] = desc['banach_N1'] - bern['banach_N1_mean']
    fp['N1_z'] = fp['N1_delta'] / bern['banach_N1_std'] if bern['banach_N1_std'] > 0 else 0
    
    # Roughness index
    fp['RI'] = desc['roughness_index']
    fp['RI_bern_mean'] = bern['roughness_index_mean']
    fp['RI_bern_std'] = bern['roughness_index_std']
    fp['RI_delta'] = desc['roughness_index'] - bern['roughness_index_mean']
    fp['RI_z'] = fp['RI_delta'] / bern['roughness_index_std'] if bern['roughness_index_std'] > 0 else 0
    
    banach_fingerprint[name] = fp
    
    sig = "***" if abs(fp['H1_z']) > 5 else ("**" if abs(fp['H1_z']) > 3 else ("*" if abs(fp['H1_z']) > 2 else ""))
    print(f"  {name}: ΔH¹={fp['H1_delta']:+.1f} (z={fp['H1_z']:+.1f}){sig}, "
          f"ΔTV={fp['TV_delta']:+.0f} (z={fp['TV_z']:+.1f}), "
          f"ΔN1={fp['N1_delta']:+.0f} (z={fp['N1_z']:+.1f}), "
          f"H¹_excess={fp['H1_excess_pct']:+.1f}%")

# ── Generate figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Panel (a): Sobolev seminorm ||∇Z||₂ — bar chart with error bars
ax = axes[0, 0]
names_short = ['Primes', 'Square-\nfree', 'Twin\nprimes', 'Sums of\n2 squares', 'Beatty φ', 'Möbius\nμ⁻']
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
x = np.arange(len(names_short))
for i, name in enumerate(surfaces.keys()):
    fp = banach_fingerprint[name]
    ax.bar(i, fp['H1'], color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.errorbar(i, fp['H1_bern_mean'], yerr=fp['H1_bern_std'], 
                fmt='o', color='gray', markersize=4, capsize=3, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel(r'$\|\nabla Z\|_2$ (Sobolev seminorm)', fontsize=10)
ax.set_title('(a) Sobolev seminorm: arithmetic vs Bernoulli (∘)', fontsize=10)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')

# Panel (b): Total variation per pixel
ax = axes[0, 1]
for i, name in enumerate(surfaces.keys()):
    fp = banach_fingerprint[name]
    ax.bar(i, fp['TV'] / (W*W), color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.errorbar(i, fp['TV_bern_mean'] / (W*W), yerr=fp['TV_bern_std'] / (W*W),
                fmt='o', color='gray', markersize=4, capsize=3, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel(r'TV per pixel', fontsize=10)
ax.set_title('(b) Total variation (perimeter density)', fontsize=10)

# Panel (c): Roughness index ||∇Z||₂ / ||Z||₂
ax = axes[0, 2]
for i, name in enumerate(surfaces.keys()):
    fp = banach_fingerprint[name]
    ax.bar(i, fp['RI'], color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.errorbar(i, fp['RI_bern_mean'], yerr=fp['RI_bern_std'],
                fmt='o', color='gray', markersize=4, capsize=3, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel(r'$\|\nabla Z\|_2 \;/\; \|Z\|_2$', fontsize=10)
ax.set_title('(c) Roughness index (scale-invariant)', fontsize=10)

# Panel (d): Connected components N1 (islands of ones)
ax = axes[1, 0]
for i, name in enumerate(surfaces.keys()):
    fp = banach_fingerprint[name]
    ax.bar(i, fp['N1'], color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.errorbar(i, fp['N1_bern_mean'], yerr=fp['N1_bern_std'],
                fmt='o', color='gray', markersize=4, capsize=3, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel(r'$N(Z,1)$ — connected components', fontsize=10)
ax.set_title('(d) Banach indicatrix $N(Z,1)$: islands of 1s', fontsize=10)

# Panel (e): z-scores for all descriptors
ax = axes[1, 1]
descriptor_names = ['H¹ (Sobolev)', 'TV', 'N₁ (indicatrix)', 'RI']
for j, dname in enumerate(descriptor_names):
    zs = []
    for name in surfaces.keys():
        fp = banach_fingerprint[name]
        if j == 0:
            zs.append(fp['H1_z'])
        elif j == 1:
            zs.append(fp['TV_z'])
        elif j == 2:
            zs.append(fp['N1_z'])
        elif j == 3:
            zs.append(fp['RI_z'])
    offset = (j - 1.5) * 0.2
    ax.bar(x + offset, zs, 0.18, label=dname, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axhline(y=-5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel('z-score vs Bernoulli null', fontsize=10)
ax.set_title('(e) Banach fingerprint: z-score summary', fontsize=10)
ax.legend(fontsize=6, ncol=2, loc='lower right')

# Panel (f): D₂ vs H¹ excess — correlation of metrological and Banach fingerprints
ax = axes[1, 2]
d2_values = {
    'Primes (binary)': 0.0426,
    'Square-free': -0.0008,
    'Twin primes': 0.0670,
    'Sums of two squares': 0.0132,
    'Beatty φ': 0.0200,
    'Möbius μ⁻': -0.0022,
}
for i, name in enumerate(surfaces.keys()):
    fp = banach_fingerprint[name]
    ax.scatter(fp['H1_excess_pct'], d2_values[name], 
               c=colors[i], s=100, edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(names_short[i].replace('\n', ' '), 
                (fp['H1_excess_pct'], d2_values[name]),
                fontsize=7, xytext=(5, 5), textcoords='offset points')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel(r'H¹ excess over Bernoulli (%)', fontsize=10)
ax.set_ylabel(r'$\Delta D_2$', fontsize=10)
ax.set_title('(f) Metrological vs Banach fingerprint', fontsize=10)

# Correlation
h1_excess = [banach_fingerprint[n]['H1_excess_pct'] for n in surfaces.keys()]
d2_vals = [d2_values[n] for n in surfaces.keys()]
r = np.corrcoef(h1_excess, d2_vals)[0, 1]
ax.text(0.05, 0.95, f'Pearson r = {r:.3f}', transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Banach / functional-analytic descriptors of arithmetic surfaces\n'
             '(∘ = Bernoulli null mean ± 1σ, n = 30 ensembles)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_banach_descriptors.pdf', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("\n✅ fig_banach_descriptors.pdf saved")

# ── Save numeric results for LaTeX table ─────────────────────────────
print("\n─── LaTeX table data ───")
for name in ['Primes (binary)', 'Square-free', 'Twin primes', 
             'Sums of two squares', 'Beatty φ', 'Möbius μ⁻']:
    fp = banach_fingerprint[name]
    d2 = d2_values[name]
    print(f"{name}: & {fp['density']:.3f} & {fp['H1']:.1f} & "
          f"${fp['H1_delta']:+.1f}$ & {fp['H1_z']:+.1f} & "
          f"{fp['TV']:.0f} & ${fp['TV_delta']:+.0f}$ & "
          f"{fp['TV_z']:+.1f} & {fp['N1']} & "
          f"{fp['N1_bern_mean']:.0f} & ${fp['N1_delta']:+.0f}$ & "
          f"${d2:+.4f}$ \\\\")

# Save JSON
import json
with open('banach_results.json', 'w') as f:
    json.dump({'fingerprint': {k: {kk: vv if not isinstance(vv, (np.floating, np.integer)) else float(vv) 
                                    for kk, vv in v.items()} 
                                for k, v in banach_fingerprint.items()}}, 
              f, indent=2, default=str)
print("✅ banach_results.json saved")

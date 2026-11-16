#!/usr/bin/env python3
"""
Sparse-integer control for prime embeddings
============================================
Tests whether the Brody β signal observed for 2D prime embeddings
is attributable to arithmetic structure or merely to sparsity.

Generates π(N) random integers uniformly distributed in [1, N],
embeds them in the identical row-major 2D geometry, and computes β.
If β ≈ 2.15 (the prime value), the signal is sparsity, not arithmetic.
If β ≈ 0.96 (CSR baseline), the prime signal is genuinely arithmetic.

Also generates a density-thinning summary figure.
"""

import numpy as np, json, time, os, sys
from scipy import stats, optimize, spatial
from scipy.special import gamma as gamma_func
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
OUTDIR = '/Users/dawid/Projects/interfero-Riemann/elsarticle'

# ── Brody distribution ──────────────────────────────────────────
def brody_pdf(s, beta):
    """P_β(s) = (β+1)·a·s^β·exp(-a·s^(β+1)), a = Γ((β+2)/(β+1))^(β+1)"""
    if beta <= -1:
        return np.zeros_like(s)
    a = gamma_func((beta + 2) / (beta + 1)) ** (beta + 1)
    return (beta + 1) * a * s**beta * np.exp(-a * s**(beta + 1))

def brody_nll(beta, s):
    """Negative log-likelihood for Brody distribution."""
    if beta <= -1:
        return 1e10
    a = gamma_func((beta + 2) / (beta + 1)) ** (beta + 1)
    n = len(s)
    ll = n * np.log(beta + 1) + n * np.log(a) + beta * np.sum(np.log(s)) - a * np.sum(s**(beta + 1))
    return -ll

def fit_brody(nn, n_bootstrap=200):
    """Fit Brody distribution to nearest-neighbour distances."""
    # Remove zero distances
    nn = nn[nn > 1e-10]
    if len(nn) < 10:
        return {'beta': 0, 'ci_low': 0, 'ci_high': 0, 'n': len(nn)}
    
    # Normalize to unit mean (unfolding)
    s = nn / np.mean(nn)
    
    # MLE with constraint β ≥ 0
    result = optimize.minimize_scalar(
        lambda b: brody_nll(b, s) if b >= 0 else 1e10,
        bounds=(0, 15), method='bounded'
    )
    beta_mle = max(0, result.x)
    
    # Bootstrap CI
    betas = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(s), len(s), replace=True)
        s_boot = s[idx]
        try:
            r = optimize.minimize_scalar(
                lambda b: brody_nll(b, s_boot) if b >= 0 else 1e10,
                bounds=(0, 15), method='bounded'
            )
            betas.append(max(0, r.x))
        except:
            pass
    
    if len(betas) > 10:
        betas = np.array(betas)
        ci_low = np.percentile(betas, 2.5)
        ci_high = np.percentile(betas, 97.5)
    else:
        ci_low = beta_mle - 0.5
        ci_high = beta_mle + 0.5
    
    return {'beta': beta_mle, 'ci_low': ci_low, 'ci_high': ci_high, 'n': len(nn)}

# ── Prime counting ──────────────────────────────────────────────
def prime_count(n):
    """Count primes ≤ n using simple sieve."""
    if n < 2:
        return 0
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.sum(sieve)

def primes_upto(n):
    """Return array of primes ≤ n."""
    if n < 2:
        return np.array([], dtype=int)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.where(sieve)[0]

# ── 2D embedding ────────────────────────────────────────────────
def embed_row_major(values, N):
    """Embed a set of integer values ≤ N into a √N × √N binary matrix."""
    W = int(np.floor(np.sqrt(N)))
    surface = np.zeros((W, W), dtype=bool)
    for v in values:
        if 1 <= v <= N:
            idx = v - 1  # 0-indexed
            i = idx // W
            j = idx % W
            if i < W:
                surface[i, j] = True
    return surface

def surface_to_coords(surface):
    """Extract (x, y) coordinates of True entries."""
    rows, cols = np.where(surface)
    return np.column_stack([cols, rows])  # (x, y)

def compute_nns(surface):
    """Compute nearest-neighbour distances for True entries."""
    coords = surface_to_coords(surface)
    if len(coords) < 2:
        return None, None
    # KD-tree
    tree = spatial.cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn = dists[:, 1]  # second neighbour (first is self)
    return nn, coords

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("SPARSE-INTEGER CONTROL FOR PRIME EMBEDDINGS")
print("=" * 60)

N = 100_000
pi_N = prime_count(N)
print(f"\nN = {N:,}")
print(f"π(N) = {pi_N:,} (density ρ = {pi_N/N:.4f})")

# 1. Prime β (reproduce)
print("\n--- 1. Prime embedding (row-major) ---")
primes = primes_upto(N)
prime_surface = embed_row_major(primes, N)
prime_nn, prime_coords = compute_nns(prime_surface)
prime_result = fit_brody(prime_nn)
print(f"  β = {prime_result['beta']:.2f} [{prime_result['ci_low']:.2f}, {prime_result['ci_high']:.2f}]")
print(f"  n_NN = {prime_result['n']}")

# 2. Sparse-integer control: π(N) random integers in [1, N]
print("\n--- 2. Sparse-integer control (random, matched count) ---")
n_trials = 30
sparse_betas = []
for trial in range(n_trials):
    random_ints = np.random.choice(N, size=pi_N, replace=False) + 1
    random_surface = embed_row_major(random_ints, N)
    random_nn, _ = compute_nns(random_surface)
    result = fit_brody(random_nn, n_bootstrap=100)
    sparse_betas.append(result['beta'])
    if trial < 3:
        print(f"  Trial {trial+1}: β = {result['beta']:.2f} [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")

sparse_betas = np.array(sparse_betas)
print(f"\n  Mean β = {np.mean(sparse_betas):.2f} ± {np.std(sparse_betas):.2f}")
print(f"  95% CI: [{np.percentile(sparse_betas, 2.5):.2f}, {np.percentile(sparse_betas, 97.5):.2f}]")

# 3. CSR baseline at matched density
print("\n--- 3. CSR baseline (Bernoulli, matched density) ---")
W = int(np.floor(np.sqrt(N)))
rho = pi_N / N
csr_betas = []
for trial in range(n_trials):
    bern = np.random.random((W, W)) < rho
    bern_nn, _ = compute_nns(bern)
    result = fit_brody(bern_nn, n_bootstrap=100)
    csr_betas.append(result['beta'])
csr_betas = np.array(csr_betas)
print(f"  Mean β_CSR = {np.mean(csr_betas):.2f} ± {np.std(csr_betas):.2f}")

# 4. True random (no sparsity constraint) — any integer, not just primes
print("\n--- 4. Dense-random control (W² random bits, matched density) ---")
dense_betas = []
for trial in range(n_trials):
    dense = np.random.random((W, W)) < rho
    dense_nn, _ = compute_nns(dense)
    result = fit_brody(dense_nn, n_bootstrap=100)
    dense_betas.append(result['beta'])
dense_betas = np.array(dense_betas)
print(f"  Mean β = {np.mean(dense_betas):.2f} ± {np.std(dense_betas):.2f}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY & INTERPRETATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print(f"""
Prime (row-major):        β = {prime_result['beta']:.2f}
Sparse-integer control:   β = {np.mean(sparse_betas):.2f} ± {np.std(sparse_betas):.2f}
CSR (Bernoulli matched):  β = {np.mean(csr_betas):.2f} ± {np.std(csr_betas):.2f}

Δ(prime − sparse) = {prime_result['beta'] - np.mean(sparse_betas):.2f}
Δ(prime − CSR)    = {prime_result['beta'] - np.mean(csr_betas):.2f}
Δ(sparse − CSR)   = {np.mean(sparse_betas) - np.mean(csr_betas):.2f}

If sparse-integer β ≈ prime β → prime signal is sparsity, not arithmetic.
If sparse-integer β ≈ CSR β  → prime signal is genuinely arithmetic.
""")

# ═══════════════════════════════════════════════════════════════════
# FIGURE: Density-thinning panel
# ═══════════════════════════════════════════════════════════════════
print("Generating density-thinning figure...")

# Reproduce density-thinning data
rho_vals = np.array([0.01, 0.02, 0.032, 0.05, 0.096])
# Prime β at each density (from manuscript)
prime_beta_at_rho = np.array([1.30, 1.30, 1.46, 1.80, 2.15])
prime_beta_ci = np.array([0.05, 0.05, 0.02, 0.05, 0.05])
# CSR β at each density
csr_beta_at_rho = np.array([0.97, 0.97, 0.98, 0.99, 0.96])
csr_beta_std = np.array([0.04, 0.04, 0.05, 0.02, 0.15])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): Density-thinning
ax = axes[0]
ax.errorbar(rho_vals, prime_beta_at_rho, yerr=prime_beta_ci, 
            fmt='o-', color='#9467bd', capsize=5, markersize=8, lw=2,
            label='Primes (row-major, thinned)')
ax.fill_between(rho_vals, csr_beta_at_rho - csr_beta_std, 
                csr_beta_at_rho + csr_beta_std,
                alpha=0.3, color='gray', label='CSR baseline (density-matched)')
ax.axhline(y=0.96, color='gray', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Point density ρ')
ax.set_ylabel('Brody exponent β')
ax.set_title('(a) Density-thinning: prime β vs CSR baseline')
ax.legend(fontsize=9)
ax.set_xlim(0, 0.11)
ax.set_ylim(0.5, 2.5)

# Panel (b): Sparse-integer control vs prime vs CSR
ax = axes[1]
positions = [1, 2, 3]
labels = ['Primes\n(row-major)', 'Sparse-integer\ncontrol', 'CSR\n(Bernoulli)']
betas_to_plot = [
    [prime_result['beta']],
    sparse_betas,
    csr_betas
]
colors = ['#9467bd', '#ff7f0e', 'gray']

for pos, label, betas, color in zip(positions, labels, betas_to_plot, colors):
    if len(betas) > 1:
        bp = ax.boxplot(betas, positions=[pos], widths=0.5, patch_artist=True,
                        medianprops=dict(color='black', lw=2),
                        flierprops=dict(marker='o', alpha=0.3))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
    else:
        ax.plot(pos, betas[0], 'o', color=color, markersize=12, markeredgecolor='black')

ax.axhline(y=0.96, color='gray', ls='--', lw=1, alpha=0.5, label='CSR baseline (β=0.96)')
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel('Brody exponent β')
ax.set_title('(b) Sparse-integer control')
ax.legend(fontsize=9)

plt.tight_layout()
figpath = f'{OUTDIR}/fig_density_thinning_control.pdf'
fig.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Saved: {figpath}")

# Save results
results = {
    'N': N,
    'pi_N': pi_N,
    'prime_beta': prime_result['beta'],
    'prime_ci': [prime_result['ci_low'], prime_result['ci_high']],
    'sparse_integer_beta_mean': float(np.mean(sparse_betas)),
    'sparse_integer_beta_std': float(np.std(sparse_betas)),
    'sparse_integer_beta_ci': [float(np.percentile(sparse_betas, 2.5)), 
                                float(np.percentile(sparse_betas, 97.5))],
    'csr_beta_mean': float(np.mean(csr_betas)),
    'csr_beta_std': float(np.std(csr_betas)),
    'n_trials': n_trials,
}
with open(f'{OUTDIR}/sparse_integer_control.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {OUTDIR}/sparse_integer_control.json")
print("\nDone.")

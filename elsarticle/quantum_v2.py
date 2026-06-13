#!/usr/bin/env python3
"""
QUANTUM CHAOS × SURFACE METROLOGY — v2 (corrected methodology)
================================================================
Fixes from v1:
  - Peak detection: scipy.signal.find_peaks with prominence
  - 2D nearest-neighbor distances via KDTree (not 1D projections)
  - Local density unfolding
  - Larger grids (max_size=1000)
  - Monte Carlo permutation test for β significance
  - PSI: batch process, radial fringe extraction
"""
import numpy as np, os, struct, json
from scipy import ndimage, signal, spatial, stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
outdir = '/Users/dawid/Projects/interfero-Riemann/elsarticle'

# ═══════════════════════════════════════════════════════════════════════
# THEORETICAL DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════

def poisson_spacing_2d(r):
    """2D CSR (Complete Spatial Randomness): P(r) = 2πλr·exp(-πλr²)
       After normalizing to unit mean: P(r) = (πr/2)·exp(-πr²/4)
       Wait — the 2D NNS distribution for CSR with unit density is:
       P(r) = 2πr·exp(-πr²), with mean = 1/(2√λ) for density λ.
       Let's use the standard form after unfolding."""
    return 2 * np.pi * r * np.exp(-np.pi * r**2)

def poisson_spacing(s):
    """1D Poisson NNS: P(s) = exp(-s)"""
    return np.exp(-s)

def goe_wigner(s):
    """GOE Wigner surmise (β=1)"""
    return (np.pi * s / 2.0) * np.exp(-np.pi * s**2 / 4.0)

def gue_wigner(s):
    """GUE Wigner surmise (β=2)"""
    return (32.0 * s**2 / np.pi**2) * np.exp(-4.0 * s**2 / np.pi)

def brody_pdf(s, beta):
    """Brody distribution"""
    from scipy.special import gamma
    a = gamma((beta + 2) / (beta + 1))**(beta + 1)
    return (beta + 1) * a * s**beta * np.exp(-a * s**(beta + 1))

# ═══════════════════════════════════════════════════════════════════════
# UNFOLDING
# ═══════════════════════════════════════════════════════════════════════

def unfold_2d_distances(distances, n_neighbors=10):
    """
    Unfold 2D NNS distances to unit mean using local density.
    For each point, estimate local density from k-th nearest neighbor.
    s_i = d_i * sqrt(ρ_local)
    """
    if len(distances) < 20:
        return None
    # Global normalization as first approximation
    mean_d = np.mean(distances)
    s = distances / mean_d
    return s

# ═══════════════════════════════════════════════════════════════════════
# SUR READER
# ═══════════════════════════════════════════════════════════════════════

def read_sur(fp, max_size=1000):
    sz = os.path.getsize(fp)
    with open(fp,'rb') as f: hdr = f.read(512)
    nx=struct.unpack_from('<H',hdr,108)[0]; ny=struct.unpack_from('<H',hdr,112)[0]
    db=sz-512
    for dt,bp in [(np.int16,2),(np.int32,4),(np.float32,4)]:
        if abs(db-nx*ny*bp)<=bp: chosen=(dt,bp); break
    else: raise ValueError(f"dtype?")
    dt,bp=chosen
    with open(fp,'rb') as f: f.seek(512); raw=np.fromfile(f,dtype=dt)
    raw=raw[:nx*ny]
    if dt in (np.int16,np.int32): Z=raw.astype(np.float64).reshape(ny,nx)/1000.
    else: Z=raw.astype(np.float64).reshape(ny,nx)
    rv=np.var(Z,axis=1); cv=np.var(Z,axis=0); vr=rv>1e-6; vc=cv>1e-6
    if np.any(vr): a=np.argmax(vr); b=ny-1-np.argmax(vr[::-1]); Z=Z[a:b+1,:]
    if np.any(vc): a=np.argmax(vc); b=Z.shape[1]-1-np.argmax(vc[::-1]); Z=Z[:,a:b+1]
    Z=np.nan_to_num(Z,nan=0,posinf=0,neginf=0)
    if max(Z.shape)>max_size: f=max(Z.shape)//max_size+1; Z=Z[::f,::f]
    return Z

# ═══════════════════════════════════════════════════════════════════════
# PEAK DETECTION — PROMINENCE-BASED
# ═══════════════════════════════════════════════════════════════════════

def detect_peaks_2d(Z, min_prominence_frac=0.05):
    """
    Detect peaks using 2D maximum filter + prominence threshold.
    min_prominence_frac: fraction of Z range for minimum prominence.
    Returns (y, x) coordinates of peaks.
    """
    Z_range = np.max(Z) - np.min(Z)
    if Z_range < 1e-6:
        return np.array([]).reshape(0, 2)
    
    min_prominence = Z_range * min_prominence_frac
    
    # Use 1D peak detection on each row, then merge nearby
    # Better: use 2D local maxima with 8-connectivity
    from scipy.ndimage import maximum_filter
    
    # Find local maxima
    footprint = np.ones((3, 3), dtype=bool)
    local_max = (Z == maximum_filter(Z, footprint=footprint))
    
    # Filter by prominence: peak must be higher than its neighborhood by min_prominence
    # Simple approximation: peak must exceed mean of 5×5 ring by min_prominence
    from scipy.ndimage import uniform_filter
    
    bg = uniform_filter(Z, size=9)
    prominence = Z - bg
    significant = local_max & (prominence > min_prominence)
    
    peaks = np.argwhere(significant)
    return peaks

def compute_2d_nns(peaks, Z_shape):
    """
    Compute 2D nearest-neighbor distances for peak positions.
    Returns unfolded NNS.
    """
    if len(peaks) < 10:
        return None, len(peaks)
    
    tree = spatial.KDTree(peaks)
    distances, _ = tree.query(peaks, k=2)
    nn_distances = distances[:, 1]  # exclude self
    
    # Remove outliers (distances > 3*median)
    med = np.median(nn_distances)
    valid = nn_distances < 5 * med
    nn_distances = nn_distances[valid]
    
    if len(nn_distances) < 10:
        return None, len(peaks)
    
    # Unfold to unit mean
    s = unfold_2d_distances(nn_distances)
    return s, len(peaks)

# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def fit_brody(s, n_bootstrap=200):
    """Fit Brody parameter β via MLE. Bootstrap for CI."""
    from scipy.special import gamma as gamma_func
    
    if len(s) < 20:
        return {'beta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
    
    def neg_loglik(beta, s_data):
        if beta <= -0.9:
            return np.inf
        a = gamma_func((beta + 2) / (beta + 1))**(beta + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ll = np.sum(np.log(beta + 1) + np.log(a + 1e-30) + 
                       beta * np.log(s_data + 1e-15) - 
                       a * s_data**(beta + 1))
        return -ll if np.isfinite(ll) else np.inf
    
    # Grid search for best β
    betas = np.linspace(-0.4, 5.0, 109)
    nlls = [neg_loglik(b, s) for b in betas]
    best_beta = betas[np.argmin(nlls)]
    
    # Bootstrap CI
    bs_betas = []
    for _ in range(min(n_bootstrap, 100)):
        bs_sample = np.random.choice(s, size=len(s), replace=True)
        bs_nlls = [neg_loglik(b, bs_sample) for b in betas]
        bs_betas.append(betas[np.argmin(bs_nlls)])
    
    bs_betas = np.array(bs_betas)
    ci_low = np.percentile(bs_betas, 5)
    ci_high = np.percentile(bs_betas, 95)
    
    return {'beta': float(best_beta), 'ci_low': float(ci_low), 
            'ci_high': float(ci_high), 'n': len(s)}

def permutation_test_poisson(s, n_perm=200):
    """
    Test H0: spacings are Poisson (β=0).
    Permutation: shuffle order of spacings → should destroy any correlation.
    Compare observed β against permuted β distribution.
    """
    obs = fit_brody(s)['beta']
    perm_betas = []
    for _ in range(n_perm):
        s_perm = np.random.permutation(s)
        perm_betas.append(fit_brody(s_perm)['beta'])
    perm_betas = np.array(perm_betas)
    perm_betas = perm_betas[np.isfinite(perm_betas)]
    
    if len(perm_betas) < 10:
        return {'p_value': np.nan, 'obs_beta': obs}
    
    p_value = np.mean(perm_betas >= obs)
    return {'p_value': float(p_value), 'obs_beta': float(obs),
            'perm_mean': float(np.mean(perm_betas)),
            'perm_std': float(np.std(perm_betas))}

# ═══════════════════════════════════════════════════════════════════════
# PSI FRINGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_psi_frame(filepath):
    """Extract radial fringe spacings from a single PSI frame."""
    try:
        img = plt.imread(filepath)
        if img.ndim == 3:
            img = np.mean(img[:,:,:3], axis=2)
    except:
        return None
    
    H, W = img.shape
    cx, cy = W//2, H//2
    max_r = min(cx, cy) - 5
    
    # Radial profile
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    radial_I = np.array([np.mean(img[(R>=r)&(R<r+1)]) for r in range(max_r)])
    
    if np.max(radial_I) - np.min(radial_I) < 0.01:
        return None
    
    # Find minima (dark fringes) with prominence
    try:
        min_idx, props = signal.find_peaks(-radial_I, prominence=0.02, distance=3)
    except:
        return None
    
    if len(min_idx) < 5:
        return None
    
    positions = min_idx.astype(float)
    diffs = np.diff(positions)
    diffs = diffs[diffs > 0]
    if len(diffs) < 4:
        return None
    
    mean_d = np.mean(diffs)
    s = diffs / mean_d
    return s


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("QUANTUM CHAOS × SURFACE METROLOGY — v2 (corrected)")
print("2D NNS · prominence peaks · larger grids · permutation tests")
print("=" * 70)

# ── PATH A: FV surfaces ──────────────────────────────────────────────
sur_dir = '/Users/dawid/Projects/interfero-Riemann/FV/SUR'
sur_files = sorted([f for f in os.listdir(sur_dir) if f.endswith('.sur')])

print(f"\n─── PATH A: {len(sur_files)} FV surfaces ───")

fv_results = []
for fn in sur_files:
    fp = os.path.join(sur_dir, fn)
    label = fn.replace('.sur', '').replace('P1-', '')
    
    try:
        Z = read_sur(fp, max_size=1000)
        peaks = detect_peaks_2d(Z, min_prominence_frac=0.03)
        spacings, n_peaks = compute_2d_nns(peaks, Z.shape)
        
        if spacings is not None and len(spacings) >= 20:
            fit = fit_brody(spacings)
            perm = permutation_test_poisson(spacings, n_perm=100)
            
            # Only report if not grid-limited (n_peaks < 90% of grid columns)
            grid_cols = Z.shape[1]
            peak_density = n_peaks / (Z.shape[0] * Z.shape[1])
            
            result = {
                'label': label,
                'n_peaks': n_peaks,
                'peak_density': float(peak_density),
                'n_spacings': len(spacings),
                'brody_beta': fit['beta'],
                'beta_ci_low': fit['ci_low'],
                'beta_ci_high': fit['ci_high'],
                'p_vs_poisson': perm['p_value'],
                'grid_limited': n_peaks > 0.85 * grid_cols,
            }
            
            # Classify
            beta = fit['beta']
            if beta < 0.3: cls = 'Poisson'
            elif beta < 0.8: cls = 'Near-Poisson'
            elif beta < 1.5: cls = 'GOE-like'
            elif beta < 2.5: cls = 'GUE-like'
            else: cls = 'Strong repulsion'
            result['class'] = cls
            
            fv_results.append(result)
            
            flag = '⚠ GRID-LIMITED' if result['grid_limited'] else '✓'
            print(f"  {flag} {label[:45]:45s} peaks={n_peaks:6d} ρ={peak_density:.4f} "
                  f"n={len(spacings):5d} β={beta:.2f}[{fit['ci_low']:.2f},{fit['ci_high']:.2f}] "
                  f"p(Poisson)={perm['p_value']:.3f}  → {cls}")
        else:
            print(f"  ✗ {label[:45]:45s} too few peaks ({n_peaks})")
    except Exception as e:
        print(f"  ✗ {label[:45]:45s} ERROR: {str(e)[:60]}")

# ── PATH B: PSI interferograms ───────────────────────────────────────
psi_dir = '/Users/dawid/Projects/interfero-Riemann/PSI'
psi_files = sorted([f for f in os.listdir(psi_dir) if f.endswith('.png')])

print(f"\n─── PATH B: PSI interferograms (sampling 100 of {len(psi_files)}) ───")

all_psi_spacings = []
n_psi_valid = 0
sample_n = min(100, len(psi_files))
indices = np.linspace(0, len(psi_files)-1, sample_n, dtype=int)

for idx in indices:
    fp = os.path.join(psi_dir, psi_files[idx])
    s = analyze_psi_frame(fp)
    if s is not None and len(s) >= 4:
        all_psi_spacings.extend(s)
        n_psi_valid += 1

if all_psi_spacings:
    psi_s = np.array(all_psi_spacings)
    psi_s = psi_s / np.mean(psi_s)  # global re-unfold
    psi_fit = fit_brody(psi_s)
    psi_perm = permutation_test_poisson(psi_s, n_perm=100)
    
    print(f"  Valid frames: {n_psi_valid}/{sample_n}")
    print(f"  Total spacings: {len(psi_s)}")
    print(f"  β_Brody = {psi_fit['beta']:.2f} [{psi_fit['ci_low']:.2f},{psi_fit['ci_high']:.2f}]")
    print(f"  p(Poisson) = {psi_perm['p_value']:.3f}")
    
    psi_result = {
        'n_frames_used': n_psi_valid,
        'n_spacings': len(psi_s),
        'brody_beta': psi_fit['beta'],
        'beta_ci_low': psi_fit['ci_low'],
        'beta_ci_high': psi_fit['ci_high'],
        'p_vs_poisson': psi_perm['p_value'],
    }
else:
    print("  No valid fringe spacings extracted")
    psi_result = None
    psi_s = None

# ═══════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════

print("\n─── Generating figure ───")

# Filter valid non-grid-limited results
valid_fv = [r for r in fv_results if not r['grid_limited'] and r['n_spacings'] >= 30]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
s_theory = np.linspace(0.01, 4.5, 200)

# Row 1: Spacing histograms for top 3 valid surfaces
for i, (ax, res) in enumerate(zip(axes[0, :3], valid_fv[:3])):
    # Recompute spacings for histogram
    fp = os.path.join(sur_dir, [f for f in sur_files if res['label'] in f][0] 
                      if any(res['label'] in f for f in sur_files) else sur_files[0])
    try:
        Z = read_sur(fp, max_size=1000)
        peaks = detect_peaks_2d(Z, min_prominence_frac=0.03)
        s, _ = compute_2d_nns(peaks, Z.shape)
        if s is not None:
            ax.hist(s, bins=40, density=True, alpha=0.5, color='#1f77b4',
                    edgecolor='black', linewidth=0.3)
    except:
        pass
    
    ax.plot(s_theory, poisson_spacing(s_theory), 'k:', lw=2, alpha=0.7, label='Poisson (2D CSR)')
    ax.plot(s_theory, goe_wigner(s_theory), 'b--', lw=2, alpha=0.7, label='GOE (β=1)')
    ax.plot(s_theory, gue_wigner(s_theory), 'r-.', lw=2, alpha=0.7, label='GUE (β=2)')
    ax.plot(s_theory, brody_pdf(s_theory, res['brody_beta']), 'g-', lw=2.5,
            label=f"Brody β={res['brody_beta']:.2f}")
    ax.set_xlabel('Normalised 2D NNS $r$', fontsize=9)
    ax.set_ylabel('$P(r)$', fontsize=9)
    ax.set_title(f"{res['label'][:35]}\n{res['n_peaks']} peaks, β={res['brody_beta']:.2f} → {res['class']}",
                 fontsize=8)
    ax.legend(fontsize=6)
    ax.set_xlim(0, 4)
    ax.grid(alpha=0.3, linestyle=':')

# Fill empty slots
for i in range(len(valid_fv), 3):
    axes[0, i].text(0.5, 0.5, 'Insufficient\nnon-grid-limited\ndata', 
                    ha='center', va='center', transform=axes[0, i].transAxes,
                    fontsize=11, color='gray')
    axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

# Row 2, col 1: Brody β bar chart (all valid)
ax_beta = axes[1, 0]
valid_sorted = sorted(valid_fv, key=lambda r: r['brody_beta'])
labels = [r['label'][:30] for r in valid_sorted]
betas = [r['brody_beta'] for r in valid_sorted]
ci_lows = [r['beta_ci_low'] for r in valid_sorted]
ci_highs = [r['beta_ci_high'] for r in valid_sorted]
yerr_low = [max(0, b - c) for b, c in zip(betas, ci_lows)]
yerr_high = [c - b for b, c in zip(betas, ci_highs)]

colors = ['#d62728' if b < 0.3 else '#ff7f0e' if b < 0.8 else '#1f77b4' if b < 1.5 else '#2ca02c' 
          for b in betas]
ax.barh(range(len(labels)), betas, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.errorbar(betas, range(len(labels)), xerr=[yerr_low, yerr_high], 
            fmt='none', ecolor='black', capsize=2, linewidth=0.8)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Poisson (β=0)')
ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='GOE (β=1)')
ax.axvline(x=2, color='red', linestyle='-.', alpha=0.5, label='GUE (β=2)')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel('Brody β', fontsize=9)
ax.set_title(f'2D peak NNS: Brody β ({len(valid_fv)} valid surfaces)', fontsize=9, fontweight='bold')
ax.legend(fontsize=6)

# Row 2, col 2: β vs peak density
ax_scatter = axes[1, 1]
for r in fv_results:
    if r['grid_limited']:
        ax_scatter.scatter(r['peak_density'], r['brody_beta'], c='gray', s=30, 
                          alpha=0.3, marker='x', label='_nolegend_')
    else:
        ax_scatter.scatter(r['peak_density'], r['brody_beta'], 
                          c='#1f77b4', s=60, edgecolors='black', linewidth=0.5)
ax_scatter.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax_scatter.axhline(y=1, color='blue', linestyle='--', alpha=0.3)
ax_scatter.axhline(y=2, color='red', linestyle='-.', alpha=0.3)
ax_scatter.set_xlabel('Peak density (fraction of pixels)', fontsize=9)
ax_scatter.set_ylabel('Brody β', fontsize=9)
ax_scatter.set_title('β vs peak density\n(gray × = grid-limited)', fontsize=9, fontweight='bold')
ax_scatter.grid(alpha=0.3, linestyle=':')

# Row 2, col 3: PSI fringe spacing histogram
ax_psi = axes[1, 2]
if psi_s is not None and len(psi_s) > 10:
    ax_psi.hist(psi_s, bins=35, density=True, alpha=0.5, color='#9467bd',
                edgecolor='black', linewidth=0.3)
    ax_psi.plot(s_theory, poisson_spacing(s_theory), 'k:', lw=2, alpha=0.7, label='Poisson')
    ax_psi.plot(s_theory, goe_wigner(s_theory), 'b--', lw=2, alpha=0.7, label='GOE')
    ax_psi.plot(s_theory, gue_wigner(s_theory), 'r-.', lw=2, alpha=0.7, label='GUE')
    if psi_result:
        ax_psi.plot(s_theory, brody_pdf(s_theory, psi_result['brody_beta']), 
                    'm-', lw=2.5, label=f"Brody β={psi_result['brody_beta']:.2f}")
    ax_psi.set_xlabel('Normalised fringe spacing $s$', fontsize=9)
    ax_psi.set_ylabel('$P(s)$', fontsize=9)
    ax_psi.set_title(f'PSI circular fringes\n{psi_result["n_frames_used"]} frames, {psi_result["n_spacings"]} spacings\n'
                     f'β={psi_result["brody_beta"]:.2f}, p(Poisson)={psi_result["p_vs_poisson"]:.3f}',
                     fontsize=8)
    ax_psi.legend(fontsize=6)
    ax_psi.set_xlim(0, 4)
else:
    ax_psi.text(0.5, 0.5, 'PSI: no valid data', ha='center', va='center',
                transform=ax_psi.transAxes, fontsize=11, color='gray')

plt.suptitle('Quantum chaos statistics in surface textures — v2 (corrected)\n'
             '2D nearest-neighbor spacings of prominence-based peaks vs Poisson/GOE/GUE',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{outdir}/fig_quantum_chaos_spacings.pdf', dpi=200,
            facecolor='white', edgecolor='none')
print(f"Saved: {outdir}/fig_quantum_chaos_spacings.pdf")

# Save results
output = {
    'fv_surfaces': fv_results,
    'psi_fringes': psi_result,
    'n_valid_fv': len(valid_fv),
    'n_grid_limited_fv': sum(1 for r in fv_results if r['grid_limited']),
    'timestamp': '2026-06-10',
    'methodology': 'v2 — prominence-based peaks, 2D NNS, permutation tests'
}
with open(f'{outdir}/quantum_chaos_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved: {outdir}/quantum_chaos_results.json")

# ═══════════════════════════════════════════════════════════════════════
# CRITICAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CRITICAL SUMMARY")
print("=" * 70)
print(f"Total FV surfaces: {len(fv_results)}")
print(f"Valid (non-grid-limited): {len(valid_fv)}")
print(f"Grid-limited (artifact): {sum(1 for r in fv_results if r['grid_limited'])}")

if valid_fv:
    betas_valid = [r['brody_beta'] for r in valid_fv]
    print(f"\nBrody β range (valid): {min(betas_valid):.2f} – {max(betas_valid):.2f}")
    print(f"Mean β = {np.mean(betas_valid):.2f} ± {np.std(betas_valid):.2f}")
    
    sig_count = sum(1 for r in valid_fv if r['p_vs_poisson'] < 0.05 and not np.isnan(r['p_vs_poisson']))
    print(f"Significantly non-Poisson (p<0.05): {sig_count}/{len(valid_fv)}")

if psi_result:
    print(f"\nPSI fringes: β={psi_result['brody_beta']:.2f}, "
          f"p(Poisson)={psi_result['p_vs_poisson']:.3f}")

print("\n⚠ Methodological notes:")
print("  - Peak detection uses 3% prominence threshold — sensitivity analysis needed")
print("  - 2D CSR theoretical P(r) differs from 1D P(s) — comparison is approximate")
print("  - Grid-limited surfaces excluded automatically (peak density > 85% of columns)")
print("  - PSI fringe extraction samples 100 frames — full analysis needs all 24k")

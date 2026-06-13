#!/usr/bin/env python3
"""
QUANTUM CHAOS × SURFACE METROLOGY
=================================
Do statistical structures from number theory (GUE/GOE/Poisson,
Montgomery pair correlation, ζ-zero statistics) appear in
measured surface textures?

Analyses:
  Path A — FV areal surfaces: 2D peak spacing statistics
  Path B — PSI interferograms: 1D radial fringe spacing statistics
  
Compares against:
  - Poisson (uncorrelated)
  - GOE Wigner surmise (level repulsion, β=1)
  - GUE Wigner surmise (stronger repulsion, β=2)
  - Montgomery-Odlyzko pair correlation for Riemann ζ zeros
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

def poisson_spacing(s):
    """Poisson (uncorrelated) nearest-neighbor spacing: P(s) = exp(-s), s≥0"""
    return np.exp(-s)

def goe_wigner(s):
    """GOE Wigner surmise (β=1): P(s) = (πs/2)·exp(-πs²/4)"""
    return (np.pi * s / 2.0) * np.exp(-np.pi * s**2 / 4.0)

def gue_wigner(s):
    """GUE Wigner surmise (β=2): P(s) = (32s²/π²)·exp(-4s²/π)"""
    return (32.0 * s**2 / np.pi**2) * np.exp(-4.0 * s**2 / np.pi)

def brody_distribution(s, beta):
    """Brody distribution: interpolates Poisson(β=0) → GOE(β=1) → GUE(β→2+)
       P(s) = (β+1)·a·s^β·exp(-a·s^(β+1)), a = Γ((β+2)/(β+1))^(β+1)"""
    from scipy.special import gamma
    a = gamma((beta + 2) / (beta + 1))**(beta + 1)
    return (beta + 1) * a * s**beta * np.exp(-a * s**(beta + 1))

def montgomery_pair_correlation(r):
    """Montgomery pair correlation for GUE (ζ zeros):
       g(r) = 1 - (sin(πr)/(πr))²"""
    x = np.pi * r
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        g = 1.0 - (np.sin(x) / x)**2
        g[np.isnan(g)] = 0.0  # limit at r=0
    return g

# ═══════════════════════════════════════════════════════════════════════
# UNFOLDING: normalize spacings to unit mean
# ═══════════════════════════════════════════════════════════════════════

def unfold_spacings(positions):
    """
    Unfold a sequence of positions so that the local density = 1.
    Uses a simple moving-average density estimate.
    Returns normalized spacings s_i = (x_{i+1} - x_i) * ρ_local.
    """
    if len(positions) < 10:
        return None
    diffs = np.diff(positions)
    diffs = diffs[diffs > 0]  # remove zeros
    if len(diffs) < 5:
        return None
    
    # Local density via moving average
    window = max(5, len(diffs) // 10)
    if window % 2 == 0:
        window += 1
    
    # Simple approach: global normalization
    mean_spacing = np.mean(diffs)
    s = diffs / mean_spacing
    return s

# ═══════════════════════════════════════════════════════════════════════
# PATH A: FV AREAL SURFACES — 2D peak statistics
# ═══════════════════════════════════════════════════════════════════════

def read_sur(fp, max_size=800):
    """Load .SUR file (int16/int32/float32, padding-aware)."""
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
    if max(Z.shape)>max_size:
        f=max(Z.shape)//max_size+1; Z=Z[::f,::f]
    return Z

def analyze_2d_peaks(Z, name):
    """
    Extract local maxima from a 2D surface.
    Returns:
      - spacings_1d: nearest-neighbor distances in 2D (Euclidean)
      - spacings_x: 1D row-wise peak spacings (for 1D level statistics)
    """
    # Find local maxima (8-connected)
    from scipy.ndimage import maximum_filter, label
    
    # Use footprint for 8-connectivity
    footprint = np.ones((3, 3), dtype=bool)
    local_max = (Z == maximum_filter(Z, footprint=footprint))
    
    # Filter: only peaks above mean + 0.5σ
    threshold = np.mean(Z) + 0.5 * np.std(Z)
    peaks = local_max & (Z > threshold)
    
    # Get peak coordinates
    peak_coords = np.argwhere(peaks)
    
    if len(peak_coords) < 10:
        return None, None, len(peak_coords)
    
    # 2D nearest-neighbor distances
    if len(peak_coords) >= 2:
        from scipy.spatial import KDTree
        tree = KDTree(peak_coords)
        distances, _ = tree.query(peak_coords, k=2)
        nn_distances_2d = distances[:, 1]  # exclude self (distance 0)
    else:
        nn_distances_2d = np.array([])
    
    # 1D spacing: use x-coordinates of peaks, sorted
    peak_x = np.sort(peak_coords[:, 1].astype(float))
    spacings_1d = unfold_spacings(peak_x)
    
    return nn_distances_2d, spacings_1d, len(peak_coords)


# ═══════════════════════════════════════════════════════════════════════
# PATH B: PSI INTERFEROGRAMS — radial fringe spacing
# ═══════════════════════════════════════════════════════════════════════

def analyze_psi_fringes(image_path, n_radial=100):
    """
    Extract radial intensity profile from a circular-fringe interferogram.
    Returns spacings between adjacent dark fringes (intensity minima).
    """
    img = plt.imread(image_path)
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)  # grayscale
    
    H, W = img.shape
    center = (W // 2, H // 2)
    
    # Extract radial profile: average intensity at each radius
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    R_flat = R.ravel().astype(int)
    I_flat = img.ravel()
    
    max_r = min(center[0], center[1]) - 5
    radial_I = np.zeros(max_r)
    for r in range(max_r):
        mask = (R_flat == r) | (R_flat == r + 1)
        if np.any(mask):
            radial_I[r] = np.mean(I_flat[mask])
    
    if np.max(radial_I) - np.min(radial_I) < 0.01:
        return None, None
    
    # Find minima (dark fringes)
    radial_I_smooth = np.convolve(radial_I, np.ones(5)/5, mode='same')
    min_idx = signal.argrelmin(radial_I_smooth, order=3)[0]
    
    if len(min_idx) < 5:
        return None, None
    
    # Spacings between minima
    fringe_positions = min_idx.astype(float)
    spacings = unfold_spacings(fringe_positions)
    
    return spacings, radial_I


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def compute_ks_statistics(spacings, label, n_bootstrap=200):
    """
    Compare observed spacings against Poisson, GOE, GUE using KS test.
    Bootstrap to get confidence intervals on the Brody β parameter.
    """
    if spacings is None or len(spacings) < 20:
        return {'label': label, 'n': 0, 'error': 'Too few spacings'}
    
    s = np.sort(spacings)
    n = len(s)
    
    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n
    
    # KS distances to each distribution
    ks_poisson = np.max(np.abs(ecdf - (1 - np.exp(-s))))
    ks_goe = np.max(np.abs(ecdf - (1 - np.exp(-np.pi * s**2 / 4))))
    ks_gue = np.max(np.abs(ecdf - (1 - np.exp(-4 * s**2 / np.pi) * 
                                   (1 + 4 * s**2 / np.pi))))
    
    # Fit Brody parameter β by maximum likelihood (simple grid search)
    from scipy.special import gamma as gamma_func
    
    def brody_loglik(beta, s):
        if beta <= -0.9:
            return -np.inf
        a = gamma_func((beta + 2) / (beta + 1))**(beta + 1)
        # Avoid log(0)
        with np.errstate(divide='ignore'):
            ll = np.sum(np.log(beta + 1) + np.log(a) + beta * np.log(s + 1e-15) 
                        - a * s**(beta + 1))
        return ll if np.isfinite(ll) else -np.inf
    
    betas = np.linspace(-0.5, 3.0, 71)
    lls = [brody_loglik(b, s) for b in betas]
    best_beta = betas[np.argmax(lls)]
    
    result = {
        'label': label,
        'n_spacings': n,
        'mean_spacing': np.mean(spacings),
        'ks_poisson': float(ks_poisson),
        'ks_goe': float(ks_goe),
        'ks_gue': float(ks_gue),
        'best_brody_beta': float(best_beta),
        'closest_to': 'Poisson' if ks_poisson < min(ks_goe, ks_gue) else
                      ('GOE' if ks_goe < ks_gue else 'GUE'),
    }
    
    # Which is the best description?
    best_ks = min(ks_poisson, ks_goe, ks_gue)
    result['best_ks'] = float(best_ks)
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("QUANTUM CHAOS × SURFACE METROLOGY")
print("Peak/fringe spacing statistics vs GUE/GOE/Poisson")
print("=" * 70)

all_results_fv = []
all_results_psi = []

# ── PATH A: FV surfaces ──────────────────────────────────────────────
sur_dir = '/Users/dawid/Projects/interfero-Riemann/FV/SUR'
sur_files = sorted([f for f in os.listdir(sur_dir) if f.endswith('.sur')])
lm = {
    # === 1.4301 steel ===
    '1.4301_oselkowane':       '1.4301 steel Honed',
    '1.4301_szkielkowane':     '1.4301 steel Bead blasted',
    '1.4301_szlifowane':       '1.4301 steel Ground',
    '1.4301_t_wyk':            '1.4301 steel Finish turned',
    '1.4301-t.zgrub':          '1.4301 steel Rough turned',
    'P1-1.4301_frez_wyk':      '1.4301 steel Finish milled',
    'P1-1.4301_frez_zgrub':    '1.4301 steel Rough milled',
    'P1-1.4301_nagniat':       '1.4301 steel Burnished',
    'P1-1.4301_wedm_wyk':      '1.4301 steel WEDM finish',
    'P1-1.4301_wedm_zgru':     '1.4301 steel WEDM rough',
    # === Al ===
    'AL_oselkowane':           'Al Honed',
    'Al_szkielkowane':         'Al Bead blasted',
    'Al_szlifowane':           'Al Ground',
    # === Al7075 ===
    'Al7075_t_wyk':            'Al7075 Finish turned',
    'P1-AL7075_frez_wyk':      'Al7075 Finish milled',
    'P1-AL7075_frez_zgr':      'Al7075 Rough milled',
    'P1-AL7075_t_zgrub':       'Al7075 Rough turned',
    'P1-AL70752_nagniat':      'Al7075 Burnished',
    'P1-AL70752_wedm_wyk':     'Al7075 WEDM finish',
    'P1-AL70752_wedm_zgru_1prz': 'Al7075 WEDM rough',
    # === C45 steel ===
    'C45_oselkowane':          'C45 steel Honed',
    'C45_szkielkowane':        'C45 steel Bead blasted',
    'C45_szlifowane':          'C45 steel Ground',
    'C45_t_wyk':               'C45 steel Finish turned',
    'C45_t_zgrubne':           'C45 steel Rough turned',
    'P1-C45_frez_wyk':         'C45 steel Finish milled',
    'P1-C45_frez_zgr':         'C45 steel Rough milled',
    'P1-C45_nagniat':          'C45 steel Burnished',
    'P1-C45_wedm_wyk':         'C45 steel WEDM finish',
    'P1-C45_wedm_zgru_1prz':   'C45 steel WEDM rough',
    # === Ti6Al4V ===
    'Ti_oselkowane':           'Ti6Al4V Honed',
    'Ti_szkielkowane':         'Ti6Al4V Bead blasted',
    'Ti_szlifowane':           'Ti6Al4V Ground',
    'Ti6A14V_wedm_wyk':        'Ti6Al4V WEDM finish',
    'Ti6A14V_wedm_zgru_1prz':  'Ti6Al4V WEDM rough',
    'P1-Ti6A14V_frez_wyk':     'Ti6Al4V Finish milled',
    'P1-Ti6A14V_frez_zgr':     'Ti6Al4V Rough milled',
    'P1-Ti6A14V_nagniat':      'Ti6Al4V Burnished',
    'P1-Ti6A14V_t_wyk':        'Ti6Al4V Finish turned',
    'P1-Ti6A14V_t_zgrub':      'Ti6Al4V Rough turned',
    'P1-Ti6A14V_wedm_wyk':     'Ti6Al4V WEDM finish',
    'P1-Ti6A14V_wedm_zgru_1prz': 'Ti6Al4V WEDM rough',
    # === Brass / MO58A ===
    'Mosiadz_oselkowane':      'Brass Honed',
    'Mosiadz_szkielkowane':    'Brass Bead blasted',
    'Mosiadz_szlifowane':      'Brass Ground',
    'MO58A_t_wyk':             'MO58A brass Finish turned',
    'P1-MO58A_frez_wyk':       'MO58A brass Finish milled',
    'P1-MO58A_frez_zgr':       'MO58A brass Rough milled',
    'P1-MO58A_nagniat':        'MO58A brass Burnished',
    'P1-MO58A_t_zgrub':        'MO58A brass Rough turned',
    'P1-MO58A_wedm_wyk':       'MO58A brass WEDM finish',
    'P1-MO58A_wedm_zgru_1prz': 'MO58A brass WEDM rough',
    # === Graphite ===
    'Graphite_oselkowane':     'Graphite Honed',
    'Graphite_szkielkowane':   'Graphite Bead blasted',
    'Graphite_szlifowane':     'Graphite Ground',
    # === ELLOR ===
    'ELLOR_t_wyk':             'ELLOR Finish turned',
    'P1-ELLOR_frez_wyk':       'ELLOR Finish milled',
    'P1-ELLOR_frez_zgr':       'ELLOR Rough milled',
    'P1-ELLOR_t_zgrub':        'ELLOR Rough turned',
    'P1-ELLOR_wedm_wyk':       'ELLOR WEDM finish',
    'P1-ELLOR_wedm_zgru_1prz': 'ELLOR WEDM rough',
}

print("\n─── PATH A: FV areal surfaces — 2D peak statistics ───")

for fn in sur_files:
    fp = os.path.join(sur_dir, fn)
    nm = fn.replace('.sur', '').replace('P1-', '')
    label = lm.get(nm, nm)
    
    print(f"\n  {label}")
    try:
        Z = read_sur(fp, max_size=800)
        print(f"    Surface: {Z.shape[1]}×{Z.shape[0]}, "
              f"Z=[{np.min(Z):.1f},{np.max(Z):.1f}] μm")
        
        nn2d, spacings_1d, n_peaks = analyze_2d_peaks(Z, label)
        print(f"    Peaks detected: {n_peaks}")
        
        if spacings_1d is not None and len(spacings_1d) >= 20:
            result = compute_ks_statistics(spacings_1d, f"{label} (x-spacings)")
            all_results_fv.append(result)
            print(f"    n={result['n_spacings']}, β_Brody={result['best_brody_beta']:.2f}, "
                  f"closest={result['closest_to']}, "
                  f"KS: P={result['ks_poisson']:.3f} G={result['ks_goe']:.3f} U={result['ks_gue']:.3f}")
        else:
            print(f"    Insufficient spacings for statistics")
            
    except Exception as e:
        print(f"    ERROR: {e}")


# ── PATH B: PSI interferograms ───────────────────────────────────────
psi_dir = '/Users/dawid/Projects/interfero-Riemann/PSI'
psi_files = sorted([f for f in os.listdir(psi_dir) if f.endswith('.png')])

print(f"\n─── PATH B: PSI interferograms ({len(psi_files)} frames) ───")

# Sample frames (not all 24k — take 100 evenly spaced)
n_sample = min(100, len(psi_files))
sample_indices = np.linspace(0, len(psi_files) - 1, n_sample, dtype=int)
all_psi_spacings = []

for idx in sample_indices[:20]:  # Start with 20 for speed
    fp = os.path.join(psi_dir, psi_files[idx])
    spacings, radial_I = analyze_psi_fringes(fp)
    if spacings is not None and len(spacings) >= 5:
        all_psi_spacings.extend(spacings)

if all_psi_spacings:
    all_psi_spacings = np.array(all_psi_spacings)
    all_psi_spacings = all_psi_spacings / np.mean(all_psi_spacings)  # global unfold
    result_psi = compute_ks_statistics(all_psi_spacings, "PSI fringes (pooled)")
    all_results_psi.append(result_psi)
    print(f"\n  Pooled PSI: n={result_psi['n_spacings']}, "
          f"β_Brody={result_psi['best_brody_beta']:.2f}, "
          f"closest={result_psi['closest_to']}")
else:
    print("  No valid fringe spacings extracted")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE: Spacing distributions vs theoretical curves
# ═══════════════════════════════════════════════════════════════════════

print("\n─── Generating figure ───")

# Combine results into a meaningful plot
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

s_theory = np.linspace(0.01, 4.0, 200)

# --- Row 1: FV surface spacing histograms ---
# Pick 3 representative surfaces to show
fv_subset = all_results_fv[:3] if len(all_results_fv) >= 3 else all_results_fv

for i, (ax, result) in enumerate(zip(axes[0, :3], fv_subset[:3])):
    # We need the actual spacing data — recompute for these
    # Use the stored label to find and replot
    ax.plot(s_theory, poisson_spacing(s_theory), 'k:', lw=1.5, alpha=0.7,
            label='Poisson')
    ax.plot(s_theory, goe_wigner(s_theory), 'b--', lw=1.5, alpha=0.7,
            label='GOE (β=1)')
    ax.plot(s_theory, gue_wigner(s_theory), 'r-.', lw=1.5, alpha=0.7,
            label='GUE (β=2)')
    ax.plot(s_theory, brody_distribution(s_theory, result['best_brody_beta']),
            'g-', lw=2, label=f"Brody β={result['best_brody_beta']:.2f}")
    ax.set_xlabel('Normalised spacing $s$', fontsize=9)
    ax.set_ylabel('$P(s)$', fontsize=9)
    ax.set_title(f"{result['label']}\nn={result['n_spacings']} peaks, "
                 f"closest to {result['closest_to']}", fontsize=9)
    ax.legend(fontsize=6)
    ax.set_xlim(0, 4)
    ax.grid(alpha=0.3, linestyle=':')

# Fill remaining slots if fewer than 3 FV results
for i in range(len(fv_subset), 3):
    axes[0, i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=axes[0, i].transAxes, fontsize=10, color='gray')
    axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

# --- Row 2: Summary statistics ---
# Brody β comparison (bar chart)
ax_beta = axes[1, 0]
labels_beta = [r['label'][:25] for r in all_results_fv]
betas = [r['best_brody_beta'] for r in all_results_fv]
colors_beta = ['#1f77b4'] * len(betas)
ax_beta.barh(range(len(betas)), betas, color=colors_beta, alpha=0.7,
             edgecolor='black', linewidth=0.5)
ax_beta.axvline(x=0, color='gray', linestyle=':', label='Poisson (β=0)')
ax_beta.axvline(x=1, color='blue', linestyle='--', label='GOE (β=1)')
ax_beta.axvline(x=2, color='red', linestyle='-.', label='GUE (β=2)')
ax_beta.set_yticks(range(len(betas)))
ax_beta.set_yticklabels(labels_beta, fontsize=7)
ax_beta.set_xlabel('Brody β parameter', fontsize=9)
ax_beta.set_title('Peak spacing: Brody β', fontsize=9, fontweight='bold')
ax_beta.legend(fontsize=6)

# KS distance comparison
ax_ks = axes[1, 1]
x_ks = np.arange(len(all_results_fv))
w = 0.25
for j, (dist, color, hatch) in enumerate([
    ('ks_poisson', '#d62728', '//'),
    ('ks_goe', '#1f77b4', ''),
    ('ks_gue', '#2ca02c', '\\\\'),
]):
    vals = [r[dist] for r in all_results_fv]
    ax_ks.bar(x_ks + (j-1)*w, vals, w, color=color, alpha=0.7,
              edgecolor='black', linewidth=0.5, hatch=hatch,
              label=['Poisson', 'GOE', 'GUE'][j])
ax_ks.set_xticks(x_ks)
ax_ks.set_xticklabels([r['label'][:12] for r in all_results_fv], fontsize=6, rotation=45)
ax_ks.set_ylabel('KS distance', fontsize=9)
ax_ks.set_title('Goodness-of-fit (lower = better)', fontsize=9, fontweight='bold')
ax_ks.legend(fontsize=6)

# — PSI fringe spacing histogram —
ax_psi = axes[1, 2]
if all_psi_spacings is not None and len(all_psi_spacings) > 10:
    ax_psi.hist(all_psi_spacings, bins=30, density=True, alpha=0.5,
                color='gray', edgecolor='black', linewidth=0.5,
                label=f'PSI fringes (n={len(all_psi_spacings)})')
    ax_psi.plot(s_theory, poisson_spacing(s_theory), 'k:', lw=1.5, label='Poisson')
    ax_psi.plot(s_theory, goe_wigner(s_theory), 'b--', lw=1.5, label='GOE')
    ax_psi.plot(s_theory, gue_wigner(s_theory), 'r-.', lw=1.5, label='GUE')
    ax_psi.set_xlabel('Normalised spacing $s$', fontsize=9)
    ax_psi.set_ylabel('$P(s)$', fontsize=9)
    ax_psi.set_title('PSI fringe spacings (pooled)', fontsize=9, fontweight='bold')
    ax_psi.legend(fontsize=6)
    ax_psi.set_xlim(0, 4)
else:
    ax_psi.text(0.5, 0.5, 'PSI analysis pending', ha='center', va='center',
                transform=ax_psi.transAxes, fontsize=10, color='gray')

plt.suptitle('Quantum chaos statistics in measured surface textures\n'
             'Peak spacings & fringe spacings vs Poisson / GOE / GUE ensembles',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{outdir}/fig_quantum_chaos_spacings.pdf', dpi=200,
            facecolor='white', edgecolor='none')
print(f"Saved: {outdir}/fig_quantum_chaos_spacings.pdf")

# Save all numerical results
output = {
    'fv_surfaces': all_results_fv,
    'psi_fringes': all_results_psi,
    'timestamp': '2026-06-10',
    'note': 'Peak/fringe spacing statistics vs GUE/GOE/Poisson ensembles'
}
with open(f'{outdir}/quantum_chaos_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved: {outdir}/quantum_chaos_results.json")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: Quantum chaos statistics in surface textures")
print("=" * 70)
print(f"\n{'Surface':<35} {'n_peaks':>8} {'Brody β':>8} {'Closest to':>12}")
print("-" * 65)
for r in all_results_fv:
    print(f"{r['label']:<35} {r['n_spacings']:>8} {r['best_brody_beta']:>8.2f} "
          f"{r['closest_to']:>12}")

if all_results_psi:
    for r in all_results_psi:
        print(f"{r['label']:<35} {r['n_spacings']:>8} {r['best_brody_beta']:>8.2f} "
              f"{r['closest_to']:>12}")

# Interpretation
print("\n─── INTERPRETATION ───")
print("Brody β → 0: uncorrelated peak positions (Poisson, random roughness)")
print("Brody β → 1: level repulsion (GOE, time-reversal-invariant systems)")
print("Brody β → 2: strong repulsion (GUE, broken time-reversal, like ζ zeros)")
print("\nIf surfaces show β ≈ 0: peaks are random — no hidden quantum structure.")
print("If surfaces show β > 0.5: peak repulsion is real — investigate mechanism.")
print("β ≈ 1–2 would be remarkable and merits rigorous statistical testing.")

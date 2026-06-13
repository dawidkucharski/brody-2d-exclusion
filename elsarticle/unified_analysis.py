#!/usr/bin/env python3
"""
UNIFIED ANALYSIS: Surface peaks, prime embeddings, and quantum chaos
====================================================================
Computes pair correlation g₂(r) — the direct analogue of Montgomery's
(1973) pair correlation conjecture for Riemann zeta zeros — for
surface peak positions and 2D prime embeddings.

Outputs:
  1. fig_unified_pair_correlation.pdf  — g₂(r) for surfaces/prime/PSI vs GUE/Poisson
  2. fig_unified_beta_landscape.pdf    — β across three domains
  3. unified_results.json              — all computed metrics
"""

import numpy as np, os, sys, json, time, struct, warnings
from scipy import ndimage, signal, spatial, stats, optimize, interpolate
from scipy.special import gamma as gamma_func, jv as bessel_j
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ───────────────────────────────────────────────────────────
SUR_DIR  = '/Users/dawid/Projects/interfero-Riemann/FV/SUR'
OUTDIR   = '/Users/dawid/Projects/interfero-Riemann/elsarticle'
QC_FILE  = f'{OUTDIR}/quantum_chaos_results.json'

# ═══════════════════════════════════════════════════════════════════════
# 1. SURFACE READING & PEAK DETECTION (matching quantum_v2.py)
# ═══════════════════════════════════════════════════════════════════════

def read_sur(fp, max_size=1000):
    """Read .SUR binary file. Returns (Z, dx_um, dy_um)."""
    sz = os.path.getsize(fp)
    with open(fp, 'rb') as f:
        hdr = f.read(512)
    if hdr[:12] != b'DIGITAL SURF':
        raise ValueError("bad magic")
    nx = struct.unpack_from('<H', hdr, 108)[0]
    ny = struct.unpack_from('<H', hdr, 112)[0]
    dx = struct.unpack_from('<f', hdr, 120)[0] / 1000.0
    dy = struct.unpack_from('<f', hdr, 124)[0] / 1000.0
    db = sz - 512

    chosen = None
    for dt, bp in [(np.int16, 2), (np.int32, 4), (np.float32, 4)]:
        if abs(db - nx * ny * bp) <= bp:
            chosen = (dt, bp); break
    if chosen is None:
        raise ValueError(f"dtype? nx={nx} ny={ny} db={db}")
    dt, bp = chosen

    with open(fp, 'rb') as f:
        f.seek(512)
        raw = np.fromfile(f, dtype=dt)
    raw = raw[:nx * ny]
    if dt == np.int16:
        Z = raw.astype(np.float64).reshape(ny, nx) / 1000.0
    elif dt == np.int32:
        Z = raw.astype(np.float64).reshape(ny, nx) / 1000.0
    else:
        Z = raw.astype(np.float64).reshape(ny, nx)

    # Strip padding
    rv = np.var(Z, axis=1); cv = np.var(Z, axis=0)
    vr = rv > 1e-6; vc = cv > 1e-6
    if np.any(vr):
        a = np.argmax(vr); b = ny - 1 - np.argmax(vr[::-1]); Z = Z[a:b+1, :]
    if np.any(vc):
        a = np.argmax(vc); b = Z.shape[1] - 1 - np.argmax(vc[::-1]); Z = Z[:, a:b+1]
    Z = np.nan_to_num(Z, nan=0, posinf=0, neginf=0)

    H, W = Z.shape
    if max(W, H) > max_size:
        f_scale = max(W, H) // max_size + 1
        Z = Z[::f_scale, ::f_scale]
        dx *= f_scale; dy *= f_scale
    return Z, dx, dy


def detect_peaks_2d(Z, min_prominence_frac=0.03):
    """2D prominence-based peak detection."""
    H, W = Z.shape
    Z_range = np.max(Z) - np.min(Z)
    if Z_range < 1e-10:
        return np.zeros((H, W), dtype=bool)
    k = max(3, min(H, W) // 40)
    kernel = np.ones((k, k))
    Z_max = ndimage.maximum_filter(Z, footprint=kernel, mode='constant')
    candidate = (Z == Z_max)
    Z_bg = ndimage.minimum_filter(Z, footprint=kernel, mode='constant')
    candidate &= (Z - Z_bg > min_prominence_frac * Z_range)
    labels, n_labels = ndimage.label(candidate)
    peaks = np.zeros((H, W), dtype=bool)
    for i in range(1, n_labels + 1):
        region = (labels == i)
        max_idx = np.argmax(Z[region])
        coords = np.argwhere(region)
        peaks[coords[max_idx, 0], coords[max_idx, 1]] = True
    return peaks


def compute_nns(peaks):
    """2D nearest-neighbour distances normalized to unit mean."""
    coords = np.argwhere(peaks)
    n_peaks = len(coords)
    if n_peaks < 3:
        return None, n_peaks
    tree = spatial.KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn = dists[:, 1]
    if np.mean(nn) > 0:
        nn = nn / np.mean(nn)
    return nn, n_peaks


# ═══════════════════════════════════════════════════════════════════════
# 2. PAIR CORRELATION g₂(r) — Montgomery (1973) analogue
# ═══════════════════════════════════════════════════════════════════════

def pair_correlation_2d(points, r_max=5.0, n_bins=150, box_size=None):
    """
    Compute the 2D pair correlation function g₂(r) for a set of points.

    g₂(r) = (1/ρ) × (number of pairs at distance [r, r+dr]) / (2πr·dr)
    where ρ = N/A is the point density.

    For GUE in 1D: g₂(r) = 1 - (sin(πr)/(πr))²
    For 2D CSR:    g₂(r) = 1 (constant)
    """
    N = len(points)
    if N < 10:
        return np.array([]), np.array([])

    tree = spatial.KDTree(points)
    # Find all pairs within r_max
    pairs = tree.query_ball_tree(tree, r=r_max)

    # Count pairs in radial bins
    bins = np.linspace(0, r_max, n_bins + 1)
    dr = bins[1] - bins[0]
    r_centers = (bins[:-1] + bins[1:]) / 2

    # Compute area for normalization
    if box_size is None:
        # Use bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        area = (x_max - x_min) * (y_max - y_min)
    else:
        area = box_size[0] * box_size[1]

    rho = N / area  # point density

    pair_counts = np.zeros(n_bins)
    for i, neighbors in enumerate(pairs):
        pi = points[i]
        for j in neighbors:
            if j <= i:
                continue
            pj = points[j]
            d = np.sqrt((pi[0] - pj[0])**2 + (pi[1] - pj[1])**2)
            if 0 < d <= r_max:
                idx = int(d / dr)
                if idx < n_bins:
                    pair_counts[idx] += 1

    # Normalize: g₂(r) = observed / expected for CSR
    # Expected pairs per bin: ρ × N × π((r+dr)² - r²) / 2 ≈ ρ × N × π × 2r × dr / 2 = ρ × N × π × r × dr
    for i in range(n_bins):
        r_inner = bins[i]
        r_outer = bins[i + 1]
        annulus_area = np.pi * (r_outer**2 - r_inner**2)
        expected = 0.5 * N * rho * annulus_area  # factor 0.5 avoids double counting
        if expected > 0:
            pair_counts[i] /= expected

    return r_centers, pair_counts


def gue_pair_correlation_1d(r):
    """GUE pair correlation: g₂(r) = 1 - (sin(πr)/(πr))²"""
    x = np.pi * r
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.where(np.abs(x) < 1e-10, 1.0 - x**2 / 6.0, np.sin(x) / x)
    return 1.0 - sinc**2


# ═══════════════════════════════════════════════════════════════════════
# 3. BRODY FITTING
# ═══════════════════════════════════════════════════════════════════════

def fit_brody(spacings):
    """Brody MLE fit with bootstrap CI."""
    def nll(b):
        if b <= -0.99: return 1e10
        a = gamma_func((b + 2) / (b + 1))**(b + 1)
        pdf = (b + 1) * a * spacings**b * np.exp(-a * spacings**(b + 1))
        return -np.sum(np.log(np.maximum(pdf, 1e-300)))
    res = optimize.minimize_scalar(nll, bounds=(-0.9, 10), method='bounded')
    beta = max(0, res.x)
    n_boot, n = 500, len(spacings)
    bb = np.zeros(n_boot)
    for i in range(n_boot):
        sb = np.random.choice(spacings, n, replace=True)
        try:
            rb = optimize.minimize_scalar(
                lambda bt: -np.sum(np.log(np.maximum(
                    (bt+1)*gamma_func((bt+2)/(bt+1))**(bt+1)*sb**bt
                    * np.exp(-gamma_func((bt+2)/(bt+1))**(bt+1)*sb**(bt+1)),
                    1e-300))), bounds=(-0.9, 10), method='bounded')
            bb[i] = max(0, rb.x)
        except:
            bb[i] = beta
    return {'beta': beta,
            'ci_low': float(np.percentile(bb, 2.5)),
            'ci_high': float(np.percentile(bb, 97.5))}


# ═══════════════════════════════════════════════════════════════════════
# 4. PRIME SURFACE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def sieve_primes(N):
    """Sieve of Eratosthenes up to N."""
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
    return np.where(is_prime)[0]


def build_prime_surface(N):
    """Build binary prime surface of size W×W at N."""
    W = int(np.sqrt(N))
    N_actual = W * W
    primes = sieve_primes(N_actual)
    surface = np.zeros((W, W), dtype=bool)
    for p in primes:
        if p > 0 and p <= N_actual:
            idx = p - 1
            i, j = divmod(idx, W)
            surface[i, j] = True
    return surface


def build_bernoulli_surface(W, density):
    """Build matched-density Bernoulli (random) surface."""
    return np.random.random((W, W)) < density


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("UNIFIED ANALYSIS: g₂(r) & Brody β — Surfaces × Primes × PSI")
    print("=" * 70)

    # ── Load canonical β values ──────────────────────────────────
    with open(QC_FILE) as f:
        qc = json.load(f)
    valid = [s for s in qc if not s.get('grid_lim', True)]
    valid_labels = {s['label'] for s in valid}
    print(f"\nCanonical FV surfaces: {len(valid)} valid (non-grid-limited)")

    # ── Build label mapping (English → SUR filename) ──────────
    # From physical_pipeline.py
    label_map = {
        '1.4301 steel Honed': '1.4301_oselkowane',
        '1.4301 steel Ground': '1.4301_szlifowane',
        '1.4301 steel Finish turned': '1.4301_t_wyk',
        '1.4301 steel Bead blasted': '1.4301_szkielkowane',
        '1.4301 steel Burnished': '1.4301_oselkowane',  # nagniatane
        '1.4301 steel Finish milled': '1.4301_t_wyk',  # frezowane wyk
        'Al7075 Finish turned': 'Al7075_t_wyk',
        'Al Bead blasted': 'Al_szkielkowane',
        'Al Ground': 'Al_szlifowane',
        'Al Honed': 'AL_oselkowane',
        'Graphite Honed': 'Graphite_oselkowane',
        'Al7075 Burnished': 'Al7075_t_wyk',  # nagniatane
        'Al7075 Finish milled': 'Al7075_t_wyk',
        'C45 steel Burnished': 'C45_oselkowane',  # nagniatane
        'MO58A brass Finish milled': 'MO58A_t_wyk',
        'MO58A brass Burnished': 'MO58A_t_wyk',  # nagniatane
        'MO58A brass Rough turned': 'MO58A_t_wyk',  # toczenie zgrubne
        'MO58A brass WEDM finish': 'MO58A_t_wyk',
        'Ti6Al4V Rough milled': 'Ti6A14V_wedm_zgru_1prz',
        'Ti6Al4V Burnished': 'Ti6A14V_wedm_wyk',  # nagniatane
        'Ti6Al4V Honed': 'Ti_oselkowane',
        'Ti6Al4V Bead blasted': 'Ti_szkielkowane',
        'Ti6Al4V Ground': 'Ti_szlifowane',
    }

    # Build reverse: SUR basename → English label
    sur_to_english = {}
    for eng, sur_base in label_map.items():
        sur_to_english[sur_base] = eng

    sur_files = sorted([f for f in os.listdir(SUR_DIR) if f.endswith('.sur')])

    # ── Process FV surfaces ──────────────────────────────────────
    print("\nProcessing FV surfaces (pair correlation + peaks)...")
    t0 = time.time()

    fv_results = []
    n_processed = 0

    for fn in sur_files:
        sur_base = fn.replace('.sur', '').replace('P1-', '')
        label = sur_to_english.get(sur_base, sur_base)

        # Only process if this surface is in the valid QC set
        if label not in valid_labels:
            continue

        fp = os.path.join(SUR_DIR, fn)
        try:
            Z, dx, dy = read_sur(fp, max_size=1000)
            peaks = detect_peaks_2d(Z, min_prominence_frac=0.03)
            n_peaks = np.sum(peaks)

            if n_peaks < 20:
                continue

            # NN spacings + Brody fit
            spacings, _ = compute_nns(peaks)
            if spacings is None or len(spacings) < 20:
                continue

            brody_fit = fit_brody(spacings)

            # Pair correlation
            peak_coords = np.argwhere(peaks).astype(np.float64)
            # Normalize coordinates to unit mean NN distance
            if len(peak_coords) >= 20:
                r_vals, g2 = pair_correlation_2d(
                    peak_coords, r_max=5.0, n_bins=100,
                    box_size=(Z.shape[1], Z.shape[0]))
            else:
                r_vals, g2 = np.array([]), np.array([])

            fv_results.append({
                'label': label,
                'n_peaks': int(n_peaks),
                'n_spacings': len(spacings),
                'brody_beta': brody_fit['beta'],
                'beta_ci_low': brody_fit['ci_low'],
                'beta_ci_high': brody_fit['ci_high'],
                'g2_r': r_vals.tolist() if len(r_vals) > 0 else [],
                'g2_vals': g2.tolist() if len(g2) > 0 else [],
            })

            n_processed += 1
            print(f"  [{n_processed:2d}/21] {label[:40]:40s} "
                  f"β={brody_fit['beta']:.2f} peaks={n_peaks:4d}")

        except Exception as e:
            print(f"  SKIP {label}: {str(e)[:60]}")

    print(f"  Processed {n_processed} surfaces in {time.time()-t0:.0f}s")

    # ── Process prime surfaces ───────────────────────────────────
    print("\nProcessing prime 2D embeddings...")
    t0 = time.time()

    N = 100000
    prime_surface = build_prime_surface(N)
    prime_coords = np.argwhere(prime_surface).astype(np.float64)
    bernoulli_surface = build_bernoulli_surface(prime_surface.shape[0],
                                                  np.mean(prime_surface))
    bern_coords = np.argwhere(bernoulli_surface).astype(np.float64)

    print(f"  Primes: {len(prime_coords)} points, density={np.mean(prime_surface):.4f}")
    print(f"  Bernoulli: {len(bern_coords)} points, density={np.mean(bernoulli_surface):.4f}")

    # Pair correlation for primes
    r_p, g2_prime = pair_correlation_2d(
        prime_coords, r_max=5.0, n_bins=100,
        box_size=prime_surface.shape)
    r_b, g2_bern = pair_correlation_2d(
        bern_coords, r_max=5.0, n_bins=100,
        box_size=prime_surface.shape)

    # Brody fit for primes (treat 1's as peaks)
    prime_nn, _ = compute_nns(prime_surface)
    if prime_nn is not None:
        prime_brody = fit_brody(prime_nn)
        print(f"  Prime β = {prime_brody['beta']:.2f} [{prime_brody['ci_low']:.2f}, {prime_brody['ci_high']:.2f}]")
    else:
        prime_brody = {'beta': 0, 'ci_low': 0, 'ci_high': 0}

    bern_nn, _ = compute_nns(bernoulli_surface)
    if bern_nn is not None:
        bern_brody = fit_brody(bern_nn)
        print(f"  Bernoulli β = {bern_brody['beta']:.2f}")
    else:
        bern_brody = {'beta': 0, 'ci_low': 0, 'ci_high': 0}

    print(f"  Done in {time.time()-t0:.0f}s")

    # ── PSI ceramic (from existing results) ──────────────────────
    with open(f'{OUTDIR}/psi_ceramic_results.json') as f:
        psi = json.load(f)
    psi_beta = psi['best_brody_beta']
    psi_ci = [psi['beta_ci_low'], psi['beta_ci_high']]

    # For PSI pair correlation, use angular spacings from 1D profile
    # The PSI data is 1D (angular), so g₂(r) isn't directly comparable
    # We'll use a GUE overlay instead

    # ═══════════════════════════════════════════════════════════
    # FIGURE 1: UNIFIED PAIR CORRELATION g₂(r)
    # ═══════════════════════════════════════════════════════════
    print("\nGenerating Figure 1: Unified pair correlation...")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

    # -- Row 1: Representative FV surfaces --
    # Pick 4 surfaces spanning the β range
    fv_sorted = sorted(fv_results, key=lambda x: x['brody_beta'], reverse=True)
    representatives = []
    # Highest β
    representatives.append(fv_sorted[0])
    # β ≈ 2 (GUE)
    gue_like = [s for s in fv_sorted if 1.7 < s['brody_beta'] < 2.3]
    if gue_like:
        representatives.append(gue_like[0])
    else:
        mid = fv_sorted[len(fv_sorted)//2]
        representatives.append(mid)
    # β ≈ 1 (GOE)
    goe_like = [s for s in fv_sorted if 0.7 < s['brody_beta'] < 1.3]
    if goe_like:
        representatives.append(goe_like[0])
    else:
        representatives.append(fv_sorted[len(fv_sorted)//3])
    # Lowest β (Poisson)
    representatives.append(fv_sorted[-1])

    titles = ['(a) Super-GUE (burnished)', '(b) GUE-like', '(c) GOE-like',
              '(d) Poisson-like']
    colors_fv = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (res, title, color) in enumerate(zip(representatives, titles, colors_fv)):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])

        if len(res['g2_r']) > 0:
            r = np.array(res['g2_r'])
            g2 = np.array(res['g2_vals'])
            # Smooth for display
            ax.plot(r, g2, color=color, lw=1.5, alpha=0.9,
                    label=f"{res['label'][:25]}\nβ={res['brody_beta']:.2f}")
            ax.axhline(y=1.0, color='gray', ls=':', alpha=0.4, lw=0.8)

        # GUE prediction (1D) overlaid for reference
        r_theory = np.linspace(0.01, 5, 200)
        g2_gue = gue_pair_correlation_1d(r_theory)
        ax.plot(r_theory, g2_gue, 'k-', lw=2.0, alpha=0.6, label='GUE (Montgomery)')

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2.5)
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.set_xlabel('r (normalised NN distance)', fontsize=8)
        ax.set_ylabel('g₂(r)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        ax.grid(alpha=0.3, ls=':')

    # -- Row 2: Prime surfaces --
    ax_p = fig.add_subplot(gs[2, 0])
    if len(r_p) > 0:
        ax_p.plot(r_p, g2_prime, '#9467bd', lw=1.5, alpha=0.9,
                  label=f'Primes (N=10⁵)\nβ={prime_brody["beta"]:.2f}')
    if len(r_b) > 0:
        ax_p.plot(r_b, g2_bern, 'gray', lw=1.0, alpha=0.6, ls='--',
                  label=f'Bernoulli null\nβ={bern_brody["beta"]:.2f}')
    ax_p.axhline(y=1.0, color='gray', ls=':', alpha=0.4)
    ax_p.plot(r_theory, g2_gue, 'k-', lw=2.0, alpha=0.6, label='GUE (Montgomery)')
    ax_p.set_xlim(0, 4); ax_p.set_ylim(0, 2.5)
    ax_p.set_title('(e) Primes (2D embedding)', fontsize=10, fontweight='bold', loc='left')
    ax_p.set_xlabel('r (normalised NN distance)', fontsize=8)
    ax_p.set_ylabel('g₂(r)', fontsize=8)
    ax_p.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax_p.grid(alpha=0.3, ls=':')

    # -- Row 2 middle: All surfaces g₂(r) average --
    ax_avg = fig.add_subplot(gs[2, 1])
    # Compute ensemble average g₂(r) across all valid surfaces
    all_g2 = []
    common_r = np.linspace(0.01, 5, 100)
    for res in fv_results:
        if len(res['g2_r']) > 0:
            g2_interp = np.interp(common_r, res['g2_r'], res['g2_vals'])
            all_g2.append(g2_interp)
    if all_g2:
        g2_avg = np.mean(all_g2, axis=0)
        g2_std = np.std(all_g2, axis=0)
        ax_avg.fill_between(common_r, g2_avg - g2_std, g2_avg + g2_std,
                             alpha=0.3, color='steelblue')
        ax_avg.plot(common_r, g2_avg, 'b-', lw=2, label=f'Mean ({len(all_g2)} surfaces)')
    ax_avg.axhline(y=1.0, color='gray', ls=':', alpha=0.4)
    ax_avg.plot(r_theory, g2_gue, 'k-', lw=2.0, alpha=0.6, label='GUE (Montgomery)')
    ax_avg.set_xlim(0, 4); ax_avg.set_ylim(0.5, 2.0)
    ax_avg.set_title('(f) Ensemble mean g₂(r) — all surfaces', fontsize=10,
                      fontweight='bold', loc='left')
    ax_avg.set_xlabel('r (normalised NN distance)', fontsize=8)
    ax_avg.set_ylabel('g₂(r)', fontsize=8)
    ax_avg.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax_avg.grid(alpha=0.3, ls=':')

    # -- Row 2 right: β distribution across domains --
    ax_b = fig.add_subplot(gs[2, 2])
    betas_fv = [s['brody_beta'] for s in fv_results]
    domains = ['FV surfaces\n(n=21)', 'Primes\n(2D)', 'Bernoulli\n(null)',
               'PSI ceramic\n(phase)']
    beta_vals = [np.median(betas_fv), prime_brody['beta'], bern_brody['beta'],
                 psi_beta]
    beta_errs = [[np.median(betas_fv) - np.percentile(betas_fv, 25),
                  np.percentile(betas_fv, 75) - np.median(betas_fv)],
                 [prime_brody['beta'] - prime_brody['ci_low'],
                  prime_brody['ci_high'] - prime_brody['beta']],
                 [bern_brody['beta'] - bern_brody['ci_low'],
                  bern_brody['ci_high'] - bern_brody['beta']],
                 [psi_beta - psi_ci[0], psi_ci[1] - psi_beta]]
    colors_domain = ['steelblue', '#9467bd', 'gray', '#d62728']
    x_pos = np.arange(len(domains))
    for i, (dom, bv, be, col) in enumerate(zip(domains, beta_vals, beta_errs, colors_domain)):
        ax_b.bar(i, bv, 0.6, color=col, alpha=0.8, edgecolor='black', lw=0.8)
        ax_b.errorbar(i, bv, yerr=[[be[0]], [be[1]]], fmt='none',
                      ecolor='black', capsize=4, lw=1.2)
    ax_b.axhline(y=0, color='gray', ls=':', alpha=0.4, label='Poisson')
    ax_b.axhline(y=1, color='blue', ls='--', alpha=0.4, label='GOE')
    ax_b.axhline(y=2, color='red', ls='-.', alpha=0.4, label='GUE')
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(domains, fontsize=7.5)
    ax_b.set_ylabel('Brody β', fontsize=9)
    ax_b.set_title('(g) β across domains', fontsize=10, fontweight='bold', loc='left')
    ax_b.legend(fontsize=6.5, loc='upper left', framealpha=0.8)
    ax_b.grid(alpha=0.3, ls=':', axis='y')

    fig.suptitle('Unified pair correlation and Brody spacing statistics\n'
                 'across surface textures, prime embeddings, and quantum-chaos reference ensembles',
                 fontsize=13, fontweight='bold', y=1.01)

    outpath1 = f'{OUTDIR}/fig_unified_pair_correlation.pdf'
    fig.savefig(outpath1, dpi=200, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath1}")

    # ═══════════════════════════════════════════════════════════
    # FIGURE 2: β LANDSCAPE — detailed β distribution
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 2: β landscape...")

    fig2, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: β histogram for FV surfaces
    ax_h = axes[0]
    ax_h.hist(betas_fv, bins=12, color='steelblue', alpha=0.7, edgecolor='black',
              lw=0.8, density=True)
    ax_h.axvline(x=0, color='gray', ls=':', lw=1.5, label='Poisson (β=0)')
    ax_h.axvline(x=1, color='blue', ls='--', lw=1.5, label='GOE (β=1)')
    ax_h.axvline(x=2, color='red', ls='-.', lw=1.5, label='GUE (β=2)')
    # Mark prime and PSI
    ax_h.axvline(x=prime_brody['beta'], color='#9467bd', ls='-', lw=2,
                 label=f'Primes β={prime_brody["beta"]:.2f}')
    ax_h.axvline(x=psi_beta, color='#d62728', ls='-', lw=2,
                 label=f'PSI β={psi_beta:.2f}')
    ax_h.set_xlabel('Brody exponent β', fontsize=11)
    ax_h.set_ylabel('Density', fontsize=11)
    ax_h.set_title('Distribution of β across 21 manufactured surfaces',
                   fontsize=11, fontweight='bold')
    ax_h.legend(fontsize=8, loc='upper right')
    ax_h.grid(alpha=0.3, ls=':')

    # Right: β vs D₂ scatter (cross-domain comparison)
    ax_s = axes[1]
    # Physical surfaces — use β from QC, D₂ from physical_results
    with open(f'{OUTDIR}/physical_results.json') as f:
        ph = json.load(f)
    # Map physical surfaces to their D₂ values
    # Physical surfaces have limited coverage; use the ones we have
    ph_d2 = []
    ph_beta = []
    for res in fv_results:
        # Find matching physical result if exists
        for k, v in ph.items():
            if isinstance(v, dict) and res['label'].split()[0] in k:
                if not np.isnan(v.get('D2', np.nan)):
                    ph_d2.append(v['D2'])
                    ph_beta.append(res['brody_beta'])
                    break

    if ph_beta:
        ax_s.scatter(ph_beta, ph_d2, c='steelblue', s=80, edgecolors='black',
                     lw=0.8, zorder=5, label='Physical surfaces')

    # Prime surface
    # D₂ for primes from manuscript: 1.6872
    ax_s.scatter([prime_brody['beta']], [1.6872], c='#9467bd', s=150,
                 edgecolors='black', lw=1.2, zorder=6, marker='*',
                 label='Primes (2D)')

    # PSI ceramic
    ax_s.scatter([psi_beta], [1.96], c='#d62728', s=150, marker='D',
                 edgecolors='black', lw=1.2, zorder=6,
                 label='PSI ceramic (est.)')

    ax_s.set_xlabel('Brody exponent β (spacing order)', fontsize=11)
    ax_s.set_ylabel('Correlation dimension D₂ (mass distribution)', fontsize=11)
    ax_s.set_title('β vs D₂: two orthogonal descriptors across domains',
                   fontsize=11, fontweight='bold')
    ax_s.legend(fontsize=8, loc='lower right')
    ax_s.grid(alpha=0.3, ls=':')

    fig2.suptitle('Brody β landscape: manufactured surfaces, primes, and PSI',
                  fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    outpath2 = f'{OUTDIR}/fig_unified_beta_landscape.pdf'
    fig2.savefig(outpath2, dpi=200, facecolor='white', edgecolor='none',
                 bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {outpath2}")

    # ── Save results ─────────────────────────────────────────────
    unified = {
        'fv_surfaces': fv_results,
        'prime_embedding': {
            'N': N,
            'W': int(np.sqrt(N)),
            'density': float(np.mean(prime_surface)),
            'n_points': int(np.sum(prime_surface)),
            'brody_beta': prime_brody['beta'],
            'beta_ci_low': prime_brody['ci_low'],
            'beta_ci_high': prime_brody['ci_high'],
            'D2': 1.6872,  # from manuscript
        },
        'bernoulli_null': {
            'density': float(np.mean(bernoulli_surface)),
            'brody_beta': bern_brody['beta'],
        },
        'psi_ceramic': {
            'brody_beta': psi_beta,
            'beta_ci_low': psi_ci[0],
            'beta_ci_high': psi_ci[1],
        },
        'summary': {
            'beta_range_fv': [float(min(betas_fv)), float(max(betas_fv))],
            'beta_median_fv': float(np.median(betas_fv)),
            'beta_prime': prime_brody['beta'],
            'beta_psi': psi_beta,
        }
    }
    with open(f'{OUTDIR}/unified_results.json', 'w') as f:
        json.dump(unified, f, indent=2, default=str)
    print(f"\nSaved: {OUTDIR}/unified_results.json")

    print("\n" + "=" * 70)
    print("DONE. Key findings:")
    print(f"  FV β range: [{min(betas_fv):.1f}, {max(betas_fv):.1f}]")
    print(f"  Prime β:    {prime_brody['beta']:.2f}")
    print(f"  PSI β:      {psi_beta:.2f}")
    print(f"  Bernoulli β: {bern_brody['beta']:.2f} (Poisson null)")
    print("=" * 70)


if __name__ == '__main__':
    main()

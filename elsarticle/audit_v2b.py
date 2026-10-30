#!/usr/bin/env python3
"""audit_v2b.py — corrected follow-ups to audit_v2.py.

1. Toroidal NN (self-pair excluded properly).
2. Box-counting D2 with per-scale mass normalisation.
3. Grassberger-Procaccia for null ensembles.
4. Ripley K nulls with the SAME n_bins=18 grid as deterministic surfaces.
5. g2 at exact lattice distances with complete shift classes.
"""
import numpy as np, json, time
from scipy import spatial
from audit_v2 import (
    W, N_POP, primes_upto, row_major_surface, ulam_even_surface,
    cantor_surface, coords, bernoulli_surface, sparse_integer_surface,
    inhomogeneous_surface, fit_brody, nn_stats)
from benchmark_descriptors import rip_k_border

rho = 9580 / N_POP

np.random.seed(2026)

# ── 1. Toroidal NN, self-excluded ────────────────────────────────
def nn_toroidal(pts, W):
    n = len(pts)
    tree = spatial.KDTree(pts)
    best = np.full(n, np.inf)
    for dx in (-W, 0, W):
        for dy in (-W, 0, W):
            shifted = pts + np.array([dx, dy])
            if dx == 0 and dy == 0:
                d, _ = tree.query(shifted, k=2)      # exclude self
                d = d[:, 1]
            else:
                d, _ = tree.query(shifted, k=1)
            best = np.minimum(best, d)
    return best

# ── 2. Box-counting D2 with per-scale mass normalisation ─────────
def d2_boxcount_corr(surf):
    H, W_ = surf.shape
    Z = surf.astype(float)
    max_k = int(np.log2(min(H, W_))) - 1
    eps_list, z2_list = [], []
    for k in range(1, max_k + 1):
        eps = 2 ** k
        nh, nw = H // eps, W_ // eps
        if nh < 2 or nw < 2:
            continue
        crop = Z[:nh * eps, :nw * eps]
        M_eps = crop.sum()
        if M_eps <= 0:
            continue
        boxes = crop.reshape(nh, eps, nw, eps)
        masses = boxes.sum(axis=(1, 3))
        z2 = np.sum((masses / M_eps) ** 2)
        eps_list.append(np.log(eps))
        z2_list.append(np.log(z2))
    eps_arr = np.array(eps_list); z2_arr = np.array(z2_list)
    A = np.vstack([eps_arr, np.ones(len(eps_arr))]).T
    (slope, intercept), res, *_ = np.linalg.lstsq(A, z2_arr, rcond=None)
    fitted = A @ np.array([slope, intercept])
    ss_res = float(np.sum((z2_arr - fitted) ** 2))
    ss_tot = float(np.sum((z2_arr - z2_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # large-epsilon subrange slope (eps >= 8)
    m8 = eps_arr >= np.log(8.0)
    s8 = np.nan
    if m8.sum() >= 3:
        A8 = np.vstack([eps_arr[m8], np.ones(m8.sum())]).T
        (s8, _), *_ = np.linalg.lstsq(A8, z2_arr[m8], rcond=None)
        s8 = float(s8)
    return {'d2': float(slope), 'r2': r2, 'd2_eps_ge8': s8,
            'eps': eps_arr.tolist(), 'logz2': z2_arr.tolist()}

# ── 3. GP via query_pairs, reused for nulls ─────────────────────
def gp_pair(pts, W, n_radii=12):
    from scipy import spatial as sp
    n = len(pts)
    if n < 100:
        return None
    tree = sp.KDTree(pts)
    radii = np.logspace(np.log10(2.0), np.log10(W / 3.0), n_radii)
    logr, logC = [], []
    for r in radii:
        pairs = tree.query_pairs(r, output_type='ndarray')
        C = 2.0 * len(pairs) / (n * (n - 1))
        if C > 0:
            logr.append(np.log(r)); logC.append(np.log(C))
    logr = np.array(logr); logC = np.array(logC)
    A = np.vstack([logr, np.ones(len(logr))]).T
    (slope, _), *_ = np.linalg.lstsq(A, logC, rcond=None)
    return float(slope)

# ── 5. g2 exact with complete shift classes ─────────────────────
CLASSES = {
    'r1':     {'m': 2, 'reps': [(1, 0), (0, 1)]},
    'rsqrt2': {'m': 4, 'reps': [(1, 1), (1, -1)]},
    'r2':     {'m': 2, 'reps': [(2, 0), (0, 2)]},
    'rsqrt5': {'m': 8, 'reps': [(1, 2), (1, -2), (2, 1), (2, -1)]},
    'r4':     {'m': 2, 'reps': [(4, 0), (0, 4)]},
}

def g2_exact_full(surf):
    N = int(surf.sum()); rho_ = surf.mean()
    out = {}
    for name, spec in CLASSES.items():
        tot = 0.0
        for dx, dy in spec['reps']:
            Ss = np.roll(np.roll(surf, dx, axis=1), dy, axis=0)
            tot += int((surf & Ss).sum()) / 2.0
        expected = N * rho_ * spec['m'] / 2.0
        out[name] = tot / expected if expected > 0 else np.nan
    return out

def main():
    t0 = time.time()
    out = {}
    primes = primes_upto(N_POP)
    surf_rm = row_major_surface(primes, N_POP, W)
    surf_ul = ulam_even_surface(primes, W)
    surf_ct = cantor_surface(primes, N_POP, W)

    # toroidal beta
    for nm, surf in [('row_major', surf_rm), ('ulam', surf_ul), ('cantor', surf_ct)]:
        pts = coords(surf)
        nn_t = nn_toroidal(pts, W)
        bf = fit_brody(nn_t, n_bootstrap=2000, rng=np.random.default_rng(0))
        print(f"{nm}: beta_toroidal={bf['beta']:.3f} [{bf['ci_low']:.3f},{bf['ci_high']:.3f}]", flush=True)
        out.setdefault('toroidal', {})[nm] = {'beta': float(bf['beta']),
                                              'ci': [float(bf['ci_low']), float(bf['ci_high'])]}

    # corrected box-counting
    for nm, surf in [('row_major', surf_rm), ('ulam', surf_ul), ('cantor', surf_ct)]:
        d = d2_boxcount_corr(surf)
        print(f"{nm}: D2_box_corr={d['d2']:.4f} (R2={d['r2']:.4f}) eps>=8: {d['d2_eps_ge8']:.4f}", flush=True)
        out.setdefault('boxcount_corr', {})[nm] = d

    # corrected g2 exact
    for nm, surf in [('row_major', surf_rm), ('ulam', surf_ul), ('cantor', surf_ct)]:
        g = g2_exact_full(surf)
        print(f"{nm}: g2_exact={ {k: round(v,3) for k,v in g.items()} }", flush=True)
        out.setdefault('g2_exact_full', {})[nm] = g

    # nulls: corrected box-counting D2 + GP + toroidal beta (subsamples)
    for name, gen in [
        ('bernoulli', lambda rng: bernoulli_surface(W, rho)),
        ('inhomogeneous', lambda rng: inhomogeneous_surface(W, 9580, rng)),
    ]:
        print(f'null {name}: corrected D2 + GP ...', flush=True)
        rng = np.random.default_rng(hash(name) % (2**32))
        d2s, d2s8, gps, bt = [], [], [], []
        for i in range(500):
            surf = gen(rng)
            d = d2_boxcount_corr(surf)
            d2s.append(d['d2']); d2s8.append(d['d2_eps_ge8'])
            if i < 100:
                gps.append(gp_pair(coords(surf), W))
            if i < 100:
                nnt = nn_toroidal(coords(surf), W)
                bt.append(float(fit_brody(nnt, n_bootstrap=0, rng=np.random.default_rng(0))['beta']))
        d2s = np.array(d2s); d2s8 = np.array(d2s8); gps = np.array(gps); bt = np.array(bt)
        print(f"   D2_corr {d2s.mean():.4f}+-{d2s.std(ddof=1):.4f} | eps>=8 {d2s8.mean():.4f}+-{d2s8.std(ddof=1):.4f} | GP {gps.mean():.4f}+-{gps.std(ddof=1):.4f} | beta_tor {bt.mean():.4f}+-{bt.std(ddof=1):.4f}", flush=True)
        out.setdefault('nulls', {})[name] = {
            'd2_corr_mean': float(d2s.mean()), 'd2_corr_sd': float(d2s.std(ddof=1)),
            'd2_ge8_mean': float(d2s8.mean()), 'd2_ge8_sd': float(d2s8.std(ddof=1)),
            'gp_mean': float(gps.mean()), 'gp_sd': float(gps.std(ddof=1)),
            'beta_toroidal_mean': float(bt.mean()), 'beta_toroidal_sd': float(bt.std(ddof=1)),
            'n': 500}

    # Ripley nulls with n_bins=18 (same grid as deterministic)
    print('Ripley nulls n_bins=18 ...', flush=True)
    bern_L = []
    for _ in range(100):
        surf = bernoulli_surface(W, rho)
        pts = coords(surf)
        _, _, L, maxdev, L_nn = rip_k_border(pts, W, W, r_max=W / 3.0, n_bins=18)
        bern_L.append({'maxdev': float(maxdev), 'L_at_nn': float(L_nn)})
    md = np.array([b['maxdev'] for b in bern_L]); ln = np.array([b['L_at_nn'] for b in bern_L])
    print(f"   maxdev {md.mean():.3f}+-{md.std(ddof=1):.3f} | L_at_nn {ln.mean():.3f}+-{ln.std(ddof=1):.3f}", flush=True)
    out['rip_k_bernoulli_18bins'] = {
        'maxdev_mean': float(md.mean()), 'maxdev_sd': float(md.std(ddof=1)),
        'L_at_nn_mean': float(ln.mean()), 'L_at_nn_sd': float(ln.std(ddof=1))}

    json.dump(out, open('/Users/dawid/Projects/interfero-Riemann/elsarticle/audit_v2b_results.json', 'w'), indent=1, default=float)
    print(f'DONE in {(time.time()-t0)/60:.1f} min', flush=True)

if __name__ == '__main__':
    main()

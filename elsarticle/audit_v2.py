#!/usr/bin/env python3
"""
audit_v2.py — full numerical audit of the prime-embedding study.

Addresses the critical review:
  1. ONE common prime population: all primes <= W^2 = 99856 (count 9580),
     window 316x316 for ALL THREE embeddings (Ulam on even side, start
     (157,158), E,N,W,S).
  2. NN boundary sensitivity: Euclidean, toroidal, interior-reference.
  3. Inhomogeneous Bernoulli null with p(v) = c/ln v.
  4. Null ensembles of 1000 realisations (beta, NNI, D2, g2).
  5. Distance-specific pair correlations at r = 1, sqrt2, 2, sqrt5, 4.
  6. D2: box-counting with regression diagnostics + Grassberger-Procaccia
     pairwise validation.
  7. Density thinning with confidence intervals and matched nulls.
  8. Finite-size beta(N) with matched nulls.
"""
import numpy as np, json, time
from scipy import optimize, spatial
from scipy.special import gamma as gamma_func

np.random.seed(2026)
OUTDIR = '/Users/dawid/Projects/interfero-Riemann/elsarticle'
W = 316
N_POP = W * W          # 99,856 — one common integer population
EPS = np.finfo(float).eps

from benchmark_descriptors import (
    fit_brody, primes_upto, row_major_surface, cantor_surface,
    bernoulli_surface, sparse_integer_surface, coords, rip_k_border)

# ── Ulam spiral on an EVEN side (verified bijection) ──────────────
def ulam_even_surface(values, W):
    side = W
    surf = np.zeros((side, side), dtype=bool)
    pos = {}
    x, y = side // 2 - 1, side // 2
    pos[1] = (x, y)
    n = 1
    step = 1
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # E, N, W, S
    total = side * side
    while n < total:
        for di, (dx, dy) in enumerate(dirs):
            for _ in range(step):
                x += dx; y += dy
                n += 1
                if n > total:
                    break
                pos[n] = (x, y)
            if di % 2 == 1:
                step += 1
            if n >= total:
                break
    for v in values:
        if v in pos:
            x, y = pos[v]
            surf[y, x] = True
    return surf

# ── Nearest-neighbour distances under 3 boundary treatments ───────
def nn_stats(pts):
    if len(pts) < 2:
        return None
    tree = spatial.KDTree(pts)
    d, _ = tree.query(pts, k=2)
    return d[:, 1]

def nn_toroidal(pts, W):
    """Nearest-neighbour distances with periodic boundary conditions."""
    if len(pts) < 2:
        return None
    tree = spatial.cKDTree(pts, boxsize=W)
    distances, _ = tree.query(pts, k=2)
    return distances[:, 1]

def nn_interior(pts, W, margin=5):
    """NN distances from interior reference points only (Euclidean)."""
    centres = pts + 0.5
    mask = ((centres[:, 0] >= margin) & (centres[:, 0] <= W - margin) &
            (centres[:, 1] >= margin) & (centres[:, 1] <= W - margin))
    ref = pts[mask]
    if len(ref) < 2:
        return None
    tree = spatial.KDTree(pts)
    d, _ = tree.query(ref, k=2)
    return d[:, 1]

def beta_summary(nn, n_bootstrap=2000, seed=0):
    rng = np.random.default_rng(seed)
    res = fit_brody(nn, n_bootstrap=n_bootstrap, rng=rng)
    return float(res['beta']), float(res['ci_low']), float(res['ci_high'])

# ── Exact lattice pair correlations via cyclic cross-correlation ──
def g2_exact(surf, shifts):
    """g2 at exact toroidal lattice distances.
    shifts: list of (name, dx, dy, k) where k = number of symmetric unit
    directions at that distance (2 for axes, 4 for diagonals)."""
    N = int(surf.sum())
    rho = surf.mean()
    out = {}
    for name, dx, dy, k in shifts:
        Ss = np.roll(np.roll(surf, dx, axis=1), dy, axis=0)
        tot = int((surf & Ss).sum())          # counts each unordered pair twice
        expected = N * rho * k / 2.0
        out[name] = (tot / 2.0) / expected if expected > 0 else np.nan
    return out

SHIFTS = [('r1', 1, 0, 2), ('rsqrt2', 1, 1, 4), ('r2', 2, 0, 2),
          ('rsqrt5', 1, 2, 4), ('r4', 4, 0, 2)]

def g2_annulus_toroidal(pts, W, r_min=1.0, r_max=3.0, n_sample=3000, seed=0):
    if len(pts) < 2 or not 0 <= r_min < r_max <= W / 2 or n_sample < 1:
        raise ValueError('invalid points, annulus radii, or sample size')
    rng = np.random.default_rng(seed)
    n = len(pts)
    rho = n / (W * W)
    tree = spatial.cKDTree(pts, boxsize=W)
    idx = rng.choice(n, size=min(n, n_sample), replace=False)
    area_ann = np.pi * (r_max ** 2 - r_min ** 2)
    outer = tree.query_ball_point(pts[idx], np.nextafter(r_max, -np.inf), return_length=True)
    inner_radius = max(0.0, np.nextafter(r_min, -np.inf))
    inner = tree.query_ball_point(pts[idx], inner_radius, return_length=True)
    pair_count = int(np.sum(outer - inner))
    return float((pair_count / len(idx)) / (rho * area_ann))

# ── Inhomogeneous Bernoulli null (prime-number-theorem density) ───
def inhomogeneous_probabilities(W, target):
    """Return probabilities and calibration with p(1)=0 and p(2)=1."""
    if W * W < 3 or not 1 <= target <= W * W - 1:
        raise ValueError('target must be between 1 and W*W - 1')
    v = np.arange(1, W * W + 1)
    wts = np.zeros_like(v, dtype=float)
    wts[2:] = 1.0 / np.log(v[2:])
    c = (target - 1) / wts.sum()
    if np.any(c * wts[2:] > 1):
        c = optimize.brentq(
            lambda value: np.minimum(1.0, value * wts[2:]).sum() + 1 - target,
            0.0, np.log(W * W))
    p = np.minimum(1.0, c * wts)
    p[1] = 1.0
    return p, float(c)

def inhomogeneous_surface(W, target, rng):
    """Draw a row-major inhomogeneous Bernoulli field with the given mean count."""
    p, _ = inhomogeneous_probabilities(W, target)
    return rng.random((W, W)) < p.reshape(W, W)

# ── D2: box counting with diagnostics ─────────────────────────────
def d2_boxcount_diag(surf):
    H, W_ = surf.shape
    Z = surf.astype(float)
    M = Z.sum()
    if M == 0:
        return None
    max_k = int(np.log2(min(H, W_))) - 1
    eps_list, z2_list = [], []
    for k in range(1, max_k + 1):
        eps = 2 ** k
        nh, nw = H // eps, W_ // eps
        if nh < 2 or nw < 2:
            continue
        boxes = Z[:nh * eps, :nw * eps].reshape(nh, eps, nw, eps)
        masses = boxes.sum(axis=(1, 3))
        z2 = np.sum((masses / M) ** 2)
        eps_list.append(np.log(eps))
        z2_list.append(np.log(z2))
    eps_arr = np.array(eps_list); z2_arr = np.array(z2_list)
    if len(eps_arr) < 3:
        return None
    A = np.vstack([eps_arr, np.ones(len(eps_arr))]).T
    (slope, intercept), res, *_ = np.linalg.lstsq(A, z2_arr, rcond=None)
    fitted = A @ np.array([slope, intercept])
    ss_res = float(np.sum((z2_arr - fitted) ** 2))
    ss_tot = float(np.sum((z2_arr - z2_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'d2': float(slope), 'r2': float(r2), 'n_boxes': len(eps_arr),
            'eps': eps_arr.tolist(), 'logz2': z2_arr.tolist()}

# ── Grassberger-Procaccia pairwise correlation sum ────────────────
def gp_correlation_dim(pts, W, n_radii=12):
    """Classical pairwise correlation sum C(r) = 2/(n(n-1)) * #{i<j: d_ij<r},
    computed exactly via KDTree query_pairs; slope of log C vs log r."""
    n = len(pts)
    if n < 100:
        return None
    tree = spatial.KDTree(pts)
    radii = np.logspace(np.log10(2.0), np.log10(W / 3.0), n_radii)
    logr, logC = [], []
    for r in radii:
        pairs = tree.query_pairs(r, output_type='ndarray')
        C = 2.0 * len(pairs) / (n * (n - 1))
        if C > 0:
            logr.append(np.log(r)); logC.append(np.log(C))
    logr = np.array(logr); logC = np.array(logC)
    if len(logr) < 4:
        return None
    A = np.vstack([logr, np.ones(len(logr))]).T
    (slope, intercept), res, *_ = np.linalg.lstsq(A, logC, rcond=None)
    fitted = A @ np.array([slope, intercept])
    ss_res = float(np.sum((logC - fitted) ** 2))
    ss_tot = float(np.sum((logC - logC.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'d2_gp': float(slope), 'r2': float(r2),
            'r_range': [float(np.exp(logr[0])), float(np.exp(logr[-1]))]}

def gp_correlation_dim_sampled(pts, W, n_sample=1000, n_radii=10, seed=0):
    """GP slope from a random subset of reference points (for nulls)."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n < 100:
        return None
    sample = rng.choice(n, size=min(n, n_sample), replace=False)
    tree = spatial.KDTree(pts)
    radii = np.logspace(np.log10(2.0), np.log10(W / 3.0), n_radii)
    logr, logC = [], []
    for r in radii:
        tot = 0.0
        for i in sample:
            tot += len(tree.query_ball_point(pts[i], r)) - 1
        C = tot / (len(sample) * (n - 1))
        if C > 0:
            logr.append(np.log(r)); logC.append(np.log(C))
    logr = np.array(logr); logC = np.array(logC)
    if len(logr) < 4:
        return None
    A = np.vstack([logr, np.ones(len(logr))]).T
    (slope, intercept), res, *_ = np.linalg.lstsq(A, logC, rcond=None)
    fitted = A @ np.array([slope, intercept])
    ss_res = float(np.sum((logC - fitted) ** 2))
    ss_tot = float(np.sum((logC - logC.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'d2_gp': float(slope), 'r2': float(r2)}

# ── Quick descriptor for null ensembles (beta, NNI, D2) ───────────
def quick_stats(surf, W, rng, fit_seed=0, bootstrap=0):
    pts = coords(surf)
    n = len(pts)
    if n < 10:
        return None
    nn = nn_stats(pts)
    res = fit_brody(nn, n_bootstrap=bootstrap, rng=np.random.default_rng(fit_seed))
    rho = n / (W * W)
    nni = float(np.mean(nn) * 2.0 * np.sqrt(rho))
    d2r = d2_boxcount_diag(surf)
    g2x = g2_exact(surf, SHIFTS)
    return {'n': n, 'beta': float(res['beta']), 'nni': nni,
            'd2': d2r['d2'] if d2r else np.nan, 'g2': g2x}

# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    res = {}
    primes = primes_upto(N_POP)
    n_p = len(primes)
    rho = n_p / N_POP
    print(f'pi({N_POP}) = {n_p}, rho = {rho:.6f}', flush=True)

    # ── deterministic surfaces (one population, one window) ───────
    surf_rm = row_major_surface(primes, N_POP, W)
    surf_ul = ulam_even_surface(primes, W)
    surf_ct = cantor_surface(primes, N_POP, W)
    for nm, s in [('row_major', surf_rm), ('ulam', surf_ul), ('cantor', surf_ct)]:
        print(f'{nm}: {s.sum()} sites', flush=True)
    res['config'] = {'W': W, 'N_POP': N_POP, 'n_primes': n_p}

    # ── deterministic local stats with boundary treatments ────────
    res['deterministic'] = {}
    for nm, surf in [('row_major', surf_rm), ('ulam', surf_ul), ('cantor', surf_ct)]:
        pts = coords(surf)
        nn_e = nn_stats(pts)
        nn_t = nn_toroidal(pts, W)
        nn_i = nn_interior(pts, W)
        be = fit_brody(nn_e, n_bootstrap=3000, rng=np.random.default_rng(0))
        bt = fit_brody(nn_t, n_bootstrap=0, rng=np.random.default_rng(0))
        bi = fit_brody(nn_i, n_bootstrap=0, rng=np.random.default_rng(0))
        rho_loc = len(pts) / (W * W)
        nni_e = float(np.mean(nn_e) * 2 * np.sqrt(rho_loc))
        nni_t = float(np.mean(nn_t) * 2 * np.sqrt(rho_loc))
        d2d = d2_boxcount_diag(surf)
        gp = gp_correlation_dim(pts, W)
        g2x = g2_exact(surf, SHIFTS)
        g2ann = g2_annulus_toroidal(pts, W)
        res['deterministic'][nm] = {
            'n': int(len(pts)),
            'beta_euclid': float(be['beta']),
            'beta_euclid_ci': [float(be['ci_low']), float(be['ci_high'])],
            'beta_toroidal': float(bt['beta']),
            'beta_interior': float(bi['beta']),
            'nni_euclid': nni_e, 'nni_toroidal': nni_t,
            'd2_boxcount': d2d, 'd2_gp': gp,
            'g2_exact': g2x, 'g2_annulus_1_3': g2ann,
        }
        print(f"{nm}: beta_e={be['beta']:.3f}[{be['ci_low']:.3f},{be['ci_high']:.3f}] "
              f"beta_t={bt['beta']:.3f} beta_i={bi['beta']:.3f} "
              f"D2={d2d['d2']:.4f} D2_gp={gp['d2_gp']:.4f}", flush=True)

    # ── Ripley K: deterministic + 100 Bernoulli nulls ─────────────
    print('Ripley K ...', flush=True)
    rip = {}
    for nm, surf in [('row_major', surf_rm), ('ulam', surf_ul), ('cantor', surf_ct)]:
        pts = coords(surf)
        r, K, L, maxdev, L_nn = rip_k_border(pts, W, W, r_max=W / 3.0, n_bins=18)
        rip[nm] = {'r': r.tolist(), 'L': L.tolist(),
                   'maxdev': float(maxdev), 'L_at_nn': float(L_nn)}
    bern_L = []
    for _ in range(100):
        surf = bernoulli_surface(W, rho)
        pts = coords(surf)
        _, _, L, maxdev, L_nn = rip_k_border(pts, W, W, r_max=W / 3.0, n_bins=12)
        bern_L.append({'L': L.tolist(), 'maxdev': maxdev, 'L_at_nn': L_nn})
    rip['bernoulli_100'] = bern_L
    res['rip_k'] = rip

    # ── Null ensembles: 1000 realisations each ────────────────────
    nulls = {}
    for name, gen in [
        ('bernoulli', lambda rng: bernoulli_surface(W, rho)),
        ('sparse_integer', lambda rng: sparse_integer_surface(n_p, N_POP, W)),
        ('inhomogeneous', lambda rng: inhomogeneous_surface(W, n_p, rng)),
    ]:
        print(f'null {name}: 1000 realisations ...', flush=True)
        rows = []
        rng = np.random.default_rng(hash(name) % (2**32))
        for i in range(1000):
            surf = gen(rng)
            q = quick_stats(surf, W, rng)
            if q:
                rows.append(q)
        nulls[name] = rows
        b = np.array([r['beta'] for r in rows])
        print(f'   beta mean={b.mean():.4f} sd={b.std(ddof=1):.4f} '
              f'p2.5={np.percentile(b,2.5):.4f} p97.5={np.percentile(b,97.5):.4f}', flush=True)
    res['nulls'] = nulls

    # ── Density thinning with matched nulls ───────────────────────
    print('density thinning ...', flush=True)
    thin = {}
    for target_rho in [0.010, 0.020, 0.032, 0.050, 0.096]:
        keep_frac = target_rho / rho
        rng = np.random.default_rng(int(target_rho * 1000))
        prime_betas = []
        for _ in range(100):
            mask = rng.random(n_p) < keep_frac
            surf = row_major_surface(primes[mask], N_POP, W)
            nn = nn_stats(coords(surf))
            if nn is not None and len(nn) > 10:
                prime_betas.append(float(fit_brody(nn, n_bootstrap=0,
                                     rng=np.random.default_rng(0))['beta']))
        null_betas = []
        for _ in range(1000):
            surf = bernoulli_surface(W, target_rho)
            q = quick_stats(surf, W, rng)
            if q:
                null_betas.append(q['beta'])
        pb = np.array(prime_betas); nb = np.array(null_betas)
        thin[str(target_rho)] = {
            'prime_n': int(n_p * keep_frac),
            'prime_beta_mean': float(pb.mean()), 'prime_beta_sd': float(pb.std(ddof=1)),
            'prime_n_reps': len(pb),
            'null_beta_mean': float(nb.mean()), 'null_beta_sd': float(nb.std(ddof=1)),
            'null_n': len(nb)}
        print(f'   rho={target_rho}: prime {pb.mean():.4f}+-{pb.std(ddof=1):.4f} '
              f'null {nb.mean():.4f}+-{nb.std(ddof=1):.4f}', flush=True)
    res['thinning'] = thin

    # ── Finite-size beta(N) with matched nulls ────────────────────
    print('finite-size ...', flush=True)
    fs = {}
    for Nfs in [10**3, 3*10**3, 10**4, 3*10**4, 10**5, 3*10**5, 10**6]:
        Wfs = int(np.floor(np.sqrt(Nfs)))
        Npop = Wfs * Wfs
        pr = primes_upto(Npop)
        surf = row_major_surface(pr, Npop, Wfs)
        nn = nn_stats(coords(surf))
        boot = 2000 if len(pr) < 60000 else 200
        bfit = fit_brody(nn, n_bootstrap=boot, rng=np.random.default_rng(0))
        rho_fs = len(pr) / Npop
        n_null = 1000 if len(pr) < 60000 else 200
        nb = []
        rng = np.random.default_rng(7)
        for _ in range(n_null):
            surfn = bernoulli_surface(Wfs, rho_fs)
            q = quick_stats(surfn, Wfs, rng)
            if q:
                nb.append(q['beta'])
        nb = np.array(nb)
        fs[str(Npop)] = {'W': Wfs, 'n_primes': int(len(pr)),
                         'beta': float(bfit['beta']),
                         'beta_ci': [float(bfit['ci_low']), float(bfit['ci_high'])],
                         'null_mean': float(nb.mean()),
                         'null_sd': float(nb.std(ddof=1)), 'null_n': int(len(nb)),
                         'delta_beta': float(bfit['beta'] - nb.mean())}
        print(f"   W={Wfs} n={len(pr)} beta={bfit['beta']:.3f} "
              f"null={nb.mean():.3f}+-{nb.std(ddof=1):.3f} "
              f"delta={bfit['beta']-nb.mean():+.3f}", flush=True)
    res['finite_size'] = fs

    with open(f'{OUTDIR}/audit_v2_results.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f'DONE in {(time.time()-t0)/60:.1f} min', flush=True)

if __name__ == '__main__':
    main()

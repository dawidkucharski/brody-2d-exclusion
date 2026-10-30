#!/usr/bin/env python3
"""
Descriptor benchmark for 2D binary prime embeddings
===================================================
Recomputes local and multiscale descriptors for the three deterministic
prime embeddings (row-major, Ulam spiral, Cantor diagonal sweep) and for
the matched-density Bernoulli null and sparse-integer control.

Descriptors per configuration:
  - n_points        : number of occupied sites
  - brody_beta      : MLE of Brody beta on unit-mean NN spacings
  - nni             : Clark-Evans nearest-neighbour index
  - rip_k_rmean     : L(r) - r at r = mean NN distance (border corrected)
  - rip_k_maxdev    : max |L(r) - r| over r in [1, 30] px
  - g2_contact      : pair correlation value at r = 1.5 px (toroidal)
  - D2              : box-counting correlation dimension

Outputs:
  - descriptor_benchmark.json
  - fig_descriptor_benchmark.pdf
"""
import numpy as np, json, time
from scipy import optimize, spatial
from scipy.special import gamma as gamma_func
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
OUTDIR = '/Users/dawid/Projects/interfero-Riemann/elsarticle'
N = 100_000          # prime limit
W = int(np.sqrt(N))  # 316 -> grid covers values 1..99856
RHO = None           # set after prime count

# ── Brody fitting (verified MLE) ──────────────────────────────────
def brody_nll(beta, s):
    if beta <= -1:
        return 1e10
    a = gamma_func((beta + 2) / (beta + 1)) ** (beta + 1)
    n = len(s)
    ll = (n * np.log(beta + 1) + n * np.log(a)
          + beta * np.sum(np.log(s)) - a * np.sum(s ** (beta + 1)))
    return -ll

def fit_brody(nn, n_bootstrap=200):
    nn = nn[nn > 1e-10]
    if len(nn) < 10:
        return {'beta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': len(nn)}
    s = nn / np.mean(nn)
    res = optimize.minimize_scalar(
        lambda b: brody_nll(b, s) if b >= 0 else 1e10,
        bounds=(0, 15), method='bounded')
    beta = max(0.0, res.x)
    bs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(s), len(s), replace=True)
        try:
            r = optimize.minimize_scalar(
                lambda b: brody_nll(b, s[idx]) if b >= 0 else 1e10,
                bounds=(0, 15), method='bounded')
            bs.append(max(0.0, r.x))
        except Exception:
            pass
    bs = np.array(bs)
    return {'beta': beta,
            'ci_low': float(np.percentile(bs, 2.5)),
            'ci_high': float(np.percentile(bs, 97.5)),
            'n': len(nn)}

# ── Primes ────────────────────────────────────────────────────────
def primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = False
    return np.where(sieve)[0]

# ── Embeddings ────────────────────────────────────────────────────
def row_major_surface(values, N, W):
    surf = np.zeros((W, W), dtype=bool)
    for v in values:
        if 1 <= v <= W * W:
            idx = v - 1
            surf[idx // W, idx % W] = True
    return surf

def ulam_surface(values, N):
    side = int(np.ceil(np.sqrt(N)))
    if side % 2 == 0:
        side += 1
    surf = np.zeros((side, side), dtype=bool)
    pos = {}
    cx = cy = side // 2
    pos[1] = (cx, cy)
    x, y = cx, cy
    step = 1
    n = 1
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # E, N, W, S
    while n < min(N, side * side):
        for di, (dx, dy) in enumerate(dirs):
            for _ in range(step):
                x += dx; y += dy
                n += 1
                if n > N or n >= side * side:
                    break
                if 0 <= x < side and 0 <= y < side:
                    pos[n] = (x, y)
            if di % 2 == 1:
                step += 1
            if n >= N or n >= side * side:
                break
    for v in values:
        if v in pos:
            x, y = pos[v]
            surf[y, x] = True
    return surf

def cantor_surface(values, N, W):
    """Diagonal sweep: cell (i,j) receives value n if it is the n-th
    cell visited along anti-diagonals i+j = 0,1,2,..."""
    surf = np.zeros((W, W), dtype=bool)
    N_use = W * W
    pos = {}
    n = 1
    for s in range(2 * W - 1):
        j0 = max(0, s - (W - 1))
        j1 = min(s, W - 1)
        for j in range(j0, j1 + 1):
            i = s - j
            if 0 <= i < W:
                pos[n] = (i, j)
                n += 1
                if n > N_use:
                    break
        if n > N_use:
            break
    for v in values:
        if v in pos:
            i, j = pos[v]
            surf[j, i] = True
    return surf

def bernoulli_surface(W, rho):
    return np.random.random((W, W)) < rho

def sparse_integer_surface(values_count, N, W):
    """pi(N) random distinct integers in [1, W*W], row-major embedded."""
    vals = np.random.choice(np.arange(1, W * W + 1), size=min(values_count, W * W),
                            replace=False)
    surf = np.zeros((W, W), dtype=bool)
    for v in vals:
        idx = v - 1
        surf[idx // W, idx % W] = True
    return surf

# ── Descriptors ───────────────────────────────────────────────────
def coords(surf):
    rows, cols = np.where(surf)
    return np.column_stack([cols, rows]).astype(float)

def nn_stats(pts):
    if len(pts) < 2:
        return None
    tree = spatial.KDTree(pts)
    d, _ = tree.query(pts, k=2)
    return d[:, 1]

def clark_evans(pts, W, H=None):
    """NNI = mean NN / expected mean NN under CSR in the window."""
    if H is None:
        H = W
    nn = nn_stats(pts)
    n = len(pts)
    area = W * H
    rho = n / area
    expected = 0.5 / np.sqrt(rho)
    return float(np.mean(nn) / expected)

def rip_k_border(pts, W, H, r_max, n_bins, n_ang=64):
    """Ripley K with isotropic border correction (circumference fraction).
    Returns r, K, LminusR, maxdev, L_at_rmean."""
    n = len(pts)
    area = W * H
    tree = spatial.KDTree(pts)
    # radius grid (log-spaced for resolution at small r)
    r_edges = np.geomspace(1.0, r_max, n_bins + 1)
    r_cent = np.sqrt(r_edges[:-1] * r_edges[1:])
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    K = np.zeros(n_bins)
    for r_i, r in enumerate(r_cent):
        # border weight per point: fraction of circumference inside window
        xs = pts[:, 0][:, None] + r * cos_a[None, :]
        ys = pts[:, 1][:, None] + r * sin_a[None, :]
        inside = (xs >= 0) & (xs <= W) & (ys >= 0) & (ys <= H)
        w = inside.mean(axis=1)
        w = np.where(w > 0, w, np.inf)
        pairs = tree.query_pairs(r, output_type='ndarray')
        if len(pairs) > 0:
            counts = np.bincount(pairs.ravel(), minlength=n)
            tot = float(np.sum(counts / w))
        else:
            tot = 0.0
        K[r_i] = area * tot / (n * n)
    L = np.sqrt(K / np.pi) - r_cent
    maxdev = float(np.max(np.abs(L)))
    r_mean = float(np.mean(nn_stats(pts)))
    # L at r = mean NN (interpolate)
    L_at_rmean = float(np.interp(r_mean, r_cent, L))
    return r_cent, K, L, maxdev, L_at_rmean

def g2_toroidal(pts, W, H, r_min=1.0, r_max=3.0):
    """Pair correlation via toroidal pair counting in annulus [r_min, r_max]."""
    n = len(pts)
    area = W * H
    rho = n / area
    tree = spatial.KDTree(pts)
    idx = np.arange(n)
    np.random.shuffle(idx)
    sample = idx[: min(n, 3000)]
    pair_count = 0
    area_ann = np.pi * (r_max ** 2 - r_min ** 2)
    for i in sample:
        p = pts[i]
        # periodic images of all points within r_max of p (torus)
        # candidates: points within r_max in any of 9 images
        cand = []
        for dx9 in (-W, 0, W):
            for dy9 in (-H, 0, H):
                qs = tree.query_ball_point(p + [dx9, dy9], r_max)
                for q in qs:
                    cand.append((q, dx9, dy9))
        for q, dx9, dy9 in cand:
            if q == i and dx9 == 0 and dy9 == 0:
                continue
            dd = np.hypot(pts[q, 0] + dx9 - p[0], pts[q, 1] + dy9 - p[1])
            if r_min <= dd < r_max:
                pair_count += 1
    g2 = (pair_count / len(sample)) / (rho * area_ann)
    return float(g2)

def d2_boxcount(surf):
    """Correlation dimension from box-counting partition function."""
    H, W_ = surf.shape
    Z = surf.astype(float)
    M = Z.sum()
    if M == 0:
        return np.nan
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
    if len(eps_list) < 3:
        return np.nan
    # regression slope
    A = np.vstack([eps_list, np.ones(len(eps_list))]).T
    slope, _ = np.linalg.lstsq(A, z2_list, rcond=None)[0]
    return float(slope)

def full_descriptors(surf, W, H=None, n_bootstrap=200):
    if H is None:
        H = W
    pts = coords(surf)
    out = {'n_points': int(len(pts))}
    if len(pts) < 10:
        for k in ('brody_beta', 'nni', 'rip_k_maxdev', 'rip_k_rmean', 'g2_contact', 'D2'):
            out[k] = np.nan
        return out
    nn = nn_stats(pts)
    fit = fit_brody(nn, n_bootstrap=n_bootstrap)
    out['brody_beta'] = float(fit['beta'])
    out['beta_ci'] = [fit['ci_low'], fit['ci_high']]
    out['nni'] = clark_evans(pts, W, H)
    r, K, L, maxdev, L_at_rmean = rip_k_border(pts, W, H, r_max=min(W, H) / 3.0, n_bins=18)
    out['rip_k_maxdev'] = maxdev
    out['rip_k_rmean'] = L_at_rmean
    out['g2_contact'] = g2_toroidal(pts, W, H)
    out['D2'] = d2_boxcount(surf)
    return out

# ── Main ──────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print('sieve...')
    primes = primes_upto(N)
    n_p = len(primes)
    rho = n_p / N
    print(f'pi({N}) = {n_p}, rho = {rho:.4f}')

    print('building embeddings...')
    surf_rm = row_major_surface(primes, N, W)
    surf_ul = ulam_surface(primes, N)
    surf_ct = cantor_surface(primes, N, W)
    print('row-major:', surf_rm.sum(), ' ulam:', surf_ul.sum(), ' cantor:', surf_ct.sum())

    results = {}
    results['row_major'] = full_descriptors(surf_rm, W)
    print('row_major done', {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in results['row_major'].items() if k != 'beta_ci'})
    results['ulam'] = full_descriptors(surf_ul, surf_ul.shape[1], surf_ul.shape[0])
    print('ulam done', {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in results['ulam'].items() if k != 'beta_ci'})
    results['cantor'] = full_descriptors(surf_ct, W)
    print('cantor done', {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in results['cantor'].items() if k != 'beta_ci'})

    # Bernoulli null: 20 realisations at rho = 9580/99856 (grid-matched)
    rho_grid = n_p / (W * W)
    print('Bernoulli null (20 reps)...')
    bern = []
    for rep in range(20):
        surf = bernoulli_surface(W, rho_grid)
        bern.append(full_descriptors(surf, W, n_bootstrap=100))
    results['bernoulli'] = bern

    # Sparse-integer control: 20 reps of pi(W^2) distinct random ints
    print('Sparse-integer control (20 reps)...')
    n_grid = sum(1 for p in primes if p <= W * W)
    sparse = []
    for rep in range(20):
        surf = sparse_integer_surface(n_grid, W * W, W)
        sparse.append(full_descriptors(surf, W, n_bootstrap=100))
    results['sparse_integer'] = sparse

    # aggregates
    def aggregate(rows):
        agg = {}
        for key in ('brody_beta', 'nni', 'rip_k_maxdev', 'rip_k_rmean', 'g2_contact', 'D2'):
            vals = np.array([r[key] for r in rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[key + '_mean'] = float(np.mean(vals))
            agg[key + '_std'] = float(np.std(vals))
        return agg
    results['bernoulli_agg'] = aggregate(bern)
    results['sparse_agg'] = aggregate(sparse)

    with open(f'{OUTDIR}/descriptor_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'saved JSON ({time.time()-t0:.0f}s)')

    make_figure(results, surf_rm, surf_ul, surf_ct)

    # summary print
    print('\n=== SUMMARY ===')
    keys = ['row_major', 'ulam', 'cantor']
    metrics = ['brody_beta', 'nni', 'rip_k_maxdev', 'rip_k_rmean', 'g2_contact', 'D2']
    print(f"{'metric':<16}{'row-major':>12}{'ulam':>12}{'cantor':>12}{'bernoulli':>12}{'sparse':>12}")
    for m in metrics:
        row = [f"{results[k][m]:.3f}" for k in keys]
        row.append(f"{results['bernoulli_agg'][m+'_mean']:.3f}+/-{results['bernoulli_agg'][m+'_std']:.3f}")
        row.append(f"{results['sparse_agg'][m+'_mean']:.3f}+/-{results['sparse_agg'][m+'_std']:.3f}")
        print(f"{m:<16}" + ''.join(f"{v:>12}" for v in row))
    print(f"\nManuscript quotes: row-major beta=2.15, ulam=1.87, cantor=1.40, bernoulli=1.47")


def make_figure(results, surf_rm, surf_ul, surf_ct):
    """Build the 3-panel descriptor benchmark figure; save PDF + PNG."""
    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    names = ['Row-major', 'Ulam', 'Cantor']
    keys = ['row_major', 'ulam', 'cantor']

    def zscore(val, mean, std):
        return (val - mean) / std if std > 0 else np.nan

    # panel 1: Brody beta with Bernoulli 95% band
    ax = axes[0]
    bs = [results[k]['brody_beta'] for k in keys]
    bb, bsig = results['bernoulli_agg']['brody_beta_mean'], results['bernoulli_agg']['brody_beta_std']
    ax.bar(names, bs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.axhspan(bb - 1.96 * bsig, bb + 1.96 * bsig, color='gray', alpha=0.3)
    ax.axhline(bb, color='gray', ls='--', lw=1)
    ax.set_ylabel('Brody $\\beta$')
    ax.set_title('Local spacing fit')
    for i, b in enumerate(bs):
        ax.text(i, b + 0.05, f'{b:.2f}', ha='center', fontsize=9)

    # panel 2: z-scores vs Bernoulli for all descriptors
    ax = axes[1]
    metrics = ['brody_beta', 'nni', 'rip_k_maxdev', 'rip_k_rmean', 'g2_contact', 'D2']
    labels = ['Brody $\\beta$', 'NNI', '$\\max|L(r)-r|$', '$L(r)-r$ at $r$=NN', '$g_2$ contact', '$D_2$']
    x = np.arange(len(metrics))
    width = 0.25
    for off, (nm, key) in zip([-width, 0, width], zip(names, keys)):
        zs = []
        for m in metrics:
            v = results[key][m]
            mu = results['bernoulli_agg'][m + '_mean']
            sd = results['bernoulli_agg'][m + '_std']
            zs.append(zscore(v, mu, sd))
        ax.bar(x + off, zs, width, label=nm, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(1.96, color='gray', ls=':', lw=0.8)
    ax.axhline(-1.96, color='gray', ls=':', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('z-score vs Bernoulli null')
    ax.set_title('Descriptor separation from null')
    ax.legend(fontsize=8)

    # panel 3: L(r)-r curves
    ax = axes[2]
    for nm, key in zip(names, keys):
        surf = {'row_major': surf_rm, 'ulam': surf_ul, 'cantor': surf_ct}[key]
        Hh, Ww = surf.shape
        pts = coords(surf)
        r, K, L, _, _ = rip_k_border(pts, Ww, Hh, r_max=min(Ww, Hh) / 3.0, n_bins=18)
        ax.plot(r, L, lw=1.8, label=nm)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('r (pixels)')
    ax.set_ylabel('$L(r) - r$')
    ax.set_title('Ripley K summary')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig_descriptor_benchmark.pdf', dpi=200, facecolor='white')
    plt.savefig(f'{OUTDIR}/fig_descriptor_benchmark.png', dpi=150, facecolor='white')
    plt.close(fig)
    print('saved figure (pdf + png)')

if __name__ == '__main__':
    import sys
    if '--figure-only' in sys.argv:
        results = json.load(open(f'{OUTDIR}/descriptor_benchmark.json'))
        primes = primes_upto(N)
        surf_rm = row_major_surface(primes, N, W)
        surf_ul = ulam_surface(primes, N)
        surf_ct = cantor_surface(primes, N, W)
        make_figure(results, surf_rm, surf_ul, surf_ct)
    else:
        main()

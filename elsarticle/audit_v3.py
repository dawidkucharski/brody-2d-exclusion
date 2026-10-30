#!/usr/bin/env python3
"""audit_v3.py — compute everything needed to fix review issues M1-M3, m1-m8."""
import json
import numpy as np
from scipy import spatial

W, N_POP, X = 316, 99856, 99856
np.random.seed(2026)

from audit_v2 import (primes_upto, row_major_surface, ulam_even_surface,
                      cantor_surface, coords, bernoulli_surface,
                      inhomogeneous_surface, fit_brody, nn_stats, nn_interior)
from audit_v2b import CLASSES

primes = primes_upto(N_POP)
surf = {'RM': row_major_surface(primes, N_POP, W),
        'UL': ulam_even_surface(primes, W),
        'CT': cantor_surface(primes, N_POP, W)}

# ── exact pair counts pi_g(x) for all gaps up to 3000 via FFT ────
print('sieve + FFT pair counts ...')
sieve = np.ones(X + 3000 + 2, dtype=bool); sieve[:2] = False
for i in range(2, int((X + 3000) ** 0.5) + 1):
    if sieve[i]:
        sieve[i*i::i] = False
pi = int(sieve[:X + 1].sum())
nfft = 1 << (int(np.log2(2 * (X + 3000))) + 1)
F = np.fft.rfft(sieve.astype(float), n=nfft)
auto = np.fft.irfft(F * np.conj(F), n=nfft)
pair_counts = np.round(auto).astype(int)   # ordered pair count at each gap
print(f'pi(x)={pi}, pi2={pair_counts[2]}, pi4={pair_counts[4]}')

# ── value grids (embedding of each integer) ──────────────────────
def value_grid_rm():
    return np.arange(1, N_POP + 1).reshape(W, W)

def value_grid_ulam():
    grid = np.zeros((W, W), dtype=int)
    x, y = W // 2 - 1, W // 2
    grid[y, x] = 1
    n = 1; step = 1
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    while n < N_POP:
        for di, (dx, dy) in enumerate(dirs):
            for _ in range(step):
                x += dx; y += dy; n += 1
                if n > N_POP: break
                grid[y, x] = n
            if di % 2 == 1: step += 1
            if n >= N_POP: break
    return grid

def value_grid_cantor():
    grid = np.zeros((W, W), dtype=int)
    n = 1
    for s in range(2 * W - 1):
        j0 = max(0, s - (W - 1)); j1 = min(s, W - 1)
        for j in range(j0, j1 + 1):
            i = s - j
            if 0 <= i < W:
                grid[j, i] = n; n += 1
                if n > N_POP: break
        if n > N_POP: break
    return grid

V = {'RM': value_grid_rm(), 'UL': value_grid_ulam(), 'CT': value_grid_cantor()}
print('value grids built')

# ── standard g2 + gap dictionaries + HL predictions ──────────────
def standard_g2_and_gaps(surf, V, name):
    N = int(surf.sum()); rho = surf.mean()
    out = {}
    for dname, spec in CLASSES.items():
        tot = 0.0
        gap_hist = {}
        m = spec['m']
        for dx, dy in spec['reps']:
            Ss = np.roll(np.roll(surf, dx, axis=1), dy, axis=0)
            c = int((surf & Ss).sum())
            tot += c
            # gap multiset: iterate over occupied pairs only (fast enough)
            occ = np.argwhere(surf)
            if name in ('UL', 'CT'):
                for (i, j) in occ:
                    v1 = V[i, j]
                    v2 = V[(i + dy) % W, (j + dx) % W]
                    if v1 == 0 or v2 == 0:
                        continue
                    g = abs(int(v1) - int(v2))
                    gap_hist[g] = gap_hist.get(g, 0) + 1
        g2 = tot / (N * rho * m / 2.0)
        out[dname] = {'g2': g2, 'gap_hist': gap_hist}
    return out

print('standard g2 and gap dictionaries ...')
G = {}
for nm in surf:
    G[nm] = standard_g2_and_gaps(surf[nm], V[nm], nm)
    print(nm, {k: round(v['g2'], 3) for k, v in G[nm].items()})

# ── HL predictions per distance class (RM: analytic gaps) ────────
print('\nHL gap predictions vs measured (row-major):')
HL = {}
for dname, spec in CLASSES.items():
    pred = 0.0
    for dx, dy in spec['reps']:
        g = abs(dy * W + dx)
        if g == 0:
            continue
        # number of cell pairs at this offset (non-wrapping handled via count)
        pred += pair_counts[g]
    meas = int(np.round(G['RM'][dname]['g2'] * (int(surf['RM'].sum()) * surf['RM'].mean() * spec['m'] / 2.0)))
    HL[dname] = {'pred': int(pred), 'meas': meas}
    print(f"  {dname}: measured={meas}  HL_pred={int(pred)}  ratio={meas/pred:.3f}")

# ── HL predictions for UL/CT from gap histograms ─────────────────
print('\nHL predictions from gap histograms (Ulam, Cantor):')
for nm in ['UL', 'CT']:
    for dname, spec in CLASSES.items():
        hist = G[nm][dname]['gap_hist']
        pred = 0.0
        for g, mcnt in hist.items():
            if 0 < g <= X:
                pred += mcnt * pair_counts[g] / (X - g)
        meas = int(np.round(G[nm][dname]['g2'] * (int(surf[nm].sum()) * surf[nm].mean() * spec['m'] / 2.0)))
        if hist:
            print(f"  {nm} {dname}: measured={meas}  HL_pred={pred:.0f}  ratio={meas/pred:.3f}  gaps={sorted(hist)[:6]}...")
print()

# ── null mean/sd of standard g2 (verify =1 everywhere) ───────────
print('null g2 (standard) ...')
sim = {k: [] for k in CLASSES}
for i in range(2000):
    s = bernoulli_surface(W, pi / N_POP)
    N = int(s.sum()); rho = s.mean()
    for dname, spec in CLASSES.items():
        tot = 0.0
        for dx, dy in spec['reps']:
            Ss = np.roll(np.roll(s, dx, axis=1), dy, axis=0)
            tot += int((s & Ss).sum())
        sim[dname].append(tot / (N * rho * spec['m'] / 2.0))
for dname in CLASSES:
    v = np.array(sim[dname])
    print(f"  {dname}: mean={v.mean():.4f} sd={v.std(ddof=1):.4f}")
print()

# ── direction-resolved r=2/r=4 for RM + null sd ──────────────────
print('direction-resolved (RM) + null sd ...')
for lab, dx, dy in [('h2', 0, 2), ('v2', 2, 0), ('h4', 0, 4), ('v4', 4, 0)]:
    s = surf['RM']
    Ss = np.roll(np.roll(s, dy, axis=0), dx, axis=1)
    c = int((s & Ss).sum())
    print(f"  {lab}: g2_dir={c/(s.sum()*s.mean()):.4f}  (pairs={c}, expected={s.sum()*s.mean():.0f})")
simd = {lab: [] for lab in ['h2', 'v2', 'h4', 'v4']}
for i in range(2000):
    s = bernoulli_surface(W, pi / N_POP)
    for lab, dx, dy in [('h2', 0, 2), ('v2', 2, 0), ('h4', 0, 4), ('v4', 4, 0)]:
        Ss = np.roll(np.roll(s, dy, axis=0), dx, axis=1)
        simd[lab].append(int((s & Ss).sum()) / (s.sum() * s.mean()))
for lab in simd:
    v = np.array(simd[lab])
    print(f"  null {lab}: mean={v.mean():.4f} sd={v.std(ddof=1):.4f}")

# ── interior-reference Bernoulli null ────────────────────────────
print('\ninterior null beta (1000 reps) ...')
ib = []
for i in range(1000):
    s = bernoulli_surface(W, pi / N_POP)
    nn_i = nn_interior(coords(s), W)
    ib.append(float(fit_brody(nn_i, n_bootstrap=0, rng=np.random.default_rng(0))['beta']))
ib = np.array(ib)
print(f"  mean={ib.mean():.4f} sd={ib.std(ddof=1):.4f}")

# ── null exceedance ──────────────────────────────────────────────
print('\nnull exceedance ...')
import json as _json
aud = _json.load(open('audit_v2_results.json'))
bn = np.array([r['beta'] for r in aud['nulls']['bernoulli'] if r])
print(f"  Bernoulli null beta: min={bn.min():.4f} max={bn.max():.4f} "
      f"p99.9={np.percentile(bn,99.9):.4f}")

# ── ablation: remove twin/cousin primes, refit beta ──────────────
print('\nablation ...')
def ablation_fit(kind, n_reps=1):
    pset = set(primes.tolist())
    removed = 0
    for p in primes:
        if kind in ('twins', 'both') and (p + 2) in pset and p + 2 <= X:
            pset.discard(p + 2); removed += 1
        if kind in ('cousins', 'both') and (p + 4) in pset and p + 4 <= X:
            pset.discard(p + 4); removed += 1
    keep = np.array(sorted(pset))
    surf_k = row_major_surface(keep, N_POP, W)
    nn = nn_stats(coords(surf_k))
    b = float(fit_brody(nn, n_bootstrap=0, rng=np.random.default_rng(0))['beta'])
    rho_k = len(keep) / N_POP
    nb = []
    for _ in range(500):
        s = bernoulli_surface(W, rho_k)
        nb.append(float(fit_brody(nn_stats(coords(s)), n_bootstrap=0,
                                  rng=np.random.default_rng(0))['beta']))
    nb = np.array(nb)
    return len(keep), b, nb.mean(), nb.std(ddof=1)
for kind in ['twins', 'cousins', 'both']:
    n, b, mu, sd = ablation_fit(kind)
    print(f"  {kind}: n={n} beta={b:.4f} null={mu:.4f}+-{sd:.4f} delta={b-mu:+.4f}")

# ── NNI bootstrap CI ─────────────────────────────────────────────
print('\nNNI bootstrap (2000) ...')
for nm in ['RM', 'UL', 'CT']:
    pts = coords(surf[nm])
    nn = nn_stats(pts)
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(2000):
        idx = rng.choice(len(nn), len(nn), replace=True)
        vals.append(np.mean(nn[idx]) * 2 * np.sqrt(len(pts) / N_POP))
    vals = np.array(vals)
    print(f"  {nm}: NNI={np.mean(vals):.4f} [{np.percentile(vals,2.5):.4f},{np.percentile(vals,97.5):.4f}]")

# ── low-density null beta trend ──────────────────────────────────
print('\nlow-density null beta ...')
for rho_t in [0.002, 0.005, 0.01]:
    vals = []
    for _ in range(500):
        s = bernoulli_surface(W, rho_t)
        nn = nn_stats(coords(s))
        vals.append(float(fit_brody(nn, n_bootstrap=0, rng=np.random.default_rng(0))['beta']))
    vals = np.array(vals)
    print(f"  rho={rho_t}: null beta={vals.mean():.4f}+-{vals.std(ddof=1):.4f}")

out = {'standard_g2': {nm: {k: v['g2'] for k, v in G[nm].items()} for nm in G},
       'g2_null_sd': {k: float(np.array(sim[k]).std(ddof=1)) for k in sim},
       'g2_null_mean': {k: float(np.array(sim[k]).mean()) for k in sim},
       'hl': HL,
       'dir_g2': {lab: None for lab in ['h2', 'v2', 'h4', 'v4']},
       'interior_null': {'mean': float(ib.mean()), 'sd': float(ib.std(ddof=1))},
       'null_max_beta': float(bn.max()), 'null_p999': float(np.percentile(bn, 99.9))}
json.dump(out, open('audit_v3_results.json', 'w'), indent=1, default=float)
print('\nDONE')

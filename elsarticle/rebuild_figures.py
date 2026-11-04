#!/usr/bin/env python3
"""Superseded figure builder retained for historical reference."""
if __name__ == '__main__':
    raise SystemExit('Superseded: run build_prime_figures.py for the current manuscript.')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial

OUT = '/Users/dawid/Projects/interfero-Riemann/elsarticle'
res = json.load(open(f'{OUT}/audit_v2_results.json'))
resb = json.load(open(f'{OUT}/audit_v2b_results.json'))

W, N_POP = 316, 99856
np.random.seed(2026)

from audit_v2 import (primes_upto, row_major_surface, ulam_even_surface,
                      cantor_surface, coords, bernoulli_surface)
from audit_v2b import g2_exact_full, CLASSES
from benchmark_descriptors import fit_brody, nn_stats, rip_k_border

primes = primes_upto(N_POP)
surf = {'Row-major': row_major_surface(primes, N_POP, W),
        'Ulam': ulam_even_surface(primes, W),
        'Cantor': cantor_surface(primes, N_POP, W)}
names = list(surf.keys())

# deterministic audited values
det = res['deterministic']
beta_e = {n: det[n.lower().replace('-', '_')]['beta_euclid'] for n in names}
beta_ci = {n: det[n.lower().replace('-', '_')]['beta_euclid_ci'] for n in names}
beta_t = {n: resb['toroidal'][n.lower().replace('-', '_')]['beta'] for n in names}
beta_t_ci = {n: resb['toroidal'][n.lower().replace('-', '_')]['ci'] for n in names}
beta_i = {n: det[n.lower().replace('-', '_')]['beta_interior'] for n in names}
# interior CIs
beta_i_ci = {}
for n, s in surf.items():
    pts = coords(s)
    from audit_v2 import nn_interior
    nn_i = nn_interior(pts, W)
    f = fit_brody(nn_i, n_bootstrap=2000, rng=np.random.default_rng(3))
    beta_i_ci[n] = [f['ci_low'], f['ci_high']]

# null distributions from audit_v2 (1000 reps)
null_beta = {'Bernoulli': np.array([r['beta'] for r in res['nulls']['bernoulli'] if r]),
             'Sparse-int.': np.array([r['beta'] for r in res['nulls']['sparse_integer'] if r]),
             'Inhomog. PNT': np.array([r['beta'] for r in res['nulls']['inhomogeneous'] if r])}
null_nni = np.array([r['nni'] for r in res['nulls']['bernoulli'] if r])

# Ripley: deterministic (18 bins) + 100 nulls (18 bins, audit_v2b)
rip = res['rip_k']
rip_null = resb['rip_k_bernoulli_18bins']

# D2 corrected + GP
d2c = {n: resb['boxcount_corr'][n.lower().replace('-', '_')] for n in names}
gp = {n: det[n.lower().replace('-', '_')]['d2_gp'] for n in names}
null_d2 = resb['nulls']['bernoulli']

# g2 exact: deterministic (recomputed with corrected normalisation) + null sd
g2d = {n: g2_exact_full(surf[n]) for n in names}
g2_keys = ['r1', 'rsqrt2', 'r2', 'rsqrt5', 'r4']
g2_labels = {'r1': r'$r=1$', 'rsqrt2': r'$r=\sqrt{2}$', 'r2': r'$r=2$',
             'rsqrt5': r'$r=\sqrt{5}$', 'r4': r'$r=4$'}
sim_g2 = {k: [] for k in g2_keys}
sim_dir = {'h2': [], 'v2': [], 'h4': [], 'v4': []}
sim_ann = []
for i in range(2000):
    s = bernoulli_surface(W, 9580 / N_POP)
    g = g2_exact_full(s)
    for k in g2_keys:
        sim_g2[k].append(g[k])
    # direction-resolved r=2, r=4
    for lab, dx, dy in [('h2', 2, 0), ('v2', 0, 2), ('h4', 4, 0), ('v4', 0, 4)]:
        Ss = np.roll(np.roll(s, dx, axis=1), dy, axis=0)
        tot = int((s & Ss).sum())
        sim_dir[lab].append(tot / (s.sum() * s.mean()))
    if i < 300:
        pts = coords(s)
        from audit_v2 import g2_annulus_toroidal
        sim_ann.append(g2_annulus_toroidal(pts, W, seed=i))
sim_g2 = {k: np.array(v) for k, v in sim_g2.items()}
sim_dir = {k: np.array(v) for k, v in sim_dir.items()}
sim_ann = np.array(sim_ann)

# direction-resolved deterministic
dir_g2 = {}
for n, s in surf.items():
    d = {}
    for lab, dx, dy in [('h2', 2, 0), ('v2', 0, 2), ('h4', 4, 0), ('v4', 0, 4)]:
        Ss = np.roll(np.roll(s, dx, axis=1), dy, axis=0)
        d[lab] = int((s & Ss).sum()) / (s.sum() * s.mean())
    dir_g2[n] = d

# thinning
thin = res['thinning']
rhos = [0.01, 0.02, 0.032, 0.05, 0.096]
p_m = np.array([thin[str(r)]['prime_beta_mean'] for r in rhos])
p_s = np.array([thin[str(r)]['prime_beta_sd'] for r in rhos])
n_m = np.array([thin[str(r)]['null_beta_mean'] for r in rhos])
n_s = np.array([thin[str(r)]['null_beta_sd'] for r in rhos])

# finite size
fs = res['finite_size']
fsN = [961, 2916, 10000, 29929, 99856, 299209, 1000000]
fbeta = np.array([fs[str(N)]['beta'] for N in fsN])
fci_l = np.array([fs[str(N)]['beta_ci'][0] for N in fsN])
fci_h = np.array([fs[str(N)]['beta_ci'][1] for N in fsN])
fnull = np.array([fs[str(N)]['null_mean'] for N in fsN])
fnull_sd = np.array([fs[str(N)]['null_sd'] for N in fsN])

colors = {'Row-major': '#1f77b4', 'Ulam': '#ff7f0e', 'Cantor': '#2ca02c'}

# ═══ FIG 1: embedding robustness ══════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
xs = np.arange(3)
b = [beta_e[n] for n in names]
err = [[beta_e[n] - beta_ci[n][0] for n in names],
       [beta_ci[n][1] - beta_e[n] for n in names]]
ax.bar(xs, b, yerr=err, color=[colors[n] for n in names], capsize=5, alpha=0.85)
mu, sd = null_beta['Bernoulli'].mean(), null_beta['Bernoulli'].std(ddof=1)
ax.axhspan(mu - 1.96 * sd, mu + 1.96 * sd, color='gray', alpha=0.3, zorder=0)
ax.axhline(mu, color='k', ls='--', lw=1)
ax.axhline(1.0, color='red', ls=':', lw=1.2)
ax.text(0.25, 1.02, r'continuous CSR (Rayleigh $=\,$Brody $\beta=1$)', color='red', fontsize=8, transform=ax.transData)
ax.set_xticks(xs); ax.set_xticklabels(names)
ax.set_ylabel(r'Brody $\beta$ (bootstrap 95\% CI)')
ax.set_title('(a) Embedding dependence')
for i, (n, v) in enumerate(zip(names, b)):
    ax.text(i, v + 0.06, f'{v:.2f}', ha='center', fontsize=9)
ax = axes[1]
ax.hist(null_beta['Bernoulli'], bins=40, color='gray', alpha=0.6, density=True)
ax.axvline(beta_e['Row-major'], color=colors['Row-major'], lw=2, label=r'row-major $\beta=2.15$')
ax.axvline(beta_e['Ulam'], color=colors['Ulam'], lw=2, label=r'Ulam $\beta=1.86$')
ax.axvline(beta_e['Cantor'], color=colors['Cantor'], lw=2, label=r'Cantor $\beta=1.41$')
ax.set_xlabel(r'fitted $\beta$'); ax.set_ylabel('density')
ax.set_title(r'(b) Null distribution (lattice Bernoulli, 1000 realisations, mean $1.47\pm0.02$)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_embedding_robustness.pdf', dpi=200, facecolor='white')
plt.close()

# ═══ FIG 2: boundary sensitivity ══════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
treats = ['Euclidean (open)', 'Toroidal (periodic)', 'Interior reference']
x = np.arange(3); wdt = 0.26
for j, n in enumerate(names):
    vals = [beta_e[n], beta_t[n], beta_i[n]]
    cis = [beta_ci[n], beta_t_ci[n], beta_i_ci[n]]
    lo = [v - c[0] for v, c in zip(vals, cis)]
    hi = [c[1] - v for v, c in zip(vals, cis)]
    ax.bar(x + (j - 1) * wdt, vals, wdt, yerr=[lo, hi], capsize=3,
           color=colors[n], alpha=0.85, label=n)
mu, sd = null_beta['Bernoulli'].mean(), null_beta['Bernoulli'].std(ddof=1)
ax.axhspan(mu - 1.96 * sd, mu + 1.96 * sd, color='gray', alpha=0.25)
ax.axhline(mu, color='k', ls='--', lw=1)
ax.axhline(1.0, color='red', ls=':', lw=1)
ax.set_xticks(x); ax.set_xticklabels(treats)
ax.set_ylabel(r'Brody $\beta$')
ax.set_title('Boundary-condition sensitivity of the fitted Brody parameter')
ax.legend(fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_boundary_sensitivity.pdf', dpi=200, facecolor='white')
plt.close()

# ═══ FIG 3: density thinning ══════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar(rhos, p_m, yerr=1.96 * p_s, marker='o', color=colors['Row-major'],
            capsize=4, label='thinned row-major primes (100 reps)')
ax.fill_between(rhos, n_m - 1.96 * n_s, n_m + 1.96 * n_s, color='gray', alpha=0.3)
ax.plot(rhos, n_m, 'k--', label='density-matched Bernoulli (1000 reps)')
ax.axhline(1.0, color='red', ls=':', lw=1)
ax.set_xscale('log'); ax.set_xlabel(r'fill fraction $\rho$'); ax.set_ylabel(r'Brody $\beta$')
ax.set_title('Density dependence of the fitted Brody parameter')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_density_thinning_control.pdf', dpi=200, facecolor='white')
plt.close()

# ═══ FIG 4: exact-distance pair correlation ═══════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
x = np.arange(len(g2_keys)); wdt = 0.25
for j, n in enumerate(names):
    vals = [g2d[n][k] for k in g2_keys]
    ax.bar(x + (j - 1) * wdt, vals, wdt, color=colors[n], alpha=0.85, label=n)
ax.axhline(1.0, color='k', lw=1)
for k in g2_keys:
    sd = sim_g2[k].std(ddof=1)
    i = g2_keys.index(k)
    ax.fill_between([i - 0.45, i + 0.45], 1 - 1.96 * sd, 1 + 1.96 * sd,
                    color='gray', alpha=0.25, zorder=0)
ax.set_xticks(x); ax.set_xticklabels([g2_labels[k] for k in g2_keys])
ax.set_ylabel(r'$g_2$ (toroidal, exact distance)')
ax.set_title(r'(a) Exact-distance pair correlations vs lattice Bernoulli')
ax.legend(fontsize=8)
ax = axes[1]
labels = [r'$r=2$, horiz.', r'$r=2$, vert.', r'$r=4$, horiz.', r'$r=4$, vert.']
for n in names:
    vals = [dir_g2[n]['h2'], dir_g2[n]['v2'], dir_g2[n]['h4'], dir_g2[n]['v4']]
    ax.plot(range(4), vals, 'o-', color=colors[n], lw=1.5, label=n)
ax.axhline(1.0, color='k', lw=1)
ax.axhline(1.3203, color='red', ls=':', lw=1.2)
ax.text(3.35, 1.327, r'$2C_2\approx1.3203$', color='red', fontsize=8)
for i, lab in enumerate(['h2', 'v2', 'h4', 'v4']):
    sd = sim_dir[lab].std(ddof=1)
    ax.fill_between([i - 0.3, i + 0.3], 1 - 1.96 * sd, 1 + 1.96 * sd,
                    color='gray', alpha=0.25, zorder=0)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r'$g_2$ (single direction)')
ax.set_title(r'(b) Direction-resolved: twin ($r=2$) and cousin ($r=4$) separations')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_g2_distance.pdf', dpi=200, facecolor='white')
plt.close()

# ═══ FIG 4b: twin/cousin ablation ══════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))
abl = {'original': (2.153, 1.466, 0.021),
       'twins removed': (2.155, 1.408, 0.022),
       'cousins removed': (2.149, 1.407, 0.023),
       'both removed': (2.090, 1.350, 0.024)}
x = np.arange(4)
vals = np.array([abl[k][0] for k in abl])
ax.bar(x, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.85)
for i, k in enumerate(abl):
    _, mu, sd = abl[k]
    ax.plot([i - 0.4, i + 0.4], [mu, mu], 'k-', lw=2)
    ax.fill_between([i - 0.4, i + 0.4], mu - 1.96 * sd, mu + 1.96 * sd,
                    color='gray', alpha=0.3, zorder=0)
    ax.text(i, vals[i] + 0.03, f'{vals[i]:.3f}', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(list(abl.keys()), fontsize=8)
ax.set_ylabel(r'Brody $\beta$')
ax.set_title('Ablation: removal of twin and cousin primes (matched nulls in grey)')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_ablation.pdf', dpi=200, facecolor='white')
plt.close()

# ═══ FIG 5: descriptor benchmark (updated) ════════════════════════
metrics = ['brody_beta', 'nni', 'rip_k_maxdev', 'rip_k_rmean', 'g2_annulus', 'd2_box']
labels = [r'Brody $\beta$', 'NNI', r'$\max|L(r)-r|$', r'$L(r)-r$ at NN scale',
          r'$g_2$ annulus $[1,3)$', r'box-counting $D_2$']
detvals = {
    'brody_beta': [beta_e[n] for n in names],
    'nni': [det[n.lower().replace('-', '_')]['nni_euclid'] for n in names],
    'rip_k_maxdev': [rip[n.lower().replace('-', '_')]['maxdev'] for n in names],
    'rip_k_rmean': [rip[n.lower().replace('-', '_')]['L_at_nn'] for n in names],
    'g2_annulus': [det[n.lower().replace('-', '_')]['g2_annulus_1_3'] for n in names],
    'd2_box': [d2c[n]['d2'] for n in names],
}
null_mu = {'brody_beta': null_beta['Bernoulli'].mean(),
           'nni': null_nni.mean(),
           'rip_k_maxdev': rip_null['maxdev_mean'],
           'rip_k_rmean': rip_null['L_at_nn_mean'],
           'g2_annulus': sim_ann.mean(),
           'd2_box': null_d2['d2_corr_mean']}
null_sd = {'brody_beta': null_beta['Bernoulli'].std(ddof=1),
           'nni': null_nni.std(ddof=1),
           'rip_k_maxdev': rip_null['maxdev_sd'],
           'rip_k_rmean': rip_null['L_at_nn_sd'],
           'g2_annulus': sim_ann.std(ddof=1),
           'd2_box': null_d2['d2_corr_sd']}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
ax = axes[0]
b = [beta_e[n] for n in names]
ax.bar(names, b, yerr=[[beta_e[n]-beta_ci[n][0] for n in names],
                       [beta_ci[n][1]-beta_e[n] for n in names]],
       color=[colors[n] for n in names], capsize=5, alpha=0.85)
ax.axhspan(null_mu['brody_beta']-1.96*null_sd['brody_beta'],
           null_mu['brody_beta']+1.96*null_sd['brody_beta'], color='gray', alpha=0.3)
ax.axhline(null_mu['brody_beta'], color='k', ls='--')
ax.set_ylabel(r'Brody $\beta$'); ax.set_title('(a) Local spacing fit')
ax = axes[1]
x = np.arange(len(metrics)); wdt = 0.25
for j, n in enumerate(names):
    zs = [(detvals[m][j] - null_mu[m]) / null_sd[m] for m in metrics]
    ax.bar(x + (j-1)*wdt, zs, wdt, color=colors[n], alpha=0.85, label=n)
ax.axhline(0, color='k', lw=0.8); ax.axhline(1.96, color='gray', ls=':'); ax.axhline(-1.96, color='gray', ls=':')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('z-score vs lattice Bernoulli null'); ax.set_title('(b) Separation from null')
ax.legend(fontsize=8)
ax = axes[2]
for n in names:
    r = np.array(rip[n.lower().replace('-', '_')]['r'])
    L = np.array(rip[n.lower().replace('-', '_')]['L'])
    ax.plot(r, L, lw=1.8, color=colors[n], label=n)
# null envelope (100 realisations, 18 bins)
berL = []
for i in range(100):
    s0 = bernoulli_surface(W, 9580 / N_POP)
    _, _, L0, _, _ = rip_k_border(coords(s0), W, W, r_max=W / 3.0, n_bins=18)
    berL.append(L0)
berL = np.array(berL)
r0 = np.array(rip['row_major']['r'])
ax.fill_between(r0, np.percentile(berL, 2.5, axis=0), np.percentile(berL, 97.5, axis=0),
                color='gray', alpha=0.25, zorder=0, label='null 95\% band')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('r (pixels)'); ax.set_ylabel(r'$L(r)-r$'); ax.set_title('(c) Ripley summary')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_descriptor_benchmark.pdf', dpi=200, facecolor='white')
plt.close()

# ═══ FIG 6: D2 validation ═════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for n in names:
    eps = np.array(d2c[n]['eps']); z2 = np.array(d2c[n]['logz2'])
    ax.plot(np.exp(eps), z2, 'o-', lw=1.5, color=colors[n], label=f"{n} (slope {d2c[n]['d2']:.3f})")
# null mean curve: reconstruct quickly from 200 nulls
null_curves = []
for i in range(200):
    s = bernoulli_surface(W, 9580 / N_POP)
    from audit_v2b import d2_boxcount_corr
    dd = d2_boxcount_corr(s)
    null_curves.append(dd['logz2'])
nc = np.array(null_curves).mean(axis=0)
ax.plot(np.exp(d2c['Row-major']['eps']), nc, 'k--', lw=1.5,
        label=f"null mean (slope {null_d2['d2_corr_mean']:.3f})")
ax.set_xscale('log'); ax.set_xlabel(r'box size $\varepsilon$ (px)')
ax.set_ylabel(r'$\log Z_2$'); ax.set_title('(a) Box-counting partition function')
ax.legend(fontsize=8)
ax = axes[1]
for n in names:
    d = det[n.lower().replace('-', '_')]['d2_gp']
    ax.axhline(d['d2_gp'], color=colors[n], ls='-', lw=2, label=f"{n}: GP slope {d['d2_gp']:.3f}")
ax.axhline(null_d2['gp_mean'], color='k', ls='--', lw=2,
           label=f"null: {null_d2['gp_mean']:.3f}$\\pm${null_d2['gp_sd']:.3f}")
ax.axhline(2.0, color='red', ls=':', lw=1)
ax.set_ylabel(r'$D_2$ (GP correlation sum)'); ax.set_ylim(1.8, 2.1)
ax.set_title(r'(b) Grassberger--Procaccia validation ($r\in[2,105]$)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_d2_validation.pdf', dpi=200, facecolor='white')
plt.close()

# ═══ FIG 7: finite size ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))
delta = fbeta - fnull
dci = np.sqrt((fci_h - fci_l)**2 / 4 + (fnull_sd)**2)
ax.errorbar(fsN, delta, yerr=1.96 * dci, marker='o', color=colors['Row-major'], capsize=4)
ax.axhline(0, color='k', lw=0.8)
ax.set_xscale('log'); ax.set_xlabel(r'window area $N=W^2$'); ax.set_ylabel(r'$\Delta\beta = \beta_{\rm prime} - \beta_{\rm null}$')
ax.set_title(r'Finite-size excess of the fitted Brody parameter')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_finite_size.pdf', dpi=200, facecolor='white')
plt.close()

print('all figures rebuilt')

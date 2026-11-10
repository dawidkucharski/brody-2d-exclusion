#!/usr/bin/env python3
"""Build vector manuscript figures from the seeded prime-pattern results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
NAMES = ('row_major', 'ulam', 'cantor')
LABELS = {'row_major': 'Row-major', 'ulam': 'Ulam', 'cantor': 'Diagonal sweep'}
COLOURS = {'row_major': '#24658a', 'ulam': '#b25c38', 'cantor': '#39805b'}
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, filename):
    fig.savefig(ROOT / filename, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def build_figures(data):
    patterns = data['patterns']
    fixed = data['fixed_null']
    config = data['config']

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    axis = axes[0]
    axis.axhspan(fixed['beta']['row_major']['q025'], fixed['beta']['row_major']['q975'],
                 color='0.86', label='Fixed-count null, 2.5–97.5%')
    for index, name in enumerate(NAMES):
        fit = patterns[name]['fits']['open']
        lower, upper = fit['resampling_interval']
        axis.errorbar(index, fit['beta'], yerr=[[fit['beta'] - lower], [upper - fit['beta']]],
                      fmt='o', color=COLOURS[name], capsize=3, markersize=6)
    axis.axhline(1, color='0.35', linestyle=':', linewidth=1, label=r'Continuous CSR: $\beta=1$')
    axis.set_xticks(range(3), [LABELS[name] for name in NAMES])
    axis.set_ylabel(r'Fitted $\beta$')
    axis.set_title('(a) Open-boundary estimates')
    axis.legend(loc='center left', bbox_to_anchor=(0.02, 0.2), frameon=False)

    axis = axes[1]
    beta = np.asarray(fixed['beta_values'])
    differences = (beta[:, 0] - beta[:, 1], beta[:, 0] - beta[:, 2], beta[:, 1] - beta[:, 2])
    keys = ('row_major__ulam', 'row_major__cantor', 'ulam__cantor')
    tick_labels = ('Row-major − Ulam', 'Row-major − diagonal sweep', 'Ulam − diagonal sweep')
    for index, (sample, key) in enumerate(zip(differences, keys)):
        lower, upper = np.quantile(sample, [0.025, 0.975])
        axis.plot([lower, upper], [index, index], color='0.35', linewidth=4,
                  solid_capstyle='butt', label='Paired null, 2.5–97.5%' if index == 0 else None)
        axis.plot(fixed['pair_tests'][key]['observed_difference'], index, 'o',
                  color='#a13b30', markersize=6, label='Observed contrast' if index == 0 else None)
    axis.axvline(0, color='0.6', linewidth=0.8)
    axis.set_yticks(range(3), tick_labels)
    axis.set_xlabel(r'Difference in fitted $\beta$')
    axis.set_title('(b) Paired embedding contrasts')
    axis.legend(loc='lower right', frameon=False)
    save(fig, 'fig_embedding_robustness.pdf')

    fig, axis = plt.subplots(figsize=(8, 4), layout='constrained')
    boundaries = ('open', 'toroidal', 'interior')
    labels = ('Open', 'Toroidal', 'Interior reference')
    nulls = (fixed['beta']['row_major'], fixed['toroidal_beta'], fixed['interior_beta'])
    for index, null in enumerate(nulls):
        axis.fill_between([index - 0.43, index + 0.43], null['q025'], null['q975'],
                          color='0.86', zorder=0)
    offsets = (-0.22, 0, 0.22)
    for offset, name in zip(offsets, NAMES):
        for index, boundary in enumerate(boundaries):
            fit = patterns[name]['fits'][boundary]
            lower, upper = fit['resampling_interval']
            axis.errorbar(index + offset, fit['beta'],
                          yerr=[[fit['beta'] - lower], [upper - fit['beta']]], fmt='o',
                          color=COLOURS[name], capsize=2, markersize=5,
                          label=LABELS[name] if index == 0 else None)
    axis.set_xticks(range(3), labels)
    axis.set_ylabel(r'Fitted $\beta$')
    axis.legend(ncol=3, frameon=False, loc='upper left')
    save(fig, 'fig_boundary_sensitivity.pdf')

    fig, axis = plt.subplots(figsize=(7, 4), layout='constrained')
    thinning = data['thinning']
    densities = np.array([item['density'] for item in thinning])
    means = np.array([item['prime_mean'] for item in thinning])
    lower = []
    upper = []
    for item in thinning:
        values = np.asarray(item['prime_values'])
        quantiles = np.quantile(values, [0.025, 0.975]) if len(values) > 1 else [values[0], values[0]]
        lower.append(item['prime_mean'] - quantiles[0])
        upper.append(quantiles[1] - item['prime_mean'])
    axis.errorbar(densities, means, yerr=[lower, upper], fmt='o-', color=COLOURS['row_major'],
                  capsize=3, linewidth=1.4, label='Prime thinning')
    null_mean = np.array([item['null']['mean'] for item in thinning])
    null_lower = np.array([item['null']['q025'] for item in thinning])
    null_upper = np.array([item['null']['q975'] for item in thinning])
    axis.fill_between(densities, null_lower, null_upper, color='0.86', label='Bernoulli null, 2.5–97.5%')
    axis.plot(densities, null_mean, '--', color='0.3', linewidth=1)
    axis.set_xscale('log')
    axis.set_xlabel('Target occupancy fraction')
    axis.set_ylabel(r'Fitted $\beta$')
    axis.legend(frameon=False)
    save(fig, 'fig_density_thinning_control.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    axis = axes[0]
    shell_keys = ('r1', 'rsqrt2', 'r2', 'rsqrt5', 'r4')
    shell_labels = (r'$1$', r'$\sqrt{2}$', r'$2$', r'$\sqrt{5}$', r'$4$')
    for index, shell in enumerate(shell_keys):
        null = fixed['shells'][shell]
        axis.fill_between([index - 0.43, index + 0.43], null['q025'], null['q975'], color='0.86')
    for offset, name in zip(offsets, NAMES):
        values = [patterns[name]['shells'][shell] for shell in shell_keys]
        axis.plot(np.arange(5) + offset, values, 'o', markersize=5, color=COLOURS[name], label=LABELS[name])
    axis.axhline(1, color='0.5', linewidth=0.8)
    axis.set_xticks(range(5), shell_labels)
    axis.set_xlabel('Toroidal lattice distance')
    axis.set_ylabel('Normalised shell count')
    axis.set_title('(a) Distance-specific counts', pad=25)
    axis.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.01))

    axis = axes[1]
    direction_keys = ('h2', 'v2', 'h4', 'v4')
    direction_labels = (r'$2$, horizontal', r'$2$, vertical', r'$4$, horizontal', r'$4$, vertical')
    for index, key in enumerate(direction_keys):
        null = fixed['direction'][key]
        axis.fill_between([index - 0.32, index + 0.32], null['q025'], null['q975'], color='0.86')
    axis.plot(range(4), [patterns['row_major']['direction'][key] for key in direction_keys],
              'o', color=COLOURS['row_major'], label='Row-major')
    axis.axhline(1, color='0.5', linewidth=0.8)
    axis.axhline(1.32032, color='#a13b30', linestyle=':', linewidth=1, label=r'Asymptotic $2C_2$')
    axis.set_xticks(range(4), direction_labels, rotation=20, ha='right')
    axis.set_ylabel('Normalised directional count')
    axis.set_title('(b) Row-major directions')
    axis.legend(frameon=False, loc='upper left')
    save(fig, 'fig_g2_distance.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained',
                             gridspec_kw={'width_ratios': [1.05, 1]})
    axis = axes[0]
    metric_names = (r'Brody $\beta$', 'Clark–Evans R', r'$L(\bar r)-\bar r$',
                    'Annulus count')
    nulls = (fixed['beta']['row_major'], fixed['nni'], fixed['ripley_at_nn'],
             fixed['annulus'])
    norm = TwoSlopeNorm(vmin=-36, vcenter=0, vmax=36)
    colourmap = plt.colormaps['RdBu_r']
    for row, name in enumerate(NAMES):
        pattern = patterns[name]
        values = (pattern['fits']['open']['beta'], pattern['nni'], pattern['ripley']['at_mean_nn'],
                  pattern['annulus'])
        for column, (value, null) in enumerate(zip(values, nulls)):
            score = (value - null['mean']) / null['sd']
            axis.add_patch(Rectangle((column, row), 1, 1, facecolor=colourmap(norm(np.clip(score, -36, 36))),
                                     edgecolor='white', linewidth=1))
            axis.text(column + 0.5, row + 0.5, f'{score:.1f}', ha='center', va='center',
                      color='white' if abs(score) > 18 else 'black', fontsize=8)
    axis.set_xlim(0, len(metric_names))
    axis.set_ylim(len(NAMES), 0)
    axis.set_xticks(np.arange(len(metric_names)) + 0.5, metric_names, rotation=35, ha='right')
    axis.set_yticks(np.arange(len(NAMES)) + 0.5, [LABELS[name] for name in NAMES])
    axis.tick_params(length=0)
    axis.set_title('(a) Standardised descriptive differences')

    axis = axes[1]
    curves = np.asarray(fixed['ripley_curves'])
    radii = np.asarray(patterns['row_major']['ripley']['radii'])
    axis.fill_between(radii, np.quantile(curves, 0.025, axis=0), np.quantile(curves, 0.975, axis=0),
                      color='0.86', label='Fixed-count null, pointwise 2.5–97.5%')
    for name in NAMES:
        axis.plot(radii, patterns[name]['ripley']['l_minus_r'], color=COLOURS[name],
                  linewidth=1.3, label=LABELS[name])
    axis.axhline(0, color='0.5', linewidth=0.8)
    axis.set_xscale('log')
    axis.set_xlabel('Radius (lattice units)')
    axis.set_ylabel(r'$L(r)-r$')
    axis.set_title('(b) Border-corrected Ripley summary')
    axis.legend(frameon=False, loc='best')
    save(fig, 'fig_descriptor_benchmark.pdf')

    fig, axis = plt.subplots(figsize=(7, 4), layout='constrained')
    finite = data['finite_size']
    population = np.array([item['population'] for item in finite])
    differences = np.array([item['difference'] for item in finite])
    for position, item in zip(population, finite):
        null = item['null']
        axis.plot([position, position], [null['q025'] - null['mean'], null['q975'] - null['mean']],
                  color='0.45', linewidth=4, solid_capstyle='butt')
    axis.plot(population, differences, 'o-', color=COLOURS['row_major'], linewidth=1.2)
    axis.axhline(0, color='0.4', linewidth=0.8)
    axis.set_xscale('log')
    axis.set_xlabel(r'Window area $W^2$')
    axis.set_ylabel(r'$\beta_{\rm prime}-\overline{\beta}_{\rm null}$')
    save(fig, 'fig_finite_size.pdf')
    print(f"Built six vector figures from {config['fixed_realisations']} paired realisations")


def main():
    with (ROOT / 'prime_spatial_results.json').open(encoding='utf-8') as stream:
        build_figures(json.load(stream))


if __name__ == '__main__':
    main()
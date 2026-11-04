#!/usr/bin/env python3
"""Power calibration of the paired edge-contrast procedure.

Under the fixed-count null the paired contrast (normalised shell counts for
the row-major and Ulam embeddings at toroidal distance sqrt(2)) is
standardised by the exact null variance of Eq. (5) of the manuscript. Under
planted alternatives, labels are selected sequentially with weight
exp(theta*k_i), where k_i is the number of already selected labels whose
distance from label i lies in a target gap set. Two alternatives are used:
G_A = {315, 317}, the integer gap classes of the row-major distance-sqrt(2)
shell, and G_B = {4}, a gap class of the Ulam distance-sqrt(2) shell. The two
shell graphs share no edges, so the sign of the paired contrast identifies
which embedding carries the planted structure. Marginal tests standardise
each embedding's count alone against its own null distribution.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prime_spatial_analysis import (SHELLS, embedding_value_grids, normalised_shell,
                                    paired_shell_moments, random_labels, seed_generator)

ROOT = Path(__file__).resolve().parent
WIDTH = 316
OCCUPIED = 9580


def fenwick_prefix(tree, index):
    """Return the sum of the first index entries of a Fenwick tree."""
    total = 0.0
    while index > 0:
        total += tree[index]
        index -= index & (-index)
    return total


def fenwick_sample(tree, generator):
    """Draw an index with probability proportional to the tree weights."""
    total = fenwick_prefix(tree, len(tree) - 1)
    if total <= 0:
        raise ValueError('empty weighted sampler')
    target = generator.random() * total
    index = 0
    bitmask = 1 << ((len(tree) - 1).bit_length() - 1)
    while bitmask:
        probe = index + bitmask
        if probe < len(tree) and tree[probe] < target:
            index = probe
            target -= tree[probe]
        bitmask >>= 1
    return index + 1


def fenwick_add(tree, index, delta):
    """Add delta to the weight of a single index."""
    while index < len(tree):
        tree[index] += delta
        index += index & (-index)


def biased_label_sample(generator, population, occupied, theta, target_gaps=(2,)):
    """Select labels sequentially with weight exp(theta * k_i)."""
    if theta < 0 or occupied > population:
        raise ValueError('theta must be non-negative and occupied <= population')
    factor = np.exp(theta)
    weights = np.ones(population + 1, dtype=float)
    tree = np.zeros(population + 1, dtype=float)
    for index in range(1, population + 1):
        fenwick_add(tree, index, 1.0)
    gap_counts = np.zeros(population + 1, dtype=np.int16)
    selected = np.zeros(population + 1, dtype=bool)
    chosen = np.empty(occupied, dtype=np.int64)
    for step in range(occupied):
        label = fenwick_sample(tree, generator)
        chosen[step] = label
        selected[label] = True
        fenwick_add(tree, label, -weights[label])
        weights[label] = 0.0
        for gap in target_gaps:
            for neighbour in (label - gap, label + gap):
                if 1 <= neighbour <= population and not selected[neighbour]:
                    delta = weights[neighbour] * (factor - 1.0)
                    fenwick_add(tree, neighbour, delta)
                    weights[neighbour] *= factor
                    gap_counts[neighbour] += 1
    labels = np.zeros(population + 1, dtype=bool)
    labels[chosen] = True
    return labels


GAP_SETS = {'row_major': (315, 317),
            'ulam': (1246, 1248, 1250, 1252, 1254, 1256, 1258)}


def run(args):
    started = time.monotonic()
    grids = embedding_value_grids(WIDTH)
    shifts = SHELLS['rsqrt2']
    moments = paired_shell_moments(grids['row_major'], grids['ulam'], shifts, OCCUPIED)
    sigma_zero = np.sqrt(moments['variance_normalised_difference'])

    rng = seed_generator(args.seed, 90)
    null_values = []
    for _ in range(args.null):
        random_pattern = random_labels(rng, WIDTH * WIDTH, OCCUPIED)
        first = normalised_shell(random_pattern[grids['row_major']], shifts)
        second = normalised_shell(random_pattern[grids['ulam']], shifts)
        null_values.append((first, second))
    null_values = np.asarray(null_values)
    null_difference = null_values[:, 0] - null_values[:, 1]
    null_standardised = null_difference / sigma_zero
    marginal_row_sd = float(np.std(null_values[:, 0], ddof=1))
    marginal_ulam_sd = float(np.std(null_values[:, 1], ddof=1))
    marginal_row_null = (null_values[:, 0] - 1.0) / marginal_row_sd
    marginal_ulam_null = (null_values[:, 1] - 1.0) / marginal_ulam_sd
    level_paired = float(np.mean(np.abs(null_standardised) >= 1.96))
    level_row = float(np.mean(np.abs(marginal_row_null) >= 1.96))
    level_ulam = float(np.mean(np.abs(marginal_ulam_null) >= 1.96))
    print(f'null: n={args.null}, sigma0={sigma_zero:.6f}, '
          f'paired level={level_paired:.4f}, marginal row/ulam levels={level_row:.4f}/{level_ulam:.4f}',
          flush=True)

    alternative = []
    for stream, theta in enumerate(args.theta):
        for owner, gap_set in GAP_SETS.items():
            rng = seed_generator(args.seed, 91 + stream * len(GAP_SETS) + list(GAP_SETS).index(owner))
            paired_z, row_z, ulam_z = [], [], []
            for _ in range(args.alternatives):
                random_pattern = biased_label_sample(rng, WIDTH * WIDTH, OCCUPIED, theta,
                                                     target_gaps=gap_set)
                first = normalised_shell(random_pattern[grids['row_major']], shifts)
                second = normalised_shell(random_pattern[grids['ulam']], shifts)
                paired_z.append((first - second) / sigma_zero)
                row_z.append((first - 1.0) / marginal_row_sd)
                ulam_z.append((second - 1.0) / marginal_ulam_sd)
            paired_z = np.asarray(paired_z)
            row_z = np.asarray(row_z)
            ulam_z = np.asarray(ulam_z)
            alternative.append({
                'theta': float(theta),
                'owner': owner,
                'gap_set': list(gap_set),
                'realisations': args.alternatives,
                'mean_paired_z': float(np.mean(paired_z)),
                'power_paired': float(np.mean(np.abs(paired_z) >= 1.96)),
                'power_marginal_row': float(np.mean(np.abs(row_z) >= 1.96)),
                'power_marginal_ulam': float(np.mean(np.abs(ulam_z) >= 1.96)),
                'paired_z': paired_z.tolist(),
            })
            print(f'theta={theta}, owner={owner}: mean paired z={np.mean(paired_z):.3f}, '
                  f'paired power={alternative[-1]["power_paired"]:.4f}, '
                  f'marginal row/ulam power={alternative[-1]["power_marginal_row"]:.4f}/'
                  f'{alternative[-1]["power_marginal_ulam"]:.4f}', flush=True)

    output = {
        'config': {'width': WIDTH, 'occupied': OCCUPIED, 'shell': 'rsqrt2',
                   'seed': args.seed, 'null_realisations': args.null,
                   'alternative_realisations': args.alternatives,
                   'theta': list(args.theta), 'sigma_zero': sigma_zero,
                   'marginal_row_sd': marginal_row_sd, 'marginal_ulam_sd': marginal_ulam_sd},
        'null': {'paired_level': level_paired, 'marginal_row_level': level_row,
                 'marginal_ulam_level': level_ulam},
        'alternative': alternative,
    }
    with (ROOT / args.output).open('w', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True, layout='constrained')
    x_values = [0.0] + list(args.theta)
    rows = {}
    for item in alternative:
        rows.setdefault(item['owner'], []).append(item)
    for axis, owner in zip(axes, ('row_major', 'ulam')):
        paired = [level_paired] + [item['power_paired'] for item in rows[owner]]
        marginal_row = [level_row] + [item['power_marginal_row'] for item in rows[owner]]
        marginal_ulam = [level_ulam] + [item['power_marginal_ulam'] for item in rows[owner]]
        axis.plot(x_values, paired, 'o-', color='#24658a', label='Paired contrast')
        axis.plot(x_values, marginal_row, 's--', color='#b25c38', label='Marginal row-major')
        axis.plot(x_values, marginal_ulam, '^:', color='#39805b', label='Marginal Ulam')
        axis.axhline(0.05, color='0.4', linestyle=':', linewidth=1, label='Nominal 5%')
        axis.set_xlabel(r'Enrichment strength $\theta$')
        axis.set_title(f"Planted gaps {GAP_SETS[owner]}, owner {owner}")
    axes[0].set_ylabel('Rejection rate at nominal 5%')
    axes[1].legend(frameon=False, loc='center right')
    figure.savefig(ROOT / args.figure, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(f'Wrote {args.output} and {args.figure} in {(time.monotonic() - started) / 60:.2f} minutes',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--null', type=int, default=2000)
    parser.add_argument('--alternatives', type=int, default=400)
    parser.add_argument('--theta', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--output', default='power_calibration.json')
    parser.add_argument('--figure', default='fig_power_calibration.pdf')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

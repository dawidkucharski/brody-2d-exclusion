#!/usr/bin/env python3
"""Independent level check of the paired and marginal tests at fixed +/-1.96.

The power calibration uses the fixed Gaussian thresholds +/-1.96 (nominal 5%,
two-sided); no empirical critical value is estimated. This script repeats the
null phase with a second seed so that the reported type-I error estimates do
not reuse the null ensemble used for the standard deviations and figures.
"""

import argparse
import json

import numpy as np

from prime_spatial_analysis import (SHELLS, embedding_value_grids, normalised_shell,
                                    paired_shell_moments, random_labels, seed_generator)

WIDTH = 316
OCCUPIED = 9580


def run(args):
    grids = embedding_value_grids(WIDTH)
    shifts = SHELLS['rsqrt2']
    moments = paired_shell_moments(grids['row_major'], grids['ulam'], shifts, OCCUPIED)
    sigma_zero = float(np.sqrt(moments['variance_normalised_difference']))
    rng = seed_generator(args.seed, 90)
    values = []
    for _ in range(args.null):
        pattern = random_labels(rng, WIDTH * WIDTH, OCCUPIED)
        first = normalised_shell(pattern[grids['row_major']], shifts)
        second = normalised_shell(pattern[grids['ulam']], shifts)
        values.append((first, second))
    values = np.asarray(values)
    difference = values[:, 0] - values[:, 1]
    row = (values[:, 0] - 1.0) / float(np.std(values[:, 0], ddof=1))
    ulam = (values[:, 1] - 1.0) / float(np.std(values[:, 1], ddof=1))
    output = {
        'seed': args.seed,
        'null_realisations': args.null,
        'threshold': 1.96,
        'sigma_zero': sigma_zero,
        'marginal_row_sd': float(np.std(values[:, 0], ddof=1)),
        'marginal_ulam_sd': float(np.std(values[:, 1], ddof=1)),
        'paired_level': float(np.mean(np.abs(difference / sigma_zero) >= 1.96)),
        'marginal_row_level': float(np.mean(np.abs(row) >= 1.96)),
        'marginal_ulam_level': float(np.mean(np.abs(ulam) >= 1.96)),
    }
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--null', type=int, default=2000)
    parser.add_argument('--output', default='power_level_check.json')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

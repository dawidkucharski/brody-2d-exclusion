#!/usr/bin/env python3
"""Independent checks for the prime-pattern analysis and its null models."""

import itertools
import json
import re
import unittest
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma


def round_decimal(value, places):
    """Round half-up so manuscript display rounding can be checked exactly."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))

from prime_spatial_analysis import (SHELLS, annulus_statistic, box_partition, brody_fit_diagnostics,
                                    coords, embedding_value_grids, fit_brody_distances,
                                    inhomogeneous_probabilities, nearest_neighbour_distances,
                                    normalised_shell, primes_upto, rip_k_border,
                                    paired_graph_moments, paired_shell_moments,
                                    pairwise_slope, shell_count, shell_edges)


def direct_brody_fit(distances):
    """Independent full-sample implementation of the continuous fit."""
    spacings = distances / np.mean(distances)

    def objective(beta):
        shape = beta + 1
        coefficient = gamma(1 + 1 / shape) ** shape
        likelihood = (len(spacings) * np.log(shape * coefficient) +
                      beta * np.log(spacings).sum() - coefficient * np.sum(spacings ** shape))
        return -float(likelihood)

    return float(minimize_scalar(objective, bounds=(0, 15), method='bounded').x)


class PrimeSpatialAnalysisTests(unittest.TestCase):
    def test_hand_checked_four_by_four_embeddings(self):
        grids = embedding_value_grids(4)
        expected = {
            'row_major': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            'ulam': [[16, 15, 14, 13], [5, 4, 3, 12], [6, 1, 2, 11], [7, 8, 9, 10]],
            'cantor': [[1, 2, 4, 7], [3, 5, 8, 11], [6, 9, 12, 14], [10, 13, 15, 16]],
        }
        for name, grid in expected.items():
            np.testing.assert_array_equal(grids[name], np.array(grid))

    def test_full_embeddings_and_sieve(self):
        width = 316
        population = width * width
        primes = primes_upto(population)
        labels = np.zeros(population + 1, dtype=bool)
        labels[primes] = True
        grids = embedding_value_grids(width)
        for name, grid in grids.items():
            np.testing.assert_array_equal(np.sort(grid.ravel()), np.arange(1, population + 1))
            self.assertEqual(int(labels[grid].sum()), 9580)
        np.testing.assert_array_equal(grids['row_major'][0], np.arange(1, width + 1))
        self.assertEqual(grids['ulam'][158, 157], 1)
        self.assertEqual(grids['ulam'][158, 158], 2)
        self.assertEqual(grids['cantor'][0, 0], 1)
        for value in (1, 2, 3, 4, 97, 9973, 99991, 100000):
            expected = value >= 2 and all(value % divisor for divisor in range(2, int(value ** 0.5) + 1))
            self.assertEqual(value in primes_upto(100000), expected)
        self.assertEqual(len(primes_upto(100000)) - len(primes), 12)

    def test_nearest_neighbours_and_periodic_annulus(self):
        points = np.array([[0., 0.], [9., 0.], [3., 0.], [0., 2.], [8., 8.], [5., 5.]])
        expected_neighbours = []
        directed_pairs = 0
        for index, point in enumerate(points):
            distances = []
            for other_index, other in enumerate(points):
                if index == other_index:
                    continue
                delta = np.abs(point - other)
                distance = np.linalg.norm(np.minimum(delta, 10 - delta))
                distances.append(distance)
                directed_pairs += 1 <= distance < 3
            expected_neighbours.append(min(distances))
        expected_annulus = directed_pairs / len(points) / ((len(points) / 100) * np.pi * 8)
        surface = np.zeros((10, 10), dtype=bool)
        surface[points[:, 1].astype(int), points[:, 0].astype(int)] = True
        self.assertAlmostEqual(annulus_statistic(surface), expected_annulus)
        np.testing.assert_allclose(np.sort(nearest_neighbour_distances(surface, 'toroidal')),
                       np.sort(expected_neighbours))

    def test_brody_fit_and_resampling(self):
        distances = np.array([1., 1., 1., 1.4, 1.4, 2., 2., 2., 3., 4.])
        seed, repetitions = 27, 30
        actual = fit_brody_distances(distances, resamples=repetitions,
                                     rng=np.random.default_rng(seed))
        self.assertAlmostEqual(actual['beta'], direct_brody_fit(distances), places=8)
        values, counts = np.unique(distances, return_counts=True)
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(repetitions):
            sampled_counts = rng.multinomial(len(distances), counts / len(distances))
            sample = np.repeat(values, sampled_counts)
            estimates.append(direct_brody_fit(sample))
        np.testing.assert_allclose(actual['resampling_interval'],
                                   np.percentile(estimates, [2.5, 97.5]), atol=1e-8)

    def test_brody_cdf_discrepancy_includes_ties(self):
        distances = np.array([1., 1., 1., 1.4, 1.4, 2., 2., 2., 3., 4.])
        sample = np.sort(distances / distances.mean())
        beta = fit_brody_distances(distances)['beta']
        points = np.unique(np.concatenate(([0.], sample, np.nextafter(sample, -np.inf))))
        shape = beta + 1
        scale = gamma(1 + 1 / shape) ** shape
        fitted = 1 - np.exp(-scale * points ** shape)
        empirical = np.searchsorted(sample, points, side='right') / len(sample)
        diagnostics = brody_fit_diagnostics(distances, beta)
        self.assertAlmostEqual(diagnostics['cdf_supremum'],
                               float(np.max(np.abs(empirical - fitted))), places=13)
        self.assertEqual(diagnostics['distinct_distances'], 5)
        self.assertAlmostEqual(diagnostics['largest_tie_fraction'], 0.3)

    def test_inhomogeneous_expected_count_and_exceptions(self):
        for width, target in ((316, 9580), (4, 13)):
            probabilities, _ = inhomogeneous_probabilities(width, target)
            self.assertEqual(probabilities[0], 0)
            self.assertEqual(probabilities[1], 1)
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
            self.assertAlmostEqual(float(probabilities.sum()), target, places=8)
            sample = np.random.default_rng(42).random(width * width) < probabilities
            self.assertFalse(sample[0])
            self.assertTrue(sample[1])
        self.assertAlmostEqual(inhomogeneous_probabilities(316, 9580)[1], 0.9961876724, places=9)

    def test_border_estimator_with_directed_pairs(self):
        points = np.array([[1., 5.], [2., 5.], [5., 5.], [9., 9.]])
        radii, estimates, _, _, _ = rip_k_border(points, 10, 10, 9, 1)
        radius = radii[0]
        references = [point for point in points if radius <= point[0] <= 10 - radius
                      and radius <= point[1] <= 10 - radius]
        count = sum(np.linalg.norm(point - other) <= radius and not np.array_equal(point, other)
                    for point in references for other in points)
        self.assertAlmostEqual(estimates[0], 100 * count / (len(points) * len(references)))
        edge_points = np.array([[0.5, 5.], [1.5, 5.]])
        _, estimates, _, maximum, _ = rip_k_border(edge_points, 10, 10, 9, 1)
        self.assertTrue(np.isnan(estimates[0]))
        self.assertTrue(np.isnan(maximum))

    def test_exact_paired_shell_variance(self):
        width, occupied = 4, 3
        grids = embedding_value_grids(width)
        pairs = (('row_major', 'ulam'), ('row_major', 'cantor'), ('ulam', 'cantor'))
        for shell in ('r1', 'rsqrt2'):
            shifts = SHELLS[shell]
            differences = {pair: [] for pair in pairs}
            normalised = []
            for chosen in itertools.combinations(range(1, width * width + 1), occupied):
                labels = np.zeros(width * width + 1, dtype=bool)
                labels[list(chosen)] = True
                surfaces = {name: labels[grid] for name, grid in grids.items()}
                counts = {name: shell_count(surface, shifts) for name, surface in surfaces.items()}
                normalised.append(normalised_shell(surfaces['row_major'], shifts))
                for first, second in pairs:
                    differences[(first, second)].append(counts[first] - counts[second])
            self.assertAlmostEqual(float(np.mean(normalised)), 1.0, places=13)
            for first, second in pairs:
                moments = paired_shell_moments(grids[first], grids[second], shifts, occupied)
                observed = differences[(first, second)]
                self.assertAlmostEqual(float(np.mean(observed)), 0.0, places=13)
                self.assertAlmostEqual(float(np.var(observed)),
                                       moments['variance_count_difference'], places=13)

    def test_irregular_graph_variance_and_regular_limit(self):
        width, population = 4, 16
        grids = embedding_value_grids(width)
        first = shell_edges(grids['row_major'], SHELLS['r1'], 'open')
        second = shell_edges(grids['ulam'], SHELLS['rsqrt2'], 'open')
        for occupied in (2, 3, 4, 14, 15, 16):
            moments = paired_graph_moments(first, second, population, occupied)
            contrasts = []
            for chosen in itertools.combinations(range(1, population + 1), occupied):
                labels = np.zeros(population + 1, dtype=bool)
                labels[list(chosen)] = True
                first_count = np.count_nonzero(labels[first // (population + 1)] &
                                               labels[first % (population + 1)])
                second_count = np.count_nonzero(labels[second // (population + 1)] &
                                                labels[second % (population + 1)])
                contrasts.append(first_count / moments['expected_first'] -
                                 second_count / moments['expected_second'])
            self.assertAlmostEqual(float(np.mean(contrasts)), 0.0, places=13)
            self.assertAlmostEqual(float(np.var(contrasts)),
                                   moments['variance_normalised_difference'], places=13)
        first = shell_edges(grids['row_major'], SHELLS['rsqrt2'])
        second = shell_edges(grids['ulam'], SHELLS['rsqrt2'])
        general = paired_graph_moments(first, second, population, 3)
        regular = paired_shell_moments(grids['row_major'], grids['ulam'], SHELLS['rsqrt2'], 3)
        self.assertAlmostEqual(general['degree_energy'], 0.0, places=14)
        self.assertAlmostEqual(general['variance_normalised_difference'],
                               regular['variance_normalised_difference'], places=13)

    def test_multiscale_descriptors_against_direct_counts(self):
        rng = np.random.default_rng(2026)
        surface = rng.random((16, 16)) < 0.2
        sizes, partitions = [], []
        for size in (2, 4, 8):
            masses = [int(surface[row:row + size, column:column + size].sum())
                      for row in range(0, 16, size) for column in range(0, 16, size)]
            sizes.append(size)
            partitions.append(sum((mass / sum(masses)) ** 2 for mass in masses))
        expected_slope = np.polyfit(np.log(sizes), np.log(partitions), 1)[0]
        self.assertAlmostEqual(box_partition(surface)['slope'], expected_slope)
        radii = np.array([1., 2., 3., 5.])
        actual = pairwise_slope(surface, radii)['correlation']
        points = coords(surface)
        pair_distances = [np.linalg.norm(first - second)
                          for first, second in itertools.combinations(points, 2)]
        expected = [2 * sum(distance <= radius for distance in pair_distances) /
                    (len(points) * (len(points) - 1)) for radius in radii]
        np.testing.assert_allclose(actual, expected)

    def test_manuscript_tables_match_recorded_results(self):
        root = Path(__file__).resolve().parent
        data = json.loads((root / 'prime_spatial_results.json').read_text(encoding='utf-8'))
        manuscript = (root / 'aurora_manuscript.tex').read_text(encoding='utf-8')

        def table_row(label, name):
            section = manuscript.split('\\label{' + label + '}', 1)[1].split('\\end{table}', 1)[0]
            for line in section.splitlines():
                if line.strip().startswith(name + ' &'):
                    return line.split('&')[1:]
            self.fail(f'No {name!r} row found in {label}')

        def numbers(text):
            return [float(value) for value in re.findall(r'[-+]?\d+(?:\.\d+)?',
                                                          text.replace('\\,', ''))]

        def close(actual, expected, tolerance=0.00051):
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)

        names = ('row_major', 'ulam', 'cantor')
        patterns = data['patterns']
        for heading, boundary in (('Open', 'open'), ('Toroidal', 'toroidal'),
                                  ('Interior reference', 'interior')):
            cells = table_row('tab:local', heading)
            for name, cell in zip(names, cells):
                fit = patterns[name]['fits'][boundary]
                close(numbers(cell), [fit['beta'], *fit['resampling_interval']])

        shells = ('r1', 'rsqrt2', 'r2', 'rsqrt5', 'r4')
        for heading, name in (('Row-major', 'row_major'), ('Ulam', 'ulam'), ('Diagonal sweep', 'cantor')):
            cells = table_row('tab:g2', heading)
            close([numbers(cell)[0] for cell in cells],
                  [patterns[name]['shells'][shell] for shell in shells])

        fixed = data['fixed_null']
        metrics = {
            'Brody $\\hat\\beta$': ([patterns[name]['fits']['open']['beta'] for name in names],
                                     fixed['beta']['row_major']),
            'Clark--Evans $R$': ([patterns[name]['nni'] for name in names], fixed['nni']),
            '$L(\\bar r)-\\bar r$': ([patterns[name]['ripley']['at_mean_nn'] for name in names],
                                    fixed['ripley_at_nn']),
            '$\\max|L(r)-r|$': ([patterns[name]['ripley']['max_abs'] for name in names],
                                 fixed['ripley_max_abs']),
            'Annulus $[1,3)$': ([patterns[name]['annulus'] for name in names], fixed['annulus']),
        }
        for heading, (values, null) in metrics.items():
            cells = table_row('tab:descriptors', heading)
            close([numbers(cell)[0] for cell in cells[:3]], values)
            close(numbers(cells[3]), [null['mean'], null['sd']])

        discrepancy_cells = table_row('tab:descriptors', 'CDF discrepancy $D_{\\rm fit}$')
        close([numbers(cell)[0] for cell in discrepancy_cells[:3]],
              [patterns[name]['brody_diagnostics']['cdf_supremum'] for name in names])

        comparisons = (('Row-major -- Ulam', 'row_major__ulam', 0, 1),
                       ('Row-major -- diagonal sweep', 'row_major__cantor', 0, 2),
                       ('Ulam -- diagonal sweep', 'ulam__cantor', 1, 2))
        simulated_shells = np.asarray(fixed['paired_sqrt2_shell_values'])
        for heading, key, first, second in comparisons:
            row = [numbers(cell)[0] for cell in table_row('tab:paired', heading)]
            test = fixed['pair_tests'][key]
            moments = data['shell_moments'][key]['rsqrt2']
            expected = [test['observed_difference'], test['max_adjusted_p_value'],
                        moments['shared_edges'], np.sqrt(moments['variance_normalised_difference']),
                        np.std(simulated_shells[:, first] - simulated_shells[:, second], ddof=1)]
            close(row, expected, tolerance=0.00051)
            open_row = [numbers(cell)[0] for cell in table_row('tab:open-variance', heading)]
            open_result = data['open_shell_validation']['contrasts'][key]
            exact = open_result['exact']
            open_expected = [exact['shared_edges'], exact['degree_energy'] * 1e7,
                             np.sqrt(exact['variance_normalised_difference']),
                             open_result['simulation']['sd']]
            close(open_row, open_expected, tolerance=0.00051)

        for item in data['finite_size']:
            cells = table_row('tab:finite', f"${item['width']}$")
            actual = [numbers(cells[0])[0], numbers(cells[1])[0],
                      *numbers(cells[2]), numbers(cells[3])[0]]
            expected = [item['occupied'], item['beta'], item['null']['mean'],
                        item['null']['sd'], item['difference']]
            close(actual, expected)

        beta_values = np.asarray(fixed['beta_values'])
        maximum = np.max(np.abs(np.column_stack((beta_values[:, 0] - beta_values[:, 1],
                                                 beta_values[:, 0] - beta_values[:, 2],
                                                 beta_values[:, 1] - beta_values[:, 2]))), axis=1)
        for _, key, _, _ in comparisons:
            test = fixed['pair_tests'][key]
            exceedances = np.count_nonzero(maximum >= abs(test['observed_difference']))
            self.assertEqual(int(exceedances), test['max_exceedances'])
            self.assertEqual((exceedances + 1) / (len(maximum) + 1), test['max_adjusted_p_value'])


class PowerCalibrationTests(unittest.TestCase):
    def test_fenwick_sampler_is_proportional(self):
        from power_calibration import fenwick_add, fenwick_sample
        generator = np.random.default_rng(7)
        tree = np.zeros(11, dtype=float)
        for index in range(1, 11):
            fenwick_add(tree, index, float(index))
        samples = np.asarray([fenwick_sample(tree, generator) for _ in range(100000)])
        probabilities = np.bincount(samples, minlength=11)[1:] / len(samples)
        np.testing.assert_allclose(probabilities, np.arange(1, 11) / 55, atol=0.004)

    def test_biased_sampler_is_uniform_at_theta_zero(self):
        from power_calibration import biased_label_sample
        generator = np.random.default_rng(11)
        population, occupied, theta, gaps = 1000, 100, 0.0, (2, 4)
        labels = biased_label_sample(generator, population, occupied, theta, gaps)
        chosen = np.flatnonzero(labels)
        self.assertEqual(len(chosen), occupied)
        self.assertEqual(len(np.unique(chosen)), occupied)
        observed = 0
        for gap in gaps:
            observed += int((labels[1:population + 1 - gap] & labels[gap + 1:]).sum())
        expected = sum((population - gap) for gap in gaps) * occupied * (occupied - 1) / (population * (population - 1))
        self.assertLess(abs(observed - expected), 4 * np.sqrt(expected))

    def test_manuscript_power_table_if_present(self):
        from power_calibration import GAP_SETS
        path = Path(__file__).resolve().parent / 'power_calibration.json'
        if not path.exists():
            self.skipTest('power_calibration.json not generated')
        data = json.loads(path.read_text())
        manuscript = (Path(__file__).resolve().parent / 'aurora_manuscript.tex').read_text()
        label_pos = manuscript.index('\\label{tab:power}')
        start = manuscript.rfind('\\begin{table}', 0, label_pos)
        end = manuscript.index('\\end{table}', label_pos) + len('\\end{table}')
        block = manuscript[start:end]
        level = data['null']
        level_line = next(line for line in block.splitlines() if 'Fixed-count null' in line)
        expected_level = [round_decimal(level['paired_level'], 3),
                          round_decimal(level['marginal_row_level'], 3),
                          round_decimal(level['marginal_ulam_level'], 3)]
        self.assertEqual([float(value) for value in re.findall(r'0\.\d{3,4}', level_line)],
                         expected_level)
        for item in data['alternative']:
            name = 'Row-major' if item['owner'] == 'row_major' else 'Ulam'
            row = next(line for line in block.splitlines()
                       if name in line and f'${item["theta"]}$' in line)
            cells = [cell for cell in row.replace('\\\\', '').split('&')][1:]
            reported = [float(re.findall(r'[-0-9.]+', cell)[-1]) for cell in cells[1:]]
            expected = [round_decimal(item['power_paired'], 3),
                        round_decimal(item['power_marginal_row'], 3),
                        round_decimal(item['power_marginal_ulam'], 3)]
            self.assertEqual(reported, expected, msg=f'{name} theta={item["theta"]}')
        self.assertEqual(set(GAP_SETS), {'row_major', 'ulam'})

    def test_independent_level_check_if_present(self):
        path = Path(__file__).resolve().parent / 'power_level_check.json'
        if not path.exists():
            self.skipTest('power_level_check.json not generated')
        level_check = json.loads(path.read_text())
        self.assertEqual(level_check['seed'], 20261102)
        self.assertEqual(level_check['null_realisations'], 2000)
        main = json.loads((Path(__file__).resolve().parent / 'power_calibration.json').read_text())
        self.assertAlmostEqual(level_check['sigma_zero'], main['config']['sigma_zero'], places=10)
        for key in ('paired_level', 'marginal_row_level', 'marginal_ulam_level'):
            self.assertGreaterEqual(level_check[key], 0.0)
            self.assertLessEqual(level_check[key], 0.12)
        manuscript = (Path(__file__).resolve().parent / 'aurora_manuscript.tex').read_text()
        expected = [round_decimal(level_check['paired_level'], 3),
                    round_decimal(level_check['marginal_row_level'], 3),
                    round_decimal(level_check['marginal_ulam_level'], 3)]
        lines = manuscript.splitlines()
        anchor = next(index for index, line in enumerate(lines) if 'marginal Ulam levels' in line)
        reported = [float(value) for value in re.findall(r'0\.\d{3,4}', lines[anchor + 1])]
        self.assertEqual(reported, expected)


class SeismicApplicationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parent / 'seismic_application.json'
        if not path.exists():
            self.skipTest('seismic_application.json not generated')
        self.data = json.loads(path.read_text())

    def test_nested_graphs_and_exact_variance(self):
        graphs = self.data['graphs']
        self.assertTrue(graphs['nested'])
        self.assertEqual(graphs['shared_edges'], graphs['rook_edges'])
        variance = self.data['variance']
        self.assertAlmostEqual(variance['exact'], variance['recomputed_from_terms'], places=12)
        self.assertAlmostEqual(variance['sigma'], np.sqrt(variance['exact']), places=12)

    def test_null_statistics_from_stored_contrasts(self):
        contrasts = np.asarray(self.data['null']['contrasts'])
        null = self.data['null']
        self.assertEqual(len(contrasts), self.data['config']['null_realisations'])
        self.assertAlmostEqual(null['mean'], float(np.mean(contrasts)), places=10)
        self.assertAlmostEqual(null['sd'], float(np.std(contrasts, ddof=1)), places=10)
        observed = self.data['counts']['observed_difference']
        exceedances = int(np.count_nonzero(np.abs(contrasts) >= abs(observed)))
        self.assertEqual(null['exceedances'], exceedances)
        self.assertAlmostEqual(null['monte_carlo_value'],
                               (exceedances + 1) / (len(contrasts) + 1), places=10)
        self.assertAlmostEqual(null['sd'], self.data['variance']['sigma'], delta=0.001)

    def test_manuscript_seismic_table(self):
        manuscript = (Path(__file__).resolve().parent / 'aurora_manuscript.tex').read_text()
        label_pos = manuscript.index('\\label{tab:seismic}')
        start = manuscript.rfind('\\begin{table}', 0, label_pos)
        end = manuscript.index('\\end{table}', label_pos) + len('\\end{table}')
        block = manuscript[start:end]

        def row_values(name):
            row = next(line for line in block.splitlines() if name in line and '&' in line)
            return [float(re.findall(r'[-0-9.]+', cell.replace('\\,', ''))[-1])
                    for cell in row.replace('\\\\', '').split('&')[1:]]

        counts = self.data['counts']
        self.assertEqual([round_decimal(counts['rook_normalised'], 3),
                          round_decimal(counts['queen_normalised'], 3)],
                         row_values('Normalised count'))
        self.assertEqual([float(counts['rook']), float(counts['queen'])],
                         row_values('Occupied edges'))
        difference_row = next(line for line in block.splitlines() if 'Difference $g' in line)
        difference_cells = [cell for cell in difference_row.replace('\\\\', '').split('&')][1:]
        self.assertEqual(round_decimal(counts['observed_difference'], 4),
                         round_decimal(float(re.findall(r'[-0-9.]+', difference_cells[0])[-1]), 4))
        monte_row = next(line for line in block.splitlines() if 'Monte Carlo value' in line)
        monte_cells = [cell for cell in monte_row.replace('\\', '').split('&')][1:]
        self.assertEqual(round_decimal(self.data['null']['monte_carlo_value'], 4),
                         round_decimal(float(re.findall(r'[-0-9.]+', monte_cells[0])[-1]), 4))


if __name__ == '__main__':
    unittest.main()
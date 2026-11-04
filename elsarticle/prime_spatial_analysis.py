#!/usr/bin/env python3
"""Reproducible spatial analysis of a common finite prime sequence."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import optimize, spatial
from scipy.special import gammaln

SHELLS = {
    'r1': ((1, 0), (0, 1)),
    'rsqrt2': ((1, 1), (1, -1)),
    'r2': ((2, 0), (0, 2)),
    'rsqrt5': ((1, 2), (1, -2), (2, 1), (2, -1)),
    'r4': ((4, 0), (0, 4)),
}


def primes_upto(limit):
    """Return all primes not exceeding the integer limit."""
    if limit < 1:
        return np.array([], dtype=int)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for divisor in range(2, int(limit ** 0.5) + 1):
        if sieve[divisor]:
            sieve[divisor * divisor:limit + 1:divisor] = False
    return np.flatnonzero(sieve)


def coords(surface):
    """Return occupied lattice indices in column-row order."""
    rows, columns = np.nonzero(surface)
    return np.column_stack((columns, rows)).astype(float)


def inhomogeneous_probabilities(width, target):
    """Calibrate label probabilities with p(1)=0 and p(2)=1."""
    population = width * width
    if population < 3 or not 1 <= target <= population - 1:
        raise ValueError('target must be between 1 and width**2 - 1')
    labels = np.arange(1, population + 1)
    weights = np.zeros(population, dtype=float)
    weights[2:] = 1.0 / np.log(labels[2:])
    calibration = (target - 1) / weights.sum()
    if np.any(calibration * weights[2:] > 1):
        calibration = optimize.brentq(
            lambda value: np.minimum(1.0, value * weights[2:]).sum() + 1 - target,
            0.0, np.log(population))
    probabilities = np.minimum(1.0, calibration * weights)
    probabilities[1] = 1.0
    return probabilities, float(calibration)


def rip_k_border(points, width, height, max_radius, bins):
    """Estimate Ripley's K by minus sampling at fixed geometric radii."""
    count = len(points)
    if count < 2 or max_radius <= 1 or bins < 1:
        raise ValueError('at least two points, max_radius > 1, and bins >= 1 are required')
    tree = spatial.cKDTree(points)
    edges = np.geomspace(1.0, max_radius, bins + 1)
    radii = np.sqrt(edges[:-1] * edges[1:])

    def estimate(radius):
        inside = ((points[:, 0] >= radius) & (points[:, 0] <= width - radius) &
                  (points[:, 1] >= radius) & (points[:, 1] <= height - radius))
        references = int(inside.sum())
        if references == 0:
            return np.nan
        neighbours = tree.query_ball_point(points[inside], radius, return_length=True) - 1
        return float(width * height * neighbours.sum() / (count * references))

    estimates = np.array([estimate(radius) for radius in radii])
    transformed = np.sqrt(estimates / np.pi) - radii
    finite = np.isfinite(transformed)
    maximum = float(np.max(np.abs(transformed[finite]))) if finite.any() else np.nan
    mean_distance = float(tree.query(points, k=2)[0][:, 1].mean())
    at_mean = float(np.sqrt(estimate(mean_distance) / np.pi) - mean_distance)
    return radii, estimates, transformed, maximum, at_mean


def fit_brody_distances(distances, resamples=0, rng=None):
    """Fit the unit-mean Brody family and optionally resample its distance histogram."""
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if len(distances) < 10:
        raise ValueError('at least ten positive finite distances are required')
    values, frequencies = np.unique(distances, return_counts=True)
    count = int(frequencies.sum())

    def fit_counts(weights):
        mean = float(np.dot(values, weights) / count)
        log_spacing = np.log(values / mean)
        sum_log_spacing = float(np.dot(weights, log_spacing))

        def negative_log_likelihood(beta):
            shape = beta + 1.0
            log_a = shape * gammaln(1.0 + 1.0 / shape)
            likelihood = (count * (np.log(shape) + log_a) + beta * sum_log_spacing -
                          np.dot(weights, np.exp(log_a + shape * log_spacing)))
            return -float(likelihood)

        result = optimize.minimize_scalar(negative_log_likelihood, bounds=(0, 15),
                                          method='bounded')
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('Brody fit did not converge')
        return float(result.x)

    result = {'beta': fit_counts(frequencies), 'n': count}
    if resamples:
        if rng is None:
            raise ValueError('a random generator is required for resampling')
        probabilities = frequencies / count
        estimates = [fit_counts(rng.multinomial(count, probabilities))
                     for _ in range(resamples)]
        result['resampling_interval'] = np.percentile(estimates, [2.5, 97.5]).tolist()
    return result


def brody_fit_diagnostics(distances, beta):
    """Describe the empirical CDF discrepancy and tied-distance mass."""
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if len(distances) < 2 or beta < 0:
        raise ValueError('positive distances and a non-negative beta are required')
    values, counts = np.unique(distances / distances.mean(), return_counts=True)
    shape = beta + 1.0
    log_a = shape * gammaln(1.0 + 1.0 / shape)
    fitted_cdf = -np.expm1(-np.exp(log_a) * values ** shape)
    empirical_right = np.cumsum(counts) / len(distances)
    empirical_left = (np.cumsum(counts) - counts) / len(distances)
    discrepancy = max(float(np.max(empirical_right - fitted_cdf)),
                      float(np.max(fitted_cdf - empirical_left)))
    return {'cdf_supremum': discrepancy, 'distinct_distances': len(values),
            'largest_tie_fraction': float(np.max(counts) / len(distances))}


def nearest_neighbour_distances(surface, boundary='open', margin=5):
    """Return nearest-neighbour distances for the selected boundary convention."""
    width, height = surface.shape[1], surface.shape[0]
    rows, columns = np.nonzero(surface)
    points = np.column_stack((columns, rows)).astype(float)
    if len(points) < 2:
        raise ValueError('at least two occupied sites are required')
    if boundary == 'toroidal':
        tree = spatial.cKDTree(points, boxsize=(width, height))
        references = points
    elif boundary in ('open', 'interior'):
        tree = spatial.cKDTree(points)
        if boundary == 'interior':
            centres = points + 0.5
            inside = ((centres[:, 0] >= margin) & (centres[:, 0] <= width - margin) &
                      (centres[:, 1] >= margin) & (centres[:, 1] <= height - margin))
            references = points[inside]
        else:
            references = points
    else:
        raise ValueError('unknown boundary convention')
    return tree.query(references, k=2)[0][:, 1]


def embedding_value_grids(width):
    """Return bijections from lattice sites to integer labels 1,...,width**2."""
    if width < 2 or width % 2:
        raise ValueError('the Ulam construction requires a positive even width')
    population = width * width
    row_major = np.arange(1, population + 1, dtype=np.int64).reshape(width, width)

    ulam = np.zeros((width, width), dtype=np.int64)
    column, row = width // 2 - 1, width // 2
    value, step = 1, 1
    ulam[row, column] = value
    directions = ((1, 0), (0, -1), (-1, 0), (0, 1))
    while value < population:
        for direction_index, (delta_column, delta_row) in enumerate(directions):
            for _ in range(step):
                column += delta_column
                row += delta_row
                value += 1
                ulam[row, column] = value
                if value == population:
                    break
            if direction_index % 2 == 1:
                step += 1
            if value == population:
                break

    cantor = np.zeros((width, width), dtype=np.int64)
    value = 1
    for diagonal in range(2 * width - 1):
        first_row = max(0, diagonal - width + 1)
        last_row = min(diagonal, width - 1)
        for row in range(first_row, last_row + 1):
            column = diagonal - row
            cantor[row, column] = value
            value += 1

    return {'row_major': row_major, 'ulam': ulam, 'cantor': cantor}


def shell_edges(value_grid, shifts, boundary='toroidal'):
    """Encode each undirected shell edge by its two integer labels."""
    population = value_grid.size
    height, width = value_grid.shape
    encoded = []
    for delta_column, delta_row in shifts:
        if boundary == 'toroidal':
            labels = value_grid.ravel()
            neighbours = np.roll(value_grid, (delta_row, delta_column), axis=(0, 1)).ravel()
        elif boundary == 'open':
            first_column, last_column = max(0, -delta_column), min(width, width - delta_column)
            first_row, last_row = max(0, -delta_row), min(height, height - delta_row)
            labels = value_grid[first_row:last_row, first_column:last_column].ravel()
            neighbours = value_grid[first_row + delta_row:last_row + delta_row,
                                    first_column + delta_column:last_column + delta_column].ravel()
        else:
            raise ValueError('unknown shell boundary convention')
        first = np.minimum(labels, neighbours)
        second = np.maximum(labels, neighbours)
        if np.any(first == second):
            raise ValueError('shell shifts must not create self-edges')
        encoded.append(first * (population + 1) + second)
    edges = np.sort(np.concatenate(encoded))
    if np.any(edges[1:] == edges[:-1]):
        raise ValueError('shell representatives must give distinct undirected edges')
    return edges


def shell_count(surface, shifts, boundary='toroidal'):
    """Count occupied undirected edges for one set of opposite-shift representatives."""
    if boundary == 'toroidal':
        return sum(int(np.count_nonzero(surface & np.roll(
            surface, (delta_row, delta_column), axis=(0, 1))))
                   for delta_column, delta_row in shifts)
    if boundary != 'open':
        raise ValueError('unknown shell boundary convention')
    height, width = surface.shape
    total = 0
    for delta_column, delta_row in shifts:
        first_column, last_column = max(0, -delta_column), min(width, width - delta_column)
        first_row, last_row = max(0, -delta_row), min(height, height - delta_row)
        first = surface[first_row:last_row, first_column:last_column]
        second = surface[first_row + delta_row:last_row + delta_row,
                         first_column + delta_column:last_column + delta_column]
        total += int(np.count_nonzero(first & second))
    return total


def normalised_shell(surface, shifts, boundary='toroidal'):
    """Normalise a shell count by its exact fixed-count random-labelling expectation."""
    population = surface.size
    occupied = int(np.count_nonzero(surface))
    if boundary == 'toroidal':
        edge_count = population * len(shifts)
    elif boundary == 'open':
        height, width = surface.shape
        edge_count = sum((width - abs(delta_column)) * (height - abs(delta_row))
                         for delta_column, delta_row in shifts)
    else:
        raise ValueError('unknown shell boundary convention')
    expected = edge_count * inclusion_probability(population, occupied, 2)
    if expected == 0:
        raise ValueError('at least two occupied sites are required')
    return shell_count(surface, shifts, boundary) / expected


def annulus_statistic(surface, r_min=1.0, r_max=3.0):
    """Compute the complete periodic annulus count with continuous-area normalisation."""
    height, width = surface.shape
    if not 0 <= r_min < r_max <= min(width, height) / 2:
        raise ValueError('annulus radii must fit inside half of the torus')
    occupied = int(np.count_nonzero(surface))
    if occupied < 2:
        raise ValueError('at least two occupied sites are required')
    limit = int(np.ceil(r_max))
    total = 0
    for delta_row in range(-limit, limit + 1):
        for delta_column in range(-limit, limit + 1):
            squared = delta_column ** 2 + delta_row ** 2
            if squared and r_min ** 2 <= squared < r_max ** 2:
                total += int(np.count_nonzero(surface & np.roll(
                    surface, (delta_row, delta_column), axis=(0, 1))))
    intensity = occupied / surface.size
    return total / (occupied * intensity * np.pi * (r_max ** 2 - r_min ** 2))


def box_partition(surface):
    """Return finite-scale box partition values and their least-squares slope."""
    height, width = surface.shape
    sizes, partition, retained = [], [], []
    for power in range(1, int(np.log2(min(width, height)))):
        size = 2 ** power
        rows, columns = height // size, width // size
        if min(rows, columns) < 2:
            continue
        cropped = surface[:rows * size, :columns * size]
        total = int(np.count_nonzero(cropped))
        if total == 0:
            continue
        masses = cropped.reshape(rows, size, columns, size).sum(axis=(1, 3))
        sizes.append(size)
        partition.append(float(np.sum((masses / total) ** 2)))
        retained.append(total)
    if len(sizes) < 2:
        raise ValueError('at least two nonempty box scales are required')
    slope = float(np.polyfit(np.log(sizes), np.log(partition), 1)[0])
    return {'sizes': sizes, 'partition': partition, 'retained': retained, 'slope': slope}


def pairwise_slope(surface, radii):
    """Compute the finite-range Grassberger-Procaccia pair-count slope."""
    rows, columns = np.nonzero(surface)
    points = np.column_stack((columns, rows)).astype(float)
    occupied = len(points)
    if occupied < 2:
        raise ValueError('at least two occupied sites are required')
    radii = np.asarray(radii, dtype=float)
    tree = spatial.cKDTree(points)
    counts = tree.count_neighbors(tree, radii) - occupied
    correlation = counts / (occupied * (occupied - 1))
    valid = correlation > 0
    if valid.sum() < 2:
        raise ValueError('at least two positive pair counts are required')
    slope = float(np.polyfit(np.log(radii[valid]), np.log(correlation[valid]), 1)[0])
    return {'radii': radii.tolist(), 'correlation': correlation.tolist(), 'slope': slope}


def inclusion_probability(population, occupied, order):
    """Inclusion probability for distinct labels under uniform fixed-count sampling."""
    if not 0 <= occupied <= population or not 0 <= order <= population:
        raise ValueError('invalid population, occupied count, or order')
    if occupied < order:
        return 0.0
    probability = 1.0
    for index in range(order):
        probability *= (occupied - index) / (population - index)
    return probability


def paired_graph_moments(first_edges, second_edges, population, occupied):
    """Exact variance of normalized edge-count differences for simple graphs."""
    if population < 4 or not 2 <= occupied <= population:
        raise ValueError('at least four labels and two occupied labels are required')
    first_edges = np.asarray(first_edges, dtype=np.int64)
    second_edges = np.asarray(second_edges, dtype=np.int64)
    if len(first_edges) == 0 or len(second_edges) == 0:
        raise ValueError('both graphs must contain at least one edge')
    degrees = []
    for edges in (first_edges, second_edges):
        first = edges // (population + 1)
        second = edges % (population + 1)
        if (np.any(first < 1) or np.any(second > population) or np.any(first >= second) or
                len(np.unique(edges)) != len(edges)):
            raise ValueError('graphs must contain distinct undirected non-self edges')
        degrees.append(np.bincount(np.concatenate((first, second)), minlength=population + 1)[1:])
    first_count, second_count = len(first_edges), len(second_edges)
    shared = len(np.intersect1d(first_edges, second_edges, assume_unique=True))
    edge_energy = 1 / first_count + 1 / second_count - 2 * shared / (first_count * second_count)
    degree_difference = degrees[0] / first_count - degrees[1] / second_count
    degree_energy = float(np.dot(degree_difference, degree_difference))
    pair_probability = inclusion_probability(population, occupied, 2)
    triple_probability = inclusion_probability(population, occupied, 3)
    disjoint_complement = (pair_probability * (population - occupied) *
                           (population - occupied - 1) / ((population - 2) * (population - 3)))
    single_complement = triple_probability * (population - occupied) / (population - 3)
    variance = (edge_energy * disjoint_complement + degree_energy * single_complement) / pair_probability ** 2
    return {
        'first_edges': first_count, 'second_edges': second_count,
        'shared_edges': shared, 'edge_energy': edge_energy, 'degree_energy': degree_energy,
        'expected_first': first_count * pair_probability,
        'expected_second': second_count * pair_probability,
        'variance_normalised_difference': variance,
    }


def paired_shell_moments(first_grid, second_grid, shifts, occupied):
    """Exact null moments for the difference between two regular shell counts."""
    if first_grid.shape != second_grid.shape or first_grid.size < 4:
        raise ValueError('embeddings must have the same window with at least four sites')
    population = first_grid.size
    if not 2 <= occupied <= population:
        raise ValueError('at least two occupied sites are required')
    first_edges = shell_edges(first_grid, shifts)
    second_edges = shell_edges(second_grid, shifts)
    if len(first_edges) != len(second_edges):
        raise ValueError('both shell graphs must have the same number of edges')
    edge_count = len(first_edges)
    shared_edges = len(np.intersect1d(first_edges, second_edges, assume_unique=True))
    pair_probability = inclusion_probability(population, occupied, 2)
    other_pair_vacant = ((population - occupied) * (population - occupied - 1) /
                         ((population - 2) * (population - 3)))
    variance = 2 * (edge_count - shared_edges) * pair_probability * other_pair_vacant
    expected_count = edge_count * pair_probability
    return {
        'edges': edge_count,
        'shared_edges': shared_edges,
        'expected_count': expected_count,
        'variance_count_difference': variance,
        'variance_normalised_difference': variance / expected_count ** 2,
    }


def summarise(values):
    """Summarise an ensemble without implying a confidence interval."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError('at least two finite ensemble values are required')
    return {'n': len(values), 'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)),
            'q025': float(np.quantile(values, 0.025)),
            'q05': float(np.quantile(values, 0.05)),
            'q95': float(np.quantile(values, 0.95)),
            'q975': float(np.quantile(values, 0.975))}


def random_labels(generator, population, occupied):
    """Draw one uniformly distributed fixed-count integer-label pattern."""
    labels = np.zeros(population + 1, dtype=bool)
    labels[generator.choice(population, size=occupied, replace=False) + 1] = True
    return labels


def seed_generator(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def calculate_pattern(surface, width, resamples, generator):
    """Compute all reported descriptors for one observed pattern."""
    occupied = int(np.count_nonzero(surface))
    distances = {boundary: nearest_neighbour_distances(surface, boundary)
                 for boundary in ('open', 'toroidal', 'interior')}
    fits = {boundary: fit_brody_distances(values, resamples=resamples, rng=generator)
            for boundary, values in distances.items()}
    radii, _, ripley, maximum, at_nn = rip_k_border(
        coords(surface) + 0.5, width, width, width / 3.0, 18)
    shells = {name: normalised_shell(surface, shifts) for name, shifts in SHELLS.items()}
    shell_counts = {name: shell_count(surface, shifts) for name, shifts in SHELLS.items()}
    direction_shifts = {'h2': ((2, 0),), 'v2': ((0, 2),),
                        'h4': ((4, 0),), 'v4': ((0, 4),)}
    return {
        'occupied': occupied,
        'fits': fits,
        'brody_diagnostics': brody_fit_diagnostics(distances['open'], fits['open']['beta']),
        'nni': float(2 * np.sqrt(occupied / surface.size) * distances['open'].mean()),
        'ripley': {'radii': radii.tolist(), 'l_minus_r': ripley.tolist(),
                   'max_abs': maximum, 'at_mean_nn': at_nn},
        'annulus': annulus_statistic(surface),
        'box': box_partition(surface),
        'gp': pairwise_slope(surface, np.geomspace(2, width / 3.0, 12)),
        'shells': shells,
        'shell_counts': shell_counts,
        'direction': {name: normalised_shell(surface, shifts)
                      for name, shifts in direction_shifts.items()},
        'direction_counts': {name: shell_count(surface, shifts)
                             for name, shifts in direction_shifts.items()},
    }


def open_shell_validation(grids, selected_labels, occupied, seed, realisations):
    """Check irregular open-shell contrasts against paired random labelling."""
    names = ('row_major', 'ulam', 'cantor')
    pairs = ((0, 1), (0, 2), (1, 2))
    population = selected_labels.size - 1
    shifts = SHELLS['rsqrt2']
    graphs = {name: shell_edges(grids[name], shifts, 'open') for name in names}
    moments = {f'{names[first]}__{names[second]}': paired_graph_moments(
        graphs[names[first]], graphs[names[second]], population, occupied)
               for first, second in pairs}
    observed = {name: normalised_shell(selected_labels[grids[name]], shifts, 'open')
                for name in names}
    rng = seed_generator(seed, 2)
    values = []
    for _ in range(realisations):
        random_pattern = random_labels(rng, population, occupied)
        values.append([normalised_shell(random_pattern[grids[name]], shifts, 'open')
                       for name in names])
    values = np.asarray(values)
    contrasts = {}
    for first, second in pairs:
        key = f'{names[first]}__{names[second]}'
        contrasts[key] = {
            'observed_difference': float(observed[names[first]] - observed[names[second]]),
            'simulation': summarise(values[:, first] - values[:, second]),
            'exact': moments[key],
        }
    return {'shell': 'rsqrt2', 'realisations': realisations, 'observed': observed,
            'contrasts': contrasts, 'paired_values': values.tolist()}


def run_analysis(args):
    """Generate every numerical result used by the prime-pattern manuscript."""
    started = time.monotonic()
    width = 316
    population = width * width
    primes = primes_upto(population)
    occupied = len(primes)
    density = occupied / population
    names = ('row_major', 'ulam', 'cantor')
    grids = embedding_value_grids(width)
    labels = np.zeros(population + 1, dtype=bool)
    labels[primes] = True
    surfaces = {name: labels[grid] for name, grid in grids.items()}
    if any(int(surface.sum()) != occupied for surface in surfaces.values()):
        raise RuntimeError('embedding occupancy does not match the prime population')

    config = {
        'width': width, 'population': population, 'primes': occupied, 'density': density,
        'seed': args.seed, 'fixed_realisations': args.paired,
        'inhomogeneous_realisations': args.inhomogeneous,
        'descriptor_realisations': args.descriptors,
        'shell_realisations': args.shells, 'boundary_realisations': args.boundaries,
        'thinning_realisations': args.thinning, 'control_realisations': args.controls,
        'large_window_realisations': args.large_windows, 'resamples': args.resamples,
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
    }
    output = {'config': config, 'patterns': {}}
    fit_rng = seed_generator(args.seed, 1)
    for name in names:
        output['patterns'][name] = calculate_pattern(surfaces[name], width, args.resamples, fit_rng)
        print(f"Observed {name}: beta={output['patterns'][name]['fits']['open']['beta']:.6f}", flush=True)

    shell_moments = {}
    for first, second in ((names[0], names[1]), (names[0], names[2]), (names[1], names[2])):
        key = f'{first}__{second}'
        shell_moments[key] = {
            shell: paired_shell_moments(grids[first], grids[second], shifts, occupied)
            for shell, shifts in SHELLS.items()}
    output['shell_moments'] = shell_moments

    rng = seed_generator(args.seed, 2)
    beta_values, nni_values, box_values, box_curves = [], [], [], []
    toroidal_values, interior_values, gp_values, ripley_values = [], [], [], []
    ripley_at_nn, ripley_max, annulus_values = [], [], []
    shell_values = {name: [] for name in SHELLS}
    direction_shifts = {'h2': ((2, 0),), 'v2': ((0, 2),),
                        'h4': ((4, 0),), 'v4': ((0, 4),)}
    direction_values = {name: [] for name in direction_shifts}
    paired_shell_values = []
    for iteration in range(args.paired):
        random_pattern = random_labels(rng, population, occupied)
        sample_surfaces = [random_pattern[grids[name]] for name in names]
        sample_distances = [nearest_neighbour_distances(surface) for surface in sample_surfaces]
        beta_values.append([fit_brody_distances(values)['beta'] for values in sample_distances])
        first_surface = sample_surfaces[0]
        nni_values.append(float(2 * np.sqrt(density) * sample_distances[0].mean()))
        if iteration < args.shells:
            for name, shifts in SHELLS.items():
                shell_values[name].append(normalised_shell(first_surface, shifts))
            for name, shifts in direction_shifts.items():
                direction_values[name].append(normalised_shell(first_surface, shifts))
            paired_shell_values.append([normalised_shell(surface, SHELLS['rsqrt2'])
                                        for surface in sample_surfaces])
        if iteration < args.boundaries:
            toroidal_values.append(fit_brody_distances(
                nearest_neighbour_distances(first_surface, 'toroidal'))['beta'])
            interior_values.append(fit_brody_distances(
                nearest_neighbour_distances(first_surface, 'interior'))['beta'])
        if iteration < args.descriptors:
            box = box_partition(first_surface)
            box_values.append(box['slope'])
            box_curves.append(box['partition'])
            gp_values.append(pairwise_slope(first_surface, np.geomspace(2, width / 3.0, 12))['slope'])
            _, _, values, maximum, at_nn = rip_k_border(
                coords(first_surface) + 0.5, width, width, width / 3.0, 18)
            ripley_values.append(values.tolist())
            ripley_at_nn.append(at_nn)
            ripley_max.append(maximum)
            annulus_values.append(annulus_statistic(first_surface))
        if (iteration + 1) % 500 == 0:
            print(f'Fixed-count paired null: {iteration + 1}/{args.paired}', flush=True)

    beta_values = np.asarray(beta_values)
    observed_betas = np.array([output['patterns'][name]['fits']['open']['beta'] for name in names])
    pairs = ((0, 1), (0, 2), (1, 2))
    differences = np.column_stack([beta_values[:, first] - beta_values[:, second]
                                   for first, second in pairs])
    observed_differences = np.array([observed_betas[first] - observed_betas[second]
                                     for first, second in pairs])
    max_differences = np.max(np.abs(differences), axis=1)
    pair_tests = {}
    for index, (first, second) in enumerate(pairs):
        key = f'{names[first]}__{names[second]}'
        threshold = abs(observed_differences[index])
        pair_tests[key] = {
            'observed_difference': float(observed_differences[index]),
            'null_difference': summarise(differences[:, index]),
            'exceedances': int(np.count_nonzero(np.abs(differences[:, index]) >= threshold)),
            'p_value': float((1 + np.count_nonzero(np.abs(differences[:, index]) >= threshold)) /
                             (args.paired + 1)),
            'max_exceedances': int(np.count_nonzero(max_differences >= threshold)),
            'max_adjusted_p_value': float((1 + np.count_nonzero(max_differences >= threshold)) /
                                          (args.paired + 1)),
        }
    global_threshold = np.max(np.abs(observed_differences))
    fixed_null = {
        'beta_values': beta_values.tolist(),
        'beta': {name: summarise(beta_values[:, index]) for index, name in enumerate(names)},
        'pair_tests': pair_tests,
        'global_max_p_value': float((1 + np.count_nonzero(max_differences >= global_threshold)) /
                                    (args.paired + 1)),
        'nni': summarise(nni_values),
        'toroidal_beta': summarise(toroidal_values),
        'interior_beta': summarise(interior_values),
        'box': summarise(box_values),
        'box_curve_mean': np.mean(box_curves, axis=0).tolist(),
        'gp': summarise(gp_values),
        'ripley_at_nn': summarise(ripley_at_nn),
        'ripley_max_abs': summarise(ripley_max),
        'ripley_curves': ripley_values,
        'annulus': summarise(annulus_values),
        'shells': {name: summarise(values) for name, values in shell_values.items()},
        'direction': {name: summarise(values) for name, values in direction_values.items()},
        'paired_sqrt2_shell_values': paired_shell_values,
    }
    output['fixed_null'] = fixed_null
    print('Fixed-count paired null completed', flush=True)
    output['open_shell_validation'] = open_shell_validation(
        grids, labels, occupied, args.seed, args.shells)
    print('Open-boundary graph variance validation completed', flush=True)

    rng = seed_generator(args.seed, 3)
    probabilities, calibration = inhomogeneous_probabilities(width, occupied)
    inhomogeneous_betas, inhomogeneous_counts = [], []
    for iteration in range(args.inhomogeneous):
        random_pattern = np.zeros(population + 1, dtype=bool)
        random_pattern[1:] = rng.random(population) < probabilities
        inhomogeneous_counts.append(int(random_pattern.sum()))
        inhomogeneous_betas.append([
            fit_brody_distances(nearest_neighbour_distances(random_pattern[grids[name]]))['beta']
            for name in names])
        if (iteration + 1) % 500 == 0:
            print(f'Inhomogeneous paired null: {iteration + 1}/{args.inhomogeneous}', flush=True)
    inhomogeneous_betas = np.asarray(inhomogeneous_betas)
    output['inhomogeneous_null'] = {
        'calibration': calibration, 'expected_count': float(probabilities.sum()),
        'count': summarise(inhomogeneous_counts),
        'beta': {name: summarise(inhomogeneous_betas[:, index])
                 for index, name in enumerate(names)},
        'beta_values': inhomogeneous_betas.tolist(),
        'differences': {f'{names[first]}__{names[second]}': summarise(
            inhomogeneous_betas[:, first] - inhomogeneous_betas[:, second])
                        for first, second in pairs},
    }

    rng = seed_generator(args.seed, 4)
    thinning = []
    for target_density in (0.01, 0.02, 0.032, 0.05, density):
        keep = min(1.0, target_density / density)
        prime_betas = []
        repetitions = 1 if keep == 1 else args.thinning
        for _ in range(repetitions):
            selected = primes[rng.random(occupied) < keep]
            field = np.zeros(population, dtype=bool)
            field[selected - 1] = True
            prime_betas.append(fit_brody_distances(
                nearest_neighbour_distances(field.reshape(width, width)))['beta'])
        null_betas = []
        for _ in range(args.controls):
            field = rng.random((width, width)) < target_density
            null_betas.append(fit_brody_distances(nearest_neighbour_distances(field))['beta'])
        null_summary = summarise(null_betas)
        thinning.append({
            'density': float(target_density), 'keep_probability': keep,
            'prime_values': prime_betas,
            'prime_mean': float(np.mean(prime_betas)),
            'prime_sd': float(np.std(prime_betas, ddof=1)) if len(prime_betas) > 1 else None,
            'null': null_summary,
            'above_null_q95': int(np.count_nonzero(np.asarray(prime_betas) > null_summary['q95'])),
        })
        print(f'Thinning density {target_density:.5f} completed', flush=True)
    output['thinning'] = thinning

    rng = seed_generator(args.seed, 5)
    primality = np.zeros(population + 1, dtype=bool)
    primality[primes] = True
    removals = {}
    for gap in (2, 4):
        eligible = primes[primes > gap]
        removals[gap] = eligible[primality[eligible - gap]]
    ablations = {}
    for name, removed in (('original', np.array([], dtype=int)),
                          ('twin_larger', removals[2]), ('cousin_larger', removals[4]),
                          ('both_larger', np.union1d(removals[2], removals[4]))):
        selected = np.setdiff1d(primes, removed, assume_unique=True)
        field = np.zeros(population, dtype=bool)
        field[selected - 1] = True
        estimate = fit_brody_distances(nearest_neighbour_distances(field.reshape(width, width)))['beta']
        if name == 'original':
            null_betas = beta_values[:args.controls, 0]
        else:
            null_betas = []
            for _ in range(args.controls):
                sample = random_labels(rng, population, len(selected))
                null_betas.append(fit_brody_distances(
                    nearest_neighbour_distances(sample[1:].reshape(width, width)))['beta'])
        ablations[name] = {'remaining': len(selected), 'removed': len(removed),
                           'beta': estimate, 'null': summarise(null_betas)}
        print(f'Ablation {name} completed', flush=True)
    output['ablation'] = ablations

    rng = seed_generator(args.seed, 6)
    finite = []
    for side in (31, 54, 100, 173, 316, 547, 1000):
        area = side * side
        local_primes = primes_upto(area)
        field = np.zeros(area, dtype=bool)
        field[local_primes - 1] = True
        beta = fit_brody_distances(nearest_neighbour_distances(field.reshape(side, side)))['beta']
        repetitions = args.controls if side <= 316 else args.large_windows
        if side == width:
            null_betas = beta_values[:repetitions, 0]
        else:
            null_betas = []
            for _ in range(repetitions):
                sample = random_labels(rng, area, len(local_primes))
                null_betas.append(fit_brody_distances(
                    nearest_neighbour_distances(sample[1:].reshape(side, side)))['beta'])
        null_summary = summarise(null_betas)
        finite.append({'width': side, 'population': area, 'occupied': len(local_primes),
                       'beta': beta, 'null': null_summary, 'difference': beta - null_summary['mean']})
        print(f'Finite window W={side} completed', flush=True)
    output['finite_size'] = finite

    destination = Path(args.output)
    with destination.open('w', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Wrote {destination} in {(time.monotonic() - started) / 60:.2f} minutes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--paired', type=int, default=4999)
    parser.add_argument('--inhomogeneous', type=int, default=1000)
    parser.add_argument('--descriptors', type=int, default=200)
    parser.add_argument('--shells', type=int, default=2000)
    parser.add_argument('--boundaries', type=int, default=1000)
    parser.add_argument('--thinning', type=int, default=500)
    parser.add_argument('--controls', type=int, default=1000)
    parser.add_argument('--large-windows', type=int, default=200)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--output', default=str(Path(__file__).with_name('prime_spatial_results.json')))
    args = parser.parse_args()
    values = (args.paired, args.inhomogeneous, args.descriptors, args.shells, args.boundaries,
              args.thinning, args.controls, args.large_windows, args.resamples)
    if min(values) < 2 or max(args.descriptors, args.shells, args.boundaries, args.controls) > args.paired:
        parser.error('all counts must be at least two; paired count must cover all fixed-null subsamples')
    run_analysis(args)


if __name__ == '__main__':
    main()
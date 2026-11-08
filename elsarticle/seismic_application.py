#!/usr/bin/env python3
"""Second application: seismic-event cells under alternative adjacency graphs.

The same finite population of grid cells over the California region is labelled
by whether it contains at least one earthquake epicentre (USGS ANSS catalogue,
M >= 2.5, 2000-01-01 to 2026-01-01). Two alternative neighbourhood graphs on
this one population are compared: rook adjacency (edge-sharing cells) and queen
adjacency (rook plus diagonal cells). The queen graph strictly contains the
rook graph, so the paired graph-contrast variance of the manuscript's Eq. (5)
is evaluated for nested, irregular, open-window graphs, and the observed
normalised count contrast is tested against the fixed-count random-labelling
null with paired Monte Carlo realisations.

Input: seismic_events_ca.csv.gz (cached USGS FDSN CSV export). If the cached
file is missing, the script fetches the catalogue in yearly windows from
https://earthquake.usgs.gov/fdsnws/event/1/query and caches it.
"""

import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prime_spatial_analysis import (inclusion_probability, paired_graph_moments,
                                    random_labels, seed_generator)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'seismic_events_ca.csv.gz'
BBOX = {'minlatitude': 32, 'maxlatitude': 42,
        'minlongitude': -124.5, 'maxlongitude': -114}
START = '2000-01-01'
END = '2026-01-01'
MIN_MAG = 2.5
URL = 'https://earthquake.usgs.gov/fdsnws/event/1/query'


def fetch_catalogue():
    """Download the catalogue in yearly windows and cache it (gzip CSV)."""
    rows = []
    for year in range(2000, 2026):
        params = urllib.parse.urlencode({
            'format': 'csv',
            'starttime': f'{year}-01-01',
            'endtime': f'{year + 1}-01-01',
            'minlatitude': BBOX['minlatitude'], 'maxlatitude': BBOX['maxlatitude'],
            'minlongitude': BBOX['minlongitude'], 'maxlongitude': BBOX['maxlongitude'],
            'minmagnitude': MIN_MAG,
            'eventtype': 'earthquake',
            'orderby': 'time',
        })
        request = urllib.request.Request(f'{URL}?{params}', headers={'User-Agent': 'SeismicApplication/1.0'})
        text = urllib.request.urlopen(request, timeout=120).read().decode('utf-8')
        rows.append(text)
        time.sleep(1)
    lines = []
    for index, block in enumerate(rows):
        for line in block.splitlines():
            if index == 0 or not line.startswith('time,'):
                lines.append(line)
    with gzip.open(CACHE, 'wt', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return lines


def load_catalogue():
    """Load (time, latitude, longitude) rows from the cached CSV."""
    if not CACHE.exists():
        fetch_catalogue()
    with gzip.open(CACHE, 'rt', encoding='utf-8') as stream:
        lines = stream.read().splitlines()
    latitudes, longitudes = [], []
    for line in lines[1:]:
        fields = line.split(',')
        try:
            longitudes.append(float(fields[2]))
            latitudes.append(float(fields[1]))
        except (ValueError, IndexError):
            continue
    return np.asarray(latitudes), np.asarray(longitudes)


def build_grid(width, latitudes, longitudes):
    """Rasterise epicentres onto a square width-by-width cell grid."""
    lon_min, lon_max = BBOX['minlongitude'], BBOX['maxlongitude']
    lat_min, lat_max = BBOX['minlatitude'], BBOX['maxlatitude']
    columns = np.minimum(np.floor((longitudes - lon_min) / (lon_max - lon_min) * width).astype(int),
                         width - 1)
    rows = np.minimum(np.floor((latitudes - lat_min) / (lat_max - lat_min) * width).astype(int),
                      width - 1)
    occupied = np.zeros((width, width), dtype=bool)
    occupied[rows, columns] = True
    return occupied


def build_neighbourhood_graphs(width):
    """Return encoded edge arrays for the open-window rook and queen graphs.

    Rook edges join cells sharing a side; queen edges additionally join cells
    touching at a corner (both diagonal orientations), so the queen graph is
    the rook graph plus the two diagonal-neighbour graphs. Each undirected
    pair is generated exactly once by construction (the four offsets are
    pairwise non-inverse, so no pair occurs twice).

    Labels are cell indices 1..M with cell (column, row) -> column + row * width + 1.
    Edges are encoded as u * (M + 1) + v with u < v.
    """
    population = width * width
    rook, queen = [], []
    for row in range(width):
        for column in range(width):
            label = column + row * width + 1
            for delta_column, delta_row in ((1, 0), (0, 1), (1, 1), (1, -1)):
                other_column, other_row = column + delta_column, row + delta_row
                if not (0 <= other_column < width and 0 <= other_row < width):
                    continue
                other = other_column + other_row * width + 1
                queen.append(min(label, other) * (population + 1) + max(label, other))
                if delta_column == 0 or delta_row == 0:
                    rook.append(min(label, other) * (population + 1) + max(label, other))
    return np.asarray(rook, dtype=np.int64), np.asarray(queen, dtype=np.int64)


def count_occupied(edges, labels):
    """Count occupied undirected edges in one boolean label pattern.

    Labels have size population + 1; an edge is encoded as u * (population + 1) + v.
    """
    return int(np.count_nonzero(labels[edges // labels.size] &
                                labels[edges % labels.size]))


def run(args):
    started = time.monotonic()
    latitudes, longitudes = load_catalogue()
    width = args.width
    population = width * width
    grid = build_grid(width, latitudes, longitudes)
    labels = np.zeros(population + 1, dtype=bool)
    labels[np.flatnonzero(grid.ravel()) + 1] = True
    occupied = int(labels.sum())
    rook_edges, queen_edges = build_neighbourhood_graphs(width)
    rook_count = count_occupied(rook_edges, labels)
    queen_count = count_occupied(queen_edges, labels)
    moments = paired_graph_moments(rook_edges, queen_edges, population, occupied)
    probability_two = inclusion_probability(population, occupied, 2)
    probability_three = inclusion_probability(population, occupied, 3)
    probability_four = inclusion_probability(population, occupied, 4)
    edge_discrepancy = moments['edge_energy']
    degree_discrepancy = moments['degree_energy']
    variance = moments['variance_normalised_difference']
    rook_normalised = rook_count / (len(rook_edges) * probability_two)
    queen_normalised = queen_count / (len(queen_edges) * probability_two)
    observed = rook_normalised - queen_normalised
    observed_z = observed / np.sqrt(variance)
    edge_term = edge_discrepancy * (probability_two - 2 * probability_three + probability_four)
    degree_term = degree_discrepancy * (probability_three - probability_four)
    variance_check = (edge_term + degree_term) / probability_two ** 2

    rng = seed_generator(args.seed, 70)
    contrasts = []
    for _ in range(args.null):
        pattern = random_labels(rng, population, occupied)
        first = count_occupied(rook_edges, pattern) / (len(rook_edges) * probability_two)
        second = count_occupied(queen_edges, pattern) / (len(queen_edges) * probability_two)
        contrasts.append(first - second)
    contrasts = np.asarray(contrasts)
    exceedances = int(np.count_nonzero(np.abs(contrasts) >= abs(observed)))
    monte_carlo_value = (exceedances + 1) / (args.null + 1)

    output = {
        'config': {'width': width, 'population': population, 'occupied': occupied,
                   'events': int(latitudes.size), 'seed': args.seed,
                   'null_realisations': args.null,
                   'catalogue': {'source': URL, 'start': START, 'end': END,
                                 'min_magnitude': MIN_MAG, 'bbox': BBOX,
                                 'cached_file': CACHE.name, 'accessed': '2026-11-04'}},
        'graphs': {'rook_edges': int(len(rook_edges)), 'queen_edges': int(len(queen_edges)),
                   'shared_edges': int(moments['shared_edges']),
                   'nested': bool(moments['shared_edges'] == len(rook_edges))},
        'counts': {'rook': rook_count, 'queen': queen_count,
                   'rook_normalised': rook_normalised, 'queen_normalised': queen_normalised,
                   'observed_difference': observed, 'observed_z': observed_z},
        'variance': {'exact': variance, 'recomputed_from_terms': variance_check,
                     'edge_discrepancy_S_e': edge_discrepancy,
                     'degree_discrepancy_S_v': degree_discrepancy,
                     'edge_term_share': edge_term / (edge_term + degree_term),
                     'degree_term_share': degree_term / (edge_term + degree_term),
                     'sigma': float(np.sqrt(variance))},
        'null': {'mean': float(np.mean(contrasts)), 'sd': float(np.std(contrasts, ddof=1)),
                 'monte_carlo_value': monte_carlo_value, 'exceedances': exceedances,
                 'contrasts': contrasts.tolist()},
    }
    with (ROOT / args.output).open('w', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    map_axis = axes[0]
    longitude_edges = np.linspace(BBOX['minlongitude'], BBOX['maxlongitude'], width + 1)
    latitude_edges = np.linspace(BBOX['minlatitude'], BBOX['maxlatitude'], width + 1)
    map_axis.pcolormesh(longitude_edges, latitude_edges, grid, cmap='Blues',
                        edgecolors='none', shading='flat')
    map_axis.set_xlim(BBOX['minlongitude'], BBOX['maxlongitude'])
    map_axis.set_ylim(BBOX['minlatitude'], BBOX['maxlatitude'])
    map_axis.set_xlabel('Longitude (deg)')
    map_axis.set_ylabel('Latitude (deg)')
    map_axis.set_title('(a) Occupied cells')
    histogram_axis = axes[1]
    sigma = float(np.sqrt(variance))
    span = max(5 * sigma, abs(observed) + sigma)
    grid_points = np.linspace(-span, span, 200)
    density = np.exp(-0.5 * (grid_points / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    histogram_axis.hist(contrasts, bins=60, density=True, color='0.82',
                        label=f'{args.null} paired null realisations')
    histogram_axis.plot(grid_points, density, color='#24658a',
                        label=f'Exact null density ($\\sigma={sigma:.4f}$)')
    histogram_axis.axvline(observed, color='#b25c38', linewidth=1.5,
                           label=f'Observed {observed:+.4f}')
    histogram_axis.set_xlabel('Normalised count difference (rook $-$ queen)')
    histogram_axis.set_ylabel('Density')
    histogram_axis.set_title('(b) Paired null contrasts')
    histogram_axis.legend(frameon=False, fontsize=8)
    figure.savefig(ROOT / args.figure, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(json.dumps({key: output[key] for key in ('graphs', 'counts', 'variance', 'null')},
                     indent=2), flush=True)
    print(f'Wrote {args.output} and {args.figure} in {(time.monotonic() - started) / 60:.2f} minutes',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--null', type=int, default=1999)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--output', default='seismic_application.json')
    parser.add_argument('--figure', default='fig_seismic_application.pdf')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

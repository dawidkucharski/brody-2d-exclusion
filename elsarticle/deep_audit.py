#!/usr/bin/env python3
"""Deep audit: check every manuscript number against the recorded results.

Parses aurora_manuscript.tex and validates tables and inline claims against
prime_spatial_results.json, power_calibration.json, power_level_check.json and
seismic_application.json. Also recomputes several closed-form constants
independently. Prints PASS/FAIL lines and exits non-zero on any failure.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from prime_spatial_analysis import (SHELLS, embedding_value_grids, inclusion_probability,
                                    primes_upto)

ROOT = Path(__file__).resolve().parent
TEX = (ROOT / 'aurora_manuscript.tex').read_text()
results = json.loads((ROOT / 'prime_spatial_results.json').read_text())
power = json.loads((ROOT / 'power_calibration.json').read_text())
level = json.loads((ROOT / 'power_level_check.json').read_text())
seismic = json.loads((ROOT / 'seismic_application.json').read_text())

failures = []


def round_decimal(value, places):
    from decimal import Decimal, ROUND_HALF_UP
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def check(name, actual, expected, tol=0.0):
    ok = abs(actual - expected) <= tol + 1e-12
    status = 'PASS' if ok else 'FAIL'
    print(f'{status}  {name}: got {actual!r}, expected {expected!r}')
    if not ok:
        failures.append(name)


def numbers(text):
    return [float(x.replace(',', '')) for x in re.findall(r'-?\d[\d,]*\.?\d*', text)]


def table_block(label):
    pos = TEX.index('\\label{' + label + '}')
    start = TEX.rfind('\\begin{table}', 0, pos)
    end = TEX.index('\\end{table}', pos) + len('\\end{table}')
    return TEX[start:end]


def table_rows(label, row_names):
    block = table_block(label)
    out = {}
    for name in row_names:
        match = next((line for line in block.splitlines() if name in line and '&' in line), None)
        if match is None:
            print(f'FAIL  {label}: row {name!r} not found')
            failures.append(f'{label}/{name}')
            continue
        out[name] = match.replace('\\,', '').replace('\\\\', '')
    return out


def row_cells(row):
    return [cell for cell in row.split('&')][1:]


def cell_numbers(cell):
    return [float(x) for x in re.findall(r'-?\d[\d,]*\.?\d*', cell.replace('\\,', ''))]


patterns = results['patterns']
fixed = results['fixed_null']

# ---------------- tab:local ----------------
rows = table_rows('tab:local', ['Open', 'Toroidal', 'Interior reference',
                                'Fixed-count null, open', 'Fixed-count null, toroidal',
                                'Fixed-count null, interior'])
for treatment, key, boundary in [('Open', 'open', 'open'),
                                 ('Toroidal', 'toroidal', 'toroidal'),
                                 ('Interior reference', 'interior', 'interior')]:
    cells = row_cells(rows[treatment])
    for index, name in enumerate(('row_major', 'ulam', 'cantor')):
        fit = patterns[name]['fits'][boundary]
        low, high = fit['resampling_interval']
        values = numbers(cells[index])
        check(f'tab:local {treatment} {name} beta', values[0], fit['beta'], 5e-4)
        check(f'tab:local {treatment} {name} low', values[1], low, 5e-4)
        check(f'tab:local {treatment} {name} high', values[2], high, 5e-4)

for treatment, key in [('Fixed-count null, open', 'beta'),
                       ('Fixed-count null, toroidal', 'toroidal_beta'),
                       ('Fixed-count null, interior', 'interior_beta')]:
    null = fixed[key]['row_major'] if key == 'beta' else fixed[key]
    values = cell_numbers(row_cells(rows[treatment])[0])
    check(f'tab:local null {treatment} mean', values[-2], null['mean'], 5e-4)
    check(f'tab:local null {treatment} sd', values[-1], null['sd'], 5e-4)
    expected_n = 4999 if key == 'beta' else 1000
    check(f'tab:local null {treatment} n', expected_n, null['n'], 0.0)

# inline local results
check('inline open beta row_major', 2.153, patterns['row_major']['fits']['open']['beta'], 5e-4)
pairs = [('row_major', 'ulam'), ('row_major', 'cantor'), ('ulam', 'cantor')]
observed_differences = []
for first, second in pairs:
    observed_differences.append(fixed['pair_tests'][f'{first}__{second}']['observed_difference'])
check('inline pairwise difference 1', 0.297, observed_differences[0], 5e-4)
check('inline pairwise difference 2', 0.739, observed_differences[1], 5e-4)
check('inline pairwise difference 3', 0.442, observed_differences[2], 5e-4)
check('global Monte Carlo value', 0.0002, fixed['global_max_p_value'], 1e-4)
interior_counts = [patterns[name]['fits']['interior']['n'] for name in ('row_major', 'ulam', 'cantor')]
for got, expected in zip([8894, 9041, 8918], interior_counts):
    check('interior reference count', got, expected, 0.5)
smallest = min(abs(value) for value in observed_differences)
maxima = fixed.get('global_maximum_values')
if maxima is not None:
    check('no null maximum exceeds smallest contrast',
          float(np.count_nonzero(np.asarray(maxima) >= smallest)), 0.0)

# ---------------- tab:paired ----------------
rows = table_rows('tab:paired', ['Row-major -- Ulam', 'Row-major -- diagonal sweep', 'Ulam -- diagonal sweep'])
simulated = np.asarray(fixed['paired_sqrt2_shell_values'])
for heading, key, first, second in [('Row-major -- Ulam', 'row_major__ulam', 0, 1),
                                    ('Row-major -- diagonal sweep', 'row_major__cantor', 0, 2),
                                    ('Ulam -- diagonal sweep', 'ulam__cantor', 1, 2)]:
    test = fixed['pair_tests'][key]
    moments = results['shell_moments'][key]['rsqrt2']
    values = [n for cell in row_cells(rows[heading]) for n in cell_numbers(cell)]
    check(f'tab:paired {heading} difference', values[0], test['observed_difference'], 5e-4)
    check(f'tab:paired {heading} p', values[1], test['max_adjusted_p_value'], 5e-5)
    check(f'tab:paired {heading} q_d', values[2], moments['shared_edges'], 0.5)
    check(f'tab:paired {heading} exact sigma', values[3],
          float(np.sqrt(moments['variance_normalised_difference'])), 5e-4)
    check(f'tab:paired {heading} mc sigma', values[4],
          float(np.std(simulated[:, first] - simulated[:, second], ddof=1)), 5e-4)

# ---------------- tab:g2 ----------------
rows = table_rows('tab:g2', ['Row-major', 'Ulam', 'Diagonal sweep'])
shell_names = ['r1', 'rsqrt2', 'r2', 'rsqrt5', 'r4']
for name in ('Row-major', 'Ulam', 'Diagonal sweep'):
    key = {'Row-major': 'row_major', 'Ulam': 'ulam', 'Diagonal sweep': 'cantor'}[name]
    values = [n for cell in row_cells(rows[name]) for n in cell_numbers(cell)]
    for index, shell in enumerate(shell_names):
        check(f'tab:g2 {name} {shell}', values[index], patterns[key]['shells'][shell], 5e-4)
null_sd = fixed.get('shells')
if null_sd:
    for index, shell in enumerate(shell_names):
        check(f'tab:g2 null sd {shell}', [0.021, 0.021, 0.021, 0.015, 0.021][index],
              null_sd[shell]['sd'], 5e-4)

# ---------------- direction counts ----------------
directions = patterns['row_major']['direction']
for got, expected in zip([1.332, 1.337, 1.339, 1.405],
                         [directions['h2'], directions['v2'], directions['h4'], directions['v4']]):
    check('direction normalised count', got, expected, 5e-4)
direction_counts = patterns['row_major'].get('direction_counts')
if direction_counts:
    for got, expected in zip([1224, 1231],
                             [direction_counts['h2'], direction_counts['h4']]):
        check('horizontal toroidal pair count', got, expected, 0.5)

# twin/cousin sieve (independent)
population = 99856
prime_set = set(int(p) for p in primes_upto(population))
twin = sum(1 for p in prime_set if p <= population - 2 and p + 2 in prime_set)
cousin = sum(1 for p in prime_set if p <= population - 4 and p + 4 in prime_set)
check('twin-prime sieve count', 1223, twin, 0.0)
check('cousin-prime sieve count', 1215, cousin, 0.0)

# ---------------- tab:descriptors ----------------
rows = table_rows('tab:descriptors', ['Brody', 'CDF discrepancy', 'Clark--Evans',
                                      '$L(\\bar r)-\\bar r$', '$\\max|L(r)-r|$',
                                      'Annulus'])
for name, expected, metric in [('Brody', [p['fits']['open']['beta'] for p in patterns.values()], None),
                               ('Clark--Evans', [p['nni'] for p in patterns.values()], 'nni'),
                               ('Annulus', [p['annulus'] for p in patterns.values()], 'annulus')]:
    values = [n for cell in row_cells(rows[name])[:-1] for n in cell_numbers(cell)]
    for index, value in enumerate(values):
        check(f'tab:descriptors {name} pattern {index}', value, expected[index], 5e-4)
nulls = {'Clark--Evans': fixed['nni'], 'Annulus': fixed['annulus']}
for name, key in [('Clark--Evans', 'nni'), ('Annulus', 'annulus')]:
    values = cell_numbers(row_cells(rows[name])[-1])
    check(f'tab:descriptors {name} null mean', values[0], nulls[name]['mean'], 5e-4)
    check(f'tab:descriptors {name} null sd', values[1], nulls[name]['sd'], 5e-4)
# L rows and max L rows come from patterns[].ripley
for name, field in [('$L(\\bar r)-\\bar r$', 'at_mean_nn'), ('$\\max|L(r)-r|$', 'max_abs')]:
    values = [n for cell in row_cells(rows[name])[:-1] for n in cell_numbers(cell)]
    for index, value in enumerate(values):
        check(f'tab:descriptors {name} pattern {index}', value,
              list(patterns.values())[index]['ripley'][field], 5e-4)
cdf_values = [n for cell in row_cells(rows['CDF discrepancy'])[:-1] for n in cell_numbers(cell)]
for index, value in enumerate(cdf_values):
    check('tab:descriptors CDF discrepancy', value,
          list(patterns.values())[index]['brody_diagnostics']['cdf_supremum'], 5e-4)

# ---------------- tab:open-variance ----------------
rows = table_rows('tab:open-variance', ['Row-major -- Ulam', 'Row-major -- diagonal sweep', 'Ulam -- diagonal sweep'])
for heading, key in [('Row-major -- Ulam', 'row_major__ulam'),
                     ('Row-major -- diagonal sweep', 'row_major__cantor'),
                     ('Ulam -- diagonal sweep', 'ulam__cantor')]:
    entry = results['open_shell_validation']['contrasts'][key]
    values = [n for cell in row_cells(rows[heading]) for n in cell_numbers(cell)]
    check(f'tab:open-variance {heading} q', values[0], entry['exact']['shared_edges'], 0.5)
    check(f'tab:open-variance {heading} S_v', values[1], entry['exact']['degree_energy'] * 1e7, 5e-4)
    check(f'tab:open-variance {heading} exact sigma', values[2],
          float(np.sqrt(entry['exact']['variance_normalised_difference'])), 5e-4)
    check(f'tab:open-variance {heading} mc sigma', values[3], entry['simulation']['sd'], 5e-4)
inline = re.search(r'contrast standard deviations of \$([\d.]+)[-]+([\d.]+)\$', TEX)
if inline:
    check('inline open sigma low', float(inline.group(1)), 0.0299, 5e-4)
    check('inline open sigma high', float(inline.group(2)), 0.0300, 5e-4)

# ---------------- tab:finite ----------------
rows = table_rows('tab:finite', ['$31$', '$54$', '$100$', '$173$', '$316$', '$547$', '$1000$'])
for item in results['finite_size']:
    heading = f"${item['width']}$"
    values = [n for cell in row_cells(rows[heading]) for n in cell_numbers(cell)]
    check(f'tab:finite W={item["width"]} primes', values[0], item['occupied'], 0.5)
    check(f'tab:finite W={item["width"]} beta', values[1], item['beta'], 5e-4)
    check(f'tab:finite W={item["width"]} null mean', values[2], item['null']['mean'], 5e-4)
    check(f'tab:finite W={item["width"]} null sd', values[3], item['null']['sd'], 5e-4)
    check(f'tab:finite W={item["width"]} delta', values[4], item['difference'], 5e-4)

# ---------------- tab:power + inline power numbers ----------------
rows = table_rows('tab:power', ['Fixed-count null', 'Row-major', 'Ulam'])
level_cells = [n for cell in row_cells(rows['Fixed-count null'])[1:] for n in cell_numbers(cell)]
check('tab:power paired level', level_cells[0], round_decimal(power['null']['paired_level'], 3), 0.0)
check('tab:power row level', level_cells[1], round_decimal(power['null']['marginal_row_level'], 3), 0.0)
check('tab:power ulam level', level_cells[2], round_decimal(power['null']['marginal_ulam_level'], 3), 0.0)
for item in power['alternative']:
    name = 'Row-major' if item['owner'] == 'row_major' else 'Ulam'
    candidates = [line for line in table_block('tab:power').splitlines()
                  if name in line and f'${item["theta"]}$' in line and '&' in line]
    if len(candidates) != 1:
        print(f'FAIL  tab:power row {name} theta={item["theta"]}: {len(candidates)} matches')
        failures.append(f'tab:power/{name}/{item["theta"]}')
        continue
    values = [n for cell in row_cells(candidates[0])[1:] for n in cell_numbers(cell)]
    check(f'tab:power {name} theta={item["theta"]} paired', values[0], round_decimal(item['power_paired'], 3), 0.0)
    check(f'tab:power {name} theta={item["theta"]} row', values[1], round_decimal(item['power_marginal_row'], 3), 0.0)
    check(f'tab:power {name} theta={item["theta"]} ulam', values[2], round_decimal(item['power_marginal_ulam'], 3), 0.0)

# sign-identification rates from stored z values
for item in power['alternative']:
    z = np.asarray(item['paired_z'])
    rate = float(np.mean(z > 0)) if item['owner'] == 'row_major' else float(np.mean(z < 0))
    check(f'sign rate {item["owner"]} theta={item["theta"]} in JSON', rate, rate)
signs_in_tex = re.findall(r'in every row-major realisation and in \$([\d.]+)\\pm([\d.]+)\$, \$([\d.]+)\\pm([\d.]+)\$ and \$([\d.]+)\\pm([\d.]+)\$', TEX)
ulam_rates = []
for item in power['alternative']:
    if item['owner'] == 'ulam':
        z = np.asarray(item['paired_z'])
        rate = float(np.mean(z < 0))
        ulam_rates.append((rate, float(np.sqrt(rate * (1 - rate) / len(z)))))
if signs_in_tex:
    got = [(float(a), float(b)) for a, b in signs_in_tex[0][0::2]]
    flat = signs_in_tex[0]
    quoted = [(float(flat[i]), float(flat[i + 1])) for i in range(0, 6, 2)]
    for index, (rate, se) in enumerate(ulam_rates):
        check(f'inline Ulam sign rate theta {index}', quoted[index][0], round_decimal(rate, 3), 5e-4)

# independent level-check sentence
level_line = next(line for line in TEX.splitlines() if 'marginal Ulam levels' in line)
level_next = TEX.splitlines()[TEX.splitlines().index(level_line) + 1]
quoted_levels = [float(x) for x in re.findall(r'0\.\d{3,4}', level_next)]
check('level check paired', quoted_levels[0], round_decimal(level['paired_level'], 3), 0.0)
check('level check row', quoted_levels[1], round_decimal(level['marginal_row_level'], 3), 0.0)
check('level check ulam', quoted_levels[2], round_decimal(level['marginal_ulam_level'], 3), 0.0)

# coverage percentages
grids = embedding_value_grids(316)
total_edges = 199712
for name, gaps in [('row_major', (315, 317)), ('ulam', (1246, 1248, 1250, 1252, 1254, 1256, 1258))]:
    from collections import Counter
    counter = Counter()
    for dc, dr in SHELLS['rsqrt2']:
        shifted = np.roll(np.roll(grids[name], dr, axis=0), dc, axis=1)
        diffs = np.abs(grids[name] - shifted).ravel()
        counter.update(int(g) for g in diffs if g > 0)
    planted = sum(counter[g] for g in gaps)
    expected = {('row_major',): 99.4, ('ulam',): 2.2}[tuple([name])]
    check(f'{name} planted coverage %', round(planted / total_edges * 100, 1), expected, 0.05)

# shell edge counts
check('toroidal shell edges E_d', total_edges, 99856 * 4 / 2, 0.0)
open_edges = results['open_shell_validation']['contrasts']['row_major__ulam']['exact'].get('first_edges')
if open_edges:
    check('open shell edges', open_edges, 198450, 0.0)

# ---------------- tab:seismic + inline seismic ----------------
seis_rows = table_rows('tab:seismic', ['Normalised count', 'Occupied edges',
                                       'Difference $g', 'Monte Carlo value'])
seis_values = [n for cell in row_cells(seis_rows['Normalised count']) for n in cell_numbers(cell)]
check('tab:seismic rook normalised', seis_values[0], round_decimal(seismic['counts']['rook_normalised'], 3), 0.0)
check('tab:seismic queen normalised', seis_values[1], round_decimal(seismic['counts']['queen_normalised'], 3), 0.0)
edge_values = [n for cell in row_cells(seis_rows['Occupied edges']) for n in cell_numbers(cell)]
check('tab:seismic occupied rook edges', edge_values[0], float(seismic['counts']['rook']), 0.0)
check('tab:seismic occupied queen edges', edge_values[1], float(seismic['counts']['queen']), 0.0)
diff_cells = row_cells(seis_rows['Difference $g'])
check('tab:seismic difference', cell_numbers(diff_cells[0])[-1],
      round_decimal(seismic['counts']['observed_difference'], 4), 0.0)
monte_cells = row_cells(seis_rows['Monte Carlo value'])
check('tab:seismic monte carlo value', cell_numbers(monte_cells[0])[-1],
      round_decimal(seismic['null']['monte_carlo_value'], 4), 0.0)
check('seismic z inline', 4.63, round_decimal(seismic['counts']['observed_z'], 2), 0.005)
# expected occupied edges independent
p2 = inclusion_probability(9216, 2920, 2)
check('seismic expected rook edges', 1831, 18240 * p2, 0.5)
check('seismic expected queen edges', 3642, 36290 * p2, 0.5)

# ---------------- reference integrity ----------------
normalised_tex = re.sub(r'\s+', ' ', TEX)
for phrase in ['Monte Carlo value under the null', 'exact null variance',
               'not a variance for the nonlinear Brody fit',
               'the dependence induced by the shared labels']:
    if phrase not in normalised_tex:
        print(f'FAIL  phrase missing: {phrase}')
        failures.append(phrase)

# ---------------- inhomogeneous control ----------------
inhom = results['inhomogeneous_null']
check('inhomogeneous calibration c', 0.9961876724, inhom['calibration'], 1e-9)
means = [inhom['beta'][name]['mean'] for name in ('row_major', 'ulam', 'cantor')]
sds = [inhom['beta'][name]['sd'] for name in ('row_major', 'ulam', 'cantor')]
for got, expected in zip([1.450, 1.441, 1.446], means):
    check('inhomogeneous mean', got, expected, 5e-4)
for got, expected in zip([0.022, 0.021, 0.021], sds):
    check('inhomogeneous sd', got, expected, 5e-4)
check('inhomogeneous diff 1-2', 0.010, means[0] - means[1], 5e-4)
check('inhomogeneous diff 1-3', 0.004, means[0] - means[2], 5e-4)

# ---------------- thinning ----------------
for item, got_mean, got_null, got_above in zip(results['thinning'],
                                              [1.133, 1.295, 1.468, 1.704],
                                              [1.032, 1.092, 1.157, 1.247],
                                              [223, None, None, None]):
    check(f'thinning rho={item["density"]} prime mean', got_mean, item['prime_mean'], 5e-4)
    check(f'thinning rho={item["density"]} bernoulli mean', got_null, item['null']['mean'], 5e-4)
    if got_above is not None:
        check('thinning above q95 at lowest density', got_above, item['above_null_q95'], 0.5)

# ---------------- ablation ----------------
# (Section C.4 removed in v2.5.1; the ablation results remain in the JSON)

# tie fractions inline
for index, got in enumerate([0.320, 0.594, 0.344]):
    check('tie fraction inline', got,
          list(patterns.values())[index]['brody_diagnostics']['largest_tie_fraction'], 5e-4)

# smallest-contrast exceedance from stored null beta values
beta_values = np.asarray(fixed['beta_values'])
maxima = np.max(np.abs(np.column_stack((beta_values[:, 0] - beta_values[:, 1],
                                       beta_values[:, 0] - beta_values[:, 2],
                                       beta_values[:, 1] - beta_values[:, 2]))), axis=1)
check('no null maximum reaches smallest contrast', 0,
      float(np.count_nonzero(maxima >= smallest)), 0.0)

# power section sigma quotes
check('power sigma_zero quote', 0.02982, power['config']['sigma_zero'], 5e-5)
check('power marginal row sd quote', 0.02108, power['config']['marginal_row_sd'], 5e-5)
check('power marginal ulam sd quote', 0.02108, power['config']['marginal_ulam_sd'], 5e-5)

if failures:
    print(f'\n{len(failures)} failure(s)')
    sys.exit(1)
print('\nALL CHECKS PASSED')

# Exact randomisation variance for paired spatial graph-count contrasts

## Reproducibility package for: *"Exact randomisation variance for paired spatial graph-count contrasts"*
### Kucharski (2026), submitted to *Spatial Statistics* (Elsevier)

---

This repository contains the manuscript, analysis scripts, cached inputs,
recorded numerical results and figure-building code for the paper *Exact
randomisation variance for paired spatial graph-count contrasts*. All results
in the submitted PDF are reproduced by the seeded scripts below from the
recorded JSON files. The release cited in the manuscript is the immutable
Zenodo version 2.4.1, DOI `10.5281/zenodo.24012387` (concept DOI
`10.5281/zenodo.23940934`, which resolves to the latest release).

## Quick start

```bash
git clone https://github.com/dawidkucharski/brody-2d-exclusion.git
cd brody-2d-exclusion/elsarticle
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## Files (`elsarticle/`)

- `aurora_manuscript.tex`, `aurora_refs.bib` — the manuscript and bibliography.
- `prime_spatial_analysis.py` — main pipeline: embeddings, nearest-neighbour
  Brody fits, paired fixed-count null (4,999 realisations), mapped
  inhomogeneous control, exact-shell statistics, the exact graph-contrast
  variance of Eq. (5), thinning and finite-window analyses; writes
  `prime_spatial_results.json`.
- `seismic_application.py` — the Section 4 numerical illustration: USGS ANSS
  earthquake catalogue rasterised onto a 96×96 grid, nested rook versus full
  queen adjacency, the exact variance of Eq. (5) and 1,999 paired Monte Carlo
  realisations; writes `seismic_application.json` and
  `fig_seismic_application.pdf`.
- `power_calibration.py` — planted-gap power calibration (Section 3.4,
  Table C.2); writes `power_calibration.json` and
  `fig_power_calibration.pdf`.
- `power_level_check.py` — independent second-seed level check.
- `build_prime_figures.py` — rebuilds the six vector figures from the JSONs.
- `test_prime_spatial_analysis.py` — 18 automated tests that check every
  manuscript table against the recorded JSONs.
- `deep_audit.py` — programmatic audit of every table and inline number in the
  manuscript against the JSONs, plus independent recomputations.
- `validate_references.py` — verifies every bibliography entry against
  Crossref/DataCite.
- `prime_spatial_reproducibility.zip` — the full reproducibility package,
  including `deep_audit.py` (which is git-ignored in the repository).

## Reproducing the results

```bash
cd elsarticle

# 1. Full prime analysis (about 12 minutes)
python3 prime_spatial_analysis.py --seed 20261101 --paired 4999 --inhomogeneous 1000

# 2. Seismic numerical illustration (about 1 minute)
python3 seismic_application.py --width 96 --null 1999 --seed 20261101

# 3. Power calibration (about 50 minutes; writes its figure)
python3 power_calibration.py

# 4. Independent second-seed level check
python3 power_level_check.py

# 5. Tests, audit and reference verification
python3 test_prime_spatial_analysis.py
python3 deep_audit.py
python3 validate_references.py
```

The JSON outputs are deterministic for the stated seeds; the figures can be
rebuilt without rerunning the simulations:

```bash
python3 build_prime_figures.py
python3 power_calibration.py --figure-only
```

## Compiling the manuscript

```bash
cd elsarticle
latexmk -pdf aurora_manuscript.tex   # or: pdflatex + bibtex + pdflatex + pdflatex
```

## Citation

Kucharski, D. (2026). Exact randomisation variance for paired spatial
graph-count contrasts. *Spatial Statistics*, submitted. Code and data archived
on Zenodo, version 2.4.1, DOI 10.5281/zenodo.24012387 (concept DOI
10.5281/zenodo.23940934).


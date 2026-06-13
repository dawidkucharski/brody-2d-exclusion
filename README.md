# Brody Spacing Statistics of Surface Textures

## Reproducibility package for: *"The Brody exponent as a measure of exclusion strength in 2D spatial point processes"*
### Kucharski (2026), Physica A: Statistical Mechanics and its Applications

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/dawidkucharski/brody-2d-exclusion.git
cd brody-2d-exclusion

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline
python3 elsarticle/unified_analysis.py
```

## Repository Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── aurora_manuscript.tex               # LaTeX manuscript (main)
├── aurora_refs.bib                     # Bibliography
├── elsarticle/
│   ├── unified_analysis.py             # Main analysis pipeline
│   ├── generate_*.py                   # Figure generation scripts
│   ├── fig_*.pdf                       # All figures (main + SI)
│   ├── fig_*.png                       # PNG versions
│   ├── *.json                          # Computed data files
│   └── spatial_pp_fitting.json         # Spatial PP model comparison
├── submission/
│   ├── cover_letter.docx               # Cover letter
│   ├── highlights.docx                 # Highlights
│   └── aurora_manuscript.pdf           # Compiled manuscript
└── data/
    └── (surface metrology .sur files - available on request)
```

## Data Files

All computed data are stored as JSON in `elsarticle/`:

| File | Description |
|------|-------------|
| `unified_results.json` | β values for FV surfaces, primes, PSI |
| `exclusion_physics_results.json` | Synthetic hard-core calibration data |
| `density_thinning_results.json` | Density-thinning experiment (5 densities × 20 trials) |
| `spatial_permutation_and_convergence.json` | Spatial permutation control + β(N) series |
| `watershed_grid_limited_results.json` | Watershed-recovered surface β values |
| `psi_ceramic_results.json` | PSI ceramic standard analysis |
| `psi_per_rev_results.json` | Per-revolution PSI analysis |
| `spatial_pp_fitting.json` | Strauss/area-interaction model comparison |
| `embedding_robustness.json` | Embedding dependence (row-major, Ulam, Cantor) |
| `sensitivity_analysis.json` | Peak-detection threshold sensitivity |
| `spatial_benchmark_results.json` | Comparison with standard spatial statistics |
| `banach_results.json` | Banach-space descriptor fingerprints |

## Figures

### Main text (5 figures)
- `fig_exclusion_physics.pdf` — β–r_excl calibration + domain overlay
- `fig_density_thinning.pdf` — Density-dependence of prime β
- `fig_unified_pair_correlation.pdf` — g₂(r) across domains
- `fig_unified_beta_landscape.pdf` — 58-surface β histogram + process boxplots
- `fig_psi_ceramic_analysis.pdf` — PSI case study

### Supplementary Information (10 figures)
- `fig_multifractal.pdf` — Fig. S1: D_q and f(α) spectra
- `fig_d2_scaling.pdf` — Fig. S2: D₂ scaling for 6 sequences
- `fig_dq_spectra.pdf` — Fig. S3: D_q for FV surfaces
- `fig_forest_plot.pdf` — Fig. S4: ΔD₂ forest plot
- `fig_psd_scaleup.pdf` — Fig. S5: PSD scale-up
- `fig_convergence.pdf` — Fig. S6: ΔD₂(N) convergence
- `fig_physical_vs_math.pdf` — Fig. S7: Physical vs. mathematical surfaces
- `fig_quantum_chaos_spacings.pdf` — Fig. S8: β for 21 prominence-based surfaces
- `fig_banach_descriptors.pdf` — Fig. S9: Banach descriptor fingerprints
- `fig_prime_convergence_aic.pdf` — Fig. S10: AIC convergence

### Additional
- `fig_embedding_robustness.pdf` — Embedding robustness (main text)
- `fig_sensitivity_analysis.pdf` — Peak-detection sensitivity (main text)

## Reproducing Results

All numerical claims in the manuscript can be verified by running:

```bash
python3 elsarticle/unified_analysis.py
```

Key verification checks:
```bash
# Verify CSR baseline
python3 -c "import json; d=json.load(open('elsarticle/exclusion_physics_results.json')); print(f'CSR β = {d[\"poisson_csr_2d\"][\"beta_mean\"]:.2f}±{d[\"poisson_csr_2d\"][\"beta_std\"]:.2f}')"

# Verify prime β
python3 -c "import json; d=json.load(open('elsarticle/unified_results.json')); print(f'Prime β = {d[\"prime_embedding\"][\"brody_beta\"]:.2f}')"

# Verify density-thinning at PSI density
python3 -c "import json; d=json.load(open('elsarticle/density_thinning_results.json')); print(f'Prime β at ρ=0.032: {d[\"rho_0.032\"][\"beta_mean\"]:.2f}±{d[\"rho_0.032\"][\"beta_std\"]:.2f}')"

# Verify PSI β
python3 -c "import json; d=json.load(open('elsarticle/psi_ceramic_results.json')); print(f'PSI β = {d[\"best_brody_beta\"]:.2f} [{d[\"beta_ci_low\"]:.2f},{d[\"beta_ci_high\"]:.2f}]')"
```

## Compiling the Manuscript

```bash
cd elsarticle
pdflatex aurora_manuscript.tex
bibtex aurora_manuscript
pdflatex aurora_manuscript.tex
pdflatex aurora_manuscript.tex
```

## License

MIT License. Surface metrology data (.sur files) available from the author upon reasonable request.

## Citation

Kucharski, D. (2026). The Brody exponent as a measure of exclusion strength in 2D spatial point processes: cross-domain consistency across manufactured surfaces, prime embeddings, and interferometric profilometry. *Physica A: Statistical Mechanics and its Applications*, under review.

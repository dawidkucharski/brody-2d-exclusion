# Option 2 Salvage Paper

## Working title
Embedding geometry controls short-range spacing statistics in 2D binary prime patterns

## Alternative titles
1. Short-range spacing and multiscale descriptors of 2D prime embeddings
2. Embedding-dependent nearest-neighbor statistics in binary prime surfaces
3. Prime embeddings in two dimensions: local spacing sensitivity versus multiscale descriptor stability

## Core claim
The publishable result is not a universal Brody-based exclusion framework for 2D point processes.

The defensible result is narrower:
- the same 1D prime sequence produces materially different 2D nearest-neighbor spacing statistics under different deterministic embeddings;
- this embedding sensitivity is strong for local spacing descriptors such as fitted Brody beta;
- the effect is not explained by sparsity alone, as shown by sparse-integer and density-matched Bernoulli controls;
- multiscale descriptors such as D2 and Banach-space measures are comparatively more stable across embeddings, so local and multiscale descriptors capture genuinely different aspects of arithmetic structure.

## Draft abstract
Deterministic embeddings of one-dimensional arithmetic sequences into two dimensions can create or suppress apparent spatial structure. This paper studies binary 2D embeddings of the prime sequence at N = 10^5 under three deterministic mappings: row-major order, the Ulam spiral, and Cantor pairing. I compare a local spacing descriptor derived from nearest-neighbor statistics with multiscale descriptors based on the correlation dimension D2 and Banach-space invariants. The main result is a sharp contrast between descriptor classes. The nearest-neighbor spacing fit is strongly embedding-dependent: the fitted Brody parameter changes from 2.15 for row-major embedding to 1.87 for the Ulam spiral and 1.40 for Cantor pairing, while a density-matched Bernoulli control gives 1.47 ± 0.02. A sparse-integer control shows that the elevated row-major value is not a trivial consequence of low fill fraction. In contrast, D2 remains positive relative to the Bernoulli null across embeddings, indicating that multiscale mass-distribution signatures of the prime sequence are more stable than short-range spacing signatures. These results show that 2D spatial statistics of arithmetic patterns are jointly determined by sequence structure and embedding geometry, and that local nearest-neighbor summaries and multiscale descriptors should not be interpreted interchangeably.

## Proposed paper scope
This version should focus almost entirely on arithmetic embeddings and controls.

Keep:
- row-major / Ulam / Cantor embedding comparison
- sparse-integer control
- density-thinning control
- Bernoulli null at matched density
- D2 comparison across embeddings
- Banach descriptors only if they help separate local versus multiscale structure
- one short discussion of why local and multiscale descriptors disagree

Drop from the main paper:
- the 58-surface FV dataset as a central result
- PSI profilometry as a central result
- the synthetic hard-core calibration as a headline contribution
- any claim that the 2D CSR baseline is newly established
- any claim that Rayleigh is outside the Brody family
- any universal interpretation of beta as a calibrated exclusion scale across domains

Possible supplementary material:
- full D2 validation details
- additional arithmetic sequences
- finite-size scans in N
- sensitivity analyses
- surface and PSI material only as secondary context, if kept at all

## Recommended structure
### 1. Introduction
- motivate the problem: a 1D arithmetic sequence can look very different under different 2D embeddings;
- distinguish local spacing structure from multiscale mass-distribution structure;
- state the specific question: which observed spatial signatures are intrinsic to the prime sequence and which are created by the embedding?

### 2. Data and embeddings
- define the binary prime surface at N = 10^5;
- define row-major, Ulam, and Cantor embeddings precisely;
- define Bernoulli and sparse-integer controls;
- state point density explicitly.

### 3. Descriptor families
- local descriptor: nearest-neighbor spacing fit summarized by fitted Brody beta;
- multiscale descriptor: D2;
- optional Banach descriptors if retained.
- explicitly state that beta is used only as an empirical fit summary, not as a universal exclusion parameter.

### 4. Results: embedding dependence of local spacing statistics
- row-major, Ulam, Cantor beta values;
- Cantor null result;
- sparse-integer control;
- density-thinning control.

### 5. Results: embedding dependence of multiscale descriptors
- D2 across embeddings;
- Bernoulli comparison;
- optional Banach descriptor summary;
- direct contrast with the beta behavior.

### 6. Discussion
- local descriptors are embedding-sensitive;
- multiscale descriptors are comparatively stable;
- arithmetic spatial structure is not a single object but depends on the observation operator (embedding + descriptor);
- implications for visual or statistical claims about prime patterns in 2D.

### 7. Conclusion
- emphasize descriptor dependence and embedding dependence;
- avoid any universal calibration language.

## Figures to reuse
Most reusable figures are already in the project.

Strong candidates:
- fig_embedding_robustness.pdf
- fig_density_thinning_control.pdf
- fig_multifractal.pdf
- fig_d2_scaling.pdf
- fig_banach_descriptors.pdf
- fig_forest_plot.pdf (only if it directly supports the arithmetic-sequence story)

Likely drop:
- fig_exclusion_physics.pdf
- fig_unified_pair_correlation.pdf
- fig_unified_beta_landscape.pdf
- fig_physical_vs_math.pdf
- fig_quantum_chaos_spacings.pdf

## Claims that must be removed or rewritten
Delete or rewrite all statements of the form:
- the 2D CSR baseline is newly corrected from 0 to 0.96;
- Rayleigh is not in the Brody family;
- Brody outperforms Rayleigh for CSR;
- beta is a calibrated cross-domain exclusion scale;
- the same calibration can be transferred unchanged across surfaces, primes, and PSI.

Replace with:
- beta is an empirical one-parameter summary of local spacing statistics;
- in arithmetic embeddings, beta is strongly embedding-dependent;
- D2 and beta respond to different spatial scales;
- sparse-integer and Bernoulli controls separate arithmetic structure from density and embedding artefacts.

## Minimum extra work before resubmission
1. Recompute and present all arithmetic-only results in a self-contained table.
2. Add one explicit section comparing what beta and D2 do measure.
3. Remove cross-domain language from title, abstract, and conclusion.
4. Audit every occurrence of "CSR baseline" and "exclusion calibration".
5. Rewrite the introduction around embedding dependence rather than universal calibration.

## Go / no-go judgment
Go, but only as a substantially narrower arithmetic-embedding paper.

Do not resubmit the current broad manuscript with cosmetic edits.

If you are unwilling to cut the surface and PSI stories from the main contribution, retire the project instead of shopping the same paper again.

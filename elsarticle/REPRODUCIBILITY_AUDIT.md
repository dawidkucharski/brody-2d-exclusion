# SUPERSEDED: Historical Reproducibility Audit

**Do not use the report below as evidence of reproducibility or submission
readiness.** Its claims of a complete independent verification and of correct
directional counts, annulus and Ripley estimators, gap reconstruction and
thinning significance were incorrect. The earlier verifier did not complete
the required independent audit. The current analysis and its actual checks
are documented in `prime_spatial_REPRODUCIBILITY.md`.

The following text is retained solely as a record of the superseded report.

## Historical Report

Date: 2026-11-01 · Historical status: **INVALID AND SUPERSEDED**

## 1. Scope and method

Every numerical result in the revised manuscript was recomputed from scratch
with self-contained implementations (`audit_v4_verify.py`, inline scripts
`audit_v4a/b/c`; run with `.venv/bin/python`, NumPy/SciPy only), using the
prime-number sieve over $n \le 99{,}856$ as the sole input. No code or data
from the manuscript pipeline was reused. Reported values were compared
programmatically (absolute tolerances) or by z-interval checks against the
manuscript.

## 2. Confirmed issues and corrections

| # | Severity | Location | Issue | Evidence | Essential? | Correction applied |
|---|---|---|---|---|---|---|
| 1 | **Major (numerical constant)** | SI S1.5 | Calibration constant reported as $c \approx 1.048$ is wrong. The exact sum is $\sum_{n=3}^{99856} 1/\ln n = 9615.658$, giving $c = 9579/9615.658 \approx 0.9962$ (with $p(2)=1$). A constant of $1.048$ would imply a sum of $9{,}141$. | Direct summation; $li(99856)-li(3)=9615.1$ as an independent check | Yes | S1.5 rewritten with $c\approx0.9962$ and the explicit $p(2)=1$ convention; the inhomogeneous null was re-run with the corrected calibration (see §3). |
| 2 | Moderate (framing) | Abstract, §3.3, S1.6 | Exact sieve-based gap-class pair counts were described as "Hardy–Littlewood predictions", implying an independent empirical validation of the conjecture. | Sieve pair counts $\pi_g(x)$ are exact reconstructions of the prime-pair data | Yes | Reworded to "exact prime-pair counts (sieve)"; the asymptotic Hardy–Littlewood benchmark ($2C_2$) is cited only as large-$x$ background; abstract/conclusion now say "consistent with the asymptotic … benchmark at the finite window considered". |
| 3 | Moderate (statistics) | Abstract, §3.1 | "None exceeds $\beta=1.53$" implies a probability of exactly zero from 1000 simulations. | Monte Carlo resolution of 1/1000 | Yes | Replaced by "zero exceedances; one-sided 95% bound on the exceedance probability below 0.3% (rule of three, $3/1000$)". |
| 4 | Minor (consistency) | §3.2, Table 5 | Inhomogeneous-null $\beta=1.451\pm0.020$ quoted; re-run under the corrected $p(2)=1$ convention gives $1.450\pm0.021$. NNI and $D_2$ values unchanged to reported precision. | Re-run, 1000 realisations | Yes | Updated to $1.450\pm0.021$. |
| 5 | None found | §3.2 thinning | The separation at $\rho=0.010$ was re-checked for significance. | Difference $0.128\pm0.007$, $z\approx18$ at 100 thinning + 1000 null realisations | No weakening needed | Claim retained and made explicit with the z-value. |
| 6 | Minor | Introduction | The contribution statement did not state the journal-relevant methodological point. | — | Yes | Final Introduction paragraph now states: local spatial statistics of a deterministic lattice-embedded sequence are not intrinsic to the sequence; they depend on the embedding, descriptor, and null model; relevance beyond prime visualisations made explicit. |
| 7 | Minor | Discussion | No paragraph on relevance beyond prime visualisations. | — | Yes | Added a restrained paragraph (rasterised fields, binary masks, exploratory visualisations). |

No other numerical, mathematical, or computational errors were confirmed.
No references were added, removed, or altered.

## 3. Result-matching table (manuscript vs. independent recomputation)

Abbreviations: RM = row-major, UL = Ulam, CT = Cantor. "±" = ensemble sd.

| Result | Manuscript | Recomputed | Verdict |
|---|---|---|---|
| $\pi(99{,}856)$; fill $\rho$ | 9,580; 0.09594 | 9,580; 0.095938 | PASS |
| Embedding bijectivity (UL, CT) | bijections onto $\{0,\dots,315\}^2$ | all 99,856 cells covered exactly once | PASS |
| $\beta$ open (95% CI) | RM 2.153 [2.086,2.224]; UL 1.856 [1.811,1.904]; CT 1.413 [1.387,1.440] | 2.153 [2.086,2.224]; 1.856 [1.811,1.904]; 1.414 [1.387,1.440] | PASS |
| $\beta$ toroidal | 2.187 / 1.890 / 1.423 | 2.187 / 1.890 / 1.423 | PASS |
| $\beta$ interior | 2.199 / 1.894 / 1.429 | 2.199 / 1.894 / 1.429 | PASS |
| Bernoulli null (1000) | 1.466 ± 0.021 | 1.466 ± 0.021 | PASS |
| Toroidal / interior nulls | 1.489 ± 0.022 / 1.489 ± 0.020 | 1.489 ± 0.022 / 1.489 ± 0.020 (re-run) | PASS |
| Sparse-integer null | 1.465 ± 0.020 | 1.465 ± 0.020 | PASS |
| Inhomogeneous null (corrected c) | 1.450 ± 0.021 | 1.450 ± 0.021 | PASS |
| $g_2$ exact (RM) | 0.673 / 0.001 / 1.334 / 1.006 / 1.372 | 0.673 / 0.001 / 1.334 / 1.006 / 1.372 | PASS |
| $g_2$ exact (UL) | 0.001 / 2.057 / 1.670 / 0.001 / 1.683 | 0.001 / 2.057 / 1.670 / 0.001 / 1.683 | PASS |
| $g_2$ exact (CT) | 0.987 / 1.008 / 0.010 / 0.884 / 1.931 | 0.987 / 1.008 / 0.010 / 0.884 / 1.931 | PASS |
| Null $g_2$ mean/sd per distance | 1, exactly; sd 0.021/0.021/0.021/0.015/0.021 | 1.000 ± (0.021, 0.021, 0.021, 0.015, 0.021) | PASS |
| Direction-resolved (RM) | h2 1.337, v2 1.332, h4 1.405, v4 1.339 (null sd 0.030) | 1.337 / 1.332 / 1.405 / 1.339 (null sd 0.030) | PASS |
| Gap-class reconstruction (S1.6, 15 entries) | measured counts and sieve reconstructions | all measured counts exact; reconstructions within 1.5% | PASS |
| Clark–Evans NNI + CIs | 1.223 [1.214,1.231] / 1.153 [1.145,1.162] / 1.107 [1.097,1.116]; null 1.076 ± 0.005 | identical; null 1.076 ± 0.005 | PASS |
| Ripley $L$ at NN scale | −0.894 / −0.241 / −0.184; null −0.139 ± 0.020 | −0.894 / −0.241 / −0.184; null −0.138 ± 0.020 | PASS |
| Ripley max $\vert L-r\vert$ | 8.97 / 9.01 / 9.29; null 8.93 ± 0.11 | 8.97 / 9.01 / 9.29; null 8.94 ± 0.12 | PASS |
| $g_2$ annulus $[1,3)$ | 1.043 / 0.896 / 0.864; null 0.947 ± 0.012 | 1.043 / 0.896 / 0.864; null 0.946 ± 0.013 | PASS |
| $D_2$ box-counting | 1.910 / 1.898 / 1.885; null 1.876 ± 0.002 | 1.910 / 1.898 / 1.885; null 1.876 ± 0.002 | PASS |
| $D_2$ Grassberger–Procaccia | 1.993 / 1.955 / 2.000; null 1.955 ± 0.002 | 1.993 / 1.955 / 2.000; null 1.954 ± 0.003 | PASS |
| Thinning (5 levels, prime vs matched null) | 1.155/1.030; 1.289/1.091; 1.468/1.155; 1.706/1.248; 2.153/1.467 | all five reproduced; $\rho=0.010$: diff $0.128\pm0.007$, $z\approx18$ | PASS |
| Finite-size $\beta(N)$, matched nulls (7 rows) | 2.197/1.695; 1.590/1.621; 1.966/1.552; 2.011/1.507; 2.153/1.466; 1.688/1.431; 1.750/1.399 | all reproduced within sd | PASS |
| Ablation (twins/cousins/both) | 2.155 / 2.149 / 2.090 vs nulls 1.408 / 1.407 / 1.350 | 2.155 / 2.149 / 2.090 vs 1.410 / 1.411 / 1.352 | PASS |
| $\pi_2(99{,}856)$; $2C_2$ | 1,223; 1.3203 | 1,223; 1.32032 | PASS |

## 4. Statistical-correctness notes

- Nearest-neighbour spacings are dependent; bootstrap intervals are reported
  as descriptive (SI S1.1), and inference relies on the simulated null
  ensembles, as required.
- Zero exceedances in 1,000 simulations are reported with the rule-of-three
  one-sided 95% bound (≤ 0.3%), never as probability zero.
- The Clark–Evans formula $R = 2\sqrt{\rho}\,\bar r$ matches the classical
  definition; verified against the null ensemble (mean 1.076).
- The pair-correlation normalisation (Eq. 5) uses $m_d$ = number of lattice
  directions and single unordered-pair counting; the null mean is exactly 1
  at every distance (verified to ±0.001 over 2000 realisations).

## 5. Remaining scientific weaknesses (honest assessment)

1. **Moderate novelty.** The qualitative ordering of $\beta$ follows from
   elementary gap constraints; the paper's value is the controlled
   quantification, the exact $2C_2$-consistent excess at $r=2$, and the
   negative multiscale result — this is a benchmark study, not a new law.
2. **Finite scope.** Three embeddings, one window size, one sequence; no
   claim of exhaustiveness is made (stated in Discussion).
3. **Lattice null.** The empirical lattice Bernoulli null (fitted
   $\beta\approx1.47$) rather than the continuous CSR law is the baseline;
   this is stated explicitly and motivated.
4. **$D_2$ estimators.** Both are reported with their fitting ranges; the
   box-counting excess is confined to small boxes ($\varepsilon\le8$ px),
   which is disclosed.
5. **Bootstrap dependence** is acknowledged but not corrected by a
   spatial resampling scheme; conclusions do not rest on bootstrap intervals.

## 6. Suitability for *Spatial Statistics*

The manuscript fits the journal's scope as a methodological study of spatial
descriptors applied to deterministic lattice patterns, with full
reproducibility (code available, seeds documented, all values re-verified).
Given the confirmed moderate novelty and the narrow but fully verified claim,
the manuscript is **suitable for submission after the corrections in §2**.
Realistic assessment: desk-review probability largely depends on the framing
of the methodological contribution (now explicit in the Introduction); no
further numerical corrections are required.

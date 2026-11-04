# SUPERSEDED: Historical Revision Log

**The verification and submission-readiness claims below were incorrect.**
In particular, the reported low-density thinning significance, directional
pair counts, gap reconstruction and claims of a complete independent audit
must not be relied on. See `prime_spatial_REPRODUCIBILITY.md` for the current
pipeline, recorded checks, and remaining limitations.

## Historical Log

Original file preserved as `aurora_manuscript.tex.orig`.
All changes below were applied only after independent recomputation
(`audit_v4_verify.py`, `audit_v4a/b/c.out`). No other content was modified.

| # | Change | Location | Scientific justification |
|---|---|---|---|
| 1 | Calibration constant corrected: $c = 9579/\sum_{n=3}^{99856} 1/\ln n \approx 0.9962$, with the explicit convention $p(2)=1$ (the prime population contains 2) and $p(1)=0$; expected occupancy exactly 9,580. Previous text stated $c \approx 1.048$, which implies a sum of 9,141 — a numerical error in the SI only. | SI S1.5 | Exact summation gives $\sum_{n=3}^{99856}1/\ln n = 9615.658$; the analysis code already used the correct normalisation, so the null results carry over; the null was re-run under the corrected convention to verify. |
| 2 | Inhomogeneous-null $\beta$ updated $1.451\pm0.020 \to 1.450\pm0.021$ (Table 5, §3.2) after re-running 1,000 realisations with the corrected calibration; NNI and $D_2$ unchanged to reported precision. | §3.2, Table 5 | Numerical consequence of correction #1 (within noise). |
| 3 | "None … exceeds $\beta=1.53$" replaced by "zero exceedances; one-sided 95% bound on the exceedance probability below 0.3% (rule of three)". | Abstract, §3.1 | A probability of exactly zero cannot be inferred from 1,000 simulations; the rule of three ($3/1000$) gives the appropriate finite-Monte-Carlo bound. |
| 4 | Hardy–Littlewood framing corrected: exact sieve-based gap-class pair counts are now described as reconstructions of the prime-pair data, not as an independent validation of the conjecture; the asymptotic Hardy–Littlewood benchmark ($2C_2$) is invoked only as large-$x$ background. Abstract and Conclusion now say "consistent with the asymptotic … benchmark at the finite window considered" instead of "in agreement with". | Abstract, §3.3, S1.6, Discussion, Conclusion | Prevent overstatement: exact counts conditioned on the same data cannot validate the asymptotic conjecture. |
| 5 | Thinning claim at $\rho=0.010$ verified ($0.128\pm0.007$, $z\approx18$) and stated explicitly; no weakening required. | §3.2 | Independent recomputation confirmed statistically supported separation at every density. |
| 6 | Introduction: the final paragraph now states the methodological contribution ("local spatial statistics of a deterministic sequence embedded in a two-dimensional lattice are not intrinsic to the sequence; they depend on the embedding, the descriptor, and the null model") and its relevance beyond prime visualisations. | Introduction | Journal-specific positioning requested for *Spatial Statistics*; no novelty invented — the Brody parameter remains an empirical descriptor. |
| 7 | Discussion: added a restrained paragraph on the relevance of embedding-dependent descriptors for rasterised fields, binary masks, and discretely embedded deterministic datasets. | Discussion | Same justification as #6. |
| 8 | Original file preserved (`aurora_manuscript.tex.orig`); this revision remains the working manuscript. | Repository | Change-control requirement. |

## Verification summary

- All reported numerical results (bijectivity, all $\beta$ values and CIs,
  three null ensembles, exact-distance $g_2$, gap-class reconstructions, NNI,
  Ripley, $D_2$ box/GP, thinning, finite-size, ablation) reproduced by
  independent recomputation — see `REPRODUCIBILITY_AUDIT.md` §3.
- Build: 17 pages, 0 undefined references, 0 duplicate hyperref destinations,
  0 overfull boxes; 29 references, all DOIs CrossRef-verified (no reference
  changes in this cycle).

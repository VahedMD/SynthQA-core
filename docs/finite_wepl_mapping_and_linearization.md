# Finite WEPL mapping and first-order approximation diagnostics

## Estimators

The benchmark reports two distinct quantities. The local first-order estimate is

\[
\Delta z_{R80}^{(1)}=-\frac{\Delta W(z_{R80,ref})}{SPR_{ref}(z_{R80,ref})}.
\]

The finite WEPL-mapped estimate is obtained by solving

\[
z_{R80,mapped}=W_{sCT}^{-1}(W_{ref}(z_{R80,ref})),\qquad
\Delta z_{R80}^{(finite)}=z_{R80,mapped}-z_{R80,ref}.
\]

The finite mapping uses the same Siddon-integrated SPR fields and sampled WEPL profiles as the benchmark. Repeated WEPL coordinates are collapsed to their first occurrence before inversion. This gives a deterministic convention for zero-contribution plateaus; a target outside the sampled WEPL range returns NaN rather than extrapolating.

The remapped dose profile is retained as a numerical consistency check. Its R80 is not an independent dose calculation: it is reconstructed from the reference dose profile through the same WEPL mapping.

## Diagnostics

For the interval between mapped and reference R80, the benchmark reports path-length-weighted reference and sCT SPR means, interval minima/maxima, and normalized SPR ranges:

\[
V_{SPR}=\frac{SPR_{max}-SPR_{min}}{\overline{SPR}}.
\]

It also reports absolute and relative first-order linearization discrepancies:

\[
\epsilon_{lin}=|\Delta z_{R80}^{(1)}-\Delta z_{R80}^{(finite)}|,
\qquad
\eta_{lin}=\frac{\epsilon_{lin}}{|\Delta z_{R80}^{(finite)}|+10^{-9}}.
\]

These are descriptive diagnostics, not prespecified pass/fail criteria. They should be evaluated across the full cohort before defining any threshold.

## Interpretation and limitations

The local approximation is expected to become less accurate when SPR varies appreciably over the finite mapping interval. Extended low-density distal paths are a plausible recurring configuration, but low density alone is not sufficient to explain failure: the relevant quantity is the spatial variation of the stopping-power gradient over the interval. The diagnostics support testing this hypothesis; they do not establish a causal anatomical mechanism without image-level review.

Neither estimator constitutes independent dose recalculation. The finite mapping is an analytical WEPL reconstruction; independent PB/Monte Carlo comparisons must remain separate in the analysis and manuscript.

## Recomputing the frozen expansion cohort

Run `python scripts/rebenchmark_expansion_dual_estimators.py` from the repository root. The script reads the existing refreshed n=30 case-level CSV only to recover its ordered patient IDs, verifies the cohort constraints (including exclusion of 1THB121 and inclusion of 1ABB143), and recomputes those same patients without reselection. It writes a new `model_a_expansion_dual_estimator_n30.csv`; the historical file is not overwritten.

# Appendix A. Validity of the local first-order WEPL-to-range approximation

> **Analysis note:** This appendix is a working draft. Numerical cohort summaries must be regenerated for both frozen cohorts with the same shared-v3 implementation before manuscript submission. In particular, the expansion-arm case 1THB121 is currently held out and must be audited separately; its earlier result must not be mixed with shared-v3 results.

## A.1. Local and finite range-shift estimators

For a reference distal range endpoint at depth \(z_{R80,\mathrm{ref}}\), the local first-order approximation is

\[
\Delta z_{R80}^{(1)} = -\frac{\Delta W(z_{R80,\mathrm{ref}})}{\mathrm{SPR}_{\mathrm{ref}}(z_{R80,\mathrm{ref}})}.
\]

This expression follows from a first-order expansion of the reference WEPL profile around the reference endpoint. It assumes that the local slope is representative over the displacement being estimated. It is therefore a local approximation, not a general finite-displacement identity.

The finite WEPL-mapped endpoint is obtained by solving

\[
W_{\mathrm{sCT}}(z_{R80,\mathrm{mapped}})
= W_{\mathrm{ref}}(z_{R80,\mathrm{ref}}),
\]

and the corresponding shift is

\[
\Delta z_{R80}^{(\mathrm{finite})}
= z_{R80,\mathrm{mapped}}-z_{R80,\mathrm{ref}}.
\]

The finite mapping retains the spatial variation of SPR along the mapped interval and does not use a single local SPR value to convert the accumulated WEPL difference into depth.

## A.2. Why the local approximation can fail

The approximation can become inaccurate when the finite mapped interval spans a substantial change in SPR. In that situation, the SPR evaluated at the reference R80 endpoint may differ markedly from the path-averaged sCT SPR over the interval connecting the mapped endpoint and the reference endpoint.

For a monotonic WEPL profile and a valid inverse mapping, the interval-averaged sCT SPR is related to the accumulated WEPL difference by

\[
\overline{\mathrm{SPR}}_{\mathrm{sCT},\,\mathrm{interval}}
= \frac{|\Delta W(z_{R80,\mathrm{ref}})|}
       {|\Delta z_{R80}^{(\mathrm{finite})}|}.
\]

This identity follows from the finite-mapping equation and the integral definition of WEPL. It provides a useful diagnostic comparison with the local reference SPR used by the first-order approximation. A large discrepancy between these two gradients indicates that the local slope may not represent the full finite-mapping interval.

A low-SPR region distal to, or overlapping, the dose fall-off can contribute to this behaviour when the reference endpoint and the finite-mapped endpoint lie on different sides of a substantial tissue-density transition. The mechanism should be described in terms of the measured SPR profile rather than assigning an anatomical tissue label unless the corresponding image has been explicitly inspected.

## A.3. Case audit: 1THB002

The shared-v3 audit of 1THB002 found the following:

| Quantity | Result |
|---|---:|
| Reference R80 | 965.198 mm |
| Finite WEPL-mapped R80 | 941.421 mm |
| Finite mapped shift | −23.777 mm |
| ΔWEPL at reference R80 | +10.667 mm |
| Reference SPR at reference R80 | 0.2061 |
| sCT SPR at reference R80 | 0.2563 |
| First-order shift | −51.770 mm |
| Absolute difference between estimators | 27.993 mm |
| Path-averaged sCT SPR over mapped interval | 0.4486 |

The local first-order estimate is consistent with its defining equation, but the local reference SPR at R80 (0.2061) is substantially lower than the path-averaged sCT SPR over the finite-mapping interval (0.4486). The interval includes a marked SPR transition: segment-level values range from approximately 0.17–0.23 near the reference endpoint to values approaching 1.1 nearer the mapped endpoint. This spatial variation explains why the local conversion substantially overestimates the magnitude of the finite mapped shift in this case.

The inverse WEPL-mapped shift (−23.777 mm) and the R80 shift extracted from the remapped reference dose profile (−23.714 mm) differed by 0.063 mm. These are two numerical checks of the same analytical WEPL-remapping construction; they are not independent dose calculations or independent physical validation.

## A.4. Diagnostic quantities

The shared implementation reports the following quantities for each beamlet:

- absolute linearization discrepancy, \(|\Delta z^{(1)}-\Delta z^{(\mathrm{finite})}|\);
- relative linearization discrepancy, normalized by the magnitude of the finite mapped shift;
- local reference and sCT SPR at the reference R80 endpoint;
- path-length-weighted mean, standard deviation, minimum, and maximum SPR over the finite-mapping interval;
- coefficient of variation and normalized range of SPR over that interval;
- inverse-map versus remapped-dose-profile endpoint discrepancy.

These are descriptive diagnostics. No universal pass/fail threshold is imposed. Any threshold proposed later must be justified from the observed distribution and, ideally, tested in an independent cohort.

## A.5. Reporting and interpretation

Cases with large first-order discrepancies should remain in the primary cohort analysis. They are not automatically invalid data points: they identify conditions under which the local approximation is inadequate. The finite WEPL-mapped estimator should be reported separately from the first-order approximation, and neither should be described as an independent Monte Carlo or pencil-beam recalculation.

The expansion-arm outlier 1THB121 was identified in an earlier analysis and subsequently held out from the refreshed 30-patient cohort pending investigation. Its profile and voxel-segment audit must be recomputed under the same shared-v3 implementation before quantitative comparison with 1THB002. Until then, it should be described as an unresolved sensitivity case, not as confirmed evidence of the same mechanism.

## Suggested concise main-text discussion

The local first-order conversion from accumulated WEPL difference to range displacement assumes that the reference SPR at the distal endpoint represents the gradient over the resulting displacement. This assumption may be violated when the finite-mapping interval spans a substantial spatial variation in SPR, including transitions between near-water-equivalent and lower-SPR regions. In such cases, the first-order estimate can differ markedly from the finite WEPL-mapped endpoint. We therefore report the local approximation and finite mapping separately and characterize their discrepancy using path-averaged SPR and interval heterogeneity diagnostics (Appendix A). These discrepancies are treated as a limitation of the local approximation rather than grounds for excluding the corresponding cases. The finite mapping remains an analytical reconstruction from the reference dose profile and is not an independent perturbed-image dose calculation.

# M1–M2 implementation notes: dual estimators and validity diagnostics

## What the shared-v3 package already implemented

- Shared Siddon-based paired WEPL integration on the CT grid.
- Reference dose-profile sampling in its own image geometry.
- First-order local R80 shift.
- Finite WEPL inverse-mapped R80 shift.
- Remapped reference-dose-profile R80 as a numerical consistency check.
- Absolute and relative first-order-versus-finite discrepancy.
- Path-length-weighted mean and min/max SPR over the finite-mapping interval.

## Changes in this patch

- Added path-length-weighted SPR standard deviation and coefficient of variation for reference and sCT over the finite-mapping interval.
- Added those fields to the existing case-level CSV schema through the shared `benchmark_result_row` flattening logic.
- Added these expansion-cohort summary statistics: median relative linearization discrepancy and median/maximum sCT interval SPR coefficient of variation.
- Added interval diagnostics to stratified cohort manifest notes.
- Added `scripts/summarize_linearization_diagnostics.py` to combine available stratified and expansion dual-estimator CSVs without selecting or excluding patients.
- Added a working appendix draft at `docs/APPENDIX_A_LINEARIZATION_VALIDITY.md`.

## Re-run sequence

1. Re-run the stratified benchmark with the frozen 12-patient list.
2. Re-run the expansion benchmark using the frozen candidate order and the shared-v3 script. Keep 1THB121 held out for a separate audit; do not silently substitute it into the n=30 cohort.
3. Run `python scripts/summarize_linearization_diagnostics.py`.
4. Audit 1THB121 with the same shared-v3 diagnostic path before deciding whether its mechanism matches 1THB002.

The summary script intentionally preserves source-file provenance and does not deduplicate patients across cohorts. It does not apply a threshold or exclude outliers.

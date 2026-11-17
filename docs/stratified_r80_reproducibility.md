# Stratified R80 benchmark: shared implementation and exact-case audit

## Purpose

`benchmark_dataset.py` and `audit_stratified_r80_case.py` now call the same
implementation in `src/synthqa_core/surrogates/stratified_r80.py`.
The benchmark records the exact selected beamlet and physical ray in its CSV.
The audit reads that row; it does not independently search for or select a
beamlet. It validates beam/ray/beamlet identifiers, energy, dose-file path,
source/target coordinates, and profile ray length before recomputing the case.

The shared implementation version is `stratified_r80_shared_v3`. The audit
refuses benchmark CSVs produced by older implementations.

## Clean-start workflow

The scripts create their output directories when needed. From the repository
root, with the project virtual environment active:

```powershell
python scripts/benchmark_dataset.py
```

This recreates:

- `outputs/reproducibility/model_a_stratified_case_level.csv`
- `outputs/reproducibility/cohort_manifest.csv`
- `figures/Fig6_FirstOrder_vs_FiniteMap.png`

The frozen 12-patient cohort is used by default. To explicitly request live
candidate discovery (which requires listing the Hugging Face repository):

```powershell
$env:SYNTHQA_REFRESH_STRATIFIED_COHORT="1"
python scripts/benchmark_dataset.py
Remove-Item Env:SYNTHQA_REFRESH_STRATIFIED_COHORT
```

## Exact-case audits

Run the audit only after the revised benchmark has completed:

```powershell
python scripts/audit_stratified_r80_case.py --patient 1THB078
python scripts/audit_stratified_r80_case.py --patient 1THB002
```

For each patient, the audit recreates:

- `outputs/reproducibility/r80_case_audits/<PATIENT>_shared_v3_audit.json`
- `outputs/reproducibility/r80_case_audits/<PATIENT>_shared_v3_profile.csv`
- `outputs/reproducibility/r80_case_audits/<PATIENT>_shared_v3_voxel_segments.csv`

A successful audit reports `PASS` only when the selected beamlet identity and
all required estimator/interval-diagnostic values reproduce the benchmark CSV
within the default absolute tolerance of `1e-6`. A mismatch exits with status
2 and must be investigated; do not manually edit the CSV or relax the
comparison tolerance to force a pass.

## Calculation conventions

- Dose profiles are sampled by trilinear interpolation in the dose image's own
  physical geometry at 2,000 uniformly spaced positions along the saved ray.
- WEPL is integrated with the shared Siddon voxel-intersection traversal on
  the CT geometry. `num_samples` controls reporting positions, not integration
  accuracy.
- Reference R80 is the distal crossing of 80% of the reference profile peak,
  linearly interpolated between the first bracketing samples after the peak.
- The first-order estimator uses the reference SPR of the voxel segment
  containing reference R80.
- FiniteMap is obtained by inverse mapping the reference R80 WEPL coordinate
  through the sCT WEPL profile.
- RemappedProfile is obtained by remapping the reference dose profile through
  the same WEPL profiles and re-extracting R80.

`FiniteMap_mm` and `RemappedProfile_mm` are analytical reconstructions from the
reference dose profile. They are not independent perturbed-image Monte Carlo
or pencil-beam dose calculations. The audit verifies computational
reproducibility, not independent physical validation.

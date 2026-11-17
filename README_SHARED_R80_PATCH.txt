SynthQA_V2 shared R80 reconciliation patch
===========================================

Files to install (preserve the directory structure):

1. scripts/benchmark_dataset.py
2. scripts/audit_stratified_r80_case.py          [new]
3. src/synthqa_core/surrogates/stratified_r80.py [new]
4. tests/test_stratified_r80_shared.py           [new regression tests]

Why this patch is needed
------------------------
The previous benchmark result CSV did not preserve the selected beam/ray/beamlet,
energy, dose-file path, and exact ray coordinates. The audit could therefore
reselect a different candidate and still look like it was auditing the same
patient. In addition, the two scripts did not share a single R80 implementation.

This patch makes the benchmark CSV the source of truth for beamlet identity and
moves profile sampling, R80 extraction, Siddon WEPL evaluation, first-order
prediction, finite inverse mapping, and remapped-profile R80 into one shared
function. The audit reads the benchmark row, validates the identifiers and
ray geometry against the patient plan, then invokes that same function.

Install
-------
Copy the four files above into the matching locations in your local project.
The audit script is new. Do not run the audit against an old benchmark CSV.

Run
---
From the project root, with the project's normal virtual environment active:

    python scripts/benchmark_dataset.py
    python scripts/audit_stratified_r80_case.py --patient 1THB078
    python scripts/audit_stratified_r80_case.py --patient 1THB002

The benchmark overwrites:
    outputs/reproducibility/model_a_stratified_case_level.csv

The audit writes:
    outputs/reproducibility/r80_case_audits/<PATIENT>_shared_v2_audit.json
    outputs/reproducibility/r80_case_audits/<PATIENT>_shared_v2_profile.csv
    outputs/reproducibility/r80_case_audits/<PATIENT>_shared_v2_voxel_segments.csv

Audit PASS means the recomputed values match the benchmark CSV within 1e-6 mm
(or the relevant scalar units) and the plan identifiers, energy, dose file,
and ray coordinates match. A mismatch exits with code 2 and must be investigated.

Important scientific interpretation
------------------------------------
FiniteMap_mm and RemappedProfile_mm remain analytical WEPL remappings of the
reference dose profile. They are not independent perturbed-image MC/PB dose
calculations. The patch addresses selection and implementation consistency; it
does not change that scientific limitation.

Validation performed in the packaging environment
--------------------------------------------------
- Python compilation: benchmark, audit, shared module, and new tests passed.
- Targeted pytest suite: 10 passed.
- Full pytest collection could not complete because SimpleITK is not installed
  in the packaging environment (tests/test_phase1.py import error). Run the full
  suite in the user's project virtual environment before treating it as fully
  validated.

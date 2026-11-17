Targeted stratified R80 audit
============================

This diagnostic reproduces the candidate-selection and profile-remapping path
used by scripts/benchmark_dataset.py for one patient. It does not append to the
cohort manifest and does not overwrite cohort benchmark outputs.

From the repository root, run:

    python scripts/audit_stratified_r80_case.py --patient 1THB002

Optional output directory:

    python scripts/audit_stratified_r80_case.py --patient 1THB002 --output outputs/reproducibility/audit/1THB002

Outputs:
- diagnostics.json: selected beamlet, image-geometry checks, R80 crossing
  brackets, exact Siddon first-order calculation, inverse-WEPL map, and local
  gradient-window sensitivity.
- profile_audit.csv: sampled reference dose, WEPL reference/sCT profiles,
  remapped dose, and an additional dose-profile sampling using the dose image's
  own geometry metadata.
- voxel_segments.csv: exact Siddon segment lengths, voxel indices, reference
  and sCT SPR, and their WEPL contributions.

The script deliberately follows benchmark_dataset.py's existing candidate
selection: first four beams, first six rays per beam, energy-ranked candidate
beamlets, and the first valid dose profile among the top four. It samples the
benchmark profile using CT metadata to reproduce the existing result, while
also checking and testing dose-image metadata independently.

No output from this script is an independent perturbed-image MC/PB calculation.
The remapped profile and inverse-WEPL map are analytical reconstructions from
the original reference dose profile.

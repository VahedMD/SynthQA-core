# API Reference

## Coordinate conventions (read first)
- SimpleITK arrays: NumPy **(Z, Y, X)**; spacing/origin: **(X, Y, Z)**. Loaders label these
  explicitly (`shape_zyx`, `spacing_xyz`) — never mix them.
- `axis` / `beam_axis`: NumPy convention `0 = Z (SI)`, `1 = Y (AP)`, `2 = X (LR)`.
- DoseRAD2026 dose grids are anisotropic: 1 × 1 × 3 mm³.
- **Sample-count rule:** any 1-D ray analysis must use the same `num_samples` for the dose
  profile and the WEPL profile (default of `compute_wepl_along_ray` is 500; benchmarks use
  2000). Mismatches raise NumPy broadcasting errors.

## Implemented modules

### `synthqa_core.io.image_loader`
`load_mha(file_path) -> dict` with keys `array (Z,Y,X)`, `spacing_xyz`, `origin_xyz`,
`direction` (3×3 flat), `shape_zyx`.

### `synthqa_core.calibration`
- `DOSERAD_HU_DENSITY_CURVE`; `HUToDensityConverter(curve_data=None).convert(hu)` → g/cm³,
  clipped to [−1024, 4000] HU.
- `DensityToSPRConverter(method="linear_approx").convert(density)` → SPR ≈ ρ_rel.
- `CalibrationProfile(name)` / `.from_json(path)`; attributes `hu_to_density`,
  `density_to_spr`.

### `synthqa_core.surrogates`
- `compute_wepl_along_axis(spr_array, spacing_xyz, axis)` → cumulative WEPL (mm), cumsum.
- `compute_wepl_along_ray(spr_array, spacing_xyz, origin_xyz, ray_source, ray_target,
  num_samples=500)` → 1-D cumulative WEPL (mm); trilinear sampling; out-of-grid SPR = 0.
- `compute_spr_error(spr_ref, spr_sct)` → ΔSPR; `compute_delta_wepl_along_axis(...)`.

### `synthqa_core.risk`
- `compute_gradient_weighted_risk(delta_wepl, mc_dose, beam_axis, spacing)` → Gy·mm.
- `find_distal_edge(dose_profile, wepl_profile, threshold_percent=80)` → (edge, max_dose),
  sub-voxel linear interpolation. (`scripts/` define a thin `find_r80` wrapper on depth.)
- `predict_range_shift(wepl_ref, wepl_sct)`; `classify_risk(delta_wepl, dose_gradient,
  gradient_threshold=0.1)` → int8 classes 0/1/2.

### Planned (not yet implemented)
`io.plan_parser`, `io.dose_loader`, `io.dataset_manifest`, `metrics.gamma_index`,
`metrics.dose_difference`, `metrics.range_metrics`, `visualization.*`.

## Scripts

| Script | Purpose | Key outputs |
|---|---|---|
| `run_proton_qa.py` | CLI QA on local triplet | risk map, classes, timing |
| `synthetic_benchmark.py` | phantom end-to-end | r, timing, Fig console summary |
| `generate_figures.py` | manuscript Figs 1–4 | `figures/Fig1–4*.png` |
| `benchmark_dataset.py` | stratified real-data benchmark; `MODE = "uniform" \| "bone_only"` | Figs 6–7, stratified table |
| `benchmark_pyradplan.py` | head-to-head vs pyRadPlan (needs `pyRadPlan==0.3.5`) | Fig 8, Table 4 |
| `reviewer_stats.py` | bootstrap CIs, Bland–Altman, subgroups, cohort expansion | Figs 9–11, stats console |

## Testing
`pytest tests/ -v` — covers loader axis ordering, water equivalence (HU 0 → ≈1.006 g/cm³),
axis-aligned and arbitrary-ray WEPL on analytic phantoms, gradient-weighted risk behavior,
R80 recovery, threshold gating.
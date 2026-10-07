# API Reference

## Coordinate conventions (read first)

- SimpleITK arrays are returned as **NumPy (Z, Y, X)**; spacing/origin are **(X, Y, Z)**.
  Every loader labels these explicitly (`shape_zyx`, `spacing_xyz`) — never mix them.
- `axis` / `beam_axis` arguments use NumPy convention: `0 = Z (SI)`, `1 = Y (AP)`, `2 = X (LR)`.
- DoseRAD2026 dose grids are anisotropic: 1 × 1 × 3 mm³. Pass the spacing of the beam axis.

---

## `synthqa_core.io.image_loader`

```python
load_mha(file_path: str) -> dict
```
Returns `{"array" (Z,Y,X), "spacing_xyz", "origin_xyz", "direction" (3×3 flat), "shape_zyx"}`.

## `synthqa_core.calibration.hu_to_density`

```python
DOSERAD_HU_DENSITY_CURVE          # 10-point DoseRAD2026 Appendix B curve
HUToDensityConverter(curve_data=None)
    .convert(hu) -> np.ndarray    # g/cm³, clipped to [-1024, 4000] HU
```

## `synthqa_core.calibration.density_to_spr`

```python
DensityToSPRConverter(method="linear_approx")
    .convert(density) -> np.ndarray   # SPR ≈ ρ_rel (water = 1)
```
`method="schneider"` reserved for the stoichiometric profile (NotImplementedError).

## `synthqa_core.calibration.profiles`

```python
CalibrationProfile(profile_name="doserad2026_default")
    .hu_to_density : HUToDensityConverter
    .density_to_spr: DensityToSPRConverter
CalibrationProfile.from_json(path)  # keys: "name", "hu_to_density_curve", "density_to_spr_method"
```

## `synthqa_core.surrogates.wepl`

```python
compute_wepl_along_axis(spr_array, spacing_xyz, axis) -> np.ndarray
    # cumulative WEPL (mm) via cumsum; beam enters at index 0

compute_wepl_along_ray(spr_array, spacing_xyz, origin_xyz,
                       ray_source, ray_target, num_samples=500) -> np.ndarray
    # 1-D cumulative WEPL (mm) along an arbitrary physical ray; trilinear sampling,
    # out-of-bounds SPR = 0
```

## `synthqa_core.surrogates.spr_error`

```python
compute_spr_error(spr_ref, spr_sct) -> np.ndarray            # ΔSPR = SPR_sCT − SPR_ref
compute_delta_wepl_along_axis(delta_spr, spacing_xyz, axis)  # accumulated ΔWEPL (mm)
```

## `synthqa_core.risk.gradient_weighted`

```python
compute_gradient_weighted_risk(delta_wepl, mc_dose, beam_axis, spacing) -> np.ndarray
    # Risk = |ΔWEPL| × |∂D/(beam axis)|   (units Gy·mm)
```

## `synthqa_core.risk.range_shift`

```python
find_distal_edge(dose_profile, wepl_profile, threshold_percent=80.0) -> (edge, max_dose)
    # R80/R90 along a 1-D profile with sub-voxel linear interpolation
predict_range_shift(wepl_ref, wepl_sct) -> float              # ΔR (mm)
```

## `synthqa_core.risk.thresholds`

```python
classify_risk(delta_wepl, dose_gradient, gradient_threshold=0.1) -> np.ndarray[int8]
    # 0 low / 1 medium (1–3 mm) / 2 high (≥3 mm), gated on |∇D| ≥ threshold
```

---

## Scripts

| Script | Purpose | Example |
|---|---|---|
| `run_proton_qa.py` | CLI QA on real data | `python scripts/run_proton_qa.py --ct ct.mha --dose dose.mha [--sct sct.mha]` |
| `synthetic_benchmark.py` | self-contained validation + timing | `python scripts/synthetic_benchmark.py` |
| `generate_figures.py` | manuscript Figures 1–4 | `python scripts/generate_figures.py` |

## Testing

```bash
pytest tests/ -v
```
Unit tests cover axis-ordering of the loader, water equivalence of the calibration
(HU 0 → ≈1.006 g/cm³), axis-aligned and arbitrary-ray WEPL on analytic phantoms,
gradient-weighted risk behavior, R80 recovery, and threshold gating.
# API Reference

## Coordinate conventions (read first)

- SimpleITK arrays are **NumPy (Z, Y, X)**; spacing/origin are **(X, Y, Z)**.
- `axis` / `beam_axis`: `0 = Z (SI)`, `1 = Y (AP)`, `2 = X (LR)`.
- **Sign convention:** ΔWEPL > 0 ⇒ proximal shift; predicted depth shift = −ΔWEPL(peak).
- DoseRAD2026 proton dose grid: 1 × 1 × 3 mm³ (anisotropic).

*(io, calibration, surrogates, risk sections unchanged from v0.1.)*

---

## `synthqa_core.metrics.gamma_index`

```python
gamma_3d(ref, evl, dose_crit=0.03, dist_crit_mm=3.0, spacing_xyz=(1.,1.,3.),
         dose_threshold=0.10) -> {"gamma": ndarray, "pass_rate": float, "n_evaluated": int}
    # global normalization; shifted-array search within dist_crit; NaN outside mask
gamma_1d(ref, evl, dx_mm, dose_crit=0.03, dist_crit_mm=3.0,
         dose_threshold=0.10) -> {"gamma": ndarray, "pass_rate": float}
```

## `synthqa_core.metrics.dose_difference`

```python
absolute_difference(ref, evl) -> ndarray                 # Gy
relative_difference_global(ref, evl) -> ndarray          # % of max(ref)
relative_difference_local(ref, evl, eps=1e-8) -> ndarray # % of local ref (NaN where ref≈0)
summary(ref, evl, dose_threshold=0.10, delta_pct=2.0) -> dict
    # mean/max |ΔD|, mean relative diff, fraction exceeding delta_pct, n_evaluated
```

## `synthqa_core.metrics.range_metrics`

```python
distal_edge(depth_mm, dose, fraction=0.8) -> float       # sub-voxel R80/R90 of a profile
distal_edge_map(dose, spacing_axis_mm, axis=0, fraction=0.8) -> 2D ndarray   # mm, vectorized
range_shift_map(ref, evl, spacing_axis_mm, axis=0, fraction=0.8) -> 2D ndarray
    # evl edge − ref edge (mm); positive = distal shift; NaN where undefined
range_shift_stats(shift_map, mask=None) -> {"mean","std","p95","n"}
```

## `synthqa_core.visualization.slice_viewer`

```python
get_slice(volume, plane="axial", index=None, spacing_xyz=(1.,1.,3.)) -> (2D, extent_mm)
plot_slice(volume, plane=..., index=..., spacing_xyz=..., cmap="bone", vmin=None,
           vmax=None, overlay=None, overlay_cmap="inferno", overlay_alpha=0.55,
           overlay_floor=None, title="", cbar_label=None) -> (fig, ax)
compare_slices(volumes, titles, plane=..., index=..., spacing_xyz=...,
               cmap="bone", vmin=None, vmax=None) -> (fig, axes)
```

## `synthqa_core.visualization.wepl_heatmap`

```python
plot_wepl(wepl2d, extent=None, cmap="viridis", title="", cbar_label="WEPL (mm)") -> (fig, ax)
plot_delta_wepl(d2d, extent=None, vmax=None, cmap="bwr", title="",
                cbar_label="ΔWEPL (mm)") -> (fig, ax)   # symmetric limits ±vmax
```

## `synthqa_core.visualization.risk_overlay`

```python
RISK_CMAP   # black→red→yellow→white
overlay_risk(ct2d, risk2d, extent=None, ct_range=(-1024,2000), risk_floor=0.01,
             alpha=0.6, contours=(), legend_handles=(), title="") -> (fig, ax)
    # contours: iterable of (array2d, level, color), e.g. R80 of ref/sCT dose
overlay_classes(ct2d, cls2d, extent=None, ct_range=(-1024,2000),
                colors=(None,"yellow","red"), alpha=0.45, title="") -> (fig, ax)
```

---

## Scripts

| Script | Purpose | Outputs |
|---|---|---|
| `run_proton_qa.py` | CLI QA on a CT/sCT/dose triplet | risk map, classes, timing |
| `synthetic_benchmark.py` | self-contained analytical validation | r, timing |
| `generate_figures.py` | phantom, Bragg profiles, overlay, scatter | Figs 1–4 |
| `benchmark_dataset.py` | stratified DoseRAD2026 benchmark (targeted HF download, bone-targeted rays, proximal-bone tracking) | Figs 6–7, sensitivity/specificity table |
| `benchmark_pyradplan.py` | head-to-head vs pyRadPlan Hong PB (energy snapping included) | Fig 8, Table 4 |
| `reviewer_stats.py` | bootstrap CIs, Bland–Altman, Mann–Whitney subgroups, cohort expansion | Figs 9–11 |

## Testing

```bash
pytest tests/ -v
```
Suites: `test_phase1.py` (loader axes, water equivalence), `test_phase2_3.py` (SPR, axis &
ray WEPL), `test_phase4.py` (risk formula, R80, thresholds),
`test_metrics_visualization.py` (gamma identity/shift, dose-difference stats,
vectorized range maps, visualization smoke tests).

## Troubleshooting

- pyRadPlan requires beamlet energies snapped to `machine.energies` (else `KeyError`).
- matplotlib ≥ 3.10: `boxplot(labels=...)` removed → `set_xticklabels`.
- HF: set `HF_TOKEN`; Windows symlink warning is harmless
  (`HF_HUB_DISABLE_SYMLINKS_WARNING=1`).
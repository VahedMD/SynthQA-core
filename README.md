# synthqa-core

**Millisecond-cost physics surrogates for synthetic CT quality assurance in proton therapy.**

`synthqa-core` is a pure-NumPy Python package that predicts and spatially localizes proton
dosimetric risk caused by synthetic CT (sCT) inaccuracies — **without running Monte Carlo**.
It is the software companion to **Article 1** of a two-paper series on synthetic CT QA for
radiotherapy (proton/hadron focus).

> **Research question.** *Can millisecond-cost physics surrogates (SPR/WEPL maps) predict
> proton range uncertainty as accurately as full Monte Carlo dose recalculation — and
> spatially localize where the risk is?*

> **Core hypothesis.** A gradient-weighted WEPL risk map, computed in milliseconds per
> beamlet, predicts the spatial distribution of proton dose errors caused by sCT
> inaccuracies, validated against Geant4 Monte Carlo and pencil-beam ground truth.

## The method in one formula

```
Risk(x, y, z) = |ΔWEPL(x, y, z)| × |∂D_ref/∂z|
```

- `ΔWEPL` — accumulated stopping-power-ratio error along the beam ray up to depth `z`
- `∂D_ref/∂z` — dose gradient of the reference (MC) dose along the beam direction

The product localizes risk to **distal edges** (high gradient), ignores entrance regions
(low gradient), and fires only on **rays that traverse the erroneous anatomy** (ΔWEPL ≠ 0).

```
CT (HU) ──HU→ρ──▶ ρ (g/cm³) ──ρ→SPR──▶ SPR ──ray integral──▶ WEPL ──▶ ΔWEPL
                                                                       │
MC dose D_ref ────────────────∂/∂z──────────────────────────▶ |∂D/∂z|  │
                                                                 └──×──▶ Risk map ──▶ QA classes
```

**Sign convention:** positive ΔWEPL (sCT overestimates SPR) ⇒ **proximal** range shift;
predicted depth shift = −ΔWEPL evaluated at the Bragg peak.

## Headline results (DoseRAD2026, Geant4 ground truth)

| Endpoint | n | Result |
|---|---|---|
| Range-shift prediction vs analytical MC (cohort expansion) | 17 | r = 0.983 (95% CI 0.955–0.996); MAE 0.333 mm (0.248–0.422) |
| Range-shift prediction vs pyRadPlan (Hong PB) | 11 | MAE 0.367 mm (0.033–0.955); Bland–Altman bias −0.24 mm, LoA [−2.13, +1.65]; excl. one boundary-crossing outlier: bias +0.05 mm, LoA [−0.26, +0.36] |
| Specificity (rays missing the perturbed bone) | 8 | mean \|predicted shift\| = 0.01 mm |
| Subgroup robustness | 11 | metal vs clean p = 0.460 (NS); thorax vs abdomen p = 0.034 → 0.056 after outlier exclusion |
| Computation time | per beamlet | surrogate ≈ 1–14 ms vs pyRadPlan ≈ 124 s (**≈ 8,900× speedup**; ~10⁵× vs Geant4) |

## Installation

```bash
git clone https://github.com/VahedMD/SynthQA-core.git && cd synthqa-core
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -e .[dev]
# optional, for the head-to-head benchmark:
pip install pyRadPlan==0.3.5 huggingface_hub
```

Requires Python ≥ 3.9. No GPU, no Monte Carlo, no web frameworks.

## Quickstart

```python
import numpy as np
from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.spr_error import compute_spr_error
from synthqa_core.surrogates.wepl import compute_wepl_along_axis
from synthqa_core.risk.gradient_weighted import compute_gradient_weighted_risk
from synthqa_core.risk.thresholds import classify_risk
from synthqa_core.metrics.range_metrics import range_shift_map
from synthqa_core.visualization.risk_overlay import overlay_classes

ct, sct, dose = (load_mha(p) for p in ["ct.mha", "sct.mha", "dose.mha"])
prof = CalibrationProfile()                      # DoseRAD2026 calibration by default
spr = lambda a: prof.density_to_spr.convert(prof.hu_to_density.convert(a))
sp = ct["spacing_xyz"]                           # (X, Y, Z); arrays are (Z, Y, X)!

dwepl = compute_wepl_along_axis(compute_spr_error(spr(ct["array"]), spr(sct["array"])), sp, axis=0)
risk  = compute_gradient_weighted_risk(dwepl, dose["array"], beam_axis=0, spacing=sp[2])
cls   = classify_risk(dwepl, np.abs(np.gradient(dose["array"], sp[2], axis=0)))
fig, ax = overlay_classes(ct["array"][:, ct["array"].shape[1]//2, :],
                          cls[:, cls.shape[1]//2, :], title="QA decision map")
```

## Repository structure

```
src/synthqa_core/
├── calibration/   HU→density (DoseRAD2026 curve), density→SPR, JSON profiles
├── surrogates/    WEPL line integrals (axes + arbitrary rays), ΔSPR/ΔWEPL
├── risk/          gradient-weighted risk, R80 range shift, clinical thresholds
├── io/            SimpleITK .mha loader
├── metrics/       3D/1D gamma index, dose-difference stats, vectorized range-shift maps
└── visualization/ slice viewer, WEPL/ΔWEPL heatmaps, risk & class overlays
scripts/
├── run_proton_qa.py         CLI QA on a CT/sCT/dose triplet
├── synthetic_benchmark.py   self-contained analytical validation + timing (Figs 1–4)
├── generate_figures.py      manuscript figures 1–4
├── benchmark_dataset.py     stratified DoseRAD2026 benchmark, targeted HF download (Figs 6–7)
├── benchmark_pyradplan.py   head-to-head vs pyRadPlan Hong PB (Fig 8, Table 4)
└── reviewer_stats.py        bootstrap CIs, Bland–Altman, subgroup tests, cohort expansion (Figs 9–11)
tests/                       pytest suite (I/O, calibration, WEPL, risk, metrics, visualization)
docs/                        physics_background.md, api_reference.md
configs/                     calibration curve JSON profiles
```

## Datasets

| Dataset | Role | License |
|---|---|---|
| [SynthRAD2025](https://zenodo.org/records/14918089) | paired CT/MRI source images | CC BY-NC 4.0 |
| [DoseRAD2026](https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026) | Geant4 proton/photon beamlet doses + beam JSONs | CC BY-NC 4.0 |

- The **DoseRAD2026 HU→density curve** (Appendix B) is the default calibration profile and
  must be used for reproducibility with the Geant4 ground truth. It is deliberately
  **non-monotonic at 120→121 HU** — do not "fix" it.
- Proton dose grid: 1 × 1 × 3 mm³ (anisotropic); MC statistical uncertainty < 2 %.
- **Targeted download:** the repo is ~850 GB; the benchmark scripts fetch only the needed
  files via `huggingface_hub.hf_hub_download` with paths
  `proton/training/{pid}/{pid}.json`, `.../image/ct.mha`, `.../dose/Dose_B{b}_R{r}_L{l}.mha`,
  cached under `~/.cache/huggingface/hub`. Set `HF_TOKEN` for higher rate limits.
- Test set embargoed until March 2030 — all validation uses the training split.
- 13 training patients carry metal implants (list in the dataset paper, Appendix C);
  they are flagged in all subgroup analyses.

## Testing & benchmarking

```bash
pytest tests/ -v                        # unit + metric + visualization tests
python scripts/synthetic_benchmark.py   # end-to-end analytical validation + timing
python scripts/generate_figures.py      # figures 1–4
python scripts/benchmark_dataset.py     # stratified DoseRAD2026 benchmark (figures 6–7)
python scripts/benchmark_pyradplan.py   # head-to-head vs pyRadPlan (figure 8, table 4)
python scripts/reviewer_stats.py        # bootstrap CIs, Bland–Altman, subgroups (figures 9–11)
```

## Documentation

- [`docs/physics_background.md`](docs/physics_background.md) — derivations, sign
  conventions, validity regime (Taylor breakdown), validation methodology, limitations.
- [`docs/api_reference.md`](docs/api_reference.md) — full API, coordinate conventions,
  script reference, troubleshooting.

## Status / roadmap

- ✅ Phases 1–6: I/O, calibration, WEPL surrogates, risk maps, validation, figures.
- ✅ Stratified real-data benchmark + pyRadPlan head-to-head + reviewer statistics.
- ✅ `metrics/` and `visualization/` modules implemented and tested.

## Troubleshooting

- **HF rate limits / slow downloads:** set `HF_TOKEN`; downloads are cached after first run.
- **Windows symlink warning from `huggingface_hub`:** harmless; set
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- **pyRadPlan `KeyError: <energy>`:** beamlet energies must be snapped to the nearest
  machine energy (`machine.energies`) before building the beam — `benchmark_pyradplan.py`
  does this automatically.
- **matplotlib ≥ 3.10:** `boxplot(labels=...)` was removed; use `set_xticklabels`.
- **Axis order:** SimpleITK arrays are (Z, Y, X); spacing/origin are (X, Y, Z). Every
  loader labels these explicitly — never mix them.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code, please also cite the
SynthRAD2025 and DoseRAD2026 datasets.

## License

Code: **MIT** (see [`LICENSE`](LICENSE)).
Underlying datasets: **CC BY-NC 4.0** — derived validation results inherit their
non-commercial terms.
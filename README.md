# synthqa-core

**Millisecond-cost physics surrogates for synthetic CT quality assurance in proton therapy.**

`synthqa-core` is a pure-NumPy Python package that predicts and spatially localizes proton
dosimetric risk caused by synthetic CT (sCT) inaccuracies — **without running Monte Carlo**.
It is the software companion to **Article 1** of a two-paper series on synthetic CT QA for
radiotherapy (proton/hadron focus).

> **Research question.** *Can millisecond-cost physics surrogates (SPR/WEPL maps) predict
> proton range uncertainty as accurately as full Monte Carlo dose recalculation — and
> spatially localize where the risk is?*

> **Core hypothesis.** A gradient-weighted WEPL risk map, computed in <10 ms per patient,
> predicts the spatial distribution of proton dose errors caused by sCT inaccuracies,
> validated against Geant4 Monte Carlo ground truth.

## The method in one formula

```
Risk(x, y, z) = |ΔWEPL(x, y, z)| × |∂D_ref/∂z|
```

- `ΔWEPL` — accumulated stopping-power-ratio error along the beam ray up to depth `z`
- `∂D_ref/∂z` — dose gradient of the reference (MC) dose along the beam direction

The product localizes risk to **distal edges** (high gradient) and ignores entrance regions
(low gradient), and to **rays that actually traverse the erroneous anatomy** (nonzero ΔWEPL).

```
CT (HU) ──HU→ρ──▶ ρ (g/cm³) ──ρ→SPR──▶ SPR ──ray integral──▶ WEPL ──▶ ΔWEPL
                                                                       │
MC dose D_ref ────────────────∂/∂z──────────────────────────▶ |∂D/∂z|  │
                                                                 └──×──▶ Risk map ──▶ QA classes
```

## Key results (synthetic phantom benchmark)

| Metric | Value |
|---|---|
| Pearson *r* (predicted risk vs. actual dose error) | **0.94** (R² = 0.89) |
| Voxels analyzed | 668,800 |
| Risk localization | distal edge only; entrance channel silent |
| Computation time | few ms per 100³ volume, single CPU core (target < 10 ms) |

## Installation

```bash
git clone <your-repo-url> && cd synthqa-core
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -e .[dev]
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

ct   = load_mha("ct.mha")
sct  = load_mha("sct.mha")
dose = load_mha("dose.mha")          # reference MC dose
prof = CalibrationProfile()          # DoseRAD2026 calibration by default

spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct["array"]))
spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(sct["array"]))
spacing = ct["spacing_xyz"]          # (X, Y, Z) — arrays are (Z, Y, X)!

dwepl = compute_wepl_along_axis(compute_spr_error(spr_ref, spr_sct), spacing, axis=0)
risk  = compute_gradient_weighted_risk(dwepl, dose["array"], beam_axis=0, spacing=spacing[2])
grad  = np.abs(np.gradient(dose["array"], spacing[2], axis=0))
cls   = classify_risk(dwepl, grad)   # 0 = low, 1 = medium, 2 = high risk
```

## Repository structure

```
src/synthqa_core/
├── calibration/   HU→density (DoseRAD2026 curve), density→SPR, JSON profiles
├── surrogates/    WEPL line integrals (axes + arbitrary rays), ΔSPR/ΔWEPL
├── risk/          gradient-weighted risk, R80 range shift, clinical thresholds
├── io/            SimpleITK .mha loader, (future: DoseRAD plan parser, manifests)
├── metrics/       (planned) gamma index, dose difference, range metrics
└── visualization/ (planned) slice viewer, WEPL heatmaps, risk overlays
scripts/
├── run_proton_qa.py        CLI entry point for real CT/sCT/dose triplets
├── synthetic_benchmark.py  self-contained end-to-end validation benchmark
└── generate_figures.py     manuscript figures 1–4
tests/                      pytest suite (phases 1–4)
docs/                       physics_background.md, api_reference.md
configs/                    calibration curve JSON profiles
```

## Datasets

| Dataset | Role | License |
|---|---|---|
| [SynthRAD2025](https://zenodo.org/records/14918089) | paired CT/MRI source images | CC BY-NC 4.0 |
| [DoseRAD2026](https://doi.org/10.5281/zenodo.19347848) | Geant4 proton beamlet doses + beam JSONs | CC BY-NC 4.0 |

- The **DoseRAD2026 HU→density curve** (Appendix B) is hard-coded as the default
  calibration profile and must be used for reproducibility with the Geant4 ground truth.
  Note it is deliberately **non-monotonic at 120→121 HU** — do not "fix" it.
- Dose grid: 1 × 1 × 3 mm³ (anisotropic). MC statistical uncertainty < 2 %.
- **Test set is embargoed until March 2030** — all validation uses the training set.
- 13 training patients carry metal implants; flag them in subgroup analyses.

## Testing & benchmarking

```bash
pytest tests/ -v                      # unit tests (phases 1–4)
python scripts/synthetic_benchmark.py # end-to-end validation + timing
python scripts/generate_figures.py    # regenerate manuscript figures
```

## Documentation

- [`docs/physics_background.md`](docs/physics_background.md) — physics derivation,
  assumptions, and limitations.
- [`docs/api_reference.md`](docs/api_reference.md) — full API and coordinate conventions.

## Status / roadmap

- ✅ Phases 1–4: I/O, calibration, WEPL surrogates, risk maps (tested).
- ✅ Phase 5–6: synthetic benchmark (r = 0.94) and figure generation.
- ⏳ Full DoseRAD2026 training-set benchmark (75 patients, 81k beamlets).
- ⏳ pyRadPlan pencil-beam comparison; gamma / range metrics modules; visualization module.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code, please also cite the
SynthRAD2025 and DoseRAD2026 datasets.

## License

Code: **MIT** (see [`LICENSE`](LICENSE)).
Underlying datasets: **CC BY-NC 4.0** — derived validation results inherit their
non-commercial terms.
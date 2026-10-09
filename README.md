# synthqa-core

**Millisecond-cost physics surrogates for synthetic CT quality assurance in proton therapy.**

`synthqa-core` predicts and spatially localizes proton dosimetric risk caused by synthetic
CT (sCT) inaccuracies **without running Monte Carlo or pencil-beam dose recalculation**.
It is the software companion to Article 1 of a two-paper series on sCT QA for radiotherapy
(proton/hadron focus).

> **Research question.** Can millisecond-cost physics surrogates (SPR/WEPL maps) predict
> proton range uncertainty as accurately as full Monte Carlo dose recalculation — and
> spatially localize where the risk is?

> **Core hypothesis.** A gradient-weighted WEPL risk map, computed in milliseconds per
> beamlet, predicts the spatial distribution of proton dose errors caused by sCT
> inaccuracies, validated against Geant4 Monte Carlo and pencil-beam ground truth.

## The method in one formula

```
Risk(x, y, z) = |ΔWEPL(x, y, z)| × |∂D_ref/∂z|
```

- `ΔWEPL` — accumulated stopping-power-ratio error along the beam ray up to depth z
- `∂D_ref/∂z` — dose gradient of the reference (MC) dose along the beam direction

The product localizes risk to **distal edges** (high gradient), to **rays that traverse
the erroneous anatomy** (nonzero ΔWEPL), and stays silent in entrance channels and
error-free tissue.

```
CT (HU) ─HU→ρ─▶ ρ (g/cm³) ─ρ→SPR─▶ SPR ─ray integral─▶ WEPL ─▶ ΔWEPL
                                                                    │
MC dose D_ref ───────────────∂/∂z──────────────────────▶ |∂D/∂z|   │
                                                            └──×──▶ Risk ─▶ QA classes / R80 shift
```

## Headline results (DoseRAD2026 proton training set)

| Validation arm | n | Metric | Result |
|---|---|---|---|
| Surrogate vs analytical MC range shift (cohort expansion) | 17 | Pearson r | **0.983** (95% CI 0.955–0.996) |
| | | MAE | **0.333 mm** (95% CI 0.248–0.422) |
| Surrogate vs pyRadPlan Hong pencil beam | 11 | MAE | 0.367 mm (95% CI 0.033–0.955) |
| | | Bland–Altman bias / LoA | −0.24 mm / [−2.13, +1.65] mm |
| | | — excluding 1 boundary case | +0.05 mm / [−0.26, +0.36] mm |
| Bone-only error, rays crossing bone (sensitivity) | 3 | r / MAE | 0.928 / 0.56 mm |
| Bone-only error, water rays (specificity) | 8 | pred vs actual shift | 0.01 vs 0.02 mm |
| Subgroup robustness | 11 | metal vs clean (Mann–Whitney) | p = 0.460 (NS) |
| Computation | 11 | surrogate vs PB (ref+sCT) | **14 ms vs 124 s ≈ 8,900× speedup** |

Synthetic-phantom voxel-wise validation: r = 0.94 (R² = 0.89).

## Installation

```bash
git clone https://github.com/VahedMD/SynthQA-core.git && cd synthqa-core
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # or: pip install -e .[dev]
# For the pencil-beam head-to-head benchmark only:
pip install pyRadPlan==0.3.5 huggingface_hub
```

Requires Python ≥ 3.9. No GPU, no Monte Carlo.

## Quickstart

```python
import numpy as np
from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.spr_error import compute_spr_error
from synthqa_core.surrogates.wepl import compute_wepl_along_axis
from synthqa_core.risk.gradient_weighted import compute_gradient_weighted_risk
from synthqa_core.risk.thresholds import classify_risk

ct   = load_mha("ct.mha");  sct = load_mha("sct.mha");  dose = load_mha("dose.mha")
prof = CalibrationProfile()                      # DoseRAD2026 calibration by default
spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct["array"]))
spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(sct["array"]))
sp = ct["spacing_xyz"]                           # (X, Y, Z); arrays are (Z, Y, X)!

dwepl = compute_wepl_along_axis(compute_spr_error(spr_ref, spr_sct), sp, axis=0)
risk  = compute_gradient_weighted_risk(dwepl, dose["array"], beam_axis=0, spacing=sp[2])
cls   = classify_risk(dwepl, np.abs(np.gradient(dose["array"], sp[2], axis=0)))
```

## Repository structure (implemented vs planned)

```
src/synthqa_core/
├── calibration/   ✅ hu_to_density (DoseRAD2026 curve), density_to_spr, JSON profiles
├── surrogates/    ✅ wepl (axis + arbitrary-ray line integrals), spr_error
├── risk/          ✅ gradient_weighted, range_shift (R80/R90), thresholds
├── io/            ✅ image_loader   ⏳ plan_parser, dose_loader, dataset_manifest
├── metrics/       ⏳ gamma_index, dose_difference, range_metrics
└── visualization/ ⏳ slice_viewer, wepl_heatmap, risk_overlay
scripts/
├── run_proton_qa.py        CLI QA on a local CT/sCT/dose triplet
├── synthetic_benchmark.py  analytical phantom end-to-end validation (r = 0.94)
├── generate_figures.py     manuscript Figures 1–4
├── benchmark_dataset.py    stratified real-data benchmark (uniform / bone-only error)
├── benchmark_pyradplan.py  head-to-head vs pyRadPlan Hong PB (Figure 8, Table 4)
└── reviewer_stats.py       bootstrap CIs, Bland–Altman, subgroups, cohort expansion
tests/                      pytest suite (phases 1–4)
configs/                    calibration curve JSON profiles
docs/                       physics_background.md, api_reference.md, benchmarking.md
```

## Datasets

| Dataset | Role | License | Access |
|---|---|---|---|
| [SynthRAD2025](https://zenodo.org/records/14918089) | paired CT/MRI source images | CC BY-NC 4.0 | Zenodo |
| [DoseRAD2026](https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026) | Geant4 proton beamlet doses + beam JSONs | CC BY-NC 4.0 | Hugging Face |

- **Targeted downloads only.** The proton subset lives at
  `proton/training/{PatientID}/{image/ct.mha, dose/Dose_B*_R*_L*.mha, {PatientID}.json}`.
  The benchmark scripts use `huggingface_hub.hf_hub_download` per file (≈30–40 MB/patient)
  and cache locally — the ~850 GB repository is never cloned. Set `HF_TOKEN` for higher
  rate limits.
- The **DoseRAD2026 HU→density curve** (Appendix B) is the default calibration profile;
  it is deliberately non-monotonic at 120→121 HU — do not "fix" it.
- Dose grid 1×1×3 mm³; MC statistical uncertainty < 2%; **test set embargoed until
  March 2030** (training-only validation); 13 training patients carry metal artifacts
  (flagged in all stratified analyses).

## Testing & reproduction

```bash
pytest tests/ -v                          # unit tests (loader axes, calibration, WEPL, risk)
python scripts/synthetic_benchmark.py     # phantom validation + timing
python scripts/generate_figures.py        # Figures 1–4
python scripts/benchmark_dataset.py       # Figures 6–7 (real patients, stratified)
python scripts/benchmark_pyradplan.py     # Figure 8, Table 4 (needs pyRadPlan)
python scripts/reviewer_stats.py          # Figures 9–11 + bootstrap/subgroup statistics
```

See [`docs/benchmarking.md`](docs/benchmarking.md) for expected downloads, runtimes,
outputs, and troubleshooting.

## Documentation

- [`docs/physics_background.md`](docs/physics_background.md) — derivation, sign
  conventions, validity regime (Taylor breakdown), error models, statistical methodology.
- [`docs/api_reference.md`](docs/api_reference.md) — full API, coordinate conventions,
  script usage.
- [`docs/benchmarking.md`](docs/benchmarking.md) — benchmark pipeline, beamlet
  pre-selection, troubleshooting.

## Status / roadmap

- ✅ Phases 1–6: I/O, calibration, WEPL surrogates, risk maps, phantom validation, figures.
- ✅ Real-data validation: stratified DoseRAD2026 cohort, cohort expansion (n = 17).
- ✅ Head-to-head vs pyRadPlan (≈8,900× speedup) and reviewer-grade statistics.
- ⏳ Manuscript (Article 1) in writing; remaining package modules (metrics, visualization,
  plan/dose loaders); full 75-patient PB arm; Article 2 (photon/VMAT extension).

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the SynthRAD2025 and DoseRAD2026
dataset publications when using the benchmarks.

## License

Code: **MIT** ([LICENSE](LICENSE)). Underlying datasets: **CC BY-NC 4.0** — derived
validation results inherit their non-commercial terms.
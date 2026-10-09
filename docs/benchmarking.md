# Benchmarking Guide

## Pipeline map (which script answers which question)
1. `synthetic_benchmark.py` — Does the formula work at all? (phantom, r = 0.94, <10 ms)
2. `generate_figures.py` — Can we show localization visually? (Figs 1–4)
3. `benchmark_dataset.py` — Does it hold on real anatomy and realistic error topologies?
   (Figs 6–7; uniform and bone-only models; sensitivity/specificity)
4. `benchmark_pyradplan.py` — Is it faster and as accurate as a deterministic engine?
   (Fig 8, Table 4; ≈8,900× speedup)
5. `reviewer_stats.py` — Are the statistics review-proof? (Figs 9–11; CIs, Bland–Altman,
   subgroups, cohort expansion n = 17)

## Data access (targeted downloads)
- Repo: `LMUK-RADONC-PHYS-RES/DoseRAD2026` (Hugging Face, CC BY-NC 4.0).
- Proton paths: `proton/training/{PID}/{PID}.json`, `.../image/ct.mha`,
  `.../dose/Dose_B{b}_R{r}_L{l}.mha`.
- `list_repo_files` discovers patient folders; `hf_hub_download` fetches single files
  (≈30–40 MB/patient) into the HF cache — the ~850 GB repo is never cloned.
- Set `HF_TOKEN` (public repo, but raises rate limits); on Windows set
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1` (script does this automatically).

## Beamlet pre-selection (avoids blind downloads)
1. Compute ray WEPL cheaply (400 samples) for candidate rays.
2. Predict water range per beamlet energy: `R_mm ≈ 0.022 · E^1.77`.
3. Rank beamlets by |R(E) − 0.9 · ray WEPL| (peak deep but inside profile).
4. Accept only profiles with peak index > 20% of samples, peak > 1e-3, and > 90% distal
   fall-off; for bone studies require proximal bone (HU > 300) before the peak.

## Expected runtimes and volumes
| Script | Downloads | Runtime (cached / uncached) |
|---|---|---|
| synthetic_benchmark | none | seconds |
| generate_figures | none | seconds |
| benchmark_dataset | ~12 patients × 3 files | minutes uncached, <1 min cached |
| benchmark_pyradplan | ~12 patients + PB runs | ~30–40 min (PB ≈ 1–4 min/patient pair) |
| reviewer_stats | ~17–30 patients | ~5–10 min |

## Troubleshooting
- **404 on patient folders:** IDs are non-contiguous; always discover via
  `list_repo_files`, never guess sequential IDs.
- **`Found 0 patients`:** wrong path prefix; proton data lives under `proton/training/`.
- **Broadcasting error (500,) vs (2000,):** mismatched `num_samples` between dose profile
  and WEPL (see api_reference conventions).
- **`nan` R80:** shifted peak left the sampled ray — check sign convention
  (`W + ΔW`, §9 of physics_background) or extend sampling.
- **numpy boolean negation error:** use `0.0 if cond else 1.0`, not `-cond`.
- **matplotlib ≥3.10 boxplot:** `labels=` removed; use `set_xticklabels`.
- **pyRadPlan `KeyError: <energy>`:** snap JSON energy to the machine energy grid before
  building the `Beam` object.

## Extending
- Change error model: `MODE`, `BONE_THRESH`, `BONE_SHIFT`, `UNIFORM_SHIFT` at the top of
  `benchmark_dataset.py`.
- Grow the cohort: raise `n_patients` in `reviewer_stats.run_cohort_expansion` and widen
  the candidate search (more beams/rays/energy layers).
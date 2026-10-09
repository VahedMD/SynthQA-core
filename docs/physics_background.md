# Physics Background

## 1. The problem: proton range in synthetic CT

Proton dose deposition terminates at the Bragg peak, whose position depends on the
integral of the relative stopping power along the beam path. Unlike photons, where HU
errors mostly perturb attenuation smoothly, a small systematic SPR bias in an sCT
**translates the distal edge** — potentially moving a high-dose region into or out of an
organ at risk. QA for sCT in proton therapy is therefore fundamentally a **range
uncertainty** problem, not merely a voxel-wise HU problem.

## 2. HU → mass density

Mass density is obtained by piecewise-linear interpolation of a calibration curve.
The default curve is the 10-point **DoseRAD2026 Appendix B** curve, because the Geant4
ground truth was generated with exactly this curve. Two reproducibility caveats:

1. The curve is **non-monotonic** between HU 120 (1.1266 g/cm³) and 121 (1.0951 g/cm³).
   This is intentional; interpolation in HU remains well-posed.
2. Out-of-range values are clipped to [−1024, 4000] HU before interpolation.

## 3. Mass density → stopping power ratio

The proton stopping power ratio follows from the Bethe equation; to first order

```
SPR ≈ ρ_rel · ([Z/A]_tissue / [Z/A]_water)
```

For soft tissue and water, Z/A ≈ 0.54–0.555, so the correction factor is ≈ 1 and
**SPR ≈ ρ_rel** (the `linear_approx` method). This is accurate for soft tissue; bone
introduces a few-percent bias, which is why a stoichiometric (Schneider-type)
conversion is planned as an alternative profile.

## 4. WEPL and ΔWEPL

The water-equivalent path length is the ray integral of SPR:

```
WEPL(z) = ∫₀ᶻ SPR(z′) dz′
```

Implementation:
- **Principal axes:** `numpy.cumsum` × voxel size — O(N), exact for axis-aligned beams.
- **Arbitrary gantry angles:** ray marching with trilinear sampling
  (`scipy.ndimage.map_coordinates`, order 1), cumulative sum × step length.

The range-relevant quantity is the *difference* between sCT and reference CT:

```
ΔWEPL(z) = ∫₀ᶻ [SPR_sCT − SPR_CT] dz′
```

At the distal edge, ΔWEPL is the **predicted range shift** ΔR (first order).

## 5. Gradient-weighted risk (the core surrogate)

Assume the sCT dose is a depth-shifted version of the reference dose,
`D_sCT(z) ≈ D_ref(z − ΔWEPL)`. A first-order Taylor expansion gives

```
ΔD(z) ≈ −(∂D_ref/∂z) · ΔWEPL   ⇒   |ΔD| ≈ |∂D_ref/∂z| · |ΔWEPL| = Risk
```

Consequences that make the map clinically meaningful:

- **Entrance channel silence.** There, |∂D/∂z| is small, so even large |ΔWEPL| yields
  negligible risk — correct, because a range shift on a flat plateau changes dose little.
- **Distal edge localization.** The fall-off has the largest gradient in a proton plan,
  so risk concentrates exactly where range errors matter.
- **Lateral selectivity.** Rays that never traverse the erroneous anatomy have
  ΔWEPL = 0 and zero risk, even where their own distal gradient is steep.

### Assumptions and limitations

1. **First-order linearization.** When |ΔWEPL| becomes a substantial fraction of the
   fall-off width (σ), curvature breaks the approximation (visible as heteroscedastic
   scatter at the high end of the risk–error correlation plot). The R80 range-shift
   metric is provided as an exact companion at the edge.
2. **Single dominant beam direction** per risk map; the gradient is taken along the ray.
   For multi-field plans, compute per-beam maps and combine (e.g., max).
3. **Density-only physics.** Nuclear interaction and LET changes from composition errors
   are ignored (acceptable at the few-percent level for sCT QA).

## 6. Risk classification

| Class | Criterion |
|---|---|
| 0 Low | \|ΔWEPL\| < 1 mm, or gradient below threshold |
| 1 Medium | 1 mm ≤ \|ΔWEPL\| < 3 mm **and** high gradient |
| 2 High | \|ΔWEPL\| ≥ 3 mm **and** high gradient |

The 1/3 mm bands are chosen against typical clinical range margins
(~3.5 %·R + 1–3 mm): a ≥3 mm localized error can consume a full margin.

## 7. Validation methodology

Because MC doses on perturbed sCTs are not available, ground truth dose error is
obtained by re-evaluating the (analytical or pencil-beam) dose on the perturbed SPR map.
Metrics: Pearson *r* and R² between `Risk` and |ΔD| over voxels with dose > 5 % of max;
R80 shifts per ray; sensitivity/specificity of the risk classes for flagging > 2 % dose
errors; wall-clock time per volume (target < 10 ms).


## 8. One-dimensional ray-tracing validation framework
DoseRAD2026 beams use 36 arbitrary gantry angles, so 3-D axis-aligned gradients are
invalid. Validation therefore extracts 1-D physical profiles:
- dose profile: trilinear sampling (`scipy.ndimage.map_coordinates`, order 1) of the 3-D
  MC beamlet dose along `ray_source → ray_target`, 2000 samples (≈0.1 mm);
- WEPL profile: `compute_wepl_along_ray` with the **same** sample count (mismatched
  `num_samples` between dose and WEPL arrays is a broadcasting error, not physics);
- beamlet pre-selection without downloads: water range–energy fit
  `R_mm(E) ≈ 0.022 · E^1.77` matched to 0.9 × total ray WEPL, so the Bragg peak lands
  deep but inside the sampled profile;
- proximal-bone accounting: bone path length is counted only **up to the Bragg peak**;
  bone distal to the peak cannot affect range and must not be counted.

## 9. Sign conventions and the shift model
With ΔWEPL = WEPL_sCT − WEPL_ref, the perturbed depth–dose is
`D_sCT(z) = D_ref` evaluated at WEPL `W(z) + ΔWEPL(z)` (i.e. `np.interp(W + ΔW, W, D_ref)`):
higher SPR stops protons earlier (proximal shift, negative depth shift), lower SPR
penetrates deeper (distal shift, positive depth shift). The predicted depth shift is
`−ΔWEPL` at the reference peak. Using `W − ΔW` inverts the physics and can push the
shifted peak outside the sampled ray (a common source of spurious `nan` R80 values).

## 10. Validity regime: the Taylor breakdown
Risk = |ΔWEPL|·|∇D| is a first-order expansion, valid while |ΔWEPL| is small compared with
the distal fall-off width. For single beamlets (fall-off ≈ 3–4 mm) and large global errors
(e.g. +50 HU ⇒ ≈10 mm shift), the voxel-wise error becomes a step function while Risk is a
bell curve: voxel-wise correlation collapses (r ≈ 0.1–0.5) even though the **range-shift
endpoint remains exact**. Consequently:
- primary endpoint = R80 displacement (valid at any shift magnitude);
- voxel-wise gated correlation is reported only in the small-shift regime
  (bone-only perturbations: r_gated ≈ 0.99);
- at heterogeneity-boundary crossings, 1-D WEPL is a lower bound (ignores lateral
  scatter) and pencil beam an upper bound; the surrogate is then a trigger for
  recalculation, not a millimetre-equivalent substitute (case 1THB002: surrogate 0.58 mm,
  MC 2.22 mm, PB 3.70 mm).

## 11. sCT error models
- **Uniform bias (+50 HU):** global systematic error; maximizes ΔWEPL variance; used for
  the range-shift and head-to-head arms.
- **Bone-only (−200 HU for HU > 300):** representative deep-learning failure (cortical bone
  underestimation on MRI); localized ΔWEPL; produces natural negative controls (rays that
  miss bone ⇒ zero shift), enabling sensitivity/specificity analysis.

## 12. Statistical methodology
- Bootstrap 95% CIs (10,000 resamples) for MAE, bias, Pearson r, and speedup.
- Bland–Altman bias and 95% limits of agreement for surrogate vs pencil beam; reported
  with and without the single boundary-crossing case.
- Mann–Whitney U on |error| for metal-vs-clean and thorax-vs-abdomen subgroups, with a
  pre-specified sensitivity analysis excluding the boundary case (composition confound:
  abdominal pencil-beam subset is dominated by negative-control water rays).
- Pearson r is scale-invariant, so per-beamlet (mGy) dose normalization does not affect
  correlation; absolute errors are reported as % of beamlet D_peak or as mm range shift.

## Limitations (carried into the manuscript)
First-order model; central-axis 1-D endpoint; density-only physics (no nuclear/LET);
per-beam maps require a combination rule for multi-field plans; single-institution
imaging; training-set-only validation (test embargo); 300 HU bone threshold heuristic;
analytical shift ground truth in the expansion arm; hardware-dependent timings.
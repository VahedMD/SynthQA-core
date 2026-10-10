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
`D_sCT(z) ≈ D_ref(z − ΔR)`. A first-order Taylor expansion gives

```
ΔD(z) ≈ −(∂D_ref/∂z) · ΔWEPL   ⇒   |ΔD| ≈ |∂D_ref/∂z| · |ΔWEPL| = Risk
```

### 5.1 Sign convention and dose reconstruction

With `ΔWEPL = WEPL_sCT − WEPL_ref`:
- ΔWEPL > 0 (sCT overestimates SPR) ⇒ protons stop earlier ⇒ **proximal** shift;
  predicted depth shift = −ΔWEPL at the Bragg peak.
- The perturbed depth–dose profile is reconstructed from the reference profile by
  re-sampling in WEPL space:
  `D_sCT(z) = D_ref evaluated at (WEPL_ref(z) + ΔWEPL(z))`,
  i.e. `np.interp(WEPL_ref + ΔWEPL, WEPL_ref, D_ref)`. (Using `−ΔWEPL` here mirrors the
  shift and is a common sign bug.)

### 5.2 Validity regime (the "Taylor breakdown")

The voxel-wise identity |ΔD| ≈ Risk requires |ΔWEPL| ≲ fall-off width. For single proton
beamlets the fall-off is only ~3–4 mm wide; a uniform +50 HU bias produces 8–12 mm shifts,
where Risk (bell-shaped in the gradient) and |ΔD| (step-shaped) decorrelate. Consequences:
- **Voxel-wise gated correlation** is meaningful only in the small-error regime
  (e.g., localized bone errors of 1–4 mm).
- **The R80 range-shift metric is exact for arbitrary shift magnitudes** and is therefore
  the primary validation endpoint; voxel-wise correlation is reported as secondary.

## 6. Risk classification

| Class | Criterion |
|---|---|
| 0 Low | \|ΔWEPL\| < 1 mm, or gradient below threshold |
| 1 Medium | 1 mm ≤ \|ΔWEPL\| < 3 mm **and** high gradient |
| 2 High | \|ΔWEPL\| ≥ 3 mm **and** high gradient |

Bands are chosen against typical clinical range margins (~3.5 %·R + 1–3 mm).

## 7. Validation methodology

Endpoint hierarchy:
1. **Range shift (primary):** predicted −ΔWEPL at the peak vs (a) R80 shift of the
   WEPL-reconstructed dose on the MC profile, and (b) R80 shift between two full
   pyRadPlan Hong pencil-beam calculations (reference CT vs perturbed sCT).
2. **Voxel-wise gated correlation (secondary):** Pearson r between Risk and |ΔD| over
   voxels with |∇D| > 10 % of max (distal edge only).
3. **Specificity controls:** rays with zero perturbed-bone path must yield ≈ 0 mm shift.

Statistical robustness:
- **Bootstrap 95 % CIs** (10,000 resamples) for MAE, bias, and r.
- **Bland–Altman** limits of agreement for surrogate-vs-PB.
- **Mann–Whitney U** subgroup tests (metal vs clean; thorax vs abdomen) with a
  pre-specified sensitivity analysis excluding boundary-crossing outliers.
- **Cohort expansion:** surrogate-vs-MC correlation on an independent random patient
  subset (n = 17) beyond the stratified cohort.

Final numbers (see README table): cohort expansion r = 0.983 (0.955–0.996),
MAE 0.333 mm; PB head-to-head MAE 0.367 mm, LoA [−2.13, +1.65] mm
(excl. outlier [−0.26, +0.36] mm); speedup ≈ 8,900× vs PB.

## 8. Gamma and dose-difference metrics

Gamma uses **global** normalization (dose criterion as fraction of reference maximum),
evaluation restricted to voxels ≥ 10 % of reference max, and a shifted-array search within
the distance criterion (no interpolation of the reference). Default criteria 3 %/3 mm and
2 %/2 mm. Dose differences are reported absolute (Gy), relative-global (% of max), and
relative-local (% of local reference, masked where reference ≈ 0).


## 9. One-dimensional ray-tracing validation framework
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

## 10. Sign conventions and the shift model
With ΔWEPL = WEPL_sCT − WEPL_ref, the perturbed depth–dose is
`D_sCT(z) = D_ref` evaluated at WEPL `W(z) + ΔWEPL(z)` (i.e. `np.interp(W + ΔW, W, D_ref)`):
higher SPR stops protons earlier (proximal shift, negative depth shift), lower SPR
penetrates deeper (distal shift, positive depth shift). The predicted depth shift is
`−ΔWEPL` at the reference peak. Using `W − ΔW` inverts the physics and can push the
shifted peak outside the sampled ray (a common source of spurious `nan` R80 values).

## 11. Validity regime: the Taylor breakdown
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

## 12. sCT error models
- **Uniform bias (+50 HU):** global systematic error; maximizes ΔWEPL variance; used for
  the range-shift and head-to-head arms.
- **Bone-only (−200 HU for HU > 300):** representative deep-learning failure (cortical bone
  underestimation on MRI); localized ΔWEPL; produces natural negative controls (rays that
  miss bone ⇒ zero shift), enabling sensitivity/specificity analysis.

## 13. Statistical methodology
- Bootstrap 95% CIs (10,000 resamples) for MAE, bias, Pearson r, and speedup.
- Bland–Altman bias and 95% limits of agreement for surrogate vs pencil beam; reported
  with and without the single boundary-crossing case.
- Mann–Whitney U on |error| for metal-vs-clean and thorax-vs-abdomen subgroups, with a
  pre-specified sensitivity analysis excluding the boundary case (composition confound:
  abdominal pencil-beam subset is dominated by negative-control water rays).
- Pearson r is scale-invariant, so per-beamlet (mGy) dose normalization does not affect
  correlation; absolute errors are reported as % of beamlet D_peak or as mm range shift.


## 14. Limitations
1. First-order linearization (see §5.2).
2. 1-D ray model: lateral scatter and range mixing at heterogeneity boundaries are not
   captured (the 1THB002-type surrogate-vs-PB divergence); the surrogate is a lower-bound
   trigger in such cases.
3. Density-only physics: composition/LET and nuclear-interaction changes ignored.
4. Single dominant beam direction per risk map; multi-field plans combine per-beam maps.




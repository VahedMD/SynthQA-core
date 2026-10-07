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
# Voxel-wise WEPL integration update

## What changed

- Replaced arbitrary-ray trilinear SPR point sampling and rectangular accumulation
  with Siddon-style ray–voxel traversal.
- Each ray is partitioned at voxel faces. For segment `i`, the contribution is
  `SPR_i * segment_length_i`.
- The same traversal geometry and segment lengths are used for paired reference
  and sCT calculations.
- `num_samples` is retained for API compatibility. It now specifies the number
  of equally spaced locations at which the exact cumulative voxel-wise integral
  is returned; it is not the integration step size.
- Principal-axis WEPL is evaluated at voxel centres, including half of the
  current voxel's path length.
- Optional SimpleITK direction cosines are supported by both WEPL integration
  and uniform physical-position dose-profile sampling. The benchmark and
  reviewer-analysis scripts now pass the CT direction explicitly.

## Numerical conventions

- Arrays: `(Z,Y,X)`.
- Spacing/origin: `(X,Y,Z)`.
- Origin: physical centre of voxel `(0,0,0)`.
- Voxel values are piecewise constant for WEPL integration.
- Ray portions outside the image have `SPR=0` as a computational boundary
  convention. This convention is applied identically to reference and sCT.
- WEPL profile output is evaluated at the same uniformly spaced physical
  positions used by the benchmark dose-profile extractors, preserving paired
  array lengths and existing R80 analysis workflows.

## Required scientific re-analysis

The new integration changes the numerical method and can change the reported
results. Re-run all result-generating workflows before using previous figures
or summary statistics in the manuscript:

1. Synthetic analytical phantom benchmark and figures.
2. Stratified DoseRAD2026 benchmark.
3. Bone-only perturbation benchmark and spatial agreement analysis.
4. pyRadPlan head-to-head benchmark.
5. The 30-patient expansion cohort analysis and reviewer statistics.
6. All derived tables, confidence intervals, Bland–Altman analyses, and figures.

The prior results are retained in the repository for traceability only and must
not be described as results of this voxel-wise implementation.

## Tests added

The updated `tests/test_phase2_3.py` covers:
- axis-aligned voxel-centre path lengths;
- exact uniform-water oblique path lengths;
- heterogeneous voxel path integration;
- anisotropic voxel spacing;
- outside-image path convention;
- paired reference/sCT intersection consistency;
- direction-cosine handling;
- independence of integration accuracy from output `num_samples`;
- matching dose/WEPL profile segment locations;
- uniform dose-profile sampling under non-identity image direction cosines.

## Independent synthetic analytical validation (2026-11-10)

The Siddon-style implementation was checked against independent analytic
geometry rather than against another implementation of the same traversal:

| Test | Analytical result | Siddon result | Absolute difference |
|---|---:|---:|---:|
| Heterogeneous axis-aligned profile (four voxels) | 1.000, 3.500, 7.000, 9.800 mm | 1.000, 3.500, 7.000, 9.800 mm | 1.8 × 10⁻¹⁵ mm (maximum) |
| Oblique ray through water and a rectangular SPR=2.25 inclusion | 29.867614475610 mm | 29.867614475610 mm | 3.6 × 10⁻¹⁵ mm |
| Paired ΔWEPL for a two-voxel +0.2 SPR slab | 0.400000000000 mm | 0.400000000000 mm | 1.7 × 10⁻¹⁶ mm |

The oblique test uses an independent parametric ray/AABB intersection to
calculate the path length through both the image volume and the heterogeneous
inclusion. It also verifies that the sum of all Siddon segment lengths equals
the complete source-to-target ray length, including portions outside the image.

The implementation uses a half-open voxel-index convention at exact voxel
faces: a ray lying parallel to an internal face is assigned to the voxel on
the positive-index side (`floor(q + 0.5)`). Such face-coincident paths are
geometrically degenerate; this tie-breaking rule makes the numerical result
deterministic and prevents double counting. It should be retained consistently
in future implementations.

The targeted test suites currently pass **23 tests**. The full suite cannot be
collected in the validation environment because `SimpleITK` is unavailable
(`tests/test_phase1.py` imports it at module collection). The legacy spatial
agreement phantom was also rerun and returned Dice=0.939, sensitivity=0.920,
and specificity=0.998. The legacy synthetic benchmark reported gated Pearson
correlation 0.9306. Its displayed 6.25 ms timing measures the gradient-weighted
risk calculation only; it does not include WEPL generation and must not be
reported as end-to-end surrogate runtime.

The analytical validation script is `scripts/validate_siddon_phantom.py`.
Run it from the repository root with:

```bash
python scripts/validate_siddon_phantom.py
python -m pytest -q tests/test_phase2_3.py tests/test_phase4.py tests/test_metrics_visualization.py
```

## Physical-coordinate propagation in benchmark scripts

The image loader returns the SimpleITK direction-cosine matrix in addition to
spacing and origin. The stratified, bone-perturbation, pyRadPlan, reviewer
statistics, and R80 diagnostic scripts now pass this direction to both the
Siddon WEPL calculation and the common trilinear dose-profile sampler. This
keeps the WEPL and dose profiles on the same physical ray for non-identity
image orientations. The dose-profile interpolation remains trilinear; only
WEPL integration uses the piecewise-constant voxel-intersection model.

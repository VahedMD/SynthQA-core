import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.calibration.hu_to_density import HUToDensityConverter
from synthqa_core.calibration.density_to_spr import DensityToSPRConverter
from synthqa_core.surrogates.wepl import (
    compute_wepl_along_axis, compute_wepl_along_ray,
    compute_delta_wepl_along_ray, trace_ray_voxels,
    sample_volume_along_ray, sample_volume_at_ray_positions,
)


def test_spr_conversion():
    hu_conv = HUToDensityConverter()
    spr_conv = DensityToSPRConverter(method="linear_approx")
    density = hu_conv.convert(0)
    spr = spr_conv.convert(density)
    assert 0.99 < spr < 1.02
    density_bone = hu_conv.convert(1000)
    spr_bone = spr_conv.convert(density_bone)
    assert spr_bone > 1.1


def test_axis_aligned_wepl_at_voxel_centres():
    spr = np.ones((3, 3, 3), dtype=np.float32)
    spacing = (2.0, 2.0, 2.0)
    # A centre-based profile starts at the entrance face, half a voxel
    # before the first centre: 1, 3, 5 mm at the three voxel centres.
    assert np.allclose(compute_wepl_along_axis(spr, spacing, axis=0)[:, 0, 0], [1, 3, 5])
    assert np.allclose(compute_wepl_along_axis(spr, spacing, axis=2)[0, 0, :], [1, 3, 5])


def test_uniform_water_ray_exact_intersection_length():
    spr = np.ones((10, 10, 10), dtype=float)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)  # voxel centres
    # Traverse from first voxel centre to last voxel centre along X.
    src, dst = np.array([0., 0., 0.]), np.array([9., 0., 0.])
    details = compute_wepl_along_ray(spr, spacing, origin, src, dst, return_details=True)
    assert np.isclose(details["segment_lengths_mm"].sum(), 9.0)
    assert np.isclose(details["wepl_exit_mm"][-1], 9.0)
    assert np.allclose(details["wepl_mid_mm"], details["distances_mid_mm"])


def test_diagonal_uniform_water_ray_exact():
    spr = np.ones((10, 10, 10), dtype=float)
    src, dst = np.array([0., 0., 0.]), np.array([9., 9., 9.])
    length = np.linalg.norm(dst-src)
    details = compute_wepl_along_ray(
        spr, (1., 1., 1.), (0., 0., 0.), src, dst, return_details=True
    )
    assert np.isclose(details["segment_lengths_mm"].sum(), length)
    assert np.isclose(details["wepl_exit_mm"][-1], length)


def test_heterogeneous_voxel_path_integral_exact():
    # Ray along X through 3 voxels with SPR [1, 2, 3], unit spacing.
    spr = np.ones((1, 1, 3), dtype=float)
    spr[0, 0, :] = [1., 2., 3.]
    src, dst = np.array([-0.5, 0., 0.]), np.array([2.5, 0., 0.])
    details = compute_wepl_along_ray(
        spr, (1., 1., 1.), (0., 0., 0.), src, dst, return_details=True
    )
    assert np.allclose(details["segment_lengths_mm"], [1., 1., 1.])
    assert np.allclose(details["spr_segments"], [1., 2., 3.])
    assert np.isclose(details["wepl_exit_mm"][-1], 6.0)
    assert np.allclose(details["wepl_mid_mm"], [0.5, 2.0, 4.5])


def test_anisotropic_oblique_ray_lengths():
    spr = np.ones((4, 4, 4), dtype=float)
    spacing = (2.0, 1.0, 3.0)
    # Physical diagonal from centre to centre; the ray crosses multiple faces.
    src = np.array([0., 0., 0.])
    dst = np.array([6., 3., 9.])
    details = compute_wepl_along_ray(
        spr, spacing, (0., 0., 0.), src, dst, return_details=True
    )
    expected = np.linalg.norm(dst-src)
    assert np.isclose(details["segment_lengths_mm"].sum(), expected)
    assert np.isclose(details["wepl_exit_mm"][-1], expected)


def test_outside_image_is_zero_and_inside_integral_is_exact():
    spr = np.ones((1, 1, 3), dtype=float)
    # Image extent along X is [-0.5, 2.5]; source/target extend 2 mm outside.
    src, dst = np.array([-2.5, 0., 0.]), np.array([4.5, 0., 0.])
    details = compute_wepl_along_ray(
        spr, (1., 1., 1.), (0., 0., 0.), src, dst, return_details=True
    )
    assert np.isclose(details["segment_lengths_mm"].sum(), 7.0)
    assert np.isclose(details["wepl_exit_mm"][-1], 3.0)
    assert np.all(details["spr_segments"][~details["inside_image"]] == 0.0)


def test_reference_sct_share_identical_intersections():
    ref = np.ones((2, 2, 4), dtype=float)
    sct = ref.copy()
    sct[:, :, 1:3] += 0.2
    result = compute_delta_wepl_along_ray(
        ref, sct, (1., 1., 2.), (0., 0., 0.),
        np.array([-0.5, 0., 0.]), np.array([3.5, 0., 0.])
    )
    assert np.isclose(result["delta_wepl_exit_mm"][-1], 0.4)
    assert np.array_equal(result["voxel_indices_zyx"],
                          compute_wepl_along_ray(
                              sct, (1., 1., 2.), (0., 0., 0.),
                              np.array([-0.5, 0., 0.]), np.array([3.5, 0., 0.]),
                              return_details=True
                          )["voxel_indices_zyx"])


def test_num_samples_changes_profile_density_not_integral_accuracy():
    spr = np.ones((1, 1, 4), dtype=float)
    spr[0, 0, 1:3] = 2.0
    src, dst = np.array([-0.5, 0., 0.]), np.array([3.5, 0., 0.])
    coarse = compute_wepl_along_ray(
        spr, (1., 1., 1.), (0., 0., 0.), src, dst, num_samples=3
    )
    dense = compute_wepl_along_ray(
        spr, (1., 1., 1.), (0., 0., 0.), src, dst, num_samples=2001
    )
    assert len(coarse) == 3 and len(dense) == 2001
    assert np.isclose(coarse[-1], 6.0)
    assert np.isclose(dense[-1], 6.0)


def test_direction_cosines_rotate_physical_ray_into_voxel_grid():
    spr = np.ones((1, 1, 3), dtype=float)
    # 90-degree rotation around Z: physical +Y corresponds to image +X.
    direction = (0., -1., 0., 1., 0., 0., 0., 0., 1.)
    src = np.array([0., -0.5, 0.])
    dst = np.array([0., 2.5, 0.])
    details = compute_wepl_along_ray(
        spr, (1., 1., 1.), (0., 0., 0.), src, dst,
        direction=direction, num_samples=None, return_details=True
    )
    assert np.isclose(details["segment_lengths_mm"].sum(), 3.0)
    assert np.isclose(details["wepl_exit_mm"][-1], 3.0)
    assert np.array_equal(details["voxel_indices_zyx"][:, 2], [0, 1, 2])


def test_dose_sampling_uses_same_segment_midpoints():
    vol = np.ones((2, 2, 4), dtype=float)
    src, dst = np.array([-0.5, 0., 0.]), np.array([3.5, 0., 0.])
    values, tr = sample_volume_along_ray(
        vol, (1., 1., 1.), (0., 0., 0.), src, dst, order=1
    )
    wepl = compute_wepl_along_ray(
        vol, (1., 1., 1.), (0., 0., 0.), src, dst
    )
    assert len(values) == len(wepl) == len(tr["segment_lengths_mm"])
    assert np.allclose(values, 1.0)
    assert np.allclose(wepl, tr["distances_mid_mm"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


def _ray_aabb_length(source, target, lower, upper):
    """Independent slab-intersection calculation for a ray segment and AABB."""
    v = np.asarray(target, float) - np.asarray(source, float)
    t0, t1 = 0.0, 1.0
    for ax in range(3):
        if abs(v[ax]) < 1e-14:
            if source[ax] < lower[ax] or source[ax] > upper[ax]:
                return 0.0
            continue
        ta = (lower[ax] - source[ax]) / v[ax]
        tb = (upper[ax] - source[ax]) / v[ax]
        t0 = max(t0, min(ta, tb))
        t1 = min(t1, max(ta, tb))
        if t1 <= t0:
            return 0.0
    return float(np.linalg.norm(v) * (t1 - t0))


def test_siddon_axis_ray_matches_voxel_centre_axis_profile():
    # 1x1x4 volume, anisotropic X spacing, heterogeneous SPR.
    spr = np.array([[[1.0, 1.5, 2.0, 0.8]]])
    sx = 2.0
    origin = (10.0, -3.0, 5.0)
    source = np.array([origin[0] - sx / 2, origin[1], origin[2]])
    targets = [np.array([origin[0] + i * sx, origin[1], origin[2]]) for i in range(4)]
    expected = compute_wepl_along_axis(spr, (sx, 1.0, 1.0), axis=2)[0, 0]
    actual = []
    for target in targets:
        detail = compute_wepl_along_ray(
            spr, (sx, 1.0, 1.0), origin, source, target,
            return_details=True
        )
        actual.append(detail["wepl_exit_mm"][-1])
    assert np.allclose(actual, expected, atol=1e-12)


def test_siddon_oblique_slab_matches_independent_analytic_chord():
    # Water background with a higher-SPR cuboid. The expected integral is
    # total in-image path length + the extra SPR-weighted chord through cuboid.
    shape = (12, 10, 8)  # Z,Y,X
    spacing = (1.5, 2.0, 2.5)  # X,Y,Z
    origin = np.array([4.0, -6.0, 3.0])
    spr = np.ones(shape, dtype=float)
    # Cuboid indices: z=3..7, y=2..6, x=2..5 (inclusive-exclusive).
    spr[3:8, 2:7, 2:6] = 2.25

    # Physical bounds are voxel outer faces, using voxel-centre origin.
    lower = origin + np.array([2 - 0.5, 2 - 0.5, 3 - 0.5]) * np.array(spacing)
    upper = origin + np.array([6 - 0.5, 7 - 0.5, 8 - 0.5]) * np.array(spacing)
    # Ray crosses the cuboid diagonally and extends beyond the image.
    source = np.array([origin[0] - 4.0, origin[1] + 2.0, origin[2] + 2.0])
    target = np.array([origin[0] + 13.0, origin[1] + 12.0, origin[2] + 19.0])

    image_lower = origin - 0.5 * np.array(spacing)
    image_upper = origin + (np.array([shape[2], shape[1], shape[0]]) - 0.5) * np.array(spacing)
    image_chord = _ray_aabb_length(source, target, image_lower, image_upper)
    cuboid_chord = _ray_aabb_length(source, target, lower, upper)
    expected = image_chord + (2.25 - 1.0) * cuboid_chord

    result = compute_wepl_along_ray(
        spr, spacing, origin, source, target, return_details=True
    )
    actual = result["wepl_exit_mm"][-1]
    assert np.isclose(result["segment_lengths_mm"].sum(), np.linalg.norm(target-source))
    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_siddon_axis_aligned_ray_on_internal_face_has_deterministic_convention():
    # A ray lying exactly on an internal Y face is assigned to the +Y voxel
    # by floor(q + 0.5). It must not double-count path length.
    spr = np.ones((1, 2, 2), dtype=float)
    spr[0, 1, :] = 3.0
    source = np.array([-0.5, 0.5, 0.0])
    target = np.array([1.5, 0.5, 0.0])
    result = compute_wepl_along_ray(
        spr, (1., 1., 1.), (0., 0., 0.), source, target,
        return_details=True
    )
    assert np.isclose(result["segment_lengths_mm"].sum(), 2.0)
    assert np.isclose(result["wepl_exit_mm"][-1], 6.0)


def test_uniform_dose_profile_sampling_respects_direction_cosines():
    vol = np.zeros((1, 1, 3), dtype=float)
    vol[0, 0, :] = [0.0, 1.0, 2.0]
    direction = (0., -1., 0., 1., 0., 0., 0., 0., 1.)
    source = np.array([0., 0., 0.])
    target = np.array([0., 2., 0.])
    values, distances = sample_volume_at_ray_positions(
        vol, (1., 1., 1.), (0., 0., 0.), source, target,
        num_samples=3, direction=direction
    )
    assert np.allclose(values, [0., 1., 2.])
    assert np.allclose(distances, [0., 1., 2.])


def test_paired_siddon_profiles_are_consistent_and_exact_at_samples():
    ref = np.ones((1, 1, 4), dtype=np.float32)
    sct = ref.copy()
    sct[0, 0, 1:3] += 0.2
    src, dst = np.array([-0.5, 0., 0.]), np.array([3.5, 0., 0.])
    paired = compute_delta_wepl_along_ray(
        ref, sct, (1., 1., 1.), (0., 0., 0.), src, dst, num_samples=9
    )
    assert len(paired["delta_wepl_samples_mm"]) == 9
    assert np.allclose(
        paired["wepl_sct_samples_mm"] - paired["wepl_ref_samples_mm"],
        paired["delta_wepl_samples_mm"], atol=1e-12
    )
    assert np.isclose(paired["delta_wepl_samples_mm"][-1], 0.4)
    assert np.isclose(paired["wepl_ref_samples_mm"][-1], 4.0)
    assert np.isclose(paired["wepl_sct_samples_mm"][-1], 4.4)

"""Independent analytical validation of Siddon-style voxel-wise WEPL."""
from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from synthqa_core.surrogates.wepl import compute_wepl_along_axis, compute_wepl_along_ray


def aabb_chord(source, target, lower, upper):
    """Independent parametric line/AABB chord length."""
    v = target - source
    t_enter, t_exit = 0.0, 1.0
    for axis in range(3):
        if abs(v[axis]) < 1e-14:
            if source[axis] < lower[axis] or source[axis] > upper[axis]:
                return 0.0
            continue
        a = (lower[axis] - source[axis]) / v[axis]
        b = (upper[axis] - source[axis]) / v[axis]
        t_enter = max(t_enter, min(a, b))
        t_exit = min(t_exit, max(a, b))
        if t_exit <= t_enter:
            return 0.0
    return float(np.linalg.norm(v) * (t_exit - t_enter))


def main():
    # Case 1: axis-aligned ray through four heterogeneous voxels.
    sx = 2.0
    spr_x = np.array([[[1.0, 1.5, 2.0, 0.8]]])
    origin = np.array([10.0, -3.0, 5.0])
    src = np.array([origin[0] - sx / 2, origin[1], origin[2]])
    expected_axis = compute_wepl_along_axis(spr_x, (sx, 1., 1.), axis=2)[0, 0]
    actual_axis = []
    for i in range(4):
        dst = np.array([origin[0] + i * sx, origin[1], origin[2]])
        detail_axis = compute_wepl_along_ray(
            spr_x, (sx, 1., 1.), origin, src, dst, return_details=True
        )
        actual_axis.append(float(detail_axis["wepl_exit_mm"][-1]))
    axis_error = float(np.max(np.abs(np.asarray(actual_axis) - expected_axis)))

    # Case 2: oblique ray through a rectangular higher-SPR inclusion.
    shape = (12, 10, 8)  # Z,Y,X
    spacing = np.array([1.5, 2.0, 2.5])  # X,Y,Z
    origin = np.array([4.0, -6.0, 3.0])
    spr = np.ones(shape, dtype=float)
    spr[3:8, 2:7, 2:6] = 2.25
    lower = origin + np.array([1.5, 1.5, 2.5]) * spacing
    upper = origin + np.array([5.5, 6.5, 7.5]) * spacing
    image_lower = origin - 0.5 * spacing
    image_upper = origin + (np.array([shape[2], shape[1], shape[0]]) - 0.5) * spacing
    source = np.array([origin[0] - 4., origin[1] + 2., origin[2] + 2.])
    target = np.array([origin[0] + 13., origin[1] + 12., origin[2] + 19.])
    expected_image = aabb_chord(source, target, image_lower, image_upper)
    expected_inclusion = aabb_chord(source, target, lower, upper)
    expected_oblique = expected_image + 1.25 * expected_inclusion
    detail = compute_wepl_along_ray(
        spr, tuple(spacing), origin, source, target, return_details=True
    )
    actual_oblique = float(detail["wepl_exit_mm"][-1])
    segment_partition_error = float(abs(detail["segment_lengths_mm"].sum() - np.linalg.norm(target-source)))

    # Case 3: paired ΔWEPL, where a known slab perturbation is traversed.
    ref = np.ones((3, 3, 6), dtype=float)
    sct = ref.copy()
    sct[:, :, 2:4] += 0.2
    paired = compute_wepl_along_ray(
        sct - ref, (1., 1., 1.), (0., 0., 0.),
        np.array([-0.5, 1., 1.]), np.array([5.5, 1., 1.]),
        return_details=True
    )
    expected_delta = 2 * 0.2 * 1.0
    actual_delta = float(paired["wepl_exit_mm"][-1])

    report = {
        "method": "Siddon-style voxel-face partition; piecewise-constant SPR",
        "axis_aligned": {
            "max_abs_difference_vs_axis_cumsum_mm": axis_error,
            "expected_profile_mm": np.asarray(expected_axis).tolist(),
            "actual_profile_mm": np.asarray(actual_axis).tolist(),
        },
        "oblique_heterogeneous": {
            "expected_wepl_mm": expected_oblique,
            "actual_wepl_mm": actual_oblique,
            "absolute_error_mm": abs(actual_oblique - expected_oblique),
            "expected_image_chord_mm": expected_image,
            "expected_inclusion_chord_mm": expected_inclusion,
            "segment_partition_error_mm": segment_partition_error,
        },
        "paired_delta_wepl": {
            "expected_delta_wepl_mm": expected_delta,
            "actual_delta_wepl_mm": actual_delta,
            "absolute_error_mm": abs(actual_delta - expected_delta),
        },
    }
    print(json.dumps(report, indent=2))
    assert axis_error < 1e-10
    assert abs(actual_oblique - expected_oblique) < 1e-10
    assert segment_partition_error < 1e-10
    assert abs(actual_delta - expected_delta) < 1e-10


if __name__ == "__main__":
    main()

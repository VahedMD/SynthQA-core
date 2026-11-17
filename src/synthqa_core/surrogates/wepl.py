"""
Voxel-wise water-equivalent path-length (WEPL) integration.

The image is represented as a piecewise-constant voxel field. Rays are
partitioned at voxel faces (Siddon-style traversal), and each segment
contributes SPR_voxel * physical_segment_length.

Coordinate conventions:
- arrays are indexed (Z, Y, X);
- spacing and origin are supplied in (X, Y, Z);
- origin denotes the physical centre of voxel (0, 0, 0);
- direction is an optional SimpleITK 3x3 direction-cosine matrix, flattened
  row-major as returned by Image.GetDirection(). Rays are in physical space.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.ndimage import map_coordinates


def _direction_matrix(direction=None) -> np.ndarray:
    if direction is None:
        return np.eye(3, dtype=float)
    matrix = np.asarray(direction, dtype=float)
    if matrix.size == 9:
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3):
        raise ValueError("direction must be a 3x3 matrix or 9-element sequence")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
        raise ValueError("direction must be orthonormal (direction-cosine matrix)")
    return matrix


def trace_ray_voxels(
    shape_zyx: Tuple[int, int, int],
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    direction=None,
) -> Dict[str, np.ndarray]:
    """Partition the complete source-to-target ray into voxel-face segments.

    Segments outside the image are retained (with voxel index -1); their
    material contribution is zero. Returned sample locations are segment
    midpoints, and lengths are exact geometric intersection lengths up to
    floating-point precision.
    """
    shape = np.asarray(shape_zyx, dtype=int)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("shape_zyx must contain three positive dimensions")
    spacing = np.asarray(spacing_xyz, dtype=float)
    origin = np.asarray(origin_xyz, dtype=float)
    source = np.asarray(ray_source, dtype=float)
    target = np.asarray(ray_target, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three positive values")
    if origin.shape != (3,) or source.shape != (3,) or target.shape != (3,):
        raise ValueError("origin, ray_source, and ray_target must be 3-vectors")

    direction_matrix = _direction_matrix(direction)
    ray_vector = target - source
    ray_length = float(np.linalg.norm(ray_vector))
    if not np.isfinite(ray_length) or ray_length <= 0:
        raise ValueError("ray_source and ray_target must define a non-zero finite ray")

    # Convert physical coordinates to continuous voxel-centre coordinates.
    q0 = (direction_matrix.T @ (source - origin)) / spacing
    q1 = (direction_matrix.T @ (target - origin)) / spacing
    dq = q1 - q0

    # Every voxel face is at a half-integer continuous index.
    t_values = [0.0, 1.0]
    for axis, n in enumerate(shape[::-1]):  # q axes are X,Y,Z; shape is Z,Y,X
        if abs(dq[axis]) < 1e-14:
            continue
        faces = np.arange(n + 1, dtype=float) - 0.5
        ts = (faces - q0[axis]) / dq[axis]
        valid = ts[(ts > 1e-12) & (ts < 1.0 - 1e-12)]
        if valid.size:
            t_values.extend(valid.tolist())

    t = np.unique(np.asarray(t_values, dtype=float))
    t.sort()
    t_mid = 0.5 * (t[:-1] + t[1:])
    dt = np.diff(t)
    q_mid = q0[None, :] + t_mid[:, None] * dq[None, :]
    xyz_idx = np.floor(q_mid + 0.5).astype(int)
    x, y, z = xyz_idx[:, 0], xyz_idx[:, 1], xyz_idx[:, 2]
    inside = (
        (x >= 0) & (x < shape[2]) &
        (y >= 0) & (y < shape[1]) &
        (z >= 0) & (z < shape[0])
    )
    indices = np.full((len(t_mid), 3), -1, dtype=int)
    indices[inside] = np.column_stack((z[inside], y[inside], x[inside]))

    # Midpoint physical coordinates and physical segment lengths.
    midpoints = source[None, :] + t_mid[:, None] * ray_vector[None, :]
    lengths = ray_length * dt
    distances_mid = ray_length * t_mid

    return {
        "t_edges": t,
        "t_midpoints": t_mid,
        "midpoints_xyz": midpoints,
        "distances_mid_mm": distances_mid,
        "segment_lengths_mm": lengths,
        "voxel_indices_zyx": indices,
        "inside_image": inside,
        "ray_length_mm": np.asarray(ray_length),
    }


def sample_volume_along_ray(
    volume: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    direction=None,
    order: int = 1,
    cval: float = 0.0,
    traversal: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Sample a 3-D volume at the midpoint of each ray–voxel segment.

    Dose can be linearly interpolated at these locations (order=1), while SPR
    integration itself remains piecewise constant per voxel.
    """
    if np.asarray(volume).ndim != 3:
        raise ValueError("volume must be a 3-D (Z, Y, X) array")
    tr = traversal or trace_ray_voxels(
        volume.shape, spacing_xyz, origin_xyz, ray_source, ray_target, direction
    )
    dmat = _direction_matrix(direction)
    points = tr["midpoints_xyz"]
    local_xyz = ((dmat.T @ (points - np.asarray(origin_xyz)).T).T
                 / np.asarray(spacing_xyz, dtype=float))
    coords_zyx = np.vstack((local_xyz[:, 2], local_xyz[:, 1], local_xyz[:, 0]))
    values = map_coordinates(
        np.asarray(volume, dtype=float), coords_zyx,
        order=int(order), mode="constant", cval=float(cval), prefilter=(order > 1)
    )
    return values, tr



def sample_volume_at_ray_positions(
    volume: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    num_samples: int = 2000,
    direction=None,
    order: int = 1,
    cval: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a 3-D volume at uniformly spaced physical positions on a ray.

    This is intended for dose-profile extraction, not WEPL integration. The
    image direction cosines are applied when converting physical points to
    continuous voxel coordinates. Returns (sampled_values, distances_mm).
    """
    data = np.asarray(volume, dtype=float)
    if data.ndim != 3:
        raise ValueError("volume must be a 3-D (Z, Y, X) array")
    if int(num_samples) < 2:
        raise ValueError("num_samples must be >= 2")
    source = np.asarray(ray_source, dtype=float)
    target = np.asarray(ray_target, dtype=float)
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("ray_source and ray_target must be 3-vectors")
    ray_vector = target - source
    ray_length = float(np.linalg.norm(ray_vector))
    if not np.isfinite(ray_length) or ray_length <= 0:
        raise ValueError("ray_source and ray_target must define a non-zero finite ray")
    spacing = np.asarray(spacing_xyz, dtype=float)
    origin = np.asarray(origin_xyz, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three positive values")
    if origin.shape != (3,):
        raise ValueError("origin_xyz must be a 3-vector")
    dmat = _direction_matrix(direction)
    t = np.linspace(0.0, 1.0, int(num_samples))
    points = source[None, :] + t[:, None] * ray_vector[None, :]
    local_xyz = ((dmat.T @ (points - origin).T).T / spacing)
    coords_zyx = np.vstack((local_xyz[:, 2], local_xyz[:, 1], local_xyz[:, 0]))
    values = map_coordinates(
        data, coords_zyx, order=int(order), mode="constant", cval=float(cval),
        prefilter=(order > 1)
    )
    return values, t * ray_length

def compute_wepl_along_ray(
    spr_array: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    num_samples: Optional[int] = None,
    direction=None,
    return_details: bool = False,
):
    """Compute cumulative WEPL at ray-segment midpoints by voxel intersection.

    Integration is performed exactly over voxel-intersection segments.
    ``num_samples`` controls only the locations at which the cumulative
    integral is reported (uniformly along the complete source-to-target ray);
    it does not control integration accuracy. If ``num_samples`` is None,
    values are returned at voxel-segment midpoints. This preserves the
    historical profile-array interface while replacing point-sampled
    integration with voxel-wise path-length accumulation.
    """
    # Preserve the source dtype; only the compact per-segment arrays are
    # promoted to float64 for accurate accumulation. Avoid copying full CTs.
    spr = np.asarray(spr_array)
    if spr.ndim != 3:
        raise ValueError("spr_array must be a 3-D (Z, Y, X) array")
    tr = trace_ray_voxels(
        spr.shape, spacing_xyz, origin_xyz, ray_source, ray_target, direction
    )
    indices = tr["voxel_indices_zyx"]
    spr_segments = np.zeros(len(indices), dtype=float)
    inside = tr["inside_image"]
    spr_segments[inside] = spr[indices[inside, 0], indices[inside, 1], indices[inside, 2]]
    contributions = spr_segments * tr["segment_lengths_mm"]
    cumulative_before = np.concatenate(([0.0], np.cumsum(contributions)[:-1]))
    wepl_mid = cumulative_before + 0.5 * contributions
    wepl_exit = np.cumsum(contributions)
    tr.update({
        "spr_segments": spr_segments,
        "wepl_mid_mm": wepl_mid,
        "wepl_exit_mm": wepl_exit,
        "delta_wepl_contributions_mm": contributions,
    })

    if num_samples is None:
        sampled_wepl = wepl_mid
        sample_t = tr["t_midpoints"]
    else:
        if int(num_samples) < 2:
            raise ValueError("num_samples must be >= 2 when specified")
        sample_t = np.linspace(0.0, 1.0, int(num_samples))
        seg_idx = np.searchsorted(tr["t_edges"], sample_t, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(contributions) - 1)
        t_start = tr["t_edges"][seg_idx]
        dist_into_segment = (sample_t - t_start) * tr["ray_length_mm"]
        before = np.concatenate(([0.0], wepl_exit[:-1]))
        sampled_wepl = before[seg_idx] + spr_segments[seg_idx] * dist_into_segment
        # Numerical endpoint guard: the last sample is the exact total integral.
        sampled_wepl[-1] = wepl_exit[-1]
    tr["sample_t"] = sample_t
    tr["wepl_samples_mm"] = sampled_wepl
    return tr if return_details else sampled_wepl


def compute_wepl_along_axis(
    spr_array: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    axis: int,
) -> np.ndarray:
    """Compute WEPL at voxel centres for a positive principal-axis beam.

    The entrance boundary is half a voxel before the first voxel centre;
    therefore the centre value includes half of the current voxel's WEPL.
    Axis mapping follows NumPy (Z,Y,X): axis 0 uses sz, 1 uses sy, 2 uses sx.
    """
    spr = np.asarray(spr_array, dtype=float)
    if spr.ndim != 3:
        raise ValueError("spr_array must be a 3-D (Z, Y, X) array")
    spacing_map = {0: spacing_xyz[2], 1: spacing_xyz[1], 2: spacing_xyz[0]}
    if axis not in spacing_map:
        raise ValueError("axis must be 0 (Z), 1 (Y), or 2 (X)")
    voxel_size = float(spacing_map[axis])
    return (np.cumsum(spr, axis=axis) - 0.5 * spr) * voxel_size


def compute_delta_wepl_along_ray(
    spr_ref: np.ndarray,
    spr_sct: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    direction=None,
    num_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute paired WEPL profiles with one shared Siddon traversal.

    SPR is piecewise constant per voxel. Both images use identical voxel
    segments and physical segment lengths. If ``num_samples`` is provided,
    cumulative reference, sCT, and delta-WEPL values are reported at uniformly
    spaced positions along the complete physical ray; this sampling does not
    affect integration accuracy.
    """
    ref = np.asarray(spr_ref)
    sct = np.asarray(spr_sct)
    if ref.ndim != 3 or sct.ndim != 3 or ref.shape != sct.shape:
        raise ValueError("spr_ref and spr_sct must be matching 3-D arrays")

    tr = trace_ray_voxels(
        ref.shape, spacing_xyz, origin_xyz, ray_source, ray_target, direction
    )
    indices = tr["voxel_indices_zyx"]
    inside = tr["inside_image"]
    ref_segments = np.zeros(len(indices), dtype=np.float64)
    sct_segments = np.zeros(len(indices), dtype=np.float64)
    ref_segments[inside] = ref[indices[inside, 0], indices[inside, 1], indices[inside, 2]]
    sct_segments[inside] = sct[indices[inside, 0], indices[inside, 1], indices[inside, 2]]

    lengths = tr["segment_lengths_mm"]
    ref_contrib = ref_segments * lengths
    sct_contrib = sct_segments * lengths
    delta_spr = sct_segments - ref_segments
    delta_contrib = delta_spr * lengths

    def cumulative_at(t_values, contributions, segment_values):
        t_values = np.asarray(t_values, dtype=float)
        seg_idx = np.searchsorted(tr["t_edges"], t_values, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(contributions) - 1)
        before = np.concatenate(([0.0], np.cumsum(contributions, dtype=np.float64)[:-1]))
        distance_into = (t_values - tr["t_edges"][seg_idx]) * float(tr["ray_length_mm"])
        return before[seg_idx] + segment_values[seg_idx] * distance_into

    if num_samples is None:
        t_samples = tr["t_midpoints"]
        ref_values = np.concatenate(([0.0], np.cumsum(ref_contrib, dtype=np.float64)[:-1])) + 0.5 * ref_contrib
        sct_values = np.concatenate(([0.0], np.cumsum(sct_contrib, dtype=np.float64)[:-1])) + 0.5 * sct_contrib
        delta_values = np.concatenate(([0.0], np.cumsum(delta_contrib, dtype=np.float64)[:-1])) + 0.5 * delta_contrib
    else:
        if int(num_samples) < 2:
            raise ValueError("num_samples must be >= 2 when specified")
        t_samples = np.linspace(0.0, 1.0, int(num_samples))
        ref_values = cumulative_at(t_samples, ref_contrib, ref_segments)
        sct_values = cumulative_at(t_samples, sct_contrib, sct_segments)
        delta_values = cumulative_at(t_samples, delta_contrib, delta_spr)
        ref_values[-1] = np.sum(ref_contrib, dtype=np.float64)
        sct_values[-1] = np.sum(sct_contrib, dtype=np.float64)
        delta_values[-1] = np.sum(delta_contrib, dtype=np.float64)

    tr.update({
        "spr_ref_segments": ref_segments,
        "spr_sct_segments": sct_segments,
        "delta_spr_segments": delta_spr,
        "wepl_ref_contributions_mm": ref_contrib,
        "wepl_sct_contributions_mm": sct_contrib,
        "delta_wepl_contributions_mm": delta_contrib,
        "wepl_ref_mid_mm": np.concatenate(([0.0], np.cumsum(ref_contrib, dtype=np.float64)[:-1])) + 0.5 * ref_contrib,
        "wepl_sct_mid_mm": np.concatenate(([0.0], np.cumsum(sct_contrib, dtype=np.float64)[:-1])) + 0.5 * sct_contrib,
        "delta_wepl_mid_mm": np.concatenate(([0.0], np.cumsum(delta_contrib, dtype=np.float64)[:-1])) + 0.5 * delta_contrib,
        "wepl_ref_exit_mm": np.cumsum(ref_contrib, dtype=np.float64),
        "wepl_sct_exit_mm": np.cumsum(sct_contrib, dtype=np.float64),
        "delta_wepl_exit_mm": np.cumsum(delta_contrib, dtype=np.float64),
        "sample_t": t_samples,
        "wepl_ref_samples_mm": ref_values,
        "wepl_sct_samples_mm": sct_values,
        "delta_wepl_samples_mm": delta_values,
    })
    return tr

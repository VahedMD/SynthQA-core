"""Distal-edge (R80/R90) extraction and range-shift maps, vectorized over columns."""
import numpy as np

def distal_edge(depth_mm, dose, fraction=0.8):
    """Scalar R{fraction*100} along a 1D profile with sub-voxel interpolation."""
    pk = int(np.argmax(dose))
    if pk == 0 or dose[pk] <= 0:
        return np.nan
    thr = fraction * dose[pk]
    below = np.where(dose[pk:] < thr)[0]
    if len(below) == 0:
        return np.nan
    i = pk + below[0]
    if 0 < i < len(dose):
        y1, y2 = dose[i-1], dose[i]
        if y1 != y2:
            return depth_mm[i-1] + (thr - y1) * (depth_mm[i] - depth_mm[i-1]) / (y2 - y1)
    return depth_mm[i]

def distal_edge_map(dose, spacing_axis_mm, axis=0, fraction=0.8):
    """
    Per-ray-column distal edge depth (mm) of a 3D dose, vectorized.
    Returns 2D map over the two axes orthogonal to `axis` (NaN where undefined).
    """
    d = np.moveaxis(dose, axis, 0)
    n = d.shape[0]
    colmax = d.max(0)
    pk = d.argmax(0)
    idxg = np.arange(n)[:, None, None]
    below = (d < fraction * colmax[None]) & (idxg >= pk[None])
    has = below.any(0) & (colmax > 0)
    first = np.where(has, below.argmax(0), 1)
    safe = np.clip(first, 1, n - 1)
    y2 = np.take_along_axis(d, safe[None], 0)[0]
    y1 = np.take_along_axis(d, (safe - 1)[None], 0)[0]
    t = np.where(y1 != y2, (fraction * colmax - y1) / np.where(y1 != y2, y2 - y1, 1.0), 0.0)
    edge = (safe - 1 + t) * spacing_axis_mm
    return np.where(has, edge, np.nan)

def range_shift_map(ref, evl, spacing_axis_mm, axis=0, fraction=0.8):
    """evl edge minus ref edge (mm); positive = deeper (distal) shift."""
    return distal_edge_map(evl, spacing_axis_mm, axis, fraction) - \
           distal_edge_map(ref, spacing_axis_mm, axis, fraction)

def range_shift_stats(shift_map, mask=None):
    v = shift_map[mask] if mask is not None else shift_map.ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": np.nan, "std": np.nan, "p95": np.nan, "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v)),
            "p95": float(np.percentile(np.abs(v), 95)), "n": int(v.size)}
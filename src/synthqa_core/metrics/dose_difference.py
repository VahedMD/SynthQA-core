"""Voxel-wise dose difference metrics."""
import numpy as np

def absolute_difference(ref, evl):
    return evl - ref

def relative_difference_global(ref, evl):
    """(evl - ref) / max(ref), in percent."""
    return 100.0 * (evl - ref) / ref.max()

def relative_difference_local(ref, evl, eps=1e-8):
    """(evl - ref) / ref, in percent; NaN where ref ~ 0."""
    return 100.0 * (evl - ref) / np.where(ref > eps, ref, np.nan)

def summary(ref, evl, dose_threshold=0.10, delta_pct=2.0):
    """Summary stats inside the ref > threshold*max region."""
    m = ref >= dose_threshold * ref.max()
    rel = relative_difference_global(ref, evl)
    return {
        "mean_abs_diff_Gy": float(np.mean(np.abs(evl - ref)[m])),
        "max_abs_diff_Gy": float(np.max(np.abs(evl - ref)[m])),
        "mean_rel_diff_pct": float(np.mean(rel[m])),
        "frac_exceeding_delta_pct": float(np.mean(np.abs(rel[m]) > delta_pct)),
        "n_evaluated": int(m.sum()),
    }
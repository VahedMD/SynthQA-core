"""Shared beamlet selection metadata and R80 calculations for stratified QA.

This module is the single implementation used by both benchmark_dataset.py
and audit_stratified_r80_case.py. Dose profiles are sampled in the dose image's
own physical geometry; WEPL is integrated on the CT geometry with one shared
Siddon traversal. The two grids need not have identical array shapes, but both
must be expressed in the same physical coordinate system.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import numpy as np

from synthqa_core.surrogates.wepl import (
    compute_wepl_along_ray,
    compute_delta_wepl_along_ray,
    sample_volume_at_ray_positions,
)
from synthqa_core.surrogates.range_mapping import (
    inverse_monotone_profile,
    interval_spr_diagnostics,
)

ANALYSIS_VERSION = "stratified_r80_shared_v3"
PROFILE_SAMPLES = 2000
PRESELECTION_SAMPLES = 400
MAX_DOSE_DOWNLOAD_CANDIDATES = 4
MIN_PRESELECTION_WEPL_MM = 10.0


def range_mm(energy_mev: float) -> float:
    """Dataset benchmark's CSDA-like range-energy fit in water."""
    return 0.022 * float(energy_mev) ** 1.77


def find_r80(dose: Sequence[float], depth_mm: Sequence[float]) -> float:
    """Find distal 80%-of-peak crossing using the benchmark's linear rule.

    The threshold is 0.8 times the maximum of the supplied profile. The first
    sample strictly below threshold after the peak brackets the distal crossing.
    This exact convention is shared by benchmark and audit.
    """
    d = np.asarray(dose, dtype=float)
    z = np.asarray(depth_mm, dtype=float)
    if d.ndim != 1 or z.ndim != 1 or d.size != z.size or d.size < 3:
        return float("nan")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(z)):
        return float("nan")
    peak = int(np.argmax(d))
    if peak == 0 or d[peak] <= 0:
        return float("nan")
    threshold = 0.8 * float(d[peak])
    below = np.flatnonzero(d[peak:] < threshold)
    if below.size == 0:
        return float("nan")
    upper = peak + int(below[0])
    lower = upper - 1
    if lower < 0 or upper >= d.size:
        return float("nan")
    y0, y1 = float(d[lower]), float(d[upper])
    if y1 == y0:
        return float(z[upper])
    return float(z[lower] + (threshold - y0) * (z[upper] - z[lower]) / (y1 - y0))


def _meta_value(meta: Mapping[str, Any], key: str, fallback=None):
    return meta.get(key, fallback)


def sample_dose_profile(
    dose_meta: Mapping[str, Any],
    ray_source_xyz: Sequence[float],
    ray_target_xyz: Sequence[float],
    num_samples: int = PROFILE_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample dose along the physical ray using dose-image metadata."""
    required = ("array", "spacing_xyz", "origin_xyz")
    missing = [k for k in required if k not in dose_meta]
    if missing:
        raise ValueError(f"Dose image metadata missing required fields: {missing}")
    return sample_volume_at_ray_positions(
        dose_meta["array"], dose_meta["spacing_xyz"], dose_meta["origin_xyz"],
        np.asarray(ray_source_xyz, dtype=float), np.asarray(ray_target_xyz, dtype=float),
        num_samples=int(num_samples), direction=dose_meta.get("direction"), order=1, cval=0.0,
    )


def is_valid_dose_profile(dose_profile: Sequence[float]) -> bool:
    """Preserve the established profile eligibility rule exactly."""
    d = np.asarray(dose_profile, dtype=float)
    if d.ndim != 1 or d.size < 3 or not np.all(np.isfinite(d)):
        return False
    peak = int(np.argmax(d))
    return bool(peak > 0.2 * len(d) and d[peak] > 1e-3 and np.min(d[peak:]) < 0.1 * d[peak])


def evaluate_selected_beamlet(
    *,
    patient_id: str,
    spr_ref: np.ndarray,
    spr_sct: np.ndarray,
    ct_spacing_xyz: Sequence[float],
    ct_origin_xyz: Sequence[float],
    ct_direction: Optional[Sequence[float]],
    dose_meta: Mapping[str, Any],
    ray_source_xyz: Sequence[float],
    ray_target_xyz: Sequence[float],
    profile_samples: int = PROFILE_SAMPLES,
) -> Dict[str, Any]:
    """Compute all stratified R80 estimators from one selected physical ray."""
    dose_profile, depth = sample_dose_profile(
        dose_meta, ray_source_xyz, ray_target_xyz, num_samples=profile_samples
    )
    if not is_valid_dose_profile(dose_profile):
        raise ValueError(f"Selected dose profile is invalid for patient {patient_id}")

    paired = compute_delta_wepl_along_ray(
        spr_ref, spr_sct, ct_spacing_xyz, ct_origin_xyz,
        np.asarray(ray_source_xyz, dtype=float), np.asarray(ray_target_xyz, dtype=float),
        direction=ct_direction, num_samples=profile_samples,
    )
    w_ref = np.asarray(paired["wepl_ref_samples_mm"], dtype=float)
    w_sct = np.asarray(paired["wepl_sct_samples_mm"], dtype=float)
    delta_wepl = np.asarray(paired["delta_wepl_samples_mm"], dtype=float)
    if len(w_ref) != len(dose_profile) or len(w_sct) != len(dose_profile):
        raise RuntimeError(
            f"Profile length mismatch for {patient_id}: dose={len(dose_profile)}, "
            f"WEPL_ref={len(w_ref)}, WEPL_sCT={len(w_sct)}"
        )
    if np.any(np.diff(w_ref) < -1e-8) or np.any(np.diff(w_sct) < -1e-8):
        raise RuntimeError(f"Non-monotone WEPL profile encountered for {patient_id}")

    r80_ref = find_r80(dose_profile, depth)
    if not np.isfinite(r80_ref):
        raise ValueError(f"Reference R80 is invalid for patient {patient_id}")

    ray_length = float(paired["ray_length_mm"])
    t_r80 = float(np.clip(r80_ref / ray_length, 0.0, 1.0))
    segment_index = int(np.searchsorted(paired["t_edges"], t_r80, side="right") - 1)
    segment_index = int(np.clip(segment_index, 0, len(paired["delta_spr_segments"]) - 1))
    delta_contrib = np.asarray(paired["delta_wepl_contributions_mm"], dtype=float)
    before_delta = np.concatenate(([0.0], np.cumsum(delta_contrib, dtype=np.float64)[:-1]))
    delta_wepl_r80 = float(
        before_delta[segment_index]
        + paired["delta_spr_segments"][segment_index]
        * (t_r80 - paired["t_edges"][segment_index]) * ray_length
    )
    spr_ref_r80 = float(paired["spr_ref_segments"][segment_index])
    spr_sct_r80 = float(paired["spr_sct_segments"][segment_index])
    first_order = -delta_wepl_r80 / spr_ref_r80 if spr_ref_r80 > 0 else float("nan")

    target_wepl = float(np.interp(r80_ref, depth, w_ref))
    finite_mapped_r80 = inverse_monotone_profile(w_sct, depth, target_wepl)
    finite_map = finite_mapped_r80 - r80_ref if np.isfinite(finite_mapped_r80) else float("nan")

    unique_w, unique_idx = np.unique(w_ref, return_index=True)
    remapped_dose = np.interp(w_sct, unique_w, dose_profile[unique_idx], left=0.0, right=0.0)
    remapped_r80 = find_r80(remapped_dose, depth)
    remapped_profile_shift = remapped_r80 - r80_ref if np.isfinite(remapped_r80) else float("nan")
    inverse_profile_error = (
        finite_mapped_r80 - remapped_r80
        if np.isfinite(finite_mapped_r80) and np.isfinite(remapped_r80) else float("nan")
    )
    linearization_error = (
        abs(first_order - finite_map)
        if np.isfinite(first_order) and np.isfinite(finite_map) else float("nan")
    )
    relative_error = (
        linearization_error / (abs(finite_map) + 1e-9)
        if np.isfinite(linearization_error) else float("nan")
    )
    interval_diag = (
        interval_spr_diagnostics(paired, finite_mapped_r80, r80_ref)
        if np.isfinite(finite_mapped_r80) else {}
    )
    effective_spr = (
        abs(delta_wepl_r80) / abs(finite_map)
        if np.isfinite(finite_map) and abs(finite_map) > 1e-9 else float("nan")
    )
    local_spr_ratio = (
        effective_spr / spr_ref_r80 if spr_ref_r80 > 0 and np.isfinite(effective_spr) else float("nan")
    )

    return {
        "patient_id": str(patient_id),
        "dose_profile": dose_profile,
        "remapped_dose_profile": remapped_dose,
        "depth_mm": depth,
        "wepl_ref_samples_mm": w_ref,
        "wepl_sct_samples_mm": w_sct,
        "delta_wepl_samples_mm": delta_wepl,
        "paired": paired,
        "r80_ref_mm": float(r80_ref),
        "r80_remapped_profile_mm": float(remapped_r80) if np.isfinite(remapped_r80) else float("nan"),
        "r80_inverse_map_mm": float(finite_mapped_r80) if np.isfinite(finite_mapped_r80) else float("nan"),
        "first_order_mm": float(first_order),
        "finite_map_mm": float(finite_map),
        "remapped_profile_shift_mm": float(remapped_profile_shift),
        "inverse_vs_profile_mm": float(inverse_profile_error),
        "delta_wepl_r80_mm": float(delta_wepl_r80),
        "spr_ref_r80": spr_ref_r80,
        "spr_sct_r80": spr_sct_r80,
        "segment_index_r80": segment_index,
        "t_r80": t_r80,
        "linearization_error_mm": float(linearization_error),
        "linearization_relative_error": float(relative_error),
        "effective_spr_interval": float(effective_spr),
        "effective_to_local_spr_ratio": float(local_spr_ratio),
        **interval_diag,
    }


def benchmark_result_row(
    *,
    patient_id: str,
    region: str,
    metal: bool,
    selection: Mapping[str, Any],
    metrics: Mapping[str, Any],
    r_gated: float,
    runtime_ms: float,
) -> Dict[str, Any]:
    """Flatten selection provenance and shared metrics into one CSV row."""
    return {
        "Patient": str(patient_id), "Region": str(region), "Metal": bool(metal),
        "AnalysisVersion": ANALYSIS_VERSION,
        "BeamIdx": int(selection["beam_idx"]), "RayIdx": int(selection["ray_idx"]),
        "BeamletIdx": int(selection["beamlet_idx"]),
        "EnergyMeV": float(selection["energy_mev"]),
        "DoseFile": str(selection["dose_file"]),
        "SelectionRank": int(selection.get("selection_rank", -1)),
        "EnergyMismatch_mm": float(selection.get("energy_mismatch_mm", np.nan)),
        "RaySourceXYZ": _json_vector(selection["ray_source_xyz"]),
        "RayTargetXYZ": _json_vector(selection["ray_target_xyz"]),
        "ProfileSamples": int(selection.get("profile_samples", PROFILE_SAMPLES)),
        "r_gated": float(r_gated) if np.isfinite(r_gated) else float("nan"),
        "FirstOrder_mm": metrics["first_order_mm"],
        "FiniteMap_mm": metrics["finite_map_mm"],
        "RemappedProfile_mm": metrics["remapped_profile_shift_mm"],
        "InverseVsProfile_mm": metrics["inverse_vs_profile_mm"],
        "R80_ref_mm": metrics["r80_ref_mm"],
        "R80_finite_map_mm": metrics["r80_inverse_map_mm"],
        "DeltaWEPL_R80_mm": metrics["delta_wepl_r80_mm"],
        "SPR_ref_R80": metrics["spr_ref_r80"],
        "SPR_sct_R80": metrics["spr_sct_r80"],
        "Linearization_Error_mm": metrics["linearization_error_mm"],
        "Linearization_Relative_Error": metrics["linearization_relative_error"],
        "Effective_SPR_interval": metrics["effective_spr_interval"],
        "Effective_to_Local_SPR_Ratio": metrics["effective_to_local_spr_ratio"],
        "Time_ms": float(runtime_ms),
        **{k: v for k, v in metrics.items() if k.startswith(("interval_", "mean_spr_", "spr_ref_", "spr_sct_")) and k not in {"spr_ref_r80", "spr_sct_r80"}},
    }


def _json_vector(values: Sequence[float]) -> str:
    import json
    return json.dumps([float(v) for v in values], separators=(",", ":"))

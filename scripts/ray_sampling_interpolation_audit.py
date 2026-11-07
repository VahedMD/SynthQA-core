#!/usr/bin/env python3
"""Ray-sampling / interpolation sensitivity audit for SynthQA WEPL mapping.

Run from the SynthQA-core repository root (or put this file in scripts/):
    python ray_sampling_interpolation_audit.py --patient 1THB121

This is a numerical-method audit only. It does NOT calculate an independent
perturbed-image MC/PB dose distribution. It reuses the selected reference dose
profile and remaps it using WEPL curves computed with different spatial
sampling resolutions and interpolation orders.

Assumptions: axis-aligned image geometry (identity direction matrix), matching
CT/dose geometry, origin/spacing in xyz, arrays in zyx. The script fails rather
than silently proceeding when these assumptions are not met.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

# Import project utilities from the expected src layout.
ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / "src", ROOT.parent / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from huggingface_hub import hf_hub_download

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"


def sample_volume_along_ray(volume, spacing_xyz, origin_xyz, source_xyz, target_xyz,
                            n_samples: int, order: int):
    """Sample zyx volume along xyz ray using scipy interpolation order 0/1/3."""
    t = np.linspace(0.0, 1.0, int(n_samples), dtype=float)
    points = source_xyz[None, :] + t[:, None] * (target_xyz - source_xyz)[None, :]
    ix = (points[:, 0] - origin_xyz[0]) / spacing_xyz[0]
    iy = (points[:, 1] - origin_xyz[1]) / spacing_xyz[1]
    iz = (points[:, 2] - origin_xyz[2]) / spacing_xyz[2]
    coords = np.vstack((iz, iy, ix))
    values = map_coordinates(np.asarray(volume, dtype=np.float64), coords,
                             order=int(order), mode="constant", cval=0.0,
                             prefilter=(int(order) > 1))
    return points, values


def cumulative_wepl(spr, spacing_xyz, origin_xyz, source_xyz, target_xyz,
                    n_samples: int, order: int):
    points, spr_line = sample_volume_along_ray(
        spr, spacing_xyz, origin_xyz, source_xyz, target_xyz, n_samples, order)
    distance = float(np.linalg.norm(target_xyz - source_xyz))
    ds = distance / (n_samples - 1)
    # Composite trapezoidal integration; WEPL[0] = 0.
    increments = 0.5 * (spr_line[:-1] + spr_line[1:]) * ds
    wepl = np.concatenate(([0.0], np.cumsum(increments)))
    return points, spr_line, wepl, distance


def dose_profile_along_ray(dose, spacing_xyz, origin_xyz, source_xyz, target_xyz, n_samples):
    _, values = sample_volume_along_ray(dose, spacing_xyz, origin_xyz,
                                        source_xyz, target_xyz, n_samples, order=1)
    return values, float(np.linalg.norm(target_xyz - source_xyz))


def find_r80(dose, depth):
    dose = np.asarray(dose, float)
    peak = int(np.argmax(dose))
    if peak == 0 or not np.isfinite(dose[peak]) or dose[peak] <= 0:
        return np.nan
    threshold = 0.8 * dose[peak]
    below = np.flatnonzero(dose[peak:] < threshold)
    if not len(below):
        return np.nan
    i = peak + int(below[0])
    if i <= 0:
        return float(depth[i])
    y0, y1 = dose[i - 1], dose[i]
    if not np.isfinite(y0 + y1) or y1 == y0:
        return float(depth[i])
    return float(depth[i - 1] + (threshold - y0) *
                 (depth[i] - depth[i - 1]) / (y1 - y0))


def inverse_map_shift(depth, w_ref, w_sct, r80_ref):
    """Plateau-aware inverse map. Average physical depths for duplicate WEPLs."""
    wr80 = float(np.interp(r80_ref, depth, w_ref))
    # np.unique sorts values; average depth on each plateau to avoid arbitrary
    # first/last-index selection. This is an explicit convention for flat runs.
    vals, inv = np.unique(w_sct, return_inverse=True)
    sums = np.bincount(inv, weights=depth)
    counts = np.bincount(inv)
    mean_depth = sums / counts
    if len(vals) < 2 or wr80 < vals[0] or wr80 > vals[-1]:
        return np.nan, wr80
    mapped = float(np.interp(wr80, vals, mean_depth))
    return mapped - float(r80_ref), wr80


def flat_run_stats(w):
    d = np.diff(w)
    return {
        "n_steps": int(len(d)),
        "n_flat_steps": int(np.sum(np.abs(d) <= 1e-10)),
        "flat_fraction": float(np.mean(np.abs(d) <= 1e-10)) if len(d) else np.nan,
        "n_negative_steps": int(np.sum(d < -1e-8)),
        "unique_fraction": float(len(np.unique(w)) / len(w)) if len(w) else np.nan,
        "min_positive_step": float(np.min(d[d > 1e-10])) if np.any(d > 1e-10) else np.nan,
        "max_step": float(np.max(d)) if len(d) else np.nan,
    }


def select_ray(patient_id, ct_arr, spacing, origin, spr_ref):
    """Mirror the prior diagnostic's candidate order and dose acceptance rule."""
    from huggingface_hub import hf_hub_download
    plan_path = hf_hub_download(REPO_ID, f"proton/training/{patient_id}/{patient_id}.json",
                                repo_type="dataset")
    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    for beam in plan.get("beams", [])[:2]:
        for ray in beam.get("rays", [])[:5]:
            rs = np.asarray(ray["ray_source"], dtype=float)
            rt = np.asarray(ray["ray_target"], dtype=float)
            _, _, wshort, _ = cumulative_wepl(spr_ref, spacing, origin, rs, rt, 400, 1)
            if wshort[-1] < 10.0:
                continue
            dose_rel = (f"proton/training/{patient_id}/dose/"
                        f"Dose_B{beam['beam_idx']}_R{ray['ray_idx']}_L0.mha")
            try:
                dose_path = hf_hub_download(REPO_ID, dose_rel, repo_type="dataset")
                dose_img = load_mha(dose_path)
            except Exception as exc:
                print(f"[skip] {dose_rel}: {exc}")
                continue
            dose_arr = np.asarray(dose_img["array"])
            d1, dist = dose_profile_along_ray(dose_arr, spacing, origin, rs, rt, 2000)
            pk = int(np.argmax(d1))
            if (pk > 0.2 * len(d1) and d1[pk] > 1e-3 and
                    np.min(d1[pk:]) < 0.1 * d1[pk]):
                # Ensure dose geometry matches CT geometry.
                for key in ("spacing_xyz", "origin_xyz"):
                    if key in dose_img and not np.allclose(dose_img[key],
                                                           spacing if key == "spacing_xyz" else origin,
                                                           atol=1e-5):
                        raise ValueError(f"Dose {key} does not match CT geometry")
                return beam, ray, rs, rt, dose_arr, dose_rel
    raise RuntimeError(f"No acceptable reference dose ray found for {patient_id}")


def run(patient_id, output_dir, resolutions, orders):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prof = CalibrationProfile()
    ct_path = hf_hub_download(REPO_ID, f"proton/training/{patient_id}/image/ct.mha",
                              repo_type="dataset")
    ct_img = load_mha(ct_path)
    ct = np.asarray(ct_img["array"])
    spacing = np.asarray(ct_img["spacing_xyz"], dtype=float)
    origin = np.asarray(ct_img["origin_xyz"], dtype=float)
    direction = ct_img.get("direction_matrix", ct_img.get("direction"))
    if direction is not None:
        direction = np.asarray(direction, dtype=float)
        if direction.size == 9 and not np.allclose(direction.reshape(3, 3), np.eye(3), atol=1e-5):
            raise ValueError("Non-identity image direction detected; transform ray coordinates before auditing")
    spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct))
    spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(ct + 50.0))
    beam, ray, rs, rt, dose_arr, dose_file = select_ray(patient_id, ct, spacing, origin, spr_ref)

    # The reference dose is sampled at 2000 points, matching the prior diagnostic.
    dose, distance = dose_profile_along_ray(dose_arr, spacing, origin, rs, rt, 2000)
    depth = np.linspace(0.0, distance, len(dose))
    r80_ref = find_r80(dose, depth)
    if not np.isfinite(r80_ref):
        raise RuntimeError("Reference dose profile has no valid distal R80 crossing")

    records = []
    profiles = []
    for n in resolutions:
        for order in orders:
            pts, sr, wr, _ = cumulative_wepl(spr_ref, spacing, origin, rs, rt, n, order)
            _, ss, ws, _ = cumulative_wepl(spr_sct, spacing, origin, rs, rt, n, order)
            # Put WEPL curves on the reference-dose depth grid for dose remapping.
            z_native = np.linspace(0.0, distance, n)
            wr_dose = np.interp(depth, z_native, wr)
            ws_dose = np.interp(depth, z_native, ws)
            # Plateau-safe remapping: use unique reference WEPL values and average
            # reference physical depth for each plateau before interpolation.
            # For dose-vs-WEPL, average dose on repeated WEPL coordinates.
            uniq_wr, inv_wr = np.unique(wr_dose, return_inverse=True)
            dose_sum = np.bincount(inv_wr, weights=dose)
            dose_count = np.bincount(inv_wr)
            dose_by_wepl = dose_sum / dose_count
            reconstructed = np.interp(ws_dose, uniq_wr, dose_by_wepl,
                                       left=0.0, right=0.0)
            r80_rec = find_r80(reconstructed, depth)
            shift, wr80 = inverse_map_shift(depth, wr_dose, ws_dose, r80_ref)
            ref_stats, sct_stats = flat_run_stats(wr), flat_run_stats(ws)
            rec = {
                "patient": patient_id, "beam_idx": beam.get("beam_idx"),
                "ray_idx": ray.get("ray_idx"), "dose_file": dose_file,
                "n_samples": int(n), "sample_spacing_mm": distance / (n - 1),
                "interpolation_order": int(order),
                "interpolation_name": {0: "nearest", 1: "linear", 3: "cubic"}[order],
                "reference_r80_mm": r80_ref, "reconstructed_r80_mm": r80_rec,
                "reconstructed_r80_shift_mm": r80_rec - r80_ref,
                "inverse_map_shift_mm": shift, "reference_wepl_at_r80_mm": wr80,
                "reference_flat_fraction": ref_stats["flat_fraction"],
                "perturbed_flat_fraction": sct_stats["flat_fraction"],
                "reference_unique_fraction": ref_stats["unique_fraction"],
                "perturbed_unique_fraction": sct_stats["unique_fraction"],
                "reference_negative_steps": ref_stats["n_negative_steps"],
                "perturbed_negative_steps": sct_stats["n_negative_steps"],
                "reference_min_positive_step": ref_stats["min_positive_step"],
                "perturbed_min_positive_step": sct_stats["min_positive_step"],
                "reference_total_wepl_mm": float(wr[-1]),
                "perturbed_total_wepl_mm": float(ws[-1]),
                "delta_total_wepl_mm": float(ws[-1] - wr[-1]),
            }
            records.append(rec)
            # Save representative pointwise curves (all settings, useful for plots).
            for j in range(n):
                profiles.append({"n_samples": n, "interpolation_order": order,
                                 "depth_mm": z_native[j], "wepl_ref_mm": wr[j],
                                 "wepl_perturbed_mm": ws[j], "spr_ref_sampled": sr[j],
                                 "spr_perturbed_sampled": ss[j]})

    result = pd.DataFrame(records)
    result.to_csv(out / "ray_sampling_interpolation_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(out / "ray_sampling_interpolation_profiles.csv", index=False)
    meta = {
        "patient": patient_id, "dose_file": dose_file,
        "beam_idx": beam.get("beam_idx"), "ray_idx": ray.get("ray_idx"),
        "ray_source_xyz": rs.tolist(), "ray_target_xyz": rt.tolist(),
        "ray_length_mm": distance, "ct_shape_zyx": list(ct.shape),
        "ct_spacing_xyz_mm": spacing.tolist(), "ct_origin_xyz_mm": origin.tolist(),
        "perturbation": "global +50 HU", "dose_reference_sampling": "linear, 2000 samples",
        "wepl_integration": "composite trapezoidal integration of sampled SPR",
        "inverse_mapping_plateau_policy": "average physical depths for duplicate WEPL coordinates",
        "dose_mapping_plateau_policy": "average reference dose for duplicate reference WEPL coordinates",
        "warning": "This is a numerical sensitivity audit of analytical WEPL remapping, not independent MC/PB validation."
    }
    (out / "ray_sampling_interpolation_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"\nSaved audit outputs to: {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="1THB121")
    parser.add_argument("--output", default=None)
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=[501, 1001, 2001, 4001, 8001],
                        help="Number of ray samples; default spans ~2, 1, 0.5, 0.25, 0.125 mm for a 1 m ray")
    parser.add_argument("--orders", nargs="+", type=int, choices=[0, 1, 3], default=[0, 1, 3],
                        help="SPR interpolation order: 0 nearest, 1 linear, 3 cubic")
    args = parser.parse_args()
    if any(n < 3 for n in args.resolutions):
        parser.error("Each resolution must be >= 3")
    default_out = ROOT / "outputs" / "reproducibility" / "audit" / args.patient / "sampling_audit"
    run(args.patient, args.output or str(default_out), sorted(set(args.resolutions)),
        sorted(set(args.orders)))

if __name__ == "__main__":
    main()

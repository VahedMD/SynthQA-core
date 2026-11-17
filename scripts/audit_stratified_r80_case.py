#!/usr/bin/env python3
"""Reproduce and audit one exact row from the shared-v3 stratified benchmark.

Run after benchmark_dataset.py:
    python scripts/audit_stratified_r80_case.py --patient 1THB078

The script does NOT independently select a beamlet. It reads BeamIdx/RayIdx/
BeamletIdx, energy, dose-file path, and ray coordinates from the benchmark CSV,
validates those identifiers against the patient's plan, then calls the same
shared evaluate_selected_beamlet() function used by the benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.stratified_r80 import (
    ANALYSIS_VERSION, PROFILE_SAMPLES, evaluate_selected_beamlet,
)

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"
DEFAULT_BENCHMARK = ROOT / "outputs" / "reproducibility" / "model_a_stratified_case_level.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "reproducibility" / "r80_case_audits"

REQUIRED_COLUMNS = [
    "Patient", "AnalysisVersion", "BeamIdx", "RayIdx", "BeamletIdx", "EnergyMeV",
    "DoseFile", "RaySourceXYZ", "RayTargetXYZ", "ProfileSamples",
    "FirstOrder_mm", "FiniteMap_mm", "RemappedProfile_mm", "InverseVsProfile_mm",
    "R80_ref_mm", "R80_finite_map_mm", "DeltaWEPL_R80_mm", "SPR_ref_R80", "SPR_sct_R80",
    "Linearization_Error_mm", "Linearization_Relative_Error",
    "Effective_SPR_interval", "Effective_to_Local_SPR_Ratio",
    "interval_length_mm", "mean_spr_ref_interval", "mean_spr_sct_interval",
    "spr_ref_min_interval", "spr_ref_max_interval", "spr_sct_min_interval",
    "spr_sct_max_interval", "spr_ref_variation_interval", "spr_sct_variation_interval",
]

COMPARE_FIELDS = {
    "FirstOrder_mm": "first_order_mm",
    "FiniteMap_mm": "finite_map_mm",
    "RemappedProfile_mm": "remapped_profile_shift_mm",
    "InverseVsProfile_mm": "inverse_vs_profile_mm",
    "R80_ref_mm": "r80_ref_mm",
    "R80_finite_map_mm": "r80_inverse_map_mm",
    "DeltaWEPL_R80_mm": "delta_wepl_r80_mm",
    "SPR_ref_R80": "spr_ref_r80",
    "SPR_sct_R80": "spr_sct_r80",
    "Linearization_Error_mm": "linearization_error_mm",
    "Linearization_Relative_Error": "linearization_relative_error",
    "Effective_SPR_interval": "effective_spr_interval",
    "Effective_to_Local_SPR_Ratio": "effective_to_local_spr_ratio",
    "interval_length_mm": "interval_length_mm",
    "mean_spr_ref_interval": "mean_spr_ref_interval",
    "mean_spr_sct_interval": "mean_spr_sct_interval",
    "spr_ref_min_interval": "spr_ref_min_interval",
    "spr_ref_max_interval": "spr_ref_max_interval",
    "spr_sct_min_interval": "spr_sct_min_interval",
    "spr_sct_max_interval": "spr_sct_max_interval",
    "spr_ref_variation_interval": "spr_ref_variation_interval",
    "spr_sct_variation_interval": "spr_sct_variation_interval",
}


def _finite_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _close(a, b, atol):
    a, b = _finite_float(a), _finite_float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= atol)


def _find_plan_objects(plan, beam_idx, ray_idx, beamlet_idx):
    for beam in plan.get("beams", []):
        if int(beam.get("beam_idx", 0)) != int(beam_idx):
            continue
        for ray in beam.get("rays", []):
            if int(ray.get("ray_idx", 0)) != int(ray_idx):
                continue
            for beamlet in ray.get("beamlets", []):
                if int(beamlet.get("beamlet_idx", 0)) == int(beamlet_idx):
                    return beam, ray, beamlet
    raise KeyError(f"Plan does not contain B{beam_idx}R{ray_idx}L{beamlet_idx}")


def _assert_vector_match(label, saved, plan_value, atol=1e-6):
    a = np.asarray(saved, dtype=float)
    b = np.asarray(plan_value, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.allclose(a, b, atol=atol, rtol=0.0):
        raise RuntimeError(f"{label} in benchmark CSV does not match plan: CSV={a}, plan={b}")


def _write_segment_csv(path: Path, metrics: dict):
    paired = metrics["paired"]
    edges = np.asarray(paired["t_edges"], dtype=float)
    ray_length = float(paired["ray_length_mm"])
    lengths = np.asarray(paired["segment_lengths_mm"], dtype=float)
    ref = np.asarray(paired["spr_ref_segments"], dtype=float)
    sct = np.asarray(paired["spr_sct_segments"], dtype=float)
    delta = np.asarray(paired["delta_spr_segments"], dtype=float)
    ref_c = np.asarray(paired["wepl_ref_contributions_mm"], dtype=float)
    sct_c = np.asarray(paired["wepl_sct_contributions_mm"], dtype=float)
    delta_c = np.asarray(paired["delta_wepl_contributions_mm"], dtype=float)
    idx = np.asarray(paired["voxel_indices_zyx"], dtype=int)
    inside = np.asarray(paired["inside_image"], dtype=bool)
    frame = pd.DataFrame({
        "segment_index": np.arange(len(lengths)),
        "t_start": edges[:-1], "t_end": edges[1:],
        "distance_start_mm": edges[:-1] * ray_length,
        "distance_end_mm": edges[1:] * ray_length,
        "segment_length_mm": lengths,
        "inside_image": inside,
        "voxel_z": idx[:, 0], "voxel_y": idx[:, 1], "voxel_x": idx[:, 2],
        "spr_ref": ref, "spr_sct": sct, "delta_spr": delta,
        "wepl_ref_contribution_mm": ref_c,
        "wepl_sct_contribution_mm": sct_c,
        "delta_wepl_contribution_mm": delta_c,
        "wepl_ref_exit_mm": np.cumsum(ref_c),
        "wepl_sct_exit_mm": np.cumsum(sct_c),
        "delta_wepl_exit_mm": np.cumsum(delta_c),
    })
    frame["contains_reference_R80"] = (
        (frame["distance_start_mm"] <= metrics["r80_ref_mm"])
        & (frame["distance_end_mm"] >= metrics["r80_ref_mm"])
    )
    frame["contains_finite_mapped_R80"] = (
        (frame["distance_start_mm"] <= metrics["r80_inverse_map_mm"])
        & (frame["distance_end_mm"] >= metrics["r80_inverse_map_mm"])
    )
    frame.to_csv(path, index=False, float_format="%.12g")


def audit(patient_id: str, benchmark_csv: Path, output_dir: Path, atol: float = 1e-6):
    if not benchmark_csv.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {benchmark_csv}. Run the revised benchmark_dataset.py first."
        )
    df = pd.read_csv(benchmark_csv, dtype={"Patient": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "This is not a shared-v3 benchmark CSV. Missing provenance columns: "
            f"{missing}. Rerun the revised benchmark_dataset.py before auditing."
        )
    selected_rows = df.loc[df["Patient"].astype(str) == str(patient_id)]
    if len(selected_rows) != 1:
        raise RuntimeError(f"Expected exactly one benchmark row for {patient_id}; found {len(selected_rows)}")
    row = selected_rows.iloc[0]
    if str(row["AnalysisVersion"]) != ANALYSIS_VERSION:
        raise RuntimeError(
            f"Benchmark AnalysisVersion={row['AnalysisVersion']!r}; expected {ANALYSIS_VERSION!r}. "
            "Rerun the revised benchmark before auditing."
        )

    beam_idx, ray_idx, beamlet_idx = int(row["BeamIdx"]), int(row["RayIdx"]), int(row["BeamletIdx"])
    ray_source_csv = json.loads(str(row["RaySourceXYZ"]))
    ray_target_csv = json.loads(str(row["RayTargetXYZ"]))
    profile_samples = int(row["ProfileSamples"])
    if profile_samples != PROFILE_SAMPLES:
        raise RuntimeError(f"CSV ProfileSamples={profile_samples}; shared implementation expects {PROFILE_SAMPLES}")

    plan_path = hf_hub_download(REPO_ID, f"proton/training/{patient_id}/{patient_id}.json", repo_type="dataset")
    ct_path = hf_hub_download(REPO_ID, f"proton/training/{patient_id}/image/ct.mha", repo_type="dataset")
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    beam, ray, beamlet = _find_plan_objects(plan, beam_idx, ray_idx, beamlet_idx)

    _assert_vector_match("RaySourceXYZ", ray_source_csv, ray["ray_source"])
    _assert_vector_match("RayTargetXYZ", ray_target_csv, ray["ray_target"])
    plan_energy = float(beamlet["energy"])
    if not np.isclose(plan_energy, float(row["EnergyMeV"]), atol=1e-8, rtol=0.0):
        raise RuntimeError(f"Energy mismatch: CSV={row['EnergyMeV']}, plan={plan_energy}")

    expected_dose_file = f"proton/training/{patient_id}/dose/Dose_B{beam_idx}_R{ray_idx}_L{beamlet_idx}.mha"
    dose_file = str(row["DoseFile"])
    if dose_file != expected_dose_file:
        raise RuntimeError(f"DoseFile mismatch: CSV={dose_file!r}; expected from selected IDs={expected_dose_file!r}")
    dose_path = hf_hub_download(REPO_ID, dose_file, repo_type="dataset")

    ct_meta = load_mha(ct_path)
    dose_meta = load_mha(dose_path)
    calibration = CalibrationProfile()
    ct_array = ct_meta["array"]
    spr_ref = calibration.density_to_spr.convert(calibration.hu_to_density.convert(ct_array))
    spr_sct = calibration.density_to_spr.convert(calibration.hu_to_density.convert(ct_array + 50.0))

    metrics = evaluate_selected_beamlet(
        patient_id=patient_id, spr_ref=spr_ref, spr_sct=spr_sct,
        ct_spacing_xyz=ct_meta["spacing_xyz"], ct_origin_xyz=ct_meta["origin_xyz"],
        ct_direction=ct_meta["direction"], dose_meta=dose_meta,
        ray_source_xyz=ray_source_csv, ray_target_xyz=ray_target_csv,
        profile_samples=profile_samples,
    )

    comparisons = {}
    all_match = True
    for csv_field, metric_field in COMPARE_FIELDS.items():
        if csv_field not in df.columns:
            continue
        benchmark_value = _finite_float(row[csv_field])
        audit_value = _finite_float(metrics[metric_field])
        matches = _close(benchmark_value, audit_value, atol)
        comparisons[csv_field] = {
            "benchmark_csv": benchmark_value,
            "audit_recomputed": audit_value,
            "absolute_difference": abs(benchmark_value - audit_value)
            if np.isfinite(benchmark_value) and np.isfinite(audit_value) else None,
            "within_tolerance": matches,
        }
        all_match = all_match and matches

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{patient_id}_shared_v3"
    profile_csv = output_dir / f"{stem}_profile.csv"
    segment_csv = output_dir / f"{stem}_voxel_segments.csv"
    report_json = output_dir / f"{stem}_audit.json"

    profile_frame = pd.DataFrame({
        "depth_mm": metrics["depth_mm"],
        "reference_dose": metrics["dose_profile"],
        "finite_remapped_dose": metrics["remapped_dose_profile"],
        "wepl_ref_mm": metrics["wepl_ref_samples_mm"],
        "wepl_sct_mm": metrics["wepl_sct_samples_mm"],
        "delta_wepl_mm": metrics["delta_wepl_samples_mm"],
    })
    profile_frame.to_csv(profile_csv, index=False, float_format="%.12g")
    _write_segment_csv(segment_csv, metrics)

    dose_ray_length = float(np.linalg.norm(np.asarray(ray_target_csv, dtype=float) - np.asarray(ray_source_csv, dtype=float)))
    sampled_ray_length = float(metrics["depth_mm"][-1])
    geometry = {
        "ct_shape_zyx": list(ct_meta["array"].shape),
        "dose_shape_zyx": list(dose_meta["array"].shape),
        "ct_spacing_xyz": list(ct_meta["spacing_xyz"]),
        "dose_spacing_xyz": list(dose_meta["spacing_xyz"]),
        "ct_origin_xyz": list(ct_meta["origin_xyz"]),
        "dose_origin_xyz": list(dose_meta["origin_xyz"]),
        "ct_direction": list(ct_meta["direction"]),
        "dose_direction": list(dose_meta["direction"]),
        "geometry_equal": bool(
            ct_meta["array"].shape == dose_meta["array"].shape
            and np.allclose(ct_meta["spacing_xyz"], dose_meta["spacing_xyz"])
            and np.allclose(ct_meta["origin_xyz"], dose_meta["origin_xyz"])
            and np.allclose(ct_meta["direction"], dose_meta["direction"])
        ),
        "dose_sampling_ray_length_mm": dose_ray_length,
        "sampled_profile_ray_length_mm": sampled_ray_length,
        "profile_ray_length_matches_plan_mm": bool(np.isclose(dose_ray_length, sampled_ray_length, atol=1e-6, rtol=0.0)),
        "note": "Dose sampling uses dose-image metadata; WEPL traversal uses CT-image metadata. Both use the exact physical ray saved in the benchmark CSV.",
    }
    if not geometry["profile_ray_length_matches_plan_mm"]:
        raise RuntimeError("Sampled dose-profile length does not match the saved physical ray length")
    selection = {
        "beam_idx": beam_idx, "ray_idx": ray_idx, "beamlet_idx": beamlet_idx,
        "energy_mev": plan_energy, "dose_file": dose_file,
        "selection_rank": int(row["SelectionRank"]) if "SelectionRank" in row else None,
        "energy_mismatch_mm": _finite_float(row["EnergyMismatch_mm"]) if "EnergyMismatch_mm" in row else None,
        "ray_source_xyz": [float(x) for x in ray_source_csv],
        "ray_target_xyz": [float(x) for x in ray_target_csv],
        "profile_samples": profile_samples,
    }
    metrics_json = {
        k: v for k, v in metrics.items()
        if k not in {"dose_profile", "remapped_dose_profile", "depth_mm",
                     "wepl_ref_samples_mm", "wepl_sct_samples_mm", "delta_wepl_samples_mm", "paired"}
    }
    report = {
        "patient_id": patient_id,
        "analysis_version": ANALYSIS_VERSION,
        "benchmark_csv": str(benchmark_csv),
        "selection": selection,
        "geometry": geometry,
        "metrics": metrics_json,
        "benchmark_comparison": comparisons,
        "comparison_absolute_tolerance": atol,
        "benchmark_reproduction_passed": bool(all_match),
        "outputs": {"profile_csv": str(profile_csv), "voxel_segments_csv": str(segment_csv)},
        "interpretation": (
            "The audit recomputed the exact beamlet recorded in the shared-v3 benchmark CSV, "
            "validated its beam/ray/beamlet identifiers, energy, dose file, and ray coordinates "
            "against the plan, and used the same shared R80 calculation function. FiniteMap and "
            "RemappedProfile remain analytical WEPL remapping, not independent perturbed-image MC/PB."
        ),
    }
    report_json.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    print(f"\n[selection] {patient_id}: B{beam_idx} R{ray_idx} L{beamlet_idx} | {plan_energy:.6f} MeV")
    print(f"[dose] {dose_file}")
    print(f"[R80] reference={metrics['r80_ref_mm']:.6f} mm | finite-map={metrics['finite_map_mm']:+.6f} mm")
    print(f"[estimators] first-order={metrics['first_order_mm']:+.6f} mm | "
          f"finite-map={metrics['finite_map_mm']:+.6f} mm | "
          f"linearization-error={metrics['linearization_error_mm']:.6f} mm")
    print(f"[consistency] inverse/profile={metrics['inverse_vs_profile_mm']:+.8f} mm")
    print("\nBenchmark CSV reproduction:")
    for field, values in comparisons.items():
        status = "PASS" if values["within_tolerance"] else "MISMATCH"
        print(f"  {status:8s} {field:34s} benchmark={values['benchmark_csv']:.10g} "
              f"audit={values['audit_recomputed']:.10g} diff={values['absolute_difference']}")
    print(f"\nAudit result: {'PASS' if all_match else 'FAIL'} (atol={atol:g})")
    print(f"Saved: {profile_csv}")
    print(f"Saved: {segment_csv}")
    print(f"Saved: {report_json}")
    return 0 if all_match else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", required=True, help="Patient ID present in the latest shared-v3 benchmark CSV")
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute numerical comparison tolerance")
    args = parser.parse_args()
    if args.atol <= 0:
        parser.error("--atol must be positive")
    return audit(args.patient, args.benchmark_csv, args.output_dir, args.atol)


if __name__ == "__main__":
    raise SystemExit(main())

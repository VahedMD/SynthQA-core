"""Stratified DoseRAD2026 benchmark using shared, auditable R80 calculations.

Run from the repository root:
    python scripts/benchmark_dataset.py

The selected beamlet identity and physical ray are written to the result CSV.
The companion audit_stratified_r80_case.py reads that row and recomputes the
same beamlet through the same shared calculation module.
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.wepl import compute_wepl_along_ray
from synthqa_core.surrogates.stratified_r80 import (
    ANALYSIS_VERSION, PROFILE_SAMPLES, PRESELECTION_SAMPLES,
    MAX_DOSE_DOWNLOAD_CANDIDATES, MIN_PRESELECTION_WEPL_MM,
    range_mm, sample_dose_profile, is_valid_dose_profile,
    evaluate_selected_beamlet, benchmark_result_row,
)
from cohort_manifest import (
    new_run, base_record, append_manifest_rows, discover_stratified_candidate_pool,
)

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"
OUT_DIR = ROOT / "outputs" / "reproducibility"
FIG_DIR = ROOT / "figures"

STRATIFIED_TARGETS = {
    ("Thorax", True): 3,
    ("Thorax", False): 3,
    ("Abdomen", True): 2,
    ("Abdomen", False): 4,
}


def discover_patients():
    """Return frozen 12-patient cohort unless explicit refresh is requested."""
    refresh = os.environ.get("SYNTHQA_REFRESH_STRATIFIED_COHORT", "0").strip().lower()
    cache_path = Path(__file__).with_name("stratified_candidate_patient_ids.txt")
    if refresh not in {"1", "true", "yes"} and cache_path.exists():
        patient_ids = [
            line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(patient_ids) != 12 or len(set(patient_ids)) != 12:
            raise RuntimeError(f"Frozen stratified cohort file must contain 12 unique IDs: {cache_path}")
        configs = discover_stratified_candidate_pool(REPO_ID, patient_ids=patient_ids)
        observed = {}
        for cfg in configs:
            key = (cfg["region"], bool(cfg["metal"]))
            observed[key] = observed.get(key, 0) + 1
        if observed != STRATIFIED_TARGETS:
            raise RuntimeError(f"Frozen stratified cohort counts {observed}; expected {STRATIFIED_TARGETS}")
        print(f"[cohort] Using frozen local stratified cohort ({len(configs)} patients): {cache_path}")
        print("[cohort] No Hugging Face repository listing required. Set "
              "SYNTHQA_REFRESH_STRATIFIED_COHORT=1 to discover replacement candidates.")
        return configs
    print("[cohort] Discovering full candidate pool from Hugging Face (network required)...")
    return discover_stratified_candidate_pool(REPO_ID)


def _plan_candidate_records(plan, spr_ref, spacing_xyz, origin_xyz, direction):
    """Build the exact ordered candidate list used for beamlet selection."""
    candidates = []
    examined_rays = 0
    examined_beamlets = 0
    short_rays = 0
    for beam in plan.get("beams", [])[:4]:
        for ray in beam.get("rays", [])[:6]:
            examined_rays += 1
            source = np.asarray(ray["ray_source"], dtype=float)
            target = np.asarray(ray["ray_target"], dtype=float)
            wepl = compute_wepl_along_ray(
                spr_ref, spacing_xyz, origin_xyz, source, target,
                num_samples=PRESELECTION_SAMPLES, direction=direction,
            )
            if float(wepl[-1]) < MIN_PRESELECTION_WEPL_MM:
                short_rays += 1
                continue
            target_wepl = 0.9 * float(wepl[-1])
            for beamlet in ray.get("beamlets", []):
                examined_beamlets += 1
                energy = float(beamlet["energy"])
                mismatch = abs(range_mm(energy) - target_wepl)
                candidates.append({
                    "energy_mismatch_mm": float(mismatch),
                    "beam_idx": int(beam.get("beam_idx", 0)),
                    "ray_idx": int(ray.get("ray_idx", 0)),
                    "beamlet_idx": int(beamlet.get("beamlet_idx", 0)),
                    "energy_mev": energy,
                    "ray_source_xyz": source,
                    "ray_target_xyz": target,
                })
    # Python's sort is stable: exact ties retain the plan's original order.
    candidates.sort(key=lambda item: item["energy_mismatch_mm"])
    return candidates, examined_rays, examined_beamlets, short_rays


def _get_plan_objects(plan, beam_idx, ray_idx, beamlet_idx):
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


def run():
    run_meta = new_run(__file__)
    calibration = CalibrationProfile()
    results, timings, manifest_rows = [], [], []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))

    patient_configs = discover_patients()
    successful = {key: 0 for key in STRATIFIED_TARGETS}
    attempted = {key: 0 for key in STRATIFIED_TARGETS}

    for selection_order, cfg in enumerate(patient_configs, start=1):
        patient_id = str(cfg["pid"])
        stratum = (cfg["region"], bool(cfg["metal"]))
        target_count = STRATIFIED_TARGETS[stratum]
        if successful[stratum] >= target_count:
            continue
        attempted[stratum] += 1
        manifest = base_record(
            run_meta, "model_a_stratified", patient_id,
            anatomical_region=cfg["region"], metal_status=bool(cfg["metal"]),
            selection_method="stable energy-mismatch ranking; first valid dose profile among top four candidates",
            perturbation_model="Model A: global HU bias", perturbation_threshold_hu="",
            perturbation_delta_hu=50, candidate_rays_examined=0, candidate_beamlets_examined=0,
            rays_rejected_short_wepl=0, dose_download_failures=0, invalid_dose_profiles=0,
            dose_engine_failures=0, eligible_rays=0, outcome_status="selected_pending",
            notes_json={
                "analysis_version": ANALYSIS_VERSION,
                "selection_order": selection_order,
                "candidate_rank_within_stratum": cfg["candidate_rank"],
                "target_count_for_stratum": target_count,
                "dose_reference": "analytical finite WEPL remapping of reference dose profile; not independent perturbed-image MC/PB",
            },
        )
        print(f"\n{cfg['region']} {patient_id} (Metal={cfg['metal']})...", end=" ", flush=True)

        try:
            plan_path = hf_hub_download(REPO_ID, f"proton/training/{patient_id}/{patient_id}.json", repo_type="dataset")
            ct_path = hf_hub_download(REPO_ID, f"proton/training/{patient_id}/image/ct.mha", repo_type="dataset")
            with open(plan_path, encoding="utf-8") as handle:
                plan = json.load(handle)
            ct_meta = load_mha(ct_path)
        except Exception as exc:
            manifest.update(outcome_status="excluded", exclusion_reason="plan_or_CT_load_failed",
                            notes_json={**manifest["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(manifest)
            print("[plan/CT load failed]")
            continue

        ct_array = ct_meta["array"]
        spacing = ct_meta["spacing_xyz"]
        origin = ct_meta["origin_xyz"]
        direction = ct_meta["direction"]
        try:
            spr_ref = calibration.density_to_spr.convert(calibration.hu_to_density.convert(ct_array))
            spr_sct = calibration.density_to_spr.convert(calibration.hu_to_density.convert(ct_array + 50.0))
            candidates, n_rays, n_beamlets, n_short = _plan_candidate_records(
                plan, spr_ref, spacing, origin, direction
            )
        except Exception as exc:
            manifest.update(outcome_status="excluded", exclusion_reason="candidate_preselection_failed",
                            notes_json={**manifest["notes_json"], "error": str(exc)[:500]})
            manifest_rows.append(manifest)
            print("[preselection failed]")
            continue

        manifest["candidate_rays_examined"] = n_rays
        manifest["candidate_beamlets_examined"] = n_beamlets
        manifest["rays_rejected_short_wepl"] = n_short
        selected = None
        selected_dose_meta = None
        dose_failures = 0
        invalid_profiles = 0
        candidate_errors = []

        for rank, candidate in enumerate(candidates[:MAX_DOSE_DOWNLOAD_CANDIDATES], start=1):
            b_idx, r_idx, l_idx = candidate["beam_idx"], candidate["ray_idx"], candidate["beamlet_idx"]
            dose_file = f"proton/training/{patient_id}/dose/Dose_B{b_idx}_R{r_idx}_L{l_idx}.mha"
            try:
                dose_path = hf_hub_download(REPO_ID, dose_file, repo_type="dataset")
                dose_meta = load_mha(dose_path)
            except Exception as exc:
                dose_failures += 1
                candidate_errors.append({"dose_file": dose_file, "error": str(exc)[:200]})
                continue
            try:
                profile, _ = sample_dose_profile(
                    dose_meta, candidate["ray_source_xyz"], candidate["ray_target_xyz"],
                    num_samples=PROFILE_SAMPLES,
                )
            except Exception as exc:
                invalid_profiles += 1
                candidate_errors.append({"dose_file": dose_file, "profile_error": str(exc)[:200]})
                continue
            if not is_valid_dose_profile(profile):
                invalid_profiles += 1
                continue
            selected = {**candidate, "dose_file": dose_file, "selection_rank": rank,
                        "profile_samples": PROFILE_SAMPLES}
            selected_dose_meta = dose_meta
            break

        manifest["dose_download_failures"] = dose_failures
        manifest["invalid_dose_profiles"] = invalid_profiles
        if selected is None:
            manifest.update(outcome_status="excluded", exclusion_reason="no_valid_dose_profile_in_top_4_candidates",
                            notes_json={**manifest["notes_json"], "candidate_errors": candidate_errors})
            manifest_rows.append(manifest)
            print("[no valid profile in top four]")
            continue

        try:
            t0 = time.perf_counter()
            metrics = evaluate_selected_beamlet(
                patient_id=patient_id, spr_ref=spr_ref, spr_sct=spr_sct,
                ct_spacing_xyz=spacing, ct_origin_xyz=origin, ct_direction=direction,
                dose_meta=selected_dose_meta,
                ray_source_xyz=selected["ray_source_xyz"], ray_target_xyz=selected["ray_target_xyz"],
                profile_samples=PROFILE_SAMPLES,
            )
            runtime_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            manifest.update(outcome_status="excluded", exclusion_reason="shared_r80_evaluation_failed",
                            notes_json={**manifest["notes_json"], "error": str(exc)[:500],
                                        "selection": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                      for k, v in selected.items()}})
            manifest_rows.append(manifest)
            print("[R80 evaluation failed]")
            continue

        dose = metrics["dose_profile"]
        remapped = metrics["remapped_dose_profile"]
        depth = metrics["depth_mm"]
        delta_wepl = metrics["delta_wepl_samples_mm"]
        gradient = np.gradient(dose, float(depth[1] - depth[0]))
        risk = np.abs(delta_wepl) * np.abs(gradient)
        dose_error = np.abs(remapped - dose)
        gate = np.abs(gradient) > 0.10 * np.abs(gradient).max()
        if gate.sum() < 15:
            gate = np.abs(gradient) > 0.02 * np.abs(gradient).max()
        r_gated = np.corrcoef(risk[gate], dose_error[gate])[0, 1] if gate.sum() > 5 else float("nan")

        result = benchmark_result_row(
            patient_id=patient_id, region=cfg["region"], metal=bool(cfg["metal"]),
            selection=selected, metrics=metrics, r_gated=r_gated, runtime_ms=runtime_ms,
        )
        results.append(result)
        timings.append(runtime_ms)
        successful[stratum] += 1

        manifest.update(
            selected_beam_idx=selected["beam_idx"], selected_ray_idx=selected["ray_idx"],
            selected_beamlet_idx=selected["beamlet_idx"], first_order_shift_mm=metrics["first_order_mm"],
            finite_map_shift_mm=metrics["finite_map_mm"], surrogate_runtime_ms=runtime_ms,
            eligible_rays=1, outcome_status="included", exclusion_reason="",
            notes_json={**manifest["notes_json"],
                "selection": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in selected.items()},
                "r80_analysis_version": ANALYSIS_VERSION,
                "r80_reference_mm": metrics["r80_ref_mm"],
                "r80_finite_map_mm": metrics["r80_inverse_map_mm"],
                "remapped_profile_shift_mm": metrics["remapped_profile_shift_mm"],
                "inverse_vs_profile_mm": metrics["inverse_vs_profile_mm"],
                "linearization_error_mm": metrics["linearization_error_mm"],
                "effective_spr_interval": metrics["effective_spr_interval"],
                "effective_to_local_spr_ratio": metrics["effective_to_local_spr_ratio"],
                "interval_spr_diagnostics": {
                    key: (float(value) if np.isfinite(value) else None)
                    for key, value in metrics.items()
                    if key.startswith(("interval_", "mean_spr_", "spr_ref_", "spr_sct_"))
                    and key not in {"spr_ref_r80", "spr_sct_r80"}
                },
                "gated_risk_error_correlation": None if not np.isfinite(r_gated) else float(r_gated),
                "gate_rule": "|gradient| > 0.10*max(|gradient|), fallback 0.02 if fewer than 15 samples",
            },
        )
        manifest_rows.append(manifest)

        color = "red" if cfg["metal"] else "blue"
        marker = "x" if cfg["region"] == "Thorax" else "o"
        plt.scatter(metrics["first_order_mm"], metrics["finite_map_mm"], s=80,
                    c=color, marker=marker, alpha=0.9)
        print(
            f"(B{selected['beam_idx']}R{selected['ray_idx']}L{selected['beamlet_idx']}) "
            f"first_order={metrics['first_order_mm']:+.3f} "
            f"finite_map={metrics['finite_map_mm']:+.3f} mm | "
            f"linearization_error={metrics['linearization_error_mm']:.3f} mm"
        )

    append_manifest_rows(manifest_rows, __file__)
    print("\nSuccessful patients by stratum:")
    for (region, metal), target in STRATIFIED_TARGETS.items():
        label = f"{region}_{'metal' if metal else 'nonmetal'}"
        print(f"  {label}: {successful[(region, metal)]}/{target} (candidates attempted: {attempted[(region, metal)]})")
    if any(successful[k] < n for k, n in STRATIFIED_TARGETS.items()):
        print("[WARNING] One or more strata did not reach the planned n=12 cohort.")

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "model_a_stratified_case_level.csv"
    df.to_csv(out_csv, index=False, float_format="%.12g")
    print(f"\nSaved {len(df)} shared-v2 stratified rows: {out_csv}")
    if df.empty:
        raise RuntimeError("No valid stratified patients were produced; inspect the manifest.")

    valid = df.dropna(subset=["FiniteMap_mm", "FirstOrder_mm"])
    corr = float(np.corrcoef(valid["FirstOrder_mm"], valid["FiniteMap_mm"])[0, 1]) if len(valid) > 1 else float("nan")
    mae = float(np.mean(np.abs(valid["FiniteMap_mm"] - valid["FirstOrder_mm"])))
    bias = float(np.mean(valid["FirstOrder_mm"] - valid["FiniteMap_mm"]))
    print("\n" + "=" * 72)
    print(f"FIRST-ORDER vs FINITE WEPL MAPPING: r={corr:.3f} | MAE={mae:.3f} mm | bias={bias:+.3f} mm | n={len(valid)}")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    limits = [min(valid["FirstOrder_mm"].min(), valid["FiniteMap_mm"].min()) - 1,
              max(valid["FirstOrder_mm"].max(), valid["FiniteMap_mm"].max()) + 1]
    plt.plot(limits, limits, "k--", alpha=0.5)
    plt.xlabel("First-order local R80 shift (mm)")
    plt.ylabel("Finite WEPL-mapped R80 shift (mm)")
    plt.title("First-order approximation versus finite WEPL mapping")
    plt.xlim(limits); plt.ylim(limits); plt.grid(ls="--", alpha=0.5)
    plt.legend(handles=[
        mlines.Line2D([], [], color="blue", marker="x", ls="None", ms=9, label="Thorax (non-metal)"),
        mlines.Line2D([], [], color="red", marker="x", ls="None", ms=9, label="Thorax (metal)"),
        mlines.Line2D([], [], color="blue", marker="o", ls="None", ms=9, label="Abdomen (non-metal)"),
        mlines.Line2D([], [], color="red", marker="o", ls="None", ms=9, label="Abdomen (metal)"),
    ], loc="best")
    plt.tight_layout()
    figure_path = FIG_DIR / "Fig6_FirstOrder_vs_FiniteMap.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Saved {figure_path}")
    print(f"Median paired-WEPL runtime: {np.median(timings):.3f} ms")
    print(f"Median linearization error: {df['Linearization_Error_mm'].median():.4f} mm")
    print(f"Median inverse/profile consistency error: {df['InverseVsProfile_mm'].abs().median():.6f} mm")


if __name__ == "__main__":
    run()

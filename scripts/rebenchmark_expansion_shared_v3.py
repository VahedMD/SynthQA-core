
"""
Rebuild and rebenchmark the Model A expansion cohort using shared R80 v3.

Run from the repository root:
    python scripts/rebenchmark_expansion_shared_v3.py

Smoke test:
    python scripts/rebenchmark_expansion_shared_v3.py --limit-patients 1

Resume / complete:
    python scripts/rebenchmark_expansion_shared_v3.py

This is a deterministic reconstruction of the expansion cohort, not a claim
that its membership is identical to the earlier cohort whose outputs were
deleted.

The finite-map reference is an analytical WEPL remapping of the reference
dose profile. It is NOT a fresh perturbed-image MC or PB calculation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


# ---------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.profiles import CalibrationProfile
from synthqa_core.surrogates.stratified_r80 import (
    ANALYSIS_VERSION,
    PROFILE_SAMPLES,
    MAX_DOSE_DOWNLOAD_CANDIDATES,
    sample_dose_profile,
    is_valid_dose_profile,
    evaluate_selected_beamlet,
    benchmark_result_row,
)

import benchmark_dataset as shared_benchmark

from cohort_manifest import (
    discover_training_patient_ids,
    patient_metadata,
)


REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"

TARGET_VALID_N = 30
RANDOM_SEED = 42
HOLDOUT_IDS = {"1THB121"}

# Exact successful 12-patient stratified cohort frozen by the project.
STRATIFIED_IDS_FILE = SCRIPTS / "stratified_candidate_patient_ids.txt"

# Fixed reviewer PB-comparison cohort. Keep this list explicit and auditable.
FIXED_PB_IDS = {
    "1THB016",
    "1THB021",
    "1THB029",
    "1THB002",
    "1THB008",
    "1THB011",
    "1ABB078",
    "1ABB138",
    "1ABB006",
    "1ABB011",
    "1ABB020",
}

OUT_DIR = ROOT / "outputs" / "reproducibility"
STRATIFIED_RESULTS_CSV = OUT_DIR / "model_a_stratified_case_level.csv"
PROFILE_DIR = OUT_DIR / "r80_expansion_shared_v3_profiles"

CANDIDATE_ORDER_CSV = (
    OUT_DIR / "expansion_candidate_order_shared_v3_seed42.csv"
)
RESULTS_CSV = (
    OUT_DIR / "model_a_expansion_shared_v3_n30.csv"
)
PROCESSING_CSV = (
    OUT_DIR / "model_a_expansion_shared_v3_processing_log.csv"
)
CANDIDATE_LOG_CSV = (
    OUT_DIR / "model_a_expansion_shared_v3_candidate_log.csv"
)
COHORT_IDS_CSV = (
    OUT_DIR / "expansion_cohort_ids_shared_v3.csv"
)
SUMMARY_JSON = (
    OUT_DIR / "model_a_expansion_shared_v3_summary.json"
)

REFERENCE_TYPE = (
    "analytical finite WEPL remapping of the reference dose profile; "
    "not independent perturbed-image MC/PB"
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically; remove schema-less empty checkpoint files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and len(df.columns) == 0:
        # A zero-byte/newline-only CSV is not a useful checkpoint and makes
        # pandas.read_csv raise EmptyDataError on the next resume.
        if path.exists():
            path.unlink()
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g")
    tmp.replace(path)


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a checkpoint, treating missing/empty files as empty dataframes."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    tmp.replace(path)


def json_vector(values) -> str:
    return json.dumps(
        [float(v) for v in values],
        separators=(",", ":"),
    )


def is_not_found_error(exc: Exception) -> bool:
    """Return True only for a confirmed missing Hub file."""
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        return status == 404
    return False


# ---------------------------------------------------------------------
# Cohort discovery and freezing
# ---------------------------------------------------------------------

def load_frozen_stratified_ids() -> set[str]:
    """Load the exact 12 successful stratified IDs used in this project."""
    if STRATIFIED_IDS_FILE.exists():
        ids = {
            line.strip()
            for line in STRATIFIED_IDS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    elif STRATIFIED_RESULTS_CSV.exists():
        df = safe_read_csv(STRATIFIED_RESULTS_CSV)
        if "Patient" not in df.columns:
            raise RuntimeError(
                f"Cannot recover frozen stratified IDs from {STRATIFIED_RESULTS_CSV}"
            )
        ids = set(df["Patient"].astype(str))
    else:
        raise FileNotFoundError(
            "The frozen stratified patient list is missing. Expected "
            f"{STRATIFIED_IDS_FILE} or {STRATIFIED_RESULTS_CSV}."
        )

    if len(ids) != 12:
        raise RuntimeError(
            f"Expected exactly 12 frozen stratified IDs; found {len(ids)}: "
            f"{sorted(ids)}"
        )
    return ids


def discover_and_freeze_candidate_order() -> pd.DataFrame:
    """Create or repair the deterministic expansion candidate order.

    Existing order is retained, but all members of the *actual frozen*
    stratified cohort, fixed PB cohort, and explicit hold-out are removed.
    This repairs the earlier nominal-quota reservation error without
    reshuffling the remaining candidates.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stratified_ids = load_frozen_stratified_ids()
    reserved_ids = stratified_ids | FIXED_PB_IDS | HOLDOUT_IDS

    if CANDIDATE_ORDER_CSV.exists():
        order = safe_read_csv(CANDIDATE_ORDER_CSV)
        required = {"CandidateOrder", "Patient", "Region", "Metal"}
        if not required.issubset(order.columns):
            raise RuntimeError(
                "Existing candidate-order CSV has an unexpected schema. "
                "Move it aside before running this shared-v3 script."
            )
        if order["Patient"].astype(str).duplicated().any():
            raise RuntimeError("Duplicate patient IDs in candidate-order CSV.")

        old_n = len(order)
        removed_ids = set(order["Patient"].astype(str)) & reserved_ids
        if removed_ids and RESULTS_CSV.exists():
            prior_results = safe_read_csv(RESULTS_CSV)
            if not prior_results.empty and "Patient" in prior_results.columns:
                successful_overlap = set(prior_results["Patient"].astype(str)) & removed_ids
                if successful_overlap:
                    raise RuntimeError(
                        "The existing expansion results contain patients that "
                        "belong to a reserved cohort: "
                        f"{sorted(successful_overlap)}. Do not silently rewrite "
                        "an already populated analysis; archive and review "
                        "these outputs first."
                    )
        order = order.loc[
            ~order["Patient"].astype(str).isin(reserved_ids)
        ].copy()
        removed = old_n - len(order)
        order = order.reset_index(drop=True)
        order["CandidateOrder"] = np.arange(1, len(order) + 1, dtype=int)
        if removed:
            atomic_csv(order, CANDIDATE_ORDER_CSV)
            order_map = dict(zip(order["Patient"].astype(str), order["CandidateOrder"].astype(int)))
            # Keep historical logs, but synchronize ranks for patients that
            # remain in the repaired candidate sequence.
            if PROCESSING_CSV.exists():
                old_processing = safe_read_csv(PROCESSING_CSV)
                if not old_processing.empty:
                    old_processing["CandidateOrder"] = old_processing.apply(
                        lambda row: order_map.get(str(row["Patient"]), row["CandidateOrder"]),
                        axis=1,
                    )
                    atomic_csv(old_processing, PROCESSING_CSV)
            if RESULTS_CSV.exists():
                old_results = safe_read_csv(RESULTS_CSV)
                if not old_results.empty and "CohortOrder" in old_results.columns:
                    old_results["CohortOrder"] = old_results["Patient"].astype(str).map(order_map)
                    atomic_csv(old_results, RESULTS_CSV)
            print(
                f"[cohort] Repaired saved candidate order: removed {removed} "
                "reserved/overlapping patient(s); retained the relative order "
                "of all remaining candidates."
            )
        else:
            print(
                f"[cohort] Reusing validated candidate order: "
                f"{CANDIDATE_ORDER_CSV}"
            )

        overlap = set(order["Patient"].astype(str)) & reserved_ids
        if overlap:
            raise RuntimeError(f"Reserved patients remain in candidate order: {sorted(overlap)}")
        return order

    print("[cohort] Discovering DoseRAD2026 training patients...")
    patient_ids = discover_training_patient_ids(REPO_ID)
    if not patient_ids:
        raise RuntimeError("No proton training patient IDs were discovered.")
    patient_ids = sorted({str(pid) for pid in patient_ids})
    eligible_ids = [pid for pid in patient_ids if pid not in reserved_ids]
    if len(eligible_ids) < TARGET_VALID_N:
        raise RuntimeError(
            f"Only {len(eligible_ids)} eligible patients remain; "
            f"cannot target {TARGET_VALID_N} valid cases."
        )

    rng = np.random.RandomState(RANDOM_SEED)
    candidate_order = [str(pid) for pid in rng.permutation(eligible_ids)]
    rows = []
    for rank, pid in enumerate(candidate_order, start=1):
        meta = patient_metadata(pid)
        rows.append({
            "CandidateOrder": rank,
            "Patient": pid,
            "Region": meta["anatomical_region"],
            "Metal": bool(meta["metal_status"]),
            "RandomSeed": RANDOM_SEED,
            "AnalysisVersion": ANALYSIS_VERSION,
        })
    order = pd.DataFrame(rows)
    atomic_csv(order, CANDIDATE_ORDER_CSV)
    print(f"[cohort] Frozen stratified IDs reserved: {len(stratified_ids)}")
    print(f"[cohort] Fixed PB patients reserved: {len(FIXED_PB_IDS)}")
    print(f"[cohort] Explicit hold-outs: {sorted(HOLDOUT_IDS)}")
    print(f"[cohort] Eligible pool: {len(eligible_ids)}")
    print(f"[cohort] Saved seeded candidate order: {CANDIDATE_ORDER_CSV}")
    return order


# ---------------------------------------------------------------------
# Existing checkpoints and resume logic
# ---------------------------------------------------------------------

def load_checkpoint():
    results = safe_read_csv(RESULTS_CSV)

    if PROCESSING_CSV.exists():
        processing = safe_read_csv(PROCESSING_CSV)
    else:
        processing = pd.DataFrame(
            columns=[
                "Patient",
                "CandidateOrder",
                "Region",
                "Metal",
                "Status",
                "Reason",
            ]
        )

    candidate_log = safe_read_csv(CANDIDATE_LOG_CSV)

    if not results.empty:
        if "AnalysisVersion" not in results.columns:
            raise RuntimeError(
                "Existing results CSV lacks AnalysisVersion. "
                "Move it aside before running this shared-v3 script."
            )

        versions = set(results["AnalysisVersion"].astype(str))
        if versions != {ANALYSIS_VERSION}:
            raise RuntimeError(
                f"Existing results contain unexpected versions: {versions}"
            )

        if results["Patient"].astype(str).duplicated().any():
            raise RuntimeError("Duplicate patient rows in results checkpoint.")

    terminal_statuses = {"included", "no_valid_profile_top4", "shared_r80_evaluation_failed", "plan_or_ct_not_found"}
    completed = set()

    if not processing.empty:
        for _, row in processing.iterrows():
            if str(row["Status"]) in terminal_statuses:
                completed.add(str(row["Patient"]))

    # Recover safely if the process stopped after writing the results file but
    # before writing its matching processing-log row.
    if not results.empty and "Patient" in results.columns:
        completed.update(results["Patient"].astype(str))

    return results, processing, candidate_log, completed


# ---------------------------------------------------------------------
# Shared-v3 dose candidate selection
# ---------------------------------------------------------------------

def select_shared_v3_beamlet(
    patient_id: str,
    plan: dict,
    spr_ref: np.ndarray,
    spacing,
    origin,
    direction,
):
    """
    Select the first valid dose profile among the shared-v3 top candidates.

    Uses the exact candidate preselection function from benchmark_dataset.py.
    """
    candidates, n_rays, n_beamlets, n_short = (
        shared_benchmark._plan_candidate_records(
            plan,
            spr_ref,
            spacing,
            origin,
            direction,
        )
    )

    attempt_rows = []

    if not candidates:
        return None, None, attempt_rows, {
            "candidate_rays_examined": n_rays,
            "candidate_beamlets_examined": n_beamlets,
            "rays_rejected_short_wepl": n_short,
        }

    for rank, candidate in enumerate(
        candidates[:MAX_DOSE_DOWNLOAD_CANDIDATES],
        start=1,
    ):
        b = candidate["beam_idx"]
        r = candidate["ray_idx"]
        l = candidate["beamlet_idx"]

        dose_file = (
            f"proton/training/{patient_id}/dose/"
            f"Dose_B{b}_R{r}_L{l}.mha"
        )

        attempt = {
            "Patient": patient_id,
            "CandidateRank": rank,
            "BeamIdx": b,
            "RayIdx": r,
            "BeamletIdx": l,
            "EnergyMeV": candidate["energy_mev"],
            "EnergyMismatch_mm": candidate["energy_mismatch_mm"],
            "DoseFile": dose_file,
            "Status": "pending",
            "Error": "",
        }

        try:
            dose_path = hf_hub_download(
                REPO_ID,
                dose_file,
                repo_type="dataset",
            )
            dose_meta = load_mha(dose_path)

        except Exception as exc:
            attempt["Error"] = str(exc)[:400]

            if is_not_found_error(exc):
                attempt["Status"] = "dose_file_not_found"
                attempt_rows.append(attempt)
                continue

            # A non-404 download failure may be a transient network or
            # authentication problem. Stop rather than silently replacing
            # this patient with another one.
            raise RuntimeError(
                f"Non-404 dose download failure for {patient_id}, "
                f"{dose_file}: {exc}"
            ) from exc

        try:
            profile, _ = sample_dose_profile(
                dose_meta,
                candidate["ray_source_xyz"],
                candidate["ray_target_xyz"],
                num_samples=PROFILE_SAMPLES,
            )
        except Exception as exc:
            attempt["Status"] = "profile_sampling_failed"
            attempt["Error"] = str(exc)[:400]
            attempt_rows.append(attempt)
            continue

        if not is_valid_dose_profile(profile):
            d = np.asarray(profile, dtype=float)
            if d.ndim == 1 and d.size:
                peak_idx = int(np.argmax(d))
                peak_value = float(d[peak_idx])
                distal_min = float(np.min(d[peak_idx:]))
                distal_ratio = (distal_min / peak_value) if peak_value > 0 else float("nan")
            else:
                peak_idx, peak_value, distal_min, distal_ratio = -1, float("nan"), float("nan"), float("nan")
            attempt.update({
                "Status": "invalid_dose_profile",
                "PeakIndex": peak_idx,
                "PeakDose": peak_value,
                "DistalMinimumDose": distal_min,
                "DistalMinimumToPeakRatio": distal_ratio,
            })
            attempt_rows.append(attempt)
            continue

        selected = {
            **candidate,
            "dose_file": dose_file,
            "selection_rank": rank,
            "profile_samples": PROFILE_SAMPLES,
        }

        attempt["Status"] = "selected"
        attempt_rows.append(attempt)

        return selected, dose_meta, attempt_rows, {
            "candidate_rays_examined": n_rays,
            "candidate_beamlets_examined": n_beamlets,
            "rays_rejected_short_wepl": n_short,
        }

    return None, None, attempt_rows, {
        "candidate_rays_examined": n_rays,
        "candidate_beamlets_examined": n_beamlets,
        "rays_rejected_short_wepl": n_short,
    }


# ---------------------------------------------------------------------
# Profile output and R80 calculation
# ---------------------------------------------------------------------

def calculate_r_gated(metrics: dict) -> float:
    dose = np.asarray(metrics["dose_profile"], dtype=float)
    remapped = np.asarray(metrics["remapped_dose_profile"], dtype=float)
    depth = np.asarray(metrics["depth_mm"], dtype=float)
    delta_wepl = np.asarray(metrics["delta_wepl_samples_mm"], dtype=float)

    if len(dose) < 3 or len(depth) != len(dose):
        return float("nan")

    step = float(depth[1] - depth[0])
    if not np.isfinite(step) or step <= 0:
        return float("nan")

    gradient = np.gradient(dose, step)
    risk = np.abs(delta_wepl) * np.abs(gradient)
    dose_error = np.abs(remapped - dose)

    max_gradient = float(np.max(np.abs(gradient)))
    if max_gradient <= 0:
        return float("nan")

    gate = np.abs(gradient) > 0.10 * max_gradient
    if gate.sum() < 15:
        gate = np.abs(gradient) > 0.02 * max_gradient

    if gate.sum() <= 5:
        return float("nan")

    if np.std(risk[gate]) == 0 or np.std(dose_error[gate]) == 0:
        return float("nan")

    return float(np.corrcoef(risk[gate], dose_error[gate])[0, 1])


def save_profile(patient_id: str, metrics: dict) -> str:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "Depth_mm": metrics["depth_mm"],
        "Dose_reference": metrics["dose_profile"],
        "Dose_finite_WEPL_remap": metrics["remapped_dose_profile"],
        "WEPL_reference_mm": metrics["wepl_ref_samples_mm"],
        "WEPL_sCT_mm": metrics["wepl_sct_samples_mm"],
        "DeltaWEPL_mm": metrics["delta_wepl_samples_mm"],
    })

    path = PROFILE_DIR / f"{patient_id}_shared_v3_profile.csv"
    df.to_csv(path, index=False, float_format="%.12g")

    return str(path.relative_to(ROOT))


def process_patient(
    cfg: dict,
    calibration: CalibrationProfile,
) -> tuple[dict | None, list[dict], dict]:
    patient_id = str(cfg["Patient"])
    region = str(cfg["Region"])
    metal = bool(cfg["Metal"])
    candidate_order = int(cfg["CandidateOrder"])

    print(
        f"\n[{candidate_order}] {patient_id} "
        f"({region}, Metal={metal})",
        flush=True,
    )

    plan_path = hf_hub_download(
        REPO_ID,
        f"proton/training/{patient_id}/{patient_id}.json",
        repo_type="dataset",
    )
    ct_path = hf_hub_download(
        REPO_ID,
        f"proton/training/{patient_id}/image/ct.mha",
        repo_type="dataset",
    )

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    ct_meta = load_mha(ct_path)

    ct_array = ct_meta["array"]
    spacing = ct_meta["spacing_xyz"]
    origin = ct_meta["origin_xyz"]
    direction = ct_meta.get("direction")

    spr_ref = calibration.density_to_spr.convert(
        calibration.hu_to_density.convert(ct_array)
    )
    spr_sct = calibration.density_to_spr.convert(
        calibration.hu_to_density.convert(ct_array + 50.0)
    )

    selected, dose_meta, attempts, counts = select_shared_v3_beamlet(
        patient_id=patient_id,
        plan=plan,
        spr_ref=spr_ref,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )

    if selected is None:
        status = {
            "Patient": patient_id,
            "CandidateOrder": candidate_order,
            "Region": region,
            "Metal": metal,
            "Status": "no_valid_profile_top4",
            "Reason": "No valid dose profile among shared-v3 top candidates",
            **counts,
        }
        return None, attempts, status

    start = time.perf_counter()

    try:
        metrics = evaluate_selected_beamlet(
            patient_id=patient_id,
            spr_ref=spr_ref,
            spr_sct=spr_sct,
            ct_spacing_xyz=spacing,
            ct_origin_xyz=origin,
            ct_direction=direction,
            dose_meta=dose_meta,
            ray_source_xyz=selected["ray_source_xyz"],
            ray_target_xyz=selected["ray_target_xyz"],
            profile_samples=PROFILE_SAMPLES,
        )
    except Exception as exc:
        status = {
            "Patient": patient_id,
            "CandidateOrder": candidate_order,
            "Region": region,
            "Metal": metal,
            "Status": "shared_r80_evaluation_failed",
            "Reason": str(exc)[:500],
            "SelectedBeamIdx": selected["beam_idx"],
            "SelectedRayIdx": selected["ray_idx"],
            "SelectedBeamletIdx": selected["beamlet_idx"],
            "SelectedEnergyMeV": selected["energy_mev"],
            "SelectedDoseFile": selected["dose_file"],
            "SelectionRank": selected["selection_rank"],
            **counts,
        }
        print(f"  [shared R80 evaluation failed] {exc}")
        return None, attempts, status

    runtime_ms = (time.perf_counter() - start) * 1000.0

    r_gated = calculate_r_gated(metrics)

    result = benchmark_result_row(
        patient_id=patient_id,
        region=region,
        metal=metal,
        selection=selected,
        metrics=metrics,
        r_gated=r_gated,
        runtime_ms=runtime_ms,
    )

    result.update({
        "CohortArm": "model_a_expansion_rebuilt_shared_v3",
        "CohortOrder": candidate_order,
        "CandidateRaysExamined": counts["candidate_rays_examined"],
        "CandidateBeamletsExamined": counts["candidate_beamlets_examined"],
        "RaysRejectedShortWEPL": counts["rays_rejected_short_wepl"],
        "RaySourceXYZ": json_vector(selected["ray_source_xyz"]),
        "RayTargetXYZ": json_vector(selected["ray_target_xyz"]),
        "DoseReferenceType": REFERENCE_TYPE,
        "ProfileAuditCSV": save_profile(patient_id, metrics),
    })

    status = {
        "Patient": patient_id,
        "CandidateOrder": candidate_order,
        "Region": region,
        "Metal": metal,
        "Status": "included",
        "Reason": "",
        "SelectedBeamIdx": selected["beam_idx"],
        "SelectedRayIdx": selected["ray_idx"],
        "SelectedBeamletIdx": selected["beamlet_idx"],
        "SelectedEnergyMeV": selected["energy_mev"],
        "SelectedDoseFile": selected["dose_file"],
        "SelectionRank": selected["selection_rank"],
        "Runtime_ms": runtime_ms,
        **counts,
    }

    print(
        f"  Selected B{selected['beam_idx']} "
        f"R{selected['ray_idx']} L{selected['beamlet_idx']} "
        f"at {selected['energy_mev']:.4f} MeV"
    )
    print(
        f"  FirstOrder={metrics['first_order_mm']:+.4f} mm | "
        f"FiniteMap={metrics['finite_map_mm']:+.4f} mm | "
        f"Error={metrics['linearization_error_mm']:.4f} mm"
    )

    return result, attempts, status


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

def make_summary(results: pd.DataFrame) -> dict:
    """Summarize paired estimators without emitting NaN/Inf JSON values."""
    if results.empty:
        return {"n": 0, "MAE_mm": None, "MedianAE_mm": None,
                "RMSE_mm": None, "Bias_FirstOrder_minus_FiniteMap_mm": None,
                "Pearson_r": None, "MaximumAbsoluteError_mm": None,
                "MedianLinearizationError_mm": None,
                "MedianLinearizationRelativeError": None,
                "MedianSPR_SCT_CV_Interval": None,
                "MaximumSPR_SCT_CV_Interval": None,
                "MedianInverseVsProfileAbsoluteError_mm": None,
                "Interpretation": REFERENCE_TYPE}

    x = results["FirstOrder_mm"].to_numpy(dtype=float)
    y = results["FiniteMap_mm"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    error = x - y
    absolute_error = np.abs(error)

    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
        else None
    )

    def finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    if len(error):
        mae = finite_or_none(np.mean(absolute_error))
        median_ae = finite_or_none(np.median(absolute_error))
        rmse = finite_or_none(np.sqrt(np.mean(error ** 2)))
        bias = finite_or_none(np.mean(error))
        max_ae = finite_or_none(np.max(absolute_error))
    else:
        mae = median_ae = rmse = bias = max_ae = None

    lin_values = results["Linearization_Error_mm"].to_numpy(dtype=float)
    lin_values = lin_values[np.isfinite(lin_values)]
    relative_values = results["Linearization_Relative_Error"].to_numpy(dtype=float)
    relative_values = relative_values[np.isfinite(relative_values)]
    spr_cv_values = results["spr_sct_cv_interval"].to_numpy(dtype=float) if "spr_sct_cv_interval" in results else np.array([])
    spr_cv_values = spr_cv_values[np.isfinite(spr_cv_values)]
    inverse_values = results["InverseVsProfile_mm"].to_numpy(dtype=float)
    inverse_values = np.abs(inverse_values[np.isfinite(inverse_values)])
    lin_median = finite_or_none(np.median(lin_values)) if len(lin_values) else None
    inverse_median = finite_or_none(np.median(inverse_values)) if len(inverse_values) else None

    return {
        "n": int(len(error)),
        "MAE_mm": mae,
        "MedianAE_mm": median_ae,
        "RMSE_mm": rmse,
        "Bias_FirstOrder_minus_FiniteMap_mm": bias,
        "Pearson_r": finite_or_none(pearson) if pearson is not None else None,
        "MaximumAbsoluteError_mm": max_ae,
        "MedianLinearizationError_mm": lin_median,
        "MedianLinearizationRelativeError": finite_or_none(np.median(relative_values)) if len(relative_values) else None,
        "MedianSPR_SCT_CV_Interval": finite_or_none(np.median(spr_cv_values)) if len(spr_cv_values) else None,
        "MaximumSPR_SCT_CV_Interval": finite_or_none(np.max(spr_cv_values)) if len(spr_cv_values) else None,
        "MedianInverseVsProfileAbsoluteError_mm": inverse_median,
        "Interpretation": REFERENCE_TYPE,
    }


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-patients",
        type=int,
        default=None,
        help="Maximum new candidate patients to process in this invocation.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    order = discover_and_freeze_candidate_order()

    results, processing, candidate_log, completed = load_checkpoint()

    included_ids = (
        set(results["Patient"].astype(str))
        if not results.empty else set()
    )

    if len(included_ids) >= TARGET_VALID_N:
        print("[cohort] Thirty valid patients already exist in the checkpoint.")
        print(f"[output] {RESULTS_CSV}")
        return

    calibration = CalibrationProfile()
    newly_processed = 0

    for _, cfg_series in order.iterrows():
        cfg = cfg_series.to_dict()
        patient_id = str(cfg["Patient"])

        if patient_id in completed or patient_id in included_ids:
            continue

        if len(included_ids) >= TARGET_VALID_N:
            break

        if (
            args.limit_patients is not None
            and newly_processed >= args.limit_patients
        ):
            print("[run] Reached --limit-patients. Checkpoint saved.")
            break

        # A non-404 plan/CT error aborts here. This is intentional:
        # we must not replace a patient because of a transient network error.
        result, attempts, status = process_patient(cfg, calibration)

        newly_processed += 1

        if attempts:
            candidate_log = pd.concat(
                [candidate_log, pd.DataFrame(attempts)],
                ignore_index=True,
            )
            atomic_csv(candidate_log, CANDIDATE_LOG_CSV)

        if result is not None:
            results = pd.concat(
                [results, pd.DataFrame([result])],
                ignore_index=True,
            )
            included_ids.add(patient_id)

        processing = pd.concat(
            [processing, pd.DataFrame([status])],
            ignore_index=True,
        )

        # Checkpoint after every completed patient.
        atomic_csv(results, RESULTS_CSV)
        atomic_csv(processing, PROCESSING_CSV)

        cohort_ids = (
            results[[
                "Patient", "Region", "Metal",
                "CohortOrder", "AnalysisVersion",
                "BeamIdx", "RayIdx", "BeamletIdx",
                "EnergyMeV", "DoseFile",
            ]].copy()
            if not results.empty else pd.DataFrame()
        )
        atomic_csv(cohort_ids, COHORT_IDS_CSV)

        print(
            f"[checkpoint] Valid cases: "
            f"{len(included_ids)}/{TARGET_VALID_N}"
        )

    # Re-read results after the loop and write the current summary.
    results = safe_read_csv(RESULTS_CSV)
    processing = safe_read_csv(PROCESSING_CSV)

    unique_n = (
        results["Patient"].astype(str).nunique()
        if not results.empty else 0
    )

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "cohort_definition": (
            "Reconstructed deterministic expansion cohort using sorted "
            "DoseRAD training IDs, RandomState(seed=42), exclusion of the "
            "actual frozen 12-patient stratified cohort and fixed PB "
            "comparison patients, and "
            "explicit hold-out of 1THB121. First 30 patients passing the "
            "shared-v3 top-four dose-profile eligibility and evaluation "
            "procedure are included."
        ),
        "random_seed": RANDOM_SEED,
        "target_valid_n": TARGET_VALID_N,
        "successful_n": int(len(results)),
        "unique_successful_n": int(unique_n),
        "held_out_patient_ids": sorted(HOLDOUT_IDS),
        "fixed_pb_ids_reserved": sorted(FIXED_PB_IDS),
        "outputs": {
            "candidate_order": str(CANDIDATE_ORDER_CSV.relative_to(ROOT)),
            "case_level": str(RESULTS_CSV.relative_to(ROOT)),
            "processing_log": str(PROCESSING_CSV.relative_to(ROOT)),
            "candidate_log": str(CANDIDATE_LOG_CSV.relative_to(ROOT)),
            "frozen_cohort_ids": str(COHORT_IDS_CSV.relative_to(ROOT)),
            "profiles": str(PROFILE_DIR.relative_to(ROOT)),
        },
        "reference_type": REFERENCE_TYPE,
        "agreement_statistics": (
            make_summary(results)
            if len(results) == TARGET_VALID_N and unique_n == TARGET_VALID_N
            else None
        ),
    }

    atomic_json(summary, SUMMARY_JSON)

    print("\n" + "=" * 68)
    print("SHARED-V3 EXPANSION REBUILD")
    print("=" * 68)
    print(f"Valid patients: {unique_n}/{TARGET_VALID_N}")
    print(f"Processing records: {len(processing)}")

    if unique_n == TARGET_VALID_N:
        print("<SUCCESS> The 30-patient shared-v3 cohort is complete.")
        print("\nSummary:")
        print(json.dumps(summary["agreement_statistics"], indent=2))
    else:
        print(
            "<INCOMPLETE> The cohort is not complete. "
            "Rerun the same command to resume from the saved checkpoints."
        )

    print("\nSaved outputs:")
    print(f"  {CANDIDATE_ORDER_CSV}")
    print(f"  {RESULTS_CSV}")
    print(f"  {PROCESSING_CSV}")
    print(f"  {CANDIDATE_LOG_CSV}")
    print(f"  {COHORT_IDS_CSV}")
    print(f"  {SUMMARY_JSON}")

    if unique_n != TARGET_VALID_N:
        print("\nNo final 30-patient expansion statistics should be reported yet.")


if __name__ == "__main__":
    main()

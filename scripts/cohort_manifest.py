"""Shared append-only cohort manifest utilities for SynthQA-core audit runs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_FIELDS = [
    "run_id", "run_timestamp_utc", "script_name", "analysis_id",
    "repository_commit", "working_tree_dirty", "script_sha256", "manifest_helper_sha256",
    "dataset_repo", "dataset_split", "patient_id",
    "anatomical_region", "metal_status", "cohort_selected", "selection_method",
    "perturbation_model", "perturbation_threshold_hu", "perturbation_delta_hu",
    "candidate_rays_examined", "candidate_beamlets_examined", "rays_rejected_short_wepl", "dose_download_failures",
    "invalid_dose_profiles", "dose_engine_failures", "eligible_rays", "selected_beam_idx", "selected_ray_idx",
    "selected_beamlet_idx", "selected_energy_mev", "outcome_status", "exclusion_reason",
    "surrogate_shift_mm", "reference_shift_mm", "surrogate_runtime_ms",
    "reference_runtime_ms", "notes_json"
]

# Shared cohort metadata conventions used by the existing stratified analyses.
# Metal labels follow the project's current METAL_PIDS membership rule; this is
# an operational code-level label, not an independently verified dataset field.
METAL_PIDS = frozenset({
    "1THB016", "1THB021", "1THB029", "1THB031", "1THB052", "1THB054",
    "1THB067", "1THB078", "1THB122", "1THB214", "1THB217", "1ABB138", "1ABB078"
})
STRATIFIED_QUOTAS = (("1THB", True, 3), ("1THB", False, 3),
                     ("1ABB", True, 2), ("1ABB", False, 4))

def patient_metadata(patient_id: str) -> dict[str, Any]:
    """Return transparent project-level labels and their provenance.

    Region is mapped from the dataset's patient-ID namespace used by the
    existing scripts. Metal status uses the repository's explicit METAL_PIDS
    convention. The source caveat is written to each manifest row.
    """
    if patient_id.startswith("1THB"):
        region = "Thorax"
        region_source = "patient-ID prefix rule used by existing stratified scripts"
    elif patient_id.startswith("1ABB"):
        region = "Abdomen"
        region_source = "patient-ID prefix rule used by existing stratified scripts"
    else:
        region = ""
        region_source = "unknown; no mapping rule available"
    if region:
        metal = patient_id in METAL_PIDS
        metal_source = ("project METAL_PIDS membership rule; not independently verified "
                        "against patient-level dataset metadata")
    else:
        metal = ""
        metal_source = "unknown; patient ID has no recognized prefix"
    return {"anatomical_region": region, "metal_status": metal,
            "anatomical_region_source": region_source, "metal_status_source": metal_source}

def discover_training_patient_ids(repo_id: str) -> list[str]:
    """Discover unique patient IDs present in the public training split."""
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted({parts[2] for path in files
                   if (parts := path.split("/")) and len(parts) >= 4
                   and parts[0] == "proton" and parts[1] == "training"})

def discover_stratified_cohort(repo_id: str, patient_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Return the deterministic patient candidates used by the stratified arms.

    The list includes candidates later excluded by analysis failures, so it can
    be reserved from the expansion cohort to prevent patient overlap.
    """
    pids = sorted(set(patient_ids)) if patient_ids is not None else discover_training_patient_ids(repo_id)
    configs: list[dict[str, Any]] = []
    for prefix, metal, quota in STRATIFIED_QUOTAS:
        stratum = [pid for pid in pids if pid.startswith(prefix) and (pid in METAL_PIDS) == metal]
        region = "Thorax" if prefix == "1THB" else "Abdomen"
        configs.extend({"pid": pid, "region": region, "metal": metal} for pid in stratum[:quota])
    return configs


def new_run(script_file: str) -> dict[str, str]:
    script_path = Path(script_file).resolve()
    repo_root = script_path.parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip())
    except Exception:
        commit = "unknown_or_not_a_git_checkout"
        dirty = "unknown"
    def sha256_file(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return "unavailable"
    helper_path = script_path.with_name("cohort_manifest.py")
    return {
        "run_id": str(uuid.uuid4()),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_name": script_path.name,
        "repository_commit": commit,
        "working_tree_dirty": dirty,
        "script_sha256": sha256_file(script_path),
        "manifest_helper_sha256": sha256_file(helper_path),
    }


def manifest_path(script_file: str) -> Path:
    repo_root = Path(script_file).resolve().parent.parent
    override = os.environ.get("SYNTHQA_MANIFEST_PATH")
    return Path(override).expanduser().resolve() if override else repo_root / "outputs" / "reproducibility" / "cohort_manifest.csv"


def append_manifest_rows(rows: Iterable[dict[str, Any]], script_file: str) -> Path | None:
    rows = list(rows)
    if not rows:
        return None
    path = manifest_path(script_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        for row in rows:
            normalized = {k: row.get(k, "") for k in MANIFEST_FIELDS}
            if isinstance(normalized.get("notes_json"), (dict, list)):
                normalized["notes_json"] = json.dumps(normalized["notes_json"], ensure_ascii=False, sort_keys=True)
            writer.writerow(normalized)
    print(f"[manifest] Appended {len(rows)} records to {path}")
    return path


def base_record(run: dict[str, str], analysis_id: str, patient_id: str, **kwargs: Any) -> dict[str, Any]:
    row = {
        **run,
        "analysis_id": analysis_id,
        "dataset_repo": "LMUK-RADONC-PHYS-RES/DoseRAD2026",
        "dataset_split": "proton/training",
        "patient_id": patient_id,
        "cohort_selected": True,
    }
    row.update(kwargs)
    return row

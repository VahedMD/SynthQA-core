#!/usr/bin/env python3
"""Run exact shared-v3 reproducibility audits for a benchmark cohort.

Run from the repository root after benchmark_dataset.py:
    python scripts/audit_stratified_r80_batch.py --exclude-patient 1THB002

Each patient is audited serially using the exact beamlet provenance recorded in
model_a_stratified_case_level.csv. Failures are recorded and do not prevent
subsequent patients from being audited. No cohort discovery or full Hugging
Face repository listing is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_stratified_r80_case import audit, DEFAULT_BENCHMARK, DEFAULT_OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude-patient", action="append", default=[],
        help="Patient ID to skip; may be supplied more than once.",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-6,
        help="Absolute tolerance for benchmark/audit numeric comparisons.",
    )
    args = parser.parse_args()
    if args.atol <= 0:
        parser.error("--atol must be positive")
    if not args.benchmark_csv.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {args.benchmark_csv}")

    frame = pd.read_csv(args.benchmark_csv, dtype={"Patient": str})
    required = {"Patient", "AnalysisVersion"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Benchmark CSV is missing required columns: {missing}")
    if frame["Patient"].duplicated().any():
        duplicates = frame.loc[frame["Patient"].duplicated(keep=False), "Patient"].tolist()
        raise RuntimeError(f"Benchmark CSV contains duplicate patient rows: {duplicates}")

    excluded = {str(x).strip() for x in args.exclude_patient if str(x).strip()}
    patient_ids = [str(x) for x in frame["Patient"].tolist() if str(x) not in excluded]
    if not patient_ids:
        raise RuntimeError("No patients remain after applying --exclude-patient")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"[batch] Benchmark: {args.benchmark_csv}")
    print(f"[batch] Patients to audit: {len(patient_ids)}")
    if excluded:
        print(f"[batch] Explicitly skipped: {', '.join(sorted(excluded))}")

    for index, patient_id in enumerate(patient_ids, start=1):
        print(f"\n[{index}/{len(patient_ids)}] Auditing {patient_id}", flush=True)
        started = time.perf_counter()
        try:
            code = audit(patient_id, args.benchmark_csv, args.output_dir, args.atol)
            status = "PASS" if code == 0 else "MISMATCH"
            error = "" if code == 0 else f"audit returned exit code {code}"
        except Exception as exc:  # keep cohort audit moving; record exact failure
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
            print(f"[batch] ERROR for {patient_id}: {error}", flush=True)
        elapsed = time.perf_counter() - started
        records.append({
            "Patient": patient_id,
            "Status": status,
            "Elapsed_s": elapsed,
            "Error": error,
            "Audit_JSON": str(args.output_dir / f"{patient_id}_shared_v3_audit.json"),
            "Profile_CSV": str(args.output_dir / f"{patient_id}_shared_v3_profile.csv"),
            "Voxel_Segments_CSV": str(args.output_dir / f"{patient_id}_shared_v3_voxel_segments.csv"),
        })

    summary = pd.DataFrame(records)
    summary_csv = args.output_dir / "stratified_shared_v3_batch_summary.csv"
    summary_json = args.output_dir / "stratified_shared_v3_batch_summary.json"
    summary.to_csv(summary_csv, index=False, float_format="%.6f")
    payload = {
        "benchmark_csv": str(args.benchmark_csv),
        "output_dir": str(args.output_dir),
        "comparison_atol": args.atol,
        "patients_requested": len(patient_ids),
        "patients_passed": int((summary["Status"] == "PASS").sum()),
        "patients_mismatched": int((summary["Status"] == "MISMATCH").sum()),
        "patients_errored": int((summary["Status"] == "ERROR").sum()),
        "explicitly_skipped": sorted(excluded),
        "results": records,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(
        f"BATCH AUDIT: PASS={payload['patients_passed']} | "
        f"MISMATCH={payload['patients_mismatched']} | "
        f"ERROR={payload['patients_errored']} | n={len(patient_ids)}"
    )
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("=" * 72)
    return 0 if payload["patients_passed"] == len(patient_ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Recompute the frozen n=30 expansion cohort with dual WEPL estimators.

Run from the repository root:
    python scripts/rebenchmark_expansion_dual_estimators.py

The existing refreshed case-level CSV is used only to recover the frozen patient
order. Patient selection is not repeated. New values are written to a new file.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reviewer_stats import run_cohort_expansion

SOURCE = ROOT / "outputs/reproducibility/refresh_excl_1THB121/model_a_expansion_case_level_n30.csv"
DEST = ROOT / "outputs/reproducibility/model_a_expansion_dual_estimator_n30.csv"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Frozen expansion cohort file not found: {SOURCE}")
    frozen = pd.read_csv(SOURCE)
    if len(frozen) != 30 or frozen["Patient"].astype(str).nunique() != 30:
        raise ValueError("Expected exactly 30 unique patients in the frozen expansion cohort")
    if "1THB121" in set(frozen["Patient"].astype(str)):
        raise ValueError("1THB121 must remain excluded from this frozen sensitivity cohort")
    if "1ABB143" not in set(frozen["Patient"].astype(str)):
        raise ValueError("Expected replacement patient 1ABB143 is missing")

    result = run_cohort_expansion(n_patients=30, fixed_patient_ids=frozen["Patient"].astype(str).tolist())
    if len(result) != 30 or result["Patient"].astype(str).nunique() != 30:
        raise RuntimeError(f"Dual-estimator recomputation incomplete: rows={len(result)}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(DEST, index=False, float_format="%.10g")
    print(f"Saved {len(result)} frozen-cohort dual-estimator rows: {DEST}")
    print("Reminder: FiniteMap_mm is analytical WEPL remapping, not independent dose recalculation.")


if __name__ == "__main__":
    main()

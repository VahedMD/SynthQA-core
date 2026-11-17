"""Summarize first-order versus finite WEPL-mapping diagnostics.

Run from the repository root. The script does not select or exclude cases.
It reports the observed distribution and ranks cases by linearization error.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / "outputs/reproducibility/model_a_stratified_case_level.csv",
    ROOT / "outputs/reproducibility/model_a_expansion_shared_v3_n30.csv",
    ROOT / "outputs/reproducibility/model_a_expansion_dual_estimator_n30.csv",
]
OUT = ROOT / "outputs/reproducibility/linearization_diagnostics_summary.csv"
REQUIRED = ["Patient", "FirstOrder_mm", "FiniteMap_mm", "Linearization_Error_mm",
            "Linearization_Relative_Error"]
OPTIONAL = ["mean_spr_sct_interval", "spr_sct_cv_interval", "spr_sct_variation_interval"]


def main():
    frames = []
    seen = set()
    for path in INPUTS:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"[skip] {path.name}: missing {missing}")
            continue
        df = df.copy()
        for column in OPTIONAL:
            if column not in df.columns:
                df[column] = np.nan
        df["SourceFile"] = path.name
        # The same patient can legitimately appear in the stratified and fixed PB
        # cohorts. Keep source provenance; do not silently deduplicate.
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No cohort CSV with the required dual-estimator diagnostics was found.")
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("Linearization_Error_mm", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, float_format="%.10g")
    print(f"Saved {len(result)} case-level diagnostic rows: {OUT}")
    for source, group in result.groupby("SourceFile", sort=False):
        e = group["Linearization_Error_mm"].dropna()
        print(f"\n{source}: n={len(group)} | median abs error={e.median():.3f} mm | "
              f"IQR={e.quantile(.75)-e.quantile(.25):.3f} mm | max={e.max():.3f} mm")
        print(group[["Patient", "FirstOrder_mm", "FiniteMap_mm", "Linearization_Error_mm",
                     "mean_spr_sct_interval", "spr_sct_cv_interval"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

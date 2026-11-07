
"""Audit case-level CSVs for SynthQA-core.

Read-only with respect to the source CSVs.
Writes audit reports to outputs/reproducibility/audit/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(pred: pd.Series, ref: pd.Series) -> dict:
    """Metrics with bias = prediction - reference."""
    pair = pd.DataFrame({"pred": pred, "ref": ref}).dropna()
    pair = pair[np.isfinite(pair["pred"]) & np.isfinite(pair["ref"])]

    if pair.empty:
        return {"n": 0}

    error = pair["pred"].to_numpy() - pair["ref"].to_numpy()
    p = pair["pred"].to_numpy()
    r = pair["ref"].to_numpy()

    return {
        "n": int(len(pair)),
        "mae_mm": float(np.mean(np.abs(error))),
        "bias_pred_minus_ref_mm": float(np.mean(error)),
        "median_absolute_error_mm": float(np.median(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "pearson_r": (
            float(np.corrcoef(p, r)[0, 1])
            if len(pair) >= 2 and np.std(p) > 0 and np.std(r) > 0
            else None
        ),
        "max_absolute_error_mm": float(np.max(np.abs(error))),
    }


def paired_bootstrap(
    pred: pd.Series,
    ref: pd.Series,
    *,
    n_boot: int = 20_000,
    seed: int = 42,
) -> dict:
    """Percentile CIs from paired resampling; reports observed point estimates."""
    data = pd.DataFrame({"pred": pred, "ref": ref}).dropna()
    data = data[
        np.isfinite(data["pred"]) & np.isfinite(data["ref"])
    ].reset_index(drop=True)

    point = metrics(data["pred"], data["ref"])
    n = len(data)

    if n < 3:
        return {"point_estimates": point, "ci95": None}

    p = data["pred"].to_numpy()
    r = data["ref"].to_numpy()
    rng = np.random.default_rng(seed)

    maes = np.empty(n_boot)
    biases = np.empty(n_boot)
    correlations = np.full(n_boot, np.nan)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        error = p[idx] - r[idx]
        maes[b] = np.mean(np.abs(error))
        biases[b] = np.mean(error)

        if np.std(p[idx]) > 0 and np.std(r[idx]) > 0:
            correlations[b] = np.corrcoef(p[idx], r[idx])[0, 1]

    def ci(values: np.ndarray) -> list[float] | None:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    return {
        "point_estimates": point,
        "ci95": {
            "mae_mm": ci(maes),
            "bias_pred_minus_ref_mm": ci(biases),
            "pearson_r": ci(correlations),
        },
        "bootstrap_replicates": n_boot,
        "seed": seed,
        "bootstrap_method": "paired percentile bootstrap",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--expansion",
        type=Path,
        default=None,
        help="Path to model_a_expansion_case_level.csv",
    )
    parser.add_argument(
        "--pb",
        type=Path,
        default=None,
        help="Path to reviewer_pb_comparison_case_level.csv",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "outputs" / "reproducibility" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    expansion_path = args.expansion or (
        root / "outputs" / "reproducibility"
        / "model_a_expansion_case_level.csv"
    )
    pb_path = args.pb or (
        root / "outputs" / "reproducibility"
        / "reviewer_pb_comparison_case_level.csv"
    )

    report: dict = {
        "repository_root": str(root),
        "bias_convention": "prediction minus reference",
        "source_files": {
            "expansion": str(expansion_path),
            "pb_comparison": str(pb_path),
        },
        "warnings": [],
    }

    # ---------------------------------------------------------------
    # A. Model A expansion cohort
    # ---------------------------------------------------------------
    if expansion_path.exists():
        exp = pd.read_csv(expansion_path)
        required = {"Patient", "Pred_mm", "Act_mm"}
        missing = required - set(exp.columns)
        if missing:
            raise ValueError(
                f"Expansion CSV missing columns: {sorted(missing)}"
            )

        exp["Patient"] = exp["Patient"].astype(str).str.strip()
        exp["Pred_mm"] = pd.to_numeric(exp["Pred_mm"], errors="coerce")
        exp["Act_mm"] = pd.to_numeric(exp["Act_mm"], errors="coerce")
        exp["Error_pred_minus_ref_mm"] = exp["Pred_mm"] - exp["Act_mm"]
        exp["Abs_error_mm"] = exp["Error_pred_minus_ref_mm"].abs()

        exp.to_csv(
            out_dir / "expansion_case_audit.csv",
            index=False,
            float_format="%.10g",
        )

        report["expansion"] = {
            "rows_in_file": int(len(exp)),
            "unique_patient_ids": int(exp["Patient"].nunique()),
            "duplicate_patient_ids": (
                exp.loc[exp["Patient"].duplicated(keep=False), "Patient"]
                .drop_duplicates().tolist()
            ),
            "missing_numeric_pairs": int(
                (exp[["Pred_mm", "Act_mm"]].notna().sum(axis=1) < 2).sum()
            ),
            "metrics_all_valid_rows": paired_bootstrap(
                exp["Pred_mm"], exp["Act_mm"]
            ),
        }

        # Full-cohort metrics and a clearly labeled diagnostic analysis
        # excluding the extreme-error case, without altering source data.
        target = exp[exp["Patient"] == "1THB121"]
        report["expansion"]["target_case_1THB121"] = (
            target[
                [
                    c for c in [
                        "Patient", "Region", "Metal", "Pred_mm", "Act_mm",
                        "Error_pred_minus_ref_mm", "Abs_error_mm",
                    ] if c in target.columns
                ]
            ].to_dict(orient="records")
        )

        without_target = exp[exp["Patient"] != "1THB121"]
        report["expansion"]["diagnostic_without_1THB121"] = metrics(
            without_target["Pred_mm"], without_target["Act_mm"]
        )

        # Rank all cases by absolute error.
        top = exp.sort_values("Abs_error_mm", ascending=False)
        top.head(10).to_csv(
            out_dir / "expansion_top_10_absolute_errors.csv",
            index=False,
            float_format="%.10g",
        )

        # Report whether region/metal metadata is missing or unexpected.
        if "Region" in exp.columns:
            report["expansion"]["region_counts"] = (
                exp["Region"].fillna("<missing>").value_counts().to_dict()
            )
        if "Metal" in exp.columns:
            report["expansion"]["metal_counts"] = (
                exp["Metal"].fillna("<missing>").astype(str)
                .value_counts().to_dict()
            )
    else:
        report["warnings"].append(
            f"Expansion CSV not found: {expansion_path}"
        )

    # ---------------------------------------------------------------
    # B. Fixed PB comparison
    # ---------------------------------------------------------------
    if pb_path.exists():
        pb = pd.read_csv(pb_path)
        required = {"Patient", "Pred_mm", "PB_shift_mm"}
        missing = required - set(pb.columns)
        if missing:
            raise ValueError(
                f"PB comparison CSV missing columns: {sorted(missing)}"
            )

        pb["Patient"] = pb["Patient"].astype(str).str.strip()
        pb["Pred_mm"] = pd.to_numeric(pb["Pred_mm"], errors="coerce")
        pb["PB_shift_mm"] = pd.to_numeric(
            pb["PB_shift_mm"], errors="coerce"
        )
        pb["Error_pred_minus_ref_mm"] = (
            pb["Pred_mm"] - pb["PB_shift_mm"]
        )
        pb["Abs_error_mm"] = pb["Error_pred_minus_ref_mm"].abs()

        pb.to_csv(
            out_dir / "pb_comparison_case_audit.csv",
            index=False,
            float_format="%.10g",
        )

        report["pb_comparison"] = {
            "rows_in_file": int(len(pb)),
            "unique_patient_ids": int(pb["Patient"].nunique()),
            "duplicate_patient_ids": (
                pb.loc[pb["Patient"].duplicated(keep=False), "Patient"]
                .drop_duplicates().tolist()
            ),
            "metrics_all_valid_rows": paired_bootstrap(
                pb["Pred_mm"], pb["PB_shift_mm"]
            ),
            "diagnostic_without_1THB002": metrics(
                pb.loc[pb["Patient"] != "1THB002", "Pred_mm"],
                pb.loc[pb["Patient"] != "1THB002", "PB_shift_mm"],
            ),
        }

        # Check runtime-derived speedup if both timing columns exist.
        if {"Surrogate_ms", "PB_total_ms"}.issubset(pb.columns):
            pb["Surrogate_ms"] = pd.to_numeric(
                pb["Surrogate_ms"], errors="coerce"
            )
            pb["PB_total_ms"] = pd.to_numeric(
                pb["PB_total_ms"], errors="coerce"
            )
            valid_times = pb.dropna(
                subset=["Surrogate_ms", "PB_total_ms"]
            )
            valid_times = valid_times[
                (valid_times["Surrogate_ms"] > 0)
                & (valid_times["PB_total_ms"] > 0)
            ]
            report["pb_comparison"]["runtime"] = {
                "n_valid": int(len(valid_times)),
                "median_surrogate_ms": float(
                    valid_times["Surrogate_ms"].median()
                ) if len(valid_times) else None,
                "median_pb_total_ms": float(
                    valid_times["PB_total_ms"].median()
                ) if len(valid_times) else None,
                "median_of_per_case_speedups": float(
                    (valid_times["PB_total_ms"]
                     / valid_times["Surrogate_ms"]).median()
                ) if len(valid_times) else None,
                "ratio_of_medians": float(
                    valid_times["PB_total_ms"].median()
                    / valid_times["Surrogate_ms"].median()
                ) if len(valid_times) else None,
            }
    else:
        report["warnings"].append(
            f"PB comparison CSV not found: {pb_path}"
        )

    report_path = out_dir / "case_level_audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Audit report written to: {report_path}")
    print(f"Audit outputs directory: {out_dir}")
    print("No source CSVs were modified.")
    print("Review the JSON report and the generated case-audit CSVs.")


if __name__ == "__main__":
    main()

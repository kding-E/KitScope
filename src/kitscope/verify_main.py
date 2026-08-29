from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .io import first_column, read_table, resolve_data_path, sha256_file
from .metrics import binary_label, binary_metrics, confusion_counts, recall_at_fpr_budget


def canonical_predictions(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    id_column = first_column(frame, ("capture_id", "sample_id"), "capture identity")
    label_column = first_column(frame, ("y", "label", "y_true", "truth"), "label")
    score_column = first_column(frame, ("session_score", "score", "prob_1", "phishing_probability"), "score")
    result = pd.DataFrame(
        {
            "capture_id": frame[id_column].astype(str),
            "y": binary_label(frame[label_column]),
            "score": pd.to_numeric(frame[score_column], errors="raise").astype(float),
        }
    )
    if "y_pred" in frame.columns:
        result["saved_y_pred"] = pd.to_numeric(frame["y_pred"], errors="raise").astype(int)
    if result["capture_id"].duplicated().any():
        raise ValueError(f"duplicate capture_id values in {path}")
    return result


def load_url_weights(path: Path, stage: str) -> pd.DataFrame:
    weights = read_table(path)
    required = {"capture_id", "label", "url_group_stage", "url_group_weight_raw"}
    missing = sorted(required.difference(weights.columns))
    if missing:
        raise ValueError(f"URL weights file is missing columns: {missing}")
    weights = weights[weights["url_group_stage"].astype(str).eq(str(stage))].copy()
    if weights.empty:
        raise ValueError(f"URL weights contain no rows for stage={stage!r}")
    weights["capture_id"] = weights["capture_id"].astype(str)
    if weights["capture_id"].duplicated().any():
        raise ValueError(f"duplicate capture_id values in URL weights: {path}")
    weights["url_group_weight_raw"] = pd.to_numeric(weights["url_group_weight_raw"], errors="raise").astype(float)
    return weights


def pct(value: float) -> float:
    return float(value) * 100.0


def rounded(value: float, decimals: int) -> float:
    return round(float(value), int(decimals))


def compare_expected(computed: dict[str, Any], expected: dict[str, Any], decimals: int) -> list[str]:
    failures: list[str] = []
    count_keys = ("n", "positives", "benign", "tp", "fp", "tn", "fn")
    metric_keys = ("f1_pct", "fpr_pct", "r_at_3_pct", "ap_pct", "u_f1_pct", "u_fpr_pct")
    for key in count_keys:
        if key in expected and int(computed[key]) != int(expected[key]):
            failures.append(f"{key}: computed={computed[key]} expected={expected[key]}")
    for key in metric_keys:
        if key in expected and rounded(computed[key], decimals) != rounded(expected[key], decimals):
            failures.append(
                f"{key}: computed={computed[key]:.6f} expected={float(expected[key]):.6f}"
            )
    return failures


def verify_method(
    method_name: str,
    spec: dict[str, Any],
    data_root: Path,
    weights: pd.DataFrame,
    fpr_budget: float,
    decimals: int,
) -> dict[str, Any]:
    prediction_path = resolve_data_path(data_root, str(spec["prediction_file"]))
    predictions = canonical_predictions(prediction_path)
    threshold = float(spec["threshold"])
    truth = predictions["y"].to_numpy(dtype=int)
    score = predictions["score"].to_numpy(dtype=float)
    deployed_pred = (score >= threshold).astype(int)
    counts = confusion_counts(truth, deployed_pred)
    main = binary_metrics(truth, score, threshold)
    rank = recall_at_fpr_budget(truth, score, fpr_budget)

    joined = predictions.merge(
        weights[["capture_id", "url_group_weight_raw"]], on="capture_id", how="left", validate="one_to_one"
    )
    if joined["url_group_weight_raw"].isna().any():
        missing = joined.loc[joined["url_group_weight_raw"].isna(), "capture_id"].head(5).tolist()
        raise ValueError(f"{method_name}: missing URL weights for captures: {missing}")
    url_metrics = binary_metrics(
        truth,
        score,
        threshold,
        sample_weight=joined["url_group_weight_raw"].to_numpy(dtype=float),
    )

    saved_decision_mismatches = None
    if "saved_y_pred" in predictions.columns:
        saved_decision_mismatches = int((predictions["saved_y_pred"].to_numpy(dtype=int) != deployed_pred).sum())

    computed = {
        "n": int(len(predictions)),
        "positives": int((truth == 1).sum()),
        "benign": int((truth == 0).sum()),
        "tp": counts.tp,
        "fp": counts.fp,
        "tn": counts.tn,
        "fn": counts.fn,
        "f1_pct": pct(main["f1"]),
        "fpr_pct": pct(main["fpr"]),
        "r_at_3_pct": pct(rank["recall"]),
        "ap_pct": pct(main["average_precision"]),
        "u_f1_pct": pct(url_metrics["f1"]),
        "u_fpr_pct": pct(url_metrics["fpr"]),
    }
    failures = compare_expected(computed, spec.get("expected", {}), decimals)
    if saved_decision_mismatches not in (None, 0):
        failures.append(f"saved_y_pred differs from threshold decisions on {saved_decision_mismatches} rows")

    return {
        "method": method_name,
        "representation": str(spec.get("representation", "")),
        "prediction_file": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "threshold": threshold,
        "r_at_3_threshold": rank["threshold"],
        "saved_decision_mismatches": saved_decision_mismatches,
        "computed": computed,
        "expected": spec.get("expected", {}),
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# KitScope Main Random Verification",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Method | F1 | FPR | R@3 | AP | U-F1 | U-FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        c = method["computed"]
        lines.append(
            "| {method} | {f1:.2f} | {fpr:.2f} | {r3:.2f} | {ap:.2f} | {uf1:.2f} | {ufpr:.2f} |".format(
                method=method["method"],
                f1=c["f1_pct"],
                fpr=c["fpr_pct"],
                r3=c["r_at_3_pct"],
                ap=c["ap_pct"],
                uf1=c["u_f1_pct"],
                ufpr=c["u_fpr_pct"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_verification(config_path: Path, data_root: Path, out_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = resolve_data_path(data_root, str(config["url_weights"]))
    weights = load_url_weights(weights_path, str(config.get("url_weight_stage", "evaluation")))
    decimals = int(config.get("rounding_decimals", 2))
    methods = [
        verify_method(
            name,
            spec,
            data_root,
            weights,
            float(config.get("fpr_budget", 0.03)),
            decimals,
        )
        for name, spec in config["methods"].items()
    ]
    failures = {method["method"]: method["failures"] for method in methods if method["failures"]}
    report = {
        "schema": "kitscope_main_random_verification",
        "status": "passed" if not failures else "failed",
        "config": str(config_path),
        "data_root": str(data_root),
        "dataset_url": config.get("dataset_url"),
        "split": config.get("split"),
        "seed": config.get("seed"),
        "fpr_budget": float(config.get("fpr_budget", 0.03)),
        "url_weights": str(weights_path),
        "url_weights_sha256": sha256_file(weights_path),
        "methods": methods,
        "failures": failures,
    }
    (out_dir / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, out_dir / "verification_report.md")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify KitScope metrics.")
    parser.add_argument("--config", type=Path, default=Path("configs/main_random.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/kitscope-release"))
    parser.add_argument("--out", type=Path, default=Path("results/main_random"))
    parser.add_argument("--no-strict", action="store_true", help="write the report but return success on mismatch")
    parser.add_argument("--print-json", action="store_true", help="print the verification report to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_verification(args.config, args.data_root, args.out)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed" and not args.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

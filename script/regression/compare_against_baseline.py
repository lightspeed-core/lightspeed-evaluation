#!/usr/bin/env python3
"""Baseline comparison script for OKP quality regression gating.

Compares a current evaluation run against a baseline run and determines
whether regressions have occurred. Two modes:

  --check-only --> Print "regression" or "ok" (for OKP image comparison)
  --fail-on-critical-regression --> Exit non-zero if critical metrics regressed (for PR gating)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CRITICAL_METRICS = {
    "ragas:faithfulness",
    "custom:answer_correctness",
    "ragas:context_precision_without_reference",
    "ragas:response_relevancy",
}

NON_CRITICAL_METRICS = {
    "custom:intent_eval",
    "custom:keywords_eval",
    "deepeval:knowledge_retention",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results against a baseline for regression detection.",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to the current run's output directory.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline run's output directory.",
    )
    parser.add_argument(
        "--output",
        help="Path to write the markdown regression summary.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help='Print "regression" or "ok" to stdout. For OKP image comparison.',
    )
    parser.add_argument(
        "--fail-on-critical-regression",
        action="store_true",
        help="Exit non-zero if any critical metric regressed. For PR gating.",
    )
    parser.add_argument(
        "--critical-delta",
        type=float,
        default=0.03,
        help="Allowed mean score drop for critical metrics (default: 0.03).",
    )
    parser.add_argument(
        "--warn-delta",
        type=float,
        default=0.03,
        help="Score drop threshold for non-critical warnings (default: 0.03).",
    )
    return parser.parse_args()


def find_and_load_summary(directory: str) -> dict[str, Any]:
    """Find and load the summary JSON file from an evaluation output directory.

    Args:
        directory: Path to an evaluation output directory.

    Returns:
        Parsed JSON content of the summary file.

    Raises:
        FileNotFoundError: If no summary JSON is found.
        RuntimeError: If multiple summary JSONs are found.
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    summary_files = list(dir_path.glob("*_summary.json"))

    if len(summary_files) == 0:
        raise FileNotFoundError(f"No *_summary.json file found in: {directory}")

    if len(summary_files) > 1:
        raise RuntimeError(
            f"Multiple summary files found in {directory}: "
            f"{[f.name for f in summary_files]}. Expected exactly one."
        )

    with open(summary_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metric_deltas(
    baseline_data: dict[str, Any],
    current_data: dict[str, Any],
    critical_delta: float,
    warn_delta: float,
) -> list[dict[str, Any]]:
    """Compute per-metric deltas between baseline and current runs.

    Args:
        baseline_data: Parsed baseline summary JSON.
        current_data: Parsed current run summary JSON.
        critical_delta: Allowed score drop for critical metrics.
        warn_delta: Score drop threshold for non-critical warnings.

    Returns:
        List of dicts, one per metric, with delta and status.
    """
    baseline_metrics = baseline_data.get("summary_stats", {}).get("by_metric", {})
    current_metrics = current_data.get("summary_stats", {}).get("by_metric", {})

    all_metrics = set(baseline_metrics.keys()) | set(current_metrics.keys())
    results = []

    for metric in sorted(all_metrics):
        baseline_stats = baseline_metrics.get(metric, {})
        current_stats = current_metrics.get(metric, {})

        baseline_mean = baseline_stats.get("score_statistics", {}).get("mean")
        current_mean = current_stats.get("score_statistics", {}).get("mean")

        baseline_pass_rate = baseline_stats.get("pass_rate")
        current_pass_rate = current_stats.get("pass_rate")

        if baseline_mean is not None and current_mean is not None:
            score_delta = current_mean - baseline_mean
        else:
            score_delta = None

        if baseline_pass_rate is not None and current_pass_rate is not None:
            pass_rate_delta = current_pass_rate - baseline_pass_rate
        else:
            pass_rate_delta = None

        is_critical = metric in CRITICAL_METRICS
        threshold = critical_delta if is_critical else warn_delta

        if score_delta is not None and score_delta < -threshold:
            status = "FAIL" if is_critical else "WARN"
        else:
            status = "PASS"

        results.append(
            {
                "metric": metric,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "score_delta": score_delta,
                "baseline_pass_rate": baseline_pass_rate,
                "current_pass_rate": current_pass_rate,
                "pass_rate_delta": pass_rate_delta,
                "is_critical": is_critical,
                "status": status,
            }
        )

    return results


def generate_markdown_summary(
    deltas: list[dict[str, Any]],
    baseline_data: dict[str, Any],
    current_data: dict[str, Any],
) -> str:
    """Generate a markdown regression summary table."""
    lines = [
        "# Regression Comparison Summary",
        "",
        f"- **Baseline:** {baseline_data['total_evaluations']} evaluations "
        f"({baseline_data['timestamp']})",
        f"- **Current:** {current_data['total_evaluations']} evaluations "
        f"({current_data['timestamp']})",
        "",
        "| Metric | Baseline | Current | Delta | Status |",
        "|---|---:|---:|---:|---|",
    ]

    for d in deltas:
        bm = f"{d['baseline_mean']:.3f}" if d["baseline_mean"] is not None else "N/A"
        cm = f"{d['current_mean']:.3f}" if d["current_mean"] is not None else "N/A"
        sd = f"{d['score_delta']:+.3f}" if d["score_delta"] is not None else "N/A"
        status = d["status"]
        if status == "FAIL":
            status_display = "**FAIL**"
        elif status == "WARN":
            status_display = "WARN"
        else:
            status_display = "PASS"
        lines.append(f"| {d['metric']} | {bm} | {cm} | {sd} | {status_display} |")

    has_critical_fail = any(d["status"] == "FAIL" and d["is_critical"] for d in deltas)
    has_warn = any(d["status"] == "WARN" for d in deltas)

    lines.append("")
    if has_critical_fail:
        lines.append("**Result: REGRESSION** (critical metrics degraded)")
    elif has_warn:
        lines.append("**Result: WARNING** (non-critical metrics degraded)")
    else:
        lines.append("**Result: PASS** (no regressions detected)")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Entry point."""
    args = parse_args()

    try:
        baseline_data = find_and_load_summary(args.baseline)
        current_data = find_and_load_summary(args.current)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print(
        f"Baseline: {baseline_data['total_evaluations']} evaluations "
        f"({baseline_data['timestamp']})"
    )
    print(
        f"Current:  {current_data['total_evaluations']} evaluations "
        f"({current_data['timestamp']})"
    )

    deltas = compute_metric_deltas(
        baseline_data,
        current_data,
        args.critical_delta,
        args.warn_delta,
    )

    has_critical_fail = any(d["status"] == "FAIL" and d["is_critical"] for d in deltas)

    if args.check_only:
        print("regression" if has_critical_fail else "ok")
        return 0

    print(
        f"\n{'Metric':<50} {'Baseline':>10} {'Current':>10} "
        f"{'Delta':>10} {'Status':>8}"
    )
    print("-" * 92)
    for d in deltas:
        bm = f"{d['baseline_mean']:.3f}" if d["baseline_mean"] is not None else "N/A"
        cm = f"{d['current_mean']:.3f}" if d["current_mean"] is not None else "N/A"
        sd = f"{d['score_delta']:+.3f}" if d["score_delta"] is not None else "N/A"
        print(f"{d['metric']:<50} {bm:>10} {cm:>10} {sd:>10} {d['status']:>8}")

    if args.output:
        markdown = generate_markdown_summary(deltas, baseline_data, current_data)
        output_path = Path(args.output)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"\nMarkdown summary written to: {output_path}")

    if args.fail_on_critical_regression and has_critical_fail:
        print("\nCritical regression detected — exiting with code 1.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

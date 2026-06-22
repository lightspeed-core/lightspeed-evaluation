#!/usr/bin/env python3
"""Three-run A/B/C comparison for regression attribution.

Compares three evaluation runs to determine whether a quality
regression was caused by the PR code change or by OKP data changes:

  Run A: main branch + old OKP image  (known-good baseline)
  Run B: main branch + new OKP image  (isolates OKP impact)
  Run C: PR branch   + new OKP image  (the PR under test)

Pairwise comparisons:
  A vs B  -->  OKP data impact
  B vs C  -->  PR code impact (controlling for OKP changes)
  A vs C  -->  Total quality change
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from script.regression.compare_against_baseline import (
    compute_metric_deltas,
    find_and_load_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Three-run A/B/C regression comparison with attribution.",
    )
    parser.add_argument(
        "--run-a",
        help="Path to Run A results (main + old OKP). Optional.",
    )
    parser.add_argument(
        "--run-b",
        help="Path to Run B results (main + new OKP). Optional.",
    )
    parser.add_argument(
        "--run-c",
        required=True,
        help="Path to Run C results (PR branch + new OKP). Required.",
    )
    parser.add_argument(
        "--baseline",
        help="Fallback baseline directory when Run A/B are unavailable.",
    )
    parser.add_argument(
        "--output",
        help="Path to write the markdown regression summary.",
    )
    parser.add_argument(
        "--fail-on-pr-regression",
        action="store_true",
        help="Exit non-zero if PR caused critical regression (B vs C).",
    )
    parser.add_argument(
        "--critical-delta",
        type=float,
        default=0.03,
        help="Allowed score drop for critical metrics (default: 0.03).",
    )
    parser.add_argument(
        "--warn-delta",
        type=float,
        default=0.03,
        help="Score drop threshold for non-critical warnings (default: 0.03).",
    )
    return parser.parse_args()


def determine_gate_verdict(
    okp_deltas: list[dict[str, Any]] | None,
    pr_deltas: list[dict[str, Any]] | None,
    total_deltas: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    """Determine the gate verdict from pairwise comparison results.

    Args:
        okp_deltas: A vs B deltas (OKP impact). None if Run A unavailable.
        pr_deltas: B vs C deltas (PR impact). None if Run B unavailable.
        total_deltas: A vs C deltas (total impact). None if Run A unavailable.

    Returns:
        Tuple of (verdict, explanation) where verdict is PASS, WARN, or FAIL.
    """
    pr_has_critical = _has_critical_regression(pr_deltas)
    okp_has_critical = _has_critical_regression(okp_deltas)
    total_has_critical = _has_critical_regression(total_deltas)

    if pr_deltas is not None and pr_has_critical:
        return ("FAIL", "PR caused critical regression (B vs C).")

    if okp_has_critical and not pr_has_critical:
        return ("WARN", "OKP data caused regression, not the PR (A vs B).")

    if total_has_critical and pr_deltas is None:
        return ("FAIL", "Critical regression detected vs baseline (A vs C).")

    has_any_warn = (
        _has_noncritical_regression(okp_deltas)
        or _has_noncritical_regression(pr_deltas)
        or _has_noncritical_regression(total_deltas)
    )
    if has_any_warn:
        return ("WARN", "Non-critical metrics degraded.")

    return ("PASS", "No critical regressions detected.")


def _has_critical_regression(deltas: list[dict[str, Any]] | None) -> bool:
    """Check if any critical metric has FAIL status."""
    if deltas is None:
        return False
    return any(d["status"] == "FAIL" and d["is_critical"] for d in deltas)


def _has_noncritical_regression(deltas: list[dict[str, Any]] | None) -> bool:
    """Check if any non-critical metric has WARN status."""
    if deltas is None:
        return False
    return any(d["status"] == "WARN" for d in deltas)


def generate_abc_markdown_summary(
    okp_deltas: list[dict[str, Any]] | None,
    pr_deltas: list[dict[str, Any]] | None,
    total_deltas: list[dict[str, Any]] | None,
    summary_a: dict[str, Any] | None,
    summary_b: dict[str, Any] | None,
    summary_c: dict[str, Any],
    verdict: str,
    explanation: str,
) -> str:
    """Generate a three-panel markdown regression report.

    Args:
        okp_deltas: A vs B deltas. None if Run A unavailable.
        pr_deltas: B vs C deltas. None if Run B unavailable.
        total_deltas: A vs C deltas. None if Run A unavailable.
        summary_a: Run A summary data. None if unavailable.
        summary_b: Run B summary data. None if unavailable.
        summary_c: Run C summary data.
        verdict: Gate verdict (PASS/WARN/FAIL).
        explanation: One-line explanation of the verdict.

    Returns:
        Markdown string with the full report.
    """
    lines = ["# Three-Run Regression Analysis", ""]

    lines.append("## Run Configuration")
    lines.append("| Run | Evaluations | Timestamp |")
    lines.append("|-----|------------:|-----------|")
    if summary_a is not None:
        lines.append(
            f"| A (main + old OKP) | {summary_a['total_evaluations']} "
            f"| {summary_a['timestamp']} |"
        )
    if summary_b is not None:
        lines.append(
            f"| B (main + new OKP) | {summary_b['total_evaluations']} "
            f"| {summary_b['timestamp']} |"
        )
    lines.append(
        f"| C (PR + new OKP) | {summary_c['total_evaluations']} "
        f"| {summary_c['timestamp']} |"
    )
    lines.append("")

    if okp_deltas is not None and summary_a is not None and summary_b is not None:
        lines.append("## OKP Data Impact (A vs B)")
        lines.append("_Did the new OKP data cause quality changes?_")
        lines.append("")
        lines.append(
            _format_comparison_table(okp_deltas, summary_a, summary_b, "Run A", "Run B")
        )

    if pr_deltas is not None and summary_b is not None:
        lines.append("## PR Impact (B vs C)")
        lines.append("_Did the PR cause quality changes, controlling for OKP data?_")
        lines.append("")
        lines.append(
            _format_comparison_table(pr_deltas, summary_b, summary_c, "Run B", "Run C")
        )

    if total_deltas is not None and summary_a is not None:
        lines.append("## Overall Impact (A vs C)")
        lines.append("_Total quality change from production baseline._")
        lines.append("")
        lines.append(
            _format_comparison_table(
                total_deltas, summary_a, summary_c, "Run A", "Run C"
            )
        )

    lines.append("## Gate Verdict")
    if verdict == "FAIL":
        lines.append(f"**FAIL** — {explanation}")
    elif verdict == "WARN":
        lines.append(f"**WARN** — {explanation}")
    else:
        lines.append(f"**PASS** — {explanation}")
    lines.append("")

    return "\n".join(lines)


def _format_comparison_table(
    deltas: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
    baseline_label: str,
    current_label: str,
) -> str:
    """Format a single comparison as a markdown table with result line."""
    lines = [
        f"| Metric | {baseline_label} | {current_label} | Delta | Status |",
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

    has_critical = any(d["status"] == "FAIL" and d["is_critical"] for d in deltas)
    has_warn = any(d["status"] == "WARN" for d in deltas)

    lines.append("")
    if has_critical:
        lines.append("**Result: REGRESSION** (critical metrics degraded)")
    elif has_warn:
        lines.append("**Result: WARNING** (non-critical metrics degraded)")
    else:
        lines.append("**Result: PASS** (no regressions detected)")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    args = parse_args()

    summary_a = None
    summary_b = None

    try:
        summary_c = find_and_load_summary(args.run_c)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"Error loading Run C: {err}", file=sys.stderr)
        return 1

    if args.run_a:
        try:
            summary_a = find_and_load_summary(args.run_a)
        except (FileNotFoundError, RuntimeError) as err:
            print(f"Warning: Run A unavailable: {err}", file=sys.stderr)

    if args.run_b:
        try:
            summary_b = find_and_load_summary(args.run_b)
        except (FileNotFoundError, RuntimeError) as err:
            print(f"Warning: Run B unavailable: {err}", file=sys.stderr)

    if summary_a is None and summary_b is None and args.baseline:
        try:
            summary_a = find_and_load_summary(args.baseline)
            print("Fallback: using baseline directory as Run A.")
        except (FileNotFoundError, RuntimeError) as err:
            print(f"Warning: Baseline unavailable: {err}", file=sys.stderr)

    cd = args.critical_delta
    wd = args.warn_delta

    okp_deltas = None
    pr_deltas = None
    total_deltas = None

    if summary_a is not None and summary_b is not None:
        okp_deltas = compute_metric_deltas(summary_a, summary_b, cd, wd)

    if summary_b is not None:
        pr_deltas = compute_metric_deltas(summary_b, summary_c, cd, wd)

    if summary_a is not None:
        total_deltas = compute_metric_deltas(summary_a, summary_c, cd, wd)

    verdict, explanation = determine_gate_verdict(okp_deltas, pr_deltas, total_deltas)

    print(f"\nGate Verdict: {verdict} — {explanation}")

    if args.output:
        markdown = generate_abc_markdown_summary(
            okp_deltas,
            pr_deltas,
            total_deltas,
            summary_a,
            summary_b,
            summary_c,
            verdict,
            explanation,
        )
        output_path = Path(args.output)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Markdown summary written to: {output_path}")

    if args.fail_on_pr_regression and verdict == "FAIL":
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

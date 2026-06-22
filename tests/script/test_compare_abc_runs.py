"""Tests for the three-run A/B/C regression comparison script."""

import sys
from pathlib import Path
from typing import Any

from script.regression.compare_abc_runs import (
    determine_gate_verdict,
    generate_abc_markdown_summary,
    main,
)
from script.regression.compare_against_baseline import compute_metric_deltas
from tests.script.conftest import make_summary, write_summary


def _deltas(
    baseline: dict[str, Any],
    current: dict[str, Any],
    critical_delta: float = 0.03,
    warn_delta: float = 0.03,
) -> list[dict[str, Any]]:
    """Shortcut to compute deltas between two summaries."""
    return compute_metric_deltas(baseline, current, critical_delta, warn_delta)


# ---------------------------------------------------------------------------
# Reusable test data builders
# ---------------------------------------------------------------------------

GOOD_SCORES: dict[str, tuple[float, float]] = {
    "ragas:faithfulness": (0.90, 100.0),
    "custom:answer_correctness": (0.85, 95.0),
    "custom:intent_eval": (0.95, 100.0),
}

SLIGHTLY_BETTER: dict[str, tuple[float, float]] = {
    "ragas:faithfulness": (0.92, 100.0),
    "custom:answer_correctness": (0.87, 95.0),
    "custom:intent_eval": (0.96, 100.0),
}

CRITICAL_REGRESSION: dict[str, tuple[float, float]] = {
    "ragas:faithfulness": (0.50, 60.0),
    "custom:answer_correctness": (0.85, 95.0),
    "custom:intent_eval": (0.95, 100.0),
}

NONCRITICAL_REGRESSION: dict[str, tuple[float, float]] = {
    "ragas:faithfulness": (0.90, 100.0),
    "custom:answer_correctness": (0.85, 95.0),
    "custom:intent_eval": (0.60, 70.0),
}


def _run_with_argv(argv: list[str]) -> int:
    """Run main() with patched sys.argv and return the exit code."""
    old_argv = sys.argv
    try:
        sys.argv = argv
        return main()
    finally:
        sys.argv = old_argv


def _setup_run_dirs(
    tmp_path: Path,
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Create temp directories with summary files for each named run.

    Args:
        tmp_path: Pytest temporary directory.
        summaries: Mapping of run name to summary data.

    Returns:
        Mapping of run name to its directory path.
    """
    dirs = {}
    for name, data in summaries.items():
        run_dir = tmp_path / name
        run_dir.mkdir()
        write_summary(run_dir, data, f"eval_{name}_summary.json")
        dirs[name] = run_dir
    return dirs


# ---------------------------------------------------------------------------
# Tests for determine_gate_verdict
# ---------------------------------------------------------------------------


class TestDetermineGateVerdict:
    """Tests for the gate verdict logic."""

    def test_all_pass(self) -> None:
        """No regressions in any pair should give PASS."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(GOOD_SCORES)
        c = make_summary(SLIGHTLY_BETTER)

        verdict, _ = determine_gate_verdict(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
        )
        assert verdict == "PASS"

    def test_pr_regression_is_fail(self) -> None:
        """Critical regression in B vs C should FAIL — the PR caused it."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(GOOD_SCORES)
        c = make_summary(CRITICAL_REGRESSION)

        verdict, explanation = determine_gate_verdict(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
        )
        assert verdict == "FAIL"
        assert "PR" in explanation

    def test_okp_regression_is_warn(self) -> None:
        """Critical regression in A vs B (OKP change) but B vs C is fine should WARN."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(CRITICAL_REGRESSION)
        c = make_summary(CRITICAL_REGRESSION)

        okp_deltas = _deltas(a, b)
        pr_deltas = _deltas(b, c)

        verdict, explanation = determine_gate_verdict(
            okp_deltas=okp_deltas,
            pr_deltas=pr_deltas,
            total_deltas=_deltas(a, c),
        )
        assert verdict == "WARN"
        assert "OKP" in explanation

    def test_both_regress_is_fail(self) -> None:
        """If both OKP and PR cause regressions, FAIL (PR is still at fault)."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(CRITICAL_REGRESSION)
        c_scores = {
            "ragas:faithfulness": (0.30, 40.0),
            "custom:answer_correctness": (0.85, 95.0),
            "custom:intent_eval": (0.95, 100.0),
        }
        c = make_summary(c_scores)

        verdict, _ = determine_gate_verdict(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
        )
        assert verdict == "FAIL"

    def test_noncritical_regression_is_warn(self) -> None:
        """Non-critical regressions should produce WARN, not FAIL."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(GOOD_SCORES)
        c = make_summary(NONCRITICAL_REGRESSION)

        verdict, _ = determine_gate_verdict(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
        )
        assert verdict == "WARN"

    def test_missing_run_a(self) -> None:
        """With no Run A, only B vs C comparison is available."""
        b = make_summary(GOOD_SCORES)
        c = make_summary(SLIGHTLY_BETTER)

        verdict, _ = determine_gate_verdict(
            okp_deltas=None,
            pr_deltas=_deltas(b, c),
            total_deltas=None,
        )
        assert verdict == "PASS"

    def test_missing_run_a_with_pr_regression(self) -> None:
        """With no Run A, B vs C regression should still FAIL."""
        b = make_summary(GOOD_SCORES)
        c = make_summary(CRITICAL_REGRESSION)

        verdict, _ = determine_gate_verdict(
            okp_deltas=None,
            pr_deltas=_deltas(b, c),
            total_deltas=None,
        )
        assert verdict == "FAIL"

    def test_missing_run_b_with_total_regression(self) -> None:
        """With no Run B, A vs C regression should FAIL (can't attribute)."""
        a = make_summary(GOOD_SCORES)
        c = make_summary(CRITICAL_REGRESSION)

        verdict, _ = determine_gate_verdict(
            okp_deltas=None,
            pr_deltas=None,
            total_deltas=_deltas(a, c),
        )
        assert verdict == "FAIL"

    def test_all_none_is_pass(self) -> None:
        """With no comparisons available, default to PASS."""
        verdict, _ = determine_gate_verdict(
            okp_deltas=None,
            pr_deltas=None,
            total_deltas=None,
        )
        assert verdict == "PASS"


# ---------------------------------------------------------------------------
# Tests for generate_abc_markdown_summary
# ---------------------------------------------------------------------------


class TestGenerateAbcMarkdownSummary:
    """Tests for the three-panel markdown report generation."""

    def test_full_report_structure(self) -> None:
        """Full three-run report should have all sections."""
        a = make_summary(GOOD_SCORES, timestamp="2026-01-01T00:00:00")
        b = make_summary(GOOD_SCORES, timestamp="2026-01-02T00:00:00")
        c = make_summary(SLIGHTLY_BETTER, timestamp="2026-01-03T00:00:00")

        md = generate_abc_markdown_summary(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
            summary_a=a,
            summary_b=b,
            summary_c=c,
            verdict="PASS",
            explanation="No critical regressions detected.",
        )

        assert "# Three-Run Regression Analysis" in md
        assert "## Run Configuration" in md
        assert "## OKP Data Impact (A vs B)" in md
        assert "## PR Impact (B vs C)" in md
        assert "## Overall Impact (A vs C)" in md
        assert "## Gate Verdict" in md
        assert "**PASS**" in md

    def test_missing_run_a_omits_sections(self) -> None:
        """Without Run A, OKP and Overall sections should be omitted."""
        b = make_summary(GOOD_SCORES)
        c = make_summary(SLIGHTLY_BETTER)

        md = generate_abc_markdown_summary(
            okp_deltas=None,
            pr_deltas=_deltas(b, c),
            total_deltas=None,
            summary_a=None,
            summary_b=b,
            summary_c=c,
            verdict="PASS",
            explanation="No critical regressions detected.",
        )

        assert "## OKP Data Impact" not in md
        assert "## Overall Impact" not in md
        assert "## PR Impact (B vs C)" in md
        assert "## Gate Verdict" in md

    def test_missing_run_b_omits_pr_section(self) -> None:
        """Without Run B, PR Impact section should be omitted."""
        a = make_summary(GOOD_SCORES)
        c = make_summary(SLIGHTLY_BETTER)

        md = generate_abc_markdown_summary(
            okp_deltas=None,
            pr_deltas=None,
            total_deltas=_deltas(a, c),
            summary_a=a,
            summary_b=None,
            summary_c=c,
            verdict="PASS",
            explanation="No critical regressions detected.",
        )

        assert "## PR Impact" not in md
        assert "## Overall Impact (A vs C)" in md

    def test_fail_verdict_shown(self) -> None:
        """FAIL verdict should appear in the Gate Verdict section."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(GOOD_SCORES)
        c = make_summary(CRITICAL_REGRESSION)

        md = generate_abc_markdown_summary(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
            summary_a=a,
            summary_b=b,
            summary_c=c,
            verdict="FAIL",
            explanation="PR caused critical regression (B vs C).",
        )

        assert "**FAIL**" in md
        assert "PR caused critical regression" in md

    def test_warn_verdict_shown(self) -> None:
        """WARN verdict should appear in the Gate Verdict section."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(CRITICAL_REGRESSION)
        c = make_summary(CRITICAL_REGRESSION)

        md = generate_abc_markdown_summary(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
            summary_a=a,
            summary_b=b,
            summary_c=c,
            verdict="WARN",
            explanation="OKP data caused regression, not the PR (A vs B).",
        )

        assert "**WARN**" in md

    def test_run_config_table_has_timestamps(self) -> None:
        """Run Configuration table should show timestamps for each run."""
        a = make_summary(GOOD_SCORES, timestamp="2026-06-01T10:00:00")
        b = make_summary(GOOD_SCORES, timestamp="2026-06-02T11:00:00")
        c = make_summary(GOOD_SCORES, timestamp="2026-06-03T12:00:00")

        md = generate_abc_markdown_summary(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
            summary_a=a,
            summary_b=b,
            summary_c=c,
            verdict="PASS",
            explanation="No critical regressions detected.",
        )

        assert "2026-06-01" in md
        assert "2026-06-02" in md
        assert "2026-06-03" in md

    def test_comparison_tables_contain_metrics(self) -> None:
        """Each comparison table should list the metric names."""
        a = make_summary(GOOD_SCORES)
        b = make_summary(GOOD_SCORES)
        c = make_summary(GOOD_SCORES)

        md = generate_abc_markdown_summary(
            okp_deltas=_deltas(a, b),
            pr_deltas=_deltas(b, c),
            total_deltas=_deltas(a, c),
            summary_a=a,
            summary_b=b,
            summary_c=c,
            verdict="PASS",
            explanation="No critical regressions detected.",
        )

        assert "ragas:faithfulness" in md
        assert "custom:answer_correctness" in md
        assert "custom:intent_eval" in md


# ---------------------------------------------------------------------------
# Tests for main() via CLI simulation
# ---------------------------------------------------------------------------


class TestMainIntegration:
    """Integration tests running the full script with temp directories."""

    def test_all_pass_returns_zero(self, tmp_path: Path) -> None:
        """No regressions should exit 0."""
        dirs = _setup_run_dirs(
            tmp_path,
            {
                "run_a": make_summary(GOOD_SCORES),
                "run_b": make_summary(GOOD_SCORES),
                "run_c": make_summary(SLIGHTLY_BETTER),
            },
        )
        rc = _run_with_argv(
            [
                "compare_abc_runs.py",
                "--run-a",
                str(dirs["run_a"]),
                "--run-b",
                str(dirs["run_b"]),
                "--run-c",
                str(dirs["run_c"]),
                "--fail-on-pr-regression",
            ]
        )
        assert rc == 0

    def test_pr_regression_returns_one(self, tmp_path: Path) -> None:
        """PR regression with --fail-on-pr-regression should exit 1."""
        dirs = _setup_run_dirs(
            tmp_path,
            {
                "run_a": make_summary(GOOD_SCORES),
                "run_b": make_summary(GOOD_SCORES),
                "run_c": make_summary(CRITICAL_REGRESSION),
            },
        )
        rc = _run_with_argv(
            [
                "compare_abc_runs.py",
                "--run-a",
                str(dirs["run_a"]),
                "--run-b",
                str(dirs["run_b"]),
                "--run-c",
                str(dirs["run_c"]),
                "--fail-on-pr-regression",
            ]
        )
        assert rc == 1

    def test_okp_regression_returns_zero(self, tmp_path: Path) -> None:
        """OKP regression (not PR's fault) should exit 0 even with gate flag."""
        dirs = _setup_run_dirs(
            tmp_path,
            {
                "run_a": make_summary(GOOD_SCORES),
                "run_b": make_summary(CRITICAL_REGRESSION),
                "run_c": make_summary(CRITICAL_REGRESSION),
            },
        )
        rc = _run_with_argv(
            [
                "compare_abc_runs.py",
                "--run-a",
                str(dirs["run_a"]),
                "--run-b",
                str(dirs["run_b"]),
                "--run-c",
                str(dirs["run_c"]),
                "--fail-on-pr-regression",
            ]
        )
        assert rc == 0

    def test_only_run_c_returns_zero(self, tmp_path: Path) -> None:
        """With only Run C, no comparisons possible — should exit 0."""
        dirs = _setup_run_dirs(
            tmp_path,
            {
                "run_c": make_summary(GOOD_SCORES),
            },
        )
        rc = _run_with_argv(
            [
                "compare_abc_runs.py",
                "--run-c",
                str(dirs["run_c"]),
            ]
        )
        assert rc == 0

    def test_writes_markdown_output(self, tmp_path: Path) -> None:
        """--output should create a markdown file."""
        output_file = tmp_path / "report.md"
        dirs = _setup_run_dirs(
            tmp_path,
            {
                "run_a": make_summary(GOOD_SCORES),
                "run_b": make_summary(GOOD_SCORES),
                "run_c": make_summary(SLIGHTLY_BETTER),
            },
        )
        rc = _run_with_argv(
            [
                "compare_abc_runs.py",
                "--run-a",
                str(dirs["run_a"]),
                "--run-b",
                str(dirs["run_b"]),
                "--run-c",
                str(dirs["run_c"]),
                "--output",
                str(output_file),
            ]
        )
        assert rc == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "# Three-Run Regression Analysis" in content

    def test_fallback_baseline(self, tmp_path: Path) -> None:
        """--baseline should be used as Run A when --run-a is not provided."""
        dirs = _setup_run_dirs(
            tmp_path,
            {
                "baseline": make_summary(GOOD_SCORES),
                "run_c": make_summary(SLIGHTLY_BETTER),
            },
        )
        rc = _run_with_argv(
            [
                "compare_abc_runs.py",
                "--run-c",
                str(dirs["run_c"]),
                "--baseline",
                str(dirs["baseline"]),
                "--fail-on-pr-regression",
            ]
        )
        assert rc == 0

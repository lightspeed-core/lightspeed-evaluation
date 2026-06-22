"""Tests for the baseline comparison regression gating script."""

from pathlib import Path
from typing import Any

import pytest

from script.regression.compare_against_baseline import (
    compute_metric_deltas,
    find_and_load_summary,
    generate_markdown_summary,
)
from tests.script.conftest import make_summary, write_summary

# ---------------------------------------------------------------------------
# Tests for find_and_load_summary
# ---------------------------------------------------------------------------


class TestFindAndLoadSummary:
    """Tests for the find_and_load_summary function."""

    def test_loads_single_summary(self, tmp_path: Path) -> None:
        """Happy path: one summary file in the directory."""
        expected = make_summary({"ragas:faithfulness": (0.9, 100.0)})
        write_summary(tmp_path, expected)

        result = find_and_load_summary(str(tmp_path))

        assert result == expected

    def test_raises_on_no_summary(self, tmp_path: Path) -> None:
        """No *_summary.json file should raise FileNotFoundError."""
        (tmp_path / "other_file.txt").write_text("not a summary")

        with pytest.raises(FileNotFoundError, match="No \\*_summary.json"):
            find_and_load_summary(str(tmp_path))

    def test_raises_on_multiple_summaries(self, tmp_path: Path) -> None:
        """Multiple summary files should raise RuntimeError."""
        data = make_summary({"ragas:faithfulness": (0.9, 100.0)})
        write_summary(tmp_path, data, "run_a_summary.json")
        write_summary(tmp_path, data, "run_b_summary.json")

        with pytest.raises(RuntimeError, match="Multiple summary files"):
            find_and_load_summary(str(tmp_path))

    def test_raises_on_missing_directory(self) -> None:
        """Non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            find_and_load_summary("/tmp/nonexistent_dir_abc123")


# ---------------------------------------------------------------------------
# Tests for compute_metric_deltas
# ---------------------------------------------------------------------------


class TestComputeMetricDeltas:
    """Tests for the compute_metric_deltas function."""

    def test_no_regression(self) -> None:
        """Scores that stay the same or improve should all be PASS."""
        baseline = make_summary(
            {
                "ragas:faithfulness": (0.85, 100.0),
                "custom:answer_correctness": (0.90, 95.0),
            }
        )
        current = make_summary(
            {
                "ragas:faithfulness": (0.88, 100.0),
                "custom:answer_correctness": (0.90, 95.0),
            }
        )

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        assert all(d["status"] == "PASS" for d in deltas)

    def test_critical_regression_is_fail(self) -> None:
        """A critical metric dropping beyond the threshold should be FAIL."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({"ragas:faithfulness": (0.80, 100.0)})

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        faith = [d for d in deltas if d["metric"] == "ragas:faithfulness"][0]
        assert faith["status"] == "FAIL"
        assert faith["is_critical"] is True
        assert faith["score_delta"] == pytest.approx(-0.10, abs=1e-6)

    def test_noncritical_regression_is_warn(self) -> None:
        """A non-critical metric dropping beyond threshold should be WARN, not FAIL."""
        baseline = make_summary({"custom:intent_eval": (0.95, 100.0)})
        current = make_summary({"custom:intent_eval": (0.80, 100.0)})

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        intent = [d for d in deltas if d["metric"] == "custom:intent_eval"][0]
        assert intent["status"] == "WARN"
        assert intent["is_critical"] is False

    def test_drop_within_threshold_is_pass(self) -> None:
        """A small drop within the allowed delta should still be PASS."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({"ragas:faithfulness": (0.88, 100.0)})

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        faith = [d for d in deltas if d["metric"] == "ragas:faithfulness"][0]
        assert faith["status"] == "PASS"

    def test_mixed_results(self) -> None:
        """Multiple metrics with different outcomes."""
        baseline = make_summary(
            {
                "ragas:faithfulness": (0.90, 100.0),
                "custom:answer_correctness": (0.85, 90.0),
                "custom:intent_eval": (0.95, 100.0),
                "ragas:response_relevancy": (0.80, 80.0),
            }
        )
        current = make_summary(
            {
                "ragas:faithfulness": (0.50, 80.0),
                "custom:answer_correctness": (0.86, 92.0),
                "custom:intent_eval": (0.70, 80.0),
                "ragas:response_relevancy": (0.82, 85.0),
            }
        )

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )
        by_metric = {d["metric"]: d["status"] for d in deltas}

        assert by_metric["ragas:faithfulness"] == "FAIL"
        assert by_metric["custom:answer_correctness"] == "PASS"
        assert by_metric["custom:intent_eval"] == "WARN"
        assert by_metric["ragas:response_relevancy"] == "PASS"

    def test_metric_only_in_baseline(self) -> None:
        """A metric present in baseline but missing from current gets None deltas."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({})

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        faith = [d for d in deltas if d["metric"] == "ragas:faithfulness"][0]
        assert faith["current_mean"] is None
        assert faith["score_delta"] is None
        assert faith["status"] == "PASS"

    def test_metric_only_in_current(self) -> None:
        """A new metric in current but not in baseline gets None deltas and PASS."""
        baseline = make_summary({})
        current = make_summary({"ragas:faithfulness": (0.90, 100.0)})

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        faith = [d for d in deltas if d["metric"] == "ragas:faithfulness"][0]
        assert faith["baseline_mean"] is None
        assert faith["score_delta"] is None
        assert faith["status"] == "PASS"

    def test_custom_thresholds(self) -> None:
        """A drop of 0.05 should FAIL with critical_delta=0.01 but PASS with 0.10."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({"ragas:faithfulness": (0.85, 100.0)})

        strict = compute_metric_deltas(
            baseline, current, critical_delta=0.01, warn_delta=0.01
        )
        assert strict[0]["status"] == "FAIL"

        lenient = compute_metric_deltas(
            baseline, current, critical_delta=0.10, warn_delta=0.10
        )
        assert lenient[0]["status"] == "PASS"

    def test_just_within_threshold_is_pass(self) -> None:
        """A drop smaller than the threshold should be PASS."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        # Drop of 0.02, threshold is 0.03 — should pass
        current = make_summary({"ragas:faithfulness": (0.88, 100.0)})

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        assert deltas[0]["status"] == "PASS"

    def test_just_beyond_threshold_is_fail(self) -> None:
        """A drop larger than the threshold should be FAIL for critical metrics."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        # Drop of 0.04, threshold is 0.03 — should fail
        current = make_summary({"ragas:faithfulness": (0.86, 100.0)})

        deltas = compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

        assert deltas[0]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Tests for generate_markdown_summary
# ---------------------------------------------------------------------------


class TestGenerateMarkdownSummary:
    """Tests for the generate_markdown_summary function."""

    def _make_deltas(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Shortcut to compute deltas from two summaries."""
        return compute_metric_deltas(
            baseline, current, critical_delta=0.03, warn_delta=0.03
        )

    def test_contains_header_and_table(self) -> None:
        """Output should have the title, metadata, and a markdown table."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({"ragas:faithfulness": (0.91, 100.0)})
        deltas = self._make_deltas(baseline, current)

        md = generate_markdown_summary(deltas, baseline, current)

        assert "# Regression Comparison Summary" in md
        assert "| Metric |" in md
        assert "ragas:faithfulness" in md

    def test_shows_regression_on_critical_fail(self) -> None:
        """Result line should say REGRESSION when a critical metric fails."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({"ragas:faithfulness": (0.50, 80.0)})
        deltas = self._make_deltas(baseline, current)

        md = generate_markdown_summary(deltas, baseline, current)

        assert "**Result: REGRESSION**" in md

    def test_shows_warning_on_noncritical_drop(self) -> None:
        """Result line should say WARNING when only non-critical metrics drop."""
        baseline = make_summary({"custom:intent_eval": (0.95, 100.0)})
        current = make_summary({"custom:intent_eval": (0.70, 80.0)})
        deltas = self._make_deltas(baseline, current)

        md = generate_markdown_summary(deltas, baseline, current)

        assert "**Result: WARNING**" in md

    def test_shows_pass_when_no_regressions(self) -> None:
        """Result line should say PASS when nothing regressed."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({"ragas:faithfulness": (0.92, 100.0)})
        deltas = self._make_deltas(baseline, current)

        md = generate_markdown_summary(deltas, baseline, current)

        assert "**Result: PASS**" in md

    def test_includes_evaluation_counts(self) -> None:
        """Summary should show evaluation counts from both runs."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary(
            {"ragas:faithfulness": (0.90, 100.0)},
            timestamp="2026-02-01T00:00:00",
        )
        deltas = self._make_deltas(baseline, current)

        md = generate_markdown_summary(deltas, baseline, current)

        assert str(baseline["total_evaluations"]) in md
        assert str(current["total_evaluations"]) in md
        assert "2026-01-01" in md
        assert "2026-02-01" in md

    def test_fail_status_is_bold(self) -> None:
        """FAIL entries in the table should be bold for visibility."""
        baseline = make_summary({"ragas:faithfulness": (0.90, 100.0)})
        current = make_summary({"ragas:faithfulness": (0.50, 80.0)})
        deltas = self._make_deltas(baseline, current)

        md = generate_markdown_summary(deltas, baseline, current)

        assert "**FAIL**" in md

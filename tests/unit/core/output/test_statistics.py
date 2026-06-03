"""Unit tests for core statistics module."""

import pandas as pd
import pytest

from lightspeed_evaluation.core.models.data import (
    EvaluationResult,
)
from lightspeed_evaluation.core.models.statistics import OverallStats
from lightspeed_evaluation.core.output.statistics import (
    bootstrap_intervals,
    compute_overall_stats,
    compute_score_statistics,
)


class TestBootstrapIntervals:
    """Tests for bootstrap_intervals function."""

    def test_bootstrap_intervals_basic(self) -> None:
        """Test basic bootstrap interval calculation."""
        series = pd.Series([0.8, 0.85, 0.9, 0.75, 0.88])

        low, mean, high = bootstrap_intervals(
            series, confidence=95, bootstrap_steps=100
        )

        assert low <= mean <= high
        assert 0 <= low <= 1
        assert 0 <= high <= 1

    def test_bootstrap_intervals_invalid_confidence(self) -> None:
        """Test bootstrap with invalid confidence value."""
        series = pd.Series([0.8, 0.85, 0.9])

        with pytest.raises(ValueError, match="Invalid confidence"):
            bootstrap_intervals(series, confidence=150)

    def test_bootstrap_intervals_negative_confidence(self) -> None:
        """Test bootstrap with negative confidence value."""
        series = pd.Series([0.8, 0.85, 0.9])

        with pytest.raises(ValueError, match="Invalid confidence"):
            bootstrap_intervals(series, confidence=-10)

    def test_bootstrap_intervals_custom_confidence(self) -> None:
        """Test bootstrap with custom confidence level."""
        series = pd.Series([0.8, 0.85, 0.9, 0.75, 0.88])

        low, mean, high = bootstrap_intervals(
            series, confidence=90, bootstrap_steps=100
        )

        assert low <= mean <= high

    def test_bootstrap_intervals_custom_steps(self) -> None:
        """Test bootstrap with custom bootstrap steps."""
        series = pd.Series([0.8, 0.85, 0.9])

        low, mean, high = bootstrap_intervals(series, confidence=95, bootstrap_steps=50)

        assert low <= mean <= high

    def test_bootstrap_intervals_valid_confidence(self) -> None:
        """Test bootstrap_intervals with valid confidence levels."""
        data = pd.Series([0.8, 0.9, 0.7, 0.85, 0.75])

        # Test default 95% confidence
        low, mean, high = bootstrap_intervals(data)
        assert low <= mean <= high
        assert isinstance(low, float)
        assert isinstance(mean, float)
        assert isinstance(high, float)

        # Test 90% confidence (should be narrower)
        low_90, mean_90, high_90 = bootstrap_intervals(data, confidence=90)
        assert low_90 <= mean_90 <= high_90
        ci_95_width = high - low
        ci_90_width = high_90 - low_90
        assert ci_90_width < ci_95_width

    def test_bootstrap_intervals_edge_cases(self) -> None:
        """Test bootstrap_intervals with edge cases."""
        # Test with single value
        single_value = pd.Series([0.5])
        low, mean, high = bootstrap_intervals(single_value)
        assert low == mean == high == 0.5

        # Test with all same values
        same_values = pd.Series([0.8, 0.8, 0.8, 0.8, 0.8])
        low, mean, high = bootstrap_intervals(same_values)
        assert abs(low - 0.8) < 0.001
        assert abs(mean - 0.8) < 0.001
        assert abs(high - 0.8) < 0.001

    def test_bootstrap_intervals_confidence_levels(self) -> None:
        """Test bootstrap_intervals with different confidence levels."""
        data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Test 0% confidence (should be very narrow)
        low_0, mean_0, high_0 = bootstrap_intervals(data, confidence=0)
        assert low_0 <= mean_0 <= high_0

        # Test 100% confidence (should be very wide)
        low_100, mean_100, high_100 = bootstrap_intervals(data, confidence=100)
        assert low_100 <= mean_100 <= high_100


class TestCalculateScoreStatistics:
    """Tests for compute_score_statistics function."""

    def test_score_statistics_multiple_scores(self) -> None:
        """Test score statistics with multiple scores includes confidence interval."""
        scores = [0.8, 0.85, 0.9, 0.75, 0.88]
        result = compute_score_statistics(scores, compute_ci=True)

        assert result.count == 5
        assert result.mean == pytest.approx(0.836, rel=1e-3)
        assert result.median == 0.85
        assert result.min_score == 0.75
        assert result.max_score == 0.9
        assert result.std > 0

        # Confidence interval should be calculated
        ci = result.confidence_interval
        assert ci is not None
        assert ci.confidence_level == 95
        assert ci.low < ci.mean < ci.high

    def test_score_statistics_two_scores(self) -> None:
        """Test score statistics with exactly 2 scores includes CI."""
        scores = [0.7, 0.9]
        result = compute_score_statistics(scores, compute_ci=True)

        assert result.count == 2
        assert result.mean == 0.8
        assert result.confidence_interval is not None

    def test_score_statistics_single_score_no_ci(self) -> None:
        """Test score statistics with single score has no confidence interval."""
        scores = [0.8]
        result = compute_score_statistics(scores)

        assert result.count == 1
        assert result.mean == 0.8
        assert result.std == 0.0  # No std for single value
        assert result.confidence_interval is None

    def test_score_statistics_empty_scores(self) -> None:
        """Test score statistics with empty list returns zeros and no CI."""
        result = compute_score_statistics([])

        assert result.count == 0
        assert result.mean == 0.0
        assert result.median == 0.0
        assert result.std == 0.0
        assert result.min_score == 0.0
        assert result.max_score == 0.0
        assert result.confidence_interval is None


class TestCalculateBasicStats:
    """Tests for compute_overall_stats function."""

    def test_basic_stats_with_results(
        self, sample_results_statistics: list[EvaluationResult]
    ) -> None:
        """Test basic stats calculation with results."""
        stats = compute_overall_stats(sample_results_statistics)

        assert stats.total == 4
        assert stats.passed == 2
        assert stats.failed == 1
        assert stats.error == 1
        assert stats.pass_rate == 50.0
        assert stats.fail_rate == 25.0
        assert stats.error_rate == 25.0

    def test_basic_stats_empty_results(self) -> None:
        """Test basic stats with empty results."""
        stats = compute_overall_stats([])

        assert stats.total == 0
        assert stats.passed == 0
        assert stats.failed == 0
        assert stats.error == 0
        assert stats.pass_rate == 0.0
        assert stats.fail_rate == 0.0
        assert stats.error_rate == 0.0

    def test_basic_stats_all_pass(self) -> None:
        """Test basic stats with all passing results."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=f"turn{i}",
                metric_identifier="metric1",
                score=0.9,
                result="PASS",
                threshold=0.7,
            )
            for i in range(5)
        ]

        stats = compute_overall_stats(results)

        assert stats.total == 5
        assert stats.passed == 5
        assert stats.pass_rate == 100.0
        assert stats.fail_rate == 0.0

    def test_basic_stats_all_fail(self) -> None:
        """Test basic stats with all failing results."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=f"turn{i}",
                metric_identifier="metric1",
                score=0.5,
                result="FAIL",
                threshold=0.7,
            )
            for i in range(3)
        ]

        stats = compute_overall_stats(results)

        assert stats.total == 3
        assert stats.failed == 3
        assert stats.fail_rate == 100.0
        assert stats.pass_rate == 0.0

    def test_basic_stats_all_error(self) -> None:
        """Test basic stats with all error results."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=f"turn{i}",
                metric_identifier="metric1",
                score=None,
                result="ERROR",
                threshold=0.7,
            )
            for i in range(2)
        ]

        stats = compute_overall_stats(results)

        assert stats.total == 2
        assert stats.error == 2
        assert stats.error_rate == 100.0

    def test_compute_overall_stats_mixed_results(self) -> None:
        """Test compute_overall_stats with mixed results."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good",
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn2",
                metric_identifier="test:metric",
                result="FAIL",
                score=0.3,
                threshold=0.7,
                reason="Poor",
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn3",
                metric_identifier="test:metric",
                result="ERROR",
                score=0.0,
                threshold=0.7,
                reason="API timeout",
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn4",
                metric_identifier="test:metric",
                result="PASS",
                score=0.9,
                threshold=0.7,
                reason="Excellent",
            ),
        ]

        stats = compute_overall_stats(results)

        expected = OverallStats(
            total=4,
            passed=2,
            failed=1,
            error=1,
            skipped=0,
            pass_rate=50.0,
            fail_rate=25.0,
            error_rate=25.0,
            skipped_rate=0.0,
            total_judge_llm_input_tokens=0,
            total_judge_llm_output_tokens=0,
            total_judge_llm_tokens=0,
            total_embedding_tokens=0,
        )
        assert stats == expected

    def test_compute_overall_stats_single_result(self) -> None:
        """Test compute_overall_stats with single result."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="FAIL",
                score=0.2,
                threshold=0.7,
                reason="Poor performance",
            )
        ]

        stats = compute_overall_stats(results)

        expected = OverallStats(
            total=1,
            passed=0,
            failed=1,
            error=0,
            skipped=0,
            pass_rate=0.0,
            fail_rate=100.0,
            error_rate=0.0,
            skipped_rate=0.0,
            total_judge_llm_input_tokens=0,
            total_judge_llm_output_tokens=0,
            total_judge_llm_tokens=0,
            total_embedding_tokens=0,
        )
        assert stats == expected

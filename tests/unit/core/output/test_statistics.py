"""Unit tests for output statistics module."""

import pytest
import pandas as pd

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.output.statistics import (
    bootstrap_intervals,
    calculate_basic_stats,
    calculate_detailed_stats,
)


@pytest.fixture
def sample_results():
    """Create sample evaluation results."""
    return [
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="metric1",
            score=0.9,
            result="PASS",
            threshold=0.7,
            reason="Good",
        ),
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn2",
            metric_identifier="metric1",
            score=0.5,
            result="FAIL",
            threshold=0.7,
            reason="Low score",
        ),
        EvaluationResult(
            conversation_group_id="conv2",
            turn_id="turn1",
            metric_identifier="metric2",
            score=0.8,
            result="PASS",
            threshold=0.7,
            reason="Good",
        ),
        EvaluationResult(
            conversation_group_id="conv2",
            turn_id="turn2",
            metric_identifier="metric2",
            score=None,
            result="ERROR",
            threshold=0.7,
            reason="Failed",
        ),
    ]


class TestBootstrapIntervals:
    """Tests for bootstrap_intervals function."""

    def test_bootstrap_intervals_basic(self):
        """Test basic bootstrap interval calculation."""
        series = pd.Series([0.8, 0.85, 0.9, 0.75, 0.88])

        low, mean, high = bootstrap_intervals(
            series, confidence=95, bootstrap_steps=100
        )

        assert low <= mean <= high
        assert 0 <= low <= 1
        assert 0 <= high <= 1

    def test_bootstrap_intervals_invalid_confidence(self):
        """Test bootstrap with invalid confidence value."""
        series = pd.Series([0.8, 0.85, 0.9])

        with pytest.raises(ValueError, match="Invalid confidence"):
            bootstrap_intervals(series, confidence=150)

    def test_bootstrap_intervals_negative_confidence(self):
        """Test bootstrap with negative confidence value."""
        series = pd.Series([0.8, 0.85, 0.9])

        with pytest.raises(ValueError, match="Invalid confidence"):
            bootstrap_intervals(series, confidence=-10)

    def test_bootstrap_intervals_custom_confidence(self):
        """Test bootstrap with custom confidence level."""
        series = pd.Series([0.8, 0.85, 0.9, 0.75, 0.88])

        low, mean, high = bootstrap_intervals(
            series, confidence=90, bootstrap_steps=100
        )

        assert low <= mean <= high

    def test_bootstrap_intervals_custom_steps(self):
        """Test bootstrap with custom bootstrap steps."""
        series = pd.Series([0.8, 0.85, 0.9])

        low, mean, high = bootstrap_intervals(series, confidence=95, bootstrap_steps=50)

        assert low <= mean <= high


class TestCalculateBasicStats:
    """Tests for calculate_basic_stats function."""

    def test_basic_stats_with_results(self, sample_results):
        """Test basic stats calculation with results."""
        stats = calculate_basic_stats(sample_results)

        assert stats["TOTAL"] == 4
        assert stats["PASS"] == 2
        assert stats["FAIL"] == 1
        assert stats["ERROR"] == 1
        assert stats["pass_rate"] == 50.0
        assert stats["fail_rate"] == 25.0
        assert stats["error_rate"] == 25.0

    def test_basic_stats_empty_results(self):
        """Test basic stats with empty results."""
        stats = calculate_basic_stats([])

        assert stats["TOTAL"] == 0
        assert stats["PASS"] == 0
        assert stats["FAIL"] == 0
        assert stats["ERROR"] == 0
        assert stats["pass_rate"] == 0.0
        assert stats["fail_rate"] == 0.0
        assert stats["error_rate"] == 0.0

    def test_basic_stats_all_pass(self):
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

        stats = calculate_basic_stats(results)

        assert stats["TOTAL"] == 5
        assert stats["PASS"] == 5
        assert stats["pass_rate"] == 100.0
        assert stats["fail_rate"] == 0.0

    def test_basic_stats_all_fail(self):
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

        stats = calculate_basic_stats(results)

        assert stats["TOTAL"] == 3
        assert stats["FAIL"] == 3
        assert stats["fail_rate"] == 100.0
        assert stats["pass_rate"] == 0.0

    def test_basic_stats_all_error(self):
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

        stats = calculate_basic_stats(results)

        assert stats["TOTAL"] == 2
        assert stats["ERROR"] == 2
        assert stats["error_rate"] == 100.0


class TestCalculateDetailedStats:
    """Tests for calculate_detailed_stats function."""

    def test_detailed_stats_with_results(self, sample_results):
        """Test detailed stats calculation."""
        stats = calculate_detailed_stats(sample_results)

        assert "by_metric" in stats
        assert "by_conversation" in stats
        assert "metric1" in stats["by_metric"]
        assert "metric2" in stats["by_metric"]
        assert "conv1" in stats["by_conversation"]
        assert "conv2" in stats["by_conversation"]

    def test_detailed_stats_empty_results(self):
        """Test detailed stats with empty results."""
        stats = calculate_detailed_stats([])

        assert stats["by_metric"] == {}
        assert stats["by_conversation"] == {}

    def test_detailed_stats_metric_breakdown(self, sample_results):
        """Test metric breakdown in detailed stats."""
        stats = calculate_detailed_stats(sample_results)

        metric1_stats = stats["by_metric"]["metric1"]
        assert metric1_stats["pass"] == 1
        assert metric1_stats["fail"] == 1

        metric2_stats = stats["by_metric"]["metric2"]
        assert metric2_stats["pass"] == 1
        assert metric2_stats["error"] == 1

    def test_detailed_stats_conversation_breakdown(self, sample_results):
        """Test conversation breakdown in detailed stats."""
        stats = calculate_detailed_stats(sample_results)

        conv1_stats = stats["by_conversation"]["conv1"]
        assert conv1_stats["pass"] == 1
        assert conv1_stats["fail"] == 1

        conv2_stats = stats["by_conversation"]["conv2"]
        assert conv2_stats["pass"] == 1
        assert conv2_stats["error"] == 1

    def test_detailed_stats_includes_rates(self, sample_results):
        """Test that detailed stats include percentage rates."""
        stats = calculate_detailed_stats(sample_results)

        metric1_stats = stats["by_metric"]["metric1"]
        assert "pass_rate" in metric1_stats
        assert "fail_rate" in metric1_stats
        assert metric1_stats["pass_rate"] == 50.0
        assert metric1_stats["fail_rate"] == 50.0

    def test_detailed_stats_single_metric(self):
        """Test detailed stats with single metric."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="single_metric",
                score=0.9,
                result="PASS",
                threshold=0.7,
            )
        ]

        stats = calculate_detailed_stats(results)

        assert len(stats["by_metric"]) == 1
        assert "single_metric" in stats["by_metric"]

"""Unit tests for output statistics module."""

import pytest
import pandas as pd

from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult, TurnData
from lightspeed_evaluation.core.output.statistics import (
    _calculate_score_statistics,
    bootstrap_intervals,
    calculate_api_token_usage,
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

    def test_bootstrap_intervals_valid_confidence(self):
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

    def test_bootstrap_intervals_edge_cases(self):
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

    def test_bootstrap_intervals_confidence_levels(self):
        """Test bootstrap_intervals with different confidence levels."""
        data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Test 0% confidence (should be very narrow)
        low_0, mean_0, high_0 = bootstrap_intervals(data, confidence=0)
        assert low_0 <= mean_0 <= high_0

        # Test 100% confidence (should be very wide)
        low_100, mean_100, high_100 = bootstrap_intervals(data, confidence=100)
        assert low_100 <= mean_100 <= high_100


class TestCalculateScoreStatistics:
    """Tests for _calculate_score_statistics function."""

    def test_score_statistics_multiple_scores(self):
        """Test score statistics with multiple scores includes confidence interval."""
        scores = [0.8, 0.85, 0.9, 0.75, 0.88]
        result = _calculate_score_statistics(scores)

        assert result["count"] == 5
        assert result["mean"] == pytest.approx(0.836, rel=1e-3)
        assert result["median"] == 0.85
        assert result["min"] == 0.75
        assert result["max"] == 0.9
        assert result["std"] > 0

        # Confidence interval should be calculated
        ci = result["confidence_interval"]
        assert ci is not None
        assert "low" in ci
        assert "mean" in ci
        assert "high" in ci
        assert ci["confidence_level"] == 95
        assert ci["low"] < ci["mean"] < ci["high"]

    def test_score_statistics_two_scores(self):
        """Test score statistics with exactly 2 scores includes CI."""
        scores = [0.7, 0.9]
        result = _calculate_score_statistics(scores)

        assert result["count"] == 2
        assert result["mean"] == 0.8
        assert result["confidence_interval"] is not None

    def test_score_statistics_single_score_no_ci(self):
        """Test score statistics with single score has no confidence interval."""
        scores = [0.8]
        result = _calculate_score_statistics(scores)

        assert result["count"] == 1
        assert result["mean"] == 0.8
        assert result["std"] == 0.0  # No std for single value
        assert result["confidence_interval"] is None

    def test_score_statistics_empty_scores(self):
        """Test score statistics with empty list returns zeros and no CI."""
        result = _calculate_score_statistics([])

        assert result["count"] == 0
        assert result["mean"] == 0.0
        assert result["median"] == 0.0
        assert result["std"] == 0.0
        assert result["min"] == 0.0
        assert result["max"] == 0.0
        assert result["confidence_interval"] is None


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

    def test_calculate_basic_stats_mixed_results(self):
        """Test calculate_basic_stats with mixed results."""
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

        stats = calculate_basic_stats(results)

        expected = {
            "TOTAL": 4,
            "PASS": 2,
            "FAIL": 1,
            "ERROR": 1,
            "SKIPPED": 0,
            "pass_rate": 50.0,
            "fail_rate": 25.0,
            "error_rate": 25.0,
            "skipped_rate": 0.0,
            "total_judge_llm_input_tokens": 0,
            "total_judge_llm_output_tokens": 0,
            "total_judge_llm_tokens": 0,
        }
        assert stats == expected

    def test_calculate_basic_stats_single_result(self):
        """Test calculate_basic_stats with single result."""
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

        stats = calculate_basic_stats(results)

        expected = {
            "TOTAL": 1,
            "PASS": 0,
            "FAIL": 1,
            "ERROR": 0,
            "SKIPPED": 0,
            "pass_rate": 0.0,
            "fail_rate": 100.0,
            "error_rate": 0.0,
            "skipped_rate": 0.0,
            "total_judge_llm_input_tokens": 0,
            "total_judge_llm_output_tokens": 0,
            "total_judge_llm_tokens": 0,
        }
        assert stats == expected


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

    def test_calculate_detailed_stats_single_metric_single_conversation(self):
        """Test calculate_detailed_stats with single metric and conversation."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good",
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn2",
                metric_identifier="ragas:faithfulness",
                result="FAIL",
                score=0.3,
                threshold=0.7,
                reason="Poor",
            ),
        ]

        stats = calculate_detailed_stats(results)

        # Check by_metric breakdown
        assert "ragas:faithfulness" in stats["by_metric"]
        metric_stats = stats["by_metric"]["ragas:faithfulness"]
        assert metric_stats["pass"] == 1
        assert metric_stats["fail"] == 1
        assert metric_stats["error"] == 0
        assert metric_stats["pass_rate"] == 50.0

        # Check by_conversation breakdown
        assert "conv1" in stats["by_conversation"]
        conv_stats = stats["by_conversation"]["conv1"]
        assert conv_stats["pass"] == 1
        assert conv_stats["fail"] == 1
        assert conv_stats["error"] == 0
        assert conv_stats["pass_rate"] == 50.0

    def test_calculate_detailed_stats_multiple_metrics_conversations(self):
        """Test calculate_detailed_stats with multiple metrics and conversations."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good",
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="ragas:relevancy",
                result="FAIL",
                score=0.3,
                threshold=0.7,
                reason="Poor",
            ),
            EvaluationResult(
                conversation_group_id="conv2",
                turn_id="turn1",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.9,
                threshold=0.7,
                reason="Excellent",
            ),
            EvaluationResult(
                conversation_group_id="conv2",
                turn_id="turn1",
                metric_identifier="ragas:relevancy",
                result="ERROR",
                score=0.0,
                threshold=0.7,
                reason="API error",
            ),
        ]

        stats = calculate_detailed_stats(results)

        # Check metrics
        assert len(stats["by_metric"]) == 2
        assert "ragas:faithfulness" in stats["by_metric"]
        assert "ragas:relevancy" in stats["by_metric"]

        faithfulness_stats = stats["by_metric"]["ragas:faithfulness"]
        assert faithfulness_stats["pass"] == 2
        assert faithfulness_stats["fail"] == 0
        assert faithfulness_stats["error"] == 0
        assert faithfulness_stats["pass_rate"] == 100.0

        relevancy_stats = stats["by_metric"]["ragas:relevancy"]
        assert relevancy_stats["pass"] == 0
        assert relevancy_stats["fail"] == 1
        assert relevancy_stats["error"] == 1
        assert relevancy_stats["pass_rate"] == 0.0
        assert relevancy_stats["fail_rate"] == 50.0
        assert relevancy_stats["error_rate"] == 50.0

        # Check conversations
        assert len(stats["by_conversation"]) == 2
        assert "conv1" in stats["by_conversation"]
        assert "conv2" in stats["by_conversation"]

        conv1_stats = stats["by_conversation"]["conv1"]
        assert conv1_stats["pass"] == 1
        assert conv1_stats["fail"] == 1
        assert conv1_stats["error"] == 0
        assert conv1_stats["pass_rate"] == 50.0

        conv2_stats = stats["by_conversation"]["conv2"]
        assert conv2_stats["pass"] == 1
        assert conv2_stats["fail"] == 0
        assert conv2_stats["error"] == 1
        assert conv2_stats["pass_rate"] == 50.0
        assert conv2_stats["error_rate"] == 50.0

    def test_calculate_detailed_stats_score_statistics(self):
        """Test calculate_detailed_stats includes score statistics."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good",
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn2",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.9,
                threshold=0.7,
                reason="Excellent",
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn3",
                metric_identifier="ragas:faithfulness",
                result="FAIL",
                score=0.3,
                threshold=0.7,
                reason="Poor",
            ),
        ]

        stats = calculate_detailed_stats(results)

        metric_stats = stats["by_metric"]["ragas:faithfulness"]
        assert "score_statistics" in metric_stats

        score_stats = metric_stats["score_statistics"]
        assert score_stats["count"] == 3
        assert score_stats["mean"] == pytest.approx(0.6667, rel=1e-3)
        assert score_stats["min"] == 0.3
        assert score_stats["max"] == 0.9
        assert score_stats["median"] == 0.8
        assert score_stats["std"] > 0  # Should have some standard deviation
        # Confidence interval should be calculated for 3+ scores
        assert "confidence_interval" in score_stats
        ci = score_stats["confidence_interval"]
        assert ci is not None
        assert "low" in ci
        assert "mean" in ci
        assert "high" in ci
        assert ci["confidence_level"] == 95

    def test_calculate_detailed_stats_no_scores(self):
        """Test calculate_detailed_stats with results that have no scores."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="ERROR",
                threshold=0.7,
                reason="API error",
            )
        ]

        stats = calculate_detailed_stats(results)

        metric_stats = stats["by_metric"]["test:metric"]
        assert "score_statistics" in metric_stats

        score_stats = metric_stats["score_statistics"]
        assert score_stats["count"] == 0
        assert score_stats["mean"] == 0.0
        assert score_stats["median"] == 0.0
        assert score_stats["std"] == 0.0
        # Confidence interval should be None when no scores
        assert score_stats["confidence_interval"] is None

    def test_calculate_detailed_stats_single_score_no_confidence_interval(self):
        """Test calculate_detailed_stats with single score has no CI (needs 2+)."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good",
            )
        ]

        stats = calculate_detailed_stats(results)

        metric_stats = stats["by_metric"]["test:metric"]
        score_stats = metric_stats["score_statistics"]
        assert score_stats["count"] == 1
        # Confidence interval should be None for single score
        assert score_stats["confidence_interval"] is None


class TestCalculateApiTokenUsage:
    """Tests for calculate_api_token_usage function."""

    def test_calculate_api_token_usage_empty_data(self):
        """Test calculate_api_token_usage with empty data."""
        result = calculate_api_token_usage([])
        assert result["total_api_input_tokens"] == 0
        assert result["total_api_output_tokens"] == 0
        assert result["total_api_tokens"] == 0

    def test_calculate_api_token_usage_single_turn(self):
        """Test calculate_api_token_usage with single turn."""
        turn = TurnData(
            turn_id="turn1",
            query="Test query",
            response="Test response",
            api_input_tokens=100,
            api_output_tokens=50,
        )
        eval_data = EvaluationData(
            conversation_group_id="conv1",
            turns=[turn],
        )
        result = calculate_api_token_usage([eval_data])
        assert result["total_api_input_tokens"] == 100
        assert result["total_api_output_tokens"] == 50
        assert result["total_api_tokens"] == 150

    def test_calculate_api_token_usage_multiple_turns(self):
        """Test calculate_api_token_usage with multiple turns."""
        turns = [
            TurnData(
                turn_id="turn1",
                query="Query 1",
                response="Response 1",
                api_input_tokens=100,
                api_output_tokens=50,
            ),
            TurnData(
                turn_id="turn2",
                query="Query 2",
                response="Response 2",
                api_input_tokens=150,
                api_output_tokens=75,
            ),
        ]
        eval_data = EvaluationData(
            conversation_group_id="conv1",
            turns=turns,
        )
        result = calculate_api_token_usage([eval_data])
        assert result["total_api_input_tokens"] == 250
        assert result["total_api_output_tokens"] == 125
        assert result["total_api_tokens"] == 375

    def test_calculate_api_token_usage_multiple_conversations(self):
        """Test calculate_api_token_usage with multiple conversations."""
        eval_data1 = EvaluationData(
            conversation_group_id="conv1",
            turns=[
                TurnData(
                    turn_id="turn1",
                    query="Q1",
                    response="R1",
                    api_input_tokens=100,
                    api_output_tokens=50,
                ),
            ],
        )
        eval_data2 = EvaluationData(
            conversation_group_id="conv2",
            turns=[
                TurnData(
                    turn_id="turn1",
                    query="Q2",
                    response="R2",
                    api_input_tokens=200,
                    api_output_tokens=100,
                ),
            ],
        )
        result = calculate_api_token_usage([eval_data1, eval_data2])
        assert result["total_api_input_tokens"] == 300
        assert result["total_api_output_tokens"] == 150
        assert result["total_api_tokens"] == 450

    def test_calculate_api_token_usage_zero_tokens(self):
        """Test calculate_api_token_usage with zero token values."""
        turn = TurnData(
            turn_id="turn1",
            query="Test",
            response="Response",
            api_input_tokens=0,
            api_output_tokens=0,
        )
        eval_data = EvaluationData(
            conversation_group_id="conv1",
            turns=[turn],
        )
        result = calculate_api_token_usage([eval_data])
        assert result["total_api_input_tokens"] == 0
        assert result["total_api_output_tokens"] == 0
        assert result["total_api_tokens"] == 0


class TestCalculateBasicStatsWithTokens:
    """Tests for calculate_basic_stats token tracking fields."""

    def test_basic_stats_includes_token_fields(self):
        """Test that basic stats includes token fields."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                threshold=0.7,
                judge_llm_input_tokens=100,
                judge_llm_output_tokens=50,
            )
        ]
        stats = calculate_basic_stats(results)
        assert "total_judge_llm_input_tokens" in stats
        assert "total_judge_llm_output_tokens" in stats
        assert "total_judge_llm_tokens" in stats

    def test_basic_stats_sums_token_values(self):
        """Test that basic stats correctly sums token values."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                threshold=0.7,
                judge_llm_input_tokens=100,
                judge_llm_output_tokens=50,
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn2",
                metric_identifier="test:metric",
                result="PASS",
                score=0.9,
                threshold=0.7,
                judge_llm_input_tokens=200,
                judge_llm_output_tokens=100,
            ),
        ]
        stats = calculate_basic_stats(results)
        assert stats["total_judge_llm_input_tokens"] == 300
        assert stats["total_judge_llm_output_tokens"] == 150
        assert stats["total_judge_llm_tokens"] == 450

    def test_basic_stats_zero_tokens_by_default(self):
        """Test that results without tokens default to zero."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                threshold=0.7,
            )
        ]
        stats = calculate_basic_stats(results)
        assert stats["total_judge_llm_input_tokens"] == 0
        assert stats["total_judge_llm_output_tokens"] == 0
        assert stats["total_judge_llm_tokens"] == 0

    def test_basic_stats_empty_results_zero_tokens(self):
        """Test that empty results have zero tokens."""
        stats = calculate_basic_stats([])
        assert stats["total_judge_llm_input_tokens"] == 0
        assert stats["total_judge_llm_output_tokens"] == 0
        assert stats["total_judge_llm_tokens"] == 0

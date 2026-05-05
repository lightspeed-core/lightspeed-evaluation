"""Unit tests for detailed statistics module."""

import pytest

from lightspeed_evaluation.core.models import (
    EvaluationResult,
)
from lightspeed_evaluation.core.output.statistics import (
    compute_detailed_stats,
)


class TestCalculateDetailedStats:
    """Tests for compute_detailed_stats function."""

    def test_detailed_stats_with_results(
        self, sample_results_statistics: list[EvaluationResult]
    ) -> None:
        """Test detailed stats calculation."""
        stats = compute_detailed_stats(sample_results_statistics).model_dump()

        assert stats["by_metric"]
        assert stats["by_conversation"]

        assert "metric1" in stats["by_metric"]
        assert "metric2" in stats["by_metric"]
        assert "conv1" in stats["by_conversation"]
        assert "conv2" in stats["by_conversation"]

    def test_detailed_stats_empty_results(self) -> None:
        """Test detailed stats with empty results."""
        stats = compute_detailed_stats([]).model_dump()

        assert not stats["by_metric"]
        assert not stats["by_conversation"]

    def test_detailed_stats_metric_breakdown(
        self, sample_results_statistics: list[EvaluationResult]
    ) -> None:
        """Test metric breakdown in detailed stats."""
        stats = compute_detailed_stats(sample_results_statistics).model_dump()

        metric1_stats = stats["by_metric"]["metric1"]
        assert metric1_stats["passed"] == 1
        assert metric1_stats["failed"] == 1

        metric2_stats = stats["by_metric"]["metric2"]
        assert metric2_stats["passed"] == 1
        assert metric2_stats["error"] == 1

    def test_detailed_stats_conversation_breakdown(
        self, sample_results_statistics: list[EvaluationResult]
    ) -> None:
        """Test conversation breakdown in detailed stats."""
        stats = compute_detailed_stats(sample_results_statistics).model_dump()

        conv1_stats = stats["by_conversation"]["conv1"]
        assert conv1_stats["passed"] == 1
        assert conv1_stats["failed"] == 1

        conv2_stats = stats["by_conversation"]["conv2"]
        assert conv2_stats["passed"] == 1
        assert conv2_stats["error"] == 1

    def test_detailed_stats_includes_rates(
        self, sample_results_statistics: list[EvaluationResult]
    ) -> None:
        """Test that detailed stats include percentage rates."""
        stats = compute_detailed_stats(sample_results_statistics).model_dump()

        metric1_stats = stats["by_metric"]["metric1"]
        assert "pass_rate" in metric1_stats
        assert "fail_rate" in metric1_stats
        assert metric1_stats["pass_rate"] == 50.0
        assert metric1_stats["fail_rate"] == 50.0

    def test_detailed_stats_single_metric(self) -> None:
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

        stats = compute_detailed_stats(results).model_dump()

        assert len(stats["by_metric"]) == 1
        assert "single_metric" in stats["by_metric"]

    def test_compute_detailed_stats_single_metric_single_conversation(self) -> None:
        """Test compute_detailed_stats with single metric and conversation."""
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

        stats = compute_detailed_stats(results).model_dump()

        # Check by_metric breakdown
        assert "ragas:faithfulness" in stats["by_metric"]
        metric_stats = stats["by_metric"]["ragas:faithfulness"]
        assert metric_stats["passed"] == 1
        assert metric_stats["failed"] == 1
        assert metric_stats["error"] == 0
        assert metric_stats["pass_rate"] == 50.0

        # Check by_conversation breakdown
        assert "conv1" in stats["by_conversation"]
        conv_stats = stats["by_conversation"]["conv1"]
        assert conv_stats["passed"] == 1
        assert conv_stats["failed"] == 1
        assert conv_stats["error"] == 0
        assert conv_stats["pass_rate"] == 50.0

    def test_compute_detailed_stats_multiple_metrics_conversations(self) -> None:
        """Test compute_detailed_stats with multiple metrics and conversations."""
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

        stats = compute_detailed_stats(results).model_dump()

        # Check metrics
        assert len(stats["by_metric"]) == 2
        assert "ragas:faithfulness" in stats["by_metric"]
        assert "ragas:relevancy" in stats["by_metric"]

        faithfulness_stats = stats["by_metric"]["ragas:faithfulness"]
        assert faithfulness_stats["passed"] == 2
        assert faithfulness_stats["failed"] == 0
        assert faithfulness_stats["error"] == 0
        assert faithfulness_stats["pass_rate"] == 100.0

        relevancy_stats = stats["by_metric"]["ragas:relevancy"]
        assert relevancy_stats["passed"] == 0
        assert relevancy_stats["failed"] == 1
        assert relevancy_stats["error"] == 1
        assert relevancy_stats["pass_rate"] == 0.0
        assert relevancy_stats["fail_rate"] == 50.0
        assert relevancy_stats["error_rate"] == 50.0

        # Check conversations
        assert len(stats["by_conversation"]) == 2
        assert "conv1" in stats["by_conversation"]
        assert "conv2" in stats["by_conversation"]

        conv1_stats = stats["by_conversation"]["conv1"]
        assert conv1_stats["passed"] == 1
        assert conv1_stats["failed"] == 1
        assert conv1_stats["error"] == 0
        assert conv1_stats["pass_rate"] == 50.0

        conv2_stats = stats["by_conversation"]["conv2"]
        assert conv2_stats["passed"] == 1
        assert conv2_stats["failed"] == 0
        assert conv2_stats["error"] == 1
        assert conv2_stats["pass_rate"] == 50.0
        assert conv2_stats["error_rate"] == 50.0

    def test_compute_detailed_stats_score_statistics(self) -> None:
        """Test compute_detailed_stats includes score statistics."""
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

        stats = compute_detailed_stats(results).model_dump()

        metric_stats = stats["by_metric"]["ragas:faithfulness"]
        assert "score_statistics" in metric_stats

        score_stats = metric_stats["score_statistics"]
        assert score_stats["count"] == 3
        assert score_stats["mean"] == pytest.approx(0.6667, rel=1e-3)
        assert score_stats["min_score"] == 0.3
        assert score_stats["max_score"] == 0.9
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

    def test_compute_detailed_stats_no_scores(self) -> None:
        """Test compute_detailed_stats with results that have no scores."""
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

        stats = compute_detailed_stats(results).model_dump()

        metric_stats = stats["by_metric"]["test:metric"]
        assert "score_statistics" in metric_stats

        score_stats = metric_stats["score_statistics"]
        assert score_stats["count"] == 0
        assert score_stats["mean"] == 0.0
        assert score_stats["median"] == 0.0
        assert score_stats["std"] == 0.0
        # Confidence interval should be None when no scores
        assert score_stats["confidence_interval"] is None

    def test_compute_detailed_stats_single_score_no_confidence_interval(self) -> None:
        """Test compute_detailed_stats with single score has no CI (needs 2+)."""
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

        stats = compute_detailed_stats(results).model_dump()

        metric_stats = stats["by_metric"]["test:metric"]
        score_stats = metric_stats["score_statistics"]
        assert score_stats["count"] == 1
        # Confidence interval should be None for single score
        assert score_stats["confidence_interval"] is None

    def test_compute_detailed_stats_by_tag(self) -> None:
        """Test compute_detailed_stats includes by_tag breakdown."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                tag="production",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                score=0.9,
                threshold=0.7,
                reason="Good",
            ),
            EvaluationResult(
                conversation_group_id="conv2",
                tag="production",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good",
            ),
            EvaluationResult(
                conversation_group_id="conv3",
                tag="staging",
                turn_id="turn1",
                metric_identifier="metric1",
                result="FAIL",
                score=0.5,
                threshold=0.7,
                reason="Below threshold",
            ),
        ]

        stats = compute_detailed_stats(results).model_dump()

        # Verify by_tag is present
        assert "by_tag" in stats
        assert "production" in stats["by_tag"]
        assert "staging" in stats["by_tag"]

        # Check production tag stats
        prod_stats = stats["by_tag"]["production"]
        assert prod_stats["passed"] == 2
        assert prod_stats["failed"] == 0
        assert prod_stats["pass_rate"] == 100.0
        assert "score_statistics" in prod_stats
        assert prod_stats["score_statistics"]["count"] == 2
        assert prod_stats["score_statistics"]["mean"] == pytest.approx(0.85)

        # Check staging tag stats
        staging_stats = stats["by_tag"]["staging"]
        assert staging_stats["passed"] == 0
        assert staging_stats["failed"] == 1
        assert staging_stats["fail_rate"] == 100.0
        assert "score_statistics" in staging_stats

    def test_compute_detailed_stats_default_tag(self) -> None:
        """Test compute_detailed_stats with default 'eval' tag."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                threshold=0.7,
            ),
        ]

        stats = compute_detailed_stats(results).model_dump()

        # Default tag should be "eval"
        assert "by_tag" in stats
        assert "eval" in stats["by_tag"]
        assert stats["by_tag"]["eval"]["passed"] == 1

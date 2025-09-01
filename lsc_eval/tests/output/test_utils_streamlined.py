"""Test cases for output utilities based on system.yaml configuration."""

import pytest
from lsc_eval.output.utils import (
    EvaluationScope,
    calculate_basic_stats,
    calculate_detailed_stats,
)
from lsc_eval.core.models import EvaluationResult, TurnData


class TestEvaluationScope:
    """Test EvaluationScope dataclass."""

    def test_evaluation_scope_creation(self):
        """Test EvaluationScope creation with different configurations."""
        # Default scope
        scope = EvaluationScope()
        assert scope.turn_idx is None
        assert scope.turn_data is None
        assert scope.is_conversation is False

        # Turn-level scope
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        turn_scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)
        assert turn_scope.turn_idx == 0
        assert turn_scope.turn_data == turn_data
        assert turn_scope.is_conversation is False

        # Conversation-level scope
        conv_scope = EvaluationScope(is_conversation=True)
        assert conv_scope.is_conversation is True


class TestCalculateBasicStats:
    """Test calculate_basic_stats function."""

    def test_calculate_basic_stats_scenarios(self):
        """Test basic stats calculation with different scenarios."""
        # Mixed results
        mixed_results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.9
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=2,
                metric_identifier="ragas:faithfulness",
                result="FAIL",
                score=0.3
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=3,
                metric_identifier="ragas:faithfulness",
                result="ERROR",
                score=None
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=4,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.85
            ),
        ]
        
        stats = calculate_basic_stats(mixed_results)
        assert stats["TOTAL"] == 4
        assert stats["PASS"] == 2
        assert stats["FAIL"] == 1
        assert stats["ERROR"] == 1
        assert stats["pass_rate"] == 50.0
        assert stats["fail_rate"] == 25.0
        assert stats["error_rate"] == 25.0

        # Empty results
        empty_stats = calculate_basic_stats([])
        assert empty_stats["TOTAL"] == 0
        assert empty_stats["pass_rate"] == 0.0


class TestCalculateDetailedStats:
    """Test calculate_detailed_stats function."""

    def test_calculate_detailed_stats_comprehensive(self):
        """Test detailed stats calculation with comprehensive data."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.9
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=2,
                metric_identifier="ragas:faithfulness",
                result="FAIL",
                score=0.3
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=1,
                metric_identifier="ragas:response_relevancy",
                result="PASS",
                score=0.85
            ),
            EvaluationResult(
                conversation_group_id="conv2",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.8
            ),
        ]

        detailed_stats = calculate_detailed_stats(results)

        # Check structure
        assert "by_metric" in detailed_stats
        assert "by_conversation" in detailed_stats

        # Check metric stats
        by_metric = detailed_stats["by_metric"]
        assert "ragas:faithfulness" in by_metric
        assert "ragas:response_relevancy" in by_metric
        
        faithfulness_stats = by_metric["ragas:faithfulness"]
        assert faithfulness_stats["pass"] == 2
        assert faithfulness_stats["fail"] == 1
        assert round(faithfulness_stats["pass_rate"], 2) == 66.67

        # Check conversation stats
        by_conversation = detailed_stats["by_conversation"]
        assert "conv1" in by_conversation
        assert "conv2" in by_conversation
        
        conv1_stats = by_conversation["conv1"]
        assert conv1_stats["pass"] == 2
        assert conv1_stats["fail"] == 1

    def test_calculate_detailed_stats_empty(self):
        """Test detailed stats with empty results."""
        detailed_stats = calculate_detailed_stats([])
        
        assert detailed_stats["by_metric"] == {}
        assert detailed_stats["by_conversation"] == {}

    def test_calculate_detailed_stats_score_statistics(self):
        """Test score statistics in detailed stats."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.9
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=2,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.8
            ),
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=3,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.7
            ),
        ]

        detailed_stats = calculate_detailed_stats(results)
        
        faithfulness_stats = detailed_stats["by_metric"]["ragas:faithfulness"]
        score_stats = faithfulness_stats["score_statistics"]
        
        assert score_stats["count"] == 3
        assert score_stats["mean"] == 0.8
        assert score_stats["median"] == 0.8
        assert score_stats["min"] == 0.7
        assert score_stats["max"] == 0.9
        assert score_stats["std"] > 0

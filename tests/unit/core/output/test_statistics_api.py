"""Unit tests for api statistics module."""

import pytest

from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult, TurnData
from lightspeed_evaluation.core.output.statistics import (
    compute_agent_token_usage,
    compute_field_numeric_stats_from_evaluation_data,
    compute_overall_stats,
)


class TestCalculateApiTokenUsage:
    """Tests for compute_agent_token_usage function."""

    def test_compute_agent_token_usage_empty_data(self) -> None:
        """Test compute_agent_token_usage with empty data."""
        result = compute_agent_token_usage([])
        assert result.total_api_input_tokens == 0
        assert result.total_api_output_tokens == 0
        assert result.total_api_tokens == 0

    def test_compute_agent_token_usage_single_turn(self) -> None:
        """Test compute_agent_token_usage with single turn."""
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
        result = compute_agent_token_usage([eval_data])
        assert result.total_api_input_tokens == 100
        assert result.total_api_output_tokens == 50
        assert result.total_api_tokens == 150

    def test_compute_agent_token_usage_multiple_turns(self) -> None:
        """Test compute_agent_token_usage with multiple turns."""
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
        result = compute_agent_token_usage([eval_data])
        assert result.total_api_input_tokens == 250
        assert result.total_api_output_tokens == 125
        assert result.total_api_tokens == 375

    def test_compute_agent_token_usage_multiple_conversations(self) -> None:
        """Test compute_agent_token_usage with multiple conversations."""
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
        result = compute_agent_token_usage([eval_data1, eval_data2])
        assert result.total_api_input_tokens == 300
        assert result.total_api_output_tokens == 150
        assert result.total_api_tokens == 450

    def test_compute_agent_token_usage_zero_tokens(self) -> None:
        """Test compute_agent_token_usage with zero token values."""
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
        result = compute_agent_token_usage([eval_data])
        assert result.total_api_input_tokens == 0
        assert result.total_api_output_tokens == 0
        assert result.total_api_tokens == 0


class TestCalculateApiLatencyStats:
    """Tests for compute_field_numeric_stats_from_evaluation_data function."""

    def test_calculate_api_latency_with_values(self) -> None:
        """Test API latency calculation with valid non-zero values."""
        eval_data = [
            EvaluationData(
                conversation_group_id="conv1",
                turns=[
                    TurnData(turn_id="turn1", query="Q1", agent_latency=1.0),
                    TurnData(turn_id="turn2", query="Q2", agent_latency=2.0),
                ],
            ),
        ]
        result = compute_field_numeric_stats_from_evaluation_data(
            eval_data, "agent_latency"
        )

        assert result is not None
        assert result.count == 2
        assert result.mean == pytest.approx(1.5)
        assert result.median == 1.5
        assert result.min_value == 1.0
        assert result.max_value == 2.0
        assert result.p95 is not None
        assert result.p99 is not None

    def test_calculate_api_latency_excludes_zeros(self) -> None:
        """Test that zero latency values are excluded (no actual API calls)."""
        eval_data = [
            EvaluationData(
                conversation_group_id="conv1",
                turns=[
                    TurnData(turn_id="turn1", query="Q1", agent_latency=0),
                    TurnData(turn_id="turn2", query="Q2", agent_latency=1.5),
                ],
            ),
        ]
        result = compute_field_numeric_stats_from_evaluation_data(
            eval_data, "agent_latency"
        )

        assert result is not None
        assert result.count == 1
        assert result.mean == 1.5

    def test_calculate_api_latency_all_zeros_returns_empty(self) -> None:
        """Test that all-zero latencies return empty count (api_enabled=False scenario)."""
        eval_data = [
            EvaluationData(
                conversation_group_id="conv1",
                turns=[
                    TurnData(turn_id="turn1", query="Q1", agent_latency=0),
                    TurnData(turn_id="turn2", query="Q2", agent_latency=0),
                ],
            ),
        ]
        result = compute_field_numeric_stats_from_evaluation_data(
            eval_data, "agent_latency"
        )

        assert result is None


class TestCalculateBasicStatsWithTokens:
    """Tests for compute_overall_stats token tracking fields."""

    def test_basic_stats_includes_token_fields(self) -> None:
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
                embedding_tokens=150,
            )
        ]
        stats = compute_overall_stats(results)
        assert hasattr(stats, "total_judge_llm_input_tokens")
        assert hasattr(stats, "total_judge_llm_output_tokens")
        assert hasattr(stats, "total_judge_llm_tokens")

        assert hasattr(stats, "total_embedding_tokens")

    def test_basic_stats_sums_token_values(self) -> None:
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
                embedding_tokens=100,
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
                embedding_tokens=250,
            ),
        ]
        stats = compute_overall_stats(results)
        assert stats.total_judge_llm_input_tokens == 300
        assert stats.total_judge_llm_output_tokens == 150
        assert stats.total_judge_llm_tokens == 450

        assert stats.total_embedding_tokens == 350
        assert stats.total_embedding_tokens == 350

    def test_basic_stats_zero_tokens_by_default(self) -> None:
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
        stats = compute_overall_stats(results)
        assert stats.total_judge_llm_input_tokens == 0
        assert stats.total_judge_llm_output_tokens == 0
        assert stats.total_judge_llm_tokens == 0

        assert stats.total_embedding_tokens == 0
        assert stats.total_embedding_tokens == 0

    def test_basic_stats_empty_results_zero_tokens(self) -> None:
        """Test that empty results have zero tokens."""
        stats = compute_overall_stats([])
        assert stats.total_judge_llm_input_tokens == 0
        assert stats.total_judge_llm_output_tokens == 0
        assert stats.total_judge_llm_tokens == 0
        assert stats.total_embedding_tokens == 0
        assert stats.total_embedding_tokens == 0

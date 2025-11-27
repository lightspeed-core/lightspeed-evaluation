"""Additional tests for data models to increase coverage."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models import TurnData, EvaluationData, EvaluationResult


class TestTurnData:
    """Additional tests for TurnData."""

    def test_turn_data_with_all_optional_fields(self):
        """Test TurnData with all optional fields populated."""
        turn = TurnData(
            turn_id="turn1",
            query="What is Python?",
            response="Python is a language",
            contexts=["context1"],
            expected_response="Expected",
            expected_keywords=[["python", "language"]],
            expected_intent="information",
            tool_calls=[[{"tool_name": "search", "arguments": {}}]],
            expected_tool_calls=[[[{"tool_name": "search", "arguments": {}}]]],
            verify_script="verify.sh",
            turn_metrics=["ragas:faithfulness"],
        )

        assert turn.turn_id == "turn1"
        assert turn.query == "What is Python?"
        assert turn.response == "Python is a language"
        assert len(turn.contexts) == 1
        assert turn.expected_response == "Expected"

    def test_turn_data_minimal_fields(self):
        """Test TurnData with only required fields."""
        turn = TurnData(
            turn_id="turn1",
            query="Test query",
        )

        assert turn.turn_id == "turn1"
        assert turn.query == "Test query"
        assert turn.response is None
        assert turn.contexts is None

    def test_turn_data_empty_turn_id_fails(self):
        """Test that empty turn_id fails validation."""
        with pytest.raises(ValidationError):
            TurnData(turn_id="", query="Test")

    def test_turn_data_empty_query_fails(self):
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError):
            TurnData(turn_id="turn1", query="")


class TestEvaluationData:
    """Additional tests for EvaluationData."""

    def test_evaluation_data_with_multiple_turns(self):
        """Test EvaluationData with multiple turns."""
        turns = [
            TurnData(turn_id="turn1", query="Query 1"),
            TurnData(turn_id="turn2", query="Query 2"),
            TurnData(turn_id="turn3", query="Query 3"),
        ]

        eval_data = EvaluationData(
            conversation_group_id="conv1",
            turns=turns,
        )

        assert eval_data.conversation_group_id == "conv1"
        assert len(eval_data.turns) == 3

    def test_evaluation_data_with_conversation_metrics(self):
        """Test EvaluationData with conversation-level metrics."""
        turn = TurnData(turn_id="turn1", query="Query")

        eval_data = EvaluationData(
            conversation_group_id="conv1",
            turns=[turn],
            conversation_metrics=["deepeval:conversation_completeness"],
        )

        assert len(eval_data.conversation_metrics) == 1

    def test_evaluation_data_empty_conversation_id_fails(self):
        """Test that empty conversation_group_id fails."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="", turns=[turn])

    def test_evaluation_data_empty_turns_fails(self):
        """Test that empty turns list fails."""
        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="conv1", turns=[])


class TestEvaluationResult:
    """Additional tests for EvaluationResult."""

    def test_evaluation_result_full_data(self):
        """Test EvaluationResult with all fields."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.95,
            threshold=0.7,
            reason="High score",
            query="What is Python?",
            response="Python is a language",
            execution_time=1.5,
        )

        assert result.conversation_group_id == "conv1"
        assert result.result == "PASS"
        assert result.score == 0.95
        assert result.execution_time == 1.5

    def test_evaluation_result_error_state(self):
        """Test EvaluationResult in ERROR state."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            result="ERROR",
            threshold=0.7,
            reason="API timeout",
        )

        assert result.result == "ERROR"
        assert result.score is None

    def test_evaluation_result_fail_state(self):
        """Test EvaluationResult in FAIL state."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="metric1",
            result="FAIL",
            score=0.5,
            threshold=0.7,
            reason="Below threshold",
        )

        assert result.result == "FAIL"
        assert result.score < result.threshold

    def test_evaluation_result_minimal_fields(self):
        """Test EvaluationResult with minimal required fields."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="metric1",
            result="PASS",
            threshold=0.7,
        )

        assert result.conversation_group_id == "conv1"
        assert result.turn_id == "turn1"
        assert result.result == "PASS"

    def test_evaluation_result_with_zero_execution_time(self):
        """Test EvaluationResult with zero execution time."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="metric1",
            result="PASS",
            threshold=0.7,
            execution_time=0,
        )

        assert result.execution_time == 0

    def test_evaluation_result_with_conversation_level(self):
        """Test EvaluationResult for conversation-level metric."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id=None,  # Conversation-level doesn't need turn_id
            metric_identifier="deepeval:conversation_completeness",
            result="PASS",
            score=0.85,
            threshold=0.7,
        )

        assert result.turn_id is None
        assert result.result == "PASS"

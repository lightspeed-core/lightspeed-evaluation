"""Tests for data models."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.data import (
    EvaluationData,
    EvaluationResult,
    TurnData,
)


class TestTurnData:
    """General tests for TurnData model."""

    def test_minimal_fields(self):
        """Test TurnData with only required fields."""
        turn = TurnData(turn_id="turn1", query="Test query")

        assert turn.turn_id == "turn1"
        assert turn.query == "Test query"
        assert turn.response is None
        assert turn.contexts is None

    def test_empty_turn_id_fails(self):
        """Test that empty turn_id fails validation."""
        with pytest.raises(ValidationError):
            TurnData(turn_id="", query="Test")

    def test_empty_query_fails(self):
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError):
            TurnData(turn_id="turn1", query="")


class TestTurnDataToolCallsValidation:
    """Test cases for TurnData expected_tool_calls field validation and conversion."""

    def test_single_set_format_conversion(self):
        """Test that single set format is converted to multiple sets format."""
        # Single set format (backward compatibility)
        turn_data = TurnData(
            turn_id="test_single",
            query="Test query",
            expected_tool_calls=[
                [{"tool_name": "test_tool", "arguments": {"key": "value"}}]
            ],
        )

        # Should be converted to multiple sets format
        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 1  # One alternative set
        assert len(expected[0]) == 1  # One sequence in the set
        assert len(expected[0][0]) == 1  # One tool call in the sequence
        assert expected[0][0][0]["tool_name"] == "test_tool"

    def test_multiple_sets_format_preserved(self):
        """Test that multiple sets format is preserved as-is."""
        # Multiple sets format
        turn_data = TurnData(
            turn_id="test_multiple",
            query="Test query",
            expected_tool_calls=[
                [[{"tool_name": "tool1", "arguments": {"key": "value1"}}]],
                [[{"tool_name": "tool2", "arguments": {"key": "value2"}}]],
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 2  # Two alternative sets
        assert expected[0][0][0]["tool_name"] == "tool1"
        assert expected[1][0][0]["tool_name"] == "tool2"

    def test_empty_alternatives_allowed(self):
        """Test that empty alternatives are allowed as fallback."""
        turn_data = TurnData(
            turn_id="test_flexible",
            query="Test query",
            expected_tool_calls=[
                [[{"tool_name": "cache_check", "arguments": {"key": "data"}}]],
                [],  # Alternative: skip tool (empty)
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 2
        assert len(expected[0]) == 1  # First set has one sequence
        assert len(expected[1]) == 0  # Second set is empty

    def test_complex_sequences(self):
        """Test complex tool call sequences."""
        turn_data = TurnData(
            turn_id="test_complex",
            query="Test query",
            expected_tool_calls=[
                [
                    [{"tool_name": "validate", "arguments": {}}],
                    [{"tool_name": "deploy", "arguments": {}}],
                ],
                [[{"tool_name": "deploy", "arguments": {}}]],
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 2
        assert len(expected[0]) == 2  # Two sequences in first set
        assert len(expected[1]) == 1  # One sequence in second set

    def test_none_expected_tool_calls(self):
        """Test that None is handled correctly."""
        turn_data = TurnData(
            turn_id="test_none", query="Test query", expected_tool_calls=None
        )
        assert turn_data.expected_tool_calls is None

    def test_regex_arguments_preserved(self):
        """Test that regex patterns in arguments are preserved."""
        turn_data = TurnData(
            turn_id="test_regex",
            query="Test query",
            expected_tool_calls=[
                [[{"tool_name": "get_pod", "arguments": {"name": "web-server-[0-9]+"}}]]
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert expected[0][0][0]["arguments"]["name"] == "web-server-[0-9]+"

    def test_invalid_format_rejected(self):
        """Test that non-list format is rejected."""
        with pytest.raises(ValidationError):
            TurnData(
                turn_id="test_invalid",
                query="Test query",
                expected_tool_calls="not_a_list",
            )

    def test_invalid_tool_call_structure_rejected(self):
        """Test that invalid tool call structure is rejected."""
        with pytest.raises(ValidationError):
            TurnData(
                turn_id="test_invalid_structure",
                query="Test query",
                expected_tool_calls=[[[{"invalid": "structure"}]]],
            )

    def test_empty_sequence_rejected(self):
        """Test that empty sequences are rejected."""
        with pytest.raises(
            ValidationError,
            match="Empty sequence at position 0 in alternative 0 is invalid",
        ):
            TurnData(
                turn_id="test_invalid_empty_sequence",
                query="Test query",
                expected_tool_calls=[[]],
            )

    def test_empty_set_as_first_element_rejected(self):
        """Test that empty set as the first element is rejected."""
        with pytest.raises(ValidationError, match="Empty set cannot be the first"):
            TurnData(
                turn_id="test_empty_sequences",
                query="Test query",
                expected_tool_calls=[[], []],
            )

    def test_multiple_empty_alternatives_rejected(self):
        """Test that multiple empty alternatives are rejected as redundant."""
        with pytest.raises(
            ValidationError, match="Found 2 empty alternatives.*redundant"
        ):
            TurnData(
                turn_id="test_redundant_empty",
                query="Test query",
                expected_tool_calls=[
                    [[{"tool_name": "tool1", "arguments": {}}]],
                    [],
                    [],
                ],
            )


class TestTurnDataFormatDetection:
    """Test cases for format detection logic."""

    def test_empty_list_rejected(self):
        """Test that empty list is rejected."""
        with pytest.raises(
            ValidationError, match="Empty set cannot be the only alternative"
        ):
            TurnData(turn_id="test", query="Test", expected_tool_calls=[])

    def test_is_single_set_format_detection(self):
        """Test detection of single set format."""
        turn_data = TurnData(
            turn_id="test",
            query="Test",
            expected_tool_calls=[
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 1  # One alternative set
        assert len(expected[0]) == 2  # Two sequences in that set


class TestTurnDataKeywordsValidation:
    """Test cases for expected_keywords validation in TurnData."""

    def test_valid_single_group(self):
        """Test valid expected_keywords with single group."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            expected_keywords=[["keyword1", "keyword2"]],
        )
        assert turn_data.expected_keywords == [["keyword1", "keyword2"]]

    def test_valid_multiple_groups(self):
        """Test valid expected_keywords with multiple groups."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            expected_keywords=[
                ["yes", "confirmed"],
                ["monitoring", "namespace"],
            ],
        )
        assert len(turn_data.expected_keywords) == 2

    def test_none_is_valid(self):
        """Test that None is valid for expected_keywords."""
        turn_data = TurnData(
            turn_id="test_turn", query="Test query", expected_keywords=None
        )
        assert turn_data.expected_keywords is None

    def test_non_list_rejected(self):
        """Test that non-list expected_keywords is rejected."""
        with pytest.raises(ValidationError, match="Input should be a valid list"):
            TurnData(
                turn_id="test_turn", query="Test query", expected_keywords="not_a_list"
            )

    def test_empty_inner_list_rejected(self):
        """Test that empty inner lists are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            TurnData(
                turn_id="test_turn",
                query="Test query",
                expected_keywords=[[], ["valid_list"]],
            )

    def test_empty_string_element_rejected(self):
        """Test that empty string elements are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty or whitespace"):
            TurnData(
                turn_id="test_turn",
                query="Test query",
                expected_keywords=[["valid_string", ""]],
            )


class TestEvaluationData:
    """Tests for EvaluationData model."""

    def test_valid_creation(self):
        """Test EvaluationData creation with valid data."""
        turns = [
            TurnData(turn_id="turn1", query="First query"),
            TurnData(turn_id="turn2", query="Second query", response="Response"),
        ]

        eval_data = EvaluationData(
            conversation_group_id="conv1",
            turns=turns,
            description="Test conversation",
            tag="test_tag",
            conversation_metrics=["deepeval:conversation_completeness"],
        )

        assert eval_data.conversation_group_id == "conv1"
        assert eval_data.tag == "test_tag"
        assert len(eval_data.turns) == 2
        assert eval_data.description == "Test conversation"
        assert len(eval_data.conversation_metrics) == 1

    def test_default_tag_value(self):
        """Test EvaluationData has correct default tag value."""
        turn = TurnData(turn_id="turn1", query="Query")
        eval_data = EvaluationData(conversation_group_id="conv1", turns=[turn])

        assert eval_data.tag == "eval"

    def test_empty_tag_rejected(self):
        """Test that empty tag is rejected."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="conv1", turns=[turn], tag="")

    def test_empty_conversation_id_rejected(self):
        """Test that empty conversation_group_id is rejected."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="", turns=[turn])

    def test_empty_turns_rejected(self):
        """Test that empty turns list is rejected."""
        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="conv1", turns=[])


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_default_values(self):
        """Test EvaluationResult has correct default values."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="metric1",
            result="PASS",
            threshold=0.7,
        )

        # Test meaningful defaults
        assert result.tag == "eval"
        assert result.score is None
        assert result.reason == ""
        assert result.execution_time == 0

    def test_explicit_tag_value(self):
        """Test EvaluationResult with explicit tag value."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            tag="custom_tag",
            turn_id="turn1",
            metric_identifier="metric1",
            result="PASS",
            threshold=0.7,
        )

        assert result.tag == "custom_tag"

    def test_empty_tag_rejected(self):
        """Test that empty tag is rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                conversation_group_id="conv1",
                tag="",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                threshold=0.7,
            )

    def test_invalid_result_status_rejected(self):
        """Test that invalid result status is rejected."""
        with pytest.raises(ValidationError, match="Result must be one of"):
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="metric1",
                result="INVALID_STATUS",
                threshold=0.7,
            )

    def test_negative_execution_time_rejected(self):
        """Test that negative execution_time is rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                threshold=0.7,
                execution_time=-1,
            )

    def test_conversation_level_metric_allows_none_turn_id(self):
        """Test that turn_id can be None for conversation-level metrics."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id=None,
            metric_identifier="deepeval:conversation_completeness",
            result="PASS",
            threshold=0.7,
        )

        assert result.turn_id is None

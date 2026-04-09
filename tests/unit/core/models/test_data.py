"""Tests for data models."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.data import (
    EvaluationData,
    EvaluationResult,
    JudgeScore,
    MetricResult,
    TurnData,
)


class TestTurnData:
    """General tests for TurnData model."""

    def test_minimal_fields(self) -> None:
        """Test TurnData with only required fields."""
        turn = TurnData(turn_id="turn1", query="Test query")

        assert turn.turn_id == "turn1"
        assert turn.query == "Test query"
        assert turn.response is None
        assert turn.contexts is None

    def test_empty_turn_id_fails(self) -> None:
        """Test that empty turn_id fails validation."""
        with pytest.raises(ValidationError):
            TurnData(turn_id="", query="Test")

    def test_empty_query_fails(self) -> None:
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError):
            TurnData(turn_id="turn1", query="")

    def test_turn_metrics_metadata_accepted_at_load(self) -> None:
        """Override dict is accepted at load; GEval is validated at eval time when resolved."""
        turn = TurnData(
            turn_id="turn1",
            query="Q",
            turn_metrics_metadata={"geval:custom": {"threshold": 0.8}},
        )
        assert turn.turn_metrics_metadata == {"geval:custom": {"threshold": 0.8}}


class TestTurnDataExtraRequestParams:
    """Test cases for TurnData extra_request_params field."""

    def test_extra_request_params_accepted(self) -> None:
        """Test TurnData accepts extra_request_params dict."""
        turn = TurnData(
            turn_id="turn1",
            query="Q",
            extra_request_params={"mode": "troubleshooting"},
        )
        assert turn.extra_request_params == {"mode": "troubleshooting"}

    def test_extra_request_params_none_default(self) -> None:
        """Test TurnData defaults extra_request_params to None."""
        turn = TurnData(turn_id="turn1", query="Q")
        assert turn.extra_request_params is None

    def test_extra_request_params_multiple_keys(self) -> None:
        """Test TurnData accepts multiple extra params."""
        turn = TurnData(
            turn_id="turn1",
            query="Q",
            extra_request_params={"mode": "ask", "custom_flag": True},
        )
        assert turn.extra_request_params == {"mode": "ask", "custom_flag": True}


class TestTurnDataToolCallsValidation:
    """Test cases for TurnData expected_tool_calls field validation and conversion."""

    def test_single_set_format_conversion(self) -> None:
        """Test that single set format is converted to multiple sets format."""
        # Single set format (backward compatibility)
        turn_data = TurnData(
            turn_id="test_single",
            query="Test query",
            expected_tool_calls=[  # pyright: ignore[reportArgumentType]
                [{"tool_name": "test_tool", "arguments": {"key": "value"}}]
            ],
        )

        # Should be converted to multiple sets format
        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert (
            len(expected) == 1  # pylint: disable=unsubscriptable-object
        )  # One alternative set
        assert (
            len(expected[0]) == 1  # pylint: disable=unsubscriptable-object
        )  # One sequence in the set
        assert (
            len(expected[0][0]) == 1  # pylint: disable=unsubscriptable-object
        )  # One tool call in the sequence
        assert (
            expected[0][0][0]["tool_name"]  # pylint: disable=unsubscriptable-object
            == "test_tool"
        )

    def test_multiple_sets_format_preserved(self) -> None:
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
        assert (
            expected[0][0][0]["tool_name"]  # pylint: disable=unsubscriptable-object
            == "tool1"
        )
        assert (
            expected[1][0][0]["tool_name"]  # pylint: disable=unsubscriptable-object
            == "tool2"
        )

    def test_empty_alternatives_allowed(self) -> None:
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
        assert (
            len(expected[0]) == 1  # pylint: disable=unsubscriptable-object
        )  # First set has one sequence
        assert (
            len(expected[1]) == 0  # pylint: disable=unsubscriptable-object
        )  # Second set is empty

    def test_complex_sequences(self) -> None:
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
        assert (
            len(expected[0]) == 2  # pylint: disable=unsubscriptable-object
        )  # Two sequences in first set
        assert (
            len(expected[1]) == 1  # pylint: disable=unsubscriptable-object
        )  # One sequence in second set

    def test_none_expected_tool_calls(self) -> None:
        """Test that None is handled correctly."""
        turn_data = TurnData(
            turn_id="test_none", query="Test query", expected_tool_calls=None
        )
        assert turn_data.expected_tool_calls is None

    def test_regex_arguments_preserved(self) -> None:
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
        assert (
            expected[0][0][0]["arguments"][  # pylint: disable=unsubscriptable-object
                "name"
            ]
            == "web-server-[0-9]+"
        )

    def test_invalid_format_rejected(self) -> None:
        """Test that non-list format is rejected."""
        with pytest.raises(ValidationError):
            TurnData(
                turn_id="test_invalid",
                query="Test query",
                expected_tool_calls="not_a_list",  # pyright: ignore[reportArgumentType]
            )

    def test_invalid_tool_call_structure_rejected(self) -> None:
        """Test that invalid tool call structure is rejected."""
        with pytest.raises(ValidationError):
            TurnData(
                turn_id="test_invalid_structure",
                query="Test query",
                expected_tool_calls=[[[{"invalid": "structure"}]]],
            )

    def test_empty_sequence_rejected(self) -> None:
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

    def test_empty_set_as_first_element_rejected(self) -> None:
        """Test that empty set as the first element is rejected."""
        with pytest.raises(ValidationError, match="Empty set cannot be the first"):
            TurnData(
                turn_id="test_empty_sequences",
                query="Test query",
                expected_tool_calls=[[], []],
            )

    def test_multiple_empty_alternatives_rejected(self) -> None:
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

    def test_empty_list_rejected(self) -> None:
        """Test that empty list is rejected."""
        with pytest.raises(
            ValidationError, match="Empty set cannot be the only alternative"
        ):
            TurnData(turn_id="test", query="Test", expected_tool_calls=[])

    def test_is_single_set_format_detection(self) -> None:
        """Test detection of single set format."""
        turn_data = TurnData(
            turn_id="test",
            query="Test",
            expected_tool_calls=[  # pyright: ignore[reportArgumentType]
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 1  # One alternative set
        assert (
            len(expected[0]) == 2  # pylint: disable=unsubscriptable-object
        )  # Two sequences in that set


class TestTurnDataExpectedResponseValidation:
    """Test cases for expected_response validation in TurnData."""

    @pytest.mark.parametrize(
        "valid_response",
        ["Single word", ["Response option 1", "Response option 2"]],
    )
    def test_valid_expected_response(self, valid_response: str | list[str]) -> None:
        """Test valid expected_response values."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            expected_response=valid_response,
        )
        assert turn_data.expected_response == valid_response

    def test_none_expected_response_valid(self) -> None:
        """Test that None is valid for expected_response."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            expected_response=None,
        )
        assert turn_data.expected_response is None

    @pytest.mark.parametrize(
        "invalid_response,match_pattern",
        [
            ("", "cannot be empty or whitespace"),
            ("   ", "cannot be empty or whitespace"),
            ([], "expected_response list cannot be empty"),
            (["valid", ""], "cannot be empty or whitespace"),
            (["valid", "   "], "cannot be empty or whitespace"),
        ],
    )
    def test_invalid_expected_response(
        self, invalid_response: str | list[str], match_pattern: str
    ) -> None:
        """Test that invalid expected_response values are rejected."""
        with pytest.raises(ValidationError, match=match_pattern):
            TurnData(
                turn_id="test_turn",
                query="Test query",
                expected_response=invalid_response,
            )


class TestTurnDataKeywordsValidation:
    """Test cases for expected_keywords validation in TurnData."""

    def test_valid_single_group(self) -> None:
        """Test valid expected_keywords with single group."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            expected_keywords=[["keyword1", "keyword2"]],
        )
        assert turn_data.expected_keywords == [["keyword1", "keyword2"]]

    def test_valid_multiple_groups(self) -> None:
        """Test valid expected_keywords with multiple groups."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            expected_keywords=[
                ["yes", "confirmed"],
                ["monitoring", "namespace"],
            ],
        )
        assert turn_data.expected_keywords is not None
        assert len(turn_data.expected_keywords) == 2

    def test_none_is_valid(self) -> None:
        """Test that None is valid for expected_keywords."""
        turn_data = TurnData(
            turn_id="test_turn", query="Test query", expected_keywords=None
        )
        assert turn_data.expected_keywords is None

    def test_non_list_rejected(self) -> None:
        """Test that non-list expected_keywords is rejected."""
        with pytest.raises(ValidationError, match="Input should be a valid list"):
            TurnData(
                turn_id="test_turn",
                query="Test query",
                expected_keywords="not_a_list",  # pyright: ignore[reportArgumentType]
            )

    def test_empty_inner_list_rejected(self) -> None:
        """Test that empty inner lists are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            TurnData(
                turn_id="test_turn",
                query="Test query",
                expected_keywords=[[], ["valid_list"]],
            )

    def test_empty_string_element_rejected(self) -> None:
        """Test that empty string elements are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty or whitespace"):
            TurnData(
                turn_id="test_turn",
                query="Test query",
                expected_keywords=[["valid_string", ""]],
            )


class TestEvaluationData:
    """Tests for EvaluationData model."""

    def test_valid_creation(self) -> None:
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
        assert eval_data.conversation_metrics is not None
        assert len(eval_data.conversation_metrics) == 1

    def test_default_tag_value(self) -> None:
        """Test EvaluationData has correct default tag value."""
        turn = TurnData(turn_id="turn1", query="Query")
        eval_data = EvaluationData(conversation_group_id="conv1", turns=[turn])

        assert eval_data.tag == "eval"

    def test_empty_tag_rejected(self) -> None:
        """Test that empty tag is rejected."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="conv1", turns=[turn], tag="")

    def test_empty_conversation_id_rejected(self) -> None:
        """Test that empty conversation_group_id is rejected."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="", turns=[turn])

    def test_empty_turns_rejected(self) -> None:
        """Test that empty turns list is rejected."""
        with pytest.raises(ValidationError):
            EvaluationData(conversation_group_id="conv1", turns=[])

    def test_conversation_metrics_metadata_accepted_at_load(self) -> None:
        """Override dict is accepted at load; GEval is validated at eval time when resolved."""
        turn = TurnData(turn_id="turn1", query="Q")
        eval_data = EvaluationData(
            conversation_group_id="conv1",
            turns=[turn],
            conversation_metrics_metadata={"geval:custom": {"threshold": 0.7}},
        )
        assert eval_data.conversation_metrics_metadata == {
            "geval:custom": {"threshold": 0.7}
        }


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_default_values(self) -> None:
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

    def test_explicit_tag_value(self) -> None:
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

    def test_empty_tag_rejected(self) -> None:
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

    def test_invalid_result_status_rejected(self) -> None:
        """Test that invalid result status is rejected."""
        with pytest.raises(ValidationError, match="Result must be one of"):
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="metric1",
                result="INVALID_STATUS",
                threshold=0.7,
            )

    def test_negative_execution_time_rejected(self) -> None:
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

    def test_conversation_level_metric_allows_none_turn_id(self) -> None:
        """Test that turn_id can be None for conversation-level metrics."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id=None,
            metric_identifier="deepeval:conversation_completeness",
            result="PASS",
            threshold=0.7,
        )

        assert result.turn_id is None


class TestJudgeScore:
    """Tests for JudgeScore model used in judge panel evaluations."""

    def test_valid_creation(self) -> None:
        """Test JudgeScore creation with valid data."""
        judge_score = JudgeScore(
            judge_id="gpt-4o-mini",
            score=0.85,
            reason="Response is accurate and relevant",
            input_tokens=150,
            output_tokens=50,
        )

        assert judge_score.judge_id == "gpt-4o-mini"
        assert judge_score.score == 0.85
        assert judge_score.reason == "Response is accurate and relevant"
        assert judge_score.input_tokens == 150
        assert judge_score.output_tokens == 50

    def test_default_values(self) -> None:
        """Test JudgeScore has correct default values."""
        judge_score = JudgeScore(judge_id="gpt-4o")

        assert judge_score.score is None
        assert judge_score.reason == ""
        assert judge_score.input_tokens == 0
        assert judge_score.output_tokens == 0

    def test_empty_judge_id_rejected(self) -> None:
        """Test that empty judge_id is rejected."""
        with pytest.raises(ValidationError):
            JudgeScore(judge_id="")

    def test_score_out_of_range_rejected(self) -> None:
        """Test that score outside 0-1 range is rejected."""
        with pytest.raises(ValidationError):
            JudgeScore(judge_id="gpt-4o", score=1.5)

        with pytest.raises(ValidationError):
            JudgeScore(judge_id="gpt-4o", score=-0.1)

    def test_negative_tokens_rejected(self) -> None:
        """Test that negative token counts are rejected."""
        with pytest.raises(ValidationError):
            JudgeScore(judge_id="gpt-4o", input_tokens=-1)

        with pytest.raises(ValidationError):
            JudgeScore(judge_id="gpt-4o", output_tokens=-1)


class TestMetricResultJudgeScores:
    """Tests for MetricResult.judge_scores field."""

    def test_metric_result_with_judge_scores(self) -> None:
        """Test MetricResult with judge_scores populated."""
        judge_scores = [
            JudgeScore(judge_id="gpt-4o-mini", score=0.8, input_tokens=100),
            JudgeScore(judge_id="gpt-4o", score=0.9, input_tokens=120),
            JudgeScore(judge_id="gemini-flash", score=0.85, input_tokens=110),
        ]

        result = MetricResult(
            result="PASS",
            score=0.85,
            threshold=0.7,
            reason="Aggregated from 3 judges",
            judge_llm_input_tokens=330,
            judge_llm_output_tokens=150,
            judge_scores=judge_scores,
        )

        assert result.judge_scores is not None
        assert len(result.judge_scores) == 3
        # pylint: disable=unsubscriptable-object
        assert result.judge_scores[0].judge_id == "gpt-4o-mini"
        assert result.judge_scores[1].score == 0.9
        assert result.score == 0.85

    def test_metric_result_without_judge_scores(self) -> None:
        """Test MetricResult without judge_scores (single judge mode)."""
        result = MetricResult(
            result="PASS",
            score=0.8,
            threshold=0.7,
            reason="Single judge evaluation",
        )

        assert result.judge_scores is None

    def test_evaluation_result_inherits_judge_scores(self) -> None:
        """Test that EvaluationResult inherits judge_scores from MetricResult."""
        judge_scores = [
            JudgeScore(judge_id="gpt-4o-mini", score=0.75),
            JudgeScore(judge_id="gpt-4o", score=0.85),
        ]

        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.8,
            threshold=0.7,
            judge_scores=judge_scores,
        )

        assert result.judge_scores is not None
        assert len(result.judge_scores) == 2

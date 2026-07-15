"""Tests for data models."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.data import (
    ConversationMetadata,
    DatasetMetadata,
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

    def test_query_populated_from_proposal_spec_request(self) -> None:
        """Test query is auto-populated from proposal_spec.request."""
        turn = TurnData(
            turn_id="turn1",
            proposal_spec={"request": "Fix the OOM issue"},
        )
        assert turn.query == "Fix the OOM issue"

    def test_query_preserved_when_both_provided(self) -> None:
        """Test explicit query is not overridden by proposal_spec.request."""
        turn = TurnData(
            turn_id="turn1",
            query="Explicit query",
            proposal_spec={"request": "Spec request"},
        )
        assert turn.query == "Explicit query"

    def test_no_query_no_proposal_spec_fails(self) -> None:
        """Test that missing query without proposal_spec raises."""
        with pytest.raises(ValidationError, match="query is required"):
            TurnData(turn_id="turn1")

    def test_no_query_proposal_spec_missing_request_fails(self) -> None:
        """Test that proposal_spec without request key raises."""
        with pytest.raises(ValidationError, match="proposal_spec must contain"):
            TurnData(turn_id="turn1", proposal_spec={"analysis": {}})

    def test_no_query_proposal_spec_empty_request_fails(self) -> None:
        """Test that proposal_spec with empty request raises."""
        with pytest.raises(ValidationError, match="proposal_spec must contain"):
            TurnData(turn_id="turn1", proposal_spec={"request": ""})

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
            tag={"test_tag"},
            conversation_metrics=["deepeval:conversation_completeness"],
        )

        assert eval_data.conversation_group_id == "conv1"
        assert eval_data.tag == {"test_tag"}
        assert len(eval_data.turns) == 2
        assert eval_data.description == "Test conversation"
        assert eval_data.conversation_metrics is not None
        assert len(eval_data.conversation_metrics) == 1

    def test_default_tag_value(self) -> None:
        """Test EvaluationData has correct default tag value."""
        turn = TurnData(turn_id="turn1", query="Query")
        eval_data = EvaluationData(conversation_group_id="conv1", turns=[turn])

        assert eval_data.tag == {"eval"}

    def test_single_string_tag_normalized_to_set(self) -> None:
        """Test that a single string tag is normalized to a set."""
        turn = TurnData(turn_id="turn1", query="Query")
        eval_data = EvaluationData(
            conversation_group_id="conv1", turns=[turn], tag="basic"  # type: ignore[arg-type]
        )

        assert eval_data.tag == {"basic"}

    def test_list_tag_accepted(self) -> None:
        """Test that a list of tags is accepted."""
        turn = TurnData(turn_id="turn1", query="Query")
        eval_data = EvaluationData(
            conversation_group_id="conv1", turns=[turn], tag={"basic", "advanced"}
        )

        assert eval_data.tag == {"basic", "advanced"}

    def test_empty_tag_list_rejected(self) -> None:
        """Test that empty list tag is rejected."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(
                conversation_group_id="conv1",
                turns=[turn],
                tag=[],  # type: ignore[arg-type]
            )

    def test_whitespace_only_tag_rejected(self) -> None:
        """Test that a list of only whitespace strings is rejected."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(
                conversation_group_id="conv1",
                turns=[turn],
                tag=["  "],  # type: ignore[arg-type]
            )

    def test_non_string_tag_items_rejected(self) -> None:
        """Test that non-string items in tag list are rejected."""
        turn = TurnData(turn_id="turn1", query="Query")

        with pytest.raises(ValidationError):
            EvaluationData(
                conversation_group_id="conv1",
                turns=[turn],
                tag=[1, "prod"],  # type: ignore[list-item]
            )

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
        assert result.tag == {"eval"}
        assert result.score is None
        assert result.reason == ""
        assert result.evaluation_latency == 0

    def test_explicit_tag_value(self) -> None:
        """Test EvaluationResult with explicit tag value."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            tag={"custom_tag"},
            turn_id="turn1",
            metric_identifier="metric1",
            result="PASS",
            threshold=0.7,
        )

        assert result.tag == {"custom_tag"}

    def test_empty_tag_list_rejected(self) -> None:
        """Test that empty tag list is rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                conversation_group_id="conv1",
                tag=[],  # type: ignore[arg-type]
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                threshold=0.7,
            )

    def test_whitespace_only_tag_rejected(self) -> None:
        """Test that a list of only whitespace strings is rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                conversation_group_id="conv1",
                tag=["  "],  # type: ignore[arg-type]
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                threshold=0.7,
            )

    def test_non_string_tag_items_rejected(self) -> None:
        """Test that non-string items in tag list are rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                conversation_group_id="conv1",
                tag=[1, "prod"],  # type: ignore[list-item]
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

    def test_negative_evaluation_latency_rejected(self) -> None:
        """Test that negative evaluation_latency is rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                threshold=0.7,
                evaluation_latency=-1,
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
            judge_input_tokens=150,
            judge_output_tokens=50,
            embedding_tokens=5,
        )

        assert judge_score.judge_id == "gpt-4o-mini"
        assert judge_score.score == 0.85
        assert judge_score.reason == "Response is accurate and relevant"
        assert judge_score.judge_input_tokens == 150
        assert judge_score.judge_output_tokens == 50
        assert judge_score.embedding_tokens == 5

    def test_default_values(self) -> None:
        """Test JudgeScore has correct default values."""
        judge_score = JudgeScore(judge_id="gpt-4o")

        assert judge_score.score is None
        assert judge_score.reason == ""
        assert judge_score.judge_input_tokens == 0
        assert judge_score.judge_output_tokens == 0
        assert judge_score.embedding_tokens == 0

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
            JudgeScore(judge_id="gpt-4o", judge_input_tokens=-1)

        with pytest.raises(ValidationError):
            JudgeScore(judge_id="gpt-4o", judge_output_tokens=-1)

        with pytest.raises(ValidationError):
            JudgeScore(judge_id="gpt-4o", embedding_tokens=-1)


class TestMetricResultJudgeScores:
    """Tests for MetricResult.judge_scores field."""

    def test_metric_result_with_judge_scores(self) -> None:
        """Test MetricResult with judge_scores populated."""
        judge_scores = [
            JudgeScore(judge_id="gpt-4o-mini", score=0.8, judge_input_tokens=100),
            JudgeScore(judge_id="gpt-4o", score=0.9, judge_input_tokens=120),
            JudgeScore(judge_id="gemini-flash", score=0.85, judge_input_tokens=110),
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


class TestEvaluationDataAgentFields:
    """Tests for agent-related fields on EvaluationData."""

    def test_agent_fields_default_to_none(self) -> None:
        """Backward compat: existing YAML without agent fields still works."""
        data = EvaluationData(
            conversation_group_id="cg1",
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        assert data.agent is None
        assert data.agent_config is None

    def test_agent_list_accepted(self) -> None:
        """Agent list is accepted."""
        data = EvaluationData(
            conversation_group_id="cg1",
            agent=["openshift_agentic_lightspeed"],
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        assert data.agent == ["openshift_agentic_lightspeed"]

    def test_agent_string_normalized_to_list(self) -> None:
        """String agent is auto-converted to single-element list."""
        data = EvaluationData.model_validate(
            {
                "conversation_group_id": "cg1",
                "agent": "ols_api",
                "turns": [{"turn_id": "t1", "query": "Q"}],
            }
        )
        assert data.agent == ["ols_api"]

    def test_duplicate_agents_deduplicated(self) -> None:
        """Duplicate agent names are silently removed."""
        data = EvaluationData(
            conversation_group_id="cg1",
            agent=["model_a", "model_b", "model_a"],
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        assert data.agent == ["model_a", "model_b"]

    def test_agent_multi_list_accepted(self) -> None:
        """Multiple agents in list accepted."""
        data = EvaluationData(
            conversation_group_id="cg1",
            agent=["model_a", "model_b"],
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        assert data.agent == ["model_a", "model_b"]

    def test_agent_config_keyed_by_agent(self) -> None:
        """Agent config keyed by agent name is accepted."""
        data = EvaluationData(
            conversation_group_id="cg1",
            agent=["model_a"],
            agent_config={"model_a": {"timeout": 1200}},
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        assert data.agent_config == {"model_a": {"timeout": 1200}}

    def test_agent_config_flat_accepted(self) -> None:
        """Flat agent config (applies to all agents) is accepted."""
        data = EvaluationData(
            conversation_group_id="cg1",
            agent=["model_a"],
            agent_config={"timeout": 1200, "namespace": "custom"},
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        assert data.agent_config == {"timeout": 1200, "namespace": "custom"}

    def test_empty_agent_list_rejected(self) -> None:
        """Empty agent list is rejected."""
        with pytest.raises(ValidationError):
            EvaluationData(
                conversation_group_id="cg1",
                agent=[],
                turns=[TurnData(turn_id="t1", query="Q")],
            )

    def test_empty_string_agent_rejected(self) -> None:
        """Empty string agent is rejected."""
        with pytest.raises(ValidationError):
            EvaluationData.model_validate(
                {
                    "conversation_group_id": "cg1",
                    "agent": "",
                    "turns": [{"turn_id": "t1", "query": "Q"}],
                }
            )

    def test_padded_agent_name_stripped(self) -> None:
        """Whitespace-padded agent name is stripped."""
        data = EvaluationData.model_validate(
            {
                "conversation_group_id": "cg1",
                "agent": " ols_api ",
                "turns": [{"turn_id": "t1", "query": "Q"}],
            }
        )
        assert data.agent == ["ols_api"]


class TestConversationMetadata:
    """Tests for ConversationMetadata model."""

    def test_defaults_to_none_and_accepts_all_fields(self) -> None:
        """Empty construction defaults to None; all fields are assignable."""
        assert ConversationMetadata().scenario_category is None

        meta = ConversationMetadata(
            scenario_category="Core/Happy path",
            use_case="RAG",
            interaction_type="Multi-turn",
            topic="networking",
            jtbd_reference="JTBD-001",
            complexity="Complex",
            data_source="Human-written",
            human_verified=True,
            verified_by="jane.doe",
            persona="admin",
            additional_metadata={"region": "EMEA"},
        )
        assert meta.scenario_category == "Core/Happy path"
        assert meta.use_case == "RAG"
        assert meta.complexity == "Complex"
        assert meta.human_verified is True
        assert meta.persona == "admin"
        assert meta.additional_metadata == {"region": "EMEA"}

    def test_extra_fields_forbidden(self) -> None:
        """ConversationMetadata rejects unknown fields."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ConversationMetadata.model_validate({"unknown": "x"})

    def test_optional_on_evaluation_data(self) -> None:
        """EvaluationData accepts metadata and defaults to None without it."""
        data = EvaluationData(
            conversation_group_id="cg1",
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        assert data.metadata is None

        data = EvaluationData(
            conversation_group_id="cg1",
            turns=[TurnData(turn_id="t1", query="Q")],
            metadata=ConversationMetadata(scenario_category="Edge Case"),
        )
        assert data.metadata is not None
        assert data.metadata.scenario_category == "Edge Case"

    def test_serialization_round_trip(self) -> None:
        """ConversationMetadata survives model_dump / model_validate."""
        data = EvaluationData(
            conversation_group_id="cg1",
            turns=[TurnData(turn_id="t1", query="Q")],
            metadata=ConversationMetadata(
                use_case="Agent/Tools", additional_metadata={"region": "EMEA"}
            ),
        )
        restored = EvaluationData.model_validate(data.model_dump(mode="json"))
        assert restored.metadata is not None
        assert restored.metadata.use_case == "Agent/Tools"


class TestDatasetMetadata:
    """Tests for DatasetMetadata model."""

    def test_defaults_to_none_and_accepts_all_fields(self) -> None:
        """Empty construction defaults to None; all fields are assignable."""
        assert DatasetMetadata().team_product is None

        meta = DatasetMetadata(
            description="OLS evaluation dataset",
            jtbd_source="JTBD-2025-Q2",
            team_product="Team LEADS / OLS",
            dataset_version="1.2.0",
            pii_confirmed_removed=True,
            generation_tools=["SDG-hub", "Ragas"],
            llms_used=["gpt-4o", "granite-3.2"],
            last_updated="2025-06-15",
            additional_metadata={"env": "staging"},
        )
        assert meta.description == "OLS evaluation dataset"
        assert meta.jtbd_source == "JTBD-2025-Q2"
        assert meta.team_product == "Team LEADS / OLS"
        assert meta.pii_confirmed_removed is True
        assert meta.generation_tools == ["SDG-hub", "Ragas"]
        assert meta.last_updated == "2025-06-15"

    def test_extra_fields_forbidden(self) -> None:
        """DatasetMetadata rejects unknown fields."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DatasetMetadata.model_validate({"bad_field": 123})

    def test_serialization_round_trip(self) -> None:
        """DatasetMetadata survives model_dump / model_validate."""
        meta = DatasetMetadata(
            team_product="OLS",
            pii_confirmed_removed=True,
            additional_metadata={"grade": "Gold"},
        )
        restored = DatasetMetadata.model_validate(meta.model_dump(mode="json"))
        assert restored.team_product == "OLS"
        assert restored.additional_metadata == {"grade": "Gold"}

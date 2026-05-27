"""Tests for custom metrics module."""

# pylint: disable=protected-access

import json

from pytest_mock import MockerFixture

from lightspeed_evaluation.core.metrics.custom.custom import CustomMetrics
from lightspeed_evaluation.core.metrics.manager import MetricLevel
from lightspeed_evaluation.core.models import EvaluationScope, TurnData
from lightspeed_evaluation.core.system.exceptions import LLMError


class TestCustomMetricsToolEval:
    """Test CustomMetrics tool_eval functionality."""

    def test_evaluate_tool_calls_with_none_tool_calls(
        self, mocker: MockerFixture
    ) -> None:
        """Test that None tool_calls is handled correctly."""
        # Mock LLM manager
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {"parameters": {}}

        custom_metrics = CustomMetrics(mock_llm_manager)

        turn_data = TurnData(
            turn_id="test_turn",
            query="hello",
            tool_calls=None,
            expected_tool_calls=[
                [[{"tool_name": "some_tool", "arguments": {}}]],  # Primary
                [],  # Alternative: no tools (should match None -> [])
            ],
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = custom_metrics.evaluate("tool_eval", None, scope)

        assert score == 1.0
        assert "Alternative 2 matched" in reason

    def test_default_config_uses_full_ordered(self, mocker: MockerFixture) -> None:
        """Test that default config uses full_match=True and ordered=True."""
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {"parameters": {}}

        custom_metrics = CustomMetrics(mock_llm_manager)

        turn_data = TurnData(
            turn_id="test_turn",
            query="hello",
            tool_calls=[
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ],
            expected_tool_calls=[
                [
                    [{"tool_name": "tool1", "arguments": {}}],
                    [{"tool_name": "tool2", "arguments": {}}],
                ]
            ],
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = custom_metrics.evaluate("tool_eval", None, scope)

        assert score == 1.0
        assert "full" in reason
        assert "ordered" in reason

    def test_config_ordered_false_from_metadata(self, mocker: MockerFixture) -> None:
        """Test that ordered=False is read from turn_metrics_metadata."""
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {"parameters": {}}

        custom_metrics = CustomMetrics(mock_llm_manager)

        turn_data = TurnData(
            turn_id="test_turn",
            query="hello",
            tool_calls=[
                [{"tool_name": "tool2", "arguments": {}}],  # Reversed order
                [{"tool_name": "tool1", "arguments": {}}],
            ],
            expected_tool_calls=[
                [
                    [{"tool_name": "tool1", "arguments": {}}],
                    [{"tool_name": "tool2", "arguments": {}}],
                ]
            ],
            turn_metrics_metadata={"custom:tool_eval": {"ordered": False}},
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = custom_metrics.evaluate("tool_eval", None, scope)

        assert score == 1.0
        assert "unordered" in reason

    def test_config_match_partial_from_metadata(self, mocker: MockerFixture) -> None:
        """Test that full_match=False is read from turn_metrics_metadata."""
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {"parameters": {}}

        custom_metrics = CustomMetrics(mock_llm_manager)

        turn_data = TurnData(
            turn_id="test_turn",
            query="hello",
            tool_calls=[
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],  # Extra tool
            ],
            expected_tool_calls=[
                [[{"tool_name": "tool1", "arguments": {}}]]  # Only expect tool1
            ],
            turn_metrics_metadata={"custom:tool_eval": {"full_match": False}},
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = custom_metrics.evaluate("tool_eval", None, scope)

        assert score == 1.0
        assert "partial" in reason
        assert "1/1 expected matched" in reason

    def test_config_from_system_defaults_via_metric_manager(
        self, mocker: MockerFixture
    ) -> None:
        """Test that config is read from system.yaml via MetricManager."""
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {"parameters": {}}

        # Mock MetricManager to return system defaults
        mock_metric_manager = mocker.Mock()
        mock_metric_manager.get_metric_metadata.return_value = {
            "ordered": False,
            "full_match": False,
        }

        custom_metrics = CustomMetrics(mock_llm_manager, mock_metric_manager)

        # TurnData WITHOUT turn_metrics_metadata - should use system defaults
        turn_data = TurnData(
            turn_id="test_turn",
            query="hello",
            tool_calls=[
                [{"tool_name": "tool2", "arguments": {}}],  # Wrong order
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "extra_tool", "arguments": {}}],  # Extra
            ],
            expected_tool_calls=[
                [
                    [{"tool_name": "tool1", "arguments": {}}],
                    [{"tool_name": "tool2", "arguments": {}}],
                ]
            ],
            # No turn_metrics_metadata - should use system defaults via MetricManager
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = custom_metrics.evaluate("tool_eval", None, scope)

        # Verify MetricManager was called with correct arguments
        mock_metric_manager.get_metric_metadata.assert_called_once_with(
            metric_identifier="custom:tool_eval",
            level=MetricLevel.TURN,
            conv_data=None,
            turn_data=turn_data,
        )

        # Should succeed with system defaults (partial + unordered)
        assert score == 1.0
        assert "partial" in reason
        assert "unordered" in reason


def _make_custom_metrics(mocker: MockerFixture) -> CustomMetrics:
    """Create a CustomMetrics instance with mocked LLM manager."""
    mock_llm_manager = mocker.Mock()
    mock_llm_manager.get_model_name.return_value = "test-model"
    mock_llm_manager.get_llm_params.return_value = {"parameters": {}}
    return CustomMetrics(mock_llm_manager)


def _make_scope(turn_data: TurnData, is_conversation: bool = False) -> EvaluationScope:
    """Create an EvaluationScope for a turn."""
    return EvaluationScope(
        turn_idx=0, turn_data=turn_data, is_conversation=is_conversation
    )


METRIC_NAME = "proposal_evaluation_correctness"

_LLM_RESPONSE_ALL_DIMS = json.dumps(
    {
        "reasoning": (
            "Diagnosis was accurate. Execution addressed root cause. "
            "Verification confirmed fix."
        ),
        "diagnosis": 0.9,
        "execution": 0.8,
        "verification": 0.7,
        "average": 0.80,
    }
)

_LLM_RESPONSE_NO_VERIFICATION = json.dumps(
    {
        "reasoning": "Diagnosis correct. Execution appropriate but no verification.",
        "diagnosis": 0.9,
        "execution": 0.8,
        "verification": None,
        "average": 0.85,
    }
)


class TestParseProposalEvalResponse:
    """Test _parse_proposal_eval_response parser."""

    def test_all_dimensions(self, mocker: MockerFixture) -> None:
        """Test parsing response with all three dimensions scored."""
        cm = _make_custom_metrics(mocker)
        score, detail = cm._parse_proposal_eval_response(_LLM_RESPONSE_ALL_DIMS)

        assert score == 0.80
        assert "diagnosis=0.90" in detail
        assert "execution=0.80" in detail
        assert "verification=0.70" in detail
        assert "avg=0.80" in detail
        assert "Diagnosis was accurate" in detail

    def test_dimension_na(self, mocker: MockerFixture) -> None:
        """Test parsing response with N/A dimension."""
        cm = _make_custom_metrics(mocker)
        score, detail = cm._parse_proposal_eval_response(_LLM_RESPONSE_NO_VERIFICATION)

        assert score == 0.85
        assert "verification=N/A" in detail
        assert "diagnosis=0.90" in detail

    def test_fallback_average_from_sub_scores(self, mocker: MockerFixture) -> None:
        """Test average is computed from sub-scores when average key is missing."""
        cm = _make_custom_metrics(mocker)
        response = json.dumps(
            {
                "reasoning": "ok",
                "diagnosis": 0.9,
                "execution": 0.7,
                "verification": None,
            }
        )

        score, detail = cm._parse_proposal_eval_response(response)

        assert score is not None
        assert abs(score - 0.80) < 0.01
        assert "avg=0.80" in detail

    def test_invalid_json_returns_none(self, mocker: MockerFixture) -> None:
        """Test that invalid JSON response returns None score."""
        cm = _make_custom_metrics(mocker)
        score, detail = cm._parse_proposal_eval_response("I cannot evaluate this.")

        assert score is None
        assert "Invalid JSON" in detail


class TestProposalEvaluationCorrectness:
    """Test custom:proposal_evaluation_correctness metric."""

    def test_returns_score_from_llm(self, mocker: MockerFixture) -> None:
        """Test successful LLM evaluation returns parsed score."""
        cm = _make_custom_metrics(mocker)
        mocker.patch.object(cm, "_call_llm", return_value=_LLM_RESPONSE_ALL_DIMS)

        turn = TurnData(
            turn_id="t1",
            query="Fix pod crash",
            response="## Request\n\nFix pod crash\n\n## Analysis\n...",
            expected_outcome="Root cause: OOMKilled. Remediation: increase memory limit.",
        )
        score, reason = cm.evaluate(METRIC_NAME, None, _make_scope(turn))

        assert score == 0.80
        assert "diagnosis=0.90" in reason
        assert "avg=0.80" in reason

    def test_conversation_level_returns_none(self, mocker: MockerFixture) -> None:
        """Test conversation-level scope returns None score."""
        cm = _make_custom_metrics(mocker)
        turn = TurnData(turn_id="t1", query="q", response="r")
        scope = _make_scope(turn, is_conversation=True)

        score, reason = cm.evaluate(METRIC_NAME, None, scope)

        assert score is None
        assert "turn-level" in reason

    def test_missing_response_returns_none(self, mocker: MockerFixture) -> None:
        """Test missing response returns None score."""
        cm = _make_custom_metrics(mocker)
        turn = TurnData(turn_id="t1", query="Fix pod crash")

        score, reason = cm.evaluate(METRIC_NAME, None, _make_scope(turn))

        assert score is None
        assert "response" in reason.lower()

    def test_missing_expected_outcome_returns_none(self, mocker: MockerFixture) -> None:
        """Test missing expected_outcome returns None score."""
        cm = _make_custom_metrics(mocker)
        turn = TurnData(
            turn_id="t1",
            query="Fix pod crash",
            response="## Analysis\nDiagnosis: memory too low",
        )

        score, reason = cm.evaluate(METRIC_NAME, None, _make_scope(turn))

        assert score is None
        assert "expected outcome" in reason.lower()

    def test_none_turn_data_returns_none(self, mocker: MockerFixture) -> None:
        """Test None turn data returns None score."""
        cm = _make_custom_metrics(mocker)
        scope = EvaluationScope(turn_idx=0, turn_data=None, is_conversation=False)

        score, reason = cm.evaluate(METRIC_NAME, None, scope)

        assert score is None
        assert "TurnData" in reason

    def test_unparseable_llm_response(self, mocker: MockerFixture) -> None:
        """Test unparseable LLM response returns None score."""
        cm = _make_custom_metrics(mocker)
        mocker.patch.object(cm, "_call_llm", return_value="I cannot evaluate this.")

        turn = TurnData(
            turn_id="t1",
            query="q",
            response="## Request\nq",
            expected_outcome="Expected outcome",
        )

        score, reason = cm.evaluate(METRIC_NAME, None, _make_scope(turn))

        assert score is None
        assert "parse" in reason.lower() or "Could not" in reason

    def test_llm_error_handled(self, mocker: MockerFixture) -> None:
        """Test LLM error is caught and returns None score."""
        cm = _make_custom_metrics(mocker)
        mocker.patch.object(cm, "_call_llm", side_effect=LLMError("timeout"))

        turn = TurnData(
            turn_id="t1",
            query="q",
            response="## Request\nq",
            expected_outcome="Expected outcome",
        )

        score, reason = cm.evaluate(METRIC_NAME, None, _make_scope(turn))

        assert score is None
        assert "timeout" in reason

    def test_prompt_contains_query_response_and_expected(
        self, mocker: MockerFixture
    ) -> None:
        """Test that the LLM prompt includes query, response, and expected."""
        cm = _make_custom_metrics(mocker)
        call_spy = mocker.patch.object(
            cm, "_call_llm", return_value=_LLM_RESPONSE_ALL_DIMS
        )

        turn = TurnData(
            turn_id="t1",
            query="Fix OOMKilled pod",
            response="## Analysis\nDiagnosis: memory too low",
            expected_outcome="Root cause: OOMKilled. Increase memory limit to 512Mi.",
        )
        cm.evaluate(METRIC_NAME, None, _make_scope(turn))

        prompt: str = call_spy.call_args[0][0]
        assert "Fix OOMKilled pod" in prompt
        assert "Diagnosis: memory too low" in prompt
        assert "Increase memory limit to 512Mi" in prompt

    def test_prompt_contains_sre_persona(self, mocker: MockerFixture) -> None:
        """Test that the prompt includes the SRE persona."""
        cm = _make_custom_metrics(mocker)
        call_spy = mocker.patch.object(
            cm, "_call_llm", return_value=_LLM_RESPONSE_ALL_DIMS
        )

        turn = TurnData(
            turn_id="t1",
            query="q",
            response="r",
            expected_outcome="e",
        )
        cm.evaluate(METRIC_NAME, None, _make_scope(turn))

        prompt: str = call_spy.call_args[0][0]
        assert "senior Site Reliability Engineer" in prompt

"""Tests for custom metrics module."""

from pytest_mock import MockerFixture
from lightspeed_evaluation.core.metrics.custom.custom import CustomMetrics
from lightspeed_evaluation.core.metrics.manager import MetricLevel
from lightspeed_evaluation.core.models import EvaluationScope, TurnData


class TestCustomMetricsToolEval:
    """Test CustomMetrics tool_eval functionality."""

    def test_evaluate_tool_calls_with_none_tool_calls(
        self, mocker: MockerFixture
    ) -> None:
        """Test that None tool_calls is handled correctly."""
        # Mock LLM manager
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {}

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
        mock_llm_manager.get_llm_params.return_value = {}

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
        mock_llm_manager.get_llm_params.return_value = {}

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
        mock_llm_manager.get_llm_params.return_value = {}

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
        assert "1/1 matched" in reason

    def test_config_from_system_defaults_via_metric_manager(
        self, mocker: MockerFixture
    ) -> None:
        """Test that config is read from system.yaml via MetricManager."""
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {}

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

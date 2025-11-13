"""Tests for custom metrics module."""

from lightspeed_evaluation.core.metrics.custom.custom import CustomMetrics
from lightspeed_evaluation.core.models import TurnData


class TestCustomMetrics:
    """Test CustomMetrics class."""

    def test_evaluate_tool_calls_with_none_tool_calls(self, mocker):
        """Test that None tool_calls is handled correctly."""
        # Mock LLM manager
        mock_llm_manager = mocker.Mock()
        mock_llm_manager.get_model_name.return_value = "test-model"
        mock_llm_manager.get_llm_params.return_value = {}

        custom_metrics = CustomMetrics(mock_llm_manager)

        # TurnData with tool_calls = None
        turn_data = TurnData(
            turn_id="test_turn",
            query="hello",
            tool_calls=None,
            expected_tool_calls=[
                [
                    [{"tool_name": "some_tool", "arguments": {}}]
                ],  # Primary: expects tool
                [],  # Alternative: no tools (should match None -> [])
            ],
        )

        # Should match the empty alternative without error
        score, reason = custom_metrics._evaluate_tool_calls(
            _conv_data=None, _turn_idx=0, turn_data=turn_data, is_conversation=False
        )

        assert score == 1.0
        assert "Alternative 2 matched" in reason

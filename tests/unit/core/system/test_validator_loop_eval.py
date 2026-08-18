# pylint: disable=protected-access

"""Unit tests for custom:loop_eval validator requirements."""

from lightspeed_evaluation.core.models import EvaluationData, TurnData
from lightspeed_evaluation.core.system.validator import (
    DataValidator,
    check_metric_required_data,
)


class TestLoopEvalRequiredData:
    """Runtime required-data checks for custom:loop_eval."""

    def test_empty_tool_calls_is_ok(self) -> None:
        """Empty tool_calls is valid for loop_eval (no loops)."""
        turn = TurnData(turn_id="1", query="Q", tool_calls=[])
        ok, msg = check_metric_required_data(turn, "custom:loop_eval")
        assert ok is True
        assert msg == ""

    def test_missing_tool_calls_is_error(self) -> None:
        """None tool_calls is missing data for loop_eval."""
        turn = TurnData(turn_id="1", query="Q", tool_calls=None)
        ok, msg = check_metric_required_data(turn, "custom:loop_eval")
        assert ok is False
        assert "tool_calls" in msg
        assert "missing" in msg


class TestLoopEvalDataValidation:
    """Load-time validation for custom:loop_eval required fields."""

    def test_empty_tool_calls_is_valid(self) -> None:
        """Empty tool_calls is valid for loop_eval at load time."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            tool_calls=[],
            turn_metrics=["custom:loop_eval"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is True
        assert not validator.validation_errors

    def test_missing_tool_calls_fails_when_api_disabled(self) -> None:
        """Missing tool_calls fails loop_eval validation when API is disabled."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            tool_calls=None,
            turn_metrics=["custom:loop_eval"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any(
            "tool_calls" in error.lower() for error in validator.validation_errors
        )

    def test_conversation_level_empty_tool_calls_is_valid(self) -> None:
        """Empty tool_calls is valid for conversation-level loop_eval."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(turn_id="1", query="Query", tool_calls=[])
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn],
            conversation_metrics=["custom:loop_eval"],
        )

        result = validator._validate_evaluation_data([conv_data])

        assert result is True
        assert not validator.validation_errors
        assert not conv_data.is_metric_invalid("custom:loop_eval")

    def test_conversation_level_missing_tool_calls_fails_when_api_disabled(
        self,
    ) -> None:
        """Conversation-level loop_eval requires tool_calls on every turn."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(turn_id="1", query="Query", tool_calls=None)
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn],
            conversation_metrics=["custom:loop_eval"],
        )

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any(
            "tool_calls" in error.lower() for error in validator.validation_errors
        )
        assert conv_data.is_metric_invalid("custom:loop_eval")

    def test_conversation_level_mixed_none_fails_when_api_disabled(self) -> None:
        """A single missing tool_calls turn fails conversation-level loop_eval."""
        validator = DataValidator(api_enabled=False)

        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[
                TurnData(turn_id="1", query="Query", tool_calls=[]),
                TurnData(turn_id="2", query="Query", tool_calls=None),
            ],
            conversation_metrics=["custom:loop_eval"],
        )

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any("TurnData 2" in error for error in validator.validation_errors)
        assert conv_data.is_metric_invalid("custom:loop_eval")

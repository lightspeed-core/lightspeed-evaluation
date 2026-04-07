"""Unit tests for the programmatic API module."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.api import (
    evaluate,
    evaluate_conversation,
    evaluate_conversation_with_summary,
    evaluate_turn,
    evaluate_turn_with_summary,
    evaluate_with_summary,
)
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.models.summary import EvaluationSummary


class TestEvaluate:
    """Unit tests for the evaluate() function."""

    def test_evaluate_success(self, mocker: MockerFixture) -> None:
        """Test successful evaluation returns results."""
        mock_loader = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.api.ConfigLoader"
        ).from_config.return_value = mock_loader

        mock_pipeline = mocker.Mock()
        mock_results = [mocker.Mock(spec=EvaluationResult)]
        mock_pipeline.run_evaluation.return_value = mock_results
        mocker.patch(
            "lightspeed_evaluation.api.EvaluationPipeline",
            return_value=mock_pipeline,
        )

        config = SystemConfig()
        data = [mocker.Mock(spec=EvaluationData)]

        results = evaluate(config, data)

        assert results == mock_results
        mock_pipeline.run_evaluation.assert_called_once_with(data)
        mock_pipeline.close.assert_called_once()

    def test_evaluate_empty_data(self) -> None:
        """Test that evaluate returns empty list for empty data."""
        config = SystemConfig()
        results = evaluate(config, [])

        assert not results

    def test_evaluate_pipeline_close_on_error(self, mocker: MockerFixture) -> None:
        """Test that pipeline.close() is called even on error."""
        mock_loader = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.api.ConfigLoader"
        ).from_config.return_value = mock_loader

        mock_pipeline = mocker.Mock()
        mock_pipeline.run_evaluation.side_effect = RuntimeError("boom")
        mocker.patch(
            "lightspeed_evaluation.api.EvaluationPipeline",
            return_value=mock_pipeline,
        )

        config = SystemConfig()
        data = [mocker.Mock(spec=EvaluationData)]

        with pytest.raises(RuntimeError, match="boom"):
            evaluate(config, data)

        mock_pipeline.close.assert_called_once()

    def test_evaluate_output_dir_passthrough(self, mocker: MockerFixture) -> None:
        """Test that output_dir is passed to EvaluationPipeline."""
        mock_loader = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.api.ConfigLoader"
        ).from_config.return_value = mock_loader

        mock_pipeline = mocker.Mock()
        mock_pipeline.run_evaluation.return_value = []
        mock_pipeline_class = mocker.patch(
            "lightspeed_evaluation.api.EvaluationPipeline",
            return_value=mock_pipeline,
        )

        config = SystemConfig()
        data = [mocker.Mock(spec=EvaluationData)]

        evaluate(config, data, output_dir="/custom/output")

        mock_pipeline_class.assert_called_once_with(mock_loader, "/custom/output")


class TestEvaluateConversation:
    """Unit tests for the evaluate_conversation() function."""

    def test_delegates_to_evaluate_with_list(self, mocker: MockerFixture) -> None:
        """Test that evaluate_conversation wraps data in a list and calls evaluate."""
        mock_evaluate = mocker.patch("lightspeed_evaluation.api.evaluate")
        mock_evaluate.return_value = [mocker.Mock(spec=EvaluationResult)]

        config = SystemConfig()
        data = mocker.Mock(spec=EvaluationData)

        results = evaluate_conversation(config, data, output_dir="/output")

        mock_evaluate.assert_called_once_with(config, [data], output_dir="/output")
        assert results == mock_evaluate.return_value

    def test_delegates_without_output_dir(self, mocker: MockerFixture) -> None:
        """Test that evaluate_conversation delegates without output_dir."""
        mock_evaluate = mocker.patch("lightspeed_evaluation.api.evaluate")
        mock_evaluate.return_value = []

        config = SystemConfig()
        data = mocker.Mock(spec=EvaluationData)

        evaluate_conversation(config, data)

        mock_evaluate.assert_called_once_with(config, [data], output_dir=None)


class TestEvaluateTurn:
    """Unit tests for the evaluate_turn() function."""

    def test_wraps_turn_in_evaluation_data(self, mocker: MockerFixture) -> None:
        """Test that evaluate_turn wraps a turn in EvaluationData and calls evaluate."""
        mock_evaluate = mocker.patch("lightspeed_evaluation.api.evaluate")
        mock_evaluate.return_value = [mocker.Mock(spec=EvaluationResult)]

        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="What is OCP?")

        results = evaluate_turn(config, turn)

        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args
        data_list = call_args[0][1]
        assert len(data_list) == 1
        assert data_list[0].conversation_group_id == "programmatic_eval"
        assert data_list[0].turns[0].turn_id == "t1"
        assert results == mock_evaluate.return_value

    def test_metrics_override(self, mocker: MockerFixture) -> None:
        """Test that metrics parameter overrides turn_metrics."""
        mock_evaluate = mocker.patch("lightspeed_evaluation.api.evaluate")
        mock_evaluate.return_value = []

        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="What is OCP?")

        evaluate_turn(config, turn, metrics=["ragas:faithfulness"])

        call_args = mock_evaluate.call_args
        data_list = call_args[0][1]
        assert data_list[0].turns[0].turn_metrics == ["ragas:faithfulness"]

    def test_default_conversation_group_id(self, mocker: MockerFixture) -> None:
        """Test the default conversation_group_id."""
        mock_evaluate = mocker.patch("lightspeed_evaluation.api.evaluate")
        mock_evaluate.return_value = []

        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="hello")

        evaluate_turn(config, turn)

        call_args = mock_evaluate.call_args
        data_list = call_args[0][1]
        assert data_list[0].conversation_group_id == "programmatic_eval"

    def test_custom_conversation_group_id(self, mocker: MockerFixture) -> None:
        """Test that custom conversation_group_id is passed through."""
        mock_evaluate = mocker.patch("lightspeed_evaluation.api.evaluate")
        mock_evaluate.return_value = []

        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="hello")

        evaluate_turn(config, turn, conversation_group_id="my_custom_group")

        call_args = mock_evaluate.call_args
        data_list = call_args[0][1]
        assert data_list[0].conversation_group_id == "my_custom_group"

    def test_does_not_mutate_original_turn(self, mocker: MockerFixture) -> None:
        """Test that the original turn is not mutated when metrics are provided."""
        mock_evaluate = mocker.patch("lightspeed_evaluation.api.evaluate")
        mock_evaluate.return_value = []

        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="hello", turn_metrics=None)

        evaluate_turn(config, turn, metrics=["ragas:faithfulness"])

        # Original turn should be unchanged
        assert turn.turn_metrics is None

    def test_invalid_metrics_format_rejected(self) -> None:
        """Test that metrics with invalid format are rejected by validators."""
        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="hello")

        with pytest.raises(ValueError, match="must be in format"):
            evaluate_turn(config, turn, metrics=["bad_format"])


class TestEvaluateWithSummary:
    """Unit tests for the evaluate_with_summary() function."""

    def test_returns_evaluation_summary(self, mocker: MockerFixture) -> None:
        """Test that evaluate_with_summary returns an EvaluationSummary."""
        mock_loader = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.api.ConfigLoader"
        ).from_config.return_value = mock_loader

        mock_pipeline = mocker.Mock()
        mock_results = [
            EvaluationResult(
                conversation_group_id="c1",
                metric_identifier="m:1",
                result="PASS",
                score=0.9,
                threshold=0.7,
            )
        ]
        mock_pipeline.run_evaluation.return_value = mock_results
        mocker.patch(
            "lightspeed_evaluation.api.EvaluationPipeline",
            return_value=mock_pipeline,
        )

        config = SystemConfig()
        data = [
            EvaluationData(
                conversation_group_id="c1",
                turns=[TurnData(turn_id="t1", query="hello")],
            )
        ]

        summary = evaluate_with_summary(config, data)

        assert isinstance(summary, EvaluationSummary)
        assert summary.overall.total == 1
        assert summary.overall.passed == 1
        assert len(summary.results) == 1

    def test_empty_data_returns_empty_summary(self) -> None:
        """Test that empty data returns a summary with zero results."""
        config = SystemConfig()
        summary = evaluate_with_summary(config, [])

        assert isinstance(summary, EvaluationSummary)
        assert summary.overall.total == 0
        assert not summary.results


class TestSummaryConversationAndTurn:
    """Unit tests for conversation and turn summary functions."""

    def test_conversation_with_summary_delegates(self, mocker: MockerFixture) -> None:
        """Test that it delegates to evaluate_with_summary with list-wrapped data."""
        mock_eval = mocker.patch("lightspeed_evaluation.api.evaluate_with_summary")
        mock_eval.return_value = mocker.Mock(spec=EvaluationSummary)

        config = SystemConfig()
        data = mocker.Mock(spec=EvaluationData)

        result = evaluate_conversation_with_summary(config, data, output_dir="/out")

        mock_eval.assert_called_once_with(
            config,
            [data],
            output_dir="/out",
            compute_confidence_intervals=False,
        )
        assert result == mock_eval.return_value

    def test_turn_with_summary_wraps_turn(self, mocker: MockerFixture) -> None:
        """Test that evaluate_turn_with_summary wraps turn and returns summary."""
        mock_eval = mocker.patch("lightspeed_evaluation.api.evaluate_with_summary")
        mock_eval.return_value = mocker.Mock(spec=EvaluationSummary)

        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="What is OCP?")

        result = evaluate_turn_with_summary(config, turn)

        mock_eval.assert_called_once()
        call_args = mock_eval.call_args
        data_list = call_args[0][1]
        assert len(data_list) == 1
        assert data_list[0].conversation_group_id == "programmatic_eval"
        assert call_args.kwargs.get("compute_confidence_intervals") is False
        assert result == mock_eval.return_value

    def test_turn_with_summary_metrics_override(self, mocker: MockerFixture) -> None:
        """Test that metrics parameter overrides turn_metrics."""
        mock_eval = mocker.patch("lightspeed_evaluation.api.evaluate_with_summary")
        mock_eval.return_value = mocker.Mock(spec=EvaluationSummary)

        config = SystemConfig()
        turn = TurnData(turn_id="t1", query="hello")

        evaluate_turn_with_summary(config, turn, metrics=["ragas:faithfulness"])

        call_args = mock_eval.call_args
        data_list = call_args[0][1]
        assert data_list[0].turns[0].turn_metrics == ["ragas:faithfulness"]

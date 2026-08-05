# pylint: disable=unused-argument,protected-access

"""Unit tests for uncaught exception handling in evaluate_metric calls."""

import logging

from _pytest.logging import LogCaptureFixture

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    TurnData,
)
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
from lightspeed_evaluation.pipeline.evaluation.processor import (
    ConversationProcessor,
)


class TestEvaluateMetricExceptionHandling:
    """Unit tests for uncaught exception handling in evaluate_metric calls."""

    def test_evaluate_turn_catches_exception_and_returns_error(
        self,
        processor: ConversationProcessor,
        mock_metrics_evaluator: MetricsEvaluator,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test _evaluate_turn catches unexpected exceptions and returns ERROR result."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        mock_metrics_evaluator.evaluate_metric.side_effect = RuntimeError(
            "The output is incomplete due to a max_tokens length limit."
        )

        with caplog.at_level(logging.ERROR):
            results = processor._evaluate_turn(
                conv_data, 0, turn_data, ["ragas:faithfulness"]
            )

        assert len(results) == 1
        assert results[0].result == "ERROR"
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[0].conversation_group_id == "test_conv"
        assert results[0].turn_id == "1"
        assert "RuntimeError" in results[0].reason
        assert "max_tokens" in results[0].reason
        assert "ragas:faithfulness evaluation failed" in caplog.text
        assert "test_conv" in caplog.text
        assert "turn 0" in caplog.text

    def test_evaluate_turn_continues_after_exception(
        self,
        processor: ConversationProcessor,
        mock_metrics_evaluator: MetricsEvaluator,
    ) -> None:
        """Test _evaluate_turn continues evaluating remaining metrics after one throws."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        mock_metrics_evaluator.evaluate_metric.side_effect = [
            RuntimeError("boom"),
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="1",
                metric_identifier="geval:correctness",
                result="PASS",
                score=0.9,
                reason="Good",
                threshold=0.8,
            ),
        ]

        results = processor._evaluate_turn(
            conv_data, 0, turn_data, ["ragas:faithfulness", "geval:correctness"]
        )

        assert len(results) == 2
        assert results[0].result == "ERROR"
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[0].conversation_group_id == "test_conv"
        assert results[0].turn_id == "1"
        assert "RuntimeError" in results[0].reason
        assert results[1].result == "PASS"
        assert results[1].metric_identifier == "geval:correctness"

    def test_evaluate_conversation_catches_exception_and_returns_error(
        self,
        processor: ConversationProcessor,
        mock_metrics_evaluator: MetricsEvaluator,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test _evaluate_conversation catches unexpected exceptions and returns ERROR."""
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[TurnData(turn_id="1", query="Q", response="R")],
        )

        mock_metrics_evaluator.evaluate_metric.side_effect = RuntimeError(
            "unexpected failure"
        )

        with caplog.at_level(logging.ERROR):
            results = processor._evaluate_conversation(
                conv_data, ["deepeval:conversation_completeness"]
            )

        assert len(results) == 1
        assert results[0].result == "ERROR"
        assert results[0].metric_identifier == "deepeval:conversation_completeness"
        assert results[0].conversation_group_id == "test_conv"
        assert "RuntimeError" in results[0].reason
        assert "unexpected failure" in results[0].reason
        assert "conversation_completeness evaluation failed" in caplog.text
        assert "test_conv" in caplog.text

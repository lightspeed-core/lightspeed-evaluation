"""Unit tests for pipeline evaluation processor module."""

import logging

import pytest

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationResult,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation.processor import (
    ConversationProcessor,
    ProcessorComponents,
)


@pytest.fixture
def config_loader(mocker):
    """Create a mock config loader with system config."""
    loader = mocker.Mock(spec=ConfigLoader)

    config = SystemConfig()
    config.default_turn_metrics_metadata = {
        "ragas:faithfulness": {"threshold": 0.7, "default": True},
        "custom:answer_correctness": {"threshold": 0.8, "default": False},
    }
    config.default_conversation_metrics_metadata = {
        "deepeval:conversation_completeness": {"threshold": 0.6, "default": True},
    }
    config.api.enabled = False

    loader.system_config = config
    return loader


@pytest.fixture
def mock_metrics_evaluator(mocker):
    """Create a mock metrics evaluator."""
    from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator

    evaluator = mocker.Mock(spec=MetricsEvaluator)

    def evaluate_metric(request):
        """Mock evaluate_metric that returns a result based on metric."""
        return EvaluationResult(
            conversation_group_id=request.conv_data.conversation_group_id,
            turn_id=request.turn_id,
            metric_identifier=request.metric_identifier,
            result="PASS",
            score=0.85,
            reason="Test evaluation",
            threshold=0.7,
        )

    evaluator.evaluate_metric.side_effect = evaluate_metric
    return evaluator


@pytest.fixture
def mock_api_amender(mocker):
    """Create a mock API data amender."""
    from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender

    amender = mocker.Mock(spec=APIDataAmender)
    return amender


@pytest.fixture
def mock_error_handler(mocker):
    """Create a mock error handler."""
    from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler

    handler = mocker.Mock(spec=EvaluationErrorHandler)
    return handler


@pytest.fixture
def mock_metric_manager(mocker):
    """Create a mock metric manager."""
    from lightspeed_evaluation.core.metrics.manager import MetricManager

    manager = mocker.Mock(spec=MetricManager)
    return manager


@pytest.fixture
def mock_script_manager(mocker):
    """Create a mock script execution manager."""
    from lightspeed_evaluation.core.script import ScriptExecutionManager

    manager = mocker.Mock(spec=ScriptExecutionManager)
    return manager


@pytest.fixture
def processor_components(
    mock_metrics_evaluator,
    mock_api_amender,
    mock_error_handler,
    mock_metric_manager,
    mock_script_manager,
):
    """Create processor components fixture."""
    return ProcessorComponents(
        metrics_evaluator=mock_metrics_evaluator,
        api_amender=mock_api_amender,
        error_handler=mock_error_handler,
        metric_manager=mock_metric_manager,
        script_manager=mock_script_manager,
    )


@pytest.fixture
def processor(config_loader, processor_components):
    """Create ConversationProcessor instance."""
    return ConversationProcessor(config_loader, processor_components)


class TestConversationProcessorEvaluateTurn:
    """Unit tests for ConversationProcessor._evaluate_turn method."""

    def test_evaluate_turn_with_valid_metrics(self, processor, mock_metrics_evaluator):
        """Test _evaluate_turn with all valid metrics."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Context"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        turn_metrics = ["ragas:faithfulness", "custom:answer_correctness"]

        results = processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Should evaluate both metrics
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)

        # Verify evaluate_metric was called twice
        assert mock_metrics_evaluator.evaluate_metric.call_count == 2

        # Verify the requests
        calls = mock_metrics_evaluator.evaluate_metric.call_args_list
        assert calls[0][0][0].metric_identifier == "ragas:faithfulness"
        assert calls[1][0][0].metric_identifier == "custom:answer_correctness"

    def test_evaluate_turn_with_invalid_metric(
        self, processor, mock_metrics_evaluator, caplog
    ):
        """Test _evaluate_turn with an invalid metric - should skip and log error."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Context"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        # Mark one metric as invalid
        turn_data.add_invalid_metric("ragas:faithfulness")

        turn_metrics = ["ragas:faithfulness", "custom:answer_correctness"]

        with caplog.at_level(logging.ERROR):
            results = processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Should only evaluate the valid metric
        assert len(results) == 1
        assert results[0].metric_identifier == "custom:answer_correctness"

        # Verify evaluate_metric was called only once (for valid metric)
        assert mock_metrics_evaluator.evaluate_metric.call_count == 1

        # Verify error was logged for invalid metric
        assert "Invalid turn metric 'ragas:faithfulness'" in caplog.text
        assert "check Validation Errors" in caplog.text

    def test_evaluate_turn_with_all_invalid_metrics(
        self, processor, mock_metrics_evaluator, caplog
    ):
        """Test _evaluate_turn with all metrics invalid - should return empty results."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Context"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        # Mark all metrics as invalid
        turn_data.add_invalid_metric("ragas:faithfulness")
        turn_data.add_invalid_metric("custom:answer_correctness")

        turn_metrics = ["ragas:faithfulness", "custom:answer_correctness"]

        with caplog.at_level(logging.ERROR):
            results = processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Should return empty results
        assert len(results) == 0

        # Verify evaluate_metric was never called
        assert mock_metrics_evaluator.evaluate_metric.call_count == 0

        # Verify errors were logged for both invalid metrics
        assert "Invalid turn metric 'ragas:faithfulness'" in caplog.text
        assert "Invalid turn metric 'custom:answer_correctness'" in caplog.text

    def test_evaluate_turn_with_mixed_valid_invalid_metrics(
        self, processor, mock_metrics_evaluator, caplog
    ):
        """Test _evaluate_turn with mix of valid and invalid metrics."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Context"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        # Mark middle metric as invalid
        turn_data.add_invalid_metric("custom:answer_correctness")

        turn_metrics = [
            "ragas:faithfulness",
            "custom:answer_correctness",
            "ragas:context_recall",
        ]

        with caplog.at_level(logging.ERROR):
            results = processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Should evaluate only the two valid metrics
        assert len(results) == 2
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[1].metric_identifier == "ragas:context_recall"

        # Verify evaluate_metric was called twice
        assert mock_metrics_evaluator.evaluate_metric.call_count == 2

        # Verify error was logged for invalid metric
        assert "Invalid turn metric 'custom:answer_correctness'" in caplog.text

    def test_evaluate_turn_with_empty_metrics(self, processor, mock_metrics_evaluator):
        """Test _evaluate_turn with empty metrics list."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        turn_metrics = []

        results = processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Should return empty results
        assert len(results) == 0

        # Verify evaluate_metric was never called
        assert mock_metrics_evaluator.evaluate_metric.call_count == 0

    def test_evaluate_turn_creates_correct_request(
        self, processor, mock_metrics_evaluator
    ):
        """Test _evaluate_turn creates correct EvaluationRequest."""
        turn_data = TurnData(
            turn_id="turn_123",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Context"],
        )
        conv_data = EvaluationData(conversation_group_id="conv_456", turns=[turn_data])

        turn_metrics = ["ragas:faithfulness"]

        processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Verify the request structure
        assert mock_metrics_evaluator.evaluate_metric.call_count == 1
        call_args = mock_metrics_evaluator.evaluate_metric.call_args[0][0]

        assert isinstance(call_args, EvaluationRequest)
        assert call_args.conv_data.conversation_group_id == "conv_456"
        assert call_args.metric_identifier == "ragas:faithfulness"
        assert call_args.turn_id == "turn_123"
        assert call_args.turn_idx == 0

    def test_evaluate_turn_handles_evaluator_returning_none(
        self, processor, mock_metrics_evaluator
    ):
        """Test _evaluate_turn handles when evaluator returns None."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        # Reset the side_effect and make evaluator return None
        mock_metrics_evaluator.evaluate_metric.side_effect = None
        mock_metrics_evaluator.evaluate_metric.return_value = None

        turn_metrics = ["ragas:faithfulness"]

        results = processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Should return empty results when evaluator returns None
        assert len(results) == 0

        # Verify evaluate_metric was still called
        assert mock_metrics_evaluator.evaluate_metric.call_count == 1

    def test_evaluate_turn_multiple_turns_correct_index(
        self, processor, mock_metrics_evaluator
    ):
        """Test _evaluate_turn uses correct turn index."""
        turn_data_1 = TurnData(turn_id="1", query="Q1", response="R1")
        turn_data_2 = TurnData(turn_id="2", query="Q2", response="R2")
        turn_data_3 = TurnData(turn_id="3", query="Q3", response="R3")

        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn_data_1, turn_data_2, turn_data_3],
        )

        turn_metrics = ["ragas:faithfulness"]

        # Evaluate second turn (index 1)
        processor._evaluate_turn(conv_data, 1, turn_data_2, turn_metrics)

        # Verify correct turn index
        call_args = mock_metrics_evaluator.evaluate_metric.call_args[0][0]
        assert call_args.turn_idx == 1
        assert call_args.turn_id == "2"

    def test_evaluate_turn_preserves_metric_order(
        self, processor, mock_metrics_evaluator
    ):
        """Test _evaluate_turn evaluates metrics in the order provided."""
        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        turn_metrics = [
            "custom:answer_correctness",
            "ragas:faithfulness",
            "ragas:context_recall",
        ]

        processor._evaluate_turn(conv_data, 0, turn_data, turn_metrics)

        # Verify metrics were evaluated in order
        assert mock_metrics_evaluator.evaluate_metric.call_count == 3
        calls = mock_metrics_evaluator.evaluate_metric.call_args_list

        assert calls[0][0][0].metric_identifier == "custom:answer_correctness"
        assert calls[1][0][0].metric_identifier == "ragas:faithfulness"
        assert calls[2][0][0].metric_identifier == "ragas:context_recall"

    def test_is_metric_invalid_functionality(self):
        """Test TurnData.is_metric_invalid and add_invalid_metric methods."""
        turn_data = TurnData(turn_id="1", query="Q", response="R")

        # Initially no metrics are invalid
        assert not turn_data.is_metric_invalid("ragas:faithfulness")
        assert not turn_data.is_metric_invalid("custom:answer_correctness")

        # Add an invalid metric
        turn_data.add_invalid_metric("ragas:faithfulness")

        # Now it should be marked as invalid
        assert turn_data.is_metric_invalid("ragas:faithfulness")
        # But other metrics should still be valid
        assert not turn_data.is_metric_invalid("custom:answer_correctness")

        # Add another invalid metric
        turn_data.add_invalid_metric("custom:answer_correctness")

        # Both should now be invalid
        assert turn_data.is_metric_invalid("ragas:faithfulness")
        assert turn_data.is_metric_invalid("custom:answer_correctness")

        # Adding same metric again should not cause issues
        turn_data.add_invalid_metric("ragas:faithfulness")
        assert turn_data.is_metric_invalid("ragas:faithfulness")

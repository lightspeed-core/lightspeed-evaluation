"""Unit tests for ConversationProcessor."""

import pytest

from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationResult,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.script import (
    ScriptExecutionError,
    ScriptExecutionManager,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
from lightspeed_evaluation.pipeline.evaluation.processor import (
    ConversationProcessor,
    ProcessorComponents,
)


@pytest.fixture
def mock_config_loader(mocker):
    """Create a mock config loader."""
    loader = mocker.Mock(spec=ConfigLoader)
    config = SystemConfig()
    config.api.enabled = False
    loader.system_config = config
    return loader


@pytest.fixture
def processor_components(mocker):
    """Create processor components."""
    metrics_evaluator = mocker.Mock(spec=MetricsEvaluator)
    api_amender = mocker.Mock(spec=APIDataAmender)
    error_handler = mocker.Mock(spec=EvaluationErrorHandler)
    metric_manager = mocker.Mock(spec=MetricManager)
    script_manager = mocker.Mock(spec=ScriptExecutionManager)

    # Default behavior for metric resolution
    metric_manager.resolve_metrics.return_value = ["ragas:faithfulness"]

    return ProcessorComponents(
        metrics_evaluator=metrics_evaluator,
        api_amender=api_amender,
        error_handler=error_handler,
        metric_manager=metric_manager,
        script_manager=script_manager,
    )


@pytest.fixture
def sample_conv_data():
    """Create sample conversation data."""
    turn1 = TurnData(
        turn_id="turn1",
        query="What is Python?",
        response="Python is a programming language.",
        contexts=["Context"],
        turn_metrics=["ragas:faithfulness"],
    )
    return EvaluationData(
        conversation_group_id="conv1",
        turns=[turn1],
    )


class TestConversationProcessor:
    """Unit tests for ConversationProcessor."""

    def test_initialization(self, mock_config_loader, processor_components):
        """Test processor initialization."""
        processor = ConversationProcessor(mock_config_loader, processor_components)

        assert processor.config_loader == mock_config_loader
        assert processor.config == mock_config_loader.system_config
        assert processor.components == processor_components

    def test_process_conversation_skips_when_no_metrics(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test processing skips when no metrics specified."""
        # Mock metric manager to return empty lists
        processor_components.metric_manager.resolve_metrics.return_value = []

        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = processor.process_conversation(sample_conv_data)

        assert len(results) == 0

    def test_process_conversation_turn_metrics(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test processing with turn-level metrics."""
        from lightspeed_evaluation.core.models import EvaluationResult

        # Configure metric manager to return turn metrics and empty conversation metrics
        def resolve_side_effect(metrics, level):
            from lightspeed_evaluation.core.metrics.manager import MetricLevel

            if level == MetricLevel.TURN:
                return ["ragas:faithfulness"]
            return []

        processor_components.metric_manager.resolve_metrics.side_effect = (
            resolve_side_effect
        )

        # Mock metrics evaluation - one result per metric per turn
        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
        )
        processor_components.metrics_evaluator.evaluate_metric.return_value = (
            mock_result
        )

        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = processor.process_conversation(sample_conv_data)

        # Each turn with metrics will produce results
        assert len(results) > 0
        assert all(r.result == "PASS" for r in results)

    def test_process_conversation_conversation_metrics(
        self, mock_config_loader, processor_components, mocker
    ):
        """Test processing with conversation-level metrics."""
        from lightspeed_evaluation.core.models import EvaluationResult

        turn1 = TurnData(turn_id="turn1", query="Q", response="R")
        conv_data = EvaluationData(
            conversation_group_id="conv1",
            turns=[turn1],
            conversation_metrics=["deepeval:conversation_completeness"],
        )

        # Mock metric resolution
        def resolve_side_effect(metrics, level):
            from lightspeed_evaluation.core.metrics.manager import MetricLevel

            if level == MetricLevel.TURN:
                return []
            return ["deepeval:conversation_completeness"]

        processor_components.metric_manager.resolve_metrics.side_effect = (
            resolve_side_effect
        )

        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id=None,
            metric_identifier="deepeval:conversation_completeness",
            score=0.75,
            result="PASS",
            threshold=0.6,
            reason="Complete",
        )
        processor_components.metrics_evaluator.evaluate_metric.return_value = (
            mock_result
        )

        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = processor.process_conversation(conv_data)

        assert len(results) == 1
        assert results[0].turn_id is None  # Conversation-level

    def test_process_conversation_with_setup_script_success(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test processing with successful setup script."""
        from lightspeed_evaluation.core.models import EvaluationResult

        sample_conv_data.setup_script = "setup.sh"
        mock_config_loader.system_config.api.enabled = True

        # Configure metric manager to return turn metrics and empty conversation metrics
        def resolve_side_effect(metrics, level):
            from lightspeed_evaluation.core.metrics.manager import MetricLevel

            if level == MetricLevel.TURN:
                return ["ragas:faithfulness"]
            return []

        processor_components.metric_manager.resolve_metrics.side_effect = (
            resolve_side_effect
        )

        processor_components.script_manager.run_script.return_value = True
        # Mock API amender to return tuple
        processor_components.api_amender.amend_single_turn.return_value = (
            None,
            "conv_123",
        )

        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
        )
        processor_components.metrics_evaluator.evaluate_metric.return_value = (
            mock_result
        )

        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = processor.process_conversation(sample_conv_data)

        processor_components.script_manager.run_script.assert_called()
        assert len(results) > 0

    def test_process_conversation_with_setup_script_failure(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test processing handles setup script failure."""
        sample_conv_data.setup_script = "setup.sh"
        mock_config_loader.system_config.api.enabled = True

        processor_components.script_manager.run_script.side_effect = (
            ScriptExecutionError("Script failed")
        )
        processor_components.error_handler.mark_all_metrics_as_error.return_value = []

        processor = ConversationProcessor(mock_config_loader, processor_components)
        processor.process_conversation(sample_conv_data)

        processor_components.error_handler.mark_all_metrics_as_error.assert_called_once()

    def test_process_conversation_with_cleanup_script(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test cleanup script is always called."""
        from lightspeed_evaluation.core.models import EvaluationResult

        sample_conv_data.cleanup_script = "cleanup.sh"
        mock_config_loader.system_config.api.enabled = True

        # Configure metric manager to return turn metrics and empty conversation metrics
        def resolve_side_effect(metrics, level):
            from lightspeed_evaluation.core.metrics.manager import MetricLevel

            if level == MetricLevel.TURN:
                return ["ragas:faithfulness"]
            return []

        processor_components.metric_manager.resolve_metrics.side_effect = (
            resolve_side_effect
        )

        processor_components.script_manager.run_script.return_value = True
        # Mock API amender to return tuple
        processor_components.api_amender.amend_single_turn.return_value = (
            None,
            "conv_123",
        )

        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
        )
        processor_components.metrics_evaluator.evaluate_metric.return_value = (
            mock_result
        )

        processor = ConversationProcessor(mock_config_loader, processor_components)
        processor.process_conversation(sample_conv_data)

        # Verify cleanup was called
        calls = processor_components.script_manager.run_script.call_args_list
        assert any("cleanup.sh" in str(call) for call in calls)

    def test_process_conversation_with_api_amendment(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test API amendment during turn processing."""
        from lightspeed_evaluation.core.models import EvaluationResult

        mock_config_loader.system_config.api.enabled = True

        # Configure metric manager to return turn metrics and empty conversation metrics
        def resolve_side_effect(metrics, level):
            from lightspeed_evaluation.core.metrics.manager import MetricLevel

            if level == MetricLevel.TURN:
                return ["ragas:faithfulness"]
            return []

        processor_components.metric_manager.resolve_metrics.side_effect = (
            resolve_side_effect
        )

        # Mock API amender
        processor_components.api_amender.amend_single_turn.return_value = (
            None,
            "conv_123",
        )

        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
        )
        processor_components.metrics_evaluator.evaluate_metric.return_value = (
            mock_result
        )

        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = processor.process_conversation(sample_conv_data)

        processor_components.api_amender.amend_single_turn.assert_called_once()
        assert len(results) > 0

    def test_process_conversation_with_api_error_cascade(
        self, mock_config_loader, processor_components, mocker
    ):
        """Test API error causes cascade failure."""
        mock_config_loader.system_config.api.enabled = True

        # Create multi-turn conversation
        turn1 = TurnData(
            turn_id="turn1",
            query="Q1",
            response="R1",
            turn_metrics=["ragas:faithfulness"],
        )
        turn2 = TurnData(
            turn_id="turn2",
            query="Q2",
            response="R2",
            turn_metrics=["ragas:faithfulness"],
        )
        conv_data = EvaluationData(
            conversation_group_id="conv1",
            turns=[turn1, turn2],
        )

        # Mock API error on first turn
        processor_components.api_amender.amend_single_turn.return_value = (
            "API Error",
            None,
        )
        processor_components.error_handler.mark_turn_metrics_as_error.return_value = []
        processor_components.error_handler.mark_cascade_error.return_value = []

        processor = ConversationProcessor(mock_config_loader, processor_components)
        processor.process_conversation(conv_data)

        # Verify cascade error handling was triggered
        processor_components.error_handler.mark_cascade_error.assert_called_once()

    def test_evaluate_turn(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test _evaluate_turn method."""
        from lightspeed_evaluation.core.models import EvaluationResult

        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
        )
        processor_components.metrics_evaluator.evaluate_metric.return_value = (
            mock_result
        )

        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = processor._evaluate_turn(
            sample_conv_data, 0, sample_conv_data.turns[0], ["ragas:faithfulness"]
        )

        assert len(results) == 1
        assert results[0].result == "PASS"

    def test_evaluate_conversation(
        self, mock_config_loader, processor_components, sample_conv_data, mocker
    ):
        """Test _evaluate_conversation method."""
        from lightspeed_evaluation.core.models import EvaluationResult

        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id=None,
            metric_identifier="deepeval:conversation_completeness",
            score=0.75,
            result="PASS",
            threshold=0.6,
            reason="Complete",
        )
        processor_components.metrics_evaluator.evaluate_metric.return_value = (
            mock_result
        )

        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = processor._evaluate_conversation(
            sample_conv_data, ["deepeval:conversation_completeness"]
        )

        assert len(results) == 1
        assert results[0].turn_id is None

    def test_run_setup_script_skips_when_api_disabled(
        self, mock_config_loader, processor_components, sample_conv_data
    ):
        """Test setup script is skipped when API disabled."""
        sample_conv_data.setup_script = "setup.sh"
        mock_config_loader.system_config.api.enabled = False

        processor = ConversationProcessor(mock_config_loader, processor_components)
        error = processor._run_setup_script(sample_conv_data)

        assert error is None
        processor_components.script_manager.run_script.assert_not_called()

    def test_run_cleanup_script_skips_when_api_disabled(
        self, mock_config_loader, processor_components, sample_conv_data
    ):
        """Test cleanup script is skipped when API disabled."""
        sample_conv_data.cleanup_script = "cleanup.sh"
        mock_config_loader.system_config.api.enabled = False

        processor = ConversationProcessor(mock_config_loader, processor_components)
        processor._run_cleanup_script(sample_conv_data)

        processor_components.script_manager.run_script.assert_not_called()

    def test_run_cleanup_script_logs_warning_on_failure(
        self, mock_config_loader, processor_components, sample_conv_data
    ):
        """Test cleanup script failure is logged as warning."""
        sample_conv_data.cleanup_script = "cleanup.sh"
        mock_config_loader.system_config.api.enabled = True

        processor_components.script_manager.run_script.return_value = False

        processor = ConversationProcessor(mock_config_loader, processor_components)
        # Should not raise, just log warning
        processor._run_cleanup_script(sample_conv_data)

    def test_get_metrics_summary(
        self, mock_config_loader, processor_components, sample_conv_data
    ):
        """Test get_metrics_summary method."""
        processor_components.metric_manager.count_metrics_for_conversation.return_value = {
            "turn_metrics": 2,
            "conversation_metrics": 1,
        }

        processor = ConversationProcessor(mock_config_loader, processor_components)
        summary = processor.get_metrics_summary(sample_conv_data)

        assert summary["turn_metrics"] == 2
        assert summary["conversation_metrics"] == 1


# Fixtures for TestConversationProcessorEvaluateTurn
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
    amender = mocker.Mock(spec=APIDataAmender)
    return amender


@pytest.fixture
def mock_error_handler(mocker):
    """Create a mock error handler."""
    handler = mocker.Mock(spec=EvaluationErrorHandler)

    # Configure create_error_result to return a proper EvaluationResult
    def create_error_result_side_effect(
        conv_id, metric_id, reason, *, turn_id=None, query=""
    ):
        return EvaluationResult(
            conversation_group_id=conv_id,
            turn_id=turn_id,
            metric_identifier=metric_id,
            result="ERROR",
            reason=reason,
            query=query,
        )

    handler.create_error_result.side_effect = create_error_result_side_effect
    return handler


@pytest.fixture
def mock_metric_manager(mocker):
    """Create a mock metric manager."""
    manager = mocker.Mock(spec=MetricManager)
    return manager


@pytest.fixture
def mock_script_manager(mocker):
    """Create a mock script execution manager."""
    manager = mocker.Mock(spec=ScriptExecutionManager)
    return manager


@pytest.fixture
def processor_components_pr(
    mock_metrics_evaluator,
    mock_api_amender,
    mock_error_handler,
    mock_metric_manager,
    mock_script_manager,
):
    """Create processor components fixture for PR tests."""
    return ProcessorComponents(
        metrics_evaluator=mock_metrics_evaluator,
        api_amender=mock_api_amender,
        error_handler=mock_error_handler,
        metric_manager=mock_metric_manager,
        script_manager=mock_script_manager,
    )


@pytest.fixture
def processor(config_loader, processor_components_pr):
    """Create ConversationProcessor instance for PR tests."""
    return ConversationProcessor(config_loader, processor_components_pr)


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
        """Test _evaluate_turn with an invalid metric - creates ERROR result and logs error."""
        import logging

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

        # Should get 2 results: 1 ERROR for invalid metric, 1 PASS for valid metric
        assert len(results) == 2

        # First result should be ERROR for invalid metric
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[0].result == "ERROR"

        # Second result should be PASS for valid metric
        assert results[1].metric_identifier == "custom:answer_correctness"
        assert results[1].result == "PASS"

        # Verify evaluate_metric was called only once (for valid metric)
        assert mock_metrics_evaluator.evaluate_metric.call_count == 1

        # Verify error was logged for invalid metric
        assert "Invalid turn metric 'ragas:faithfulness'" in caplog.text
        assert "check Validation Errors" in caplog.text

    def test_evaluate_turn_with_all_invalid_metrics(
        self, processor, mock_metrics_evaluator, caplog
    ):
        """Test _evaluate_turn with all metrics invalid - returns ERROR results."""
        import logging

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

        # Should return ERROR results for both invalid metrics
        assert len(results) == 2
        assert all(r.result == "ERROR" for r in results)
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[1].metric_identifier == "custom:answer_correctness"

        # Verify evaluate_metric was never called
        assert mock_metrics_evaluator.evaluate_metric.call_count == 0

        # Verify errors were logged for both invalid metrics
        assert "Invalid turn metric 'ragas:faithfulness'" in caplog.text
        assert "Invalid turn metric 'custom:answer_correctness'" in caplog.text

    def test_evaluate_turn_with_mixed_valid_invalid_metrics(
        self, processor, mock_metrics_evaluator, caplog
    ):
        """Test _evaluate_turn with mix of valid and invalid metrics."""
        import logging

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

        # Should get 3 results: 2 valid metrics (PASS) and 1 invalid metric (ERROR)
        assert len(results) == 3
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[0].result == "PASS"
        assert results[1].metric_identifier == "custom:answer_correctness"
        assert results[1].result == "ERROR"
        assert results[2].metric_identifier == "ragas:context_recall"
        assert results[2].result == "PASS"

        # Verify evaluate_metric was called twice (for valid metrics only)
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


class TestSkipOnFailure:
    """Unit tests for skip_on_failure feature."""

    @pytest.fixture
    def multi_turn_conv_data(self):
        """Create conversation data with multiple turns."""
        turns = [
            TurnData(
                turn_id=f"turn{i}",
                query=f"Q{i}",
                response=f"R{i}",
                turn_metrics=["custom:answer_correctness"],
            )
            for i in range(1, 4)
        ]
        return EvaluationData(
            conversation_group_id="multi_turn_conv",
            turns=turns,
            conversation_metrics=["deepeval:conversation_completeness"],
        )

    @pytest.fixture
    def config_loader_factory(self, mocker):
        """Factory to create config loader with configurable skip_on_failure."""

        def _create(skip_on_failure: bool):
            loader = mocker.Mock(spec=ConfigLoader)
            config = SystemConfig()
            config.api.enabled = False
            config.core.skip_on_failure = skip_on_failure
            loader.system_config = config
            return loader

        return _create

    @pytest.mark.parametrize(
        "system_skip,conv_skip,expected",
        [
            (True, None, True),  # System enabled, no override
            (False, None, False),  # System disabled, no override
            (False, True, True),  # System disabled, conv enables
            (True, False, False),  # System enabled, conv disables
        ],
    )
    def test_is_skip_on_failure_enabled(
        self,
        config_loader_factory,
        processor_components,
        system_skip,
        conv_skip,
        expected,
    ):
        """Test skip_on_failure resolution from system config and conversation override."""
        conv_data = EvaluationData(
            conversation_group_id="test",
            turns=[TurnData(turn_id="1", query="Q")],
            skip_on_failure=conv_skip,
        )
        processor = ConversationProcessor(
            config_loader_factory(system_skip), processor_components
        )
        assert processor._is_skip_on_failure_enabled(conv_data) is expected

    @pytest.mark.parametrize(
        "results_status,expected",
        [
            (["PASS", "FAIL"], True),  # Has FAIL
            (["PASS", "ERROR"], True),  # Has ERROR
            (["PASS", "PASS"], False),  # All PASS
        ],
    )
    def test_has_failure(
        self, mock_config_loader, processor_components, results_status, expected
    ):
        """Test _has_failure detection for FAIL and ERROR results."""
        processor = ConversationProcessor(mock_config_loader, processor_components)
        results = [
            EvaluationResult(
                conversation_group_id="test", metric_identifier=f"m{i}", result=status
            )
            for i, status in enumerate(results_status)
        ]
        assert processor._has_failure(results) is expected

    @pytest.mark.parametrize("skip_enabled,expect_skip", [(True, True), (False, False)])
    def test_skip_on_failure_behavior(
        self,
        config_loader_factory,
        processor_components,
        multi_turn_conv_data,
        skip_enabled,
        expect_skip,
    ):
        """Test skip_on_failure skips remaining turns when enabled, continues when disabled."""
        # Configure metric manager
        processor_components.metric_manager.resolve_metrics.side_effect = [
            ["custom:answer_correctness"],  # turn1
            ["custom:answer_correctness"],  # turn2
            ["custom:answer_correctness"],  # turn3
            ["deepeval:conversation_completeness"],  # conversation level
        ]

        if expect_skip:
            # When skip enabled: turn1 PASS, turn2 FAIL, then skip
            processor_components.metrics_evaluator.evaluate_metric.side_effect = [
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    turn_id="turn1",
                    metric_identifier="custom:answer_correctness",
                    result="PASS",
                ),
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    turn_id="turn2",
                    metric_identifier="custom:answer_correctness",
                    result="FAIL",
                ),
            ]
            processor_components.error_handler.mark_cascade_skipped.return_value = [
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    turn_id="turn3",
                    metric_identifier="custom:answer_correctness",
                    result="SKIPPED",
                ),
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    metric_identifier="deepeval:conversation_completeness",
                    result="SKIPPED",
                ),
            ]
        else:
            # When skip disabled: all turns evaluated
            processor_components.metrics_evaluator.evaluate_metric.side_effect = [
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    turn_id="turn1",
                    metric_identifier="custom:answer_correctness",
                    result="PASS",
                ),
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    turn_id="turn2",
                    metric_identifier="custom:answer_correctness",
                    result="FAIL",
                ),
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    turn_id="turn3",
                    metric_identifier="custom:answer_correctness",
                    result="PASS",
                ),
                EvaluationResult(
                    conversation_group_id="multi_turn_conv",
                    metric_identifier="deepeval:conversation_completeness",
                    result="PASS",
                ),
            ]

        processor = ConversationProcessor(
            config_loader_factory(skip_enabled), processor_components
        )
        results = processor.process_conversation(multi_turn_conv_data)

        assert len(results) == 4
        if expect_skip:
            assert [r.result for r in results] == ["PASS", "FAIL", "SKIPPED", "SKIPPED"]
            processor_components.error_handler.mark_cascade_skipped.assert_called_once()
        else:
            assert [r.result for r in results] == ["PASS", "FAIL", "PASS", "PASS"]
            processor_components.error_handler.mark_cascade_skipped.assert_not_called()

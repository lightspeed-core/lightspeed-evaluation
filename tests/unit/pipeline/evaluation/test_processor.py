"""Unit tests for ConversationProcessor."""

import pytest

from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.models import EvaluationData, SystemConfig, TurnData
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
        processor_components.error_handler.mark_cascade_failure.return_value = []

        processor = ConversationProcessor(mock_config_loader, processor_components)
        processor.process_conversation(conv_data)

        # Verify cascade error handling was triggered
        processor_components.error_handler.mark_cascade_failure.assert_called_once()

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

# pylint: disable=redefined-outer-name

"""Pytest configuration and fixtures for evaluation tests."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import (
    EvaluationData,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.core.models import EvaluationResult, EvaluationRequest
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
from lightspeed_evaluation.pipeline.evaluation.processor import (
    ProcessorComponents,
    ConversationProcessor,
)


@pytest.fixture
def config_loader(mocker: MockerFixture) -> ConfigLoader:
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
    config.api.enabled = True

    loader.system_config = config
    return loader


@pytest.fixture
def mock_metric_manager(mocker: MockerFixture) -> MetricManager:
    """Create a mock metric manager."""
    manager = mocker.Mock(spec=MetricManager)

    def get_threshold(
        metric_id: str,
        _level: str,
        _conv_data: EvaluationData | None = None,
        _turn_data: TurnData | None = None,
    ) -> float:
        thresholds = {
            "ragas:faithfulness": 0.7,
            "custom:answer_correctness": 0.8,
            "deepeval:conversation_completeness": 0.6,
        }
        return thresholds.get(metric_id, 0.5)

    manager.get_effective_threshold.side_effect = get_threshold
    # Mock get_metric_metadata to return None (no metadata) to support iteration
    # in _extract_metadata_for_csv
    manager.get_metric_metadata.return_value = None
    return manager


@pytest.fixture
def mock_script_manager(mocker: MockerFixture) -> ScriptExecutionManager:
    """Create a mock script execution manager."""
    manager = mocker.Mock(spec=ScriptExecutionManager)
    return manager


@pytest.fixture
def mock_config_loader(mocker: MockerFixture) -> ConfigLoader:
    """Create a mock config loader with system config."""
    loader = mocker.Mock(spec=ConfigLoader)

    config = SystemConfig()
    config.api.enabled = False
    config.output.output_dir = "/tmp/test_output"
    config.output.base_filename = "test"
    config.core.max_threads = 2

    loader.system_config = config
    return loader


@pytest.fixture
def sample_evaluation_data() -> list[EvaluationData]:
    """Create sample evaluation data."""
    turn1 = TurnData(
        turn_id="turn1",
        query="What is Python?",
        response="Python is a programming language.",
        contexts=["Python context"],
        turn_metrics=["ragas:faithfulness"],
    )
    conv_data = EvaluationData(
        conversation_group_id="conv1",
        turns=[turn1],
    )
    return [conv_data]


@pytest.fixture
def processor_components(mocker: MockerFixture) -> ProcessorComponents:
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
def sample_conv_data() -> EvaluationData:
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


@pytest.fixture
def mock_metrics_evaluator(mocker: MockerFixture) -> MetricsEvaluator:
    """Create a mock metrics evaluator."""
    evaluator = mocker.Mock(spec=MetricsEvaluator)

    def evaluate_metric(request: EvaluationRequest) -> EvaluationResult:
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
def mock_api_amender(mocker: MockerFixture) -> APIDataAmender:
    """Create a mock API data amender."""
    amender = mocker.Mock(spec=APIDataAmender)
    return amender


@pytest.fixture
def mock_error_handler(mocker: MockerFixture) -> EvaluationErrorHandler:
    """Create a mock error handler."""
    handler = mocker.Mock(spec=EvaluationErrorHandler)

    # Configure create_error_result to return a proper EvaluationResult
    def create_error_result_side_effect(
        conv_id: str,
        metric_id: str,
        reason: str,
        *,
        turn_id: str | None = None,
        query: str = "",
    ) -> EvaluationResult:
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
def processor_components_pr(
    mock_metrics_evaluator: MetricsEvaluator,
    mock_api_amender: APIDataAmender,
    mock_error_handler: EvaluationErrorHandler,
    mock_metric_manager: MetricManager,
    mock_script_manager: ScriptExecutionManager,
) -> ProcessorComponents:
    """Create processor components fixture for PR tests."""
    return ProcessorComponents(
        metrics_evaluator=mock_metrics_evaluator,
        api_amender=mock_api_amender,
        error_handler=mock_error_handler,
        metric_manager=mock_metric_manager,
        script_manager=mock_script_manager,
    )


@pytest.fixture
def processor(
    config_loader: ConfigLoader,
    processor_components_pr: ProcessorComponents,
) -> ConversationProcessor:
    """Create ConversationProcessor instance for PR tests."""
    return ConversationProcessor(config_loader, processor_components_pr)

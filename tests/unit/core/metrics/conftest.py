# pylint: disable=redefined-outer-name

"""Pytest configuration and fixtures for metrics tests."""

import sys
from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
from lightspeed_evaluation.core.metrics.nlp import NLPMetrics
from lightspeed_evaluation.core.metrics.ragas import RagasMetrics
from lightspeed_evaluation.core.models import (
    EmbeddingConfig,
    EvaluationScope,
    SystemConfig,
    TurnData,
)


@pytest.fixture
def system_config() -> SystemConfig:
    """Create a test system config with metrics metadata."""
    config = SystemConfig()

    # Set up test metrics metadata
    config.default_turn_metrics_metadata = {
        "ragas:faithfulness": {
            "threshold": 0.7,
            "default": True,
            "description": "Test",
        },
        "ragas:response_relevancy": {
            "threshold": 0.8,
            "default": False,
            "description": "Test",
        },
        "custom:answer_correctness": {
            "threshold": 0.75,
            "default": True,
            "description": "Test",
        },
    }

    config.default_conversation_metrics_metadata = {
        "deepeval:conversation_completeness": {
            "threshold": 0.6,
            "default": True,
            "description": "Test",
        },
        "deepeval:conversation_relevancy": {
            "threshold": 0.7,
            "default": False,
            "description": "Test",
        },
    }

    return config


@pytest.fixture
def nlp_metrics() -> NLPMetrics:
    """Create NLPMetrics instance."""
    return NLPMetrics()


@pytest.fixture
def sample_turn_data() -> TurnData:
    """Create sample TurnData for testing."""
    return TurnData(
        turn_id="test_turn",
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        expected_response="The capital of France is Paris.",
    )


@pytest.fixture
def sample_scope(
    sample_turn_data: TurnData,
) -> EvaluationScope:
    """Create sample EvaluationScope for turn-level evaluation."""
    return EvaluationScope(
        turn_idx=0,
        turn_data=sample_turn_data,
        is_conversation=False,
    )


@pytest.fixture
def conversation_scope(
    sample_turn_data: TurnData,
) -> EvaluationScope:
    """Create sample EvaluationScope for conversation-level evaluation."""
    return EvaluationScope(
        turn_idx=0,
        turn_data=sample_turn_data,
        is_conversation=True,
    )


@pytest.fixture
def mock_bleu_scorer(mocker: MockerFixture) -> MockerFixture:
    """Mock sacrebleu BLEU with configurable return value.

    Uses sys.modules injection to mock sacrebleu without requiring it to be installed.
    """
    mock_result = mocker.MagicMock()
    mock_result.score = 85.0  # sacrebleu returns 0-100 scale

    mock_scorer_instance = mocker.MagicMock()
    mock_scorer_instance.corpus_score = mocker.MagicMock(return_value=mock_result)

    mock_bleu_class = mocker.MagicMock(return_value=mock_scorer_instance)

    # Create a fake sacrebleu module and inject it into sys.modules
    mock_sacrebleu = mocker.MagicMock()
    mock_sacrebleu.BLEU = mock_bleu_class
    mocker.patch.dict(sys.modules, {"sacrebleu": mock_sacrebleu})

    return mock_scorer_instance


@pytest.fixture
def mock_rouge_scorer(mocker: MockerFixture) -> MockerFixture:
    """Mock RougeScore with configurable return value.

    Returns different scores for precision, recall, fmeasure.
    Uses ragas 0.4+ API with score() method returning MetricResult.
    """
    # Create mock MetricResult objects for each mode
    mock_results = [
        mocker.MagicMock(value=0.95),  # precision
        mocker.MagicMock(value=0.89),  # recall
        mocker.MagicMock(value=0.92),  # fmeasure
    ]

    mock_scorer_instance = mocker.MagicMock()
    mock_scorer_instance.score = mocker.MagicMock(side_effect=mock_results)

    mocker.patch(
        "lightspeed_evaluation.core.metrics.nlp.RougeScore",
        return_value=mock_scorer_instance,
    )
    return mock_scorer_instance


@pytest.fixture
def mock_similarity_scorer(mocker: MockerFixture) -> MockerFixture:
    """Mock NonLLMStringSimilarity with configurable return value.

    Uses ragas 0.4+ API with score() method returning MetricResult.
    """
    mock_result = mocker.MagicMock(value=0.78)

    mock_scorer_instance = mocker.MagicMock()
    mock_scorer_instance.score = mocker.MagicMock(return_value=mock_result)

    mocker.patch(
        "lightspeed_evaluation.core.metrics.nlp.NonLLMStringSimilarity",
        return_value=mock_scorer_instance,
    )
    return mock_scorer_instance


@pytest.fixture
def mock_llm_manager(mocker: MockerFixture) -> Any:
    """Create a mock LLMManager for DeepEval tests."""
    mock_manager = mocker.MagicMock()
    mock_config = mocker.MagicMock()
    mock_config.cache_enabled = False
    mock_manager.get_config.return_value = mock_config
    mock_manager.get_model_name.return_value = "gpt-4"
    mock_manager.get_llm_params.return_value = {"num_retries": 3}
    return mock_manager


@pytest.fixture
def mock_metric_manager(mocker: MockerFixture) -> Any:
    """Create a mock MetricManager."""
    return mocker.MagicMock()


@pytest.fixture
def mock_deepeval_llm_manager(mocker: MockerFixture) -> Any:
    """Create a mock DeepEvalLLMManager."""
    mock_manager = mocker.MagicMock()
    mock_llm = mocker.MagicMock()
    mock_manager.get_llm.return_value = mock_llm
    mock_manager.flush_deepevals_pending_tasks.return_value = None
    return mock_manager


@pytest.fixture
def mock_conv_data(mocker: MockerFixture) -> Any:
    """Create mock conversation data with turns."""
    turn1 = mocker.MagicMock()
    turn1.query = "What is AI?"
    turn1.response = "AI stands for Artificial Intelligence."

    turn2 = mocker.MagicMock()
    turn2.query = "Can you explain more?"
    turn2.response = "AI is the simulation of human intelligence by machines."

    conv_data = mocker.MagicMock()
    conv_data.turns = [turn1, turn2]
    return conv_data


@pytest.fixture
def deepeval_metrics(
    mock_llm_manager: Any,
    mock_metric_manager: Any,
    mock_deepeval_llm_manager: Any,
    mocker: MockerFixture,
) -> DeepEvalMetrics:
    """Create DeepEvalMetrics instance with mocked dependencies."""
    mocker.patch(
        "lightspeed_evaluation.core.metrics.deepeval.DeepEvalLLMManager",
        return_value=mock_deepeval_llm_manager,
    )
    mocker.patch("lightspeed_evaluation.core.metrics.deepeval.GEvalHandler")

    return DeepEvalMetrics(
        llm_manager=mock_llm_manager,
        metric_manager=mock_metric_manager,
    )


@pytest.fixture
def mock_ragas_deps(mocker: MockerFixture) -> dict[str, Any]:
    """Mock all heavy dependencies needed to construct RagasMetrics."""
    mock_llm_manager = mocker.MagicMock()
    mock_llm_config = mocker.MagicMock()
    mock_llm_config.cache_enabled = False
    mock_llm_manager.get_config.return_value = mock_llm_config

    mock_embedding_manager = mocker.MagicMock(spec=EmbeddingManager)
    mock_embedding_manager.config = EmbeddingConfig(
        provider="openai", model="text-embedding-3-small", cache_enabled=False
    )

    mocker.patch("lightspeed_evaluation.core.metrics.ragas.RagasLLMManager")

    return {
        "llm_manager": mock_llm_manager,
        "embedding_manager": mock_embedding_manager,
    }


@pytest.fixture
def ragas_metrics(mock_ragas_deps: dict[str, Any]) -> RagasMetrics:
    """Create RagasMetrics with mocked dependencies."""
    return RagasMetrics(**mock_ragas_deps)


@pytest.fixture
def turn_scope() -> EvaluationScope:
    """Create a turn-level evaluation scope."""
    return EvaluationScope(
        turn_idx=0,
        turn_data=TurnData(
            turn_id="t1",
            query="What is Python?",
            response="A programming language.",
            expected_response="A programming language.",
        ),
        is_conversation=False,
    )

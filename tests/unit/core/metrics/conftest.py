"""Pytest configuration and fixtures for metrics tests."""

import sys

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.metrics.nlp import NLPMetrics
from lightspeed_evaluation.core.models import EvaluationScope, TurnData, SystemConfig


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
def sample_scope(  # pylint: disable=redefined-outer-name
    sample_turn_data: TurnData,
) -> EvaluationScope:
    """Create sample EvaluationScope for turn-level evaluation."""
    return EvaluationScope(
        turn_idx=0,
        turn_data=sample_turn_data,
        is_conversation=False,
    )


@pytest.fixture
def conversation_scope(  # pylint: disable=redefined-outer-name
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
    """
    mock_scorer_instance = mocker.MagicMock()
    # Return scores for precision, recall, fmeasure (called in that order)
    mock_scorer_instance.single_turn_score = mocker.MagicMock(
        side_effect=[0.95, 0.89, 0.92]
    )
    mocker.patch(
        "lightspeed_evaluation.core.metrics.nlp.RougeScore",
        return_value=mock_scorer_instance,
    )
    return mock_scorer_instance


@pytest.fixture
def mock_similarity_scorer(mocker: MockerFixture) -> MockerFixture:
    """Mock NonLLMStringSimilarity with configurable return value."""
    mock_scorer_instance = mocker.MagicMock()
    mock_scorer_instance.single_turn_score = mocker.MagicMock(return_value=0.78)
    mocker.patch(
        "lightspeed_evaluation.core.metrics.nlp.NonLLMStringSimilarity",
        return_value=mock_scorer_instance,
    )
    return mock_scorer_instance

"""Pytest configuration and fixtures for output tests."""

import pytest
from pytest_mock import MockerFixture
from lightspeed_evaluation.core.models import EvaluationResult


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    """Create sample evaluation results."""
    return [
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
            query="What is Python?",
            response="Python is a programming language",
        ),
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn2",
            metric_identifier="ragas:answer_relevancy",
            score=0.60,
            result="FAIL",
            threshold=0.7,
            reason="Low score",
            query="How?",
            response="It works",
        ),
    ]


@pytest.fixture
def sample_results_statistics() -> list[EvaluationResult]:
    """Create sample evaluation results."""
    return [
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="metric1",
            score=0.9,
            result="PASS",
            threshold=0.7,
            reason="Good",
        ),
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn2",
            metric_identifier="metric1",
            score=0.5,
            result="FAIL",
            threshold=0.7,
            reason="Low score",
        ),
        EvaluationResult(
            conversation_group_id="conv2",
            turn_id="turn1",
            metric_identifier="metric2",
            score=0.8,
            result="PASS",
            threshold=0.7,
            reason="Good",
        ),
        EvaluationResult(
            conversation_group_id="conv2",
            turn_id="turn2",
            metric_identifier="metric2",
            score=None,
            result="ERROR",
            threshold=0.7,
            reason="Failed",
        ),
    ]


@pytest.fixture
def mock_system_config(mocker: MockerFixture) -> MockerFixture:
    """Create mock system config."""
    config = mocker.Mock()
    config.output.enabled_outputs = ["csv", "json", "txt"]
    config.output.csv_columns = [
        "conversation_group_id",
        "turn_id",
        "metric_identifier",
        "result",
        "score",
    ]
    config.visualization.enabled_graphs = []
    # Mock model_fields to support iteration in _write_config_params and _build_config_dict
    config.model_fields.keys.return_value = []
    return config

# pylint: disable=redefined-outer-name

"""Pytest configuration and fixtures for script tests."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from script.run_multi_provider_eval import MultiProviderEvaluationRunner


@pytest.fixture
def script_path() -> Path:
    """Return the path to the compare_evaluations.py script."""
    # Test is in tests/script/, script is in project_root/script/
    return Path(__file__).parent.parent.parent / "script" / "compare_evaluations.py"


@pytest.fixture
def sample_evaluation_data() -> tuple[list[dict], list[dict]]:
    """Return sample evaluation data for testing."""
    sample_results1 = [
        {
            "conversation_group_id": "conv1",
            "turn_id": "1",
            "metric_identifier": "ragas:faithfulness",
            "result": "PASS",
            "score": 0.8,
            "threshold": 0.7,
            "execution_time": 1.0,
        },
        {
            "conversation_group_id": "conv1",
            "turn_id": "2",
            "metric_identifier": "ragas:faithfulness",
            "result": "PASS",
            "score": 0.9,
            "threshold": 0.7,
            "execution_time": 1.2,
        },
    ]

    sample_results2 = [
        {
            "conversation_group_id": "conv1",
            "turn_id": "1",
            "metric_identifier": "ragas:faithfulness",
            "result": "PASS",
            "score": 0.85,
            "threshold": 0.7,
            "execution_time": 1.1,
        },
        {
            "conversation_group_id": "conv1",
            "turn_id": "2",
            "metric_identifier": "ragas:faithfulness",
            "result": "FAIL",
            "score": 0.6,
            "threshold": 0.7,
            "execution_time": 1.0,
        },
    ]

    return sample_results1, sample_results2


@pytest.fixture
def temp_config_files(tmp_path: Path) -> dict:
    """Create temporary configuration files for testing."""
    # Create multi_eval_config.yaml
    providers_config = {
        "providers": {
            "openai": {
                "models": ["gpt-4o-mini", "gpt-4-turbo"],
            },
            "watsonx": {
                "models": ["ibm/granite-13b-chat-v2"],
            },
        },
        "settings": {"output_base": str(tmp_path / "eval_output")},
    }
    providers_path = tmp_path / "multi_eval_config.yaml"
    with open(providers_path, "w", encoding="utf-8") as f:
        yaml.dump(providers_config, f)

    # Create system.yaml
    system_config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
        },
        "api": {"enabled": False},
        "output": {"output_dir": "./eval_output"},
    }
    system_path = tmp_path / "system.yaml"
    with open(system_path, "w", encoding="utf-8") as f:
        yaml.dump(system_config, f)

    # Create evaluation_data.yaml
    eval_data = [
        {
            "conversation_group_id": "test_conv",
            "turns": [
                {
                    "turn_id": "turn_1",
                    "query": "Test query",
                    "response": "Test response",
                    "contexts": ["Context 1"],
                    "expected_response": "Expected",
                    "turn_metrics": ["ragas:response_relevancy"],
                }
            ],
        }
    ]
    eval_path = tmp_path / "evaluation_data.yaml"
    with open(eval_path, "w", encoding="utf-8") as f:
        yaml.dump(eval_data, f)

    return {
        "providers_config": providers_path,
        "system_config": system_path,
        "eval_data": eval_path,
        "output_dir": tmp_path / "eval_output",
    }


@pytest.fixture
def runner(
    temp_config_files: dict,
) -> MultiProviderEvaluationRunner:
    """Create a MultiProviderEvaluationRunner instance for testing."""
    return MultiProviderEvaluationRunner(
        providers_config_path=str(temp_config_files["providers_config"]),
        system_config_path=str(temp_config_files["system_config"]),
        eval_data_path=str(temp_config_files["eval_data"]),
    )


@pytest.fixture
def sample_evaluation_summary() -> dict[str, Any]:
    """Create a sample evaluation summary JSON for testing analysis."""
    return {
        "timestamp": "2025-01-01T12:00:00",
        "total_evaluations": 10,
        "summary_stats": {
            "overall": {
                "TOTAL": 10,
                "PASS": 8,
                "FAIL": 2,
                "ERROR": 0,
                "pass_rate": 80.0,  # Percentage format
                "fail_rate": 20.0,
                "error_rate": 0.0,
            },
            "by_metric": {
                "ragas:faithfulness": {
                    "pass": 4,
                    "fail": 0,
                    "error": 0,
                    "pass_rate": 100.0,
                    "fail_rate": 0.0,
                    "error_rate": 0.0,
                    "score_statistics": {
                        "mean": 0.95,
                        "median": 0.95,
                        "std": 0.02,
                        "min": 0.92,
                        "max": 0.98,
                        "count": 4,
                    },
                },
                "ragas:response_relevancy": {
                    "pass": 4,
                    "fail": 2,
                    "error": 0,
                    "pass_rate": 66.67,
                    "fail_rate": 33.33,
                    "error_rate": 0.0,
                    "score_statistics": {
                        "mean": 0.75,
                        "median": 0.78,
                        "std": 0.12,
                        "min": 0.55,
                        "max": 0.88,
                        "count": 6,
                    },
                },
            },
        },
        "results": [
            {
                "conversation_group_id": "conv1",
                "turn_id": "turn1",
                "metric_identifier": "ragas:faithfulness",
                "result": "PASS",
                "score": 0.95,
                "threshold": 0.8,
                "execution_time": 1.0,
            },
            {
                "conversation_group_id": "conv1",
                "turn_id": "turn2",
                "metric_identifier": "ragas:response_relevancy",
                "result": "PASS",
                "score": 0.85,
                "threshold": 0.7,
                "execution_time": 1.2,
            },
        ]
        * 5,  # Repeat to get 10 results
    }

"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from lightspeed_evaluation.core import EvaluationData, LLMConfig, SystemConfig, TurnData
from lightspeed_evaluation.core.llm.manager import LLMManager


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def config_dir():
    """Provide configuration directory."""
    return Path(__file__).parent.parent / "config"


@pytest.fixture
def sample_system_config():
    """Provide a sample SystemConfig for testing."""
    return SystemConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.0,
        llm_max_tokens=512,
        output_dir="./test_output",
        base_filename="test_evaluation",
        include_graphs=True,
    )


@pytest.fixture
def sample_llm_config():
    """Provide a sample LLMConfig for testing."""
    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        timeout=300,
        num_retries=3,
    )


@pytest.fixture
def sample_turn_data():
    """Provide sample TurnData for testing."""
    return TurnData(
        turn_id="1",
        query="What is Python?",
        response="Python is a high-level programming language.",
        contexts=[
            "Python is a programming language created by Guido van Rossum.",
            "Python is widely used for web development, data science, and automation.",
        ],
        expected_response="Python is a high-level programming language used for various applications.",
    )


@pytest.fixture
def sample_evaluation_data(sample_turn_data):
    """Provide sample EvaluationData for testing."""
    # Add turn_metrics to the turn data
    sample_turn_data.turn_metrics = ["ragas:faithfulness", "ragas:response_relevancy"]

    return EvaluationData(
        conversation_group_id="test_conversation",
        description="Test conversation for evaluation",
        conversation_metrics=["deepeval:conversation_completeness"],
        turns=[sample_turn_data],
    )


@pytest.fixture
def mock_llm_manager():
    """Provide a mock LLM manager."""
    manager = MagicMock(spec=LLMManager)
    manager.get_model_name.return_value = "gpt-4o-mini"
    manager.get_litellm_params.return_value = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout": 300,
        "num_retries": 3,
    }
    manager.config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        timeout=300,
        num_retries=3,
    )
    return manager


@pytest.fixture
def temp_config_files():
    """Create temporary configuration files for testing."""
    system_config_data = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": 300,
            "num_retries": 3,
        },
        "environment": {
            "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
            "LITELLM_LOG_LEVEL": "ERROR",
        },
        "logging": {"source_level": "INFO", "package_level": "ERROR"},
        "metrics_metadata": {
            "turn_level": {
                "ragas:faithfulness": {
                    "threshold": 0.8,
                    "description": "How faithful the response is to the provided context",
                    "default": False,
                },
                "ragas:response_relevancy": {
                    "threshold": 0.8,
                },
                "custom:answer_correctness": {
                    "threshold": 0.7,
                },
            },
            "conversation_level": {
                "deepeval:conversation_completeness": {
                    "threshold": 0.7,
                    "description": "How completely the conversation addresses user intentions",
                    "default": False,
                }
            },
        },
        "output": {
            "base_directory": "./test_output",
            "base_filename": "test_evaluation",
            "formats": {"csv": True, "json": True, "txt": True},
            "include_graphs": True,
        },
        "visualization": {"figsize": [12, 8], "dpi": 300},
    }

    eval_data = [
        {
            "conversation_group_id": "test_conv_1",
            "description": "Test conversation 1",
            "conversation_metrics": [],
            "conversation_metrics_metadata": {},
            "turns": [
                {
                    "turn_id": "1",
                    "query": "What is machine learning?",
                    "response": "Machine learning is a subset of AI.",
                    "contexts": ["Machine learning is a method of data analysis."],
                    "expected_response": "Machine learning is a subset of artificial intelligence.",
                    "turn_metrics": ["ragas:faithfulness", "ragas:response_relevancy"],
                    "turn_metrics_metadata": {},
                }
            ],
        },
        {
            "conversation_group_id": "test_conv_2",
            "description": "Test conversation 2",
            "conversation_metrics": ["deepeval:conversation_completeness"],
            "conversation_metrics_metadata": {},
            "turns": [
                {
                    "turn_id": "1",
                    "query": "Explain neural networks",
                    "response": "Neural networks are computing systems inspired by biological neural networks.",
                    "contexts": ["Neural networks consist of interconnected nodes."],
                    "expected_response": "Neural networks are computational models inspired by the human brain.",
                    "turn_metrics": ["custom:answer_correctness"],
                    "turn_metrics_metadata": {},
                },
                {
                    "turn_id": "2",
                    "query": "What are the applications?",
                    "response": "Neural networks are used in image recognition, NLP, and more.",
                    "contexts": [
                        "Applications include computer vision and natural language processing."
                    ],
                    "expected_response": "Applications include computer vision, NLP, and pattern recognition.",
                    "turn_metrics": None,
                    "turn_metrics_metadata": {},
                },
            ],
        },
    ]

    # Create temporary files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as system_file:
        yaml.dump(system_config_data, system_file, default_flow_style=False)
        system_config_path = system_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as eval_file:
        yaml.dump(eval_data, eval_file, default_flow_style=False)
        eval_data_path = eval_file.name

    yield {"system_config": system_config_path, "eval_data": eval_data_path}

    # Cleanup
    os.unlink(system_config_path)
    os.unlink(eval_data_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set required environment variables for testing
    test_env_vars = {
        "OPENAI_API_KEY": "test-api-key-for-testing",
        "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
        "DEEPEVAL_DISABLE_PROGRESS_BAR": "YES",
        "LITELLM_LOG_LEVEL": "ERROR",
    }

    # Store original values
    original_values = {}
    for key, value in test_env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "config: mark test as configuration-related")
    config.addinivalue_line("markers", "metrics: mark test as metrics-related")
    config.addinivalue_line("markers", "output: mark test as output-related")
    config.addinivalue_line("markers", "cli: mark test as CLI-related")


# Custom pytest collection hook to organize tests
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and locations."""
    for item in items:
        # Add markers based on test file names
        if "test_config" in item.fspath.basename:
            item.add_marker(pytest.mark.config)
        elif "test_metrics" in item.fspath.basename:
            item.add_marker(pytest.mark.metrics)
        elif "test_cli" in item.fspath.basename:
            item.add_marker(pytest.mark.cli)
        elif "test_output" in item.fspath.basename:
            item.add_marker(pytest.mark.output)

        # Add markers based on test names
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

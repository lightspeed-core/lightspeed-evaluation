"""Pytest configuration and fixtures for LSC Evaluation Framework tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import yaml


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_system_config() -> Dict[str, Any]:
    """Sample system configuration based on system.yaml."""
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": 300,
            "num_retries": 3
        },
        "environment": {
            "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
            "DEEPEVAL_DISABLE_PROGRESS_BAR": "YES",
            "LITELLM_LOG_LEVEL": "ERROR"
        },
        "logging": {
            "source_level": "INFO",
            "package_level": "ERROR",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "show_timestamps": True,
            "package_overrides": {
                "httpx": "ERROR",
                "urllib3": "ERROR",
                "requests": "ERROR",
                "matplotlib": "ERROR",
                "openai": "ERROR",
                "LiteLLM": "WARNING",
                "DeepEval": "WARNING"
            }
        },
        "metrics_metadata": {
            "turn_level": {
                "ragas:faithfulness": {
                    "threshold": 0.8,
                    "type": "turn",
                    "description": "How faithful the response is to the provided context",
                    "framework": "ragas"
                },
                "ragas:response_relevancy": {
                    "threshold": 0.8,
                    "type": "turn",
                    "description": "How relevant the response is to the question",
                    "framework": "ragas"
                },
                "ragas:context_recall": {
                    "threshold": 0.8,
                    "type": "turn",
                    "description": "Did we fetch every fact the answer needs?",
                    "framework": "ragas"
                },
                "custom:answer_correctness": {
                    "threshold": 0.75,
                    "type": "turn",
                    "description": "Correctness vs expected answer using custom LLM evaluation",
                    "framework": "custom"
                }
            },
            "conversation_level": {
                "deepeval:conversation_completeness": {
                    "threshold": 0.8,
                    "type": "conversation",
                    "description": "How completely the conversation addresses user intentions",
                    "framework": "deepeval"
                },
                "deepeval:conversation_relevancy": {
                    "threshold": 0.7,
                    "type": "conversation",
                    "description": "How relevant the conversation is to the topic/context",
                    "framework": "deepeval"
                }
            }
        },
        "output": {
            "base_directory": "./eval_output",
            "base_filename": "evaluation",
            "formats": {
                "csv": True,
                "json": True,
                "txt": True
            },
            "include_graphs": True,
            "csv_columns": [
                "conversation_group_id",
                "turn_id",
                "metric_identifier",
                "score",
                "threshold",
                "result",
                "reason",
                "query",
                "response",
                "execution_time"
            ]
        },
        "visualization": {
            "figsize": [12, 8],
            "dpi": 300,
            "enabled_graphs": [
                "pass_rates",
                "score_distribution",
                "conversation_heatmap",
                "status_breakdown"
            ]
        }
    }


@pytest.fixture
def sample_evaluation_data() -> list[Dict[str, Any]]:
    """Sample evaluation data based on evaluation_data.yaml."""
    return [
        {
            "conversation_group_id": "conv_group_1",
            "description": "Test conversation group 1",
            "turn_metrics": [
                "ragas:faithfulness",
                "ragas:response_relevancy"
            ],
            "turn_metrics_metadata": {
                "ragas:faithfulness": {
                    "threshold": 0.9
                }
            },
            "conversation_metrics": [],
            "conversation_metrics_metadata": {},
            "turns": [
                {
                    "turn_id": 1,
                    "query": "What is machine learning?",
                    "response": "Machine learning is a subset of artificial intelligence.",
                    "contexts": [
                        {"content": "Machine learning involves algorithms that learn from data."},
                        {"content": "AI encompasses various techniques including machine learning."}
                    ],
                    "expected_response": "Machine learning is a method of data analysis."
                }
            ]
        },
        {
            "conversation_group_id": "conv_group_2",
            "description": "Test conversation group 2",
            "turn_metrics": [
                "ragas:context_recall",
                "custom:answer_correctness"
            ],
            "turn_metrics_metadata": {},
            "conversation_metrics": [
                "deepeval:conversation_completeness"
            ],
            "conversation_metrics_metadata": {},
            "turns": [
                {
                    "turn_id": 1,
                    "query": "How does deep learning work?",
                    "response": "Deep learning uses neural networks with multiple layers.",
                    "contexts": [
                        {"content": "Neural networks are inspired by biological neurons."}
                    ],
                    "expected_response": "Deep learning employs multi-layered neural networks."
                },
                {
                    "turn_id": 2,
                    "query": "What are the applications?",
                    "response": "Deep learning is used in image recognition, NLP, and more.",
                    "contexts": [
                        {"content": "Computer vision uses deep learning for image analysis."}
                    ],
                    "expected_response": "Applications include computer vision and natural language processing."
                }
            ]
        }
    ]


@pytest.fixture
def system_config_file(temp_dir: Path, sample_system_config: Dict[str, Any]) -> Path:
    """Create a temporary system config file."""
    config_file = temp_dir / "system.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_system_config, f)
    return config_file


@pytest.fixture
def evaluation_data_file(temp_dir: Path, sample_evaluation_data: list[Dict[str, Any]]) -> Path:
    """Create a temporary evaluation data file."""
    data_file = temp_dir / "evaluation_data.yaml"
    with open(data_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_evaluation_data, f)
    return data_file


@pytest.fixture
def invalid_system_config() -> Dict[str, Any]:
    """Invalid system configuration for testing error cases."""
    return {
        "llm": {
            "provider": "",  # Invalid empty provider
            "model": "gpt-4o-mini",
            "temperature": -1.0,  # Invalid temperature
            "max_tokens": -100,  # Invalid max_tokens
            "timeout": 0,  # Invalid timeout
            "num_retries": -1  # Invalid retries
        },
        "logging": {
            "source_level": "INVALID_LEVEL",  # Invalid log level
            "package_level": "ERROR"
        }
    }


@pytest.fixture
def invalid_evaluation_data() -> list[Dict[str, Any]]:
    """Invalid evaluation data for testing error cases."""
    return [
        {
            "conversation_group_id": "",  # Invalid empty ID
            "turn_metrics": [
                "invalid:metric"  # Invalid metric format
            ],
            "turns": []  # Invalid empty turns
        },
        {
            "conversation_group_id": "valid_id",
            "turn_metrics": [
                "ragas:faithfulness"  # Requires contexts
            ],
            "turns": [
                {
                    "turn_id": 0,  # Invalid turn_id
                    "query": "",  # Invalid empty query
                    "response": "Valid response",
                    "contexts": []  # Missing required contexts
                }
            ]
        }
    ]


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before and after each test."""
    # Store original values
    original_env = {}
    env_vars_to_clean = [
        "DEEPEVAL_TELEMETRY_OPT_OUT",
        "DEEPEVAL_DISABLE_PROGRESS_BAR",
        "LITELLM_LOG_LEVEL"
    ]
    
    for var in env_vars_to_clean:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]
    
    yield
    
    # Restore original values
    for var in env_vars_to_clean:
        if var in os.environ:
            del os.environ[var]
        if var in original_env:
            os.environ[var] = original_env[var]

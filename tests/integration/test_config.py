"""Simplified integration tests for configuration components."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from lightspeed_evaluation.core.config import (
    ConfigLoader,
    DataValidator,
    setup_environment_variables,
)

class TestDataValidator:
    """Test data validation functionality."""

    def test_data_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        assert validator is not None

class TestConfigLoaderIntegration:
    """Test ConfigLoader integration scenarios."""

    def test_full_config_loading_pipeline(self):
        """Test complete configuration loading pipeline."""
        # Create temporary system config
        system_config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 1024,
                "timeout": 300,
                "num_retries": 3,
            },
            "output": {
                "base_directory": "./test_output",
                "include_graphs": True,
            },
            "logging": {
                "source_level": "INFO",
            },
            "metrics_metadata": {
                "turn_level": {
                    "ragas:faithfulness": {
                        "threshold": 0.8,
                        "type": "turn",
                        "description": "How faithful the response is to the provided context",
                        "framework": "ragas",
                    }
                },
                "conversation_level": {},
            },
        }

        # Create temporary evaluation data
        eval_data = {
            "conversation_group_id": "integration_test",
            "description": "Integration test conversation",
            "turn_metrics": ["ragas:faithfulness"],
            "conversation_metrics": [],
            "turns": [
                {
                    "turn_id": 1,
                    "query": "What is Kubernetes?",
                    "response": "Kubernetes is a container orchestration platform.",
                    "contexts": [{"content": "Kubernetes documentation context"}],
                    "expected_response": "Kubernetes is an open-source container orchestration system.",
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as system_f:
            yaml.dump(system_config_data, system_f)
            system_config_path = system_f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as eval_f:
            yaml.dump([eval_data], eval_f)
            eval_data_path = eval_f.name

        try:
            with patch("lightspeed_evaluation.core.config.loader.setup_logging"):
                loader = ConfigLoader()

                # Load system config
                system_config = loader.load_system_config(system_config_path)
                assert system_config.llm_provider == "openai"
                assert system_config.llm_model == "gpt-4o-mini"
                assert system_config.output_dir == "./test_output"

                # Load evaluation data
                validator = DataValidator()
                evaluation_data = validator.load_evaluation_data(eval_data_path)
                assert len(evaluation_data) == 1
                assert evaluation_data[0].conversation_group_id == "integration_test"
                assert len(evaluation_data[0].turns) == 1

        finally:
            os.unlink(system_config_path)
            os.unlink(eval_data_path)

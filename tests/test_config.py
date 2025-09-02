"""Tests for configuration components."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from lightspeed_evaluation.core.config import (
    ConfigLoader,
    DataValidator,
    EvaluationData,
    LLMConfig,
    SystemConfig,
    TurnData,
    setup_environment_variables,
)


class TestSystemConfig:
    """Test SystemConfig model."""

    def test_system_config_defaults(self):
        """Test SystemConfig with default values."""
        config = SystemConfig()
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.llm_temperature == 0.0
        assert config.output_dir == "./eval_output"
        assert config.include_graphs is True

    def test_system_config_custom_values(self):
        """Test SystemConfig with custom values."""
        config = SystemConfig(
            llm_provider="anthropic",
            llm_model="claude-3-sonnet",
            llm_temperature=0.5,
            output_dir="./custom_output",
            include_graphs=False
        )
        
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-sonnet"
        assert config.llm_temperature == 0.5
        assert config.output_dir == "./custom_output"
        assert config.include_graphs is False


class TestLLMConfig:
    """Test LLMConfig model."""

    def test_llm_config_validation(self):
        """Test LLMConfig validation."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            num_retries=2
        )
        
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout == 60
        assert config.num_retries == 2

    def test_llm_config_invalid_temperature(self):
        """Test LLMConfig with invalid temperature."""
        with pytest.raises(ValueError):
            LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=3.0  # Invalid: > 2.0
            )

    def test_llm_config_from_dict(self):
        """Test creating LLMConfig from dictionary."""
        config_dict = {
            "provider": "anthropic",
            "model": "claude-3-haiku",
            "temperature": 0.3,
            "max_tokens": 800,
            "timeout": 120,
            "num_retries": 1
        }
        
        config = LLMConfig.from_dict(config_dict)
        
        assert config.provider == "anthropic"
        assert config.model == "claude-3-haiku"
        assert config.temperature == 0.3


class TestTurnData:
    """Test TurnData model."""

    def test_valid_turn_data(self):
        """Test valid TurnData creation."""
        turn = TurnData(
            turn_id=1,
            query="What is AI?",
            response="AI is artificial intelligence.",
            contexts=[{"content": "AI context"}],
            expected_response="AI stands for artificial intelligence."
        )
        
        assert turn.turn_id == 1
        assert turn.query == "What is AI?"
        assert turn.response == "AI is artificial intelligence."
        assert len(turn.contexts) == 1
        assert turn.contexts[0]["content"] == "AI context"

    def test_turn_data_validation_empty_query(self):
        """Test TurnData validation with empty query."""
        with pytest.raises(ValueError, match="Query and response cannot be empty"):
            TurnData(
                turn_id=1,
                query="",
                response="Valid response"
            )

    def test_turn_data_validation_invalid_turn_id(self):
        """Test TurnData validation with invalid turn_id."""
        with pytest.raises(ValueError, match="Turn ID must be positive"):
            TurnData(
                turn_id=0,  # Invalid: must be positive
                query="Valid query",
                response="Valid response"
            )

    def test_turn_data_context_validation(self):
        """Test TurnData context validation."""
        with pytest.raises(ValueError, match='Context 0 must have a "content" field'):
            TurnData(
                turn_id=1,
                query="Valid query",
                response="Valid response",
                contexts=[{"invalid": "no content field"}]
            )


class TestEvaluationData:
    """Test EvaluationData model."""

    def test_valid_evaluation_data(self):
        """Test valid EvaluationData creation."""
        turn = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )
        
        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            description="Test conversation",
            turn_metrics=["ragas:faithfulness"],
            conversation_metrics=["deepeval:completeness"],
            turns=[turn]
        )
        
        assert eval_data.conversation_group_id == "test_conv"
        assert eval_data.description == "Test conversation"
        assert len(eval_data.turn_metrics) == 1
        assert len(eval_data.conversation_metrics) == 1
        assert len(eval_data.turns) == 1

    def test_evaluation_data_empty_conversation_id(self):
        """Test EvaluationData with empty conversation_group_id."""
        with pytest.raises(ValueError, match="Conversation group ID cannot be empty"):
            EvaluationData(
                conversation_group_id="",
                turns=[TurnData(turn_id=1, query="q", response="r")]
            )

    def test_evaluation_data_empty_turns(self):
        """Test EvaluationData with empty turns."""
        with pytest.raises(ValueError, match="Conversation must have at least one turn"):
            EvaluationData(
                conversation_group_id="test_conv",
                turns=[]
            )

    def test_evaluation_data_invalid_metric_format(self):
        """Test EvaluationData with invalid metric format."""
        turn = TurnData(turn_id=1, query="q", response="r")
        
        with pytest.raises(ValueError, match='must be in format "framework:metric_name"'):
            EvaluationData(
                conversation_group_id="test_conv",
                turn_metrics=["invalid_metric"],  # Missing colon
                turns=[turn]
            )


class TestConfigLoader:
    """Test ConfigLoader functionality."""

    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        
        assert loader.system_config is None
        assert loader.evaluation_data is None
        assert loader.logger is None

    @patch('lightspeed_evaluation.core.config.loader.setup_logging')
    def test_load_system_config_with_mock(self, mock_setup_logging):
        """Test loading system config with mocked dependencies."""
        # Create temporary config file
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.5
            },
            "output": {
                "base_directory": "./test_output"
            },
            "logging": {
                "source_level": "DEBUG"
            },
            "metrics_metadata": {
                "turn_level": {},
                "conversation_level": {}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loader = ConfigLoader()
            system_config = loader.load_system_config(config_path)
            
            assert system_config.llm_provider == "openai"
            assert system_config.llm_model == "gpt-4"
            assert system_config.llm_temperature == 0.5
            assert system_config.output_dir == "./test_output"
            
        finally:
            os.unlink(config_path)


class TestEnvironmentSetup:
    """Test environment variable setup."""

    def test_setup_environment_variables_success(self):
        """Test successful environment variable setup."""
        config_data = {
            "environment": {
                "TEST_VAR": "test_value",
                "ANOTHER_VAR": "another_value"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Clear any existing values
            os.environ.pop("TEST_VAR", None)
            os.environ.pop("ANOTHER_VAR", None)
            
            setup_environment_variables(config_path)
            
            assert os.environ.get("TEST_VAR") == "test_value"
            assert os.environ.get("ANOTHER_VAR") == "another_value"
            
        finally:
            os.unlink(config_path)
            # Clean up
            os.environ.pop("TEST_VAR", None)
            os.environ.pop("ANOTHER_VAR", None)

    def test_setup_environment_variables_fallback(self):
        """Test environment variable setup with fallback."""
        # Test with non-existent file
        setup_environment_variables("nonexistent_config.yaml")
        
        # Should set fallback values
        assert os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") == "YES"
        assert os.environ.get("LITELLM_LOG_LEVEL") == "ERROR"


class TestDataValidator:
    """Test DataValidator functionality."""

    def test_data_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        
        assert validator.validation_errors == []
        assert validator.evaluation_data is None

    def test_load_evaluation_data_from_yaml(self):
        """Test loading evaluation data from YAML file."""
        eval_data = [
            {
                "conversation_group_id": "test_conv",
                "turn_metrics": ["ragas:faithfulness"],
                "conversation_metrics": [],
                "turns": [
                    {
                        "turn_id": 1,
                        "query": "Test query",
                        "response": "Test response",
                        "contexts": [
                            {"content": "Test context for faithfulness metric"}
                        ]
                    }
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(eval_data, f)
            data_path = f.name
        
        try:
            # Mock the metric validation to avoid the validation error
            with patch('lightspeed_evaluation.core.config.validator.TURN_LEVEL_METRICS', {"ragas:faithfulness"}):
                with patch('lightspeed_evaluation.core.config.validator.CONVERSATION_LEVEL_METRICS', set()):
                    validator = DataValidator()
                    loaded_data = validator.load_evaluation_data(data_path)
                    
                    assert len(loaded_data) == 1
                    assert loaded_data[0].conversation_group_id == "test_conv"
                    assert len(loaded_data[0].turns) == 1
            
        finally:
            os.unlink(data_path)


class TestConfigurationScenarios:
    """Test realistic configuration scenarios."""

    def test_system_config_with_different_providers(self):
        """Test SystemConfig with different LLM providers."""
        providers_config = [
            {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
            {"provider": "anthropic", "model": "claude-3-sonnet", "temperature": 0.1},
        ]
        
        for config_data in providers_config:
            config = SystemConfig(
                llm_provider=config_data["provider"],
                llm_model=config_data["model"],
                llm_temperature=config_data["temperature"]
            )
            
            assert config.llm_provider == config_data["provider"]
            assert config.llm_model == config_data["model"]
            assert config.llm_temperature == config_data["temperature"]

    def test_evaluation_data_with_multiple_metrics(self):
        """Test EvaluationData with comprehensive metric configurations."""
        eval_data = EvaluationData(
            conversation_group_id="comprehensive_eval",
            description="Full evaluation with multiple metrics",
            turn_metrics=[
                "ragas:faithfulness",
                "ragas:response_relevancy", 
                "custom:answer_correctness"
            ],
            conversation_metrics=[
                "deepeval:conversation_completeness"
            ],
            turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.85},
                "custom:answer_correctness": {"threshold": 0.80}
            },
            conversation_metrics_metadata={
                "deepeval:conversation_completeness": {"threshold": 0.75}
            },
            turns=[
                TurnData(
                    turn_id=1,
                    query="What are the benefits of cloud computing?",
                    response="Cloud computing offers scalability, cost-effectiveness, and accessibility.",
                    contexts=[
                        {"content": "Cloud computing provides on-demand access to computing resources."},
                        {"content": "Benefits include reduced infrastructure costs and improved scalability."}
                    ],
                    expected_response="Cloud computing provides scalable, cost-effective computing resources."
                )
            ]
        )
        
        assert len(eval_data.turn_metrics) == 3
        assert len(eval_data.conversation_metrics) == 1
        assert len(eval_data.turns) == 1
        assert eval_data.turn_metrics_metadata["ragas:faithfulness"]["threshold"] == 0.85

    def test_turn_data_with_rich_context(self):
        """Test TurnData with comprehensive context information."""
        turn = TurnData(
            turn_id=1,
            query="How does machine learning model training work?",
            response="Machine learning model training involves feeding data to algorithms that learn patterns and make predictions.",
            contexts=[
                {"content": "Machine learning training requires large datasets and computational resources."},
                {"content": "The training process involves iterative optimization of model parameters."},
                {"content": "Validation datasets help prevent overfitting during training."}
            ],
            expected_response="ML training feeds data to algorithms to learn patterns through iterative optimization."
        )
        
        assert len(turn.contexts) == 3
        assert all("content" in ctx for ctx in turn.contexts)
        assert "machine learning" in turn.query.lower()
        assert "training" in turn.response.lower()

    def test_load_evaluation_data_invalid_yaml(self):
        """Test loading invalid YAML evaluation data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            data_path = f.name
        
        try:
            validator = DataValidator()
            with pytest.raises((ValueError, yaml.YAMLError)):
                validator.load_evaluation_data(data_path)
                
        finally:
            os.unlink(data_path)


class TestConfigurationScenarios:
    """Test realistic configuration scenarios."""

    def test_system_config_with_different_providers(self):
        """Test SystemConfig with different LLM providers."""
        providers_config = [
            {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
            {"provider": "anthropic", "model": "claude-3-sonnet", "temperature": 0.1},
        ]
        
        for config_data in providers_config:
            config = SystemConfig(
                llm_provider=config_data["provider"],
                llm_model=config_data["model"],
                llm_temperature=config_data["temperature"]
            )
            
            assert config.llm_provider == config_data["provider"]
            assert config.llm_model == config_data["model"]
            assert config.llm_temperature == config_data["temperature"]

    def test_evaluation_data_with_multiple_metrics(self):
        """Test EvaluationData with comprehensive metric configurations."""
        eval_data = EvaluationData(
            conversation_group_id="comprehensive_eval",
            description="Full evaluation with multiple metrics",
            turn_metrics=[
                "ragas:faithfulness",
                "ragas:response_relevancy", 
                "custom:answer_correctness"
            ],
            conversation_metrics=[
                "deepeval:conversation_completeness"
            ],
            turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.85},
                "custom:answer_correctness": {"threshold": 0.80}
            },
            conversation_metrics_metadata={
                "deepeval:conversation_completeness": {"threshold": 0.75}
            },
            turns=[
                TurnData(
                    turn_id=1,
                    query="What are the benefits of cloud computing?",
                    response="Cloud computing offers scalability, cost-effectiveness, and accessibility.",
                    contexts=[
                        {"content": "Cloud computing provides on-demand access to computing resources."},
                        {"content": "Benefits include reduced infrastructure costs and improved scalability."}
                    ],
                    expected_response="Cloud computing provides scalable, cost-effective computing resources."
                )
            ]
        )
        
        assert len(eval_data.turn_metrics) == 3
        assert len(eval_data.conversation_metrics) == 1
        assert len(eval_data.turns) == 1
        assert eval_data.turn_metrics_metadata["ragas:faithfulness"]["threshold"] == 0.85

    def test_turn_data_with_rich_context(self):
        """Test TurnData with comprehensive context information."""
        turn = TurnData(
            turn_id=1,
            query="How does machine learning model training work?",
            response="Machine learning model training involves feeding data to algorithms that learn patterns and make predictions.",
            contexts=[
                {"content": "Machine learning training requires large datasets and computational resources."},
                {"content": "The training process involves iterative optimization of model parameters."},
                {"content": "Validation datasets help prevent overfitting during training."}
            ],
            expected_response="ML training feeds data to algorithms to learn patterns through iterative optimization."
        )
        
        assert len(turn.contexts) == 3
        assert all("content" in ctx for ctx in turn.contexts)
        assert "machine learning" in turn.query.lower()
        assert "training" in turn.response.lower()

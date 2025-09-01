"""Test cases for ConfigLoader class based on system.yaml configuration."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest
import yaml

from lsc_eval.core.config_loader import (
    ConfigLoader,
    SystemConfig,
    setup_environment_variables,
    setup_logging,
    populate_metric_mappings,
    validate_metrics,
    TURN_LEVEL_METRICS,
    CONVERSATION_LEVEL_METRICS,
    DEFAULT_CSV_COLUMNS,
)


@pytest.mark.config
@pytest.mark.unit
class TestSetupEnvironmentVariables:
    """Test environment variable setup from system config."""

    def test_setup_environment_variables_success(self, system_config_file: Path):
        """Test successful environment variable setup."""
        # Clear any existing env vars
        env_vars = ["DEEPEVAL_TELEMETRY_OPT_OUT", "DEEPEVAL_DISABLE_PROGRESS_BAR", "LITELLM_LOG_LEVEL"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

        setup_environment_variables(str(system_config_file))

        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
        assert os.environ["DEEPEVAL_DISABLE_PROGRESS_BAR"] == "YES"
        assert os.environ["LITELLM_LOG_LEVEL"] == "ERROR"

    def test_setup_environment_variables_file_not_found(self):
        """Test fallback when config file is not found."""
        setup_environment_variables("/nonexistent/path/system.yaml")

        # Should set fallback values
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
        assert os.environ["DEEPEVAL_DISABLE_PROGRESS_BAR"] == "YES"
        assert os.environ["LITELLM_LOG_LEVEL"] == "ERROR"

    def test_setup_environment_variables_invalid_yaml(self, temp_dir: Path):
        """Test fallback when YAML is invalid."""
        invalid_config = temp_dir / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        setup_environment_variables(str(invalid_config))

        # Should set fallback values
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
        assert os.environ["DEEPEVAL_DISABLE_PROGRESS_BAR"] == "YES"
        assert os.environ["LITELLM_LOG_LEVEL"] == "ERROR"

    def test_setup_environment_variables_missing_environment_section(self, temp_dir: Path):
        """Test when environment section is missing from config."""
        config_without_env = {"llm": {"provider": "openai"}}
        config_file = temp_dir / "no_env.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_without_env, f)

        # Clear environment first
        env_vars = ["DEEPEVAL_TELEMETRY_OPT_OUT", "DEEPEVAL_DISABLE_PROGRESS_BAR", "LITELLM_LOG_LEVEL"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

        setup_environment_variables(str(config_file))

        # When environment section is missing (but file loads successfully), 
        # no environment variables should be set (no fallback triggered)
        assert "DEEPEVAL_TELEMETRY_OPT_OUT" not in os.environ


@pytest.mark.config
@pytest.mark.unit
class TestSetupLogging:
    """Test logging configuration setup."""

    def test_setup_logging_default_config(self):
        """Test logging setup with default configuration."""
        logging_config = {
            "source_level": "INFO",
            "package_level": "ERROR",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "show_timestamps": True,
            "package_overrides": {
                "httpx": "WARNING",
                "openai": "ERROR"
            }
        }

        logger = setup_logging(logging_config)

        assert logger.name == "lsc_eval"
        assert logger.level == logging.INFO
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.ERROR

    def test_setup_logging_variations(self):
        """Test logging setup with different configurations."""
        # Test DEBUG level
        debug_config = {"source_level": "DEBUG", "package_level": "WARNING"}
        logger = setup_logging(debug_config)
        assert logger.level == logging.DEBUG

        # Test invalid level
        invalid_config = {"source_level": "INVALID_LEVEL", "package_level": "ERROR"}
        with pytest.raises(AttributeError):
            setup_logging(invalid_config)

        # Test package overrides
        override_config = {
            "source_level": "INFO",
            "package_level": "ERROR", 
            "package_overrides": {"urllib3": "WARNING", "requests": "DEBUG"}
        }
        setup_logging(override_config)
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("requests").level == logging.DEBUG


@pytest.mark.config
@pytest.mark.unit
class TestPopulateMetricMappings:
    """Test metric mapping population from system config."""

    def test_populate_metric_mappings_success(self, sample_system_config: Dict[str, Any]):
        """Test successful metric mapping population."""
        metrics_metadata = sample_system_config["metrics_metadata"]
        
        populate_metric_mappings(metrics_metadata)

        # Check turn-level metrics
        expected_turn_metrics = {
            "ragas:faithfulness",
            "ragas:response_relevancy", 
            "ragas:context_recall",
            "custom:answer_correctness"
        }
        assert TURN_LEVEL_METRICS == expected_turn_metrics

        # Check conversation-level metrics
        expected_conversation_metrics = {
            "deepeval:conversation_completeness",
            "deepeval:conversation_relevancy"
        }
        assert CONVERSATION_LEVEL_METRICS == expected_conversation_metrics

    def test_populate_metric_mappings_edge_cases(self):
        """Test metric mapping with edge cases."""
        # Test empty metadata
        populate_metric_mappings({})
        assert len(TURN_LEVEL_METRICS) == 0
        assert len(CONVERSATION_LEVEL_METRICS) == 0

        # Test missing conversation section
        partial_metadata = {"turn_level": {"ragas:faithfulness": {"type": "turn"}}}
        populate_metric_mappings(partial_metadata)
        assert "ragas:faithfulness" in TURN_LEVEL_METRICS
        assert len(CONVERSATION_LEVEL_METRICS) == 0

    def test_populate_metric_mappings_incorrect_type(self):
        """Test metric mapping with incorrect type field."""
        metrics_metadata = {
            "turn_level": {
                "ragas:faithfulness": {"type": "conversation"},  # Wrong type
                "ragas:response_relevancy": {"type": "turn"}
            },
            "conversation_level": {
                "deepeval:completeness": {"type": "turn"}  # Wrong type
            }
        }

        populate_metric_mappings(metrics_metadata)

        # Only correctly typed metrics should be included
        assert "ragas:response_relevancy" in TURN_LEVEL_METRICS
        assert "ragas:faithfulness" not in TURN_LEVEL_METRICS
        assert len(CONVERSATION_LEVEL_METRICS) == 0


@pytest.mark.config
@pytest.mark.validation
@pytest.mark.unit
class TestValidateMetrics:
    """Test metric validation against available metrics."""

    def test_validate_metrics_success(self, sample_system_config: Dict[str, Any]):
        """Test successful metric validation."""
        # First populate the mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        turn_metrics = ["ragas:faithfulness", "ragas:response_relevancy"]
        conversation_metrics = ["deepeval:conversation_completeness"]

        errors = validate_metrics(turn_metrics, conversation_metrics)

        assert len(errors) == 0

    def test_validate_metrics_invalid_turn_metrics(self, sample_system_config: Dict[str, Any]):
        """Test validation with invalid turn metrics."""
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        turn_metrics = ["ragas:faithfulness", "invalid:metric", "another:invalid"]
        conversation_metrics = []

        errors = validate_metrics(turn_metrics, conversation_metrics)

        assert len(errors) == 2
        assert "'invalid:metric' is not a recognized turn-level metric" in errors
        assert "'another:invalid' is not a recognized turn-level metric" in errors

    def test_validate_metrics_invalid_conversation_metrics(self, sample_system_config: Dict[str, Any]):
        """Test validation with invalid conversation metrics."""
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        turn_metrics = []
        conversation_metrics = ["deepeval:conversation_completeness", "invalid:conversation_metric"]

        errors = validate_metrics(turn_metrics, conversation_metrics)

        assert len(errors) == 1
        assert "'invalid:conversation_metric' is not a recognized conversation-level metric" in errors

    def test_validate_metrics_empty_mappings(self):
        """Test validation when metric mappings are empty."""
        # Clear mappings
        TURN_LEVEL_METRICS.clear()
        CONVERSATION_LEVEL_METRICS.clear()

        turn_metrics = ["ragas:faithfulness"]
        conversation_metrics = ["deepeval:completeness"]

        errors = validate_metrics(turn_metrics, conversation_metrics)

        assert len(errors) == 2


@pytest.mark.config
@pytest.mark.models
@pytest.mark.unit
class TestSystemConfig:
    """Test SystemConfig Pydantic model."""

    def test_system_config_default_values(self):
        """Test SystemConfig with default values."""
        config = SystemConfig()

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.llm_temperature == 0.0
        assert config.llm_max_tokens == 512
        assert config.llm_timeout == 300
        assert config.llm_retries == 3
        assert config.logging_source_level == "INFO"
        assert config.output_dir == "./eval_output"
        assert config.csv_columns == []

    def test_system_config_custom_values(self):
        """Test SystemConfig with custom values."""
        config = SystemConfig(
            llm_provider="anthropic",
            llm_model="claude-3-sonnet",
            llm_temperature=0.5,
            llm_max_tokens=1024,
            output_dir="/custom/output",
            csv_columns=["custom", "columns"]
        )

        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-sonnet"
        assert config.llm_temperature == 0.5
        assert config.llm_max_tokens == 1024
        assert config.output_dir == "/custom/output"
        assert config.csv_columns == ["custom", "columns"]

    def test_system_config_validation(self):
        """Test SystemConfig field validation."""
        # Test valid config
        config = SystemConfig(
            llm_temperature=1.0,
            llm_max_tokens=2048,
            visualization_figsize=[10, 6],
            visualization_dpi=150
        )

        assert config.llm_temperature == 1.0
        assert config.llm_max_tokens == 2048
        assert config.visualization_figsize == [10, 6]
        assert config.visualization_dpi == 150


@pytest.mark.config
@pytest.mark.integration
class TestConfigLoader:
    """Test ConfigLoader class functionality."""

    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()

        assert loader.system_config is None
        assert loader.evaluation_data is None
        assert loader.logger is None

    def test_load_system_config_success(self, system_config_file: Path):
        """Test successful system config loading."""
        loader = ConfigLoader()
        
        config = loader.load_system_config(str(system_config_file))

        assert isinstance(config, SystemConfig)
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.llm_temperature == 0.0
        assert config.logging_source_level == "INFO"
        assert config.output_dir == "./eval_output"
        assert loader.logger is not None

    def test_load_system_config_file_not_found(self):
        """Test system config loading with non-existent file."""
        loader = ConfigLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_system_config("/nonexistent/config.yaml")

    def test_load_system_config_invalid_yaml(self, temp_dir: Path):
        """Test system config loading with invalid YAML."""
        invalid_config = temp_dir / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        loader = ConfigLoader()

        with pytest.raises(yaml.YAMLError):
            loader.load_system_config(str(invalid_config))

    def test_load_system_config_partial_config(self, temp_dir: Path):
        """Test system config loading with partial configuration."""
        partial_config = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-3-haiku"
            },
            "logging": {
                "source_level": "DEBUG"
            }
        }

        config_file = temp_dir / "partial.yaml"
        with open(config_file, "w") as f:
            yaml.dump(partial_config, f)

        loader = ConfigLoader()
        config = loader.load_system_config(str(config_file))

        # Should use provided values
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-haiku"
        assert config.logging_source_level == "DEBUG"

        # Should use defaults for missing values
        assert config.llm_temperature == 0.0
        assert config.llm_max_tokens == 512
        assert config.output_dir == "./eval_output"

    def test_get_llm_config_dict_success(self, system_config_file: Path):
        """Test getting LLM config as dictionary."""
        loader = ConfigLoader()
        loader.load_system_config(str(system_config_file))

        llm_config = loader.get_llm_config_dict()

        expected_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }

        assert llm_config == expected_config

    def test_get_llm_config_dict_no_config_loaded(self):
        """Test getting LLM config when no config is loaded."""
        loader = ConfigLoader()

        with pytest.raises(ValueError, match="System config not loaded"):
            loader.get_llm_config_dict()

    def test_metric_mappings_populated_after_load(self, system_config_file: Path):
        """Test that metric mappings are populated after loading config."""
        # Clear existing mappings
        TURN_LEVEL_METRICS.clear()
        CONVERSATION_LEVEL_METRICS.clear()

        loader = ConfigLoader()
        loader.load_system_config(str(system_config_file))

        # Mappings should be populated
        assert len(TURN_LEVEL_METRICS) > 0
        assert len(CONVERSATION_LEVEL_METRICS) > 0
        assert "ragas:faithfulness" in TURN_LEVEL_METRICS
        assert "deepeval:conversation_completeness" in CONVERSATION_LEVEL_METRICS

    def test_logger_configured_after_load(self, system_config_file: Path):
        """Test that logger is properly configured after loading config."""
        loader = ConfigLoader()
        config = loader.load_system_config(str(system_config_file))

        assert loader.logger is not None
        assert loader.logger.name == "lsc_eval"
        assert loader.logger.level == logging.INFO

    @patch('lsc_eval.core.config_loader.setup_logging')
    def test_load_system_config_logging_setup_called(self, mock_setup_logging: MagicMock, system_config_file: Path):
        """Test that setup_logging is called during config loading."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        loader = ConfigLoader()
        loader.load_system_config(str(system_config_file))

        mock_setup_logging.assert_called_once()
        assert loader.logger == mock_logger

    def test_default_csv_columns_constant(self):
        """Test that DEFAULT_CSV_COLUMNS contains expected columns."""
        expected_columns = [
            "conversation_group_id",
            "turn_id",
            "metric_identifier",
            "result",
            "score",
            "threshold",
            "reason",
            "execution_time",
        ]

        for column in expected_columns:
            assert column in DEFAULT_CSV_COLUMNS

    def test_config_loader_with_custom_csv_columns(self, temp_dir: Path):
        """Test config loading with custom CSV columns."""
        config_with_custom_columns = {
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "output": {
                "csv_columns": ["custom_col1", "custom_col2", "custom_col3"]
            },
            "metrics_metadata": {"turn_level": {}, "conversation_level": {}}
        }

        config_file = temp_dir / "custom_columns.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_with_custom_columns, f)

        loader = ConfigLoader()
        config = loader.load_system_config(str(config_file))

        assert config.csv_columns == ["custom_col1", "custom_col2", "custom_col3"]

    def test_config_loader_metrics_metadata_extraction(self, system_config_file: Path):
        """Test that metrics metadata is properly extracted."""
        loader = ConfigLoader()
        config = loader.load_system_config(str(system_config_file))

        # Check that default metrics metadata is populated
        assert len(config.default_turn_metrics_metadata) > 0
        assert len(config.default_conversation_metrics_metadata) > 0
        
        # Check specific metrics
        assert "ragas:faithfulness" in config.default_turn_metrics_metadata
        assert "deepeval:conversation_completeness" in config.default_conversation_metrics_metadata
        
        # Check metadata structure
        faithfulness_meta = config.default_turn_metrics_metadata["ragas:faithfulness"]
        assert faithfulness_meta["threshold"] == 0.8
        assert faithfulness_meta["type"] == "turn"
        assert faithfulness_meta["framework"] == "ragas"

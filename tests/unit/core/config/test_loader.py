"""Unit tests for core.config.loader module."""

import logging
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import yaml

from lightspeed_evaluation.core.config.loader import (
    DEFAULT_CSV_COLUMNS,
    ConfigLoader,
    setup_environment_variables,
    setup_logging,
)


class TestSetupEnvironmentVariables:
    """Unit tests for setup_environment_variables function."""

    def test_setup_environment_variables_success(self):
        """Test successful environment variable setup from config."""
        config_data = {
            "environment": {
                "TEST_VAR_1": "value1",
                "TEST_VAR_2": "value2",
                "NUMERIC_VAR": 123,
                "BOOLEAN_VAR": True,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Clear any existing test variables
            for var in ["TEST_VAR_1", "TEST_VAR_2", "NUMERIC_VAR", "BOOLEAN_VAR"]:
                os.environ.pop(var, None)

            setup_environment_variables(config_path)

            assert os.environ.get("TEST_VAR_1") == "value1"
            assert os.environ.get("TEST_VAR_2") == "value2"
            assert os.environ.get("NUMERIC_VAR") == "123"
            assert os.environ.get("BOOLEAN_VAR") == "True"

        finally:
            os.unlink(config_path)
            # Clean up
            for var in ["TEST_VAR_1", "TEST_VAR_2", "NUMERIC_VAR", "BOOLEAN_VAR"]:
                os.environ.pop(var, None)

    def test_setup_environment_variables_missing_environment_section(self):
        """Test setup with config missing environment section."""
        config_data = {
            "llm": {"provider": "openai"},
            "output": {"base_directory": "./output"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Should not raise an exception
            setup_environment_variables(config_path)

        finally:
            os.unlink(config_path)

    def test_setup_environment_variables_file_not_found(self):
        """Test setup with non-existent config file."""
        with patch("builtins.print") as mock_print:
            setup_environment_variables("nonexistent_config.yaml")

            # Should set fallback values
            assert os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") == "YES"
            assert os.environ.get("DEEPEVAL_DISABLE_PROGRESS_BAR") == "YES"
            assert os.environ.get("LITELLM_LOG_LEVEL") == "ERROR"

            # Should print warning
            mock_print.assert_any_call("Using fallback environment settings...")

    def test_setup_environment_variables_invalid_yaml(self):
        """Test setup with invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with patch("builtins.print") as mock_print:
                setup_environment_variables(config_path)

                # Should set fallback values
                assert os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") == "YES"
                assert os.environ.get("LITELLM_LOG_LEVEL") == "ERROR"

                # Should print warning
                mock_print.assert_any_call("Using fallback environment settings...")

        finally:
            os.unlink(config_path)

    def test_setup_environment_variables_overwrites_existing(self):
        """Test that setup overwrites existing environment variables."""
        config_data = {"environment": {"EXISTING_VAR": "new_value"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Set an existing value
            os.environ["EXISTING_VAR"] = "old_value"

            setup_environment_variables(config_path)

            # Should be overwritten
            assert os.environ.get("EXISTING_VAR") == "new_value"

        finally:
            os.unlink(config_path)
            os.environ.pop("EXISTING_VAR", None)


class TestSetupLogging:
    """Unit tests for setup_logging function."""

    def test_setup_logging_default_config(self):
        """Test setup_logging with default configuration."""
        logging_config = {}

        with patch("logging.basicConfig") as mock_basic_config:
            logger = setup_logging(logging_config)

            # Should use defaults
            mock_basic_config.assert_called_once_with(
                level=logging.INFO,  # default source_level
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                force=True,
            )

            assert isinstance(logger, logging.Logger)

    def test_setup_logging_custom_config(self):
        """Test setup_logging with custom configuration."""
        logging_config = {
            "source_level": "DEBUG",
            "package_level": "ERROR",
            "format": "%(levelname)s: %(message)s",
        }

        with patch("logging.basicConfig") as mock_basic_config:
            logger = setup_logging(logging_config)

            mock_basic_config.assert_called_once_with(
                level=logging.DEBUG, format="%(levelname)s: %(message)s", force=True
            )

            assert isinstance(logger, logging.Logger)

    def test_setup_logging_with_package_overrides(self):
        """Test setup_logging with package-specific overrides."""
        logging_config = {
            "source_level": "INFO",
            "package_level": "WARNING",
            "package_overrides": {"urllib3": "ERROR", "custom_package": "DEBUG"},
        }

        with patch("logging.basicConfig"):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                logger = setup_logging(logging_config)

                # Should have called getLogger for overridden packages
                assert mock_get_logger.call_count >= 2

                # Should have set levels on loggers
                assert mock_logger.setLevel.call_count >= 2

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid log level."""
        logging_config = {"source_level": "INVALID_LEVEL"}

        with pytest.raises(AttributeError):
            setup_logging(logging_config)

    def test_setup_logging_case_insensitive_levels(self):
        """Test setup_logging with different case log levels."""
        logging_config = {"source_level": "debug", "package_level": "warning"}

        with patch("logging.basicConfig") as mock_basic_config:
            logger = setup_logging(logging_config)

            mock_basic_config.assert_called_once_with(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                force=True,
            )


class TestDefaultCsvColumns:
    """Unit tests for DEFAULT_CSV_COLUMNS constant."""

    def test_default_csv_columns_exists(self):
        """Test that DEFAULT_CSV_COLUMNS is defined and has expected columns."""
        assert isinstance(DEFAULT_CSV_COLUMNS, list)
        assert len(DEFAULT_CSV_COLUMNS) > 0

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

    def test_default_csv_columns_no_duplicates(self):
        """Test that DEFAULT_CSV_COLUMNS has no duplicate entries."""
        assert len(DEFAULT_CSV_COLUMNS) == len(set(DEFAULT_CSV_COLUMNS))

    def test_default_csv_columns_all_strings(self):
        """Test that all DEFAULT_CSV_COLUMNS entries are strings."""
        for column in DEFAULT_CSV_COLUMNS:
            assert isinstance(column, str)
            assert len(column) > 0


class TestConfigLoader:
    """Unit tests for ConfigLoader class."""

    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization with defaults."""
        loader = ConfigLoader()

        assert loader.system_config is None
        assert loader.evaluation_data is None
        assert loader.logger is None

    @patch("lightspeed_evaluation.core.config.loader.setup_logging")
    def test_load_system_config_success(self, mock_setup_logging):
        """Test successful system config loading."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        config_data = {
            "llm": {"provider": "openai", "model": "gpt-4", "temperature": 0.5},
            "output": {"base_directory": "./test_output", "include_graphs": True},
            "logging": {"source_level": "DEBUG"},
            "metrics_metadata": {
                "turn_level": {"ragas:faithfulness": {"threshold": 0.8}},
                "conversation_level": {"deepeval:completeness": {"threshold": 0.9}},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader()
            system_config = loader.load_system_config(config_path)

            assert system_config.llm_provider == "openai"
            assert system_config.llm_model == "gpt-4"
            assert system_config.llm_temperature == 0.5
            assert system_config.output_dir == "./test_output"
            assert system_config.include_graphs is True

            # Should have set up logging
            mock_setup_logging.assert_called_once()
            assert loader.logger == mock_logger

        finally:
            os.unlink(config_path)

    def test_load_system_config_file_not_found(self):
        """Test system config loading with non-existent file."""
        loader = ConfigLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_system_config("nonexistent_config.yaml")

    def test_load_system_config_invalid_yaml(self):
        """Test system config loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            loader = ConfigLoader()

            with pytest.raises(yaml.YAMLError):
                loader.load_system_config(config_path)

        finally:
            os.unlink(config_path)

    @patch("lightspeed_evaluation.core.config.loader.setup_logging")
    def test_load_system_config_minimal(self, mock_setup_logging):
        """Test system config loading with minimal configuration."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        config_data = {
            "llm": {"provider": "openai"},
            "logging": {"source_level": "INFO"},
            "metrics_metadata": {"turn_level": {}, "conversation_level": {}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader()
            system_config = loader.load_system_config(config_path)

            # Should use defaults for missing fields
            assert system_config.llm_provider == "openai"
            assert system_config.llm_model == "gpt-4o-mini"  # default
            assert system_config.llm_temperature == 0.0  # default
            assert system_config.output_dir == "./eval_output"  # default

        finally:
            os.unlink(config_path)

    def test_config_loader_state_persistence(self):
        """Test that ConfigLoader maintains state between operations."""
        loader = ConfigLoader()

        # Initially empty
        assert loader.system_config is None
        assert loader.evaluation_data is None

        # After loading, should persist
        with patch("lightspeed_evaluation.core.config.loader.setup_logging"):
            config_data = {
                "llm": {"provider": "openai"},
                "logging": {"source_level": "INFO"},
                "metrics_metadata": {"turn_level": {}, "conversation_level": {}},
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config_data, f)
                config_path = f.name

            try:
                system_config = loader.load_system_config(config_path)

                # State should be updated
                assert loader.system_config == system_config
                assert loader.logger is not None

            finally:
                os.unlink(config_path)

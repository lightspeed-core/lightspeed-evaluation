"""Unit tests for system setup module."""

import logging

from lightspeed_evaluation.core.models import LoggingConfig
from lightspeed_evaluation.core.system.setup import (
    setup_environment_variables,
    setup_logging,
)


class TestSetupEnvironmentVariables:
    """Tests for environment variable setup."""

    def test_setup_default_environment_variables(self, mocker):
        """Test setting up default environment variables."""
        import os

        config_data = {}

        # Use mocker to patch os.environ
        mocker.patch.dict(os.environ, {}, clear=True)

        setup_environment_variables(config_data)

        # Verify default values were set
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
        assert os.environ["DEEPEVAL_DISABLE_PROGRESS_BAR"] == "YES"
        assert os.environ["LITELLM_LOG"] == "ERROR"
        assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"

    def test_setup_custom_environment_variables(self, mocker):
        """Test setting up custom environment variables."""
        import os

        config_data = {
            "environment": {
                "CUSTOM_VAR": "custom_value",
                "ANOTHER_VAR": "another_value",
            }
        }

        mocker.patch.dict(os.environ, {}, clear=True)

        setup_environment_variables(config_data)

        # Verify custom values were set
        assert os.environ["CUSTOM_VAR"] == "custom_value"
        assert os.environ["ANOTHER_VAR"] == "another_value"
        # Defaults should still be set
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"

    def test_setup_environment_variables_override_defaults(self, mocker):
        """Test overriding default environment variables."""
        import os

        config_data = {"environment": {"LITELLM_LOG": "DEBUG"}}

        mocker.patch.dict(os.environ, {}, clear=True)

        setup_environment_variables(config_data)

        # Verify override worked
        assert os.environ["LITELLM_LOG"] == "DEBUG"

    def test_setup_environment_variables_handles_key_error(self, mocker, capsys):
        """Test handling of KeyError during environment setup."""
        import os

        config_data = {"environment": None}  # This will cause issues

        mocker.patch.dict(os.environ, {}, clear=True)

        # Should not raise, should fallback to defaults
        setup_environment_variables(config_data)

        # Verify defaults were still set
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"

        # Verify warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "fallback" in captured.out

    def test_setup_environment_variables_handles_type_error(self, mocker, capsys):
        """Test handling of TypeError during environment setup."""
        import os

        config_data = {"environment": "invalid_type"}

        mocker.patch.dict(os.environ, {}, clear=True)

        setup_environment_variables(config_data)

        # Defaults should be set despite error
        assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


class TestSetupLogging:
    """Tests for logging setup."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logging_config = LoggingConfig(
            source_level="INFO",
            package_level="WARNING",
            log_format="%(levelname)s: %(message)s",
        )

        logger = setup_logging(logging_config)

        assert logger is not None
        assert logger.name == "lightspeed_evaluation"
        assert logger.level == logging.INFO

    def test_setup_logging_debug_level(self):
        """Test logging setup with DEBUG level."""
        logging_config = LoggingConfig(
            source_level="DEBUG",
            package_level="ERROR",
            log_format="%(message)s",
        )

        logger = setup_logging(logging_config)

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_package_overrides(self):
        """Test logging setup with package overrides."""
        logging_config = LoggingConfig(
            source_level="INFO",
            package_level="WARNING",
            log_format="%(message)s",
            package_overrides={
                "httpx": "ERROR",
                "urllib3": "CRITICAL",
            },
        )

        logger = setup_logging(logging_config)

        # Verify main logger
        assert logger.level == logging.INFO

        # Verify package overrides
        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level == logging.ERROR

        urllib3_logger = logging.getLogger("urllib3")
        assert urllib3_logger.level == logging.CRITICAL

    def test_setup_logging_sets_default_noisy_packages(self):
        """Test that noisy packages get default levels set."""
        logging_config = LoggingConfig(
            source_level="INFO",
            package_level="ERROR",
            log_format="%(message)s",
        )

        setup_logging(logging_config)

        # Check that default noisy packages were configured
        matplotlib_logger = logging.getLogger("matplotlib")
        assert matplotlib_logger.level == logging.ERROR

    def test_setup_logging_handles_invalid_override_level(self, capsys):
        """Test handling of invalid log level in overrides."""
        logging_config = LoggingConfig(
            source_level="INFO",
            package_level="WARNING",
            log_format="%(message)s",
            package_overrides={
                "test_package": "INVALID_LEVEL",
            },
        )

        # Should not raise, just print warning
        logger = setup_logging(logging_config)

        assert logger is not None
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Invalid" in captured.out

    def test_setup_logging_error_level(self):
        """Test logging setup with ERROR level."""
        logging_config = LoggingConfig(
            source_level="ERROR",
            package_level="CRITICAL",
            log_format="%(levelname)s: %(message)s",
        )

        logger = setup_logging(logging_config)

        assert logger.level == logging.ERROR

    def test_setup_logging_custom_format(self):
        """Test logging with custom format."""
        custom_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging_config = LoggingConfig(
            source_level="INFO",
            package_level="WARNING",
            log_format=custom_format,
        )

        logger = setup_logging(logging_config)

        assert logger is not None
        # Format is applied to root logger, not easy to verify directly
        # but at least verify it doesn't crash

    def test_setup_logging_warning_level(self):
        """Test logging setup with WARNING level."""
        logging_config = LoggingConfig(
            source_level="WARNING",
            package_level="ERROR",
            log_format="%(message)s",
        )

        logger = setup_logging(logging_config)

        assert logger.level == logging.WARNING

    def test_setup_logging_applies_to_all_default_packages(self):
        """Test that all default noisy packages get configured."""
        logging_config = LoggingConfig(
            source_level="INFO",
            package_level="WARNING",
            log_format="%(message)s",
        )

        setup_logging(logging_config)

        # Test a few default packages
        packages = ["httpx", "requests", "LiteLLM", "ragas"]
        for package in packages:
            pkg_logger = logging.getLogger(package)
            assert pkg_logger.level == logging.WARNING

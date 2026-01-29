"""Unit tests for system setup module."""

import logging
import os

from pytest_mock import MockerFixture
from _pytest.capture import CaptureFixture

from lightspeed_evaluation.core.models import LoggingConfig
from lightspeed_evaluation.core.system.setup import (
    setup_environment_variables,
    setup_logging,
)


class TestSetupEnvironmentVariables:
    """Tests for environment variable setup."""

    def test_setup_default_environment_variables(self, mocker: MockerFixture) -> None:
        """Test setting up default environment variables."""
        config_data: dict = {}

        # Use mocker to patch os.environ
        mocker.patch.dict(os.environ, {}, clear=True)
        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = "/path/to/certifi/cacert.pem"

        setup_environment_variables(config_data)

        # Verify default values were set
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
        assert os.environ["DEEPEVAL_DISABLE_PROGRESS_BAR"] == "YES"
        assert os.environ["LITELLM_LOG"] == "ERROR"
        assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
        assert os.environ["SSL_CERTIFI_BUNDLE"] == "/path/to/certifi/cacert.pem"
        mock_where.assert_called_once()

    def test_setup_custom_environment_variables(self, mocker: MockerFixture) -> None:
        """Test setting up custom environment variables."""
        config_data = {
            "environment": {
                "CUSTOM_VAR": "custom_value",
                "ANOTHER_VAR": "another_value",
            }
        }

        mocker.patch.dict(os.environ, {}, clear=True)
        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = "/path/to/certifi/cacert.pem"

        setup_environment_variables(config_data)

        # Verify custom values were set
        assert os.environ["CUSTOM_VAR"] == "custom_value"
        assert os.environ["ANOTHER_VAR"] == "another_value"
        # Defaults should still be set
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
        assert os.environ["SSL_CERTIFI_BUNDLE"] == "/path/to/certifi/cacert.pem"

    def test_setup_environment_variables_override_defaults(
        self, mocker: MockerFixture
    ) -> None:
        """Test overriding default environment variables."""
        config_data = {"environment": {"LITELLM_LOG": "DEBUG"}}

        mocker.patch.dict(os.environ, {}, clear=True)
        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = "/path/to/certifi/cacert.pem"

        setup_environment_variables(config_data)

        # Verify override worked
        assert os.environ["LITELLM_LOG"] == "DEBUG"
        assert os.environ["SSL_CERTIFI_BUNDLE"] == "/path/to/certifi/cacert.pem"

    def test_setup_environment_variables_handles_key_error(
        self, mocker: MockerFixture, capsys: CaptureFixture
    ) -> None:
        """Test handling of KeyError during environment setup."""
        config_data = {"environment": None}  # This will cause issues

        mocker.patch.dict(os.environ, {}, clear=True)
        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = "/path/to/certifi/cacert.pem"

        # Should not raise, should fallback to defaults
        setup_environment_variables(config_data)

        # Verify defaults were still set
        assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"
        assert os.environ["SSL_CERTIFI_BUNDLE"] == "/path/to/certifi/cacert.pem"

        # Verify warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "fallback" in captured.out

    def test_setup_environment_variables_handles_type_error(
        self, mocker: MockerFixture
    ) -> None:
        """Test handling of TypeError during environment setup."""
        config_data = {"environment": "invalid_type"}

        mocker.patch.dict(os.environ, {}, clear=True)
        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = "/path/to/certifi/cacert.pem"

        setup_environment_variables(config_data)

        # Defaults should be set despite error
        assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
        assert os.environ["SSL_CERTIFI_BUNDLE"] == "/path/to/certifi/cacert.pem"

    def test_setup_ssl_certifi_bundle_set_when_ssl_cert_file_is_none(
        self, mocker: MockerFixture
    ) -> None:
        """Test SSL_CERTIFI_BUNDLE is still set even when ssl_cert_file is None."""
        config_data = {"llm": {"ssl_verify": True, "ssl_cert_file": None}}
        mocker.patch.dict(os.environ, {}, clear=True)
        mock_where = mocker.patch(
            "lightspeed_evaluation.core.system.ssl_certifi.certifi.where"
        )
        mock_where.return_value = "/path/to/certifi/cacert.pem"

        setup_environment_variables(config_data)

        assert os.environ["SSL_CERTIFI_BUNDLE"] == "/path/to/certifi/cacert.pem"
        mock_where.assert_called()


class TestSetupLogging:
    """Tests for logging setup."""

    def test_setup_logging_basic(self) -> None:
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

    def test_setup_logging_debug_level(self) -> None:
        """Test logging setup with DEBUG level."""
        logging_config = LoggingConfig(
            source_level="DEBUG",
            package_level="ERROR",
            log_format="%(message)s",
        )

        logger = setup_logging(logging_config)

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_package_overrides(self) -> None:
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

    def test_setup_logging_sets_default_noisy_packages(self) -> None:
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

    def test_setup_logging_handles_invalid_override_level(
        self, capsys: CaptureFixture
    ) -> None:
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

    def test_setup_logging_error_level(self) -> None:
        """Test logging setup with ERROR level."""
        logging_config = LoggingConfig(
            source_level="ERROR",
            package_level="CRITICAL",
            log_format="%(levelname)s: %(message)s",
        )

        logger = setup_logging(logging_config)

        assert logger.level == logging.ERROR

    def test_setup_logging_custom_format(self) -> None:
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

    def test_setup_logging_warning_level(self) -> None:
        """Test logging setup with WARNING level."""
        logging_config = LoggingConfig(
            source_level="WARNING",
            package_level="ERROR",
            log_format="%(message)s",
        )

        logger = setup_logging(logging_config)

        assert logger.level == logging.WARNING

    def test_setup_logging_applies_to_all_default_packages(self) -> None:
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

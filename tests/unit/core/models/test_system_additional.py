"""Additional tests for system configuration models."""

import os
import tempfile
import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import (
    LLMConfig,
    EmbeddingConfig,
    APIConfig,
    OutputConfig,
    VisualizationConfig,
    LoggingConfig,
)


class TestLLMConfig:
    """Additional tests for LLMConfig."""

    def test_temperature_validation_min(self) -> None:
        """Test temperature minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

    def test_temperature_validation_max(self) -> None:
        """Test temperature maximum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)

    def test_timeout_validation(self) -> None:
        """Test timeout minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(timeout=0)

    def test_num_retries_validation(self) -> None:
        """Test num_retries minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(num_retries=-1)

    def test_ssl_verify_default(self) -> None:
        """Test ssl_verify has correct default value."""
        config = LLMConfig()
        assert config.ssl_verify is True

    def test_ssl_verify_false(self) -> None:
        """Test ssl_verify can be set to False."""
        config = LLMConfig(ssl_verify=False)
        assert config.ssl_verify is False

    def test_ssl_cert_file_default(self) -> None:
        """Test ssl_cert_file defaults to None."""
        config = LLMConfig()
        assert config.ssl_cert_file is None

    def test_ssl_cert_file_valid_path(self) -> None:
        """Test ssl_cert_file with valid certificate file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".crt", delete=False) as f:
            cert_path = f.name
            f.write("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----\n")

        try:
            config = LLMConfig(ssl_cert_file=cert_path)
            assert config.ssl_cert_file is not None
            assert config.ssl_cert_file == os.path.abspath(cert_path)
            assert os.path.isabs(config.ssl_cert_file)
        finally:
            os.unlink(cert_path)

    def test_ssl_cert_file_expands_env_variables(self, mocker: MockerFixture) -> None:
        """Test ssl_cert_file expands environment variables."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".crt", delete=False) as f:
            cert_path = f.name

        try:
            test_dir = os.path.dirname(cert_path)
            test_filename = os.path.basename(cert_path)
            mocker.patch.dict(os.environ, {"TEST_CERT_DIR": test_dir})

            env_path = f"$TEST_CERT_DIR/{test_filename}"
            config = LLMConfig(ssl_cert_file=env_path)
            assert config.ssl_cert_file == os.path.abspath(cert_path)
        finally:
            os.unlink(cert_path)

    def test_ssl_cert_file_nonexistent_raises_error(self) -> None:
        """Test ssl_cert_file validation fails for non-existent file."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(ssl_cert_file="/tmp/nonexistent_cert_12345.crt")

        assert "not found" in str(exc_info.value).lower()

    def test_ssl_cert_file_directory_raises_error(self) -> None:
        """Test ssl_cert_file validation fails for directory paths."""
        temp_dir = tempfile.gettempdir()
        with pytest.raises(ValidationError):
            LLMConfig(ssl_cert_file=temp_dir)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self) -> None:
        """Test default embedding configuration."""
        config = EmbeddingConfig()

        assert config.provider is not None
        assert config.model is not None
        assert config.cache_enabled is True

    def test_custom_embedding_model(self) -> None:
        """Test custom embedding model configuration."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
        )

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"


class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_api_config(self) -> None:
        """Test default API configuration."""
        config = APIConfig()

        assert isinstance(config.enabled, bool)
        assert isinstance(config.cache_enabled, bool)
        assert config.timeout > 0

    def test_custom_api_config(self) -> None:
        """Test custom API configuration."""
        config = APIConfig(
            enabled=True,
            api_base="https://custom.api.com",
            timeout=300,
        )

        assert config.enabled is True
        assert config.api_base == "https://custom.api.com"
        assert config.timeout == 300

    def test_timeout_validation(self) -> None:
        """Test API timeout validation."""
        with pytest.raises(ValidationError):
            APIConfig(timeout=0)


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_output_config(self) -> None:
        """Test default output configuration."""
        config = OutputConfig()

        assert "csv" in config.enabled_outputs
        assert len(config.csv_columns) > 0

    def test_custom_output_config(self) -> None:
        """Test custom output configuration."""
        config = OutputConfig(
            enabled_outputs=["json"],
            csv_columns=["conversation_group_id", "result"],
        )

        assert config.enabled_outputs == ["json"]
        assert len(config.csv_columns) == 2

    def test_minimal_csv_columns(self) -> None:
        """Test with minimal CSV columns."""
        config = OutputConfig(csv_columns=["result"])
        assert len(config.csv_columns) >= 1


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_visualization_config(self) -> None:
        """Test default visualization configuration."""
        config = VisualizationConfig()

        assert isinstance(config.enabled_graphs, list)
        assert config.dpi > 0
        assert len(config.figsize) == 2

    def test_custom_visualization_config(self) -> None:
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            enabled_graphs=["pass_rates", "score_distribution"],
            dpi=150,
            figsize=(12, 8),  # pyright: ignore[reportArgumentType]
        )

        assert "pass_rates" in config.enabled_graphs
        assert "score_distribution" in config.enabled_graphs
        assert config.dpi == 150
        assert config.figsize == [12, 8]  # Pydantic converts tuple to list

    def test_dpi_validation(self) -> None:
        """Test DPI validation."""
        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=0)


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_logging_config(self) -> None:
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.source_level is not None
        assert config.package_level is not None
        assert isinstance(config.package_overrides, dict)

    def test_custom_logging_config(self) -> None:
        """Test custom logging configuration."""
        config = LoggingConfig(
            source_level="DEBUG",
            package_level="ERROR",
            package_overrides={"httpx": "CRITICAL"},
        )

        assert config.source_level == "DEBUG"
        assert config.package_level == "ERROR"
        assert config.package_overrides["httpx"] == "CRITICAL"

    def test_show_timestamps_toggle(self) -> None:
        """Test show_timestamps configuration."""
        config1 = LoggingConfig(show_timestamps=True)
        config2 = LoggingConfig(show_timestamps=False)

        assert config1.show_timestamps is True
        assert config2.show_timestamps is False

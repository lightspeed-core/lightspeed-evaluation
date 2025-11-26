"""Additional tests for system configuration models."""

import pytest
from pydantic import ValidationError

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

    def test_temperature_validation_min(self):
        """Test temperature minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

    def test_temperature_validation_max(self):
        """Test temperature maximum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)

    def test_timeout_validation(self):
        """Test timeout minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(timeout=0)

    def test_num_retries_validation(self):
        """Test num_retries minimum validation."""
        with pytest.raises(ValidationError):
            LLMConfig(num_retries=-1)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()

        assert config.provider is not None
        assert config.model is not None
        assert config.cache_enabled is True

    def test_custom_embedding_model(self):
        """Test custom embedding model configuration."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
        )

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"


class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_api_config(self):
        """Test default API configuration."""
        config = APIConfig()

        assert isinstance(config.enabled, bool)
        assert isinstance(config.cache_enabled, bool)
        assert config.timeout > 0

    def test_custom_api_config(self):
        """Test custom API configuration."""
        config = APIConfig(
            enabled=True,
            api_base="https://custom.api.com",
            timeout=300,
        )

        assert config.enabled is True
        assert config.api_base == "https://custom.api.com"
        assert config.timeout == 300

    def test_timeout_validation(self):
        """Test API timeout validation."""
        with pytest.raises(ValidationError):
            APIConfig(timeout=0)


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_output_config(self):
        """Test default output configuration."""
        config = OutputConfig()

        assert "csv" in config.enabled_outputs
        assert len(config.csv_columns) > 0

    def test_custom_output_config(self):
        """Test custom output configuration."""
        config = OutputConfig(
            enabled_outputs=["json"],
            csv_columns=["conversation_group_id", "result"],
        )

        assert config.enabled_outputs == ["json"]
        assert len(config.csv_columns) == 2

    def test_minimal_csv_columns(self):
        """Test with minimal CSV columns."""
        config = OutputConfig(csv_columns=["result"])
        assert len(config.csv_columns) >= 1


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_visualization_config(self):
        """Test default visualization configuration."""
        config = VisualizationConfig()

        assert isinstance(config.enabled_graphs, list)
        assert config.dpi > 0
        assert len(config.figsize) == 2

    def test_custom_visualization_config(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            enabled_graphs=["pass_rates", "score_distribution"],
            dpi=150,
            figsize=(12, 8),
        )

        assert "pass_rates" in config.enabled_graphs
        assert "score_distribution" in config.enabled_graphs
        assert config.dpi == 150
        assert config.figsize == [12, 8]  # Pydantic converts tuple to list

    def test_dpi_validation(self):
        """Test DPI validation."""
        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=0)


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.source_level is not None
        assert config.package_level is not None
        assert isinstance(config.package_overrides, dict)

    def test_custom_logging_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            source_level="DEBUG",
            package_level="ERROR",
            package_overrides={"httpx": "CRITICAL"},
        )

        assert config.source_level == "DEBUG"
        assert config.package_level == "ERROR"
        assert config.package_overrides["httpx"] == "CRITICAL"

    def test_show_timestamps_toggle(self):
        """Test show_timestamps configuration."""
        config1 = LoggingConfig(show_timestamps=True)
        config2 = LoggingConfig(show_timestamps=False)

        assert config1.show_timestamps is True
        assert config2.show_timestamps is False

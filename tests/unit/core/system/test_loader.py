"""Unit tests for core system loader module."""

import tempfile
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import SystemConfig
from lightspeed_evaluation.core.storage import get_file_config
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.core.system.loader import ConfigLoader


class TestConfigLoader:
    """Unit tests for ConfigLoader."""

    def test_load_default_system_yaml_loads_config_successfully(self) -> None:
        """Test that default config/system.yaml can be loaded without validation errors."""
        default_system_yaml_path = Path(__file__).parents[4] / "config" / "system.yaml"
        assert (
            default_system_yaml_path.exists()
        ), f"system.yaml not found at {default_system_yaml_path}"

        loader = ConfigLoader()

        # This should not raise any validation errors
        config = loader.load_system_config(str(default_system_yaml_path))

        # Basic sanity checks
        assert config is not None
        assert loader.system_config is not None

    def test_load_system_config_file_not_found(self) -> None:
        """Test loading non-existent config file raises error."""
        loader = ConfigLoader()

        with pytest.raises(ValueError, match="file not found"):
            loader.load_system_config("/nonexistent/config.yaml")

    def test_load_system_config_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: [[[")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(ValueError, match="Invalid YAML syntax"):
                loader.load_system_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_empty_file(self) -> None:
        """Test loading empty YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(ValueError, match="Empty or invalid"):
                loader.load_system_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_not_dict(self) -> None:
        """Test loading YAML with non-dict root raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(ValueError, match="must be a dictionary"):
                loader.load_system_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_minimal_valid(self) -> None:
        """Test loading minimal valid config."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

metrics_metadata:
  turn_level:
    ragas:faithfulness:
      threshold: 0.7
      default: true
      description: "Test metric"
  conversation_level: {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)

            assert config is not None
            assert loader.system_config is not None
            assert loader.logger is not None
            assert config.llm.provider == "openai"
            assert config.llm.model == "gpt-4o-mini"
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_with_all_sections(self) -> None:
        """Test loading config with all sections."""
        yaml_content = """
core:
  max_threads: 4

llm:
  provider: openai
  model: gpt-4
  temperature: 0.7

embedding:
  provider: openai
  model: text-embedding-3-small

api:
  enabled: false

storage:
  - type: "file"
    output_dir: ./test_output
    enabled_outputs:
      - csv
      - json

logging:
  source_level: DEBUG
  package_level: WARNING

visualization:
  figsize: [10, 6]
  dpi: 200

metrics_metadata:
  turn_level:
    ragas:faithfulness:
      threshold: 0.7
      default: true
      description: "Test"
  conversation_level:
    deepeval:conversation_completeness:
      threshold: 0.6
      default: true
      description: "Test"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)

            assert config.core.max_threads == 4
            assert config.llm.provider == "openai"
            assert config.llm.model == "gpt-4"
            assert config.llm.temperature == 0.7
            assert config.embedding.provider == "openai"
            assert config.api.enabled is False
            file_config = get_file_config(config.storage)
            assert file_config.output_dir == "./test_output"
            assert "csv" in file_config.enabled_outputs
            assert config.logging.source_level == "DEBUG"
            assert config.visualization.figsize == [10, 6]
            assert config.visualization.dpi == 200
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_populates_metrics_on_config(self) -> None:
        """Test that loading config populates metric names on SystemConfig."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

metrics_metadata:
  turn_level:
    ragas:faithfulness:
      threshold: 0.7
      default: true
      description: "Test"
    custom:answer_correctness:
      threshold: 0.8
      default: false
      description: "Test"
  conversation_level:
    deepeval:completeness:
      threshold: 0.6
      default: true
      description: "Test"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)

            # Metric names are now computed properties on SystemConfig
            assert "ragas:faithfulness" in config.turn_level_metric_names
            assert "custom:answer_correctness" in config.turn_level_metric_names
            assert "deepeval:completeness" in config.conversation_level_metric_names

            # Check config has metadata
            assert "ragas:faithfulness" in config.default_turn_metrics_metadata
            assert (
                "deepeval:completeness" in config.default_conversation_metrics_metadata
            )
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_with_defaults(self) -> None:
        """Test that missing sections use defaults."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

metrics_metadata:
  turn_level: {}
  conversation_level: {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)

            # Check defaults are applied
            assert config.llm.temperature == 0.0  # Default
            assert config.llm.max_tokens == 512  # Default
            assert get_file_config(config.storage).output_dir == "./eval_output"
            assert config.logging.show_timestamps is True  # Default
        finally:
            Path(temp_path).unlink()

    def test_create_system_config_missing_metrics_metadata(self) -> None:
        """Test creating config when metrics_metadata is missing."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)

            # Should handle missing metrics_metadata gracefully
            assert not config.default_turn_metrics_metadata
            assert not config.default_conversation_metrics_metadata
        finally:
            Path(temp_path).unlink()

    def test_create_system_config_partial_metrics_metadata(self) -> None:
        """Test creating config with partial metrics_metadata."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

metrics_metadata:
  turn_level:
    ragas:faithfulness:
      threshold: 0.7
      default: true
      description: "Test"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)

            # Should handle missing conversation_level
            assert len(config.default_turn_metrics_metadata) > 0
            assert not config.default_conversation_metrics_metadata
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_empty_sections(self) -> None:
        """Test loading config with empty sections."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

core: {}
api: {}
storage: []
logging: {}

metrics_metadata:
  turn_level: {}
  conversation_level: {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)

            # Should use defaults for empty sections
            assert config.core.max_threads is None
            assert config.api.enabled is True  # Default is True
            assert get_file_config(config.storage).output_dir == "./eval_output"
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_unknown_storage_type_raises(self) -> None:
        """Unknown storage backend type must fail fast."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

storage:
  - type: mongo
    database: x

metrics_metadata:
  turn_level: {}
  conversation_level: {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(
                ConfigurationError, match="Unknown storage backend type"
            ):
                loader.load_system_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_invalid_geval_metadata_fails(self) -> None:
        """Test that invalid GEval metadata in system config causes load to fail."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

metrics_metadata:
  turn_level:
    geval:bad_metric: {}
  conversation_level: {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            with pytest.raises(ConfigurationError) as exc_info:
                loader.load_system_config(temp_path)
            # GEval requires non-empty criteria; validator wraps as ConfigurationError
            assert (
                "criteria" in str(exc_info.value).lower()
                or "geval" in str(exc_info.value).lower()
            )
        finally:
            Path(temp_path).unlink()


class TestConfigLoaderFromConfig:
    """Unit tests for ConfigLoader.from_config classmethod."""

    def test_from_config_sets_system_config(self) -> None:
        """Test that from_config sets the system_config attribute."""
        config = SystemConfig()
        loader = ConfigLoader.from_config(config)

        assert loader.system_config is config

    def test_from_config_calls_setup_logging(self, mocker: MockerFixture) -> None:
        """Test that from_config calls setup_logging."""
        mock_setup_logging = mocker.patch(
            "lightspeed_evaluation.core.system.loader.setup_logging"
        )
        config = SystemConfig()

        ConfigLoader.from_config(config)

        mock_setup_logging.assert_called_once_with(config.logging)

    def test_from_config_calls_setup_environment_variables(
        self, mocker: MockerFixture
    ) -> None:
        """Test that from_config calls setup_environment_variables."""
        mock_setup_env = mocker.patch(
            "lightspeed_evaluation.core.system.loader.setup_environment_variables"
        )
        config = SystemConfig()

        ConfigLoader.from_config(config)

        mock_setup_env.assert_called_once()

    def test_from_config_no_longer_calls_populate_metric_mappings(self) -> None:
        """Test that from_config does not call populate_metric_mappings.

        Metric sets are now derived from SystemConfig properties, so
        the old global populate step is no longer needed.
        """
        config = SystemConfig(
            default_turn_metrics_metadata={"ragas:faithfulness": {"threshold": 0.7}},
        )
        loader = ConfigLoader.from_config(config)

        assert loader.system_config is config
        assert "ragas:faithfulness" in config.turn_level_metric_names

    def test_from_config_builds_ssl_config_data(self, mocker: MockerFixture) -> None:
        """Test that from_config builds config data with SSL fields for env setup."""
        mock_setup_env = mocker.patch(
            "lightspeed_evaluation.core.system.loader.setup_environment_variables"
        )
        config = SystemConfig()
        config.llm.ssl_verify = True
        config.llm.ssl_cert_file = "/path/to/cert.pem"

        ConfigLoader.from_config(config)

        config_data = mock_setup_env.call_args[0][0]
        assert config_data["llm"]["ssl_verify"] is True
        assert config_data["llm"]["ssl_cert_file"] == "/path/to/cert.pem"

    def test_from_config_sets_logger(self, mocker: MockerFixture) -> None:
        """Test that from_config sets the logger attribute."""
        mock_logger = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.system.loader.setup_logging",
            return_value=mock_logger,
        )
        config = SystemConfig()

        loader = ConfigLoader.from_config(config)

        assert loader.logger is mock_logger

    def test_from_config_returns_config_loader_instance(self) -> None:
        """Test that from_config returns a ConfigLoader instance."""
        config = SystemConfig()
        loader = ConfigLoader.from_config(config)

        assert isinstance(loader, ConfigLoader)


class TestConfigLoaderQualityScore:
    """Unit tests for ConfigLoader quality_score feature."""

    def test_quality_score_absent_results_in_none(self) -> None:
        """Test missing quality_score section results in None."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = ConfigLoader().load_system_config(temp_path)
            assert config.quality_score is None
        finally:
            Path(temp_path).unlink()

    def test_quality_score_config_loaded(self) -> None:
        """Test quality_score section is parsed and attached to SystemConfig."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

metrics_metadata:
  turn_level:
    ragas:faithfulness:
      threshold: 0.7
      default: true
  conversation_level: {}

quality_score:
  metrics:
    - ragas:faithfulness
  default: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = ConfigLoader().load_system_config(temp_path)
            assert config.quality_score is not None
            assert config.quality_score.metrics == ["ragas:faithfulness"]
            assert config.quality_score.default is False
            # Metric default is untouched
            assert (
                config.default_turn_metrics_metadata["ragas:faithfulness"]["default"]
                is True
            )
        finally:
            Path(temp_path).unlink()

    def test_quality_score_default_true_sets_default_on_metrics(self) -> None:
        """Test quality_score.default: true sets default: true on turn-level metrics."""
        yaml_content = """
llm:
  provider: openai

metrics_metadata:
  turn_level:
    ragas:faithfulness:
      threshold: 0.7
      default: true
    custom:correctness:
      threshold: 0.8
      default: false
    other:timeliness:
      threshold: 0.75
      default: false
  conversation_level:
    deepeval:completeness:
      threshold: 0.6
      default: false

quality_score:
  metrics:
    - ragas:faithfulness
    - custom:correctness
  default: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = ConfigLoader().load_system_config(temp_path)
            assert (
                config.default_turn_metrics_metadata["ragas:faithfulness"]["default"]
                is True
            )
            assert (
                config.default_turn_metrics_metadata["custom:correctness"]["default"]
                is True
            )
            assert (
                config.default_turn_metrics_metadata["other:timeliness"]["default"]
                is False
            )
            assert (
                config.default_conversation_metrics_metadata["deepeval:completeness"][
                    "default"
                ]
                is False
            )
        finally:
            Path(temp_path).unlink()

    def test_quality_score_default_true_with_undefined_metric_fails(self) -> None:
        """Test quality_score.default: true with undefined metric raises ConfigurationError."""
        yaml_content = """
llm:
  provider: openai

metrics_metadata:
  turn_level: {}
  conversation_level: {}

quality_score:
  metrics:
    - nonexistent:metric
  default: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(
                ConfigurationError,
                match=(
                    "Metric 'nonexistent:metric' is listed in "
                    "quality_score.metrics but not defined"
                ),
            ):
                ConfigLoader().load_system_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestConfigLoaderAgents:
    """Tests for agents configuration loading."""

    def test_load_yaml_with_agents_block(self) -> None:
        """YAML with agents: block is parsed correctly."""
        yaml_content = """
api:
  enabled: false
agents:
  default:
    agent:
      - ols_api
  ols_api:
    type: http_api
    api_base: http://localhost:8080
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)
            assert config.agents is not None
            assert config.agents.default.agent == ["ols_api"]
            assert "ols_api" in config.agents.agents
        finally:
            Path(temp_path).unlink()

    def test_load_yaml_with_string_agent_backward_compat(self) -> None:
        """YAML with string agent is auto-converted to list."""
        yaml_content = """
api:
  enabled: false
agents:
  default:
    agent: ols_api
  ols_api:
    type: http_api
    api_base: http://localhost:8080
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)
            assert config.agents is not None
            assert config.agents.default.agent == ["ols_api"]
        finally:
            Path(temp_path).unlink()

    def test_load_yaml_api_only_auto_migrates(self) -> None:
        """Existing api-only YAML auto-migrates to agents."""
        yaml_content = """
api:
  enabled: true
  api_base: http://legacy:8080
  timeout: 500
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_system_config(temp_path)
            assert config.agents is not None
            assert config.agents.default.agent == ["http_api"]
            assert config.agents.agents["http_api"].api_base == "http://legacy:8080"
            assert config.agents.agents["http_api"].timeout == 500
            # api field also preserved
            assert config.api.enabled is True
            assert config.api.api_base == "http://legacy:8080"
        finally:
            Path(temp_path).unlink()

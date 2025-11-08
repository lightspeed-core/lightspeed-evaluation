"""Unit tests for core system loader module."""

import tempfile
from pathlib import Path

import pytest

from lightspeed_evaluation.core.system.loader import (
    ConfigLoader,
    populate_metric_mappings,
    TURN_LEVEL_METRICS,
    CONVERSATION_LEVEL_METRICS,
)
from lightspeed_evaluation.core.models import SystemConfig


class TestPopulateMetricMappings:
    """Unit tests for populate_metric_mappings function."""

    def test_populate_metric_mappings_turn_level(self):
        """Test populating turn-level metrics."""
        config = SystemConfig()
        config.default_turn_metrics_metadata = {
            "ragas:faithfulness": {"threshold": 0.7},
            "custom:answer_correctness": {"threshold": 0.8},
        }
        config.default_conversation_metrics_metadata = {}

        populate_metric_mappings(config)

        assert "ragas:faithfulness" in TURN_LEVEL_METRICS
        assert "custom:answer_correctness" in TURN_LEVEL_METRICS

    def test_populate_metric_mappings_conversation_level(self):
        """Test populating conversation-level metrics."""
        config = SystemConfig()
        config.default_turn_metrics_metadata = {}
        config.default_conversation_metrics_metadata = {
            "deepeval:conversation_completeness": {"threshold": 0.6},
            "deepeval:conversation_relevancy": {"threshold": 0.7},
        }

        populate_metric_mappings(config)

        assert "deepeval:conversation_completeness" in CONVERSATION_LEVEL_METRICS
        assert "deepeval:conversation_relevancy" in CONVERSATION_LEVEL_METRICS

    def test_populate_metric_mappings_clears_previous(self):
        """Test that populate clears previous mappings."""
        config1 = SystemConfig()
        config1.default_turn_metrics_metadata = {"metric1": {}}
        config1.default_conversation_metrics_metadata = {}

        populate_metric_mappings(config1)
        assert "metric1" in TURN_LEVEL_METRICS

        # Populate with different config
        config2 = SystemConfig()
        config2.default_turn_metrics_metadata = {"metric2": {}}
        config2.default_conversation_metrics_metadata = {}

        populate_metric_mappings(config2)

        # Should only have metric2 now
        assert "metric2" in TURN_LEVEL_METRICS
        # metric1 should be cleared
        assert "metric1" not in TURN_LEVEL_METRICS


class TestConfigLoader:
    """Unit tests for ConfigLoader."""

    def test_load_system_config_file_not_found(self):
        """Test loading non-existent config file raises error."""
        loader = ConfigLoader()

        with pytest.raises(ValueError, match="file not found"):
            loader.load_system_config("/nonexistent/config.yaml")

    def test_load_system_config_invalid_yaml(self):
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

    def test_load_system_config_empty_file(self):
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

    def test_load_system_config_not_dict(self):
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

    def test_load_system_config_minimal_valid(self):
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

    def test_load_system_config_with_all_sections(self):
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

output:
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
            assert config.output.output_dir == "./test_output"
            assert "csv" in config.output.enabled_outputs
            assert config.logging.source_level == "DEBUG"
            assert config.visualization.figsize == [10, 6]
            assert config.visualization.dpi == 200
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_populates_metrics(self):
        """Test that loading config populates global metric mappings."""
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

            # Check that metrics were populated
            assert "ragas:faithfulness" in TURN_LEVEL_METRICS
            assert "custom:answer_correctness" in TURN_LEVEL_METRICS
            assert "deepeval:completeness" in CONVERSATION_LEVEL_METRICS

            # Check config has metadata
            assert "ragas:faithfulness" in config.default_turn_metrics_metadata
            assert (
                "deepeval:completeness" in config.default_conversation_metrics_metadata
            )
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_with_defaults(self):
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
            assert config.output.output_dir == "./eval_output"  # Default
            assert config.logging.show_timestamps is True  # Default
        finally:
            Path(temp_path).unlink()

    def test_create_system_config_missing_metrics_metadata(self):
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
            assert config.default_turn_metrics_metadata == {}
            assert config.default_conversation_metrics_metadata == {}
        finally:
            Path(temp_path).unlink()

    def test_create_system_config_partial_metrics_metadata(self):
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
            assert config.default_conversation_metrics_metadata == {}
        finally:
            Path(temp_path).unlink()

    def test_load_system_config_empty_sections(self):
        """Test loading config with empty sections."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o-mini

core: {}
api: {}
output: {}
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
            assert config.output.output_dir == "./eval_output"
        finally:
            Path(temp_path).unlink()

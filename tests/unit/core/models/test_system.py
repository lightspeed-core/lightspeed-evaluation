"""Tests for system configuration models."""

import os
import tempfile
import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import (
    JudgePanelConfig,
    LLMConfig,
    LLMPoolConfig,
    SystemConfig,
    EmbeddingConfig,
    APIConfig,
    VisualizationConfig,
)
from lightspeed_evaluation.core.storage import FileBackendConfig
from lightspeed_evaluation.core.models.system import (
    GEvalConfig,
    GEvalRubricConfig,
    LLMDefaultsConfig,
    LLMParametersConfig,
    LLMProviderConfig,
    LoggingConfig,
)
from lightspeed_evaluation.core.system.exceptions import ConfigurationError


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_defaults_and_validation(self) -> None:
        """Test default values and field validations."""
        # Test defaults
        config = LLMConfig()
        assert config.ssl_verify is True
        assert config.ssl_cert_file is None

        # Test validation bounds
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValidationError):
            LLMConfig(timeout=0)
        with pytest.raises(ValidationError):
            LLMConfig(num_retries=-1)

    def test_parameters_auto_populated_from_fields(self) -> None:
        """Legacy path: parameters dict is built from explicit fields."""
        config = LLMConfig(temperature=0.3, max_tokens=1024)
        assert config.parameters == {
            "temperature": 0.3,
            "max_completion_tokens": 1024,
        }

    def test_parameters_from_pool_path(self) -> None:
        """Pool path: model_copy sets parameters with extras."""
        config = LLMConfig(temperature=0.5, max_tokens=2048)
        params = {"temperature": 0.5, "max_completion_tokens": 2048, "top_p": 0.9}
        config = config.model_copy(update={"parameters": params})
        assert config.parameters == params
        assert config.parameters["top_p"] == 0.9

    def test_parameters_stripped_from_yaml_input(self) -> None:
        """User-provided parameters in YAML is stripped, rebuilt from fields."""
        config = LLMConfig(
            temperature=0.3,
            max_tokens=1024,
            parameters={"top_p": 0.9},
        )
        assert "top_p" not in config.parameters
        assert config.parameters == {
            "temperature": 0.3,
            "max_completion_tokens": 1024,
        }

    def test_ssl_cert_file_handling(self, mocker: MockerFixture) -> None:
        """Test ssl_cert_file validation and path expansion."""
        # Create temp cert file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".crt", delete=False) as f:
            cert_path = f.name
            f.write("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----\n")

        try:
            # Valid path - converts to absolute
            config = LLMConfig(ssl_cert_file=cert_path)
            assert config.ssl_cert_file == os.path.abspath(cert_path)

            # Environment variable expansion
            test_dir = os.path.dirname(cert_path)
            test_filename = os.path.basename(cert_path)
            mocker.patch.dict(os.environ, {"TEST_CERT_DIR": test_dir})
            config = LLMConfig(ssl_cert_file=f"$TEST_CERT_DIR/{test_filename}")
            assert config.ssl_cert_file == os.path.abspath(cert_path)
        finally:
            os.unlink(cert_path)

        # Non-existent file fails
        with pytest.raises(ConfigurationError, match="(?i)not found"):
            LLMConfig(ssl_cert_file="/tmp/nonexistent_cert_12345.crt")

        # Directory fails
        with pytest.raises(ConfigurationError):
            LLMConfig(ssl_cert_file=tempfile.gettempdir())


class TestBasicConfigModels:
    """Tests for basic config models."""

    def test_embedding_config(self) -> None:
        """Test EmbeddingConfig defaults and custom values."""
        default = EmbeddingConfig()
        assert default.provider is not None
        assert default.cache_enabled is True

        custom = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
        assert custom.provider == "openai"
        assert custom.model == "text-embedding-3-small"

    def test_api_config(self) -> None:
        """Test APIConfig defaults and validation."""
        default = APIConfig()
        assert isinstance(default.enabled, bool)
        assert default.timeout > 0

        custom = APIConfig(enabled=True, api_base="https://custom.api.com", timeout=300)
        assert custom.api_base == "https://custom.api.com"

        with pytest.raises(ValidationError):
            APIConfig(timeout=0)

    def test_file_backend_config(self) -> None:
        """Test FileBackendConfig defaults and custom values."""
        default = FileBackendConfig()
        assert "csv" in default.enabled_outputs
        assert len(default.csv_columns) > 0

        custom = FileBackendConfig(enabled_outputs=["json"], csv_columns=["result"])
        assert custom.enabled_outputs == ["json"]

    def test_visualization_config(self) -> None:
        """Test VisualizationConfig defaults and validation."""
        default = VisualizationConfig()
        assert default.dpi > 0
        assert len(default.figsize) == 2

        custom = VisualizationConfig(dpi=150, figsize=[12, 8])
        assert custom.dpi == 150

        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=0)

    def test_logging_config(self) -> None:
        """Test LoggingConfig defaults and custom values."""
        default = LoggingConfig()
        assert default.source_level is not None

        custom = LoggingConfig(
            source_level="DEBUG",
            package_overrides={"httpx": "CRITICAL"},
            show_timestamps=True,
        )
        assert custom.source_level == "DEBUG"
        assert custom.package_overrides["httpx"] == "CRITICAL"


class TestLLMParametersConfig:
    """Tests for LLMParametersConfig model."""

    def test_defaults_and_validation(self) -> None:
        """Test defaults and field validations."""
        params = LLMParametersConfig()
        assert params.temperature is None
        assert params.max_completion_tokens is None

        # Validation
        with pytest.raises(ValidationError):
            LLMParametersConfig(temperature=2.5)
        with pytest.raises(ValidationError):
            LLMParametersConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMParametersConfig(max_completion_tokens=0)

    def test_extra_params_and_to_dict(self) -> None:
        """Test extra parameters allowed and to_dict method."""
        params = LLMParametersConfig.model_validate(
            {"temperature": 0.5, "top_p": 0.9, "frequency_penalty": 0.5}
        )
        assert params.temperature == 0.5
        # Access extra fields via model_dump
        dump = params.model_dump()
        assert dump["top_p"] == 0.9
        assert dump["frequency_penalty"] == 0.5

        # to_dict excludes None by default
        result = params.to_dict()
        assert "temperature" in result
        assert "max_completion_tokens" not in result

        # to_dict(exclude_none=False) returns only explicitly set fields
        result = params.to_dict(exclude_none=False)
        assert "max_completion_tokens" not in result  # not set, not included
        assert result["temperature"] == 0.5

        # Explicitly setting a field to None includes it (for override/drop)
        params_with_null = LLMParametersConfig.model_validate(
            {"temperature": None, "top_p": 0.9}
        )
        result = params_with_null.to_dict(exclude_none=False)
        assert result["temperature"] is None  # explicitly set to None
        assert result["top_p"] == 0.9


class TestLLMProviderConfig:
    """Tests for LLMProviderConfig model."""

    def test_minimal_and_full_config(self) -> None:
        """Test minimal required fields and full config."""
        # Minimal - only provider is required
        entry = LLMProviderConfig(provider="openai")
        assert entry.provider == "openai"
        assert entry.model is None
        assert entry.ssl_verify is None
        assert entry.api_base is None
        assert entry.api_key_path is None
        assert entry.timeout is None

        # Full config with all fields
        entry = LLMProviderConfig(
            provider="hosted_vllm",
            model="gpt-oss-20b",
            ssl_verify=True,
            api_base="https://vllm.example.com/v1",
            api_key_path="/secrets/key.txt",
            parameters=LLMParametersConfig(temperature=0.5),
            timeout=600,
        )
        assert entry.model == "gpt-oss-20b"
        assert entry.api_base == "https://vllm.example.com/v1"
        assert entry.api_key_path == "/secrets/key.txt"
        assert entry.parameters.temperature == 0.5

    def test_provider_required(self) -> None:
        """Test that provider field is required."""
        with pytest.raises(ValidationError):
            LLMProviderConfig.model_validate({})


class TestLLMPoolConfig:
    """Tests for LLMPoolConfig model."""

    def test_pool_basics(self) -> None:
        """Test pool creation and model ID retrieval."""
        pool = LLMPoolConfig(
            models={
                "gpt-4o-mini": LLMProviderConfig(provider="openai"),
                "gpt-4o": LLMProviderConfig(provider="openai"),
            }
        )
        assert pool.defaults is not None
        assert len(pool.get_model_ids()) == 2

    def test_resolve_llm_config(self) -> None:
        """Test resolving LLMConfig with defaults and overrides."""
        pool = LLMPoolConfig(
            defaults=LLMDefaultsConfig(
                timeout=600,
                cache_dir=".caches/llm",
                parameters=LLMParametersConfig(
                    temperature=0.1, max_completion_tokens=512
                ),
            ),
            models={
                "gpt-4o-mini": LLMProviderConfig(provider="openai"),
                "gpt-4o": LLMProviderConfig(
                    provider="openai",
                    parameters=LLMParametersConfig(
                        temperature=0.5, max_completion_tokens=2048
                    ),
                    timeout=300,
                ),
            },
        )

        # Uses defaults
        resolved = pool.resolve_llm_config("gpt-4o-mini")
        assert isinstance(resolved, LLMConfig)
        assert resolved.provider == "openai"
        assert resolved.model == "gpt-4o-mini"  # Defaults to key
        assert resolved.timeout == 600
        assert resolved.temperature == 0.1
        assert resolved.cache_dir == ".caches/llm/gpt-4o-mini"

        # With cache suffix
        resolved = pool.resolve_llm_config("gpt-4o-mini", cache_suffix="judge_0")
        assert resolved.cache_dir == ".caches/llm/judge_0"

        # Overrides take precedence
        resolved = pool.resolve_llm_config("gpt-4o")
        assert resolved.temperature == 0.5
        assert resolved.max_tokens == 2048
        assert resolved.timeout == 300

        # Unknown model raises error
        with pytest.raises(ConfigurationError, match="Model 'unknown' not found"):
            pool.resolve_llm_config("unknown")

    def test_resolve_llm_config_passes_extras(self) -> None:
        """Extra parameters from LLMParametersConfig flow into LLMConfig.parameters."""
        pool = LLMPoolConfig(
            defaults=LLMDefaultsConfig(
                parameters=LLMParametersConfig.model_validate(
                    {"temperature": 0.1, "max_completion_tokens": 512, "top_p": 0.9}
                ),
            ),
            models={
                "gpt-4o-mini": LLMProviderConfig(
                    provider="openai",
                    parameters=LLMParametersConfig.model_validate(
                        {"frequency_penalty": 0.5}
                    ),
                ),
            },
        )
        resolved = pool.resolve_llm_config("gpt-4o-mini")

        # Explicit fields set from merged params
        assert resolved.temperature == 0.1
        assert resolved.max_tokens == 512

        # Extras present in parameters dict
        assert resolved.parameters["top_p"] == 0.9
        assert resolved.parameters["frequency_penalty"] == 0.5
        assert resolved.parameters["temperature"] == 0.1
        assert resolved.parameters["max_completion_tokens"] == 512

    def test_resolve_llm_config_individual_overrides_default_extras(self) -> None:
        """Individual model extras override default extras."""
        pool = LLMPoolConfig(
            defaults=LLMDefaultsConfig(
                parameters=LLMParametersConfig.model_validate({"top_p": 0.9}),
            ),
            models={
                "model-a": LLMProviderConfig(
                    provider="openai",
                    parameters=LLMParametersConfig.model_validate({"top_p": 0.5}),
                ),
            },
        )
        resolved = pool.resolve_llm_config("model-a")
        assert resolved.parameters["top_p"] == 0.5

    def test_resolve_llm_config_none_drops_default(self) -> None:
        """Individual model setting a param to null removes the default value."""
        pool = LLMPoolConfig(
            defaults=LLMDefaultsConfig(
                parameters=LLMParametersConfig.model_validate(
                    {"temperature": 0.3, "top_p": 0.9}
                ),
            ),
            models={
                "model-a": LLMProviderConfig(
                    provider="openai",
                    parameters=LLMParametersConfig.model_validate(
                        {"temperature": None}
                    ),
                ),
            },
        )
        resolved = pool.resolve_llm_config("model-a")
        # temperature was explicitly nulled -> absent from parameters (not sent to LLM)
        assert "temperature" not in resolved.parameters
        # top_p from defaults survives
        assert resolved.parameters["top_p"] == 0.9

    def test_custom_model_id_and_ssl(self) -> None:
        """Test custom model IDs and SSL settings."""
        pool = LLMPoolConfig(
            models={
                "gpt-4o-eval": LLMProviderConfig(
                    provider="openai",
                    model="gpt-4o",  # Actual model differs from key
                    parameters=LLMParametersConfig(temperature=0.0),
                ),
                "gpt-oss-prod": LLMProviderConfig(
                    provider="hosted_vllm", model="gpt-oss-20b", ssl_verify=True
                ),
                "gpt-oss-staging": LLMProviderConfig(
                    provider="hosted_vllm", model="gpt-oss-20b", ssl_verify=False
                ),
            }
        )

        # Custom model ID
        eval_config = pool.resolve_llm_config("gpt-4o-eval")
        assert eval_config.model == "gpt-4o"
        assert eval_config.temperature == 0.0

        # SSL settings
        assert pool.resolve_llm_config("gpt-oss-prod").ssl_verify is True
        assert pool.resolve_llm_config("gpt-oss-staging").ssl_verify is False

    def test_forbidden_keys_rejected_at_load_time(self) -> None:
        """Forbidden keys in parameters raise ConfigurationError at config load."""
        # Direct LLMParametersConfig
        with pytest.raises(ConfigurationError, match="Keys not allowed"):
            LLMParametersConfig.model_validate({"timeout": 60})

        with pytest.raises(ConfigurationError, match="Keys not allowed"):
            LLMParametersConfig.model_validate({"provider": "openai"})

        # At defaults level
        with pytest.raises(ConfigurationError, match="Keys not allowed"):
            LLMDefaultsConfig(
                parameters=LLMParametersConfig.model_validate({"model": "gpt-4"})
            )

        # At model level
        with pytest.raises(ConfigurationError, match="Keys not allowed"):
            LLMProviderConfig(
                provider="openai",
                parameters=LLMParametersConfig.model_validate({"num_retries": 5}),
            )


class TestJudgePanelConfig:
    """Tests for JudgePanelConfig model."""

    def test_valid_configurations(self) -> None:
        """Test valid judge panel configurations."""
        # Single judge
        panel = JudgePanelConfig(judges=["gpt-4o-mini"])
        assert len(panel.judges) == 1
        assert panel.enabled_metrics is None
        assert panel.aggregation_strategy == "max"  # Default is max

        # Multiple judges with metrics
        panel = JudgePanelConfig(
            judges=["gpt-4o-mini", "gpt-4o"],
            enabled_metrics=["ragas:faithfulness", "custom:correctness"],
            aggregation_strategy="max",
        )
        assert len(panel.judges) == 2
        assert panel.enabled_metrics is not None
        assert len(panel.enabled_metrics) == 2

        for strategy in ["max", "average", "majority_vote"]:
            panel = JudgePanelConfig(
                judges=["gpt-4o-mini"], aggregation_strategy=strategy
            )
            assert panel.aggregation_strategy == strategy

    def test_invalid_configurations(self) -> None:
        """Test invalid configurations are rejected."""
        # Empty judges
        with pytest.raises(ValidationError, match="(?i)at least 1 item"):
            JudgePanelConfig(judges=[])

        # Dict format rejected (must be string IDs)
        with pytest.raises(ValidationError):
            JudgePanelConfig.model_validate({"judges": [{"provider": "openai"}]})

        # Invalid metric format - no colon
        with pytest.raises(ValidationError, match="framework:metric_name"):
            JudgePanelConfig(judges=["gpt-4o-mini"], enabled_metrics=["invalid"])

        # Invalid metric format - empty parts
        with pytest.raises(ValidationError, match="framework:metric_name"):
            JudgePanelConfig(judges=["gpt-4o-mini"], enabled_metrics=[":metric"])
        with pytest.raises(ValidationError, match="framework:metric_name"):
            JudgePanelConfig(judges=["gpt-4o-mini"], enabled_metrics=["framework:"])

        # Invalid aggregation strategy
        with pytest.raises(ValidationError, match="(?i)aggregation_strategy"):
            JudgePanelConfig(judges=["gpt-4o-mini"], aggregation_strategy="invalid")


class TestSystemConfigWithLLMPoolAndJudgePanel:
    """Tests for SystemConfig with llm_pool and judge_panel."""

    def test_new_fields_are_optional(self) -> None:
        """Test SystemConfig works without new fields (llm_pool, judge_panel)."""
        config = SystemConfig()
        assert config.llm is not None  # Existing field still works
        assert config.llm_pool is None
        assert config.judge_panel is None

    def test_with_pool_and_panel(self) -> None:
        """Test SystemConfig with llm_pool and judge_panel."""
        pool = LLMPoolConfig(
            defaults=LLMDefaultsConfig(
                parameters=LLMParametersConfig(
                    temperature=0.0, max_completion_tokens=512
                )
            ),
            models={
                "gpt-4o-mini": LLMProviderConfig(provider="openai"),
                "gpt-4o": LLMProviderConfig(
                    provider="openai",
                    parameters=LLMParametersConfig(max_completion_tokens=1024),
                ),
            },
        )
        panel = JudgePanelConfig(judges=["gpt-4o-mini", "gpt-4o"])

        config = SystemConfig(llm_pool=pool, judge_panel=panel)

        # Pool and panel configured
        assert config.llm_pool is not None
        assert config.judge_panel is not None

        # get_judge_configs returns tuples of (pool_key, resolved_config)
        judge_configs = config.get_judge_configs()
        assert len(judge_configs) == 2
        # Each entry is a tuple of (pool_key, LLMConfig)
        assert all(
            isinstance(entry, tuple) and len(entry) == 2 for entry in judge_configs
        )
        assert all(isinstance(entry[1], LLMConfig) for entry in judge_configs)
        # Check pool keys match the judge panel judges list
        assert judge_configs[0][0] == "gpt-4o-mini"
        assert judge_configs[1][0] == "gpt-4o"
        # Check configs
        assert judge_configs[0][1].model == "gpt-4o-mini"
        assert judge_configs[0][1].cache_dir.endswith("judge_0")
        assert judge_configs[1][1].max_tokens == 1024
        assert judge_configs[1][1].cache_dir.endswith("judge_1")

        # get_llm_config works
        llm_config = config.get_llm_config("gpt-4o-mini")
        assert llm_config.provider == "openai"

    def test_error_branches(self) -> None:
        """Test error handling in get_judge_configs and get_llm_config."""
        # get_llm_config without pool raises ConfigurationError
        config = SystemConfig()
        with pytest.raises(ConfigurationError, match="llm_pool.*not configured"):
            config.get_llm_config("gpt-4o-mini")

        # get_judge_configs with panel but no pool raises ConfigurationError
        config = SystemConfig(judge_panel=JudgePanelConfig(judges=["gpt-4o-mini"]))
        with pytest.raises(ConfigurationError, match="llm_pool.*not defined"):
            config.get_judge_configs()

        # get_judge_configs with invalid judge ID raises error
        pool = LLMPoolConfig(
            models={"gpt-4o-mini": LLMProviderConfig(provider="openai")}
        )
        panel = JudgePanelConfig(judges=["gpt-4o-mini", "nonexistent"])
        config = SystemConfig(llm_pool=pool, judge_panel=panel)
        with pytest.raises(ConfigurationError, match="Model 'nonexistent' not found"):
            config.get_judge_configs()


class TestGEvalRubricValidation:
    """Tests for GEval rubric and config validation."""

    def test_rubric_score_range_valid(self) -> None:
        """Valid score_range [0, 10] and min <= max accepted."""
        r = GEvalRubricConfig(score_range=(0, 10), expected_outcome="Full marks.")
        assert r.score_range == (0, 10)
        r2 = GEvalRubricConfig(score_range=(3, 7), expected_outcome="Partial.")
        assert r2.score_range == (3, 7)

    def test_rubric_score_range_min_max_swapped_fails(self) -> None:
        """score_range with min > max fails."""
        with pytest.raises(ValueError, match="min must be <= max"):
            GEvalRubricConfig(score_range=(5, 3), expected_outcome="X")

    def test_rubric_score_range_out_of_bounds_fails(self) -> None:
        """score_range outside 0-10 fails."""
        with pytest.raises(ValueError, match="between 0 and 10"):
            GEvalRubricConfig(score_range=(-1, 5), expected_outcome="X")
        with pytest.raises(ValueError, match="between 0 and 10"):
            GEvalRubricConfig(score_range=(0, 11), expected_outcome="X")

    def test_geval_config_rubrics_non_overlapping(self) -> None:
        """Non-overlapping rubric ranges accepted."""
        config = GEvalConfig.from_metadata(
            {
                "criteria": "Check correctness.",
                "rubrics": [
                    {"score_range": [0, 3], "expected_outcome": "Low"},
                    {"score_range": [4, 7], "expected_outcome": "Mid"},
                    {"score_range": [8, 10], "expected_outcome": "High"},
                ],
            }
        )
        assert config.rubrics is not None
        assert len(config.rubrics) == 3

    def test_geval_config_rubrics_adjacent_non_overlapping_accepted(self) -> None:
        """Adjacent but non-overlapping ranges [0,3] and [4,7] are accepted."""
        config = GEvalConfig.from_metadata(
            {
                "criteria": "Check.",
                "rubrics": [
                    {"score_range": [0, 3], "expected_outcome": "Low"},
                    {"score_range": [4, 7], "expected_outcome": "High"},
                ],
            }
        )
        assert config.rubrics is not None
        assert len(config.rubrics) == 2
        assert config.rubrics[0].score_range == (0, 3)
        assert config.rubrics[1].score_range == (4, 7)

    def test_geval_config_rubrics_overlapping_fails(self) -> None:
        """Overlapping rubric ranges fail validation."""
        with pytest.raises(ConfigurationError, match="overlap"):
            GEvalConfig.from_metadata(
                {
                    "criteria": "Check.",
                    "rubrics": [
                        {"score_range": [0, 5], "expected_outcome": "A"},
                        {"score_range": [4, 10], "expected_outcome": "B"},
                    ],
                }
            )

    def test_system_config_geval_validation_error_wrapped_as_configuration_error(
        self,
    ) -> None:
        """ValidationError from GEval/rubric validation is wrapped as ConfigurationError.

        GEvalConfig.from_metadata can raise pydantic.ValidationError (e.g. from
        invalid rubric structure). The field validator catches it and re-raises
        ConfigurationError so callers get a consistent config-failure exception type.
        """
        with pytest.raises(
            ConfigurationError, match="Invalid GEval config for 'geval:bad'"
        ):
            SystemConfig(
                default_turn_metrics_metadata={
                    "geval:bad": {
                        "criteria": "Some criteria",
                        "rubrics": [
                            {"score_range": "not-a-list", "expected_outcome": "X"},
                        ],
                    },
                }
            )


class TestSystemConfigMetricNameProperties:
    """Tests for SystemConfig.turn_level_metric_names and conversation_level_metric_names."""

    def test_turn_level_metric_names_returns_correct_set(self) -> None:
        """Test turn_level_metric_names returns keys from metadata dict."""
        config = SystemConfig(
            default_turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.7},
                "custom:answer_correctness": {"threshold": 0.8},
            },
        )
        assert config.turn_level_metric_names == {
            "ragas:faithfulness",
            "custom:answer_correctness",
        }

    def test_conversation_level_metric_names_returns_correct_set(self) -> None:
        """Test conversation_level_metric_names returns keys from metadata dict."""
        config = SystemConfig(
            default_conversation_metrics_metadata={
                "deepeval:conversation_completeness": {"threshold": 0.6},
                "deepeval:conversation_relevancy": {"threshold": 0.7},
            },
        )
        assert config.conversation_level_metric_names == {
            "deepeval:conversation_completeness",
            "deepeval:conversation_relevancy",
        }

    def test_empty_metadata_returns_empty_sets(self) -> None:
        """Test that empty metadata dicts return empty sets."""
        config = SystemConfig()
        assert config.turn_level_metric_names == set()
        assert config.conversation_level_metric_names == set()

    def test_metric_names_are_derived_not_stored(self) -> None:
        """Test that metric names are computed properties, not stored state."""
        config = SystemConfig(
            default_turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.7},
            },
        )
        names1 = config.turn_level_metric_names
        names2 = config.turn_level_metric_names
        assert names1 == names2
        # Each call returns a fresh set (not the same object)
        assert names1 is not names2

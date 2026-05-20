"""LLM configuration models."""

import logging
import os
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from lightspeed_evaluation.core.constants import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_SSL_CERT_FILE,
    DEFAULT_SSL_VERIFY,
)
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Keys not allowed in llm_pool.parameters
# These are either request-envelope fields or managed via dedicated config fields
FORBIDDEN_PARAMETER_KEYS = frozenset(
    {
        # Request envelope fields
        "model",
        "messages",
        "n",
        # Operational fields (set via dedicated config)
        "timeout",
        "num_retries",
        "ssl_verify",
        "ssl_cert_file",
        "cache_enabled",
        "cache_dir",
        # Provider/client fields
        "provider",
        "client",
    }
)


class LLMConfig(BaseModel):
    """LLM configuration from system configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        min_length=1,
        description="Provider name, e.g., openai, azure, watsonx etc..",
    )
    model: str = Field(
        default=DEFAULT_LLM_MODEL,
        min_length=1,
        description="Model identifier or deployment name",
    )
    ssl_verify: bool = Field(
        default=DEFAULT_SSL_VERIFY,
        description="Verify SSL certificates for HTTPS connections. Can be True/False",
    )
    ssl_cert_file: Optional[str] = Field(
        default=DEFAULT_SSL_CERT_FILE,
        description="Path to custom CA certificate file for SSL verification",
    )

    @model_validator(mode="after")
    def validate_ssl_cert_file(self) -> "LLMConfig":
        """Validate SSL certificate file exists if provided."""
        if self.ssl_cert_file is not None:
            cert_path = self.ssl_cert_file

            # Expand environment variables and user paths
            cert_path = os.path.expandvars(os.path.expanduser(cert_path))

            # Check if file exists
            if not os.path.isfile(cert_path):
                raise ConfigurationError(
                    f"SSL certificate file not found: '{cert_path}'. "
                    f"Original path: '{self.ssl_cert_file}'. "
                    "Please provide a valid path to a CA certificate file "
                    "or set ssl_cert_file to null."
                )

            # Update to absolute path for consistency
            self.ssl_cert_file = os.path.abspath(cert_path)

        return self

    temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=DEFAULT_LLM_MAX_TOKENS, ge=1, description="Maximum tokens in response"
    )
    timeout: int = Field(
        default=DEFAULT_API_TIMEOUT, ge=1, description="Request timeout in seconds"
    )
    num_retries: int = Field(
        default=DEFAULT_LLM_RETRIES,
        ge=0,
        description="Retry attempts for failed requests",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Location of cached 'LLM as a judge' queries",
    )
    cache_enabled: bool = Field(
        default=True, description="Is caching of 'LLM as a judge' queries enabled?"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal: dynamic LLM parameters for API calls",
    )

    @model_validator(mode="before")
    @classmethod
    def strip_user_parameters(cls, data: Any) -> Any:
        """Strip parameters from YAML/dict input.

        Dynamic parameters via YAML are only supported through llm_pool.
        For legacy llm:, parameters is always built from explicit fields.
        """
        if isinstance(data, dict):
            data.pop("parameters", None)
        return data

    @model_validator(mode="after")
    def build_parameters_from_fields(self) -> "LLMConfig":
        """Build parameters dict from explicit temperature and max_tokens.

        For the pool path, resolve_llm_config() overrides this via
        model_copy(update=...) after construction.
        """
        self.parameters = {
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        return self


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default=DEFAULT_EMBEDDING_PROVIDER,
        min_length=1,
        description="Provider name, e.g., huggingface, openai",
    )
    model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model identifier",
    )
    provider_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description="Embedding provider arguments, e.g. model_kwargs: device:cpu",
    )
    cache_enabled: bool = Field(
        default=True, description="Is caching of embedding queries enabled?"
    )

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        allowed = {"openai", "huggingface", "gemini"}
        if v not in allowed:
            raise ValueError(
                f"Unsupported embedding provider '{v}'. Allowed: {sorted(allowed)}"
            )
        return v


class LLMParametersConfig(BaseModel):
    """Dynamic parameters passed to LLM API calls.

    These parameters are passed directly to the LLM provider.
    All fields are optional - unset fields inherit from parent level.
    Uses extra="allow" to pass through any provider-specific parameters.
    """

    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens in response",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_forbidden_keys(cls, data: Any) -> Any:
        """Reject keys that must be set via dedicated config fields, not parameters."""
        if not isinstance(data, dict):
            return data
        forbidden_found = FORBIDDEN_PARAMETER_KEYS & data.keys()
        if forbidden_found:
            raise ConfigurationError(
                f"Keys not allowed in parameters: {forbidden_found}. "
                f"Use dedicated config fields instead."
            )
        return data

    def to_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        """Convert parameters to dict for passing to LLM.

        Args:
            exclude_none: If True, exclude None values from output.
                If False, include only explicitly set fields (uses model_fields_set
                to distinguish user-provided None from unset defaults).

        Returns:
            Dict of parameters ready for LLM API call
        """
        if exclude_none:
            return {k: v for k, v in self.model_dump().items() if v is not None}
        # Include only fields the user explicitly set (including None overrides)
        return {
            k: v for k, v in self.model_dump().items() if k in self.model_fields_set
        }


class LLMDefaultsConfig(BaseModel):
    """Global default settings for all LLMs in the pool.

    These are shared defaults that apply to all LLMs unless overridden
    at the provider or model level.
    """

    model_config = ConfigDict(extra="forbid")

    cache_enabled: bool = Field(
        default=True,
        description="Is caching of LLM queries enabled?",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Base cache directory",
    )

    timeout: int = Field(
        default=DEFAULT_API_TIMEOUT,
        ge=1,
        description="Request timeout in seconds",
    )

    num_retries: int = Field(
        default=DEFAULT_LLM_RETRIES,
        ge=0,
        description="Retry attempts for failed requests",
    )

    # Default dynamic parameters
    parameters: LLMParametersConfig = Field(
        default_factory=lambda: LLMParametersConfig(
            temperature=DEFAULT_LLM_TEMPERATURE,
            max_completion_tokens=DEFAULT_LLM_MAX_TOKENS,
        ),
        description="Default dynamic parameters for LLM calls",
    )


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider/model in the pool.

    Contains model-specific settings. Cache and retry settings are managed
    at the pool defaults level, not per-model.

    The dict key is the unique model ID used for referencing.
    """

    model_config = ConfigDict(extra="forbid")

    # Required: Provider type
    provider: str = Field(
        min_length=1,
        description="Provider type (e.g., openai, watsonx, gemini, hosted_vllm)",
    )

    # Model identity (optional - defaults to dict key)
    model: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Actual model name. If not set, uses the dict key as model name.",
    )

    # SSL settings (optional - inherit from defaults or use system defaults)
    ssl_verify: Optional[bool] = Field(
        default=None,
        description="Verify SSL certificates. Inherits from defaults if not set.",
    )
    ssl_cert_file: Optional[str] = Field(
        default=None,
        description="Path to custom CA certificate file",
    )

    # API endpoint/key configuration (optional - falls back to environment variable)
    api_base: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Base URL for the API endpoint. "
            "If not set, falls back to provider-specific environment variable."
        ),
    )
    api_key_path: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Path to text file containing the API key for this model. "
            "If not set, falls back to provider-specific environment variable."
        ),
    )

    # Dynamic parameters (passed to LLM API)
    parameters: LLMParametersConfig = Field(
        default_factory=LLMParametersConfig,
        description="Dynamic parameters for this model (merged with defaults)",
    )

    # Timeout can be model-specific (some models are slower)
    timeout: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override timeout for this model",
    )


class LLMPoolConfig(BaseModel):
    """Pool of LLM configurations for reuse across the system.

    Provides a centralized place to define all LLM configurations,
    which can be referenced by judge_panel, agents, or other components.

    Cache and retry settings are managed at the defaults level only.
    Model entries contain model-specific settings (provider, parameters, SSL).
    """

    model_config = ConfigDict(extra="forbid")

    defaults: LLMDefaultsConfig = Field(
        default_factory=LLMDefaultsConfig,
        description="Global default settings for all LLMs (cache, retry, parameters)",
    )
    models: dict[str, LLMProviderConfig] = Field(
        default_factory=dict,
        description="Model configurations. Key is unique model ID for referencing.",
    )

    def get_model_ids(self) -> list[str]:
        """Get all available model IDs."""
        return list(self.models.keys())

    def resolve_llm_config(
        self, model_id: str, cache_suffix: Optional[str] = None
    ) -> LLMConfig:
        """Resolve a model ID to a fully configured LLMConfig.

        Resolution order: defaults -> model entry (for model-specific fields)

        Args:
            model_id: Model identifier (key in models dict)
            cache_suffix: Optional suffix for cache directory (e.g., "judge_0")

        Returns:
            Fully resolved LLMConfig

        Raises:
            ConfigurationError: If model_id not found
        """
        if model_id not in self.models:
            raise ConfigurationError(
                f"Model '{model_id}' not found in llm_pool.models. "
                f"Available: {list(self.models.keys())}"
            )
        entry = self.models[model_id]

        # Merge parameters: defaults -> individual (individual overrides defaults).
        # None in individual explicitly removes the default's value.
        # Note: Forbidden keys are rejected at LLMParametersConfig load time.
        merged_params: dict[str, Any] = {}
        merged_params.update(self.defaults.parameters.to_dict(exclude_none=True))
        merged_params.update(entry.parameters.to_dict(exclude_none=False))
        merged_params = {k: v for k, v in merged_params.items() if v is not None}

        # Build cache_dir from defaults with model-specific suffix
        # Note: cache_dir is guaranteed to be set by global_cache_setup()
        suffix = cache_suffix if cache_suffix else model_id
        base_cache_dir = self.defaults.cache_dir or ""
        cache_dir = os.path.join(base_cache_dir, suffix)

        config = LLMConfig(
            provider=entry.provider,
            model=entry.model or model_id,
            temperature=merged_params.get("temperature", DEFAULT_LLM_TEMPERATURE),
            max_tokens=merged_params.get(
                "max_completion_tokens", DEFAULT_LLM_MAX_TOKENS
            ),
            timeout=(
                entry.timeout if entry.timeout is not None else self.defaults.timeout
            ),
            num_retries=self.defaults.num_retries,
            ssl_verify=(
                entry.ssl_verify if entry.ssl_verify is not None else DEFAULT_SSL_VERIFY
            ),
            ssl_cert_file=entry.ssl_cert_file,
            cache_enabled=self.defaults.cache_enabled,
            cache_dir=cache_dir,
            # Note: api_base and api_key_path are not propagated yet - requires LLMConfig extension
        )
        return config.model_copy(update={"parameters": merged_params})


class JudgePanelConfig(BaseModel):
    """Judge panel configuration for multi-LLM evaluation.

    References models from LLM pool by model ID (the key in llm_pool.models).
    Each judge ID must correspond to a key in the llm_pool.models dictionary.
    """

    model_config = ConfigDict(extra="forbid")

    judges: list[str] = Field(
        ...,
        min_length=1,
        description="List of model IDs (keys from llm_pool.models). At least one required.",
    )
    enabled_metrics: Optional[list[str]] = Field(
        default=None,
        description=(
            "Metrics that should use the judge panel. "
            "If None, all metrics use the panel. "
            "If empty list, no metrics use the panel."
        ),
    )
    aggregation_strategy: str = Field(
        default="max",
        description=(
            "Strategy for aggregating scores: 'max', 'average', or "
            "'majority_vote' (average reported; PASS if a strict majority vote)."
        ),
    )

    @field_validator("enabled_metrics")
    @classmethod
    def validate_enabled_metrics(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate enabled_metrics format (framework:metric_name)."""
        if v is not None:
            for metric in v:
                if not metric or ":" not in metric:
                    raise ValueError(
                        f'Metric "{metric}" must be in format "framework:metric_name"'
                    )
                parts = metric.split(":", 1)
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    raise ValueError(
                        f'Metric "{metric}" must be in format "framework:metric_name"'
                    )
        return v

    @field_validator("aggregation_strategy")
    @classmethod
    def validate_aggregation_strategy(cls, v: str) -> str:
        """Validate aggregation_strategy is a supported value."""
        allowed = ["max", "average", "majority_vote"]
        if v not in allowed:
            raise ValueError(
                f"Unsupported aggregation_strategy '{v}'. Allowed: {allowed}"
            )
        return v


class GEvalRubricConfig(BaseModel):
    """Single rubric entry: score range 0-10 and expected outcome text."""

    model_config = ConfigDict(extra="forbid")

    score_range: tuple[int, int] = Field(
        ...,
        description="[min, max] score range (0-10); non-overlapping",
    )
    expected_outcome: str = Field(
        ...,
        min_length=1,
        description="Expected outcome for this score range",
    )

    @field_validator("score_range")
    @classmethod
    def validate_score_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Ensure score_range is [min, max] with 0 <= min <= max <= 10."""
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError("score_range must be [min, max] with two integers")
        low, high = int(v[0]), int(v[1])
        if low > high:
            raise ValueError(f"score_range min must be <= max, got [{low}, {high}]")
        if not (0 <= low <= 10 and 0 <= high <= 10):
            raise ValueError(
                f"score_range values must be between 0 and 10, got [{low}, {high}]"
            )
        return (low, high)


class GEvalConfig(BaseModel):
    """Validated GEval metric configuration (criteria required; rest optional)."""

    model_config = ConfigDict(extra="forbid")

    criteria: str = Field(..., min_length=1, description="Required evaluation criteria")
    evaluation_params: list[str] = Field(
        default_factory=list,
        description="Field names to include (e.g. query, response, expected_response)",
    )
    evaluation_steps: list[str] | None = Field(
        default=None,
        description="Optional step-by-step evaluation instructions",
    )
    rubrics: list[GEvalRubricConfig] | None = Field(
        default=None,
        description="Optional score ranges (0-10) with expected_outcome",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for pass/fail",
    )

    @model_validator(mode="after")
    def validate_rubrics_non_overlapping(self) -> "GEvalConfig":
        """Ensure rubric score ranges do not overlap."""
        rubs: list[GEvalRubricConfig] = self.rubrics if self.rubrics else []
        if len(rubs) <= 1:
            return self
        ranges = [r.score_range for r in rubs]
        for i, (a, b) in enumerate(ranges):
            for j, (c, d) in enumerate(ranges):
                if i >= j:
                    continue
                # Overlap if not (b < c or d < a)
                if not (b < c or d < a):
                    raise ConfigurationError(
                        f"Rubric score ranges must not overlap: "
                        f"[{a}, {b}] and [{c}, {d}] overlap"
                    )
        return self

    @classmethod
    def from_metadata(cls, raw: dict[str, Any]) -> "GEvalConfig":
        """Build GEvalConfig from raw metadata dict.

        Args:
            raw: Metadata dict with at least "criteria" (required). May include
                evaluation_params, evaluation_steps, rubrics, threshold.

        Returns:
            Validated GEvalConfig instance.

        Raises:
            ValueError: If raw is not a dict or criteria is missing/empty
                (only these pre-model_validate checks raise bare ValueError).
            ValidationError: If rubric or config fields fail Pydantic validation:
                wrong types (e.g. score_range, expected_outcome), invalid structure.
            ConfigurationError: If rubric score ranges overlap (model validator
                raises ConfigurationError directly, bypassing Pydantic wrapping).
        """
        if not isinstance(raw, dict):
            raise ValueError("GEval config must be a dict")
        criteria = raw.get("criteria")
        if not criteria or not isinstance(criteria, str) or not criteria.strip():
            raise ValueError("GEval requires non-empty 'criteria' in configuration")
        data: dict[str, Any] = {
            "criteria": criteria.strip(),
            "evaluation_params": raw.get("evaluation_params") or [],
            "evaluation_steps": raw.get("evaluation_steps"),
            "threshold": raw.get("threshold", 0.5),
        }
        raw_rubrics = raw.get("rubrics")
        if raw_rubrics and isinstance(raw_rubrics, list):
            data["rubrics"] = [
                GEvalRubricConfig.model_validate(item) for item in raw_rubrics
            ]
        else:
            data["rubrics"] = None
        return cls.model_validate(data)

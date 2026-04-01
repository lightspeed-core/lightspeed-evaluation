"""System configuration models."""

import os
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from lightspeed_evaluation.core.constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_CACHE_DIR,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_VERSION,
    DEFAULT_API_NUM_RETRIES,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENDPOINT_TYPE,
    DEFAULT_LLM_CACHE_DIR,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SSL_VERIFY,
    DEFAULT_SSL_CERT_FILE,
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_PACKAGE_LEVEL,
    DEFAULT_LOG_SHOW_TIMESTAMPS,
    DEFAULT_LOG_SOURCE_LEVEL,
    DEFAULT_VISUALIZATION_DPI,
    DEFAULT_VISUALIZATION_FIGSIZE,
    SUPPORTED_ENDPOINT_TYPES,
    SUPPORTED_GRAPH_TYPES,
)
from lightspeed_evaluation.core.storage.config import StorageBackendConfig
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

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
    cache_dir: str = Field(
        default=DEFAULT_LLM_CACHE_DIR,
        min_length=1,
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
    cache_dir: str = Field(
        default=DEFAULT_EMBEDDING_CACHE_DIR,
        min_length=1,
        description="Location of cached embedding queries",
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


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server authentication."""

    model_config = ConfigDict(extra="forbid")

    env_var: str = Field(
        ...,
        min_length=1,
        description="Environment variable containing the token/key",
    )
    header_name: Optional[str] = Field(
        default=None,
        description="Custom header name (optional, defaults to 'Authorization')",
    )


class MCPHeadersConfig(BaseModel):
    """Configuration for MCP headers functionality."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable MCP headers functionality",
    )
    servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP server configurations",
    )

    @model_validator(mode="after")
    def _validate_env_vars_when_enabled(self) -> "MCPHeadersConfig":
        """Validate that environment variables are set when MCP headers are enabled."""
        if self.enabled and self.servers:
            missing_vars = []
            for server_name, server_config in self.servers.items():
                if not os.getenv(server_config.env_var):
                    missing_vars.append(f"{server_name}: {server_config.env_var}")

            if missing_vars:
                missing_list = ", ".join(missing_vars)
                msg = (
                    "MCP headers are enabled but required environment variables are not set: "
                    f"{missing_list}"
                )
                raise ValueError(msg)

        return self


class APIConfig(BaseModel):
    """API configuration for dynamic data generation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable API-based data generation")
    api_base: str = Field(
        default=DEFAULT_API_BASE,
        description="Base URL for API requests (without version)",
    )
    version: str = Field(
        default=DEFAULT_API_VERSION, description="API version (e.g., v1, v2)"
    )
    endpoint_type: str = Field(
        default=DEFAULT_ENDPOINT_TYPE,
        description="API endpoint type (streaming or query)",
    )
    timeout: int = Field(
        default=DEFAULT_API_TIMEOUT, ge=1, description="Request timeout in seconds"
    )
    provider: Optional[str] = Field(default=None, description="LLM provider for API")
    model: Optional[str] = Field(default=None, description="LLM model for API")
    no_tools: Optional[bool] = Field(
        default=None, description="Disable tool usage in API calls"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for API calls"
    )
    extra_request_params: Optional[dict[str, Any]] = Field(default=None)
    cache_dir: str = Field(
        default=DEFAULT_API_CACHE_DIR,
        min_length=1,
        description="Location of cached lightspeed-stack queries",
    )
    cache_enabled: bool = Field(
        default=True, description="Is caching of lightspeed-stack queries enabled?"
    )
    mcp_headers: Optional[MCPHeadersConfig] = Field(
        default=None, description="MCP headers configuration for authentication"
    )
    num_retries: int = Field(
        default=DEFAULT_API_NUM_RETRIES,
        ge=0,
        description=(
            "Maximum number of retry attempts for API calls on "
            "429 Too Many Requests errors"
        ),
    )

    @field_validator("endpoint_type")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        """Validate endpoint type is supported."""
        if v not in SUPPORTED_ENDPOINT_TYPES:
            raise ValueError(f"Endpoint type must be one of {SUPPORTED_ENDPOINT_TYPES}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    source_level: str = Field(
        default=DEFAULT_LOG_SOURCE_LEVEL, description="Source code logging level"
    )
    package_level: str = Field(
        default=DEFAULT_LOG_PACKAGE_LEVEL, description="Package logging level"
    )
    log_format: str = Field(
        default=DEFAULT_LOG_FORMAT, description="Log message format"
    )
    show_timestamps: bool = Field(
        default=DEFAULT_LOG_SHOW_TIMESTAMPS, description="Show timestamps in logs"
    )
    package_overrides: dict[str, str] = Field(
        default_factory=dict, description="Package-specific log level overrides"
    )


class VisualizationConfig(BaseModel):
    """Visualization configuration for graphs and charts."""

    model_config = ConfigDict(extra="forbid")

    figsize: list[int] = Field(
        default=DEFAULT_VISUALIZATION_FIGSIZE, description="Figure size [width, height]"
    )
    dpi: int = Field(
        default=DEFAULT_VISUALIZATION_DPI, ge=50, description="Resolution in DPI"
    )
    enabled_graphs: list[str] = Field(
        default=[],
        description="List of graph types to generate",
    )

    @field_validator("enabled_graphs")
    @classmethod
    def validate_enabled_graphs(cls, v: list[str]) -> list[str]:
        """Validate that all enabled graphs are supported."""
        for graph_type in v:
            if graph_type not in SUPPORTED_GRAPH_TYPES:
                raise ValueError(
                    f"Unsupported graph type: {graph_type}. "
                    f"Supported types: {SUPPORTED_GRAPH_TYPES}"
                )
        return v


class CoreConfig(BaseModel):
    """Core evaluation configuration (e.g., concurrency limits)."""

    model_config = ConfigDict(extra="forbid")

    max_threads: Optional[int] = Field(
        default=None,
        description="Maximum threads for multithreading eval",
        gt=0,
    )
    fail_on_invalid_data: bool = Field(
        default=True,
        description="If False don't fail on invalid conversations",
    )
    skip_on_failure: bool = Field(
        default=False,
        description="Skip remaining turns in conversation when a turn evaluation fails",
    )


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
    cache_dir: str = Field(
        default=DEFAULT_LLM_CACHE_DIR,
        min_length=1,
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
        suffix = cache_suffix if cache_suffix else model_id
        cache_dir = os.path.join(self.defaults.cache_dir, suffix)

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


class SystemConfig(BaseModel):
    """System configuration using individual config models."""

    model_config = ConfigDict(extra="forbid")

    # Individual configuration models
    core: CoreConfig = Field(
        default_factory=CoreConfig, description="Core eval configuration"
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")

    # LLM Pool - shared pool of LLM configurations
    llm_pool: Optional[LLMPoolConfig] = Field(
        default=None,
        description=(
            "Pool of LLM configurations. Define models once, "
            "reference by ID in judge_panel or other components."
        ),
    )

    # Judge Panel - references models from llm_pool
    judge_panel: Optional[JudgePanelConfig] = Field(
        default=None,
        description=(
            "Optional judge panel configuration. "
            "References models from 'llm_pool' by ID. "
            "If not provided, the single 'llm' configuration is used."
        ),
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embeddings configuration"
    )
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    storage: list[StorageBackendConfig] = Field(
        default_factory=list,
        description="Storage backends for evaluation results (file and/or database)",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization configuration"
    )

    # Default metrics metadata from system config
    default_turn_metrics_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Default turn metrics metadata"
    )
    default_conversation_metrics_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Default conversation metrics metadata"
    )

    @field_validator(
        "default_turn_metrics_metadata", "default_conversation_metrics_metadata"
    )
    @classmethod
    def validate_default_metrics_metadata_geval(
        cls, v: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Validate GEval entries at load; keep storing as dict (result discarded).

        We call GEvalConfig.from_metadata(meta) only for its validation side
        effect (fail fast on invalid system config). The returned config is
        discarded; the raw dict is stored. At evaluation time the manager
        may merge overrides with this dict, and the handler re-validates
        via from_metadata on the merged result.

        Raises:
            ConfigurationError: When a geval:* entry has invalid config (e.g.
                missing criteria, invalid rubric structure, overlapping rubrics).
                Re-raised from ValueError, ValidationError, or ConfigurationError
                for a consistent config-failure exception type with metric context.
        """
        if not v:
            return v
        for metric_id, meta in v.items():
            if metric_id.startswith("geval:") and isinstance(meta, dict):
                try:
                    GEvalConfig.from_metadata(meta)
                except (ValueError, ValidationError, ConfigurationError) as e:
                    raise ConfigurationError(
                        f"Invalid GEval config for '{metric_id}': {e!s}"
                    ) from e
        return v

    @property
    def turn_level_metric_names(self) -> set[str]:
        """Return turn-level metric names derived from metadata keys."""
        return set(self.default_turn_metrics_metadata.keys())

    @property
    def conversation_level_metric_names(self) -> set[str]:
        """Return conversation-level metric names derived from metadata keys."""
        return set(self.default_conversation_metrics_metadata.keys())

    def get_judge_configs(self) -> list[tuple[str, LLMConfig]]:
        """Get resolved LLMConfig for all judges with their pool keys.

        Returns:
            List of (pool_key, LLMConfig) tuples for each judge.
            If judge_panel is configured, resolves from llm_pool.
            Otherwise, returns single entry with "primary" as key.
        """
        if not self.judge_panel:
            return [("primary", self.llm)]

        if not self.llm_pool:
            raise ConfigurationError(
                "judge_panel is configured but 'llm_pool' is not defined. "
                "Please define the llm_pool section with models."
            )

        configs = []
        for idx, judge_id in enumerate(self.judge_panel.judges):
            cache_suffix = f"judge_{idx}"
            config = self.llm_pool.resolve_llm_config(
                judge_id, cache_suffix=cache_suffix
            )
            configs.append((judge_id, config))
        return configs

    def get_llm_config(
        self, model_id: str, cache_suffix: Optional[str] = None
    ) -> LLMConfig:
        """Get resolved LLMConfig for a specific model from the pool.

        Args:
            model_id: Model identifier (key in llm_pool.models)
            cache_suffix: Optional suffix for cache directory

        Returns:
            Fully resolved LLMConfig

        Raises:
            ConfigurationError: If llm_pool not configured or model not found
        """
        if not self.llm_pool:
            raise ConfigurationError(
                f"Cannot resolve model '{model_id}' - 'llm_pool' is not configured."
            )
        return self.llm_pool.resolve_llm_config(model_id, cache_suffix=cache_suffix)

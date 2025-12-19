"""System configuration models."""

import os
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lightspeed_evaluation.core.constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_CACHE_DIR,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_VERSION,
    DEFAULT_BASE_FILENAME,
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
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VISUALIZATION_DPI,
    DEFAULT_VISUALIZATION_FIGSIZE,
    SUPPORTED_CSV_COLUMNS,
    SUPPORTED_ENDPOINT_TYPES,
    SUPPORTED_GRAPH_TYPES,
    SUPPORTED_OUTPUT_TYPES,
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
                raise ValueError(
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
    cache_dir: str = Field(
        default=DEFAULT_API_CACHE_DIR,
        min_length=1,
        description="Location of cached lightspeed-stack queries",
    )
    cache_enabled: bool = Field(
        default=True, description="Is caching of lightspeed-stack queries enabled?"
    )

    @field_validator("endpoint_type")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        """Validate endpoint type is supported."""
        if v not in SUPPORTED_ENDPOINT_TYPES:
            raise ValueError(f"Endpoint type must be one of {SUPPORTED_ENDPOINT_TYPES}")
        return v


class OutputConfig(BaseModel):
    """Output configuration for evaluation results."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(
        default=DEFAULT_OUTPUT_DIR, description="Output directory for results"
    )
    base_filename: str = Field(
        default=DEFAULT_BASE_FILENAME, description="Base filename for output files"
    )
    enabled_outputs: list[str] = Field(
        default=SUPPORTED_OUTPUT_TYPES,
        description="List of enabled output types: csv, json, txt",
    )
    csv_columns: list[str] = Field(
        default=SUPPORTED_CSV_COLUMNS,
        description="CSV columns to include in detailed results",
    )

    @field_validator("csv_columns")
    @classmethod
    def validate_csv_columns(cls, v: list[str]) -> list[str]:
        """Validate that all CSV columns are supported."""
        for column in v:
            if column not in SUPPORTED_CSV_COLUMNS:
                raise ValueError(
                    f"Unsupported CSV column: {column}. "
                    f"Supported columns: {SUPPORTED_CSV_COLUMNS}"
                )
        return v

    @field_validator("enabled_outputs")
    @classmethod
    def validate_enabled_outputs(cls, v: list[str]) -> list[str]:
        """Validate that all enabled outputs are supported."""
        for output_type in v:
            if output_type not in SUPPORTED_OUTPUT_TYPES:
                raise ValueError(
                    f"Unsupported output type: {output_type}. "
                    f"Supported types: {SUPPORTED_OUTPUT_TYPES}"
                )
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


class SystemConfig(BaseModel):
    """System configuration using individual config models."""

    model_config = ConfigDict(extra="forbid")

    # Individual configuration models
    core: CoreConfig = Field(
        default_factory=CoreConfig, description="Core eval configuration"
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embeddings configuration"
    )
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
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

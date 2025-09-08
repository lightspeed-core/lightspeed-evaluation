"""System configuration models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_TIMEOUT,
    DEFAULT_BASE_FILENAME,
    DEFAULT_ENDPOINT_TYPE,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_PACKAGE_LEVEL,
    DEFAULT_LOG_SHOW_TIMESTAMPS,
    DEFAULT_LOG_SOURCE_LEVEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VISUALIZATION_DPI,
    DEFAULT_VISUALIZATION_FIGSIZE,
    SUPPORTED_ENDPOINT_TYPES,
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


class APIConfig(BaseModel):
    """API configuration for dynamic data generation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable API-based data generation")
    api_base: str = Field(
        default=DEFAULT_API_BASE, description="Base URL for API requests"
    )
    endpoint_type: str = Field(
        default=DEFAULT_ENDPOINT_TYPE,
        description="API endpoint type (streaming or query)",
    )
    timeout: int = Field(
        default=DEFAULT_API_TIMEOUT, ge=1, description="Request timeout in seconds"
    )
    provider: str = Field(
        default=DEFAULT_LLM_PROVIDER, description="LLM provider for API"
    )
    model: str = Field(default=DEFAULT_LLM_MODEL, description="LLM model for API")
    no_tools: Optional[bool] = Field(
        default=None, description="Disable tool usage in API calls"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for API calls"
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
    enabled_outputs: List[str] = Field(
        default=["csv", "json", "txt", "graphs"],
        description="List of enabled output types: csv, json, txt, graphs",
    )


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
    package_overrides: Dict[str, str] = Field(
        default_factory=dict, description="Package-specific log level overrides"
    )


class VisualizationConfig(BaseModel):
    """Visualization configuration for graphs and charts."""

    model_config = ConfigDict(extra="forbid")

    figsize: List[int] = Field(
        default=DEFAULT_VISUALIZATION_FIGSIZE, description="Figure size [width, height]"
    )
    dpi: int = Field(
        default=DEFAULT_VISUALIZATION_DPI, ge=50, description="Resolution in DPI"
    )


class SystemConfig(BaseModel):
    """System configuration using individual config models."""

    model_config = ConfigDict(extra="forbid")

    # Individual configuration models
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
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
    default_turn_metrics_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Default turn metrics metadata"
    )
    default_conversation_metrics_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Default conversation metrics metadata"
    )

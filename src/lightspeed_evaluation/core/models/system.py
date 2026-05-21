# pylint: disable=too-many-lines
"""System configuration models."""

import logging
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
    DEFAULT_AGENT_CACHE_SUBDIR,
    DEFAULT_CACHE_BASE_DIR,
    DEFAULT_LLM_CACHE_SUBDIR,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_PACKAGE_LEVEL,
    DEFAULT_LOG_SHOW_TIMESTAMPS,
    DEFAULT_LOG_SOURCE_LEVEL,
    DEFAULT_VISUALIZATION_DPI,
    DEFAULT_VISUALIZATION_FIGSIZE,
    SUPPORTED_GRAPH_TYPES,
)
from lightspeed_evaluation.core.models.agents import (
    AgentsConfig,
    HttpApiBaseFields,
    MCPHeadersConfig,
)
from lightspeed_evaluation.core.models.llm import (
    EmbeddingConfig,
    GEvalConfig,
    JudgePanelConfig,
    LLMConfig,
    LLMPoolConfig,
)
from lightspeed_evaluation.core.storage.config import StorageBackendConfig
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class APIConfig(HttpApiBaseFields):
    """API configuration for dynamic data generation."""

    enabled: bool = Field(default=True, description="Enable API-based data generation")
    mcp_headers: Optional[MCPHeadersConfig] = Field(
        default=None, description="MCP headers configuration for authentication"
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
    cache_enabled: bool = Field(
        default=True,
        description="Global caching toggle for embeddings, API, and LLM as a judge queries",
    )
    cache_base_dir: str = Field(
        default=DEFAULT_CACHE_BASE_DIR,
        min_length=1,
        description="Base directory for all evaluation caches (embeddings, API, LLM judge)",
    )


class QualityScoreConfig(BaseModel):
    """Quality score configuration."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=list,
        description="List of metric identifiers to use for quality score computation",
    )
    default: bool = Field(
        default=False,
        description="If true, set default: true for all metrics in the list",
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Ensure metrics list is not empty and contains no duplicates."""
        if len(v) == 0:
            raise ValueError(
                "Quality score metrics list cannot be empty. "
                "Either specify at least one metric or "
                "remove the quality_score section from configuration."
            )
        if len(v) != len(set(v)):
            duplicates = [m for m in v if v.count(m) > 1]
            raise ValueError(
                f"Quality score metrics contains duplicates: {set(duplicates)}. "
                "Each metric should appear only once."
            )
        return v


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
    agents: Optional[AgentsConfig] = Field(
        default=None,
        description=(
            "Agents configuration. When present, takes precedence over api: block. "
            "Auto-populated from api: when absent."
        ),
    )
    storage: list[StorageBackendConfig] = Field(
        default_factory=list,
        description=(
            "Storage backends for evaluation results (file, database, and/or langfuse export)"
        ),
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization configuration"
    )

    # Quality score configuration
    quality_score: Optional[QualityScoreConfig] = Field(
        default=None, description="Quality score configuration"
    )

    # Default metrics metadata from system config
    default_turn_metrics_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Default turn metrics metadata"
    )
    default_conversation_metrics_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Default conversation metrics metadata"
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_api_to_agents(cls, data: Any) -> Any:
        """Auto-migrate api: to agents: when agents: is absent."""
        if not isinstance(data, dict):
            return data
        if data.get("agents") is not None or data.get("api") is None:
            return data

        api_data = data["api"]
        if isinstance(api_data, dict):
            api_enabled = api_data.get("enabled", True)
            agent_fields = {k: v for k, v in api_data.items() if k != "enabled"}
        elif isinstance(api_data, BaseModel):
            api_enabled = getattr(api_data, "enabled", True)
            agent_fields = api_data.model_dump(exclude={"enabled"})
        else:
            return data

        agent_fields["type"] = "http_api"
        data["agents"] = {
            "enabled": api_enabled,
            "default": {"agent": "http_api" if api_enabled else None},
            "http_api": agent_fields,
        }
        return data

    @model_validator(mode="after")
    def validate_agents_config(self) -> "SystemConfig":
        """Validate that default.agent references a defined agent."""
        if self.agents is not None:
            default_agent = self.agents.default.agent
            if default_agent is not None and default_agent not in self.agents.agents:
                raise ConfigurationError(
                    f"Default agent '{default_agent}' not found in agents "
                    f"definitions. Available: {list(self.agents.agents.keys())}"
                )
        return self

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

    @model_validator(mode="after")
    def validate_quality_score_metrics(self) -> "SystemConfig":
        """Validate quality_score metrics exist in metrics_metadata.

        Raises:
            ConfigurationError: When quality_score contains metrics not defined
                in turn or conversation level metrics_metadata.
        """
        if not self.quality_score:
            return self

        # Combine all available metrics from both turn and conversation level metadata
        all_metrics = set(self.default_turn_metrics_metadata.keys()) | set(
            self.default_conversation_metrics_metadata.keys()
        )

        # Check for invalid metrics
        invalid = [m for m in self.quality_score.metrics if m not in all_metrics]
        if invalid:
            raise ConfigurationError(
                f"Invalid quality_score metrics: {invalid}. "
                "Must be defined in default_turn_metrics_metadata or "
                "default_conversation_metrics_metadata."
            )

        return self

    @property
    def turn_level_metric_names(self) -> set[str]:
        """Return turn-level metric names derived from metadata keys."""
        return set(self.default_turn_metrics_metadata.keys())

    @property
    def conversation_level_metric_names(self) -> set[str]:
        """Return conversation-level metric names derived from metadata keys."""
        return set(self.default_conversation_metrics_metadata.keys())

    @model_validator(mode="after")
    def global_cache_setup(self) -> "SystemConfig":
        """Apply global cache_enabled toggle using AND logic with component-level toggles.

        Also appends component-specific subdirectories to the global cache_dir.

        Precedence: llm_pool (if judge_panel configured) > llm (legacy fallback).
        Only the active LLM configuration is updated to avoid confusion.
        """
        # Core cache settings
        global_cache_enabled = self.core.cache_enabled
        global_cache_base_dir = self.core.cache_base_dir
        # Build cache paths
        llm_cache_path = os.path.join(global_cache_base_dir, DEFAULT_LLM_CACHE_SUBDIR)
        api_cache_path = os.path.join(global_cache_base_dir, DEFAULT_AGENT_CACHE_SUBDIR)

        # If component-level caching is enabled, warn user
        has_api_cache_config = self.api.cache_enabled is False or self.api.cache_dir
        has_embedding_cache_config = (
            self.embedding.cache_enabled is False or self.embedding.cache_dir
        )
        has_llm_pool_cache_config = self.llm_pool and (
            self.llm_pool.defaults.cache_enabled is False
            or self.llm_pool.defaults.cache_dir
        )
        has_agents_cache_config = self.agents and any(
            (
                getattr(agent, "cache_enabled", None) is False
                or getattr(agent, "cache_dir", None)
            )
            for agent in self.agents.agents.values()
        )
        if (
            has_api_cache_config
            or has_embedding_cache_config
            or has_llm_pool_cache_config
            or has_agents_cache_config
        ):
            logger.warning(
                "DEPRECATION: Component-level caching is deprecated. "
                "Please use 'core.cache_enabled' and 'core.cache_base_dir' instead."
            )

        # LLM as a judge cache setup
        # Apply cache settings based on which LLM config is actually used
        # judge_panel uses llm_pool, no judge_panel uses legacy llm
        if self.llm_pool:
            # llm_pool is configured - set its defaults
            self.llm_pool.defaults.cache_enabled = (
                global_cache_enabled and self.llm_pool.defaults.cache_enabled
            )
            self.llm_pool.defaults.cache_dir = (
                self.llm_pool.defaults.cache_dir or llm_cache_path
            )
            # LEGACY support: sync llm config with pool defaults
            self.llm.cache_enabled = self.llm_pool.defaults.cache_enabled
            self.llm.cache_dir = self.llm_pool.defaults.cache_dir
        else:
            # LEGACY support
            # No llm_pool -> uses legacy llm config only
            logger.warning(
                "DEPRECATION: 'llm' configuration is deprecated. "
                "Please migrate to 'llm_pool' format."
            )
            self.llm.cache_enabled = global_cache_enabled and self.llm.cache_enabled
            self.llm.cache_dir = self.llm.cache_dir or llm_cache_path

        # Embedding cache setup
        # Embedding cache is shared with LLM as a judge
        self.embedding.cache_enabled = (
            global_cache_enabled and self.embedding.cache_enabled
        )

        # Agents cache setup
        if self.agents:
            for agent_name, agent_def in self.agents.agents.items():
                if hasattr(agent_def, "cache_enabled"):
                    agent_def.cache_enabled = (
                        global_cache_enabled and agent_def.cache_enabled
                    )
                if hasattr(agent_def, "cache_dir"):
                    agent_cache_path = os.path.join(api_cache_path, agent_name)
                    agent_def.cache_dir = agent_def.cache_dir or agent_cache_path
        # API cache setup (LEGACY support)
        self.api.cache_enabled = global_cache_enabled and self.api.cache_enabled
        self.api.cache_dir = self.api.cache_dir or api_cache_path

        return self

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

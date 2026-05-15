"""Agent configuration models for the evaluation framework."""

import os
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lightspeed_evaluation.core.constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_NUM_RETRIES,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_VERSION,
    DEFAULT_ENDPOINT_TYPE,
    SUPPORTED_AGENT_TYPES,
    SUPPORTED_ENDPOINT_TYPES,
)
from lightspeed_evaluation.core.system.exceptions import ConfigurationError


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
                    "MCP headers are enabled but required environment "
                    "variables are not set: "
                    f"{missing_list}"
                )
                raise ValueError(msg)

        return self


class HttpApiBaseFields(BaseModel):
    """Shared HTTP API connection fields.

    Base class for both ``HttpApiAgentConfig`` (agents layer) and
    ``APIConfig`` (legacy api: block) to avoid duplicate field definitions.
    """

    model_config = ConfigDict(extra="forbid")

    api_base: str = Field(
        default=DEFAULT_API_BASE,
        description="Base URL for API requests (without version)",
    )
    version: str = Field(
        default=DEFAULT_API_VERSION, description="API version (e.g., v1, v2)"
    )
    endpoint_type: str = Field(
        default=DEFAULT_ENDPOINT_TYPE,
        description="API endpoint type (streaming / query / infer)",
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
    cache_dir: Optional[str] = Field(
        default=None,
        description="Location of cached API queries",
    )
    cache_enabled: bool = Field(
        default=True, description="Is caching of API queries enabled?"
    )
    num_retries: int = Field(
        default=DEFAULT_API_NUM_RETRIES,
        ge=0,
        description="Maximum number of retry attempts for 429 errors",
    )

    @field_validator("endpoint_type")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        """Validate endpoint type is supported."""
        if v not in SUPPORTED_ENDPOINT_TYPES:
            raise ValueError(f"Endpoint type must be one of {SUPPORTED_ENDPOINT_TYPES}")
        return v


class HttpApiAgentConfig(HttpApiBaseFields):
    """Configuration for an HTTP API agent."""

    type: Literal["http_api"] = Field(
        default="http_api", description="Agent type identifier"
    )
    mcp_headers: Optional[MCPHeadersConfig] = Field(
        default=None,
        description="MCP headers configuration for authentication",
    )


class ProposalAgentConfig(BaseModel):
    """Configuration for a Proposal CRD-based agent."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["proposal"] = "proposal"
    namespace: str
    auto_approve: bool = True
    cleanup_proposals: bool = True
    timeout: int = Field(default=900, gt=0)
    poll_interval: int = Field(default=2, gt=0)


# Discriminated union of all agent config types; extend by adding new
# config classes to support additional agent types.
AgentDefinition = Union[HttpApiAgentConfig, ProposalAgentConfig]


class AgentDefaultConfig(BaseModel):
    """Default agent selection and shared configuration."""

    model_config = ConfigDict(extra="forbid")

    agent: Optional[str] = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
        description="Name of the default agent when eval_data doesn't specify one",
    )
    agent_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Shared default agent config overrides applied to all agents",
    )


class AgentsConfig(BaseModel):
    """Top-level agents configuration container.

    Parses a flat YAML namespace where ``default`` is a reserved key holding
    shared config and agent selection, and all other keys with a ``type`` field
    are agent definitions.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable agent-based API calls instead of using pre-filled data",
    )
    default: AgentDefaultConfig = Field(default_factory=AgentDefaultConfig)
    agents: dict[str, AgentDefinition] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def extract_agent_definitions(cls, data: Any) -> Any:
        """Extract named agent definitions from the flat YAML namespace."""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        agents: dict[str, Any] = {}
        default_data = data.pop("default", {})
        remaining: dict[str, Any] = {}

        for key, value in data.items():
            if key == "agents":
                agents.update(value)
            elif isinstance(value, dict) and "type" in value:
                agents[key] = value
            else:
                remaining[key] = value

        result: dict[str, Any] = {"default": default_data, "agents": agents}
        result.update(remaining)
        return result

    @model_validator(mode="after")
    def validate_agent_types(self) -> "AgentsConfig":
        """Validate that all agent definitions have supported types."""
        for name, agent_def in self.agents.items():
            if agent_def.type not in SUPPORTED_AGENT_TYPES:
                raise ConfigurationError(
                    f"Agent '{name}' has unsupported type '{agent_def.type}'. "
                    f"Supported types: {SUPPORTED_AGENT_TYPES}"
                )
        return self

    def resolve_agent_config(
        self,
        agent_name: Optional[str] = None,
        agent_config_override: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve final agent configuration from the 2-level priority chain.

        Per-key merge in ascending priority order — higher levels override
        matching keys while non-overlapping keys from lower levels survive:

        1. ``default.agent_config`` (lowest)
        2. ``agent_config_override`` from eval data (highest)

        The agent definition's typed fields form the base; override dicts
        are applied on top.

        Args:
            agent_name: Explicit agent name. Falls back to default.agent.
            agent_config_override: Per-evaluation config overrides (highest priority).

        Returns:
            Tuple of (agent_name, merged_config_dict).

        Raises:
            ConfigurationError: If no agent can be resolved or agent not found.
        """
        name = agent_name or self.default.agent
        if name is None:
            raise ConfigurationError(
                "No agent specified and no default agent configured"
            )

        if name not in self.agents:
            raise ConfigurationError(
                f"Agent '{name}' not found in agents configuration. "
                f"Available agents: {list(self.agents.keys())}"
            )

        agent_def = self.agents[name]

        effective: dict[str, Any] = {}
        if self.default.agent_config:
            effective.update(self.default.agent_config)
        if agent_config_override:
            effective.update(agent_config_override)

        base_config = agent_def.model_dump()
        base_config.update(effective)

        return name, base_config

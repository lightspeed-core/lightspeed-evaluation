"""Tests for agent configuration models."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_CACHE_DIR,
    DEFAULT_API_NUM_RETRIES,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_VERSION,
    DEFAULT_ENDPOINT_TYPE,
)
from lightspeed_evaluation.core.models.agents import (
    AgentDefaultConfig,
    AgentsConfig,
    HttpApiAgentConfig,
    MCPHeadersConfig,
    MCPServerConfig,
)
from lightspeed_evaluation.core.system.exceptions import ConfigurationError


class TestHttpApiAgentConfig:
    """Tests for HttpApiAgentConfig model."""

    def test_defaults_match_api_config(self) -> None:
        """Verify defaults match the existing APIConfig constants."""
        config = HttpApiAgentConfig()
        assert config.type == "http_api"
        assert config.api_base == DEFAULT_API_BASE
        assert config.version == DEFAULT_API_VERSION
        assert config.endpoint_type == DEFAULT_ENDPOINT_TYPE
        assert config.timeout == DEFAULT_API_TIMEOUT
        assert config.cache_dir == DEFAULT_API_CACHE_DIR
        assert config.cache_enabled is True
        assert config.num_retries == DEFAULT_API_NUM_RETRIES
        assert config.provider is None
        assert config.model is None
        assert config.no_tools is None
        assert config.system_prompt is None
        assert config.extra_request_params is None

    def test_custom_values(self) -> None:
        """Test HttpApiAgentConfig with custom values."""
        config = HttpApiAgentConfig(
            api_base="https://custom.api.com",
            timeout=600,
            provider="openai",
            model="gpt-4o",
        )
        assert config.api_base == "https://custom.api.com"
        assert config.timeout == 600
        assert config.provider == "openai"
        assert config.model == "gpt-4o"

    def test_endpoint_type_validation(self) -> None:
        """Test invalid endpoint_type is rejected."""
        with pytest.raises(ValidationError, match="Endpoint type"):
            HttpApiAgentConfig(endpoint_type="invalid")

    def test_timeout_must_be_positive(self) -> None:
        """Test timeout must be >= 1."""
        with pytest.raises(ValidationError):
            HttpApiAgentConfig(timeout=0)

    def test_num_retries_must_be_non_negative(self) -> None:
        """Test num_retries must be >= 0."""
        with pytest.raises(ValidationError):
            HttpApiAgentConfig(num_retries=-1)

    def test_mcp_headers_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test mcp_headers field is accepted."""
        monkeypatch.setenv("TOKEN", "dummy")
        headers = MCPHeadersConfig(
            enabled=True,
            servers={"mcp1": MCPServerConfig(env_var="TOKEN")},
        )
        config = HttpApiAgentConfig(mcp_headers=headers)
        assert config.mcp_headers is not None
        assert config.mcp_headers.enabled is True
        assert "mcp1" in config.mcp_headers.servers

    def test_mcp_headers_defaults_to_none(self) -> None:
        """Test mcp_headers defaults to None."""
        config = HttpApiAgentConfig()
        assert config.mcp_headers is None

    def test_extra_forbid(self) -> None:
        """Test unknown fields are rejected."""
        with pytest.raises(ValidationError):
            HttpApiAgentConfig.model_validate({"unknown_field": "value"})


class TestAgentDefaultConfig:
    """Tests for AgentDefaultConfig model."""

    def test_defaults(self) -> None:
        """Test all fields default to None."""
        config = AgentDefaultConfig()
        assert config.agent is None
        assert config.agent_config is None

    def test_with_agent_config(self) -> None:
        """Test setting agent and agent_config."""
        config = AgentDefaultConfig(
            agent="my_agent",
            agent_config={"timeout": 600, "num_retries": 5},
        )
        assert config.agent == "my_agent"
        assert config.agent_config == {"timeout": 600, "num_retries": 5}

    def test_empty_agent_name_rejected(self) -> None:
        """Test empty string agent name is rejected."""
        with pytest.raises(ValidationError):
            AgentDefaultConfig(agent="")

    def test_whitespace_agent_name_rejected(self) -> None:
        """Test whitespace-only agent name is rejected."""
        with pytest.raises(ValidationError):
            AgentDefaultConfig(agent="   ")


class TestAgentsConfig:
    """Tests for AgentsConfig model."""

    def test_parses_flat_yaml_namespace(self) -> None:
        """Test YAML-style flat namespace is parsed correctly."""
        config = AgentsConfig.model_validate(
            {
                "default": {"agent": "ols_api"},
                "ols_api": {
                    "type": "http_api",
                    "api_base": "http://localhost:8080",
                },
            }
        )
        assert config.default.agent == "ols_api"
        assert "ols_api" in config.agents
        assert config.agents["ols_api"].type == "http_api"
        assert config.agents["ols_api"].api_base == "http://localhost:8080"

    def test_multiple_agents(self) -> None:
        """Test parsing multiple named agents."""
        config = AgentsConfig.model_validate(
            {
                "default": {"agent": "primary"},
                "primary": {
                    "type": "http_api",
                    "api_base": "http://primary:8080",
                },
                "secondary": {
                    "type": "http_api",
                    "api_base": "http://secondary:8080",
                },
            }
        )
        assert len(config.agents) == 2
        assert config.agents["primary"].api_base == "http://primary:8080"
        assert config.agents["secondary"].api_base == "http://secondary:8080"

    def test_empty_config(self) -> None:
        """Test empty config is valid."""
        config = AgentsConfig()
        assert config.default.agent is None
        assert not config.agents

    def test_default_only(self) -> None:
        """Test config with only default section."""
        config = AgentsConfig.model_validate(
            {"default": {"agent_config": {"timeout": 600}}}
        )
        assert config.default.agent_config == {"timeout": 600}
        assert not config.agents

    def test_unknown_top_level_key_rejected(self) -> None:
        """Test that non-agent, non-default keys are rejected."""
        with pytest.raises(ValidationError):
            AgentsConfig.model_validate(
                {
                    "default": {"agent": "ols_api"},
                    "ols_api": {"type": "http_api"},
                    "not_a_dict": "invalid_value",
                }
            )

    def test_input_dict_not_mutated(self) -> None:
        """Test that the input dict is not mutated by the model validator."""
        input_data = {
            "default": {"agent": "ols_api"},
            "ols_api": {"type": "http_api"},
        }
        original_keys = set(input_data.keys())
        AgentsConfig.model_validate(input_data)
        assert set(input_data.keys()) == original_keys
        assert "default" in input_data


class TestAgentsConfigResolve:
    """Tests for AgentsConfig.resolve_agent_config method."""

    def _make_config(self) -> AgentsConfig:
        """Create a basic AgentsConfig for testing."""
        return AgentsConfig.model_validate(
            {
                "default": {"agent": "ols_api"},
                "ols_api": {"type": "http_api", "api_base": "http://localhost:8080"},
            }
        )

    def test_resolve_default_agent(self) -> None:
        """Test resolving with the default agent."""
        config = self._make_config()
        name, resolved = config.resolve_agent_config()
        assert name == "ols_api"
        assert resolved["type"] == "http_api"
        assert resolved["api_base"] == "http://localhost:8080"

    def test_resolve_explicit_agent(self) -> None:
        """Test resolving with explicit agent name."""
        config = AgentsConfig.model_validate(
            {
                "default": {"agent": "primary"},
                "primary": {"type": "http_api"},
                "secondary": {
                    "type": "http_api",
                    "api_base": "http://secondary:9090",
                },
            }
        )
        name, resolved = config.resolve_agent_config(agent_name="secondary")
        assert name == "secondary"
        assert resolved["api_base"] == "http://secondary:9090"

    def test_resolve_with_overrides(self) -> None:
        """Test per-evaluation config overrides take priority."""
        config = self._make_config()
        _, resolved = config.resolve_agent_config(
            agent_config_override={"timeout": 1200, "api_base": "http://override:8080"}
        )
        assert resolved["timeout"] == 1200
        assert resolved["api_base"] == "http://override:8080"

    def test_resolve_missing_agent_raises(self) -> None:
        """Test resolving a non-existent agent raises error."""
        config = self._make_config()
        with pytest.raises(ConfigurationError, match="not found"):
            config.resolve_agent_config(agent_name="nonexistent")

    def test_resolve_no_default_no_name_raises(self) -> None:
        """Test resolving without default or name raises error."""
        config = AgentsConfig.model_validate(
            {
                "default": {},
                "ols_api": {"type": "http_api"},
            }
        )
        with pytest.raises(ConfigurationError, match="No agent specified"):
            config.resolve_agent_config()

    def test_default_agent_config_applied(self) -> None:
        """Test default agent_config applies when no higher levels exist."""
        config = AgentsConfig.model_validate(
            {
                "default": {
                    "agent": "ols_api",
                    "agent_config": {"timeout": 900, "num_retries": 5},
                },
                "ols_api": {"type": "http_api"},
            }
        )
        _, resolved = config.resolve_agent_config()
        assert resolved["timeout"] == 900
        assert resolved["num_retries"] == 5

    def test_eval_data_overrides_default_keys(self) -> None:
        """Test eval_data agent_config overrides matching default keys."""
        config = AgentsConfig.model_validate(
            {
                "default": {
                    "agent": "ols_api",
                    "agent_config": {"timeout": 900, "provider": "aws"},
                },
                "ols_api": {"type": "http_api"},
            }
        )
        _, resolved = config.resolve_agent_config(
            agent_config_override={"timeout": 300}
        )
        assert resolved["timeout"] == 300
        assert resolved["provider"] == "aws"

    def test_two_level_merge(self) -> None:
        """Test both levels merge per-key in ascending priority order."""
        config = AgentsConfig.model_validate(
            {
                "default": {
                    "agent": "ols_api",
                    "agent_config": {
                        "timeout": 900,
                        "provider": "aws",
                        "num_retries": 3,
                    },
                },
                "ols_api": {
                    "type": "http_api",
                    "provider": "azure",
                },
            }
        )
        _, resolved = config.resolve_agent_config(
            agent_config_override={"timeout": 300}
        )
        assert resolved["timeout"] == 300
        assert resolved["provider"] == "aws"
        assert resolved["num_retries"] == 3

    def test_no_agent_config_at_any_level(self) -> None:
        """Test agent definition typed defaults stand when no overrides exist."""
        config = self._make_config()
        _, resolved = config.resolve_agent_config()
        assert resolved["timeout"] == DEFAULT_API_TIMEOUT
        assert resolved["num_retries"] == DEFAULT_API_NUM_RETRIES

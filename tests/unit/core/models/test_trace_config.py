"""Tests for trace source configuration models and validation."""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from lightspeed_evaluation.core.models.agents import AgentsConfig
from lightspeed_evaluation.core.models.system import SystemConfig
from lightspeed_evaluation.core.models.trace_config import (
    CollectorLogsTraceSourceConfig,
    LangfuseTraceSourceConfig,
    MLflowTraceSourceConfig,
    OtelTraceSourceConfig,
    TraceSourceConfig,
)
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

_SOURCE_ADAPTER = TypeAdapter(TraceSourceConfig)


class TestSourceConfigs:
    """Per-type parsing, defaults, and required fields."""

    def test_langfuse_defaults(self) -> None:
        """Langfuse entries load with polling defaults."""
        config = _SOURCE_ADAPTER.validate_python({"type": "langfuse"})
        assert isinstance(config, LangfuseTraceSourceConfig)
        assert config.host is None
        assert config.persistence_timeout == 10
        assert config.poll_interval == 2

    def test_mlflow_defaults(self) -> None:
        """MLflow entries load with polling defaults."""
        config = _SOURCE_ADAPTER.validate_python({"type": "mlflow"})
        assert isinstance(config, MLflowTraceSourceConfig)
        assert config.tracking_uri is None
        assert config.persistence_timeout == 10

    def test_collector_requires_namespace_and_service(self) -> None:
        """Collector entries require namespace and service name."""
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python({"type": "collector_logs"})

    def test_collector_defaults(self) -> None:
        """Collector entries load with transport and polling defaults."""
        config = _SOURCE_ADAPTER.validate_python(
            {
                "type": "collector_logs",
                "namespace": "openshift-lightspeed",
                "service_name": "lightspeed-otel-collector",
            }
        )
        assert isinstance(config, CollectorLogsTraceSourceConfig)
        assert config.transport == "kubernetes_service_proxy"
        assert config.service_port == "8080"
        assert config.persistence_timeout == 30
        assert config.stable_reads == 2

    def test_otel_requires_endpoint(self) -> None:
        """OTEL entries require an endpoint."""
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python({"type": "otel"})

    def test_otel_defaults(self) -> None:
        """OTEL entries load with correlation defaults."""
        config = _SOURCE_ADAPTER.validate_python(
            {"type": "otel", "endpoint": "http://jaeger:16686"}
        )
        assert isinstance(config, OtelTraceSourceConfig)
        assert config.query_by == "span_attribute"
        assert config.attribute_key == "gen_ai.conversation.id"

    def test_otel_rejects_unknown_query_mode(self) -> None:
        """OTEL query_by only accepts defined correlation modes."""
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python(
                {"type": "otel", "endpoint": "http://jaeger:16686", "query_by": "magic"}
            )

    def test_unknown_type_rejected(self) -> None:
        """Unknown source types are rejected."""
        with pytest.raises(ValidationError, match="literal|type"):
            _SOURCE_ADAPTER.validate_python({"type": "jaeger"})

    def test_extra_fields_rejected(self) -> None:
        """Extra fields (e.g. secrets in YAML) are rejected."""
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python(
                {"type": "langfuse", "secret_key": "in-yaml"}
            )

    def test_non_positive_intervals_rejected(self) -> None:
        """Non-positive polling intervals are rejected."""
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python({"type": "langfuse", "poll_interval": 0})


def _system_config(
    agents: dict[str, Any], trace_sources: dict[str, Any]
) -> SystemConfig:
    return SystemConfig.model_validate(
        {"agents": agents, "trace_sources": trace_sources}
    )


class TestSystemRegistry:
    """System-level trace_sources registry and cross-validation."""

    def test_registry_parses_multiple_types(self) -> None:
        """The registry parses entries of different types."""
        config = _system_config(
            agents={},
            trace_sources={
                "lf": {"type": "langfuse"},
                "collector": {
                    "type": "collector_logs",
                    "namespace": "ns",
                    "service_name": "svc",
                },
            },
        )
        assert isinstance(config.trace_sources["lf"], LangfuseTraceSourceConfig)
        assert isinstance(
            config.trace_sources["collector"], CollectorLogsTraceSourceConfig
        )

    def test_configs_without_trace_fields_load_unchanged(self) -> None:
        """Configs without trace fields load exactly as before."""
        config = SystemConfig.model_validate({})
        assert config.trace_sources == {}

    def test_agent_reference_resolves(self) -> None:
        """A declared agent reference resolves against the registry."""
        config = _system_config(
            agents={
                "default": {"agent": ["a1"]},
                "a1": {"type": "http_api", "trace_source": "lf"},
            },
            trace_sources={"lf": {"type": "langfuse"}},
        )
        assert config.agents is not None
        assert config.agents.agents["a1"].trace_source == "lf"

    def test_unknown_agent_reference_fails_with_location(self) -> None:
        """Unknown agent references fail naming the agent."""
        with pytest.raises(ConfigurationError, match=r"agents\.a1.*missing_src"):
            _system_config(
                agents={
                    "default": {"agent": ["a1"]},
                    "a1": {"type": "http_api", "trace_source": "missing_src"},
                },
                trace_sources={"lf": {"type": "langfuse"}},
            )

    def test_unknown_default_reference_fails(self) -> None:
        """Unknown default references fail naming the location."""
        with pytest.raises(
            ConfigurationError, match=r"agents\.default\.agent_config.*ghost"
        ):
            _system_config(
                agents={
                    "default": {
                        "agent": ["a1"],
                        "agent_config": {"trace_source": "ghost"},
                    },
                    "a1": {"type": "http_api"},
                },
                trace_sources={"lf": {"type": "langfuse"}},
            )

    @pytest.mark.parametrize("reference", ["none", ""])
    def test_disable_sentinels_accepted(self, reference: str) -> None:
        """Disable sentinels are accepted without registry entries."""
        config = _system_config(
            agents={
                "default": {"agent": ["a1"]},
                "a1": {"type": "http_api", "trace_source": reference},
            },
            trace_sources={"lf": {"type": "langfuse"}},
        )
        assert config.agents is not None
        assert config.agents.agents["a1"].trace_source == reference

    @pytest.mark.parametrize("reserved", ["none", "", "   "])
    def test_reserved_registry_keys_rejected(self, reserved: str) -> None:
        """Reserved names in the registry fail at load time."""
        with pytest.raises(ConfigurationError, match="reserved"):
            _system_config(
                agents={"default": {"agent": ["a1"]}, "a1": {"type": "http_api"}},
                trace_sources={reserved: {"type": "langfuse"}},
            )

    def test_reference_without_registry_fails(self) -> None:
        """References fail when the registry is empty."""
        with pytest.raises(ConfigurationError, match="unknown trace_source"):
            _system_config(
                agents={
                    "default": {"agent": ["a1"]},
                    "a1": {"type": "http_api", "trace_source": "lf"},
                },
                trace_sources={},
            )

    def test_non_string_default_reference_fails_cleanly(self) -> None:
        """A non-string default trace_source fails with a clear error."""
        with pytest.raises(ConfigurationError, match="must be a string"):
            _system_config(
                agents={
                    "default": {
                        "agent": ["a1"],
                        "agent_config": {"trace_source": {"type": "langfuse"}},
                    },
                    "a1": {"type": "http_api"},
                },
                trace_sources={},
            )

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_string_overrides_rejected(self, blank: str) -> None:
        """Blank host and endpoint overrides are rejected."""
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python({"type": "langfuse", "host": blank})
        with pytest.raises(ValidationError):
            _SOURCE_ADAPTER.validate_python(
                {
                    "type": "otel",
                    "endpoint": "http://jaeger:16686",
                    "attribute_key": blank,
                }
            )


class TestAgentInheritance:
    """trace_source flows through the 3-level agent config merge."""

    @staticmethod
    def _agents_config(payload: dict[str, Any]) -> AgentsConfig:
        return AgentsConfig.model_validate(payload)

    def test_absent_inherits_default(self) -> None:
        """Absent agent value inherits the default agent config."""
        agents = self._agents_config(
            {
                "default": {
                    "agent": ["a1"],
                    "agent_config": {"trace_source": "lf"},
                },
                "a1": {"type": "http_api"},
            }
        )
        _, merged = agents.resolve_agent_config()
        assert merged["trace_source"] == "lf"

    def test_agent_value_overrides_default(self) -> None:
        """Agent-level values override the default agent config."""
        agents = self._agents_config(
            {
                "default": {
                    "agent": ["a1"],
                    "agent_config": {"trace_source": "lf"},
                },
                "a1": {"type": "http_api", "trace_source": "collector"},
            }
        )
        _, merged = agents.resolve_agent_config()
        assert merged["trace_source"] == "collector"

    def test_none_overrides_inherited_default(self) -> None:
        """The none sentinel overrides an inherited default."""
        agents = self._agents_config(
            {
                "default": {
                    "agent": ["a1"],
                    "agent_config": {"trace_source": "lf"},
                },
                "a1": {"type": "http_api", "trace_source": "none"},
            }
        )
        _, merged = agents.resolve_agent_config()
        assert merged["trace_source"] == "none"

    def test_unset_everywhere_stays_none(self) -> None:
        """Unset everywhere resolves to no trace source."""
        agents = self._agents_config(
            {"default": {"agent": ["a1"]}, "a1": {"type": "http_api"}}
        )
        _, merged = agents.resolve_agent_config()
        assert merged["trace_source"] is None

    def test_agentic_run_agent_accepts_trace_source(self) -> None:
        """AgenticRun agents accept a trace source reference."""
        agents = self._agents_config(
            {
                "default": {"agent": ["run1"]},
                "run1": {
                    "type": "openshift_agentic_run",
                    "namespace": "ns",
                    "trace_source": "collector",
                },
            }
        )
        assert agents.agents["run1"].trace_source == "collector"

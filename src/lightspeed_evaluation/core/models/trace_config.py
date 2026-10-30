"""Trace source configuration models.

Each entry in the system-level ``trace_sources`` registry selects the adapter
that captures live traces during evaluation. Entries are discriminated by
``type``. Credentials are supplied via environment variables, never in YAML.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LangfuseTraceSourceConfig(BaseModel):
    """Query traces from a Langfuse server via the Langfuse SDK.

    Credentials come from LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and
    LANGFUSE_HOST; the optional host field overrides the environment for a
    different instance.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["langfuse"] = "langfuse"
    host: Optional[str] = Field(
        default=None,
        min_length=1,
        pattern=r"\S+",
        description="Langfuse host override (defaults to LANGFUSE_HOST)",
    )
    persistence_timeout: int = Field(
        default=10,
        gt=0,
        description="Seconds to wait for traces to persist before reading",
    )
    poll_interval: int = Field(
        default=2, gt=0, description="Seconds between persistence polls"
    )


class MLflowTraceSourceConfig(BaseModel):
    """Query traces from an MLflow tracking server via the MLflow API."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mlflow"] = "mlflow"
    tracking_uri: Optional[str] = Field(
        default=None,
        min_length=1,
        pattern=r"\S+",
        description="MLflow tracking URI override (defaults to MLFLOW_TRACKING_URI)",
    )
    persistence_timeout: int = Field(
        default=10,
        gt=0,
        description="Seconds to wait for traces to persist before reading",
    )
    poll_interval: int = Field(
        default=2, gt=0, description="Seconds between persistence polls"
    )


class CollectorLogsTraceSourceConfig(BaseModel):
    """Query templog records from an OTEL Collector via its admin API."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["collector_logs"] = "collector_logs"
    transport: Literal["kubernetes_service_proxy"] = "kubernetes_service_proxy"
    namespace: str = Field(
        ...,
        min_length=1,
        pattern=r"\S+",
        description="Kubernetes namespace hosting the collector service",
    )
    service_name: str = Field(
        ...,
        min_length=1,
        pattern=r"\S+",
        description="Collector service name",
    )
    service_port: str = Field(
        default="8080",
        min_length=1,
        pattern=r"\S+",
        description="Collector service port",
    )
    persistence_timeout: int = Field(
        default=30,
        gt=0,
        description="Seconds to wait for records to persist before reading",
    )
    stable_reads: int = Field(
        default=2,
        gt=0,
        description="Consecutive identical reads required before a snapshot is stable",
    )


class OtelTraceSourceConfig(BaseModel):
    """Query native OTEL spans from Jaeger, Tempo, or a compatible backend."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["otel"] = "otel"
    endpoint: str = Field(
        ...,
        min_length=1,
        pattern=r"\S+",
        description="Backend query endpoint (e.g. Jaeger API URL)",
    )
    query_by: Literal["span_attribute", "trace_id"] = Field(
        default="span_attribute",
        description="How traces are correlated to conversations",
    )
    attribute_key: str = Field(
        default="gen_ai.conversation.id",
        min_length=1,
        pattern=r"\S+",
        description="Span attribute holding the conversation identifier",
    )


TraceSourceConfig = Annotated[
    LangfuseTraceSourceConfig
    | MLflowTraceSourceConfig
    | CollectorLogsTraceSourceConfig
    | OtelTraceSourceConfig,
    Field(discriminator="type"),
]

"""Trace ingestion and normalization.

Platform adapters convert already-fetched trace payloads (MLflow, Langfuse,
OpenTelemetry) into the internal span models defined in
``core.models.trace``. The capture pipeline planned in the trace ingestion
spec will also live here.
"""

from lightspeed_evaluation.core.tracing.langfuse_adapter import (
    span_from_langfuse,
    trace_from_langfuse,
)
from lightspeed_evaluation.core.tracing.mlflow_adapter import (
    span_from_mlflow,
    trace_from_mlflow,
)
from lightspeed_evaluation.core.tracing.otel_adapter import (
    span_from_otel,
    trace_from_otel,
)
from lightspeed_evaluation.core.tracing.session import (
    session_from_traces,
)

__all__ = [
    "session_from_traces",
    "span_from_langfuse",
    "span_from_mlflow",
    "span_from_otel",
    "trace_from_langfuse",
    "trace_from_mlflow",
    "trace_from_otel",
]

"""Platform-agnostic internal trace/span models.

This module defines the contract between platform-specific trace ingestion
(MLflow, Langfuse, OpenTelemetry) and platform-agnostic trace evaluation.

These models are intentionally unused by the evaluation pipeline today.
Ingestion adapters convert already-fetched platform payloads into this shape;
evaluation code can consume ``Trace`` / ``TraceSession`` without knowing the
source platform.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpanType(str, Enum):
    """Standard span types normalized across observability platforms."""

    AGENT = "AGENT"
    CHAIN = "CHAIN"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    GENERATION = "GENERATION"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"
    EVENT = "EVENT"
    UNKNOWN = "UNKNOWN"


class SpanStatus(str, Enum):
    """Normalized span execution status."""

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class TracePlatform(str, Enum):
    """Observability platform that produced the original trace payload.

    Distinct from a future user-defined ``source`` in eval data, which may
    describe how traces were selected or labeled for evaluation.
    """

    MLFLOW = "mlflow"
    LANGFUSE = "langfuse"
    OTEL = "otel"
    UNKNOWN = "unknown"


# Platform type strings → SpanType. Keys are upper-cased at lookup time.
SPAN_TYPE_MAP: dict[str, SpanType] = {
    # Shared / MLflow span types
    "AGENT": SpanType.AGENT,
    "CHAIN": SpanType.CHAIN,
    "TOOL": SpanType.TOOL,
    "RETRIEVER": SpanType.RETRIEVER,
    "LLM": SpanType.GENERATION,
    "CHAT_MODEL": SpanType.GENERATION,
    "GENERATION": SpanType.GENERATION,
    "EMBEDDING": SpanType.EMBEDDING,
    "RERANKER": SpanType.RERANKER,
    "GUARDRAIL": SpanType.GUARDRAIL,
    "EVALUATOR": SpanType.EVALUATOR,
    "PARSER": SpanType.CHAIN,
    "UNKNOWN": SpanType.UNKNOWN,
    # Langfuse observation types
    "SPAN": SpanType.CHAIN,
    "EVENT": SpanType.EVENT,
    # OpenTelemetry GenAI / common attribute values
    "CHAT": SpanType.GENERATION,
    "TEXT_COMPLETION": SpanType.GENERATION,
    "EXECUTE_TOOL": SpanType.TOOL,
    "INVOKE_AGENT": SpanType.AGENT,
    "RETRIEVE": SpanType.RETRIEVER,
    "RERANK": SpanType.RERANKER,
}


def normalize_span_type(raw: str | None) -> SpanType:
    """Map a platform-specific type string to :class:`SpanType`.

    Args:
        raw: Platform type string (e.g. ``LLM``, ``execute_tool``,
            ``gen_ai.operation.name`` value). ``None`` or blank maps to
            :attr:`SpanType.UNKNOWN`.

    Returns:
        Normalized :class:`SpanType`, or :attr:`SpanType.UNKNOWN` when the
        input is empty or unmapped.
    """
    if raw is None or not str(raw).strip():
        return SpanType.UNKNOWN
    key = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    # Strip common prefixes (e.g. gen_ai.operation.name values already normalized)
    if key.startswith("GEN_AI_"):
        key = key.removeprefix("GEN_AI_")
    return SPAN_TYPE_MAP.get(key, SpanType.UNKNOWN)


class TokenUsage(BaseModel):
    """Optional token usage for LLM-related spans."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: Optional[int] = Field(
        default=None, ge=0, description="Prompt / input tokens"
    )
    output_tokens: Optional[int] = Field(
        default=None, ge=0, description="Completion / output tokens"
    )
    total_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total tokens when provided by the platform",
    )


class ModelInfo(BaseModel):
    """Optional model identity for LLM-related spans."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, description="Model name or ID")
    provider: Optional[str] = Field(
        default=None, description="Provider (e.g. openai, anthropic)"
    )
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=0, description="Max generation tokens"
    )


class LLMSpanData(BaseModel):
    """Optional LLM-specific fields attached to a span."""

    model_config = ConfigDict(extra="forbid")

    model: Optional[ModelInfo] = Field(default=None, description="Model information")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Token counts")
    cost_usd: Optional[float] = Field(
        default=None, ge=0.0, description="Estimated or reported cost in USD"
    )


class Span(BaseModel):
    """Normalized span: contract unit between ingestion and evaluation.

    Parent-child relationships are represented via ``parent_span_id``. The
    optional ``children`` list is populated by :meth:`Trace.build_span_tree`
    and is not required when constructing spans from flat platform payloads.
    """

    model_config = ConfigDict(extra="forbid")

    span_id: str = Field(..., min_length=1, description="Unique span identifier")
    trace_id: str = Field(..., min_length=1, description="Owning trace identifier")
    parent_span_id: Optional[str] = Field(
        default=None, description="Parent span ID; None for root spans"
    )
    name: str = Field(..., min_length=1, description="Human-readable span name")
    span_type: SpanType = Field(
        default=SpanType.UNKNOWN, description="Normalized span type"
    )
    status: SpanStatus = Field(default=SpanStatus.UNSET, description="Execution status")
    status_message: Optional[str] = Field(
        default=None, description="Status detail or error message"
    )
    start_time: Optional[datetime] = Field(default=None, description="Span start time")
    end_time: Optional[datetime] = Field(default=None, description="Span end time")
    latency: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Span duration in seconds (all span types)",
    )
    inputs: Optional[Any] = Field(default=None, description="Span inputs / arguments")
    outputs: Optional[Any] = Field(default=None, description="Span outputs / results")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary span metadata"
    )
    tags: list[str] = Field(default_factory=list, description="Span tags / labels")
    llm: Optional[LLMSpanData] = Field(
        default=None, description="Optional LLM-specific attributes"
    )
    children: list[Span] = Field(
        default_factory=list,
        description="Child spans (populated when building a tree; omit for flat lists)",
    )


class Trace(BaseModel):
    """Normalized single trace containing a flat list of spans."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(..., min_length=1, description="Unique trace identifier")
    name: Optional[str] = Field(default=None, description="Trace display name")
    session_id: Optional[str] = Field(
        default=None, description="Session that groups related traces"
    )
    user_id: Optional[str] = Field(default=None, description="Optional end-user ID")
    platform: TracePlatform = Field(
        default=TracePlatform.UNKNOWN, description="Originating observability platform"
    )
    start_time: Optional[datetime] = Field(default=None, description="Trace start time")
    end_time: Optional[datetime] = Field(default=None, description="Trace end time")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Trace-level metadata"
    )
    tags: list[str] = Field(default_factory=list, description="Trace-level tags")
    spans: list[Span] = Field(
        default_factory=list,
        description="Flat span list; parent-child via parent_span_id",
    )

    @model_validator(mode="after")
    def _validate_spans(self) -> Trace:
        """Ensure spans belong to this trace and form a valid parent graph.

        Rejects mismatched ``trace_id`` values, duplicate ``span_id`` values,
        and cyclic ``parent_span_id`` chains (including self-parenting).
        """
        seen_ids: set[str] = set()
        for span in self.spans:
            if span.trace_id != self.trace_id:
                raise ValueError(
                    f"span {span.span_id!r} has trace_id {span.trace_id!r}, "
                    f"expected {self.trace_id!r}"
                )
            if span.span_id in seen_ids:
                raise ValueError(f"duplicate span_id {span.span_id!r} in trace")
            seen_ids.add(span.span_id)

        # Follow only parents that exist in this trace (missing parents are roots).
        parent_of = {
            span.span_id: span.parent_span_id
            for span in self.spans
            if span.parent_span_id is not None and span.parent_span_id in seen_ids
        }
        for start_id in parent_of:
            seen_path: set[str] = set()
            current: Optional[str] = start_id
            while current is not None:
                if current in seen_path:
                    raise ValueError(
                        f"cyclic parent_span_id chain involving {current!r}"
                    )
                seen_path.add(current)
                current = parent_of.get(current)
        return self

    def build_span_tree(self) -> list[Span]:
        """Return root spans with ``children`` populated from ``parent_span_id``.

        Spans whose parent is missing from the flat list are treated as roots.
        Does not mutate the flat ``spans`` list on this model. Duplicate IDs and
        parent cycles are rejected by model validation before this runs.

        Returns:
            Root :class:`Span` instances with ``children`` nested according to
            ``parent_span_id`` relationships.
        """
        by_id: dict[str, Span] = {
            span.span_id: span.model_copy(update={"children": []}, deep=True)
            for span in self.spans
        }
        roots: list[Span] = []
        for span in by_id.values():
            parent_id = span.parent_span_id
            if parent_id and parent_id in by_id:
                by_id[parent_id].children.append(span)
            else:
                roots.append(span)
        return roots


class TraceSession(BaseModel):
    """Session-grouped multi-trace input for evaluation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1, description="Session identifier")
    traces: list[Trace] = Field(
        default_factory=list, description="Traces belonging to this session"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Session-level metadata"
    )
    tags: list[str] = Field(default_factory=list, description="Session-level tags")

    @model_validator(mode="after")
    def _align_trace_session_ids(self) -> TraceSession:
        """Fill missing trace.session_id and reject conflicting values."""
        aligned: list[Trace] = []
        for trace in self.traces:
            if trace.session_id is None:
                aligned.append(trace.model_copy(update={"session_id": self.session_id}))
            elif trace.session_id != self.session_id:
                raise ValueError(
                    f"trace {trace.trace_id!r} has session_id {trace.session_id!r}, "
                    f"expected {self.session_id!r}"
                )
            else:
                aligned.append(trace)
        self.traces = aligned
        return self

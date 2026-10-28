"""Tests for internal trace models and platform adapters."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.trace import (
    ModelInfo,
    Span,
    SpanEvent,
    SpanStatus,
    SpanType,
    TokenUsage,
    Trace,
    TracePlatform,
    TraceSession,
    normalize_span_type,
)
from lightspeed_evaluation.core.models.trace_adapters import (
    _as_datetime,
    session_from_traces,
    span_from_langfuse,
    span_from_mlflow,
    span_from_otel,
    trace_from_langfuse,
    trace_from_mlflow,
    trace_from_otel,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError


class TestAsDatetime:
    """UTC normalization for all supported time representations."""

    def test_naive_iso_assumes_utc(self) -> None:
        """ISO strings without timezone are treated as UTC."""
        result = _as_datetime("2024-01-01T12:00:00")
        assert result == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

    def test_zulu_and_offset_iso_normalize_to_utc(self) -> None:
        """Z and explicit offsets both land on UTC-aware datetimes."""
        assert _as_datetime("2024-01-01T12:00:00Z") == datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC
        )
        assert _as_datetime("2024-01-01T12:00:00+05:30") == datetime(
            2024, 1, 1, 6, 30, 0, tzinfo=UTC
        )

    def test_naive_datetime_and_epoch_are_utc(self) -> None:
        """Naive datetime objects and epoch values are UTC-aware."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        assert _as_datetime(naive) == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        epoch = _as_datetime(1_700_000_000)
        assert epoch is not None
        assert epoch.tzinfo == UTC


class TestTraceTreeAndSession:
    """Tree relationships, type mapping, and session grouping."""

    def test_normalize_span_type(self) -> None:
        """Map shared platform type strings."""
        assert normalize_span_type("TOOL") is SpanType.TOOL
        assert normalize_span_type("llm") is SpanType.GENERATION
        assert normalize_span_type("GENERATION") is SpanType.GENERATION
        assert normalize_span_type("SPAN") is SpanType.CHAIN
        assert normalize_span_type("execute_tool") is SpanType.TOOL
        assert normalize_span_type(None) is SpanType.UNKNOWN
        assert normalize_span_type("custom-xyz") is SpanType.UNKNOWN

    def test_build_span_tree(self) -> None:
        """Parent-child links produce a span tree."""
        trace = Trace(
            trace_id="t1",
            spans=[
                Span(
                    span_id="root", trace_id="t1", name="root", span_type=SpanType.AGENT
                ),
                Span(
                    span_id="child",
                    trace_id="t1",
                    parent_span_id="root",
                    name="child",
                    span_type=SpanType.TOOL,
                ),
                Span(
                    span_id="orphan",
                    trace_id="t1",
                    parent_span_id="missing",
                    name="orphan",
                    span_type=SpanType.EVENT,
                ),
            ],
        )
        roots = trace.build_span_tree()
        assert {r.span_id for r in roots} == {"root", "orphan"}
        root = next(r for r in roots if r.span_id == "root")
        assert len(root.children) == 1
        assert root.children[0].span_id == "child"
        # Flat list unchanged
        assert all(not s.children for s in trace.spans)

    def test_mismatched_span_trace_id_rejected(self) -> None:
        """Spans must belong to the owning trace."""
        with pytest.raises(ValidationError, match="trace_id"):
            Trace(
                trace_id="t1",
                spans=[Span(span_id="s1", trace_id="other", name="s1")],
            )

    def test_duplicate_span_id_rejected(self) -> None:
        """Duplicate span_id values cannot silently overwrite in the tree."""
        with pytest.raises(ValidationError, match="duplicate span_id"):
            Trace(
                trace_id="t1",
                spans=[
                    Span(span_id="s1", trace_id="t1", name="first"),
                    Span(span_id="s1", trace_id="t1", name="second"),
                ],
            )

    def test_self_parent_span_rejected(self) -> None:
        """A span cannot be its own parent."""
        with pytest.raises(ValidationError, match="cyclic parent_span_id"):
            Trace(
                trace_id="t1",
                spans=[
                    Span(
                        span_id="s1",
                        trace_id="t1",
                        parent_span_id="s1",
                        name="loop",
                    )
                ],
            )

    def test_multi_span_parent_cycle_rejected(self) -> None:
        """Parent chains that form a cycle are rejected."""
        with pytest.raises(ValidationError, match="cyclic parent_span_id"):
            Trace(
                trace_id="t1",
                spans=[
                    Span(
                        span_id="a",
                        trace_id="t1",
                        parent_span_id="c",
                        name="a",
                    ),
                    Span(
                        span_id="b",
                        trace_id="t1",
                        parent_span_id="a",
                        name="b",
                    ),
                    Span(
                        span_id="c",
                        trace_id="t1",
                        parent_span_id="b",
                        name="c",
                    ),
                ],
            )

    def test_trace_session_aligns_session_id(self) -> None:
        """Session fills missing trace.session_id values."""
        session = TraceSession(
            session_id="sess-1",
            traces=[Trace(trace_id="t1", spans=[])],
        )
        assert session.traces[0].session_id == "sess-1"

    def test_trace_session_rejects_conflict(self) -> None:
        """Conflicting session IDs are rejected."""
        with pytest.raises(ValidationError, match="session_id"):
            TraceSession(
                session_id="sess-1",
                traces=[Trace(trace_id="t1", session_id="other", spans=[])],
            )

    def test_session_from_traces(self) -> None:
        """Group multiple traces under one session."""
        t1 = Trace(trace_id="a", platform=TracePlatform.MLFLOW, spans=[])
        t2 = Trace(trace_id="b", platform=TracePlatform.LANGFUSE, spans=[])
        session = session_from_traces("sess-x", [t1, t2], tags=["nightly"])
        assert session.session_id == "sess-x"
        assert len(session.traces) == 2
        assert session.tags == ["nightly"]
        assert all(t.session_id == "sess-x" for t in session.traces)


class TestMLflowAdapter:
    """MLflow payload → internal model."""

    def test_trace_from_mlflow_sample_shape(self) -> None:
        """Convert MLflow-style spans payload."""
        payload = {
            "spans": [
                {
                    "name": "agent-run",
                    "context": {
                        "span_id": "span-1",
                        "trace_id": "trace-abc",
                    },
                    "parent_id": None,
                    "start_time": 1_700_000_000_000_000_000,
                    "end_time": 1_700_000_000_500_000_000,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.spanType": '"AGENT"',
                        "mlflow.spanInputs": '{"query": "hi"}',
                        "mlflow.spanOutputs": '{"response": "hello"}',
                        "conversation_group_id": '"conv_1"',
                    },
                },
                {
                    "name": "llm-call",
                    "context": {
                        "span_id": "span-2",
                        "trace_id": "trace-abc",
                    },
                    "parent_id": "span-1",
                    "start_time": 1_700_000_000_100_000_000,
                    "end_time": 1_700_000_000_400_000_000,
                    "status_code": "OK",
                    "attributes": {
                        "mlflow.spanType": '"LLM"',
                        "input_tokens": "10",
                        "output_tokens": "20",
                    },
                },
            ],
            "request": '{"query": "hi"}',
            "response": '{"response": "hello"}',
        }
        trace = trace_from_mlflow(payload)
        assert trace.trace_id == "trace-abc"
        assert trace.platform is TracePlatform.MLFLOW
        assert len(trace.spans) == 2
        root = next(s for s in trace.spans if s.span_id == "span-1")
        child = next(s for s in trace.spans if s.span_id == "span-2")
        assert root.span_type is SpanType.AGENT
        assert root.source is TracePlatform.MLFLOW
        assert root.attributes == {}
        assert root.metadata["mlflow.inputs"] == {"query": "hi"}
        assert root.metadata["mlflow.conversation_group_id"] == "conv_1"
        assert child.parent_span_id == "span-1"
        assert child.span_type is SpanType.GENERATION
        assert child.llm is not None
        assert child.llm.token_usage is not None
        assert child.llm.token_usage.input_tokens == 10
        assert len(trace.build_span_tree()) == 1

    def test_span_from_mlflow_requires_ids(self) -> None:
        """Missing identity fields raise DataValidationError."""
        with pytest.raises(DataValidationError, match="trace_id"):
            span_from_mlflow({"context": {"span_id": "s1"}, "name": "x"})

    def test_non_llm_span_has_latency_without_llm_block(self) -> None:
        """Tool/other spans get latency in seconds; timing alone does not create llm."""
        span = span_from_mlflow(
            {
                "name": "tool-call",
                "context": {"span_id": "s-tool", "trace_id": "t1"},
                "start_time": 1_700_000_000_000_000_000,
                "end_time": 1_700_000_000_100_000_000,
                "status_code": "OK",
                "attributes": {"mlflow.spanType": '"TOOL"'},
            }
        )
        assert span.span_type is SpanType.TOOL
        assert span.latency == pytest.approx(0.1)
        assert span.llm is None

    def test_llm_span_keeps_derived_latency(self) -> None:
        """LLM spans include span-level latency derived from start/end."""
        span = span_from_mlflow(
            {
                "name": "llm-call",
                "context": {"span_id": "s-llm", "trace_id": "t1"},
                "start_time": 1_700_000_000_000_000_000,
                "end_time": 1_700_000_000_250_000_000,
                "status_code": "OK",
                "attributes": {
                    "mlflow.spanType": '"LLM"',
                    "input_tokens": "3",
                    "output_tokens": "4",
                },
            }
        )
        assert span.llm is not None
        assert span.llm.token_usage is not None
        assert span.latency == pytest.approx(0.25)

    def test_malformed_numeric_attrs_do_not_abort_conversion(self) -> None:
        """Bad provider numerics are dropped instead of raising."""
        span = span_from_mlflow(
            {
                "name": "llm-call",
                "context": {"span_id": "s-llm", "trace_id": "t1"},
                "status_code": "OK",
                "attributes": {
                    "mlflow.spanType": '"LLM"',
                    "input_tokens": "not-a-number",
                    "output_tokens": {"nested": True},
                    "cost_usd": "n/a",
                    "model": "gpt-test",
                },
            }
        )
        assert span.llm is not None
        assert span.llm.model is not None
        assert span.llm.model.name == "gpt-test"
        assert span.llm.token_usage is None
        assert span.llm.cost_usd is None

    def test_zero_cost_and_zero_tokens_are_preserved(self) -> None:
        """Zero is a valid reported value and must not be treated as missing."""
        span = span_from_mlflow(
            {
                "name": "llm-call",
                "context": {"span_id": "s-llm", "trace_id": "t1"},
                "status_code": "OK",
                "attributes": {
                    "mlflow.spanType": '"LLM"',
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                },
            }
        )
        assert span.llm is not None
        assert span.llm.token_usage is not None
        assert span.llm.token_usage.input_tokens == 0
        assert span.llm.token_usage.output_tokens == 0
        assert span.llm.cost_usd == 0.0

    def test_negative_cost_is_dropped(self) -> None:
        """Negative cost is invalid for ge=0 fields and is omitted."""
        span = span_from_mlflow(
            {
                "name": "llm-call",
                "context": {"span_id": "s-llm", "trace_id": "t1"},
                "status_code": "OK",
                "attributes": {
                    "mlflow.spanType": '"LLM"',
                    "input_tokens": 1,
                    "cost_usd": -1.5,
                },
            }
        )
        assert span.llm is not None
        assert span.llm.cost_usd is None
        assert span.llm.token_usage is not None
        assert span.llm.token_usage.input_tokens == 1

    def test_negative_tokens_are_dropped(self) -> None:
        """Negative token counts are omitted; valid counts are kept."""
        span = span_from_mlflow(
            {
                "name": "llm-call",
                "context": {"span_id": "s-llm", "trace_id": "t1"},
                "status_code": "OK",
                "attributes": {
                    "mlflow.spanType": '"LLM"',
                    "input_tokens": -3,
                    "output_tokens": 4,
                    "total_tokens": -1,
                },
            }
        )
        assert span.llm is not None
        assert span.llm.token_usage is not None
        assert span.llm.token_usage.input_tokens is None
        assert span.llm.token_usage.output_tokens == 4
        assert span.llm.token_usage.total_tokens is None


class TestTokenAndModelConstraints:
    """Direct model constraints for non-negative token fields."""

    def test_negative_token_usage_rejected(self) -> None:
        """TokenUsage rejects negative counts."""
        with pytest.raises(ValidationError):
            TokenUsage(input_tokens=-1)
        with pytest.raises(ValidationError):
            TokenUsage(output_tokens=-1)
        with pytest.raises(ValidationError):
            TokenUsage(total_tokens=-1)

    def test_negative_max_tokens_rejected(self) -> None:
        """ModelInfo rejects negative max_tokens."""
        with pytest.raises(ValidationError):
            ModelInfo(max_tokens=-1)


class TestLangfuseAdapter:
    """Langfuse payload → internal model."""

    def test_trace_from_langfuse(self) -> None:
        """Convert Langfuse trace with observations."""
        payload = {
            "id": "lf-trace-1",
            "name": "chat",
            "sessionId": "sess-9",
            "userId": "user-1",
            "tags": ["prod"],
            "metadata": {"env": "test"},
            "observations": [
                {
                    "id": "obs-1",
                    "traceId": "lf-trace-1",
                    "type": "SPAN",
                    "name": "pipeline",
                    "startTime": "2024-01-01T00:00:00Z",
                    "endTime": "2024-01-01T00:00:02Z",
                    "level": "DEFAULT",
                    "input": {"q": "hi"},
                    "output": {"a": "yo"},
                },
                {
                    "id": "obs-2",
                    "traceId": "lf-trace-1",
                    "parentObservationId": "obs-1",
                    "type": "GENERATION",
                    "name": "chat-completion",
                    "model": "gpt-4o-mini",
                    "usage": {"input": 5, "output": 7, "total": 12},
                    "calculatedTotalCost": 0.001,
                    "latency": 0.25,
                    "level": "DEFAULT",
                },
            ],
        }
        trace = trace_from_langfuse(payload)
        assert trace.session_id == "sess-9"
        assert trace.user_id == "user-1"
        assert trace.platform is TracePlatform.LANGFUSE
        gen = next(s for s in trace.spans if s.span_id == "obs-2")
        assert gen.span_type is SpanType.GENERATION
        assert gen.parent_span_id == "obs-1"
        assert gen.llm is not None
        assert gen.llm.model is not None
        assert gen.llm.model.name == "gpt-4o-mini"
        assert gen.llm.token_usage is not None
        assert gen.llm.token_usage.total_tokens == 12
        assert gen.llm.cost_usd == 0.001
        assert gen.latency == pytest.approx(0.25)
        pipeline = next(s for s in trace.spans if s.span_id == "obs-1")
        assert pipeline.latency == pytest.approx(2.0)
        assert pipeline.llm is None

    def test_span_from_langfuse_error_level(self) -> None:
        """ERROR level maps to SpanStatus.ERROR."""
        span = span_from_langfuse(
            {
                "id": "obs-err",
                "traceId": "t1",
                "type": "EVENT",
                "name": "fail",
                "level": "ERROR",
                "statusMessage": "boom",
            }
        )
        assert span.status is SpanStatus.ERROR
        assert span.status_message == "boom"

    def test_langfuse_span_without_llm_signal_has_no_llm_block(self) -> None:
        """Langfuse SPAN keeps latency on the span; timing alone does not populate llm."""
        span = span_from_langfuse(
            {
                "id": "obs-1",
                "traceId": "t1",
                "type": "SPAN",
                "name": "pipeline",
                "startTime": "2024-01-01T00:00:00Z",
                "endTime": "2024-01-01T00:00:02Z",
                "latency": 2.0,
                "level": "DEFAULT",
            }
        )
        assert span.latency == pytest.approx(2.0)
        assert span.llm is None

    def test_langfuse_naive_iso_times_are_utc(self) -> None:
        """Naive ISO timestamps from Langfuse are normalized to UTC."""
        span = span_from_langfuse(
            {
                "id": "obs-1",
                "traceId": "t1",
                "type": "SPAN",
                "name": "pipeline",
                "startTime": "2024-01-01T00:00:00",
                "endTime": "2024-01-01T00:00:02",
                "level": "DEFAULT",
            }
        )
        assert span.start_time == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert span.end_time == datetime(2024, 1, 1, 0, 0, 2, tzinfo=UTC)
        assert span.latency == pytest.approx(2.0)


class TestOtelAdapter:
    """OpenTelemetry payload → internal model."""

    def test_trace_from_otel_flat_spans(self) -> None:
        """Convert flat OTEL span list with GenAI attributes."""
        payload = {
            "traceId": "otel-trace-1",
            "spans": [
                {
                    "traceId": "otel-trace-1",
                    "spanId": "s-root",
                    "name": "agent",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000002000000000",
                    "status": {"code": "STATUS_CODE_OK"},
                    "attributes": [
                        {
                            "key": "gen_ai.operation.name",
                            "value": {"stringValue": "invoke_agent"},
                        },
                        {"key": "session.id", "value": {"stringValue": "sess-otel"}},
                    ],
                },
                {
                    "traceId": "otel-trace-1",
                    "spanId": "s-llm",
                    "parentSpanId": "s-root",
                    "name": "chat",
                    "startTimeUnixNano": "1700000000500000000",
                    "endTimeUnixNano": "1700000001500000000",
                    "status": {"code": "STATUS_CODE_OK"},
                    "attributes": {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": "gpt-4.1",
                        "gen_ai.system": "openai",
                        "gen_ai.usage.input_tokens": 11,
                        "gen_ai.usage.output_tokens": 13,
                        "gen_ai.prompt": "hello",
                        "gen_ai.completion": "world",
                    },
                },
            ],
        }
        trace = trace_from_otel(payload)
        assert trace.platform is TracePlatform.OTEL
        assert trace.session_id == "sess-otel"
        assert isinstance(trace.start_time, datetime)
        assert trace.start_time.tzinfo == UTC
        root = next(s for s in trace.spans if s.span_id == "s-root")
        assert root.latency == pytest.approx(2.0)
        llm = next(s for s in trace.spans if s.span_id == "s-llm")
        assert llm.span_type is SpanType.GENERATION
        assert llm.source is TracePlatform.OTEL
        assert llm.attributes["generation.prompt"] == "hello"
        assert llm.attributes["generation.completion"] == "world"
        assert llm.attributes["model.provider"] == "openai"
        assert llm.attributes["tokens.input"] == 11
        assert llm.latency == pytest.approx(1.0)
        assert llm.llm is not None
        assert llm.llm.model is not None
        assert llm.llm.model.provider == "openai"
        assert llm.llm.token_usage is not None
        assert llm.llm.token_usage.input_tokens == 11

    def test_span_from_otel_requires_ids(self) -> None:
        """Missing spanId raises DataValidationError."""
        with pytest.raises(DataValidationError, match="spanId"):
            span_from_otel({"traceId": "t1", "name": "x"})

    def test_otel_unwraps_array_kvlist_and_bytes_values(self) -> None:
        """OTLP arrayValue, kvlistValue, and bytesValue are unwrapped."""
        span = span_from_otel(
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "attrs",
                "attributes": [
                    {
                        "key": "tags",
                        "value": {
                            "arrayValue": {
                                "values": [
                                    {"stringValue": "a"},
                                    {"stringValue": "b"},
                                ]
                            }
                        },
                    },
                    {
                        "key": "meta",
                        "value": {
                            "kvlistValue": {
                                "values": [
                                    {
                                        "key": "env",
                                        "value": {"stringValue": "test"},
                                    },
                                    {
                                        "key": "count",
                                        "value": {"intValue": "2"},
                                    },
                                ]
                            }
                        },
                    },
                    {
                        "key": "payload",
                        "value": {"bytesValue": "aGVsbG8="},
                    },
                ],
            }
        )
        assert span.metadata["otel.tags"] == ["a", "b"]
        assert span.metadata["otel.meta"] == {"env": "test", "count": 2}
        assert span.metadata["otel.payload"] == "aGVsbG8="


@pytest.mark.parametrize(
    "adapter", [span_from_mlflow, span_from_langfuse, span_from_otel]
)
def test_structured_retrieval_documents(
    adapter: Callable[[dict[str, Any]], Span],
) -> None:
    """Common document shapes normalize identically across sources."""
    span = adapter(
        {
            "traceId": "t",
            "trace_id": "t",
            "spanId": "s",
            "span_id": "s",
            "id": "s",
            "type": "RETRIEVER",
            "input": {"query": "pods"},
            "output": {
                "documents": [
                    {
                        "page_content": "text",
                        "metadata": {"source": "guide"},
                        "score": 0.0,
                    }
                ]
            },
            "attributes": {
                "mlflow.spanType": "RETRIEVER",
                "mlflow.spanInputs": {"query": "pods"},
                "mlflow.spanOutputs": [
                    {"text": "text", "source": "guide", "score": 0.0}
                ],
                "input.value": {"query": "pods"},
                "output.value": [{"content": "text", "source": "guide", "score": 0.0}],
            },
        }
    )
    assert span.attributes["retrieval.query"] == "pods"
    assert span.attributes["retrieval.chunks"] == ["text"]
    assert span.attributes["retrieval.sources"] == ["guide"]
    assert span.attributes["retrieval.scores"] == [0.0]


@pytest.mark.parametrize("result", [[], {}, "", False, 0])
def test_otel_preserves_explicit_empty_result(result: Any) -> None:
    """Only None triggers the input/output fallback."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "name": "display",
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.call.id": "call-1",
                "input.value": "{}",
                "output.value": result,
            },
        }
    )
    assert span.attributes["tool.name"] == "search"
    assert span.attributes["tool.arguments"] == {}
    assert span.attributes["tool.result"] == result
    assert span.attributes["tool.call_id"] == "call-1"


def test_langfuse_json_and_metadata_tool_name() -> None:
    """Metadata provides tool identity when observation name is missing."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "TOOL",
            "metadata": {"tool_name": "search", "custom": 1},
            "input": '{"query":"pods"}',
            "output": '{"results":[]}',
        }
    )
    assert span.attributes["tool.name"] == "search"
    assert span.attributes["tool.arguments"] == {"query": "pods"}
    assert span.attributes["tool.result"] == {"results": []}
    assert span.metadata == {"langfuse.custom": 1}


def test_langfuse_temperature_matches_llm() -> None:
    """Temperature is preserved in both standard and typed interfaces."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "GENERATION",
            "modelParameters": {"temperature": 0.0},
        }
    )
    assert span.attributes["model.temperature"] == 0.0
    assert span.llm is not None and span.llm.model is not None
    assert span.llm.model.temperature == 0.0


def test_otel_reasoning_tokens() -> None:
    """Reasoning usage is normalized separately from output tokens."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.usage.reasoning.output_tokens": "20",
            },
        }
    )
    assert span.attributes["tokens.reasoning"] == 20


def test_mlflow_span_preserves_events_and_namespaced_extras() -> None:
    """MLflow sub-span events and unmapped attributes remain available."""
    span = span_from_mlflow(
        {
            "context": {"trace_id": "t", "span_id": "s"},
            "attributes": {
                "mlflow.spanType": "LLM",
                "mlflow.spanCustom": 1,
                "llm.usage.prompt_tokens": 0,
                "mlflow.spanCost": 0.0,
            },
            "events": [
                {
                    "name": "checkpoint",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "attributes": {"step": 1},
                }
            ],
        }
    )
    assert span.metadata["mlflow.spanCustom"] == 1
    assert span.events[0].attributes == {"step": 1}
    assert span.events[0].timestamp == datetime(2024, 1, 1, tzinfo=UTC)
    assert span.attributes["tokens.input"] == 0
    assert span.attributes["cost.usd"] == 0.0
    assert span.llm is not None and span.llm.token_usage is not None
    assert span.llm.token_usage.input_tokens == span.attributes["tokens.input"]


def test_event_span_semantic_attributes() -> None:
    """EVENT spans expose event body and phase, not just metadata."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "EVENT",
            "name": "phase_transition",
            "input": '{"phase":"analysis"}',
        }
    )
    assert span.attributes == {
        "event.name": "phase_transition",
        "event.phase": "analysis",
        "event.body": {"phase": "analysis"},
    }


class TestStandardAttributes:
    """Tests for standard attribute keys, source, events (spec §3)."""

    def test_mlflow_tool_span_maps_standard_attributes(self) -> None:
        """MLflow TOOL span maps inputs/outputs to tool.* keys."""
        span = span_from_mlflow(
            {
                "context": {"trace_id": "t1", "span_id": "s1"},
                "name": "search_knowledge_base",
                "status_code": "ERROR",
                "status_message": "timeout",
                "attributes": {
                    "mlflow.spanType": "TOOL",
                    "mlflow.spanInputs": {"query": "user profile"},
                    "mlflow.spanOutputs": {"results": []},
                },
            }
        )
        assert span.span_type is SpanType.TOOL
        assert span.source is TracePlatform.MLFLOW
        assert span.attributes["tool.name"] == "search_knowledge_base"
        assert span.attributes["tool.arguments"] == {"query": "user profile"}
        assert span.attributes["tool.result"] == {"results": []}
        assert span.attributes["tool.error"] == "timeout"

    def test_langfuse_generation_span_maps_standard_attributes(self) -> None:
        """Langfuse GENERATION span maps input/output/usage to standard keys."""
        span = span_from_langfuse(
            {
                "id": "obs-1",
                "traceId": "t1",
                "type": "GENERATION",
                "name": "llm-call",
                "model": "gpt-4",
                "usage": {"input": 50, "output": 100, "total": 150},
                "input": [{"role": "user", "content": "hi"}],
                "output": "The answer",
            }
        )
        assert span.span_type is SpanType.GENERATION
        assert span.source is TracePlatform.LANGFUSE
        assert span.attributes["generation.prompt"] == [
            {"role": "user", "content": "hi"}
        ]
        assert span.attributes["generation.completion"] == "The answer"
        assert span.attributes["model.name"] == "gpt-4"
        assert span.attributes["tokens.input"] == 50
        assert span.attributes["tokens.output"] == 100
        assert span.attributes["tokens.total"] == 150

    def test_langfuse_unmapped_type_keeps_data_in_metadata(self) -> None:
        """Langfuse span types without standard mapping keep data in metadata."""
        span = span_from_langfuse(
            {
                "id": "obs-1",
                "traceId": "t1",
                "type": "SPAN",
                "name": "pipeline",
                "input": {"q": 1},
                "output": {"r": 2},
            }
        )
        assert not span.attributes
        assert span.metadata["langfuse.inputs"] == {"q": 1}
        assert span.metadata["langfuse.outputs"] == {"r": 2}

    def test_otel_events_mapped_to_span_events(self) -> None:
        """OTEL span events convert to SpanEvent instances."""
        span = span_from_otel(
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "agent",
                "events": [
                    {
                        "name": "exception",
                        "timeUnixNano": "1700000000000000000",
                        "attributes": [
                            {
                                "key": "exception.type",
                                "value": {"stringValue": "ValueError"},
                            }
                        ],
                    },
                    {"attributes": [{"key": "x", "value": {"stringValue": "y"}}]},
                ],
            }
        )
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
        assert event.timestamp is not None
        assert event.attributes["exception.type"] == "ValueError"

    def test_otel_consumed_attrs_removed_from_metadata_for_mapped_types(self) -> None:
        """GenAI data keys are consumed into attributes, not duplicated in metadata."""
        span = span_from_otel(
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "chat",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.prompt": "hello",
                    "gen_ai.completion": "world",
                    "gen_ai.usage.input_tokens": 5,
                    "custom.key": "kept",
                },
            }
        )
        assert span.span_type is SpanType.GENERATION
        assert span.source is TracePlatform.OTEL
        assert span.attributes["generation.prompt"] == "hello"
        assert "gen_ai.prompt" not in span.metadata
        assert "gen_ai.usage.input_tokens" not in span.metadata
        assert span.metadata["otel.custom.key"] == "kept"

    def test_mlflow_retriever_span_maps_standard_attributes(self) -> None:
        """MLflow RETRIEVER span maps inputs/outputs to retrieval.* keys."""
        span = span_from_mlflow(
            {
                "context": {"trace_id": "t1", "span_id": "s1"},
                "name": "retrieve_docs",
                "attributes": {
                    "mlflow.spanType": "RETRIEVER",
                    "mlflow.spanInputs": "find user information",
                    "mlflow.spanOutputs": ["chunk 1", "chunk 2"],
                },
            }
        )
        assert span.span_type is SpanType.RETRIEVER
        assert span.attributes["retrieval.query"] == "find user information"
        assert span.attributes["retrieval.chunks"] == ["chunk 1", "chunk 2"]

    def test_span_event_model_rejects_extra_fields(self) -> None:
        """SpanEvent forbids extra fields."""
        with pytest.raises(ValidationError):
            SpanEvent.model_validate({"name": "e", "unknown": 1})

    def test_span_rejects_inputs_outputs(self) -> None:
        """Removed inputs/outputs fields are rejected (extra=forbid)."""
        with pytest.raises(ValidationError):
            Span.model_validate(
                {"span_id": "s", "trace_id": "t", "name": "n", "inputs": 1}
            )

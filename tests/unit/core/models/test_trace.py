"""Tests for internal trace models and platform adapters."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.trace import (
    ModelInfo,
    Span,
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
        assert root.inputs == {"query": "hi"}
        assert root.metadata["conversation_group_id"] == "conv_1"
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
        assert llm.inputs == "hello"
        assert llm.outputs == "world"
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
        assert span.metadata["tags"] == ["a", "b"]
        assert span.metadata["meta"] == {"env": "test", "count": 2}
        assert span.metadata["payload"] == "aGVsbG8="

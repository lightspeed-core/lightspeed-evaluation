"""Tests for otel trace conversion."""

from datetime import UTC, datetime
from typing import Any

import pytest

from lightspeed_evaluation.core.models.trace import (
    SpanType,
    TracePlatform,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.core.tracing.otel_adapter import (
    span_from_otel,
    trace_from_otel,
)


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
        assert span.tags == ["a", "b"]
        assert span.metadata["otel.meta"] == {"env": "test", "count": 2}
        assert span.metadata["otel.payload"] == "aGVsbG8="


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


def test_otel_events_mapped_to_span_events() -> None:
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


def test_otel_consumed_attrs_removed_from_metadata_for_mapped_types() -> None:
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


def test_otel_retriever_json_string_values_preserved() -> None:
    """Retriever metadata keeps original strings; semantics stay parsed."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "attributes": [
                {
                    "key": "openinference.span.kind",
                    "value": {"stringValue": "RETRIEVER"},
                },
                {"key": "input.value", "value": {"stringValue": '{"query": "q"}'}},
                {"key": "output.value", "value": {"stringValue": '[{"text": "a"}]'}},
            ],
        }
    )
    assert span.attributes["retrieval.query"] == "q"
    assert span.attributes["retrieval.chunks"] == ["a"]
    assert span.metadata["otel.input.value"] == '{"query": "q"}'
    assert span.metadata["otel.output.value"] == '[{"text": "a"}]'


def test_otel_non_retriever_metadata_stays_parsed() -> None:
    """Non-retriever spans keep the established parsed metadata form."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                {"key": "input.value", "value": {"stringValue": '{"q": 1}'}},
            ],
        }
    )
    assert span.metadata["otel.input.value"] == {"q": 1}


@pytest.mark.parametrize("metadata", ["not-a-dict", 42, ["list"], True])
def test_otel_trace_non_dict_metadata_preserved(metadata: Any) -> None:
    """Malformed trace metadata must not abort conversion."""
    trace = trace_from_otel(
        {"spans": [{"traceId": "t", "spanId": "s"}], "metadata": metadata}
    )
    assert trace.metadata["otel.metadata"] == metadata


def test_otel_session_and_trace_id_from_any_span() -> None:
    """Span order must not affect identity or session resolution."""
    trace = trace_from_otel(
        {
            "spans": [
                {"spanId": "a"},
                {
                    "traceId": "t",
                    "spanId": "b",
                    "attributes": {"session.id": "sess-x"},
                },
            ]
        }
    )
    assert trace.trace_id == "t"
    assert trace.session_id == "sess-x"
    assert {span.trace_id for span in trace.spans} == {"t"}


def test_otel_wrapper_supports_snake_case() -> None:
    """Wrapper and span helper support the same identity aliases."""
    trace = trace_from_otel({"spans": [{"trace_id": "t", "span_id": "s"}]})
    assert trace.trace_id == trace.spans[0].trace_id == "t"


def test_otel_explicit_nanosecond_units_and_epoch_zero() -> None:
    """Small OTLP timestamps are nanoseconds even near the epoch."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "startTimeUnixNano": 0,
            "endTimeUnixNano": 1_000_000_000,
            "events": [{"name": "checkpoint", "timeUnixNano": 0}],
        }
    )
    assert span.start_time == datetime(1970, 1, 1, tzinfo=UTC)
    assert span.latency == 1
    assert span.events[0].timestamp == span.start_time


def test_unused_and_invalid_aliases_retained() -> None:
    """Only successfully selected aliases are consumed."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.usage.input_tokens": "bad",
                "gen_ai.usage.prompt_tokens": 3,
                "gen_ai.request.model": "request",
                "gen_ai.response.model": "response",
            },
        }
    )
    assert span.attributes["tokens.input"] == 3
    assert span.metadata["otel.gen_ai.usage.input_tokens"] == "bad"
    assert span.metadata["otel.gen_ai.response.model"] == "response"
    assert "otel.gen_ai.usage.prompt_tokens" not in span.metadata


def test_fields_inapplicable_to_span_type_retained() -> None:
    """Tool-specific fields on an AGENT span must not disappear."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.call.id": "c",
            },
        }
    )
    assert span.metadata["otel.gen_ai.tool.name"] == "search"
    assert span.metadata["otel.gen_ai.tool.call.id"] == "c"


def test_unknown_type_retained() -> None:
    """Unknown classification remains available for future normalization."""
    span = span_from_otel(
        {
            "traceId": "t",
            "spanId": "s",
            "attributes": {"gen_ai.operation.name": "custom"},
        }
    )
    assert span.metadata["otel.gen_ai.operation.name"] == "custom"

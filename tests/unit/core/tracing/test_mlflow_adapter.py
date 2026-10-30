"""Tests for mlflow trace conversion."""

from datetime import UTC, datetime
from typing import Any

import pytest

from lightspeed_evaluation.core.models.trace import (
    SpanType,
    TracePlatform,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.core.tracing.mlflow_adapter import (
    span_from_mlflow,
    trace_from_mlflow,
)


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
        assert root.metadata["mlflow.spanInputs"] == {"query": "hi"}
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


def test_mlflow_tool_span_maps_standard_attributes() -> None:
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


def test_mlflow_retriever_span_maps_standard_attributes() -> None:
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


def test_mlflow_span_fallback_identity_consistent() -> None:
    """Fallback identity from the first span stays consistent for all spans."""
    trace = trace_from_mlflow(
        {
            "spans": [
                {"context": {"trace_id": "t2", "span_id": "a"}},
                {"context": {"trace_id": "t2", "span_id": "b"}},
            ]
        }
    )
    assert trace.trace_id == "t2"
    assert {span.trace_id for span in trace.spans} == {"t2"}


def test_mlflow_request_id_and_otel_context_id_coexist() -> None:
    """MLflow request ID and OTel context trace ID are distinct identifiers."""
    span = span_from_mlflow(
        {
            "context": {"span_id": "s", "trace_id": "0af7651916cd43dd8448eb211c80319c"},
            "attributes": {
                "mlflow.spanType": "TOOL",
                "mlflow.traceRequestId": '"f65bbc9e6f2f4d1f8a5c1e5f4d0e7d9b"',
            },
        }
    )
    assert span.trace_id == "f65bbc9e6f2f4d1f8a5c1e5f4d0e7d9b"
    assert (
        span.metadata["mlflow.span_context_trace_id"]
        == "0af7651916cd43dd8448eb211c80319c"
    )


def test_mlflow_trace_uses_request_id_identity() -> None:
    """Trace identity resolves from the request ID, not the OTel context ID."""
    trace = trace_from_mlflow(
        {
            "info": {"request_id": "f65bbc9e6f2f4d1f8a5c1e5f4d0e7d9b"},
            "spans": [
                {
                    "context": {
                        "span_id": "s",
                        "trace_id": "0af7651916cd43dd8448eb211c80319c",
                    },
                    "attributes": {
                        "mlflow.traceRequestId": '"f65bbc9e6f2f4d1f8a5c1e5f4d0e7d9b"',
                    },
                }
            ],
        }
    )
    assert trace.trace_id == "f65bbc9e6f2f4d1f8a5c1e5f4d0e7d9b"
    assert trace.spans[0].trace_id == trace.trace_id


def test_large_integer_tokens_do_not_round() -> None:
    """Decimal integer strings must not lose precision through float coercion."""
    count = 9007199254740993
    span = span_from_mlflow(
        {
            "trace_id": "t",
            "span_id": "s",
            "attributes": {"mlflow.spanType": "LLM", "input_tokens": str(count)},
        }
    )
    assert span.attributes["tokens.input"] == count


@pytest.mark.parametrize("value", ["bad", "inf", "-inf", "NaN", -1, 1.5, True])
def test_failed_numeric_mapping_retained(value: Any) -> None:
    """Unusable numeric fields remain available instead of being excluded."""
    span = span_from_mlflow(
        {
            "trace_id": "t",
            "span_id": "s",
            "attributes": {"mlflow.spanType": "LLM", "input_tokens": value},
        }
    )
    assert "tokens.input" not in span.attributes
    assert span.metadata["mlflow.input_tokens"] == value


def test_mlflow_retriever_json_string_values_preserved() -> None:
    """Retriever metadata keeps original JSON strings; semantics stay parsed."""
    span = span_from_mlflow(
        {
            "trace_id": "t",
            "span_id": "s",
            "attributes": {
                "mlflow.spanType": "RETRIEVER",
                "mlflow.spanInputs": '{"query": "q"}',
                "mlflow.spanOutputs": '[{"text": "a"}]',
            },
        }
    )
    assert span.attributes["retrieval.query"] == "q"
    assert span.attributes["retrieval.chunks"] == ["a"]
    assert span.metadata["mlflow.spanInputs"] == '{"query": "q"}'
    assert span.metadata["mlflow.spanOutputs"] == '[{"text": "a"}]'


def test_mlflow_non_retriever_metadata_stays_parsed() -> None:
    """Non-retriever spans keep the established parsed metadata form."""
    span = span_from_mlflow(
        {
            "trace_id": "t",
            "span_id": "s",
            "attributes": {"mlflow.spanType": "AGENT", "mlflow.spanInputs": '{"q": 1}'},
        }
    )
    assert span.metadata["mlflow.spanInputs"] == {"q": 1}


def test_mlflow_session_id_falls_back_to_top_level() -> None:
    """Tag dictionary without a session tag must not shadow top-level IDs."""
    trace = trace_from_mlflow(
        {
            "info": {"request_id": "r", "tags": {"other": "x"}},
            "session_id": "sess-top",
            "spans": [{"context": {"trace_id": "r", "span_id": "s"}}],
        }
    )
    assert trace.session_id == "sess-top"


def test_mlflow_request_id_resolved_from_any_span() -> None:
    """Identity is found even when the first span carries no identifiers."""
    trace = trace_from_mlflow(
        {
            "spans": [
                {"span_id": "a"},
                {"context": {"trace_id": "t2", "span_id": "b"}},
            ]
        }
    )
    assert trace.trace_id == "t2"
    assert {span.trace_id for span in trace.spans} == {"t2"}


@pytest.mark.parametrize("metadata", ["not-a-dict", 42, ["list"], True])
def test_mlflow_trace_non_dict_metadata_preserved(metadata: Any) -> None:
    """Malformed trace-info metadata must not abort conversion."""
    trace = trace_from_mlflow({"info": {"metadata": metadata, "request_id": "r"}})
    assert trace.metadata["mlflow.metadata"] == metadata


def test_unused_latency_alias_remains() -> None:
    """Choosing seconds does not consume an unused millisecond field."""
    span = span_from_mlflow(
        {
            "trace_id": "t",
            "span_id": "s",
            "attributes": {"latency": 0, "latency_ms": 1000},
        }
    )
    assert span.latency == 0
    assert span.metadata["mlflow.latency_ms"] == 1000

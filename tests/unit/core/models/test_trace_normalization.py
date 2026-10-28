"""Regression coverage for trace identity and loss-aware normalization."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from lightspeed_evaluation.core.models.trace import Span
from lightspeed_evaluation.core.models.trace_adapters import (
    _as_datetime,
    span_from_langfuse,
    span_from_mlflow,
    span_from_otel,
    trace_from_langfuse,
    trace_from_mlflow,
    trace_from_otel,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError

SpanAdapter = Callable[..., Span]


@pytest.mark.parametrize(
    "adapter", [span_from_mlflow, span_from_langfuse, span_from_otel]
)
def test_conflicting_span_identity_rejected(adapter: SpanAdapter) -> None:
    """An enclosing trace cannot overwrite explicit span identity."""
    with pytest.raises(DataValidationError, match="conflicts"):
        adapter({"trace_id": "other", "span_id": "s", "id": "s"}, trace_id="t")


@pytest.mark.parametrize(
    "adapter", [span_from_mlflow, span_from_langfuse, span_from_otel]
)
def test_missing_span_identity_filled(adapter: SpanAdapter) -> None:
    """Enclosing trace supplies only missing identity."""
    assert adapter({"span_id": "s", "id": "s"}, trace_id="t").trace_id == "t"


@pytest.mark.parametrize(
    "adapter, payload",
    [
        (
            trace_from_otel,
            {
                "spans": [
                    {"traceId": "t1", "spanId": "a"},
                    {"traceId": "t2", "spanId": "b"},
                ]
            },
        ),
        (
            trace_from_mlflow,
            {"trace_id": "t1", "spans": [{"trace_id": "t2", "span_id": "s"}]},
        ),
        (
            trace_from_mlflow,
            {
                "spans": [
                    {"context": {"trace_id": "t1", "span_id": "a"}},
                    {"context": {"trace_id": "t2", "span_id": "b"}},
                ]
            },
        ),
        (
            trace_from_langfuse,
            {"id": "t1", "observations": [{"traceId": "t2", "id": "s"}]},
        ),
    ],
)
def test_trace_wrappers_reject_mixed_identity(
    adapter: Callable[..., Any], payload: dict[str, Any]
) -> None:
    """All trace wrappers reject conflicting span IDs."""
    with pytest.raises(DataValidationError, match="conflicts"):
        adapter(payload)


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


def test_otel_wrapper_supports_snake_case() -> None:
    """Wrapper and span helper support the same identity aliases."""
    trace = trace_from_otel({"spans": [{"trace_id": "t", "span_id": "s"}]})
    assert trace.trace_id == trace.spans[0].trace_id == "t"


@pytest.mark.parametrize("metadata", ["not-a-dict", 42, ["list"], True])
def test_non_dict_retrieval_metadata_does_not_crash(metadata: Any) -> None:
    """Truth-y non-dictionary document metadata must not abort conversion."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "output": [{"text": "a", "metadata": metadata}],
        }
    )
    assert span.attributes["retrieval.chunks"] == ["a"]
    assert "retrieval.sources" not in span.attributes


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


@pytest.mark.parametrize(
    "output",
    ['[{"unknown": 1}]', {"unknown": "payload"}, [{"unknown": "document"}], 42],
)
def test_unrecognized_retrieval_output_preserved(output: Any) -> None:
    """Unrecognized retrieval shapes are not mistaken for empty results."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "output": output,
        }
    )
    assert "retrieval.chunks" not in span.attributes
    assert span.metadata["langfuse.outputs"] == output


def test_event_with_input_and_output_preserves_input() -> None:
    """EVENT body prefers the output; the input stays available in metadata."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "EVENT",
            "name": "phase",
            "input": {"phase": "x", "extra": 1},
            "output": {"result": "done"},
        }
    )
    assert span.attributes["event.body"] == {"result": "done"}
    assert span.metadata["langfuse.inputs"] == {"phase": "x", "extra": 1}


def test_event_with_input_only_keeps_single_body() -> None:
    """EVENT with only an input uses it as the body without duplication."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "EVENT",
            "name": "phase",
            "input": {"phase": "x"},
        }
    )
    assert span.attributes["event.body"] == {"phase": "x"}
    assert "langfuse.inputs" not in span.metadata


def test_retriever_extra_input_keys_preserved() -> None:
    """Retrieval input keys beyond the standard set stay in metadata."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "input": {"query": "q", "filters": {"ns": "prod"}},
            "output": [{"text": "a"}],
        }
    )
    assert span.attributes["retrieval.query"] == "q"
    assert span.metadata["langfuse.inputs"] == {"query": "q", "filters": {"ns": "prod"}}


def test_retriever_unrecognized_input_dict_preserved() -> None:
    """A retrieval input dict without standard keys is not lost."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "input": {"q": "search-term"},
            "output": [{"text": "a"}],
        }
    )
    assert span.metadata["langfuse.inputs"] == {"q": "search-term"}


def test_retriever_standard_input_not_duplicated() -> None:
    """Fully mapped retrieval input is not copied into metadata."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "input": {"query": "q", "top_k": 3},
            "output": [{"text": "a"}],
        }
    )
    assert span.attributes["retrieval.query"] == "q"
    assert span.attributes["retrieval.top_k"] == 3
    assert "langfuse.inputs" not in span.metadata


def test_recognized_json_string_retrieval_output_not_duplicated() -> None:
    """A parseable JSON-string output classifies identically in both paths."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "output": '[{"text": "chunk"}]',
        }
    )
    assert span.attributes["retrieval.chunks"] == ["chunk"]
    assert "langfuse.outputs" not in span.metadata


def test_partial_retrieval_preserves_raw_output() -> None:
    """Mixed recognized/unrecognized documents keep raw output in metadata."""
    documents = [{"text": "kept"}, {"unknown": "dropped-by-parsing"}]
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "output": documents,
        }
    )
    assert span.attributes["retrieval.chunks"] == ["kept"]
    assert span.metadata["langfuse.outputs"] == documents


def test_retrieval_columns_stay_aligned() -> None:
    """Incomplete optional columns remain in raw metadata, not misleading lists."""
    documents = [{"text": "a"}, {"text": "b", "source": "doc-b", "score": 0}]
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "RETRIEVER",
            "output": documents,
        }
    )
    assert span.attributes["retrieval.chunks"] == ["a", "b"]
    assert "retrieval.sources" not in span.attributes
    assert "retrieval.scores" not in span.attributes
    assert span.metadata["langfuse.outputs"] == documents


@pytest.mark.parametrize(
    "adapter", [span_from_mlflow, span_from_langfuse, span_from_otel]
)
def test_string_tag_is_one_tag(adapter: SpanAdapter) -> None:
    """String tags are not split into characters."""
    span = adapter({"trace_id": "t", "span_id": "s", "id": "s", "tags": "prod"})
    assert span.tags == ["prod"]


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


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 10**100, "invalid"])
def test_invalid_timestamps_do_not_crash(value: Any) -> None:
    """Unrepresentable optional timestamps normalize to missing."""
    assert _as_datetime(value) is None


@pytest.mark.parametrize(
    "adapter, payload",
    [
        (trace_from_otel, {"spans": [{"traceId": "t", "spanId": "s"}]}),
        (trace_from_mlflow, {"trace_id": "t", "spans": []}),
        (trace_from_langfuse, {"id": "t", "observations": []}),
    ],
)
def test_trace_tag_is_not_split(
    adapter: Callable[..., Any], payload: dict[str, Any]
) -> None:
    """Trace-level tags use the same representation as span-level tags."""
    assert adapter({**payload, "tags": "prod"}).tags == ["prod"]


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

"""Cross-platform tests for shared span mapping contracts."""

from collections.abc import Callable
from typing import Any

import pytest

from lightspeed_evaluation.core.models.trace import (
    Span,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError
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

SpanAdapter = Callable[..., Span]


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


@pytest.mark.parametrize(
    "adapter", [span_from_mlflow, span_from_langfuse, span_from_otel]
)
def test_string_tag_is_one_tag(adapter: SpanAdapter) -> None:
    """String tags are not split into characters."""
    span = adapter({"trace_id": "t", "span_id": "s", "id": "s", "tags": "prod"})
    assert span.tags == ["prod"]


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


@pytest.mark.parametrize("adapter", [span_from_mlflow, span_from_otel])
@pytest.mark.parametrize(
    ("top_tags", "expected", "retained"),
    [
        (None, ["attribute"], False),
        (["top"], ["top"], True),
        ([], [], True),
        (42, ["attribute"], False),
    ],
)
def test_only_selected_tags_consumed(
    adapter: SpanAdapter, top_tags: object, expected: list[str], retained: bool
) -> None:
    """Metadata snapshot follows consumption; unused attribute tags survive."""
    span = adapter(
        {
            "trace_id": "t",
            "span_id": "s",
            "tags": top_tags,
            "attributes": {"tags": ["attribute"]},
        }
    )
    assert span.tags == expected
    key = f"{span.source.value}.tags"
    assert (key in span.metadata) is retained
    if retained:
        assert span.metadata[key] == ["attribute"]


@pytest.mark.parametrize(
    "adapter", [span_from_mlflow, span_from_langfuse, span_from_otel]
)
def test_non_generation_llm_data_kept(adapter: SpanAdapter) -> None:
    """Existing typed LLM data remains supported independently of span type."""
    span = adapter(
        {
            "trace_id": "t",
            "span_id": "s",
            "model": "model",
            "usage": {"input": 0},
            "attributes": {
                "model": "model",
                "input_tokens": 0,
                "gen_ai.request.model": "model",
                "gen_ai.usage.input_tokens": 0,
            },
        }
    )
    assert span.llm is not None
    assert span.llm.model is not None
    assert span.llm.model.name == "model"
    assert span.llm.token_usage is not None
    assert span.llm.token_usage.input_tokens == 0

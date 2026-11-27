"""Tests for platform-neutral span input/output access."""

from lightspeed_evaluation.core.tracing import (
    span_from_langfuse,
    span_from_mlflow,
    span_input,
    span_output,
)


def test_unmapped_langfuse_span_reads_raw_payload() -> None:
    """CHAIN-like spans expose input/output kept in namespaced metadata."""
    span = span_from_langfuse(
        {
            "id": "o",
            "traceId": "t",
            "type": "SPAN",
            "input": {"q": 1},
            "output": {"r": 2},
        }
    )
    assert span_input(span) == {"q": 1}
    assert span_output(span) == {"r": 2}


def test_unmapped_mlflow_span_reads_raw_payload() -> None:
    """The same accessor works for MLflow's namespaced keys."""
    span = span_from_mlflow(
        {
            "name": "chain",
            "context": {"span_id": "s", "trace_id": "t"},
            "attributes": {"mlflow.spanType": "CHAIN"},
            "inputs": {"q": 1},
            "outputs": {"r": 2},
        }
    )
    assert span_input(span) == {"q": 1}
    assert span_output(span) == {"r": 2}


def test_tool_span_reads_standard_attributes() -> None:
    """Typed spans resolve through standard semantic attributes."""
    span = span_from_langfuse(
        {
            "id": "o",
            "traceId": "t",
            "type": "TOOL",
            "name": "search",
            "input": {"q": "x"},
            "output": "doc",
        }
    )
    assert span_input(span) == {"q": "x"}
    assert span_output(span) == "doc"


def test_missing_payload_is_none() -> None:
    """Spans without input/output return None."""
    span = span_from_langfuse({"id": "o", "traceId": "t", "type": "SPAN"})
    assert span_input(span) is None
    assert span_output(span) is None

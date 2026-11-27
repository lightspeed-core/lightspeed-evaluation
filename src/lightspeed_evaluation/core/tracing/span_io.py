"""Platform-neutral access to a normalized span's input and output payloads.

Adapters map input/output into standard attributes only for span types with a
semantic schema (TOOL, GENERATION, RETRIEVER, EVENT). For every other type
(CHAIN, AGENT, ...) the raw payload is kept in namespaced metadata whose key
depends on the source platform (``langfuse.output``, ``mlflow.spanOutputs``,
``otel.output.value``, ...). These helpers hide that difference so consumers
never need to know which platform produced a span.
"""

from typing import Any

from lightspeed_evaluation.core.models.trace import Span, SpanType, TracePlatform
from lightspeed_evaluation.core.tracing._helpers import _first_present, _parse_jsonish

# Namespaced metadata keys where each adapter keeps unmapped input/output.
_RAW_INPUT_KEYS: dict[TracePlatform, tuple[str, ...]] = {
    TracePlatform.LANGFUSE: ("langfuse.input", "langfuse.inputs"),
    TracePlatform.MLFLOW: ("mlflow.spanInputs", "mlflow.inputs"),
    TracePlatform.OTEL: ("otel.input.value", "otel.inputs"),
}
_RAW_OUTPUT_KEYS: dict[TracePlatform, tuple[str, ...]] = {
    TracePlatform.LANGFUSE: ("langfuse.output", "langfuse.outputs"),
    TracePlatform.MLFLOW: ("mlflow.spanOutputs", "mlflow.outputs"),
    TracePlatform.OTEL: ("otel.output.value", "otel.outputs"),
}

_STANDARD_INPUT_KEY = {
    SpanType.TOOL: "tool.arguments",
    SpanType.GENERATION: "generation.prompt",
}
_STANDARD_OUTPUT_KEY = {
    SpanType.TOOL: "tool.result",
    SpanType.GENERATION: "generation.completion",
    SpanType.EVENT: "event.body",
}


def _raw_value(span: Span, keys: dict[TracePlatform, tuple[str, ...]]) -> Any:
    """Return the first namespaced raw payload present for the span's platform."""
    candidates = keys.get(span.source, ())
    return _parse_jsonish(
        _first_present(*(span.metadata.get(key) for key in candidates))
    )


def span_input(span: Span) -> Any:
    """Return a span's input payload regardless of span type or source platform.

    Args:
        span: Normalized span.

    Returns:
        The standard-attribute input when the span type defines one, otherwise
        the raw input preserved by the adapter, or ``None`` when absent.
    """
    key = _STANDARD_INPUT_KEY.get(span.span_type)
    standard = span.attributes.get(key) if key else None
    return _first_present(standard, _raw_value(span, _RAW_INPUT_KEYS))


def span_output(span: Span) -> Any:
    """Return a span's output payload regardless of span type or source platform.

    Args:
        span: Normalized span.

    Returns:
        The standard-attribute output when the span type defines one, otherwise
        the raw output preserved by the adapter, or ``None`` when absent.
    """
    key = _STANDARD_OUTPUT_KEY.get(span.span_type)
    standard = span.attributes.get(key) if key else None
    return _first_present(standard, _raw_value(span, _RAW_OUTPUT_KEYS))

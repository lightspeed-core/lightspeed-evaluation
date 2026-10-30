"""Tests for internal trace models and validation."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.trace import (
    ModelInfo,
    Span,
    SpanEvent,
    SpanType,
    TokenUsage,
    Trace,
    TraceSession,
    normalize_span_type,
)


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


def test_span_event_model_rejects_extra_fields() -> None:
    """SpanEvent forbids extra fields."""
    with pytest.raises(ValidationError):
        SpanEvent.model_validate({"name": "e", "unknown": 1})


def test_span_rejects_inputs_outputs() -> None:
    """Removed inputs/outputs fields are rejected (extra=forbid)."""
    with pytest.raises(ValidationError):
        Span.model_validate({"span_id": "s", "trace_id": "t", "name": "n", "inputs": 1})

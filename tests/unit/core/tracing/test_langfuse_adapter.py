"""Tests for langfuse trace conversion."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest

from lightspeed_evaluation.core.models.trace import (
    SpanStatus,
    SpanType,
    TracePlatform,
)
from lightspeed_evaluation.core.tracing.langfuse_adapter import (
    span_from_langfuse,
    trace_from_langfuse,
)


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


def test_langfuse_generation_span_maps_standard_attributes() -> None:
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
    assert span.attributes["generation.prompt"] == [{"role": "user", "content": "hi"}]
    assert span.attributes["generation.completion"] == "The answer"
    assert span.attributes["model.name"] == "gpt-4"
    assert span.attributes["tokens.input"] == 50
    assert span.attributes["tokens.output"] == 100
    assert span.attributes["tokens.total"] == 150


def test_langfuse_unmapped_type_keeps_data_in_metadata() -> None:
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
    assert span.metadata["langfuse.input"] == {"q": 1}
    assert span.metadata["langfuse.output"] == {"r": 2}


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
    assert span.metadata["langfuse.output"] == output


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
    assert span.metadata["langfuse.input"] == {"phase": "x", "extra": 1}


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
    assert "langfuse.input" not in span.metadata


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
    assert span.metadata["langfuse.input"] == {"query": "q", "filters": {"ns": "prod"}}


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
    assert span.metadata["langfuse.input"] == {"q": "search-term"}


def test_retriever_raw_input_always_available() -> None:
    """Fully mapped retrieval input/output still remain in metadata."""
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
    # Retrieval raw values are always preserved: normalization can be partial.
    assert span.metadata["langfuse.input"] == {"query": "q", "top_k": 3}
    assert span.metadata["langfuse.output"] == [{"text": "a"}]


def test_recognized_json_string_retrieval_output_normalized_and_kept() -> None:
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
    assert span.metadata["langfuse.output"] == '[{"text": "chunk"}]'


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
    assert span.metadata["langfuse.output"] == documents


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
    assert span.metadata["langfuse.output"] == documents


def test_partial_nested_langfuse_mapping_preserves_extras() -> None:
    """Unknown and invalid nested usage/model settings remain in metadata."""
    span = span_from_langfuse(
        {
            "traceId": "t",
            "id": "s",
            "type": "GENERATION",
            "model": "primary",
            "usage": {"input": "bad", "promptTokens": 4, "custom": 7},
            "modelParameters": {"model": "alternate", "temperature": 0, "custom": 8},
        }
    )
    assert span.attributes["tokens.input"] == 4
    assert span.attributes["model.temperature"] == 0
    assert span.metadata["langfuse.usage"] == {"input": "bad", "custom": 7}
    assert span.metadata["langfuse.modelParameters"] == {
        "model": "alternate",
        "custom": 8,
    }


def test_retrieval_nested_document_metadata_preserved() -> None:
    """Extras nested inside recognized documents are not lost."""
    documents = [{"text": "a", "source": "s", "metadata": {"document_id": 5}}]
    span = span_from_langfuse(
        {"traceId": "t", "id": "s", "type": "RETRIEVER", "output": documents}
    )
    assert span.attributes["retrieval.sources"] == ["s"]
    assert span.metadata["langfuse.output"] == documents


def test_retrieval_uncoercible_score_preserved() -> None:
    """A present-but-invalid score stays available in the raw output."""
    documents = [{"text": "a", "source": "s", "score": "NaN"}]
    span = span_from_langfuse(
        {"traceId": "t", "id": "s", "type": "RETRIEVER", "output": documents}
    )
    assert "retrieval.scores" not in span.attributes
    assert span.metadata["langfuse.output"] == documents


@pytest.mark.parametrize("span_type", ["TOOL", "GENERATION", "EVENT"])
def test_langfuse_json_string_parsed_once(span_type: str) -> None:
    """JSON-encoded text stays text rather than being recursively decoded."""
    text = '{"query": "literal text"}'
    span = span_from_langfuse(
        {"traceId": "t", "id": "s", "type": span_type, "input": json.dumps(text)}
    )
    key = {
        "TOOL": "tool.arguments",
        "GENERATION": "generation.prompt",
        "EVENT": "event.body",
    }[span_type]
    assert span.attributes[key] == text


@pytest.mark.parametrize("metadata", ["not-a-dict", 42, ["list"], True])
def test_langfuse_non_dict_metadata_preserved(metadata: Any) -> None:
    """Malformed span metadata must not abort conversion."""
    span = span_from_langfuse(
        {"traceId": "t", "id": "s", "type": "SPAN", "metadata": metadata}
    )
    assert span.metadata["langfuse.metadata"] == metadata


def test_langfuse_trace_snake_case_times() -> None:
    """Trace start/end accept the snake_case aliases used by spans."""
    trace = trace_from_langfuse(
        {
            "id": "t",
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:01:00",
        }
    )
    assert trace.start_time == datetime(2024, 1, 1, tzinfo=UTC)
    assert trace.end_time == datetime(2024, 1, 1, 0, 1, tzinfo=UTC)


@pytest.mark.parametrize("metadata", ["not-a-dict", 42, ["list"], True])
def test_langfuse_trace_non_dict_metadata_preserved(metadata: Any) -> None:
    """Malformed trace metadata must not abort conversion."""
    trace = trace_from_langfuse({"id": "t", "metadata": metadata})
    assert trace.metadata["langfuse.metadata"] == metadata


@pytest.mark.parametrize("message_key", ["statusMessage", "status_message"])
def test_langfuse_status_message_alias_sets_ok(message_key: str) -> None:
    """A present status message yields OK for both alias spellings."""
    span = span_from_langfuse(
        {"traceId": "t", "id": "s", "type": "SPAN", message_key: "detail"}
    )
    assert span.status is SpanStatus.OK
    assert span.status_message == "detail"


def test_langfuse_metadata_collision_and_input_immutability() -> None:
    """Source input cannot overwrite colliding custom metadata without backup."""
    raw = {
        "traceId": "t",
        "id": "s",
        "type": "RETRIEVER",
        "input": {"query": "q"},
        "metadata": {"input": {"custom": True}},
        "usage": {"input": 0, "extra": 1},
        "modelParameters": {"temperature": 0, "extra": 2},
    }
    original = deepcopy(raw)
    span = span_from_langfuse(raw)
    assert span.metadata["langfuse.input"] == {"query": "q"}
    assert span.metadata["langfuse.metadata"] == {"input": {"custom": True}}
    assert span.metadata["langfuse.usage"] == {"extra": 1}
    assert span.metadata["langfuse.modelParameters"] == {"extra": 2}
    assert raw == original

"""Tests for observation-based turn/signal extraction."""

from typing import Any

from lightspeed_evaluation.core.ingestion.config import IngestConfig, SignalSpec
from lightspeed_evaluation.core.ingestion.extractor import (
    build_conversations,
    extract_signal,
    extract_turn,
    trace_document,
)
from lightspeed_evaluation.core.ingestion.langfuse_ingest import (
    traces_from_observations,
)
from lightspeed_evaluation.core.models.trace import Trace
from lightspeed_evaluation.core.tracing import trace_from_mlflow


def _config(**overrides: object) -> IngestConfig:
    """Build an ingest config with the default exit_reason signal + interpretation."""
    base: dict[str, object] = {
        "source": {"type": "file", "path": "unused.json"},
        "signals": {
            "exit_reason": {
                "query": (
                    "observations[?name=='supervisor']"
                    ".output.intent.exit_reason | [-1]"
                ),
                "role": "model",
                "values": {
                    "resolved": {
                        "polarity": "positive",
                        "strength": "weak",
                        "type": "inferred",
                    },
                    "escalated": {
                        "polarity": "negative",
                        "strength": "strong",
                        "type": "deterministic",
                    },
                },
            }
        },
    }
    base.update(overrides)
    return IngestConfig.model_validate(base)


def _langgraph_obs(trace_id: str, session: str, messages: list, t: str) -> dict:
    """A root LangGraph observation carrying a cumulative message list."""
    return {
        "traceId": trace_id,
        "sessionId": session,
        "name": "LangGraph",
        "type": "CHAIN",
        "startTime": t,
        "output": {"messages": messages},
    }


def _supervisor_obs(trace_id: str, session: str, exit_reason: str, t: str) -> dict:
    """A supervisor observation carrying intent.exit_reason."""
    return {
        "traceId": trace_id,
        "sessionId": session,
        "name": "supervisor",
        "type": "CHAIN",
        "startTime": t,
        "output": {"intent": {"exit_reason": exit_reason}},
    }


def test_extract_turn_reads_human_and_ai_messages() -> None:
    """Query comes from the human message, response from the ai message."""
    msgs: list[dict[str, Any]] = [
        {"type": "human", "content": "pods crashlooping"},
        {"type": "ai", "content": "restart the deployment"},
    ]
    turn = extract_turn(msgs, set(), 2000)
    assert turn["query"] == "pods crashlooping"
    assert turn["response"] == "restart the deployment"


def test_extract_turn_collects_tool_calls_and_contexts() -> None:
    """Tool calls and context-tool outputs are extracted."""
    msgs: list[dict[str, Any]] = [
        {"type": "human", "content": "q"},
        {
            "type": "ai",
            "content": "a",
            "tool_calls": [{"name": "search_knowledge_base", "args": {"q": "x"}}],
        },
        {"name": "search_knowledge_base", "content": "DOC TEXT"},
    ]
    turn = extract_turn(msgs, {"search_knowledge_base"}, 2000)
    assert turn["tool_calls"] == [
        [{"tool_name": "search_knowledge_base", "arguments": {"q": "x"}}]
    ]
    assert turn["contexts"] == ["DOC TEXT"]


def test_extract_signal_reads_supervisor_exit_reason() -> None:
    """A JMESPath query reaches into the trace's observations."""
    trace = {"observations": [_supervisor_obs("tr1", "s1", "resolved", "1")]}
    spec = SignalSpec(
        query="observations[?name=='supervisor'].output.intent.exit_reason | [-1]"
    )
    assert extract_signal(trace, spec) == "resolved"


def test_extract_signal_applies_map() -> None:
    """Raw values are remapped via `map`."""
    trace = {"observations": [_supervisor_obs("tr1", "s1", "turn_complete", "1")]}
    spec = SignalSpec(
        query="observations[?name=='supervisor'].output.intent.exit_reason | [-1]",
        map={"turn_complete": "none"},
    )
    assert extract_signal(trace, spec) == "none"


def _convs(observations: list[dict], config: IngestConfig) -> list[dict]:
    """Normalize Langfuse observations, then build conversations."""
    return build_conversations(traces_from_observations(observations), config)


def test_traces_from_observations_groups_and_sorts() -> None:
    """Observations group by traceId into Traces, spans ordered by startTime."""
    obs = [
        _supervisor_obs("tr1", "s1", "resolved", "2"),
        _langgraph_obs("tr1", "s1", [], "1"),
    ]
    traces = traces_from_observations(obs)
    assert len(traces) == 1
    assert isinstance(traces[0], Trace)
    assert traces[0].trace_id == "tr1"
    assert traces[0].session_id == "s1"
    assert [s.name for s in traces[0].spans] == ["LangGraph", "supervisor"]


def test_traces_from_observations_dedupes_and_drops_orphans() -> None:
    """Duplicate ids are dropped and observations without a traceId ignored."""
    obs = [
        {**_supervisor_obs("tr1", "s1", "resolved", "1"), "id": "o1"},
        {**_supervisor_obs("tr1", "s1", "resolved", "1"), "id": "o1"},
        {"name": "orphan", "type": "SPAN"},
    ]
    traces = traces_from_observations(obs)
    assert [s.span_id for s in traces[0].spans] == ["o1"]


def test_traces_from_observations_skips_invalid_trace(caplog: Any) -> None:
    """A trace that fails normalization is skipped, others are kept."""
    obs = [
        {
            **_supervisor_obs("bad", "s1", "x", "1"),
            "id": "a",
            "parentObservationId": "b",
        },
        {
            **_supervisor_obs("bad", "s1", "x", "2"),
            "id": "b",
            "parentObservationId": "a",
        },
        _supervisor_obs("good", "s2", "resolved", "3"),
    ]
    traces = traces_from_observations(obs)
    assert [t.trace_id for t in traces] == ["good"]
    assert "Skipping trace bad" in caplog.text


def test_trace_document_exposes_spans_and_legacy_observations() -> None:
    """Signal queries can address spans as `spans` or the legacy `observations`."""
    trace = traces_from_observations([_supervisor_obs("tr1", "s1", "resolved", "1")])[0]
    doc = trace_document(trace)
    assert doc["spans"] is doc["observations"]
    assert doc["spans"][0]["output"] == {"intent": {"exit_reason": "resolved"}}
    spec = SignalSpec(
        query="spans[?name=='supervisor'].output.intent.exit_reason | [0]"
    )
    assert extract_signal(doc, spec) == "resolved"


def test_build_conversations_is_platform_agnostic() -> None:
    """An MLflow trace goes through the same turn/signal assembly."""
    msgs = [
        {"type": "human", "content": "q"},
        {"type": "ai", "content": "r"},
    ]
    trace = trace_from_mlflow(
        {
            "info": {"trace_id": "m1"},
            "data": {
                "spans": [
                    {
                        "name": "LangGraph",
                        "context": {"span_id": "s1", "trace_id": "m1"},
                        "attributes": {"mlflow.spanType": "CHAIN"},
                        "outputs": {"messages": msgs},
                    },
                    {
                        "name": "supervisor",
                        "context": {"span_id": "s2", "trace_id": "m1"},
                        "parent_id": "s1",
                        "attributes": {"mlflow.spanType": "CHAIN"},
                        "outputs": {"intent": {"exit_reason": "resolved"}},
                    },
                ]
            },
        }
    )
    convs = build_conversations([trace], _config())
    turn = convs[0]["turns"][0]
    assert convs[0]["conversation_group_id"] == "m1"
    assert (turn["query"], turn["response"]) == ("q", "r")
    assert turn["signals"]["exit_reason"]["value"] == "resolved"


def test_build_conversations_single_turn() -> None:
    """One trace becomes one conversation/turn with query/response/signals."""
    msgs = [
        {"type": "human", "content": "pods crashlooping"},
        {"type": "ai", "content": "restart it"},
    ]
    obs = [
        _langgraph_obs("tr1", "s1", msgs, "1"),
        _supervisor_obs("tr1", "s1", "resolved", "2"),
    ]
    convs = _convs(obs, _config())
    assert len(convs) == 1
    turn = convs[0]["turns"][0]
    assert convs[0]["conversation_group_id"] == "s1"
    assert turn["query"] == "pods crashlooping"
    assert turn["response"] == "restart it"
    # Signal is enriched at ingest with polarity/strength/type.
    assert turn["signals"]["exit_reason"] == {
        "value": "resolved",
        "role": "model",
        "polarity": "positive",
        "strength": 0.5,
        "type": "inferred",
    }


def test_build_conversations_multiturn_diffs_messages() -> None:
    """Within a session, cumulative messages diff into separate turns."""
    t1_msgs = [
        {"type": "human", "content": "q1"},
        {"type": "ai", "content": "r1"},
    ]
    t2_msgs = t1_msgs + [
        {"type": "human", "content": "q2"},
        {"type": "ai", "content": "r2"},
    ]
    obs = [
        _langgraph_obs("tr1", "s1", t1_msgs, "1"),
        _supervisor_obs("tr1", "s1", "turn_complete", "1b"),
        _langgraph_obs("tr2", "s1", t2_msgs, "2"),
        _supervisor_obs("tr2", "s1", "resolved", "2b"),
    ]
    convs = _convs(obs, _config())
    assert len(convs) == 1
    turns = convs[0]["turns"]
    assert [t["query"] for t in turns] == ["q1", "q2"]
    assert [t["response"] for t in turns] == ["r1", "r2"]
    assert turns[1]["signals"]["exit_reason"]["value"] == "resolved"
    assert turns[1]["signals"]["exit_reason"]["polarity"] == "positive"


def test_extract_turn_does_not_duplicate_tool_calls() -> None:
    """Calls present in both LangChain shapes are emitted once."""
    msg: dict[str, Any] = {
        "type": "ai",
        "content": "",
        "tool_calls": [{"name": "search", "args": {"q": "x"}}],
        "additional_kwargs": {
            "tool_calls": [{"function": {"name": "search", "arguments": '{"q": "x"}'}}]
        },
    }
    turn = extract_turn([msg], set(), 2000)
    assert turn["tool_calls"] == [[{"tool_name": "search", "arguments": {"q": "x"}}]]


def test_extract_turn_reads_kwargs_tool_calls_when_top_level_empty() -> None:
    """Provider kwargs are the fallback when the top-level list is empty."""
    msg: dict[str, Any] = {
        "type": "ai",
        "content": "",
        "tool_calls": [],
        "additional_kwargs": {
            "tool_calls": [{"function": {"name": "search", "arguments": '{"q": "x"}'}}]
        },
    }
    turn = extract_turn([msg], set(), 2000)
    assert turn["tool_calls"] == [[{"tool_name": "search", "arguments": {"q": "x"}}]]


def test_build_conversations_skips_session_with_missing_root(caplog: Any) -> None:
    """A session with a trace lacking the root span is skipped, not half-emitted."""
    msgs = [{"type": "human", "content": "q"}, {"type": "ai", "content": "r"}]
    obs = [
        _langgraph_obs("tr1", "s1", msgs, "1"),
        _supervisor_obs("tr2", "s1", "resolved", "2"),  # no LangGraph root
        _langgraph_obs("tr3", "s2", msgs, "3"),
    ]
    convs = _convs(obs, _config())
    assert [c["conversation_group_id"] for c in convs] == ["s2"]
    assert "Skipped 1 incomplete session" in caplog.text

"""Tests for observation-based turn/signal extraction."""

from typing import Any

from lightspeed_evaluation.core.ingestion.config import IngestConfig, SignalSpec
from lightspeed_evaluation.core.ingestion.extractor import (
    build_conversations,
    extract_signal,
    extract_turn,
    group_into_traces,
)


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


def test_group_into_traces_groups_and_sorts() -> None:
    """Observations group by traceId, ordered by startTime within a trace."""
    obs = [
        _supervisor_obs("tr1", "s1", "resolved", "2"),
        _langgraph_obs("tr1", "s1", [], "1"),
    ]
    traces = group_into_traces(obs)
    assert len(traces) == 1
    assert traces[0]["id"] == "tr1"
    assert traces[0]["sessionId"] == "s1"
    assert [o["name"] for o in traces[0]["observations"]] == ["LangGraph", "supervisor"]


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
    convs = build_conversations(obs, _config())
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
    convs = build_conversations(obs, _config())
    assert len(convs) == 1
    turns = convs[0]["turns"]
    assert [t["query"] for t in turns] == ["q1", "q2"]
    assert [t["response"] for t in turns] == ["r1", "r2"]
    assert turns[1]["signals"]["exit_reason"]["value"] == "resolved"
    assert turns[1]["signals"]["exit_reason"]["polarity"] == "positive"

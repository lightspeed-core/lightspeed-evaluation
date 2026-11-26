"""Reconstruct evaluation turns and signals from Langfuse observations.

One trace == one turn. Turn content (query / response / tool_calls / contexts)
is derived from the root observation's cumulative ``messages`` list, diffed per
turn within a session (logic reused from the curate_demo build scripts). Signals
are extracted declaratively via JMESPath (:class:`SignalSpec`), so adding or
re-pointing a signal is a config edit.
"""

import json
import logging
from collections import defaultdict
from typing import Any

import jmespath
from jmespath.exceptions import JMESPathError

from lightspeed_evaluation.core.ingestion.config import IngestConfig, SignalSpec

logger = logging.getLogger(__name__)


# --- signal extraction (config-driven) ----------------------------------------


def _search(query: str, data: Any) -> Any:
    """Evaluate a JMESPath query, logging (not raising) on a bad expression."""
    try:
        return jmespath.search(query, data)
    except JMESPathError as e:
        logger.warning("Invalid JMESPath query %r: %s", query, e)
        return None


def extract_signal(trace: dict[str, Any], spec: SignalSpec) -> Any:
    """Extract and map one signal's raw value from a trace payload."""
    value = _search(spec.query, trace)
    if value is None:
        return None
    if spec.map:
        return spec.map.get(str(value), value)
    return value


def _enrich_signal(value: Any, spec: SignalSpec) -> dict[str, Any]:
    """Attach polarity/strength/type to a raw value using the signal's config.

    The enriched object travels on the turn (and onto results), so curation reads
    interpreted signals without re-reading this config. Unlisted values are
    neutral with zero weight.
    """
    enriched: dict[str, Any] = {"value": value}
    if spec.role is not None:
        enriched["role"] = spec.role
    value_spec = spec.values.get(str(value))
    if value_spec is not None:
        enriched["polarity"] = value_spec.polarity
        enriched["strength"] = value_spec.strength
        enriched["type"] = value_spec.type
    else:
        enriched["polarity"] = "neutral"
        enriched["strength"] = 0.0
        enriched["type"] = None
    return enriched


def extract_signals(
    trace: dict[str, Any], signals: dict[str, SignalSpec]
) -> dict[str, Any]:
    """Extract and interpret all declared signals present in a trace (absent omitted)."""
    out: dict[str, Any] = {}
    for name, spec in signals.items():
        value = extract_signal(trace, spec)
        if value is not None:
            out[name] = _enrich_signal(value, spec)
    return out


# --- turn content (reused message-diff logic) ---------------------------------


def _tool_calls_from_ai(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from an AI message (two LangChain shapes)."""
    step: list[dict[str, Any]] = []
    # Shape 1: additional_kwargs.tool_calls -> {function: {name, arguments}}
    for tc in (message.get("additional_kwargs", {}) or {}).get("tool_calls", []) or []:
        fn = tc.get("function", {}) or {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                pass
        step.append({"tool_name": name, "arguments": args or {}})
    # Shape 2: top-level tool_calls -> {name, args}
    for tc in message.get("tool_calls", []) or []:
        name = tc.get("name")
        if name:
            step.append({"tool_name": name, "arguments": tc.get("args") or {}})
    return step


def extract_turn(
    new_messages: list[dict[str, Any]],
    context_tools: set[str],
    context_cap: int,
) -> dict[str, Any]:
    """From a turn's new messages, build query/response/tool_calls/contexts."""
    query, response = "", ""
    tool_calls: list[list[dict[str, Any]]] = []
    contexts: list[str] = []

    for m in new_messages:
        if not isinstance(m, dict):
            continue
        mtype = m.get("type") or ""
        if mtype == "human":
            query = str(m.get("content") or "")
        elif mtype == "ai":
            content = m.get("content")
            if content and str(content).strip():
                response = str(content)
            step = _tool_calls_from_ai(m)
            if step:
                tool_calls.append(step)
        if (
            m.get("name") in context_tools
            and m.get("content")
            and str(m.get("content")).strip()
        ):
            text = str(m.get("content"))
            contexts.append(text[:context_cap] if context_cap else text)

    turn: dict[str, Any] = {"query": query or None, "response": response or None}
    if tool_calls:
        turn["tool_calls"] = tool_calls
    if contexts:
        turn["contexts"] = contexts
    return turn


# --- grouping + assembly ------------------------------------------------------


def group_into_traces(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group flat observations into trace payloads, sorted within each trace.

    Returns one dict per trace: ``{id, sessionId, startTime, observations}``
    with observations ordered by ``startTime``.
    """
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        trace_id = obs.get("traceId") or obs.get("trace_id")
        if trace_id:
            by_trace[trace_id].append(obs)

    traces: list[dict[str, Any]] = []
    for trace_id, obs_list in by_trace.items():
        obs_list.sort(key=lambda o: o.get("startTime") or o.get("start_time") or "")
        session_id = next(
            (
                o.get("sessionId") or o.get("session_id")
                for o in obs_list
                if o.get("sessionId") or o.get("session_id")
            ),
            None,
        )
        traces.append(
            {
                "id": trace_id,
                "sessionId": session_id,
                "startTime": (
                    obs_list[0].get("startTime") or obs_list[0].get("start_time")
                    if obs_list
                    else None
                ),
                "observations": obs_list,
            }
        )
    traces.sort(key=lambda t: t.get("startTime") or "")
    return traces


def _root_messages(trace: dict[str, Any], root_name: str) -> list[dict[str, Any]]:
    """Return the cumulative message list from a trace's root observation."""
    for obs in trace["observations"]:
        if obs.get("name") == root_name:
            output = obs.get("output")
            if isinstance(output, dict):
                messages = output.get("messages")
                if isinstance(messages, list):
                    return messages
    return []


def build_conversations(
    observations: list[dict[str, Any]], config: IngestConfig
) -> list[dict[str, Any]]:
    """Build evaluation-data conversations from flat observations.

    Traces are grouped by session; within a session each trace is one turn, whose
    content comes from diffing the root observation's cumulative messages and
    whose signals are extracted via config.
    """
    traces = group_into_traces(observations)
    context_tools = set(config.turn.context_tools)

    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for trace in traces:
        group = trace.get("sessionId") or trace["id"]
        if group not in sessions:
            order.append(group)
        sessions[group].append(trace)

    conversations: list[dict[str, Any]] = []
    for group in order:
        turn_entries = _build_session_turns(sessions[group], config, context_tools)
        if turn_entries:
            conversations.append(
                {
                    "conversation_group_id": group,
                    "metadata": {"data_source": "Production log"},
                    "turns": turn_entries,
                }
            )
    return conversations


def _build_session_turns(
    traces: list[dict[str, Any]],
    config: IngestConfig,
    context_tools: set[str],
) -> list[dict[str, Any]]:
    """Build the ordered turn list for one session (message-diff per trace)."""
    turn_entries: list[dict[str, Any]] = []
    prev_len = 0
    for idx, trace in enumerate(traces, start=1):
        messages = _root_messages(trace, config.turn.root_observation)
        new_messages = messages[prev_len:]
        prev_len = len(messages)

        turn = extract_turn(new_messages, context_tools, config.turn.context_cap)
        turn = {"turn_id": str(idx), **turn}
        signals = extract_signals(trace, config.signals)
        if signals:
            turn["signals"] = signals
        turn_entries.append(turn)
    return turn_entries

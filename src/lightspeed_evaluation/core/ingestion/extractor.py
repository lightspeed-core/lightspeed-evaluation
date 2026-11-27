"""Reconstruct evaluation turns and signals from normalized traces.

Input is the platform-agnostic :class:`Trace` model produced by the
``core.tracing`` adapters, so this module is independent of the trace backend.
One trace == one turn. Turn content (query / response / tool_calls / contexts)
is derived from the root span's cumulative ``messages`` list, diffed per turn
within a session. Signals are extracted declaratively via JMESPath
(:class:`SignalSpec`) over :func:`trace_document`, so adding or re-pointing a
signal is a config edit.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Optional

import jmespath
from jmespath.exceptions import JMESPathError

from lightspeed_evaluation.core.ingestion.config import IngestConfig, SignalSpec
from lightspeed_evaluation.core.models.trace import Trace, TraceSession
from lightspeed_evaluation.core.tracing import (
    session_from_traces,
    span_input,
    span_output,
)

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
    """Extract tool calls from an AI message (two LangChain shapes).

    LangChain can carry the same calls in both ``tool_calls`` and
    ``additional_kwargs.tool_calls``; the top-level list wins and the provider
    kwargs are read only when it is empty, so calls are never duplicated.
    """
    step: list[dict[str, Any]] = []
    # Shape 1: top-level tool_calls -> {name, args}
    for tc in message.get("tool_calls", []) or []:
        name = tc.get("name")
        if name:
            step.append({"tool_name": name, "arguments": tc.get("args") or {}})
    if step:
        return step
    # Shape 2: additional_kwargs.tool_calls -> {function: {name, arguments}}
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


# --- assembly from normalized traces -----------------------------------------


def trace_document(trace: Trace) -> dict[str, Any]:
    """Build the platform-neutral document that signal JMESPath queries run on.

    Spans are exposed under ``spans`` and, for backward compatibility with
    existing configs, under ``observations``. Each span carries ``name``,
    ``type``, ``input``, ``output``, ``attributes`` and ``metadata`` resolved the
    same way for every source platform.
    """
    spans = [
        {
            "id": span.span_id,
            "parent_id": span.parent_span_id,
            "name": span.name,
            "type": span.span_type.value,
            "status": span.status.value,
            "input": span_input(span),
            "output": span_output(span),
            "attributes": span.attributes,
            "metadata": span.metadata,
            "tags": span.tags,
        }
        for span in trace.spans
    ]
    return {
        "id": trace.trace_id,
        "session_id": trace.session_id,
        "name": trace.name,
        "metadata": trace.metadata,
        "tags": trace.tags,
        "spans": spans,
        "observations": spans,
    }


def _root_messages(trace: Trace, root_name: str) -> Optional[list[dict[str, Any]]]:
    """Return the cumulative message list from a trace's root span.

    Returns ``None`` when the root span (or its message list) is missing.
    """
    for span in trace.spans:
        if span.name == root_name:
            output = span_output(span)
            if isinstance(output, dict):
                messages = output.get("messages")
                if isinstance(messages, list):
                    return messages
    return None


def _group_sessions(traces: list[Trace]) -> list[TraceSession]:
    """Group traces by session (trace id when sessionless), in first-seen order."""
    grouped: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        grouped[trace.session_id or trace.trace_id].append(trace)
    return [session_from_traces(key, members) for key, members in grouped.items()]


def build_conversations(
    traces: list[Trace], config: IngestConfig
) -> list[dict[str, Any]]:
    """Build evaluation-data conversations from normalized traces.

    Traces are grouped by session; within a session each trace is one turn, whose
    content comes from diffing the root span's cumulative messages and whose
    signals are extracted via config. Traces must be ordered by start time.

    A session with any trace lacking the root span's messages is skipped with a
    warning: diffing across the gap would emit empty or merged turns.
    """
    context_tools = set(config.turn.context_tools)
    conversations: list[dict[str, Any]] = []
    sessions = _group_sessions(traces)
    for session in sessions:
        turn_entries = _build_session_turns(session.traces, config, context_tools)
        if turn_entries is None:
            continue
        conversations.append(
            {
                "conversation_group_id": session.session_id,
                "metadata": {"data_source": "Production log"},
                "turns": turn_entries,
            }
        )
    skipped = len(sessions) - len(conversations)
    if skipped:
        logger.warning(
            "Skipped %d incomplete session(s): root span %r missing in a trace",
            skipped,
            config.turn.root_observation,
        )
    return conversations


def _build_session_turns(
    traces: list[Trace],
    config: IngestConfig,
    context_tools: set[str],
) -> Optional[list[dict[str, Any]]]:
    """Build the ordered turn list for one session (message-diff per trace).

    Returns ``None`` when a trace lacks the root span's messages.
    """
    turn_entries: list[dict[str, Any]] = []
    prev_len = 0
    for idx, trace in enumerate(traces, start=1):
        messages = _root_messages(trace, config.turn.root_observation)
        if messages is None:
            logger.warning(
                "Session %s: trace %s has no %r messages",
                trace.session_id,
                trace.trace_id,
                config.turn.root_observation,
            )
            return None
        new_messages = messages[prev_len:]
        prev_len = len(messages)

        turn = extract_turn(new_messages, context_tools, config.turn.context_cap)
        turn = {"turn_id": str(idx), **turn}
        signals = extract_signals(trace_document(trace), config.signals)
        if signals:
            turn["signals"] = signals
        turn_entries.append(turn)
    return turn_entries

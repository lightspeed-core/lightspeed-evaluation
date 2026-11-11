"""SCA (Support Case Agent) SSE stream parsing and post-processing utilities."""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from lightspeed_evaluation.core.system.exceptions import APIError

logger = logging.getLogger(__name__)


@dataclass
class SCAStreamResult:
    """Aggregated data from a single SCA SSE stream."""

    text: str = ""
    conversation_id: str | None = None
    trace_id: str | None = None
    tool_calls: list[list[dict[str, Any]]] = field(default_factory=list)
    chat_widgets: dict[str, Any] = field(default_factory=dict)


def parse_sca_tool_call(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a normalised tool-call dict from an SCA event payload."""
    tool_name = data.get("name") or data.get("tool_name") or ""
    arguments = data.get("args") or data.get("arguments") or {}
    if not tool_name:
        return None
    tool_call: dict[str, Any] = {"tool_name": tool_name, "arguments": arguments}
    result = data.get("result")
    if result is not None:
        tool_call["result"] = result
    return tool_call


def _is_tool_node(node: dict[str, Any]) -> bool:
    """Check whether a trace node represents a tool call."""
    node_type = node.get("type", "").lower()
    return node_type == "tool" or "tool" in node.get("name", "").lower()


def extract_tool_calls_from_trace(
    trace_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract tool calls from a trace object.

    Supports two schemas:
    1. New SCA schema: trace has ``event == "tool_call"`` with ``tool_name``
       directly on the trace dict.
    2. Legacy node-tree schema: trace contains a ``nodes`` list with nested
       tool entries identified by type or name.
    """
    tool_calls: list[dict[str, Any]] = []

    if trace_data.get("event") == "tool_call":
        tool_name = trace_data.get("tool_name", "")
        if tool_name:
            tool_calls.append({"tool_name": tool_name, "arguments": {}})
        return tool_calls

    nodes = trace_data.get("nodes", [])
    if not isinstance(nodes, list):
        return tool_calls

    candidates = list(_iter_tool_candidates(nodes))
    for candidate in candidates:
        tc = parse_sca_tool_call(candidate)
        if tc:
            tool_calls.append(tc)
    return tool_calls


def _iter_tool_candidates(
    nodes: list[Any],
) -> Iterator[dict[str, Any]]:
    """Yield dicts from *nodes* (and their children) that look like tool calls."""
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if _is_tool_node(node):
            yield node
        children = node.get("children") or node.get("nodes") or []
        if not isinstance(children, list):
            continue
        for child in children:
            if isinstance(child, dict) and _is_tool_node(child):
                yield child


def iter_sca_events(response: httpx.Response) -> Iterator[dict[str, Any]]:
    """Decode complete SSE frames, ignoring heartbeat comments."""
    data_lines: list[str] = []
    event_name = ""

    for line in response.iter_lines():
        if line.startswith(":"):
            continue

        if not line:
            if data_lines:
                yield _decode_sca_event(data_lines, event_name)
            data_lines = []
            event_name = ""
            continue

        sse_field, _, value = line.partition(":")
        value = value.removeprefix(" ")

        if sse_field == "data":
            data_lines.append(value)
        elif sse_field == "event":
            event_name = value


def _decode_sca_event(data_lines: list[str], event_name: str) -> dict[str, Any]:
    """Parse an event payload, using the SSE event name as a fallback."""
    try:
        event = json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise APIError("Invalid JSON in SCA SSE event") from exc

    if not isinstance(event, dict):
        raise APIError("Expected an object in SCA SSE event")

    event["event"] = event.get("event") or event_name
    return event


def _parse_json_field(value: Any) -> Any:
    """Parse a value that may be a JSON string, returning the original on failure."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def extract_citation_contexts(chat_widgets: dict[str, Any]) -> list[str]:
    """Pull citation URLs from chat_widgets into context strings."""
    citations = chat_widgets.get("citations", [])
    if not isinstance(citations, list):
        return []
    contexts: list[str] = []
    for cite in citations:
        if isinstance(cite, dict):
            url = cite.get("view_uri", "")
            if url:
                contexts.append(url)
    return contexts


def enrich_from_trace(
    stream_result: SCAStreamResult,
    observations: list[dict[str, Any]],
) -> None:
    """Enrich tool calls with trace observations.

    Merges ``observations`` (from the SCA trace API) into
    ``stream_result.tool_calls`` with the actual arguments and results
    returned by the Hydra search_knowledge_base tool.

    Best-effort: callers should log and ignore failures so they never
    block the main evaluation flow.
    """
    if observations:
        logger.info(
            "Enriching tool calls with %d trace observations",
            len(observations),
        )
        merge_trace_tool_calls(stream_result, observations)
    else:
        logger.info(
            "No trace observations found",
        )


def merge_trace_tool_calls(
    stream_result: SCAStreamResult,
    observations: list[dict[str, Any]],
) -> None:
    """Merge trace observations into existing tool call sequences."""
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        tool_entry: dict[str, Any] = {
            "tool_name": obs.get("name", "search_knowledge_base"),
            "arguments": _parse_json_field(obs.get("input", "{}")),
        }
        output = obs.get("output")
        if output is not None:
            tool_entry["result"] = _parse_json_field(output)

        merged = False
        for tc_seq in stream_result.tool_calls:
            if (
                tc_seq
                and tc_seq[0].get("tool_name") == "search_knowledge_base"
                and not tc_seq[0].get("arguments")
            ):
                tc_seq[0].update(tool_entry)
                merged = True
                break
        if not merged:
            stream_result.tool_calls.append([tool_entry])


def read_sca_stream(response: httpx.Response) -> SCAStreamResult:
    """Assemble assistant tokens, tool calls, and metadata from the stream."""
    chunks: list[str] = []
    result = SCAStreamResult()

    for event in iter_sca_events(response):
        _update_metadata(result, event)
        _dispatch_event(result, event, chunks)
        _collect_trace_tool_calls(result, event)

    if not result.text and chunks:
        result.text = "".join(chunks)
    if result.text:
        return result
    raise APIError("SCA stream completed without assistant text")


def _update_metadata(result: SCAStreamResult, event: dict[str, Any]) -> None:
    """Update conversation and trace IDs from an event."""
    result.conversation_id = event.get("conversation_id") or result.conversation_id
    tid = event.get("trace_id")
    if tid:
        result.trace_id = tid


def _dispatch_event(
    result: SCAStreamResult,
    event: dict[str, Any],
    chunks: list[str],
) -> None:
    """Route an SSE event by type, updating *result* and *chunks*."""
    event_type = event["event"]

    if event_type == "error":
        raise APIError(f"SCA stream error: {event.get('data')}")

    if event_type == "token":
        token = event.get("data")
        if not isinstance(token, str):
            raise APIError("SCA token data must be a string")
        chunks.append(token)

    elif event_type == "tool_call":
        tool_data = event.get("data", event)
        if isinstance(tool_data, dict):
            tc = parse_sca_tool_call(tool_data.get("token", tool_data))
            if tc:
                result.tool_calls.append([tc])

    elif event_type == "end":
        result.text = "".join(chunks)
        widgets = event.get("chat_widgets")
        if isinstance(widgets, dict):
            result.chat_widgets = widgets


def _collect_trace_tool_calls(result: SCAStreamResult, event: dict[str, Any]) -> None:
    """Extract tool calls from the trace field."""
    trace_data = event.get("trace")
    if not isinstance(trace_data, dict):
        return
    extracted = extract_tool_calls_from_trace(trace_data)
    for tc in extracted:
        result.tool_calls.append([tc])

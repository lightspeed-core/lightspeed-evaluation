"""Streaming parser for API client."""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DATA_PREFIX = "data: "
CONTENT_EVENTS = frozenset(("token", "tool_call", "tool_result", "turn_complete"))


@dataclass
class _PerformanceTracker:
    """Tracks streaming performance metrics (TTFT, throughput)."""

    stream_start_time: float = field(default_factory=time.perf_counter)
    time_to_first_token: Optional[float] = None
    _first_content_received: bool = field(default=False, init=False, repr=False)

    def capture_ttft(self) -> None:
        """Capture time to first token if not already captured."""
        if not self._first_content_received:
            self.time_to_first_token = time.perf_counter() - self.stream_start_time
            self._first_content_received = True
            logger.debug("Time to first token: %.3f seconds", self.time_to_first_token)

    def get_metrics(self, output_tokens: int) -> tuple[float, Optional[float]]:
        """Calculate streaming duration and tokens per second."""
        streaming_duration = time.perf_counter() - self.stream_start_time
        tokens_per_second = self._calculate_tokens_per_second(
            output_tokens, streaming_duration
        )
        return streaming_duration, tokens_per_second

    def _calculate_tokens_per_second(
        self, output_tokens: int, total_duration: float
    ) -> Optional[float]:
        """Calculate tokens per second, excluding TTFT from throughput calculation."""
        if output_tokens <= 0 or self.time_to_first_token is None:
            return None
        generation_time = total_duration - self.time_to_first_token
        if generation_time <= 0:
            return None
        tokens_per_second = output_tokens / generation_time
        logger.debug(
            "Streaming performance: %.3f tokens/sec (%d tokens in %.3f sec)",
            tokens_per_second,
            output_tokens,
            generation_time,
        )
        return tokens_per_second


# Streaming event handlers
@dataclass
class StreamingContext:
    """Context for streaming response parsing."""

    conversation_id: str = ""
    final_response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    perf: _PerformanceTracker = field(default_factory=_PerformanceTracker)

    def to_response_dict(self) -> dict[str, Any]:
        """Convert context to response dictionary."""
        streaming_duration, tokens_per_second = self.perf.get_metrics(
            self.output_tokens
        )
        return {
            "response": self.final_response,
            "tool_calls": _format_tool_sequences(self.tool_calls),
            "conversation_id": self.conversation_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "time_to_first_token": self.perf.time_to_first_token,
            "streaming_duration": streaming_duration,
            "tokens_per_second": tokens_per_second,
        }


def _handle_start(ctx: StreamingContext, event_data: dict[str, Any]) -> None:
    if "conversation_id" in event_data:
        ctx.conversation_id = event_data["conversation_id"].strip()
        logger.debug("Found conversation_id: %s", ctx.conversation_id)


def _handle_turn_complete(ctx: StreamingContext, event_data: dict[str, Any]) -> None:
    if "token" in event_data:
        ctx.final_response = event_data["token"].strip()
        logger.debug("Found final response (%d characters)", len(ctx.final_response))


def _handle_tool_call(ctx: StreamingContext, event_data: dict[str, Any]) -> None:
    # Support both formats:
    # - Legacy: {"event": "tool_call", "data": {"token": {"tool_name": ..., "arguments": ...}}}
    # - New: {"event": "tool_call", "data": {"name": ..., "args": ..., "id": ...}}
    tool_data = event_data.get("token", event_data)
    tool_call = _parse_tool_call(tool_data)
    if tool_call:
        tool_id = event_data.get("id")
        ctx.tool_calls.append(tool_call)
        if tool_id:
            ctx.tool_calls_by_id[tool_id] = tool_call
        logger.debug("Found tool call: %s", tool_call)


def _handle_tool_result(ctx: StreamingContext, event_data: dict[str, Any]) -> None:
    tool_id = event_data.get("id")
    content = event_data.get("content")
    if not tool_id or content is None:
        logger.debug("Tool result missing id or content: %s", event_data)
        return
    if tool_id in ctx.tool_calls_by_id:
        ctx.tool_calls_by_id[tool_id]["result"] = content
        logger.debug("Associated result with tool call %s", tool_id)
    else:
        logger.debug("No matching tool call found for result id: %s", tool_id)


def _handle_end(ctx: StreamingContext, event_data: dict[str, Any]) -> None:
    ctx.input_tokens = event_data.get("input_tokens", 0)
    ctx.output_tokens = event_data.get("output_tokens", 0)


_STREAMING_EVENT_HANDLERS: dict[
    str, Callable[[StreamingContext, dict[str, Any]], None]
] = {
    "start": _handle_start,
    "turn_complete": _handle_turn_complete,
    "tool_call": _handle_tool_call,
    "tool_result": _handle_tool_result,
    "end": _handle_end,
}


# Responses streaming event handlers
@dataclass
class ResponsesStreamingContext:
    """Context for /responses streaming endpoint parsing."""

    conversation_id: str = ""
    final_response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rag_chunks: list[dict[str, str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    perf: _PerformanceTracker = field(default_factory=_PerformanceTracker)

    def to_response_dict(self) -> dict[str, Any]:
        """Convert context to response dictionary."""
        streaming_duration, tokens_per_second = self.perf.get_metrics(
            self.output_tokens
        )
        return {
            "response": self.final_response,
            "conversation_id": self.conversation_id,
            "tool_calls": _format_tool_sequences(self.tool_calls),
            "rag_chunks": self.rag_chunks,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "time_to_first_token": self.perf.time_to_first_token,
            "streaming_duration": streaming_duration,
            "tokens_per_second": tokens_per_second,
        }


def _responses_handle_created(
    ctx: ResponsesStreamingContext, data: dict[str, Any]
) -> None:
    ctx.conversation_id = data.get("response", {}).get("conversation", "")
    if not ctx.conversation_id:
        logger.debug("response.created event missing conversation field")


def _responses_handle_text_delta(
    ctx: ResponsesStreamingContext, _data: dict[str, Any]
) -> None:
    ctx.perf.capture_ttft()


def _responses_handle_output_item_done(
    ctx: ResponsesStreamingContext, data: dict[str, Any]
) -> None:
    item = data.get("item", {})
    item_handlers = {
        "mcp_call": _responses_handle_mcp_call_item,
        "file_search_call": _responses_handle_file_search_item,
    }
    handler = item_handlers.get(item.get("type", ""))
    if handler:
        handler(ctx, item)


def _responses_handle_mcp_call_item(
    ctx: ResponsesStreamingContext, item: dict[str, Any]
) -> None:
    raw_args = item.get("arguments") or {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    tool_call: dict[str, Any] = {
        "tool_name": item.get("name", ""),
        "arguments": raw_args,
    }
    output = item.get("output")
    if output is not None:
        tool_call["result"] = output
    error = item.get("error")
    if error is not None:
        tool_call["error"] = error
    ctx.tool_calls.append(tool_call)


def _responses_handle_file_search_item(
    ctx: ResponsesStreamingContext, item: dict[str, Any]
) -> None:
    for result in item.get("results", []):
        text = result.get("text")
        if text:
            ctx.rag_chunks.append({"content": text})
    ctx.tool_calls.append(
        {
            "tool_name": "file_search",
            "arguments": {"queries": item.get("queries", [])},
        }
    )


def _responses_handle_completed(
    ctx: ResponsesStreamingContext, data: dict[str, Any]
) -> None:
    resp = data.get("response", {})
    usage = resp.get("usage", {})
    ctx.input_tokens = usage.get("input_tokens", 0)
    ctx.output_tokens = usage.get("output_tokens", 0)
    ctx.final_response = resp.get("output_text", "")


_RESPONSES_EVENT_HANDLERS: dict[
    str, Callable[[ResponsesStreamingContext, dict[str, Any]], None]
] = {
    "response.created": _responses_handle_created,
    "response.output_text.delta": _responses_handle_text_delta,
    "response.output_item.done": _responses_handle_output_item_done,
    "response.completed": _responses_handle_completed,
}


# Public parse functions


def parse_streaming_response(response: httpx.Response) -> dict[str, Any]:
    """Parse a /streaming SSE response into a response dict with performance metrics."""
    ctx = StreamingContext()

    for line in response.iter_lines():
        line = line.strip()
        if not line or not line.startswith(DATA_PREFIX):
            continue

        parsed_data = _parse_sse_line(line.replace(DATA_PREFIX, ""))
        if not parsed_data:
            continue

        event, event_data = parsed_data

        if event == "error" and "token" in event_data:
            error_message = event_data["token"]
            logger.error("Received error event from streaming API: %s", error_message)
            raise ValueError(f"Streaming API error: {error_message}")

        if event in CONTENT_EVENTS:
            ctx.perf.capture_ttft()

        handler = _STREAMING_EVENT_HANDLERS.get(event)
        if handler:
            handler(ctx, event_data)

    _validate_streaming_response(ctx)
    return ctx.to_response_dict()


def parse_responses_streaming(response: httpx.Response) -> dict[str, Any]:
    """Parse a /responses SSE response into a response dict with performance metrics."""
    ctx = ResponsesStreamingContext()

    for line in response.iter_lines():
        line = line.strip()
        if not line or not line.startswith(DATA_PREFIX):
            continue
        raw = line[len(DATA_PREFIX) :]
        if raw == "[DONE]":
            break
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON from responses streaming: %s", raw)
            continue

        handler = _RESPONSES_EVENT_HANDLERS.get(data.get("type", ""))
        if handler:
            handler(ctx, data)

    _validate_responses_response(ctx)
    return ctx.to_response_dict()


# Shared helpers


def _parse_sse_line(json_data: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Parse a SSE line and return event and data."""
    try:
        data = json.loads(json_data)
        event = data.get("event", "")
        event_data = data.get("data", {})
        return event, event_data
    except json.JSONDecodeError:
        logger.debug("Failed to parse JSON from streaming response: %s", json_data)
        return None


def _parse_tool_call(event_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Parse tool call from streaming event data.

    Extracts tool_name, arguments, and optionally result from the event data.
    Supports both formats:
    - New format: {name: "...", args: {...}}
    - Legacy format: {tool_name: "...", arguments: {...}}

    Args:
        event_data: Dictionary containing tool call data from the API.

    Returns:
        Dictionary with tool_name, arguments, and optional result, or None if invalid.
    """
    try:
        tool_name = event_data.get("name") or event_data.get("tool_name") or ""
        arguments = event_data.get("args") or event_data.get("arguments") or {}

        if not tool_name:
            logger.debug("Tool call missing name/tool_name field")
            return None

        if arguments is None:
            logger.debug("Tool call missing args/arguments field for %s", tool_name)
            return None

        tool_call: dict[str, Any] = {"tool_name": tool_name, "arguments": arguments}

        result = event_data.get("result")
        if result is not None:
            tool_call["result"] = result
            logger.debug("Tool call '%s' has result: %s", tool_name, result)

        return tool_call

    except (ValueError, IndexError, AttributeError) as e:
        logger.debug("Failed to parse tool call '%s': %s", event_data, e)
        return None


def _format_tool_sequences(
    tool_calls: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Format tool calls into sequences."""
    if not tool_calls:
        return []

    # TODO(future): Handle Parallel tool calls
    sequences = [[tool_call] for tool_call in tool_calls]
    logger.debug(
        "Formatted %d tool calls into %d sequences", len(tool_calls), len(sequences)
    )
    return sequences


def _validate_streaming_response(ctx: StreamingContext) -> None:
    """Validate that required streaming response fields are present."""
    if not ctx.final_response:
        raise ValueError("No final response found in streaming output")
    if not ctx.conversation_id:
        raise ValueError("No Conversation ID found")


def _validate_responses_response(ctx: ResponsesStreamingContext) -> None:
    """Validate that required /responses streaming fields are present."""
    if not ctx.final_response:
        raise ValueError("No final response found in responses streaming output")
    if not ctx.conversation_id:
        raise ValueError("No conversation_id found in responses streaming output")

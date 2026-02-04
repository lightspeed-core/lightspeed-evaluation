"""Streaming response parser for API client."""

import json
import logging
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)

DATA_PREFIX = "data: "
CONTENT_EVENTS = ("token", "tool_call", "tool_result", "turn_complete")


class _PerformanceTracker(BaseModel):
    """Tracks streaming performance metrics (TTFT, throughput)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stream_start_time: float = Field(default_factory=time.perf_counter)
    time_to_first_token: Optional[float] = Field(default=None)
    _first_content_received: bool = PrivateAttr(default=False)

    def capture_ttft(self) -> None:
        """Capture time to first token if not already captured."""
        if not self._first_content_received:
            self.time_to_first_token = time.perf_counter() - self.stream_start_time
            self._first_content_received = True
            logger.debug("Time to first token: %.3f seconds", self.time_to_first_token)

    def get_metrics(self, output_tokens: int) -> tuple[float, Optional[float]]:
        """Calculate streaming duration and tokens per second."""
        streaming_duration = time.perf_counter() - self.stream_start_time
        tokens_per_second = _calculate_tokens_per_second(
            output_tokens, self.time_to_first_token, streaming_duration
        )
        return streaming_duration, tokens_per_second


class StreamingContext(BaseModel):
    """Context for streaming response parsing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversation_id: str = Field(default="")
    final_response: str = Field(default="")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls_by_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    perf: _PerformanceTracker = Field(default_factory=_PerformanceTracker)

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


def _calculate_tokens_per_second(
    output_tokens: int, ttft: Optional[float], total_duration: float
) -> Optional[float]:
    """Calculate tokens per second, excluding TTFT from throughput calculation."""
    if output_tokens <= 0 or ttft is None:
        return None
    generation_time = total_duration - ttft
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


def parse_streaming_response(  # pylint: disable=too-many-branches
    response: httpx.Response,
) -> dict[str, Any]:
    """Parse streaming response and extract data.

    Captures performance metrics including:
    - Time to First Token (TTFT): Time from request start to first content token
    - Streaming duration: Total time to receive all tokens
    - Tokens per second: Output throughput calculation

    Args:
        response: The httpx streaming response object to parse.

    Returns:
        Dictionary containing parsed response data with keys:
            - response: Final response text
            - conversation_id: Conversation tracking ID
            - tool_calls: List of tool call sequences
            - input_tokens: Number of input tokens used
            - output_tokens: Number of output tokens generated
            - time_to_first_token: TTFT in seconds (None if not captured)
            - streaming_duration: Total streaming time in seconds
            - tokens_per_second: Output throughput (None if not calculable)

    Raises:
        APIError: If an error event is received from the streaming API.
        DataValidationError: If required response fields are missing.
    """
    ctx = StreamingContext()

    for line in response.iter_lines():
        line = line.strip()
        if not line or not line.startswith(DATA_PREFIX):
            continue

        parsed_data = _parse_sse_line(line.replace(DATA_PREFIX, ""))
        if not parsed_data:
            continue

        event, event_data = parsed_data

        if event in CONTENT_EVENTS:
            ctx.perf.capture_ttft()

        _process_event(ctx, event, event_data)

    _validate_response(ctx)
    return ctx.to_response_dict()


def _process_event(ctx: StreamingContext, event: str, event_data: dict) -> None:
    """Process a single streaming event and update context."""
    if event == "error" and "token" in event_data:
        error_message = event_data["token"]
        logger.error("Received error event from streaming API: %s", error_message)
        raise ValueError(f"Streaming API error: {error_message}")
    if event == "start" and "conversation_id" in event_data:
        ctx.conversation_id = event_data["conversation_id"].strip()
        logger.debug("Found conversation_id: %s", ctx.conversation_id)
    elif event == "turn_complete" and "token" in event_data:
        ctx.final_response = event_data["token"].strip()
        logger.debug("Found final response (%d characters)", len(ctx.final_response))
    elif event == "tool_call":
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
    elif event == "tool_result":
        _process_tool_result(ctx, event_data)
    elif event == "end":
        ctx.input_tokens = event_data.get("input_tokens", 0)
        ctx.output_tokens = event_data.get("output_tokens", 0)


def _validate_response(ctx: StreamingContext) -> None:
    """Validate that required response fields are present."""
    if not ctx.final_response:
        raise ValueError("No final response found in streaming output")
    if not ctx.conversation_id:
        raise ValueError("No Conversation ID found")


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
    """Parse tool call from event data.

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
        # Support both "name"/"args" (new format) and "tool_name"/"arguments" (legacy)
        tool_name = event_data.get("name") or event_data.get("tool_name") or ""
        arguments = event_data.get("args") or event_data.get("arguments") or {}

        if not tool_name:
            logger.debug("Tool call missing name/tool_name field")
            return None

        # Arguments can be empty dict, but field should exist
        if arguments is None:
            logger.debug("Tool call missing args/arguments field for %s", tool_name)
            return None

        tool_call: dict[str, Any] = {"tool_name": tool_name, "arguments": arguments}

        # Capture tool result if present (optional field)
        result = event_data.get("result")
        if result is not None:
            tool_call["result"] = result
            logger.debug("Tool call '%s' has result: %s", tool_name, result)

        return tool_call

    except (ValueError, IndexError, AttributeError) as e:
        logger.debug("Failed to parse tool call '%s': %s", event_data, e)
        return None


def _process_tool_result(ctx: StreamingContext, event_data: dict[str, Any]) -> None:
    """Process tool result event and associate with corresponding tool call.

    Args:
        ctx: The streaming context to update.
        event_data: Dictionary containing tool result data with id and content.
    """
    tool_id = event_data.get("id")
    content = event_data.get("content")

    if not tool_id or content is None:
        logger.debug("Tool result missing id or content: %s", event_data)
        return

    # Find and update the corresponding tool call
    if tool_id in ctx.tool_calls_by_id:
        ctx.tool_calls_by_id[tool_id]["result"] = content
        logger.debug("Associated result with tool call %s", tool_id)
    else:
        logger.debug("No matching tool call found for result id: %s", tool_id)


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

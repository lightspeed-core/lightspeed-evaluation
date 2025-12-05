"""Streaming response parser for API client."""

import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DATA_PREFIX = "data: "


def parse_streaming_response(  # pylint: disable=too-many-branches
    response: httpx.Response,
) -> dict[str, Any]:
    """Parse streaming response and extract data."""
    conversation_id = ""
    final_response = ""
    tool_calls: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0

    for line in response.iter_lines():
        line = line.strip()
        if not line or not line.startswith(DATA_PREFIX):
            continue

        json_data = line.replace(DATA_PREFIX, "")  # Remove data prefix
        parsed_data = _parse_sse_line(json_data)

        if not parsed_data:
            continue

        event, event_data = parsed_data

        if event == "error" and "token" in event_data:
            error_message = event_data["token"]
            logger.error("Received error event from streaming API: %s", error_message)
            raise ValueError(f"Streaming API error: {error_message}")
        if event == "start" and "conversation_id" in event_data:
            conversation_id = event_data["conversation_id"].strip()
            logger.debug("Found conversation_id: %s", conversation_id)
        elif event == "turn_complete" and "token" in event_data:
            final_response = event_data["token"].strip()
            logger.debug("Found final response (%d characters)", len(final_response))
        elif event == "tool_call" and "token" in event_data:
            tool_call = _parse_tool_call(event_data["token"])
            if tool_call:
                tool_calls.append(tool_call)
                logger.debug("Found tool call: %s", tool_call)
        elif event == "end":
            # Extract token counts from end event (provided by lightspeed-stack)
            if "input_tokens" in event_data:
                input_tokens = event_data["input_tokens"]
            if "output_tokens" in event_data:
                output_tokens = event_data["output_tokens"]

    if not final_response:
        raise ValueError("No final response found in streaming output")
    if not conversation_id:
        raise ValueError("No Conversation ID found")

    tool_sequences = _format_tool_sequences(tool_calls)

    return {
        "response": final_response,
        "tool_calls": tool_sequences,
        "conversation_id": conversation_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


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


def _parse_tool_call(token: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Parse tool call from token."""
    try:
        tool_name = token.get("tool_name")
        arguments = token.get("arguments")

        if not tool_name:
            logger.debug("Tool call missing tool_name field")
            return None

        # Only process tool calls that explicitly have arguments field
        if arguments is None:
            logger.debug("Tool call missing arguments field for %s", tool_name)
            return None

        return {"tool_name": tool_name, "arguments": arguments}

    except (ValueError, IndexError, AttributeError) as e:
        logger.debug("Failed to parse tool call '%s': %s", token, e)
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

"""Streaming response parser utilities for agent API."""

import json
import logging
from typing import Any, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

DATA_PREFIX = "data: "
TOOL_PREFIX = "Tool:"
ARGUMENTS_SEPARATOR = " arguments:"


def parse_streaming_response(response: httpx.Response) -> dict[str, Any]:
    """Parse streaming response and extract data."""
    conversation_id = ""
    final_response = ""
    tool_calls = []

    for line in response.iter_lines():
        line = line.strip()
        if not line or not line.startswith(DATA_PREFIX):
            continue

        json_data = line[len(DATA_PREFIX) :]  # Remove data prefix
        parsed_data = _parse_streaming_line(json_data)

        if not parsed_data:
            continue

        event, event_data = parsed_data

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

    if not final_response:
        raise ValueError("No final response found in streaming output")
    if not conversation_id:
        raise ValueError("No Conversation ID found")

    tool_sequences = _format_tool_sequences(tool_calls)

    return {
        "response": final_response,
        "tool_calls": tool_sequences,
        "conversation_id": conversation_id,
    }


def _parse_streaming_line(json_data: str) -> Optional[Tuple[str, dict]]:
    """Parse a single streaming line and return event and data."""
    try:
        data = json.loads(json_data)
        event = data.get("event", "")
        event_data = data.get("data", {})
        return event, event_data
    except json.JSONDecodeError:
        logger.debug("Failed to parse JSON from streaming response: %s", json_data)
        return None


def _parse_tool_call(token: str) -> Optional[dict[str, Any]]:
    """Parse tool call from token."""
    # format: 'Tool:<name> arguments:{...}'
    try:
        token = token.strip()
        if not token.startswith(TOOL_PREFIX) or ARGUMENTS_SEPARATOR not in token:
            return None

        parts = token.split(ARGUMENTS_SEPARATOR, 1)
        if len(parts) != 2:
            return None

        tool_name = parts[0][len(TOOL_PREFIX) :]  # Remove Tool prefix
        arguments_str = parts[1]

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse tool arguments for '%s': invalid JSON '%s'",
                tool_name,
                arguments_str,
            )
            arguments = {}  # If JSON parsing fails, treat as empty arguments

        return {"name": tool_name, "arguments": arguments}
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

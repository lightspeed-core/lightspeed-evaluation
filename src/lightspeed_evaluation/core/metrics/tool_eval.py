"""Custom tool call evaluation utilities."""

import logging
import re
from typing import Any, Callable, Dict, List

from ..models import ToolCallData

logger = logging.getLogger(__name__)


def evaluate_tool_calls(
    expected: List[List[ToolCallData]], actual: List[List[Dict[str, Any]]]
) -> tuple[bool, str]:
    """Evaluate tool calls using the custom:tool_eval metric.

    Args:
        expected: Expected tool calls structure (List[List[ToolCallData]])
        actual: Actual tool calls from API response (List[List[Dict[str, Any]]])

    Returns:
        tuple: (success: bool, details: str)
    """
    try:
        # Convert expected ToolCallData objects to dict format for comparison
        expected_dicts = []
        for sequence in expected:
            sequence_dicts = []
            for tool_call in sequence:
                sequence_dicts.append(
                    {"name": tool_call.name, "arguments": tool_call.arguments}
                )
            expected_dicts.append(sequence_dicts)

        success = compare_tool_calls(expected_dicts, actual)

        if success:
            details = "Tool calls match expected structure and arguments"
        else:
            details = "Tool calls do not match expected structure or arguments"

        return success, details

    except (AttributeError, TypeError, ValueError) as e:
        logger.error("Error during tool evaluation: %s", e)
        return False, f"Tool evaluation error: {str(e)}"


def compare_tool_calls(
    expected: List[List[Dict[str, Any]]], actual: List[List[Dict[str, Any]]]
) -> bool:
    """Compare expected and actual tool calls."""
    return _compare_lists(
        expected,
        actual,
        _compare_tool_call_sequence,
        "Tool calls count mismatch: expected %d, got %d",
    )


def _compare_tool_call_sequence(
    expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]
) -> bool:
    """Compare a single sequence of tool calls."""
    return _compare_lists(
        expected,
        actual,
        _compare_single_tool_call,
        "Tool call count mismatch in sequence: expected %d, got %d",
    )


def _compare_lists(
    expected: List,
    actual: List,
    compare_func: Callable[[Any, Any], bool],
    mismatch_message: str,
) -> bool:
    """Compare two lists using a comparison function."""
    if len(expected) != len(actual):
        logger.debug(mismatch_message, len(expected), len(actual))
        return False

    for i, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        if not compare_func(expected_item, actual_item):
            logger.debug("Item %d does not match in %s", i, compare_func.__name__)
            return False
    return True


def _compare_single_tool_call(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """Compare a single tool call."""
    expected_name = expected.get("name")
    actual_name = actual.get("name")

    if expected_name != actual_name:
        logger.debug(
            "Tool name mismatch: expected '%s', got '%s'",
            expected_name,
            actual_name,
        )
        return False

    expected_args = expected.get("arguments", {})
    actual_args = actual.get("arguments", {})

    return _compare_tool_arguments(expected_args, actual_args)


def _compare_tool_arguments(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """Compare tool arguments name & value (regex pattern for the value)."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        logger.debug(
            "Argument type mismatch: expected dict, got %s and %s",
            type(expected),
            type(actual),
        )
        return False

    # Compare each expected argument
    for key, expected_value in expected.items():
        if key not in actual:
            logger.debug("Missing argument key: '%s'", key)
            return False

        actual_value = actual[key]

        expected_str = str(expected_value)
        actual_str = str(actual_value)

        # Use regex search for flexible matching
        # This is a quick work-around, enhance this to use both regex & exact match.
        try:
            if not re.search(expected_str, actual_str):
                logger.debug(
                    "Argument value mismatch for '%s': pattern '%s' not found in '%s'",
                    key,
                    expected_str,
                    actual_str,
                )
                return False
        except re.error as e:
            logger.debug(
                "Invalid regex pattern '%s' for key '%s': %s", expected_str, key, e
            )
            # If regex is invalid, fail the comparison
            return False

    # Check for extra keys in actual
    extra_keys = set(actual.keys()) - set(expected.keys())
    if extra_keys:
        logger.debug("Additional argument keys: %s", extra_keys)
        return False

    return True


def format_tool_calls_for_logging(tool_calls: List[List[Dict[str, Any]]]) -> str:
    """Format tool calls for logging purposes."""
    if not tool_calls:
        return "No tool calls"

    formatted = []
    for i, sequence in enumerate(tool_calls):
        sequence_str = f"Sequence {i+1}:"
        for j, tool_call in enumerate(sequence):
            tool_name = tool_call.get("name", "unknown")
            args = tool_call.get("arguments", {})
            sequence_str += f"\n  Tool {j+1}: {tool_name}({args})"
        formatted.append(sequence_str)

    return "\n".join(formatted)

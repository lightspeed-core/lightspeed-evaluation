"""Custom tool call evaluation utilities."""

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


def evaluate_tool_calls(
    expected: list[list[list[dict[str, Any]]]],
    actual: list[list[dict[str, Any]]],
) -> tuple[bool, str]:
    """Evaluate tool calls using the custom:tool_eval metric.

    Args:
        expected: Expected tool calls structure (with alternatives)
        actual: Actual tool calls from API response

    Returns:
        tuple: (success: bool, details: str)
    """
    try:
        # Try each set until one matches
        for i, expected_set in enumerate(expected):
            if compare_tool_calls(expected_set, actual):
                return _create_success_message(i, expected_set)

        # If all sets fail, return failure status & message
        return _create_failure_message(expected, actual)

    except (AttributeError, TypeError, ValueError) as e:
        logger.error("Error during tool evaluation: %s", e)
        return False, f"Tool evaluation error: {str(e)}"


def compare_tool_calls(
    expected: list[list[dict[str, Any]]], actual: list[list[dict[str, Any]]]
) -> bool:
    """Compare expected and actual tool calls."""
    return _compare_lists(
        expected,
        actual,
        _compare_tool_call_sequence,
        "Tool calls count mismatch: expected %d, got %d",
    )


def _compare_tool_call_sequence(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> bool:
    """Compare a single sequence of tool calls."""
    return _compare_lists(
        expected,
        actual,
        _compare_single_tool_call,
        "Tool call count mismatch in sequence: expected %d, got %d",
    )


def _compare_lists(
    expected: list,
    actual: list,
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


def _compare_single_tool_call(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare a single tool call."""
    expected_name = expected.get("tool_name")
    actual_name = actual.get("tool_name")

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


def _compare_tool_arguments(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
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


def _create_success_message(
    index: int, expected_set: list[list[dict[str, Any]]]
) -> tuple[bool, str]:
    """Create success message based on match type."""
    pattern_type = "Primary pattern" if index == 0 else f"Alternative {index + 1}"

    # Determine message based on what matched
    if len(expected_set) == 0:
        # Empty alternative matched - index 0 can never be empty due to constraints
        message = "No tool calls made (valid alternate skip scenario)"
    else:
        message = "Tool calls match expected structure and arguments"

    return True, f"{pattern_type} matched: {message}"


def _create_failure_message(
    expected: list[list[list[dict[str, Any]]]], actual: list[list[dict[str, Any]]]
) -> tuple[bool, str]:
    """Create failure message with helpful context."""
    # If we reach here, none of the alternatives matched

    if len(actual) == 0:
        return (
            False,
            "No actual tool calls made and this is not set as an expected alternative",
        )

    return (
        False,
        f"Tool calls made but didn't match any of the {len(expected)} expected pattern(s)",
    )


def format_tool_calls_for_logging(tool_calls: list[list[dict[str, Any]]]) -> str:
    """Format tool calls for logging purposes."""
    if not tool_calls:
        return "No tool calls"

    formatted = []
    for i, sequence in enumerate(tool_calls):
        sequence_str = f"Sequence {i+1}:"
        for j, tool_call in enumerate(sequence):
            tool_name = tool_call.get("tool_name", "unknown")
            arguments = tool_call.get("arguments", {})
            sequence_str += f"\n  Tool {j+1}: {tool_name}({arguments})"
        formatted.append(sequence_str)

    return "\n".join(formatted)

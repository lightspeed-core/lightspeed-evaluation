"""Custom tool call evaluation utilities."""

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


def evaluate_tool_calls(
    expected: list[list[list[dict[str, Any]]]],
    actual: list[list[dict[str, Any]]],
    ordered: bool = True,
) -> tuple[bool, str]:
    """Evaluate tool calls using the custom:tool_eval metric.

    Args:
        expected: Expected tool calls structure (with alternatives)
        actual: Actual tool calls from API response
        ordered: If True, tool calls must match in order. If False, order is ignored.

    Returns:
        tuple: (success: bool, details: str)
    """
    try:
        # Try each set until one matches
        for i, expected_set in enumerate(expected):
            if compare_tool_calls(expected_set, actual, ordered=ordered):
                return _create_success_message(i, expected_set, ordered=ordered)

        # If all sets fail, return failure status & message
        return _create_failure_message(expected, actual)

    except (AttributeError, TypeError, ValueError) as e:
        logger.error("Error during tool evaluation: %s", e)
        return False, f"Tool evaluation error: {str(e)}"


def compare_tool_calls(
    expected: list[list[dict[str, Any]]],
    actual: list[list[dict[str, Any]]],
    ordered: bool = True,
) -> bool:
    """Compare expected and actual tool calls.

    Args:
        expected: Expected tool call sequences
        actual: Actual tool call sequences
        ordered: If True, sequences must match in order. If False, order is ignored.

    Returns:
        True if tool calls match according to the ordered setting
    """
    if ordered:
        return _compare_lists_ordered(
            expected,
            actual,
            _compare_tool_call_sequence,
            "Tool calls count mismatch: expected %d, got %d",
        )
    return _compare_lists_unordered(
        expected,
        actual,
        _compare_tool_call_sequence,
        "Tool calls count mismatch: expected %d, got %d",
    )


def _compare_tool_call_sequence(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> bool:
    """Compare a single sequence of tool calls.

    Note: Within a sequence, tool calls are always compared in order.
    The `ordered` option only affects the order of sequences, not tools within a sequence.

    Args:
        expected: Expected tool calls in the sequence
        actual: Actual tool calls in the sequence

    Returns:
        True if the sequence matches
    """
    return _compare_lists_ordered(
        expected,
        actual,
        _compare_single_tool_call,
        "Tool call count mismatch in sequence: expected %d, got %d",
    )


def _compare_lists_ordered(
    expected: list,
    actual: list,
    compare_func: Callable[[Any, Any], bool],
    mismatch_message: str,
) -> bool:
    """Compare two lists in order using a comparison function.

    Args:
        expected: Expected list items
        actual: Actual list items
        compare_func: Function to compare individual items
        mismatch_message: Message format for length mismatch

    Returns:
        True if all items match in order
    """
    if len(expected) != len(actual):
        logger.debug(mismatch_message, len(expected), len(actual))
        return False

    for i, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        if not compare_func(expected_item, actual_item):
            logger.debug("Item %d does not match in %s", i, compare_func.__name__)
            return False
    return True


def _get_sort_key(item: Any) -> tuple[str, ...]:
    """Create a sort key from a tool call sequence or single tool call.

    Args:
        item: A tool call sequence (list of dicts) or single tool call (dict)

    Returns:
        Tuple of tool names for sorting
    """
    if isinstance(item, list):
        # Item is a sequence of tool calls - create key from tool names
        return tuple(tc.get("tool_name", "") for tc in item if isinstance(tc, dict))
    if isinstance(item, dict):
        # Item is a single tool call
        return (item.get("tool_name", ""),)
    return (str(item),)


def _compare_lists_unordered(
    expected: list,
    actual: list,
    compare_func: Callable[[Any, Any], bool],
    mismatch_message: str,
) -> bool:
    """Compare two lists without considering order.

    Sorts both lists by tool names and then performs ordered comparison.
    Time complexity: O(n log n) for sorting + O(n) for comparison.

    Args:
        expected: Expected list items
        actual: Actual list items
        compare_func: Function to compare individual items
        mismatch_message: Message format for length mismatch

    Returns:
        True if all expected items have a matching actual item
    """
    if len(expected) != len(actual):
        logger.debug(mismatch_message, len(expected), len(actual))
        return False

    if not expected:
        return True

    # Sort both lists by tool names for deterministic comparison
    sorted_expected = sorted(expected, key=_get_sort_key)
    sorted_actual = sorted(actual, key=_get_sort_key)

    return _compare_lists_ordered(
        sorted_expected, sorted_actual, compare_func, mismatch_message
    )


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
    index: int,
    expected_set: list[list[dict[str, Any]]],
    ordered: bool = True,
) -> tuple[bool, str]:
    """Create success message based on match type.

    Args:
        index: Index of the matched alternative (0 = primary)
        expected_set: The matched expected tool call set
        ordered: Whether ordered matching was used

    Returns:
        Tuple of (True, success message)
    """
    pattern_type = "Primary pattern" if index == 0 else f"Alternative {index + 1}"
    order_mode = "ordered" if ordered else "unordered"

    # Determine message based on what matched
    if len(expected_set) == 0:
        # Empty alternative matched - index 0 can never be empty due to constraints
        message = "No tool calls made (valid alternate skip scenario)"
    else:
        message = f"Tool calls match expected structure and arguments ({order_mode})"

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

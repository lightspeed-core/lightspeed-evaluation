"""Tool call evaluation utilities."""

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
            logger.debug("Invalid regex pattern '%s' for key '%s': %s", expected_str, key, e)
            # If regex is invalid, fail the comparison
            return False

    # Check for extra keys in actual
    extra_keys = set(actual.keys()) - set(expected.keys())
    if extra_keys:
        logger.debug("Additional argument keys: %s", extra_keys)
        return False

    return True

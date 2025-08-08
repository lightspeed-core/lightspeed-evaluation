"""Tool call evaluation utilities."""

import logging
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


def _compare_tool_call_sequence(expected: list[dict], actual: list[dict]) -> bool:
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
            logger.debug("Item %d does not match", i)
            return False
    return True


def _compare_single_tool_call(expected: dict, actual: dict) -> bool:
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


def _compare_tool_arguments(expected: dict, actual: dict) -> bool:
    """Compare tool arguments."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        logger.debug(
            "Argument type mismatch: expected dict, got %s and %s",
            type(expected),
            type(actual),
        )
        return False
    if len(expected) != len(actual):
        logger.debug(
            "Argument count mismatch: expected %d args, got %d args",
            len(expected),
            len(actual),
        )
        return False

    # Direct comparison is not done to have better debugging ability
    for key, expected_value in expected.items():
        if key not in actual:
            logger.debug("Missing argument key: '%s'", key)
            return False
        actual_value = actual[key]
        if expected_value != actual_value:
            logger.debug(
                "Argument value mismatch for '%s': expected %s, got %s",
                key,
                expected_value,
                actual_value,
            )
            return False

    return True

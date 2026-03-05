"""Custom tool call evaluation utilities."""

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


def evaluate_tool_calls(
    expected: list[list[list[dict[str, Any]]]],
    actual: list[list[dict[str, Any]]],
    ordered: bool = True,
    full_match: bool = True,
) -> tuple[bool, str]:
    """Evaluate tool calls using the custom:tool_eval metric.

    Args:
        expected: Expected tool calls structure (with alternatives)
        actual: Actual tool calls from API response
        ordered: If True, tool calls must match in order. If False, order is ignored.
        full_match: True requires all expected to match all actual (1:1).
            False requires all expected tools to be present (subset matching, extras allowed).

    Returns:
        tuple: (success: bool, details: str)
    """
    try:
        # Try each set until one matches, track best result for failure reporting
        best_result: dict[str, Any] = {}
        best_matched = -1

        for i, expected_set in enumerate(expected):
            result = compare_tool_calls(
                expected_set, actual, ordered=ordered, full_match=full_match
            )
            if result["success"]:
                return _create_success_message(
                    i,
                    expected_set,
                    ordered=ordered,
                    full_match=full_match,
                    match_stats=result.get("stats"),
                )

            # Track the best matching alternative (most matched tools)
            stats = result.get("stats", {})
            matched = stats.get("matched", 0)
            if matched > best_matched:
                best_matched = matched
                best_result = result

        # If all sets fail, return failure with best match stats
        return _create_failure_message(
            expected,
            actual,
            ordered=ordered,
            full_match=full_match,
            best_stats=best_result.get("stats"),
        )

    except (AttributeError, TypeError, ValueError) as e:
        logger.error("Error during tool evaluation: %s", e)
        return False, f"Tool evaluation error: {str(e)}"


def compare_tool_calls(
    expected: list[list[dict[str, Any]]],
    actual: list[list[dict[str, Any]]],
    ordered: bool = True,
    full_match: bool = True,
) -> dict[str, Any]:
    """Compare expected and actual tool calls.

    Args:
        expected: Expected tool call sequences
        actual: Actual tool call sequences
        ordered: Controls sequence normalization before comparison.
            If True, sequences are compared in their original order.
            If False, both lists are sorted before comparison.
        full_match: Determines matching strictness.
            True (default): Requires exact 1:1 match between expected and actual.
            False: Requires all expected tools to be found in actual (subset matching).
                   Extra actual tools beyond expected are allowed.

    Note:
        In partial match mode (full_match=False), the ordered parameter affects
        normalization (sorting) but greedy matching finds items regardless of
        their relative positions. For full match mode, ordered=True requires
        exact positional matching while ordered=False allows any order.

    Returns:
        Dict with 'success' bool and optional 'stats' for partial match
    """
    expected_normalized, actual_normalized = _normalize_sequences(
        expected, actual, ordered
    )
    mismatch_message = "Tool calls count mismatch: expected %d, got %d"

    if not full_match:
        matched, total, extra_tools, unmatched_expected = _compare_partial(
            expected_normalized, actual_normalized
        )
        # Partial match succeeds if all expected tools matched (subset matching)
        success = matched == total
        return {
            "success": success,
            "stats": {
                "matched": matched,
                "total": total,
                "unmatched": total - matched,
                "extra_actual_tools": extra_tools,
                "unmatched_expected_tools": unmatched_expected,
            },
        }

    # Full match (default)
    if len(expected_normalized) != len(actual_normalized):
        logger.debug(mismatch_message, len(expected_normalized), len(actual_normalized))
        return {"success": False}

    success = _compare_lists_ordered(
        expected_normalized,
        actual_normalized,
        _compare_tool_call_sequence,
        mismatch_message,
    )
    return {"success": success}


def _normalize_sequences(
    expected: list[list[dict[str, Any]]],
    actual: list[list[dict[str, Any]]],
    ordered: bool,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Normalize sequences for ordered/unordered comparisons."""
    if ordered:
        return expected, actual

    sorted_expected = sorted(expected, key=_get_sort_key)
    sorted_actual = sorted(actual, key=_get_sort_key)
    return sorted_expected, sorted_actual


def _compare_partial(
    expected: list[list[dict[str, Any]]],
    actual: list[list[dict[str, Any]]],
) -> tuple[int, int, list[str], list[str]]:
    """Compare tool calls with partial matching.

    Counts how many expected sequences are found in actual.
    Extra actual sequences are allowed (not required to match).
    Uses greedy matching - each actual can only be matched once.

    Note: Ordering is handled by caller via _normalize_sequences.
    This function uses the same core logic regardless of order mode.

    Args:
        expected: Expected tool call sequences (pre-normalized)
        actual: Actual tool call sequences (pre-normalized)

    Returns:
        Tuple of (matched_count, total_expected, extra_actual_tools, unmatched_expected_tools)
    """
    if not expected:
        extra_tools = [_get_sequence_tool_names(seq) for seq in actual]
        return (0, 0, extra_tools, [])

    matched = 0
    used_indices: set[int] = set()
    matched_expected_indices: set[int] = set()

    for i, expected_seq in enumerate(expected):
        for j, actual_seq in enumerate(actual):
            if j not in used_indices and _compare_tool_call_sequence(
                expected_seq, actual_seq
            ):
                matched += 1
                used_indices.add(j)
                matched_expected_indices.add(i)
                break

    extra_tools = [
        _get_sequence_tool_names(actual[i])
        for i in range(len(actual))
        if i not in used_indices
    ]
    unmatched_expected = [
        _get_sequence_tool_names(expected[i])
        for i in range(len(expected))
        if i not in matched_expected_indices
    ]
    logger.debug(
        "Partial match: %d/%d expected sequences found, extra: %s, unmatched: %s",
        matched,
        len(expected),
        extra_tools,
        unmatched_expected,
    )
    return (matched, len(expected), extra_tools, unmatched_expected)


def _get_sequence_tool_names(sequence: list[dict[str, Any]]) -> str:
    """Get tool names from a sequence as a comma-separated string."""
    names = [tc.get("tool_name", "unknown") for tc in sequence]
    return ", ".join(names) if len(names) > 1 else (names[0] if names else "unknown")


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
    if len(expected) != len(actual):
        logger.debug(
            "Tool call count mismatch in sequence: expected %d, got %d",
            len(expected),
            len(actual),
        )
        return False

    for i, expected_call in enumerate(expected):
        if not _compare_single_tool_call(expected_call, actual[i]):
            logger.debug("Tool call %d does not match in sequence", i)
            return False

    return True


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


def _compare_single_tool_call(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare a single tool call including name, arguments, and optional result.

    Args:
        expected: Expected tool call with tool_name, arguments, and optional result.
        actual: Actual tool call from API response.

    Returns:
        True if the tool call matches (name, arguments, and result if specified).
    """
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

    if not _compare_tool_arguments(expected_args, actual_args):
        return False

    # Compare result if expected (optional field)
    if not _compare_tool_result(expected, actual):
        return False

    return True


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


def _compare_tool_result(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare tool call result if expected result is specified.

    The result field is optional. If not specified in expected, comparison is skipped.
    When specified, uses regex pattern matching for flexible validation.

    Args:
        expected: Expected tool call dict with optional 'result' field.
        actual: Actual tool call dict with optional 'result' field.

    Returns:
        True if result matches or no expected result specified.
    """
    expected_result = expected.get("result")

    # If no expected result specified, skip comparison (pass)
    if expected_result is None:
        return True

    actual_result = actual.get("result")

    # Expected result specified but actual has no result
    if actual_result is None:
        logger.debug(
            "Expected result '%s' but actual tool call has no result",
            expected_result,
        )
        return False

    # Convert to strings for regex matching
    expected_str = str(expected_result)
    actual_str = str(actual_result)

    # Use regex search for flexible matching (same as arguments)
    try:
        if not re.search(expected_str, actual_str):
            logger.debug(
                "Tool result mismatch: pattern '%s' not found in '%s'",
                expected_str,
                actual_str,
            )
            return False
    except re.error as e:
        logger.debug("Invalid regex pattern '%s' for result: %s", expected_str, e)
        return False

    logger.debug("Tool result matched: '%s'", actual_str)
    return True


def _get_mode_suffix(ordered: bool, full_match: bool) -> str:
    """Get the mode suffix string for messages.

    Args:
        ordered: Whether ordered matching was used
        full_match: Whether full or partial matching was used

    Returns:
        Mode suffix like "(partial, ordered)"
    """
    match_mode = "full" if full_match else "partial"
    order_mode = "ordered" if ordered else "unordered"
    return f"({match_mode}, {order_mode})"


def _format_match_stats(
    match_stats: dict[str, Any],
    ordered: bool,
    full_match: bool,
) -> str:
    """Format match statistics into a human-readable string.

    Args:
        match_stats: Dict with matched/total/unmatched/extra_actual_tools/unmatched_expected_tools
        ordered: Whether ordered matching was used
        full_match: Whether full or partial matching was used

    Returns:
        Formatted statistics string
    """
    matched = match_stats["matched"]
    total = match_stats["total"]
    unmatched = match_stats["unmatched"]

    # Extra actual tools (from actual that weren't used)
    extra_tools: list[str] = match_stats.get("extra_actual_tools", [])
    extra_count = len(extra_tools)
    extra_info = f"[{', '.join(extra_tools)}]" if extra_tools else "none"

    # Handle empty expected (no tool calls expected for this alternative)
    if total == 0:
        if extra_count > 0:
            return (
                f"No expected tool calls (skip scenario), "
                f"but {extra_count} actual: {extra_info} "
                f"{_get_mode_suffix(ordered, full_match)}"
            )
        return (
            f"No expected tool calls (skip scenario) "
            f"{_get_mode_suffix(ordered, full_match)}"
        )

    # Unmatched expected tools (from expected that didn't match)
    unmatched_expected: list[str] = match_stats.get("unmatched_expected_tools", [])
    unmatched_info = (
        f"[{', '.join(unmatched_expected)}]" if unmatched_expected else "none"
    )

    return (
        f"{matched}/{total} expected matched, {unmatched} unmatched: {unmatched_info}, "
        f"{extra_count} extra in response: {extra_info} "
        f"{_get_mode_suffix(ordered, full_match)}"
    )


def _create_success_message(
    index: int,
    expected_set: list[list[dict[str, Any]]],
    ordered: bool = True,
    full_match: bool = True,
    match_stats: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Create success message based on match type.

    Args:
        index: Index of the matched alternative (0 = primary)
        expected_set: The matched expected tool call set
        ordered: Whether ordered matching was used
        full_match: Whether full or partial matching was used
        match_stats: Optional dict with matched/total/unmatched/extra_actual_tools

    Returns:
        Tuple of (True, success message)
    """
    pattern_type = "Primary pattern" if index == 0 else f"Alternative {index + 1}"

    # Check match_stats first to include extra tools info (for partial match mode)
    if match_stats:
        message = f"Tool calls: {_format_match_stats(match_stats, ordered, full_match)}"
    elif len(expected_set) == 0:
        message = "No tool calls made (valid alternate skip scenario)"
    else:
        message = (
            f"Tool calls match expected structure and arguments "
            f"{_get_mode_suffix(ordered, full_match)}"
        )

    return True, f"{pattern_type} matched: {message}"


def _create_failure_message(
    expected: list[list[list[dict[str, Any]]]],
    actual: list[list[dict[str, Any]]],
    ordered: bool = True,
    full_match: bool = True,
    best_stats: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Create failure message with helpful context.

    Args:
        expected: Expected tool call patterns (with alternatives)
        actual: Actual tool calls from API response
        ordered: Whether ordered matching was used
        full_match: Whether full or partial matching was used
        best_stats: Stats from best matching alternative (most matched tools)

    Returns:
        Tuple of (False, failure message)
    """
    if len(actual) == 0:
        return (
            False,
            "No actual tool calls made and this is not set as an expected alternative",
        )

    base_msg = (
        f"Tool calls made but didn't match any of the {len(expected)} "
        f"expected pattern(s)"
    )

    if best_stats and not full_match:
        return (
            False,
            f"{base_msg}: {_format_match_stats(best_stats, ordered, full_match)}",
        )

    return (False, base_msg)


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

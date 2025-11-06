"""Keywords evaluation utilities."""

from typing import Any, Optional

from lightspeed_evaluation.core.models import TurnData


def _validate_inputs(
    is_conversation: bool, turn_data: Optional[TurnData]
) -> Optional[tuple[Optional[float], str]]:
    """Validate inputs for keywords evaluation."""
    if is_conversation:
        return None, "Keywords eval is a turn-level metric"

    if turn_data is None:
        return None, "TurnData is required for keywords eval evaluation"

    if not turn_data.expected_keywords:
        return None, "No expected keywords provided for keywords eval evaluation"

    if not turn_data.response:
        return 0.0, "No response provided for keywords eval evaluation"

    return None


def _check_keyword_list(
    keyword_list: list[str], response_lower: str
) -> tuple[list[str], bool]:
    """Check if all keywords in a list match the response."""
    matched_keywords = []
    all_matched = True

    for keyword in keyword_list:
        if keyword.lower() in response_lower:
            matched_keywords.append(keyword)
        else:
            all_matched = False

    return matched_keywords, all_matched


def _create_success_result(
    list_index: int, matched_keywords: list[str]
) -> tuple[float, str]:
    """Create success result for keywords evaluation."""
    matched_str = ", ".join(f"'{kw}'" for kw in matched_keywords)
    reason = (
        f"Keywords eval successful: Option {list_index + 1} - "
        f"all keywords matched: {matched_str}"
    )
    return 1.0, reason


def _create_failure_result(
    expected_keywords: list[list[str]], response_lower: str
) -> tuple[float, str]:
    """Create failure result for keywords evaluation."""
    failed_details = []

    for list_index, keyword_list in enumerate(expected_keywords):
        matched_keywords, _ = _check_keyword_list(keyword_list, response_lower)
        unmatched_keywords = [
            kw for kw in keyword_list if kw.lower() not in response_lower
        ]

        if unmatched_keywords:
            unmatched_str = ", ".join(f"'{kw}'" for kw in unmatched_keywords)
            matched_str = (
                ", ".join(f"'{kw}'" for kw in matched_keywords)
                if matched_keywords
                else "none"
            )
            failed_details.append(
                f"Option {list_index + 1}: unmatched [{unmatched_str}], matched [{matched_str}]"
            )

    reason = f"Keywords eval failed: All options failed - {'; '.join(failed_details)}"
    return 0.0, reason


def evaluate_keywords(
    _conv_data: Any,
    _turn_idx: Optional[int],
    turn_data: Optional[TurnData],
    is_conversation: bool,
) -> tuple[Optional[float], str]:
    """Evaluate keywords using substring matching with sequential list checking.

    Logic: Check first option - if all keywords match, evaluation succeeds.
    If first option fails, try next alternative, and so on.
    If all alternatives fail, evaluation fails.

    Args:
        _conv_data: Conversation data (unused)
        _turn_idx: Turn index (unused)
        turn_data: Turn data containing response and expected keywords
        is_conversation: Whether this is conversation-level evaluation

    Returns:
        tuple: (score: float, reason: str)
            - score: 1.0 if any keyword list has all keywords matched, 0.0 otherwise
            - reason: Detailed explanation of evaluation results
    """
    # Validate inputs
    validation_result = _validate_inputs(is_conversation, turn_data)
    if validation_result:
        return validation_result

    if (
        turn_data is None
        or turn_data.response is None
        or turn_data.expected_keywords is None
    ):
        return None, "Invalid turn data after validation"

    response_lower = turn_data.response.lower()

    # Check each expected keywords list
    for list_index, keyword_list in enumerate(turn_data.expected_keywords):
        matched_keywords, all_matched = _check_keyword_list(
            keyword_list, response_lower
        )

        if all_matched:
            return _create_success_result(list_index, matched_keywords)

    # If we reach here, all alternatives failed
    return _create_failure_result(turn_data.expected_keywords, response_lower)

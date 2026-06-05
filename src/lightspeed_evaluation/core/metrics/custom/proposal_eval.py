"""Proposal status evaluation for CRD-based agent workflows."""

from typing import Any, Optional

from lightspeed_evaluation.core.models import TurnData
from lightspeed_evaluation.core.proposal import derive_phase


def _check_phase(
    expected: dict[str, Any],
    conditions: list[dict[str, Any]],
    proposal_spec: Optional[dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check exact phase match."""
    phase = expected.get("phase")
    if phase is None:
        return None

    actual = derive_phase(conditions, proposal_spec)
    if actual == phase:
        return True, f"Phase matches: {actual}"
    return False, f"Phase mismatch: expected '{phase}', got '{actual}'"


def _check_phase_in(
    expected: dict[str, Any],
    conditions: list[dict[str, Any]],
    proposal_spec: Optional[dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check phase membership in a list."""
    phase_in = expected.get("phase_in")
    if phase_in is None:
        return None

    actual = derive_phase(conditions, proposal_spec)
    if actual in phase_in:
        return True, f"Phase '{actual}' in {phase_in}"
    return False, f"Phase '{actual}' not in {phase_in}"


def _check_conditions(
    expected: dict[str, Any],
    conditions: list[dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check specific condition assertions."""
    expected_conditions = expected.get("conditions")
    if expected_conditions is None:
        return None

    by_type = {c["type"]: c for c in conditions if isinstance(c, dict) and "type" in c}

    for exp_cond in expected_conditions:
        cond_type = exp_cond.get("type")
        if cond_type is None:
            return False, "Condition assertion missing 'type' field"

        actual_cond = by_type.get(cond_type)
        if actual_cond is None:
            return False, f"Condition '{cond_type}' not found in proposal status"

        exp_status = exp_cond.get("status")
        if exp_status is not None and actual_cond.get("status") != exp_status:
            return (
                False,
                f"Condition '{cond_type}' status: "
                f"expected '{exp_status}', got '{actual_cond.get('status')}'",
            )

        exp_reason = exp_cond.get("reason")
        if exp_reason is not None and actual_cond.get("reason") != exp_reason:
            return (
                False,
                f"Condition '{cond_type}' reason: "
                f"expected '{exp_reason}', got '{actual_cond.get('reason')}'",
            )

    return True, "All condition assertions passed"


def _check_verification(
    expected: dict[str, Any],
    conditions: list[dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check verification-specific assertions."""
    verification = expected.get("verification")
    if verification is None:
        return None

    by_type = {c["type"]: c for c in conditions if isinstance(c, dict) and "type" in c}
    verified = by_type.get("Verified")

    if verified is None:
        return False, "Verified condition not found in proposal status"

    passed = verification.get("passed")
    if passed is not None:
        actual_passed = verified.get("status") == "True"
        if actual_passed != passed:
            return (
                False,
                f"Verification passed: expected {passed}, got {actual_passed}",
            )

    summary_contains = verification.get("summary_contains")
    if summary_contains is not None:
        message = verified.get("message", "")
        if summary_contains.lower() not in message.lower():
            return (
                False,
                f"Verification summary does not contain '{summary_contains}': "
                f"got '{message[:200]}'",
            )

    return True, "Verification assertions passed"


def evaluate_proposal_status(
    _conv_data: Any,
    _turn_idx: Optional[int],
    turn_data: Optional[TurnData],
    is_conversation: bool,
) -> tuple[Optional[float], str]:
    """Evaluate proposal status against expected assertions.

    Args:
        _conv_data: Conversation data (unused).
        _turn_idx: Turn index (unused).
        turn_data: Turn data with proposal_status and expected_proposal_status.
        is_conversation: Whether this is conversation-level evaluation.

    Returns:
        Tuple of (score, reason). Score is 1.0 if all checks pass, 0.0 on
        first failure, None if metric should be skipped.
    """
    if is_conversation:
        return None, "Proposal status is a turn-level metric"

    if turn_data is None:
        return None, "TurnData is required for proposal status evaluation"

    if not turn_data.expected_proposal_status:
        return None, "No expected_proposal_status provided"

    if not turn_data.proposal_status:
        return 0.0, "proposal_status not populated by driver"

    expected = turn_data.expected_proposal_status
    conditions = turn_data.proposal_status.get("conditions", [])
    proposal_spec = turn_data.proposal_spec

    checks = [
        _check_phase(expected, conditions, proposal_spec),
        _check_phase_in(expected, conditions, proposal_spec),
        _check_conditions(expected, conditions),
        _check_verification(expected, conditions),
    ]

    reasons: list[str] = []
    for result in checks:
        if result is None:
            continue
        passed, reason = result
        if not passed:
            return 0.0, reason
        reasons.append(reason)

    return 1.0, "; ".join(reasons) if reasons else "All checks passed"

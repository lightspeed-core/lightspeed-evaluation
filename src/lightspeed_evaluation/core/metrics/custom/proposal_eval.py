"""Proposal status evaluation for CRD-based agent workflows."""

import re
from datetime import datetime
from typing import Any, Optional

from lightspeed_evaluation.core.models import TurnData
from lightspeed_evaluation.core.proposal import derive_phase

_NON_TERMINAL_PHASES: frozenset[str] = frozenset({"StepStarted"})


def _parse_duration(duration_str: str) -> float:
    """Parse a Go-style duration string into total seconds.

    Args:
        duration_str: Duration like "5m", "2m30s", "1h", "1h30m15s".

    Returns:
        Total seconds as a float.

    Raises:
        ValueError: If the format is unrecognized or empty.
    """
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", duration_str)
    if not match or not any(match.groups()):
        msg = f"Unrecognized duration format: '{duration_str}'"
        raise ValueError(msg)
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return float(hours * 3600 + minutes * 60 + seconds)


def _get_result_phase(result: dict[str, Any]) -> Optional[str]:
    """Extract the outcome phase from a step result.

    Checks a direct ``phase`` field first.  When absent, looks for
    a ``Completed`` condition by type (matching the operator's
    kubebuilder printcolumn ``conditions[?(@.type=="Completed")].reason``).
    Falls back to the first condition's reason for in-progress results
    that only carry a ``Started`` condition.

    Args:
        result: A single step result dictionary.

    Returns:
        The phase string, or None if not found.
    """
    phase = result.get("phase")
    if phase is not None:
        return phase
    conditions = result.get("conditions", [])
    for cond in conditions:
        if isinstance(cond, dict) and cond.get("type") == "Completed":
            return cond.get("reason")
    if conditions and isinstance(conditions[0], dict):
        return conditions[0].get("reason")
    return None


def _latest_terminal_result(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the latest result with a terminal phase.

    Skips results whose phase is non-terminal (e.g. ``StepStarted``
    from an in-progress retry) and returns the most recent result
    that has reached a terminal phase.  Falls back to the last result
    when every entry is non-terminal.

    Args:
        results: Non-empty list of step result dictionaries.

    Returns:
        The chosen result dictionary.
    """
    for result in reversed(results):
        phase = _get_result_phase(result)
        if phase is None or phase not in _NON_TERMINAL_PHASES:
            return result
    return results[-1]


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


def _check_max_duration(
    expected: dict[str, Any],
    conditions: list[dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check that total elapsed time across conditions is within limit."""
    max_duration = expected.get("max_duration")
    if max_duration is None:
        return None

    timestamps: list[datetime] = []
    for cond in conditions:
        ts = cond.get("lastTransitionTime") if isinstance(cond, dict) else None
        if ts is not None:
            timestamps.append(datetime.fromisoformat(ts))

    if not timestamps:
        return False, "No lastTransitionTime found in conditions"

    elapsed = (max(timestamps) - min(timestamps)).total_seconds()
    limit = _parse_duration(max_duration)
    if elapsed <= limit:
        return True, f"Duration {elapsed:.0f}s within limit {max_duration}"
    return False, f"Duration {elapsed:.0f}s exceeds limit {max_duration} ({limit:.0f}s)"


def _check_max_attempts(
    expected: dict[str, Any],
    conditions: list[dict[str, Any]],
    proposal_status: dict[str, Any],
) -> Optional[tuple[bool, str]]:
    """Check that the number of execution attempts is within limit."""
    max_attempts = expected.get("max_attempts")
    if max_attempts is None:
        return None

    actual = proposal_status.get("attempts")
    if actual is None:
        actual = (
            sum(
                1
                for c in conditions
                if isinstance(c, dict) and c.get("reason") == "RetryingExecution"
            )
            + 1
        )

    if actual <= max_attempts:
        return True, f"Attempts {actual} within limit {max_attempts}"
    return False, f"Attempts {actual} exceeds limit {max_attempts}"


def _check_analysis_component(
    comp_type: str,
    expected_comp: dict[str, Any],
    actual_by_type: dict[str, dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check a single expected component against actual components by type."""
    if expected_comp.get("absent"):
        present = comp_type in actual_by_type
        detail = "should be absent but is present" if present else "correctly absent"
        return not present, f"Component '{comp_type}' {detail}"

    actual = actual_by_type.get(comp_type)
    if actual is None:
        return False, f"Component '{comp_type}' not found"

    match = expected_comp.get("match")
    if match is not None:
        for key, value in match.items():
            if actual.get(key) != value:
                return (
                    False,
                    f"Component '{comp_type}' field '{key}': "
                    f"expected '{value}', got '{actual.get(key)}'",
                )

    match_contains = expected_comp.get("match_contains")
    if match_contains is not None:
        for key, substring in match_contains.items():
            actual_val = str(actual.get(key, ""))
            if substring.lower() not in actual_val.lower():
                return (
                    False,
                    f"Component '{comp_type}' field '{key}' does not contain "
                    f"'{substring}': got '{actual_val[:200]}'",
                )

    required = expected_comp.get("required")
    if required is not None:
        for key in required:
            if key not in actual:
                return (
                    False,
                    f"Component '{comp_type}' missing required field '{key}'",
                )

    return True, f"Component '{comp_type}' assertions passed"


def _check_analysis_option(
    idx: int,
    expected_opt: dict[str, Any],
    actual_opt: dict[str, Any],
) -> Optional[tuple[bool, str]]:
    """Check assertions on a single analysis option by index."""
    risk_in = expected_opt.get("risk_in")
    if risk_in is not None:
        actual_risk = actual_opt.get("remediationPlan", {}).get("risk", "")
        if actual_risk.lower() not in [r.lower() for r in risk_in]:
            return False, f"Option[{idx}] risk '{actual_risk}' not in {risk_in}"

    confidence_in = expected_opt.get("confidence_in")
    if confidence_in is not None:
        actual_conf = actual_opt.get("diagnosis", {}).get("confidence", "")
        if actual_conf.lower() not in [c.lower() for c in confidence_in]:
            return (
                False,
                f"Option[{idx}] confidence '{actual_conf}' not in {confidence_in}",
            )

    diagnosis_contains = expected_opt.get("diagnosis_contains")
    if diagnosis_contains is not None:
        summary = actual_opt.get("diagnosis", {}).get("summary", "")
        for substring in diagnosis_contains:
            if substring.lower() not in summary.lower():
                return (
                    False,
                    f"Option[{idx}] diagnosis does not contain "
                    f"'{substring}': got '{summary[:200]}'",
                )

    components = expected_opt.get("components")
    if components is not None:
        actual_by_type = {
            c["type"]: c
            for c in actual_opt.get("components", [])
            if isinstance(c, dict) and "type" in c
        }
        for exp_comp in components:
            ctype = exp_comp.get("type")
            if ctype is None:
                return False, f"Option[{idx}] component assertion missing 'type'"
            result = _check_analysis_component(ctype, exp_comp, actual_by_type)
            if result is not None and not result[0]:
                return False, f"Option[{idx}]: {result[1]}"

    return True, f"Option[{idx}] assertions passed"


def _check_analysis(
    expected: dict[str, Any],
    proposal_results: Optional[dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check analysis-specific assertions (options, risk, confidence, components)."""
    analysis_expected = expected.get("analysis")
    if analysis_expected is None:
        return None

    if not proposal_results:
        return False, "No proposal_results available for analysis check"

    analysis_results = [
        r for r in proposal_results.get("analysis", []) if isinstance(r, dict)
    ]
    latest_analysis = (
        _latest_terminal_result(analysis_results) if analysis_results else {}
    )
    actual_options: list[dict[str, Any]] = [
        opt for opt in latest_analysis.get("options", []) if isinstance(opt, dict)
    ]

    min_options = analysis_expected.get("min_options")
    if min_options is not None and len(actual_options) < min_options:
        return (
            False,
            f"Analysis has {len(actual_options)} options, "
            f"expected at least {min_options}",
        )

    expected_options = analysis_expected.get("options", [])
    for idx, exp_opt in enumerate(expected_options):
        if idx >= len(actual_options):
            return (
                False,
                f"Option[{idx}] expected but only {len(actual_options)} options present",
            )
        result = _check_analysis_option(idx, exp_opt, actual_options[idx])
        if result is not None:
            passed, reason = result
            if not passed:
                return False, reason

    return True, "Analysis assertions passed"


def _check_execution(
    expected: dict[str, Any],
    proposal_results: Optional[dict[str, Any]],
) -> Optional[tuple[bool, str]]:
    """Check execution-specific assertions."""
    execution_expected = expected.get("execution")
    if execution_expected is None:
        return None

    if not proposal_results:
        return False, "No proposal_results available for execution check"

    execution_results = [
        r for r in proposal_results.get("execution", []) if isinstance(r, dict)
    ]
    if not execution_results:
        return False, "No execution results available"

    latest_execution = _latest_terminal_result(execution_results)
    phase = execution_expected.get("phase")
    if phase is not None:
        actual_phase = _get_result_phase(latest_execution) or "Unknown"
        if actual_phase != phase:
            return (
                False,
                f"Execution phase: expected '{phase}', got '{actual_phase}'",
            )

    return True, "Execution assertions passed"


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
    proposal_status = turn_data.proposal_status
    conditions = proposal_status.get("conditions", [])
    proposal_spec = turn_data.proposal_spec
    proposal_results = turn_data.proposal_results

    checks = [
        _check_phase(expected, conditions, proposal_spec),
        _check_phase_in(expected, conditions, proposal_spec),
        _check_max_duration(expected, conditions),
        _check_max_attempts(expected, conditions, proposal_status),
        _check_analysis(expected, proposal_results),
        _check_execution(expected, proposal_results),
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

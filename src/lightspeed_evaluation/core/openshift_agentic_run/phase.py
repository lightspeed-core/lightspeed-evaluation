"""Derive terminal phase from AgenticRun CRD conditions."""

from typing import Any, Optional


def derive_phase(
    conditions: list[dict[str, Any]],
    openshift_agentic_run_spec: Optional[dict[str, Any]] = None,
) -> str:
    """Derive the terminal phase from CRD conditions.

    Args:
        conditions: List of condition dicts from openshift_agentic_run_status.
        openshift_agentic_run_spec: AgenticRun spec to determine the last expected step.

    Returns:
        Phase string: Completed, Failed, Denied, Escalated, Escalating, or InProgress.
    """
    by_type = {c["type"]: c for c in conditions if isinstance(c, dict) and "type" in c}

    # Check terminal conditions derived from Denied / Escalated.
    terminal_checks: list[tuple[str, str, str]] = [
        ("Denied", "True", "Denied"),
        ("Escalated", "True", "Escalated"),
        ("Escalated", "False", "Failed"),
        ("Escalated", "Unknown", "Escalating"),
    ]
    for cond_type, expected_status, phase in terminal_checks:
        if by_type.get(cond_type, {}).get("status") == expected_status:
            return phase

    for c in conditions:
        if isinstance(c, dict) and (
            c.get("type") in {"Analyzed", "Executed", "Verified"}
            and c.get("status") == "False"
        ):
            return "Failed"

    step_to_condition = {"verification": "Verified", "execution": "Executed"}
    if openshift_agentic_run_spec:
        last = next(
            (
                cond
                for step, cond in step_to_condition.items()
                if step in openshift_agentic_run_spec
            ),
            "Analyzed",
        )
    else:
        last = "Analyzed"
        for step in ("Verified", "Executed", "Analyzed"):
            if by_type.get(step, {}).get("status") == "True":
                last = step
                break

    if by_type.get(last, {}).get("status") == "True":
        return "Completed"

    return "InProgress"

"""Derive terminal phase from Proposal CRD conditions."""

from typing import Any, Optional


def derive_phase(
    conditions: list[dict[str, Any]],
    proposal_spec: Optional[dict[str, Any]] = None,
) -> str:
    """Derive the terminal phase from CRD conditions.

    Args:
        conditions: List of condition dicts from proposal_status.
        proposal_spec: Proposal spec to determine the last expected step.

    Returns:
        Phase string: Completed, Failed, Denied, Escalated, or InProgress.
    """
    by_type = {c["type"]: c for c in conditions if isinstance(c, dict) and "type" in c}

    if by_type.get("Denied", {}).get("status") == "True":
        return "Denied"
    if by_type.get("Escalated", {}).get("status") == "True":
        return "Escalated"

    for c in conditions:
        if isinstance(c, dict) and (
            c.get("type") in {"Analyzed", "Executed", "Verified"}
            and c.get("status") == "False"
            and c.get("reason") != "RetryingExecution"
        ):
            return "Failed"

    step_to_condition = {"verification": "Verified", "execution": "Executed"}
    if proposal_spec:
        last = next(
            (cond for step, cond in step_to_condition.items() if step in proposal_spec),
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

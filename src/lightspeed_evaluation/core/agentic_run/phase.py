"""Derive terminal phase from AgenticRun CRD conditions."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def derive_phase(
    conditions: list[dict[str, Any]],
    agentic_run_spec: Optional[dict[str, Any]] = None,
    proposal_spec: Optional[dict[str, Any]] = None,
) -> str:
    """Derive the terminal phase from CRD conditions.

    Args:
        conditions: List of condition dicts from agentic_run_status.
        agentic_run_spec: AgenticRun spec to determine the last expected step.
        proposal_spec: Deprecated alias for agentic_run_spec.

    Returns:
        Phase string: Completed, Failed, Denied, Escalated, or InProgress.
    """
    # Handle backward compatibility: proposal_spec is deprecated alias
    if proposal_spec is not None:
        if agentic_run_spec is not None:
            logger.warning(
                "Both 'agentic_run_spec' and 'proposal_spec' provided. "
                "Using 'agentic_run_spec'. 'proposal_spec' is deprecated."
            )
        else:
            logger.warning(
                "Parameter 'proposal_spec' is deprecated. Use 'agentic_run_spec' instead."
            )
            agentic_run_spec = proposal_spec

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
    if agentic_run_spec:
        last = next(
            (
                cond
                for step, cond in step_to_condition.items()
                if step in agentic_run_spec
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

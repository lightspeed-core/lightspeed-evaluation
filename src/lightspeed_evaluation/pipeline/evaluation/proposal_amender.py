"""ProposalAmender — fetches child Result CRs and enriches TurnData."""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional

from lightspeed_evaluation.core.models import TurnData
from lightspeed_evaluation.pipeline.evaluation.cli import CLIClient

logger = logging.getLogger(__name__)

STEP_RESOURCES: dict[str, str] = {
    "analysis": "analysisresults",
    "execution": "executionresults",
    "verification": "verificationresults",
    "escalation": "escalationresults",
}


class ProposalAmender:
    """Fetches child Result CRs and enriches TurnData with structured results and summary."""

    def __init__(self, cli_client: CLIClient) -> None:
        """Initialize with a CLIClient for fetching child CRs."""
        self._cli = cli_client

    def amend(
        self, turn_data: TurnData, proposal_status: dict[str, Any]
    ) -> Optional[str]:
        """Amend turn_data in-place with proposal results and Markdown summary.

        Returns:
            Error message on failure, None on success.
        """
        try:
            return self._do_amend(turn_data, proposal_status)
        except (KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            return f"ProposalAmender error: {exc}"

    def _do_amend(
        self, turn_data: TurnData, proposal_status: dict[str, Any]
    ) -> Optional[str]:
        """Internal amend logic."""
        turn_data.proposal_status = proposal_status

        steps = proposal_status.get("steps", {})
        if not steps:
            turn_data.proposal_results = {}
            turn_data.response = self.build_summary(turn_data, {})
            return None

        results: dict[str, list[dict[str, Any]]] = {}
        for step_name, resource_plural in STEP_RESOURCES.items():
            step_data = steps.get(step_name)
            if step_data is None:
                continue
            refs = step_data.get("results", [])
            step_results: list[dict[str, Any]] = []
            for ref in refs:
                ref_name = ref.get("name", "")
                if not ref_name:
                    continue
                cr, err = self._cli.get_resource(resource_plural, ref_name)
                if err:
                    logger.warning(
                        "Failed to fetch %s/%s: %s",
                        resource_plural,
                        ref_name,
                        err,
                    )
                    continue
                status = cr.get("status", {})
                if status:
                    step_results.append(status)
            results[step_name] = step_results

        turn_data.proposal_results = results
        turn_data.response = self.build_summary(turn_data, results)

        return None

    @staticmethod
    def build_summary(
        turn_data: TurnData,
        results: dict[str, list[dict[str, Any]]],
    ) -> str:
        """Build a Markdown workflow summary from structured results."""
        sections: list[str] = []

        sections.append(f"## Request\n\n{turn_data.query or 'N/A'}")

        analysis_results = results.get("analysis", [])
        if analysis_results:
            sections.append(_build_analysis_section(analysis_results))

        execution_results = results.get("execution", [])
        if execution_results:
            sections.append(_build_execution_section(execution_results))

        verification_results = results.get("verification", [])
        if verification_results:
            sections.append(_build_verification_section(verification_results))

        escalation_results = results.get("escalation", [])
        if escalation_results:
            sections.append(_build_escalation_section(escalation_results))

        sections.append(_build_outcome_section(turn_data))

        return "\n\n".join(sections)


def _build_analysis_section(analysis_results: list[dict[str, Any]]) -> str:
    """Build the Analysis section from AnalysisResult statuses."""
    lines: list[str] = ["## Analysis"]
    for result_status in analysis_results:
        options = result_status.get("options", [])
        if not options:
            failure = result_status.get("failureReason", "")
            if failure:
                lines.append(f"\n**Failed:** {failure}")
            continue
        lines.append(f"\n{len(options)} option(s) proposed")
        for idx, option in enumerate(options):
            label = "(Approved)" if idx == 0 else ""
            title = option.get("title", "Untitled")
            lines.append(f"\n### Option {idx} {label}: {title}".rstrip())
            _append_diagnosis(lines, option.get("diagnosis", {}))
            _append_proposal(lines, option.get("proposal", {}))

    return "\n".join(lines)


def _append_diagnosis(lines: list[str], diagnosis: dict[str, Any]) -> None:
    """Append diagnosis details to output lines."""
    if not diagnosis:
        return
    summary = diagnosis.get("summary", "")
    confidence = diagnosis.get("confidence", "")
    root_cause = diagnosis.get("rootCause", "")
    if summary:
        lines.append(f"**Diagnosis:** {summary} (Confidence: {confidence})")
    if root_cause:
        lines.append(f"**Root Cause:** {root_cause}")


def _append_proposal(lines: list[str], proposal: dict[str, Any]) -> None:
    """Append proposal details to output lines."""
    if not proposal:
        return
    actions = proposal.get("actions", [])
    if actions:
        lines.append("**Proposed Actions:**")
        for i, action in enumerate(actions, 1):
            a_type = action.get("type", "")
            a_desc = action.get("description", "")
            lines.append(f"{i}. [{a_type}] {a_desc}")
    risk = proposal.get("risk", "")
    reversible = proposal.get("reversible", "")
    if risk or reversible:
        lines.append(f"**Risk:** {risk} | **Reversible:** {reversible}")
    impact = proposal.get("estimatedImpact", "")
    if impact:
        lines.append(f"**Estimated Impact:** {impact}")


def _build_execution_section(execution_results: list[dict[str, Any]]) -> str:
    """Build the Execution section from ExecutionResult statuses."""
    lines: list[str] = ["## Execution"]
    for result_status in execution_results:
        failure = result_status.get("failureReason", "")
        if failure:
            lines.append(f"\n**Failed:** {failure}")
            continue
        actions = result_status.get("actionsTaken", [])
        if actions:
            lines.append("\n**Actions Taken:**")
            for i, action in enumerate(actions, 1):
                lines.append(_format_execution_action(i, action))
        verification = result_status.get("verification", {})
        if verification:
            condition_outcome = verification.get("conditionOutcome", "")
            ver_summary = verification.get("summary", "")
            lines.append(f"**Post-execution:** {condition_outcome} — {ver_summary}")
    return "\n".join(lines)


def _format_execution_action(index: int, action: dict[str, Any]) -> str:
    """Format a single execution action line."""
    a_type = action.get("type", "")
    a_desc = action.get("description", "")
    outcome = action.get("outcome", "")
    line = f"{index}. [{a_type}] {a_desc} → {outcome}"
    output = action.get("output", "")
    if output:
        line += f"\n   Output: {output}"
    error = action.get("error", "")
    if error:
        line += f"\n   Error: {error}"
    return line


def _build_verification_section(
    verification_results: list[dict[str, Any]],
) -> str:
    """Build the Verification section from VerificationResult statuses."""
    lines: list[str] = ["## Verification"]
    for result_status in verification_results:
        failure = result_status.get("failureReason", "")
        if failure:
            lines.append(f"\n**Failed:** {failure}")
            continue
        checks = result_status.get("checks", [])
        if checks:
            lines.append("\n**Checks:**")
            for check in checks:
                name = check.get("name", "")
                source = check.get("source", "")
                value = check.get("value", "")
                check_result = check.get("result", "")
                lines.append(f"- {name} ({source}): {value} → {check_result}")
        ver_summary = result_status.get("summary", "")
        if ver_summary:
            lines.append(f"**Summary:** {ver_summary}")
    return "\n".join(lines)


def _build_escalation_section(
    escalation_results: list[dict[str, Any]],
) -> str:
    """Build the Escalation section from EscalationResult statuses."""
    lines: list[str] = ["## Escalation"]
    for result_status in escalation_results:
        esc_summary = result_status.get("summary", "")
        if esc_summary:
            lines.append(f"\n**Summary:** {esc_summary}")
        content = result_status.get("content", "")
        if content:
            lines.append(f"\n{content}")
        failure = result_status.get("failureReason", "")
        if failure:
            lines.append(f"\n**Failed:** {failure}")
    return "\n".join(lines)


def _build_outcome_section(turn_data: TurnData) -> str:
    """Build the Outcome section from proposal_status conditions."""
    conditions = []
    if turn_data.proposal_status:
        conditions = turn_data.proposal_status.get("conditions", [])
    if not conditions:
        return "## Outcome\n\nNo conditions available"
    messages = [c.get("message", "") for c in conditions if c.get("message")]
    summary = "; ".join(messages) if messages else "No summary available"
    return f"## Outcome\n\n{summary}"

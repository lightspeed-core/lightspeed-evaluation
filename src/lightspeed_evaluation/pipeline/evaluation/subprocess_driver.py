"""Subprocess-based agent driver for CRD lifecycle management."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from enum import StrEnum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from lightspeed_evaluation.core.models import TurnData
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.pipeline.evaluation.driver import AgentDriver

logger = logging.getLogger(__name__)


class TerminalOutcome(StrEnum):
    """Terminal outcomes for a Proposal CR lifecycle.

    These are driver-level labels, not CRD API values. The CRD exposes
    conditions with statuses that the driver interprets:

    - Analyzed:  True = analysis succeeded, False = failed, Unknown = in progress
    - Executed:  True = execution succeeded, False = failed, Unknown = in progress
    - Verified:  True = verification passed, False = failed, Unknown = in progress
    - Denied:    True = user denied a step (terminal)
    - Escalated: True = escalation complete (terminal), False = failed, Unknown = in progress

    Special reason: RetryingExecution (Verified=False triggers retry, not failure).
    """

    COMPLETED = "Completed"
    FAILED = "Failed"
    DENIED = "Denied"
    ESCALATED = "Escalated"


CRD_GROUP = "agentic.openshift.io"
CRD_VERSION = "v1alpha1"
CRD_KIND = "Proposal"
CRD_PLURAL = "proposals"
CRD_API_VERSION = f"{CRD_GROUP}/{CRD_VERSION}"

CLI_COMMAND_TIMEOUT = 30


class SubprocessAgentConfig(BaseModel):
    """Configuration for a subprocess-based CRD agent."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["subprocess"] = "subprocess"
    namespace: str
    auto_approve: bool = True
    cleanup_proposals: bool = True
    timeout: int = Field(default=900, gt=0)
    poll_interval: int = Field(default=2, gt=0)


class SubprocessDriver(AgentDriver):
    """Driver that manages Proposal CR lifecycle via oc/kubectl CLI."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the subprocess driver."""
        super().__init__(config)
        self._config = SubprocessAgentConfig.model_validate(config)
        self._cli = self._resolve_cli()

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate subprocess driver configuration."""
        SubprocessAgentConfig.model_validate(config)
        if not shutil.which("oc") and not shutil.which("kubectl"):
            raise ConfigurationError("Neither 'oc' nor 'kubectl' found on PATH")

    def execute_turn(
        self, turn_data: TurnData, conversation_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute a Proposal CR lifecycle for a single turn."""
        # Proposal CR lifecycle:
        # 1. Build Proposal CR from TurnData fields
        # 2. Apply CR to cluster via oc/kubectl
        # 3. Poll status — read .status.conditions each interval
        # 4. Auto-approve — create ProposalApproval when Analyzed=True
        # 5. Terminal detection — break on completed/failed/denied/escalated
        # 6. Amend TurnData — set response and proposal_status in-place
        # 7. Cleanup — delete Proposal CR if cleanup_proposals enabled
        suffix = uuid.uuid4().hex[:8]
        cr_name = (
            f"eval-{conversation_id}-{suffix}" if conversation_id else f"eval-{suffix}"
        )
        proposal_spec = turn_data.proposal_spec or {}
        manifest = self._build_proposal_cr(turn_data, cr_name)

        result = self._apply(manifest)
        if result.returncode != 0:
            return (
                f"Failed to apply Proposal CR: {result.stderr.strip()}",
                None,
            )

        approved = False
        outcome: Optional[TerminalOutcome] = None
        status_dict: dict[str, Any] = {}
        start = time.monotonic()

        while time.monotonic() - start < self._config.timeout:
            time.sleep(self._config.poll_interval)
            status_dict, err = self._get_status(cr_name)
            if err:
                self._cleanup(cr_name)
                return (err, None)
            conditions = status_dict.get("conditions", [])

            if (
                self._config.auto_approve
                and not approved
                and self._should_approve(conditions)
            ):
                self._apply(self._build_approval_cr(cr_name, proposal_spec))
                approved = True

            outcome = self._is_terminal(conditions, proposal_spec)
            if outcome is not None:
                break
        else:
            self._cleanup(cr_name)
            return (
                f"Timeout after {self._config.timeout}s for '{cr_name}'",
                None,
            )

        turn_data.response = self._extract_summary(status_dict)
        turn_data.proposal_status = status_dict
        self._cleanup(cr_name)

        if outcome == TerminalOutcome.COMPLETED:
            return (None, None)
        return (
            f"Proposal '{cr_name}' terminated with outcome: {outcome}",
            None,
        )

    @staticmethod
    def _resolve_cli() -> str:
        """Resolve oc or kubectl binary path."""
        return shutil.which("oc") or shutil.which("kubectl") or ""

    def _run_cli(
        self,
        args: list[str],
        stdin: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a CLI command and return the result."""
        return subprocess.run(
            [self._cli, *args],
            input=stdin,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=CLI_COMMAND_TIMEOUT,
            check=False,
        )

    def _apply(self, manifest: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        """Apply a CR manifest via stdin."""
        return self._run_cli(["apply", "-f", "-"], stdin=json.dumps(manifest))

    def _get_status(self, cr_name: str) -> tuple[dict[str, Any], Optional[str]]:
        """Get Proposal CR status."""
        result = self._run_cli(
            [
                "get",
                CRD_PLURAL,
                cr_name,
                "-n",
                self._config.namespace,
                "-o",
                "json",
            ]
        )
        if result.returncode != 0:
            return {}, f"Failed to get status for '{cr_name}': {result.stderr.strip()}"
        try:
            cr = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return {}, f"Failed to parse status JSON for '{cr_name}': {exc}"
        return cr.get("status", {}), None

    def _delete(self, cr_name: str) -> None:
        """Delete a Proposal CR."""
        self._run_cli(
            [
                "delete",
                CRD_PLURAL,
                cr_name,
                "-n",
                self._config.namespace,
                "--ignore-not-found",
            ]
        )

    def _cleanup(self, cr_name: str) -> None:
        """Delete the Proposal CR if cleanup is enabled."""
        if not self._config.cleanup_proposals:
            return
        try:
            self._delete(cr_name)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to clean up Proposal CR '%s'", cr_name)

    def _build_proposal_cr(self, turn_data: TurnData, cr_name: str) -> dict[str, Any]:
        """Build Proposal CR manifest from TurnData."""
        spec: dict[str, Any] = {"request": turn_data.query}
        if turn_data.proposal_spec:
            spec.update(turn_data.proposal_spec)
        spec.setdefault("analysis", {})
        return {
            "apiVersion": CRD_API_VERSION,
            "kind": CRD_KIND,
            "metadata": {
                "name": cr_name,
                "namespace": self._config.namespace,
            },
            "spec": spec,
        }

    def _build_approval_cr(
        self, cr_name: str, proposal_spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Build ProposalApproval CR manifest."""
        stages: list[dict[str, Any]] = [
            {"type": "Analysis", "decision": "Approved"},
        ]
        if "execution" in proposal_spec:
            stages.append(
                {
                    "type": "Execution",
                    "decision": "Approved",
                    "execution": {"option": 0},
                }
            )
        if "verification" in proposal_spec:
            stages.append({"type": "Verification", "decision": "Approved"})
        return {
            "apiVersion": CRD_API_VERSION,
            "kind": "ProposalApproval",
            "metadata": {
                "name": cr_name,
                "namespace": self._config.namespace,
            },
            "spec": {"stages": stages},
        }

    @staticmethod
    def _should_approve(conditions: list[dict[str, Any]]) -> bool:
        """Check if conditions indicate the proposal is ready for approval."""
        by_type = {c["type"]: c for c in conditions}
        analyzed = by_type.get("Analyzed")
        return analyzed is not None and analyzed.get("status") == "True"

    @staticmethod
    def _is_terminal(
        conditions: list[dict[str, Any]], proposal_spec: dict[str, Any]
    ) -> Optional[TerminalOutcome]:
        """Check if conditions indicate a terminal state."""
        by_type = {c["type"]: c for c in conditions}
        if by_type.get("Denied", {}).get("status") == "True":
            return TerminalOutcome.DENIED
        if by_type.get("Escalated", {}).get("status") == "True":
            return TerminalOutcome.ESCALATED
        for c in conditions:
            if c.get("status") == "False" and c.get("reason") != "RetryingExecution":
                return TerminalOutcome.FAILED
        if "verification" in proposal_spec:
            last = "Verified"
        elif "execution" in proposal_spec:
            last = "Executed"
        else:
            last = "Analyzed"
        if by_type.get(last, {}).get("status") == "True":
            return TerminalOutcome.COMPLETED
        return None

    @staticmethod
    def _extract_summary(status_dict: dict[str, Any]) -> str:
        """Extract a human-readable summary from analysis results."""
        conditions = status_dict.get("conditions", [])
        messages = [c["message"] for c in conditions if c.get("message")]
        return "; ".join(messages) if messages else "No summary available"

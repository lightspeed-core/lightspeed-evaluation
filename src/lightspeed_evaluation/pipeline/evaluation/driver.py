"""Agent driver implementations for evaluation pipeline."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Optional, cast

from lightspeed_evaluation.core.api import APIClient
from lightspeed_evaluation.core.models import (
    APIConfig,
    HttpApiAgentConfig,
    ProposalAgentConfig,
    TurnData,
)
from lightspeed_evaluation.core.proposal import derive_phase
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
from lightspeed_evaluation.pipeline.evaluation.cli import KubeCLI
from lightspeed_evaluation.pipeline.evaluation.proposal_amender import (
    ProposalAmender,
)

logger = logging.getLogger(__name__)


class AgentDriver(ABC):
    """Abstract driver interface for agent execution."""

    def __init__(self, config: dict[str, Any], *, enabled: bool = True) -> None:
        """Initialize the driver with validated config."""
        self._enabled = enabled
        self._config = self.validate_config(config)

    @abstractmethod
    def execute_turn(
        self, turn_data: TurnData, conversation_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute a single turn and amend data in place.

        Returns:
            Tuple of (error_message, updated_conversation_id).
        """

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> Any:
        """Validate agent configuration and return driver-specific parsed config."""

    def close(self) -> None:
        """Release any resources held by the driver."""

    @property
    def enabled(self) -> bool:
        """Whether the driver should execute."""
        return self._enabled


class HttpApiDriver(AgentDriver):
    """Driver that enriches turn data via the HTTP API."""

    def __init__(self, config: dict[str, Any], *, enabled: bool = True) -> None:
        """Initialize the HTTP API driver with validated config."""
        super().__init__(config, enabled=enabled)
        self._api_client = (
            self._create_api_client(cast(HttpApiAgentConfig, self._config))
            if enabled
            else None
        )
        self._amender = APIDataAmender(self._api_client) if self._api_client else None

    def execute_turn(
        self, turn_data: TurnData, conversation_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute the HTTP API driver for a single turn.

        Returns:
            Tuple of (error_message, updated_conversation_id).
        """
        if not self._enabled or self._amender is None:
            return None, conversation_id
        return self._amender.amend_single_turn(turn_data, conversation_id)

    def validate_config(self, config: dict[str, Any]) -> HttpApiAgentConfig:
        """Validate HTTP API driver configuration."""
        return HttpApiAgentConfig.model_validate(config)

    def close(self) -> None:
        """Close the underlying API client."""
        if self._api_client:
            self._api_client.close()

    def _create_api_client(self, config: HttpApiAgentConfig) -> Optional[APIClient]:
        api_config = APIConfig.model_validate(config.model_dump(exclude={"type"}))
        return APIClient(api_config)


# ---------------------------------------------------------------------------
# Proposal driver — CRD-based agent lifecycle via oc/kubectl CLI
# ---------------------------------------------------------------------------


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


class ProposalDriver(AgentDriver):
    """Driver that manages Proposal CR lifecycle via oc/kubectl CLI."""

    def __init__(self, config: dict[str, Any], *, enabled: bool = True) -> None:
        """Initialize the proposal driver."""
        super().__init__(config, enabled=enabled)
        self._cli = self._resolve_cli()
        self._kube_cli = KubeCLI(
            cli_path=self._cli,
            namespace=self._config.namespace,
            timeout=self._config.cli_timeout,
        )
        self._amender = ProposalAmender(self._kube_cli)

    def validate_config(self, config: dict[str, Any]) -> ProposalAgentConfig:
        """Validate proposal driver configuration."""
        parsed = ProposalAgentConfig.model_validate(config)
        if self._enabled and not shutil.which("oc") and not shutil.which("kubectl"):
            raise ConfigurationError("Neither 'oc' nor 'kubectl' found on PATH")
        return parsed

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
        safe_id = (
            re.sub(r"[^a-z0-9-]", "", conversation_id.lower())[:50]
            if conversation_id
            else ""
        )
        cr_name = f"eval-{safe_id}-{suffix}" if safe_id else f"eval-{suffix}"
        proposal_spec = turn_data.proposal_spec or {}
        manifest = self._build_proposal_cr(turn_data, cr_name)

        result = self._apply(manifest)
        if result.returncode != 0:
            return (
                f"Failed to apply Proposal CR: {result.stderr.strip()}",
                None,
            )

        if self._config.auto_approve:
            err = self._approve_when_ready(cr_name, proposal_spec)
            if err:
                self._cleanup(cr_name)
                return (err, None)

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

            outcome = self._is_terminal(conditions, proposal_spec)
            if outcome is not None:
                break
        else:
            self._cleanup(cr_name)
            return (
                f"Timeout after {self._config.timeout}s for '{cr_name}'",
                None,
            )

        self._amend_turn_data(turn_data, status_dict)
        self._cleanup(cr_name)
        return self._outcome_to_result(outcome, cr_name)

    def _amend_turn_data(
        self, turn_data: TurnData, status_dict: dict[str, Any]
    ) -> None:
        """Amend turn data from proposal status, with fallback on amender failure."""
        amend_err = self._amender.amend(turn_data, status_dict)
        if amend_err:
            logger.warning("ProposalAmender failed: %s", amend_err)
            if not turn_data.response:
                turn_data.response = self._extract_summary(status_dict)
            if not turn_data.proposal_status:
                turn_data.proposal_status = status_dict

    # Failed/Escalated → error (pipeline marks remaining turns as ERROR,
    # metrics are NOT evaluated). Denied/Completed → no error (metrics run).
    _OUTCOME_ERRORS: dict[TerminalOutcome, str] = {
        TerminalOutcome.FAILED: "Proposal '{cr_name}' execution failed",
        TerminalOutcome.ESCALATED: (
            "Proposal '{cr_name}' escalated after verification failure"
        ),
    }

    @staticmethod
    def _outcome_to_result(
        outcome: Optional[TerminalOutcome], cr_name: str
    ) -> tuple[Optional[str], None]:
        """Map a terminal outcome to an (error_message, None) result tuple."""
        template = ProposalDriver._OUTCOME_ERRORS.get(outcome)  # type: ignore[arg-type]
        if template:
            return (template.format(cr_name=cr_name), None)
        if outcome != TerminalOutcome.COMPLETED:
            logger.warning(
                "Proposal '%s' terminated with outcome: %s", cr_name, outcome
            )
        return (None, None)

    @staticmethod
    def _resolve_cli() -> str:
        """Resolve oc or kubectl binary path."""
        return shutil.which("oc") or shutil.which("kubectl") or ""

    def _apply(self, manifest: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        """Apply a CR manifest via stdin."""
        return self._kube_cli.apply(manifest)

    def _get_status(self, cr_name: str) -> tuple[dict[str, Any], Optional[str]]:
        """Get Proposal CR status."""
        cr, err = self._kube_cli.get_resource(CRD_PLURAL, cr_name)
        if err:
            return {}, f"Failed to get status for '{cr_name}': {err}"
        return cr.get("status", {}), None

    def _delete(self, cr_name: str) -> None:
        """Delete a Proposal CR."""
        self._kube_cli.delete(CRD_PLURAL, cr_name)

    def _cleanup(self, cr_name: str) -> None:
        """Delete the Proposal CR if cleanup is enabled."""
        if not self._config.cleanup_proposals:
            return
        try:
            self._delete(cr_name)
            logger.info("Cleaned up Proposal CR '%s'", cr_name)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("Failed to clean up Proposal CR '%s': %s", cr_name, exc)

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
        analysis_params: dict[str, Any] = {}
        if "analysis" in proposal_spec and isinstance(proposal_spec["analysis"], dict):
            agent = proposal_spec["analysis"].get("agent")
            if agent:
                analysis_params["agent"] = agent
        if not analysis_params:
            analysis_params["agent"] = "default"

        stages: list[dict[str, Any]] = [
            {"type": "Analysis", "decision": "Approved", "analysis": analysis_params},
        ]
        if "execution" in proposal_spec:
            exec_params: dict[str, Any] = {"option": 0}
            if isinstance(proposal_spec["execution"], dict):
                agent = proposal_spec["execution"].get("agent")
                if agent:
                    exec_params["agent"] = agent
            stages.append(
                {
                    "type": "Execution",
                    "decision": "Approved",
                    "execution": exec_params,
                }
            )
        if "verification" in proposal_spec:
            verif_params: dict[str, Any] = {}
            if isinstance(proposal_spec["verification"], dict):
                agent = proposal_spec["verification"].get("agent")
                if agent:
                    verif_params["agent"] = agent
            if not verif_params:
                verif_params["agent"] = "default"
            stages.append(
                {
                    "type": "Verification",
                    "decision": "Approved",
                    "verification": verif_params,
                }
            )
        return {
            "apiVersion": CRD_API_VERSION,
            "kind": "ProposalApproval",
            "metadata": {
                "name": cr_name,
                "namespace": self._config.namespace,
            },
            "spec": {"stages": stages},
        }

    def _approve_when_ready(
        self, cr_name: str, proposal_spec: dict[str, Any]
    ) -> Optional[str]:
        """Wait for Proposal CR to exist on the cluster, then approve all stages."""
        start = time.monotonic()
        while time.monotonic() - start < self._config.timeout:
            _, err = self._get_status(cr_name)
            if err is None:
                break
            time.sleep(self._config.poll_interval)
        else:
            return f"Proposal '{cr_name}' not found within {self._config.timeout}s"

        approval = self._build_approval_cr(cr_name, proposal_spec)
        result = self._apply(approval)
        if result.returncode != 0:
            return f"Failed to apply ProposalApproval: {result.stderr.strip()}"
        return None

    _PHASE_TO_OUTCOME: dict[str, TerminalOutcome] = {
        "Completed": TerminalOutcome.COMPLETED,
        "Failed": TerminalOutcome.FAILED,
        "Denied": TerminalOutcome.DENIED,
        "Escalated": TerminalOutcome.ESCALATED,
    }

    @staticmethod
    def _is_terminal(
        conditions: list[dict[str, Any]], proposal_spec: dict[str, Any]
    ) -> Optional[TerminalOutcome]:
        """Check if conditions indicate a terminal state."""
        phase = derive_phase(conditions, proposal_spec or None)
        return ProposalDriver._PHASE_TO_OUTCOME.get(phase)

    @staticmethod
    def _extract_summary(status_dict: dict[str, Any]) -> str:
        """Extract a human-readable summary from analysis results.

        Degraded fallback: only called when ProposalAmender.amend() fails.
        Reads only condition messages, ignoring proposal_results and child
        Result CRs (analysis/execution/verification). The full rich summary
        is built by ProposalAmender.build_summary() in the happy path.
        """
        conditions = status_dict.get("conditions", [])
        messages = [c["message"] for c in conditions if c.get("message")]
        return "; ".join(messages) if messages else "No summary available"

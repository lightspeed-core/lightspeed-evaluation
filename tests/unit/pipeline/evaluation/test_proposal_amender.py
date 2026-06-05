"""Unit tests for ProposalAmender module."""

import subprocess
from typing import Any, Optional

from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import TurnData
from lightspeed_evaluation.pipeline.evaluation.cli import CLIClient, KubeCLI
from lightspeed_evaluation.pipeline.evaluation.proposal_amender import (
    ProposalAmender,
)


class MockCLI(CLIClient):
    """Mock CLIClient that returns pre-configured resources."""

    def __init__(self, resources: Optional[dict[str, dict[str, Any]]] = None) -> None:
        """Initialize with a map of resource name -> full CR dict."""
        super().__init__(timeout=30)
        self._resources: dict[str, dict[str, Any]] = resources or {}

    def run(
        self,
        args: list[str],
        stdin: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Not used in amender tests."""
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    def get_resource(
        self,
        resource_plural: str,
        name: str,
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Return pre-configured resource or error."""
        key = f"{resource_plural}/{name}"
        if key in self._resources:
            return self._resources[key], None
        return {}, f"not found: {key}"

    def apply(
        self,
        manifest: dict[str, Any],
    ) -> subprocess.CompletedProcess[str]:
        """Not used in amender tests."""
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def delete(self, resource_plural: str, name: str) -> None:
        """Not used in amender tests."""


def _make_turn(query: str = "Fix pod crash") -> TurnData:
    """Create a minimal TurnData for testing."""
    return TurnData(turn_id="t1", query=query)


def _get_results(turn: TurnData) -> dict[str, Any]:
    """Extract proposal_results, failing if None."""
    assert turn.proposal_results is not None
    return dict(turn.proposal_results)


def _get_response(turn: TurnData) -> str:
    """Extract response, failing if None."""
    assert turn.response is not None
    return str(turn.response)


DIAGNOSIS_STATUS: dict[str, Any] = {
    "conditions": [{"type": "Completed", "status": "True", "reason": "Succeeded"}],
    "options": [
        {
            "title": "Increase memory limit",
            "summary": "Bump memory to 512Mi",
            "diagnosis": {
                "summary": "OOMKilled due to low memory",
                "confidence": "High",
                "rootCause": "Memory limit 256Mi too low",
            },
            "proposal": {
                "description": "Patch deployment memory",
                "actions": [
                    {"type": "patch", "description": "Set memory limit to 512Mi"},
                ],
                "risk": "Low",
                "reversible": "Reversible",
                "estimatedImpact": "Brief pod restart",
            },
        },
        {
            "title": "Scale horizontally",
            "diagnosis": {
                "summary": "High load causing OOM",
                "confidence": "Medium",
                "rootCause": "Single replica under load",
            },
            "proposal": {
                "description": "Add replicas",
                "actions": [
                    {"type": "scale", "description": "Scale to 3 replicas"},
                ],
                "risk": "Low",
                "reversible": "Reversible",
                "estimatedImpact": "No downtime",
            },
        },
    ],
}

EXECUTION_STATUS: dict[str, Any] = {
    "conditions": [{"type": "Completed", "status": "True", "reason": "Succeeded"}],
    "actionsTaken": [
        {
            "type": "patch",
            "description": "Set memory limit to 512Mi",
            "outcome": "Succeeded",
            "output": "deployment.apps/web patched",
        },
    ],
    "verification": {
        "conditionOutcome": "Improved",
        "summary": "Pod running with new limits",
    },
}

VERIFICATION_STATUS: dict[str, Any] = {
    "conditions": [{"type": "Completed", "status": "True", "reason": "Succeeded"}],
    "checks": [
        {
            "name": "pod-running",
            "source": "oc",
            "value": "Running",
            "result": "Passed",
        },
    ],
    "summary": "All checks passed",
}


class TestAmendAnalysisOnly:
    """Test ProposalAmender with analysis-only workflow."""

    def test_populates_proposal_results(self) -> None:
        """Test amend populates proposal_results with analysis data."""
        cli = MockCLI({"analysisresults/ar-1": {"status": DIAGNOSIS_STATUS}})
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [{"type": "Analyzed", "status": "True", "message": "Done"}],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-1", "outcome": "Succeeded"}],
                },
            },
        }

        err = amender.amend(turn, status)

        assert err is None
        assert turn.proposal_status == status
        results = _get_results(turn)
        assert "analysis" in results
        assert len(results["analysis"]) == 1
        assert results["analysis"][0]["options"][0]["title"] == "Increase memory limit"

    def test_response_contains_diagnosis(self) -> None:
        """Test Markdown summary includes diagnosis details."""
        cli = MockCLI({"analysisresults/ar-1": {"status": DIAGNOSIS_STATUS}})
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [{"type": "Analyzed", "status": "True", "message": "Done"}],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-1", "outcome": "Succeeded"}],
                },
            },
        }

        amender.amend(turn, status)

        response = _get_response(turn)
        assert "OOMKilled due to low memory" in response
        assert "Memory limit 256Mi too low" in response
        assert "Confidence: High" in response

    def test_option_zero_marked_approved(self) -> None:
        """Test option 0 is marked as (Approved) in summary."""
        cli = MockCLI({"analysisresults/ar-1": {"status": DIAGNOSIS_STATUS}})
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-1", "outcome": "Succeeded"}],
                },
            },
        }

        amender.amend(turn, status)

        response = _get_response(turn)
        assert "(Approved)" in response
        assert "### Option 0 (Approved): Increase memory limit" in response
        assert "### Option 1 : Scale horizontally" in response

    def test_no_execution_or_verification_in_results(self) -> None:
        """Test analysis-only workflow has no execution/verification keys."""
        cli = MockCLI({"analysisresults/ar-1": {"status": DIAGNOSIS_STATUS}})
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-1", "outcome": "Succeeded"}],
                },
            },
        }

        amender.amend(turn, status)

        results = _get_results(turn)
        assert "execution" not in results
        assert "verification" not in results

    def test_proposal_phases_analysis_only(self) -> None:
        """Test proposal_phases contains only 'analysis' for analysis-only workflow."""
        cli = MockCLI({"analysisresults/ar-1": {"status": DIAGNOSIS_STATUS}})
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-1", "outcome": "Succeeded"}],
                },
            },
        }

        amender.amend(turn, status)

        assert turn.proposal_phases == ["analysis"]


class TestAmendFullPipeline:
    """Test ProposalAmender with analysis + execution + verification."""

    def _make_cli(self) -> MockCLI:
        """Create MockCLI with all three result CRs."""
        return MockCLI(
            {
                "analysisresults/ar-1": {"status": DIAGNOSIS_STATUS},
                "executionresults/er-1": {"status": EXECUTION_STATUS},
                "verificationresults/vr-1": {"status": VERIFICATION_STATUS},
            }
        )

    def _make_status(self) -> dict[str, Any]:
        """Create proposal status with all three steps."""
        return {
            "conditions": [
                {"type": "Analyzed", "status": "True"},
                {"type": "Executed", "status": "True"},
                {"type": "Verified", "status": "True", "message": "All passed"},
            ],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-1", "outcome": "Succeeded"}],
                },
                "execution": {
                    "results": [{"name": "er-1", "outcome": "Succeeded"}],
                },
                "verification": {
                    "results": [{"name": "vr-1", "outcome": "Succeeded"}],
                },
            },
        }

    def test_all_results_populated(self) -> None:
        """Test all three step results are populated."""
        amender = ProposalAmender(self._make_cli())
        turn = _make_turn()

        amender.amend(turn, self._make_status())

        results = _get_results(turn)
        assert "analysis" in results
        assert "execution" in results
        assert "verification" in results
        assert len(results["analysis"]) == 1
        assert len(results["execution"]) == 1
        assert len(results["verification"]) == 1

    def test_execution_in_summary(self) -> None:
        """Test Markdown includes execution actions."""
        amender = ProposalAmender(self._make_cli())
        turn = _make_turn()

        amender.amend(turn, self._make_status())

        response = _get_response(turn)
        assert "## Execution" in response
        assert "Set memory limit to 512Mi" in response
        assert "Succeeded" in response
        assert "deployment.apps/web patched" in response

    def test_verification_in_summary(self) -> None:
        """Test Markdown includes verification checks."""
        amender = ProposalAmender(self._make_cli())
        turn = _make_turn()

        amender.amend(turn, self._make_status())

        response = _get_response(turn)
        assert "## Verification" in response
        assert "pod-running" in response
        assert "Passed" in response
        assert "All checks passed" in response

    def test_proposal_phases_full_pipeline(self) -> None:
        """Test proposal_phases lists all three phases."""
        amender = ProposalAmender(self._make_cli())
        turn = _make_turn()

        amender.amend(turn, self._make_status())

        assert turn.proposal_phases == ["analysis", "execution", "verification"]


class TestAmendEdgeCases:
    """Test ProposalAmender edge cases."""

    def test_empty_steps(self) -> None:
        """Test status with no steps produces empty results."""
        cli = MockCLI()
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {"conditions": []}

        amender.amend(turn, status)

        assert not turn.proposal_results
        assert not turn.proposal_phases
        response = _get_response(turn)
        assert "## Request" in response

    def test_step_with_no_results(self) -> None:
        """Test step present but no result refs gives empty list."""
        cli = MockCLI()
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [],
            "steps": {
                "analysis": {"results": []},
            },
        }

        amender.amend(turn, status)

        results = _get_results(turn)
        assert results["analysis"] == []

    def test_failed_fetch_logged_and_skipped(self, mocker: MockerFixture) -> None:
        """Test failed CR fetch is logged and result is skipped."""
        cli = MockCLI()
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [],
            "steps": {
                "analysis": {
                    "results": [{"name": "missing-cr", "outcome": "Succeeded"}],
                },
            },
        }
        mock_logger = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.proposal_amender.logger"
        )

        err = amender.amend(turn, status)

        assert err is None
        results = _get_results(turn)
        assert results["analysis"] == []
        mock_logger.warning.assert_called_once()

    def test_analysis_with_failure_reason(self) -> None:
        """Test analysis result with failureReason."""
        failed_status: dict[str, Any] = {
            "conditions": [{"type": "Completed", "status": "False"}],
            "failureReason": "LLM timeout",
        }
        cli = MockCLI({"analysisresults/ar-fail": {"status": failed_status}})
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [{"type": "Analyzed", "status": "False"}],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-fail", "outcome": "Failed"}],
                },
            },
        }

        amender.amend(turn, status)

        response = _get_response(turn)
        assert "LLM timeout" in response

    def test_escalation_results(self) -> None:
        """Test escalation step results are captured."""
        esc_status: dict[str, Any] = {
            "conditions": [],
            "summary": "Requires manual intervention",
            "content": "Cluster admin must review",
        }
        cli = MockCLI({"escalationresults/esc-1": {"status": esc_status}})
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [{"type": "Escalated", "status": "True"}],
            "steps": {
                "escalation": {
                    "results": [{"name": "esc-1", "outcome": "Succeeded"}],
                },
            },
        }

        amender.amend(turn, status)

        results = _get_results(turn)
        assert "escalation" in results
        response = _get_response(turn)
        assert "Requires manual intervention" in response
        assert "Cluster admin must review" in response

    def test_outcome_section_from_conditions(self) -> None:
        """Test outcome section uses condition messages."""
        cli = MockCLI()
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [
                {"type": "Analyzed", "status": "True", "message": "Analysis complete"},
                {"type": "Executed", "status": "True", "message": "Execution done"},
            ],
        }

        amender.amend(turn, status)

        response = _get_response(turn)
        assert "## Outcome" in response
        assert "Analysis complete; Execution done" in response

    def test_subprocess_error_caught_by_amend(self, mocker: MockerFixture) -> None:
        """Test that subprocess errors are caught by the broadened except clause."""
        cli = MockCLI()
        amender = ProposalAmender(cli)
        turn = _make_turn()
        status: dict[str, Any] = {
            "conditions": [],
            "steps": {
                "analysis": {
                    "results": [{"name": "ar-1", "outcome": "Succeeded"}],
                },
            },
        }
        mocker.patch.object(
            cli,
            "get_resource",
            side_effect=subprocess.SubprocessError("unexpected"),
        )

        err = amender.amend(turn, status)

        assert err is not None
        assert "ProposalAmender error" in err


class TestKubeCLITimeoutHandling:
    """Test KubeCLI normalizes subprocess.TimeoutExpired to CompletedProcess."""

    def test_run_returns_completed_process_on_timeout(
        self, mocker: MockerFixture
    ) -> None:
        """Test that TimeoutExpired is caught and a failing CompletedProcess returned."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="oc get pods", timeout=30),
        )

        cli = KubeCLI(cli_path="oc", namespace="test-ns", timeout=30)
        result = cli.run(["get", "pods"])

        assert result.returncode == 1
        assert result.stdout == ""
        assert "timed out after 30s" in result.stderr

    def test_get_resource_returns_error_on_timeout(self, mocker: MockerFixture) -> None:
        """Test that get_resource returns an error tuple when run times out."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="oc get pods", timeout=30),
        )

        cli = KubeCLI(cli_path="oc", namespace="test-ns", timeout=30)
        resource, err = cli.get_resource("pods", "my-pod")

        assert resource == {}
        assert err is not None
        assert "timed out" in err

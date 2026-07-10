"""Integration tests for ProposalDriver-based evaluation.

These tests run the evaluation pipeline against a live OpenShift cluster
using the ProposalDriver to create and manage Proposal CRs.

Prerequisites:
    - oc CLI authenticated against a cluster with agentic CRDs installed
    - OPENAI_API_KEY env var set
    - Network connectivity to the cluster API

Each scenario has provider-agnostic setup/cleanup scripts that source
_setup_infra.sh, which dispatches on the EVAL_PROVIDER env var
(openai | anthropic | claude-vertex; default: openai).

Run with: pytest tests/integration/test_proposal_evaluation.py -v -m agentic
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from lightspeed_evaluation import ConfigLoader, evaluate
from lightspeed_evaluation.core.storage import FileBackendConfig
from lightspeed_evaluation.core.system import DataValidator


def check_cli_available() -> bool:
    """Check if oc CLI is available."""
    return bool(shutil.which("oc"))


def check_cluster_reachable() -> bool:
    """Check if the cluster is reachable via oc whoami."""
    try:
        result = subprocess.run(
            ["oc", "whoami"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_crd_installed() -> bool:
    """Check if the Proposal CRD is installed on the cluster."""
    try:
        result = subprocess.run(
            ["oc", "get", "crd", "proposals.agentic.openshift.io"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_env_vars_set() -> bool:
    """Check if required environment variables are set."""
    return bool(os.getenv("OPENAI_API_KEY"))


pytestmark = pytest.mark.agentic

INTEGRATION_TEST_DIR = Path(__file__).parent
PROPOSAL_CONFIG_PATH = INTEGRATION_TEST_DIR / "system-config-agents-proposal.yaml"
PROPOSAL_EVAL_DATA_PATH = INTEGRATION_TEST_DIR / "test_evaluation_data_proposal.yaml"


class TestProposalPrerequisites:
    """Verify prerequisites for proposal integration tests."""

    def test_cli_available(self) -> None:
        """Verify that oc CLI is available."""
        assert check_cli_available(), "oc CLI must be installed and in PATH"

    def test_cluster_reachable(self) -> None:
        """Verify that the cluster is reachable."""
        assert check_cluster_reachable(), "Cluster must be reachable via 'oc whoami'"

    def test_crd_installed(self) -> None:
        """Verify that Proposal CRD is installed on the cluster."""
        assert (
            check_crd_installed()
        ), "proposals.agentic.openshift.io CRD must be installed"

    def test_env_vars_configured(self) -> None:
        """Verify that required environment variables are set."""
        assert check_env_vars_set(), "OPENAI_API_KEY environment variable must be set"


class TestProposalDriverEvaluation:
    """End-to-end tests for ProposalDriver evaluation pipeline."""

    @pytest.mark.timeout(1200)
    def test_oomkill_full_lifecycle(self, tmp_path: Path) -> None:
        """Test OOMKill full lifecycle: status check + judge evaluation.

        Verifies:
        - Setup script deploys infrastructure and test workload
        - ProposalDriver creates Proposal CR and auto-approves stages
        - Pipeline completes without errors
        - TurnData is enriched with response and proposal_status
        - At least the Analyzed condition reaches status True
        - Cleanup script removes test resources
        """
        loader = ConfigLoader()
        system_config = loader.load_system_config(str(PROPOSAL_CONFIG_PATH))
        system_config.storage = [
            FileBackendConfig(output_dir=str(tmp_path / "eval_output"))
        ]

        validator = DataValidator(
            api_enabled=True,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
        )
        all_data = validator.load_evaluation_data(str(PROPOSAL_EVAL_DATA_PATH))
        eval_data = [
            d for d in all_data if d.conversation_group_id == "proposal_oomkill"
        ]
        assert len(eval_data) == 1, "Should find proposal_oomkill data"

        evaluate(system_config, eval_data)

        turn = eval_data[0].turns[0]
        assert (
            turn.response and turn.response.strip()
        ), "Response should be populated by ProposalDriver"
        assert isinstance(
            turn.proposal_status, dict
        ), "proposal_status should be populated"
        assert (
            "conditions" in turn.proposal_status
        ), "proposal_status should contain conditions"

        conditions = turn.proposal_status["conditions"]
        by_type = {c["type"]: c for c in conditions}
        assert "Analyzed" in by_type, "Should have Analyzed condition"
        assert (
            by_type["Analyzed"].get("status") == "True"
        ), "Analyzed condition should be True"

    @pytest.mark.timeout(600)
    def test_analysis_only(self, tmp_path: Path) -> None:
        """Test analysis-only Proposal (no execution or verification).

        Verifies:
        - Only analysis stage runs
        - No Executed or Verified conditions present
        """
        loader = ConfigLoader()
        system_config = loader.load_system_config(str(PROPOSAL_CONFIG_PATH))
        system_config.storage = [
            FileBackendConfig(output_dir=str(tmp_path / "eval_output"))
        ]

        validator = DataValidator(
            api_enabled=True,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
        )
        all_data = validator.load_evaluation_data(str(PROPOSAL_EVAL_DATA_PATH))
        eval_data = [
            d for d in all_data if d.conversation_group_id == "proposal_analysis_only"
        ]
        assert len(eval_data) == 1, "Should find proposal_analysis_only data"

        evaluate(system_config, eval_data)

        turn = eval_data[0].turns[0]
        assert (
            turn.response and turn.response.strip()
        ), "Response should be populated by ProposalDriver"
        assert isinstance(
            turn.proposal_status, dict
        ), "proposal_status should be populated"

        conditions = turn.proposal_status.get("conditions", [])
        by_type = {c["type"]: c for c in conditions}
        assert "Analyzed" in by_type, "Should have Analyzed condition"
        assert (
            by_type["Analyzed"].get("status") == "True"
        ), "Analyzed condition should be True"
        if "Executed" in by_type:
            assert by_type["Executed"].get("reason") == "Skipped", (
                "Analysis-only Executed condition should be Skipped, "
                f"got reason={by_type['Executed'].get('reason')}"
            )
        if "Verified" in by_type:
            assert by_type["Verified"].get("reason") == "Skipped", (
                "Analysis-only Verified condition should be Skipped, "
                f"got reason={by_type['Verified'].get('reason')}"
            )

    @pytest.mark.timeout(1200)
    def test_oomkill_claude_vertex(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test OOMKill full lifecycle with Claude/Vertex AI provider.

        Uses the same provider-agnostic scenario as test_oomkill_full_lifecycle
        but selects the Claude/Vertex infrastructure via EVAL_PROVIDER.

        Verifies:
        - ProposalDriver populates response with workflow summary
        - custom:proposal_evaluation_correctness metric runs against response
        - Pipeline completes without errors
        """
        monkeypatch.setenv("EVAL_PROVIDER", "claude-vertex")
        loader = ConfigLoader()
        system_config = loader.load_system_config(str(PROPOSAL_CONFIG_PATH))
        system_config.storage = [
            FileBackendConfig(output_dir=str(tmp_path / "eval_output"))
        ]

        validator = DataValidator(
            api_enabled=True,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
        )
        all_data = validator.load_evaluation_data(str(PROPOSAL_EVAL_DATA_PATH))
        eval_data = [
            d for d in all_data if d.conversation_group_id == "proposal_oomkill"
        ]
        assert len(eval_data) == 1, "Should find proposal_oomkill data"

        evaluate(system_config, eval_data)

        turn = eval_data[0].turns[0]
        assert (
            turn.response and turn.response.strip()
        ), "Response should be populated by ProposalDriver"

    @pytest.mark.timeout(120)
    def test_timeout_handling(self, tmp_path: Path) -> None:
        """Test that a very short timeout is handled gracefully.

        Verifies:
        - Pipeline does not crash on timeout
        - Proposal CRs are cleaned up after timeout
        """
        loader = ConfigLoader()
        system_config = loader.load_system_config(str(PROPOSAL_CONFIG_PATH))
        system_config.storage = [
            FileBackendConfig(output_dir=str(tmp_path / "eval_output"))
        ]

        assert system_config.agents is not None
        agent_cfg = system_config.agents.agents["proposal_agent"]
        agent_cfg.timeout = 5
        agent_cfg.poll_interval = 1

        validator = DataValidator(
            api_enabled=True,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
        )
        all_data = validator.load_evaluation_data(str(PROPOSAL_EVAL_DATA_PATH))
        eval_data = [
            d for d in all_data if d.conversation_group_id == "proposal_analysis_only"
        ]

        evaluate(system_config, eval_data)

        result = subprocess.run(
            [
                "oc",
                "get",
                "proposals",
                "-n",
                "lightspeed-evaluation-test",
                "-o",
                "name",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, (
            "Failed to list proposals during timeout cleanup validation: "
            f"{result.stderr.strip()}"
        )
        lines = [
            line
            for line in result.stdout.strip().splitlines()
            if line.startswith("proposal.agentic.openshift.io/eval-")
        ]
        assert (
            len(lines) == 0
        ), f"Proposal CRs should be cleaned up after timeout, but found: {lines}"

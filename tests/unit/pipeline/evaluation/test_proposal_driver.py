# pylint: disable=protected-access

"""Unit tests for proposal agent driver module."""

from typing import Any

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import ProposalAgentConfig, TurnData
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.pipeline.evaluation.driver import (
    ProposalDriver,
    TerminalOutcome,
)

MODULE = "lightspeed_evaluation.pipeline.evaluation.driver"

VALID_CONFIG: dict[str, Any] = {
    "type": "proposal",
    "namespace": "openshift-lightspeed",
}

SPEC_ANALYSIS_ONLY: dict[str, Any] = {"analysis": {}}
SPEC_WITH_EXEC: dict[str, Any] = {"analysis": {}, "execution": {}}
SPEC_FULL: dict[str, Any] = {"analysis": {}, "execution": {}, "verification": {}}


def _cond(
    cond_type: str,
    status: str,
    reason: str = "",
    message: str = "",
) -> dict[str, Any]:
    """Build a CRD condition dict for tests."""
    c: dict[str, Any] = {"type": cond_type, "status": status}
    if reason:
        c["reason"] = reason
    if message:
        c["message"] = message
    return c


# ── Config validation ────────────────────────────────────────────────


class TestProposalAgentConfig:
    """Unit tests for ProposalAgentConfig Pydantic model."""

    def test_valid_config_all_fields(self) -> None:
        """Test config with all fields explicit."""
        config = ProposalAgentConfig.model_validate(
            {
                "type": "proposal",
                "namespace": "ns",
                "auto_approve": False,
                "cleanup_proposals": False,
                "timeout": 60,
                "cli_timeout": 15,
                "poll_interval": 5,
            }
        )
        assert config.namespace == "ns"
        assert config.auto_approve is False
        assert config.cleanup_proposals is False
        assert config.timeout == 60
        assert config.cli_timeout == 15
        assert config.poll_interval == 5

    def test_valid_config_defaults(self) -> None:
        """Test config with only required fields uses defaults."""
        config = ProposalAgentConfig.model_validate(VALID_CONFIG)
        assert config.auto_approve is True
        assert config.cleanup_proposals is True
        assert config.timeout == 900
        assert config.cli_timeout == 30
        assert config.poll_interval == 2

    def test_missing_namespace(self) -> None:
        """Test missing required namespace raises ValidationError."""
        with pytest.raises(ValidationError):
            ProposalAgentConfig.model_validate({"type": "proposal"})

    def test_extra_field_rejected(self) -> None:
        """Test extra fields raise ValidationError."""
        with pytest.raises(ValidationError):
            ProposalAgentConfig.model_validate({**VALID_CONFIG, "extra": "bad"})

    def test_invalid_timeout_zero(self) -> None:
        """Test timeout=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            ProposalAgentConfig.model_validate({**VALID_CONFIG, "timeout": 0})

    def test_invalid_timeout_negative(self) -> None:
        """Test negative timeout raises ValidationError."""
        with pytest.raises(ValidationError):
            ProposalAgentConfig.model_validate({**VALID_CONFIG, "timeout": -1})

    def test_invalid_cli_timeout_zero(self) -> None:
        """Test cli_timeout=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            ProposalAgentConfig.model_validate({**VALID_CONFIG, "cli_timeout": 0})

    def test_invalid_cli_timeout_negative(self) -> None:
        """Test negative cli_timeout raises ValidationError."""
        with pytest.raises(ValidationError):
            ProposalAgentConfig.model_validate({**VALID_CONFIG, "cli_timeout": -1})

    def test_invalid_poll_interval_zero(self) -> None:
        """Test poll_interval=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            ProposalAgentConfig.model_validate({**VALID_CONFIG, "poll_interval": 0})


# ── Condition helpers ────────────────────────────────────────────────


class TestIsTerminal:  # pylint: disable=too-few-public-methods
    """Unit tests for ProposalDriver._is_terminal."""

    @pytest.mark.parametrize(
        "conditions, spec, expected",
        [
            # Not terminal — keep polling
            ([], SPEC_FULL, None),
            ([_cond("Analyzed", "Unknown")], SPEC_FULL, None),
            ([_cond("Analyzed", "True")], SPEC_FULL, None),
            (
                [_cond("Analyzed", "True"), _cond("Executed", "Unknown")],
                SPEC_FULL,
                None,
            ),
            (
                [_cond("Analyzed", "True"), _cond("Executed", "True")],
                SPEC_FULL,
                None,
            ),
            (
                [
                    _cond("Analyzed", "True"),
                    _cond("Executed", "True"),
                    _cond("Verified", "Unknown"),
                ],
                SPEC_FULL,
                None,
            ),
            # RetryingExecution — not a failure
            (
                [
                    _cond("Analyzed", "True"),
                    _cond("Executed", "True"),
                    _cond("Verified", "False", "RetryingExecution"),
                ],
                SPEC_FULL,
                None,
            ),
            # Completed — last expected step True
            (
                [_cond("Analyzed", "True")],
                SPEC_ANALYSIS_ONLY,
                TerminalOutcome.COMPLETED,
            ),
            (
                [_cond("Analyzed", "True"), _cond("Executed", "True")],
                SPEC_WITH_EXEC,
                TerminalOutcome.COMPLETED,
            ),
            (
                [
                    _cond("Analyzed", "True"),
                    _cond("Executed", "True"),
                    _cond("Verified", "True"),
                ],
                SPEC_FULL,
                TerminalOutcome.COMPLETED,
            ),
            # Failed — any condition False (no RetryingExecution)
            (
                [_cond("Analyzed", "False")],
                SPEC_FULL,
                TerminalOutcome.FAILED,
            ),
            (
                [_cond("Analyzed", "True"), _cond("Executed", "False")],
                SPEC_FULL,
                TerminalOutcome.FAILED,
            ),
            (
                [
                    _cond("Analyzed", "True"),
                    _cond("Executed", "True"),
                    _cond("Verified", "False"),
                ],
                SPEC_FULL,
                TerminalOutcome.FAILED,
            ),
            # Denied — priority over other conditions
            ([_cond("Denied", "True")], SPEC_FULL, TerminalOutcome.DENIED),
            (
                [_cond("Denied", "True"), _cond("Analyzed", "True")],
                SPEC_FULL,
                TerminalOutcome.DENIED,
            ),
            # Escalated — priority over other conditions
            (
                [_cond("Escalated", "True")],
                SPEC_FULL,
                TerminalOutcome.ESCALATED,
            ),
            (
                [_cond("Escalated", "True"), _cond("Verified", "True")],
                SPEC_FULL,
                TerminalOutcome.ESCALATED,
            ),
        ],
        ids=[
            "no-conditions",
            "analyzing",
            "analyzed-not-terminal-full",
            "executing",
            "executed-not-terminal-full",
            "verifying",
            "retrying-execution",
            "completed-analysis-only",
            "completed-with-exec",
            "completed-full",
            "failed-analysis",
            "failed-execution",
            "failed-verification",
            "denied",
            "denied-beats-analyzed",
            "escalated",
            "escalated-beats-verified",
        ],
    )
    def test_is_terminal(
        self,
        conditions: list[dict[str, Any]],
        spec: dict[str, Any],
        expected: TerminalOutcome | None,
    ) -> None:
        """Test _is_terminal returns correct terminal outcome."""
        assert ProposalDriver._is_terminal(conditions, spec) == expected


# ── CR manifest building ────────────────────────────────────────────


class TestBuildCR:
    """Unit tests for CR manifest construction."""

    def _make_driver(self, mocker: MockerFixture) -> ProposalDriver:
        """Create a driver with mocked CLI resolution."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        return ProposalDriver(VALID_CONFIG)

    def test_proposal_cr_query_only(self, mocker: MockerFixture) -> None:
        """Test Proposal CR with query only, no proposal_spec."""
        driver = self._make_driver(mocker)
        turn = TurnData(turn_id="t1", query="Pod is crash looping")
        cr = driver._build_proposal_cr(turn, "eval-abc123")

        assert cr["apiVersion"] == "agentic.openshift.io/v1alpha1"
        assert cr["kind"] == "Proposal"
        assert cr["metadata"]["name"] == "eval-abc123"
        assert cr["metadata"]["namespace"] == "openshift-lightspeed"
        assert cr["spec"]["request"] == "Pod is crash looping"
        assert cr["spec"]["analysis"] == {}

    def test_proposal_cr_with_spec(self, mocker: MockerFixture) -> None:
        """Test Proposal CR with full proposal_spec."""
        driver = self._make_driver(mocker)
        turn = TurnData(
            turn_id="t1",
            query="Pod is crash looping",
            proposal_spec={
                "targetNamespaces": ["production"],
                "analysis": {"agent": "smart"},
                "execution": {"agent": "default"},
            },
        )
        cr = driver._build_proposal_cr(turn, "eval-abc123")

        assert cr["spec"]["request"] == "Pod is crash looping"
        assert cr["spec"]["targetNamespaces"] == ["production"]
        assert cr["spec"]["analysis"] == {"agent": "smart"}
        assert cr["spec"]["execution"] == {"agent": "default"}

    def test_proposal_cr_defaults_analysis(self, mocker: MockerFixture) -> None:
        """Test Proposal CR defaults analysis to empty dict."""
        driver = self._make_driver(mocker)
        turn = TurnData(
            turn_id="t1",
            query="Q",
            proposal_spec={"execution": {}},
        )
        cr = driver._build_proposal_cr(turn, "eval-x")

        assert cr["spec"]["analysis"] == {}

    def test_approval_cr_analysis_only(self, mocker: MockerFixture) -> None:
        """Test ProposalApproval with analysis-only spec."""
        driver = self._make_driver(mocker)
        cr = driver._build_approval_cr("eval-abc", SPEC_ANALYSIS_ONLY)

        assert cr["kind"] == "ProposalApproval"
        assert len(cr["spec"]["stages"]) == 1
        assert cr["spec"]["stages"][0]["type"] == "Analysis"
        assert cr["spec"]["stages"][0]["decision"] == "Approved"
        assert cr["spec"]["stages"][0]["analysis"] == {"agent": "default"}

    def test_approval_cr_full(self, mocker: MockerFixture) -> None:
        """Test ProposalApproval with all three stages."""
        driver = self._make_driver(mocker)
        cr = driver._build_approval_cr("eval-abc", SPEC_FULL)

        assert len(cr["spec"]["stages"]) == 3
        types = [s["type"] for s in cr["spec"]["stages"]]
        assert types == ["Analysis", "Execution", "Verification"]
        assert cr["spec"]["stages"][0]["analysis"] == {"agent": "default"}
        assert cr["spec"]["stages"][1]["execution"] == {"option": 0}
        assert cr["spec"]["stages"][2]["verification"] == {"agent": "default"}

    def test_approval_cr_with_agent_refs(self, mocker: MockerFixture) -> None:
        """Test ProposalApproval passes agent names from proposal_spec."""
        driver = self._make_driver(mocker)
        spec: dict[str, Any] = {
            "analysis": {"agent": "eval-default"},
            "execution": {"agent": "eval-default"},
            "verification": {"agent": "eval-default"},
        }
        cr = driver._build_approval_cr("eval-abc", spec)

        assert cr["spec"]["stages"][0]["analysis"] == {"agent": "eval-default"}
        assert cr["spec"]["stages"][1]["execution"]["agent"] == "eval-default"
        assert cr["spec"]["stages"][2]["verification"] == {"agent": "eval-default"}


# ── Extract summary ─────────────────────────────────────────────────


class TestExtractSummary:
    """Unit tests for ProposalDriver._extract_summary."""

    def test_with_messages(self) -> None:
        """Test summary from condition messages."""
        status = {
            "conditions": [
                _cond("Analyzed", "True", message="Analysis done"),
                _cond("Executed", "True", message="Execution ok"),
            ]
        }
        result = ProposalDriver._extract_summary(status)
        assert result == "Analysis done; Execution ok"

    def test_no_messages(self) -> None:
        """Test fallback when conditions have no messages."""
        status = {"conditions": [_cond("Analyzed", "True")]}
        assert ProposalDriver._extract_summary(status) == "No summary available"

    def test_empty_status(self) -> None:
        """Test fallback for empty status dict."""
        assert ProposalDriver._extract_summary({}) == "No summary available"


# ── Driver lifecycle ─────────────────────────────────────────────────


class TestProposalDriver:
    """Unit tests for ProposalDriver init, validate_config, enabled, close."""

    def test_validate_config_with_oc(self, mocker: MockerFixture) -> None:
        """Test driver resolves oc as primary CLI."""
        mock_shutil = mocker.patch(f"{MODULE}.shutil")
        mock_shutil.which.side_effect = lambda cmd: (
            "/usr/bin/oc" if cmd == "oc" else None
        )
        driver = ProposalDriver(VALID_CONFIG)
        assert driver._cli == "/usr/bin/oc"

    def test_validate_config_kubectl_fallback(self, mocker: MockerFixture) -> None:
        """Test driver falls back to kubectl when oc not found."""
        mock_shutil = mocker.patch(f"{MODULE}.shutil")
        mock_shutil.which.side_effect = lambda cmd: (
            "/usr/bin/kubectl" if cmd == "kubectl" else None
        )
        driver = ProposalDriver(VALID_CONFIG)
        assert driver._cli == "/usr/bin/kubectl"

    def test_validate_config_neither_found(self, mocker: MockerFixture) -> None:
        """Test ConfigurationError when neither oc nor kubectl found."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = None
        with pytest.raises(ConfigurationError, match="Neither 'oc' nor 'kubectl'"):
            ProposalDriver(VALID_CONFIG)

    def test_validate_config_invalid(self, mocker: MockerFixture) -> None:
        """Test ValidationError for missing required fields."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        with pytest.raises(ValidationError):
            ProposalDriver({"type": "proposal"})

    def test_enabled_always_true(self, mocker: MockerFixture) -> None:
        """Test enabled property defaults to True (inherited)."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        driver = ProposalDriver(VALID_CONFIG)
        assert driver.enabled is True

    def test_close_is_noop(self, mocker: MockerFixture) -> None:
        """Test close does nothing (no persistent connections)."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        driver = ProposalDriver(VALID_CONFIG)
        driver.close()


# ── Cleanup ──────────────────────────────────────────────────────────


class TestCleanup:
    """Unit tests for ProposalDriver._cleanup."""

    def test_cleanup_calls_delete(self, mocker: MockerFixture) -> None:
        """Test cleanup delegates to _delete when enabled."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        driver = ProposalDriver(VALID_CONFIG)
        mock_delete = mocker.patch.object(driver, "_delete")

        driver._cleanup("eval-test")

        mock_delete.assert_called_once_with("eval-test")

    def test_cleanup_disabled(self, mocker: MockerFixture) -> None:
        """Test cleanup skips _delete when cleanup_proposals=False."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        driver = ProposalDriver({**VALID_CONFIG, "cleanup_proposals": False})
        mock_delete = mocker.patch.object(driver, "_delete")

        driver._cleanup("eval-test")

        mock_delete.assert_not_called()

    def test_cleanup_failure_logged(self, mocker: MockerFixture) -> None:
        """Test cleanup logs warning on _delete failure."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        driver = ProposalDriver(VALID_CONFIG)
        mocker.patch.object(driver, "_delete", side_effect=OSError("boom"))
        mock_logger = mocker.patch(f"{MODULE}.logger")

        driver._cleanup("eval-test")

        mock_logger.warning.assert_called_once()


# ── execute_turn ─────────────────────────────────────────────────────


class TestExecuteTurn:
    """Unit tests for ProposalDriver.execute_turn."""

    @pytest.fixture()
    def driver(self, mocker: MockerFixture) -> ProposalDriver:
        """Create a driver with mocked shutil and uuid."""
        mocker.patch(f"{MODULE}.shutil").which.return_value = "/usr/bin/oc"
        mocker.patch(f"{MODULE}.uuid").uuid4.return_value.hex = "abcd1234"
        return ProposalDriver({**VALID_CONFIG, "timeout": 10, "poll_interval": 1})

    def test_happy_path_completed(
        self, mocker: MockerFixture, driver: ProposalDriver
    ) -> None:
        """Test successful full lifecycle returns no error."""
        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.0, 1.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        terminal_status: dict[str, Any] = {
            "conditions": [
                _cond("Analyzed", "True", message="Analysis done"),
                _cond("Executed", "True"),
                _cond("Verified", "True", message="Passed"),
            ]
        }
        mocker.patch.object(driver, "_get_status", return_value=(terminal_status, None))
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Fix pod", proposal_spec=SPEC_FULL)
        error, conv_id = driver.execute_turn(turn)

        assert error is None
        assert conv_id is None
        response = str(turn.response)
        assert "Analysis done" in response
        assert "Passed" in response
        assert turn.proposal_status == terminal_status
        driver._cleanup.assert_called_once_with("eval-abcd1234")

    def test_apply_failure(self, mocker: MockerFixture, driver: ProposalDriver) -> None:
        """Test apply failure returns error without polling."""
        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=1, stderr="connection refused")

        turn = TurnData(turn_id="t1", query="Q")
        error, conv_id = driver.execute_turn(turn)

        assert error == "Failed to apply Proposal CR: connection refused"
        assert conv_id is None

    def test_timeout(self, mocker: MockerFixture, driver: ProposalDriver) -> None:
        """Test timeout returns error when no terminal condition reached."""
        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.0, 1.0, 11.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        non_terminal: dict[str, Any] = {"conditions": [_cond("Analyzed", "Unknown")]}
        mocker.patch.object(driver, "_get_status", return_value=(non_terminal, None))
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Q", proposal_spec=SPEC_FULL)
        error, conv_id = driver.execute_turn(turn)

        assert error is not None
        assert "Timeout after 10s" in error
        assert conv_id is None
        driver._cleanup.assert_called_once()

    def test_auto_approve(self, mocker: MockerFixture, driver: ProposalDriver) -> None:
        """Test auto-approve sends approval before polling."""
        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.0, 1.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        status_ready: dict[str, Any] = {"conditions": []}
        status_terminal: dict[str, Any] = {
            "conditions": [
                _cond("Analyzed", "True"),
                _cond("Executed", "True"),
                _cond("Verified", "True"),
            ]
        }
        mocker.patch.object(
            driver,
            "_get_status",
            side_effect=[
                (status_ready, None),
                (status_terminal, None),
            ],
        )
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Q", proposal_spec=SPEC_FULL)
        error, _ = driver.execute_turn(turn)

        assert error is None
        assert mock_apply.call_count == 2

    def test_auto_approve_disabled(
        self, mocker: MockerFixture, driver: ProposalDriver
    ) -> None:
        """Test auto-approve skipped when disabled."""
        driver._config.auto_approve = False

        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 1.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        status: dict[str, Any] = {"conditions": [_cond("Analyzed", "True")]}
        mocker.patch.object(driver, "_get_status", return_value=(status, None))
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Q")
        driver.execute_turn(turn)

        assert mock_apply.call_count == 1

    def test_failed_terminal(
        self, mocker: MockerFixture, driver: ProposalDriver
    ) -> None:
        """Test failed proposal returns error with outcome."""
        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.0, 1.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        status: dict[str, Any] = {
            "conditions": [_cond("Analyzed", "False", message="LLM error")]
        }
        mocker.patch.object(driver, "_get_status", return_value=(status, None))
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Q", proposal_spec=SPEC_FULL)
        error, _ = driver.execute_turn(turn)

        assert error is not None
        assert "failed" in error
        assert turn.proposal_status == status

    def test_denied_terminal(
        self, mocker: MockerFixture, driver: ProposalDriver
    ) -> None:
        """Test denied proposal populates turn data and returns no error."""
        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.0, 1.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        status: dict[str, Any] = {
            "conditions": [_cond("Denied", "True", message="User denied")]
        }
        mocker.patch.object(driver, "_get_status", return_value=(status, None))
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Q", proposal_spec=SPEC_FULL)
        error, _ = driver.execute_turn(turn)

        assert error is None
        assert turn.proposal_status == status

    def test_get_status_error(
        self, mocker: MockerFixture, driver: ProposalDriver
    ) -> None:
        """Test get_status error triggers cleanup and returns error."""
        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.0, 1.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        mocker.patch.object(
            driver,
            "_get_status",
            side_effect=[
                ({}, None),
                ({}, "Failed to get status for 'eval-abcd1234'"),
            ],
        )
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Q")
        error, _ = driver.execute_turn(turn)

        assert error is not None
        assert "Failed to get status" in error
        driver._cleanup.assert_called_once()

    def test_conversation_id_in_cr_name(
        self, mocker: MockerFixture, driver: ProposalDriver
    ) -> None:
        """Test conversation_id is included in CR name."""
        mock_time = mocker.patch(f"{MODULE}.time")
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.0, 1.0]

        mock_apply = mocker.patch.object(driver, "_apply")
        mock_apply.return_value = mocker.Mock(returncode=0)

        status: dict[str, Any] = {"conditions": [_cond("Analyzed", "True")]}
        mocker.patch.object(driver, "_get_status", return_value=(status, None))
        mocker.patch.object(driver, "_cleanup")

        turn = TurnData(turn_id="t1", query="Q")
        driver.execute_turn(turn, conversation_id="conv-42")

        driver._cleanup.assert_called_once_with("eval-conv-42-abcd1234")

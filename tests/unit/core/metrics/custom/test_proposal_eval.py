"""Unit tests for proposal status evaluation metric."""

from typing import Any, Optional

from lightspeed_evaluation.core.metrics.custom.proposal_eval import (
    _derive_phase,
    evaluate_proposal_status,
)
from lightspeed_evaluation.core.models import TurnData


def _make_turn(
    expected_proposal_status: Optional[dict[str, Any]] = None,
    proposal_status: Optional[dict[str, Any]] = None,
    proposal_spec: Optional[dict[str, Any]] = None,
) -> TurnData:
    """Build a minimal TurnData for testing."""
    return TurnData(
        turn_id="t1",
        query="test query",
        expected_proposal_status=expected_proposal_status,
        proposal_status=proposal_status,
        proposal_spec=proposal_spec,
    )


class TestValidation:
    """Input validation guards."""

    def test_conversation_level_returns_none(self) -> None:
        """Conversation-level evaluation returns None (turn-level only)."""
        turn = _make_turn(expected_proposal_status={"phase": "Completed"})
        score, reason = evaluate_proposal_status(None, None, turn, True)
        assert score is None
        assert "turn-level" in reason

    def test_none_turn_data_returns_none(self) -> None:
        """Missing turn_data returns None."""
        score, reason = evaluate_proposal_status(None, None, None, False)
        assert score is None
        assert "TurnData" in reason

    def test_missing_expected_returns_none(self) -> None:
        """Missing expected_proposal_status returns None (skip)."""
        turn = _make_turn()
        score, reason = evaluate_proposal_status(None, None, turn, False)
        assert score is None
        assert "expected_proposal_status" in reason

    def test_missing_proposal_status_returns_zero(self) -> None:
        """Missing proposal_status (driver didn't populate) returns 0.0."""
        turn = _make_turn(expected_proposal_status={"phase": "Completed"})
        score, reason = evaluate_proposal_status(None, None, turn, False)
        assert score == 0.0
        assert "not populated" in reason


class TestDerivePhase:
    """Phase derivation from CRD conditions."""

    def test_completed_analysis_only(self) -> None:
        """Analysis-only with Analyzed=True derives Completed."""
        conditions = [{"type": "Analyzed", "status": "True"}]
        assert _derive_phase(conditions, {"analysis": {}}) == "Completed"

    def test_completed_full_lifecycle(self) -> None:
        """Full lifecycle with all conditions True derives Completed."""
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Executed", "status": "True"},
            {"type": "Verified", "status": "True"},
        ]
        spec: dict[str, Any] = {"analysis": {}, "execution": {}, "verification": {}}
        assert _derive_phase(conditions, spec) == "Completed"

    def test_completed_execution_no_verification(self) -> None:
        """Execution without verification with Executed=True derives Completed."""
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Executed", "status": "True"},
        ]
        spec: dict[str, Any] = {"analysis": {}, "execution": {}}
        assert _derive_phase(conditions, spec) == "Completed"

    def test_failed_condition(self) -> None:
        """Any condition with status False derives Failed."""
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Executed", "status": "False", "reason": "Error"},
        ]
        assert _derive_phase(conditions) == "Failed"

    def test_retrying_execution_not_failed(self) -> None:
        """RetryingExecution reason does not count as failure."""
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Verified", "status": "False", "reason": "RetryingExecution"},
        ]
        spec: dict[str, Any] = {"analysis": {}, "execution": {}, "verification": {}}
        assert _derive_phase(conditions, spec) == "InProgress"

    def test_denied(self) -> None:
        """Denied=True derives Denied."""
        conditions = [{"type": "Denied", "status": "True"}]
        assert _derive_phase(conditions) == "Denied"

    def test_escalated(self) -> None:
        """Escalated=True derives Escalated."""
        conditions = [{"type": "Escalated", "status": "True"}]
        assert _derive_phase(conditions) == "Escalated"

    def test_in_progress(self) -> None:
        """Unknown status derives InProgress."""
        conditions = [{"type": "Analyzed", "status": "Unknown"}]
        assert _derive_phase(conditions) == "InProgress"

    def test_no_proposal_spec_infers_last_step(self) -> None:
        """Without proposal_spec, infers last step from conditions."""
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Executed", "status": "True"},
            {"type": "Verified", "status": "True"},
        ]
        assert _derive_phase(conditions, None) == "Completed"


class TestPhaseCheck:
    """Phase exact match assertion."""

    def test_phase_match_pass(self) -> None:
        """Exact phase match returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={"phase": "Completed"},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_spec={"analysis": {}},
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Phase matches" in reason

    def test_phase_match_fail(self) -> None:
        """Phase mismatch returns 0.0 with details."""
        turn = _make_turn(
            expected_proposal_status={"phase": "Completed"},
            proposal_status={
                "conditions": [
                    {"type": "Analyzed", "status": "False", "reason": "Error"},
                ],
            },
            proposal_spec={"analysis": {}},
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "Phase mismatch" in reason
        assert "'Completed'" in reason
        assert "'Failed'" in reason


class TestPhaseInCheck:
    """Phase membership assertion."""

    def test_phase_in_pass(self) -> None:
        """Phase in allowed list returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={"phase_in": ["Completed", "Escalated"]},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_spec={"analysis": {}},
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_phase_in_fail(self) -> None:
        """Phase not in allowed list returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={"phase_in": ["Completed"]},
            proposal_status={
                "conditions": [{"type": "Denied", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "not in" in reason


class TestConditionsCheck:
    """Specific condition assertions."""

    def test_condition_status_pass(self) -> None:
        """Condition status match returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_status={
                "conditions": [
                    {"type": "Analyzed", "status": "True", "reason": "Done"},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "condition assertions passed" in reason

    def test_condition_status_fail(self) -> None:
        """Condition status mismatch returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "False"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "Analyzed" in reason
        assert "status" in reason

    def test_condition_reason_pass(self) -> None:
        """Condition reason match returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={
                "conditions": [
                    {"type": "Executed", "status": "True", "reason": "Skipped"},
                ],
            },
            proposal_status={
                "conditions": [
                    {"type": "Executed", "status": "True", "reason": "Skipped"},
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_condition_reason_fail(self) -> None:
        """Condition reason mismatch returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={
                "conditions": [
                    {"type": "Executed", "reason": "Skipped"},
                ],
            },
            proposal_status={
                "conditions": [
                    {"type": "Executed", "status": "True", "reason": "Done"},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "reason" in reason

    def test_condition_not_found(self) -> None:
        """Missing condition type returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={
                "conditions": [{"type": "Verified", "status": "True"}],
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "not found" in reason

    def test_condition_missing_type_field(self) -> None:
        """Condition assertion without type field returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={
                "conditions": [{"status": "True"}],
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "missing 'type'" in reason


class TestVerificationCheck:
    """Verification-specific assertions."""

    def test_verification_passed_true(self) -> None:
        """Verification passed=True with Verified=True returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={"verification": {"passed": True}},
            proposal_status={
                "conditions": [{"type": "Verified", "status": "True"}],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_verification_passed_false(self) -> None:
        """Verification passed=True with Verified=False returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={"verification": {"passed": True}},
            proposal_status={
                "conditions": [
                    {"type": "Verified", "status": "False", "reason": "Error"},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "Verification passed" in reason

    def test_verification_summary_contains_pass(self) -> None:
        """Verification summary_contains match returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={
                "verification": {"summary_contains": "replicas running"},
            },
            proposal_status={
                "conditions": [
                    {
                        "type": "Verified",
                        "status": "True",
                        "message": "3 replicas running successfully",
                    },
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_verification_summary_contains_fail(self) -> None:
        """Verification summary_contains mismatch returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={
                "verification": {"summary_contains": "replicas running"},
            },
            proposal_status={
                "conditions": [
                    {
                        "type": "Verified",
                        "status": "True",
                        "message": "Pod restarted",
                    },
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "does not contain" in reason

    def test_verification_condition_missing(self) -> None:
        """Verification check with missing Verified condition returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={"verification": {"passed": True}},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "not found" in reason

    def test_verification_summary_case_insensitive(self) -> None:
        """Verification summary_contains is case-insensitive."""
        turn = _make_turn(
            expected_proposal_status={
                "verification": {"summary_contains": "REPLICAS"},
            },
            proposal_status={
                "conditions": [
                    {
                        "type": "Verified",
                        "status": "True",
                        "message": "3 replicas running",
                    },
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0


class TestMultipleChecks:
    """Combined assertions."""

    def test_all_checks_pass(self) -> None:
        """All checks passing returns 1.0 with combined reasons."""
        turn = _make_turn(
            expected_proposal_status={
                "phase": "Completed",
                "conditions": [{"type": "Analyzed", "status": "True"}],
                "verification": {"passed": True},
            },
            proposal_status={
                "conditions": [
                    {"type": "Analyzed", "status": "True"},
                    {"type": "Executed", "status": "True"},
                    {"type": "Verified", "status": "True"},
                ],
            },
            proposal_spec={"analysis": {}, "execution": {}, "verification": {}},
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Phase matches" in reason
        assert "condition assertions passed" in reason
        assert "Verification assertions passed" in reason

    def test_fail_fast_on_first_failure(self) -> None:
        """First failing check returns 0.0 without running subsequent checks."""
        turn = _make_turn(
            expected_proposal_status={
                "phase": "Completed",
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_status={
                "conditions": [
                    {"type": "Analyzed", "status": "False", "reason": "Error"},
                ],
            },
            proposal_spec={"analysis": {}},
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "Phase mismatch" in reason
        assert "condition" not in reason.lower()

    def test_empty_expected_skips(self) -> None:
        """Empty expected_proposal_status dict is treated as no expectations."""
        turn = _make_turn(
            expected_proposal_status={},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score is None
        assert "expected_proposal_status" in reason

    def test_empty_conditions_list(self) -> None:
        """Empty conditions list with phase check fails."""
        turn = _make_turn(
            expected_proposal_status={"phase": "Completed"},
            proposal_status={"conditions": []},
            proposal_spec={"analysis": {}},
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "Phase mismatch" in reason

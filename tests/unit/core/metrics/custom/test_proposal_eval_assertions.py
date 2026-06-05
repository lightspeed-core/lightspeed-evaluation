"""Unit tests for new proposal status assertion checks.

Covers: _parse_duration, max_duration, max_attempts, analysis
(with component/option helpers), execution, and check ordering.
"""

from typing import Any, Optional

import pytest

from lightspeed_evaluation.core.metrics.custom.proposal_eval import (
    _check_analysis_component,
    _parse_duration,
    evaluate_proposal_status,
)
from lightspeed_evaluation.core.models import TurnData


def _make_turn(
    expected_proposal_status: Optional[dict[str, Any]] = None,
    proposal_status: Optional[dict[str, Any]] = None,
    proposal_spec: Optional[dict[str, Any]] = None,
    proposal_results: Optional[dict[str, Any]] = None,
) -> TurnData:
    """Build a minimal TurnData for testing."""
    return TurnData(
        turn_id="t1",
        query="test query",
        expected_proposal_status=expected_proposal_status,
        proposal_status=proposal_status,
        proposal_spec=proposal_spec,
        proposal_results=proposal_results,
    )


class TestParseDuration:
    """Go-style duration string parser."""

    def test_minutes_only(self) -> None:
        """Parse minutes-only duration."""
        assert _parse_duration("5m") == 300.0

    def test_seconds_only(self) -> None:
        """Parse seconds-only duration."""
        assert _parse_duration("30s") == 30.0

    def test_hours_only(self) -> None:
        """Parse hours-only duration."""
        assert _parse_duration("1h") == 3600.0

    def test_combined_minutes_seconds(self) -> None:
        """Parse combined minutes and seconds."""
        assert _parse_duration("2m30s") == 150.0

    def test_combined_hours_minutes(self) -> None:
        """Parse combined hours and minutes."""
        assert _parse_duration("1h5m") == 3900.0

    def test_combined_all(self) -> None:
        """Parse combined hours, minutes, and seconds."""
        assert _parse_duration("1h30m15s") == 5415.0

    def test_invalid_format(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized duration"):
            _parse_duration("5x")

    def test_empty_string(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized duration"):
            _parse_duration("")


class TestMaxDurationCheck:
    """Max duration assertion on condition timestamps."""

    def test_within_limit_pass(self) -> None:
        """Elapsed time within limit returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={"max_duration": "5m"},
            proposal_status={
                "conditions": [
                    {
                        "type": "Analyzed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:00:00Z",
                    },
                    {
                        "type": "Executed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:03:00Z",
                    },
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Duration" in reason
        assert "within limit" in reason

    def test_exceeded_fail(self) -> None:
        """Elapsed time exceeding limit returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={"max_duration": "5m"},
            proposal_status={
                "conditions": [
                    {
                        "type": "Analyzed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:00:00Z",
                    },
                    {
                        "type": "Executed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:10:00Z",
                    },
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "exceeds limit" in reason

    def test_no_timestamps_fail(self) -> None:
        """Conditions without lastTransitionTime returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={"max_duration": "5m"},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "No lastTransitionTime" in reason

    def test_skip_when_not_specified(self) -> None:
        """No max_duration in expected skips the check."""
        turn = _make_turn(
            expected_proposal_status={"phase": "Completed"},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_spec={"analysis": {}},
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_boundary_equal(self) -> None:
        """Elapsed time exactly at limit passes (less-than-or-equal)."""
        turn = _make_turn(
            expected_proposal_status={"max_duration": "3m"},
            proposal_status={
                "conditions": [
                    {
                        "type": "Analyzed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:00:00Z",
                    },
                    {
                        "type": "Executed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:03:00Z",
                    },
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "within limit" in reason


class TestMaxAttemptsCheck:
    """Max attempts assertion."""

    def test_within_limit_pass(self) -> None:
        """Attempts within limit returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={"max_attempts": 3},
            proposal_status={
                "attempts": 2,
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Attempts 2 within limit 3" in reason

    def test_exceeded_fail(self) -> None:
        """Attempts exceeding limit returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={"max_attempts": 3},
            proposal_status={
                "attempts": 4,
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "exceeds limit" in reason

    def test_from_status_field(self) -> None:
        """Reads attempts from proposal_status.attempts when available."""
        turn = _make_turn(
            expected_proposal_status={"max_attempts": 5},
            proposal_status={
                "attempts": 1,
                "conditions": [
                    {
                        "type": "Executed",
                        "status": "False",
                        "reason": "RetryingExecution",
                    },
                    {"type": "Analyzed", "status": "True"},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Attempts 1" in reason

    def test_inferred_from_conditions(self) -> None:
        """Infers attempts from RetryingExecution conditions + 1."""
        turn = _make_turn(
            expected_proposal_status={"max_attempts": 3},
            proposal_status={
                "conditions": [
                    {
                        "type": "Executed",
                        "status": "False",
                        "reason": "RetryingExecution",
                    },
                    {
                        "type": "Verified",
                        "status": "False",
                        "reason": "RetryingExecution",
                    },
                    {"type": "Analyzed", "status": "True"},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Attempts 3" in reason

    def test_skip_when_not_specified(self) -> None:
        """No max_attempts in expected skips the check."""
        turn = _make_turn(
            expected_proposal_status={"phase": "Completed"},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_spec={"analysis": {}},
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0


class TestAnalysisCheck:
    """Analysis assertion checks (options, risk, confidence, components)."""

    def test_skip_when_not_specified(self) -> None:
        """No analysis in expected skips the check."""
        turn = _make_turn(
            expected_proposal_status={"phase": "Completed"},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_spec={"analysis": {}},
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_min_options_pass(self) -> None:
        """Enough options passes min_options check."""
        turn = _make_turn(
            expected_proposal_status={"analysis": {"min_options": 1}},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [{"options": [{"diagnosis": {}, "proposal": {}}]}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Analysis assertions passed" in reason

    def test_min_options_fail(self) -> None:
        """Too few options fails min_options check."""
        turn = _make_turn(
            expected_proposal_status={"analysis": {"min_options": 2}},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={"analysis": [{"options": [{"diagnosis": {}}]}]},
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "1 options" in reason
        assert "at least 2" in reason

    def test_no_proposal_results_fail(self) -> None:
        """Missing proposal_results fails analysis check."""
        turn = _make_turn(
            expected_proposal_status={"analysis": {"min_options": 1}},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "No proposal_results" in reason

    def test_risk_in_pass(self) -> None:
        """Risk value in allowed list passes."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [{"risk_in": ["low", "medium"]}],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {"options": [{"proposal": {"risk": "Low"}, "diagnosis": {}}]},
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_risk_in_fail(self) -> None:
        """Risk value not in allowed list fails."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [{"risk_in": ["low", "medium"]}],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {"options": [{"proposal": {"risk": "High"}, "diagnosis": {}}]},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "risk" in reason.lower()
        assert "High" in reason

    def test_confidence_in_pass(self) -> None:
        """Confidence value in allowed list passes."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [{"confidence_in": ["medium", "high"]}],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {
                        "options": [
                            {"diagnosis": {"confidence": "High"}, "proposal": {}}
                        ]
                    },
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_confidence_in_fail(self) -> None:
        """Confidence value not in allowed list fails."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [{"confidence_in": ["medium", "high"]}],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {"options": [{"diagnosis": {"confidence": "Low"}, "proposal": {}}]},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "confidence" in reason.lower()

    def test_diagnosis_contains_pass(self) -> None:
        """Diagnosis summary containing all substrings passes."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [{"diagnosis_contains": ["crash", "image"]}],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {
                        "options": [
                            {
                                "diagnosis": {
                                    "summary": "Pod crash due to bad image pull",
                                },
                                "proposal": {},
                            },
                        ],
                    },
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_diagnosis_contains_fail(self) -> None:
        """Diagnosis summary missing a substring fails."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [{"diagnosis_contains": ["crash", "image"]}],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {
                        "options": [
                            {"diagnosis": {"summary": "Pod crash"}, "proposal": {}},
                        ],
                    },
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "does not contain" in reason
        assert "'image'" in reason

    def test_component_match_pass(self) -> None:
        """Component exact match passes."""
        result = _check_analysis_component(
            "remediation_summary",
            {"match": {"action": "Scale", "replicas": 3}},
            {
                "remediation_summary": {
                    "type": "remediation_summary",
                    "action": "Scale",
                    "replicas": 3,
                }
            },
        )
        assert result is not None
        passed, _ = result
        assert passed

    def test_component_match_fail(self) -> None:
        """Component exact match mismatch fails."""
        result = _check_analysis_component(
            "remediation_summary",
            {"match": {"action": "Scale"}},
            {
                "remediation_summary": {
                    "type": "remediation_summary",
                    "action": "Restart",
                }
            },
        )
        assert result is not None
        passed, reason = result
        assert not passed
        assert "'action'" in reason

    def test_component_absent_pass(self) -> None:
        """Absent component correctly not present passes."""
        result = _check_analysis_component(
            "destructive_action",
            {"absent": True},
            {"remediation_summary": {"type": "remediation_summary"}},
        )
        assert result is not None
        passed, _ = result
        assert passed

    def test_component_absent_fail(self) -> None:
        """Absent component that is present fails."""
        result = _check_analysis_component(
            "destructive_action",
            {"absent": True},
            {
                "destructive_action": {
                    "type": "destructive_action",
                    "detail": "drop table",
                }
            },
        )
        assert result is not None
        passed, reason = result
        assert not passed
        assert "should be absent" in reason

    def test_component_required_pass(self) -> None:
        """Required fields all present passes."""
        result = _check_analysis_component(
            "risk_assessment",
            {"required": ["mitigation_steps", "summary"]},
            {
                "risk_assessment": {
                    "type": "risk_assessment",
                    "mitigation_steps": ["rollback"],
                    "summary": "low risk",
                },
            },
        )
        assert result is not None
        passed, _ = result
        assert passed

    def test_component_required_fail(self) -> None:
        """Missing required field fails."""
        result = _check_analysis_component(
            "risk_assessment",
            {"required": ["mitigation_steps"]},
            {
                "risk_assessment": {
                    "type": "risk_assessment",
                    "summary": "low risk",
                }
            },
        )
        assert result is not None
        passed, reason = result
        assert not passed
        assert "missing required" in reason

    def test_component_match_contains_pass(self) -> None:
        """Component substring match passes."""
        result = _check_analysis_component(
            "risk_assessment",
            {"match_contains": {"summary": "low risk"}},
            {
                "risk_assessment": {
                    "type": "risk_assessment",
                    "summary": "This is a low risk change",
                }
            },
        )
        assert result is not None
        passed, _ = result
        assert passed

    def test_component_match_contains_fail(self) -> None:
        """Component substring match mismatch fails."""
        result = _check_analysis_component(
            "risk_assessment",
            {"match_contains": {"summary": "low risk"}},
            {
                "risk_assessment": {
                    "type": "risk_assessment",
                    "summary": "high risk detected",
                }
            },
        )
        assert result is not None
        passed, reason = result
        assert not passed
        assert "does not contain" in reason

    def test_uses_latest_analysis_result_on_retry(self) -> None:
        """On retry, only options from the latest analysis result are checked."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [{"risk_in": ["low"]}],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {"options": [{"proposal": {"risk": "High"}, "diagnosis": {}}]},
                    {"options": [{"proposal": {"risk": "Low"}, "diagnosis": {}}]},
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_option_index_out_of_range(self) -> None:
        """Expected option at index beyond actual options fails."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {
                    "options": [
                        {"risk_in": ["low"]},
                        {"risk_in": ["low"]},
                    ],
                },
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [
                    {"options": [{"proposal": {"risk": "Low"}, "diagnosis": {}}]},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "Option[1]" in reason
        assert "1 options present" in reason


class TestExecutionCheck:
    """Execution phase assertion checks."""

    def test_phase_match_pass(self) -> None:
        """Execution phase match returns 1.0."""
        turn = _make_turn(
            expected_proposal_status={"execution": {"phase": "Succeeded"}},
            proposal_status={
                "conditions": [{"type": "Executed", "status": "True"}],
            },
            proposal_results={
                "execution": [{"phase": "Succeeded"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Execution assertions passed" in reason

    def test_phase_mismatch_fail(self) -> None:
        """Execution phase mismatch returns 0.0."""
        turn = _make_turn(
            expected_proposal_status={"execution": {"phase": "Succeeded"}},
            proposal_status={
                "conditions": [{"type": "Executed", "status": "True"}],
            },
            proposal_results={
                "execution": [{"phase": "Failed"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "Execution phase" in reason
        assert "'Succeeded'" in reason
        assert "'Failed'" in reason

    def test_phase_from_conditions_fallback(self) -> None:
        """Falls back to condition reason when no phase field."""
        turn = _make_turn(
            expected_proposal_status={"execution": {"phase": "Succeeded"}},
            proposal_status={
                "conditions": [{"type": "Executed", "status": "True"}],
            },
            proposal_results={
                "execution": [
                    {
                        "conditions": [
                            {
                                "type": "Completed",
                                "status": "True",
                                "reason": "Succeeded",
                            }
                        ]
                    },
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Execution assertions passed" in reason

    def test_skip_when_not_specified(self) -> None:
        """No execution in expected skips the check."""
        turn = _make_turn(
            expected_proposal_status={"phase": "Completed"},
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_spec={"analysis": {}},
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

    def test_uses_latest_execution_result_on_retry(self) -> None:
        """On retry, the latest execution result determines the phase."""
        turn = _make_turn(
            expected_proposal_status={"execution": {"phase": "Succeeded"}},
            proposal_status={
                "conditions": [{"type": "Executed", "status": "True"}],
            },
            proposal_results={
                "execution": [
                    {"phase": "Failed"},
                    {"phase": "Succeeded"},
                ],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Execution assertions passed" in reason

    def test_no_execution_results_fail(self) -> None:
        """Missing execution results fails."""
        turn = _make_turn(
            expected_proposal_status={"execution": {"phase": "Succeeded"}},
            proposal_status={
                "conditions": [{"type": "Executed", "status": "True"}],
            },
            proposal_results={"analysis": []},
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "No execution results" in reason


class TestCheckOrdering:
    """Verify checks run in spec order and fail fast correctly."""

    def test_timing_fails_before_analysis(self) -> None:
        """Max duration failure reported before analysis check."""
        turn = _make_turn(
            expected_proposal_status={
                "max_duration": "1m",
                "analysis": {"min_options": 1},
            },
            proposal_status={
                "conditions": [
                    {
                        "type": "Analyzed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:00:00Z",
                    },
                    {
                        "type": "Executed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:05:00Z",
                    },
                ],
            },
            proposal_results={
                "analysis": [{"options": [{"diagnosis": {}}]}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "exceeds limit" in reason

    def test_analysis_fails_before_execution(self) -> None:
        """Analysis failure reported before execution check."""
        turn = _make_turn(
            expected_proposal_status={
                "analysis": {"min_options": 5},
                "execution": {"phase": "Succeeded"},
            },
            proposal_status={
                "conditions": [{"type": "Analyzed", "status": "True"}],
            },
            proposal_results={
                "analysis": [{"options": [{"diagnosis": {}}]}],
                "execution": [{"phase": "Succeeded"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 0.0
        assert "options" in reason
        assert "at least 5" in reason

    def test_all_new_checks_pass(self) -> None:
        """All new checks passing returns 1.0 with combined reasons."""
        turn = _make_turn(
            expected_proposal_status={
                "max_duration": "10m",
                "max_attempts": 3,
                "analysis": {"min_options": 1},
                "execution": {"phase": "Succeeded"},
            },
            proposal_status={
                "attempts": 1,
                "conditions": [
                    {
                        "type": "Analyzed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:00:00Z",
                    },
                    {
                        "type": "Executed",
                        "status": "True",
                        "lastTransitionTime": "2024-01-15T10:03:00Z",
                    },
                ],
            },
            proposal_results={
                "analysis": [
                    {"options": [{"diagnosis": {}, "proposal": {}}]},
                ],
                "execution": [{"phase": "Succeeded"}],
            },
        )
        score, reason = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0
        assert "Duration" in reason
        assert "Attempts" in reason
        assert "Analysis assertions passed" in reason
        assert "Execution assertions passed" in reason

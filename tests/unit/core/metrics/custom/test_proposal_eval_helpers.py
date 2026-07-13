"""Unit tests for proposal_eval helper functions.

Covers: _get_result_phase, _latest_terminal_result.
"""

from typing import Any, Optional

from lightspeed_evaluation.core.metrics.custom.proposal_eval import (
    _get_result_phase,
    _latest_terminal_result,
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


class TestGetResultPhase:
    """Phase extraction helper."""

    def test_direct_phase_field(self) -> None:
        """Extracts phase from top-level field."""
        assert _get_result_phase({"phase": "Succeeded"}) == "Succeeded"

    def test_completed_condition_preferred_over_started(self) -> None:
        """Reads Completed condition reason, not Started."""
        result: dict[str, Any] = {
            "conditions": [
                {"type": "Started", "reason": "StepStarted"},
                {"type": "Completed", "reason": "Succeeded"},
            ]
        }
        assert _get_result_phase(result) == "Succeeded"

    def test_completed_condition_regardless_of_order(self) -> None:
        """Finds Completed condition even if it comes first."""
        result: dict[str, Any] = {
            "conditions": [
                {"type": "Completed", "reason": "Failed"},
                {"type": "Started", "reason": "StepStarted"},
            ]
        }
        assert _get_result_phase(result) == "Failed"

    def test_only_started_condition(self) -> None:
        """Falls back to Started reason when no Completed condition."""
        result: dict[str, Any] = {
            "conditions": [{"type": "Started", "reason": "StepStarted"}]
        }
        assert _get_result_phase(result) == "StepStarted"

    def test_only_completed_condition(self) -> None:
        """Works with only Completed condition (no start time)."""
        result: dict[str, Any] = {
            "conditions": [{"type": "Completed", "reason": "Failed"}]
        }
        assert _get_result_phase(result) == "Failed"

    def test_none_when_empty(self) -> None:
        """Returns None for result with no phase info."""
        assert _get_result_phase({}) is None
        assert _get_result_phase({"conditions": []}) is None


class TestLatestTerminalResult:
    """Non-terminal result skipping."""

    def test_returns_last_when_terminal(self) -> None:
        """Returns last result when it has a terminal phase."""
        results: list[dict[str, Any]] = [
            {
                "conditions": [
                    {"type": "Started", "reason": "StepStarted"},
                    {"type": "Completed", "reason": "Failed"},
                ]
            },
            {
                "conditions": [
                    {"type": "Started", "reason": "StepStarted"},
                    {"type": "Completed", "reason": "Succeeded"},
                ]
            },
        ]
        assert _latest_terminal_result(results) is results[1]

    def test_skips_trailing_in_progress(self) -> None:
        """Skips trailing in-progress result and returns previous completed."""
        results: list[dict[str, Any]] = [
            {
                "conditions": [
                    {"type": "Started", "reason": "StepStarted"},
                    {"type": "Completed", "reason": "Succeeded"},
                ]
            },
            {
                "conditions": [
                    {"type": "Started", "reason": "StepStarted"},
                ]
            },
        ]
        assert _latest_terminal_result(results) is results[0]

    def test_skips_multiple_non_terminal(self) -> None:
        """Skips multiple trailing in-progress results."""
        results: list[dict[str, Any]] = [
            {
                "conditions": [
                    {"type": "Started", "reason": "StepStarted"},
                    {"type": "Completed", "reason": "Succeeded"},
                ]
            },
            {"conditions": [{"type": "Started", "reason": "StepStarted"}]},
            {"conditions": [{"type": "Started", "reason": "StepStarted"}]},
        ]
        assert _latest_terminal_result(results) is results[0]

    def test_fallback_when_all_non_terminal(self) -> None:
        """Falls back to last when every result is in-progress."""
        results: list[dict[str, Any]] = [
            {"conditions": [{"type": "Started", "reason": "StepStarted"}]},
            {"conditions": [{"type": "Started", "reason": "StepStarted"}]},
        ]
        assert _latest_terminal_result(results) is results[-1]

    def test_no_phase_is_terminal(self) -> None:
        """Result with no phase field and no conditions is treated as terminal."""
        results: list[dict[str, Any]] = [
            {"conditions": [{"type": "Started", "reason": "StepStarted"}]},
            {"options": [{"diagnosis": {}}]},
        ]
        assert _latest_terminal_result(results) is results[1]

    def test_analysis_skips_trailing_non_terminal(self) -> None:
        """A trailing in-progress analysis result is skipped end-to-end."""
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
                    {
                        "options": [
                            {"remediationPlan": {"risk": "Low"}, "diagnosis": {}}
                        ]
                    },
                    {"conditions": [{"type": "Started", "reason": "StepStarted"}]},
                ],
            },
        )
        score, _ = evaluate_proposal_status(None, 0, turn, False)
        assert score == 1.0

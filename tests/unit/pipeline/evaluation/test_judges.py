# pylint: disable=redefined-outer-name,too-few-public-methods,protected-access

"""Unit tests for JudgeOrchestrator - multi-judge evaluation and aggregation."""

from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import JudgeScore
from lightspeed_evaluation.pipeline.evaluation.judges import JudgeOrchestrator


def _make_orchestrator(
    mocker: MockerFixture,
    strategy: str,
    status_determiner: Any = None,
) -> JudgeOrchestrator:
    """Create a JudgeOrchestrator with a given aggregation strategy."""
    mock_manager = mocker.MagicMock()
    mock_manager.system_config.judge_panel.aggregation_strategy = strategy
    return JudgeOrchestrator(
        llm_manager=mock_manager,
        primary_handlers={},
        handler_factory=mocker.MagicMock(),
        status_determiner=status_determiner or (lambda s, _: "PASS"),
    )


class TestAggregateScores:
    """Tests for JudgeOrchestrator.aggregate_scores method."""

    @pytest.fixture
    def mock_llm_manager(self, mocker: MockerFixture) -> Any:
        """Create a mock LLM manager without system_config (no panel)."""
        manager = mocker.MagicMock()
        manager.system_config = None
        manager.judge_id = "primary"
        return manager

    @pytest.fixture
    def orchestrator(
        self, mock_llm_manager: Any, mocker: MockerFixture
    ) -> JudgeOrchestrator:
        """Create JudgeOrchestrator with mocked dependencies."""
        return JudgeOrchestrator(
            llm_manager=mock_llm_manager,
            primary_handlers={"ragas": mocker.MagicMock()},
            handler_factory=mocker.MagicMock(),
            status_determiner=lambda score, _: "PASS" if score >= 0.5 else "FAIL",
        )

    def test_single_judge_returns_direct_score(
        self, orchestrator: JudgeOrchestrator
    ) -> None:
        """Single judge returns its score and reason directly."""
        judge_scores = [
            JudgeScore(judge_id="judge-1", score=0.85, reason="Good answer")
        ]

        score, reason, override = orchestrator.aggregate_scores(judge_scores)

        assert score == 0.85
        assert reason == "Good answer"
        assert override is None

    def test_single_judge_with_none_score(
        self, orchestrator: JudgeOrchestrator
    ) -> None:
        """Single judge with None score returns None and original reason."""
        judge_scores = [
            JudgeScore(
                judge_id="judge-1", score=None, reason="Evaluation error: timeout"
            )
        ]

        score, reason, override = orchestrator.aggregate_scores(judge_scores)

        assert score is None
        assert reason == "Evaluation error: timeout"
        assert override is None

    def test_multiple_judges_returns_max(self, orchestrator: JudgeOrchestrator) -> None:
        """Multiple judges return max score."""
        judge_scores = [
            JudgeScore(judge_id="judge-1", score=0.6, reason="Okay"),
            JudgeScore(judge_id="judge-2", score=0.9, reason="Great"),
            JudgeScore(judge_id="judge-3", score=0.75, reason="Good"),
        ]

        score, reason, override = orchestrator.aggregate_scores(judge_scores)

        assert score == 0.9
        assert "Max of 3 judges" in reason
        assert "0.900" in reason
        assert override is None

    def test_multiple_judges_with_one_failure(
        self, orchestrator: JudgeOrchestrator
    ) -> None:
        """Multiple judges with one failure returns max of valid scores."""
        judge_scores = [
            JudgeScore(judge_id="judge-1", score=0.7, reason="Good"),
            JudgeScore(judge_id="judge-2", score=None, reason="Error"),
            JudgeScore(judge_id="judge-3", score=0.85, reason="Great"),
        ]

        score, reason, override = orchestrator.aggregate_scores(judge_scores)

        assert score == 0.85
        assert "Max of 2 judges" in reason
        assert override is None

    def test_all_judges_failed(self, orchestrator: JudgeOrchestrator) -> None:
        """All judges failing returns None with appropriate message."""
        judge_scores = [
            JudgeScore(judge_id="judge-1", score=None, reason="Error 1"),
            JudgeScore(judge_id="judge-2", score=None, reason="Error 2"),
        ]

        score, reason, override = orchestrator.aggregate_scores(judge_scores)

        assert score is None
        assert reason == "All judges failed to produce a score"
        assert override is None


class TestAggregationStrategies:
    """Tests for average and majority_vote aggregation."""

    def test_average_strategy(self, mocker: MockerFixture) -> None:
        """Average strategy returns mean of valid scores."""
        orch = _make_orchestrator(mocker, "average")
        judge_scores = [
            JudgeScore(judge_id="j1", score=0.6, reason="A"),
            JudgeScore(judge_id="j2", score=0.8, reason="B"),
        ]
        score, reason, override = orch.aggregate_scores(judge_scores, 0.7)
        assert score == pytest.approx(0.7)
        assert "Average of 2 judges" in reason
        assert override is None

    def test_majority_vote_pass(self, mocker: MockerFixture) -> None:
        """Majority vote: PASS when strict majority of judges meet metric threshold."""
        # status_determiner returns FAIL to prove override takes precedence
        orch = _make_orchestrator(
            mocker, "majority_vote", status_determiner=lambda s, _: "FAIL"
        )
        judge_scores = [
            JudgeScore(judge_id="j1", score=0.9, reason="A"),
            JudgeScore(judge_id="j2", score=0.9, reason="B"),
            JudgeScore(judge_id="j3", score=0.2, reason="C"),
        ]
        score, reason, override = orch.aggregate_scores(judge_scores, 0.7)
        assert score == pytest.approx((0.9 + 0.9 + 0.2) / 3)
        assert override == "PASS"
        assert "Majority vote (2/3" in reason

    def test_majority_vote_fail(self, mocker: MockerFixture) -> None:
        """Majority vote: FAIL when majority do not meet threshold."""
        orch = _make_orchestrator(mocker, "majority_vote")
        judge_scores = [
            JudgeScore(judge_id="j1", score=0.8, reason="A"),
            JudgeScore(judge_id="j2", score=0.2, reason="B"),
            JudgeScore(judge_id="j3", score=0.2, reason="C"),
        ]
        score, reason, override = orch.aggregate_scores(judge_scores, 0.7)
        assert override == "FAIL"
        assert "Majority vote (1/3" in reason
        assert score == pytest.approx((0.8 + 0.2 + 0.2) / 3)

    def test_majority_vote_none_threshold_uses_default_half(
        self, mocker: MockerFixture
    ) -> None:
        """Majority vote with threshold=None uses 0.5; also tests 2-judge tie (1/2 → FAIL)."""
        orch = _make_orchestrator(mocker, "majority_vote")
        judge_scores = [
            JudgeScore(judge_id="j1", score=0.4, reason="A"),
            JudgeScore(judge_id="j2", score=0.6, reason="B"),
        ]
        score, reason, override = orch.aggregate_scores(judge_scores, None)
        assert score == pytest.approx(0.5)
        assert override == "FAIL"  # 1/2 pass at 0.5, not strict majority
        assert "0.500 (default)" in reason

    def test_majority_vote_even_split_tie_is_fail(self, mocker: MockerFixture) -> None:
        """Tie on 4 judges (2/4 pass): strict majority needs >2, so FAIL."""
        orch = _make_orchestrator(mocker, "majority_vote")
        judge_scores = [
            JudgeScore(judge_id="j1", score=0.9, reason="A"),
            JudgeScore(judge_id="j2", score=0.8, reason="B"),
            JudgeScore(judge_id="j3", score=0.3, reason="C"),
            JudgeScore(judge_id="j4", score=0.2, reason="D"),
        ]
        score, reason, override = orch.aggregate_scores(judge_scores, 0.7)
        assert override == "FAIL"
        assert "Majority vote (2/4" in reason
        assert score == pytest.approx(0.55)

    def test_majority_vote_three_of_four_pass(self, mocker: MockerFixture) -> None:
        """3/4 judges pass: strict majority (3 > 2), so PASS."""
        orch = _make_orchestrator(
            mocker, "majority_vote", status_determiner=lambda s, _: "FAIL"
        )
        judge_scores = [
            JudgeScore(judge_id="j1", score=0.9, reason="A"),
            JudgeScore(judge_id="j2", score=0.8, reason="B"),
            JudgeScore(judge_id="j3", score=0.7, reason="C"),
            JudgeScore(judge_id="j4", score=0.2, reason="D"),
        ]
        score, reason, override = orch.aggregate_scores(judge_scores, 0.7)
        assert override == "PASS"
        assert "Majority vote (3/4" in reason
        assert score == pytest.approx(0.65)


class TestHandlerCaching:
    """Tests for handler caching by judge_id."""

    def test_handler_cached_per_judge_id(self, mocker: MockerFixture) -> None:
        """Handlers are cached by judge_id, not model name."""
        mock_manager = mocker.MagicMock()
        mock_manager.system_config = None
        mock_manager.judge_id = "primary"

        handler_factory = mocker.MagicMock()
        handler_factory.return_value = mocker.MagicMock()

        orchestrator = JudgeOrchestrator(
            llm_manager=mock_manager,
            primary_handlers={"ragas": mocker.MagicMock()},
            handler_factory=handler_factory,
            status_determiner=lambda s, _: "PASS",
        )

        # Create two judge managers with same model but different judge_ids
        judge_alpha = mocker.MagicMock()
        judge_alpha.judge_id = "judge-alpha"

        judge_beta = mocker.MagicMock()
        judge_beta.judge_id = "judge-beta"

        # Get handlers for both judges
        orchestrator._get_handler_for_judge("ragas", judge_alpha)
        orchestrator._get_handler_for_judge("ragas", judge_beta)

        # Factory should be called twice (once per unique judge_id)
        assert handler_factory.call_count == 2

    def test_primary_handler_reused(self, mocker: MockerFixture) -> None:
        """Primary manager reuses pre-initialized handlers."""
        mock_manager = mocker.MagicMock()
        mock_manager.system_config = None
        mock_manager.judge_id = "primary"

        primary_handler = mocker.MagicMock()
        handler_factory = mocker.MagicMock()

        orchestrator = JudgeOrchestrator(
            llm_manager=mock_manager,
            primary_handlers={"ragas": primary_handler},
            handler_factory=handler_factory,
            status_determiner=lambda s, _: "PASS",
        )

        # Get handler for primary manager
        handler = orchestrator._get_handler_for_judge("ragas", mock_manager)

        # Should return the pre-initialized handler
        assert handler is primary_handler
        # Factory should not be called
        handler_factory.assert_not_called()

    def test_cached_handler_returned_on_second_call(
        self, mocker: MockerFixture
    ) -> None:
        """Same handler is returned for same judge_id on subsequent calls."""
        mock_manager = mocker.MagicMock()
        mock_manager.system_config = None
        mock_manager.judge_id = "primary"

        cached_handler = mocker.MagicMock()
        handler_factory = mocker.MagicMock(return_value=cached_handler)

        orchestrator = JudgeOrchestrator(
            llm_manager=mock_manager,
            primary_handlers={},
            handler_factory=handler_factory,
            status_determiner=lambda s, _: "PASS",
        )

        judge = mocker.MagicMock()
        judge.judge_id = "judge-1"

        # Get handler twice
        handler1 = orchestrator._get_handler_for_judge("ragas", judge)
        handler2 = orchestrator._get_handler_for_judge("ragas", judge)

        # Should be same instance
        assert handler1 is handler2
        # Factory called only once
        assert handler_factory.call_count == 1

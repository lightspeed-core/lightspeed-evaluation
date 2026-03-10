# pylint: disable=redefined-outer-name,too-few-public-methods,protected-access

"""Unit tests for JudgeOrchestrator - multi-judge evaluation and aggregation."""

import logging
from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import JudgeScore
from lightspeed_evaluation.pipeline.evaluation.judges import JudgeOrchestrator


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
    def mock_llm_manager_with_panel(self, mocker: MockerFixture) -> Any:
        """Create a mock LLM manager with panel config set to max strategy."""
        manager = mocker.MagicMock()
        manager.system_config.judge_panel.aggregation_strategy = "max"
        manager.judge_id = "primary"
        return manager

    @pytest.fixture
    def mock_llm_manager_with_average_strategy(self, mocker: MockerFixture) -> Any:
        """Create a mock LLM manager with panel config set to average strategy."""
        manager = mocker.MagicMock()
        manager.system_config.judge_panel.aggregation_strategy = "average"
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

        score, reason = orchestrator.aggregate_scores(judge_scores)

        assert score == 0.85
        assert reason == "Good answer"

    def test_single_judge_with_none_score(
        self, orchestrator: JudgeOrchestrator
    ) -> None:
        """Single judge with None score returns None and original reason."""
        judge_scores = [
            JudgeScore(
                judge_id="judge-1", score=None, reason="Evaluation error: timeout"
            )
        ]

        score, reason = orchestrator.aggregate_scores(judge_scores)

        assert score is None
        assert reason == "Evaluation error: timeout"

    def test_multiple_judges_returns_max(self, orchestrator: JudgeOrchestrator) -> None:
        """Multiple judges return max score."""
        judge_scores = [
            JudgeScore(judge_id="judge-1", score=0.6, reason="Okay"),
            JudgeScore(judge_id="judge-2", score=0.9, reason="Great"),
            JudgeScore(judge_id="judge-3", score=0.75, reason="Good"),
        ]

        score, reason = orchestrator.aggregate_scores(judge_scores)

        assert score == 0.9
        assert "Max of 3 judges" in reason
        assert "0.900" in reason

    def test_multiple_judges_with_one_failure(
        self, orchestrator: JudgeOrchestrator
    ) -> None:
        """Multiple judges with one failure returns max of valid scores."""
        judge_scores = [
            JudgeScore(judge_id="judge-1", score=0.7, reason="Good"),
            JudgeScore(judge_id="judge-2", score=None, reason="Error"),
            JudgeScore(judge_id="judge-3", score=0.85, reason="Great"),
        ]

        score, reason = orchestrator.aggregate_scores(judge_scores)

        assert score == 0.85
        assert "Max of 2 judges" in reason

    def test_all_judges_failed(self, orchestrator: JudgeOrchestrator) -> None:
        """All judges failing returns None with appropriate message."""
        judge_scores = [
            JudgeScore(judge_id="judge-1", score=None, reason="Error 1"),
            JudgeScore(judge_id="judge-2", score=None, reason="Error 2"),
        ]

        score, reason = orchestrator.aggregate_scores(judge_scores)

        assert score is None
        assert reason == "All judges failed to produce a score"


class TestAggregationStrategyWarning:
    """Tests for aggregation strategy warning deduplication."""

    def test_warning_emitted_once_for_unimplemented_strategy(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning for unimplemented strategy is emitted only once."""
        mock_manager = mocker.MagicMock()
        mock_manager.system_config.judge_panel.aggregation_strategy = "average"
        mock_manager.judge_id = "primary"

        orchestrator = JudgeOrchestrator(
            llm_manager=mock_manager,
            primary_handlers={},
            handler_factory=mocker.MagicMock(),
            status_determiner=lambda s, _: "PASS",
        )

        judge_scores = [
            JudgeScore(judge_id="j1", score=0.8, reason="A"),
            JudgeScore(judge_id="j2", score=0.7, reason="B"),
        ]

        with caplog.at_level(logging.WARNING):
            # Call aggregate multiple times
            orchestrator.aggregate_scores(judge_scores)
            orchestrator.aggregate_scores(judge_scores)
            orchestrator.aggregate_scores(judge_scores)

        # Warning should appear only once
        warning_messages = [
            r.message for r in caplog.records if "not yet implemented" in r.message
        ]
        assert len(warning_messages) == 1
        assert "average" in warning_messages[0]

    def test_no_warning_for_max_strategy(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning emitted when using implemented 'max' strategy."""
        mock_manager = mocker.MagicMock()
        mock_manager.system_config.judge_panel.aggregation_strategy = "max"
        mock_manager.judge_id = "primary"

        orchestrator = JudgeOrchestrator(
            llm_manager=mock_manager,
            primary_handlers={},
            handler_factory=mocker.MagicMock(),
            status_determiner=lambda s, _: "PASS",
        )

        judge_scores = [
            JudgeScore(judge_id="j1", score=0.8, reason="A"),
            JudgeScore(judge_id="j2", score=0.7, reason="B"),
        ]

        with caplog.at_level(logging.WARNING):
            orchestrator.aggregate_scores(judge_scores)

        warning_messages = [
            r.message for r in caplog.records if "not yet implemented" in r.message
        ]
        assert len(warning_messages) == 0


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

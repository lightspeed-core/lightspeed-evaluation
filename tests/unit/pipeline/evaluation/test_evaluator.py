# pylint: disable=protected-access,redefined-outer-name,too-many-arguments,too-many-positional-arguments,too-many-lines, too-many-public-methods

"""Unit tests for pipeline evaluation evaluator module."""

from typing import Optional

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.llm.token_tracker import TokenTracker
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationScope,
    MetricResult,
    TurnData,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import EvaluationError
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator


class TestMetricsEvaluator:
    """Unit tests for MetricsEvaluator."""

    def test_initialization(
        self,
        evaluator: MetricsEvaluator,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
    ) -> None:
        """Test evaluator initialization."""
        assert evaluator.config_loader == config_loader
        assert evaluator.metric_manager == mock_metric_manager
        assert (
            len(evaluator.handlers) == 6
        )  # ragas, deepeval, geval, custom, script, nlp

    def test_initialization_raises_error_without_config(
        self,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
    ) -> None:
        """Test initialization fails without system config."""
        loader = ConfigLoader()
        loader.system_config = None

        with pytest.raises(RuntimeError, match="Uninitialized system_config"):
            MetricsEvaluator(loader, mock_metric_manager, mock_script_manager)

    def test_evaluate_metric_turn_level_pass(self, evaluator: MetricsEvaluator) -> None:
        """Test evaluating turn-level metric that passes."""
        evaluator.handlers["ragas"].evaluate.return_value = (0.85, "Good faithfulness")

        turn_data = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Context"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "PASS"
        assert result.score == 0.85
        assert result.threshold == 0.7
        assert result.reason == "Good faithfulness"
        assert result.conversation_group_id == "test_conv"
        assert result.turn_id == "1"
        assert result.metric_identifier == "ragas:faithfulness"
        assert result.query == "What is Python?"
        assert result.response == "Python is a programming language."
        assert result.contexts == '["Context"]'
        # Verify execution_time is populated
        assert result.execution_time >= 0.0
        assert result.evaluation_latency >= 0.0
        assert result.agent_latency >= 0.0
        assert result.execution_time == result.evaluation_latency + result.agent_latency

    def test_evaluate_metric_turn_level_fail(self, evaluator: MetricsEvaluator) -> None:
        """Test evaluating turn-level metric that fails."""
        evaluator.handlers["ragas"].evaluate.return_value = (
            0.3,
            "Low faithfulness score",
        )

        turn_data = TurnData(
            turn_id="1", query="Query", response="Response", contexts=["Context"]
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "FAIL"
        assert result.score == 0.3
        assert result.threshold == 0.7

    def test_evaluate_metric_missing_required_data_returns_error(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """When required data is missing or empty, return ERROR and skip metric processing."""
        mock_ragas = evaluator.handlers["ragas"]

        turn_data = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            contexts=None,  # Required for faithfulness but missing
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert result.score is None
        assert "contexts" in result.reason
        assert "missing or empty" in result.reason
        mock_ragas.evaluate.assert_not_called()

    def test_evaluate_metric_conversation_level(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test evaluating conversation-level metric."""
        evaluator.handlers["deepeval"].evaluate.return_value = (
            0.75,
            "Complete conversation",
        )

        turn1 = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            api_input_tokens=10,
            api_output_tokens=5,
        )
        turn2 = TurnData(
            turn_id="2",
            query="Q2",
            response="R2",
            api_input_tokens=20,
            api_output_tokens=15,
        )
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )
        request = EvaluationRequest.for_conversation(
            conv_data, "deepeval:conversation_completeness"
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "PASS"
        assert result.score == 0.75
        assert result.turn_id is None  # Conversation-level
        assert result.api_input_tokens == 30
        assert result.api_output_tokens == 20

    def test_evaluate_metric_unsupported_framework(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test unsupported framework returns ERROR and aggregates API tokens across turns."""
        turn1 = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            api_input_tokens=5,
            api_output_tokens=2,
        )
        turn2 = TurnData(
            turn_id="2",
            query="Q2",
            response="R2",
            api_input_tokens=7,
            api_output_tokens=3,
        )
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )
        request = EvaluationRequest.for_conversation(conv_data, "unknown:metric")

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert result.score is None
        assert "Unsupported framework" in result.reason
        assert result.api_input_tokens == 12
        assert result.api_output_tokens == 5

    def test_evaluate_metric_returns_none_score(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test handling when metric evaluation returns None score."""
        evaluator.handlers["ragas"].evaluate.return_value = (None, "Evaluation failed")

        turn_data = TurnData(turn_id="1", query="Q", response="R", contexts=["C"])
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert result.score is None
        assert result.reason == "Evaluation failed"

    def test_evaluate_metric_exception_handling(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test exception handling during metric evaluation.

        Note: Even on error, turn data fields (query, response, contexts) should be
        preserved in the result for debugging and analysis purposes.
        """
        evaluator.handlers["ragas"].evaluate.side_effect = EvaluationError(
            "Unexpected error"
        )

        turn_data = TurnData(turn_id="1", query="Q", response="R", contexts=["C"])
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert "Evaluation error" in result.reason
        assert "Unexpected error" in result.reason
        # Turn data should be preserved even on error for debugging
        assert result.query == "Q"
        assert result.response == "R"
        assert result.contexts == '["C"]'
        assert result.expected_response is None

    def test_evaluate_metric_skip_script_when_api_disabled(
        self, evaluator: MetricsEvaluator, config_loader: ConfigLoader
    ) -> None:
        """Test script metrics are skipped when API is disabled."""
        assert config_loader.system_config is not None
        config_loader.system_config.agents = None

        turn_data = TurnData(turn_id="1", query="Q", response="R")
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "script:action_eval", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert (
            result is None
        )  # Should return None when API is disabled for script metrics

    def test_determine_status_with_threshold(self, evaluator: MetricsEvaluator) -> None:
        """Test _determine_status method."""
        assert evaluator._determine_status(0.8, 0.7) == "PASS"
        assert evaluator._determine_status(0.7, 0.7) == "PASS"  # Equal passes
        assert evaluator._determine_status(0.6, 0.7) == "FAIL"

    def test_determine_status_without_threshold(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test _determine_status uses default 0.5 when threshold is None."""
        assert evaluator._determine_status(0.6, None) == "PASS"
        assert evaluator._determine_status(0.4, None) == "FAIL"

    @pytest.mark.parametrize(
        "metric_identifier",
        ["ragas:context_recall", "custom:answer_correctness", "nlp:rouge"],
    )
    def test_evaluate_with_expected_response_list(
        self, evaluator: MetricsEvaluator, metric_identifier: str
    ) -> None:
        """Test _evaluate_wrapper() with list expected_response for metric that requires it."""
        framework = metric_identifier.split(":")[0]
        evaluator.handlers[framework].evaluate.side_effect = [
            (0.3, "Low score"),
            (0.85, "High score"),
        ]

        turn_data = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            expected_response=["A", "B"],
            contexts=["C"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(conv_data, metric_identifier, 0, turn_data)
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        metric_result = evaluator._evaluate_wrapper(request, scope, 0.7)

        assert metric_result.score == 0.85
        assert metric_result.reason == "High score"
        assert metric_result.result == "PASS"
        assert evaluator.handlers[framework].evaluate.call_count == 2

    def test_evaluate_with_expected_response_list_fail(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test _evaluate_wrapper() with list expected_response for metric that requires it."""
        scores_reasons = [(0.3, "Score 1"), (0.65, "Score 2"), (0.45, "Score 3")]
        evaluator.handlers["ragas"].evaluate.side_effect = scores_reasons

        turn_data = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            expected_response=["A", "B", "D"],
            contexts=["C"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:context_recall", 0, turn_data
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        metric_result = evaluator._evaluate_wrapper(request, scope, 0.7)
        reason_combined = "\n".join(
            [f"{score}; {reason}" for score, reason in scores_reasons]
        )

        assert metric_result.score == 0.65
        assert metric_result.reason == reason_combined
        assert metric_result.result == "FAIL"
        assert evaluator.handlers["ragas"].evaluate.call_count == 3

    def test_evaluate_with_expected_response_string(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test _evaluate_wrapper() with string expected_response."""
        evaluator.handlers["ragas"].evaluate.return_value = (0.85, "Good score")

        turn_data = TurnData(
            turn_id="1", query="Q", response="R", expected_response="A", contexts=["C"]
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:context_recall", 0, turn_data
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        metric_result = evaluator._evaluate_wrapper(request, scope, 0.7)

        assert metric_result.score == 0.85
        assert metric_result.reason == "Good score"
        assert metric_result.result == "PASS"
        assert evaluator.handlers["ragas"].evaluate.call_count == 1

    @pytest.mark.parametrize(
        "metric_identifier", ["ragas:faithfulness", "geval:technical_accuracy"]
    )
    @pytest.mark.parametrize(
        "expected_response",
        [None, "string", ["string1", "string2"]],
        ids=["none", "string", "string_list"],
    )
    def test_evaluate_with_expected_response_not_needed(
        self,
        evaluator: MetricsEvaluator,
        metric_identifier: str,
        expected_response: str | list[str] | None,
    ) -> None:
        """Test _evaluate_wrapper() with metric that does not require expected_response."""
        framework = metric_identifier.split(":")[0]
        evaluator.handlers[framework].evaluate.side_effect = [
            (0.3, "Low score"),
            (0.85, "High score"),
        ]

        turn_data = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            expected_response=expected_response,
            contexts=["C"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(conv_data, metric_identifier, 0, turn_data)
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        metric_result = evaluator._evaluate_wrapper(request, scope, 0.7)

        assert metric_result.score == 0.3
        assert metric_result.reason == "Low score"
        assert metric_result.result == "FAIL"
        assert evaluator.handlers[framework].evaluate.call_count == 1

    def test_evaluate_multiple_expected_responses_error_preserves_tokens(
        self, evaluator: MetricsEvaluator, mocker: MockerFixture
    ) -> None:
        """Test token preservation when error occurs during multiple expected responses evaluation.

        Scenario: First iteration succeeds with tokens, second iteration fails.
        Expected: Error result should preserve tokens from first iteration.
        """
        evaluator.handlers["ragas"].evaluate.side_effect = [
            (0.3, "First iteration failed threshold"),
            EvaluationError("LLM error in second iteration"),
        ]

        original_evaluate = evaluator._evaluate

        def mock_evaluate_with_tokens(
            request: EvaluationRequest,
            scope: EvaluationScope,
            token_tracker: TokenTracker,
            threshold: Optional[float],
        ) -> MetricResult:
            result = original_evaluate(request, scope, token_tracker, threshold)
            result.judge_llm_input_tokens = 150
            result.judge_llm_output_tokens = 50
            return result

        mocker.patch.object(
            evaluator, "_evaluate", side_effect=mock_evaluate_with_tokens
        )

        turn_data = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            expected_response=["A", "B"],
            contexts=["C"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:context_recall", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.score == 0.3  # From first iteration
        assert result.judge_llm_input_tokens == 300  # 150 + 150
        assert result.judge_llm_output_tokens == 100  # 50 + 50
        assert "error" in result.reason.lower()

    def test_evaluate_single_path_error_preserves_tokens(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test token preservation when error occurs in single evaluation path.

        Scenario: Single evaluation call fails but tokens were tracked.
        Expected: Error result should preserve any tokens captured.
        """
        evaluator.handlers["ragas"].evaluate.side_effect = EvaluationError(
            "LLM connection failed"
        )

        turn_data = TurnData(
            turn_id="1", query="Q", response="R", expected_response="A", contexts=["C"]
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert "LLM connection failed" in result.reason
        assert result.judge_llm_input_tokens >= 0
        assert result.judge_llm_output_tokens >= 0

    def test_multiple_expected_responses_error_no_double_counting(
        self, evaluator: MetricsEvaluator, mocker: MockerFixture
    ) -> None:
        """Test token counts use deltas not cumulative totals when error on iteration 2+."""
        evaluator.handlers["ragas"].evaluate.side_effect = [
            (0.3, "First iteration"),
            EvaluationError("Second iteration failed"),
        ]

        call_count = [0]

        def mock_evaluate_with_tokens(
            _request: EvaluationRequest,
            _scope: EvaluationScope,
            token_tracker: TokenTracker,
            threshold: Optional[float],
        ) -> MetricResult:
            call_count[0] += 1
            if call_count[0] == 1:
                token_tracker.add_judge_tokens(100, 50)
                token_tracker.add_embedding_tokens(20)
                return MetricResult(
                    result="FAIL",
                    score=0.3,
                    threshold=threshold,
                    reason="First iteration",
                    judge_llm_input_tokens=100,
                    judge_llm_output_tokens=50,
                    embedding_tokens=20,
                )
            token_tracker.add_judge_tokens(150, 75)
            token_tracker.add_embedding_tokens(30)
            raise EvaluationError("Second iteration failed")

        mocker.patch.object(
            evaluator, "_evaluate", side_effect=mock_evaluate_with_tokens
        )

        turn_data = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            expected_response=["A", "B"],
            contexts=["C"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:context_recall", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert "Second iteration failed" in result.reason
        assert result.judge_llm_input_tokens == 250  # 100+150
        assert result.judge_llm_output_tokens == 125  # 50+75
        assert result.embedding_tokens == 50  # 20+30

    def test_execution_time_calculation(self, evaluator: MetricsEvaluator) -> None:
        """Test execution_time is correctly calculated as evaluation_latency + agent_latency."""
        mock_ragas = evaluator.handlers["ragas"]
        mock_ragas.evaluate.return_value = (0.85, "Good score")

        turn_data = TurnData(
            turn_id="1", query="Q", response="R", contexts=["C"], agent_latency=1.5
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.agent_latency == 1.5
        assert result.evaluation_latency > 0.0
        assert result.execution_time == result.evaluation_latency + result.agent_latency
        assert result.execution_time >= 1.5

    def test_execution_time_in_error_result(self, evaluator: MetricsEvaluator) -> None:
        """Test execution_time is populated even in ERROR results."""
        turn_data = TurnData(turn_id="1", query="Q", response="R", agent_latency=2.0)
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(conv_data, "unknown:metric", 0, turn_data)

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert result.agent_latency == 2.0
        assert result.evaluation_latency > 0.0
        assert result.execution_time == result.evaluation_latency + result.agent_latency
        assert result.execution_time >= 2.0

    def test_execution_time_conversation_level_sums_agent_latency(
        self, evaluator: MetricsEvaluator
    ) -> None:
        """Test execution_time uses the summed agent_latency for conversation-level metrics."""
        mock_deepeval = evaluator.handlers["deepeval"]
        mock_deepeval.evaluate.return_value = (0.75, "Good conversation")

        turn1 = TurnData(turn_id="1", query="Q1", response="R1", agent_latency=1.0)
        turn2 = TurnData(turn_id="2", query="Q2", response="R2", agent_latency=3.0)
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )
        request = EvaluationRequest.for_conversation(
            conv_data, "deepeval:conversation_completeness"
        )

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.agent_latency == 4.0
        assert result.evaluation_latency > 0.0
        assert result.execution_time == result.evaluation_latency + result.agent_latency
        assert result.execution_time >= 4.0

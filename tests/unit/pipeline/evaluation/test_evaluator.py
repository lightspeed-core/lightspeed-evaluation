"""Unit tests for pipeline evaluation evaluator module."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.llm.custom import TokenTracker
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationScope,
    TurnData,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator


class TestMetricsEvaluator:
    """Unit tests for MetricsEvaluator."""

    def test_initialization(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test evaluator initialization."""
        # Mock the metric handlers
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.NLPMetrics")

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

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

    def test_evaluate_metric_turn_level_pass(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test evaluating turn-level metric that passes."""
        # Mock the handlers
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )

        mock_ragas = mocker.Mock()
        mock_ragas.evaluate.return_value = (0.85, "Good faithfulness")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics",
            return_value=mock_ragas,
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

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

    def test_evaluate_metric_turn_level_fail(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test evaluating turn-level metric that fails."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )

        mock_ragas = mocker.Mock()
        mock_ragas.evaluate.return_value = (0.3, "Low faithfulness score")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics",
            return_value=mock_ragas,
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
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

    def test_evaluate_metric_conversation_level(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test evaluating conversation-level metric."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics")

        mock_deepeval = mocker.Mock()
        mock_deepeval.evaluate.return_value = (0.75, "Complete conversation")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics",
            return_value=mock_deepeval,
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

        turn_data = TurnData(turn_id="1", query="Q", response="R")
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        request = EvaluationRequest.for_conversation(
            conv_data, "deepeval:conversation_completeness"
        )

        result = evaluator.evaluate_metric(request)
        assert result is not None
        assert result.result == "PASS"
        assert result.score == 0.75
        assert result.turn_id is None  # Conversation-level

    def test_evaluate_metric_unsupported_framework(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test evaluating metric with unsupported framework."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

        turn_data = TurnData(turn_id="1", query="Q", response="R")
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        request = EvaluationRequest.for_turn(conv_data, "unknown:metric", 0, turn_data)

        result = evaluator.evaluate_metric(request)

        assert result is not None
        assert result.result == "ERROR"
        assert result.score is None
        assert "Unsupported framework" in result.reason

    def test_evaluate_metric_returns_none_score(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test handling when metric evaluation returns None score."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )

        mock_ragas = mocker.Mock()
        mock_ragas.evaluate.return_value = (None, "Evaluation failed")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics",
            return_value=mock_ragas,
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

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
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test exception handling during metric evaluation."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )

        mock_ragas = mocker.Mock()
        mock_ragas.evaluate.side_effect = Exception("Unexpected error")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics",
            return_value=mock_ragas,
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
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

        assert result.query == "Q"
        assert result.response == "R"
        assert result.contexts is None
        assert result.expected_response is None

    def test_evaluate_metric_skip_script_when_api_disabled(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test script metrics are skipped when API is disabled."""
        assert config_loader.system_config is not None
        config_loader.system_config.api.enabled = False

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

        turn_data = TurnData(turn_id="1", query="Q", response="R")
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        request = EvaluationRequest.for_turn(
            conv_data, "script:action_eval", 0, turn_data
        )

        result = evaluator.evaluate_metric(request)

        # Should return None when API is disabled for script metrics
        assert result is None

    def test_determine_status_with_threshold(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test _determine_status method."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

        # Test PASS
        assert (
            evaluator._determine_status(0.8, 0.7)  # pylint: disable=protected-access
            == "PASS"
        )
        assert (
            evaluator._determine_status(0.7, 0.7)  # pylint: disable=protected-access
            == "PASS"
        )  # Equal passes

        # Test FAIL
        assert (
            evaluator._determine_status(0.6, 0.7)  # pylint: disable=protected-access
            == "FAIL"
        )

    def test_determine_status_without_threshold(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test _determine_status uses default 0.5 when threshold is None."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

        # Should use 0.5 as default
        assert (
            evaluator._determine_status(0.6, None)  # pylint: disable=protected-access
            == "PASS"
        )
        assert (
            evaluator._determine_status(0.4, None)  # pylint: disable=protected-access
            == "FAIL"
        )

    def _setup_evaluate_test(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
        mock_return: tuple[float, str] | list[tuple[float, str]],
    ) -> tuple[MetricsEvaluator, dict]:
        """Helper to setup common mocks for _evaluate_wrapper() tests.

        Returns:
            tuple: (evaluator, mock_handlers) where mock_handlers is a dict with keys:
                   'ragas', 'geval', 'custom', 'script', 'nlp'
        """
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )

        # Create a helper to setup mock with return values
        def create_mock_handler(  # type: ignore[no-untyped-def]
            mocker: MockerFixture,
            mock_return: tuple[float, str] | list[tuple[float, str]],
        ):
            mock = mocker.Mock()
            if isinstance(mock_return, list):
                mock.evaluate.side_effect = mock_return
            else:
                mock.evaluate.return_value = mock_return
            return mock

        # Setup all handler mocks
        mock_ragas = create_mock_handler(mocker, mock_return)
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.RagasMetrics",
            return_value=mock_ragas,
        )

        mock_deepeval = create_mock_handler(mocker, mock_return)
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.DeepEvalMetrics",
            return_value=mock_deepeval,
        )

        mock_custom = create_mock_handler(mocker, mock_return)
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.CustomMetrics",
            return_value=mock_custom,
        )

        mock_nlp = create_mock_handler(mocker, mock_return)
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.NLPMetrics",
            return_value=mock_nlp,
        )

        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.ScriptEvalMetrics"
        )

        evaluator = MetricsEvaluator(
            config_loader, mock_metric_manager, mock_script_manager
        )

        # Return evaluator and dict of all mocks
        mock_handlers = {
            "ragas": mock_ragas,
            "geval": mock_deepeval,
            "custom": mock_custom,
            "nlp": mock_nlp,
        }

        return evaluator, mock_handlers

    @pytest.mark.parametrize(
        "metric_identifier",
        ["ragas:context_recall", "custom:answer_correctness", "nlp:rouge"],
    )
    def test_evaluate_with_expected_response_list(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
        metric_identifier: str,
    ) -> None:
        """Test _evaluate_wrapper() with list expected_response for metric that requires it."""
        evaluator, mock_handlers = self._setup_evaluate_test(
            config_loader,
            mock_metric_manager,
            mock_script_manager,
            mocker,
            [(0.3, "Low score"), (0.85, "High score")],
        )

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

        metric_result = evaluator._evaluate_wrapper(  # pylint: disable=protected-access
            request, scope, 0.7
        )

        assert metric_result.score == 0.85
        assert metric_result.reason == "High score"
        assert metric_result.result == "PASS"

        # Check the appropriate handler was called based on metric framework
        framework = metric_identifier.split(":")[0]
        assert mock_handlers[framework].evaluate.call_count == 2

    def test_evaluate_with_expected_response_list_fail(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test _evaluate_wrapper() with list expected_response for metric that requires it."""
        scores_reasons = [(0.3, "Score 1"), (0.65, "Score 2"), (0.45, "Score 3")]
        evaluator, mock_handlers = self._setup_evaluate_test(
            config_loader,
            mock_metric_manager,
            mock_script_manager,
            mocker,
            scores_reasons,
        )

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

        metric_result = evaluator._evaluate_wrapper(  # pylint: disable=protected-access
            request, scope, 0.7
        )
        reason_combined = "\n".join(
            [f"{score}; {reason}" for score, reason in scores_reasons]
        )

        assert metric_result.score == 0.65
        assert metric_result.reason == reason_combined
        assert metric_result.result == "FAIL"
        assert mock_handlers["ragas"].evaluate.call_count == 3

    def test_evaluate_with_expected_response_string(
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
    ) -> None:
        """Test _evaluate_wrapper() with string expected_response."""
        evaluator, mock_handlers = self._setup_evaluate_test(
            config_loader,
            mock_metric_manager,
            mock_script_manager,
            mocker,
            (0.85, "Good score"),
        )

        turn_data = TurnData(
            turn_id="1", query="Q", response="R", expected_response="A", contexts=["C"]
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:context_recall", 0, turn_data
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        metric_result = evaluator._evaluate_wrapper(  # pylint: disable=protected-access
            request, scope, 0.7
        )

        assert metric_result.score == 0.85
        assert metric_result.reason == "Good score"
        assert metric_result.result == "PASS"
        assert mock_handlers["ragas"].evaluate.call_count == 1

    @pytest.mark.parametrize(
        "metric_identifier", ["ragas:faithfulness", "geval:technical_accuracy"]
    )
    @pytest.mark.parametrize(
        "expected_response",
        [None, "string", ["string1", "string2"]],
        ids=["none", "string", "string_list"],
    )
    def test_evaluate_with_expected_response_not_needed(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        config_loader: ConfigLoader,
        mock_metric_manager: MetricManager,
        mock_script_manager: ScriptExecutionManager,
        mocker: MockerFixture,
        metric_identifier: str,
        expected_response: str | list[str] | None,
    ) -> None:
        """Test _evaluate_wrapper() with metric that does not require expected_response."""
        evaluator, mock_handlers = self._setup_evaluate_test(
            config_loader,
            mock_metric_manager,
            mock_script_manager,
            mocker,
            [(0.3, "Low score"), (0.85, "High score")],
        )

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

        metric_result = evaluator._evaluate_wrapper(  # pylint: disable=protected-access
            request, scope, 0.7
        )

        assert metric_result.score == 0.3
        assert metric_result.reason == "Low score"
        assert metric_result.result == "FAIL"

        # Check the appropriate handler was called based on metric
        framework = metric_identifier.split(":")[0]
        assert mock_handlers[framework].evaluate.call_count == 1


class TestTokenTracker:
    """Unit tests for TokenTracker class."""

    def test_token_tracker_initialization(self) -> None:
        """Test TokenTracker initializes with zero counts."""
        tracker = TokenTracker()
        input_tokens, output_tokens = tracker.get_counts()
        assert input_tokens == 0
        assert output_tokens == 0

    def test_token_tracker_get_counts_returns_tuple(self) -> None:
        """Test get_counts returns a tuple."""
        tracker = TokenTracker()
        result = tracker.get_counts()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_token_tracker_reset(self) -> None:
        """Test reset clears token counts."""
        tracker = TokenTracker()
        tracker.input_tokens = 100
        tracker.output_tokens = 50
        tracker.reset()
        assert tracker.get_counts() == (0, 0)

    def test_token_tracker_start_stop(self) -> None:
        """Test start and stop methods."""
        tracker = TokenTracker()
        tracker.start()
        assert tracker._callback_registered is True  # pylint: disable=protected-access
        tracker.stop()
        assert tracker._callback_registered is False  # pylint: disable=protected-access

    def test_token_tracker_double_start(self) -> None:
        """Test calling start twice doesn't register callback twice."""
        tracker = TokenTracker()
        tracker.start()
        tracker.start()  # Should not fail
        assert tracker._callback_registered is True  # pylint: disable=protected-access
        tracker.stop()

    def test_token_tracker_double_stop(self) -> None:
        """Test calling stop twice doesn't fail."""
        tracker = TokenTracker()
        tracker.start()
        tracker.stop()
        tracker.stop()  # Should not fail
        assert tracker._callback_registered is False  # pylint: disable=protected-access

    def test_token_tracker_independent_instances(self) -> None:
        """Test multiple TokenTracker instances are independent."""
        tracker1 = TokenTracker()
        tracker2 = TokenTracker()
        tracker1.input_tokens = 100
        tracker1.output_tokens = 50
        tracker2.input_tokens = 200
        tracker2.output_tokens = 100
        assert tracker1.get_counts() == (100, 50)
        assert tracker2.get_counts() == (200, 100)

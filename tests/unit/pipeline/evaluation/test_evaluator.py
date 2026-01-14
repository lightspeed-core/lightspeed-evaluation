"""Unit tests for pipeline evaluation evaluator module."""

import pytest

from lightspeed_evaluation.core.llm.custom import TokenTracker
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationScope,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator


@pytest.fixture
def config_loader(mocker):
    """Create a mock config loader with system config."""
    loader = mocker.Mock(spec=ConfigLoader)

    config = SystemConfig()
    config.default_turn_metrics_metadata = {
        "ragas:faithfulness": {"threshold": 0.7, "default": True},
        "custom:answer_correctness": {"threshold": 0.8, "default": False},
    }
    config.default_conversation_metrics_metadata = {
        "deepeval:conversation_completeness": {"threshold": 0.6, "default": True},
    }
    config.api.enabled = True

    loader.system_config = config
    return loader


@pytest.fixture
def mock_metric_manager(mocker):
    """Create a mock metric manager."""
    from lightspeed_evaluation.core.metrics.manager import MetricManager

    manager = mocker.Mock(spec=MetricManager)

    def get_threshold(metric_id, level, conv_data=None, turn_data=None):
        thresholds = {
            "ragas:faithfulness": 0.7,
            "custom:answer_correctness": 0.8,
            "deepeval:conversation_completeness": 0.6,
        }
        return thresholds.get(metric_id, 0.5)

    manager.get_effective_threshold.side_effect = get_threshold
    return manager


@pytest.fixture
def mock_script_manager(mocker):
    """Create a mock script execution manager."""
    from lightspeed_evaluation.core.script import ScriptExecutionManager

    manager = mocker.Mock(spec=ScriptExecutionManager)
    return manager


class TestMetricsEvaluator:
    """Unit tests for MetricsEvaluator."""

    def test_initialization(
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        self, mock_metric_manager, mock_script_manager
    ):
        """Test initialization fails without system config."""
        loader = ConfigLoader()
        loader.system_config = None

        with pytest.raises(RuntimeError, match="Uninitialized system_config"):
            MetricsEvaluator(loader, mock_metric_manager, mock_script_manager)

    def test_evaluate_metric_turn_level_pass(
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
        """Test script metrics are skipped when API is disabled."""
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
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        assert evaluator._determine_status(0.8, 0.7) == "PASS"
        assert evaluator._determine_status(0.7, 0.7) == "PASS"  # Equal passes

        # Test FAIL
        assert evaluator._determine_status(0.6, 0.7) == "FAIL"

    def test_determine_status_without_threshold(
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
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
        assert evaluator._determine_status(0.6, None) == "PASS"
        assert evaluator._determine_status(0.4, None) == "FAIL"

    def _setup_evaluate_test(
        self,
        config_loader,
        mock_metric_manager,
        mock_script_manager,
        mocker,
        mock_return,
    ):
        """Helper to setup common mocks for evaluate() tests."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.evaluator.LLMManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.evaluator.EmbeddingManager"
        )
        mock_ragas = mocker.Mock()
        if isinstance(mock_return, list):
            mock_ragas.evaluate.side_effect = mock_return
        else:
            mock_ragas.evaluate.return_value = mock_return
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
        return evaluator, mock_ragas

    def test_evaluate_with_expected_response_list(
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
        """Test evaluate() with list expected_response for metric that requires it."""
        evaluator, mock_ragas = self._setup_evaluate_test(
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
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:context_recall", 0, turn_data
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = evaluator.evaluate(request, scope, 0.7)

        assert score == 0.85 and reason == "High score"
        assert mock_ragas.evaluate.call_count == 2

    def test_evaluate_with_expected_response_string(
        self, config_loader, mock_metric_manager, mock_script_manager, mocker
    ):
        """Test evaluate() with string expected_response."""
        evaluator, mock_ragas = self._setup_evaluate_test(
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

        score, reason = evaluator.evaluate(request, scope, 0.7)

        assert score == 0.85 and reason == "Good score"
        assert mock_ragas.evaluate.call_count == 1

    @pytest.mark.parametrize(
        "expected_response",
        [None, "string", ["string1", "string2"]],
        ids=["none", "string", "string_list"],
    )
    def test_evaluate_with_expected_response_not_needed(
        self,
        config_loader,
        mock_metric_manager,
        mock_script_manager,
        mocker,
        expected_response,
    ):
        """Test evaluate() with metric that does not require expected_response."""
        evaluator, mock_ragas = self._setup_evaluate_test(
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
        request = EvaluationRequest.for_turn(
            conv_data, "ragas:faithfulness", 0, turn_data
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = evaluator.evaluate(request, scope, 0.7)

        assert score == 0.3 and reason == "Low score"
        assert mock_ragas.evaluate.call_count == 1


class TestTokenTracker:
    """Unit tests for TokenTracker class."""

    def test_token_tracker_initialization(self):
        """Test TokenTracker initializes with zero counts."""
        tracker = TokenTracker()
        input_tokens, output_tokens = tracker.get_counts()
        assert input_tokens == 0
        assert output_tokens == 0

    def test_token_tracker_get_counts_returns_tuple(self):
        """Test get_counts returns a tuple."""
        tracker = TokenTracker()
        result = tracker.get_counts()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_token_tracker_reset(self):
        """Test reset clears token counts."""
        tracker = TokenTracker()
        tracker.input_tokens = 100
        tracker.output_tokens = 50
        tracker.reset()
        assert tracker.get_counts() == (0, 0)

    def test_token_tracker_start_stop(self):
        """Test start and stop methods."""
        tracker = TokenTracker()
        tracker.start()
        assert tracker._callback_registered is True
        tracker.stop()
        assert tracker._callback_registered is False

    def test_token_tracker_double_start(self):
        """Test calling start twice doesn't register callback twice."""
        tracker = TokenTracker()
        tracker.start()
        tracker.start()  # Should not fail
        assert tracker._callback_registered is True
        tracker.stop()

    def test_token_tracker_double_stop(self):
        """Test calling stop twice doesn't fail."""
        tracker = TokenTracker()
        tracker.start()
        tracker.stop()
        tracker.stop()  # Should not fail
        assert tracker._callback_registered is False

    def test_token_tracker_independent_instances(self):
        """Test multiple TokenTracker instances are independent."""
        tracker1 = TokenTracker()
        tracker2 = TokenTracker()
        tracker1.input_tokens = 100
        tracker1.output_tokens = 50
        tracker2.input_tokens = 200
        tracker2.output_tokens = 100
        assert tracker1.get_counts() == (100, 50)
        assert tracker2.get_counts() == (200, 100)

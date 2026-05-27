# pylint: disable=protected-access, unused-variable, too-few-public-methods
"""Unit tests for DeepEval metrics handler."""

import logging
from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
from lightspeed_evaluation.core.models import EvaluationScope


class TestDeepEvalMetrics:
    """Tests for DeepEvalMetrics class."""

    def test_evaluate_metric_none_score_warning(
        self,
        deepeval_metrics: DeepEvalMetrics,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that None score from metric produces warning."""

        # Create a mock metric that returns None score
        class TestMetric:
            """Test metric spec for mocking."""

            def measure(self, test_case: Any) -> None:
                """Measure method required by DeepEval metrics."""

        mock_metric = mocker.MagicMock(spec=TestMetric)
        mock_metric.score = None
        mock_metric.reason = "LLM judge failed to parse response"

        # Create a mock test case
        mock_test_case = mocker.MagicMock()

        with caplog.at_level(logging.WARNING):
            _, reason = deepeval_metrics._evaluate_metric(mock_metric, mock_test_case)

        # Verify reason is preserved
        assert reason == "LLM judge failed to parse response"

        # Verify warning was logged
        assert "TestMetric metric returned None score" in caplog.text
        assert "rate limiting, timeout" in caplog.text

    def test_evaluate_metric_with_valid_score(
        self, deepeval_metrics: DeepEvalMetrics, mocker: MockerFixture
    ) -> None:
        """Test that valid score is returned as-is."""
        mock_metric = mocker.MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = "Good response quality"

        mock_test_case = mocker.MagicMock()

        score, reason = deepeval_metrics._evaluate_metric(mock_metric, mock_test_case)

        assert score == 0.85
        assert reason == "Good response quality"

    def test_evaluate_metric_without_reason(
        self, deepeval_metrics: DeepEvalMetrics, mocker: MockerFixture
    ) -> None:
        """Test that metric without reason gets default reason."""
        mock_metric = mocker.MagicMock()
        mock_metric.score = 0.75
        # Simulate metric without reason attribute
        del mock_metric.reason

        mock_test_case = mocker.MagicMock()

        score, reason = deepeval_metrics._evaluate_metric(mock_metric, mock_test_case)

        assert score == 0.75
        assert "Score: 0.75" in reason

    def test_evaluate_metric_none_score_without_reason(
        self,
        deepeval_metrics: DeepEvalMetrics,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test None score without reason gets appropriate default message."""
        mock_metric = mocker.MagicMock()
        mock_metric.score = None
        # Simulate metric without reason
        del mock_metric.reason

        mock_test_case = mocker.MagicMock()

        with caplog.at_level(logging.WARNING):
            _, reason = deepeval_metrics._evaluate_metric(mock_metric, mock_test_case)

        assert reason == "No score returned"

    def test_initialization_with_cache(
        self,
        mock_llm_manager: Any,
        mock_metric_manager: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test initialization when cache is enabled."""
        mock_config = mocker.MagicMock()
        mock_config.cache_enabled = True
        mock_config.cache_dir = "/tmp/test_cache"
        mock_llm_manager.get_config.return_value = mock_config

        mock_litellm = mocker.patch(
            "lightspeed_evaluation.core.metrics.deepeval.litellm"
        )
        mock_litellm.cache = None
        mock_cache_class = mocker.patch(
            "lightspeed_evaluation.core.metrics.deepeval.Cache"
        )
        mocker.patch("lightspeed_evaluation.core.metrics.deepeval.DeepEvalLLMManager")
        mocker.patch("lightspeed_evaluation.core.metrics.deepeval.GEvalHandler")

        DeepEvalMetrics(
            llm_manager=mock_llm_manager, metric_manager=mock_metric_manager
        )

        mock_cache_class.assert_called_once()

    def test_evaluate_conversation_completeness(
        self,
        deepeval_metrics: DeepEvalMetrics,
        mock_conv_data: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test conversation completeness evaluation."""
        mock_metric_class = mocker.patch(
            "lightspeed_evaluation.core.metrics.deepeval.ConversationCompletenessMetric"
        )
        mocker.patch.object(deepeval_metrics, "_build_conversational_test_case")
        mocker.patch.object(
            deepeval_metrics, "_evaluate_metric", return_value=(0.85, "Complete")
        )

        score, reason = deepeval_metrics._evaluate_conversation_completeness(
            conv_data=mock_conv_data,
            _turn_idx=None,
            _turn_data=None,
            is_conversation=True,
        )

        assert score == 0.85
        mock_metric_class.assert_called_once()

    def test_evaluate_conversation_relevancy(
        self,
        deepeval_metrics: DeepEvalMetrics,
        mock_conv_data: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test conversation relevancy evaluation."""
        mock_metric_class = mocker.patch(
            "lightspeed_evaluation.core.metrics.deepeval.TurnRelevancyMetric"
        )
        mocker.patch.object(deepeval_metrics, "_build_conversational_test_case")
        mocker.patch.object(
            deepeval_metrics, "_evaluate_metric", return_value=(0.90, "Relevant")
        )

        score, reason = deepeval_metrics._evaluate_conversation_relevancy(
            conv_data=mock_conv_data,
            _turn_idx=None,
            _turn_data=None,
            is_conversation=True,
        )

        assert score == 0.90
        mock_metric_class.assert_called_once()

    def test_evaluate_knowledge_retention(
        self,
        deepeval_metrics: DeepEvalMetrics,
        mock_conv_data: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test knowledge retention evaluation."""
        mock_metric_class = mocker.patch(
            "lightspeed_evaluation.core.metrics.deepeval.KnowledgeRetentionMetric"
        )
        mocker.patch.object(deepeval_metrics, "_build_conversational_test_case")
        mocker.patch.object(
            deepeval_metrics, "_evaluate_metric", return_value=(0.75, "Retained")
        )

        score, reason = deepeval_metrics._evaluate_knowledge_retention(
            conv_data=mock_conv_data,
            _turn_idx=None,
            _turn_data=None,
            is_conversation=True,
        )

        assert score == 0.75
        mock_metric_class.assert_called_once()

    def test_evaluate_standard_metric_exception_handling(
        self,
        deepeval_metrics: DeepEvalMetrics,
        mock_conv_data: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test evaluate handles exceptions from standard metrics."""
        # Set up the exception to be raised
        deepeval_metrics.supported_metrics["conversation_completeness"] = (
            mocker.MagicMock(side_effect=ValueError("Test error"))
        )

        scope = EvaluationScope(turn_idx=None, turn_data=None, is_conversation=True)
        score, reason = deepeval_metrics.evaluate(
            "conversation_completeness", mock_conv_data, scope
        )

        assert score is None
        assert "evaluation failed" in reason
        assert "Test error" in reason

    def test_evaluate_routes_to_geval(
        self,
        deepeval_metrics: DeepEvalMetrics,
        mock_conv_data: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test evaluate routes to GEval handler for custom metrics."""
        mock_geval = mocker.MagicMock()
        mock_geval.evaluate.return_value = (0.92, "Custom")
        deepeval_metrics.geval_handler = mock_geval

        scope = EvaluationScope(turn_idx=None, turn_data=None, is_conversation=True)
        deepeval_metrics.evaluate("geval:custom_metric", mock_conv_data, scope)

        mock_geval.evaluate.assert_called_once_with(
            metric_name="custom_metric",
            conv_data=mock_conv_data,
            _turn_idx=None,
            turn_data=None,
            is_conversation=True,
        )

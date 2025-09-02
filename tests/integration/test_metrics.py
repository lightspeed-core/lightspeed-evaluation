"""Tests for metrics components."""

import os
from unittest.mock import MagicMock, patch

import pytest

from lightspeed_evaluation.core.config import EvaluationData, TurnData
from lightspeed_evaluation.core.llm.manager import LLMConfig, LLMManager
from lightspeed_evaluation.core.metrics.custom import CustomMetrics
from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
from lightspeed_evaluation.core.metrics.ragas import RagasMetrics
from lightspeed_evaluation.core.output.statistics import EvaluationScope

class TestCustomMetrics:
    """Test Custom Metrics functionality."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        manager = MagicMock(spec=LLMManager)
        manager.get_model_name.return_value = "gpt-4o-mini"
        manager.get_litellm_params.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": 300,
            "num_retries": 3,
        }
        return manager

    def test_custom_metrics_initialization(self, mock_llm_manager):
        """Test CustomMetrics initialization."""
        metrics = CustomMetrics(mock_llm_manager)

        assert metrics.model_name == "gpt-4o-mini"
        assert "answer_correctness" in metrics.supported_metrics

    @patch("lightspeed_evaluation.core.metrics.custom.litellm.completion")
    def test_answer_correctness_evaluation(self, mock_completion, mock_llm_manager):
        """Test answer correctness evaluation with expected response."""
        # Mock LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Score: 0.85\nReason: Response accurately describes Python as a programming language"
        )
        mock_completion.return_value = mock_response

        metrics = CustomMetrics(mock_llm_manager)

        turn_data = TurnData(
            turn_id=1,
            query="What is Python?",
            response="Python is a programming language used for web development, data science, and automation.",
            contexts=[{"content": "Python is a high-level programming language."}],
            expected_response="Python is a high-level programming language used for various applications.",
        )

        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = metrics.evaluate("answer_correctness", None, scope)

        assert score == 0.85
        assert "Custom answer correctness" in reason

    def test_score_parsing_different_formats(self, mock_llm_manager):
        """Test parsing scores in different formats."""
        metrics = CustomMetrics(mock_llm_manager)

        # Test different score formats
        test_cases = [
            ("Score: 0.75\nReason: Good", 0.75),
            ("8.5/10 - Excellent response", 0.85),
            ("Rating: 4 out of 5", 0.8),
            ("The score is 90%", 0.9),
        ]

        for response_text, expected_score in test_cases:
            score, reason = metrics._parse_score_response(response_text)
            assert score == expected_score

    def test_unsupported_metric(self, mock_llm_manager):
        """Test evaluation of unsupported metric."""
        metrics = CustomMetrics(mock_llm_manager)

        scope = EvaluationScope(is_conversation=False)
        score, reason = metrics.evaluate("unsupported_metric", None, scope)

        assert score is None
        assert "Unsupported custom metric" in reason

class TestRagasMetrics:
    """Test Ragas Metrics functionality."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        manager = MagicMock(spec=LLMManager)
        manager.get_model_name.return_value = "gpt-4o-mini"
        manager.get_litellm_params.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
        }
        return manager

    @patch("lightspeed_evaluation.core.metrics.ragas.RagasLLMManager")
    def test_ragas_metrics_initialization(
        self, mock_ragas_llm_manager, mock_llm_manager
    ):
        """Test RagasMetrics initialization."""
        metrics = RagasMetrics(mock_llm_manager)

        # Verify that RagasLLMManager was called with correct parameters
        mock_ragas_llm_manager.assert_called_once_with(
            "gpt-4o-mini", mock_llm_manager.get_litellm_params()
        )

        assert "faithfulness" in metrics.supported_metrics
        assert "response_relevancy" in metrics.supported_metrics
        assert "context_recall" in metrics.supported_metrics

    def test_faithfulness_evaluation_with_context(self, mock_llm_manager):
        """Test faithfulness evaluation with proper context data."""
        with patch("lightspeed_evaluation.core.metrics.ragas.RagasLLMManager"):
            metrics = RagasMetrics(mock_llm_manager)

            # Mock the _evaluate_metric method directly
            with patch.object(
                metrics,
                "_evaluate_metric",
                return_value=(0.92, "Ragas faithfulness: 0.92"),
            ):
                turn_data = TurnData(
                    turn_id=1,
                    query="What are the benefits of renewable energy?",
                    response="Renewable energy reduces carbon emissions and provides sustainable power generation.",
                    contexts=[
                        {
                            "content": "Renewable energy sources like solar and wind power help reduce greenhouse gas emissions."
                        },
                        {
                            "content": "Sustainable energy systems provide long-term environmental benefits."
                        },
                    ],
                )

                scope = EvaluationScope(
                    turn_idx=0, turn_data=turn_data, is_conversation=False
                )

                score, reason = metrics.evaluate("faithfulness", None, scope)

                assert score == 0.92
                assert "Ragas faithfulness" in reason

    def test_response_relevancy_evaluation(self, mock_llm_manager):
        """Test response relevancy evaluation."""
        with patch("lightspeed_evaluation.core.metrics.ragas.RagasLLMManager"):
            metrics = RagasMetrics(mock_llm_manager)

            with patch.object(
                metrics,
                "_evaluate_metric",
                return_value=(0.88, "Ragas response relevancy: 0.88"),
            ):
                turn_data = TurnData(
                    turn_id=1,
                    query="How does machine learning work?",
                    response="Machine learning uses algorithms to learn patterns from data and make predictions.",
                )

                scope = EvaluationScope(
                    turn_idx=0, turn_data=turn_data, is_conversation=False
                )

                score, reason = metrics.evaluate("response_relevancy", None, scope)

                assert score == 0.88
                assert "response relevancy" in reason

    def test_conversation_level_metric_error(self, mock_llm_manager):
        """Test error when using turn-level metric for conversation."""
        with patch("lightspeed_evaluation.core.metrics.ragas.RagasLLMManager"):
            metrics = RagasMetrics(mock_llm_manager)

            scope = EvaluationScope(is_conversation=True)
            score, reason = metrics.evaluate("faithfulness", None, scope)

            assert score is None
            assert "turn-level metric" in reason


class TestDeepEvalMetrics:
    """Test DeepEval Metrics functionality."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        manager = MagicMock(spec=LLMManager)
        manager.get_model_name.return_value = "gpt-4o-mini"
        manager.get_litellm_params.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
        }
        return manager

    @patch("lightspeed_evaluation.core.metrics.deepeval.DeepEvalLLMManager")
    def test_deepeval_metrics_initialization(
        self, mock_deepeval_llm_manager, mock_llm_manager
    ):
        """Test DeepEvalMetrics initialization."""
        metrics = DeepEvalMetrics(mock_llm_manager)

        # Verify that DeepEvalLLMManager was called with correct parameters
        mock_deepeval_llm_manager.assert_called_once_with(
            "gpt-4o-mini", mock_llm_manager.get_litellm_params()
        )

        assert "conversation_completeness" in metrics.supported_metrics
        assert "conversation_relevancy" in metrics.supported_metrics
        assert "knowledge_retention" in metrics.supported_metrics

    @patch("lightspeed_evaluation.core.metrics.deepeval.ConversationCompletenessMetric")
    def test_conversation_completeness_evaluation(
        self, mock_metric_class, mock_llm_manager
    ):
        """Test conversation completeness evaluation with multi-turn conversation."""
        # Mock metric instance
        mock_metric = MagicMock()
        mock_metric.score = 0.82
        mock_metric.reason = "Conversation addresses user needs comprehensively"
        mock_metric_class.return_value = mock_metric

        with patch("lightspeed_evaluation.core.metrics.deepeval.DeepEvalLLMManager"):
            metrics = DeepEvalMetrics(mock_llm_manager)

            conv_data = EvaluationData(
                conversation_group_id="customer_support_conv",
                turns=[
                    TurnData(
                        turn_id=1,
                        query="I need help with my account",
                        response="I can help you with your account. What specific issue are you experiencing?",
                    ),
                    TurnData(
                        turn_id=2,
                        query="I can't log in",
                        response="Let me help you reset your password. Please check your email for instructions.",
                    ),
                    TurnData(
                        turn_id=3,
                        query="I got the email, thanks!",
                        response="Great! Is there anything else I can help you with today?",
                    ),
                ],
            )

            scope = EvaluationScope(is_conversation=True)

            score, reason = metrics.evaluate(
                "conversation_completeness", conv_data, scope
            )

            assert score == 0.82
            assert "comprehensively" in reason

    def test_turn_level_metric_error(self, mock_llm_manager):
        """Test error when using conversation-level metric for turn."""
        with patch("lightspeed_evaluation.core.metrics.deepeval.DeepEvalLLMManager"):
            metrics = DeepEvalMetrics(mock_llm_manager)

            scope = EvaluationScope(is_conversation=False)
            score, reason = metrics.evaluate("conversation_completeness", None, scope)

            assert score is None
            assert "conversation-level metric" in reason


class TestMetricsIntegration:
    """Integration tests for metrics components."""

    @pytest.mark.integration
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_metrics_manager_integration(self):
        """Test integration between different metric types."""
        from lightspeed_evaluation.drivers.evaluation import MetricsManager

        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        llm_manager = LLMManager(config)

        metrics_manager = MetricsManager(llm_manager)

        # Verify all handlers are initialized
        assert "ragas" in metrics_manager.handlers
        assert "deepeval" in metrics_manager.handlers
        assert "custom" in metrics_manager.handlers

        # Verify supported frameworks
        frameworks = metrics_manager.get_supported_frameworks()
        assert "ragas" in frameworks
        assert "deepeval" in frameworks
        assert "custom" in frameworks

    def test_evaluation_scope_factory_methods(self):
        """Test EvaluationScope creation for different scenarios."""
        # Turn-level scope
        turn_data = TurnData(
            turn_id=1,
            query="What is machine learning?",
            response="Machine learning is a subset of AI that enables computers to learn from data.",
        )

        turn_scope = EvaluationScope(
            turn_idx=0, turn_data=turn_data, is_conversation=False
        )

        assert turn_scope.turn_idx == 0
        assert turn_scope.turn_data == turn_data
        assert turn_scope.is_conversation is False

        # Conversation-level scope
        conv_scope = EvaluationScope(is_conversation=True)

        assert conv_scope.turn_idx is None
        assert conv_scope.turn_data is None
        assert conv_scope.is_conversation is True

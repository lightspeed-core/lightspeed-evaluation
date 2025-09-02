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


class TestLLMManager:
    """Test LLM Manager functionality."""

    def test_llm_manager_initialization(self):
        """Test LLM Manager initialization with OpenAI."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=512,
            timeout=300,
            num_retries=3,
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config)

            assert manager.config == config
            assert manager.model_name == "gpt-4o-mini"

    def test_llm_manager_missing_api_key(self):
        """Test LLM Manager with missing API key."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception, match="OPENAI_API_KEY"):
                LLMManager(config)

    def test_get_litellm_params(self):
        """Test getting LiteLLM parameters."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=512,
            timeout=300,
            num_retries=3,
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config)
            params = manager.get_litellm_params()

            assert params["model"] == "gpt-4o-mini"
            assert params["temperature"] == 0.0
            assert params["max_tokens"] == 512
            assert params["timeout"] == 300
            assert params["num_retries"] == 3


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


class TestRealWorldScenarios:
    """Test real-world evaluation scenarios."""

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

    @patch("lightspeed_evaluation.core.metrics.custom.litellm.completion")
    def test_technical_documentation_evaluation(
        self, mock_completion, mock_llm_manager
    ):
        """Test evaluation of technical documentation responses."""
        # Mock LLM response for technical accuracy
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Score: 0.88\nReason: Response provides accurate technical information about Kubernetes with proper context and examples"
        )
        mock_completion.return_value = mock_response

        metrics = CustomMetrics(mock_llm_manager)

        turn_data = TurnData(
            turn_id=1,
            query="How do I deploy a microservice using Kubernetes?",
            response="To deploy a microservice using Kubernetes, you need to create a Deployment manifest that specifies the container image, replicas, and resource requirements. Then use kubectl apply to deploy it to your cluster. You'll also need a Service to expose the microservice to other components.",
            contexts=[
                {
                    "content": "Kubernetes deployments manage the lifecycle of containerized applications and ensure desired state."
                },
                {
                    "content": "Services in Kubernetes provide stable network endpoints for accessing pods."
                },
            ],
            expected_response="Deploy microservices in Kubernetes by creating Deployment and Service manifests, then applying them with kubectl.",
        )

        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = metrics.evaluate("answer_correctness", None, scope)

        assert score == 0.88
        assert "technical information" in reason

    def test_customer_support_conversation_evaluation(self, mock_llm_manager):
        """Test evaluation of customer support conversation completeness."""
        with patch("lightspeed_evaluation.core.metrics.deepeval.DeepEvalLLMManager"):
            with patch(
                "lightspeed_evaluation.core.metrics.deepeval.ConversationCompletenessMetric"
            ) as mock_metric_class:
                # Mock metric for customer support scenario
                mock_metric = MagicMock()
                mock_metric.score = 0.91
                mock_metric.reason = "Conversation fully addresses customer issue with clear resolution steps"
                mock_metric_class.return_value = mock_metric

                metrics = DeepEvalMetrics(mock_llm_manager)

                conv_data = EvaluationData(
                    conversation_group_id="customer_billing_issue",
                    turns=[
                        TurnData(
                            turn_id=1,
                            query="I was charged twice for my subscription",
                            response="I understand your concern about the duplicate charge. Let me look into your account to investigate this billing issue.",
                        ),
                        TurnData(
                            turn_id=2,
                            query="When will this be resolved?",
                            response="I can see the duplicate charge in your account. I'm processing a refund right now, which should appear in 3-5 business days.",
                        ),
                        TurnData(
                            turn_id=3,
                            query="Thank you for the help",
                            response="You're welcome! I've sent you a confirmation email with the refund details. Is there anything else I can help you with today?",
                        ),
                    ],
                )

                scope = EvaluationScope(is_conversation=True)

                score, reason = metrics.evaluate(
                    "conversation_completeness", conv_data, scope
                )

                assert score == 0.91
                assert "fully addresses" in reason

    def test_code_explanation_faithfulness(self, mock_llm_manager):
        """Test faithfulness evaluation for code explanation scenarios."""
        with patch("lightspeed_evaluation.core.metrics.ragas.RagasLLMManager"):
            metrics = RagasMetrics(mock_llm_manager)

            with patch.object(
                metrics,
                "_evaluate_metric",
                return_value=(0.94, "Ragas faithfulness: 0.94"),
            ):
                turn_data = TurnData(
                    turn_id=1,
                    query="Explain what this Python function does: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                    response="This is a recursive function that calculates the nth Fibonacci number. It uses the base case where if n is 0 or 1, it returns n directly. Otherwise, it recursively calls itself with n-1 and n-2 and adds the results together.",
                    contexts=[
                        {
                            "content": "The Fibonacci sequence is defined as F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2) for n>1."
                        },
                        {
                            "content": "Recursive functions call themselves with modified parameters until reaching a base case."
                        },
                    ],
                )

                scope = EvaluationScope(
                    turn_idx=0, turn_data=turn_data, is_conversation=False
                )

                score, reason = metrics.evaluate("faithfulness", None, scope)

                assert score == 0.94
                assert "faithfulness" in reason

    @patch("lightspeed_evaluation.core.metrics.custom.litellm.completion")
    def test_multilingual_content_evaluation(self, mock_completion, mock_llm_manager):
        """Test evaluation of responses involving multilingual content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Score: 0.82\nReason: Response correctly explains the concept in English while acknowledging the Spanish term"
        )
        mock_completion.return_value = mock_response

        metrics = CustomMetrics(mock_llm_manager)

        turn_data = TurnData(
            turn_id=1,
            query="What does 'inteligencia artificial' mean and how is it used in technology?",
            response="'Inteligencia artificial' is Spanish for 'artificial intelligence'. It refers to computer systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving. It's widely used in technology for applications like machine learning, natural language processing, and computer vision.",
            contexts=[
                {
                    "content": "Artificial intelligence (AI) encompasses machine learning, neural networks, and automated decision-making systems."
                }
            ],
            expected_response="Inteligencia artificial means artificial intelligence in Spanish, referring to computer systems that simulate human intelligence for various technological applications.",
        )

        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = metrics.evaluate("answer_correctness", None, scope)

        assert score == 0.82
        assert "Spanish term" in reason

    def test_complex_multi_turn_technical_conversation(self, mock_llm_manager):
        """Test evaluation of complex multi-turn technical conversations."""
        with patch("lightspeed_evaluation.core.metrics.deepeval.DeepEvalLLMManager"):
            with patch(
                "lightspeed_evaluation.core.metrics.deepeval.KnowledgeRetentionMetric"
            ) as mock_metric_class:
                mock_metric = MagicMock()
                mock_metric.score = 0.87
                mock_metric.reason = "Good knowledge retention across technical discussion about Docker and Kubernetes"
                mock_metric_class.return_value = mock_metric

                metrics = DeepEvalMetrics(mock_llm_manager)

                conv_data = EvaluationData(
                    conversation_group_id="docker_kubernetes_discussion",
                    turns=[
                        TurnData(
                            turn_id=1,
                            query="What's the difference between Docker and Kubernetes?",
                            response="Docker is a containerization platform that packages applications, while Kubernetes is an orchestration system that manages Docker containers at scale.",
                        ),
                        TurnData(
                            turn_id=2,
                            query="How do they work together in a microservices architecture?",
                            response="In microservices, Docker containers package individual services, and Kubernetes orchestrates these containers, handling deployment, scaling, and service discovery across the cluster.",
                        ),
                        TurnData(
                            turn_id=3,
                            query="What about the networking between these Docker containers you mentioned?",
                            response="Kubernetes provides networking through Services and Ingress controllers. Each Docker container gets an IP address, and Services create stable endpoints for communication between the containerized microservices.",
                        ),
                    ],
                )

                scope = EvaluationScope(is_conversation=True)

                score, reason = metrics.evaluate(
                    "knowledge_retention", conv_data, scope
                )

                assert score == 0.87
                assert "knowledge retention" in reason

    def test_evaluation_with_incomplete_responses(self, mock_llm_manager):
        """Test evaluation of incomplete or partial responses."""
        with patch("lightspeed_evaluation.core.metrics.ragas.RagasLLMManager"):
            metrics = RagasMetrics(mock_llm_manager)

            with patch.object(
                metrics,
                "_evaluate_metric",
                return_value=(0.34, "Ragas response relevancy: 0.34"),
            ):
                turn_data = TurnData(
                    turn_id=1,
                    query="Explain the complete process of photosynthesis including light and dark reactions",
                    response="Photosynthesis uses sunlight.",  # Incomplete response
                )

                scope = EvaluationScope(
                    turn_idx=0, turn_data=turn_data, is_conversation=False
                )

                score, reason = metrics.evaluate("response_relevancy", None, scope)

                assert score == 0.34  # Low score for incomplete response
                assert "response relevancy" in reason

    @patch("lightspeed_evaluation.core.metrics.custom.litellm.completion")
    def test_evaluation_with_edge_case_scoring(self, mock_completion, mock_llm_manager):
        """Test evaluation with edge case scoring scenarios."""
        # Test different score formats that might come from LLM
        test_cases = [
            ("Perfect score: 1.0\nReason: Excellent", 1.0),
            ("Score: 0\nReason: Completely incorrect", 0.0),
            ("Rating: 7.5 out of 10", 0.75),
            ("85% accuracy", 0.85),
            ("Score: 0.999", 0.999),
        ]

        metrics = CustomMetrics(mock_llm_manager)

        for response_text, expected_score in test_cases:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = response_text
            mock_completion.return_value = mock_response

            turn_data = TurnData(
                turn_id=1,
                query="Test query",
                response="Test response",
                expected_response="Expected response",
            )

            scope = EvaluationScope(
                turn_idx=0, turn_data=turn_data, is_conversation=False
            )

            score, reason = metrics.evaluate("answer_correctness", None, scope)

            assert score == expected_score, f"Failed for response: {response_text}"

"""Tests for NLP metrics module.

This module tests the NLP-based evaluation metrics:
- BLEU Score
- ROUGE Score
- Semantic Similarity (NonLLMStringSimilarity)

Tests use pytest mocker fixture to mock Ragas scorer classes,
avoiding dependency on optional packages.

Note: Data validation (required fields) is handled by DataValidator
in the pipeline. These tests focus on metric evaluation logic.
"""

import sys

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.constants import (
    ROUGE_TYPE_ROUGE1,
    SIMILARITY_JARO_WINKLER,
)
from lightspeed_evaluation.core.metrics.nlp import NLPMetrics
from lightspeed_evaluation.core.models import EvaluationScope, TurnData
from lightspeed_evaluation.core.system.exceptions import MetricError


class TestNLPMetricsInit:  # pylint: disable=too-few-public-methods
    """Test NLPMetrics initialization."""

    def test_initialization(self, nlp_metrics: NLPMetrics) -> None:
        """Test that NLPMetrics initializes correctly."""
        assert nlp_metrics is not None
        assert "bleu" in nlp_metrics.supported_metrics
        assert "rouge" in nlp_metrics.supported_metrics
        assert "semantic_similarity_distance" in nlp_metrics.supported_metrics


class TestNLPMetricsValidation:
    """Tests for metric-level validation."""

    def test_conversation_level_rejected(
        self, nlp_metrics: NLPMetrics, conversation_scope: EvaluationScope
    ) -> None:
        """Test that NLP metrics reject conversation-level evaluation."""
        score, reason = nlp_metrics.evaluate("bleu", None, conversation_scope)

        assert score is None
        assert "turn-level metric" in reason

    def test_unsupported_metric(
        self, nlp_metrics: NLPMetrics, sample_scope: EvaluationScope
    ) -> None:
        """Test evaluate with unsupported metric name."""
        score, reason = nlp_metrics.evaluate("unsupported_metric", None, sample_scope)

        assert score is None
        assert "Unsupported NLP metric" in reason


class TestBLEUScore:
    """Tests for BLEU score metric."""

    def test_bleu_successful_evaluation(
        self,
        nlp_metrics: NLPMetrics,
        sample_scope: EvaluationScope,
        mock_bleu_scorer: MockerFixture,
    ) -> None:
        """Test BLEU score with valid inputs."""
        assert mock_bleu_scorer is not None  # Fixture sets up the mock
        score, reason = nlp_metrics.evaluate("bleu", None, sample_scope)

        assert score is not None
        assert score == pytest.approx(0.85, abs=0.01)
        assert "NLP BLEU" in reason

    def test_bleu_with_custom_ngram(
        self, nlp_metrics: NLPMetrics, mocker: MockerFixture
    ) -> None:
        """Test BLEU score with custom max_ngram configuration."""
        mock_result = mocker.MagicMock()
        mock_result.score = 90.0

        mock_scorer_instance = mocker.MagicMock()
        mock_scorer_instance.corpus_score = mocker.MagicMock(return_value=mock_result)

        mock_bleu_class = mocker.MagicMock(return_value=mock_scorer_instance)
        mock_sacrebleu = mocker.MagicMock()
        mock_sacrebleu.BLEU = mock_bleu_class
        mocker.patch.dict(sys.modules, {"sacrebleu": mock_sacrebleu})

        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="Test response with some words",
            expected_response="Test response with some words",
            turn_metrics_metadata={
                "nlp:bleu": {
                    "max_ngram": 2,  # Use bigrams
                }
            },
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = nlp_metrics.evaluate("bleu", None, scope)

        assert score is not None
        assert score == pytest.approx(0.90, abs=0.01)
        assert "BLEU-2" in reason
        # Verify BLEU was initialized with max_ngram_order=2
        mock_bleu_class.assert_called_once_with(max_ngram_order=2)

    def test_bleu_with_invalid_ngram_uses_default(
        self, nlp_metrics: NLPMetrics, mocker: MockerFixture
    ) -> None:
        """Test BLEU score falls back to default when invalid max_ngram provided."""
        mock_result = mocker.MagicMock()
        mock_result.score = 85.0

        mock_scorer_instance = mocker.MagicMock()
        mock_scorer_instance.corpus_score = mocker.MagicMock(return_value=mock_result)

        mock_bleu_class = mocker.MagicMock(return_value=mock_scorer_instance)
        mock_sacrebleu = mocker.MagicMock()
        mock_sacrebleu.BLEU = mock_bleu_class
        mocker.patch.dict(sys.modules, {"sacrebleu": mock_sacrebleu})

        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="Test response",
            expected_response="Test response",
            turn_metrics_metadata={
                "nlp:bleu": {
                    "max_ngram": 10,  # Invalid: must be 1-4
                }
            },
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = nlp_metrics.evaluate("bleu", None, scope)

        assert score is not None
        assert "BLEU-4" in reason  # Falls back to default
        mock_bleu_class.assert_called_once_with(max_ngram_order=4)


class TestROUGEScore:
    """Tests for ROUGE score metric."""

    def test_rouge_successful_evaluation(
        self,
        nlp_metrics: NLPMetrics,
        sample_scope: EvaluationScope,
        mock_rouge_scorer: MockerFixture,
    ) -> None:
        """Test ROUGE score with valid inputs."""
        assert mock_rouge_scorer is not None  # Fixture sets up the mock
        score, reason = nlp_metrics.evaluate("rouge", None, sample_scope)

        assert score is not None
        assert score == pytest.approx(0.92, abs=0.01)
        assert "NLP ROUGE" in reason
        # Should include all three modes
        assert "precision" in reason
        assert "recall" in reason
        assert "fmeasure" in reason

    def test_rouge_with_custom_rouge_type(
        self, nlp_metrics: NLPMetrics, mocker: MockerFixture
    ) -> None:
        """Test ROUGE score with custom rouge_type via turn_metrics_metadata."""
        mock_scorer_instance = mocker.MagicMock()
        # Return different scores for each mode (precision, recall, fmeasure)
        mock_scorer_instance.single_turn_score = mocker.MagicMock(
            side_effect=[0.9, 0.8, 0.85]
        )
        mocker.patch(
            "lightspeed_evaluation.core.metrics.nlp.RougeScore",
            return_value=mock_scorer_instance,
        )

        turn_data = TurnData(
            turn_id="test_turn",
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            expected_response="The capital of France is Paris.",
            turn_metrics_metadata={
                "nlp:rouge": {
                    "rouge_type": ROUGE_TYPE_ROUGE1,
                }
            },
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = nlp_metrics.evaluate("rouge", None, scope)

        # fmeasure should be the primary score
        assert score == pytest.approx(0.85, abs=0.01)
        assert "rouge1" in reason
        # All modes should be in the reason
        assert "precision" in reason
        assert "recall" in reason
        assert "fmeasure" in reason


class TestSemanticSimilarityDistance:
    """Tests for string distance similarity (NonLLMStringSimilarity) metric."""

    def test_semantic_similarity_distance_successful_evaluation(
        self,
        nlp_metrics: NLPMetrics,
        sample_scope: EvaluationScope,
        mock_similarity_scorer: MockerFixture,
    ) -> None:
        """Test string distance similarity with valid inputs."""
        assert mock_similarity_scorer is not None  # Fixture sets up the mock
        score, reason = nlp_metrics.evaluate(
            "semantic_similarity_distance", None, sample_scope
        )

        assert score is not None
        assert score == pytest.approx(0.78, abs=0.01)
        assert "NLP String Distance" in reason

    def test_semantic_similarity_distance_with_custom_measure(
        self, nlp_metrics: NLPMetrics, mocker: MockerFixture
    ) -> None:
        """Test string distance similarity with custom distance measure config."""
        mock_scorer_instance = mocker.MagicMock()
        mock_scorer_instance.single_turn_score = mocker.MagicMock(return_value=0.95)
        mocker.patch(
            "lightspeed_evaluation.core.metrics.nlp.NonLLMStringSimilarity",
            return_value=mock_scorer_instance,
        )

        turn_data = TurnData(
            turn_id="test_turn",
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            expected_response="The capital of France is Paris.",
            turn_metrics_metadata={
                "nlp:semantic_similarity_distance": {
                    "distance_measure": SIMILARITY_JARO_WINKLER,
                }
            },
        )
        scope = EvaluationScope(turn_idx=0, turn_data=turn_data, is_conversation=False)

        score, reason = nlp_metrics.evaluate(
            "semantic_similarity_distance", None, scope
        )

        assert score is not None
        assert "jaro_winkler" in reason


class TestMetricErrorHandling:
    """Tests for error handling across all NLP metrics."""

    def test_bleu_failure_raises_metric_error(
        self,
        nlp_metrics: NLPMetrics,
        sample_scope: EvaluationScope,
        mocker: MockerFixture,
    ) -> None:
        """Test that BLEU raises MetricError when scoring fails."""
        mock_scorer_instance = mocker.MagicMock()
        mock_scorer_instance.corpus_score = mocker.MagicMock(
            side_effect=RuntimeError("Test error")
        )

        mock_bleu_class = mocker.MagicMock(return_value=mock_scorer_instance)
        mock_sacrebleu = mocker.MagicMock()
        mock_sacrebleu.BLEU = mock_bleu_class
        mocker.patch.dict(sys.modules, {"sacrebleu": mock_sacrebleu})

        with pytest.raises(MetricError) as exc_info:
            nlp_metrics.evaluate("bleu", None, sample_scope)

        assert "evaluation failed" in str(exc_info.value)

    @pytest.mark.parametrize(
        "metric_name,scorer_path",
        [
            ("rouge", "lightspeed_evaluation.core.metrics.nlp.RougeScore"),
            (
                "semantic_similarity_distance",
                "lightspeed_evaluation.core.metrics.nlp.NonLLMStringSimilarity",
            ),
        ],
    )
    def test_ragas_metric_failure_raises_metric_error(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        nlp_metrics: NLPMetrics,
        sample_scope: EvaluationScope,
        mocker: MockerFixture,
        metric_name: str,
        scorer_path: str,
    ) -> None:
        """Test that Ragas-based metrics raise MetricError when scoring fails."""
        mock_scorer_instance = mocker.MagicMock()
        mock_scorer_instance.single_turn_score = mocker.MagicMock(
            side_effect=RuntimeError("Test error")
        )
        mocker.patch(scorer_path, return_value=mock_scorer_instance)

        with pytest.raises(MetricError) as exc_info:
            nlp_metrics.evaluate(metric_name, None, sample_scope)

        assert "evaluation failed" in str(exc_info.value)

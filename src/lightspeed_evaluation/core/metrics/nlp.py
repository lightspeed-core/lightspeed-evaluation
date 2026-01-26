"""NLP-based metrics evaluation using Ragas non-LLM metrics.

This module provides text comparison metrics that don't require LLM calls:
- BLEU Score: Measures n-gram overlap between response and reference
- ROUGE Score: Measures recall-oriented n-gram overlap
- Semantic Similarity: Measures string similarity using distance measures
"""

import logging
from typing import Any, Optional

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMStringSimilarity, RougeScore

from lightspeed_evaluation.core.constants import (
    DEFAULT_BLEU_MAX_NGRAM,
    DISTANCE_MEASURE_MAP,
    MAX_BLEU_NGRAM,
    MIN_BLEU_NGRAM,
    ROUGE_MODE_FMEASURE,
    ROUGE_MODE_PRECISION,
    ROUGE_MODE_RECALL,
    ROUGE_TYPE_ROUGEL,
    SIMILARITY_LEVENSHTEIN,
    SUPPORTED_SIMILARITY_MEASURES,
)
from lightspeed_evaluation.core.models import EvaluationScope, TurnData
from lightspeed_evaluation.core.system.exceptions import MetricError

logger = logging.getLogger(__name__)


class NLPMetrics:  # pylint: disable=too-few-public-methods
    """Handles NLP-based metrics evaluation using Ragas non-LLM metrics.

    These metrics compare the generated response with the expected_response
    using traditional NLP techniques without requiring LLM calls.
    """

    def __init__(self) -> None:
        """Initialize NLP Metrics.

        No LLM or embedding manager required for these metrics.
        """
        self.supported_metrics = {
            "bleu": self._evaluate_bleu,
            "rouge": self._evaluate_rouge,
            "semantic_similarity_distance": self._evaluate_semantic_similarity_distance,
        }

        logger.info("NLP Metrics initialized")

    def _extract_turn_data(self, turn_data: Optional[TurnData]) -> tuple[str, str]:
        """Extract response and expected_response from turn data.

        Args:
            turn_data: Turn data containing response and expected_response

        Returns:
            Tuple of (response, expected_response)
        """
        response = turn_data.response if turn_data else ""
        expected_response = turn_data.expected_response if turn_data else ""

        # For NLP metrics, expected_response must be a string (not a list)
        if isinstance(expected_response, list):
            raise MetricError(
                "NLP metrics require expected_response to be a string, not a list"
            )

        return response or "", expected_response or ""

    def _get_metric_metadata(
        self, turn_data: Optional[TurnData], metric_key: str
    ) -> dict[str, Any]:
        """Get metric-specific metadata from turn data.

        Args:
            turn_data: Turn data containing turn_metrics_metadata
            metric_key: The metric key (e.g., "nlp:rouge", "nlp:semantic_similarity_distance")

        Returns:
            Dictionary of metric-specific metadata, or empty dict if not found
        """
        turn_metadata = turn_data.turn_metrics_metadata if turn_data else {}
        return (turn_metadata or {}).get(metric_key, {})

    def _run_score(self, scorer: Any, response: str, reference: str) -> float:
        """Run scoring using Ragas synchronous API.

        Args:
            scorer: The Ragas scorer instance
            response: The generated response
            reference: The expected/reference response

        Returns:
            The score value

        Raises:
            ValueError: If input validation fails
            TypeError: If type conversion fails
        """
        sample = SingleTurnSample(response=response, reference=reference)
        result = scorer.single_turn_score(sample)
        return float(result)

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        scope: EvaluationScope,
    ) -> tuple[Optional[float], str]:
        """Evaluate an NLP metric.

        Args:
            metric_name: Name of the metric to evaluate
            conv_data: Conversation data (not used for NLP metrics)
            scope: Evaluation scope containing turn data

        Returns:
            Tuple of (score, reason) where score is 0-1 and reason is descriptive

        Raises:
            MetricError: When metric evaluation fails unexpectedly
        """
        if metric_name not in self.supported_metrics:
            logger.warning("Unsupported NLP metric requested: %s", metric_name)
            return None, f"Unsupported NLP metric: {metric_name}"

        try:
            score, reason = self.supported_metrics[metric_name](
                conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
            )
            if score is not None:
                logger.debug(
                    "NLP metric %s evaluated successfully: score=%.4f",
                    metric_name,
                    score,
                )
            return score, reason
        except ImportError as e:
            # Handle missing optional dependencies first - fail fast with clear message
            error_msg = str(e)
            if any(
                pkg in error_msg.lower() for pkg in ["sacrebleu", "rouge", "rapidfuzz"]
            ):
                install_hint = (
                    "NLP metrics require optional dependencies. "
                    "Install with: pip install 'lightspeed-evaluation[nlp-metrics]' "
                    "or: uv pip install sacrebleu rouge-score rapidfuzz"
                )
                logger.error(install_hint)
                raise MetricError(install_hint) from e
            raise MetricError(f"NLP {metric_name} evaluation failed: {str(e)}") from e
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(
                "NLP %s evaluation failed: %s", metric_name, str(e), exc_info=True
            )
            raise MetricError(f"NLP {metric_name} evaluation failed: {str(e)}") from e

    def _evaluate_bleu(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate BLEU score.

        BLEU (Bilingual Evaluation Understudy) measures how many n-grams
        in the response match the reference text.

        Supports configurable max_ngram (1-4, default 4 for standard BLEU-4).

        Args:
            _conv_data: Conversation data (unused)
            _turn_idx: Turn index (unused)
            turn_data: Turn data containing response and expected_response
            is_conversation: Whether this is conversation-level evaluation

        Returns:
            Tuple of (score, reason)
        """
        if is_conversation:
            return None, "BLEU is a turn-level metric"

        response, reference = self._extract_turn_data(turn_data)

        # Get n-gram configuration from metadata
        metadata = self._get_metric_metadata(turn_data, "nlp:bleu")
        max_ngram = metadata.get("max_ngram", DEFAULT_BLEU_MAX_NGRAM)

        # Validate n-gram range
        if not MIN_BLEU_NGRAM <= max_ngram <= MAX_BLEU_NGRAM:
            logger.warning(
                "Invalid max_ngram=%d, must be %d-%d. Using default %d.",
                max_ngram,
                MIN_BLEU_NGRAM,
                MAX_BLEU_NGRAM,
                DEFAULT_BLEU_MAX_NGRAM,
            )
            max_ngram = DEFAULT_BLEU_MAX_NGRAM

        # Use sacrebleu.BLEU directly for n-gram configuration support
        # Ragas BleuScore uses corpus_bleu which doesn't support max_ngram_order
        # Import lazily to avoid import errors when sacrebleu is not installed
        # pylint: disable=import-outside-toplevel,import-error
        from sacrebleu import BLEU  # type: ignore[import-not-found]

        bleu = BLEU(max_ngram_order=max_ngram)
        result = bleu.corpus_score([response], [[reference]])
        score = result.score / 100.0  # sacrebleu returns 0-100 scale

        logger.debug("BLEU-%d score computed: %.4f", max_ngram, score)
        return score, f"NLP BLEU-{max_ngram}: {score:.4f}"

    def _evaluate_rouge(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate ROUGE score.

        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
        recall-oriented n-gram overlap between response and reference.

        Always calculates precision, recall, and fmeasure. Returns fmeasure
        as the primary score since it balances precision and recall.

        Args:
            _conv_data: Conversation data (unused)
            _turn_idx: Turn index (unused)
            turn_data: Turn data containing response and expected_response
            is_conversation: Whether this is conversation-level evaluation

        Returns:
            Tuple of (score, reason) where score is the fmeasure
        """
        if is_conversation:
            return None, "ROUGE is a turn-level metric"

        response, reference = self._extract_turn_data(turn_data)
        metadata = self._get_metric_metadata(turn_data, "nlp:rouge")
        rouge_type = metadata.get("rouge_type", ROUGE_TYPE_ROUGEL)

        # Calculate all three modes - fmeasure requires precision and recall
        rouge_modes = (ROUGE_MODE_PRECISION, ROUGE_MODE_RECALL, ROUGE_MODE_FMEASURE)
        scores = {}
        for mode in rouge_modes:
            scorer = RougeScore(
                rouge_type=rouge_type,
                mode=mode,  # type: ignore[arg-type]
            )
            scores[mode] = self._run_score(scorer, response, reference)

        # Build reason string with all scores
        scores_str = ", ".join(f"{m}={s:.4f}" for m, s in scores.items())
        reason = f"NLP ROUGE ({rouge_type}): {scores_str}"

        logger.debug("ROUGE score computed (%s): %s", rouge_type, scores)
        return scores[ROUGE_MODE_FMEASURE], reason

    def _evaluate_semantic_similarity_distance(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate string similarity using character-level distance measures.

        WARNING: This metric uses string distance (Levenshtein, Jaro, etc.),
        NOT semantic meaning. LLM responses with correct meaning but different
        wording will score low. Consider using LLM-based metrics like
        custom:answer_correctness for semantic comparison.

        Args:
            _conv_data: Conversation data (unused)
            _turn_idx: Turn index (unused)
            turn_data: Turn data containing response and expected_response
            is_conversation: Whether this is conversation-level evaluation

        Returns:
            Tuple of (score, reason)
        """
        if is_conversation:
            return None, "Semantic similarity distance is a turn-level metric"

        response, reference = self._extract_turn_data(turn_data)

        # Get configuration from turn_metrics_metadata if available
        metadata = self._get_metric_metadata(
            turn_data, "nlp:semantic_similarity_distance"
        )
        distance_measure_str = metadata.get("distance_measure", SIMILARITY_LEVENSHTEIN)

        if distance_measure_str not in SUPPORTED_SIMILARITY_MEASURES:
            raise ValueError(f"Invalid distance measure: {distance_measure_str}")

        distance_measure = DISTANCE_MEASURE_MAP[distance_measure_str]
        scorer = NonLLMStringSimilarity(distance_measure=distance_measure)
        score = self._run_score(scorer, response, reference)
        logger.debug(
            "String similarity distance computed (%s): %.4f",
            distance_measure_str,
            score,
        )
        return (
            score,
            f"NLP String Distance ({distance_measure_str}): {score:.4f}",
        )

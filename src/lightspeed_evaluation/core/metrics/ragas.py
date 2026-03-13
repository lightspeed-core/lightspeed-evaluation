"""Ragas metrics evaluation using LLM Manager with Ragas 0.4+ API."""

import errno
import math
from typing import Any, Optional

import litellm
from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    ContextUtilization,
    Faithfulness,
)

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.embedding.ragas import RagasEmbeddingManager
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.llm.ragas import RagasLLMManager
from lightspeed_evaluation.core.models import EvaluationScope, TurnData


def _clamp_score(score: float) -> float:
    """Clamp score to [0, 1] range to handle floating-point precision issues.

    Ragas metrics can sometimes return values like 1.0000000000000007 due to
    floating-point arithmetic, which would fail Pydantic validation.
    """
    return max(0.0, min(1.0, score))


class RagasMetrics:  # pylint: disable=too-few-public-methods
    """Handles Ragas metrics evaluation using LLM Manager with Ragas 0.4+ API.

    Ragas 0.4+ uses collections-based metrics with direct .score() calls.
    We use the synchronous score() method since litellm.completion is synchronous.
    """

    def __init__(self, llm_manager: LLMManager, embedding_manager: EmbeddingManager):
        """Initialize with LLM Manager.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters
            embedding_manager: Pre-configured EmbeddingManager with validated parameters
        """
        # Modifying global litellm cache as there is no clear way how to do it per model
        # We can't use Ragas's Cacher here, because we use litellm here and that conflicts with
        # litellm cache from DeepEval code
        if llm_manager.get_config().cache_enabled and litellm.cache is None:
            cache_dir = llm_manager.get_config().cache_dir
            litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=cache_dir)

        # Create Ragas LLM Manager for metric configuration
        self.llm_manager = RagasLLMManager(llm_manager)
        self.embedding_manager = RagasEmbeddingManager(embedding_manager)

        self.supported_metrics = {
            # Response evaluation metrics
            "faithfulness": self._evaluate_faithfulness,
            "response_relevancy": self._evaluate_response_relevancy,
            # Context/Retrieval evaluation metrics
            "context_recall": self._evaluate_context_recall,
            "context_relevance": self._evaluate_context_relevance,
            "context_precision_with_reference": self._evaluate_context_precision_with_reference,
            "context_precision_without_reference": (
                self._evaluate_context_precision_without_reference
            ),
        }

    def _extract_turn_data(
        self, turn_data: Optional[TurnData]
    ) -> tuple[str, str, list[str]]:
        """Extract turn data."""
        if turn_data is None:
            return "", "", []

        query = turn_data.query
        response = turn_data.response
        contexts = [
            ctx.get("content", str(ctx)) if isinstance(ctx, dict) else str(ctx)
            for ctx in turn_data.contexts or []
        ]
        return query, response or "", contexts

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        scope: EvaluationScope,
    ) -> tuple[Optional[float], str]:
        """Evaluate a Ragas metric."""
        if metric_name not in self.supported_metrics:
            return None, f"Unsupported Ragas metric: {metric_name}"

        try:
            result = self.supported_metrics[metric_name](
                conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
            )
        except BrokenPipeError as e:
            return (
                None,
                f"Ragas {metric_name} evaluation failed due to network/LLM "
                f"connection issue: {str(e)}",
            )
        except OSError as e:
            err_msg = f"Ragas {metric_name} evaluation failed: {str(e)}"
            if e.errno == errno.EPIPE:
                err_msg = (
                    f"Ragas {metric_name} evaluation failed due to broken pipe "
                    f"(network/LLM timeout): {str(e)}"
                )
            return None, err_msg
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            return None, f"Ragas {metric_name} evaluation failed: {str(e)}"

        if result[0] is not None and math.isnan(result[0]):
            return (
                None,
                f"Ragas {metric_name} evaluation failed due to malformed "
                "output from the LLM",
            )

        return result

    def _evaluate_response_relevancy(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate response relevancy using Ragas 0.4+ score()."""
        if is_conversation:
            return None, "Response relevancy is a turn-level metric"

        query, response, _ = self._extract_turn_data(turn_data)

        metric = AnswerRelevancy(
            llm=self.llm_manager.get_llm(),
            embeddings=self.embedding_manager.embeddings,
        )

        result = metric.score(user_input=query, response=response)

        score = _clamp_score(float(result.value))
        return score, f"Ragas response relevancy: {score:.2f}"

    def _evaluate_faithfulness(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate faithfulness using Ragas 0.4+ score()."""
        if is_conversation:
            return None, "Faithfulness is a turn-level metric"

        query, response, contexts = self._extract_turn_data(turn_data)

        metric = Faithfulness(llm=self.llm_manager.get_llm())

        result = metric.score(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
        )

        score = _clamp_score(float(result.value))
        return score, f"Ragas faithfulness: {score:.2f}"

    def _evaluate_context_precision_without_reference(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate context precision without reference using Ragas 0.4+ score()."""
        if is_conversation:
            return None, "Context precision without reference is a turn-level metric"

        query, response, contexts = self._extract_turn_data(turn_data)

        metric = ContextUtilization(llm=self.llm_manager.get_llm())

        result = metric.score(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
        )

        score = _clamp_score(float(result.value))
        return score, f"Ragas context precision without reference: {score:.2f}"

    def _evaluate_context_precision_with_reference(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate context precision with reference using Ragas 0.4+ score()."""
        if is_conversation:
            return None, "Context precision with reference is a turn-level metric"

        if turn_data is None:
            return None, "TurnData is required for context precision with reference"

        query, _, contexts = self._extract_turn_data(turn_data)

        metric = ContextPrecision(llm=self.llm_manager.get_llm())

        result = metric.score(
            user_input=query,
            reference=turn_data.expected_response or "",
            retrieved_contexts=contexts,
        )

        score = _clamp_score(float(result.value))
        return score, f"Ragas context precision with reference: {score:.2f}"

    def _evaluate_context_recall(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate context recall using Ragas 0.4+ score()."""
        if is_conversation:
            return None, "Context recall is a turn-level metric"

        if turn_data is None:
            return None, "TurnData is required for context recall"

        query, _, contexts = self._extract_turn_data(turn_data)

        metric = ContextRecall(llm=self.llm_manager.get_llm())

        result = metric.score(
            user_input=query,
            retrieved_contexts=contexts,
            reference=turn_data.expected_response or "",
        )

        score = _clamp_score(float(result.value))
        return score, f"Ragas context recall: {score:.2f}"

    def _evaluate_context_relevance(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate context relevance using Ragas 0.4+ score()."""
        if is_conversation:
            return None, "Context relevance is a turn-level metric"

        query, _, contexts = self._extract_turn_data(turn_data)

        metric = ContextRelevance(llm=self.llm_manager.get_llm())

        result = metric.score(
            user_input=query,
            retrieved_contexts=contexts,
        )

        score = _clamp_score(float(result.value))
        return score, f"Ragas context relevance: {score:.2f}"

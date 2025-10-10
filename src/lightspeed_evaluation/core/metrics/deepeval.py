"""DeepEval metrics evaluation using LLM Manager."""

from typing import Any, Optional

import litellm
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import Turn as DeepEvalTurn
from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType

from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.models import EvaluationScope, TurnData


class DeepEvalMetrics:  # pylint: disable=too-few-public-methods
    """Handles DeepEval metrics evaluation using LLM Manager."""

    def __init__(self, llm_manager: LLMManager):
        """Initialize with LLM Manager.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters
        """
        if llm_manager.get_config().cache_enabled and litellm.cache is None:
            cache_dir = llm_manager.get_config().cache_dir
            # Modifying global litellm cache as there is no clear way how to do it per model
            # Checking if the litellm.cache as there is potential conflict with Ragas code
            litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=cache_dir)

        # Create LLM Manager for DeepEval metrics
        self.llm_manager = DeepEvalLLMManager(
            llm_manager.get_model_name(), llm_manager.get_llm_params()
        )

        self.supported_metrics = {
            "conversation_completeness": self._evaluate_conversation_completeness,
            "conversation_relevancy": self._evaluate_conversation_relevancy,
            "knowledge_retention": self._evaluate_knowledge_retention,
        }

    def _build_conversational_test_case(self, conv_data: Any) -> ConversationalTestCase:
        """Build ConversationalTestCase from conversation data."""
        turns = []
        for turn_data in conv_data.turns:
            # Add user turn
            turns.append(DeepEvalTurn(role="user", content=turn_data.query))
            # Add assistant turn
            turns.append(DeepEvalTurn(role="assistant", content=turn_data.response))

        return ConversationalTestCase(turns=turns)

    def _evaluate_metric(self, metric: Any, test_case: Any) -> tuple[float, str]:
        """Evaluate and get result."""
        metric.measure(test_case)

        reason = (
            metric.reason
            if hasattr(metric, "reason") and metric.reason
            else f"Score: {metric.score:.2f}"
        )
        return metric.score, reason

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        scope: EvaluationScope,
    ) -> tuple[Optional[float], str]:
        """Evaluate a DeepEval metric."""
        if metric_name not in self.supported_metrics:
            return None, f"Unsupported DeepEval metric: {metric_name}"

        try:
            return self.supported_metrics[metric_name](
                conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
            )
        except (ValueError, AttributeError, KeyError) as e:
            return None, f"DeepEval {metric_name} evaluation failed: {str(e)}"

    def _evaluate_conversation_completeness(
        self,
        conv_data: Any,
        _turn_idx: Optional[int],
        _turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate conversation completeness."""
        if not is_conversation:
            return None, "Conversation completeness is a conversation-level metric"

        test_case = self._build_conversational_test_case(conv_data)
        metric = ConversationCompletenessMetric(model=self.llm_manager.get_llm())

        return self._evaluate_metric(metric, test_case)

    def _evaluate_conversation_relevancy(
        self,
        conv_data: Any,
        _turn_idx: Optional[int],
        _turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate conversation relevancy using DeepEval TurnRelevancyMetric."""
        if not is_conversation:
            return None, "Conversation relevancy is a conversation-level metric"

        if not conv_data.turns:
            return None, "No conversation turns available for relevancy evaluation"

        # Use common helper methods
        test_case = self._build_conversational_test_case(conv_data)
        metric = TurnRelevancyMetric(model=self.llm_manager.get_llm())

        return self._evaluate_metric(metric, test_case)

    def _evaluate_knowledge_retention(
        self,
        conv_data: Any,
        _turn_idx: Optional[int],
        _turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate knowledge retention."""
        if not is_conversation:
            return None, "Knowledge retention is a conversation-level metric"

        if len(conv_data.turns) < 2:
            return None, "Knowledge retention requires at least 2 turns"

        test_case = self._build_conversational_test_case(conv_data)
        metric = KnowledgeRetentionMetric(model=self.llm_manager.get_llm())

        return self._evaluate_metric(metric, test_case)

"""DeepEval metrics evaluation using LLM Manager.

This module provides integration with DeepEval metrics including:
1. Standard DeepEval metrics (conversation completeness, relevancy, knowledge retention)
2. GEval integration for configurable custom evaluation criteria
"""

import logging
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
from lightspeed_evaluation.core.metrics.geval import GEvalHandler
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.models import EvaluationScope, TurnData

logger = logging.getLogger(__name__)


class DeepEvalMetrics:  # pylint: disable=too-few-public-methods
    """Handles DeepEval metrics evaluation using LLM Manager.

    This class provides a unified interface for both standard DeepEval metrics
    and GEval (configurable custom metrics). It shares LLM resources between
    both evaluation types for efficiency.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        metric_manager: MetricManager,
    ):
        """Initialize with LLM Manager.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters
            metric_manager: MetricManager for accessing metric metadata
        """
        # Setup cache if enabled (shared across all DeepEval operations)
        if llm_manager.get_config().cache_enabled and litellm.cache is None:
            cache_dir = llm_manager.get_config().cache_dir
            # Modifying global litellm cache as there is no clear way how to do it per model
            # Checking if the litellm.cache as there is potential conflict with Ragas code
            litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=cache_dir)

        # Create shared LLM Manager for all DeepEval metrics (standard + GEval)
        self.llm_manager = DeepEvalLLMManager(
            llm_manager.get_model_name(), llm_manager.get_llm_params()
        )

        # Initialize GEval handler with shared LLM manager and metric manager
        self.geval_handler = GEvalHandler(
            deepeval_llm_manager=self.llm_manager,
            metric_manager=metric_manager,
        )

        # Standard DeepEval metrics routing
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
        """Evaluate a DeepEval metric (standard or GEval).

        This method routes evaluation to either:
        - Standard DeepEval metrics (hardcoded implementations)
        - GEval metrics (configuration-driven custom metrics)

        Args:
            metric_name: Name of metric (for GEval, this should NOT include "geval:" prefix)
            conv_data: Conversation data object
            scope: EvaluationScope containing turn info and conversation flag

        Returns:
            Tuple of (score, reason)
        """
        # Route to standard DeepEval metrics
        if metric_name in self.supported_metrics:
            try:
                return self.supported_metrics[metric_name](
                    conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
                )
            except (ValueError, AttributeError, KeyError) as e:
                return None, f"DeepEval {metric_name} evaluation failed: {str(e)}"

        # Otherwise, assume it's a GEval metric
        normalized_metric_name = (
            metric_name.split(":", 1)[1]
            if metric_name.startswith("geval:")
            else metric_name
        )
        return self.geval_handler.evaluate(
            metric_name=normalized_metric_name,
            conv_data=conv_data,
            _turn_idx=scope.turn_idx,
            turn_data=scope.turn_data,
            is_conversation=scope.is_conversation,
        )

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

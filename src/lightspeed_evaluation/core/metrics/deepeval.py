"""DeepEval metrics evaluation using LLM Manager."""

from typing import Any, Dict, Optional, Tuple

from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import Turn as DeepEvalTurn

from ..config import TurnData
from ..llm.deepeval import DeepEvalLLMManager
from ..llm.manager import LLMManager
from ..output.statistics import EvaluationScope


class DeepEvalMetrics:
    """Handles DeepEval metrics evaluation using LLM Manager."""

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize with LLM Manager.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters
        """
        # Create LLM Manager for DeepEval metrics
        self.llm_manager = DeepEvalLLMManager(
            llm_manager.get_model_name(), llm_manager.get_litellm_params()
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

    def _evaluate_metric(self, metric: Any, test_case: Any) -> Tuple[float, str]:
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
    ) -> Tuple[Optional[float], str]:
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
    ) -> Tuple[Optional[float], str]:
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
    ) -> Tuple[Optional[float], str]:
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
    ) -> Tuple[Optional[float], str]:
        """Evaluate knowledge retention."""
        if not is_conversation:
            return None, "Knowledge retention is a conversation-level metric"

        if len(conv_data.turns) < 2:
            return None, "Knowledge retention requires at least 2 turns"

        test_case = self._build_conversational_test_case(conv_data)
        metric = KnowledgeRetentionMetric(model=self.llm_manager.get_llm())

        return self._evaluate_metric(metric, test_case)

    @classmethod
    def from_system_config(cls, system_config: Dict[str, Any]) -> "DeepEvalMetrics":
        """Create DeepEvalMetrics from system configuration."""
        llm_manager = LLMManager.from_system_config(system_config)
        return cls(llm_manager)

"""Ragas metrics evaluation using LLM Manager."""

from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextRelevance,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from ..llm.manager import LLMManager
from ..llm.ragas import RagasLLMManager
from ..models import EvaluationScope, TurnData


# Decide if Dataset will be used or not ?
class RagasMetrics:
    """Handles Ragas metrics evaluation using LLM Manager."""

    def __init__(self, llm_manager: LLMManager):
        """Initialize with LLM Manager.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters
        """
        # Create Ragas LLM Manager for metric configuration
        self.llm_manager = RagasLLMManager(
            llm_manager.get_model_name(), llm_manager.get_litellm_params()
        )

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
    ) -> Tuple[str, str, List[str]]:
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

    def _evaluate_metric(
        self,
        metric_class: Any,
        dataset_dict: Dict[str, Any],
        result_key: str,
        metric_name: str,
    ) -> Tuple[Optional[float], str]:
        """Evaluate metric with configured LLM."""
        dataset = Dataset.from_dict(dataset_dict)

        # Configure metric with LLM
        metric_instance = metric_class(llm=self.llm_manager.get_llm())

        result = evaluate(dataset, metrics=[metric_instance])
        df = result.to_pandas()
        score = df[result_key].iloc[0]
        return score, f"Ragas {metric_name}: {score:.2f}"

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        scope: EvaluationScope,
    ) -> Tuple[Optional[float], str]:
        """Evaluate a Ragas metric."""
        if metric_name not in self.supported_metrics:
            return None, f"Unsupported Ragas metric: {metric_name}"

        try:
            return self.supported_metrics[metric_name](
                conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
            )
        except BrokenPipeError as e:
            return (
                None,
                f"Ragas {metric_name} evaluation failed due to network/LLM "
                f"connection issue: {str(e)}",
            )
        except OSError as e:
            if e.errno == 32:  # Broken pipe
                return (
                    None,
                    f"Ragas {metric_name} evaluation failed due to broken pipe "
                    f"(network/LLM timeout): {str(e)}",
                )
            return None, f"Ragas {metric_name} evaluation failed: {str(e)}"
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            return None, f"Ragas {metric_name} evaluation failed: {str(e)}"

    def _evaluate_response_relevancy(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> Tuple[Optional[float], str]:
        """Evaluate response relevancy."""
        if is_conversation:
            return None, "Response relevancy is a turn-level metric"

        query, response, _ = self._extract_turn_data(turn_data)

        dataset_dict = {"question": [query], "answer": [response]}

        return self._evaluate_metric(
            ResponseRelevancy, dataset_dict, "answer_relevancy", "response relevancy"
        )

    def _evaluate_faithfulness(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> Tuple[Optional[float], str]:
        """Evaluate faithfulness."""
        if is_conversation:
            return None, "Faithfulness is a turn-level metric"

        query, response, contexts = self._extract_turn_data(turn_data)

        dataset_dict = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts],
        }

        return self._evaluate_metric(
            Faithfulness, dataset_dict, "faithfulness", "faithfulness"
        )

    def _evaluate_context_precision_without_reference(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> Tuple[Optional[float], str]:
        """Evaluate context precision without reference."""
        if is_conversation:
            return None, "Context precision without reference is a turn-level metric"

        query, response, contexts = self._extract_turn_data(turn_data)

        dataset_dict = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts],
        }

        return self._evaluate_metric(
            LLMContextPrecisionWithoutReference,
            dataset_dict,
            "llm_context_precision_without_reference",
            "context precision without reference",
        )

    def _evaluate_context_precision_with_reference(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> Tuple[Optional[float], str]:
        """Evaluate context precision with reference."""
        if is_conversation:
            return None, "Context precision with reference is a turn-level metric"

        if turn_data is None:
            return None, "TurnData is required for context precision with reference"

        query, response, contexts = self._extract_turn_data(turn_data)

        dataset_dict = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts],
            "ground_truth": [turn_data.expected_response],
        }

        return self._evaluate_metric(
            LLMContextPrecisionWithReference,
            dataset_dict,
            "llm_context_precision_with_reference",
            "context precision with reference",
        )

    def _evaluate_context_recall(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> Tuple[Optional[float], str]:
        """Evaluate context recall."""
        if is_conversation:
            return None, "Context recall is a turn-level metric"

        if turn_data is None:
            return None, "TurnData is required for context recall"

        query, response, contexts = self._extract_turn_data(turn_data)

        dataset_dict = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts],
            "ground_truth": [turn_data.expected_response],
        }

        return self._evaluate_metric(
            LLMContextRecall, dataset_dict, "context_recall", "LLM context recall"
        )

    def _evaluate_context_relevance(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> Tuple[Optional[float], str]:
        """Evaluate context relevance."""
        if is_conversation:
            return None, "Context relevance is a turn-level metric"

        query, _, contexts = self._extract_turn_data(turn_data)

        dataset_dict = {"question": [query], "contexts": [contexts]}

        return self._evaluate_metric(
            ContextRelevance, dataset_dict, "nv_context_relevance", "context relevance"
        )

    @classmethod
    def from_system_config(cls, system_config: Dict[str, Any]) -> "RagasMetrics":
        """Create RagasMetrics from system configuration."""
        llm_manager = LLMManager.from_system_config(system_config)
        return cls(llm_manager)

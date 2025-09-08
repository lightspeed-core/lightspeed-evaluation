"""Metrics evaluation module - handles individual metric evaluation."""

import logging
import time
from typing import Any, Dict, List, Optional

from ...core.llm.manager import LLMManager
from ...core.metrics.custom import CustomMetrics
from ...core.metrics.deepeval import DeepEvalMetrics
from ...core.metrics.ragas import RagasMetrics
from ...core.models import EvaluationData, EvaluationResult, TurnData
from ...core.output.statistics import EvaluationScope
from ...core.system import ConfigLoader

logger = logging.getLogger(__name__)


class EvaluationRequest:
    """Evaluation request containing conversation data and metric information."""

    def __init__(self, conv_data: EvaluationData, metric_identifier: str):
        """Initialize evaluation request with conversation data and metric identifier."""
        self.conv_data = conv_data
        self.metric_identifier = metric_identifier
        self.is_conversation = False
        self.turn_idx: Optional[int] = None
        self.turn_data: Optional[TurnData] = None
        self.turn_id: Optional[str] = None

    @classmethod
    def for_turn(
        cls,
        conv_data: EvaluationData,
        metric_identifier: str,
        turn_idx: int,
        turn_data: TurnData,
    ) -> "EvaluationRequest":
        """Create request for turn-level evaluation."""
        request = cls(conv_data, metric_identifier)
        request.is_conversation = False
        request.turn_idx = turn_idx
        request.turn_data = turn_data
        request.turn_id = turn_data.turn_id
        return request

    @classmethod
    def for_conversation(
        cls, conv_data: EvaluationData, metric_identifier: str
    ) -> "EvaluationRequest":
        """Create request for conversation-level evaluation."""
        request = cls(conv_data, metric_identifier)
        request.is_conversation = True
        return request

    def get_summary(self) -> str:
        """Get summary string for logging."""
        if self.is_conversation:
            return f"Conversation {self.conv_data.conversation_group_id} - {self.metric_identifier}"
        return f"Turn {self.turn_id} - {self.metric_identifier}"

    def get_evaluation_context(self) -> Dict[str, Any]:
        """Get context information for this evaluation request."""
        context = {
            "conversation_id": self.conv_data.conversation_group_id,
            "metric": self.metric_identifier,
            "is_conversation_level": self.is_conversation,
        }
        if not self.is_conversation and self.turn_data:
            context.update(
                {
                    "turn_id": self.turn_data.turn_id,
                    "query": self.turn_data.query,
                    "has_response": bool(self.turn_data.response),
                    "has_contexts": bool(self.turn_data.contexts),
                }
            )
        return context


class MetricsEvaluator:
    """Handles individual metric evaluation with proper scoring and status determination."""

    def __init__(self, llm_manager: LLMManager, config_loader: ConfigLoader) -> None:
        """Initialize with LLM manager and config."""
        self.config_loader = config_loader
        self.config = config_loader.system_config

        # Initialize metric handlers
        self.ragas_metrics = RagasMetrics(llm_manager)
        self.deepeval_metrics = DeepEvalMetrics(llm_manager)
        self.custom_metrics = CustomMetrics(llm_manager)

        # Metric routing map
        self.handlers = {
            "ragas": self.ragas_metrics,
            "deepeval": self.deepeval_metrics,
            "custom": self.custom_metrics,
        }

    def evaluate_metric(self, request: EvaluationRequest) -> Optional[EvaluationResult]:
        """Evaluate a single metric and return result."""
        start_time = time.time()
        try:
            logger.debug("Evaluating: %s", request.get_summary())

            # Parse framework and metric
            framework, metric_name = request.metric_identifier.split(":", 1)

            # Route to appropriate handler
            if framework not in self.handlers:
                execution_time = time.time() - start_time
                return self._create_error_result(
                    request, f"Unsupported framework: {framework}", execution_time
                )

            # Create evaluation scope
            evaluation_scope = EvaluationScope(
                turn_idx=request.turn_idx,
                turn_data=request.turn_data,
                is_conversation=request.is_conversation,
            )

            # Evaluate metric
            score, reason = self.handlers[framework].evaluate(  # type: ignore
                metric_name, request.conv_data, evaluation_scope
            )

            execution_time = time.time() - start_time

            if score is None:
                return self._create_error_result(request, reason, execution_time)

            # Get threshold and determine status
            threshold = self._get_effective_threshold(
                request.conv_data, request.metric_identifier, request.is_conversation
            )
            status = self._determine_status(score, threshold)

            return EvaluationResult(
                conversation_group_id=request.conv_data.conversation_group_id,
                turn_id=request.turn_id,
                metric_identifier=request.metric_identifier,
                result=status,
                score=score,
                threshold=threshold,
                reason=reason,
                query=request.turn_data.query if request.turn_data else "",
                response=(
                    request.turn_data.response
                    if request.turn_data and request.turn_data.response
                    else ""
                ),
                execution_time=execution_time,
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Any evaluation error should result in ERROR status
            execution_time = time.time() - start_time
            return self._create_error_result(
                request, f"Evaluation error: {e}", execution_time
            )

    def _create_error_result(
        self, request: EvaluationRequest, reason: str, execution_time: float
    ) -> EvaluationResult:
        """Create an ERROR result for failed evaluation."""
        return EvaluationResult(
            conversation_group_id=request.conv_data.conversation_group_id,
            turn_id=request.turn_id,
            metric_identifier=request.metric_identifier,
            result="ERROR",
            score=None,
            threshold=None,
            reason=reason,
            query=request.turn_data.query if request.turn_data else "",
            response=(
                request.turn_data.response
                if request.turn_data and request.turn_data.response
                else ""
            ),
            execution_time=execution_time,
        )

    def _get_effective_threshold(
        self, conv_data: EvaluationData, metric_identifier: str, is_conversation: bool
    ) -> Optional[float]:
        """Get effective threshold for metric (conversation-specific or system default)."""
        # Check conversation-specific metadata first
        if is_conversation:
            metadata = (conv_data.conversation_metrics_metadata or {}).get(
                metric_identifier, {}
            )
        else:
            metadata = (conv_data.turn_metrics_metadata or {}).get(
                metric_identifier, {}
            )

        if "threshold" in metadata:
            return metadata["threshold"]

        # Fall back to system defaults
        if self.config is None:
            raise ValueError("SystemConfig must be loaded")
        if is_conversation:
            default_metadata = (
                self.config.default_conversation_metrics_metadata or {}
            ).get(metric_identifier, {})
        else:
            default_metadata = (self.config.default_turn_metrics_metadata or {}).get(
                metric_identifier, {}
            )

        return default_metadata.get("threshold")

    def _determine_status(self, score: float, threshold: Optional[float]) -> str:
        """Determine evaluation status based on score and threshold."""
        if threshold is None:
            threshold = 0
        return "PASS" if score >= threshold else "FAIL"

    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported evaluation frameworks."""
        return list(self.handlers.keys())

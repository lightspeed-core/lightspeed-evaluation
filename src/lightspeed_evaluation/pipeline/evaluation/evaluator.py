"""Metrics evaluation module - handles individual metric evaluation."""

import logging
import time
from typing import Optional

from ...core.llm.manager import LLMManager
from ...core.metrics.custom import CustomMetrics
from ...core.metrics.deepeval import DeepEvalMetrics
from ...core.metrics.manager import MetricLevel, MetricManager
from ...core.metrics.ragas import RagasMetrics
from ...core.models import EvaluationRequest, EvaluationResult, EvaluationScope
from ...core.system import ConfigLoader

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """Handles individual metric evaluation with proper scoring and status determination."""

    def __init__(
        self,
        llm_manager: LLMManager,
        config_loader: ConfigLoader,
        metric_manager: MetricManager,
    ) -> None:
        """Initialize with LLM manager, config, and metric manager."""
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

        self.metric_manager = metric_manager

    def evaluate_metric(self, request: EvaluationRequest) -> Optional[EvaluationResult]:
        """Evaluate a single metric and return result."""
        start_time = time.time()

        try:
            # Create logging summary
            if request.is_conversation:
                summary = (
                    f"Conversation {request.conv_data.conversation_group_id} - "
                    f"{request.metric_identifier}"
                )
            else:
                summary = f"Turn {request.turn_id} - {request.metric_identifier}"
            logger.debug("Evaluating: %s", summary)

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

            # Get threshold
            level = (
                MetricLevel.CONVERSATION
                if request.is_conversation
                else MetricLevel.TURN
            )
            threshold = self.metric_manager.get_effective_threshold(
                request.metric_identifier, level, request.conv_data, request.turn_data
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
                response=request.turn_data.response or "" if request.turn_data else "",
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
            response=request.turn_data.response or "" if request.turn_data else "",
            execution_time=execution_time,
        )

    def _determine_status(self, score: float, threshold: Optional[float]) -> str:
        """Determine evaluation status based on score and threshold."""
        if threshold is None:
            threshold = 0.5  # This will also handle binary metrics
        return "PASS" if score >= threshold else "FAIL"

    def get_supported_frameworks(self) -> list[str]:
        """Get list of supported evaluation frameworks."""
        return list(self.handlers.keys())

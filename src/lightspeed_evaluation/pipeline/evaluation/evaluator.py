"""Metrics evaluation module - handles individual metric evaluation."""

import logging
import time
from typing import Optional

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.llm.custom import TokenTracker
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.metrics.custom import CustomMetrics
from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
from lightspeed_evaluation.core.metrics.manager import MetricLevel, MetricManager
from lightspeed_evaluation.core.metrics.ragas import RagasMetrics
from lightspeed_evaluation.core.metrics.script import ScriptEvalMetrics
from lightspeed_evaluation.core.models import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationScope,
)
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.core.system import ConfigLoader

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """Handles individual metric evaluation with proper scoring and status determination."""

    def __init__(
        self,
        config_loader: ConfigLoader,
        metric_manager: MetricManager,
        script_manager: ScriptExecutionManager,
    ) -> None:
        """Initialize Metric Evaluator."""
        self.config_loader = config_loader
        if config_loader.system_config is None:
            raise RuntimeError("Uninitialized system_config")

        llm_manager = LLMManager.from_system_config(config_loader.system_config)
        embedding_manager = EmbeddingManager.from_system_config(
            config_loader.system_config
        )

        # Initialize metric handlers
        self.ragas_metrics = RagasMetrics(llm_manager, embedding_manager)
        self.deepeval_metrics = DeepEvalMetrics(
            llm_manager,
            metric_manager=metric_manager,
        )
        self.custom_metrics = CustomMetrics(llm_manager)
        self.script_eval_metrics = ScriptEvalMetrics(script_manager)

        # Metric routing map
        self.handlers = {
            "ragas": self.ragas_metrics,
            "deepeval": self.deepeval_metrics,
            # Note: geval metrics are routed through deepeval_metrics handler
            "geval": self.deepeval_metrics,
            "custom": self.custom_metrics,
            "script": self.script_eval_metrics,
        }

        self.metric_manager = metric_manager

    def evaluate_metric(  # pylint: disable=too-many-locals
        self, request: EvaluationRequest
    ) -> Optional[EvaluationResult]:
        """Evaluate a single metric and return result.

        Tracks judge LLM token usage during evaluation and includes token counts
        in the result.

        Args:
            request: Evaluation request containing conversation data and metric
                identifier.

        Returns:
            EvaluationResult with score, status, and token usage, or None if the
            metric should be skipped (e.g., script metrics when API is disabled).
        """
        start_time = time.time()

        # Initialize token tracker for this evaluation
        token_tracker = TokenTracker()

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

            # Skip script metrics if API is disabled
            if (
                framework == "script"
                and self.config_loader.system_config is not None
                and not self.config_loader.system_config.api.enabled
            ):
                # Don't generate result for script metrics when API disabled
                return None

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

            # Start token tracking
            token_tracker.start()

            # Evaluate metric
            score, reason = self.handlers[framework].evaluate(  # type: ignore
                metric_name, request.conv_data, evaluation_scope
            )

            # Stop token tracking
            token_tracker.stop()

            execution_time = time.time() - start_time

            # Get token counts
            judge_input_tokens, judge_output_tokens = token_tracker.get_counts()

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
                judge_llm_input_tokens=judge_input_tokens,
                judge_llm_output_tokens=judge_output_tokens,
                api_input_tokens=(
                    request.turn_data.api_input_tokens if request.turn_data else 0
                ),
                api_output_tokens=(
                    request.turn_data.api_output_tokens if request.turn_data else 0
                ),
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Any evaluation error should result in ERROR status
            execution_time = time.time() - start_time
            # Stop token tracking on error
            token_tracker.stop()
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
            api_input_tokens=(
                request.turn_data.api_input_tokens if request.turn_data else 0
            ),
            api_output_tokens=(
                request.turn_data.api_output_tokens if request.turn_data else 0
            ),
        )

    def _determine_status(self, score: float, threshold: Optional[float]) -> str:
        """Determine evaluation status based on score and threshold."""
        if threshold is None:
            threshold = 0.5  # This will also handle binary metrics
        return "PASS" if score >= float(threshold) else "FAIL"

    def get_supported_frameworks(self) -> list[str]:
        """Get list of supported evaluation frameworks."""
        return list(self.handlers.keys())

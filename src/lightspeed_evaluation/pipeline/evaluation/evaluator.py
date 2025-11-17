"""Metrics evaluation module - handles individual metric evaluation.

Uses lazy loading to only import and instantiate metric frameworks when
they are actually used during evaluation.
"""

import logging
import time
from typing import Any, Optional

from lightspeed_evaluation.core.metrics.manager import MetricLevel, MetricManager
from lightspeed_evaluation.core.models import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationScope,
)
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.core.system import ConfigLoader

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """Handles individual metric evaluation with proper scoring and status determination.

    Metrics are lazy-loaded: framework-specific evaluators are only instantiated
    when a metric from that framework is first evaluated.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        metric_manager: MetricManager,
        script_manager: ScriptExecutionManager,
    ) -> None:
        """Initialize Metric Evaluator with lazy loading support."""
        self.config_loader = config_loader
        if config_loader.system_config is None:
            raise RuntimeError("Uninitialized system_config")

        self.script_manager = script_manager
        self.metric_manager = metric_manager

        # Lazy-loaded handlers - initialized on first use
        self._handlers: dict[str, Any] = {}
        self._llm_manager = None
        self._embedding_manager = None

    def _get_llm_manager(self) -> Any:
        """Lazy initialization of LLM manager."""
        # pylint: disable=import-outside-toplevel
        if self._llm_manager is None:
            # Import only when needed
            from lightspeed_evaluation.core.llm.manager import LLMManager

            self._llm_manager = LLMManager.from_system_config(
                self.config_loader.system_config  # type: ignore
            )
        return self._llm_manager

    def _get_embedding_manager(self) -> Any:
        """Lazy initialization of embedding manager."""
        # pylint: disable=import-outside-toplevel
        if self._embedding_manager is None:
            # Import only when needed
            from lightspeed_evaluation.core.embedding.manager import EmbeddingManager

            self._embedding_manager = EmbeddingManager.from_system_config(
                self.config_loader.system_config  # type: ignore
            )
        return self._embedding_manager

    def _get_handler(self, framework: str) -> Any:
        """Get or create handler for the specified framework.

        Handlers are lazy-loaded: only instantiated when first requested.
        """
        # pylint: disable=import-outside-toplevel
        if framework not in self._handlers:
            if framework == "ragas":
                # Import and instantiate only when needed
                from lightspeed_evaluation.core.metrics.ragas import RagasMetrics

                logger.debug("Initializing RagasMetrics handler (lazy load)")
                self._handlers["ragas"] = RagasMetrics(
                    self._get_llm_manager(), self._get_embedding_manager()
                )
            elif framework == "deepeval":
                from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics

                logger.debug("Initializing DeepEvalMetrics handler (lazy load)")
                self._handlers["deepeval"] = DeepEvalMetrics(self._get_llm_manager())
            elif framework == "custom":
                from lightspeed_evaluation.core.metrics.custom import CustomMetrics

                logger.debug("Initializing CustomMetrics handler (lazy load)")
                self._handlers["custom"] = CustomMetrics(self._get_llm_manager())
            elif framework == "script":
                from lightspeed_evaluation.core.metrics.script import ScriptEvalMetrics

                logger.debug("Initializing ScriptEvalMetrics handler (lazy load)")
                self._handlers["script"] = ScriptEvalMetrics(self.script_manager)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

        return self._handlers[framework]

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

            # Skip script metrics if API is disabled
            if (
                framework == "script"
                and self.config_loader.system_config is not None
                and not self.config_loader.system_config.api.enabled
            ):
                # Don't generate result for script metrics when API disabled
                return None

            # Get handler (lazy-loaded)
            try:
                handler = self._get_handler(framework)
            except ValueError as e:
                execution_time = time.time() - start_time
                return self._create_error_result(request, str(e), execution_time)

            # Create evaluation scope
            evaluation_scope = EvaluationScope(
                turn_idx=request.turn_idx,
                turn_data=request.turn_data,
                is_conversation=request.is_conversation,
            )

            # Evaluate metric
            score, reason = handler.evaluate(  # type: ignore
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
        return "PASS" if score >= float(threshold) else "FAIL"

    def get_supported_frameworks(self) -> list[str]:
        """Get list of supported evaluation frameworks."""
        return ["ragas", "deepeval", "custom", "script"]

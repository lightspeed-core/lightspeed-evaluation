"""Metrics evaluation module - handles individual metric evaluation."""

import json
import logging
import time
from typing import Any, Optional

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.llm.custom import TokenTracker
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.metrics.custom import CustomMetrics
from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
from lightspeed_evaluation.core.metrics.manager import MetricLevel, MetricManager
from lightspeed_evaluation.core.metrics.nlp import NLPMetrics
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


def _to_json_str(value: Any) -> Optional[str]:
    """Convert any value to JSON string. Returns None for empty values."""
    if value is None or value == [] or value == {}:
        return None
    try:
        return json.dumps(value, indent=None, default=str)
    except (TypeError, ValueError):
        return str(value)


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
        self.metric_manager = metric_manager

        if config_loader.system_config is None:
            raise RuntimeError("Uninitialized system_config")

        llm_manager = LLMManager.from_system_config(config_loader.system_config)
        embedding_manager = EmbeddingManager.from_system_config(
            config_loader.system_config
        )

        # Initialize metric handlers and routing map
        self.handlers = {
            "nlp": NLPMetrics(),
            "ragas": RagasMetrics(llm_manager, embedding_manager),
            "deepeval": DeepEvalMetrics(llm_manager, metric_manager=metric_manager),
            "geval": DeepEvalMetrics(llm_manager, metric_manager=metric_manager),
            "custom": CustomMetrics(llm_manager, metric_manager=metric_manager),
            "script": ScriptEvalMetrics(script_manager),
        }

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

            turn_data = request.turn_data
            return EvaluationResult(
                conversation_group_id=request.conv_data.conversation_group_id,
                tag=request.conv_data.tag,
                turn_id=request.turn_id,
                metric_identifier=request.metric_identifier,
                result=status,
                score=score,
                threshold=threshold,
                reason=reason,
                query=turn_data.query if turn_data else "",
                response=turn_data.response or "" if turn_data else "",
                execution_time=execution_time,
                api_input_tokens=(
                    request.turn_data.api_input_tokens if request.turn_data else 0
                ),
                api_output_tokens=(
                    request.turn_data.api_output_tokens if request.turn_data else 0
                ),
                judge_llm_input_tokens=judge_input_tokens,
                judge_llm_output_tokens=judge_output_tokens,
                # Streaming performance metrics
                time_to_first_token=(
                    turn_data.time_to_first_token if turn_data else None
                ),
                streaming_duration=(
                    turn_data.streaming_duration if turn_data else None
                ),
                tokens_per_second=(turn_data.tokens_per_second if turn_data else None),
                tool_calls=_to_json_str(turn_data.tool_calls) if turn_data else None,
                contexts=_to_json_str(turn_data.contexts) if turn_data else None,
                expected_response=turn_data.expected_response if turn_data else None,
                expected_intent=turn_data.expected_intent if turn_data else None,
                expected_keywords=(
                    _to_json_str(turn_data.expected_keywords) if turn_data else None
                ),
                expected_tool_calls=(
                    _to_json_str(turn_data.expected_tool_calls) if turn_data else None
                ),
                metrics_metadata=self._extract_metadata_for_csv(request),
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
        turn_data = request.turn_data
        return EvaluationResult(
            conversation_group_id=request.conv_data.conversation_group_id,
            tag=request.conv_data.tag,
            turn_id=request.turn_id,
            metric_identifier=request.metric_identifier,
            result="ERROR",
            score=None,
            threshold=None,
            reason=reason,
            query=turn_data.query if turn_data else "",
            response=turn_data.response or "" if turn_data else "",
            execution_time=execution_time,
            api_input_tokens=turn_data.api_input_tokens if turn_data else 0,
            api_output_tokens=turn_data.api_output_tokens if turn_data else 0,
            # Streaming performance metrics
            time_to_first_token=turn_data.time_to_first_token if turn_data else None,
            streaming_duration=turn_data.streaming_duration if turn_data else None,
            tokens_per_second=turn_data.tokens_per_second if turn_data else None,
            metrics_metadata=self._extract_metadata_for_csv(request),
        )

    def _determine_status(self, score: float, threshold: Optional[float]) -> str:
        """Determine evaluation status based on score and threshold."""
        if threshold is None:
            threshold = 0.5  # This will also handle binary metrics
        return "PASS" if score >= float(threshold) else "FAIL"

    def _extract_metadata_for_csv(self, request: EvaluationRequest) -> Optional[str]:
        """Extract metadata fields for CSV column (excluding threshold and metric_identifier).

        Args:
            request: Evaluation request containing metric and level information.

        Returns:
            JSON-encoded string of metadata fields, or None if no metadata found.
        """
        level = (
            MetricLevel.CONVERSATION if request.is_conversation else MetricLevel.TURN
        )

        # Get full metadata dictionary
        metadata = self.metric_manager.get_metric_metadata(
            request.metric_identifier, level, request.conv_data, request.turn_data
        )

        if not metadata:
            return None

        # Exclude fields that are already in CSV columns
        filtered_metadata = {
            k: v
            for k, v in metadata.items()
            if k not in ("threshold", "metric_identifier")
        }

        if not filtered_metadata:
            return None

        # Serialize to JSON
        return _to_json_str(filtered_metadata)

    def get_supported_frameworks(self) -> list[str]:
        """Get list of supported evaluation frameworks."""
        return list(self.handlers.keys())

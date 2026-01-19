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
from lightspeed_evaluation.core.system.exceptions import EvaluationError
from lightspeed_evaluation.core.system.validator import METRIC_REQUIREMENTS

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
            "custom": CustomMetrics(llm_manager),
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
            framework = request.metric_identifier.split(":", 1)[0]

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

            # Get threshold
            level = (
                MetricLevel.CONVERSATION
                if request.is_conversation
                else MetricLevel.TURN
            )
            threshold = self.metric_manager.get_effective_threshold(
                request.metric_identifier, level, request.conv_data, request.turn_data
            )

            # Evaluate metric
            score, reason, status, judge_input_tokens, judge_output_tokens = (
                self.evaluate(request, evaluation_scope, threshold)
            )

            execution_time = time.time() - start_time

            if score is None:
                return self._create_error_result(request, reason, execution_time)

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
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Any evaluation error should result in ERROR status
            execution_time = time.time() - start_time
            return self._create_error_result(
                request, f"Evaluation error: {e}", execution_time
            )

    def evaluate(  # pylint: disable=too-many-locals,too-many-statements
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        threshold: Optional[float],
    ) -> tuple[Optional[float], str, str, int, int]:
        """Evaluate metric logic, handling expected_response lists.

        Supports multiple expected_response values for metrics that require it.
        Iterates through alternatives until PASS or returns max score on failure.

        Args:
            request: Evaluation request containing conversation data and metric.
            evaluation_scope: Scope with turn data and evaluation context.
            threshold: Score threshold for PASS/FAIL determination.

        Returns:
            Tuple of (score, reason, status, judge_input_tokens, judge_output_tokens).

        Raises:
            RuntimeError: When required fields for metric are missing.
            EvaluationError: When metric evaluation fails unexpectedly.
        """
        # Parse framework and metric info
        framework, metric_name = request.metric_identifier.split(":", 1)

        # Initialize token tracker
        token_tracker = TokenTracker()
        # Initialize helper variables
        score = None
        reason = ""
        status = "FAIL"
        judge_input_tokens, judge_output_tokens = 0, 0

        # Decision logic for expected_response handling
        has_expected_response_in_requirements = (
            request.metric_identifier in METRIC_REQUIREMENTS
            and "expected_response"
            in METRIC_REQUIREMENTS[request.metric_identifier]["required_fields"]
        )
        metric_has_no_requirements = (
            request.metric_identifier not in METRIC_REQUIREMENTS
        )
        multiple_expected_responses = (
            evaluation_scope.turn_data is not None
            and evaluation_scope.turn_data.expected_response is not None
            and isinstance(evaluation_scope.turn_data.expected_response, list)
        )

        try:
            ## Multiple expected_responses handling
            if has_expected_response_in_requirements and multiple_expected_responses:
                # This check satisfies the linter but is logically redundant
                if (
                    evaluation_scope.turn_data is None
                    or evaluation_scope.turn_data.expected_response is None
                ):
                    raise RuntimeError(
                        f"Metric '{request.metric_identifier}' requires 'expected_response' field. "
                        f"Could not proceed with evaluation."
                    )
                score_max = -float("inf")
                reason_acc = ""

                for idx, expected_response in enumerate(
                    evaluation_scope.turn_data.expected_response
                ):
                    logger.debug(
                        "Running evaluation with expected_response %d/%d: %s",
                        idx + 1,
                        len(evaluation_scope.turn_data.expected_response),
                        expected_response,
                    )
                    alt_turn_data = evaluation_scope.turn_data.model_copy(
                        update={"expected_response": expected_response}
                    )
                    alt_scope = EvaluationScope(
                        turn_idx=evaluation_scope.turn_idx,
                        turn_data=alt_turn_data,
                        is_conversation=evaluation_scope.is_conversation,
                    )

                    # Evaluate metric
                    token_tracker.reset()
                    token_tracker.start()
                    score, reason = self.handlers[framework].evaluate(
                        metric_name, request.conv_data, alt_scope
                    )
                    token_tracker.stop()

                    # Accumulate token counts
                    input_tokens, output_tokens = token_tracker.get_counts()
                    judge_input_tokens += input_tokens
                    judge_output_tokens += output_tokens
                    logger.debug(
                        "Cumulative judge input tokens: %s, Cumulative judge output tokens: %s",
                        judge_input_tokens,
                        judge_output_tokens,
                    )

                    # Determine next steps
                    if score is not None:
                        status = self._determine_status(score, threshold)
                    if status == "PASS":
                        # Expected response PASSED
                        break
                    # Expected response did not PASS; keep track of highest score
                    score_max = max(
                        score_max, score if score is not None else score_max
                    )
                    reason_acc += f"{score}; {reason}\n"

                # If no PASS found, return highest score and accumulated reasons
                if status != "PASS":
                    score = score_max if score_max != -float("inf") else None
                    reason = reason_acc.strip()

            # For other metrics missing in METRIC_REQUIREMENTS (GEval/Deepeval)
            # multiple expected_responses handling is not supported.
            # Will evaluate only first expected_response from the list.
            elif metric_has_no_requirements and multiple_expected_responses:
                # This check satisfies the linter but is logically redundant
                if (
                    evaluation_scope.turn_data is None
                    or evaluation_scope.turn_data.expected_response is None
                ):
                    raise RuntimeError(
                        f"Metric '{request.metric_identifier}' requires 'expected_response' field. "
                        f"Could not proceed with evaluation."
                    )
                first_expected_response = evaluation_scope.turn_data.expected_response[
                    0
                ]
                logger.debug(
                    "Running evaluation with expected_response: %s",
                    first_expected_response,
                )
                alt_turn_data = evaluation_scope.turn_data.model_copy(
                    update={"expected_response": first_expected_response}
                )
                alt_scope = EvaluationScope(
                    turn_idx=evaluation_scope.turn_idx,
                    turn_data=alt_turn_data,
                    is_conversation=evaluation_scope.is_conversation,
                )

                token_tracker.start()
                score, reason = self.handlers[framework].evaluate(
                    metric_name, request.conv_data, alt_scope
                )
                token_tracker.stop()
                judge_input_tokens, judge_output_tokens = token_tracker.get_counts()

                if score is not None:
                    status = self._determine_status(score, threshold)

            ## Single expected_response handling or not supported
            else:
                token_tracker.start()
                score, reason = self.handlers[framework].evaluate(
                    metric_name, request.conv_data, evaluation_scope
                )
                token_tracker.stop()
                judge_input_tokens, judge_output_tokens = token_tracker.get_counts()

                if score is not None:
                    status = self._determine_status(score, threshold)

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Stop token tracking on error
            token_tracker.stop()
            raise EvaluationError(e) from e

        return score, reason, status, judge_input_tokens, judge_output_tokens

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
        )

    def _determine_status(self, score: float, threshold: Optional[float]) -> str:
        """Determine evaluation status based on score and threshold."""
        if threshold is None:
            threshold = 0.5  # This will also handle binary metrics
        return "PASS" if score >= float(threshold) else "FAIL"

    def get_supported_frameworks(self) -> list[str]:
        """Get list of supported evaluation frameworks."""
        return list(self.handlers.keys())

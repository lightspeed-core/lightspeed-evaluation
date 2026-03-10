"""Metrics evaluation module - handles individual metric evaluation."""

import json
import logging
import time
from typing import Any, Optional

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.llm.token_tracker import TokenTracker
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
    MetricResult,
)
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import (
    ConfigurationError,
    EvaluationError,
)
from lightspeed_evaluation.core.system.validator import (
    METRIC_REQUIREMENTS,
    check_metric_required_data,
)
from lightspeed_evaluation.core.constants import NON_LLM_FRAMEWORKS
from lightspeed_evaluation.pipeline.evaluation.judges import JudgeOrchestrator

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

        self.system_config = config_loader.system_config
        self.llm_manager = LLMManager.from_system_config(config_loader.system_config)
        self.embedding_manager = EmbeddingManager.from_system_config(
            config_loader.system_config
        )

        # Initialize default metric handlers (used for primary judge or non-panel metrics)
        self.handlers = {
            "nlp": NLPMetrics(),
            "ragas": RagasMetrics(self.llm_manager, self.embedding_manager),
            "deepeval": DeepEvalMetrics(
                self.llm_manager, metric_manager=metric_manager
            ),
            "geval": DeepEvalMetrics(self.llm_manager, metric_manager=metric_manager),
            "custom": CustomMetrics(self.llm_manager, metric_manager=metric_manager),
            "script": ScriptEvalMetrics(script_manager),
        }

        # Judge orchestrator handles multi-judge evaluation and aggregation
        self.judge_orchestrator = JudgeOrchestrator(
            llm_manager=self.llm_manager,
            primary_handlers=self.handlers,
            handler_factory=self._create_handler_for_judge,
            status_determiner=self._determine_status,
        )

    def _create_handler_for_judge(
        self, framework: str, judge_manager: LLMManager
    ) -> Any:
        """Create a metric handler for a non-primary judge.

        Used by JudgeOrchestrator to create handlers for panel judges.
        """
        if framework == "ragas":
            return RagasMetrics(judge_manager, self.embedding_manager)
        if framework in ("deepeval", "geval"):
            return DeepEvalMetrics(judge_manager, metric_manager=self.metric_manager)
        if framework == "custom":
            return CustomMetrics(judge_manager, metric_manager=self.metric_manager)
        raise ConfigurationError(f"Unsupported LLM framework for panel: {framework}")

    def evaluate_metric(  # pylint: disable=too-many-locals
        self, request: EvaluationRequest
    ) -> Optional[EvaluationResult]:
        """Evaluate a single metric and return complete evaluation result.

        Args:
            request: Evaluation request with conversation data and metric identifier.

        Returns:
            EvaluationResult with score, result, token usage, and execution time,
            or None if metric should be skipped (e.g., script metrics when API disabled).
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

            # Check required data for metric (after API call); skip with ERROR if missing
            if (
                request.turn_data is not None
                and request.metric_identifier in METRIC_REQUIREMENTS
            ):
                ok, msg = check_metric_required_data(
                    request.turn_data, request.metric_identifier
                )
                if not ok:
                    execution_time = time.time() - start_time
                    logger.warning(
                        "Skipping metric due to missing required data: %s", msg
                    )
                    return self._create_error_result(request, msg, execution_time)

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
            metric_result = self._evaluate_wrapper(request, evaluation_scope, threshold)

            execution_time = time.time() - start_time

            turn_data = request.turn_data
            return EvaluationResult(
                **metric_result.model_dump(),
                conversation_group_id=request.conv_data.conversation_group_id,
                tag=request.conv_data.tag,
                turn_id=request.turn_id,
                metric_identifier=request.metric_identifier,
                metric_metadata=self._extract_metadata_for_csv(request),
                query=turn_data.query if turn_data else "",
                response=turn_data.response or "" if turn_data else "",
                execution_time=execution_time,
                api_input_tokens=(
                    request.turn_data.api_input_tokens if request.turn_data else 0
                ),
                api_output_tokens=(
                    request.turn_data.api_output_tokens if request.turn_data else 0
                ),
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

        except EvaluationError as e:
            # Any evaluation error should result in ERROR status
            execution_time = time.time() - start_time
            return self._create_error_result(
                request, f"Evaluation error: {e}", execution_time
            )

    def _will_use_panel(self, metric_identifier: str) -> bool:
        """Check if panel of judges will be used for this metric.

        Args:
            metric_identifier: Metric identifier (e.g., 'ragas:faithfulness').

        Returns:
            True if panel evaluation will be used, False otherwise.
        """
        framework = metric_identifier.split(":", 1)[0]

        # Non-LLM frameworks never use panel
        if framework in NON_LLM_FRAMEWORKS:
            return False

        # should_use_panel_for_metric handles the no-panel case internally
        return self.llm_manager.should_use_panel_for_metric(metric_identifier)

    def _evaluate_wrapper(  # pylint: disable=too-many-locals,too-many-statements
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        threshold: Optional[float],
    ) -> MetricResult:
        """Wrap evaluation logic with token tracking and multiple expected response handling.

        Args:
            request: Evaluation request with conversation and metric data.
            evaluation_scope: Scope containing turn/conversation context.
            threshold: Optional score threshold for pass/fail determination.

        Returns:
            MetricResult containing score, result, reason, and judge llm token usage.

        Note:
            Multiple expected_responses handling is NOT supported with panel of judges.
            When panel is used, only the first expected_response is evaluated.
        """
        # Initialize token tracker for this evaluation
        token_tracker = TokenTracker()
        token_tracker.start()

        try:
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

            # Check if panel of judges will be used
            uses_panel = self._will_use_panel(request.metric_identifier)

            # Panel of judges: skip multiple expected responses (use first only)
            # This is a design choice to avoid N judges x M expected_responses complexity
            if uses_panel and multiple_expected_responses:
                logger.warning(
                    "Conv %s: Multiple expected_responses not supported with judge panel. "
                    "Using first expected_response only. This is a known limitation.",
                    request.conv_data.conversation_group_id,
                )
                metric_result = self._evaluate_panel_single_expected_response(
                    request, evaluation_scope, token_tracker, threshold
                )

            ## Multiple expected_responses handling (single judge only)
            elif has_expected_response_in_requirements and multiple_expected_responses:
                metric_result = self._evaluate_multiple_expected_responses(
                    request, evaluation_scope, token_tracker, threshold
                )

            # For other metrics missing in METRIC_REQUIREMENTS (GEval/Deepeval)
            # multiple expected_responses handling is not supported.
            # Will evaluate only first expected_response from the list.
            elif metric_has_no_requirements and multiple_expected_responses:
                metric_result = (
                    self._evaluate_multiple_expected_responses_not_supported(
                        request, evaluation_scope, token_tracker, threshold
                    )
                )

            ## Single expected_response handling or not supported
            else:
                try:
                    metric_result = self._evaluate(
                        request, evaluation_scope, token_tracker, threshold
                    )
                except EvaluationError as e:
                    logger.error(
                        "Conv %s: Error during evaluation: %s",
                        request.conv_data.conversation_group_id,
                        str(e),
                    )
                    input_tokens, output_tokens = token_tracker.get_counts()
                    metric_result = MetricResult(
                        result="ERROR",
                        score=None,
                        threshold=threshold,
                        reason=f"Evaluation error: {e}",
                        judge_llm_input_tokens=input_tokens,
                        judge_llm_output_tokens=output_tokens,
                    )
        finally:
            # Ensure callback is unregistered even on error
            token_tracker.stop()

        return metric_result

    def _evaluate_multiple_expected_responses(
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
        threshold: Optional[float],
    ) -> MetricResult:
        """Evaluate metric against multiple expected responses.

        Stop on first PASS. If none PASS, return highest score and accumulated reasons
        with according scores.

        Args:
            request: Evaluation request with conversation and metric data.
            evaluation_scope: Scope with turn data containing list of expected responses.
            token_tracker: Token tracker for accumulating judge LLM usage.
            threshold: Optional score threshold for pass/fail determination.

        Returns:
            MetricResult with highest score or first PASS, with accumulated judge llm token counts.
        """
        # Initialize helper variables
        judge_llm_input_tokens, judge_llm_output_tokens = 0, 0

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
        metric_result = None

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

            # Evaluate metric - catch exceptions to preserve accumulated tokens
            try:
                metric_result = self._evaluate(
                    request, alt_scope, token_tracker, threshold
                )
            except EvaluationError as e:
                # Include tokens from failing call (LLM may have used tokens before error)
                # Add current iteration's tokens to accumulated total
                judge_llm_input_tokens += token_tracker.get_counts()[0]
                judge_llm_output_tokens += token_tracker.get_counts()[1]
                logger.error(
                    "Conv %s: Error during evaluation iteration %d: %s",
                    request.conv_data.conversation_group_id,
                    idx + 1,
                    str(e),
                )
                return MetricResult(
                    result="ERROR",
                    score=None,
                    threshold=threshold,
                    reason=f"Evaluation error at iteration {idx + 1}: {e}",
                    judge_llm_input_tokens=judge_llm_input_tokens,
                    judge_llm_output_tokens=judge_llm_output_tokens,
                )

            # Accumulate token counts
            judge_llm_input_tokens += metric_result.judge_llm_input_tokens
            judge_llm_output_tokens += metric_result.judge_llm_output_tokens
            logger.debug(
                "Conv %s: Cumulative judge input tokens: %s, Cumulative judge output tokens: %s",
                request.conv_data.conversation_group_id,
                judge_llm_input_tokens,
                judge_llm_output_tokens,
            )
            logger.debug("Metric result: %s", metric_result)

            # Determine next steps
            if metric_result.result == "PASS":
                # Expected response PASSED
                break
            # Expected response did not PASS; keep track of highest score
            score_max = max(
                score_max,
                metric_result.score if metric_result.score is not None else score_max,
            )
            reason_acc += f"{metric_result.score}; {metric_result.reason}\n"

        # This should never happen due to the empty list check above, but satisfies type checker
        if metric_result is None:
            raise RuntimeError(
                f"Metric '{request.metric_identifier}' requires 'expected_response' field. "
                f"Could not proceed with evaluation."
            )

        # EvaluateResult actualization
        # If no PASS found, return highest score and accumulated reasons
        if metric_result.result != "PASS":
            metric_result.score = score_max if score_max != -float("inf") else None
            metric_result.reason = reason_acc.strip()

        # Update token counts in final result
        metric_result.judge_llm_input_tokens = judge_llm_input_tokens
        metric_result.judge_llm_output_tokens = judge_llm_output_tokens

        return metric_result

    def _evaluate_multiple_expected_responses_not_supported(
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
        threshold: Optional[float],
    ) -> MetricResult:
        """Evaluate metric using only first expected response from list.

        Args:
            request: Evaluation request with conversation and metric data.
            evaluation_scope: Scope with turn data containing list of expected responses.
            token_tracker: Token tracker for judge LLM usage.
            threshold: Optional score threshold for pass/fail determination.

        Returns:
            MetricResult from evaluating against first expected response only.
        """
        # This check satisfies the linter but is logically redundant
        if (
            evaluation_scope.turn_data is None
            or evaluation_scope.turn_data.expected_response is None
        ):
            raise RuntimeError(
                f"Metric '{request.metric_identifier}' requires 'expected_response' field. "
                f"Could not proceed with evaluation."
            )
        first_expected_response = evaluation_scope.turn_data.expected_response[0]
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

        # Evaluate metric - catch exceptions to return error with captured tokens
        try:
            metric_result = self._evaluate(request, alt_scope, token_tracker, threshold)
        except EvaluationError as e:
            logger.error(
                "Conv %s: Error during evaluation: %s",
                request.conv_data.conversation_group_id,
                str(e),
            )
            input_tokens, output_tokens = token_tracker.get_counts()
            return MetricResult(
                result="ERROR",
                score=None,
                threshold=threshold,
                reason=f"Evaluation error: {e}",
                judge_llm_input_tokens=input_tokens,
                judge_llm_output_tokens=output_tokens,
            )

        return metric_result

    def _evaluate_panel_single_expected_response(
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
        threshold: Optional[float],
    ) -> MetricResult:
        """Evaluate with panel of judges using only first expected response.

        Panel of judges does not support multiple expected responses due to
        complexity (N judges x M expected_responses). Only the first expected
        response from the list is used.

        Args:
            request: Evaluation request with conversation and metric data.
            evaluation_scope: Scope with turn data containing list of expected responses.
            token_tracker: Token tracker for judge LLM usage.
            threshold: Optional score threshold for pass/fail determination.

        Returns:
            MetricResult from panel evaluation against first expected response only.
        """
        if (
            evaluation_scope.turn_data is None
            or evaluation_scope.turn_data.expected_response is None
        ):
            raise RuntimeError(
                f"Metric '{request.metric_identifier}' requires 'expected_response' field. "
                f"Could not proceed with evaluation."
            )

        first_expected_response = evaluation_scope.turn_data.expected_response[0]
        logger.debug(
            "Panel evaluation with first expected_response only: %s",
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

        try:
            metric_result = self._evaluate(request, alt_scope, token_tracker, threshold)
        except EvaluationError as e:
            logger.error(
                "Conv %s: Error during panel evaluation: %s",
                request.conv_data.conversation_group_id,
                str(e),
            )
            input_tokens, output_tokens = token_tracker.get_counts()
            return MetricResult(
                result="ERROR",
                score=None,
                threshold=threshold,
                reason=f"Panel evaluation error: {e}",
                judge_llm_input_tokens=input_tokens,
                judge_llm_output_tokens=output_tokens,
            )

        return metric_result

    def _evaluate(
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
        threshold: Optional[float],
    ) -> MetricResult:
        """Execute metric evaluation using appropriate framework handler.

        Supports multi-judge evaluation when judge panel is configured and
        metric is in enabled_metrics. Otherwise, uses single judge evaluation.

        Args:
            request: Evaluation request with conversation and metric data.
            evaluation_scope: Scope containing turn/conversation context.
            token_tracker: Token tracker for recording judge LLM usage.
            threshold: Optional score threshold for pass/fail determination.

        Returns:
            MetricResult with score, result, reason, and judge llm token counts.
            For multi-judge evaluation, includes per-judge scores in judge_scores field.
        """
        framework = request.metric_identifier.split(":", 1)[0]

        # Non-LLM metrics: no judge LLM involved (nlp, script)
        if framework in NON_LLM_FRAMEWORKS:
            return self._evaluate_non_llm(request, evaluation_scope, threshold)

        # LLM metrics: delegate to judge orchestrator
        return self.judge_orchestrator.evaluate_with_judges(
            request, evaluation_scope, token_tracker, threshold
        )

    def _evaluate_non_llm(
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        threshold: Optional[float],
    ) -> MetricResult:
        """Execute non-LLM metric evaluation (nlp, script).

        These metrics don't use judge LLM, so judge_scores is None and
        token counts are always 0.
        """
        framework, metric_name = request.metric_identifier.split(":", 1)

        status = "ERROR"
        score, reason = self.handlers[framework].evaluate(
            metric_name, request.conv_data, evaluation_scope
        )

        if score is not None:
            status = self._determine_status(score, threshold)

        return MetricResult(
            result=status,
            score=score,
            threshold=threshold,
            reason=reason,
            judge_llm_input_tokens=0,
            judge_llm_output_tokens=0,
            judge_scores=None,  # No judge LLM involved
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
            metric_metadata=self._extract_metadata_for_csv(request),
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

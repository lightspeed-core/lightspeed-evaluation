"""Judge orchestration module - handles multi-judge evaluation and aggregation."""

import logging
from typing import Any, Callable, Optional

from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.llm.token_tracker import TokenTracker
from lightspeed_evaluation.core.models import (
    EvaluationRequest,
    EvaluationScope,
    JudgeScore,
    MetricResult,
)
from lightspeed_evaluation.core.system.exceptions import EvaluationError

logger = logging.getLogger(__name__)

# Type aliases
HandlerFactory = Callable[[str, LLMManager], Any]
StatusDeterminer = Callable[[float, Optional[float]], str]


class JudgeOrchestrator:
    """Orchestrates evaluation across single or multiple judges.

    Handles:
    - Getting appropriate judges for a metric (panel vs single)
    - Creating/caching handlers per judge
    - Collecting scores from all judges
    - Aggregating results
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        primary_handlers: dict[str, Any],
        handler_factory: HandlerFactory,
        status_determiner: StatusDeterminer,
    ) -> None:
        """Initialize Judge Orchestrator.

        Args:
            llm_manager: Primary LLM manager (may have judge panel configured)
            primary_handlers: Pre-initialized handlers for primary judge
            handler_factory: Function to create handlers for non-primary judges
            status_determiner: Function to determine pass/fail from score and threshold
        """
        self.llm_manager = llm_manager
        self.primary_handlers = primary_handlers
        self._handler_factory = handler_factory
        self._status_determiner = status_determiner

        # Cache for judge-specific handlers (keyed by judge_id -> framework -> handler)
        self._judge_handlers: dict[str, dict[str, Any]] = {}

        # Flag to avoid warning spam for unimplemented aggregation strategies
        self._warned_aggregation_strategy = False

    def evaluate_with_judges(
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
        threshold: Optional[float],
    ) -> MetricResult:
        """Execute metric evaluation using appropriate judges.

        For metrics in enabled_metrics (or when enabled_metrics is None),
        uses all panel judges. Otherwise uses only primary judge.
        Always returns a list of JudgeScore entries (even for single judge).

        Args:
            request: Evaluation request (contains metric_identifier and conv_data)
            evaluation_scope: Scope with turn/conversation context
            token_tracker: Token usage tracker
            threshold: Score threshold for pass/fail

        Returns:
            MetricResult with aggregated score and individual judge scores
        """
        # Extract framework and metric_name from identifier
        framework, metric_name = request.metric_identifier.split(":", 1)

        # Get judges for this metric (panel or single based on enabled_metrics)
        judge_managers = self.llm_manager.get_judges_for_metric(
            request.metric_identifier
        )

        logger.debug(
            "Evaluating %s with %d judge(s)",
            request.metric_identifier,
            len(judge_managers),
        )

        # Evaluate with each judge
        judge_scores, total_input, total_output = self._evaluate_all_judges(
            judge_managers,
            framework,
            metric_name,
            request,
            evaluation_scope,
            token_tracker,
        )

        # Aggregate scores
        aggregated_score, aggregated_reason = self.aggregate_scores(judge_scores)

        if aggregated_score is not None:
            status = self._status_determiner(aggregated_score, threshold)
        else:
            status = "ERROR"

        return MetricResult(
            result=status,
            score=aggregated_score,
            threshold=threshold,
            reason=aggregated_reason,
            judge_llm_input_tokens=total_input,
            judge_llm_output_tokens=total_output,
            judge_scores=judge_scores,
        )

    def _evaluate_all_judges(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        judge_managers: list[LLMManager],
        framework: str,
        metric_name: str,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
    ) -> tuple[list[JudgeScore], int, int]:
        """Evaluate metric with all judges and collect results.

        Returns:
            Tuple of (judge_scores, total_input_tokens, total_output_tokens)
        """
        judge_scores: list[JudgeScore] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for judge_manager in judge_managers:
            score_entry, input_tokens, output_tokens = self._evaluate_single_judge(
                judge_manager,
                framework,
                metric_name,
                request,
                evaluation_scope,
                token_tracker,
            )
            judge_scores.append(score_entry)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        return judge_scores, total_input_tokens, total_output_tokens

    def _evaluate_single_judge(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        judge_manager: LLMManager,
        framework: str,
        metric_name: str,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
    ) -> tuple[JudgeScore, int, int]:
        """Evaluate metric with a single judge.

        Returns:
            Tuple of (JudgeScore, input_tokens, output_tokens)
        """
        # Use judge_id from manager (pool key) - ensures uniqueness even when
        # multiple pool entries use the same underlying model
        judge_id = judge_manager.judge_id
        token_tracker.reset()

        try:
            handler = self._get_handler_for_judge(framework, judge_manager)
            score, reason = handler.evaluate(
                metric_name, request.conv_data, evaluation_scope
            )
            input_tokens, output_tokens = token_tracker.get_counts()

            logger.debug(
                "Judge %s: score=%s, tokens=%d/%d",
                judge_id,
                score,
                input_tokens,
                output_tokens,
            )

            return (
                JudgeScore(
                    judge_id=judge_id,
                    score=score,
                    reason=reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                input_tokens,
                output_tokens,
            )

        except EvaluationError as e:
            # Catch expected evaluation errors (LLM errors, metric errors, etc.)
            # Let unexpected exceptions (ConfigurationError, bugs) propagate
            logger.error("Judge %s failed: %s", judge_id, e)
            input_tokens, output_tokens = token_tracker.get_counts()

            return (
                JudgeScore(
                    judge_id=judge_id,
                    score=None,
                    reason=f"Evaluation error: {e}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                input_tokens,
                output_tokens,
            )

    def _get_handler_for_judge(self, framework: str, judge_manager: LLMManager) -> Any:
        """Get or create a metric handler for a specific judge.

        For primary manager, reuses existing handlers from primary_handlers.
        For other judges, handlers are cached by judge_id (pool key) to avoid recreation.
        """
        # Reuse existing handler if this is the primary manager
        if judge_manager is self.llm_manager:
            return self.primary_handlers[framework]

        # Use judge_id (pool key) for caching - ensures unique handlers even when
        # multiple pool entries use the same underlying model
        judge_id = judge_manager.judge_id

        # Check cache
        if judge_id in self._judge_handlers:
            if framework in self._judge_handlers[judge_id]:
                return self._judge_handlers[judge_id][framework]
        else:
            self._judge_handlers[judge_id] = {}

        # Create new handler using the factory
        handler = self._handler_factory(framework, judge_manager)
        self._judge_handlers[judge_id][framework] = handler
        return handler

    def aggregate_scores(
        self, judge_scores: list[JudgeScore]
    ) -> tuple[Optional[float], str]:
        """Aggregate scores from multiple judges.

        Currently implements 'max' strategy only. Other strategies (average,
        majority_vote) are planned for future implementation.
        For single judge, returns the judge's score and reason directly (even if None).

        Args:
            judge_scores: List of JudgeScore from all judges

        Returns:
            Tuple of (aggregated_score, aggregated_reason)
        """
        # Warn once if non-max strategy is configured (not yet implemented)
        panel = (
            self.llm_manager.system_config.judge_panel
            if self.llm_manager.system_config
            else None
        )
        if (
            panel
            and panel.aggregation_strategy != "max"
            and not self._warned_aggregation_strategy
        ):
            logger.warning(
                "Aggregation strategy '%s' is not yet implemented. "
                "Using 'max' instead. Other strategies coming soon.",
                panel.aggregation_strategy,
            )
            self._warned_aggregation_strategy = True

        # Single judge: return its score and reason directly (even if None)
        if len(judge_scores) == 1:
            return judge_scores[0].score, judge_scores[0].reason

        # Multiple judges: filter valid scores and aggregate
        # Extract scores directly, filtering out None values
        scores: list[float] = [js.score for js in judge_scores if js.score is not None]

        if not scores:
            # All judges failed
            return None, "All judges failed to produce a score"

        # Multiple judges: max score
        max_score = max(scores)
        return max_score, f"Max of {len(scores)} judges: {max_score:.3f}"

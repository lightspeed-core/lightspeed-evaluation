"""Judge orchestration module - handles multi-judge evaluation and aggregation."""

import logging
from statistics import mean
from typing import Any, Callable, Optional

from lightspeed_evaluation.core.constants import DEFAULT_METRIC_THRESHOLD
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

    def evaluate_with_judges(
        self,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
        threshold: Optional[float],
    ) -> MetricResult:
        """Evaluate a metric with the appropriate judges and aggregate results.

        Uses all panel judges for enabled metrics, or the primary judge only.

        Args:
            request: Contains metric identifier and conversation data.
            evaluation_scope: Turn or conversation context for the evaluation.
            token_tracker: Tracks token usage per judge call.
            threshold: Metric pass threshold from metadata. Defaults to 0.5
                when not specified.

        Returns:
            MetricResult with aggregated score and individual judge scores.
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
        judge_scores, token_totals = self._evaluate_all_judges(
            judge_managers,
            framework,
            metric_name,
            request,
            evaluation_scope,
            token_tracker,
        )

        # Aggregate scores (majority_vote sets PASS/FAIL without re-applying threshold to mean)
        aggregated_score, aggregated_reason, status_override = self.aggregate_scores(
            judge_scores, threshold
        )

        if status_override is not None:
            status = status_override
        elif aggregated_score is not None:
            status = self._status_determiner(aggregated_score, threshold)
        else:
            status = "ERROR"

        return MetricResult(
            result=status,
            score=aggregated_score,
            threshold=threshold,
            reason=aggregated_reason,
            judge_llm_input_tokens=token_totals["judge_input_tokens"],
            judge_llm_output_tokens=token_totals["judge_output_tokens"],
            embedding_tokens=token_totals["embedding_tokens"],
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
    ) -> tuple[list[JudgeScore], dict[str, int]]:
        """Evaluate metric with all judges and collect results.

        Args:
            judge_managers: LLM managers for each judge to evaluate with.
            framework: Metric framework name (e.g. ragas, deepeval).
            metric_name: Name of the metric within the framework.
            request: Contains conversation data for evaluation.
            evaluation_scope: Turn or conversation context.
            token_tracker: Tracks token usage per judge call.

        Returns:
            Tuple of (judge_scores, token_totals) where token_totals is a dict
                containing judge_input_tokens, judge_output_tokens, and
                embedding_tokens.
        """
        judge_scores: list[JudgeScore] = []
        token_totals = {
            "judge_input_tokens": 0,
            "judge_output_tokens": 0,
            "embedding_tokens": 0,
        }

        for judge_manager in judge_managers:
            score_entry = self._evaluate_single_judge(
                judge_manager,
                framework,
                metric_name,
                request,
                evaluation_scope,
                token_tracker,
            )
            judge_scores.append(score_entry)
            token_totals["judge_input_tokens"] += score_entry.judge_input_tokens
            token_totals["judge_output_tokens"] += score_entry.judge_output_tokens
            token_totals["embedding_tokens"] += score_entry.embedding_tokens

        return judge_scores, token_totals

    def _evaluate_single_judge(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        judge_manager: LLMManager,
        framework: str,
        metric_name: str,
        request: EvaluationRequest,
        evaluation_scope: EvaluationScope,
        token_tracker: TokenTracker,
    ) -> JudgeScore:
        """Evaluate metric with a single judge.

        On evaluation error, returns a JudgeScore with score=None instead
        of propagating the exception.

        Args:
            judge_manager: LLM manager for this judge.
            framework: Metric framework name (e.g. ragas, deepeval).
            metric_name: Name of the metric within the framework.
            request: Contains conversation data for evaluation.
            evaluation_scope: Turn or conversation context.
            token_tracker: Tracks token usage for this call.

        Returns:
            JudgeScore: Contains score, reason, and token usage for this judge.
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
            judge_input_tokens, judge_output_tokens = token_tracker.get_judge_counts()
            embedding_tokens = token_tracker.get_embedding_counts()

            logger.debug(
                "Judge %s: score=%s, tokens=%d/%d, embeddings=%d",
                judge_id,
                score,
                judge_input_tokens,
                judge_output_tokens,
                embedding_tokens,
            )

            return JudgeScore(
                judge_id=judge_id,
                score=score,
                reason=reason,
                judge_input_tokens=judge_input_tokens,
                judge_output_tokens=judge_output_tokens,
                embedding_tokens=embedding_tokens,
            )

        except EvaluationError as e:
            # Catch expected evaluation errors (LLM errors, metric errors, etc.)
            # Let unexpected exceptions (ConfigurationError, bugs) propagate
            logger.error("Judge %s failed: %s", judge_id, e)
            judge_input_tokens, judge_output_tokens = token_tracker.get_judge_counts()
            embedding_tokens = token_tracker.get_embedding_counts()

            return JudgeScore(
                judge_id=judge_id,
                score=None,
                reason=f"Evaluation error: {e}",
                judge_input_tokens=judge_input_tokens,
                judge_output_tokens=judge_output_tokens,
                embedding_tokens=embedding_tokens,
            )

    def _get_handler_for_judge(self, framework: str, judge_manager: LLMManager) -> Any:
        """Get or create a metric handler for a specific judge.

        Reuses existing handlers for the primary manager. For other judges,
        creates and caches handlers by judge_id to avoid recreation.

        Args:
            framework: Metric framework name (e.g. ragas, deepeval).
            judge_manager: LLM manager for the judge.

        Returns:
            The metric handler instance for the given framework and judge.
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
        self,
        judge_scores: list[JudgeScore],
        threshold: Optional[float] = None,
    ) -> tuple[Optional[float], str, Optional[str]]:
        """Aggregate scores from multiple judges based on the panel strategy.

        Judges that errored (score is None) are excluded from aggregation.
        With a single judge, its score and reason are returned directly.

        Strategies for multiple judges:
            - max: uses the highest valid score.
            - average: uses the mean of valid scores.
            - majority_vote: reports the mean as the score but determines
              pass/fail by strict majority of judges meeting the threshold.

        Args:
            judge_scores: List of score entries, one per judge.
            threshold: Metric pass threshold. Defaults to 0.5 when not set.

        Returns:
            Tuple of (aggregated_score, reason, status_override).
            status_override is only set for majority_vote (PASS/FAIL);
            None for other strategies.
        """
        panel = (
            self.llm_manager.system_config.judge_panel
            if self.llm_manager.system_config
            else None
        )
        strategy = panel.aggregation_strategy if panel else "max"

        # Single judge: return its score and reason directly (even if None)
        if len(judge_scores) == 1:
            return judge_scores[0].score, judge_scores[0].reason, None

        # Multiple judges: collect valid scores
        scores: list[float] = [js.score for js in judge_scores if js.score is not None]

        if not scores:
            return None, "All judges failed to produce a score", None

        n_valid = len(scores)

        if strategy == "average":
            avg_score = mean(scores)
            return (
                avg_score,
                f"Average of {n_valid} judges: {avg_score:.3f}",
                None,
            )

        if strategy == "majority_vote":
            return self._aggregate_majority_vote(scores, threshold)

        # max (default)
        max_score = max(scores)
        return (
            max_score,
            f"Max of {n_valid} judges: {max_score:.3f}",
            None,
        )

    def _aggregate_majority_vote(
        self,
        scores: list[float],
        threshold: Optional[float],
    ) -> tuple[float, str, Optional[str]]:
        """Aggregate using majority vote strategy.

        Reports the mean of valid scores as the score. Pass/fail is decided by
        whether a strict majority of judges individually meet the threshold
        (passes > n/2). Ties fail (e.g. 1/2 is FAIL, 2/3 is PASS).

        A status_override is returned so the caller does not re-compare the
        mean against the metric threshold.

        Args:
            scores: Valid (non-None) judge scores.
            threshold: Metric pass threshold. Defaults to 0.5 when not set.

        Returns:
            Tuple of (mean_score, reason, status_override).
        """
        n_valid = len(scores)
        avg_score = mean(scores)
        effective_threshold = (
            float(threshold) if threshold is not None else DEFAULT_METRIC_THRESHOLD
        )
        passes = sum(1 for s in scores if s >= effective_threshold)
        majority_pass = passes > n_valid / 2
        status_override = "PASS" if majority_pass else "FAIL"
        thr_label = (
            f"{effective_threshold:.3f}"
            if threshold is not None
            else f"{effective_threshold:.3f} (default)"
        )
        reason = (
            f"Majority vote ({passes}/{n_valid} pass threshold {thr_label}): "
            f"mean score {avg_score:.3f}"
        )
        return avg_score, reason, status_override

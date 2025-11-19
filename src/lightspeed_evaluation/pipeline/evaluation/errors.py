"""Error handling module for evaluation errors."""

import logging

from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult, TurnData

logger = logging.getLogger(__name__)


class EvaluationErrorHandler:
    """Handles error scenarios during evaluation."""

    def __init__(self) -> None:
        """Initialize error handler."""
        self.results: list[EvaluationResult] = []

    def mark_all_metrics_as_error(
        self,
        conv_data: EvaluationData,
        error_reason: str,
        resolved_turn_metrics: list[list[str]],
        resolved_conversation_metrics: list[str],
    ) -> list[EvaluationResult]:
        """Mark all turn and conversation metrics as ERROR when there is an error.

        Args:
            conv_data: Conversation data
            error_reason: Reason for error
            resolved_turn_metrics: Pre-resolved turn metrics
            resolved_conversation_metrics: Pre-resolved conversation metrics

        Returns:
            list[EvaluationResult]: ERROR results for all metrics
        """
        logger.warning(
            "Marking all metrics as ERROR for conversation %s: %s",
            conv_data.conversation_group_id,
            error_reason,
        )
        error_results = []

        # Mark all turn-level metrics as ERROR
        for turn_data, turn_metrics in zip(conv_data.turns, resolved_turn_metrics):
            for metric_identifier in turn_metrics:
                error_result = EvaluationResult(
                    conversation_group_id=conv_data.conversation_group_id,
                    turn_id=turn_data.turn_id,
                    metric_identifier=metric_identifier,
                    result="ERROR",
                    score=None,
                    threshold=None,
                    reason=error_reason,
                    query=turn_data.query,
                    response="",
                    execution_time=0.0,
                )
                error_results.append(error_result)

        # Mark all conversation-level metrics as ERROR
        for metric_identifier in resolved_conversation_metrics:
            error_result = EvaluationResult(
                conversation_group_id=conv_data.conversation_group_id,
                turn_id=None,  # Conversation-level metric
                metric_identifier=metric_identifier,
                result="ERROR",
                score=None,
                threshold=None,
                reason=error_reason,
                query="",
                response="",
                execution_time=0.0,
            )
            error_results.append(error_result)

        # Store results internally for summary tracking
        self.results.extend(error_results)
        return error_results

    def mark_turn_metrics_as_error(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        conv_data: EvaluationData,
        turn_idx: int,
        turn_data: TurnData,
        turn_metrics: list[str],
        error_reason: str,
    ) -> list[EvaluationResult]:
        """Mark all metrics for a single turn as ERROR.

        Args:
            conv_data: Conversation data
            turn_idx: Index of the turn
            turn_data: Turn data
            turn_metrics: Metrics for this turn
            error_reason: Reason for error

        Returns:
            list[EvaluationResult]: ERROR results for this turn's metrics
        """
        logger.warning(
            "Marking turn %d metrics as ERROR for conversation %s: %s",
            turn_idx,
            conv_data.conversation_group_id,
            error_reason,
        )
        error_results = []

        # Mark all turn-level metrics as ERROR
        for metric_identifier in turn_metrics:
            error_result = EvaluationResult(
                conversation_group_id=conv_data.conversation_group_id,
                turn_id=turn_data.turn_id,
                metric_identifier=metric_identifier,
                result="ERROR",
                score=None,
                threshold=None,
                reason=error_reason,
                query=turn_data.query,
                response="",
                execution_time=0.0,
            )
            error_results.append(error_result)

        # Store results internally for summary tracking
        self.results.extend(error_results)
        return error_results

    def get_error_summary(self) -> dict[str, int]:
        """Get summary of error results collected."""
        return {
            "total_errors": len(self.results),
            "turn_errors": len([r for r in self.results if r.turn_id is not None]),
            "conversation_errors": len([r for r in self.results if r.turn_id is None]),
        }

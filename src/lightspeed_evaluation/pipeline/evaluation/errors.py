"""Error handling module for evaluation errors."""

import logging
from typing import Optional

from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult, TurnData

logger = logging.getLogger(__name__)


class EvaluationErrorHandler:
    """Handles error scenarios during evaluation."""

    def __init__(self) -> None:
        """Initialize error handler."""
        self.results: list[EvaluationResult] = []

    def create_error_result(  # pylint: disable=too-many-arguments
        self,
        conv_id: str,
        metric_id: str,
        reason: str,
        *,
        turn_id: Optional[str] = None,
        query: str = "",
    ) -> EvaluationResult:
        """Create an ERROR EvaluationResult with common defaults.

        Args:
            conv_id: Conversation group ID
            metric_id: Metric identifier
            reason: Error reason
            turn_id: Turn ID (None for conversation-level)
            query: Query text
        """
        return EvaluationResult(
            conversation_group_id=conv_id,
            turn_id=turn_id,
            metric_identifier=metric_id,
            result="ERROR",
            reason=reason,
            query=query,
        )

    def mark_all_metrics_as_error(
        self,
        conv_data: EvaluationData,
        error_reason: str,
        resolved_turn_metrics: list[list[str]],
        resolved_conversation_metrics: list[str],
    ) -> list[EvaluationResult]:
        """Mark all turn and conversation metrics as ERROR.

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
        error_results: list[EvaluationResult] = []

        # Mark all turn-level metrics as ERROR
        for turn_data, turn_metrics in zip(conv_data.turns, resolved_turn_metrics):
            for metric_id in turn_metrics:
                error_results.append(
                    self.create_error_result(
                        conv_data.conversation_group_id,
                        metric_id,
                        error_reason,
                        turn_id=turn_data.turn_id,
                        query=turn_data.query,
                    )
                )

        # Mark all conversation-level metrics as ERROR
        for metric_id in resolved_conversation_metrics:
            error_results.append(
                self.create_error_result(
                    conv_data.conversation_group_id, metric_id, error_reason
                )
            )

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
            turn_idx: Index of the turn (for logging)
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
        error_results = [
            self.create_error_result(
                conv_data.conversation_group_id,
                metric_id,
                error_reason,
                turn_id=turn_data.turn_id,
                query=turn_data.query,
            )
            for metric_id in turn_metrics
        ]
        self.results.extend(error_results)
        return error_results

    def mark_cascade_failure(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        conv_data: EvaluationData,
        failed_turn_idx: int,
        resolved_turn_metrics: list[list[str]],
        resolved_conversation_metrics: list[str],
        error_reason: str,
    ) -> list[EvaluationResult]:
        """Mark remaining turns and conversation metrics as ERROR (cascade failure).

        Args:
            conv_data: Conversation data
            failed_turn_idx: Index of the turn that failed
            resolved_turn_metrics: Resolved metrics for all turns
            resolved_conversation_metrics: Resolved conversation metrics
            error_reason: Reason for error

        Returns:
            list[EvaluationResult]: ERROR results for remaining turns and conversation
        """
        logger.warning(
            "Marking remaining turns (%d onwards) and conversation metrics as ERROR for %s: %s",
            failed_turn_idx + 1,
            conv_data.conversation_group_id,
            error_reason,
        )
        error_results: list[EvaluationResult] = []

        # Mark remaining turns as ERROR (from failed_turn_idx + 1 onwards)
        for turn_idx in range(failed_turn_idx + 1, len(conv_data.turns)):
            turn_data = conv_data.turns[turn_idx]
            for metric_id in resolved_turn_metrics[turn_idx]:
                error_results.append(
                    self.create_error_result(
                        conv_data.conversation_group_id,
                        metric_id,
                        error_reason,
                        turn_id=turn_data.turn_id,
                        query=turn_data.query,
                    )
                )

        # Mark conversation-level metrics as ERROR
        for metric_id in resolved_conversation_metrics:
            error_results.append(
                self.create_error_result(
                    conv_data.conversation_group_id, metric_id, error_reason
                )
            )

        self.results.extend(error_results)
        return error_results

    def get_error_summary(self) -> dict[str, int]:
        """Get summary of error results collected."""
        return {
            "total_errors": len(self.results),
            "turn_errors": len([r for r in self.results if r.turn_id is not None]),
            "conversation_errors": len([r for r in self.results if r.turn_id is None]),
        }

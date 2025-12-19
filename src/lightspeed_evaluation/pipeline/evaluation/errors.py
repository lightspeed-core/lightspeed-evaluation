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

    def _create_result(  # pylint: disable=too-many-arguments
        self,
        conv_id: str,
        metric_id: str,
        reason: str,
        result_status: str,
        *,
        turn_id: Optional[str] = None,
        query: str = "",
    ) -> EvaluationResult:
        """Create an EvaluationResult with specified status.

        Args:
            conv_id: Conversation group ID
            metric_id: Metric identifier
            reason: Reason for the result
            result_status: Result status (ERROR, SKIPPED, etc.)
            turn_id: Turn ID (None for conversation-level)
            query: Query text
        """
        return EvaluationResult(
            conversation_group_id=conv_id,
            turn_id=turn_id,
            metric_identifier=metric_id,
            result=result_status,
            reason=reason,
            query=query,
        )

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
        return self._create_result(
            conv_id, metric_id, reason, "ERROR", turn_id=turn_id, query=query
        )

    def create_skipped_result(  # pylint: disable=too-many-arguments
        self,
        conv_id: str,
        metric_id: str,
        reason: str,
        *,
        turn_id: Optional[str] = None,
        query: str = "",
    ) -> EvaluationResult:
        """Create a SKIPPED EvaluationResult.

        Args:
            conv_id: Conversation group ID
            metric_id: Metric identifier
            reason: Skip reason
            turn_id: Turn ID (None for conversation-level)
            query: Query text
        """
        return self._create_result(
            conv_id, metric_id, reason, "SKIPPED", turn_id=turn_id, query=query
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

    def _mark_cascade(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        conv_data: EvaluationData,
        failed_turn_idx: int,
        resolved_turn_metrics: list[list[str]],
        resolved_conversation_metrics: list[str],
        reason: str,
        result_status: str,
    ) -> list[EvaluationResult]:
        """Internal helper for cascade marking with configurable status.

        Args:
            conv_data: Conversation data
            failed_turn_idx: Index of the turn that failed
            resolved_turn_metrics: Resolved metrics for all turns
            resolved_conversation_metrics: Resolved conversation metrics
            reason: Reason for the result
            result_status: Result status (ERROR or SKIPPED)

        Returns:
            list[EvaluationResult]: Results for remaining turns and conversation
        """
        results: list[EvaluationResult] = []

        # Mark remaining turns (from failed_turn_idx + 1 onwards)
        for turn_idx in range(failed_turn_idx + 1, len(conv_data.turns)):
            turn_data = conv_data.turns[turn_idx]
            for metric_id in resolved_turn_metrics[turn_idx]:
                results.append(
                    self._create_result(
                        conv_data.conversation_group_id,
                        metric_id,
                        reason,
                        result_status,
                        turn_id=turn_data.turn_id,
                        query=turn_data.query,
                    )
                )

        # Mark conversation-level metrics
        for metric_id in resolved_conversation_metrics:
            results.append(
                self._create_result(
                    conv_data.conversation_group_id, metric_id, reason, result_status
                )
            )

        self.results.extend(results)
        return results

    def mark_cascade_error(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        conv_data: EvaluationData,
        failed_turn_idx: int,
        resolved_turn_metrics: list[list[str]],
        resolved_conversation_metrics: list[str],
        error_reason: str,
    ) -> list[EvaluationResult]:
        """Mark remaining turns and conversation metrics as ERROR (cascade error).

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
        return self._mark_cascade(
            conv_data,
            failed_turn_idx,
            resolved_turn_metrics,
            resolved_conversation_metrics,
            error_reason,
            "ERROR",
        )

    def mark_cascade_skipped(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        conv_data: EvaluationData,
        failed_turn_idx: int,
        resolved_turn_metrics: list[list[str]],
        resolved_conversation_metrics: list[str],
        skip_reason: str,
    ) -> list[EvaluationResult]:
        """Mark remaining turns and conversation metrics as SKIPPED (skip on eval failure).

        Args:
            conv_data: Conversation data
            failed_turn_idx: Index of the turn that failed
            resolved_turn_metrics: Resolved metrics for all turns
            resolved_conversation_metrics: Resolved conversation metrics
            skip_reason: Reason for skipping

        Returns:
            list[EvaluationResult]: SKIPPED results for remaining turns and conversation
        """
        logger.info(
            "Skipping remaining turns (%d onwards) and conversation metrics for %s: %s",
            failed_turn_idx + 1,
            conv_data.conversation_group_id,
            skip_reason,
        )
        return self._mark_cascade(
            conv_data,
            failed_turn_idx,
            resolved_turn_metrics,
            resolved_conversation_metrics,
            skip_reason,
            "SKIPPED",
        )

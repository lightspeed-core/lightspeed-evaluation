"""Error handling module for evaluation errors."""

import logging
from typing import Dict, List

from ...core.models import EvaluationData, EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationErrorHandler:
    """Handles error scenarios during evaluation."""

    def __init__(self) -> None:
        """Initialize error handler."""
        self.results: List[EvaluationResult] = []

    def mark_all_metrics_as_error(
        self, conv_data: EvaluationData, error_reason: str
    ) -> List[EvaluationResult]:
        """Mark all turn and conversation metrics as ERROR when there is an error.

        Returns:
            List[EvaluationResult]: ERROR results for all metrics
        """
        logger.warning(
            "Marking all metrics as ERROR for conversation %s: %s",
            conv_data.conversation_group_id,
            error_reason,
        )
        error_results = []

        # Mark all turn-level metrics as ERROR
        if conv_data.turn_metrics:
            for turn_data in conv_data.turns:
                for metric_identifier in conv_data.turn_metrics:
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
        if conv_data.conversation_metrics:
            for metric_identifier in conv_data.conversation_metrics:
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

        return error_results

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error results collected."""
        return {
            "total_errors": len(self.results),
            "turn_errors": len([r for r in self.results if r.turn_id is not None]),
            "conversation_errors": len([r for r in self.results if r.turn_id is None]),
        }

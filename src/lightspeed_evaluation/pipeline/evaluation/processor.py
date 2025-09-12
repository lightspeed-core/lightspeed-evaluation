"""Conversation processing module - handles conversation and turn processing."""

import logging

from ...core.models import EvaluationData, EvaluationRequest, EvaluationResult, TurnData
from ...core.system import ConfigLoader
from .amender import APIDataAmender
from .errors import EvaluationErrorHandler
from .evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


class ConversationProcessor:
    """Processes individual conversations - handles both turn and conversation metrics."""

    def __init__(
        self,
        config_loader: ConfigLoader,
        metrics_evaluator: MetricsEvaluator,
        api_amender: APIDataAmender,
        error_handler: EvaluationErrorHandler,
    ):
        """Initialize with required components."""
        self.config_loader = config_loader
        self.config = config_loader.system_config
        self.metrics_evaluator = metrics_evaluator
        self.api_amender = api_amender
        self.error_handler = error_handler

    def process_conversation(self, conv_data: EvaluationData) -> list[EvaluationResult]:
        """Process single conversation - handle turn and conversation level metrics.

        Returns:
            list[EvaluationResult]: Results from processing this conversation
        """
        logger.info("Evaluating conversation: %s", conv_data.conversation_group_id)
        results: list[EvaluationResult] = []

        # Skip if no metrics specified at any level
        if not conv_data.turn_metrics and not conv_data.conversation_metrics:
            logger.debug("No metrics specified, skipping")
            return results

        # Amend with API data if enabled
        if self.config is None:
            raise ValueError("SystemConfig must be loaded")
        api_error_occurred = False
        if self.config.api.enabled:
            logger.debug("Amending data via API")
            api_error_occurred = self.api_amender.amend_conversation_data(conv_data)

        # If API error occurred, mark all metrics as ERROR and skip evaluation
        if api_error_occurred:
            logger.error("API error detected - marking all metrics as ERROR")
            error_results = self.error_handler.mark_all_metrics_as_error(
                conv_data, "API error during data amendment"
            )
            return error_results

        # Process turn-level metrics
        if conv_data.turn_metrics:
            logger.debug("Processing turn-level metrics: %s", conv_data.turn_metrics)
            for turn_idx, turn_data in enumerate(conv_data.turns):
                turn_results = self._evaluate_turn(conv_data, turn_idx, turn_data)
                results.extend(turn_results)

        # Process conversation-level metrics
        if conv_data.conversation_metrics:
            logger.debug(
                "Processing conversation-level metrics: %s",
                conv_data.conversation_metrics,
            )
            conv_results = self._evaluate_conversation(conv_data)
            results.extend(conv_results)

        return results

    def _evaluate_turn(
        self, conv_data: EvaluationData, turn_idx: int, turn_data: TurnData
    ) -> list[EvaluationResult]:
        """Evaluate single turn with specified turn metrics."""
        results = []
        for metric_identifier in conv_data.turn_metrics or []:
            request = EvaluationRequest.for_turn(
                conv_data, metric_identifier, turn_idx, turn_data
            )
            result = self.metrics_evaluator.evaluate_metric(request)
            if result:
                results.append(result)
        return results

    def _evaluate_conversation(
        self, conv_data: EvaluationData
    ) -> list[EvaluationResult]:
        """Evaluate conversation-level metrics."""
        results = []
        for metric_identifier in conv_data.conversation_metrics or []:
            request = EvaluationRequest.for_conversation(conv_data, metric_identifier)
            result = self.metrics_evaluator.evaluate_metric(request)
            if result:
                results.append(result)
        return results

    def get_metrics_summary(self, conv_data: EvaluationData) -> dict[str, int]:
        """Get summary of metrics to be evaluated for a conversation."""
        summary = {
            "turn_metrics": len(conv_data.turn_metrics or []),
            "conversation_metrics": len(conv_data.conversation_metrics or []),
            "total_turns": len(conv_data.turns),
        }
        return summary

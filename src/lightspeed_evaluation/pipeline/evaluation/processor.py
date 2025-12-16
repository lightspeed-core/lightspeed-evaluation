"""Conversation processing module - handles conversation and turn processing."""

import logging
from dataclasses import dataclass
from typing import Optional

from lightspeed_evaluation.core.metrics.manager import MetricLevel, MetricManager
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationResult,
    TurnData,
)
from lightspeed_evaluation.core.script import (
    ScriptExecutionError,
    ScriptExecutionManager,
)
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ProcessorComponents:
    """Components required for conversation processing."""

    metrics_evaluator: MetricsEvaluator
    api_amender: APIDataAmender
    error_handler: EvaluationErrorHandler
    metric_manager: MetricManager
    script_manager: ScriptExecutionManager


@dataclass
class TurnProcessingContext:
    """Context for processing turns within a conversation."""

    conv_data: EvaluationData
    resolved_turn_metrics: list[list[str]]
    resolved_conversation_metrics: list[str]
    conversation_id: Optional[str] = None


class ConversationProcessor:
    """Processes individual conversations - handles both turn and conversation metrics."""

    def __init__(self, config_loader: ConfigLoader, components: ProcessorComponents):
        """Initialize with config loader and components."""
        self.config_loader = config_loader
        self.config = config_loader.system_config
        self.components = components

    def process_conversation(self, conv_data: EvaluationData) -> list[EvaluationResult]:
        """Process single conversation - handle turn and conversation level metrics.

        Returns:
            list[EvaluationResult]: Results from processing this conversation
        """
        logger.info("Evaluating conversation: %s", conv_data.conversation_group_id)

        # Build processing context with resolved metrics
        ctx = self._build_processing_context(conv_data)

        # Skip if no metrics specified at any level
        if not self._has_metrics_to_evaluate(ctx):
            logger.debug(
                "No metrics to evaluate (no defaults or explicit metrics), skipping"
            )
            return []

        # Run setup script if provided
        setup_error = self._run_setup_script(conv_data)
        if setup_error:
            return self._handle_setup_failure(ctx, setup_error)

        try:
            if self.config is None:
                raise ValueError("SystemConfig must be loaded")

            return self._process_turns_and_conversation(ctx)

        finally:
            # Always run cleanup script (if provided) regardless of results
            self._run_cleanup_script(conv_data)

    def _build_processing_context(
        self, conv_data: EvaluationData
    ) -> TurnProcessingContext:
        """Build processing context with resolved metrics."""
        resolved_turn_metrics = [
            self.components.metric_manager.resolve_metrics(
                turn_data.turn_metrics, MetricLevel.TURN
            )
            for turn_data in conv_data.turns
        ]
        resolved_conversation_metrics = self.components.metric_manager.resolve_metrics(
            conv_data.conversation_metrics, MetricLevel.CONVERSATION
        )
        return TurnProcessingContext(
            conv_data=conv_data,
            resolved_turn_metrics=resolved_turn_metrics,
            resolved_conversation_metrics=resolved_conversation_metrics,
        )

    def _has_metrics_to_evaluate(self, ctx: TurnProcessingContext) -> bool:
        """Check if there are any metrics to evaluate."""
        has_turn_metrics = any(bool(metrics) for metrics in ctx.resolved_turn_metrics)
        has_conversation_metrics = bool(ctx.resolved_conversation_metrics)
        return has_turn_metrics or has_conversation_metrics

    def _handle_setup_failure(
        self, ctx: TurnProcessingContext, setup_error: str
    ) -> list[EvaluationResult]:
        """Handle setup script failure - mark all metrics as ERROR."""
        logger.error("Setup script failed - marking all metrics as ERROR")
        error_results = self.components.error_handler.mark_all_metrics_as_error(
            ctx.conv_data,
            f"Setup script failed: {setup_error}",
            resolved_turn_metrics=ctx.resolved_turn_metrics,
            resolved_conversation_metrics=ctx.resolved_conversation_metrics,
        )
        # Attempt cleanup even when setup failed
        self._run_cleanup_script(ctx.conv_data)
        return error_results

    def _process_turns_and_conversation(
        self, ctx: TurnProcessingContext
    ) -> list[EvaluationResult]:
        """Process all turns and conversation-level metrics."""
        results: list[EvaluationResult] = []

        # Process each turn individually (API call + evaluation)
        for turn_idx, (turn_data, turn_metrics) in enumerate(
            zip(ctx.conv_data.turns, ctx.resolved_turn_metrics)
        ):
            # Handle API call if enabled
            if self.config and self.config.api.enabled:
                api_error = self._process_turn_api(ctx, turn_idx, turn_data)
                if api_error:
                    # API failure - mark current turn and cascade to remaining
                    api_error_results = self._handle_api_error(ctx, turn_idx, api_error)
                    results.extend(api_error_results)
                    return results

            # Evaluate turn-level metrics
            if turn_metrics:
                logger.debug("Processing turn %d metrics: %s", turn_idx, turn_metrics)
                turn_results = self._evaluate_turn(
                    ctx.conv_data, turn_idx, turn_data, turn_metrics
                )
                results.extend(turn_results)

        # Process conversation-level metrics
        if ctx.resolved_conversation_metrics:
            logger.debug(
                "Processing conversation-level metrics: %s",
                ctx.resolved_conversation_metrics,
            )
            conv_results = self._evaluate_conversation(
                ctx.conv_data, ctx.resolved_conversation_metrics
            )
            results.extend(conv_results)

        return results

    def _process_turn_api(
        self, ctx: TurnProcessingContext, turn_idx: int, turn_data: TurnData
    ) -> Optional[str]:
        """Process API call for a single turn. Returns error message if failed."""
        logger.debug("Processing turn %d: %s", turn_idx, turn_data.turn_id)
        api_error_message, ctx.conversation_id = (
            self.components.api_amender.amend_single_turn(
                turn_data, ctx.conversation_id
            )
        )
        logger.debug(
            "✅ API Call completed for turn %d: %s", turn_idx, turn_data.turn_id
        )
        return api_error_message

    def _handle_api_error(
        self,
        ctx: TurnProcessingContext,
        turn_idx: int,
        api_error_message: str,
    ) -> list[EvaluationResult]:
        """Handle API error - mark current turn and cascade to remaining."""
        logger.error(
            "API error for turn %d - marking current turn, "
            "remaining turns, and conversation as ERROR",
            turn_idx,
        )
        results: list[EvaluationResult] = []

        # Derive turn data and metrics from context
        turn_data = ctx.conv_data.turns[turn_idx]
        turn_metrics = ctx.resolved_turn_metrics[turn_idx]

        # Mark current turn as ERROR
        current_turn_errors = self.components.error_handler.mark_turn_metrics_as_error(
            ctx.conv_data, turn_idx, turn_data, turn_metrics, api_error_message
        )
        results.extend(current_turn_errors)

        # Mark remaining turns and conversation metrics as ERROR
        cascade_error_reason = (
            f"Cascade failure from turn {turn_idx + 1} API error: {api_error_message}"
        )
        remaining_errors = self.components.error_handler.mark_cascade_failure(
            ctx.conv_data,
            turn_idx,
            ctx.resolved_turn_metrics,
            ctx.resolved_conversation_metrics,
            cascade_error_reason,
        )
        results.extend(remaining_errors)

        return results

    def _evaluate_turn(
        self,
        conv_data: EvaluationData,
        turn_idx: int,
        turn_data: TurnData,
        turn_metrics: list[str],
    ) -> list[EvaluationResult]:
        """Evaluate single turn with specified turn metrics."""
        results = []

        for metric_identifier in turn_metrics:
            if turn_data.is_metric_invalid(metric_identifier):
                error_reason = f"Invalid turn metric '{metric_identifier}', check Validation Errors"
                logger.error(error_reason)
                results.append(
                    self.components.error_handler.create_error_result(
                        conv_data.conversation_group_id,
                        metric_identifier,
                        error_reason,
                        turn_id=turn_data.turn_id,
                        query=turn_data.query,
                    )
                )
                continue

            request = EvaluationRequest.for_turn(
                conv_data, metric_identifier, turn_idx, turn_data
            )
            result = self.components.metrics_evaluator.evaluate_metric(request)
            if result:
                results.append(result)
        return results

    def _evaluate_conversation(
        self, conv_data: EvaluationData, conversation_metrics: list[str]
    ) -> list[EvaluationResult]:
        """Evaluate conversation-level metrics."""
        results = []

        for metric_identifier in conversation_metrics:
            if conv_data.is_metric_invalid(metric_identifier):
                error_reason = (
                    f"Invalid metric '{metric_identifier}', check Validation Errors"
                )
                logger.error(error_reason)
                results.append(
                    self.components.error_handler.create_error_result(
                        conv_data.conversation_group_id, metric_identifier, error_reason
                    )
                )
                continue

            request = EvaluationRequest.for_conversation(conv_data, metric_identifier)
            result = self.components.metrics_evaluator.evaluate_metric(request)
            if result:
                results.append(result)
        return results

    def _run_setup_script(self, conv_data: EvaluationData) -> Optional[str]:
        """Run setup script for conversation."""
        setup_script = conv_data.setup_script
        if not setup_script:
            return None

        # Skip script execution if API is disabled
        if self.config and not self.config.api.enabled:
            logger.debug("Skipping setup script (API disabled): %s", setup_script)
            return None

        try:
            logger.debug("Running setup script: %s", setup_script)
            success = self.components.script_manager.run_script(setup_script)
            if not success:
                error_msg = f"Setup script returned non-zero exit code: {setup_script}"
                logger.error(error_msg)
                return error_msg
            logger.debug("Setup script completed successfully")
            return None
        except ScriptExecutionError as e:
            logger.error("Setup script failed: %s", e)
            return str(e)

    def _run_cleanup_script(self, conv_data: EvaluationData) -> None:
        """Run cleanup script for conversation.

        Cleanup failures are logged as warnings but don't affect evaluation results.
        """
        cleanup_script = conv_data.cleanup_script
        if not cleanup_script:
            return

        # Skip script execution if API is disabled
        if self.config and not self.config.api.enabled:
            logger.debug("Skipping cleanup script (API disabled): %s", cleanup_script)
            return

        logger.debug("Running cleanup script: %s", cleanup_script)
        try:
            success = self.components.script_manager.run_script(cleanup_script)
            if success:
                logger.debug(
                    "Cleanup script completed successfully: %s", cleanup_script
                )
            else:
                logger.warning(
                    "Cleanup script failed (non-critical): %s", cleanup_script
                )
        except ScriptExecutionError as e:
            logger.warning("Cleanup script failed: %s", e)

    def get_metrics_summary(self, conv_data: EvaluationData) -> dict[str, int]:
        """Get summary of metrics to be evaluated for a conversation."""
        return self.components.metric_manager.count_metrics_for_conversation(conv_data)

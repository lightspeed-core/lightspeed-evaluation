"""Metrics mapping for evaluation."""

from enum import Enum
from typing import Any, Optional

from lightspeed_evaluation.core.models.data import EvaluationData, TurnData
from lightspeed_evaluation.core.models.system import SystemConfig


class MetricLevel(Enum):
    """Metric level enumeration."""

    TURN = "turn"
    CONVERSATION = "conversation"


class MetricManager:
    """Manager for both turn and conversation metrics."""

    def __init__(self, system_config: SystemConfig):
        """Initialize with system configuration."""
        self.system_config = system_config

    def resolve_metrics(
        self, metrics: Optional[list[str]], level: MetricLevel
    ) -> list[str]:
        """Resolve metrics mapping.

        Options:
        - None: use system defaults (metrics with default=true)
        - []: skip evaluation completely
        - [metrics...]: use specified metrics from turn data

        Args:
            metrics: The metrics configuration (None, [], or list of metrics)
            level: Whether this is TURN or CONVERSATION level

        Returns:
            List of metrics to evaluate
        """
        if metrics is None:
            # None = use system defaults
            return self._extract_default_metrics(level)
        if metrics == []:
            # [] = explicitly skip evaluation
            return []
        # Use specified metrics as-is
        return metrics

    def get_metric_metadata(
        self,
        metric_identifier: str,
        level: MetricLevel,
        conv_data: Optional[EvaluationData] = None,
        turn_data: Optional[TurnData] = None,
    ) -> Optional[dict[str, Any]]:
        """Get full metric metadata with priority hierarchy.

        This method returns the complete metadata dictionary for a metric,
        including all fields (threshold, criteria, evaluation_steps, etc.).

        Priority:
        1. Level-specific metadata (turn-specific for turns, conversation-specific for convs)
        2. System defaults

        Args:
            metric_identifier: The metric to get metadata for
            level: Whether this is TURN or CONVERSATION level
            conv_data: Conversation data for conversation-level metadata
            turn_data: Turn data for turn-specific metadata

        Returns:
            Full metadata dictionary or None if not found
        """
        # Check level-specific metadata first
        level_metadata = self._get_level_metadata(level, conv_data, turn_data)
        if metric_identifier in level_metadata:
            return level_metadata[metric_identifier]

        # Fall back to system defaults
        system_metadata = self._get_system_metadata(level)
        return system_metadata.get(metric_identifier)

    def get_effective_threshold(
        self,
        metric_identifier: str,
        level: MetricLevel,
        conv_data: Optional[EvaluationData] = None,
        turn_data: Optional[TurnData] = None,
    ) -> Optional[float]:
        """Get effective threshold with priority hierarchy.

        Priority:
        1. Level-specific metadata (turn-specific for turns, conversation-specific for convs)
        2. System defaults

        Args:
            metric_identifier: The metric to get threshold for
            level: Whether this is TURN or CONVERSATION level
            conv_data: Conversation data for conversation-level metadata
            turn_data: Turn data for turn-specific metadata

        Returns:
            Effective threshold or None if not found
        """
        # Use the unified metadata getter
        metadata = self.get_metric_metadata(
            metric_identifier, level, conv_data, turn_data
        )
        return metadata.get("threshold") if metadata else None

    def _get_level_metadata(
        self,
        level: MetricLevel,
        conv_data: Optional[EvaluationData],
        turn_data: Optional[TurnData],
    ) -> dict[str, dict[str, Any]]:
        """Get level-specific metadata (turn or conversation level)."""
        if level == MetricLevel.TURN and turn_data and turn_data.turn_metrics_metadata:
            return turn_data.turn_metrics_metadata
        if (
            level == MetricLevel.CONVERSATION
            and conv_data
            and conv_data.conversation_metrics_metadata
        ):
            return conv_data.conversation_metrics_metadata
        return {}

    def _get_system_metadata(self, level: MetricLevel) -> dict[str, dict[str, Any]]:
        """Get system-level metadata for the given level."""
        if level == MetricLevel.TURN:
            return self.system_config.default_turn_metrics_metadata
        return self.system_config.default_conversation_metrics_metadata

    def _extract_default_metrics(self, level: MetricLevel) -> list[str]:
        """Extract metrics that have default=true from metadata."""
        metrics_metadata = self._get_system_metadata(level)

        default_metrics = []
        for metric_name, metadata in metrics_metadata.items():
            if metadata.get("default", False):  # default=false if not specified
                default_metrics.append(metric_name)
        return default_metrics

    def count_metrics_for_conversation(
        self, conv_data: EvaluationData
    ) -> dict[str, int]:
        """Count total metrics that would be evaluated for a conversation."""
        # Count turn metrics
        total_turn_metrics = 0
        for turn_data in conv_data.turns:
            turn_metrics = self.resolve_metrics(
                turn_data.turn_metrics, MetricLevel.TURN
            )
            total_turn_metrics += len(turn_metrics)

        # Count conversation metrics
        conversation_metrics = self.resolve_metrics(
            conv_data.conversation_metrics, MetricLevel.CONVERSATION
        )

        return {
            "turn_metrics": total_turn_metrics,
            "conversation_metrics": len(conversation_metrics),
            "total_turns": len(conv_data.turns),
        }

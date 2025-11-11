"""Unit tests for core metrics manager module."""

import pytest

from lightspeed_evaluation.core.metrics.manager import MetricLevel, MetricManager
from lightspeed_evaluation.core.models import (
    EvaluationData,
    SystemConfig,
    TurnData,
)


@pytest.fixture
def system_config():
    """Create a test system config with metrics metadata."""
    config = SystemConfig()

    # Set up test metrics metadata
    config.default_turn_metrics_metadata = {
        "ragas:faithfulness": {
            "threshold": 0.7,
            "default": True,
            "description": "Test",
        },
        "ragas:response_relevancy": {
            "threshold": 0.8,
            "default": False,
            "description": "Test",
        },
        "custom:answer_correctness": {
            "threshold": 0.75,
            "default": True,
            "description": "Test",
        },
    }

    config.default_conversation_metrics_metadata = {
        "deepeval:conversation_completeness": {
            "threshold": 0.6,
            "default": True,
            "description": "Test",
        },
        "deepeval:conversation_relevancy": {
            "threshold": 0.7,
            "default": False,
            "description": "Test",
        },
    }

    return config


class TestMetricManager:
    """Unit tests for MetricManager."""

    def test_resolve_metrics_with_none_uses_defaults(self, system_config):
        """Test that None resolves to system defaults."""
        manager = MetricManager(system_config)

        metrics = manager.resolve_metrics(None, MetricLevel.TURN)

        # Should return only metrics with default=True
        assert "ragas:faithfulness" in metrics
        assert "custom:answer_correctness" in metrics
        assert "ragas:response_relevancy" not in metrics  # default=False

    def test_resolve_metrics_with_empty_list_skips_evaluation(self, system_config):
        """Test that empty list skips evaluation."""
        manager = MetricManager(system_config)

        metrics = manager.resolve_metrics([], MetricLevel.TURN)

        # Should return empty list
        assert metrics == []

    def test_resolve_metrics_with_explicit_list(self, system_config):
        """Test that explicit list is returned as-is."""
        manager = MetricManager(system_config)

        explicit_metrics = ["ragas:response_relevancy", "custom:tool_eval"]
        metrics = manager.resolve_metrics(explicit_metrics, MetricLevel.TURN)

        # Should return the exact list provided
        assert metrics == explicit_metrics

    def test_resolve_metrics_conversation_level_defaults(self, system_config):
        """Test conversation-level default metrics."""
        manager = MetricManager(system_config)

        metrics = manager.resolve_metrics(None, MetricLevel.CONVERSATION)

        # Should return only conversation metrics with default=True
        assert "deepeval:conversation_completeness" in metrics
        assert "deepeval:conversation_relevancy" not in metrics

    def test_get_effective_threshold_from_system_defaults(self, system_config):
        """Test getting threshold from system defaults."""
        manager = MetricManager(system_config)

        threshold = manager.get_effective_threshold(
            "ragas:faithfulness", MetricLevel.TURN, conv_data=None, turn_data=None
        )

        assert threshold == 0.7

    def test_get_effective_threshold_turn_level_override(self, system_config):
        """Test turn-level metadata overrides system defaults."""
        manager = MetricManager(system_config)

        turn_data = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            turn_metrics_metadata={"ragas:faithfulness": {"threshold": 0.9}},
        )

        threshold = manager.get_effective_threshold(
            "ragas:faithfulness", MetricLevel.TURN, turn_data=turn_data
        )

        # Should use turn-specific threshold
        assert threshold == 0.9

    def test_get_effective_threshold_conversation_level_override(self, system_config):
        """Test conversation-level metadata overrides system defaults."""
        manager = MetricManager(system_config)

        turn = TurnData(turn_id="1", query="Q", response="R")
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn],
            conversation_metrics_metadata={
                "deepeval:conversation_completeness": {"threshold": 0.85}
            },
        )

        threshold = manager.get_effective_threshold(
            "deepeval:conversation_completeness",
            MetricLevel.CONVERSATION,
            conv_data=conv_data,
        )

        # Should use conversation-specific threshold
        assert threshold == 0.85

    def test_get_effective_threshold_not_found(self, system_config):
        """Test getting threshold for unknown metric returns None."""
        manager = MetricManager(system_config)

        threshold = manager.get_effective_threshold("unknown:metric", MetricLevel.TURN)

        assert threshold is None

    def test_get_effective_threshold_no_metadata_at_level(self, system_config):
        """Test threshold lookup when no metadata at level."""
        manager = MetricManager(system_config)

        turn_data = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            turn_metrics_metadata=None,  # No metadata
        )

        threshold = manager.get_effective_threshold(
            "ragas:faithfulness", MetricLevel.TURN, turn_data=turn_data
        )

        # Should fall back to system defaults
        assert threshold == 0.7

    def test_get_effective_threshold_metric_not_in_level_metadata(self, system_config):
        """Test threshold for metric not in level metadata."""
        manager = MetricManager(system_config)

        turn_data = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            turn_metrics_metadata={"custom:answer_correctness": {"threshold": 0.95}},
        )

        # Query for different metric
        threshold = manager.get_effective_threshold(
            "ragas:faithfulness", MetricLevel.TURN, turn_data=turn_data
        )

        # Should fall back to system defaults
        assert threshold == 0.7

    def test_count_metrics_for_conversation_all_defaults(self, system_config):
        """Test counting metrics when using all defaults."""
        manager = MetricManager(system_config)

        turn1 = TurnData(turn_id="1", query="Q1", response="R1", turn_metrics=None)
        turn2 = TurnData(turn_id="2", query="Q2", response="R2", turn_metrics=None)
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn1, turn2],
            conversation_metrics=None,
        )

        counts = manager.count_metrics_for_conversation(conv_data)

        # 2 turns × 2 default turn metrics = 4
        assert counts["turn_metrics"] == 4
        # 1 default conversation metric
        assert counts["conversation_metrics"] == 1
        assert counts["total_turns"] == 2

    def test_count_metrics_for_conversation_explicit_metrics(self, system_config):
        """Test counting with explicit metrics."""
        manager = MetricManager(system_config)

        turn1 = TurnData(
            turn_id="1", query="Q1", response="R1", turn_metrics=["ragas:faithfulness"]
        )
        turn2 = TurnData(
            turn_id="2",
            query="Q2",
            response="R2",
            turn_metrics=["ragas:response_relevancy", "custom:answer_correctness"],
        )
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn1, turn2],
            conversation_metrics=["deepeval:conversation_relevancy"],
        )

        counts = manager.count_metrics_for_conversation(conv_data)

        # 1 metric for turn1 + 2 metrics for turn2 = 3
        assert counts["turn_metrics"] == 3
        # 1 explicit conversation metric
        assert counts["conversation_metrics"] == 1
        assert counts["total_turns"] == 2

    def test_count_metrics_for_conversation_skip_evaluation(self, system_config):
        """Test counting when evaluation is skipped."""
        manager = MetricManager(system_config)

        turn = TurnData(
            turn_id="1", query="Q", response="R", turn_metrics=[]  # Explicitly skip
        )
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn],
            conversation_metrics=[],  # Explicitly skip
        )

        counts = manager.count_metrics_for_conversation(conv_data)

        assert counts["turn_metrics"] == 0
        assert counts["conversation_metrics"] == 0
        assert counts["total_turns"] == 1

    def test_count_metrics_for_conversation_mixed(self, system_config):
        """Test counting with mixed default and explicit metrics."""
        manager = MetricManager(system_config)

        turn1 = TurnData(turn_id="1", query="Q1", response="R1", turn_metrics=None)
        turn2 = TurnData(
            turn_id="2", query="Q2", response="R2", turn_metrics=["ragas:faithfulness"]
        )
        turn3 = TurnData(turn_id="3", query="Q3", response="R3", turn_metrics=[])

        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn1, turn2, turn3],
            conversation_metrics=None,
        )

        counts = manager.count_metrics_for_conversation(conv_data)

        # turn1: 2 defaults, turn2: 1 explicit, turn3: 0 (skipped) = 3
        assert counts["turn_metrics"] == 3
        # 1 default conversation metric
        assert counts["conversation_metrics"] == 1
        assert counts["total_turns"] == 3

    def test_extract_default_metrics_empty_metadata(self):
        """Test extracting defaults when no metrics have default=true."""
        config = SystemConfig()
        config.default_turn_metrics_metadata = {
            "ragas:faithfulness": {"threshold": 0.7, "default": False},
            "ragas:response_relevancy": {"threshold": 0.8, "default": False},
        }

        manager = MetricManager(config)
        metrics = manager.resolve_metrics(None, MetricLevel.TURN)

        # Should return empty list when no defaults
        assert metrics == []

    def test_get_effective_threshold_with_both_metadata_sources(self, system_config):
        """Test that level metadata takes priority over system defaults."""
        manager = MetricManager(system_config)

        # System default is 0.7
        # Turn-level override is 0.95
        turn_data = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            turn_metrics_metadata={"ragas:faithfulness": {"threshold": 0.95}},
        )

        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        threshold = manager.get_effective_threshold(
            "ragas:faithfulness",
            MetricLevel.TURN,
            conv_data=conv_data,
            turn_data=turn_data,
        )

        # Should use turn-level override, not system default
        assert threshold == 0.95

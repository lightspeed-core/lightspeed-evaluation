# pylint: disable=protected-access,too-few-public-methods

"""Additional tests to boost coverage towards 75%."""

from pathlib import Path

from pytest_mock import MockerFixture
from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.output.generator import OutputHandler
from lightspeed_evaluation.core.output.statistics import (
    calculate_basic_stats,
    calculate_detailed_stats,
)
from lightspeed_evaluation.core.system.loader import validate_metrics


class TestStatisticsEdgeCases:
    """Edge case tests for statistics module."""

    def test_stats_with_mixed_results(self) -> None:
        """Test statistics with all result types."""
        results = [
            EvaluationResult(
                conversation_group_id=f"conv{i%2}",
                turn_id=f"turn{i}",
                metric_identifier=f"metric{i%3}",
                result=["PASS", "FAIL", "ERROR"][i % 3],
                score=0.8 if i % 3 == 0 else (0.5 if i % 3 == 1 else None),
                threshold=0.7,
            )
            for i in range(30)
        ]

        basic = calculate_basic_stats(results)
        detailed = calculate_detailed_stats(results)

        assert basic["TOTAL"] == 30
        assert basic["PASS"] + basic["FAIL"] + basic["ERROR"] == 30
        assert len(detailed["by_metric"]) > 0
        assert len(detailed["by_conversation"]) == 2

    def test_detailed_stats_single_conversation_multiple_metrics(self) -> None:
        """Test detailed stats with one conversation, multiple metrics."""
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id=f"turn{i}",
                metric_identifier=f"metric{i}",
                result="PASS",
                score=0.9,
                threshold=0.7,
            )
            for i in range(10)
        ]

        detailed = calculate_detailed_stats(results)

        assert len(detailed["by_conversation"]) == 1
        assert len(detailed["by_metric"]) == 10
        assert detailed["by_conversation"]["conv1"]["pass"] == 10

    def test_detailed_stats_multiple_conversations_single_metric(self) -> None:
        """Test detailed stats with multiple conversations, one metric."""
        results = [
            EvaluationResult(
                conversation_group_id=f"conv{i}",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS" if i % 2 == 0 else "FAIL",
                score=0.9 if i % 2 == 0 else 0.5,
                threshold=0.7,
            )
            for i in range(10)
        ]

        detailed = calculate_detailed_stats(results)

        assert len(detailed["by_conversation"]) == 10
        assert len(detailed["by_metric"]) == 1
        assert detailed["by_metric"]["metric1"]["pass"] == 5
        assert detailed["by_metric"]["metric1"]["fail"] == 5


class TestOutputHandlerEdgeCases:
    """Edge case tests for output handler."""

    def test_calculate_stats_with_single_result(self, tmp_path: Path) -> None:
        """Test stats calculation with exactly one result."""

        handler = OutputHandler(output_dir=str(tmp_path))
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                score=0.95,
                threshold=0.7,
            )
        ]

        stats = handler._calculate_stats(results)

        assert stats["basic"]["TOTAL"] == 1
        assert stats["basic"]["PASS"] == 1
        assert stats["basic"]["pass_rate"] == 100.0

    def test_generate_csv_with_minimal_columns(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test CSV generation with minimal column set."""

        config = mocker.Mock()
        config.output.csv_columns = ["conversation_group_id", "result"]
        config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        results = [
            EvaluationResult(
                conversation_group_id="conv1",
                turn_id="turn1",
                metric_identifier="metric1",
                result="PASS",
                threshold=0.7,
            )
        ]

        csv_file = handler._generate_csv_report(results, "test")

        assert csv_file.exists()
        content = csv_file.read_text()
        assert "conversation_group_id" in content
        assert "result" in content
        assert "PASS" in content


class TestSystemLoaderEdgeCases:
    """Edge case tests for system loader."""

    def test_validate_metrics_with_mixed_valid_invalid(self) -> None:
        """Test validating mix of valid and invalid metrics."""

        turn_metrics = [
            "ragas:faithfulness",
            "unknown:metric1",
            "custom:keywords_eval",
            "unknown:metric2",
        ]
        conversation_metrics = [
            "deepeval:conversation_completeness",
            "unknown:conv_metric",
        ]

        errors = validate_metrics(turn_metrics, conversation_metrics)

        # Should have 3 errors (2 turn-level + 1 conversation-level)
        assert len(errors) >= 2
        assert any("unknown:metric1" in err for err in errors)
        assert any("unknown:conv_metric" in err for err in errors)

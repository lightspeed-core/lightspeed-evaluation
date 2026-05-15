# pylint: disable=protected-access,too-few-public-methods

"""Additional tests to boost coverage towards 75%."""

from pathlib import Path

from pytest_mock import MockerFixture
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.models.summary import EvaluationSummary
from lightspeed_evaluation.core.output.generator import OutputHandler
from lightspeed_evaluation.core.output.statistics import (
    compute_overall_stats,
    compute_detailed_stats,
)
from lightspeed_evaluation.core.system.validator import DataValidator
from lightspeed_evaluation.core.storage import FileBackendConfig


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

        basic = compute_overall_stats(results)
        detailed = compute_detailed_stats(results).model_dump()

        assert basic.total == 30
        assert basic.passed + basic.failed + basic.error == 30
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

        detailed = compute_detailed_stats(results).model_dump()

        assert len(detailed["by_conversation"]) == 1
        assert len(detailed["by_metric"]) == 10
        assert detailed["by_conversation"]["conv1"]["passed"] == 10

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

        detailed = compute_detailed_stats(results).model_dump()

        assert len(detailed["by_conversation"]) == 10
        assert len(detailed["by_metric"]) == 1
        assert detailed["by_metric"]["metric1"]["passed"] == 5
        assert detailed["by_metric"]["metric1"]["failed"] == 5


class TestOutputHandlerEdgeCases:
    """Edge case tests for output handler."""

    def test_summary_with_single_result(self) -> None:
        """Test summary creation with exactly one result."""
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

        summary = EvaluationSummary.from_results(results)

        assert summary.overall.total == 1
        assert summary.overall.passed == 1
        assert summary.overall.pass_rate == 100.0

    def test_generate_csv_with_minimal_columns(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test CSV generation with minimal column set."""
        file_config = FileBackendConfig(csv_columns=["conversation_group_id", "result"])
        config = mocker.Mock()
        config.storage = [file_config]
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
        """Test validating mix of valid and invalid metrics via DataValidator."""
        config = SystemConfig(
            default_turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.7},
                "custom:keywords_eval": {"threshold": 0.5},
            },
            default_conversation_metrics_metadata={
                "deepeval:conversation_completeness": {"threshold": 0.6},
            },
        )
        validator = DataValidator(api_enabled=False, system_config=config)

        turn = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            contexts=["C"],
            expected_keywords=[["kw"]],
            turn_metrics=[
                "ragas:faithfulness",
                "unknown:metric1",
                "custom:keywords_eval",
                "unknown:metric2",
            ],
        )
        conv = EvaluationData(
            conversation_group_id="test",
            turns=[turn],
            conversation_metrics=[
                "deepeval:conversation_completeness",
                "unknown:conv_metric",
            ],
        )

        validator._validate_evaluation_data([conv])

        # Should have 3 errors (2 unknown turn-level + 1 unknown conversation-level)
        assert len(validator.validation_errors) == 3
        assert any("unknown:metric1" in err for err in validator.validation_errors)
        assert any("unknown:metric2" in err for err in validator.validation_errors)
        assert any("unknown:conv_metric" in err for err in validator.validation_errors)

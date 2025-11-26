"""Unit tests for output generator."""

import json

import pytest

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.output.generator import OutputHandler


@pytest.fixture
def sample_results():
    """Create sample evaluation results."""
    return [
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
            query="What is Python?",
            response="Python is a programming language",
        ),
        EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn2",
            metric_identifier="ragas:answer_relevancy",
            score=0.60,
            result="FAIL",
            threshold=0.7,
            reason="Low score",
            query="How?",
            response="It works",
        ),
    ]


@pytest.fixture
def mock_system_config(mocker):
    """Create mock system config."""
    config = mocker.Mock()
    config.output.enabled_outputs = ["csv", "json", "txt"]
    config.output.csv_columns = [
        "conversation_group_id",
        "turn_id",
        "metric_identifier",
        "result",
        "score",
    ]
    config.visualization.enabled_graphs = []
    return config


class TestOutputHandler:
    """Tests for OutputHandler."""

    def test_initialization(self, tmp_path):
        """Test handler initialization."""
        handler = OutputHandler(output_dir=str(tmp_path), base_filename="test")

        assert handler.output_dir == tmp_path
        assert handler.base_filename == "test"
        assert tmp_path.exists()

    def test_calculate_stats_with_results(self, tmp_path, sample_results):
        """Test statistics calculation."""
        handler = OutputHandler(output_dir=str(tmp_path))
        stats = handler._calculate_stats(sample_results)

        assert stats["basic"]["TOTAL"] == 2
        assert stats["basic"]["PASS"] == 1
        assert stats["basic"]["FAIL"] == 1
        assert "detailed" in stats

    def test_calculate_stats_empty(self, tmp_path):
        """Test statistics with empty results."""
        handler = OutputHandler(output_dir=str(tmp_path))
        stats = handler._calculate_stats([])

        assert stats["basic"]["TOTAL"] == 0
        assert stats["detailed"]["by_metric"] == {}

    def test_generate_csv_report(self, tmp_path, sample_results, mock_system_config):
        """Test CSV generation."""
        handler = OutputHandler(
            output_dir=str(tmp_path),
            system_config=mock_system_config,
        )

        csv_file = handler._generate_csv_report(sample_results, "test")

        assert csv_file.exists()
        assert csv_file.suffix == ".csv"

        # Verify content
        content = csv_file.read_text()
        assert "conversation_group_id" in content
        assert "conv1" in content

    def test_generate_json_summary(self, tmp_path, sample_results):
        """Test JSON summary generation."""
        handler = OutputHandler(output_dir=str(tmp_path))
        stats = handler._calculate_stats(sample_results)

        json_file = handler._generate_json_summary(
            sample_results, "test", stats["basic"], stats["detailed"]
        )

        assert json_file.exists()

        # Verify structure
        with open(json_file) as f:
            data = json.load(f)

        assert "summary_stats" in data or "results" in data
        assert data.get("total_evaluations") == 2 or len(data.get("results", [])) == 2

    def test_generate_text_summary(self, tmp_path, sample_results):
        """Test text summary generation."""
        handler = OutputHandler(output_dir=str(tmp_path))
        stats = handler._calculate_stats(sample_results)

        txt_file = handler._generate_text_summary(
            sample_results, "test", stats["basic"], stats["detailed"]
        )

        assert txt_file.exists()

        content = txt_file.read_text()
        assert "TOTAL" in content or "Summary" in content

    def test_get_output_directory(self, tmp_path):
        """Test get output directory."""
        handler = OutputHandler(output_dir=str(tmp_path))

        assert handler.get_output_directory() == tmp_path

    def test_generate_reports_creates_files(
        self, tmp_path, sample_results, mock_system_config, mocker
    ):
        """Test that generate_reports creates output files."""
        mock_now = mocker.Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_now.isoformat.return_value = "2025-01-01T12:00:00"

        mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        ).now.return_value = mock_now

        handler = OutputHandler(
            output_dir=str(tmp_path),
            base_filename="eval",
            system_config=mock_system_config,
        )

        handler.generate_reports(sample_results)

        # Check files exist
        assert (tmp_path / "eval_20250101_120000_detailed.csv").exists()
        assert (tmp_path / "eval_20250101_120000_summary.json").exists()
        assert (tmp_path / "eval_20250101_120000_summary.txt").exists()

    def test_generate_reports_with_empty_results(
        self, tmp_path, mock_system_config, mocker
    ):
        """Test generating reports with no results."""
        mock_now = mocker.Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_now.isoformat.return_value = "2025-01-01T12:00:00"

        mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        ).now.return_value = mock_now

        handler = OutputHandler(
            output_dir=str(tmp_path),
            system_config=mock_system_config,
        )

        # Should not crash
        handler.generate_reports([])

    def test_generate_individual_reports_csv_only(
        self, tmp_path, sample_results, mocker
    ):
        """Test generating only CSV."""
        config = mocker.Mock()
        config.output.enabled_outputs = ["csv"]
        config.output.csv_columns = ["conversation_group_id", "result"]
        config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        stats = handler._calculate_stats(sample_results)

        handler._generate_individual_reports(sample_results, "test", ["csv"], stats)

        assert (tmp_path / "test_detailed.csv").exists()

    def test_generate_individual_reports_json_only(
        self, tmp_path, sample_results, mocker
    ):
        """Test generating only JSON."""
        config = mocker.Mock()
        config.output.enabled_outputs = ["json"]
        config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        stats = handler._calculate_stats(sample_results)

        handler._generate_individual_reports(sample_results, "test", ["json"], stats)

        assert (tmp_path / "test_summary.json").exists()

    def test_generate_individual_reports_txt_only(
        self, tmp_path, sample_results, mocker
    ):
        """Test generating only TXT."""
        config = mocker.Mock()
        config.output.enabled_outputs = ["txt"]
        config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        stats = handler._calculate_stats(sample_results)

        handler._generate_individual_reports(sample_results, "test", ["txt"], stats)

        assert (tmp_path / "test_summary.txt").exists()

    def test_csv_with_all_columns(self, tmp_path, sample_results, mocker):
        """Test CSV with all available columns."""
        config = mocker.Mock()
        config.output.csv_columns = [
            "conversation_group_id",
            "turn_id",
            "metric_identifier",
            "result",
            "score",
            "threshold",
            "reason",
            "query",
            "response",
        ]
        config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        csv_file = handler._generate_csv_report(sample_results, "test")

        content = csv_file.read_text()
        assert "query" in content
        assert "response" in content
        assert "Python" in content

    def test_generate_reports_without_config(self, tmp_path, sample_results, mocker):
        """Test generating reports without system config."""
        mock_now = mocker.Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_now.isoformat.return_value = "2025-01-01T12:00:00"

        mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        ).now.return_value = mock_now

        handler = OutputHandler(output_dir=str(tmp_path))
        handler.generate_reports(sample_results)

        # Should use defaults
        assert (tmp_path / "evaluation_20250101_120000_detailed.csv").exists()

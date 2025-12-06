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
        api_tokens = {
            "total_api_input_tokens": 100,
            "total_api_output_tokens": 200,
            "total_api_tokens": 300,
        }

        json_file = handler._generate_json_summary(
            sample_results, "test", stats["basic"], stats["detailed"], api_tokens
        )

        assert json_file.exists()

        # Verify structure
        with open(json_file) as f:
            data = json.load(f)

        assert "summary_stats" in data or "results" in data
        assert data.get("total_evaluations") == 2 or len(data.get("results", [])) == 2
        # Verify API token usage is included in summary_stats.overall
        assert "summary_stats" in data
        assert data["summary_stats"]["overall"]["total_api_tokens"] == 300

    def test_generate_text_summary(self, tmp_path, sample_results):
        """Test text summary generation."""
        handler = OutputHandler(output_dir=str(tmp_path))
        stats = handler._calculate_stats(sample_results)
        api_tokens = {
            "total_api_input_tokens": 100,
            "total_api_output_tokens": 200,
            "total_api_tokens": 300,
        }

        txt_file = handler._generate_text_summary(
            sample_results, "test", stats["basic"], stats["detailed"], api_tokens
        )

        assert txt_file.exists()

        content = txt_file.read_text()
        assert "TOTAL" in content or "Summary" in content
        # Verify API token usage is included
        assert "Token Usage (API Calls)" in content

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


class TestOutputHandlerInitialization:
    """Additional tests for OutputHandler initialization and configuration."""

    def test_output_handler_initialization_default(self, tmp_path, mocker):
        """Test OutputHandler initialization with default parameters."""
        mock_print = mocker.patch("builtins.print")

        handler = OutputHandler(output_dir=str(tmp_path))

        assert handler.output_dir == tmp_path
        assert handler.base_filename == "evaluation"
        assert handler.system_config is None
        assert handler.output_dir.exists()

        mock_print.assert_called_with(f"✅ Output handler initialized: {tmp_path}")

    def test_output_handler_initialization_custom(self, tmp_path, mocker):
        """Test OutputHandler initialization with custom parameters."""
        system_config = mocker.Mock()
        system_config.llm.provider = "openai"

        mocker.patch("builtins.print")

        handler = OutputHandler(
            output_dir=str(tmp_path),
            base_filename="custom_eval",
            system_config=system_config,
        )

        assert handler.output_dir == tmp_path
        assert handler.base_filename == "custom_eval"
        assert handler.system_config == system_config

    def test_output_handler_creates_directory(self, tmp_path, mocker):
        """Test that OutputHandler creates output directory if it doesn't exist."""
        output_path = tmp_path / "new_output_dir"

        mocker.patch("builtins.print")

        handler = OutputHandler(output_dir=str(output_path))

        assert handler.output_dir.exists()
        assert handler.output_dir.is_dir()

    def test_generate_csv_with_specific_results(self, tmp_path, mocker):
        """Test CSV report generation with specific results."""
        results = [
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good performance",
                query="What is OpenShift?",
                response="OpenShift is a container platform.",
                execution_time=1.5,
                contexts='["OpenShift context"]',
                expected_keywords='[["OpenShift", "container"]]',
            ),
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="turn2",
                metric_identifier="test:metric",
                result="FAIL",
                score=0.3,
                threshold=0.7,
                reason="Poor performance",
                query="How to deploy?",
                response="Use oc apply.",
                execution_time=0.8,
                expected_response="Use oc apply -f deployment.yaml",
            ),
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="turn3",
                metric_identifier="ragas:response_relevancy",
                result="ERROR",
                score=None,
                threshold=None,
                reason="API connection failed",
                query="Create namespace",
                response="",
                execution_time=0.0,
            ),
        ]

        mocker.patch("builtins.print")

        handler = OutputHandler(output_dir=str(tmp_path))
        csv_file = handler._generate_csv_report(results, "test_eval")

        assert csv_file.exists()
        assert csv_file.suffix == ".csv"

        # Read and verify CSV content
        import csv as csv_module

        with open(csv_file, encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        assert rows[0]["conversation_group_id"] == "test_conv"
        assert rows[0]["result"] == "PASS"
        assert rows[0]["query"] == "What is OpenShift?"
        assert rows[0]["response"] == "OpenShift is a container platform."
        assert rows[0]["contexts"] == '["OpenShift context"]'
        assert rows[0]["expected_keywords"] == '[["OpenShift", "container"]]'

        assert rows[1]["result"] == "FAIL"
        assert rows[1]["expected_response"] == "Use oc apply -f deployment.yaml"

        # ERROR result - Additional fields are kept as empty
        assert rows[2]["result"] == "ERROR"
        assert rows[2]["query"] == "Create namespace"
        assert rows[2]["contexts"] == ""

    def test_csv_columns_configuration(self, tmp_path, mocker):
        """Test that CSV uses configured columns."""
        results = [
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="turn1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Good performance",
            )
        ]

        mocker.patch("builtins.print")

        # Test with custom system config
        system_config = mocker.Mock()
        system_config.output.csv_columns = ["conversation_group_id", "result", "score"]
        system_config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=system_config)
        csv_file = handler._generate_csv_report(results, "test_eval")

        # Read CSV headers
        import csv as csv_module

        with open(csv_file, encoding="utf-8") as f:
            reader = csv_module.reader(f)
            headers = next(reader)

        assert headers == ["conversation_group_id", "result", "score"]

    def test_filename_timestamp_format(self, tmp_path, mocker):
        """Test that generated filenames include proper timestamps."""
        results = []

        mocker.patch("builtins.print")

        handler = OutputHandler(output_dir=str(tmp_path), base_filename="test")

        # Mock datetime to get predictable timestamps
        mock_datetime = mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        )
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        csv_file = handler._generate_csv_report(results, "test_20240101_120000")

        assert "test_20240101_120000" in csv_file.name
        assert csv_file.suffix == ".csv"

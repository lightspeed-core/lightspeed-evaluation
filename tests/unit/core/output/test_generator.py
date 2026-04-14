# pylint: disable=protected-access

"""Unit tests for output generator."""

import json
from pathlib import Path

import csv as csv_module
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.models.summary import EvaluationSummary
from lightspeed_evaluation.core.output.generator import OutputHandler
from lightspeed_evaluation.core.storage import FileBackendConfig


class TestOutputHandler:
    """Tests for OutputHandler."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test handler initialization."""
        handler = OutputHandler(output_dir=str(tmp_path), base_filename="test")

        assert handler.output_dir == tmp_path
        assert handler.base_filename == "test"
        assert tmp_path.exists()

    def test_generate_json_summary(
        self, tmp_path: Path, sample_results: list[EvaluationResult]
    ) -> None:
        """Test JSON summary generation."""
        handler = OutputHandler(output_dir=str(tmp_path))
        summary = EvaluationSummary.from_results(sample_results)

        json_file = handler._generate_json_summary_from_model(summary, "test")

        assert json_file.exists()

        # Verify structure
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "summary_stats" in data or "results" in data
        assert data.get("total_evaluations") == 2 or len(data.get("results", [])) == 2
        # Verify API token usage is included in summary_stats.overall
        assert "summary_stats" in data
        assert data["summary_stats"]["overall"]["total_api_tokens"] == 0

    def test_generate_text_summary(
        self, tmp_path: Path, sample_results: list[EvaluationResult]
    ) -> None:
        """Test text summary generation."""
        handler = OutputHandler(output_dir=str(tmp_path))
        summary = EvaluationSummary.from_results(sample_results)

        txt_file = handler._generate_text_summary_from_model(summary, "test")

        assert txt_file.exists()

        content = txt_file.read_text()
        assert "Summary" in content
        # Verify API token usage is included
        assert "Token Usage (API Calls)" in content

    def test_get_output_directory(self, tmp_path: Path) -> None:
        """Test get output directory."""
        handler = OutputHandler(output_dir=str(tmp_path))

        assert handler.get_output_directory() == tmp_path

    def test_generate_reports_creates_files(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mock_system_config: MockerFixture,
        mocker: MockerFixture,
    ) -> None:
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
        self, tmp_path: Path, mock_system_config: MockerFixture, mocker: MockerFixture
    ) -> None:
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
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test generating only CSV."""
        file_config = FileBackendConfig(
            enabled_outputs=["csv"], csv_columns=["conversation_group_id", "result"]
        )
        config = mocker.Mock()
        config.storage = [file_config]
        config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        summary = EvaluationSummary.from_results(sample_results)

        handler._generate_individual_reports(sample_results, "test", ["csv"], summary)

        assert (tmp_path / "test_detailed.csv").exists()

    def test_generate_individual_reports_json_only(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test generating only JSON."""
        file_config = FileBackendConfig(enabled_outputs=["json"])
        config = mocker.Mock()
        config.storage = [file_config]
        config.visualization.enabled_graphs = []
        config.model_fields.keys.return_value = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        summary = EvaluationSummary.from_results(sample_results)

        handler._generate_individual_reports(sample_results, "test", ["json"], summary)

        assert (tmp_path / "test_summary.json").exists()

    def test_generate_individual_reports_txt_only(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test generating only TXT."""
        file_config = FileBackendConfig(enabled_outputs=["txt"])
        config = mocker.Mock()
        config.storage = [file_config]
        config.visualization.enabled_graphs = []
        config.model_fields.keys.return_value = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        summary = EvaluationSummary.from_results(sample_results)

        handler._generate_individual_reports(sample_results, "test", ["txt"], summary)

        assert (tmp_path / "test_summary.txt").exists()

    def test_csv_with_all_columns(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test CSV with all available columns."""
        file_config = FileBackendConfig(
            csv_columns=[
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
        )
        config = mocker.Mock()
        config.storage = [file_config]
        config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=config)
        csv_file = handler._generate_csv_report(sample_results, "test")

        content = csv_file.read_text()
        assert "query" in content
        assert "response" in content
        assert "Python" in content

    def test_generate_reports_without_config(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
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

    def test_output_handler_initialization_default(self, tmp_path: Path) -> None:
        """Test OutputHandler initialization with default parameters."""
        handler = OutputHandler(output_dir=str(tmp_path))

        assert handler.output_dir == tmp_path
        assert handler.base_filename == "evaluation"
        assert handler.system_config is None
        assert handler.output_dir.exists()

    def test_output_handler_initialization_custom(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test OutputHandler initialization with custom parameters."""
        system_config = mocker.Mock()
        system_config.llm.provider = "openai"

        handler = OutputHandler(
            output_dir=str(tmp_path),
            base_filename="custom_eval",
            system_config=system_config,
        )

        assert handler.output_dir == tmp_path
        assert handler.base_filename == "custom_eval"
        assert handler.system_config == system_config

    def test_output_handler_creates_directory(self, tmp_path: Path) -> None:
        """Test that OutputHandler creates output directory if it doesn't exist."""
        output_path = tmp_path / "new_output_dir"

        handler = OutputHandler(output_dir=str(output_path))

        assert handler.output_dir.exists()
        assert handler.output_dir.is_dir()

    def test_generate_csv_with_specific_results(self, tmp_path: Path) -> None:
        """Test CSV report generation with specific results."""
        metric_metadata = '{"max_ngram": 4}'
        results = [
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="turn1",
                metric_identifier="nlp:bleu",
                metric_metadata=metric_metadata,
                result="PASS",
                score=0.8,
                threshold=0.7,
                reason="Score is 0.8",
                query="What is OpenShift?",
                response="OpenShift is a container platform.",
                execution_time=1.5,
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

        handler = OutputHandler(output_dir=str(tmp_path))
        csv_file = handler._generate_csv_report(results, "test_eval")

        assert csv_file.exists()
        assert csv_file.suffix == ".csv"

        with open(csv_file, encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        assert rows[0]["conversation_group_id"] == "test_conv"
        assert rows[0]["result"] == "PASS"
        assert rows[0]["query"] == "What is OpenShift?"
        assert rows[0]["response"] == "OpenShift is a container platform."
        assert rows[0]["metric_metadata"] == metric_metadata

        assert rows[1]["result"] == "FAIL"
        assert rows[1]["expected_response"] == "Use oc apply -f deployment.yaml"

        # ERROR result - Additional fields are kept as empty
        assert rows[2]["result"] == "ERROR"
        assert rows[2]["query"] == "Create namespace"
        assert rows[2]["contexts"] == ""

    def test_csv_columns_configuration(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
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

        # Test with custom system config
        file_config = FileBackendConfig(
            csv_columns=["conversation_group_id", "result", "score"]
        )
        system_config = mocker.Mock()
        system_config.storage = [file_config]
        system_config.visualization.enabled_graphs = []

        handler = OutputHandler(output_dir=str(tmp_path), system_config=system_config)
        csv_file = handler._generate_csv_report(results, "test_eval")

        with open(csv_file, encoding="utf-8") as f:
            reader = csv_module.reader(f)
            headers = next(reader)

        assert headers == ["conversation_group_id", "result", "score"]

    def test_filename_timestamp_format(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that generated filenames include proper timestamps."""
        results: list[EvaluationResult] = []

        handler = OutputHandler(output_dir=str(tmp_path), base_filename="test")

        # Mock datetime to get predictable timestamps
        mocker.patch("lightspeed_evaluation.core.output.generator.datetime")

        csv_file = handler._generate_csv_report(results, "test_20240101_120000")

        assert "test_20240101_120000" in csv_file.name
        assert csv_file.suffix == ".csv"


class TestOutputHandlerSave:
    """Tests for OutputHandler.save() method."""

    def test_save_json_only(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test saving with JSON format only."""
        mock_now = mocker.Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_now.isoformat.return_value = "2025-01-01T12:00:00"
        mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        ).now.return_value = mock_now

        handler = OutputHandler(output_dir=str(tmp_path))
        summary = EvaluationSummary.from_results(sample_results)

        files = handler.save(summary, formats=["json"])

        assert len(files) == 1
        assert files[0].suffix == ".json"
        assert files[0].exists()

    def test_save_all_formats(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test saving with all formats."""
        mock_now = mocker.Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_now.isoformat.return_value = "2025-01-01T12:00:00"
        mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        ).now.return_value = mock_now

        handler = OutputHandler(output_dir=str(tmp_path))
        summary = EvaluationSummary.from_results(sample_results)

        files = handler.save(summary, formats=["csv", "json", "txt"])

        assert len(files) == 3

    def test_save_custom_output_dir(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test saving to a custom output directory."""
        mock_now = mocker.Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_now.isoformat.return_value = "2025-01-01T12:00:00"
        mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        ).now.return_value = mock_now

        custom_dir = tmp_path / "custom_output"
        handler = OutputHandler(output_dir=str(tmp_path))
        summary = EvaluationSummary.from_results(sample_results)

        files = handler.save(summary, formats=["json"], output_dir=str(custom_dir))

        assert len(files) == 1
        assert custom_dir.exists()
        assert files[0].parent == custom_dir

    def test_save_does_not_mutate_output_dir(
        self,
        tmp_path: Path,
        sample_results: list[EvaluationResult],
        mocker: MockerFixture,
    ) -> None:
        """Test that save() does not mutate the handler's output_dir."""
        mock_now = mocker.Mock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_now.isoformat.return_value = "2025-01-01T12:00:00"
        mocker.patch(
            "lightspeed_evaluation.core.output.generator.datetime"
        ).now.return_value = mock_now

        handler = OutputHandler(output_dir=str(tmp_path))
        summary = EvaluationSummary.from_results(sample_results)

        custom_dir = tmp_path / "custom"
        handler.save(summary, formats=["json"], output_dir=str(custom_dir))

        # handler.output_dir should remain unchanged (save() uses a local variable)
        assert handler.output_dir == tmp_path

"""Unit tests for core.output.generator module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.output.generator import OutputHandler


class TestOutputHandler:
    """Unit tests for OutputHandler class."""

    def test_output_handler_initialization_default(self):
        """Test OutputHandler initialization with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("builtins.print") as mock_print:
                handler = OutputHandler(output_dir=temp_dir)

                assert handler.output_dir == Path(temp_dir)
                assert handler.base_filename == "evaluation"
                assert handler.system_config is None
                assert handler.output_dir.exists()

                mock_print.assert_called_with(
                    f"✅ Output handler initialized: {temp_dir}"
                )

    def test_output_handler_initialization_custom(self):
        """Test OutputHandler initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system_config = {"llm": {"provider": "openai"}}

            with patch("builtins.print"):
                handler = OutputHandler(
                    output_dir=temp_dir,
                    base_filename="custom_eval",
                    system_config=system_config,
                )

                assert handler.output_dir == Path(temp_dir)
                assert handler.base_filename == "custom_eval"
                assert handler.system_config == system_config

    def test_output_handler_creates_directory(self):
        """Test that OutputHandler creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "new_output_dir"

            with patch("builtins.print"):
                handler = OutputHandler(output_dir=str(output_path))

                assert handler.output_dir.exists()
                assert handler.output_dir.is_dir()

    def test_generate_csv_report(self):
        """Test CSV report generation."""
        results = [
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="1",
                metric_identifier="test:metric",
                result="PASS",
                score=0.8,
                reason="Good performance",
                execution_time=1.5,
            ),
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="2",
                metric_identifier="test:metric",
                result="FAIL",
                score=0.3,
                reason="Poor performance",
                execution_time=0.8,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("builtins.print"):
                handler = OutputHandler(output_dir=temp_dir)
                csv_file = handler._generate_csv_report(results, "test_eval")

                assert csv_file.exists()
                assert csv_file.suffix == ".csv"

                # Read and verify CSV content
                import csv as csv_module

                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv_module.DictReader(f)
                    rows = list(reader)

                assert len(rows) == 2
                assert rows[0]["conversation_group_id"] == "test_conv"
                assert rows[0]["result"] == "PASS"
                assert rows[1]["result"] == "FAIL"

    def test_filename_timestamp_format(self):
        """Test that generated filenames include proper timestamps."""
        results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("builtins.print"):
                handler = OutputHandler(output_dir=temp_dir, base_filename="test")

                # Mock datetime to get predictable timestamps
                with patch(
                    "lightspeed_evaluation.core.output.generator.datetime"
                ) as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = (
                        "20240101_120000"
                    )

                    csv_file = handler._generate_csv_report(
                        results, "test_20240101_120000"
                    )

                    assert "test_20240101_120000" in csv_file.name
                    assert csv_file.suffix == ".csv"

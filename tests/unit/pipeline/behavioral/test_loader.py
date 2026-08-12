"""Tests for NxM behavioral loader."""

import json
from pathlib import Path

from lightspeed_evaluation.pipeline.behavioral.loader import load_run_data


class TestLoadRunData:
    """Tests for load_run_data."""

    def test_loads_summary_and_quality(self, tmp_path: Path) -> None:
        """Loads both summary.json and quality_report.json."""
        summary = {"summary_stats": {"overall": {"total": 10, "passed": 8}}}
        quality = {"quality_score": 0.85}
        (tmp_path / "evaluation_20260805_summary.json").write_text(json.dumps(summary))
        (tmp_path / "evaluation_20260805_quality_report.json").write_text(
            json.dumps(quality)
        )

        result = load_run_data(str(tmp_path), run_index=1)

        assert result is not None
        assert result.run_index == 1
        assert result.summary["summary_stats"]["overall"]["total"] == 10
        assert result.quality is not None
        assert result.quality["quality_score"] == 0.85

    def test_loads_summary_without_quality(self, tmp_path: Path) -> None:
        """Works when quality_report.json is absent."""
        summary = {"summary_stats": {"overall": {"total": 5}}}
        (tmp_path / "evaluation_20260805_summary.json").write_text(json.dumps(summary))

        result = load_run_data(str(tmp_path), run_index=2)

        assert result is not None
        assert result.quality is None

    def test_returns_none_for_missing_dir(self) -> None:
        """Returns None when directory does not exist."""
        result = load_run_data("/nonexistent/path", run_index=1)
        assert result is None

    def test_returns_none_for_missing_summary(self, tmp_path: Path) -> None:
        """Returns None when no summary.json exists."""
        (tmp_path / "other_file.txt").write_text("data")

        result = load_run_data(str(tmp_path), run_index=1)
        assert result is None

    def test_handles_corrupt_json(self, tmp_path: Path) -> None:
        """Returns None when summary.json is invalid JSON."""
        (tmp_path / "evaluation_20260805_summary.json").write_text("not json")

        result = load_run_data(str(tmp_path), run_index=1)
        assert result is None

    def test_loads_case_results_from_csv(self, tmp_path: Path) -> None:
        """Loads per-case results from detailed CSV."""
        summary = {"summary_stats": {"overall": {"TOTAL": 2}}}
        (tmp_path / "evaluation_20260805_summary.json").write_text(json.dumps(summary))
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result\n"
            "conv_group_1,turn_1,ragas:response_relevancy,PASS\n"
            "conv_group_2,turn_1,custom:answer_correctness,FAIL\n"
        )
        (tmp_path / "evaluation_20260805_detailed.csv").write_text(csv_content)

        result = load_run_data(str(tmp_path), run_index=1)

        assert result is not None
        assert result.case_results is not None
        assert len(result.case_results) == 2
        assert result.case_results[0]["result"] == "PASS"
        assert result.case_results[0]["conversation_group_id"] == "conv_group_1"
        assert (
            result.case_results[1]["metric_identifier"] == "custom:answer_correctness"
        )

    def test_no_csv_returns_none_case_results(self, tmp_path: Path) -> None:
        """case_results is None when no CSV exists."""
        summary = {"summary_stats": {"overall": {"TOTAL": 1}}}
        (tmp_path / "evaluation_20260805_summary.json").write_text(json.dumps(summary))

        result = load_run_data(str(tmp_path), run_index=1)

        assert result is not None
        assert result.case_results is None

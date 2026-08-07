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

"""Tests for NxM behavioral report."""

import json
from pathlib import Path

from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    EvalMetadata,
    EvalReport,
)
from lightspeed_evaluation.pipeline.behavioral.report import save_report


def _make_report() -> EvalReport:
    """Build a minimal EvalReport for testing."""
    return EvalReport(
        summary=EvalMetadata(
            timestamp="20260810_120000",
            total_agents=1,
            total_runs=2,
            repeat=1,
        ),
        agents={
            "model_a": AgentConsolidated(
                agent_name="model_a",
                runs_requested=2,
                runs_succeeded=2,
                conversations_count=5,
                overall={"pass_rate_mean": 85.0, "pass_rate_std": 5.0},
            ),
        },
    )


class TestSaveReport:
    """Tests for save_report()."""

    def test_writes_json(self, tmp_path: Path) -> None:
        """Saves eval_report.json with correct structure."""
        report = _make_report()

        path = save_report(report, str(tmp_path))

        assert Path(path).exists()
        assert Path(path).name == "eval_report.json"
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "summary" in data
        assert data["summary"]["timestamp"] == "20260810_120000"
        assert "agents" in data
        assert "model_a" in data["agents"]

    def test_excludes_none_fields(self, tmp_path: Path) -> None:
        """None fields are excluded from output."""
        report = _make_report()

        path = save_report(report, str(tmp_path))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "comparison" not in data
        assert "quality_score" not in data["agents"]["model_a"]

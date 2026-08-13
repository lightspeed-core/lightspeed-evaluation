"""Tests for NxM behavioral report."""

import json
from pathlib import Path

from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    EvalMetadata,
    EvalReport,
    RunSummary,
)
from lightspeed_evaluation.pipeline.behavioral.report import (
    _AGENT_REPORT_FIELDS,
    _EVAL_REPORT_AGENT_FIELDS,
    save_agent_report,
    save_report,
)


def _make_agent() -> AgentConsolidated:
    """Build an AgentConsolidated with all fields populated."""
    return AgentConsolidated(
        agent_name="model_a",
        runs_requested=2,
        runs_succeeded=2,
        conversations_count=5,
        overall={"pass_rate_mean": 85.0, "pass_rate_std": 5.0},
        by_metric={"ragas:faithfulness": {"mean": 0.9, "std": 0.05}},
        by_conversation={"conv_1": {"pass_rate_mean": 80.0}},
        quality_score={"mean": 0.85, "metrics": {}},
        per_run=[RunSummary(run_index=1, total=10, passed=8)],
    )


def _make_report() -> EvalReport:
    """Build a minimal EvalReport."""
    return EvalReport(
        summary=EvalMetadata(
            timestamp="2026-08-11T04:45:36+00:00",
            total_agents=1,
            total_runs=2,
            repeat=1,
        ),
        agents={"model_a": _make_agent()},
    )


class TestSaveAgentReport:
    """Tests for save_agent_report()."""

    def test_has_aggregated_data(self, tmp_path: Path) -> None:
        """Agent report contains cross-run aggregations."""
        path = save_agent_report(_make_agent(), str(tmp_path))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["agent_name"] == "model_a"
        assert data["overall"]["pass_rate_mean"] == 85.0
        assert "ragas:faithfulness" in data["by_metric"]
        assert "conv_1" in data["by_conversation"]
        assert "quality_score" in data

    def test_excludes_per_run(self, tmp_path: Path) -> None:
        """Per-run snapshots are in run-level files, not agent report."""
        path = save_agent_report(_make_agent(), str(tmp_path))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "per_run" not in data


class TestSaveReport:
    """Tests for save_report()."""

    def test_agent_headlines_only(self, tmp_path: Path) -> None:
        """Eval report has agent identity and overall stats."""
        path = save_report(_make_report(), str(tmp_path))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = data["agents"]["model_a"]
        assert "overall" in agent
        assert "agent_name" in agent
        assert "by_metric" not in agent
        assert "by_conversation" not in agent
        assert "per_run" not in agent
        assert "quality_score" not in agent

    def test_has_summary(self, tmp_path: Path) -> None:
        """Eval report has metadata."""
        path = save_report(_make_report(), str(tmp_path))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["summary"]["timestamp"] == "2026-08-11T04:45:36+00:00"
        assert data["summary"]["total_agents"] == 1


def test_field_sets_cover_all_model_fields() -> None:
    """Guard: every AgentConsolidated field is in a field set or explicitly excluded."""
    all_fields = set(AgentConsolidated.model_fields.keys())
    covered = _AGENT_REPORT_FIELDS | _EVAL_REPORT_AGENT_FIELDS | {"per_run"}
    assert (
        all_fields == covered
    ), f"New fields not in any field set: {all_fields - covered}"

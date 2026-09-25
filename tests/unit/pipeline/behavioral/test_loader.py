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


class TestTurnStats:
    """Tests for per-turn numeric extraction from detailed CSV."""

    @staticmethod
    def _write_summary(tmp_path: Path) -> None:
        """Write a minimal summary.json so load_run_data succeeds."""
        summary = {"summary_stats": {"overall": {"TOTAL": 1}}}
        (tmp_path / "eval_summary.json").write_text(json.dumps(summary))

    def test_extracts_turn_stats(self, tmp_path: Path) -> None:
        """Valid CSV yields separate agent and judge entries."""
        self._write_summary(tmp_path)
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result,"
            "api_input_tokens,api_output_tokens,"
            "judge_llm_input_tokens,judge_llm_output_tokens\n"
            "conv_1,t1,m1,PASS,100,50,200,80\n"
            "conv_1,t2,m1,FAIL,200,80,300,120\n"
        )
        (tmp_path / "eval_detailed.csv").write_text(csv_content)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is not None
        agent_entries = [s for s in result.turn_stats if "agent_input_tokens" in s]
        judge_entries = [s for s in result.turn_stats if "judge_input_tokens" in s]
        assert len(agent_entries) == 2
        assert len(judge_entries) == 2
        assert agent_entries[0]["agent_input_tokens"] == 100.0
        assert agent_entries[0]["agent_output_tokens"] == 50.0
        assert judge_entries[0]["judge_input_tokens"] == 200.0
        assert judge_entries[0]["judge_output_tokens"] == 80.0

    def test_agent_dedup_across_metrics(self, tmp_path: Path) -> None:
        """Same turn with multiple metrics produces one agent entry."""
        self._write_summary(tmp_path)
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result,"
            "api_input_tokens,api_output_tokens,"
            "judge_llm_input_tokens,judge_llm_output_tokens\n"
            "conv_1,t1,m1,PASS,100,50,200,80\n"
            "conv_1,t1,m2,PASS,100,50,350,140\n"
            "conv_1,t1,m3,FAIL,100,50,400,160\n"
        )
        (tmp_path / "eval_detailed.csv").write_text(csv_content)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is not None
        agent_entries = [s for s in result.turn_stats if "agent_input_tokens" in s]
        judge_entries = [s for s in result.turn_stats if "judge_input_tokens" in s]
        assert len(agent_entries) == 1
        assert len(judge_entries) == 3

    def test_conversation_level_rows_skip_agent(self, tmp_path: Path) -> None:
        """Rows without turn_id do not produce agent entries."""
        self._write_summary(tmp_path)
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result,"
            "api_input_tokens,api_output_tokens,"
            "judge_llm_input_tokens,judge_llm_output_tokens\n"
            "conv_1,,m1,PASS,500,200,100,40\n"
            "conv_1,t1,m1,PASS,100,50,200,80\n"
        )
        (tmp_path / "eval_detailed.csv").write_text(csv_content)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is not None
        agent_entries = [s for s in result.turn_stats if "agent_input_tokens" in s]
        judge_entries = [s for s in result.turn_stats if "judge_input_tokens" in s]
        assert len(agent_entries) == 1
        assert agent_entries[0]["agent_input_tokens"] == 100.0
        assert len(judge_entries) == 2

    def test_agent_first_nonzero_wins(self, tmp_path: Path) -> None:
        """First nonzero value per agent field is kept across metric rows."""
        self._write_summary(tmp_path)
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result,"
            "api_input_tokens,api_output_tokens\n"
            "conv_1,t1,m1,PASS,0,50\n"
            "conv_1,t1,m2,PASS,100,80\n"
        )
        (tmp_path / "eval_detailed.csv").write_text(csv_content)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is not None
        agent_entries = [s for s in result.turn_stats if "agent_output_tokens" in s]
        assert len(agent_entries) == 1
        assert agent_entries[0]["agent_output_tokens"] == 50.0
        assert agent_entries[0]["agent_input_tokens"] == 100.0

    def test_skips_non_numeric_values(self, tmp_path: Path) -> None:
        """Non-numeric or empty values are excluded from that field."""
        self._write_summary(tmp_path)
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result,"
            "api_input_tokens,api_output_tokens\n"
            "conv_1,t1,m1,PASS,,50\n"
            "conv_1,t2,m1,PASS,bad,80\n"
            "conv_1,t3,m1,PASS,300,\n"
        )
        (tmp_path / "eval_detailed.csv").write_text(csv_content)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is not None
        for ts in result.turn_stats:
            for val in ts.values():
                assert isinstance(val, float)

    def test_zero_values_excluded(self, tmp_path: Path) -> None:
        """Zero values are excluded (indicates no data)."""
        self._write_summary(tmp_path)
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result,"
            "api_input_tokens,api_output_tokens\n"
            "conv_1,t1,m1,PASS,0,50\n"
            "conv_1,t2,m1,PASS,100,80\n"
        )
        (tmp_path / "eval_detailed.csv").write_text(csv_content)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is not None
        agent_entries = [s for s in result.turn_stats if "agent_output_tokens" in s]
        assert len(agent_entries) == 2
        assert "agent_input_tokens" not in agent_entries[0]
        assert agent_entries[0]["agent_output_tokens"] == 50.0
        assert agent_entries[1]["agent_input_tokens"] == 100.0

    def test_no_numeric_columns(self, tmp_path: Path) -> None:
        """CSV without numeric columns → turn_stats is None."""
        self._write_summary(tmp_path)
        csv_content = (
            "conversation_group_id,turn_id,metric_identifier,result\n"
            "conv_1,t1,m1,PASS\n"
        )
        (tmp_path / "eval_detailed.csv").write_text(csv_content)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is None

    def test_no_csv_returns_none_turn_stats(self, tmp_path: Path) -> None:
        """turn_stats is None when no CSV exists."""
        self._write_summary(tmp_path)
        result = load_run_data(str(tmp_path), 0)

        assert result is not None
        assert result.turn_stats is None

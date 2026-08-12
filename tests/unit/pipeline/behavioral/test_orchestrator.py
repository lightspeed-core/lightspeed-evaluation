"""Tests for NxM behavioral evaluation orchestrator."""

from pathlib import Path
from typing import Any

from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models.agents import AgentsConfig
from lightspeed_evaluation.core.models.data import (
    EvaluationData,
    EvaluationResult,
    TurnData,
)
from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    RunResult,
    RunSummary,
)
from lightspeed_evaluation.pipeline.behavioral.orchestrator import (
    _build_agent_set,
    _build_and_save_report,
    _clone_config_for_run,
    _filter_conversations,
    _make_summary,
    _pin_conversations_to_agent,
    run,
)


class TestBuildAgentSet:
    """Tests for agent set building from config + eval data."""

    def test_default_agents_only(self) -> None:
        """Default agents with no eval data overrides."""
        data = [
            EvaluationData(
                conversation_group_id="c1",
                turns=[TurnData(turn_id="t1", query="Q")],
            ),
        ]
        result = _build_agent_set(["model_a", "model_b"], data)
        assert result == ["model_a", "model_b"]

    def test_eval_data_adds_agents(self) -> None:
        """Eval data agent overrides add to the set."""
        data = [
            EvaluationData(
                conversation_group_id="c1",
                agent=["model_c"],
                turns=[TurnData(turn_id="t1", query="Q")],
            ),
        ]
        result = _build_agent_set(["model_a"], data)
        assert result == ["model_a", "model_c"]

    def test_duplicates_removed(self) -> None:
        """Duplicate agents across default and eval data are deduplicated."""
        data = [
            EvaluationData(
                conversation_group_id="c1",
                agent=["model_a", "model_b"],
                turns=[TurnData(turn_id="t1", query="Q")],
            ),
        ]
        result = _build_agent_set(["model_a"], data)
        assert result == ["model_a", "model_b"]

    def test_empty_eval_data(self) -> None:
        """No eval data returns just default agents."""
        result = _build_agent_set(["model_a"], [])
        assert result == ["model_a"]


class TestFilterConversations:
    """Tests for conversation filtering per agent."""

    def _make_conv(
        self, conv_id: str, agent: list[str] | None = None
    ) -> EvaluationData:
        return EvaluationData(
            conversation_group_id=conv_id,
            agent=agent,
            turns=[TurnData(turn_id="t1", query="Q")],
        )

    def test_no_agent_uses_default(self) -> None:
        """Conversation without agent uses default agents."""
        convs = [self._make_conv("c1")]
        result = _filter_conversations(convs, "model_a", ["model_a", "model_b"])
        assert len(result) == 1

    def test_no_agent_excluded_if_not_in_default(self) -> None:
        """Conversation without agent excluded if agent not in defaults."""
        convs = [self._make_conv("c1")]
        result = _filter_conversations(convs, "model_c", ["model_a"])
        assert len(result) == 0

    def test_pinned_conversation_included(self) -> None:
        """Conversation pinned to agent is included."""
        convs = [self._make_conv("c1", agent=["model_a"])]
        result = _filter_conversations(convs, "model_a", ["model_b"])
        assert len(result) == 1

    def test_pinned_conversation_excluded(self) -> None:
        """Conversation pinned to different agent is excluded."""
        convs = [self._make_conv("c1", agent=["model_b"])]
        result = _filter_conversations(convs, "model_a", ["model_a"])
        assert len(result) == 0

    def test_mixed_conversations(self) -> None:
        """Mix of pinned and default conversations filtered correctly."""
        convs = [
            self._make_conv("c1"),
            self._make_conv("c2", agent=["model_a"]),
            self._make_conv("c3", agent=["model_b"]),
        ]
        result = _filter_conversations(convs, "model_a", ["model_a"])
        assert len(result) == 2
        assert {c.conversation_group_id for c in result} == {"c1", "c2"}


class TestPinConversationsToAgent:
    """Tests for pinning multi-agent conversations to current run agent."""

    def test_single_agent_unchanged(self) -> None:
        """Single-agent conversation left unchanged."""
        conv = EvaluationData(
            conversation_group_id="c1",
            agent=["model_a"],
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        result = _pin_conversations_to_agent([conv], "model_a")
        assert result[0].agent == ["model_a"]

    def test_multi_agent_pinned(self) -> None:
        """Multi-agent conversation pinned to current run agent."""
        conv = EvaluationData(
            conversation_group_id="c1",
            agent=["model_a", "model_c"],
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        result = _pin_conversations_to_agent([conv], "model_c")
        assert result[0].agent == ["model_c"]

    def test_no_agent_unchanged(self) -> None:
        """Conversation with no agent left unchanged."""
        conv = EvaluationData(
            conversation_group_id="c1",
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        result = _pin_conversations_to_agent([conv], "model_a")
        assert result[0].agent is None

    def test_original_not_mutated(self) -> None:
        """Original conversation not mutated."""
        conv = EvaluationData(
            conversation_group_id="c1",
            agent=["model_a", "model_c"],
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        _pin_conversations_to_agent([conv], "model_c")
        assert conv.agent == ["model_a", "model_c"]

    def test_agent_config_preserved(self) -> None:
        """Agent config is preserved through pinning."""
        conv = EvaluationData(
            conversation_group_id="c1",
            agent=["model_a", "model_c"],
            agent_config={"model_a": {"timeout": 300}, "model_c": {"timeout": 600}},
            turns=[TurnData(turn_id="t1", query="Q")],
        )
        result = _pin_conversations_to_agent([conv], "model_c")
        assert result[0].agent == ["model_c"]
        assert result[0].agent_config == {
            "model_a": {"timeout": 300},
            "model_c": {"timeout": 600},
        }


class TestCloneConfigForRun:
    """Tests for config cloning."""

    def _base_config(self) -> dict:
        return {
            "agents": {
                "enabled": True,
                "default": {"agent": ["model_a", "model_b"]},
                "agents": {
                    "model_a": {"type": "http_api", "cache_enabled": True},
                    "model_b": {"type": "http_api", "cache_enabled": True},
                },
            },
        }

    def test_sets_single_agent(self) -> None:
        """Cloned config has only the target agent in default."""
        cloned = _clone_config_for_run(
            self._base_config(), "model_a", False, "/tmp/out"
        )
        assert cloned["agents"]["default"]["agent"] == ["model_a"]

    def test_does_not_mutate_original(self) -> None:
        """Original config is not modified."""
        original = self._base_config()
        _clone_config_for_run(original, "model_a", False, "/tmp/out")
        assert original["agents"]["default"]["agent"] == ["model_a", "model_b"]

    def test_disables_cache(self) -> None:
        """Cache disabled when requested."""
        cloned = _clone_config_for_run(self._base_config(), "model_a", True, "/tmp/out")
        assert cloned["agents"]["agents"]["model_a"]["cache_enabled"] is False
        assert cloned["agents"]["agents"]["model_b"]["cache_enabled"] is True

    def test_cache_not_disabled_when_not_requested(self) -> None:
        """Cache stays enabled when not disabled."""
        cloned = _clone_config_for_run(
            self._base_config(), "model_a", False, "/tmp/out"
        )
        assert cloned["agents"]["agents"]["model_a"]["cache_enabled"] is True

    def test_injects_file_backend_when_missing(self) -> None:
        """File backend injected when no storage configured."""
        config = self._base_config()
        config["storage"] = []
        cloned = _clone_config_for_run(config, "model_a", False, "/tmp/run_out")
        file_entries = [
            s
            for s in cloned["storage"]
            if isinstance(s, dict) and s.get("type") == "file"
        ]
        assert len(file_entries) == 1
        assert file_entries[0]["output_dir"] == "/tmp/run_out"

    def test_does_not_inject_file_backend_when_present(self) -> None:
        """File backend not injected when already configured."""
        config = self._base_config()
        config["storage"] = [{"type": "file", "output_dir": "/existing"}]
        cloned = _clone_config_for_run(config, "model_a", False, "/tmp/run_out")
        file_entries = [
            s
            for s in cloned["storage"]
            if isinstance(s, dict) and s.get("type") == "file"
        ]
        assert len(file_entries) == 1
        assert file_entries[0]["output_dir"] == "/existing"


class TestMakeSummary:
    """Tests for _make_summary token deduplication."""

    def test_api_tokens_deduplicated_per_turn(self) -> None:
        """API tokens counted once per turn even with multiple metrics."""
        results = [
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t1",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.9,
                api_input_tokens=100,
                api_output_tokens=50,
                judge_llm_input_tokens=200,
                judge_llm_output_tokens=80,
                embedding_tokens=10,
            ),
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t1",
                metric_identifier="ragas:relevancy",
                result="PASS",
                score=0.8,
                api_input_tokens=100,
                api_output_tokens=50,
                judge_llm_input_tokens=150,
                judge_llm_output_tokens=60,
                embedding_tokens=5,
            ),
        ]
        summary = _make_summary(results)
        assert summary.agent_input_tokens == 100
        assert summary.agent_output_tokens == 50
        assert summary.judge_input_tokens == 350
        assert summary.judge_output_tokens == 140
        assert summary.embedding_tokens == 15

    def test_different_turns_counted_separately(self) -> None:
        """API tokens from different turns are summed."""
        results = [
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t1",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                api_input_tokens=100,
                api_output_tokens=50,
            ),
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t2",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                api_input_tokens=200,
                api_output_tokens=80,
            ),
        ]
        summary = _make_summary(results)
        assert summary.agent_input_tokens == 300
        assert summary.agent_output_tokens == 130

    def test_latency_averages_positive_values(self) -> None:
        """Agent latency is the mean of positive per-turn latencies."""
        results = [
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t1",
                metric_identifier="m1",
                result="PASS",
                agent_latency=2.0,
            ),
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t2",
                metric_identifier="m1",
                result="PASS",
                agent_latency=4.0,
            ),
        ]
        summary = _make_summary(results)
        assert summary.agent_latency == 3.0

    def test_latency_zero_when_no_positive(self) -> None:
        """Returns 0.0 when no turn has positive latency."""
        results = [
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t1",
                metric_identifier="m1",
                result="PASS",
                agent_latency=0.0,
            ),
        ]
        summary = _make_summary(results)
        assert summary.agent_latency == 0.0


class TestRunOrchestrator:
    """Integration tests for the orchestrator run function."""

    def _make_config_mock(
        self, mocker: MockerFixture, agents_dict: dict[str, Any]
    ) -> Any:
        """Create a mock SystemConfig with agents."""
        agents_config = AgentsConfig.model_validate(agents_dict)
        config = mocker.MagicMock(spec=["agents", "core", "model_dump"])
        config.agents = agents_config
        config.core.max_threads = 1

        full_dict = {"agents": {**agents_dict, "enabled": True}}
        for k, v in agents_dict.items():
            if k != "default" and isinstance(v, dict) and "type" in v:
                full_dict["agents"][k] = v
        config.model_dump.return_value = full_dict
        return config

    def _mock_pipeline(self, mocker: MockerFixture) -> Any:
        """Mock EvaluationPipeline and ConfigLoader to return test results."""
        mock_pipeline = mocker.MagicMock()
        mock_pipeline.run_evaluation.return_value = [
            EvaluationResult(
                conversation_group_id="c1",
                turn_id="t1",
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.9,
            ),
        ]
        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.EvaluationPipeline",
            return_value=mock_pipeline,
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.ConfigLoader"
        )
        return mock_pipeline

    def test_2x2_calls_pipeline_4_times(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Orchestrator calls pipeline per agent x repeat."""
        mock_pipeline = self._mock_pipeline(mocker)

        config = self._make_config_mock(
            mocker,
            {
                "default": {"agent": ["model_a", "model_b"], "repeat": 2},
                "model_a": {"type": "http_api"},
                "model_b": {"type": "http_api"},
            },
        )

        results = run(
            config,
            [
                EvaluationData(
                    conversation_group_id="c1",
                    turns=[TurnData(turn_id="t1", query="Q")],
                )
            ],
            str(tmp_path),
        )

        assert len(results) == 4
        assert all(r.success for r in results)
        assert mock_pipeline.run_evaluation.call_count == 4

    def test_1x1_produces_one_result(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """1x1 produces one run result with nested output path."""
        mock_pipeline = self._mock_pipeline(mocker)
        mock_pipeline.run_evaluation.return_value = []

        config = self._make_config_mock(
            mocker,
            {
                "default": {"agent": ["model_a"]},
                "model_a": {"type": "http_api"},
            },
        )

        results = run(
            config,
            [
                EvaluationData(
                    conversation_group_id="c1",
                    turns=[TurnData(turn_id="t1", query="Q")],
                )
            ],
            str(tmp_path),
        )

        assert len(results) == 1
        assert results[0].agent_name == "model_a"
        assert results[0].run_index == 1
        output_path = Path(results[0].output_dir)
        assert output_path.name == "run_1"
        assert output_path.parent.name == "model_a"
        assert output_path.parent.parent.name.startswith("eval_")

    def test_failed_run_does_not_stop_others(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """One failing run doesn't prevent others from completing."""
        call_count = 0

        def side_effect(*_args: Any, **_kwargs: Any) -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure")
            return []

        mock_pipeline = self._mock_pipeline(mocker)
        mock_pipeline.run_evaluation.side_effect = side_effect

        config = self._make_config_mock(
            mocker,
            {
                "default": {"agent": ["model_a"], "repeat": 3},
                "model_a": {"type": "http_api"},
            },
        )

        results = run(
            config,
            [
                EvaluationData(
                    conversation_group_id="c1",
                    turns=[TurnData(turn_id="t1", query="Q")],
                )
            ],
            str(tmp_path),
        )

        assert len(results) == 3
        assert sum(1 for r in results if not r.success) == 1
        assert sum(1 for r in results if r.success) == 2

    def test_output_directory_structure(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Verify output directory follows eval_<timestamp>/agent/run_N structure."""
        mock_pipeline = self._mock_pipeline(mocker)
        mock_pipeline.run_evaluation.return_value = []

        config = self._make_config_mock(
            mocker,
            {
                "default": {"agent": ["model_a"], "repeat": 2},
                "model_a": {"type": "http_api"},
            },
        )

        results = run(
            config,
            [
                EvaluationData(
                    conversation_group_id="c1",
                    turns=[TurnData(turn_id="t1", query="Q")],
                )
            ],
            str(tmp_path),
        )

        assert len(results) == 2
        for r in results:
            output_path = Path(r.output_dir)
            assert output_path.name == f"run_{r.run_index}"
            assert output_path.parent.name == "model_a"
            assert output_path.parent.parent.name.startswith("eval_")
            assert output_path.is_dir()


class TestBuildAndSaveReport:
    """Tests for _build_and_save_report integration."""

    def test_generates_report(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """Successful runs produce eval_report.json."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.load_run_data",
            return_value=mocker.Mock(
                summary={"summary_stats": {"overall": {"pass_rate": 80.0}}},
                quality=None,
                run_index=1,
            ),
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.consolidate",
            return_value=AgentConsolidated(
                agent_name="model_a",
                runs_requested=1,
                runs_succeeded=1,
                conversations_count=5,
            ),
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.compare_agents",
            return_value=None,
        )
        mock_save = mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.save_report"
        )
        mock_agent_save = mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.save_agent_report"
        )

        results = [
            RunResult(
                agent_name="model_a",
                run_index=1,
                output_dir=str(tmp_path),
                success=True,
                summary=RunSummary(total=10, passed=8),
            ),
        ]
        _build_and_save_report(
            results, str(tmp_path), repeat=1, timestamp="20260810_120000"
        )

        mock_agent_save.assert_called_once()
        assert mock_agent_save.call_args[0][1].endswith("model_a")
        mock_save.assert_called_once()
        report = mock_save.call_args[0][0]
        assert "model_a" in report.agents
        assert report.agents["model_a"].runs_requested == 1
        assert report.comparison is None
        assert report.summary.total_agents == 1
        assert report.summary.timestamp == "2026-08-10T12:00:00+00:00"

    def test_failure_does_not_raise(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Report generation failure is logged at ERROR, not raised."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.load_run_data",
            side_effect=RuntimeError("disk error"),
        )
        mock_logger = mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.logger"
        )

        results = [
            RunResult(
                agent_name="model_a",
                run_index=1,
                output_dir=str(tmp_path),
                success=True,
                summary=RunSummary(total=10, passed=8),
            ),
        ]
        _build_and_save_report(
            results, str(tmp_path), repeat=1, timestamp="20260810_120000"
        )

        mock_logger.error.assert_called_once()

    def test_mixed_success_and_failure(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Only successful runs with loadable data are consolidated."""
        mock_load = mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.load_run_data"
        )
        mock_load.return_value = mocker.Mock(
            summary={"summary_stats": {"overall": {"pass_rate": 90.0}}},
            quality=None,
            run_index=1,
        )

        mock_consolidate = mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.consolidate"
        )
        mock_consolidate.return_value = AgentConsolidated(
            agent_name="model_a",
            runs_requested=2,
            runs_succeeded=1,
            conversations_count=5,
        )

        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.compare_agents",
            return_value=None,
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.behavioral.orchestrator.save_report"
        )

        results = [
            RunResult(
                agent_name="model_a",
                run_index=1,
                output_dir=str(tmp_path),
                success=True,
            ),
            RunResult(
                agent_name="model_a",
                run_index=2,
                output_dir="",
                success=False,
                error="crashed",
            ),
        ]
        _build_and_save_report(
            results, str(tmp_path), repeat=2, timestamp="20260810_120000"
        )

        mock_load.assert_called_once()
        mock_consolidate.assert_called_once()
        assert len(mock_consolidate.call_args[0][1]) == 1

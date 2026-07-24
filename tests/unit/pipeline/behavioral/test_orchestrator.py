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
from lightspeed_evaluation.pipeline.behavioral.orchestrator import (
    _build_agent_set,
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
        assert summary["api_input_tokens"] == 100
        assert summary["api_output_tokens"] == 50
        assert summary["judge_llm_input_tokens"] == 350
        assert summary["judge_llm_output_tokens"] == 140
        assert summary["embedding_tokens"] == 15

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
        assert summary["api_input_tokens"] == 300
        assert summary["api_output_tokens"] == 130


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
        """1x1 produces one run result."""
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
        assert results[0].output_dir == str(tmp_path)

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
            assert f"model_a/run_{r.run_index}" in r.output_dir
            assert "eval_" in r.output_dir
            assert Path(r.output_dir).is_dir()

"""Tests for NxM behavioral comparison."""

from typing import Any

from lightspeed_evaluation.pipeline.behavioral.comparison import compare_agents
from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    ComparisonResult,
    RunSummary,
)


def _make_agent(name: str, **kwargs: Any) -> AgentConsolidated:
    """Build an AgentConsolidated for testing.

    Keyword args: pass_rate, latency, tokens, metrics, conversations, per_run.
    """
    overall: dict[str, float | None] = {
        "pass_rate_mean": kwargs.get("pass_rate", 80.0),
        "agent_latency_mean": kwargs.get("latency", 1.5),
        "agent_tokens_mean": kwargs.get("tokens", 1000.0),
    }
    by_metric: dict[str, dict[str, float | None]] = {}
    metrics = kwargs.get("metrics")
    if metrics:
        for metric_name, mean in metrics.items():
            by_metric[metric_name] = {"mean": mean, "std": 0.0}

    return AgentConsolidated(
        agent_name=name,
        runs_requested=2,
        runs_succeeded=2,
        conversations_count=kwargs.get("conversations", 5),
        overall=overall,
        by_metric=by_metric,
        per_run=kwargs.get("per_run", []),
    )


class TestCompareAgents:
    """Tests for compare_agents()."""

    def test_single_agent_returns_none(self) -> None:
        """No comparison for a single agent."""
        agents = {"a": _make_agent("a")}
        assert compare_agents(agents) is None

    def test_two_agents_produces_deltas(self) -> None:
        """Two agents produce one pairwise delta."""
        agents = {
            "a": _make_agent("a", pass_rate=90.0),
            "b": _make_agent("b", pass_rate=80.0),
        }
        result = compare_agents(agents)

        assert isinstance(result, ComparisonResult)
        assert len(result.deltas) == 1
        delta = result.deltas[0]
        assert delta.agent_a == "a"
        assert delta.agent_b == "b"
        assert delta.pass_rate_mean_delta == 10.0

    def test_rankings_by_pass_rate(self) -> None:
        """Rankings order by pass rate (higher is better)."""
        agents = {
            "a": _make_agent("a", pass_rate=70.0),
            "b": _make_agent("b", pass_rate=90.0),
        }
        result = compare_agents(agents)

        assert result is not None
        assert result.rankings.by_pass_rate == ["b", "a"]

    def test_rankings_by_latency(self) -> None:
        """Rankings order by latency (lower is better)."""
        agents = {
            "a": _make_agent("a", latency=2.0),
            "b": _make_agent("b", latency=1.0),
        }
        result = compare_agents(agents)

        assert result is not None
        assert result.rankings.by_latency == ["b", "a"]

    def test_metric_deltas_discovered(self) -> None:
        """Per-metric deltas are computed for shared metrics."""
        agents = {
            "a": _make_agent("a", metrics={"ragas:faith": 0.9, "custom:acc": 0.8}),
            "b": _make_agent("b", metrics={"ragas:faith": 0.7, "custom:acc": 0.85}),
        }
        result = compare_agents(agents)

        assert result is not None
        delta = result.deltas[0]
        assert abs(delta.score_deltas["ragas:faith"] - 0.2) < 1e-9
        assert abs(delta.score_deltas["custom:acc"] - (-0.05)) < 1e-9

    def test_incomparable_agents(self) -> None:
        """Agents with different conversation counts are flagged."""
        agents = {
            "a": _make_agent("a", conversations=5),
            "b": _make_agent("b", conversations=3),
        }
        result = compare_agents(agents)

        assert result is not None
        assert "a" in result.incomparable
        assert "b" in result.incomparable

    def test_three_agents_pairwise(self) -> None:
        """Three agents produce three pairwise deltas."""
        agents = {
            "a": _make_agent("a"),
            "b": _make_agent("b"),
            "c": _make_agent("c"),
        }
        result = compare_agents(agents)

        assert result is not None
        assert len(result.deltas) == 3

    def test_significance_with_per_run_data(self) -> None:
        """Significance tests run when per_run data is available."""
        runs_a = [RunSummary(total=10, passed=9, run_index=i) for i in range(5)]
        runs_b = [RunSummary(total=10, passed=2, run_index=i) for i in range(5)]
        agents = {
            "a": _make_agent("a", pass_rate=90.0, per_run=runs_a),
            "b": _make_agent("b", pass_rate=20.0, per_run=runs_b),
        }
        result = compare_agents(agents)

        assert result is not None
        delta = result.deltas[0]
        assert delta.significance is not None
        assert len(delta.significance) >= 1
        fisher = next((s for s in delta.significance if s.test == "fisher_exact"), None)
        assert fisher is not None
        assert fisher.significant is True

    def test_per_metric_significance(self) -> None:
        """Per-metric Mann-Whitney detects difference with realistic variance."""
        metric = "ragas:faithfulness"
        scores_a = [0.90, 0.85, 0.95, 0.88, 0.92]
        scores_b = [0.40, 0.35, 0.45, 0.38, 0.42]
        runs_a = [
            RunSummary(total=10, passed=9, run_index=i, by_metric={metric: s})
            for i, s in enumerate(scores_a)
        ]
        runs_b = [
            RunSummary(total=10, passed=4, run_index=i, by_metric={metric: s})
            for i, s in enumerate(scores_b)
        ]
        agents = {
            "a": _make_agent(
                "a", pass_rate=90.0, metrics={metric: 0.9}, per_run=runs_a
            ),
            "b": _make_agent(
                "b", pass_rate=40.0, metrics={metric: 0.4}, per_run=runs_b
            ),
        }
        result = compare_agents(agents)

        assert result is not None
        sig = result.deltas[0].significance
        assert sig is not None
        metric_test = next(
            (s for s in sig if s.test == "mann_whitney_u" and s.metric == metric),
            None,
        )
        assert metric_test is not None
        assert metric_test.significant is True

    def test_no_significance_without_per_run(self) -> None:
        """No significance tests when per_run is empty."""
        agents = {
            "a": _make_agent("a"),
            "b": _make_agent("b"),
        }
        result = compare_agents(agents)

        assert result is not None
        assert result.deltas[0].significance is None

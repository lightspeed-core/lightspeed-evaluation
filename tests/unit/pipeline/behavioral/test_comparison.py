"""Tests for NxM behavioral comparison."""

from typing import Any

from lightspeed_evaluation.pipeline.behavioral.comparison import compare_agents
from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    ComparisonResult,
)


def _make_agent(name: str, **kwargs: Any) -> AgentConsolidated:
    """Build an AgentConsolidated for testing.

    Keyword args: pass_rate, latency, tokens, metrics, conversations.
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

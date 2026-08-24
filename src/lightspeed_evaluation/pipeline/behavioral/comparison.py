"""Compare consolidated agents: pairwise deltas and per-dimension rankings."""

import itertools
from typing import Optional

from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    ComparisonResult,
    PairwiseDelta,
    Rankings,
    SignificanceResult,
)
from lightspeed_evaluation.pipeline.behavioral.statistics import (
    metric_significance,
    significance_tests,
)


def compare_agents(
    agents: dict[str, AgentConsolidated],
    alpha: float = 0.05,
) -> Optional[ComparisonResult]:
    """Compare multiple agents: deltas, rankings, incomparable list.

    Per-metric deltas are data-driven (discovered from by_metric keys).
    Overall deltas use fixed structural keys (pass rate, latency, tokens).

    Args:
        agents: Agent name → AgentConsolidated mapping.
        alpha: Significance level for statistical tests.

    Returns:
        ComparisonResult with deltas, rankings, incomparable. None if < 2 agents.
    """
    if len(agents) < 2:
        return None

    agent_list = sorted(agents.values(), key=lambda a: a.agent_name)
    return ComparisonResult(
        deltas=_compute_deltas(agent_list, alpha),
        rankings=_compute_rankings(agent_list),
        incomparable=_find_incomparable(agent_list),
    )


def _compute_deltas(
    agents: list[AgentConsolidated],
    alpha: float,
) -> list[PairwiseDelta]:
    """Compute pairwise deltas between all agent pairs."""
    return [_pair_delta(a, b, alpha) for a, b in itertools.combinations(agents, 2)]


def _pair_delta(
    agent_a: AgentConsolidated,
    agent_b: AgentConsolidated,
    alpha: float,
) -> PairwiseDelta:
    """Compute delta between two agents (a - b)."""
    shared_metrics = set(agent_a.by_metric) & set(agent_b.by_metric)
    score_deltas: dict[str, float] = {}
    for metric in shared_metrics:
        mean_a = agent_a.by_metric[metric].get("mean")
        mean_b = agent_b.by_metric[metric].get("mean")
        if mean_a is not None and mean_b is not None:
            score_deltas[metric] = mean_a - mean_b

    sig = _compute_significance(agent_a, agent_b, alpha)

    return PairwiseDelta(
        agent_a=agent_a.agent_name,
        agent_b=agent_b.agent_name,
        pass_rate_mean_delta=_overall_delta(agent_a, agent_b, "pass_rate_mean"),
        agent_latency_mean_delta=_overall_delta(agent_a, agent_b, "agent_latency_mean"),
        agent_tokens_mean_delta=_overall_delta(agent_a, agent_b, "agent_tokens_mean"),
        score_deltas=score_deltas,
        significance=sig or None,
    )


def _compute_significance(
    agent_a: AgentConsolidated,
    agent_b: AgentConsolidated,
    alpha: float,
) -> list[SignificanceResult]:
    """Run significance tests: overall pass/fail + per-metric scores."""
    pass_counts_a = [r.passed for r in agent_a.per_run]
    totals_a = [r.total for r in agent_a.per_run]
    pass_counts_b = [r.passed for r in agent_b.per_run]
    totals_b = [r.total for r in agent_b.per_run]
    results = significance_tests(
        pass_counts_a, totals_a, pass_counts_b, totals_b, alpha=alpha
    )

    shared_metrics = set(agent_a.by_metric) & set(agent_b.by_metric)
    for metric in sorted(shared_metrics):
        scores_a = _per_run_metric_scores(agent_a, metric)
        scores_b = _per_run_metric_scores(agent_b, metric)
        sig = metric_significance(scores_a, scores_b, metric, alpha)
        if sig is not None:
            results.append(sig)

    return results


def _per_run_metric_scores(agent: AgentConsolidated, metric: str) -> list[float]:
    """Extract per-run scores for a specific metric."""
    return [
        r.by_metric[metric]
        for r in agent.per_run
        if r.by_metric and metric in r.by_metric
    ]


def _overall_delta(
    agent_a: AgentConsolidated,
    agent_b: AgentConsolidated,
    key: str,
) -> Optional[float]:
    """Compute delta for a single overall key, or None if either is missing."""
    val_a = agent_a.overall.get(key)
    val_b = agent_b.overall.get(key)
    if val_a is not None and val_b is not None:
        return val_a - val_b
    return None


def _compute_rankings(agents: list[AgentConsolidated]) -> Rankings:
    """Compute per-dimension rankings (best first)."""
    by_metric: dict[str, list[str]] = {}

    all_metrics: set[str] = set()
    for agent in agents:
        all_metrics.update(agent.by_metric.keys())
    for metric in sorted(all_metrics):
        scored: list[tuple[str, float]] = [
            (a.agent_name, val)
            for a in agents
            if metric in a.by_metric
            and (val := a.by_metric[metric].get("mean")) is not None
        ]
        if scored:
            by_metric[metric] = [
                name for name, _ in sorted(scored, key=lambda x: x[1], reverse=True)
            ]

    return Rankings(
        by_pass_rate=_rank(agents, "pass_rate_mean", reverse=True),
        by_latency=_rank(agents, "agent_latency_mean"),
        by_tokens=_rank(agents, "agent_tokens_mean"),
        by_metric=by_metric,
    )


def _rank(
    agents: list[AgentConsolidated],
    key: str,
    reverse: bool = False,
) -> list[str]:
    """Rank agents by an overall key (lower is better unless reverse)."""
    scored: list[tuple[str, float]] = [
        (a.agent_name, val) for a in agents if (val := a.overall.get(key)) is not None
    ]
    if not scored:
        return []
    return [name for name, _ in sorted(scored, key=lambda x: x[1], reverse=reverse)]


def _find_incomparable(agents: list[AgentConsolidated]) -> list[str]:
    """Find agents that can't be fairly compared (different conversation sets).

    Returns all agent names if conversation counts differ, empty if uniform.
    """
    if not agents:
        return []
    counts = {a.agent_name: a.conversations_count for a in agents}
    if len(set(counts.values())) <= 1:
        return []
    return list(counts.keys())

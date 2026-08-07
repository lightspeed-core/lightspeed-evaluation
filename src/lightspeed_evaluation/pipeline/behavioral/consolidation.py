"""Consolidate per-run data into AgentConsolidated.

Data-driven: discovers metrics from summary.json, never hardcodes
metric names or dimension lists. Adding a new metric to the pipeline
flows through automatically.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from lightspeed_evaluation.pipeline.behavioral.loader import RunData
from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    RunSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class _CollectedData:
    """Intermediate collection of per-run values for aggregation."""

    pass_rates: list[float] = field(default_factory=list)
    metric_scores: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    conv_pass_rates: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    latencies: list[float] = field(default_factory=list)
    per_run: list[RunSummary] = field(default_factory=list)


def consolidate(
    agent_name: str,
    runs: list[RunData],
    runs_requested: int,
) -> AgentConsolidated:
    """Aggregate per-run data into a single AgentConsolidated.

    Args:
        agent_name: Agent identifier.
        runs: Loaded run data (summary + quality per run).
        runs_requested: Total runs requested (including failures).

    Returns:
        AgentConsolidated with overall stats, per-metric breakdowns,
        per-conversation breakdowns, quality score, and per-run snapshots.
    """
    if not runs:
        return AgentConsolidated(
            agent_name=agent_name,
            runs_requested=runs_requested,
            runs_succeeded=0,
            conversations_count=0,
        )

    collected = _collect_run_data(runs)
    overall = _build_overall(collected.pass_rates, collected.latencies)
    overall["total_agent_tokens"] = sum(
        r.agent_input_tokens + r.agent_output_tokens for r in collected.per_run
    )
    overall["total_judge_tokens"] = sum(
        r.judge_input_tokens + r.judge_output_tokens for r in collected.per_run
    )
    overall["total_embedding_tokens"] = sum(
        float(r.embedding_tokens) for r in collected.per_run
    )

    return AgentConsolidated(
        agent_name=agent_name,
        runs_requested=runs_requested,
        runs_succeeded=len(runs),
        conversations_count=len(collected.conv_pass_rates),
        overall=overall,
        by_metric=_build_by_metric(collected.metric_scores),
        by_conversation=_build_by_conversation(collected.conv_pass_rates),
        quality_score=_build_quality(runs),
        per_run=collected.per_run,
    )


def _collect_run_data(runs: list[RunData]) -> _CollectedData:
    """Iterate runs once, collecting all dimensions for aggregation."""
    data = _CollectedData()

    for run in runs:
        data.per_run.append(_extract_run_snapshot(run))

        stats = run.summary.get("summary_stats", {})
        overall = stats.get("overall", {})

        pr = overall.get("pass_rate")
        if pr is not None:
            data.pass_rates.append(pr)

        _collect_metric_scores(stats, data.metric_scores)
        _collect_conversation_pass_rates(stats, data.conv_pass_rates)

        lat_mean = stats.get("agent_latency_stats", {}).get("mean")
        if lat_mean is not None and lat_mean > 0:
            data.latencies.append(lat_mean)

    data.metric_scores = dict(data.metric_scores)
    data.conv_pass_rates = dict(data.conv_pass_rates)
    return data


def _extract_run_snapshot(run: RunData) -> RunSummary:
    """Build a RunSummary snapshot from loaded run data."""
    stats = run.summary.get("summary_stats", {})
    overall = stats.get("overall", {})

    by_metric_scores: dict[str, float] = {}
    for name, metric_data in stats.get("by_metric", {}).items():
        score_stats = metric_data.get("score_statistics", {})
        mean = score_stats.get("mean")
        if mean is not None:
            by_metric_scores[name] = mean

    quality_score = None
    if run.quality:
        quality_score = run.quality.get("quality_score")

    return RunSummary(
        run_index=run.run_index,
        total=overall.get("total", 0),
        passed=overall.get("passed", 0),
        failed=overall.get("failed", 0),
        error=overall.get("error", 0),
        skipped=overall.get("skipped", 0),
        pass_rate=overall.get("pass_rate"),
        by_metric=by_metric_scores or None,
        quality_score=quality_score,
        agent_latency=stats.get("agent_latency_stats", {}).get("mean") or 0.0,
        agent_input_tokens=overall.get("total_api_input_tokens", 0),
        agent_output_tokens=overall.get("total_api_output_tokens", 0),
        judge_input_tokens=overall.get("total_judge_llm_input_tokens", 0),
        judge_output_tokens=overall.get("total_judge_llm_output_tokens", 0),
        embedding_tokens=overall.get("total_embedding_tokens", 0),
    )


def _collect_metric_scores(
    stats: dict[str, Any],
    metric_scores: dict[str, list[float]],
) -> None:
    """Collect per-metric mean scores from a run's summary stats."""
    for name, metric_data in stats.get("by_metric", {}).items():
        score_stats = metric_data.get("score_statistics", {})
        mean = score_stats.get("mean")
        if mean is not None:
            metric_scores[name].append(mean)


def _collect_conversation_pass_rates(
    stats: dict[str, Any],
    conv_pass_rates: dict[str, list[float]],
) -> None:
    """Collect per-conversation pass rates from a run's summary stats."""
    for conv_id, conv_data in stats.get("by_conversation", {}).items():
        pr = conv_data.get("pass_rate")
        if pr is not None:
            conv_pass_rates[conv_id].append(pr)


def _build_overall(
    pass_rates: list[float],
    latencies: list[float],
) -> dict[str, Optional[float]]:
    """Build statistical aggregations for pass rates and latencies."""
    result: dict[str, Optional[float]] = {}

    if pass_rates:
        result["pass_rate_mean"] = statistics.mean(pass_rates)
        result["pass_rate_std"] = (
            statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0.0
        )

    if latencies:
        result["agent_latency_mean"] = statistics.mean(latencies)
        result["agent_latency_std"] = (
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        )

    return result


def _build_by_metric(
    metric_scores: dict[str, list[float]],
) -> dict[str, dict[str, Optional[float]]]:
    """Build per-metric aggregated stats."""
    result: dict[str, dict[str, Optional[float]]] = {}
    for name, scores in metric_scores.items():
        entry: dict[str, Optional[float]] = {
            "mean": statistics.mean(scores),
            "min": min(scores),
            "max": max(scores),
        }
        entry["std"] = statistics.stdev(scores) if len(scores) > 1 else 0.0
        result[name] = entry
    return result


def _build_by_conversation(
    conv_pass_rates: dict[str, list[float]],
) -> dict[str, dict[str, Optional[float]]]:
    """Build per-conversation aggregated pass rates."""
    result: dict[str, dict[str, Optional[float]]] = {}
    for conv_id, rates in conv_pass_rates.items():
        entry: dict[str, Optional[float]] = {
            "pass_rate_mean": statistics.mean(rates),
        }
        entry["pass_rate_std"] = statistics.stdev(rates) if len(rates) > 1 else 0.0
        result[conv_id] = entry
    return result


def _build_quality(runs: list[RunData]) -> Optional[dict[str, Any]]:
    """Build aggregated quality score from per-run quality reports."""
    scores: list[float] = []
    metric_weights: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run in runs:
        if not run.quality:
            continue
        qs = run.quality.get("quality_score")
        if qs is not None:
            scores.append(qs)
        for metric_id, metric_data in run.quality.get("quality_metrics", {}).items():
            metric_weights[metric_id].append(metric_data)

    if not scores:
        return None

    result: dict[str, Any] = {
        "mean": statistics.mean(scores),
    }
    if len(scores) > 1:
        result["std"] = statistics.stdev(scores)

    if metric_weights:
        metrics: dict[str, Any] = {}
        for metric_id, entries in metric_weights.items():
            means = [e.get("mean", 0.0) for e in entries if e.get("mean") is not None]
            weights = [w for e in entries if (w := e.get("weight")) is not None]
            metrics[metric_id] = {
                "mean": statistics.mean(means) if means else None,
                "weight": statistics.mean(weights) if weights else None,
            }
        result["metrics"] = metrics

    return result

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

import numpy as np

from lightspeed_evaluation.pipeline.behavioral.loader import RunData
from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    RunSummary,
)
from lightspeed_evaluation.pipeline.behavioral.statistics import (
    confidence_interval,
    pass_at_k,
    pass_hat_k,
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
    latency_percentiles: dict[str, list[float]] = field(
        default_factory=lambda: {"p95": [], "p99": []}
    )
    turn_stats: list[dict[str, float]] = field(default_factory=list)
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
    num_runs = len(collected.per_run)

    total_agent_in = sum(r.agent_input_tokens for r in collected.per_run)
    total_agent_out = sum(r.agent_output_tokens for r in collected.per_run)
    overall["total_agent_input_tokens"] = total_agent_in
    overall["total_agent_output_tokens"] = total_agent_out
    if num_runs > 0:
        overall["agent_input_tokens_mean"] = total_agent_in / num_runs
        overall["agent_output_tokens_mean"] = total_agent_out / num_runs
        overall["agent_tokens_mean"] = (total_agent_in + total_agent_out) / num_runs

    for pct in ("p95", "p99"):
        vals = collected.latency_percentiles[pct]
        if vals:
            overall[f"agent_latency_{pct}_max"] = max(vals)

    if collected.turn_stats:
        token_pcts = _compute_token_percentiles(collected.turn_stats)
        overall.update(token_pcts.get("agent", {}))

    by_metric = _build_by_metric(collected.metric_scores)
    by_conversation = _build_by_conversation(collected.conv_pass_rates)
    if num_runs > 1:
        _wire_pass_metrics(runs, num_runs, overall, by_metric, by_conversation)

    eval_costs = _build_eval_costs(collected.per_run, collected.turn_stats)

    return AgentConsolidated(
        agent_name=agent_name,
        runs_requested=runs_requested,
        runs_succeeded=len(runs),
        conversations_count=len(collected.conv_pass_rates),
        overall=overall,
        by_metric=by_metric,
        by_conversation=by_conversation,
        eval_costs=eval_costs,
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

        lat_stats = stats.get("agent_latency_stats", {})
        lat_mean = lat_stats.get("mean")
        if lat_mean is not None and lat_mean > 0:
            data.latencies.append(lat_mean)
        for pct in ("p95", "p99"):
            val = lat_stats.get(pct)
            if val is not None and val > 0:
                data.latency_percentiles[pct].append(val)

        if run.turn_stats:
            data.turn_stats.extend(run.turn_stats)

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
        total=overall.get("TOTAL", 0),
        passed=overall.get("PASS", 0),
        failed=overall.get("FAIL", 0),
        error=overall.get("ERROR", 0),
        skipped=overall.get("SKIPPED", 0),
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
        ci = confidence_interval(pass_rates)
        if ci is not None:
            result["pass_rate_ci_low"] = max(0.0, ci[0])
            result["pass_rate_ci_high"] = min(100.0, ci[1])

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
            "runs_count": len(scores),
            "mean": statistics.mean(scores),
            "min": min(scores),
            "max": max(scores),
        }
        entry["std"] = statistics.stdev(scores) if len(scores) > 1 else 0.0
        ci = confidence_interval(scores)
        if ci is not None:
            entry["ci_low"] = ci[0]
            entry["ci_high"] = ci[1]
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


def _collect_case_counts(
    runs: list[RunData],
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
    """Count PASS and total (PASS+FAIL) per case across runs."""
    case_pass: dict[tuple[str, str, str], int] = defaultdict(int)
    case_total: dict[tuple[str, str, str], int] = defaultdict(int)

    for run in runs:
        if not run.case_results:
            continue
        for case in run.case_results:
            if case["result"] not in ("PASS", "FAIL"):
                continue
            key = (
                case["conversation_group_id"],
                case["turn_id"],
                case["metric_identifier"],
            )
            case_total[key] += 1
            if case["result"] == "PASS":
                case_pass[key] += 1

    return case_pass, case_total


def _wire_pass_metrics(
    runs: list[RunData],
    num_runs: int,
    overall: dict[str, Optional[float]],
    by_metric: dict[str, dict[str, Optional[float]]],
    by_conversation: dict[str, dict[str, Optional[float]]],
) -> None:
    """Wire pass@k, pass@1, and pass^k into overall and breakdowns."""
    case_pass, case_total = _collect_case_counts(runs)
    if not case_total:
        return

    keys = sorted(case_total.keys())
    passes = [case_pass.get(key, 0) for key in keys]
    totals_list = [case_total[key] for key in keys]
    by_metric_groups = _group_cases(keys, case_pass, case_total, idx=2)
    by_conv_groups = _group_cases(keys, case_pass, case_total, idx=0)

    for label, func, k in _pass_metric_specs(num_runs):
        overall[label] = func(passes, totals_list, k=k)
        _distribute_to_groups(func, k, label, by_metric_groups, by_metric)
        _distribute_to_groups(func, k, label, by_conv_groups, by_conversation)


def _pass_metric_specs(
    num_runs: int,
) -> list[tuple[str, Any, int]]:
    """Return (label, function, k) specs for all pass metrics."""
    return [
        ("pass_at_k", pass_at_k, num_runs),
        ("pass_at_1", pass_at_k, 1),
        ("pass_hat_k", pass_hat_k, num_runs),
    ]


def _distribute_to_groups(
    func: Any,
    k: int,
    label: str,
    groups: dict[str, tuple[list[int], list[int]]],
    target: dict[str, dict[str, Optional[float]]],
) -> None:
    """Distribute a pass metric computation to grouped breakdowns."""
    for group, (g_pass, g_total) in groups.items():
        if group in target:
            target[group][label] = func(g_pass, g_total, k=k)


def _group_cases(
    keys: list[tuple[str, str, str]],
    case_pass: dict[tuple[str, str, str], int],
    case_total: dict[tuple[str, str, str], int],
    idx: int,
) -> dict[str, tuple[list[int], list[int]]]:
    """Group case pass/total counts by a tuple index (0=conv_id, 2=metric)."""
    grouped_pass: dict[str, list[int]] = defaultdict(list)
    grouped_total: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        group = key[idx]
        grouped_pass[group].append(case_pass.get(key, 0))
        grouped_total[group].append(case_total[key])
    return {g: (grouped_pass[g], grouped_total[g]) for g in grouped_pass}


_AGENT_TOKEN_FIELDS = ["agent_input_tokens", "agent_output_tokens"]
_JUDGE_TOKEN_FIELDS = ["judge_input_tokens", "judge_output_tokens"]


def _compute_token_percentiles(
    turn_stats: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute agent token p95/p99 from pooled per-turn values across all runs.

    Returns dict with "agent" sub-dict containing p95_max/p99_max keyed fields.
    """
    result: dict[str, dict[str, float]] = {"agent": {}}
    for field_name in _AGENT_TOKEN_FIELDS:
        values = [ts[field_name] for ts in turn_stats if field_name in ts]
        if not values:
            continue
        arr = np.array(values)
        result["agent"][f"{field_name}_p95_max"] = float(np.percentile(arr, 95))
        result["agent"][f"{field_name}_p99_max"] = float(np.percentile(arr, 99))
    return result


def _build_eval_costs(
    per_run: list[RunSummary],
    turn_stats: list[dict[str, float]],
) -> Optional[dict[str, float]]:
    """Build eval costs section with judge/embedding totals and max."""
    total_judge_in = sum(r.judge_input_tokens for r in per_run)
    total_judge_out = sum(r.judge_output_tokens for r in per_run)
    total_embed = sum(float(r.embedding_tokens) for r in per_run)

    if total_judge_in == 0 and total_judge_out == 0 and total_embed == 0:
        if not turn_stats:
            return None

    costs: dict[str, float] = {
        "total_judge_input_tokens": total_judge_in,
        "total_judge_output_tokens": total_judge_out,
        "total_embedding_tokens": total_embed,
    }

    if turn_stats:
        for field_name in _JUDGE_TOKEN_FIELDS:
            values = [ts[field_name] for ts in turn_stats if field_name in ts]
            if values:
                key = field_name.replace("judge_", "max_judge_")
                costs[key] = max(values)

    return costs

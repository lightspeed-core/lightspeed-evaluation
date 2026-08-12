"""Save per-agent and comparison reports."""

import json
import os

from lightspeed_evaluation.pipeline.behavioral.models import (
    AgentConsolidated,
    EvalReport,
)

_AGENT_REPORT_FIELDS = {
    "agent_name",
    "runs_requested",
    "runs_succeeded",
    "conversations_count",
    "overall",
    "by_metric",
    "by_conversation",
    "quality_score",
}

_EVAL_REPORT_AGENT_FIELDS = {
    "agent_name",
    "runs_requested",
    "runs_succeeded",
    "conversations_count",
    "overall",
}


def save_agent_report(agent: AgentConsolidated, agent_dir: str) -> str:
    """Save per-agent consolidation report.

    Contains cross-run aggregations only. Per-run detail is in the
    run-level files. Adjust _AGENT_REPORT_FIELDS to expand.

    Args:
        agent: Consolidated agent data.
        agent_dir: Agent's output directory (eval_<ts>/agent_name/).

    Returns:
        Path to the saved file.
    """
    os.makedirs(agent_dir, exist_ok=True)
    path = os.path.join(agent_dir, "agent_report.json")
    data = agent.model_dump(exclude_none=True)
    data = {k: v for k, v in data.items() if k in _AGENT_REPORT_FIELDS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def save_report(report: EvalReport, output_dir: str) -> str:
    """Save evaluation report with agent headlines and comparison.

    Per-agent fields controlled by _EVAL_REPORT_AGENT_FIELDS.
    Full agent detail is in agent_report.json.

    Args:
        report: The EvalReport to save.
        output_dir: Directory to write the file (typically eval_<ts>/).

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_report.json")
    data = report.model_dump(exclude_none=True)
    data["agents"] = {
        name: {k: v for k, v in agent.items() if k in _EVAL_REPORT_AGENT_FIELDS}
        for name, agent in data["agents"].items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

"""Models for NxM behavioral evaluation orchestration."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunContext(BaseModel):
    """Serializable context for a single evaluation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config_dict: dict[str, Any]
    eval_data_dicts: list[dict[str, Any]]
    default_agents: list[str]
    agent_name: str
    run_index: int
    run_output_dir: str
    extra: Optional[dict[str, Any]] = None


class RunSummary(BaseModel):
    """Per-run summary for orchestrator and eval_report.json.

    Orchestrator sets status counts + tokens + latency.
    Consolidation adds pass_rate, by_metric, quality_score for the report.

    Field order: status → metrics → quality → agent → judge → embedding.
    """

    # Status counts
    total: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0
    skipped: int = 0
    # Metrics (set by consolidation, not orchestrator)
    run_index: Optional[int] = None
    pass_rate: Optional[float] = None
    by_metric: Optional[dict[str, float]] = None
    # Quality (set by consolidation)
    quality_score: Optional[float] = None
    # Agent
    agent_latency: float = 0.0
    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    # Judge
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    # Embedding
    embedding_tokens: int = 0


class RunResult(BaseModel):
    """Result metadata from a single evaluation run."""

    agent_name: str
    run_index: int
    output_dir: str
    success: bool = False
    error: Optional[str] = None
    summary: RunSummary = Field(default_factory=RunSummary)


class AgentConsolidated(BaseModel):
    """Aggregated stats for one agent across N runs.

    Field order: identity → overall → breakdowns → quality → runs.
    overall includes pass_rate, latency, and token stats (flat, prefixed).
    quality is separate because it has its own composition (weighted metrics).
    """

    # Identity
    agent_name: str
    runs_requested: int
    runs_succeeded: int
    conversations_count: int
    # Aggregated stats (pass_rate + latency + tokens in one flat dict)
    overall: dict[str, Optional[float]] = {}
    # Breakdowns
    by_metric: dict[str, dict[str, Optional[float]]] = {}
    by_conversation: dict[str, dict[str, Optional[float]]] = {}
    # Quality score (separate — weighted aggregation with metric composition)
    quality_score: Optional[dict[str, Any]] = None
    # Per-run snapshots
    per_run: list[RunSummary] = []


class EvalReport(BaseModel):
    """Top-level evaluation report."""

    summary: dict[str, Any]
    agents: dict[str, AgentConsolidated]
    comparison: Optional[dict[str, Any]] = None

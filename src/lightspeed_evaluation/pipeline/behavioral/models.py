"""Models for NxM behavioral evaluation orchestration."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunContext:
    """Serializable context for a single evaluation run."""

    config_dict: dict[str, Any]
    eval_data_dicts: list[dict[str, Any]]
    default_agents: list[str]
    agent_name: str
    run_index: int
    run_output_dir: str
    extra: Optional[dict[str, Any]] = None


@dataclass
class RunResult:
    """Result metadata from a single evaluation run."""

    agent_name: str
    run_index: int
    output_dir: str
    success: bool = False
    error: Optional[str] = None
    summary: Optional[dict[str, Any]] = field(default_factory=dict)

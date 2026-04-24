"""Context passed to optional evaluation completion hooks (e.g. Langfuse)."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class EvaluationRunContext:
    """Metadata for a single evaluation run.

    Emitted to optional callbacks such as ``on_complete`` in :func:`evaluate` so
    integrations (Langfuse, custom dashboards) can label the run.
    """

    run_name: str
    original_data_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

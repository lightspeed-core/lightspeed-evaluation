"""Storage protocol interface for evaluation results.

:class:`BaseStorageBackend` is a :class:`~typing.Protocol` describing the pipeline
storage surface. Implementations inherit default no-op lifecycle hooks and must
implement :attr:`backend_name`.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol
from uuid import uuid4

from lightspeed_evaluation.core.models.data import EvaluationData, EvaluationResult


@dataclass
class RunInfo:
    """Information about an evaluation run.

    Attributes:
        run_id: Unique identifier for the evaluation run (auto-generated UUID).
        name: Human-readable name for the run.
        started_at: Timestamp when the run started (auto-generated).
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseStorageBackend(Protocol):
    """Protocol for storage backends with default no-op lifecycle hooks.

    Implementations must provide :attr:`backend_name`. Override any lifecycle
    method that should perform work (for example :class:`SQLStorageBackend`).
    """

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        ...  # pylint: disable=unnecessary-ellipsis  # Required for pyright

    def initialize(self, run_info: RunInfo) -> None:
        """Prepare the backend for a new evaluation run.

        Args:
            run_info: Information about the evaluation run.

        Raises:
            StorageError: If initialization fails.
        """

    def save_result(self, result: EvaluationResult) -> None:
        """Save a single evaluation result incrementally.

        Args:
            result: The evaluation result to save.

        Raises:
            StorageError: If saving fails.
        """

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Save all evaluation results in batch.

        Args:
            results: List of all evaluation results.

        Raises:
            StorageError: If saving fails.
        """

    def finalize(self) -> None:
        """Finalize the backend after evaluation completes.

        Raises:
            StorageError: If finalization fails.
        """

    def close(self) -> None:
        """Close the backend and release resources."""

    def set_evaluation_context(
        self, evaluation_data: Optional[list[EvaluationData]] = None
    ) -> None:
        """Optional full evaluation dataset for backends that need it at finalize."""
        _ = evaluation_data

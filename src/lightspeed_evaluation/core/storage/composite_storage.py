"""Multi-backend storage for the evaluation pipeline.

Uses :class:`BaseStorageBackend` from :mod:`protocol` for shared no-op defaults.
Each child receives the same lifecycle and optional evaluation context.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from lightspeed_evaluation.core.models.data import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.storage.protocol import BaseStorageBackend, RunInfo


class NoOpStorageBackend(BaseStorageBackend):
    """Backend that only supplies ``backend_name``; lifecycle methods are inherited no-ops."""

    def __init__(self, backend_name: str = "noop") -> None:
        """Initialize a no-op storage backend.

        Args:
            backend_name: Identifier for logging and composite naming (e.g. ``noop``,
                ``file``).
        """
        self._backend_name = backend_name

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        return self._backend_name


class CompositeStorageBackend(BaseStorageBackend):
    """Delegates to multiple backends in order."""

    def __init__(self, backends: Sequence[BaseStorageBackend]) -> None:
        """Initialize composite storage.

        Args:
            backends: Non-empty sequence of backends to receive the same lifecycle
                calls in list order.

        Raises:
            ValueError: If ``backends`` is empty.
        """
        if not backends:
            raise ValueError("CompositeStorageBackend requires at least one backend")
        self._backends = list(backends)

    @property
    def backend_name(self) -> str:
        """Return backend names joined with ``+`` (for diagnostics)."""
        return "+".join(b.backend_name for b in self._backends)

    def initialize(self, run_info: RunInfo) -> None:
        """Call ``initialize`` on every child backend."""
        for backend in self._backends:
            backend.initialize(run_info)

    def save_result(self, result: EvaluationResult) -> None:
        """Call ``save_result`` on every child backend."""
        for backend in self._backends:
            backend.save_result(result)

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Call ``save_run`` on every child backend."""
        for backend in self._backends:
            backend.save_run(results)

    def finalize(self) -> None:
        """Call ``finalize`` on every child backend."""
        for backend in self._backends:
            backend.finalize()

    def close(self) -> None:
        """Call ``close`` on every child backend."""
        for backend in self._backends:
            backend.close()

    def set_evaluation_context(
        self, evaluation_data: Optional[list[EvaluationData]] = None
    ) -> None:
        """Forward evaluation data to each child (e.g. file report backends)."""
        for backend in self._backends:
            backend.set_evaluation_context(evaluation_data)

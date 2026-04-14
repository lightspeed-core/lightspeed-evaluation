"""File storage backend: writes evaluation reports for one file config entry."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from lightspeed_evaluation.core.models.data import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.storage.config import FileBackendConfig
from lightspeed_evaluation.core.storage.protocol import BaseStorageBackend, RunInfo

if TYPE_CHECKING:
    from lightspeed_evaluation.core.models.system import SystemConfig

logger = logging.getLogger(__name__)


class FileStorageBackend(BaseStorageBackend):
    """Persists evaluation results to disk via :class:`OutputHandler` for one file entry."""

    def __init__(
        self,
        file_config: FileBackendConfig,
        system_config: SystemConfig,
        output_dir_override: Optional[str] = None,
    ) -> None:
        """Create a file backend for one ``storage`` list entry.

        Args:
            file_config: Output paths and report options for this backend.
            system_config: Full system configuration (for report content and graphs).
            output_dir_override: Optional CLI ``--output-dir`` overriding ``output_dir``.
        """
        self._file_config = file_config
        self._system_config = system_config
        self._output_dir_override = output_dir_override
        self._accumulated: list[EvaluationResult] = []
        self._evaluation_data: Optional[list[EvaluationData]] = None
        self._run_info: Optional[RunInfo] = None

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        return "file"

    def initialize(self, run_info: RunInfo) -> None:
        """Start a new run; clear accumulated results."""
        self._run_info = run_info
        self._accumulated.clear()

    def set_evaluation_context(
        self, evaluation_data: Optional[list[EvaluationData]] = None
    ) -> None:
        """Store full evaluation data for report generation (e.g. API token stats)."""
        self._evaluation_data = evaluation_data

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Accumulate batch results; reports are written in :meth:`finalize`."""
        self._accumulated.extend(results)

    def finalize(self) -> None:
        """Generate reports from accumulated results."""
        if not self._accumulated:
            logger.info(
                "File storage backend: no results to persist (run_id=%s)",
                self._run_info.run_id if self._run_info else "unknown",
            )
            return

        # Deferred import: generator pulls storage package; top-level OutputHandler
        # would circular-import storage during package startup.
        # pylint: disable-next=import-outside-toplevel
        from lightspeed_evaluation.core.output import OutputHandler

        output_dir = self._output_dir_override or self._file_config.output_dir
        output_handler = OutputHandler(
            output_dir=output_dir,
            base_filename=self._file_config.base_filename,
            system_config=self._system_config,
            file_config=self._file_config,
        )
        logger.info(
            "File storage backend: generating reports under %s",
            output_handler.output_dir,
        )
        output_handler.generate_reports(
            self._accumulated, evaluation_data=self._evaluation_data
        )

    def close(self) -> None:
        """Clear run state."""
        self._accumulated.clear()
        self._evaluation_data = None
        self._run_info = None

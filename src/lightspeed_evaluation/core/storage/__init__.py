"""Storage module for persisting evaluation results.

This module provides storage backends for saving evaluation results
to various destinations (databases). Uses a protocol-based design
for extensibility.

Example usage:
    from lightspeed_evaluation.core.storage import SQLStorageBackend, RunInfo

    # Create and initialize backend
    backend = SQLStorageBackend("sqlite:///./results.db")
    backend.initialize(RunInfo(name="my_evaluation"))

    # Save results incrementally
    for result in results:
        backend.save_result(result)

    # Finalize and close
    backend.finalize()
    backend.close()
"""

from lightspeed_evaluation.core.storage.exceptions import StorageError
from lightspeed_evaluation.core.storage.protocol import RunInfo, StorageProtocol
from lightspeed_evaluation.core.storage.sql_storage import (
    EvaluationResultDB,
    SQLStorageBackend,
)

__all__ = [
    "StorageProtocol",
    "RunInfo",
    "SQLStorageBackend",
    "EvaluationResultDB",
    "StorageError",
]

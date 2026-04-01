"""Storage module for persisting evaluation results.

This module provides storage backends for saving evaluation results
to various destinations (files, databases). Uses a protocol-based design
for extensibility.

Example usage:
    from lightspeed_evaluation.core.storage import (
        SQLStorageBackend,
        RunInfo,
        FileBackendConfig,
        get_file_config,
    )

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

from lightspeed_evaluation.core.storage.config import (
    DatabaseBackendConfig,
    FileBackendConfig,
    StorageBackendConfig,
)
from lightspeed_evaluation.core.storage.factory import (
    create_database_backend,
    get_database_config,
    get_file_config,
)
from lightspeed_evaluation.core.storage.protocol import RunInfo, StorageProtocol
from lightspeed_evaluation.core.storage.sql_storage import (
    EvaluationResultDB,
    SQLStorageBackend,
)
from lightspeed_evaluation.core.system.exceptions import StorageError

__all__ = [
    "StorageProtocol",
    "RunInfo",
    "SQLStorageBackend",
    "EvaluationResultDB",
    "StorageError",
    "FileBackendConfig",
    "DatabaseBackendConfig",
    "StorageBackendConfig",
    "create_database_backend",
    "get_database_config",
    "get_file_config",
]

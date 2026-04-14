"""Factory functions for creating storage backends from configuration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote_plus

from lightspeed_evaluation.core.storage.composite_storage import (
    CompositeStorageBackend,
    NoOpStorageBackend,
)
from lightspeed_evaluation.core.storage.config import (
    DatabaseBackendConfig,
    FileBackendConfig,
    StorageBackendConfig,
)
from lightspeed_evaluation.core.storage.file_storage import FileStorageBackend
from lightspeed_evaluation.core.storage.protocol import BaseStorageBackend
from lightspeed_evaluation.core.storage.sql_storage import SQLStorageBackend
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

if TYPE_CHECKING:
    from lightspeed_evaluation.core.models.system import SystemConfig

logger = logging.getLogger(__name__)

DEFAULT_PORTS: dict[str, int] = {
    "postgres": 5432,
    "mysql": 3306,
}


def _build_connection_url(config: DatabaseBackendConfig) -> str:
    """Build SQLAlchemy connection URL from config."""
    if config.type == "sqlite":
        return f"sqlite:///{config.database}"

    driver = "postgresql" if config.type == "postgres" else "mysql+pymysql"
    port = config.port or DEFAULT_PORTS[config.type]

    # URL-encode credentials to handle special characters
    user = quote_plus(config.user) if config.user else ""
    password = quote_plus(config.password) if config.password else ""
    host = quote_plus(config.host) if config.host else ""

    return f"{driver}://{user}:{password}@{host}:{port}/{config.database}"


def create_database_backend(
    config: StorageBackendConfig,
) -> Optional[SQLStorageBackend]:
    """Create a database storage backend from configuration."""
    if not isinstance(config, DatabaseBackendConfig):
        return None

    connection_url = _build_connection_url(config)
    backend = SQLStorageBackend(
        connection_url=connection_url,
        table_name=config.table_name,
        backend_name=config.type,
    )
    logger.debug("Created %s storage backend", config.type)
    return backend


def get_database_config(
    storage_configs: Sequence[StorageBackendConfig],
) -> Optional[DatabaseBackendConfig]:
    """Extract database configuration from storage configs list."""
    for config in storage_configs:
        if isinstance(config, DatabaseBackendConfig):
            return config
    return None


def get_file_config(
    storage_configs: Sequence[StorageBackendConfig],
) -> FileBackendConfig:
    """Extract file configuration from storage configs list.

    Returns the first FileBackendConfig found, or a default one if none exists.
    """
    for config in storage_configs:
        if isinstance(config, FileBackendConfig):
            return config
    return FileBackendConfig()


def create_pipeline_storage_backend(
    storage_configs: Sequence[StorageBackendConfig],
    *,
    system_config: Optional[SystemConfig] = None,
    output_dir_override: Optional[str] = None,
) -> BaseStorageBackend:
    """Build the storage backend used by the evaluation pipeline.

    Maps each configured backend to a protocol implementation: database
    entries become SQL backends; file entries become :class:`FileStorageBackend`
    (requires ``system_config``). Multiple backends are wrapped in a composite.

    Returns:
        A single ``BaseStorageBackend`` instance; never None.
    """
    backends: list[BaseStorageBackend] = []
    for config in storage_configs:
        if isinstance(config, DatabaseBackendConfig):
            db_backend = create_database_backend(config)
            if db_backend is not None:
                logger.info("Pipeline storage: database backend (%s)", config.type)
                backends.append(db_backend)
        elif isinstance(config, FileBackendConfig):
            if system_config is not None:
                logger.info(
                    "Pipeline storage: file backend (output_dir=%s)", config.output_dir
                )
                backends.append(
                    FileStorageBackend(
                        file_config=config,
                        system_config=system_config,
                        output_dir_override=output_dir_override,
                    )
                )
            else:
                raise ConfigurationError(
                    "File storage entries in ``storage`` require ``system_config`` "
                    "when building the pipeline storage backend."
                )
        else:
            raise ConfigurationError(
                f"Unknown storage backend type: {type(config).__name__!r}"
            )

    if not backends:
        return NoOpStorageBackend()
    if len(backends) == 1:
        return backends[0]
    return CompositeStorageBackend(backends)

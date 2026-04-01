"""Factory functions for creating storage backends from configuration."""

import logging
from collections.abc import Sequence
from typing import Optional
from urllib.parse import quote_plus

from lightspeed_evaluation.core.storage.config import (
    DatabaseBackendConfig,
    FileBackendConfig,
    StorageBackendConfig,
)
from lightspeed_evaluation.core.storage.sql_storage import SQLStorageBackend

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

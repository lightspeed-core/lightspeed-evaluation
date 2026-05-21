"""Tests for composite storage and pipeline storage factory."""

import pytest

from lightspeed_evaluation.core.models import LLMConfig, SystemConfig, EvaluationResult
from lightspeed_evaluation.core.storage import (
    BaseStorageBackend,
    CompositeStorageBackend,
    FileStorageBackend,
    NoOpStorageBackend,
    RunInfo,
    SQLStorageBackend,
    create_pipeline_storage_backend,
    get_langfuse_storage_config,
)
from lightspeed_evaluation.core.storage.config import (
    DatabaseBackendConfig,
    FileBackendConfig,
    LangfuseBackendConfig,
)
from lightspeed_evaluation.core.system.exceptions import ConfigurationError


def _minimal_system_config() -> SystemConfig:
    """Minimal valid SystemConfig for factory tests."""
    return SystemConfig(llm=LLMConfig(provider="openai", model="gpt-4o-mini"))


class TestCreatePipelineStorageBackend:
    """Tests for create_pipeline_storage_backend."""

    def test_empty_config_returns_noop(self) -> None:
        """No backends configured -> single no-op backend."""
        backend = create_pipeline_storage_backend([])
        assert isinstance(backend, NoOpStorageBackend)
        assert backend.backend_name == "noop"

    def test_file_only_without_system_config_raises(self) -> None:
        """File backend requires SystemConfig so misconfiguration fails fast."""
        with pytest.raises(ConfigurationError, match="File storage entries"):
            create_pipeline_storage_backend([FileBackendConfig()])

    def test_file_only_with_system_config_returns_file_backend(self) -> None:
        """File backend with SystemConfig uses FileStorageBackend in the pipeline."""
        backend = create_pipeline_storage_backend(
            [FileBackendConfig()],
            system_config=_minimal_system_config(),
        )
        assert isinstance(backend, FileStorageBackend)
        backend.close()

    def test_sqlite_returns_sql_backend(self) -> None:
        """Database config yields SQL storage implementation."""
        backend = create_pipeline_storage_backend(
            [DatabaseBackendConfig(type="sqlite", database=":memory:")]
        )
        assert isinstance(backend, SQLStorageBackend)
        backend.close()

    def test_file_and_sqlite_returns_composite(self) -> None:
        """Multiple backends are composed."""
        backend = create_pipeline_storage_backend(
            [
                FileBackendConfig(),
                DatabaseBackendConfig(type="sqlite", database=":memory:"),
            ],
            system_config=_minimal_system_config(),
        )
        assert isinstance(backend, CompositeStorageBackend)
        assert "file" in backend.backend_name
        assert "sqlite" in backend.backend_name
        backend.close()

    def test_langfuse_only_returns_noop(self) -> None:
        """Langfuse is not pipeline incremental storage -> no-op backend."""
        backend = create_pipeline_storage_backend(
            [LangfuseBackendConfig(host="https://langfuse.example")]
        )
        assert isinstance(backend, NoOpStorageBackend)

    def test_file_and_langfuse_returns_file_only(self) -> None:
        """Langfuse entry is skipped when composing pipeline storage."""
        backend = create_pipeline_storage_backend(
            [
                FileBackendConfig(),
                LangfuseBackendConfig(host="https://langfuse.example"),
            ],
            system_config=_minimal_system_config(),
        )
        assert isinstance(backend, FileStorageBackend)
        backend.close()

    def test_get_langfuse_storage_config_returns_first(self) -> None:
        """First langfuse row wins (same convention as file/database getters)."""
        lf1 = LangfuseBackendConfig(host="https://a.example.com")
        lf2 = LangfuseBackendConfig(host="https://b.example.com")
        assert get_langfuse_storage_config([lf1, lf2]) is lf1
        assert get_langfuse_storage_config([]) is None
        assert get_langfuse_storage_config([FileBackendConfig()]) is None


class TestCompositeStorageBackend:  # pylint: disable=too-few-public-methods
    """Tests for CompositeStorageBackend."""

    def test_delegates_lifecycle(self) -> None:
        """Initialize / finalize / close run on all children."""
        calls: list[str] = []

        class TrackingBackend(BaseStorageBackend):
            """Records selected lifecycle calls; other hooks use base no-ops."""

            @property
            def backend_name(self) -> str:
                """Return a fixed name."""
                return "track"

            def initialize(self, run_info: RunInfo) -> None:
                """Record initialize."""
                _ = run_info
                calls.append("init")

            def save_run(self, results: list[EvaluationResult]) -> None:
                """Record save_run."""
                _ = results
                calls.append("save_run")

            def finalize(self) -> None:
                """Record finalize."""
                calls.append("finalize")

            def close(self) -> None:
                """Record close."""
                calls.append("close")

        t1, t2 = TrackingBackend(), TrackingBackend()
        composite = CompositeStorageBackend([t1, t2])
        run = RunInfo(name="t")
        composite.initialize(run)
        composite.save_run([])
        composite.finalize()
        composite.close()

        assert calls == [
            "init",
            "init",
            "save_run",
            "save_run",
            "finalize",
            "finalize",
            "close",
            "close",
        ]

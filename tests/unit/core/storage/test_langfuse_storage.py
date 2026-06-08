# pylint: disable=protected-access
"""Tests for Langfuse storage backend."""

from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models.data import EvaluationResult
from lightspeed_evaluation.core.storage import create_pipeline_storage_backend
from lightspeed_evaluation.core.storage.config import LangfuseBackendConfig
from lightspeed_evaluation.core.storage.langfuse_storage import LangfuseStorageBackend
from lightspeed_evaluation.core.storage.protocol import RunInfo
from lightspeed_evaluation.core.system.loader import ConfigLoader

_RESULT_DEFAULTS: dict = {
    "conversation_group_id": "conv_1",
    "turn_id": "turn_1",
    "metric_identifier": "ragas:answer_relevancy",
    "result": "PASS",
    "score": 0.85,
    "reason": "Looks good",
    "query": "What is OpenShift?",
    "response": "OpenShift is a Kubernetes platform.",
}


def _make_result(**overrides: Any) -> EvaluationResult:
    """Build a minimal EvaluationResult for testing."""
    return EvaluationResult(**{**_RESULT_DEFAULTS, **overrides})


class TestLangfuseStorageBackend:
    """Unit tests for LangfuseStorageBackend."""

    def test_backend_name(self) -> None:
        """Backend name is 'langfuse'."""
        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        assert backend.backend_name == "langfuse"

    def test_save_run_accumulates_results(self) -> None:
        """save_run extends internal results list."""
        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend.save_run([_make_result(), _make_result()])
        assert len(backend._results) == 2

    def test_initialize_creates_client_with_config(self, mocker: MockerFixture) -> None:
        """initialize() creates a Langfuse client with explicit credentials."""
        mock_langfuse_cls = mocker.MagicMock()
        mock_module = mocker.MagicMock()
        mock_module.Langfuse = mock_langfuse_cls

        mocker.patch(
            "lightspeed_evaluation.core.storage.langfuse_storage._HAS_LANGFUSE",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.langfuse_storage.importlib.import_module",
            return_value=mock_module,
        )

        config = LangfuseBackendConfig(
            host="https://cloud.langfuse.com",
            public_key="pk-test",
            secret_key="sk-test",
        )
        backend = LangfuseStorageBackend(config)
        backend.initialize(RunInfo(name="test_run"))

        mock_langfuse_cls.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://cloud.langfuse.com",
        )
        assert backend._client is not None

    def test_initialize_logs_error_when_sdk_missing(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """initialize() logs error and sets client=None when langfuse not installed."""
        mocker.patch(
            "lightspeed_evaluation.core.storage.langfuse_storage._HAS_LANGFUSE",
            False,
        )

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        with caplog.at_level("ERROR"):
            backend.initialize(RunInfo(name="test"))

        assert "langfuse is not installed" in caplog.text
        assert backend._client is None

    def test_initialize_catches_client_error(self, mocker: MockerFixture) -> None:
        """initialize() catches client construction errors gracefully."""
        mock_module = mocker.MagicMock()
        mock_module.Langfuse.side_effect = ConnectionError("refused")

        mocker.patch(
            "lightspeed_evaluation.core.storage.langfuse_storage._HAS_LANGFUSE",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.langfuse_storage.importlib.import_module",
            return_value=mock_module,
        )

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend.initialize(RunInfo(name="test"))
        assert backend._client is None

    def test_finalize_creates_trace_and_scores(self, mocker: MockerFixture) -> None:
        """finalize() creates a trace span and scores via v4 create_score API."""
        mock_client = mocker.MagicMock()
        mock_span = mocker.MagicMock()
        mock_span.trace_id = "trace-abc-123"
        mock_client.start_as_current_observation.return_value.__enter__ = (
            mocker.MagicMock(return_value=mock_span)
        )
        mock_client.start_as_current_observation.return_value.__exit__ = (
            mocker.MagicMock(return_value=False)
        )

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="eval_run")
        backend._results = [
            _make_result(metric_identifier="ragas:relevancy", score=0.9),
            _make_result(metric_identifier="custom:accuracy", score=0.3, result="FAIL"),
        ]

        backend.finalize()

        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert call_kwargs["as_type"] == "span"
        assert "eval_run" in call_kwargs["name"]

        assert mock_client.create_score.call_count == 2
        first_score = mock_client.create_score.call_args_list[0].kwargs
        assert first_score["trace_id"] == "trace-abc-123"
        assert first_score["name"] == "ragas:relevancy"
        assert first_score["value"] == pytest.approx(0.9)
        assert first_score["data_type"] == "NUMERIC"

        mock_client.flush.assert_called_once()

    def test_finalize_skips_none_scores(self, mocker: MockerFixture) -> None:
        """finalize() skips results with score=None (ERROR/SKIPPED)."""
        mock_client = mocker.MagicMock()
        mock_span = mocker.MagicMock()
        mock_span.trace_id = "trace-xyz"
        mock_client.start_as_current_observation.return_value.__enter__ = (
            mocker.MagicMock(return_value=mock_span)
        )
        mock_client.start_as_current_observation.return_value.__exit__ = (
            mocker.MagicMock(return_value=False)
        )

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="test")
        backend._results = [
            _make_result(score=None, result="ERROR"),
            _make_result(score=0.8),
        ]

        backend.finalize()

        assert mock_client.create_score.call_count == 1

    def test_finalize_noop_when_no_client(self) -> None:
        """finalize() is a no-op when client failed to initialize."""
        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = None
        backend._results = [_make_result()]
        backend.finalize()

    def test_close_shuts_down_client(self, mocker: MockerFixture) -> None:
        """close() calls shutdown and sets client to None."""
        mock_client = mocker.MagicMock()
        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client

        backend.close()

        mock_client.shutdown.assert_called_once()
        assert backend._client is None


class TestLangfuseFactoryAndLoader:
    """Integration tests for factory and config loader."""

    def test_factory_creates_langfuse_backend(self) -> None:
        """create_pipeline_storage_backend handles LangfuseBackendConfig."""
        backend = create_pipeline_storage_backend([LangfuseBackendConfig()])
        assert isinstance(backend, LangfuseStorageBackend)
        backend.close()

    def test_loader_parses_langfuse_config(self) -> None:
        """ConfigLoader._parse_storage_config handles type='langfuse'."""
        loader = ConfigLoader()
        configs = loader._parse_storage_config(
            [{"type": "langfuse", "host": "https://cloud.langfuse.com"}]
        )
        assert len(configs) == 1
        assert isinstance(configs[0], LangfuseBackendConfig)
        assert configs[0].host == "https://cloud.langfuse.com"

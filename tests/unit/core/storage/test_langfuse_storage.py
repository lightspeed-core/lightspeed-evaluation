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
    "expected_response": "OpenShift is a Kubernetes platform.",
}


def _make_result(**overrides: Any) -> EvaluationResult:
    """Build a minimal EvaluationResult for testing."""
    return EvaluationResult(**{**_RESULT_DEFAULTS, **overrides})


def _wire_mock_span(mock_client: Any, mocker: MockerFixture, trace_id: str) -> None:
    """Attach a context-manager span mock to a Langfuse client mock."""
    mock_span = mocker.MagicMock()
    mock_span.trace_id = trace_id
    mock_client.start_as_current_observation.return_value.__enter__ = mocker.MagicMock(
        return_value=mock_span
    )
    mock_client.start_as_current_observation.return_value.__exit__ = mocker.MagicMock(
        return_value=False
    )


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
        _wire_mock_span(mock_client, mocker, "trace-abc-123")

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
        mock_client.create_dataset.assert_not_called()
        mock_client.create_dataset_item.assert_not_called()

    def test_finalize_skips_none_scores(self, mocker: MockerFixture) -> None:
        """finalize() skips results with score=None (ERROR/SKIPPED)."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-xyz")

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

    def test_finalize_scores_only_when_no_dataset_name(
        self, mocker: MockerFixture
    ) -> None:
        """Without dataset_name, only scores are exported."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-scores-only")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="scores_only")
        backend._results = [_make_result()]

        backend.finalize()

        assert mock_client.create_score.call_count == 1
        mock_client.get_dataset.assert_not_called()
        mock_client.create_dataset.assert_not_called()
        mock_client.create_dataset_item.assert_not_called()

    def test_finalize_blank_dataset_name_is_scores_only(
        self, mocker: MockerFixture
    ) -> None:
        """Blank/whitespace dataset_name is treated as unset (scores-only)."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-blank")

        config = LangfuseBackendConfig(dataset_name="   ")
        assert config.dataset_name is None

        backend = LangfuseStorageBackend(config)
        backend._client = mock_client
        backend._run_info = RunInfo(name="blank_ds")
        backend._results = [_make_result()]

        backend.finalize()

        mock_client.create_dataset_item.assert_not_called()
        assert mock_client.create_score.call_count == 1

    def test_finalize_exports_dataset_items_when_name_set(
        self, mocker: MockerFixture
    ) -> None:
        """With dataset_name, upserts turn-deduped items linked to the score trace."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-ds-123")
        mock_client.get_dataset.side_effect = RuntimeError("not found")

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="dataset_run")
        backend._results = [
            _make_result(metric_identifier="ragas:relevancy", score=0.9),
            _make_result(
                metric_identifier="custom:accuracy",
                score=0.3,
                result="FAIL",
            ),
            _make_result(
                conversation_group_id="conv_2",
                turn_id="turn_1",
                metric_identifier="ragas:relevancy",
                score=0.7,
                query="What is Kubernetes?",
            ),
        ]

        backend.finalize()

        assert mock_client.create_score.call_count == 3
        mock_client.create_dataset.assert_called_once_with(
            name="eval-baseline-v1",
            metadata={"source": "lightspeed-evaluation"},
        )
        # Two unique (conversation, turn) pairs despite three metric rows.
        assert mock_client.create_dataset_item.call_count == 2

        first_item = mock_client.create_dataset_item.call_args_list[0].kwargs
        assert first_item["dataset_name"] == "eval-baseline-v1"
        assert first_item["id"] == "eval-baseline-v1::conv_1::turn_1"
        assert first_item["source_trace_id"] == "trace-ds-123"
        assert first_item["input"]["query"] == "What is OpenShift?"
        assert first_item["expected_output"]["expected_response"] == (
            "OpenShift is a Kubernetes platform."
        )
        assert first_item["metadata"]["run_name"] == "dataset_run"
        assert first_item["metadata"]["response"] == (
            "OpenShift is a Kubernetes platform."
        )

        second_item = mock_client.create_dataset_item.call_args_list[1].kwargs
        assert second_item["id"] == "eval-baseline-v1::conv_2::turn_1"
        assert second_item["source_trace_id"] == "trace-ds-123"

        assert mock_client.flush.call_count == 2

    def test_finalize_reuses_existing_dataset(self, mocker: MockerFixture) -> None:
        """Existing dataset name is reused; items are still upserted."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-reuse")
        mock_client.get_dataset.return_value = mocker.MagicMock(name="existing")

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="reuse_run")
        backend._results = [_make_result()]

        backend.finalize()

        mock_client.get_dataset.assert_called_once_with("eval-baseline-v1")
        mock_client.create_dataset.assert_not_called()
        mock_client.create_dataset_item.assert_called_once()
        item = mock_client.create_dataset_item.call_args.kwargs
        assert item["id"] == "eval-baseline-v1::conv_1::turn_1"
        assert mock_client.create_score.call_count == 1

    def test_finalize_dataset_failure_does_not_abort(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dataset export failures are logged; scores still succeed."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-fail-ds")
        # get fails → create attempted → create/get race also fails; outer catch.
        mock_client.get_dataset.side_effect = ConnectionError("dataset down")
        mock_client.create_dataset.side_effect = ConnectionError("dataset down")

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-broken")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="fail_ds")
        backend._results = [_make_result()]

        with caplog.at_level("ERROR"):
            backend.finalize()

        assert mock_client.create_score.call_count == 1
        mock_client.create_dataset.assert_called_once()
        assert "failed to export dataset items" in caplog.text
        assert "eval-broken" in caplog.text

    def test_finalize_get_failure_creates_dataset(self, mocker: MockerFixture) -> None:
        """Any get_dataset failure falls through to create_dataset."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-nf")
        mock_client.get_dataset.side_effect = RuntimeError("missing")

        backend = LangfuseStorageBackend(LangfuseBackendConfig(dataset_name="eval-new"))
        backend._client = mock_client
        backend._run_info = RunInfo(name="nf_run")
        backend._results = [_make_result()]

        backend.finalize()

        mock_client.create_dataset.assert_called_once_with(
            name="eval-new",
            metadata={"source": "lightspeed-evaluation"},
        )
        assert mock_client.create_dataset_item.call_count == 1
        assert mock_client.create_score.call_count == 1


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

    def test_loader_parses_dataset_name(self) -> None:
        """ConfigLoader parses optional dataset_name onto LangfuseBackendConfig."""
        loader = ConfigLoader()
        configs = loader._parse_storage_config(
            [
                {
                    "type": "langfuse",
                    "host": "https://cloud.langfuse.com",
                    "dataset_name": "eval-baseline-v1",
                }
            ]
        )
        assert len(configs) == 1
        assert isinstance(configs[0], LangfuseBackendConfig)
        assert configs[0].dataset_name == "eval-baseline-v1"

    def test_dataset_name_blank_normalized_to_none(self) -> None:
        """Blank dataset_name normalizes to None on the config model."""
        config = LangfuseBackendConfig(dataset_name="")
        assert config.dataset_name is None

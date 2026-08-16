# pylint: disable=protected-access
"""Tests for Langfuse storage backend."""

from typing import Any

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models.data import EvaluationResult
from lightspeed_evaluation.core.storage import create_pipeline_storage_backend
from lightspeed_evaluation.core.storage.config import LangfuseBackendConfig
from lightspeed_evaluation.core.storage.langfuse_storage import (
    LangfuseStorageBackend,
    _compute_aggregate_scores,
    _is_missing_field_validation_error,
)
from lightspeed_evaluation.core.storage.protocol import RunInfo
from lightspeed_evaluation.core.system.loader import ConfigLoader


def _score_names(mock_client: Any) -> list[str]:
    """Return create_score name kwargs in call order."""
    return [c.kwargs["name"] for c in mock_client.create_score.call_args_list]


def _metric_score_count(mock_client: Any) -> int:
    """Count per-metric evaluation scores (exclude aggregate/status scores)."""
    return sum(
        1
        for name in _score_names(mock_client)
        if not name.startswith("aggregate/") and not name.startswith("eval/")
    )


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


def _wire_mock_span(mock_client: Any, mocker: MockerFixture, *trace_ids: str) -> Any:
    """Attach context-manager span mock(s) with distinct trace IDs per call."""
    ids = list(trace_ids) or ["trace-abc-123"]
    spans = []
    for trace_id in ids:
        span = mocker.MagicMock()
        span.trace_id = trace_id
        spans.append(span)

    # Advance through provided IDs; reuse the last ID if more spans are created.
    span_iter = iter(spans)

    def _enter(_cm: Any = None) -> Any:
        try:
            return next(span_iter)
        except StopIteration:
            return spans[-1]

    mock_client.start_as_current_observation.return_value.__enter__ = mocker.MagicMock(
        side_effect=_enter
    )
    mock_client.start_as_current_observation.return_value.__exit__ = mocker.MagicMock(
        return_value=False
    )
    return spans[0]


class TestLangfuseStorageBackend:
    """Unit tests for LangfuseStorageBackend."""

    def test_backend_name(self) -> None:
        """Backend name is 'langfuse'."""
        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        assert backend.backend_name == "langfuse"

    def test_save_run_accumulates_results(self) -> None:
        """save_run extends internal results list when client is unset."""
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
        """finalize() writes per-turn scores plus a run-level aggregate span."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-turn-1", "trace-agg-1")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="eval_run", run_id="abcd1234-5678")
        backend._results = [
            _make_result(metric_identifier="ragas:relevancy", score=0.9),
            _make_result(metric_identifier="custom:accuracy", score=0.3, result="FAIL"),
        ]

        backend.finalize()

        # One turn span + one aggregate span.
        assert mock_client.start_as_current_observation.call_count == 2
        turn_span = mock_client.start_as_current_observation.call_args_list[0].kwargs
        assert turn_span["as_type"] == "span"
        assert "conv_1" in turn_span["name"]
        agg_span = mock_client.start_as_current_observation.call_args_list[1].kwargs
        assert "aggregate" in agg_span["name"]
        assert agg_span["input"]["result_count"] == 2

        assert _metric_score_count(mock_client) == 2
        first_score = mock_client.create_score.call_args_list[0].kwargs
        assert first_score["trace_id"] == "trace-turn-1"
        assert first_score["name"] == "ragas:relevancy"
        assert first_score["value"] == pytest.approx(0.9)
        assert first_score["data_type"] == "NUMERIC"
        assert first_score["metadata"]["conversation_group_id"] == "conv_1"
        assert first_score["metadata"]["query"] == "What is OpenShift?"
        assert first_score["metadata"]["score"] == pytest.approx(0.9)

        score_names = _score_names(mock_client)
        assert "aggregate/mean_score" in score_names
        assert "aggregate/pass_rate" in score_names
        assert "aggregate/result_count" in score_names

        mock_client.flush.assert_called_once()
        mock_client.create_dataset.assert_not_called()
        mock_client.create_dataset_item.assert_not_called()
        mock_client.api.dataset_run_items.create.assert_not_called()

    def test_finalize_skips_none_scores(self, mocker: MockerFixture) -> None:
        """finalize() skips results with score=None (ERROR/SKIPPED)."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-xyz", "trace-agg")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="test")
        backend._results = [
            _make_result(score=None, result="ERROR"),
            _make_result(score=0.8),
        ]

        backend.finalize()

        assert _metric_score_count(mock_client) == 1

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

    def test_finalize_without_dataset_name_still_writes_turns(
        self, mocker: MockerFixture
    ) -> None:
        """Without dataset_name, per-turn traces and aggregates are still exported."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-turn", "trace-agg")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="no_dataset", run_id="aabbccdd-0001")
        backend._results = [_make_result()]

        backend.finalize()

        assert _metric_score_count(mock_client) == 1
        assert "aggregate/mean_score" in _score_names(mock_client)
        mock_client.get_dataset.assert_not_called()
        mock_client.create_dataset.assert_not_called()
        mock_client.create_dataset_item.assert_not_called()
        mock_client.api.dataset_run_items.create.assert_not_called()

    def test_finalize_blank_dataset_name_skips_dataset_link(
        self, mocker: MockerFixture
    ) -> None:
        """Blank/whitespace dataset_name skips Dataset linking only."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-blank", "trace-agg")

        config = LangfuseBackendConfig(dataset_name="   ")
        assert config.dataset_name is None

        backend = LangfuseStorageBackend(config)
        backend._client = mock_client
        backend._run_info = RunInfo(name="blank_ds", run_id="blank001-0001")
        backend._results = [_make_result()]

        backend.finalize()

        mock_client.create_dataset_item.assert_not_called()
        mock_client.api.dataset_run_items.create.assert_not_called()
        assert _metric_score_count(mock_client) == 1
        assert "aggregate/result_count" in _score_names(mock_client)

    def test_finalize_exports_dataset_run_when_name_set(
        self, mocker: MockerFixture
    ) -> None:
        """With dataset_name, links items + run items on top of per-turn traces."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(
            mock_client, mocker, "trace-turn-1", "trace-turn-2", "trace-agg"
        )
        mock_client.get_dataset.side_effect = RuntimeError("not found")

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="dataset_run", run_id="deadbeef-0001")
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

        mock_client.create_dataset.assert_called_once_with(
            name="eval-baseline-v1",
            metadata={"source": "lightspeed-evaluation"},
        )
        # Two turn spans + one aggregate span.
        assert mock_client.start_as_current_observation.call_count == 3
        assert mock_client.create_dataset_item.call_count == 2
        assert mock_client.api.dataset_run_items.create.call_count == 2
        assert _metric_score_count(mock_client) == 3
        assert "aggregate/mean_score" in _score_names(mock_client)

        first_item = mock_client.create_dataset_item.call_args_list[0].kwargs
        assert first_item["dataset_name"] == "eval-baseline-v1"
        assert first_item["id"] == "eval-baseline-v1::conv_1::turn_1"
        assert first_item["source_trace_id"] == "trace-turn-1"
        assert first_item["input"]["query"] == "What is OpenShift?"
        assert first_item["expected_output"]["expected_response"] == (
            "OpenShift is a Kubernetes platform."
        )
        assert first_item["metadata"]["source"] == "lightspeed-evaluation"
        assert "response" not in first_item["metadata"]

        first_run_item = mock_client.api.dataset_run_items.create.call_args_list[
            0
        ].kwargs
        assert first_run_item["run_name"] == "dataset_run__deadbeef"
        assert first_run_item["dataset_item_id"] == "eval-baseline-v1::conv_1::turn_1"
        assert first_run_item["trace_id"] == "trace-turn-1"

        second_run_item = mock_client.api.dataset_run_items.create.call_args_list[
            1
        ].kwargs
        assert second_run_item["trace_id"] == "trace-turn-2"
        assert second_run_item["trace_id"] != first_run_item["trace_id"]

        second_item = mock_client.create_dataset_item.call_args_list[1].kwargs
        assert second_item["id"] == "eval-baseline-v1::conv_2::turn_1"
        assert second_item["source_trace_id"] == "trace-turn-2"

        turn_span = mock_client.start_as_current_observation.call_args_list[0].kwargs
        assert turn_span["input"]["conversation_group_id"] == "conv_1"
        assert turn_span["output"]["response"] == (
            "OpenShift is a Kubernetes platform."
        )
        assert "csv_rows" in turn_span["metadata"]
        assert len(turn_span["metadata"]["csv_rows"]) == 2
        assert turn_span["metadata"]["csv_rows"][0]["metric_identifier"] == (
            "ragas:relevancy"
        )
        assert "query" in turn_span["metadata"]["csv_turn"]
        assert "metric_identifier" not in turn_span["metadata"]["csv_turn"]

        mock_client.flush.assert_called()

    def test_finalize_reuses_existing_dataset(self, mocker: MockerFixture) -> None:
        """Existing dataset name is reused; run items are still created."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-reuse", "trace-agg")
        mock_client.get_dataset.return_value = mocker.MagicMock(name="existing")

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="reuse_run", run_id="aabbccdd-9999")
        backend._results = [_make_result()]

        backend.finalize()

        mock_client.get_dataset.assert_called_once_with("eval-baseline-v1")
        mock_client.create_dataset.assert_not_called()
        mock_client.create_dataset_item.assert_called_once()
        mock_client.api.dataset_run_items.create.assert_called_once()
        item = mock_client.create_dataset_item.call_args.kwargs
        assert item["id"] == "eval-baseline-v1::conv_1::turn_1"
        assert _metric_score_count(mock_client) == 1
        assert "aggregate/mean_score" in _score_names(mock_client)

    def test_finalize_dataset_ensure_failure_keeps_turn_export(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dataset ensure failure still writes per-turn scores and aggregates."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-fail-ds", "trace-agg")
        mock_client.get_dataset.side_effect = ConnectionError("dataset down")
        mock_client.create_dataset.side_effect = ConnectionError("dataset down")

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-broken")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="fail_ds", run_id="fail0001-0001")
        backend._results = [_make_result()]

        with caplog.at_level("ERROR"):
            backend.finalize()

        assert "failed to ensure dataset" in caplog.text
        assert "eval-broken" in caplog.text
        assert _metric_score_count(mock_client) == 1
        assert "aggregate/mean_score" in _score_names(mock_client)
        mock_client.api.dataset_run_items.create.assert_not_called()

    def test_finalize_partial_dataset_link_continues_turns(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Mid-run dataset link failure continues other turns and aggregates."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-t1", "trace-t2", "trace-agg")
        mock_client.get_dataset.return_value = mocker.MagicMock(name="existing")
        mock_client.create_dataset_item.side_effect = [
            None,
            RuntimeError("dataset item create failed"),
        ]

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-partial")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="partial_run", run_id="partial01-0001")
        backend._results = [
            _make_result(conversation_group_id="conv_1", turn_id="turn_1"),
            _make_result(conversation_group_id="conv_2", turn_id="turn_1"),
        ]

        with caplog.at_level("ERROR"):
            backend.finalize()

        assert "dataset link failed" in caplog.text
        # Two turn metric scores + aggregates; no duplicated metric pass.
        assert _metric_score_count(mock_client) == 2
        assert "aggregate/mean_score" in _score_names(mock_client)
        assert mock_client.api.dataset_run_items.create.call_count == 1

    def test_finalize_get_failure_creates_dataset(self, mocker: MockerFixture) -> None:
        """Any get_dataset failure falls through to create_dataset."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-nf", "trace-agg")
        mock_client.get_dataset.side_effect = RuntimeError("missing")

        backend = LangfuseStorageBackend(LangfuseBackendConfig(dataset_name="eval-new"))
        backend._client = mock_client
        backend._run_info = RunInfo(name="nf_run", run_id="11223344-5566")
        backend._results = [_make_result()]

        backend.finalize()

        mock_client.create_dataset.assert_called_once_with(
            name="eval-new",
            metadata={"source": "lightspeed-evaluation"},
        )
        assert mock_client.create_dataset_item.call_count == 1
        assert mock_client.api.dataset_run_items.create.call_count == 1
        assert _metric_score_count(mock_client) == 1

    def test_finalize_tolerates_media_references_parse_error(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SDK/server skew on media_references must not block dataset runs."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-media", "trace-agg")
        mock_client.get_dataset.return_value = mocker.MagicMock(name="existing")
        mock_client.create_dataset_item.side_effect = (
            ValidationError.from_exception_data(
                "DatasetItem",
                [
                    {
                        "type": "missing",
                        "loc": ("media_references",),
                        "input": {},
                    }
                ],
            )
        )

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="compat_run", run_id="cafebabe-0001")
        backend._results = [_make_result()]

        with caplog.at_level("WARNING"):
            backend.finalize()

        assert "media_references" in caplog.text
        mock_client.api.dataset_run_items.create.assert_called_once()
        assert _metric_score_count(mock_client) == 1
        mock_client.flush.assert_called()

    def test_finalize_does_not_treat_payload_validation_as_skew(
        self, mocker: MockerFixture
    ) -> None:
        """Non-missing ValidationError on media_references must not look like success."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-payload", "trace-agg")
        mock_client.get_dataset.return_value = mocker.MagicMock(name="existing")
        mock_client.create_dataset_item.side_effect = (
            ValidationError.from_exception_data(
                "DatasetItem",
                [
                    {
                        "type": "list_type",
                        "loc": ("media_references",),
                        "input": "bad",
                    }
                ],
            )
        )

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="payload_run", run_id="badf00d0-0001")
        backend._results = [_make_result()]

        backend.finalize()

        # Upsert failed for real; do not create a run item against a missing item.
        mock_client.api.dataset_run_items.create.assert_not_called()
        # Per-turn + aggregate export still completes.
        assert _metric_score_count(mock_client) == 1
        assert "aggregate/mean_score" in _score_names(mock_client)

    def test_finalize_continues_after_run_item_payload_validation(
        self, mocker: MockerFixture
    ) -> None:
        """Non-skew run-item ValidationError is logged; export continues."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-run-item", "trace-agg")
        mock_client.get_dataset.return_value = mocker.MagicMock(name="existing")
        mock_client.api.dataset_run_items.create.side_effect = (
            ValidationError.from_exception_data(
                "DatasetRunItem",
                [
                    {
                        "type": "string_type",
                        "loc": ("trace_id",),
                        "input": 123,
                    }
                ],
            )
        )

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="run_item_bad", run_id="feedface-0001")
        backend._results = [_make_result()]

        backend.finalize()

        mock_client.create_dataset_item.assert_called_once()
        assert mock_client.api.dataset_run_items.create.call_count == 1
        assert _metric_score_count(mock_client) == 1
        assert "aggregate/mean_score" in _score_names(mock_client)

    def test_finalize_tolerates_run_item_media_references_skew(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing media_references on run-item response is treated as skew."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-run-skew", "trace-agg")
        mock_client.get_dataset.return_value = mocker.MagicMock(name="existing")
        mock_client.api.dataset_run_items.create.side_effect = (
            ValidationError.from_exception_data(
                "DatasetRunItem",
                [
                    {
                        "type": "missing",
                        "loc": ("media_references",),
                        "input": {},
                    }
                ],
            )
        )

        backend = LangfuseStorageBackend(
            LangfuseBackendConfig(dataset_name="eval-baseline-v1")
        )
        backend._client = mock_client
        backend._run_info = RunInfo(name="run_item_skew", run_id="0ddba11f-0001")
        backend._results = [_make_result()]

        with caplog.at_level("WARNING"):
            backend.finalize()

        assert "media_references" in caplog.text
        mock_client.flush.assert_called()
        assert _metric_score_count(mock_client) == 1
        assert "aggregate/mean_score" in _score_names(mock_client)


class TestLangfuseStorageHelpers:
    """Unit tests for Langfuse storage helpers and export resilience."""

    def test_save_run_exports_turns_incrementally(self, mocker: MockerFixture) -> None:
        """save_run writes per-turn traces immediately (before finalize)."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-incremental")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="inc_run", run_id="inc00001-0001")

        backend.save_run([_make_result()])

        assert mock_client.start_as_current_observation.call_count == 1
        assert _metric_score_count(mock_client) == 1
        mock_client.create_dataset_item.assert_not_called()
        assert "aggregate/mean_score" not in _score_names(mock_client)
        assert "eval/success" not in _score_names(mock_client)

    def test_finalize_records_eval_status(self, mocker: MockerFixture) -> None:
        """finalize(success=...) records eval_status and eval/success score."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-turn", "trace-agg")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="status_run", run_id="status01-0001")
        backend._results = [_make_result()]

        backend.finalize(success=False)

        agg_span = mock_client.start_as_current_observation.call_args_list[-1].kwargs
        assert agg_span["metadata"]["eval_status"] == "failed"
        assert "eval/success" in _score_names(mock_client)
        success_score = next(
            c.kwargs
            for c in mock_client.create_score.call_args_list
            if c.kwargs["name"] == "eval/success"
        )
        assert success_score["value"] == pytest.approx(0.0)

    def test_finalize_turn_trace_failure_still_writes_aggregates(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A later turn-trace failure must not skip the aggregate summary."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-t1", "trace-agg")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="turn_fail", run_id="turnfail-0001")
        backend._results = [
            _make_result(conversation_group_id="conv_1", turn_id="turn_1"),
            _make_result(conversation_group_id="conv_2", turn_id="turn_1"),
        ]

        original = backend._write_turn_trace_and_scores
        call_count = {"n": 0}

        def _flaky_turn(**kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ConnectionError("turn span failed")
            return original(**kwargs)

        mocker.patch.object(
            backend, "_write_turn_trace_and_scores", side_effect=_flaky_turn
        )

        with caplog.at_level("ERROR"):
            backend.finalize()

        assert "turn export failed" in caplog.text
        assert _metric_score_count(mock_client) == 1
        assert "aggregate/mean_score" in _score_names(mock_client)
        assert mock_client.start_as_current_observation.call_count >= 2

    def test_finalize_late_metric_appends_to_existing_trace(
        self, mocker: MockerFixture
    ) -> None:
        """Late metric rows for an exported turn append scores to that trace."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-turn-1", "trace-agg")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="late_metric", run_id="late-0001")

        first = _make_result(metric_identifier="ragas:answer_relevancy", score=0.8)
        backend.save_run([first])
        backend.save_result(
            _make_result(metric_identifier="ragas:faithfulness", score=0.7)
        )
        backend.finalize()

        assert _metric_score_count(mock_client) == 2
        metric_calls = [
            c
            for c in mock_client.create_score.call_args_list
            if not c.kwargs["name"].startswith("aggregate/")
            and not c.kwargs["name"].startswith("eval/")
        ]
        assert {c.kwargs["name"] for c in metric_calls} == {
            "ragas:answer_relevancy",
            "ragas:faithfulness",
        }
        assert {c.kwargs["trace_id"] for c in metric_calls} == {"trace-turn-1"}
        # One turn span + one aggregate span (no second turn span for late metric).
        assert mock_client.start_as_current_observation.call_count == 2

    def test_finalize_aggregate_failure_still_flushes(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Aggregate write errors must not skip the final SDK flush."""
        mock_client = mocker.MagicMock()
        _wire_mock_span(mock_client, mocker, "trace-t1")

        backend = LangfuseStorageBackend(LangfuseBackendConfig())
        backend._client = mock_client
        backend._run_info = RunInfo(name="agg_fail", run_id="aggfail-0001")
        backend._results = [_make_result()]

        mocker.patch.object(
            backend,
            "_write_aggregate_summary",
            side_effect=ConnectionError("aggregate failed"),
        )

        with caplog.at_level("ERROR"):
            backend.finalize()

        assert "failed to write aggregate summary" in caplog.text
        mock_client.flush.assert_called_once()

    def test_matches_structured_missing_field(self) -> None:
        """Only type=missing with field in loc is treated as skew."""
        exc = ValidationError.from_exception_data(
            "DatasetItem",
            [{"type": "missing", "loc": ("media_references",), "input": {}}],
        )
        assert _is_missing_field_validation_error(exc, "media_references") is True

    def test_rejects_non_missing_errors_mentioning_field(self) -> None:
        """Payload/type errors must not match via substring of the message."""
        exc = ValidationError.from_exception_data(
            "DatasetItem",
            [{"type": "list_type", "loc": ("media_references",), "input": "x"}],
        )
        assert _is_missing_field_validation_error(exc, "media_references") is False
        assert "media_references" in str(exc)

    def test_compute_aggregate_scores_overall_and_per_metric(self) -> None:
        """Aggregates include overall and per-metric mean/pass rate."""
        results = [
            _make_result(metric_identifier="ragas:relevancy", score=0.9, result="PASS"),
            _make_result(metric_identifier="custom:accuracy", score=0.3, result="FAIL"),
        ]
        aggregates = _compute_aggregate_scores(results)
        assert aggregates["aggregate/result_count"] == pytest.approx(2.0)
        assert aggregates["aggregate/mean_score"] == pytest.approx(0.6)
        assert aggregates["aggregate/pass_rate"] == pytest.approx(0.5)
        assert aggregates["aggregate/mean_score/ragas:relevancy"] == pytest.approx(0.9)
        assert aggregates["aggregate/pass_rate/custom:accuracy"] == pytest.approx(0.0)

    def test_compute_aggregate_scores_skips_missing_numeric_scores(self) -> None:
        """ERROR/SKIPPED rows without scores still count toward result_count."""
        results = [
            _make_result(score=None, result="ERROR"),
            _make_result(score=0.8, result="PASS"),
        ]
        aggregates = _compute_aggregate_scores(results)
        assert aggregates["aggregate/result_count"] == pytest.approx(2.0)
        assert aggregates["aggregate/mean_score"] == pytest.approx(0.8)
        assert aggregates["aggregate/pass_rate"] == pytest.approx(1.0)


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

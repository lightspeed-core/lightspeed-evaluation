# pylint: disable=protected-access
"""Tests for MLflow storage backend."""

from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import LLMConfig, SystemConfig
from lightspeed_evaluation.core.models.data import EvaluationResult, JudgeScore
from lightspeed_evaluation.core.storage import create_pipeline_storage_backend
from lightspeed_evaluation.core.storage.config import MLflowBackendConfig
from lightspeed_evaluation.core.storage.mlflow_storage import (
    MLflowStorageBackend,
    _registry_model_name,
)
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


def _create_initialized_backend(mocker: MockerFixture) -> MLflowStorageBackend:
    """Create an MLflow backend with a mocked client ready for use."""
    mock_client = mocker.MagicMock()
    mock_client.start_run.return_value = mocker.MagicMock()

    backend = MLflowStorageBackend(MLflowBackendConfig())
    backend._client = mock_client
    backend._run = mocker.MagicMock()
    backend._run_info = RunInfo(name="test_run")
    return backend


class TestMLflowStorageBackend:
    """Unit tests for MLflowStorageBackend."""

    def test_backend_name(self) -> None:
        """Backend name is 'mlflow'."""
        backend = MLflowStorageBackend(MLflowBackendConfig())
        assert backend.backend_name == "mlflow"

    def test_results_count_starts_at_zero(self) -> None:
        """results_count is 0 before any results are saved."""
        backend = MLflowStorageBackend(MLflowBackendConfig())
        assert backend.results_count == 0

    def test_initialize_starts_mlflow_run(self, mocker: MockerFixture) -> None:
        """initialize() sets experiment and starts an MLflow run."""
        mock_mlflow = mocker.MagicMock()
        mock_run = mocker.MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            return_value=mock_mlflow,
        )

        config = MLflowBackendConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
        )
        backend = MLflowStorageBackend(config)
        backend.initialize(RunInfo(name="test_run"))

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.set_tag.assert_called_with("eval_status", "running")
        assert backend._run is not None
        assert backend._client is not None

    def test_initialize_logs_run_params(self, mocker: MockerFixture) -> None:
        """initialize() logs run_name and eval_run_id as params."""
        mock_mlflow = mocker.MagicMock()
        mock_mlflow.start_run.return_value = mocker.MagicMock()

        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            return_value=mock_mlflow,
        )

        run_info = RunInfo(name="my_eval")
        backend = MLflowStorageBackend(MLflowBackendConfig())
        backend.initialize(run_info)

        params = mock_mlflow.log_params.call_args[0][0]
        assert params["run_name"] == "my_eval"
        assert params["eval_run_id"] == run_info.run_id

    def test_initialize_skips_tracking_uri_when_none(
        self, mocker: MockerFixture
    ) -> None:
        """initialize() does not call set_tracking_uri when URI is None."""
        mock_mlflow = mocker.MagicMock()
        mock_mlflow.start_run.return_value = mocker.MagicMock()

        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            return_value=mock_mlflow,
        )

        config = MLflowBackendConfig(tracking_uri=None)
        backend = MLflowStorageBackend(config)
        backend.initialize(RunInfo(name="test"))

        mock_mlflow.set_tracking_uri.assert_not_called()

    def test_initialize_logs_error_when_sdk_missing(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """initialize() logs error and sets client=None when mlflow not installed."""
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            False,
        )

        backend = MLflowStorageBackend(MLflowBackendConfig())
        with caplog.at_level("ERROR"):
            backend.initialize(RunInfo(name="test"))

        assert "mlflow is not installed" in caplog.text
        assert backend._client is None

    def test_initialize_catches_client_error(self, mocker: MockerFixture) -> None:
        """initialize() catches MLflow connection errors gracefully."""
        mock_mlflow = mocker.MagicMock()
        mock_mlflow.start_run.side_effect = ConnectionError("refused")

        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            return_value=mock_mlflow,
        )

        backend = MLflowStorageBackend(MLflowBackendConfig())
        backend.initialize(RunInfo(name="test"))
        assert backend._run is None
        assert backend._client is None

    def test_close_ends_run_as_failed(self, mocker: MockerFixture) -> None:
        """close() without finalize ends the run as FAILED."""
        mock_client = mocker.MagicMock()
        backend = MLflowStorageBackend(MLflowBackendConfig())
        backend._client = mock_client
        backend._run = mocker.MagicMock()

        backend.close()

        mock_client.set_tag.assert_called_with("eval_status", "failed")
        mock_client.end_run.assert_called_once_with(status="FAILED")
        assert backend._run is None

    def test_close_noop_when_no_run(self) -> None:
        """close() is a no-op when no run is active."""
        backend = MLflowStorageBackend(MLflowBackendConfig())
        backend._client = None
        backend._run = None
        backend.close()


class TestMLflowMetricLogging:
    """Tests for incremental metric logging behavior."""

    def test_save_result_logs_metrics_immediately(self, mocker: MockerFixture) -> None:
        """save_result() logs metrics to MLflow immediately with step."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(_make_result(score=0.9))

        backend._client.log_metrics.assert_called_once()
        call_kwargs = backend._client.log_metrics.call_args
        metrics = call_kwargs[0][0]
        assert metrics["score/ragas_answer_relevancy"] == pytest.approx(0.9)
        assert metrics["pass/ragas_answer_relevancy"] == 1.0
        assert call_kwargs[1]["step"] == 0
        assert backend.results_count == 1

    def test_save_result_increments_step(self, mocker: MockerFixture) -> None:
        """save_result() increments step for each result."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(_make_result())
        backend.save_result(_make_result())
        backend.save_result(_make_result())

        assert backend.results_count == 3
        calls = backend._client.log_metrics.call_args_list
        assert calls[0][1]["step"] == 0
        assert calls[1][1]["step"] == 1
        assert calls[2][1]["step"] == 2

    def test_save_run_logs_each_result_immediately(self, mocker: MockerFixture) -> None:
        """save_run() logs each result in the batch immediately."""
        backend = _create_initialized_backend(mocker)

        results = [_make_result(score=0.8), _make_result(score=0.6, result="FAIL")]
        backend.save_run(results)

        assert backend._client.log_metrics.call_count == 2
        assert backend.results_count == 2

        first_call = backend._client.log_metrics.call_args_list[0]
        assert first_call[0][0]["score/ragas_answer_relevancy"] == pytest.approx(0.8)
        assert first_call[0][0]["pass/ragas_answer_relevancy"] == 1.0
        assert first_call[1]["step"] == 0

        second_call = backend._client.log_metrics.call_args_list[1]
        assert second_call[0][0]["score/ragas_answer_relevancy"] == pytest.approx(0.6)
        assert second_call[0][0]["pass/ragas_answer_relevancy"] == 0.0
        assert second_call[1]["step"] == 1

    def test_save_result_skips_none_score(self, mocker: MockerFixture) -> None:
        """save_result() skips score metric when score is None."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(_make_result(score=None, result="ERROR"))

        metrics = backend._client.log_metrics.call_args[0][0]
        assert "score/ragas_answer_relevancy" not in metrics
        assert metrics["pass/ragas_answer_relevancy"] == 0.0

    def test_save_result_noop_when_no_client(self) -> None:
        """save_result() is a no-op when client failed to initialize."""
        backend = MLflowStorageBackend(MLflowBackendConfig())
        backend._client = None
        backend._run = None
        backend.save_result(_make_result())
        assert backend.results_count == 0

    def test_save_result_logs_token_metrics(self, mocker: MockerFixture) -> None:
        """save_result() logs token usage as MLflow metrics with metric_id."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(
                api_input_tokens=100,
                api_output_tokens=50,
                judge_llm_input_tokens=200,
                judge_llm_output_tokens=80,
                embedding_tokens=30,
            )
        )

        metrics = backend._client.log_metrics.call_args[0][0]
        assert metrics["tokens/api_input/ragas_answer_relevancy"] == 100.0
        assert metrics["tokens/api_output/ragas_answer_relevancy"] == 50.0
        assert metrics["tokens/judge_input/ragas_answer_relevancy"] == 200.0
        assert metrics["tokens/judge_output/ragas_answer_relevancy"] == 80.0
        assert metrics["tokens/embedding/ragas_answer_relevancy"] == 30.0

    def test_save_result_logs_latency_metrics(self, mocker: MockerFixture) -> None:
        """save_result() logs latency as MLflow metrics with metric_id."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(evaluation_latency=1.5, execution_time=2.0, agent_latency=0.8)
        )

        metrics = backend._client.log_metrics.call_args[0][0]
        assert metrics["latency/ragas_answer_relevancy"] == pytest.approx(1.5)
        assert metrics["execution_time/ragas_answer_relevancy"] == pytest.approx(2.0)
        assert metrics["agent_latency/ragas_answer_relevancy"] == pytest.approx(0.8)

    def test_save_result_logs_judge_panel_scores(self, mocker: MockerFixture) -> None:
        """save_result() preserves per-judge scores as MLflow metrics."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(
                judge_scores=[
                    JudgeScore(judge_id="gpt-4o-mini", score=0.9, reason="good"),
                    JudgeScore(judge_id="gpt-4.1-mini", score=0.8, reason="ok"),
                ],
            )
        )

        metrics = backend._client.log_metrics.call_args[0][0]
        assert metrics["judge/gpt-4o-mini/ragas_answer_relevancy"] == pytest.approx(0.9)
        assert metrics["judge/gpt-4.1-mini/ragas_answer_relevancy"] == pytest.approx(
            0.8
        )

    def test_save_result_logs_streaming_metrics(self, mocker: MockerFixture) -> None:
        """save_result() logs streaming performance metrics with metric_id."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(
                time_to_first_token=0.25,
                streaming_duration=3.5,
                tokens_per_second=42.0,
            )
        )

        metrics = backend._client.log_metrics.call_args[0][0]
        assert metrics["time_to_first_token/ragas_answer_relevancy"] == pytest.approx(
            0.25
        )
        assert metrics["streaming_duration/ragas_answer_relevancy"] == pytest.approx(
            3.5
        )
        assert metrics["tokens_per_second/ragas_answer_relevancy"] == pytest.approx(
            42.0
        )

    def test_finalize_logs_aggregate_metrics(self, mocker: MockerFixture) -> None:
        """finalize() computes and logs aggregate metrics."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(score=0.8, result="PASS", evaluation_latency=1.0)
        )
        backend.save_result(
            _make_result(score=0.4, result="FAIL", evaluation_latency=2.0)
        )

        backend._client.log_metrics.reset_mock()
        backend.finalize()

        aggregate_call = backend._client.log_metrics.call_args[0][0]
        assert aggregate_call["aggregate/mean_score"] == pytest.approx(0.6)
        assert aggregate_call["aggregate/pass_rate"] == pytest.approx(0.5)
        assert aggregate_call["aggregate/mean_latency"] == pytest.approx(1.5)
        assert aggregate_call["aggregate/result_count"] == 2.0

    def test_finalize_logs_aggregate_token_totals(self, mocker: MockerFixture) -> None:
        """finalize() logs total token counts including embedding as aggregates."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(
                api_input_tokens=100, api_output_tokens=50, embedding_tokens=20
            )
        )
        backend.save_result(
            _make_result(
                api_input_tokens=200, api_output_tokens=100, embedding_tokens=30
            )
        )

        backend._client.log_metrics.reset_mock()
        backend.finalize()

        aggregate_call = backend._client.log_metrics.call_args[0][0]
        assert aggregate_call["aggregate/total_api_input_tokens"] == 300.0
        assert aggregate_call["aggregate/total_api_output_tokens"] == 150.0
        assert aggregate_call["aggregate/total_embedding_tokens"] == 50.0

    def test_finalize_ends_run_as_finished(self, mocker: MockerFixture) -> None:
        """finalize(success=True) ends the run as FINISHED with complete tag."""
        backend = _create_initialized_backend(mocker)
        backend.save_result(_make_result())

        backend.finalize(success=True)

        backend._client.set_tag.assert_called_with("eval_status", "complete")
        backend._client.end_run.assert_called_once_with(status="FINISHED")
        assert backend._run is None

    def test_finalize_ends_run_as_failed(self, mocker: MockerFixture) -> None:
        """finalize(success=False) ends the run as FAILED with failed tag."""
        backend = _create_initialized_backend(mocker)
        backend.save_result(_make_result())

        backend.finalize(success=False)

        backend._client.set_tag.assert_called_with("eval_status", "failed")
        backend._client.end_run.assert_called_once_with(status="FAILED")
        assert backend._run is None

    def test_finalize_failed_still_logs_aggregates(self, mocker: MockerFixture) -> None:
        """finalize(success=False) still writes aggregates from incremental data."""
        backend = _create_initialized_backend(mocker)
        backend.save_result(_make_result(score=0.8, result="PASS"))
        backend._client.log_metrics.reset_mock()

        backend.finalize(success=False)

        aggregate_call = backend._client.log_metrics.call_args[0][0]
        assert aggregate_call["aggregate/mean_score"] == pytest.approx(0.8)
        assert aggregate_call["aggregate/result_count"] == 1.0

    def test_finalize_noop_when_no_client(self) -> None:
        """finalize() is a no-op when client failed to initialize."""
        backend = MLflowStorageBackend(MLflowBackendConfig())
        backend._client = None
        backend._run = None
        backend.finalize()

    def test_finalize_skips_aggregates_with_no_results(
        self, mocker: MockerFixture
    ) -> None:
        """finalize() skips aggregate logging when no results were saved."""
        backend = _create_initialized_backend(mocker)

        backend.finalize()

        backend._client.end_run.assert_called_once_with(status="FINISHED")
        backend._client.log_metrics.assert_not_called()

    def test_finalize_registers_models_as_provider_gt_model(
        self, mocker: MockerFixture
    ) -> None:
        """finalize() registers judge and embedding as provider>model."""
        mock_pyfunc = mocker.MagicMock()
        mock_pyfunc.PythonModel = type("PythonModel", (), {})
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            side_effect=lambda name: (
                mock_pyfunc if name == "mlflow.pyfunc" else mocker.MagicMock()
            ),
        )

        system_config = SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini")
        )
        backend = _create_initialized_backend(mocker)
        backend._system_config = system_config
        backend.save_result(_make_result())

        backend.finalize(success=True)

        assert backend._client.pyfunc.log_model.call_count >= 2
        registered_names = {
            call.kwargs["registered_model_name"]
            for call in backend._client.pyfunc.log_model.call_args_list
        }
        assert registered_names == {
            "openai>gpt-4o-mini",
            "openai>text-embedding-3-small",
        }

    def test_registry_model_name_uses_gt_separator(self) -> None:
        """Registered model names use provider>model (MLflow forbids /)."""
        assert _registry_model_name("openai", "gpt-4o-mini") == "openai>gpt-4o-mini"
        assert _registry_model_name("  watsonx ", " granite ") == "watsonx>granite"
        assert _registry_model_name("", "gpt") == ""


class TestMLflowPerMetricAggregates:
    """Tests for per-metric aggregate metrics logged at finalize."""

    def test_per_metric_pass_rates(self, mocker: MockerFixture) -> None:
        """finalize() logs pass_rate for each distinct metric."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(
                metric_identifier="ragas:faithfulness", result="PASS", score=0.9
            )
        )
        backend.save_result(
            _make_result(
                metric_identifier="ragas:faithfulness", result="FAIL", score=0.3
            )
        )
        backend.save_result(
            _make_result(metric_identifier="custom:tool_eval", result="PASS", score=1.0)
        )

        backend._client.log_metrics.reset_mock()
        backend.finalize()

        agg = backend._client.log_metrics.call_args[0][0]
        assert agg["aggregate/pass_rate/ragas_faithfulness"] == pytest.approx(0.5)
        assert agg["aggregate/pass_rate/custom_tool_eval"] == pytest.approx(1.0)

    def test_per_metric_mean_scores(self, mocker: MockerFixture) -> None:
        """finalize() logs mean_score for each distinct metric."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(metric_identifier="ragas:faithfulness", score=0.8)
        )
        backend.save_result(
            _make_result(metric_identifier="ragas:faithfulness", score=0.6)
        )
        backend.save_result(
            _make_result(metric_identifier="custom:tool_eval", score=1.0)
        )

        backend._client.log_metrics.reset_mock()
        backend.finalize()

        agg = backend._client.log_metrics.call_args[0][0]
        assert agg["aggregate/mean_score/ragas_faithfulness"] == pytest.approx(0.7)
        assert agg["aggregate/mean_score/custom_tool_eval"] == pytest.approx(1.0)

    def test_per_metric_none_score_excluded(self, mocker: MockerFixture) -> None:
        """Per-metric mean_score excludes results with score=None."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(
                metric_identifier="ragas:faithfulness", score=0.8, result="PASS"
            )
        )
        backend.save_result(
            _make_result(
                metric_identifier="ragas:faithfulness", score=None, result="ERROR"
            )
        )

        backend._client.log_metrics.reset_mock()
        backend.finalize()

        agg = backend._client.log_metrics.call_args[0][0]
        assert agg["aggregate/mean_score/ragas_faithfulness"] == pytest.approx(0.8)
        assert agg["aggregate/pass_rate/ragas_faithfulness"] == pytest.approx(0.5)


class TestMLflowResultsTable:
    """Tests for log_table() results artifact."""

    def test_finalize_logs_table(self, mocker: MockerFixture) -> None:
        """finalize() calls log_table with columnar dict."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(_make_result(score=0.9, result="PASS"))
        backend.save_result(_make_result(score=0.4, result="FAIL"))

        backend.finalize()

        backend._client.log_table.assert_called_once()
        call_kwargs = backend._client.log_table.call_args
        table_data = call_kwargs[1]["data"]
        assert isinstance(table_data, dict)
        assert len(table_data["result"]) == 2
        assert call_kwargs[1]["artifact_file"] == "evaluation_results.json"

    def test_table_has_expected_columns(self, mocker: MockerFixture) -> None:
        """Table dict contains the expected column keys."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(_make_result())

        backend.finalize()

        table = backend._client.log_table.call_args[1]["data"]
        expected_keys = {
            "conversation_group_id",
            "turn_id",
            "metric",
            "result",
            "score",
            "threshold",
            "reason",
            "query",
            "response",
            "expected_response",
            "execution_time",
            "evaluation_latency",
            "agent_latency",
            "time_to_first_token",
            "streaming_duration",
            "tokens_per_second",
            "api_input_tokens",
            "api_output_tokens",
            "judge_llm_input_tokens",
            "judge_llm_output_tokens",
            "embedding_tokens",
            "judge_scores",
        }
        assert expected_keys.issubset(table.keys())

    def test_table_column_values(self, mocker: MockerFixture) -> None:
        """Table column values match the evaluation result."""
        backend = _create_initialized_backend(mocker)

        backend.save_result(
            _make_result(
                score=0.85,
                result="PASS",
                api_input_tokens=100,
                api_output_tokens=50,
                judge_llm_input_tokens=200,
                judge_llm_output_tokens=80,
                embedding_tokens=30,
                evaluation_latency=1.5,
                execution_time=2.0,
                agent_latency=0.8,
                time_to_first_token=0.25,
                streaming_duration=3.5,
                tokens_per_second=42.0,
            )
        )

        backend.finalize()

        table = backend._client.log_table.call_args[1]["data"]
        assert table["conversation_group_id"][0] == "conv_1"
        assert table["metric"][0] == "ragas:answer_relevancy"
        assert table["result"][0] == "PASS"
        assert table["score"][0] == pytest.approx(0.85)
        assert table["query"][0] == "What is OpenShift?"
        assert table["api_input_tokens"][0] == 100
        assert table["api_output_tokens"][0] == 50
        assert table["judge_llm_input_tokens"][0] == 200
        assert table["judge_llm_output_tokens"][0] == 80
        assert table["embedding_tokens"][0] == 30
        assert table["evaluation_latency"][0] == pytest.approx(1.5)
        assert table["execution_time"][0] == pytest.approx(2.0)
        assert table["agent_latency"][0] == pytest.approx(0.8)
        assert table["time_to_first_token"][0] == pytest.approx(0.25)
        assert table["streaming_duration"][0] == pytest.approx(3.5)
        assert table["tokens_per_second"][0] == pytest.approx(42.0)

    def test_table_not_logged_with_no_results(self, mocker: MockerFixture) -> None:
        """finalize() skips log_table when no results exist."""
        backend = _create_initialized_backend(mocker)

        backend.finalize()

        backend._client.log_table.assert_not_called()


class TestMLflowRunParams:
    """Tests for run params logged from system_config."""

    def test_params_include_system_config(self, mocker: MockerFixture) -> None:
        """initialize() logs judge_model and embedding_model from system_config."""
        mock_mlflow = mocker.MagicMock()
        mock_mlflow.start_run.return_value = mocker.MagicMock()

        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            return_value=mock_mlflow,
        )

        mock_config = mocker.MagicMock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4o-mini"
        mock_config.embedding.provider = "openai"
        mock_config.embedding.model = "text-embedding-3-small"
        mock_config.judge_panel = None
        mock_config.agents = None

        backend = MLflowStorageBackend(MLflowBackendConfig(), system_config=mock_config)
        backend.initialize(RunInfo(name="test"))

        params = mock_mlflow.log_params.call_args[0][0]
        assert params["judge_model"] == "openai/gpt-4o-mini"
        assert params["embedding_model"] == "openai/text-embedding-3-small"

    def test_params_include_judge_panel(self, mocker: MockerFixture) -> None:
        """initialize() logs judge_panel and aggregation_strategy."""
        mock_mlflow = mocker.MagicMock()
        mock_mlflow.start_run.return_value = mocker.MagicMock()

        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            return_value=mock_mlflow,
        )

        mock_config = mocker.MagicMock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4o-mini"
        mock_config.embedding.provider = "openai"
        mock_config.embedding.model = "text-embedding-3-small"
        mock_config.judge_panel.judges = ["gpt-4.1-mini", "gpt-4o-mini"]
        mock_config.judge_panel.aggregation_strategy = "max"
        mock_config.agents = None

        backend = MLflowStorageBackend(MLflowBackendConfig(), system_config=mock_config)
        backend.initialize(RunInfo(name="test"))

        params = mock_mlflow.log_params.call_args[0][0]
        assert params["judge_panel"] == "gpt-4.1-mini, gpt-4o-mini"
        assert params["judge_aggregation"] == "max"

    def test_params_without_system_config(self, mocker: MockerFixture) -> None:
        """initialize() logs basic params when system_config is None."""
        mock_mlflow = mocker.MagicMock()
        mock_mlflow.start_run.return_value = mocker.MagicMock()

        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage._HAS_MLFLOW",
            True,
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.mlflow_storage.importlib.import_module",
            return_value=mock_mlflow,
        )

        backend = MLflowStorageBackend(MLflowBackendConfig())
        backend.initialize(RunInfo(name="test"))

        params = mock_mlflow.log_params.call_args[0][0]
        assert "run_name" in params
        assert "eval_run_id" in params
        assert "judge_model" not in params


class TestMLflowFactoryAndLoader:
    """Integration tests for factory and config loader."""

    def test_factory_creates_mlflow_backend(self) -> None:
        """create_pipeline_storage_backend handles MLflowBackendConfig."""
        backend = create_pipeline_storage_backend([MLflowBackendConfig()])
        assert isinstance(backend, MLflowStorageBackend)
        backend.close()

    def test_loader_parses_mlflow_config(self) -> None:
        """ConfigLoader._parse_storage_config handles type='mlflow'."""
        loader = ConfigLoader()
        configs = loader._parse_storage_config(
            [
                {
                    "type": "mlflow",
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "my_eval",
                }
            ]
        )
        assert len(configs) == 1
        assert isinstance(configs[0], MLflowBackendConfig)
        assert configs[0].tracking_uri == "http://localhost:5000"
        assert configs[0].experiment_name == "my_eval"

    def test_loader_parses_mlflow_config_defaults(self) -> None:
        """ConfigLoader._parse_storage_config uses defaults for mlflow."""
        loader = ConfigLoader()
        configs = loader._parse_storage_config([{"type": "mlflow"}])
        assert len(configs) == 1
        assert isinstance(configs[0], MLflowBackendConfig)
        assert configs[0].tracking_uri is None
        assert configs[0].experiment_name == "lightspeed_evaluation"

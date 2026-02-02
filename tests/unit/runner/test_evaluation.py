# pylint: disable=unused-argument
"""Unit tests for runner/evaluation.py."""

import argparse
import os
from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.runner.evaluation import _clear_caches, main, run_evaluation


def _make_eval_args(**kwargs: Any) -> argparse.Namespace:
    """Helper to create eval_args namespace with defaults."""
    defaults = {
        "system_config": "config/system.yaml",
        "eval_data": "config/evaluation_data.yaml",
        "output_dir": None,
        "tags": None,
        "conv_ids": None,
        "cache_warmup": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestRunEvaluation:
    """Unit tests for run_evaluation function."""

    def test_run_evaluation_success(
        self,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test successful evaluation run."""
        # Mock ConfigLoader
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.api.enabled = False
        mock_config.output.output_dir = "/tmp/output"
        mock_config.output.base_filename = "test"
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mock_config_loader_class = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader_class.return_value = mock_loader

        # Mock evaluation data
        mock_eval_data = [mocker.Mock()]

        # Mock DataValidator (imported inside function)
        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = mock_eval_data

        # Mock EvaluationPipeline (imported inside function)
        mock_pipeline = mocker.Mock()
        mock_result = mocker.Mock()
        mock_result.result = "PASS"
        mock_pipeline.run_evaluation.return_value = [mock_result]

        mock_pipeline_class = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.EvaluationPipeline"
        )
        mock_pipeline_class.return_value = mock_pipeline

        # Mock OutputHandler (imported inside function)
        mock_output_handler = mocker.Mock()
        mock_output_handler.output_dir = "/tmp/output"
        mock_output_class = mocker.patch(
            "lightspeed_evaluation.core.output.OutputHandler"
        )
        mock_output_class.return_value = mock_output_handler

        # Mock calculate_basic_stats (imported inside function)
        mock_stats = mocker.patch(
            "lightspeed_evaluation.core.output.statistics.calculate_basic_stats"
        )
        mock_stats.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
            "total_judge_llm_input_tokens": 100,
            "total_judge_llm_output_tokens": 50,
            "total_judge_llm_tokens": 150,
        }

        result = run_evaluation(_make_eval_args())

        assert result is not None
        assert result["TOTAL"] == 1
        assert result["PASS"] == 1
        mock_pipeline.close.assert_called_once()

    def test_run_evaluation_with_output_dir_override(
        self,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test evaluation with custom output directory."""
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.api.enabled = False
        mock_config.output.output_dir = "/tmp/output"
        mock_config.output.base_filename = "test"
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mock_config_loader_class = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader_class.return_value = mock_loader

        mock_eval_data = [mocker.Mock()]
        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = mock_eval_data

        mock_pipeline = mocker.Mock()
        mock_pipeline.run_evaluation.return_value = []
        mock_pipeline_class = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.EvaluationPipeline"
        )
        mock_pipeline_class.return_value = mock_pipeline

        mock_output_handler = mocker.Mock()
        mock_output_handler.output_dir = "/custom/output"
        mock_output_class = mocker.patch(
            "lightspeed_evaluation.core.output.OutputHandler"
        )
        mock_output_class.return_value = mock_output_handler

        mock_stats = mocker.patch(
            "lightspeed_evaluation.core.output.statistics.calculate_basic_stats"
        )
        mock_stats.return_value = {
            "TOTAL": 0,
            "PASS": 0,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
            "total_judge_llm_input_tokens": 0,
            "total_judge_llm_output_tokens": 0,
            "total_judge_llm_tokens": 0,
        }

        run_evaluation(_make_eval_args(output_dir="/custom/output"))

        # Verify custom output dir was used
        mock_pipeline_class.assert_called_once()
        call_args = mock_pipeline_class.call_args
        assert call_args[0][1] == "/custom/output"

    def test_run_evaluation_file_not_found(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test evaluation handles FileNotFoundError."""
        mock_config_loader = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader.return_value.load_system_config.side_effect = (
            FileNotFoundError("Config not found")
        )

        result = run_evaluation(_make_eval_args(system_config="missing.yaml"))

        assert result is None
        captured = capsys.readouterr()
        assert "Evaluation failed" in captured.out

    def test_run_evaluation_value_error(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test evaluation handles ValueError."""
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.api.enabled = False
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mock_config_loader_class = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader_class.return_value = mock_loader

        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.side_effect = ValueError(
            "Invalid data"
        )

        result = run_evaluation(_make_eval_args(eval_data="invalid.yaml"))

        assert result is None
        captured = capsys.readouterr()
        assert "Evaluation failed" in captured.out

    def test_run_evaluation_with_errors_in_results(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test evaluation reports errors in results."""
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.api.enabled = False
        mock_config.output.output_dir = "/tmp/output"
        mock_config.output.base_filename = "test"
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mock_config_loader_class = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader_class.return_value = mock_loader

        mock_eval_data = [mocker.Mock()]
        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = mock_eval_data

        mock_pipeline = mocker.Mock()
        mock_pipeline.run_evaluation.return_value = []
        mock_pipeline_class = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.EvaluationPipeline"
        )
        mock_pipeline_class.return_value = mock_pipeline

        mock_output_handler = mocker.Mock()
        mock_output_handler.output_dir = "/tmp/output"
        mock_output_class = mocker.patch(
            "lightspeed_evaluation.core.output.OutputHandler"
        )
        mock_output_class.return_value = mock_output_handler

        mock_stats = mocker.patch(
            "lightspeed_evaluation.core.output.statistics.calculate_basic_stats"
        )
        mock_stats.return_value = {
            "TOTAL": 10,
            "PASS": 5,
            "FAIL": 2,
            "ERROR": 3,
            "SKIPPED": 0,
            "total_judge_llm_input_tokens": 500,
            "total_judge_llm_output_tokens": 250,
            "total_judge_llm_tokens": 750,
        }

        result = run_evaluation(_make_eval_args())

        assert result is not None
        assert result["ERROR"] == 3
        captured = capsys.readouterr()
        assert "3 evaluations had errors" in captured.out

    def test_run_evaluation_closes_pipeline_on_exception(
        self,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test pipeline is closed even if evaluation fails."""
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.api.enabled = False
        mock_config.output.output_dir = "/tmp/output"
        mock_config.output.base_filename = "test"
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mock_config_loader_class = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader_class.return_value = mock_loader

        mock_eval_data = [mocker.Mock()]
        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = mock_eval_data

        mock_pipeline = mocker.Mock()
        mock_pipeline.run_evaluation.side_effect = RuntimeError("Processing error")
        mock_pipeline_class = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.EvaluationPipeline"
        )
        mock_pipeline_class.return_value = mock_pipeline

        result = run_evaluation(_make_eval_args())

        # Should close pipeline even on error
        mock_pipeline.close.assert_called_once()
        assert result is None

    def test_run_evaluation_with_empty_filter_result(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test evaluation returns empty result when filter matches nothing."""
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.api.enabled = False
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader",
            return_value=mock_loader,
        )

        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = []

        result = run_evaluation(_make_eval_args(tags=["nonexistent"]))

        assert result is not None
        assert result["TOTAL"] == 0
        mock_validator.return_value.load_evaluation_data.assert_called_once_with(
            "config/evaluation_data.yaml", tags=["nonexistent"], conv_ids=None
        )

        # Verify warning message appears
        captured = capsys.readouterr()
        assert "No conversation groups matched the filter criteria" in captured.out

    def test_run_evaluation_with_filter_parameters(self, mocker: MockerFixture) -> None:
        """Test that filter parameters are correctly passed to DataValidator."""
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.api.enabled = False
        mock_config.output.output_dir = "/tmp/output"
        mock_config.output.base_filename = "test"
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader",
            return_value=mock_loader,
        )

        mock_eval_data = mocker.Mock()
        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = [mock_eval_data]

        mock_pipeline = mocker.Mock()
        mock_pipeline.run_evaluation.return_value = []
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.EvaluationPipeline",
            return_value=mock_pipeline,
        )

        mock_output_handler = mocker.Mock()
        mock_output_handler.output_dir = "/tmp/output"
        mocker.patch(
            "lightspeed_evaluation.core.output.OutputHandler",
            return_value=mock_output_handler,
        )

        mocker.patch(
            "lightspeed_evaluation.core.output.statistics.calculate_basic_stats",
            return_value={
                "TOTAL": 1,
                "PASS": 1,
                "FAIL": 0,
                "ERROR": 0,
                "SKIPPED": 0,
                "total_judge_llm_input_tokens": 100,
                "total_judge_llm_output_tokens": 50,
                "total_judge_llm_tokens": 150,
            },
        )

        run_evaluation(_make_eval_args(tags=["basic"], conv_ids=["conv_1"]))

        mock_validator.return_value.load_evaluation_data.assert_called_once_with(
            "config/evaluation_data.yaml",
            tags=["basic"],
            conv_ids=["conv_1"],
        )


class TestClearCaches:
    """Unit tests for _clear_caches function."""

    def test_clear_caches_with_all_caches_enabled(
        self, tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test clearing all cache directories when all caches are enabled."""
        # Create test cache directories with files
        llm_cache = tmp_path / "llm_cache"
        api_cache = tmp_path / "api_cache"
        embedding_cache = tmp_path / "embedding_cache"

        llm_cache.mkdir()
        api_cache.mkdir()
        embedding_cache.mkdir()

        (llm_cache / "test1.db").write_text("test")
        (api_cache / "test2.db").write_text("test")
        (embedding_cache / "test3.db").write_text("test")

        # Mock system config
        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = str(llm_cache)
        mock_config.api.enabled = True
        mock_config.api.cache_enabled = True
        mock_config.api.cache_dir = str(api_cache)
        mock_config.embedding.cache_enabled = True
        mock_config.embedding.cache_dir = str(embedding_cache)

        # Call clear caches
        _clear_caches(mock_config)

        # Verify directories were cleared and recreated
        assert llm_cache.exists()
        assert api_cache.exists()
        assert embedding_cache.exists()
        assert not (llm_cache / "test1.db").exists()
        assert not (api_cache / "test2.db").exists()
        assert not (embedding_cache / "test3.db").exists()

        # Verify output messages
        captured = capsys.readouterr()
        assert "Cleared LLM Judge cache" in captured.out
        assert "Cleared API cache" in captured.out
        assert "Cleared Embedding cache" in captured.out

    def test_clear_caches_with_only_llm_cache_enabled(
        self, tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test clearing only LLM cache when others are disabled."""
        llm_cache = tmp_path / "llm_cache"
        llm_cache.mkdir()
        (llm_cache / "test.db").write_text("test")

        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = str(llm_cache)
        mock_config.api.enabled = False
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        _clear_caches(mock_config)

        assert llm_cache.exists()
        assert not (llm_cache / "test.db").exists()

        captured = capsys.readouterr()
        assert "Cleared LLM Judge cache" in captured.out
        assert "Cleared API cache" not in captured.out
        assert "Cleared Embedding cache" not in captured.out

    def test_clear_caches_when_no_caches_enabled(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test clearing caches when none are enabled."""
        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = False
        mock_config.api.enabled = False
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        _clear_caches(mock_config)

        captured = capsys.readouterr()
        assert "No caches enabled to clear" in captured.out

    def test_clear_caches_creates_nonexistent_directories(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that cache directories are created if they don't exist."""
        llm_cache = tmp_path / "new_llm_cache"

        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = str(llm_cache)
        mock_config.api.enabled = False
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        # Directory doesn't exist yet
        assert not llm_cache.exists()

        _clear_caches(mock_config)

        # Directory should be created
        assert llm_cache.exists()
        assert llm_cache.is_dir()

    def test_clear_caches_refuses_root_directory(self, mocker: MockerFixture) -> None:
        """Test that clearing root directory raises DataValidationError."""
        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = "/"  # Dangerous: root directory
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        # Should raise DataValidationError
        with pytest.raises(
            DataValidationError, match="Refusing to delete unsafe cache directory"
        ):
            _clear_caches(mock_config)

        # Verify root directory still exists
        assert os.path.exists("/")

    def test_clear_caches_refuses_current_directory_relative(
        self, mocker: MockerFixture
    ) -> None:
        """Test that clearing current directory (.) raises DataValidationError."""
        cwd = os.getcwd()

        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = "."  # Dangerous: current directory
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        # Should raise DataValidationError
        with pytest.raises(
            DataValidationError, match="Refusing to delete unsafe cache directory"
        ):
            _clear_caches(mock_config)

        # Verify current directory still exists
        assert os.path.exists(cwd)
        assert os.path.exists(__file__)

    def test_clear_caches_refuses_current_directory_absolute(
        self, mocker: MockerFixture
    ) -> None:
        """Test that clearing current directory (absolute path) raises error."""
        cwd = os.getcwd()

        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = cwd  # Dangerous: current directory as absolute path
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        # Should raise DataValidationError
        with pytest.raises(
            DataValidationError, match="Refusing to delete unsafe cache directory"
        ):
            _clear_caches(mock_config)

        # Verify current directory still exists
        assert os.path.exists(cwd)

    def test_clear_caches_refuses_symlink_to_current_directory(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that symlink to current directory is blocked."""
        cwd = os.getcwd()

        # Create a symlink pointing to current directory
        symlink = tmp_path / "link_to_cwd"
        symlink.symlink_to(cwd)

        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = str(symlink)  # Symlink to current directory
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        # Should raise DataValidationError (resolved path equals cwd)
        with pytest.raises(
            DataValidationError, match="Refusing to delete unsafe cache directory"
        ):
            _clear_caches(mock_config)

        # Verify current directory still exists
        assert os.path.exists(cwd)

    def test_clear_caches_refuses_symlink_to_root(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that symlink to root directory is blocked."""
        # Create a symlink pointing to root
        symlink = tmp_path / "link_to_root"
        symlink.symlink_to("/")

        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = str(symlink)  # Symlink to root
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False

        # Should raise DataValidationError (resolved path equals /)
        with pytest.raises(
            DataValidationError, match="Refusing to delete unsafe cache directory"
        ):
            _clear_caches(mock_config)

        # Verify root directory still exists
        assert os.path.exists("/")

    def test_clear_caches_with_api_cache_enabled_but_api_disabled(
        self, tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test that API cache IS cleared even when API is disabled."""
        # This changed in evaluation.py line 29:
        # "We clear the api cache even if the Lightspeed core api is disabled"
        api_cache = tmp_path / "api_cache"
        api_cache.mkdir()
        (api_cache / "test.db").write_text("test")

        mock_config = mocker.Mock()
        mock_config.llm.cache_enabled = False
        mock_config.api.cache_enabled = True  # Cache enabled
        mock_config.api.cache_dir = str(api_cache)
        mock_config.embedding.cache_enabled = False

        _clear_caches(mock_config)

        # API cache SHOULD be cleared (even though api.enabled might be False)
        assert api_cache.exists()
        assert not (api_cache / "test.db").exists()

        captured = capsys.readouterr()
        assert "Cleared API cache" in captured.out


class TestRunEvaluationCacheWarmup:
    """Unit tests for run_evaluation with cache warmup."""

    def test_run_evaluation_with_cache_warmup_flag(
        self, tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        # pylint: disable=too-many-locals
        """Test that cache warmup clears caches before evaluation."""
        # Setup cache directories
        llm_cache = tmp_path / "llm_cache"
        llm_cache.mkdir()
        (llm_cache / "old_cache.db").write_text("old data")

        # Mock ConfigLoader
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = str(llm_cache)
        mock_config.api.enabled = False
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False
        mock_config.output.output_dir = "/tmp/output"
        mock_config.output.base_filename = "test"
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mock_config_loader_class = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader_class.return_value = mock_loader

        # Mock evaluation data
        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = [mocker.Mock()]

        # Mock pipeline
        mock_pipeline = mocker.Mock()
        mock_result = mocker.Mock()
        mock_result.result = "PASS"
        mock_pipeline.run_evaluation.return_value = [mock_result]
        mock_pipeline_class = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.EvaluationPipeline"
        )
        mock_pipeline_class.return_value = mock_pipeline

        # Mock output handler
        mock_output_handler = mocker.Mock()
        mock_output_handler.output_dir = "/tmp/output"
        mock_output_class = mocker.patch(
            "lightspeed_evaluation.core.output.OutputHandler"
        )
        mock_output_class.return_value = mock_output_handler

        # Mock stats
        mock_stats = mocker.patch(
            "lightspeed_evaluation.core.output.statistics.calculate_basic_stats"
        )
        mock_stats.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
            "total_judge_llm_input_tokens": 100,
            "total_judge_llm_output_tokens": 50,
            "total_judge_llm_tokens": 150,
        }

        # Run evaluation with cache warmup flag
        result = run_evaluation(_make_eval_args(cache_warmup=True))

        # Verify cache was cleared
        assert not (llm_cache / "old_cache.db").exists()
        assert llm_cache.exists()

        # Verify output contains warmup message
        captured = capsys.readouterr()
        assert "Cache warmup mode: Clearing existing caches" in captured.out
        assert "Cleared LLM Judge cache" in captured.out

        # Verify evaluation still ran successfully
        assert result is not None
        assert result["PASS"] == 1

    def test_run_evaluation_without_cache_warmup_flag(
        self, tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        # pylint: disable=too-many-locals
        """Test that caches are NOT cleared when warmup flag is false."""
        # Setup cache directory with existing file
        llm_cache = tmp_path / "llm_cache"
        llm_cache.mkdir()
        (llm_cache / "existing_cache.db").write_text("existing data")

        # Mock ConfigLoader
        mock_loader = mocker.Mock()
        mock_config = mocker.Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model = "gpt-4"
        mock_config.llm.cache_enabled = True
        mock_config.llm.cache_dir = str(llm_cache)
        mock_config.api.enabled = False
        mock_config.api.cache_enabled = False
        mock_config.embedding.cache_enabled = False
        mock_config.output.output_dir = "/tmp/output"
        mock_config.output.base_filename = "test"
        mock_loader.system_config = mock_config
        mock_loader.load_system_config.return_value = mock_config

        mock_config_loader_class = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.ConfigLoader"
        )
        mock_config_loader_class.return_value = mock_loader

        mock_eval_data = [mocker.Mock()]
        mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
        mock_validator.return_value.load_evaluation_data.return_value = mock_eval_data

        mock_pipeline = mocker.Mock()
        mock_pipeline.run_evaluation.return_value = []
        mock_pipeline_class = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.EvaluationPipeline"
        )
        mock_pipeline_class.return_value = mock_pipeline

        mock_output_handler = mocker.Mock()
        mock_output_handler.output_dir = "/tmp/output"
        mock_output_class = mocker.patch(
            "lightspeed_evaluation.core.output.OutputHandler"
        )
        mock_output_class.return_value = mock_output_handler

        mock_stats = mocker.patch(
            "lightspeed_evaluation.core.output.statistics.calculate_basic_stats"
        )
        mock_stats.return_value = {
            "TOTAL": 0,
            "PASS": 0,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
            "total_judge_llm_input_tokens": 0,
            "total_judge_llm_output_tokens": 0,
            "total_judge_llm_tokens": 0,
        }

        # Run evaluation WITHOUT cache warmup flag
        run_evaluation(_make_eval_args(cache_warmup=False))

        # Verify cache was NOT cleared
        assert (llm_cache / "existing_cache.db").exists()

        # Verify no warmup message in output
        captured = capsys.readouterr()
        assert "Cache warmup mode" not in captured.out
        assert "Cleared LLM Judge cache" not in captured.out


class TestMain:
    """Unit tests for main CLI function."""

    def test_main_default_args(self, mocker: MockerFixture) -> None:
        """Test main with default arguments."""
        mocker.patch(
            "sys.argv",
            ["lightspeed-eval"],
        )

        mock_run = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.run_evaluation"
        )
        mock_run.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }

        exit_code = main()

        assert exit_code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.system_config == "config/system.yaml"
        assert args.eval_data == "config/evaluation_data.yaml"
        assert args.output_dir is None

    def test_main_custom_args(self, mocker: MockerFixture) -> None:
        """Test main with custom arguments."""
        mocker.patch(
            "sys.argv",
            [
                "lightspeed-eval",
                "--system-config",
                "custom/system.yaml",
                "--eval-data",
                "custom/eval.yaml",
                "--output-dir",
                "/custom/output",
            ],
        )

        mock_run = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.run_evaluation"
        )
        mock_run.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }

        exit_code = main()

        assert exit_code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.system_config == "custom/system.yaml"
        assert args.eval_data == "custom/eval.yaml"
        assert args.output_dir == "/custom/output"

    def test_main_returns_error_on_failure(self, mocker: MockerFixture) -> None:
        """Test main returns error code on failure."""
        mocker.patch(
            "sys.argv",
            ["lightspeed-eval"],
        )
        mock_run = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.run_evaluation"
        )
        mock_run.return_value = None  # Indicates failure

        exit_code = main()

        assert exit_code == 1

    @pytest.mark.parametrize(
        "args,expected_tags,expected_conv_ids",
        [
            (["--tags", "basic", "advanced"], ["basic", "advanced"], None),
            (["--conv-ids", "conv_1", "conv_2"], None, ["conv_1", "conv_2"]),
            (
                ["--tags", "basic", "--conv-ids", "conv_special"],
                ["basic"],
                ["conv_special"],
            ),
        ],
    )
    def test_main_with_filters(
        self,
        mocker: MockerFixture,
        args: list[str],
        expected_tags: list[str] | None,
        expected_conv_ids: list[str] | None,
    ) -> None:
        """Test main with filter arguments."""
        mocker.patch("sys.argv", ["lightspeed-eval"] + args)

        mock_run = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.run_evaluation"
        )
        mock_run.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }

        exit_code = main()

        assert exit_code == 0
        mock_run.assert_called_once()
        eval_args = mock_run.call_args[0][0]
        assert eval_args.tags == expected_tags
        assert eval_args.conv_ids == expected_conv_ids

    def test_main_with_cache_warmup_flag(self, mocker: MockerFixture) -> None:
        """Test main with --cache-warmup flag."""
        mocker.patch(
            "sys.argv",
            ["lightspeed-eval", "--cache-warmup"],
        )

        mock_run = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.run_evaluation"
        )
        mock_run.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }

        exit_code = main()

        assert exit_code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.cache_warmup is True

    def test_main_without_cache_warmup_flag(self, mocker: MockerFixture) -> None:
        """Test main without --cache-warmup flag defaults to False."""
        mocker.patch(
            "sys.argv",
            ["lightspeed-eval"],
        )

        mock_run = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.run_evaluation"
        )
        mock_run.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }

        exit_code = main()

        assert exit_code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.cache_warmup is False

    def test_main_with_cache_warmup_and_filters(self, mocker: MockerFixture) -> None:
        """Test main with both --cache-warmup and filter flags."""
        mocker.patch(
            "sys.argv",
            [
                "lightspeed-eval",
                "--cache-warmup",
                "--tags",
                "basic",
                "--conv-ids",
                "conv_1",
            ],
        )

        mock_run = mocker.patch(
            "lightspeed_evaluation.runner.evaluation.run_evaluation"
        )
        mock_run.return_value = {
            "TOTAL": 1,
            "PASS": 1,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }

        exit_code = main()

        assert exit_code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.cache_warmup is True
        assert args.tags == ["basic"]
        assert args.conv_ids == ["conv_1"]

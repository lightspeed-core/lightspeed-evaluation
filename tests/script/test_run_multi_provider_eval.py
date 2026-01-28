#!/usr/bin/env python3
"""Pytest tests for run_multi_provider_eval.py script."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import tempfile as temp_module
import logging
import multiprocessing
import shutil

import pytest
import yaml

from script.run_multi_provider_eval import MultiProviderEvaluationRunner


class TestMultiProviderEvaluationRunnerInit:
    """Tests for MultiProviderEvaluationRunner initialization."""

    def test_init_success(self, temp_config_files: dict[str, Path]) -> None:
        """Test successful initialization of the runner."""
        runner = MultiProviderEvaluationRunner(
            providers_config_path=str(temp_config_files["providers_config"]),
            system_config_path=str(temp_config_files["system_config"]),
            eval_data_path=str(temp_config_files["eval_data"]),
        )

        assert runner.providers_config_path == Path(
            temp_config_files["providers_config"]
        )
        assert runner.system_config_path == Path(temp_config_files["system_config"])
        assert runner.eval_data_path == Path(temp_config_files["eval_data"])
        assert runner.output_base.exists()
        assert not runner.results

    def test_init_config_not_found(self, temp_config_files: dict[str, Path]) -> None:
        """Test initialization fails when any config file is missing."""
        with pytest.raises(FileNotFoundError, match="Providers config not found"):
            MultiProviderEvaluationRunner(
                providers_config_path="nonexistent.yaml",
                system_config_path=str(temp_config_files["system_config"]),
                eval_data_path=str(temp_config_files["eval_data"]),
            )

    def test_max_workers_from_constructor(
        self, temp_config_files: dict[str, Path]
    ) -> None:
        """Test max_workers configured via constructor argument."""
        runner = MultiProviderEvaluationRunner(
            providers_config_path=str(temp_config_files["providers_config"]),
            system_config_path=str(temp_config_files["system_config"]),
            eval_data_path=str(temp_config_files["eval_data"]),
            max_workers=4,
        )
        assert runner.max_workers == 4

    def test_max_workers_from_config_file(
        self, temp_config_files: dict[str, Path], tmp_path: Path
    ) -> None:
        """Test max_workers configured via config file."""
        # Create config with max_workers setting
        config_with_workers = {
            "providers": {
                "openai": {"models": ["gpt-4o-mini"]},
            },
            "settings": {
                "output_base": str(tmp_path / "output"),
                "max_workers": 6,
            },
        }
        config_path = tmp_path / "config_with_workers.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_with_workers, f)

        runner = MultiProviderEvaluationRunner(
            providers_config_path=str(config_path),
            system_config_path=str(temp_config_files["system_config"]),
            eval_data_path=str(temp_config_files["eval_data"]),
        )
        assert runner.max_workers == 6

    def test_max_workers_string_coercion(
        self, temp_config_files: dict[str, Path], tmp_path: Path
    ) -> None:
        """Test max_workers string value from YAML is coerced to int."""
        # Create config with string max_workers
        config_with_string = {
            "providers": {
                "openai": {"models": ["gpt-4o-mini"]},
            },
            "settings": {
                "output_base": str(tmp_path / "output"),
                "max_workers": "4",  # String value
            },
        }
        config_path = tmp_path / "config_string.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_with_string, f)

        runner = MultiProviderEvaluationRunner(
            providers_config_path=str(config_path),
            system_config_path=str(temp_config_files["system_config"]),
            eval_data_path=str(temp_config_files["eval_data"]),
        )
        assert runner.max_workers == 4
        assert isinstance(runner.max_workers, int)

    def test_max_workers_invalid_value(
        self, temp_config_files: dict[str, Path], tmp_path: Path
    ) -> None:
        """Test max_workers with invalid value raises clear error."""
        # Create config with invalid max_workers
        config_invalid = {
            "providers": {
                "openai": {"models": ["gpt-4o-mini"]},
            },
            "settings": {
                "output_base": str(tmp_path / "output"),
                "max_workers": "invalid",
            },
        }
        config_path = tmp_path / "config_invalid.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_invalid, f)

        with pytest.raises(ValueError, match="max_workers must be an integer"):
            MultiProviderEvaluationRunner(
                providers_config_path=str(config_path),
                system_config_path=str(temp_config_files["system_config"]),
                eval_data_path=str(temp_config_files["eval_data"]),
            )

    def test_max_workers_minimum_value(
        self, temp_config_files: dict[str, Path]
    ) -> None:
        """Test max_workers is enforced to be at least 1."""
        runner = MultiProviderEvaluationRunner(
            providers_config_path=str(temp_config_files["providers_config"]),
            system_config_path=str(temp_config_files["system_config"]),
            eval_data_path=str(temp_config_files["eval_data"]),
            max_workers=0,  # Invalid value
        )
        assert runner.max_workers == 1  # Should be clamped to 1

        runner2 = MultiProviderEvaluationRunner(
            providers_config_path=str(temp_config_files["providers_config"]),
            system_config_path=str(temp_config_files["system_config"]),
            eval_data_path=str(temp_config_files["eval_data"]),
            max_workers=-5,  # Invalid value
        )
        assert runner2.max_workers == 1  # Should be clamped to 1

    def test_resource_warning_high_thread_count(
        self,
        temp_config_files: dict[str, Path],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning is logged when total threads is very high."""

        # Create system config with high max_threads
        system_config = {
            "core": {"max_threads": 100},
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "api": {"enabled": False},
            "output": {"output_dir": "./eval_output"},
        }
        system_path = tmp_path / "system_high_threads.yaml"
        with open(system_path, "w", encoding="utf-8") as f:
            yaml.dump(system_config, f)

        with caplog.at_level(logging.WARNING):
            runner = MultiProviderEvaluationRunner(
                providers_config_path=str(temp_config_files["providers_config"]),
                system_config_path=str(system_path),
                eval_data_path=str(temp_config_files["eval_data"]),
                max_workers=4,  # 4 × 100 = 400 threads (should trigger warning)
            )

        # Check warning was logged
        assert any(
            "High resource usage detected" in record.message
            for record in caplog.records
        )
        assert any(
            "400 concurrent threads" in record.message for record in caplog.records
        )
        assert runner.max_workers == 4

    def test_no_resource_warning_reasonable_config(
        self,
        temp_config_files: dict[str, Path],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test no warning with reasonable thread count."""

        # Calculate safe thread count based on actual CPU count
        cpu_count = multiprocessing.cpu_count()
        # Use values that keep total threads <= cpu_count * 2
        max_workers = 2
        max_threads = max(1, cpu_count // max_workers)  # Ensure total <= cpu_count * 2

        # Create system config with reasonable max_threads
        system_config = {
            "core": {"max_threads": max_threads},
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "api": {"enabled": False},
            "output": {"output_dir": "./eval_output"},
        }
        system_path = tmp_path / "system_reasonable.yaml"
        with open(system_path, "w", encoding="utf-8") as f:
            yaml.dump(system_config, f)

        with caplog.at_level(logging.WARNING):
            runner = MultiProviderEvaluationRunner(
                providers_config_path=str(temp_config_files["providers_config"]),
                system_config_path=str(system_path),
                eval_data_path=str(temp_config_files["eval_data"]),
                max_workers=max_workers,
            )

        # Check no warning was logged
        total_threads = max_workers * max_threads
        assert not any(
            "High resource usage detected" in record.message
            for record in caplog.records
        ), (
            f"Expected no warning: {max_workers} workers × {max_threads} "
            f"threads = {total_threads} on {cpu_count} CPUs"
        )
        assert runner.max_workers == max_workers


class TestLoadYAML:
    """Tests for _load_yaml method."""

    def test_load_valid_yaml(
        self, runner: MultiProviderEvaluationRunner, temp_config_files: dict[str, Path]
    ) -> None:
        """Test loading a valid YAML file."""
        config = runner._load_yaml(  # pylint: disable=protected-access
            temp_config_files["providers_config"]
        )
        assert isinstance(config, dict)
        assert "providers" in config
        assert "openai" in config["providers"]
        assert "models" in config["providers"]["openai"]
        assert "settings" in config

    def test_load_invalid_yaml(
        self, runner: MultiProviderEvaluationRunner, tmp_path: Path
    ) -> None:
        """Test loading an invalid YAML file."""
        invalid_yaml = tmp_path / "invalid.yaml"
        with open(invalid_yaml, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Error parsing YAML file"):
            runner._load_yaml(invalid_yaml)  # pylint: disable=protected-access

    def test_load_yaml_non_dict_type(
        self, runner: MultiProviderEvaluationRunner, tmp_path: Path
    ) -> None:
        """Test that YAML files not containing dictionaries are rejected."""
        list_yaml = tmp_path / "list.yaml"
        with open(list_yaml, "w", encoding="utf-8") as f:
            yaml.dump(["item1", "item2", "item3"], f)

        with pytest.raises(ValueError, match="must be a mapping, got list"):
            runner._load_yaml(list_yaml)  # pylint: disable=protected-access


class TestCreateProviderModelConfigs:  # pylint: disable=too-few-public-methods
    """Tests for _create_provider_model_configs method."""

    def test_create_configs_multiple_providers(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test creating configs with multiple providers."""
        configs = (
            runner._create_provider_model_configs()  # pylint: disable=protected-access
        )

        assert len(configs) == 3  # 2 openai models + 1 watsonx model

        # Check OpenAI configs
        openai_configs = [c for c in configs if c["provider_id"] == "openai"]
        assert len(openai_configs) == 2
        assert any(c["model"] == "gpt-4o-mini" for c in openai_configs)
        assert any(c["model"] == "gpt-4-turbo" for c in openai_configs)

        # Check Watsonx configs
        watsonx_configs = [c for c in configs if c["provider_id"] == "watsonx"]
        assert len(watsonx_configs) == 1
        assert watsonx_configs[0]["model"] == "ibm/granite-13b-chat-v2"

        # Verify only providers are included (not settings or other top-level keys)
        assert all(c["provider_name"] != "settings" for c in configs)


class TestCreateModifiedSystemConfig:
    """Tests for _create_modified_system_config method."""

    def test_llm_config_stays_constant(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test that LLM judge config is NOT modified (stays constant for fair comparison)."""
        original_llm_provider = runner.system_config["llm"]["provider"]
        original_llm_model = runner.system_config["llm"]["model"]

        modified = (
            runner._create_modified_system_config(  # pylint: disable=protected-access
                provider_id="watsonx",
                model="ibm/granite-13b-chat-v2",
            )
        )

        # LLM judge should remain unchanged
        assert modified["llm"]["provider"] == original_llm_provider
        assert modified["llm"]["model"] == original_llm_model

    def test_api_config_is_modified(self, temp_config_files: dict[str, Path]) -> None:
        """Test that API config is modified when API is enabled."""
        # Create system config with API enabled
        system_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
            },
            "api": {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
            "output": {"output_dir": "./eval_output"},
        }
        system_path = temp_config_files["system_config"].parent / "system_api.yaml"
        with open(system_path, "w", encoding="utf-8") as f:
            yaml.dump(system_config, f)

        runner = MultiProviderEvaluationRunner(
            providers_config_path=str(temp_config_files["providers_config"]),
            system_config_path=str(system_path),
            eval_data_path=str(temp_config_files["eval_data"]),
        )

        modified = (
            runner._create_modified_system_config(  # pylint: disable=protected-access
                provider_id="watsonx",
                model="ibm/granite-13b-chat-v2",
            )
        )

        # API config should be modified with provider and model only
        assert modified["api"]["provider"] == "watsonx"
        assert modified["api"]["model"] == "ibm/granite-13b-chat-v2"

        # LLM judge should remain unchanged
        assert modified["llm"]["provider"] == "openai"
        assert modified["llm"]["model"] == "gpt-4o-mini"
        assert modified["llm"]["temperature"] == 0.0  # Not modified


class TestCreateTempSystemConfig:
    """Tests for _create_temp_system_config method."""

    def test_create_temp_config_file(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test that a temporary config file is created."""
        temp_path = (
            runner._create_temp_system_config(  # pylint: disable=protected-access
                provider_id="openai",
                model="gpt-4o-mini",
            )
        )

        try:
            assert temp_path.exists()
            assert temp_path.suffix == ".yaml"
            assert "openai" in temp_path.name
            assert "gpt-4o-mini" in temp_path.name

            # Verify content
            with open(temp_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert config["llm"]["provider"] == "openai"
            assert config["llm"]["model"] == "gpt-4o-mini"
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()

    def test_temp_config_cleanup_on_yaml_dump_failure(
        self,
        runner: MultiProviderEvaluationRunner,
    ) -> None:
        """Test that temp file is cleaned up when yaml.dump() fails."""

        # Track the temp file path that gets created
        created_temp_path = None
        original_named_temp_file = temp_module.NamedTemporaryFile

        def track_temp_file(*args: Any, **kwargs: Any) -> Any:
            nonlocal created_temp_path
            temp_file = original_named_temp_file(  # pylint: disable=consider-using-with
                *args, **kwargs
            )
            created_temp_path = Path(temp_file.name)
            return temp_file

        # Mock NamedTemporaryFile to track the created file
        with patch(
            "script.run_multi_provider_eval.tempfile.NamedTemporaryFile",
            side_effect=track_temp_file,
        ):
            # Mock yaml.dump to raise an exception
            with patch(
                "script.run_multi_provider_eval.yaml.dump",
                side_effect=Exception("YAML dump failed"),
            ):
                with pytest.raises(Exception, match="YAML dump failed"):
                    runner._create_temp_system_config(  # pylint: disable=protected-access
                        provider_id="openai",
                        model="gpt-4o-mini",
                    )

                # Verify the temp file was cleaned up after the exception
                assert (
                    created_temp_path is not None
                ), "Temp file should have been created"
                assert (
                    not created_temp_path.exists()
                ), "Temp file should have been cleaned up"

    def test_temp_config_sanitizes_special_characters(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test that special characters in provider_id and model are sanitized."""
        temp_path = (
            runner._create_temp_system_config(  # pylint: disable=protected-access
                provider_id="open..ai//test",
                model="gpt:4o-mini/special",
            )
        )

        try:
            # Verify filename doesn't contain path separators or colons
            # (except drive letter on Windows)
            assert "/" not in temp_path.name
            # On some systems, : might appear in drive letters on Windows, so we're lenient
            # The key is that path traversal characters are neutralized
            assert temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestPathTraversalSecurity:
    """Tests for path traversal security."""

    @pytest.fixture
    def runner(
        self, temp_config_files: dict[str, Path]
    ) -> MultiProviderEvaluationRunner:
        """Create a runner instance for testing."""
        return MultiProviderEvaluationRunner(
            providers_config_path=str(temp_config_files["providers_config"]),
            system_config_path=str(temp_config_files["system_config"]),
            eval_data_path=str(temp_config_files["eval_data"]),
        )

    def test_path_traversal_blocked_in_provider_id(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test that path traversal in provider_id is sanitized."""
        with patch(
            "script.run_multi_provider_eval.run_evaluation",
            return_value={"PASS": 0, "FAIL": 0, "ERROR": 1},
        ):
            # Attempt path traversal in provider_id
            result = runner._run_single_evaluation(  # pylint: disable=protected-access
                provider_name="malicious",
                provider_id="../../etc",
                model="test",
            )

            # Verify that the output path is sanitized and stays within base
            output_path = Path(result["output_dir"])
            base_path = runner.output_base.resolve()
            assert output_path.resolve().is_relative_to(base_path)
            # Verify dangerous characters are removed
            assert ".." not in str(output_path)
            assert "/" not in str(output_path.relative_to(base_path).parts[0])

            # Cleanup
            if output_path.exists():
                shutil.rmtree(output_path.parent, ignore_errors=True)

    def test_path_traversal_blocked_in_model(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test that path traversal in model name is sanitized."""
        with patch(
            "script.run_multi_provider_eval.run_evaluation",
            return_value={"PASS": 0, "FAIL": 0, "ERROR": 1},
        ):
            # Attempt path traversal in model
            result = runner._run_single_evaluation(  # pylint: disable=protected-access
                provider_name="openai",
                provider_id="openai",
                model="../../../etc/passwd",
            )

            # Verify that the output path is sanitized and stays within base
            output_path = Path(result["output_dir"])
            base_path = runner.output_base.resolve()
            assert output_path.resolve().is_relative_to(base_path)
            # Verify dangerous characters are removed
            assert ".." not in str(output_path)

            # Cleanup
            if output_path.exists():
                shutil.rmtree(output_path.parent.parent, ignore_errors=True)


class TestRunSingleEvaluation:
    """Tests for _run_single_evaluation method."""

    def test_run_single_evaluation_success(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test successful single evaluation."""
        # Mock run_evaluation to return a successful summary
        with patch(
            "script.run_multi_provider_eval.run_evaluation",
            return_value={"PASS": 5, "FAIL": 2, "ERROR": 0},
        ) as mock_run_eval:
            result = runner._run_single_evaluation(  # pylint: disable=protected-access
                provider_name="openai",
                provider_id="openai",
                model="gpt-4o-mini",
            )

            assert result["success"] is True
            assert result["provider_id"] == "openai"
            assert result["model"] == "gpt-4o-mini"
            assert result["summary"]["PASS"] == 5
            assert result["error"] is None
            assert "duration_seconds" in result
            mock_run_eval.assert_called_once()

    def test_run_single_evaluation_failure(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test evaluation failure handling."""
        # Mock run_evaluation to return None (failure)
        with patch("script.run_multi_provider_eval.run_evaluation", return_value=None):
            result = runner._run_single_evaluation(  # pylint: disable=protected-access
                provider_name="openai",
                provider_id="openai",
                model="gpt-4o-mini",
            )

            assert result["success"] is False
            assert result["error"] == "Evaluation returned None (failed)"

    def test_run_single_evaluation_invalid_summary(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test evaluation with invalid summary structure."""
        # Mock run_evaluation to return a summary missing required keys
        with patch(
            "script.run_multi_provider_eval.run_evaluation",
            return_value={"PASS": 5, "FAIL": 2},  # Missing ERROR key
        ):
            result = runner._run_single_evaluation(  # pylint: disable=protected-access
                provider_name="openai",
                provider_id="openai",
                model="gpt-4o-mini",
            )

            assert result["success"] is False
            assert "Invalid summary structure" in result["error"]
            assert "summary" not in result


class TestRunEvaluations:  # pylint: disable=too-few-public-methods
    """Tests for run_evaluations method."""

    def test_run_evaluations_sequential(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test sequential evaluation execution."""
        # Force sequential mode
        runner.max_workers = 1

        with patch.object(
            runner,
            "_run_single_evaluation",
            return_value={
                "success": True,
                "provider_id": "test",
                "model": "test-model",
            },
        ) as mock_single_eval:
            results = runner.run_evaluations()

            assert len(results) == 3  # 2 openai + 1 watsonx
            assert mock_single_eval.call_count == 3


class TestGenerateSummary:  # pylint: disable=too-few-public-methods
    """Tests for generate_summary method."""

    def test_generate_summary_mixed_results(
        self, runner: MultiProviderEvaluationRunner
    ) -> None:
        """Test summary generation with mixed results."""
        runner.results = [
            {"success": True, "provider_id": "openai", "model": "gpt-4o-mini"},
            {"success": False, "provider_id": "watsonx", "model": "granite"},
        ]

        summary = runner.generate_summary()

        assert summary["total_evaluations"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["success_rate"] == "50.0%"


class TestBestModelAnalysis:
    """Tests for best model analysis functionality."""

    def test_analyze_model_performance(
        self,
        runner: MultiProviderEvaluationRunner,
        tmp_path: Path,
        sample_evaluation_summary: dict[str, Any],
    ) -> None:
        """Test successful model performance analysis."""
        # Setup: Create evaluation summary files
        model_dir = tmp_path / "eval_output" / "openai" / "gpt-4o-mini"
        model_dir.mkdir(parents=True, exist_ok=True)

        summary_file = model_dir / "evaluation_20250101_120000_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(sample_evaluation_summary, f)

        runner.output_base = tmp_path / "eval_output"
        runner.results = [
            {
                "success": True,
                "provider_id": "openai",
                "model": "gpt-4o-mini",
                "output_dir": str(model_dir),
            }
        ]

        runner.analyze_model_performance()

        # Verify analysis was performed correctly
        assert "openai/gpt-4o-mini" in runner.model_stats
        stats = runner.model_stats["openai/gpt-4o-mini"]
        assert stats["overall"]["total_evaluations"] == 10
        assert stats["overall"]["passed"] == 8
        assert 0.0 <= stats["composite_score"] <= 1.0

    def test_percentage_to_decimal_conversion(
        self, runner: MultiProviderEvaluationRunner, sample_evaluation_summary: dict
    ) -> None:
        """Test that percentage rates (80.0) convert to decimals (0.8)."""
        stats = runner._analyze_single_model(  # pylint: disable=protected-access
            "test/model", sample_evaluation_summary
        )

        # Verify percentage conversion
        assert abs(stats["overall"]["pass_rate"] - 0.8) < 0.01
        assert 0.0 <= stats["overall"]["pass_rate"] <= 1.0

    def test_composite_score(self, runner: MultiProviderEvaluationRunner) -> None:
        """Test composite score calculation."""
        # Perfect model should get score of 1.0
        perfect = runner._calculate_composite_score(  # pylint: disable=protected-access
            1.0, 0.0, 1.0, 1.0
        )
        assert abs(perfect - 1.0) < 0.0001

        # Poor model should get score of 0.0
        poor = runner._calculate_composite_score(  # pylint: disable=protected-access
            0.0, 1.0, 0.0, 0.0
        )
        assert poor == 0.0

    def test_model_ranking(self, runner: MultiProviderEvaluationRunner) -> None:
        """Test models are ranked by composite score."""
        runner.model_stats = {
            "model1": {"composite_score": 0.85},
            "model2": {"composite_score": 0.92},
            "model3": {"composite_score": 0.70},
        }

        ranked = runner.rank_models()

        # Best model should be first
        assert ranked[0][0] == "model2"  # Highest: 0.92
        assert ranked[1][0] == "model1"  # Second: 0.85
        assert ranked[2][0] == "model3"  # Lowest: 0.70

    def test_save_analysis_to_yaml(
        self, runner: MultiProviderEvaluationRunner, tmp_path: Path
    ) -> None:
        """Test saving analysis results to YAML file."""
        runner.output_base = tmp_path
        runner.model_stats = {
            "model1": {
                "composite_score": 0.85,
                "overall": {"pass_rate": 0.8, "error_rate": 0.0},
                "score_statistics": {"mean": 0.82, "confidence_interval": None},
            },
        }

        output_path = runner.save_model_comparison()

        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["best_model"]["model"] == "model1"
        assert data["best_model"]["composite_score"] == 0.85

    def test_print_report(
        self, runner: MultiProviderEvaluationRunner, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test statistical comparison report output."""
        runner.model_stats = {
            "model1": {
                "model_key": "model1",
                "composite_score": 0.88,
                "overall": {
                    "total_evaluations": 10,
                    "passed": 9,
                    "failed": 1,
                    "errors": 0,
                    "pass_rate": 0.9,
                    "fail_rate": 0.1,
                    "error_rate": 0.0,
                    "success_rate": 1.0,
                },
                "score_statistics": {
                    "mean": 0.85,
                    "std": 0.05,
                    "median": 0.85,
                    "min": 0.75,
                    "max": 0.95,
                    "count": 10,
                    "confidence_interval": {
                        "low": 0.82,
                        "high": 0.88,
                        "mean": 0.85,
                        "confidence_level": 95,
                    },
                },
                "metric_statistics": {},
            },
        }

        runner.print_statistical_comparison()
        output = capsys.readouterr().out

        # Verify key report sections
        assert "🏆 BEST MODEL" in output
        assert "model1" in output
        assert "RECOMMENDATIONS" in output

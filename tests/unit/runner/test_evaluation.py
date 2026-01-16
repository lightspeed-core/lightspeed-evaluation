"""Unit tests for runner/evaluation.py."""

import argparse

from lightspeed_evaluation.runner.evaluation import main, run_evaluation


def _make_eval_args(**kwargs) -> argparse.Namespace:
    """Helper to create eval_args namespace with defaults."""
    defaults = {
        "system_config": "config/system.yaml",
        "eval_data": "config/evaluation_data.yaml",
        "output_dir": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestRunEvaluation:
    """Unit tests for run_evaluation function."""

    def test_run_evaluation_success(self, mocker, capsys):
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

    def test_run_evaluation_with_output_dir_override(self, mocker, capsys):
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

    def test_run_evaluation_file_not_found(self, mocker, capsys):
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

    def test_run_evaluation_value_error(self, mocker, capsys):
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

    def test_run_evaluation_with_errors_in_results(self, mocker, capsys):
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

        assert result["ERROR"] == 3
        captured = capsys.readouterr()
        assert "3 evaluations had errors" in captured.out

    def test_run_evaluation_closes_pipeline_on_exception(self, mocker, capsys):
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


class TestMain:
    """Unit tests for main CLI function."""

    def test_main_default_args(self, mocker):
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

    def test_main_custom_args(self, mocker):
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

    def test_main_returns_error_on_failure(self, mocker):
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

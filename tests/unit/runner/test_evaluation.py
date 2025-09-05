"""Unit tests for runner.evaluation module."""

from unittest.mock import MagicMock, patch

import pytest

from lightspeed_evaluation.runner.evaluation import main, run_evaluation


class TestRunEvaluation:
    """Test the run_evaluation function."""

    @patch("lightspeed_evaluation.runner.evaluation.calculate_basic_stats")
    @patch("lightspeed_evaluation.runner.evaluation.OutputHandler")
    @patch("lightspeed_evaluation.runner.evaluation.EvaluationDriver")
    @patch("lightspeed_evaluation.runner.evaluation.DataValidator")
    @patch("lightspeed_evaluation.runner.evaluation.ConfigLoader")
    @patch("builtins.print")  # Suppress output
    def test_run_evaluation_success(
        self,
        mock_print,
        mock_config_loader_class,
        mock_data_validator_class,
        mock_driver_class,
        mock_output_handler_class,
        mock_calc_stats,
    ):
        """Test successful evaluation run."""
        # Setup simple mocks
        mock_loader = MagicMock()
        mock_config_loader_class.return_value = mock_loader
        mock_system_config = MagicMock()
        mock_system_config.llm_provider = "openai"
        mock_system_config.llm_model = "gpt-4o-mini"
        mock_system_config.output_dir = "./test_output"
        mock_loader.load_system_config.return_value = mock_system_config

        mock_validator = MagicMock()
        mock_data_validator_class.return_value = mock_validator
        mock_validator.load_evaluation_data.return_value = [MagicMock()]

        mock_driver = MagicMock()
        mock_driver_class.return_value = mock_driver
        mock_driver.run_evaluation.return_value = [MagicMock()]

        mock_output_handler = MagicMock()
        mock_output_handler_class.return_value = mock_output_handler

        mock_calc_stats.return_value = {"TOTAL": 5, "PASS": 4, "FAIL": 1, "ERROR": 0}

        # Run the function
        result = run_evaluation("system.yaml", "eval_data.yaml")

        # Assertions
        assert result == {"TOTAL": 5, "PASS": 4, "FAIL": 1, "ERROR": 0}
        mock_loader.load_system_config.assert_called_once_with("system.yaml")
        mock_validator.load_evaluation_data.assert_called_once_with("eval_data.yaml")

    @patch("lightspeed_evaluation.runner.evaluation.DataValidator")
    @patch("lightspeed_evaluation.runner.evaluation.ConfigLoader")
    @patch("builtins.print")  # Suppress output
    def test_run_evaluation_data_failure(
        self, mock_print, mock_config_loader_class, mock_data_validator_class
    ):
        """Test evaluation failure during data loading."""
        mock_loader = MagicMock()
        mock_config_loader_class.return_value = mock_loader
        mock_loader.load_system_config.return_value = MagicMock()

        mock_validator = MagicMock()
        mock_data_validator_class.return_value = mock_validator
        mock_validator.load_evaluation_data.side_effect = ValueError("Invalid data")

        result = run_evaluation("system.yaml", "invalid_data.yaml")

        assert result is None


class TestMain:
    """Test the main CLI function."""

    @patch("lightspeed_evaluation.runner.evaluation.run_evaluation")
    @patch("lightspeed_evaluation.runner.evaluation.setup_environment_variables")
    @patch("lightspeed_evaluation.runner.evaluation.Path")
    def test_main_success(self, mock_path, mock_setup_env, mock_run_eval):
        """Test successful main function execution."""
        mock_path.return_value.exists.return_value = True
        mock_run_eval.return_value = {"TOTAL": 5, "PASS": 4, "FAIL": 1, "ERROR": 0}

        with patch(
            "sys.argv",
            [
                "lightspeed-eval",
                "--system-config",
                "system.yaml",
                "--eval-data",
                "eval_data.yaml",
            ],
        ):
            result = main()

        assert result == 0
        mock_setup_env.assert_called_once_with("system.yaml")
        mock_run_eval.assert_called_once_with("system.yaml", "eval_data.yaml", None)

    @patch("lightspeed_evaluation.runner.evaluation.run_evaluation")
    @patch("lightspeed_evaluation.runner.evaluation.setup_environment_variables")
    @patch("lightspeed_evaluation.runner.evaluation.Path")
    def test_main_with_custom_output_dir(
        self, mock_path, mock_setup_env, mock_run_eval
    ):
        """Test main function with custom output directory."""
        mock_path.return_value.exists.return_value = True
        mock_run_eval.return_value = {"TOTAL": 3, "PASS": 2, "FAIL": 1, "ERROR": 0}

        with patch(
            "sys.argv",
            [
                "lightspeed-eval",
                "--system-config",
                "system.yaml",
                "--eval-data",
                "eval_data.yaml",
                "--output-dir",
                "/custom",
            ],
        ):
            result = main()

        assert result == 0
        mock_run_eval.assert_called_once_with(
            "system.yaml", "eval_data.yaml", "/custom"
        )

    def test_main_help_argument(self):
        """Test main function with help argument."""
        with patch("sys.argv", ["lightspeed-eval", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

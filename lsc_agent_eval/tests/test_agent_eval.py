"""Tests for agent evaluation CLI."""

import sys
from pytest_mock import MockerFixture

import pytest

from lsc_agent_eval.agent_eval import _args_parser, main


class TestArgsParser:
    """Test argument parser."""

    def test_args_parser_minimal(self):
        """Test argument parser with minimal required arguments."""
        args = [
            "--eval_data_yaml",
            "test.yaml",
            "--agent_provider",
            "openai",
            "--agent_model",
            "gpt-4",
        ]

        parsed = _args_parser(args)

        assert parsed.eval_data_yaml == "test.yaml"
        assert parsed.agent_provider == "openai"
        assert parsed.agent_model == "gpt-4"
        assert parsed.agent_endpoint == "http://localhost:8080"  # default
        assert parsed.result_dir == "eval_output/"  # default
        assert parsed.endpoint_type == "streaming"  # default

    def test_args_parser_all_arguments(self):
        """Test argument parser with all arguments."""
        args = [
            "--eval_data_yaml",
            "test.yaml",
            "--agent_endpoint",
            "http://custom:9090",
            "--agent_provider",
            "watsonx",
            "--agent_model",
            "granite-3-8b-instruct",
            "--agent_auth_token_file",
            "token.txt",
            "--judge_provider",
            "openai",
            "--judge_model",
            "gpt-4",
            "--kubeconfig",
            "~/kubeconfig",
            "--result_dir",
            "custom_results/",
            "--endpoint_type",
            "query",
        ]

        parsed = _args_parser(args)

        assert parsed.eval_data_yaml == "test.yaml"
        assert parsed.agent_endpoint == "http://custom:9090"
        assert parsed.agent_provider == "watsonx"
        assert parsed.agent_model == "granite-3-8b-instruct"
        assert parsed.agent_auth_token_file == "token.txt"
        assert parsed.judge_provider == "openai"
        assert parsed.judge_model == "gpt-4"
        assert parsed.kubeconfig == "~/kubeconfig"
        assert parsed.result_dir == "custom_results/"
        assert parsed.endpoint_type == "query"

    def test_args_parser_missing_required(self):
        """Test argument parser with missing required arguments."""
        args = [
            "--eval_data_yaml",
            "test.yaml",
            "--agent_provider",
            "openai",
            # Missing agent_model
        ]

        with pytest.raises(SystemExit):
            _args_parser(args)

    def test_args_parser_help(self):
        """Test argument parser help."""
        args = ["--help"]

        with pytest.raises(SystemExit):
            _args_parser(args)

    def test_args_parser_optional_judge_args(self):
        """Test argument parser with optional judge arguments."""
        args = [
            "--eval_data_yaml",
            "test.yaml",
            "--agent_provider",
            "openai",
            "--agent_model",
            "gpt-4",
            "--judge_provider",
            "watsonx",
            # Missing judge_model - should be allowed
        ]

        parsed = _args_parser(args)

        assert parsed.judge_provider == "watsonx"
        assert parsed.judge_model is None

    def test_args_parser_invalid_endpoint_type(self):
        """Test argument parser with invalid endpoint type."""
        args = [
            "--eval_data_yaml",
            "test_data.yaml",
            "--agent_provider",
            "test_provider",
            "--agent_model",
            "test_model",
            "--endpoint_type",
            "invalid",
        ]

        with pytest.raises(SystemExit):
            _args_parser(args)


class TestMain:
    """Test main function."""

    def test_main_success(self, mocker: MockerFixture):
        """Test successful main execution."""
        # Setup mocks
        mock_logging_config = mocker.patch(
            "lsc_agent_eval.agent_eval.logging.basicConfig"
        )
        mock_args_parser = mocker.patch("lsc_agent_eval.agent_eval._args_parser")
        mock_agent_goal_eval = mocker.patch("lsc_agent_eval.agent_eval.AgentGoalEval")

        mock_args = mocker.Mock()
        mock_args_parser.return_value = mock_args
        mock_evaluator = mocker.Mock()
        mock_agent_goal_eval.return_value = mock_evaluator

        # Run main
        main()

        # Verify components were called
        mock_args_parser.assert_called_once_with(sys.argv[1:])
        mock_agent_goal_eval.assert_called_once_with(mock_args)
        mock_evaluator.run_evaluation.assert_called_once()
        mock_logging_config.assert_called_once()

    def test_main_logging_configuration(self, mocker: MockerFixture):
        """Test main logging configuration."""
        # Setup mocks
        mock_logging_config = mocker.patch(
            "lsc_agent_eval.agent_eval.logging.basicConfig"
        )
        mock_args_parser = mocker.patch("lsc_agent_eval.agent_eval._args_parser")
        mock_agent_goal_eval = mocker.patch("lsc_agent_eval.agent_eval.AgentGoalEval")

        mock_args = mocker.Mock()
        mock_args_parser.return_value = mock_args
        mock_evaluator = mocker.Mock()
        mock_agent_goal_eval.return_value = mock_evaluator

        # Run main
        main()

        # Verify logging was configured correctly
        import logging

        mock_logging_config.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def test_main_evaluator_initialization(self, mocker: MockerFixture):
        """Test main evaluator initialization."""
        # Setup mocks
        mocker.patch("lsc_agent_eval.agent_eval.logging.basicConfig")
        mock_args_parser = mocker.patch("lsc_agent_eval.agent_eval._args_parser")
        mock_agent_goal_eval = mocker.patch("lsc_agent_eval.agent_eval.AgentGoalEval")

        mock_args = mocker.Mock()
        mock_args_parser.return_value = mock_args
        mock_evaluator = mocker.Mock()
        mock_agent_goal_eval.return_value = mock_evaluator

        # Run main
        main()

        # Verify evaluator was initialized with parsed args
        mock_agent_goal_eval.assert_called_once_with(mock_args)

    def test_main_evaluator_execution(self, mocker: MockerFixture):
        """Test main evaluator execution."""
        # Setup mocks
        mocker.patch("lsc_agent_eval.agent_eval.logging.basicConfig")
        mock_args_parser = mocker.patch("lsc_agent_eval.agent_eval._args_parser")
        mock_agent_goal_eval = mocker.patch("lsc_agent_eval.agent_eval.AgentGoalEval")

        mock_args = mocker.Mock()
        mock_args_parser.return_value = mock_args
        mock_evaluator = mocker.Mock()
        mock_agent_goal_eval.return_value = mock_evaluator

        # Run main
        main()

        # Verify evaluator execution was called
        mock_evaluator.run_evaluation.assert_called_once()

    def test_main_with_real_argv(self, mocker: MockerFixture):
        """Test main with real sys.argv."""
        # Setup mocks
        mocker.patch("lsc_agent_eval.agent_eval.logging.basicConfig")
        mock_args_parser = mocker.patch("lsc_agent_eval.agent_eval._args_parser")
        mock_agent_goal_eval = mocker.patch("lsc_agent_eval.agent_eval.AgentGoalEval")
        mocker.patch(
            "sys.argv",
            [
                "agent_eval",
                "--eval_data_yaml",
                "test.yaml",
                "--agent_provider",
                "openai",
                "--agent_model",
                "gpt-4",
            ],
        )

        mock_args = mocker.Mock()
        mock_args_parser.return_value = mock_args
        mock_evaluator = mocker.Mock()
        mock_agent_goal_eval.return_value = mock_evaluator

        # Run main
        main()

        # Verify args parser was called with correct arguments
        mock_args_parser.assert_called_once_with(
            [
                "--eval_data_yaml",
                "test.yaml",
                "--agent_provider",
                "openai",
                "--agent_model",
                "gpt-4",
            ]
        )

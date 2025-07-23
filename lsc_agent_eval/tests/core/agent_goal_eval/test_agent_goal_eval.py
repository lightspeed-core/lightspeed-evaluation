"""Tests for agent goal evaluation orchestrator."""

from unittest.mock import Mock, patch

import pytest

from lsc_agent_eval.core.agent_goal_eval.agent_goal_eval import AgentGoalEval
from lsc_agent_eval.core.agent_goal_eval.models import (
    EvaluationDataConfig,
    EvaluationResult,
)


class TestAgentGoalEval:
    """Test AgentGoalEval orchestrator."""

    @pytest.fixture
    def mock_args(self):
        """Mock evaluation arguments."""
        args = Mock()
        args.eval_data_yaml = "test_data.yaml"
        args.agent_endpoint = "http://localhost:8080"
        args.agent_auth_token_file = None
        args.agent_provider = "openai"
        args.agent_model = "gpt-4"
        args.judge_provider = "openai"
        args.judge_model = "gpt-4"
        args.kubeconfig = None
        args.result_dir = "results/"
        return args

    @pytest.fixture
    def sample_configs(self):
        """Sample evaluation configurations."""
        return [
            EvaluationDataConfig(
                eval_id="test_001",
                eval_query="What is Kubernetes?",
                eval_type="judge-llm",
                expected_response="Kubernetes is a container orchestration platform",
            ),
            EvaluationDataConfig(
                eval_id="test_002",
                eval_query="Deploy nginx",
                eval_type="script",
                eval_verify_script="./verify.sh",
            ),
        ]

    @pytest.fixture
    def sample_results(self):
        """Sample evaluation results."""
        return [
            EvaluationResult(
                eval_id="test_001",
                query="What is Kubernetes?",
                response="Kubernetes is a container orchestration platform",
                eval_type="judge-llm",
                result="PASS",
            ),
            EvaluationResult(
                eval_id="test_002",
                query="Deploy nginx",
                response="kubectl create deployment nginx --image=nginx",
                eval_type="script",
                result="PASS",
            ),
        ]

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_init_with_judge_manager(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
    ):
        """Test initialization with judge manager."""
        AgentGoalEval(mock_args)

        # Verify all components were initialized
        mock_config_manager.assert_called_once_with("test_data.yaml")
        mock_agent_client.assert_called_once_with("http://localhost:8080", None)
        mock_judge_manager.assert_called_once_with("openai", "gpt-4")
        mock_evaluation_runner.assert_called_once_with(
            mock_agent_client.return_value,
            mock_judge_manager.return_value,
            kubeconfig=None,
        )
        mock_results_manager.assert_called_once_with("results/")

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_init_without_judge_manager(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_agent_client,
        mock_config_manager,
        mock_args,
    ):
        """Test initialization without judge manager."""
        mock_args.judge_provider = None
        mock_args.judge_model = None

        evaluator = AgentGoalEval(mock_args)

        # Verify judge manager was not created
        assert evaluator.judge_manager is None
        mock_evaluation_runner.assert_called_once_with(
            mock_agent_client.return_value,
            None,
            kubeconfig=None,
        )

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_init_with_kubeconfig(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
    ):
        """Test initialization with kubeconfig."""
        mock_args.kubeconfig = "~/kubeconfig"

        AgentGoalEval(mock_args)

        mock_evaluation_runner.assert_called_once_with(
            mock_agent_client.return_value,
            mock_judge_manager.return_value,
            kubeconfig="~/kubeconfig",
        )

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_run_evaluation_success(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
        sample_configs,
        sample_results,
    ):
        """Test successful evaluation execution."""
        # Setup mocks
        mock_config_manager.return_value.get_eval_data.return_value = sample_configs
        mock_evaluation_runner.return_value.run_evaluation.side_effect = sample_results

        evaluator = AgentGoalEval(mock_args)

        # Capture print output
        with patch("builtins.print") as mock_print:
            evaluator.run_evaluation()

        # Verify evaluations were run
        assert mock_evaluation_runner.return_value.run_evaluation.call_count == 2

        # Verify results were saved
        mock_results_manager.return_value.save_results.assert_called_once_with(
            sample_results
        )

        # Verify summary was printed
        mock_print.assert_called()

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_run_evaluation_with_errors(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
        sample_configs,
        capsys,
    ):
        """Test evaluation execution with errors."""
        # Setup results with errors
        results_with_errors = [
            EvaluationResult(
                eval_id="test_001",
                query="What is Kubernetes?",
                response="Kubernetes is a container orchestration platform",
                eval_type="judge-llm",
                result="PASS",
            ),
            EvaluationResult(
                eval_id="test_002",
                query="Deploy nginx",
                response="",
                eval_type="script",
                result="ERROR",
                error="Script execution failed",
            ),
        ]

        mock_config_manager.return_value.get_eval_data.return_value = sample_configs
        mock_evaluation_runner.return_value.run_evaluation.side_effect = (
            results_with_errors
        )

        evaluator = AgentGoalEval(mock_args)

        evaluator.run_evaluation()

        # Capture stdout/stderr output
        captured = capsys.readouterr()

        # Verify error messages are printed to stdout
        assert "✅ test_001: PASS" in captured.out
        assert "⚠️  test_002: ERROR" in captured.out
        assert "   Query: Deploy nginx" in captured.out
        assert "   Evaluation type: script" in captured.out
        assert "   Response: " in captured.out
        assert "   Error message: Script execution failed" in captured.out

        # Verify evaluations were run
        assert mock_evaluation_runner.return_value.run_evaluation.call_count == 2

        # Verify results were saved
        mock_results_manager.return_value.save_results.assert_called_once_with(
            results_with_errors
        )

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_run_evaluation_exception(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
    ):
        """Test evaluation execution with exception."""
        mock_config_manager.return_value.get_eval_data.side_effect = Exception(
            "Config error"
        )

        evaluator = AgentGoalEval(mock_args)

        with patch(
            "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.logger"
        ) as mock_logger:
            with pytest.raises(Exception, match="Config error"):
                evaluator.run_evaluation()

        # Verify error was logged
        mock_logger.error.assert_called()
        args, kwargs = mock_logger.error.call_args
        assert args[0] == "Evaluation failed: %s"
        assert str(args[1]) == "Config error"

    def test_print_summary_all_pass(self, mock_args):
        """Test print summary with all passing results."""
        results = [
            EvaluationResult("test_001", "query1", "response1", "judge-llm", "PASS"),
            EvaluationResult("test_002", "query2", "response2", "script", "PASS"),
        ]

        with (
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"
            ),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager"),
        ):

            evaluator = AgentGoalEval(mock_args)

            with patch("builtins.print") as mock_print:
                evaluator._print_summary(results)

            # Check that summary was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            summary_text = "\n".join(print_calls)

            assert "Total Evaluations: 2" in summary_text
            assert "Passed: 2" in summary_text
            assert "Failed: 0" in summary_text
            assert "Errored: 0" in summary_text
            assert "Success Rate: 100.0%" in summary_text

    def test_print_summary_mixed_results(self, mock_args):
        """Test print summary with mixed results."""
        results = [
            EvaluationResult("test_001", "query1", "response1", "judge-llm", "PASS"),
            EvaluationResult("test_002", "query2", "response2", "script", "FAIL"),
            EvaluationResult("test_003", "query3", "response3", "script", "ERROR"),
        ]

        with (
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"
            ),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager"),
        ):

            evaluator = AgentGoalEval(mock_args)

            with patch("builtins.print") as mock_print:
                evaluator._print_summary(results)

            # Check that summary was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            summary_text = "\n".join(print_calls)

            assert "Total Evaluations: 3" in summary_text
            assert "Passed: 1" in summary_text
            assert "Failed: 1" in summary_text
            assert "Errored: 1" in summary_text
            assert "Success Rate: 33.3%" in summary_text

    def test_cleanup_with_client(self, mock_args):
        """Test cleanup method with client."""
        with (
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"
            ) as mock_client_class,
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"
            ),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager"),
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client

            evaluator = AgentGoalEval(mock_args)
            evaluator._cleanup()

            # Verify client was closed
            mock_client.close.assert_called_once()

    def test_cleanup_exception(self, mock_args):
        """Test cleanup method with exception."""
        with (
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"
            ) as mock_client_class,
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"
            ),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"
            ),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager"),
        ):

            mock_client = Mock()
            mock_client.close.side_effect = OSError("Cleanup error")
            mock_client_class.return_value = mock_client

            evaluator = AgentGoalEval(mock_args)

            with patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.logger"
            ) as mock_logger:
                evaluator._cleanup()

            # Verify warning was logged
            mock_logger.warning.assert_called()
            args, kwargs = mock_logger.warning.call_args
            assert args[0] == "Error during cleanup: %s"
            assert str(args[1]) == "Cleanup error"

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_run_evaluation_cleanup_called(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
        sample_configs,
        sample_results,
    ):
        """Test that cleanup is called even on success."""
        mock_config_manager.return_value.get_eval_data.return_value = sample_configs
        mock_evaluation_runner.return_value.run_evaluation.side_effect = sample_results

        evaluator = AgentGoalEval(mock_args)

        with patch.object(evaluator, "_cleanup") as mock_cleanup:
            evaluator.run_evaluation()

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

    @patch(
        "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"
    )
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_run_evaluation_cleanup_called_on_exception(
        self,
        mock_results_manager,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
    ):
        """Test that cleanup is called even on exception."""
        mock_config_manager.return_value.get_eval_data.side_effect = Exception(
            "Config error"
        )

        evaluator = AgentGoalEval(mock_args)

        with patch.object(evaluator, "_cleanup") as mock_cleanup:
            with pytest.raises(Exception):
                evaluator.run_evaluation()

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

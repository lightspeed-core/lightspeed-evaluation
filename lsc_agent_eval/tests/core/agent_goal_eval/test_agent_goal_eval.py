"""Tests for agent goal evaluation orchestrator."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from lsc_agent_eval.core.agent_goal_eval.agent_goal_eval import AgentGoalEval
from lsc_agent_eval.core.agent_goal_eval.models import (
    ConversationDataConfig,
    EvaluationDataConfig,
    EvaluationResult,
    EvaluationStats,
)
from lsc_agent_eval.core.utils.exceptions import AgentEvaluationError


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
    def sample_conversation(self):
        """Sample conversation data configuration."""
        return ConversationDataConfig(
            conversation_group="test_conv",
            conversation=[
                EvaluationDataConfig(
                    eval_id="test_001",
                    eval_query="What is Openshift?",
                    eval_types=["response_eval:accuracy"],
                    expected_response="OpenShift is Red Hat's enterprise Kubernetes platform.",
                ),
                EvaluationDataConfig(
                    eval_id="test_002",
                    eval_query="Deploy nginx",
                    eval_types=["response_eval:sub-string"],
                    expected_keywords=["nginx", "deployment"],
                ),
            ],
        )

    @pytest.fixture
    def sample_results(self):
        """Sample evaluation results."""
        return [
            EvaluationResult(
                eval_id="test_001",
                query="What is Kubernetes?",
                response="Kubernetes is a container orchestration platform",
                eval_type="response_eval:accuracy",
                result="PASS",
                conversation_group="test_conv",
                conversation_id="conv-id-123",
            ),
            EvaluationResult(
                eval_id="test_002",
                query="Deploy nginx",
                response="oc create deployment nginx --image=nginx",
                eval_type="response_eval:sub-string",
                result="PASS",
                conversation_group="test_conv",
                conversation_id="conv-id-123",
            ),
        ]

    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner")
    def test_init_with_judge_manager(
        self,
        mock_script_runner,
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
        mock_script_runner.assert_called_once_with(None)
        mock_evaluation_runner.assert_called_once_with(
            mock_agent_client.return_value,
            mock_script_runner.return_value,
            mock_judge_manager.return_value,
        )

    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner")
    def test_init_without_judge_manager(
        self,
        mock_script_runner,
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
            mock_script_runner.return_value,
            None,
        )

    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner")
    def test_init_with_kubeconfig(
        self,
        mock_script_runner,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
    ):
        """Test initialization with kubeconfig."""
        mock_args.kubeconfig = "~/kubeconfig"

        AgentGoalEval(mock_args)

        mock_script_runner.assert_called_once_with("~/kubeconfig")
        mock_evaluation_runner.assert_called_once_with(
            mock_agent_client.return_value,
            mock_script_runner.return_value,
            mock_judge_manager.return_value,
        )

    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner")
    @patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ResultsManager")
    def test_run_evaluation_success(
        self,
        mock_results_manager,
        mock_script_runner,
        mock_evaluation_runner,
        mock_judge_manager,
        mock_agent_client,
        mock_config_manager,
        mock_args,
        sample_conversation,
        sample_results,
    ):
        """Test successful evaluation execution."""
        # Setup mocks
        mock_config_manager.return_value.get_conversations.return_value = [sample_conversation]
        mock_config_manager.return_value.get_eval_count.return_value = 2
        mock_evaluation_runner.return_value.run_evaluation.side_effect = [
            [sample_results[0]],
            [sample_results[1]],
        ]

        # Mock results manager
        mock_results_mgr_instance = MagicMock()
        mock_results_manager.return_value = mock_results_mgr_instance
        mock_stats = EvaluationStats.from_results(sample_results)
        mock_results_mgr_instance.get_results_stats.return_value = mock_stats

        evaluator = AgentGoalEval(mock_args)

        # Capture print output
        with patch("builtins.print") as mock_print:
            evaluator.run_evaluation()

        # Verify evaluations were run
        assert mock_evaluation_runner.return_value.run_evaluation.call_count == 2

        # Verify results were saved
        mock_results_mgr_instance.save_results.assert_called_once_with(mock_args.result_dir)

        # Verify summary was printed
        mock_print.assert_called()

    def test_get_result_summary_success(self, mock_args):
        """Test result summary with available results."""
        with (
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner"),
        ):
            evaluator = AgentGoalEval(mock_args)
            evaluator.result_summary = {"TOTAL": 5, "PASS": 3, "FAIL": 1, "ERROR": 1}

            result = evaluator.get_result_summary()

            assert result == {"TOTAL": 5, "PASS": 3, "FAIL": 1, "ERROR": 1}

    def test_get_result_summary_no_results(self, mock_args):
        """Test result summary with no available results."""
        with (
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner"),
        ):
            evaluator = AgentGoalEval(mock_args)

            with pytest.raises(AgentEvaluationError, match="No results available"):
                evaluator.get_result_summary()

    def test_cleanup_with_client(self, mock_args):
        """Test cleanup method with client."""
        with (
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"
            ) as mock_client_class,
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner"),
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
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentGoalEvalDataManager"),
            patch(
                "lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.AgentHttpClient"
            ) as mock_client_class,
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.JudgeModelManager"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.EvaluationRunner"),
            patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.ScriptRunner"),
        ):

            mock_client = Mock()
            mock_client.close.side_effect = OSError("Cleanup error")
            mock_client_class.return_value = mock_client

            evaluator = AgentGoalEval(mock_args)

            with patch("lsc_agent_eval.core.agent_goal_eval.agent_goal_eval.logger") as mock_logger:
                evaluator._cleanup()

            # Verify warning was logged
            mock_logger.warning.assert_called()
            args, kwargs = mock_logger.warning.call_args
            assert args[0] == "Error during cleanup: %s"
            assert str(args[1]) == "Cleanup error"

"""Tests for evaluation runner."""

from unittest.mock import Mock, patch

import pytest

from lsc_agent_eval.core.agent_goal_eval.evaluator import EvaluationRunner
from lsc_agent_eval.core.agent_goal_eval.models import (
    EvaluationDataConfig,
    EvaluationResult,
)
from lsc_agent_eval.core.utils.api_client import AgentHttpClient
from lsc_agent_eval.core.utils.exceptions import AgentAPIError, ScriptExecutionError
from lsc_agent_eval.core.utils.judge import JudgeModelManager


class TestEvaluationRunner:
    """Test EvaluationRunner."""

    @pytest.fixture
    def mock_agent_client(self):
        """Mock agent client."""
        mock_client = Mock(spec=AgentHttpClient)
        mock_client.query_agent.return_value = "Test agent response"
        return mock_client

    @pytest.fixture
    def mock_judge_manager(self):
        """Mock judge manager."""
        mock_judge = Mock(spec=JudgeModelManager)
        mock_judge.evaluate_response.return_value = "1"
        return mock_judge

    @pytest.fixture
    def sample_config_judge_llm(self):
        """Sample judge-llm evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_001",
            eval_query="What is Kubernetes?",
            eval_type="judge-llm",
            expected_response="Kubernetes is a container orchestration platform",
        )

    @pytest.fixture
    def sample_config_script(self):
        """Sample script evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_002",
            eval_query="Deploy nginx",
            eval_type="script",
            eval_verify_script="./verify.sh",
        )

    @pytest.fixture
    def sample_config_substring(self):
        """Sample substring evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_003",
            eval_query="What is Docker?",
            eval_type="sub-string",
            expected_key_words=["container", "docker"],
        )

    def test_init(self, mock_agent_client, mock_judge_manager):
        """Test EvaluationRunner initialization."""
        runner = EvaluationRunner(
            mock_agent_client, mock_judge_manager, kubeconfig="~/kubeconfig"
        )

        assert runner.agent_client == mock_agent_client
        assert runner.judge_manager == mock_judge_manager
        assert runner.kubeconfig == "~/kubeconfig"

    def test_init_without_judge_manager(self, mock_agent_client):
        """Test EvaluationRunner initialization without judge manager."""
        runner = EvaluationRunner(mock_agent_client)

        assert runner.agent_client == mock_agent_client
        assert runner.judge_manager is None

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_judge_llm_success(
        self,
        mock_script_runner,
        mock_agent_client,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test successful judge-llm evaluation."""
        # Mock agent response
        mock_agent_client.query_agent.return_value = (
            "Kubernetes is a container orchestration platform"
        )

        # Mock judge response
        mock_judge_manager.evaluate_response.return_value = "1"

        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        result = runner.run_evaluation(sample_config_judge_llm, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_001"
        assert result.result == "PASS"
        assert result.eval_type == "judge-llm"
        assert result.error is None

        # Verify agent was queried
        mock_agent_client.query_agent.assert_called_once_with(
            "What is Kubernetes?", "openai", "gpt-4"
        )

        # Verify judge was called
        mock_judge_manager.evaluate_response.assert_called_once()

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_script_success(
        self, mock_script_runner_class, mock_agent_client, sample_config_script
    ):
        """Test successful script evaluation."""
        # Mock agent response
        mock_agent_client.query_agent.return_value = (
            "kubectl create deployment nginx --image=nginx"
        )

        # Mock script runner instance
        mock_script_runner_instance = Mock()
        mock_script_runner_instance.run_script.return_value = True
        mock_script_runner_class.return_value = mock_script_runner_instance

        runner = EvaluationRunner(mock_agent_client)
        result = runner.run_evaluation(sample_config_script, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_002"
        assert result.result == "PASS"
        assert result.eval_type == "script"
        assert result.error is None

        # Verify ScriptRunner was created with the right kubeconfig
        mock_script_runner_class.assert_called_with(kubeconfig=None)
        # Verify script was executed
        mock_script_runner_instance.run_script.assert_called_once_with("./verify.sh")

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_script_failure(
        self, mock_script_runner_class, mock_agent_client, sample_config_script
    ):
        """Test script evaluation failure."""
        # Mock agent response
        mock_agent_client.query_agent.return_value = (
            "kubectl create deployment nginx --image=nginx"
        )

        # Mock script runner instance returning failure
        mock_script_runner_instance = Mock()
        mock_script_runner_instance.run_script.return_value = False
        mock_script_runner_class.return_value = mock_script_runner_instance

        runner = EvaluationRunner(mock_agent_client)
        result = runner.run_evaluation(sample_config_script, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_002"
        assert result.result == "FAIL"
        assert result.eval_type == "script"
        assert result.error is None

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_script_with_kubeconfig(
        self, mock_script_runner_class, mock_agent_client, sample_config_script
    ):
        """Test script evaluation with kubeconfig."""
        # Mock agent response
        mock_agent_client.query_agent.return_value = (
            "kubectl create deployment nginx --image=nginx"
        )

        # Mock script runner instance
        mock_script_runner_instance = Mock()
        mock_script_runner_instance.run_script.return_value = True
        mock_script_runner_class.return_value = mock_script_runner_instance

        runner = EvaluationRunner(mock_agent_client, kubeconfig="~/kubeconfig")
        result = runner.run_evaluation(sample_config_script, "openai", "gpt-4")

        assert result.result == "PASS"

        # Verify ScriptRunner was created with kubeconfig
        mock_script_runner_class.assert_called_with(kubeconfig="~/kubeconfig")
        # Verify script was executed
        mock_script_runner_instance.run_script.assert_called_once_with("./verify.sh")

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_script_execution_error(
        self, mock_script_runner_class, mock_agent_client, sample_config_script
    ):
        """Test script evaluation with execution error."""
        # Mock agent response
        mock_agent_client.query_agent.return_value = (
            "kubectl create deployment nginx --image=nginx"
        )

        # Mock script runner instance raising error
        mock_script_runner_instance = Mock()
        mock_script_runner_instance.run_script.side_effect = ScriptExecutionError(
            "Script failed"
        )
        mock_script_runner_class.return_value = mock_script_runner_instance

        runner = EvaluationRunner(mock_agent_client)
        result = runner.run_evaluation(sample_config_script, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_002"
        assert result.result == "FAIL"
        assert result.error is None

    def test_run_evaluation_substring_success(
        self, mock_agent_client, sample_config_substring
    ):
        """Test successful substring evaluation."""
        # Mock agent response containing expected keywords
        mock_agent_client.query_agent.return_value = "Docker is a container platform"

        runner = EvaluationRunner(mock_agent_client)
        result = runner.run_evaluation(sample_config_substring, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_003"
        assert result.result == "PASS"
        assert result.eval_type == "sub-string"

    def test_run_evaluation_substring_failure(
        self, mock_agent_client, sample_config_substring
    ):
        """Test substring evaluation failure."""
        # Mock agent response not containing expected keywords
        mock_agent_client.query_agent.return_value = "This is about virtual machines"

        runner = EvaluationRunner(mock_agent_client)
        result = runner.run_evaluation(sample_config_substring, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_003"
        assert result.result == "FAIL"
        assert result.eval_type == "sub-string"

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_with_setup_script(
        self, mock_script_runner_class, mock_agent_client, mock_judge_manager
    ):
        """Test evaluation with setup script."""
        config = EvaluationDataConfig(
            eval_id="test_setup",
            eval_query="Test query",
            eval_type="judge-llm",
            expected_response="Test response",
            eval_setup_script="./setup.sh",
        )

        # Mock script runner instance for setup
        mock_script_runner_instance = Mock()
        mock_script_runner_instance.run_script.return_value = True
        mock_script_runner_class.return_value = mock_script_runner_instance

        # Mock agent and judge responses
        mock_agent_client.query_agent.return_value = "Test response"
        mock_judge_manager.evaluate_response.return_value = "1"

        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        result = runner.run_evaluation(config, "openai", "gpt-4")

        assert result.result == "PASS"
        # Verify setup script was called
        mock_script_runner_instance.run_script.assert_called_with("./setup.sh")

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_setup_script_failure(
        self, mock_script_runner_class, mock_agent_client, mock_judge_manager
    ):
        """Test evaluation with setup script failure."""
        config = EvaluationDataConfig(
            eval_id="test_setup_fail",
            eval_query="Test query",
            eval_type="judge-llm",
            expected_response="Test response",
            eval_setup_script="./setup.sh",
        )

        # Mock failing setup script execution
        mock_script_runner_instance = Mock()
        mock_script_runner_instance.run_script.return_value = False
        mock_script_runner_class.return_value = mock_script_runner_instance

        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        result = runner.run_evaluation(config, "openai", "gpt-4")

        assert result.result == "ERROR"
        assert "Setup script failed" in result.error

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_with_cleanup_script(
        self, mock_script_runner_class, mock_agent_client, mock_judge_manager
    ):
        """Test evaluation with cleanup script."""
        config = EvaluationDataConfig(
            eval_id="test_cleanup",
            eval_query="Test query",
            eval_type="judge-llm",
            expected_response="Test response",
            eval_cleanup_script="./cleanup.sh",
        )

        # Mock successful cleanup script execution
        mock_script_runner_instance = Mock()
        mock_script_runner_instance.run_script.return_value = True
        mock_script_runner_class.return_value = mock_script_runner_instance

        # Mock agent and judge responses
        mock_agent_client.query_agent.return_value = "Test response"
        mock_judge_manager.evaluate_response.return_value = "1"

        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        result = runner.run_evaluation(config, "openai", "gpt-4")

        assert result.result == "PASS"
        # Verify cleanup script was called
        mock_script_runner_instance.run_script.assert_called_with("./cleanup.sh")

    def test_run_evaluation_agent_api_error(
        self, mock_agent_client, mock_judge_manager, sample_config_judge_llm
    ):
        """Test evaluation with agent API error."""
        # Mock agent API error
        mock_agent_client.query_agent.side_effect = AgentAPIError(
            "API connection failed"
        )

        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        result = runner.run_evaluation(sample_config_judge_llm, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.result == "ERROR"
        assert "API connection failed" in result.error

    def test_run_evaluation_unknown_type(self, mock_agent_client):
        """Test evaluation with unknown evaluation type."""
        config = EvaluationDataConfig(
            eval_id="test_unknown",
            eval_query="Test query",
            eval_type="unknown-type",
        )

        # Mock agent response
        mock_agent_client.query_agent.return_value = "Test response"

        runner = EvaluationRunner(mock_agent_client)
        result = runner.run_evaluation(config, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.result == "FAIL"

    def test_get_judge_manager(self, mock_agent_client, mock_judge_manager):
        """Test get_judge_manager method."""
        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        assert runner.get_judge_manager() == mock_judge_manager

        runner_no_judge = EvaluationRunner(mock_agent_client)
        assert runner_no_judge.get_judge_manager() is None

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_judge_llm_failure(
        self,
        mock_script_runner,
        mock_agent_client,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test judge-llm evaluation failure."""
        # Mock agent response
        mock_agent_client.query_agent.return_value = "Some incorrect response"

        # Mock judge response indicating failure
        mock_judge_manager.evaluate_response.return_value = "0"

        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        result = runner.run_evaluation(sample_config_judge_llm, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_001"
        assert result.result == "FAIL"
        assert result.eval_type == "judge-llm"
        assert result.error is None

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.ScriptRunner")
    def test_run_evaluation_judge_llm_error(
        self,
        mock_script_runner,
        mock_agent_client,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test judge-llm evaluation error."""
        # Mock agent response
        mock_agent_client.query_agent.return_value = "Some incorrect response"

        # Mock judge response indicating failure
        mock_judge_manager.evaluate_response.return_value = "00"

        runner = EvaluationRunner(mock_agent_client, mock_judge_manager)
        result = runner.run_evaluation(sample_config_judge_llm, "openai", "gpt-4")

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_001"
        assert result.result == "ERROR"
        assert result.eval_type == "judge-llm"
        assert result.error == (
            "Invalid response from the judge model. "
            "Expected value either 0/1. Actual value: 00"
        )

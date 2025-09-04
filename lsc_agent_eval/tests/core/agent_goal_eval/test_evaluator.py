"""Tests for evaluation runner."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from lsc_agent_eval.core.agent_goal_eval.evaluator import EvaluationRunner
from lsc_agent_eval.core.agent_goal_eval.models import (
    EvaluationDataConfig,
    EvaluationResult,
)
from lsc_agent_eval.core.agent_goal_eval.script_runner import ScriptRunner
from lsc_agent_eval.core.utils.api_client import AgentHttpClient
from lsc_agent_eval.core.utils.exceptions import AgentAPIError, ScriptExecutionError
from lsc_agent_eval.core.utils.judge import JudgeModelManager


class TestEvaluationRunner:
    """Test EvaluationRunner."""

    @pytest.fixture
    def mock_agent_client(self):
        """Mock agent client."""
        mock_client = Mock(spec=AgentHttpClient)

        # Mock agent API: return conversation_id from input or generate one
        def mock_agent_response(api_input, **kwargs):
            return {
                "response": "Test agent response",
                "conversation_id": api_input.get(
                    "conversation_id", "generated-conversation-id"
                ),
                "tool_calls": [],  # Always return empty tool sequences by default
            }

        mock_client.streaming_query_agent.side_effect = mock_agent_response
        mock_client.query_agent.side_effect = mock_agent_response
        return mock_client

    @pytest.fixture
    def mock_script_runner(self):
        """Mock script runner."""
        mock_runner = Mock(spec=ScriptRunner)
        mock_runner.run_script.return_value = True
        return mock_runner

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
            eval_query="What is Openshift Virtualization?",
            eval_types=["response_eval:accuracy"],
            expected_response="OpenShift Virtualization is an extension of the OpenShift Container Platform",
        )

    @pytest.fixture
    def get_test_script_path(self):
        """Create a temporary test script file and cleanup."""
        # Setup
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write('#!/bin/bash\necho "test script"\nexit 0')
            script_path = f.name
        os.chmod(script_path, 0o755)

        yield script_path

        # Cleanup
        os.unlink(script_path)

    @pytest.fixture
    def sample_config_script(self, get_test_script_path):
        """Sample script evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_002",
            eval_query="Deploy nginx",
            eval_types=["action_eval"],
            eval_verify_script=get_test_script_path,
        )

    @pytest.fixture
    def sample_config_substring(self):
        """Sample sub-string evaluation configuration."""
        return EvaluationDataConfig(
            eval_id="test_003",
            eval_query="What is Podman?",
            eval_types=["response_eval:sub-string"],
            expected_keywords=["container", "podman"],
        )

    def test_init(self, mock_agent_client, mock_script_runner, mock_judge_manager):
        """Test EvaluationRunner initialization."""
        runner = EvaluationRunner(
            mock_agent_client,
            mock_script_runner,
            mock_judge_manager,
        )

        assert runner.agent_client == mock_agent_client
        assert runner.script_runner == mock_script_runner
        assert runner.judge_manager == mock_judge_manager

    def test_init_without_judge_manager(self, mock_agent_client, mock_script_runner):
        """Test EvaluationRunner initialization without judge manager."""
        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        assert runner.agent_client == mock_agent_client
        assert runner.script_runner == mock_script_runner
        assert runner.judge_manager is None

    def test_run_evaluation_judge_llm_success(
        self,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test successful judge-llm evaluation."""
        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        results = runner.run_evaluation(
            sample_config_judge_llm,
            "watsonx",
            "ibm/granite-3-3-8b-instruct",
            "conv-id-123",
        )

        assert len(results) == 1

        result = results[0]
        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_001"
        assert result.query == "What is Openshift Virtualization?"
        assert result.eval_type == "response_eval:accuracy"
        assert result.result == "PASS"
        assert result.conversation_id == "conv-id-123"
        assert result.error is None
        assert result.tool_calls is None

        # Verify agent was called
        mock_agent_client.streaming_query_agent.assert_called_once_with(
            {
                "query": "What is Openshift Virtualization?",
                "provider": "watsonx",
                "model": "ibm/granite-3-3-8b-instruct",
                "conversation_id": "conv-id-123",
            }
        )
        mock_agent_client.query_agent.assert_not_called()

        # Verify judge was called
        mock_judge_manager.evaluate_response.assert_called_once()

    def test_run_evaluation_judge_llm_failure(
        self,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test failed judge-llm evaluation."""
        # Mock judge to return 0 (failure)
        mock_judge_manager.evaluate_response.return_value = "0"

        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        results = runner.run_evaluation(
            sample_config_judge_llm,
            "openai",
            "gpt-4",
            "conv-id-123",
        )

        result = results[0]
        assert result.result == "FAIL"
        assert result.error is None

    def test_run_evaluation_script_success(
        self, mock_agent_client, mock_script_runner, sample_config_script
    ):
        """Test successful script evaluation."""
        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_script,
            "openai",
            "gpt-4",
            "conv-id-123",
        )[0]

        assert isinstance(result, EvaluationResult)
        assert result.eval_id == "test_002"
        assert result.eval_type == "action_eval"
        assert result.result == "PASS"
        assert result.error is None

        # Verify agent was called
        mock_agent_client.streaming_query_agent.assert_called_once()

        # Verify script was run
        mock_script_runner.run_script.assert_called_once_with(
            sample_config_script.eval_verify_script
        )

    def test_run_evaluation_script_failure(
        self, mock_agent_client, mock_script_runner, sample_config_script
    ):
        """Test failed script evaluation."""
        # Mock script to return False (failure)
        mock_script_runner.run_script.return_value = False

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_script,
            "openai",
            "gpt-4",
            "conv-id-123",
        )[0]

        assert result.result == "FAIL"
        assert result.error is None

    def test_run_evaluation_script_execution_error(
        self, mock_agent_client, mock_script_runner, sample_config_script
    ):
        """Test script evaluation with execution error."""
        # Mock script to raise exception
        mock_script_runner.run_script.side_effect = ScriptExecutionError(
            "Script failed"
        )

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        result = runner.run_evaluation(
            sample_config_script,
            "openai",
            "gpt-4",
            "conv-id-123",
        )[0]

        assert result.result == "ERROR"
        assert "Script failed" in result.error

    def test_run_evaluation_substring_success(
        self, mock_agent_client, mock_script_runner, sample_config_substring
    ):
        """Test successful sub-string evaluation."""

        # Mock agent response containing expected keywords
        def mock_streaming_query_agent(api_input, timeout=300):
            return {
                "response": "Podman is an open-source container engine developed by Red Hat",
                "conversation_id": api_input.get(
                    "conversation_id", "test-conversation-id"
                ),
                "tool_calls": [],
            }

        mock_agent_client.streaming_query_agent.side_effect = mock_streaming_query_agent

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        results = runner.run_evaluation(
            sample_config_substring,
            "openai",
            "gpt-4",
            "conv-id-123",
        )

        result = results[0]
        assert result.eval_id == "test_003"
        assert result.result == "PASS"
        assert result.eval_type == "response_eval:sub-string"
        assert result.error is None

    def test_run_evaluation_substring_failure(
        self, mock_agent_client, mock_script_runner, sample_config_substring
    ):
        """Test sub-string evaluation failure."""

        # Mock agent response not containing expected keywords
        def mock_streaming_query_agent(api_input, timeout=300):
            return {
                "response": "No information available",
                "conversation_id": api_input.get(
                    "conversation_id", "test-conversation-id"
                ),
                "tool_calls": [],
            }

        mock_agent_client.streaming_query_agent.side_effect = mock_streaming_query_agent

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        results = runner.run_evaluation(
            sample_config_substring,
            "openai",
            "gpt-4",
            None,
        )

        result = results[0]
        assert result.eval_id == "test_003"
        assert result.result == "FAIL"
        assert result.eval_type == "response_eval:sub-string"
        assert result.error is None

    def test_run_evaluation_agent_api_error(
        self, mock_agent_client, mock_script_runner, sample_config_judge_llm
    ):
        """Test evaluation with agent API error."""
        # Mock agent client to raise API error
        mock_agent_client.streaming_query_agent.side_effect = AgentAPIError(
            "API connection failed"
        )

        runner = EvaluationRunner(mock_agent_client, mock_script_runner)

        results = runner.run_evaluation(
            sample_config_judge_llm,
            "openai",
            "gpt-4",
            "conv-id-123",
        )

        result = results[0]
        assert result.eval_id == "test_001"
        assert result.result == "ERROR"
        assert result.eval_type == "response_eval:accuracy"
        assert "API connection failed" in result.error

    def test_substring_evaluation_logic(
        self, mock_agent_client, mock_script_runner, mock_judge_manager
    ):
        """Test sub-string evaluation with different keyword combinations."""
        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        config = EvaluationDataConfig(
            eval_id="substring_test",
            eval_query="Test query",
            eval_types=["response_eval:sub-string"],
            expected_keywords=["keyword1", "keyword2"],
        )

        # Test all keywords present - should PASS
        def mock_streaming_query_agent_all_keywords(api_input, timeout=300):
            return {
                "response": "Response with keyword1 and keyword2",
                "conversation_id": api_input.get(
                    "conversation_id", "test-conversation-id"
                ),
                "tool_calls": [],
            }

        mock_agent_client.streaming_query_agent.side_effect = (
            mock_streaming_query_agent_all_keywords
        )
        result = runner.run_evaluation(config, "openai", "gpt-4", "conv-id-123")
        assert result[0].result == "PASS"

        # Test some keywords missing (only one present) - should FAIL
        def mock_streaming_query_agent_one_keyword(api_input, timeout=300):
            return {
                "response": "Response with only keyword1",
                "conversation_id": api_input.get(
                    "conversation_id", "test-conversation-id"
                ),
                "tool_calls": [],
            }

        mock_agent_client.streaming_query_agent.side_effect = (
            mock_streaming_query_agent_one_keyword
        )
        result = runner.run_evaluation(config, "openai", "gpt-4", "conv-id-123")
        assert result[0].result == "FAIL"

        # Test no keywords present - should FAIL
        def mock_streaming_query_agent_no_keywords(api_input, timeout=300):
            return {
                "response": "Response with no matching terms",
                "conversation_id": api_input.get(
                    "conversation_id", "test-conversation-id"
                ),
                "tool_calls": [],
            }

        mock_agent_client.streaming_query_agent.side_effect = (
            mock_streaming_query_agent_no_keywords
        )
        result = runner.run_evaluation(config, "openai", "gpt-4", "conv-id-123")
        assert result[0].result == "FAIL"

        # Test case insensitive matching
        def mock_streaming_query_agent_case_insensitive(api_input, timeout=300):
            return {
                "response": "Response with KEYWORD1 and Keyword2",
                "conversation_id": api_input.get(
                    "conversation_id", "test-conversation-id"
                ),
                "tool_calls": [],
            }

        mock_agent_client.streaming_query_agent.side_effect = (
            mock_streaming_query_agent_case_insensitive
        )
        result = runner.run_evaluation(config, "openai", "gpt-4", "conv-id-123")
        assert result[0].result == "PASS"

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.compare_tool_calls")
    def test_tool_eval_success(
        self,
        mock_compare_tool_calls,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
    ):
        """Test tool evaluation with success."""
        mock_compare_tool_calls.return_value = True
        mock_agent_client.streaming_query_agent.side_effect = (
            lambda api_input, **kwargs: {
                "response": "Available versions listed",
                "conversation_id": "conv-tools-1",
                "tool_calls": [[{"tool_name": "list_versions", "arguments": {}}]],
            }
        )

        config = EvaluationDataConfig(
            eval_id="tools_test",
            eval_query="List available versions",
            eval_types=["tool_eval"],
            expected_tool_calls=[[{"tool_name": "list_versions", "arguments": {}}]],
        )

        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )
        results = runner.run_evaluation(config, "openai", "gpt-4")

        assert len(results) == 1
        assert results[0].result == "PASS"
        assert results[0].eval_type == "tool_eval"
        assert results[0].tool_calls == [
            [{"tool_name": "list_versions", "arguments": {}}]
        ]
        mock_compare_tool_calls.assert_called_once_with(
            [[{"tool_name": "list_versions", "arguments": {}}]],
            [[{"tool_name": "list_versions", "arguments": {}}]],
        )

    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.compare_tool_calls")
    def test_tool_eval_failure(
        self,
        mock_compare_tool_calls,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
    ):
        """Test tool evaluation with failure."""
        mock_compare_tool_calls.return_value = False
        mock_agent_client.streaming_query_agent.side_effect = (
            lambda api_input, **kwargs: {
                "response": "Tool call failed",
                "conversation_id": "conv-tools-2",
                "tool_calls": [[{"tool_name": "wrong_tool", "arguments": {}}]],
            }
        )

        config = EvaluationDataConfig(
            eval_id="tools_fail_test",
            eval_query="Use correct tool",
            eval_types=["tool_eval"],
            expected_tool_calls=[[{"tool_name": "correct_tool", "arguments": {}}]],
        )

        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )
        results = runner.run_evaluation(config, "openai", "gpt-4")

        assert len(results) == 1
        assert results[0].result == "FAIL"
        assert results[0].eval_type == "tool_eval"
        assert results[0].tool_calls == [[{"tool_name": "wrong_tool", "arguments": {}}]]
        mock_compare_tool_calls.assert_called_once_with(
            [[{"tool_name": "correct_tool", "arguments": {}}]],
            [[{"tool_name": "wrong_tool", "arguments": {}}]],
        )

    def test_conversation_id_propagation(
        self, mock_agent_client, mock_script_runner, mock_judge_manager
    ):
        """Test that conversation ID is properly propagated to results."""
        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        config = EvaluationDataConfig(
            eval_id="conv_id_test",
            eval_query="Test query",
            eval_types=["response_eval:accuracy"],
            expected_response="Test response",
        )

        test_conv_id = "conv-id-456"
        result = runner.run_evaluation(config, "openai", "gpt-4", test_conv_id)

        assert result[0].conversation_id == test_conv_id

        # Verify ID was passed to agent client
        mock_agent_client.streaming_query_agent.assert_called_once_with(
            {
                "query": "Test query",
                "provider": "openai",
                "model": "gpt-4",
                "conversation_id": test_conv_id,
            }
        )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.compare_tool_calls")
    def test_multiple_eval_types_all_pass(
        self,
        mock_compare_tool_calls,
        mock_is_file,
        mock_exists,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
    ):
        """Test evaluation with multiple eval types where all pass."""
        mock_agent_client.streaming_query_agent.side_effect = (
            lambda api_input, timeout=300: {
                "response": "Successfully created openshift-lightspeed namespace",
                "conversation_id": "conv-123",
                "tool_calls": [
                    [
                        {
                            "tool_name": "create_namespace",
                            "arguments": {"name": "lightspeed"},
                        }
                    ]
                ],
            }
        )
        mock_script_runner.run_script.return_value = True
        mock_judge_manager.evaluate_response.return_value = "1"
        mock_compare_tool_calls.return_value = True

        config = EvaluationDataConfig(
            eval_id="multi_pass_test",
            eval_query="create openshift-lightspeed namespace",
            eval_types=[
                "action_eval",
                "response_eval:sub-string",
                "response_eval:accuracy",
                "tool_eval",
            ],
            eval_verify_script="sample_data/script/conv4/eval1/verify.sh",
            expected_keywords=[
                "successfully",
                "created",
                "lightspeed",
            ],  # All present in response
            expected_response="openshift-lightspeed namespace is created successfully",
            expected_tool_calls=[
                [{"tool_name": "create_namespace", "arguments": {"name": "lightspeed"}}]
            ],
        )

        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )
        results = runner.run_evaluation(config, "ollama", "gpt-oss:20b")

        # Should get 4 results, one for each eval_type
        assert len(results) == 4

        assert all(
            r.result == "PASS" for r in results
        ), f"Expected all PASS, got: {[(r.eval_type, r.result) for r in results]}"

        eval_types = [r.eval_type for r in results]
        assert "action_eval" in eval_types
        assert "response_eval:sub-string" in eval_types
        assert "response_eval:accuracy" in eval_types
        assert "tool_eval" in eval_types

        assert all(r.eval_id == "multi_pass_test" for r in results)
        assert all(r.query == "create openshift-lightspeed namespace" for r in results)
        assert all(r.conversation_id == "conv-123" for r in results)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("lsc_agent_eval.core.agent_goal_eval.evaluator.compare_tool_calls")
    def test_multiple_eval_types_mixed_results(
        self,
        mock_compare_tool_calls,
        mock_is_file,
        mock_exists,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
    ):
        """Test evaluation with multiple eval types having mixed results."""
        # Mock mixed results
        mock_agent_client.streaming_query_agent.side_effect = (
            lambda api_input, timeout=300: {
                "response": "Sorry, I can't create openshift-lightspeed namespace",
                "conversation_id": "conv-456",
                "tool_calls": [[{"tool_name": "wrong_tool", "arguments": {}}]],
            }
        )
        mock_script_runner.run_script.return_value = False  # Script fails
        mock_judge_manager.evaluate_response.return_value = "0"  # Accuracy fails
        mock_compare_tool_calls.return_value = False  # Tool eval fails

        config = EvaluationDataConfig(
            eval_id="multi_mixed_test",
            eval_query="create openshift-lightspeed namespace",
            eval_types=[
                "action_eval",
                "response_eval:sub-string",
                "response_eval:accuracy",
                "tool_eval",
            ],
            eval_verify_script="sample_data/script/conv4/eval1/verify.sh",
            expected_keywords=["lightspeed"],  # Only this should pass
            expected_response="openshift-lightspeed namespace is created successfully",
            expected_tool_calls=[
                [{"tool_name": "create_namespace", "arguments": {"name": "lightspeed"}}]
            ],
        )

        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )
        results = runner.run_evaluation(config, "some-provider", "some-model")

        assert len(results) == 4

        result_by_type = {r.eval_type: r for r in results}
        assert result_by_type["action_eval"].result == "FAIL"
        assert result_by_type["action_eval"].error is None
        assert result_by_type["response_eval:sub-string"].result == "PASS"
        assert result_by_type["response_eval:sub-string"].error is None
        assert result_by_type["response_eval:accuracy"].result == "FAIL"
        assert result_by_type["response_eval:accuracy"].error is None
        assert result_by_type["tool_eval"].result == "FAIL"
        assert result_by_type["tool_eval"].error is None

        assert all(r.eval_id == "multi_mixed_test" for r in results)
        assert all(r.query == "create openshift-lightspeed namespace" for r in results)
        assert all(r.conversation_id == "conv-456" for r in results)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_multiple_eval_types_with_error(
        self,
        mock_is_file,
        mock_exists,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
    ):
        """Test evaluation with multiple eval types where some have errors."""
        mock_agent_client.streaming_query_agent.side_effect = (
            lambda api_input, timeout=300: {
                "response": "Sorry, I can't create openshift-lightspeed namespace",
                "conversation_id": "conv-789",
                "tool_calls": [],
            }
        )
        # Script execution error
        mock_script_runner.run_script.side_effect = ScriptExecutionError(
            "Script file not found"
        )
        mock_judge_manager.evaluate_response.return_value = "0"

        config = EvaluationDataConfig(
            eval_id="multi_error_test",
            eval_query="create openshift-lightspeed namespace",
            eval_types=[
                "action_eval",
                "response_eval:sub-string",
                "response_eval:accuracy",
            ],
            eval_verify_script="sample_data/script/conv4/eval1/verify.sh",
            expected_keywords=["openshift"],
            expected_response="openshift-lightspeed namespace is created successfully",
        )

        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )
        results = runner.run_evaluation(config, "some-provider", "some-model")

        assert len(results) == 3

        result_by_type = {r.eval_type: r for r in results}

        assert result_by_type["action_eval"].result == "ERROR"
        assert "Script file not found" in result_by_type["action_eval"].error

        assert result_by_type["response_eval:sub-string"].result == "PASS"

        assert result_by_type["response_eval:accuracy"].result == "FAIL"

    def test_run_evaluation_streaming_vs_query_endpoints(
        self,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test that both streaming and query endpoint modes work correctly."""
        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )

        # Test streaming mode (default)
        results_streaming = runner.run_evaluation(
            sample_config_judge_llm,
            "watsonx",
            "ibm/granite-3-3-8b-instruct",
            "conv-id-123",
            endpoint_type="streaming",
        )

        # Test query mode
        results_query = runner.run_evaluation(
            sample_config_judge_llm,
            "watsonx",
            "ibm/granite-3-3-8b-instruct",
            "conv-id-123",
            endpoint_type="query",
        )

        # Both should return the same results
        assert len(results_streaming) == len(results_query) == 1
        assert results_streaming[0].result == results_query[0].result == "PASS"

        mock_agent_client.streaming_query_agent.assert_called_once()
        mock_agent_client.query_agent.assert_called_once()

    def test_run_evaluation_invalid_endpoint_type_returns_error(
        self,
        mock_agent_client,
        mock_script_runner,
        mock_judge_manager,
        sample_config_judge_llm,
    ):
        """Test invalid endpoint type."""
        runner = EvaluationRunner(
            mock_agent_client, mock_script_runner, mock_judge_manager
        )
        results = runner.run_evaluation(
            sample_config_judge_llm,
            "openai",
            "gpt-4",
            "conv-id-xyz",
            endpoint_type="invalid-endpoint",
        )
        assert len(results) == 1
        assert results[0].result == "ERROR"
        assert results[0].error is not None
        # No agent call should have been made
        mock_agent_client.streaming_query_agent.assert_not_called()
        mock_agent_client.query_agent.assert_not_called()

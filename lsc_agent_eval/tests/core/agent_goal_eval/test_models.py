"""Tests for agent evaluation data models."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from lsc_agent_eval.core.agent_goal_eval.models import (
    ConversationDataConfig,
    EvaluationDataConfig,
    EvaluationResult,
    EvaluationStats,
)


class TestEvaluationResult:
    """Test EvaluationResult data class."""

    def test_evaluation_result_creation(self):
        """Test creating EvaluationResult instance."""
        result = EvaluationResult(
            eval_id="test_001",
            query="What is Kubernetes?",
            response="Kubernetes is a container orchestration platform",
            eval_type="response_eval:accuracy",
            result="PASS",
            error=None,
        )

        assert result.eval_id == "test_001"
        assert result.query == "What is Kubernetes?"
        assert result.response == "Kubernetes is a container orchestration platform"
        assert result.eval_type == "response_eval:accuracy"
        assert result.result == "PASS"
        assert result.error is None

    def test_evaluation_result_with_error(self):
        """Test EvaluationResult with error."""
        result = EvaluationResult(
            eval_id="test_002",
            query="Deploy nginx",
            response="",
            eval_type="action_eval",
            result="ERROR",
            error="Script execution failed",
        )

        assert result.eval_id == "test_002"
        assert result.query == "Deploy nginx"
        assert result.response == ""
        assert result.eval_type == "action_eval"
        assert result.result == "ERROR"
        assert result.error == "Script execution failed"

    def test_evaluation_result_defaults(self):
        """Test EvaluationResult with default values."""
        result = EvaluationResult(
            eval_id="test_003",
            query="Test query",
            response="Test response",
            eval_type="response_eval:sub-string",
            result="PASS",
        )

        assert result.error is None

    def test_evaluation_result_invalid_result_type(self):
        """Test EvaluationResult with invalid result type."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                eval_id="test_004",
                query="Test query",
                response="Test response",
                eval_type="response_eval:accuracy",
                result="INVALID",
            )

        assert "Result must be one of" in str(exc_info.value)

    def test_evaluation_result_invalid_eval_type(self):
        """Test EvaluationResult with invalid eval type."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                eval_id="test_005",
                query="Test query",
                response="Test response",
                eval_type="invalid-type",
                result="PASS",
            )

        assert "eval_type must be one of" in str(exc_info.value)

    def test_evaluation_results_multiple_eval_types(self):
        """Test multiple EvaluationResult instances with different eval types for same query."""
        # Mock results
        results = [
            EvaluationResult(
                eval_id="multi_001",
                query="create openshift-lightspeed namespace",
                response="Sorry, I can't create openshift-lightspeed namespace",
                eval_type="action_eval",
                result="ERROR",
                conversation_group="test_conv",
                conversation_id="conv-123",
            ),
            EvaluationResult(
                eval_id="multi_001",
                query="create openshift-lightspeed namespace",
                response="Sorry, I can't create openshift-lightspeed namespace",
                eval_type="response_eval:sub-string",
                result="PASS",
                conversation_group="test_conv",
                conversation_id="conv-123",
            ),
            EvaluationResult(
                eval_id="multi_001",
                query="create openshift-lightspeed namespace",
                response="Sorry, I can't create openshift-lightspeed namespace",
                eval_type="response_eval:accuracy",
                result="FAIL",
                conversation_group="test_conv",
                conversation_id="conv-123",
            ),
        ]

        # Verify all results have same core info but different eval_types
        assert all(r.eval_id == "multi_001" for r in results)
        assert all(r.query == "create openshift-lightspeed namespace" for r in results)
        assert all(r.conversation_id == "conv-123" for r in results)

        # Verify different eval_types and results
        eval_types = [r.eval_type for r in results]
        assert "action_eval" in eval_types
        assert "response_eval:sub-string" in eval_types
        assert "response_eval:accuracy" in eval_types

        # Verify individual results
        assert results[0].result == "ERROR"  # action_eval
        assert results[1].result == "PASS"  # response_eval:sub-string
        assert results[2].result == "FAIL"  # response_eval:accuracy


class TestEvaluationDataConfig:
    """Test EvaluationDataConfig data class."""

    def test_evaluation_data_config_judge_llm(self):
        """Test EvaluationDataConfig for judge-llm evaluation."""
        config = EvaluationDataConfig(
            eval_id="judge_test",
            eval_query="Explain containers",
            eval_types=["response_eval:accuracy"],
            expected_response="Containers are lightweight virtualization",
        )

        assert config.eval_id == "judge_test"
        assert config.eval_query == "Explain containers"
        assert config.eval_types == ["response_eval:accuracy"]
        assert config.expected_response == "Containers are lightweight virtualization"
        assert config.expected_keywords is None
        assert config.eval_verify_script is None
        assert config.description is None

    @patch("builtins.open", mock_open())
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_evaluation_data_config_script(self, mock_exists, mock_is_file):
        """Test EvaluationDataConfig for script evaluation."""
        config = EvaluationDataConfig(
            eval_id="script_test",
            eval_query="Deploy nginx pod",
            eval_types=["action_eval"],
            eval_verify_script="/mock/script/path.sh",
        )

        assert config.eval_id == "script_test"
        assert config.eval_query == "Deploy nginx pod"
        assert config.eval_types == ["action_eval"]
        assert config.expected_response is None
        assert config.expected_keywords is None
        assert isinstance(config.eval_verify_script, Path)

        # Verify path validation was called
        mock_exists.assert_called()
        mock_is_file.assert_called()

    def test_evaluation_data_config_substring(self):
        """Test EvaluationDataConfig for sub-string evaluation."""
        config = EvaluationDataConfig(
            eval_id="substring_test",
            eval_query="List container benefits",
            eval_types=["response_eval:sub-string"],
            expected_keywords=["isolation", "portability", "efficiency"],
        )

        assert config.eval_id == "substring_test"
        assert config.eval_query == "List container benefits"
        assert config.eval_types == ["response_eval:sub-string"]
        assert config.expected_response is None
        assert config.expected_keywords == ["isolation", "portability", "efficiency"]
        assert config.eval_verify_script is None

    def test_evaluation_data_config_with_description(self):
        """Test EvaluationDataConfig with description."""
        config = EvaluationDataConfig(
            eval_id="full_test",
            eval_query="What is OpenShift?",
            eval_types=["response_eval:accuracy"],
            expected_response="OpenShift is a Kubernetes platform",
            description="Test evaluation for OpenShift knowledge",
        )

        assert config.eval_id == "full_test"
        assert config.eval_query == "What is OpenShift?"
        assert config.eval_types == ["response_eval:accuracy"]
        assert config.expected_response == "OpenShift is a Kubernetes platform"
        assert config.description == "Test evaluation for OpenShift knowledge"
        assert config.expected_keywords is None
        assert config.eval_verify_script is None

    @patch("builtins.open", mock_open())
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_evaluation_data_config_multiple_eval_types(
        self, mock_exists, mock_is_file
    ):
        """Test EvaluationDataConfig with multiple eval types."""
        config = EvaluationDataConfig(
            eval_id="multi_001",
            eval_query="create openshift-lightspeed namespace",
            eval_types=[
                "action_eval",
                "response_eval:sub-string",
                "response_eval:accuracy",
            ],
            eval_verify_script="/mock/script/path.sh",
            expected_keywords=["openshift-lightspeed", "created"],
            expected_response="openshift-lightspeed namespace is successfully created",
            description="Multi-evaluation test",
        )

        assert config.eval_id == "multi_001"
        assert config.eval_query == "create openshift-lightspeed namespace"
        assert len(config.eval_types) == 3
        assert "action_eval" in config.eval_types
        assert "response_eval:sub-string" in config.eval_types
        assert "response_eval:accuracy" in config.eval_types
        assert config.expected_keywords == ["openshift-lightspeed", "created"]
        assert (
            config.expected_response
            == "openshift-lightspeed namespace is successfully created"
        )
        assert config.description == "Multi-evaluation test"
        assert isinstance(config.eval_verify_script, Path)

    def test_evaluation_data_config_tool_eval(self):
        """Test EvaluationDataConfig for tools evaluation."""
        expected_tool_calls = [[{"tool_name": "list_versions", "arguments": {}}]]

        config = EvaluationDataConfig(
            eval_id="test_tools",
            eval_query="List available versions",
            eval_types=["tool_eval"],
            expected_tool_calls=expected_tool_calls,
        )

        assert config.eval_id == "test_tools"
        assert config.eval_query == "List available versions"
        assert config.eval_types == ["tool_eval"]
        assert config.expected_tool_calls == expected_tool_calls
        assert config.expected_response is None
        assert config.expected_keywords is None
        assert config.eval_verify_script is None

    def test_evaluation_data_config_tool_missing_expected_tool_calls(self):
        """Test EvaluationDataConfig for tools evaluation without expected_tool_calls."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_tools_missing",
                eval_query="List versions",
                eval_types=["tool_eval"],
            )

        assert "requires non-empty 'expected_tool_calls'" in str(exc_info.value)

    def test_evaluation_data_config_tools_empty_expected_tool_calls(self):
        """Test EvaluationDataConfig for tools evaluation with empty expected_tool_calls."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_tools_empty",
                eval_query="List versions",
                eval_types=["tool_eval"],
                expected_tool_calls=[],
            )

        assert "at least 1 item" in str(exc_info.value)

    def test_evaluation_data_config_complex_tools(self):
        """Test EvaluationDataConfig with complex tool calls."""
        expected_tool_calls = [
            [
                {
                    "tool_name": "create_pod",
                    "arguments": {"name": "test-pod", "image": "nginx"},
                },
                {
                    "tool_name": "apply_service",
                    "arguments": {"pod_name": "test-pod", "port": 80},
                },
            ],
            [
                {
                    "tool_name": "oc_get",
                    "arguments": {"oc_get_args": ["pods"]},
                }
            ],
        ]

        config = EvaluationDataConfig(
            eval_id="complex_tools_test",
            eval_query="Create pod with service and check status",
            eval_types=["tool_eval"],
            expected_tool_calls=expected_tool_calls,
        )

        assert config.expected_tool_calls == expected_tool_calls
        assert len(config.expected_tool_calls) == 2
        assert len(config.expected_tool_calls[0]) == 2  # Two tools in first sequence
        assert len(config.expected_tool_calls[1]) == 1  # One tool in second sequence

    def test_evaluation_data_config_missing_eval_type(self):
        """Test EvaluationDataConfig with missing eval_type (should fail)."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_001",
                eval_query="What is Kubernetes?",
            )

        assert "Field required" in str(exc_info.value)

    def test_evaluation_data_config_judge_llm_missing_expected_response(self):
        """Test judge-llm evaluation missing expected_response."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_judge",
                eval_query="Test query",
                eval_types=["response_eval:accuracy"],
            )

        assert "requires non-empty 'expected_response'" in str(exc_info.value)

    def test_evaluation_data_config_substring_missing_keywords(self):
        """Test sub-string evaluation missing expected_keywords."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_substring",
                eval_query="Test query",
                eval_types=["response_eval:sub-string"],
            )

        assert "requires non-empty 'expected_keywords'" in str(exc_info.value)

    def test_evaluation_data_config_script_missing_verify_script(self):
        """Test script evaluation missing eval_verify_script."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_script",
                eval_query="Test query",
                eval_types=["action_eval"],
            )

        assert "requires non-empty 'eval_verify_script'" in str(exc_info.value)

    def test_evaluation_data_config_script_nonexistent_file(self):
        """Test script evaluation with non-existent script file."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_script",
                eval_query="Test query",
                eval_types=["action_eval"],
                eval_verify_script="/non/existent/script.sh",
            )

        assert "file not found" in str(exc_info.value)


class TestConversationDataConfig:
    """Test Conversation data config."""

    def test_conversation_config_minimal(self):
        """Test creating minimal Conversation data config."""
        config = ConversationDataConfig(
            conversation_group="test_conv",
            conversation=[
                EvaluationDataConfig(
                    eval_id="test_001",
                    eval_query="What is Kubernetes?",
                    eval_types=["response_eval:accuracy"],
                    expected_response="Kubernetes is a platform",
                )
            ],
        )

        assert config.conversation_group == "test_conv"
        assert len(config.conversation) == 1
        assert config.conversation[0].eval_id == "test_001"
        assert config.description is None
        assert config.setup_script is None
        assert config.cleanup_script is None

    @patch("builtins.open", mock_open())
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_conversation_config_with_scripts(self, mock_exists, mock_is_file):
        """Test Conversation data config with setup and cleanup scripts."""
        config = ConversationDataConfig(
            conversation_group="test_conv_scripts",
            description="Test conversation with scripts",
            setup_script="/mock/setup.sh",
            cleanup_script="/mock/cleanup.sh",
            conversation=[
                EvaluationDataConfig(
                    eval_id="test_001",
                    eval_query="Test query",
                    eval_types=["response_eval:accuracy"],
                    expected_response="Test response",
                )
            ],
        )

        assert config.conversation_group == "test_conv_scripts"
        assert config.description == "Test conversation with scripts"
        assert isinstance(config.setup_script, Path)
        assert isinstance(config.cleanup_script, Path)

    def test_conversation_config_empty_group_name(self):
        """Test Conversation data config with empty group name."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationDataConfig(
                conversation_group="   ",  # Empty after strip
                conversation=[
                    EvaluationDataConfig(
                        eval_id="test_001",
                        eval_query="Test query",
                        eval_types=["response_eval:accuracy"],
                        expected_response="Test response",
                    )
                ],
            )

        assert "cannot be empty" in str(exc_info.value)

    def test_conversation_config_nonexistent_script(self):
        """Test Conversation data config with non-existent script."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationDataConfig(
                conversation_group="test_conv",
                setup_script="/non/existent/setup.sh",
                conversation=[
                    EvaluationDataConfig(
                        eval_id="test_001",
                        eval_query="Test query",
                        eval_types=["response_eval:accuracy"],
                        expected_response="Test response",
                    )
                ],
            )

        assert "file not found" in str(exc_info.value)

    def test_conversation_config_duplicate_eval_ids(self):
        """Test Conversation data config with duplicate eval_ids."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationDataConfig(
                conversation_group="test_conv",
                conversation=[
                    EvaluationDataConfig(
                        eval_id="duplicate_id",
                        eval_query="First query",
                        eval_types=["response_eval:accuracy"],
                        expected_response="First response",
                    ),
                    EvaluationDataConfig(
                        eval_id="duplicate_id",
                        eval_query="Second query",
                        eval_types=["response_eval:accuracy"],
                        expected_response="Second response",
                    ),
                ],
            )

        assert "Duplicate eval_id" in str(exc_info.value)


class TestEvaluationStats:
    """Test Evaluation statistics data class."""

    def test_evaluation_stats_from_results(self):
        """Test Evaluation statistics creation method."""
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Query 1",
                response="Response 1",
                eval_type="response_eval:accuracy",
                result="PASS",
                conversation_group="conv1",
            ),
            EvaluationResult(
                eval_id="test_002",
                query="Query 2",
                response="Response 2",
                eval_type="action_eval",
                result="FAIL",
                conversation_group="conv1",
            ),
            EvaluationResult(
                eval_id="test_003",
                query="Query 3",
                response="Response 3",
                eval_type="response_eval:sub-string",
                result="PASS",
                conversation_group="conv2",
            ),
        ]

        stats = EvaluationStats.from_results(results)

        assert stats.total_evaluations == 3
        assert stats.total_conversations == 2
        assert stats.passed == 2
        assert stats.failed == 1
        assert stats.errored == 0
        assert abs(stats.success_rate - 66.67) < 0.01

        # Check stats by conversation
        assert "conv1" in stats.by_conversation
        assert "conv2" in stats.by_conversation
        assert stats.by_conversation["conv1"]["total"] == 2
        assert stats.by_conversation["conv1"]["passed"] == 1
        assert stats.by_conversation["conv2"]["total"] == 1
        assert stats.by_conversation["conv2"]["passed"] == 1

        # Check stats by eval_type
        assert "response_eval:accuracy" in stats.by_eval_type
        assert "action_eval" in stats.by_eval_type
        assert "response_eval:sub-string" in stats.by_eval_type

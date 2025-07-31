"""Tests for agent evaluation models."""

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
            eval_type="judge-llm",
            result="PASS",
            error=None,
        )

        assert result.eval_id == "test_001"
        assert result.query == "What is Kubernetes?"
        assert result.response == "Kubernetes is a container orchestration platform"
        assert result.eval_type == "judge-llm"
        assert result.result == "PASS"
        assert result.error is None

    def test_evaluation_result_with_error(self):
        """Test EvaluationResult with error."""
        result = EvaluationResult(
            eval_id="test_002",
            query="Deploy nginx",
            response="",
            eval_type="script",
            result="ERROR",
            error="Script execution failed",
        )

        assert result.eval_id == "test_002"
        assert result.query == "Deploy nginx"
        assert result.response == ""
        assert result.eval_type == "script"
        assert result.result == "ERROR"
        assert result.error == "Script execution failed"

    def test_evaluation_result_defaults(self):
        """Test EvaluationResult with default values."""
        result = EvaluationResult(
            eval_id="test_003",
            query="Test query",
            response="Test response",
            eval_type="sub-string",
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
                eval_type="judge-llm",
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


class TestEvaluationDataConfig:
    """Test EvaluationDataConfig data class."""

    def test_evaluation_data_config_judge_llm(self):
        """Test EvaluationDataConfig for judge-llm evaluation."""
        config = EvaluationDataConfig(
            eval_id="judge_test",
            eval_query="Explain containers",
            eval_type="judge-llm",
            expected_response="Containers are lightweight virtualization",
        )

        assert config.eval_id == "judge_test"
        assert config.eval_query == "Explain containers"
        assert config.eval_type == "judge-llm"
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
            eval_type="script",
            eval_verify_script="/mock/script/path.sh",
        )

        assert config.eval_id == "script_test"
        assert config.eval_query == "Deploy nginx pod"
        assert config.eval_type == "script"
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
            eval_type="sub-string",
            expected_keywords=["isolation", "portability", "efficiency"],
        )

        assert config.eval_id == "substring_test"
        assert config.eval_query == "List container benefits"
        assert config.eval_type == "sub-string"
        assert config.expected_response is None
        assert config.expected_keywords == ["isolation", "portability", "efficiency"]
        assert config.eval_verify_script is None

    def test_evaluation_data_config_with_description(self):
        """Test EvaluationDataConfig with description."""
        config = EvaluationDataConfig(
            eval_id="full_test",
            eval_query="What is OpenShift?",
            eval_type="judge-llm",
            expected_response="OpenShift is a Kubernetes platform",
            description="Test evaluation for OpenShift knowledge",
        )

        assert config.eval_id == "full_test"
        assert config.eval_query == "What is OpenShift?"
        assert config.eval_type == "judge-llm"
        assert config.expected_response == "OpenShift is a Kubernetes platform"
        assert config.description == "Test evaluation for OpenShift knowledge"
        assert config.expected_keywords is None
        assert config.eval_verify_script is None

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
                eval_type="judge-llm",
            )

        assert "requires non-empty 'expected_response'" in str(exc_info.value)

    def test_evaluation_data_config_substring_missing_keywords(self):
        """Test sub-string evaluation missing expected_keywords."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_substring",
                eval_query="Test query",
                eval_type="sub-string",
            )

        assert "requires non-empty 'expected_keywords'" in str(exc_info.value)

    def test_evaluation_data_config_script_missing_verify_script(self):
        """Test script evaluation missing eval_verify_script."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_script",
                eval_query="Test query",
                eval_type="script",
            )

        assert "requires non-empty 'eval_verify_script'" in str(exc_info.value)

    def test_evaluation_data_config_script_nonexistent_file(self):
        """Test script evaluation with non-existent script file."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationDataConfig(
                eval_id="test_script",
                eval_query="Test query",
                eval_type="script",
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
                    eval_type="judge-llm",
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
                    eval_type="judge-llm",
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
                        eval_type="judge-llm",
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
                        eval_type="judge-llm",
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
                        eval_type="judge-llm",
                        expected_response="First response",
                    ),
                    EvaluationDataConfig(
                        eval_id="duplicate_id",
                        eval_query="Second query",
                        eval_type="judge-llm",
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
                eval_type="judge-llm",
                result="PASS",
                conversation_group="conv1",
            ),
            EvaluationResult(
                eval_id="test_002",
                query="Query 2",
                response="Response 2",
                eval_type="script",
                result="FAIL",
                conversation_group="conv1",
            ),
            EvaluationResult(
                eval_id="test_003",
                query="Query 3",
                response="Response 3",
                eval_type="sub-string",
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
        assert "judge-llm" in stats.by_eval_type
        assert "script" in stats.by_eval_type
        assert "sub-string" in stats.by_eval_type

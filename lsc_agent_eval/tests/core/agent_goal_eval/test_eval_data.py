"""Tests for evaluation data manager."""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from lsc_agent_eval.core.agent_goal_eval.eval_data import AgentGoalEvalDataManager
from lsc_agent_eval.core.agent_goal_eval.models import (
    ConversationDataConfig,
    EvaluationDataConfig,
)
from lsc_agent_eval.core.utils.exceptions import EvaluationDataError


class TestAgentGoalEvalDataManager:
    """Test AgentGoalEvalDataManager."""

    @pytest.fixture
    def valid_conversation_yaml_content(self):
        """Valid YAML content with conversation-based structure."""
        return """
- conversation_group: conv1
  description: Test namespace detection using substring matching
  conversation:
    - eval_id: eval1
      eval_query: is there a openshift-monitoring namespace ?
      eval_types:
        - response_eval:sub-string
      expected_keywords:
        - "yes"
        - openshift-monitoring
      description: Check for openshift-monitoring namespace existence

- conversation_group: conv2
  description: Test namespace detection using LLM judge
  conversation:
    - eval_id: eval1
      eval_query: is there an openshift-lightspeed namespace?
      eval_types: [response_eval:accuracy]
      expected_response: there is an openshift-lightspeed namespace.
      description: Verify openshift-lightspeed namespace with LLM evaluation
"""

    @pytest.fixture
    def multiturn_conversation_yaml_content(self):
        """Valid YAML content with multi-turn conversation."""
        return """
- conversation_group: conv1
  description: Basic conversation flow testing cluster operations
  conversation:
    - eval_id: eval1
      eval_query: Hi!
      eval_types:
        - response_eval:accuracy
      expected_response: Hello! I'm an AI assistant for the Assisted Installer.
      description: Initial greeting to start conversation
    - eval_id: eval2
      eval_query: Get me active clusters
      eval_types:
        - response_eval:accuracy
      expected_response: Active clusters are x1, x2.
      description: Request for cluster information
    - eval_id: eval3
      eval_query: Thank you
      eval_types:
        - response_eval:accuracy
      expected_response: You're welcome!
      description: Closing statement
"""

    def test_init_success(self, valid_conversation_yaml_content):
        """Test successful initialization."""
        with (
            patch("builtins.open", mock_open(read_data=valid_conversation_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")

            assert manager.eval_data_file == "test.yaml"
            assert len(manager.conversations) == 2
            assert len(manager.get_conversations()) == 2
            assert manager.get_eval_count() == 2

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(EvaluationDataError, match="Eval data file not found"):
            AgentGoalEvalDataManager("nonexistent.yaml")

    def test_validate_eval_data_file_not_yaml(self):
        """Test loading invalid YAML file."""
        invalid_yaml = "invalid: yaml: content: ["

        with (
            patch("builtins.open", mock_open(read_data=invalid_yaml)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match="Invalid YAML"):
                AgentGoalEvalDataManager("test.yaml")

    def test_load_eval_data_file_read_error(self):
        """Test loading when file read fails."""
        with (
            patch("builtins.open", side_effect=IOError("Read error")),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match="Error loading eval data file"):
                AgentGoalEvalDataManager("test.yaml")

    def test_load_eval_data_not_list(self):
        """Test loading YAML that is not a list."""
        non_list_yaml = yaml.dump({"not": "a list"})

        with (
            patch("builtins.open", mock_open(read_data=non_list_yaml)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match="must contain a list"):
                AgentGoalEvalDataManager("test.yaml")

    def test_load_eval_data_empty_list(self):
        """Test loading YAML file with empty list."""
        empty_yaml = yaml.dump([])

        with (
            patch("builtins.open", mock_open(read_data=empty_yaml)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match="must contain at least one conversation"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_conversation_missing_group(self):
        """Test validation with missing conversation_group."""
        invalid_data = [
            {
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_query": "test query",
                        "eval_types": ["response_eval:accuracy"],
                        "expected_response": "test response",
                    }
                ]
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*Field required.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_conversation_missing_conversation_list(self):
        """Test validation with missing conversation list."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*Field required.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_missing_eval_id(self):
        """Test validation with missing eval_id."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
                "conversation": [
                    {
                        "eval_query": "test query",
                        "eval_types": ["response_eval:accuracy"],
                        "expected_response": "test response",
                    }
                ],
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*Field required.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_missing_eval_query(self):
        """Test validation with missing eval_query."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_types": ["response_eval:accuracy"],
                        "expected_response": "test response",
                    }
                ],
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*Field required.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_missing_eval_type(self):
        """Test validation with missing eval_type."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_query": "test query",
                        "expected_response": "test response",
                    }
                ],
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*Field required.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_invalid_eval_type(self):
        """Test validation with invalid eval_type."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_query": "test query",
                        "eval_types": ["invalid_type"],
                        "expected_response": "test response",
                    }
                ],
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*eval_type must be one of.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_judge_llm_missing_expected_response(self):
        """Test validation for judge-llm missing expected_response."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_query": "test query",
                        "eval_types": ["response_eval:accuracy"],
                    }
                ],
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*expected_response.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_sub_string_missing_keywords(self):
        """Test validation for sub-string missing expected_keywords."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_query": "test query",
                        "eval_types": ["response_eval:sub-string"],
                    }
                ],
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*expected_keywords.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_script_missing_verify_script(self):
        """Test validation for script missing eval_verify_script."""
        invalid_data = [
            {
                "conversation_group": "test_conv",
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_query": "test query",
                        "eval_types": ["action_eval"],
                    }
                ],
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match=".*eval_verify_script.*"):
                AgentGoalEvalDataManager("test.yaml")

    def test_duplicate_conversation_groups(self):
        """Test validation with duplicate conversation_group names."""
        invalid_data = [
            {
                "conversation_group": "duplicate_group",
                "conversation": [
                    {
                        "eval_id": "test1",
                        "eval_query": "test query 1",
                        "eval_types": ["response_eval:accuracy"],
                        "expected_response": "test response 1",
                    }
                ],
            },
            {
                "conversation_group": "duplicate_group",
                "conversation": [
                    {
                        "eval_id": "test2",
                        "eval_query": "test query 2",
                        "eval_types": ["response_eval:accuracy"],
                        "expected_response": "test response 2",
                    }
                ],
            },
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(EvaluationDataError, match="Duplicate conversation_group"):
                AgentGoalEvalDataManager("test.yaml")

    def test_get_conversations(self, valid_conversation_yaml_content):
        """Test get conversations method."""
        with (
            patch("builtins.open", mock_open(read_data=valid_conversation_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            conversations = manager.get_conversations()

            assert len(conversations) == 2
            assert isinstance(conversations[0], ConversationDataConfig)
            assert conversations[0].conversation_group == "conv1"
            assert conversations[1].conversation_group == "conv2"

    def test_get_eval_data_via_conversations(self, valid_conversation_yaml_content):
        """Test getting evaluation data via conversations."""
        with (
            patch("builtins.open", mock_open(read_data=valid_conversation_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            conversations = manager.get_conversations()

            eval_data = []
            for conversation in conversations:
                eval_data.extend(conversation.conversation)

            assert len(eval_data) == 2
            assert isinstance(eval_data[0], EvaluationDataConfig)
            assert eval_data[0].eval_id == "eval1"
            assert eval_data[1].eval_id == "eval1"

    def test_get_eval_count(self, valid_conversation_yaml_content):
        """Test get_eval_count method."""
        with (
            patch("builtins.open", mock_open(read_data=valid_conversation_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            count = manager.get_eval_count()

            assert count == 2

    def test_conversation_count_via_conversations(self, valid_conversation_yaml_content):
        """Test getting conversation count via conversations list."""
        with (
            patch("builtins.open", mock_open(read_data=valid_conversation_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            count = len(manager.conversations)

            assert count == 2

    def test_multiturn_conversation_loading(self, multiturn_conversation_yaml_content):
        """Test loading multi-turn conversation."""
        with (
            patch(
                "builtins.open",
                mock_open(read_data=multiturn_conversation_yaml_content),
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")

            assert len(manager.conversations) == 1
            assert manager.get_eval_count() == 3

            conversations = manager.get_conversations()
            conv = conversations[0]
            assert conv.conversation_group == "conv1"
            assert len(conv.conversation) == 3
            assert conv.conversation[0].eval_id == "eval1"
            assert conv.conversation[1].eval_id == "eval2"
            assert conv.conversation[2].eval_id == "eval3"

    def test_load_real_yaml_file_integration(self):
        """Integration test with a real temporary YAML file."""
        eval_data = [
            {
                "conversation_group": "integration_test",
                "description": "Integration test conversation",
                "conversation": [
                    {
                        "eval_id": "integration_test_eval",
                        "eval_query": "Test query",
                        "eval_types": ["response_eval:accuracy"],
                        "expected_response": "Test response",
                        "description": "Integration test evaluation",
                    }
                ],
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(eval_data, f)
            temp_file_path = f.name

        try:
            manager = AgentGoalEvalDataManager(temp_file_path)

            assert len(manager.conversations) == 1
            assert manager.get_eval_count() == 1

            conversations = manager.get_conversations()
            assert conversations[0].conversation_group == "integration_test"
            assert conversations[0].description == "Integration test conversation"

        finally:
            Path(temp_file_path).unlink()  # Clean up temporary file

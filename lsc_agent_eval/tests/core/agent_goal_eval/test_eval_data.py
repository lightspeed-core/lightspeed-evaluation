"""Tests for evaluation data manager."""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from lsc_agent_eval.core.agent_goal_eval.eval_data import AgentGoalEvalDataManager
from lsc_agent_eval.core.agent_goal_eval.models import EvaluationDataConfig
from lsc_agent_eval.core.utils.exceptions import ConfigurationError


class TestAgentGoalEvalDataManager:
    """Test AgentGoalEvalDataManager."""

    @pytest.fixture
    def valid_eval_data(self):
        """Valid evaluation data for testing."""
        return [
            {
                "eval_id": "test_001",
                "eval_query": "What is Kubernetes?",
                "eval_type": "judge-llm",
                "expected_response": "Kubernetes is a container orchestration platform",
            },
            {
                "eval_id": "test_002",
                "eval_query": "Deploy nginx",
                "eval_type": "script",
                "eval_verify_script": "./scripts/verify_nginx.sh",
            },
            {
                "eval_id": "test_003",
                "eval_query": "Show pods",
                "eval_type": "sub-string",
                "expected_keywords": ["pod", "running"],
            },
        ]

    @pytest.fixture
    def valid_yaml_content(self, valid_eval_data):
        """Valid YAML content as string."""
        return yaml.dump(valid_eval_data)

    def test_init_success(self, valid_yaml_content):
        """Test successful initialization."""
        with (
            patch("builtins.open", mock_open(read_data=valid_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")

            assert len(manager.eval_data) == 3
            assert manager.eval_data_file == Path("test.yaml")
            assert isinstance(manager.eval_data[0], EvaluationDataConfig)

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ConfigurationError, match="Eval data file not found"):
                AgentGoalEvalDataManager("nonexistent.yaml")

    def test_init_path_not_file(self):
        """Test initialization when path is not a file."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=False),
        ):

            with pytest.raises(ConfigurationError, match="path is not a file"):
                AgentGoalEvalDataManager("directory/")

    def test_validate_eval_data_file_exists(self):
        """Test file validation when file exists."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
            patch("builtins.open", mock_open(read_data="[]")),
        ):

            # Should not raise exception
            manager = AgentGoalEvalDataManager("test.yaml")
            assert manager.eval_data_file == Path("test.yaml")

    def test_load_eval_data_invalid_yaml(self):
        """Test loading invalid YAML content."""
        invalid_yaml = "invalid: yaml: content: ["

        with (
            patch("builtins.open", mock_open(read_data=invalid_yaml)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(ConfigurationError, match="Invalid YAML"):
                AgentGoalEvalDataManager("test.yaml")

    def test_load_eval_data_not_list(self):
        """Test loading YAML that is not a list."""
        yaml_dict = yaml.dump({"key": "value"})

        with (
            patch("builtins.open", mock_open(read_data=yaml_dict)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(ConfigurationError, match="must contain a list"):
                AgentGoalEvalDataManager("test.yaml")

    def test_load_eval_data_empty_list(self):
        """Test loading empty evaluation list."""
        empty_yaml = yaml.dump([])

        with (
            patch("builtins.open", mock_open(read_data=empty_yaml)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            assert len(manager.eval_data) == 0

    def test_load_eval_data_file_read_error(self):
        """Test loading when file read fails."""
        with (
            patch("builtins.open", side_effect=IOError("Read error")),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(
                ConfigurationError, match="Error loading eval data file"
            ):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_data_missing_eval_id(self):
        """Test validation with missing eval_id."""
        invalid_data = [{"eval_query": "test query"}]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(
                ConfigurationError, match="Missing required field 'eval_id'"
            ):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_data_missing_eval_query(self):
        """Test validation with missing eval_query."""
        invalid_data = [{"eval_id": "test_001"}]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(
                ConfigurationError, match="Missing required field 'eval_query'"
            ):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_data_invalid_eval_type(self):
        """Test validation with invalid eval_type."""
        invalid_data = [
            {
                "eval_id": "test_001",
                "eval_query": "test query",
                "eval_type": "invalid_type",
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(
                ConfigurationError, match="Invalid eval_type: invalid_type"
            ):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_data_judge_llm_missing_expected_response(self):
        """Test validation for judge-llm type missing expected_response."""
        invalid_data = [
            {
                "eval_id": "test_001",
                "eval_query": "test query",
                "eval_type": "judge-llm",
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(
                ConfigurationError, match="requires 'expected_response' field"
            ):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_data_sub_string_missing_keywords(self):
        """Test validation for sub-string type missing expected_keywords."""
        invalid_data = [
            {
                "eval_id": "test_001",
                "eval_query": "test query",
                "eval_type": "sub-string",
            }
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(
                ConfigurationError, match="requires 'expected_keywords' field"
            ):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_data_script_missing_verify_script(self):
        """Test validation for script type missing eval_verify_script."""
        invalid_data = [
            {"eval_id": "test_001", "eval_query": "test query", "eval_type": "script"}
        ]
        yaml_content = yaml.dump(invalid_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            with pytest.raises(
                ConfigurationError, match="requires 'eval_verify_script' field"
            ):
                AgentGoalEvalDataManager("test.yaml")

    def test_validate_eval_data_default_eval_type(self):
        """Test validation with default eval_type (judge-llm)."""
        data_with_default_type = [
            {
                "eval_id": "test_001",
                "eval_query": "test query",
                "expected_response": "test response",
                # eval_type not specified, should default to judge-llm
            }
        ]
        yaml_content = yaml.dump(data_with_default_type)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            assert len(manager.eval_data) == 1
            assert manager.eval_data[0].eval_type == "judge-llm"

    def test_get_eval_data(self, valid_yaml_content):
        """Test get_eval_data method."""
        with (
            patch("builtins.open", mock_open(read_data=valid_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            eval_data = manager.get_eval_data()

            assert isinstance(eval_data, list)
            assert len(eval_data) == 3
            assert all(isinstance(item, EvaluationDataConfig) for item in eval_data)
            assert eval_data[0].eval_id == "test_001"
            assert eval_data[1].eval_id == "test_002"
            assert eval_data[2].eval_id == "test_003"

    def test_get_eval_count(self, valid_yaml_content):
        """Test get_eval_count method."""
        with (
            patch("builtins.open", mock_open(read_data=valid_yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            count = manager.get_eval_count()

            assert count == 3
            assert count == len(manager.eval_data)

    def test_get_eval_count_empty(self):
        """Test get_eval_count with empty data."""
        empty_yaml = yaml.dump([])

        with (
            patch("builtins.open", mock_open(read_data=empty_yaml)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            count = manager.get_eval_count()

            assert count == 0

    def test_judge_llm_validation_success(self):
        """Test successful validation for judge-llm type."""
        judge_llm_data = [
            {
                "eval_id": "test_judge",
                "eval_query": "What is Docker?",
                "eval_type": "judge-llm",
                "expected_response": "Docker is a containerization platform",
            }
        ]
        yaml_content = yaml.dump(judge_llm_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            assert len(manager.eval_data) == 1
            assert manager.eval_data[0].eval_type == "judge-llm"
            assert (
                manager.eval_data[0].expected_response
                == "Docker is a containerization platform"
            )

    def test_script_validation_success(self):
        """Test successful validation for script type."""
        script_data = [
            {
                "eval_id": "test_script",
                "eval_query": "Deploy application",
                "eval_type": "script",
                "eval_verify_script": "./verify_deployment.sh",
            }
        ]
        yaml_content = yaml.dump(script_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            assert len(manager.eval_data) == 1
            assert manager.eval_data[0].eval_type == "script"
            assert manager.eval_data[0].eval_verify_script == "./verify_deployment.sh"

    def test_sub_string_validation_success(self):
        """Test successful validation for sub-string type."""
        sub_string_data = [
            {
                "eval_id": "test_substring",
                "eval_query": "List services",
                "eval_type": "sub-string",
                "expected_keywords": ["service", "active", "running"],
            }
        ]
        yaml_content = yaml.dump(sub_string_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            assert len(manager.eval_data) == 1
            assert manager.eval_data[0].eval_type == "sub-string"
            assert manager.eval_data[0].expected_keywords == [
                "service",
                "active",
                "running",
            ]

    def test_mixed_eval_types(self):
        """Test loading data with mixed evaluation types."""
        mixed_data = [
            {
                "eval_id": "judge_test",
                "eval_query": "What is Kubernetes?",
                "eval_type": "judge-llm",
                "expected_response": "Container orchestration",
            },
            {
                "eval_id": "script_test",
                "eval_query": "Deploy nginx",
                "eval_type": "script",
                "eval_verify_script": "./verify.sh",
            },
            {
                "eval_id": "substring_test",
                "eval_query": "List pods",
                "eval_type": "sub-string",
                "expected_keywords": ["pod", "running"],
            },
        ]
        yaml_content = yaml.dump(mixed_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            assert len(manager.eval_data) == 3

            types = [item.eval_type for item in manager.eval_data]
            assert "judge-llm" in types
            assert "script" in types
            assert "sub-string" in types

    def test_eval_data_with_optional_fields(self):
        """Test evaluation data with optional fields."""
        data_with_optional = [
            {
                "eval_id": "test_with_optional",
                "eval_query": "Deploy app",
                "eval_type": "script",
                "eval_verify_script": "./verify.sh",
                "eval_setup_script": "./setup.sh",
                "eval_cleanup_script": "./cleanup.sh",
            }
        ]
        yaml_content = yaml.dump(data_with_optional)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):

            manager = AgentGoalEvalDataManager("test.yaml")
            assert len(manager.eval_data) == 1
            eval_item = manager.eval_data[0]
            assert eval_item.eval_setup_script == "./setup.sh"
            assert eval_item.eval_cleanup_script == "./cleanup.sh"

    def test_load_real_yaml_file_integration(self):
        """Integration test with a real temporary YAML file."""
        eval_data = [
            {
                "eval_id": "integration_test",
                "eval_query": "Test query",
                "eval_type": "judge-llm",
                "expected_response": "Test response",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(eval_data, f)
            temp_file_path = f.name

        try:
            manager = AgentGoalEvalDataManager(temp_file_path)
            assert len(manager.eval_data) == 1
            assert manager.eval_data[0].eval_id == "integration_test"
        finally:
            Path(temp_file_path).unlink()  # Clean up temporary file

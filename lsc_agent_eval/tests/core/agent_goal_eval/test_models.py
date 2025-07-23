"""Tests for agent evaluation models."""

from lsc_agent_eval.core.agent_goal_eval.models import (
    EvaluationDataConfig,
    EvaluationResult,
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


class TestEvaluationDataConfig:
    """Test EvaluationDataConfig data class."""

    def test_evaluation_data_config_minimal(self):
        """Test creating minimal EvaluationDataConfig."""
        config = EvaluationDataConfig(
            eval_id="test_001",
            eval_query="What is Kubernetes?",
        )

        assert config.eval_id == "test_001"
        assert config.eval_query == "What is Kubernetes?"
        assert config.eval_type == "judge-llm"  # default
        assert config.expected_response is None
        assert config.expected_keywords is None
        assert config.eval_setup_script is None
        assert config.eval_verify_script is None
        assert config.eval_cleanup_script is None

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
        assert config.eval_setup_script is None
        assert config.eval_verify_script is None
        assert config.eval_cleanup_script is None

    def test_evaluation_data_config_script(self):
        """Test EvaluationDataConfig for script evaluation."""
        config = EvaluationDataConfig(
            eval_id="script_test",
            eval_query="Deploy nginx pod",
            eval_type="script",
            eval_setup_script="./setup.sh",
            eval_verify_script="./verify.sh",
            eval_cleanup_script="./cleanup.sh",
        )

        assert config.eval_id == "script_test"
        assert config.eval_query == "Deploy nginx pod"
        assert config.eval_type == "script"
        assert config.expected_response is None
        assert config.expected_keywords is None
        assert config.eval_setup_script == "./setup.sh"
        assert config.eval_verify_script == "./verify.sh"
        assert config.eval_cleanup_script == "./cleanup.sh"

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
        assert config.eval_setup_script is None
        assert config.eval_verify_script is None
        assert config.eval_cleanup_script is None

    def test_evaluation_data_config_all_fields(self):
        """Test EvaluationDataConfig with all fields."""
        config = EvaluationDataConfig(
            eval_id="full_test",
            eval_query="What is OpenShift?",
            eval_type="judge-llm",
            expected_response="OpenShift is a Kubernetes platform",
            expected_keywords=["kubernetes", "platform", "container"],
            eval_setup_script="./setup.sh",
            eval_verify_script="./verify.sh",
            eval_cleanup_script="./cleanup.sh",
        )

        assert config.eval_id == "full_test"
        assert config.eval_query == "What is OpenShift?"
        assert config.eval_type == "judge-llm"
        assert config.expected_response == "OpenShift is a Kubernetes platform"
        assert config.expected_keywords == ["kubernetes", "platform", "container"]
        assert config.eval_setup_script == "./setup.sh"
        assert config.eval_verify_script == "./verify.sh"
        assert config.eval_cleanup_script == "./cleanup.sh"

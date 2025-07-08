"""Tests for custom exceptions."""

from lsc_agent_eval.core.utils.exceptions import (
    AgentAPIError,
    AgentEvaluationError,
    ConfigurationError,
    JudgeModelError,
    ScriptExecutionError,
)


class TestAgentEvaluationError:
    """Test base AgentEvaluationError."""

    def test_agent_evaluation_error_creation(self):
        """Test creating AgentEvaluationError."""
        error = AgentEvaluationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_agent_evaluation_error_inheritance(self):
        """Test AgentEvaluationError inheritance."""
        error = AgentEvaluationError("Test error")
        assert isinstance(error, AgentEvaluationError)
        assert isinstance(error, Exception)


class TestConfigurationError:
    """Test ConfigurationError."""

    def test_configuration_error_creation(self):
        """Test creating ConfigurationError."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, AgentEvaluationError)

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("Config error")
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, AgentEvaluationError)
        assert isinstance(error, Exception)


class TestAgentAPIError:
    """Test AgentAPIError."""

    def test_agent_api_error_creation(self):
        """Test creating AgentAPIError."""
        error = AgentAPIError("API connection failed")
        assert str(error) == "API connection failed"
        assert isinstance(error, AgentAPIError)
        assert isinstance(error, AgentEvaluationError)

    def test_agent_api_error_inheritance(self):
        """Test AgentAPIError inheritance."""
        error = AgentAPIError("API error")
        assert isinstance(error, AgentAPIError)
        assert isinstance(error, AgentEvaluationError)
        assert isinstance(error, Exception)


class TestScriptExecutionError:
    """Test ScriptExecutionError."""

    def test_script_execution_error_creation(self):
        """Test creating ScriptExecutionError."""
        error = ScriptExecutionError("Script failed with exit code 1")
        assert str(error) == "Script failed with exit code 1"
        assert isinstance(error, ScriptExecutionError)
        assert isinstance(error, AgentEvaluationError)

    def test_script_execution_error_inheritance(self):
        """Test ScriptExecutionError inheritance."""
        error = ScriptExecutionError("Script error")
        assert isinstance(error, ScriptExecutionError)
        assert isinstance(error, AgentEvaluationError)
        assert isinstance(error, Exception)


class TestJudgeModelError:
    """Test JudgeModelError."""

    def test_judge_model_error_creation(self):
        """Test creating JudgeModelError."""
        error = JudgeModelError("Judge model initialization failed")
        assert str(error) == "Judge model initialization failed"
        assert isinstance(error, JudgeModelError)
        assert isinstance(error, AgentEvaluationError)

    def test_judge_model_error_inheritance(self):
        """Test JudgeModelError inheritance."""
        error = JudgeModelError("Judge error")
        assert isinstance(error, JudgeModelError)
        assert isinstance(error, AgentEvaluationError)
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """Test exception hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from AgentEvaluationError."""
        exceptions = [
            ConfigurationError("config error"),
            AgentAPIError("api error"),
            ScriptExecutionError("script error"),
            JudgeModelError("judge error"),
        ]

        for exc in exceptions:
            assert isinstance(exc, AgentEvaluationError)
            assert isinstance(exc, Exception)

    def test_exception_with_none_message(self):
        """Test exceptions with None message."""
        error = AgentEvaluationError(None)
        assert str(error) == "None"

    def test_exception_with_empty_message(self):
        """Test exceptions with empty message."""
        error = AgentEvaluationError("")
        assert str(error) == ""

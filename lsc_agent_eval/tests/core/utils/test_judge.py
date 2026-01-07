"""Tests for judge model manager."""

import os
import re
from pytest_mock import MockerFixture

import pytest

from lsc_agent_eval.core.utils.exceptions import JudgeModelError
from lsc_agent_eval.core.utils.judge import JudgeModelManager


class TestJudgeModelManager:
    """Test JudgeModelManager."""

    def test_init_openai_success(self, mocker: MockerFixture):
        """Test initializing OpenAI judge model."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        judge = JudgeModelManager("openai", "gpt-4")

        assert judge.judge_provider == "openai"
        assert judge.judge_model == "gpt-4"
        assert judge.model_name == "gpt-4"

    def test_init_openai_missing_key(self, mocker: MockerFixture):
        """Test initializing OpenAI judge model without API key."""
        mocker.patch.dict(os.environ, {}, clear=True)
        with pytest.raises(
            JudgeModelError, match="OPENAI_API_KEY environment variable is required"
        ):
            JudgeModelManager("openai", "gpt-4")

    def test_init_azure_success(self, mocker: MockerFixture):
        """Test initializing Azure judge model."""
        mocker.patch.dict(
            os.environ,
            {
                "AZURE_API_KEY": "test-key",
                "AZURE_API_BASE": "https://test.openai.azure.com",
                "AZURE_DEPLOYMENT_NAME": "gpt-4-deployment",
            },
        )
        judge = JudgeModelManager("azure", "gpt-4")

        assert judge.judge_provider == "azure"
        assert judge.judge_model == "gpt-4"
        assert judge.model_name == "azure/gpt-4-deployment"

    def test_init_watsonx_success(self, mocker: MockerFixture):
        """Test initializing Watsonx judge model."""
        mocker.patch.dict(
            os.environ,
            {
                "WATSONX_API_KEY": "test-key",
                "WATSONX_API_BASE": "https://test.watsonx.ibm.com",
                "WATSONX_PROJECT_ID": "test-project",
            },
        )
        judge = JudgeModelManager("watsonx", "granite-3-8b-instruct")

        assert judge.judge_provider == "watsonx"
        assert judge.judge_model == "granite-3-8b-instruct"
        assert judge.model_name == "watsonx/granite-3-8b-instruct"

    def test_init_generic_provider(self, mocker: MockerFixture):
        """Test initializing generic provider."""
        mock_logger = mocker.patch("lsc_agent_eval.core.utils.judge.logger")
        judge = JudgeModelManager("xyz", "abcd")

        assert judge.judge_provider == "xyz"
        assert judge.judge_model == "abcd"
        assert judge.model_name == "xyz/abcd"
        mock_logger.warning.assert_called_once_with(
            "Using generic provider format for %s", "xyz"
        )

    def test_init_setup_failure(self, mocker: MockerFixture):
        """Test judge model setup failure."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        mocker.patch(
            "lsc_agent_eval.core.utils.judge.JudgeModelManager._setup_litellm",
            side_effect=Exception("Setup error"),
        )

        with pytest.raises(
            JudgeModelError, match="Failed to setup JudgeLLM using LiteLLM"
        ):
            JudgeModelManager("openai", "gpt-4")

    def test_evaluate_response_success(self, mocker: MockerFixture):
        """Test successful response evaluation."""
        mock_litellm = mocker.patch("lsc_agent_eval.core.utils.judge.litellm")
        # Mock LiteLLM completion response
        mock_response = mocker.Mock()
        mock_message = mocker.Mock()
        mock_message.content = "1"
        mock_choice = mocker.Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_litellm.completion.return_value = mock_response

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        judge = JudgeModelManager("openai", "gpt-4")
        result = judge.evaluate_response("Test prompt")

        assert result == "1"
        mock_litellm.completion.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.0,
            timeout=300,
        )

    def test_evaluate_response_invalid_structure(self, mocker: MockerFixture):
        """Test response evaluation with invalid response structure."""
        mock_litellm = mocker.patch("lsc_agent_eval.core.utils.judge.litellm")
        # Mock LiteLLM completion response with invalid structure
        mock_response = mocker.Mock()
        mock_response.choices = None
        mock_litellm.completion.return_value = mock_response

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        judge = JudgeModelManager("openai", "gpt-4")
        with pytest.raises(
            JudgeModelError,
            match="No valid response from Judge Model. Check full response\n",
        ):
            judge.evaluate_response("Test prompt")

    def test_evaluate_response_timeout_retry(self, mocker: MockerFixture):
        """Test response evaluation with timeout and retry."""
        mock_litellm = mocker.patch("lsc_agent_eval.core.utils.judge.litellm")
        mock_sleep = mocker.patch("lsc_agent_eval.core.utils.judge.sleep")

        # Mock timeout on first call, success on second
        mock_response = mocker.Mock()
        mock_message = mocker.Mock()
        mock_message.content = "1"
        mock_choice = mocker.Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_litellm.completion.side_effect = [
            TimeoutError("Request timeout"),
            mock_response,
        ]

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        mock_logger = mocker.patch("lsc_agent_eval.core.utils.judge.logger")
        judge = JudgeModelManager("openai", "gpt-4")
        result = judge.evaluate_response("Test prompt")

        assert result == "1"
        assert mock_litellm.completion.call_count == 2
        mock_logger.warning.assert_called_once()
        mock_sleep.assert_called_once()

    def test_evaluate_response_max_retries_exceeded(self, mocker: MockerFixture):
        """Test response evaluation with max retries exceeded."""
        mock_litellm = mocker.patch("lsc_agent_eval.core.utils.judge.litellm")
        mock_litellm.completion.side_effect = TimeoutError("Request timeout")

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        judge = JudgeModelManager("openai", "gpt-4")

        with pytest.raises(
            JudgeModelError, match="Judge model evaluation failed after 3 attempts"
        ):
            judge.evaluate_response("Test prompt")

    def test_evaluate_response_with_whitespace(self, mocker: MockerFixture):
        """Test response evaluation with whitespace in content."""
        mock_litellm = mocker.patch("lsc_agent_eval.core.utils.judge.litellm")
        # Mock LiteLLM completion response with whitespace
        mock_response = mocker.Mock()
        mock_message = mocker.Mock()
        mock_message.content = "  1  \n"
        mock_choice = mocker.Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_litellm.completion.return_value = mock_response

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        judge = JudgeModelManager("openai", "gpt-4")
        result = judge.evaluate_response("Test prompt")

        assert result == "1"

    def test_evaluate_response_empty_content(self, mocker: MockerFixture):
        """Test response evaluation with empty content."""
        mock_litellm = mocker.patch("lsc_agent_eval.core.utils.judge.litellm")
        # Mock LiteLLM completion response with empty content
        mock_response = mocker.Mock()
        mock_message = mocker.Mock()
        mock_message.content = ""
        mock_choice = mocker.Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        # Set all fallback attributes to empty
        mock_litellm.completion.return_value = mock_response

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        judge = JudgeModelManager("openai", "gpt-4")
        with pytest.raises(
            JudgeModelError,
            match=re.escape(
                "No valid response from Judge Model. "
                f"Check full response\n{mock_response}"
            ),
        ):
            judge.evaluate_response("Test prompt")

    def test_get_model_name(self, mocker: MockerFixture):
        """Test get_model_name method."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        judge = JudgeModelManager("openai", "gpt-4")
        assert judge.get_model_name() == "gpt-4"

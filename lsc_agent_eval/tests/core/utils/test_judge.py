"""Tests for judge model manager."""

import os
import re
from unittest.mock import Mock, patch

import pytest

from lsc_agent_eval.core.utils.exceptions import JudgeModelError
from lsc_agent_eval.core.utils.judge import JudgeModelManager


class TestJudgeModelManager:
    """Test JudgeModelManager."""

    def test_init_openai_success(self):
        """Test initializing OpenAI judge model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            judge = JudgeModelManager("openai", "gpt-4")

            assert judge.judge_provider == "openai"
            assert judge.judge_model == "gpt-4"
            assert judge.model_name == "gpt-4"

    def test_init_openai_missing_key(self):
        """Test initializing OpenAI judge model without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                JudgeModelError, match="OPENAI_API_KEY environment variable is required"
            ):
                JudgeModelManager("openai", "gpt-4")

    def test_init_azure_success(self):
        """Test initializing Azure judge model."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4-deployment",
            },
        ):
            judge = JudgeModelManager("azure", "gpt-4")

            assert judge.judge_provider == "azure"
            assert judge.judge_model == "gpt-4"
            assert judge.model_name == "azure/gpt-4-deployment"

    def test_init_watsonx_success(self):
        """Test initializing Watsonx judge model."""
        with patch.dict(
            os.environ,
            {
                "WATSONX_API_KEY": "test-key",
                "WATSONX_API_BASE": "https://test.watsonx.ibm.com",
                "WATSONX_PROJECT_ID": "test-project",
            },
        ):
            judge = JudgeModelManager("watsonx", "granite-3-8b-instruct")

            assert judge.judge_provider == "watsonx"
            assert judge.judge_model == "granite-3-8b-instruct"
            assert judge.model_name == "watsonx/granite-3-8b-instruct"

    def test_init_generic_provider(self):
        """Test initializing generic provider."""
        with patch("lsc_agent_eval.core.utils.judge.logger") as mock_logger:
            judge = JudgeModelManager("xyz", "abcd")

            assert judge.judge_provider == "xyz"
            assert judge.judge_model == "abcd"
            assert judge.model_name == "xyz/abcd"
            mock_logger.warning.assert_called_once_with(
                "Using generic provider format for %s", "xyz"
            )

    def test_init_setup_failure(self):
        """Test judge model setup failure."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch(
                "lsc_agent_eval.core.utils.judge.JudgeModelManager._setup_litellm",
                side_effect=Exception("Setup error"),
            ),
        ):

            with pytest.raises(
                JudgeModelError, match="Failed to setup JudgeLLM using LiteLLM"
            ):
                JudgeModelManager("openai", "gpt-4")

    @patch("lsc_agent_eval.core.utils.judge.litellm")
    def test_evaluate_response_success(self, mock_litellm):
        """Test successful response evaluation."""
        # Mock LiteLLM completion response
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "1"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_litellm.completion.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            judge = JudgeModelManager("openai", "gpt-4")
            result = judge.evaluate_response("Test prompt")

            assert result == "1"
            mock_litellm.completion.assert_called_once_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test prompt"}],
                temperature=0.0,
                timeout=300,
            )

    @patch("lsc_agent_eval.core.utils.judge.litellm")
    def test_evaluate_response_invalid_structure(self, mock_litellm):
        """Test response evaluation with invalid response structure."""
        # Mock LiteLLM completion response with invalid structure
        mock_response = Mock()
        mock_response.choices = None
        mock_litellm.completion.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            judge = JudgeModelManager("openai", "gpt-4")
            with pytest.raises(
                JudgeModelError,
                match="No valid response from Judge Model. Check full response\n",
            ):
                judge.evaluate_response("Test prompt")

    @patch("lsc_agent_eval.core.utils.judge.litellm")
    @patch("lsc_agent_eval.core.utils.judge.sleep")
    def test_evaluate_response_timeout_retry(self, mock_sleep, mock_litellm):
        """Test response evaluation with timeout and retry."""
        # Mock timeout on first call, success on second
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "1"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_litellm.completion.side_effect = [
            TimeoutError("Request timeout"),
            mock_response,
        ]

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("lsc_agent_eval.core.utils.judge.logger") as mock_logger,
        ):
            judge = JudgeModelManager("openai", "gpt-4")
            result = judge.evaluate_response("Test prompt")

            assert result == "1"
            assert mock_litellm.completion.call_count == 2
            mock_logger.warning.assert_called_once()
            mock_sleep.assert_called_once()

    @patch("lsc_agent_eval.core.utils.judge.litellm")
    def test_evaluate_response_max_retries_exceeded(self, mock_litellm):
        """Test response evaluation with max retries exceeded."""
        mock_litellm.completion.side_effect = TimeoutError("Request timeout")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            judge = JudgeModelManager("openai", "gpt-4")

            with pytest.raises(
                JudgeModelError, match="Judge model evaluation failed after 3 attempts"
            ):
                judge.evaluate_response("Test prompt")

    @patch("lsc_agent_eval.core.utils.judge.litellm")
    def test_evaluate_response_with_whitespace(self, mock_litellm):
        """Test response evaluation with whitespace in content."""
        # Mock LiteLLM completion response with whitespace
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "  1  \n"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_litellm.completion.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            judge = JudgeModelManager("openai", "gpt-4")
            result = judge.evaluate_response("Test prompt")

            assert result == "1"

    @patch("lsc_agent_eval.core.utils.judge.litellm")
    def test_evaluate_response_empty_content(self, mock_litellm):
        """Test response evaluation with empty content."""
        # Mock LiteLLM completion response with empty content
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = ""
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        # Set all fallback attributes to empty
        mock_litellm.completion.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            judge = JudgeModelManager("openai", "gpt-4")
            with pytest.raises(
                JudgeModelError,
                match=re.escape(
                    "No valid response from Judge Model. "
                    f"Check full response\n{mock_response}"
                ),
            ):
                judge.evaluate_response("Test prompt")

    def test_get_model_name(self):
        """Test get_model_name method."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            judge = JudgeModelManager("openai", "gpt-4")
            assert judge.get_model_name() == "gpt-4"

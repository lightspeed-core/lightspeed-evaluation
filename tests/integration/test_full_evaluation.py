# pylint: disable=redefined-outer-name,too-many-arguments,too-many-positional-arguments,import-outside-toplevel
"""End-to-End Integration tests for LightSpeed Evaluation Framework.

These tests run the complete evaluation pipeline with real services:
- Real Lightspeed-stack API on localhost:8080
- Real OpenAI API (requires OPENAI_API_KEY)
- Real evaluation metrics (Ragas, DeepEval, etc.)

Prerequisites:
    - Lightspeed-stack API running on localhost:8080
    - OPENAI_API_KEY environment variable set
    - Network connectivity for API calls

Run with: pytest tests/integration/ -v -m integration
"""

import os
from pathlib import Path

import httpx
import pytest

from lightspeed_evaluation import ConfigLoader, evaluate
from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.storage import FileBackendConfig


def check_api_available() -> bool:
    """Check if Lightspeed-stack API is available on localhost:8080."""
    try:
        # Check root endpoint since /health may not exist
        response = httpx.get("http://localhost:8080/v1/models", timeout=2.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def check_openai_key_available() -> bool:
    """Check if OPENAI_API_KEY is set in environment."""
    return bool(os.getenv("OPENAI_API_KEY"))


# Mark ALL tests in this file as integration tests
# These tests will NOT run by default - must explicitly run with: pytest -m integration
pytestmark = pytest.mark.integration


@pytest.fixture
def integration_test_dir() -> Path:
    """Get the integration test directory path."""
    return Path(__file__).parent


@pytest.fixture
def eval_data_path(integration_test_dir: Path) -> Path:
    """Get path to test evaluation data file."""
    return integration_test_dir / "test_evaluation_data.yaml"


@pytest.fixture
def query_config_path(integration_test_dir: Path) -> Path:
    """Get path to query endpoint system config file."""
    return integration_test_dir / "system-config-query.yaml"


@pytest.fixture
def streaming_config_path(integration_test_dir: Path) -> Path:
    """Get path to streaming endpoint system config file."""
    return integration_test_dir / "system-config-streaming.yaml"


class TestFullEvaluation:
    """End-to-end tests for full evaluation with both query and streaming endpoints."""

    @pytest.mark.parametrize(
        "config_fixture,endpoint_type",
        [
            ("query_config_path", "query"),
            ("streaming_config_path", "streaming"),
        ],
    )
    def test_full_evaluation_endpoint(  # pylint: disable=too-many-locals
        self,
        config_fixture: str,
        endpoint_type: str,
        eval_data_path: Path,
        request: pytest.FixtureRequest,
        tmp_path: Path,
    ) -> None:
        """Test complete evaluation with both query and streaming endpoints.

        This test verifies:
        - System config loads correctly
        - Evaluation data loads correctly
        - API calls are made to localhost:8080
        - LLM judge evaluates responses
        - Pipeline executes without errors
        - Results are PASS (evaluation succeeds)

        Args:
            config_fixture: Name of the fixture providing config path
            endpoint_type: Type of endpoint ('query' or 'streaming')
            eval_data_path: Path to evaluation data YAML
            request: Pytest fixture request object
            tmp_path: Temporary directory for output
        """
        # Get the actual config path from the fixture
        config_path = request.getfixturevalue(config_fixture)

        # Load configuration
        loader = ConfigLoader()
        system_config = loader.load_system_config(str(config_path))

        # Override output directory to use temporary path
        file_config = FileBackendConfig(output_dir=str(tmp_path / "eval_output"))
        system_config.storage = [file_config]

        # Verify endpoint type matches expectation
        assert (
            system_config.api.endpoint_type == endpoint_type
        ), f"Config should use {endpoint_type} endpoint"

        # Load evaluation data
        from lightspeed_evaluation.core.system import DataValidator

        validator = DataValidator(
            api_enabled=system_config.api.enabled,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
        )
        evaluation_data = validator.load_evaluation_data(str(eval_data_path))

        # Verify evaluation data loaded
        assert len(evaluation_data) > 0, "Evaluation data should not be empty"
        assert evaluation_data[0].conversation_group_id == "conv_group_1"
        assert len(evaluation_data[0].turns) > 0

        # Run evaluation (makes real API calls)
        results = evaluate(system_config, evaluation_data)

        # Verify results
        assert isinstance(results, list), "Results should be a list"
        assert len(results) > 0, "Should have at least one result"

        # Verify all results are EvaluationResult instances
        for result in results:
            assert isinstance(result, EvaluationResult)
            assert result.conversation_group_id == "conv_group_1"
            assert result.turn_id == "turn_id1"
            assert result.metric_identifier is not None

        # Verify we have the expected metric
        metric_identifiers = [r.metric_identifier for r in results]
        assert "ragas:response_relevancy" in metric_identifiers

        # Find the response_relevancy result
        relevancy_result = next(
            r for r in results if r.metric_identifier == "ragas:response_relevancy"
        )

        # Verify the evaluation PASSED
        assert relevancy_result.result == "PASS", (
            f"Evaluation should PASS but got {relevancy_result.result}. "
            f"Reason: {relevancy_result.reason}"
        )

        # Verify threshold is correct (from test data)
        assert relevancy_result.threshold == 0.9, "Threshold should match test data"

        # Verify score is above threshold
        assert relevancy_result.response.strip(), "Should have response from API"
        assert (
            relevancy_result.score >= relevancy_result.threshold  # type: ignore
        ), f"Score {relevancy_result.score} should be >= threshold {relevancy_result.threshold}"

        # Verify we got a response from the API
        assert relevancy_result.response is not None, "Should have response from API"
        assert relevancy_result.query == "What is the capital of France?"

        # Verify API token usage is tracked
        assert (
            relevancy_result.api_input_tokens > 0
            or relevancy_result.api_output_tokens > 0
        ), "API token usage should be tracked"

    @pytest.mark.parametrize(
        "config_fixture,endpoint_type",
        [
            ("query_config_path", "query"),
            ("streaming_config_path", "streaming"),
        ],
    )
    def test_api_response_enrichment(  # pylint: disable=too-many-locals
        self,
        config_fixture: str,
        endpoint_type: str,
        eval_data_path: Path,
        request: pytest.FixtureRequest,
        tmp_path: Path,
    ) -> None:
        """Test that API responses properly enrich evaluation data.

        Verifies:
        - API returns responses for queries
        - Token usage is tracked
        - Contexts are retrieved (if available)
        - Response is non-empty

        Args:
            config_fixture: Name of the fixture providing config path
            endpoint_type: Type of endpoint ('query' or 'streaming')
            eval_data_path: Path to evaluation data YAML
            request: Pytest fixture request object
            tmp_path: Temporary directory for output
        """
        # Get the actual config path from the fixture
        config_path = request.getfixturevalue(config_fixture)

        loader = ConfigLoader()
        system_config = loader.load_system_config(str(config_path))
        file_config = FileBackendConfig(output_dir=str(tmp_path / "eval_output"))
        system_config.storage = [file_config]

        # Verify endpoint type matches expectation
        assert (
            system_config.api.endpoint_type == endpoint_type
        ), f"Config should use {endpoint_type} endpoint"

        from lightspeed_evaluation.core.system import DataValidator

        validator = DataValidator(
            api_enabled=system_config.api.enabled,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
        )
        evaluation_data = validator.load_evaluation_data(str(eval_data_path))

        # Run evaluation
        results = evaluate(system_config, evaluation_data)

        # Verify at least one result has API data
        has_api_data = any(
            r.response.strip() and (r.api_input_tokens > 0 or r.api_output_tokens > 0)
            for r in results
        )
        assert has_api_data, "At least one result should have API response data"

        # Find a result with response data
        result_with_response = next(
            r
            for r in results
            if r.response.strip()
            and (r.api_input_tokens > 0 or r.api_output_tokens > 0)
        )

        # Verify response is not empty
        assert (
            len(result_with_response.response) > 0
        ), "API response should not be empty"

        # Verify token usage
        assert (
            result_with_response.api_input_tokens > 0
        ), "Input tokens should be tracked"
        assert (
            result_with_response.api_output_tokens > 0
        ), "Output tokens should be tracked"


class TestIntegrationConfiguration:
    """Tests for integration test configuration and prerequisites."""

    def test_api_connectivity(self) -> None:
        """Verify that the Lightspeed-stack API is accessible."""
        assert (
            check_api_available()
        ), "Lightspeed-stack API should be available on localhost:8080"

    def test_openai_key_configured(self) -> None:
        """Verify that OPENAI_API_KEY is configured."""
        assert (
            check_openai_key_available()
        ), "OPENAI_API_KEY environment variable should be set"

# pylint: disable=redefined-outer-name,too-many-arguments,too-many-positional-arguments,too-few-public-methods
"""Offline evaluation e2e tests — requires OPENAI_API_KEY, no live Lightspeed API.

Covers all offline-capable metric families:
  - nlp: bleu, rouge, semantic_similarity_distance
  - custom (no LLM): keywords_eval, tool_eval
  - custom (LLM): answer_correctness, intent_eval
  - ragas: response_relevancy, faithfulness, context_recall, context_precision,
           context_utilization, context_relevance
  - geval: technical_accuracy
  - deepeval: conversation_completeness, conversation_relevancy, knowledge_retention

The pipeline runs ONCE per config format (session scope) — all tests share the cached
results, so LLM calls are not repeated across test cases.

Tests run against both config formats:
  - system-config-offline.yaml          (deprecated top-level llm:)
  - system-config-offline-llm-pool.yaml (modern llm_pool: + judge_panel:)

Run with: pytest tests/e2e/ -v -m offline_e2e
"""

import os
import tempfile
from pathlib import Path

import pytest

from lightspeed_evaluation import ConfigLoader, evaluate
from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.storage import FileBackendConfig
from lightspeed_evaluation.core.system import DataValidator

pytestmark = pytest.mark.offline_e2e


_E2E_DIR = Path(__file__).parent

_CONV_IDS = {
    "nlp": "nlp_metrics",
    "keywords": "keywords_eval",
    "tool": "tool_eval",
    "mrr": "mrr_metrics",
    "loop": "loop_eval_no_loop",
    "ragas_response": "ragas_response_metrics",
    "ragas_context": "ragas_context_metrics",
    "custom_llm": "custom_llm_metrics",
    "geval": "geval_metrics",
    "deepeval": "deepeval_conv_metrics",
}


@pytest.fixture(
    scope="session",
    params=[
        "system-config-offline.yaml",
        "system-config-offline-llm-pool.yaml",
    ],
)
def pipeline_results(request: pytest.FixtureRequest) -> list[EvaluationResult]:
    """Run the full offline pipeline once per config format and cache results.

    Session-scoped so LLM calls happen exactly twice total (once per config),
    regardless of how many test cases consume the results.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip(
            "OPENAI_API_KEY is not set — offline_e2e tests require a real OpenAI key"
        )

    config_path = _E2E_DIR / request.param
    eval_data_path = _E2E_DIR / "test_offline_evaluation_data.yaml"

    loader = ConfigLoader()
    system_config = loader.load_system_config(str(config_path))

    # Use a temp dir for output that lives for the session
    output_dir = tempfile.mkdtemp(prefix=f"e2e_offline_{request.param}_")
    system_config.storage = [FileBackendConfig(output_dir=output_dir)]

    assert system_config.agents is not None
    assert not system_config.agents.enabled

    validator = DataValidator(
        api_enabled=False,
        fail_on_invalid_data=system_config.core.fail_on_invalid_data,
    )
    evaluation_data = validator.load_evaluation_data(str(eval_data_path))
    return evaluate(system_config, evaluation_data)


def _results_for(
    results: list[EvaluationResult], conv_id: str
) -> list[EvaluationResult]:
    return [r for r in results if r.conversation_group_id == conv_id]


def _assert_valid_result(result: EvaluationResult) -> None:
    """Common structural assertions for any EvaluationResult."""
    assert isinstance(result, EvaluationResult)
    assert result.metric_identifier is not None
    assert result.result in (
        "PASS",
        "FAIL",
    ), f"Metric execution error for {result.metric_identifier}: result='{result.result}'"
    assert (
        result.score is not None
    ), f"Score is None for {result.metric_identifier} with result {result.result}"
    assert (
        0.0 <= result.score <= 1.0
    ), f"Score {result.score} out of [0, 1] for {result.metric_identifier}"


class TestRagasResponseMetrics:
    """End-to-end tests for ragas:response_relevancy and ragas:faithfulness."""

    def test_response_relevancy_produces_valid_result(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """ragas:response_relevancy returns a score in [0, 1]."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["ragas_response"])
            if r.metric_identifier == "ragas:response_relevancy"
        )
        _assert_valid_result(result)

    def test_faithfulness_produces_valid_result(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """ragas:faithfulness returns a score in [0, 1]."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["ragas_response"])
            if r.metric_identifier == "ragas:faithfulness"
        )
        _assert_valid_result(result)

    def test_no_api_tokens_used(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """Offline mode makes no agent API calls — api_input/output_tokens are 0."""
        for result in _results_for(pipeline_results, _CONV_IDS["ragas_response"]):
            assert result.api_input_tokens == 0
            assert result.api_output_tokens == 0

    def test_judge_llm_tokens_tracked(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """LLM judge tokens are tracked for ragas metrics."""
        ragas_results = _results_for(pipeline_results, _CONV_IDS["ragas_response"])
        has_tokens = any(
            (r.judge_llm_input_tokens or 0) > 0 or (r.judge_llm_output_tokens or 0) > 0
            for r in ragas_results
        )
        assert has_tokens, "At least one ragas result should have judge LLM token usage"


class TestRagasContextMetrics:
    """End-to-end tests for ragas context metrics."""

    @pytest.mark.parametrize(
        "metric_id",
        [
            "ragas:context_recall",
            "ragas:context_precision",
            "ragas:context_utilization",
            "ragas:context_relevance",
        ],
    )
    def test_context_metric_produces_valid_result(
        self,
        metric_id: str,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """Each ragas context metric returns a score in [0, 1]."""
        result = next(
            (
                r
                for r in _results_for(pipeline_results, _CONV_IDS["ragas_context"])
                if r.metric_identifier == metric_id
            ),
            None,
        )
        assert result is not None, f"{metric_id} result not found in pipeline output"
        _assert_valid_result(result)


class TestCustomLLMMetrics:
    """End-to-end tests for custom:answer_correctness and custom:intent_eval."""

    def test_answer_correctness_produces_valid_result(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """custom:answer_correctness returns a score in [0, 1] with judge tokens tracked."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["custom_llm"])
            if r.metric_identifier == "custom:answer_correctness"
        )
        _assert_valid_result(result)
        assert (
            result.judge_llm_input_tokens or 0
        ) > 0, "custom:answer_correctness should track judge LLM input tokens"

    def test_intent_eval_produces_valid_result(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """custom:intent_eval returns a score in [0, 1] with judge tokens tracked."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["custom_llm"])
            if r.metric_identifier == "custom:intent_eval"
        )
        _assert_valid_result(result)
        assert (
            result.judge_llm_input_tokens or 0
        ) > 0, "custom:intent_eval should track judge LLM input tokens"


class TestGEvalMetrics:
    """End-to-end tests for geval:technical_accuracy."""

    def test_technical_accuracy_produces_valid_result(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """geval:technical_accuracy returns a score in [0, 1] with judge tokens tracked."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["geval"])
            if r.metric_identifier == "geval:technical_accuracy"
        )
        _assert_valid_result(result)
        assert (
            result.judge_llm_input_tokens or 0
        ) > 0, "geval:technical_accuracy should track judge LLM input tokens"


class TestDeepEvalMetrics:
    """End-to-end tests for deepeval conversation metrics."""

    @pytest.mark.parametrize(
        "metric_id",
        [
            "deepeval:conversation_completeness",
            "deepeval:conversation_relevancy",
            "deepeval:knowledge_retention",
        ],
    )
    def test_conversation_metric_produces_valid_result(
        self,
        metric_id: str,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """Each deepeval conversation metric returns a score in [0, 1]."""
        result = next(
            (
                r
                for r in _results_for(pipeline_results, _CONV_IDS["deepeval"])
                if r.metric_identifier == metric_id
            ),
            None,
        )
        assert result is not None, f"{metric_id} result not found in pipeline output"
        _assert_valid_result(result)


class TestNLPMetrics:
    """End-to-end tests for nlp:bleu, nlp:rouge, nlp:semantic_similarity_distance."""

    @pytest.mark.parametrize(
        "metric_id",
        [
            "nlp:bleu",
            "nlp:rouge",
            "nlp:semantic_similarity_distance",
        ],
    )
    def test_nlp_metric_produces_valid_result(
        self,
        metric_id: str,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """Each NLP metric returns a score in [0, 1] with no LLM judge call."""
        result = next(
            (
                r
                for r in _results_for(pipeline_results, _CONV_IDS["nlp"])
                if r.metric_identifier == metric_id
            ),
            None,
        )
        assert result is not None, f"{metric_id} result not found in pipeline output"
        _assert_valid_result(result)
        assert result.api_input_tokens == 0
        assert result.api_output_tokens == 0


class TestCustomNoLLMMetrics:
    """End-to-end tests for custom:keywords_eval and custom:tool_eval."""

    def test_keywords_eval_matching_response_passes(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """custom:keywords_eval passes when response contains all expected keywords."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["keywords"])
            if r.metric_identifier == "custom:keywords_eval"
        )
        _assert_valid_result(result)
        assert result.result == "PASS", (
            f"Expected PASS for matching keywords but got {result.result}. "
            f"Reason: {result.reason}"
        )

    def test_tool_eval_exact_match_passes(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """custom:tool_eval passes when tool_calls exactly match expected_tool_calls."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["tool"])
            if r.metric_identifier == "custom:tool_eval"
        )
        _assert_valid_result(result)
        assert result.result == "PASS", (
            f"Expected PASS for matching tool calls but got {result.result}. "
            f"Reason: {result.reason}"
        )

    def test_loop_eval_no_loop_passes(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """custom:loop_eval passes when tool sequence contains no repeated loops."""
        result = next(
            r
            for r in _results_for(pipeline_results, _CONV_IDS["loop"])
            if r.metric_identifier == "custom:loop_eval"
        )
        _assert_valid_result(result)
        assert result.result == "PASS", (
            f"Expected PASS for loop-free tool sequence but got {result.result}. "
            f"Reason: {result.reason}"
        )


class TestMRRMetric:
    """End-to-end test for nlp:mrr."""

    def test_mrr_produces_valid_result(
        self,
        pipeline_results: list[EvaluationResult],
    ) -> None:
        """nlp:mrr returns a score in [0, 1] when the top context matches."""
        result = next(
            (
                r
                for r in _results_for(pipeline_results, _CONV_IDS["mrr"])
                if r.metric_identifier == "nlp:mrr"
            ),
            None,
        )
        assert result is not None, "nlp:mrr result not found in pipeline output"
        _assert_valid_result(result)
        assert result.api_input_tokens == 0
        assert result.api_output_tokens == 0

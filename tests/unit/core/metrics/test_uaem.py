# pylint: disable=redefined-outer-name,too-few-public-methods

"""Unit tests for the uaem_lib metric framework (Ragas-style).

Skipped unless the ``uaem-poc`` extra is installed, since the framework
depends on the metric library it ships.
"""

import json
from pathlib import Path
from typing import Any, Optional, cast

import pytest
from pytest_mock import MockerFixture

pytest.importorskip("lib_side", reason="requires the uaem-poc extra")

# pylint: disable=wrong-import-position
from lightspeed_evaluation.core.metrics.uaem import (
    RawScoreSink,
    UAEMLibMetrics,
    build_traces,
    call_json,
    to_lse_score,
)
from lightspeed_evaluation.core.metrics.uaem import (
    judge as judge_module,
)
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationScope,
    TurnData,
)

UAEM_METRIC_NAMES = [
    "req_coverage_prop_model",
    "scope_containment_prop_model",
    "granularity_fit_score_model",
]


class StubLLM:
    """Stands in for BaseCustomLLM, replying based on the prompt it is given."""

    def __init__(self, *_: Any, **__: Any) -> None:
        """Record the model name and the calls made."""
        self.model_name = "openai/gpt-4o-mini"
        self.calls: list[tuple[str, Optional[str]]] = []

    def call(
        self,
        prompt: str,
        return_single: bool = True,
        system_prompt: Optional[str] = None,
        **_: Any,
    ) -> str:
        """Return a canned structured reply for whichever judge prompt arrived."""
        del return_single
        self.calls.append((prompt, system_prompt))
        if "requirements" in prompt:
            return '{"total_requirements": 4, "requirements_satisfied": 3}'
        if "deviate materially" in prompt:
            return '{"count": 1}'
        return '{"label": -1}'


@pytest.fixture
def llm_manager(mocker: MockerFixture) -> Any:
    """Minimal stand-in for LLMManager."""
    manager = mocker.MagicMock()
    manager.get_model_name.return_value = "openai/gpt-4o-mini"
    manager.get_llm_params.return_value = {}
    return manager


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> StubLLM:
    """Replace BaseCustomLLM wherever the handler constructs one."""
    instance = StubLLM()
    monkeypatch.setattr(judge_module, "BaseCustomLLM", lambda *a, **k: instance)
    return instance


@pytest.fixture
def conv_data() -> EvaluationData:
    """Two-turn conversation covering both evaluation scopes."""
    return EvaluationData(
        conversation_group_id="conv1",
        turns=[
            TurnData(
                turn_id="t1",
                query="Explain A and B.",
                response="A is this. B is unrelated trivia. A matters. B is odd.",
            ),
            TurnData(turn_id="t2", query="And C?", response="C is that."),
        ],
    )


def turn_scope(conv_data: EvaluationData, index: int = 0) -> EvaluationScope:
    """Scope for a single turn."""
    return EvaluationScope(turn_idx=index, turn_data=conv_data.turns[index])


def conversation_scope() -> EvaluationScope:
    """Scope for the whole conversation."""
    return EvaluationScope(is_conversation=True)


class TestTraceMapping:
    """Tests for build_traces."""

    def test_turn_level_yields_one_trace(self, conv_data: EvaluationData) -> None:
        """Turn-level evaluation maps the turn to a single trace."""
        traces = build_traces(conv_data, turn_scope(conv_data))

        assert len(traces) == 1
        assert traces[0]["input"] == "Explain A and B."

    def test_conversation_level_yields_one_trace_per_turn(
        self, conv_data: EvaluationData
    ) -> None:
        """Conversation-level evaluation maps every usable turn."""
        assert len(build_traces(conv_data, conversation_scope())) == 2

    def test_turns_without_a_response_are_skipped(self) -> None:
        """A turn with no response cannot be scored and is dropped."""
        data = EvaluationData(
            conversation_group_id="c",
            turns=[
                TurnData(turn_id="t1", query="q"),
                TurnData(turn_id="t2", query="q2", response="r"),
            ],
        )

        assert len(build_traces(data, conversation_scope())) == 1


class TestScoreContract:
    """Tests for to_lse_score."""

    def test_zero_one_metrics_pass_through(self) -> None:
        """A proportion is reported unchanged."""
        assert to_lse_score(0.73, score_min=0.0, score_max=1.0) == 0.73

    def test_signed_scale_is_normalised_as_one_minus_abs(self) -> None:
        """A signed scale where 0 is ideal maps to 1 - |x|."""
        assert to_lse_score(-0.5, score_min=-1.0, score_max=1.0) == 0.5
        assert to_lse_score(0.0, score_min=-1.0, score_max=1.0) == 1.0
        assert to_lse_score(1.0, score_min=-1.0, score_max=1.0) == 0.0


class TestStructuredOutputRetry:
    """Tests for call_json."""

    def test_retries_once_on_an_unparsable_reply(self) -> None:
        """A judge that wraps JSON in prose gets one corrective retry."""
        # pylint: disable=import-outside-toplevel
        from lib_side.granularity_fit_score import GranularityLabel

        output_model = GranularityLabel

        class FlakyLLM:
            """Replies with prose first, then valid JSON."""

            def __init__(self) -> None:
                self.calls = 0

            def call(
                self,
                prompt: str,
                return_single: bool = True,
                system_prompt: Optional[str] = None,
                **_: Any,
            ) -> str:
                """Return prose on the first call and JSON on the second."""
                del prompt, return_single, system_prompt
                self.calls += 1
                return "sure thing!" if self.calls == 1 else '{"label": 1}'

        llm = FlakyLLM()
        parsed = call_json(
            cast(Any, llm),
            system_prompt="s",
            user_prompt="u",
            output_model=output_model,
        )

        assert parsed.label == 1
        assert llm.calls == 2


class TestHandler:
    """Tests for the uaem_lib handler."""

    @pytest.mark.usefixtures("stub_llm")
    @pytest.mark.parametrize("metric_name", UAEM_METRIC_NAMES)
    def test_every_metric_scores_within_the_contract(
        self,
        metric_name: str,
        conv_data: EvaluationData,
        llm_manager: Any,
        tmp_path: Path,
    ) -> None:
        """Every metric reports a score inside [0, 1]."""
        sink = RawScoreSink(tmp_path / "raw.jsonl")
        handler = UAEMLibMetrics(llm_manager, None, raw_sink=sink)

        score, reason = handler.evaluate(metric_name, conv_data, turn_scope(conv_data))

        assert score is not None
        assert 0.0 <= score <= 1.0
        assert metric_name in reason

    @pytest.mark.usefixtures("stub_llm")
    def test_unknown_metrics_are_rejected(
        self, conv_data: EvaluationData, llm_manager: Any
    ) -> None:
        """An unsupported metric name returns a null score and a reason."""
        handler = UAEMLibMetrics(llm_manager, None)

        score, reason = handler.evaluate("nope", conv_data, turn_scope(conv_data))

        assert score is None
        assert "Unsupported" in reason

    def test_system_prompt_is_sent_as_a_system_message(
        self, conv_data: EvaluationData, stub_llm: StubLLM, llm_manager: Any
    ) -> None:
        """The judge persona is a system message, not a prompt prefix."""
        handler = UAEMLibMetrics(llm_manager, None)

        handler.evaluate(
            "granularity_fit_score_model", conv_data, turn_scope(conv_data)
        )

        _, system_prompt = stub_llm.calls[-1]
        assert system_prompt is not None
        assert "evaluation judge" in system_prompt

    @pytest.mark.usefixtures("stub_llm")
    def test_sink_records_the_untransformed_score(
        self, conv_data: EvaluationData, llm_manager: Any, tmp_path: Path
    ) -> None:
        """The rollup needs -1.0, not the 0.0 that gets reported.

        Field names are the metric library's contract (``lib_side.report``):
        ``model_nm``, ``metric``, ``score``.
        """
        path = tmp_path / "raw.jsonl"
        sink = RawScoreSink(path)
        handler = UAEMLibMetrics(llm_manager, None, raw_sink=sink)

        handler.evaluate(
            "granularity_fit_score_model", conv_data, turn_scope(conv_data)
        )

        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["model_nm"] == "uaem_lib"
        assert row["metric"] == "granularity_fit_score_model"
        assert row["score"] == -1.0
        assert row["lse_score"] == 0.0
        assert row["turn_id"] == "t1"

    def test_conversation_level_scores_every_usable_turn(
        self, conv_data: EvaluationData, stub_llm: StubLLM, llm_manager: Any
    ) -> None:
        """Conversation scope judges each turn and averages across them."""
        handler = UAEMLibMetrics(llm_manager, None)

        score, _ = handler.evaluate(
            "req_coverage_prop_model", conv_data, conversation_scope()
        )

        assert score == pytest.approx(0.75)
        assert len(stub_llm.calls) == 2

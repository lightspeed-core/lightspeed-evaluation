"""Shared pieces for both UAEM metric frameworks.

Trace mapping, the [0, 1] score contract and the raw-score sink are the same
under either library architecture, so they live here rather than being
charged to one of them.
"""

import json
import re
import threading
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from lightspeed_evaluation.core.llm.custom import BaseCustomLLM
from lightspeed_evaluation.core.models import EvaluationScope

M = TypeVar("M", bound=BaseModel)

FRAMEWORK = "uaem_lib"

UAEM_METRICS = (
    "req_coverage_prop_model",
    "scope_containment_prop_model",
    "granularity_fit_score_model",
)

DEFAULT_RAW_SCORE_PATH = "results/uaem_raw_scores.jsonl"

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON for the requested schema. "
    "Reply with ONLY the JSON object, no prose and no markdown fences."
)
_SINK_LOCK = threading.Lock()


def build_traces(conv_data: Any, scope: EvaluationScope) -> list[dict[str, str]]:
    """Map evaluation data onto the library's {input, output} trace shape.

    Turn level: one trace from scope.turn_data.
    Conversation level: one trace per turn that has both query and response.

    Args:
        conv_data: Conversation data for the evaluation.
        scope: Turn or conversation context.

    Returns:
        A list of {"input": ..., "output": ...} dictionaries, possibly empty.
    """
    if scope.is_conversation:
        return [
            {"input": turn.query or "", "output": turn.response or ""}
            for turn in conv_data.turns
            if turn.query and turn.response
        ]
    turn_data = scope.turn_data
    if turn_data is None or not turn_data.query or not turn_data.response:
        return []
    return [{"input": turn_data.query, "output": turn_data.response}]


def parse_json(output_model: type[M], raw: str) -> M:
    """Parse a judge reply into ``output_model``, tolerating markdown fences."""
    return output_model.model_validate_json(_FENCE.sub("", raw).strip())


def call_json(
    llm: BaseCustomLLM,
    *,
    system_prompt: str,
    user_prompt: str,
    output_model: type[M],
) -> M:
    """Call the judge and parse a structured reply, retrying once.

    Passes ``output_model`` as ``response_format`` so litellm asks the
    provider to constrain its output to that schema when the model supports
    structured outputs. Some providers ignore or only loosely honor this, so
    a prompt-text parse failure still gets one corrective retry.

    Args:
        llm: Configured judge client.
        system_prompt: System message for the judge.
        user_prompt: Rendered judge prompt.
        output_model: Pydantic model the reply must satisfy.

    Returns:
        The parsed judge reply.

    Raises:
        ValidationError: If the retry also fails to parse.
    """
    raw = llm.call(
        user_prompt,
        return_single=True,
        system_prompt=system_prompt,
        response_format=output_model,
    )
    try:
        return parse_json(output_model, str(raw))
    except ValidationError:
        raw = llm.call(
            user_prompt + _RETRY_SUFFIX,
            return_single=True,
            system_prompt=system_prompt,
            response_format=output_model,
        )
        return parse_json(output_model, str(raw))


def to_lse_score(
    raw_score: float,
    *,
    score_min: Optional[float],
    score_max: Optional[float],
) -> float:
    """Map a metric's native score onto the framework's [0, 1] score contract.

    A [0, 1] metric passes through. A symmetric signed scale where 0 is the
    ideal (granularity fit, [-1, 1]) maps to ``1 - |x|`` — the same transform
    the UAEM taxonomy applies at rollup, which is why the rollup reads raw
    scores from the sink rather than reported scores, or the transform lands
    twice.

    Args:
        raw_score: Score on the metric's native scale.
        score_min: Minimum of the native scale, when declared.
        score_max: Maximum of the native scale, when declared.

    Returns:
        The score clamped into [0, 1].
    """
    del score_max
    value = (
        1.0 - abs(raw_score) if score_min is not None and score_min < 0 else raw_score
    )
    return max(0.0, min(1.0, value))


class ScoredMetric(NamedTuple):
    """One metric result, on both the native and the reported scale."""

    metric_name: str
    raw_score: float
    lse_score: float


class RawScoreSink:  # pylint: disable=too-few-public-methods
    """Append untransformed scores to JSONL for the post-run UAEM rollup.

    The rollup itself lives in the metric library (``lib_side.report``),
    which owns this file's schema: ``model_nm``, ``metric`` and ``score``
    are required, anything else is ignored. Only the write side is here,
    because only the write side knows about ``EvaluationScope``.
    """

    def __init__(self, path: str | Path, framework: str = FRAMEWORK) -> None:
        """Initialize the sink, creating the parent directory if needed."""
        self.path = Path(path)
        self.framework = framework
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self, scored: ScoredMetric, conv_data: Any, scope: EvaluationScope
    ) -> None:
        """Record one metric result on its native scale.

        Args:
            scored: Metric name with its native and reported scores.
            conv_data: Conversation data, for the group id.
            scope: Turn or conversation context.
        """
        row = {
            "model_nm": self.framework,
            "metric": scored.metric_name,
            "conversation_group_id": conv_data.conversation_group_id,
            "turn_id": (
                None
                if scope.is_conversation or scope.turn_data is None
                else scope.turn_data.turn_id
            ),
            "level": "conversation" if scope.is_conversation else "turn",
            "score": scored.raw_score,
            "lse_score": scored.lse_score,
        }
        with _SINK_LOCK, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

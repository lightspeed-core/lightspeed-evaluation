"""Shared helpers for the ``uaem_lib`` metrics framework.

Trace mapping, structured judge calls, and rendering the pass rule a metric
declares so it can travel on the result.
"""

import re
from typing import Any, Optional, TypeVar

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

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON for the requested schema. "
    "Reply with ONLY the JSON object, no prose and no markdown fences."
)


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


def format_pass_condition(condition: Any) -> Optional[str]:
    """Render the pass rule a metric declares, on its native scale.

    ``metrics.schema.TracePassCondition`` has no string form, and the
    framework's own ``threshold`` describes the reported [0, 1] score
    instead. Carrying the native rule as text keeps a reader able to see
    what the native score was judged against.

    Args:
        condition: The metric's ``default_trace_pass_condition``, or None.

    Returns:
        E.g. ``"gte(0.9)"`` or ``"between(0.0, 0.0)"``; None when the
        metric declares no condition.
    """
    if condition is None:
        return None
    operator = condition.operator.value
    if operator == "between":
        return f"{operator}({condition.lower}, {condition.upper})"
    return f"{operator}({condition.value})"

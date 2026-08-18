"""Deterministic agent loop detection from tool-call sequences.

Detects exact consecutive repeats, same-tool thrashing, and excessive
parent-child depth. The evaluation pipeline (``custom:loop_eval``) reads
``TurnData.tool_calls`` only. :func:`evaluate_loops_from_trace` is a
programmatic helper for the internal trace model; it is not eval-data input
and is not called by ``lightspeed-eval``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Optional

from lightspeed_evaluation.core.models import EvaluationData, TurnData
from lightspeed_evaluation.core.models.trace import Span, SpanType, Trace
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

DEFAULT_EXACT_LOOP_THRESHOLD = 3
DEFAULT_SOFT_LOOP_THRESHOLD = 3
DEFAULT_MAX_RECURSIVE_DEPTH = 10

_PARENT_KEYS = ("parent_span_id", "parent_id")
_SPAN_ID_KEYS = ("span_id", "id")


@dataclass(frozen=True)
class LoopEvalConfig:
    """Thresholds for loop and depth detection.

    All values must be integers ``>= 1``. Direct construction with an invalid
    threshold raises :class:`ConfigurationError` so scoring cannot divide by
    zero. YAML and metric metadata should use :func:`config_from_metadata`,
    which substitutes defaults for missing or invalid keys.

    Raises:
        ConfigurationError: If any threshold is not an integer ``>= 1``.
    """

    exact_loop_threshold: int = DEFAULT_EXACT_LOOP_THRESHOLD
    soft_loop_threshold: int = DEFAULT_SOFT_LOOP_THRESHOLD
    max_recursive_depth: int = DEFAULT_MAX_RECURSIVE_DEPTH

    def __post_init__(self) -> None:
        """Reject non-positive or non-integer thresholds."""
        for name in (
            "exact_loop_threshold",
            "soft_loop_threshold",
            "max_recursive_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(
                    f"LoopEvalConfig.{name} must be an integer >= 1, got {value!r}"
                )


LOOP_EVAL_DEFAULTS: dict[str, Any] = asdict(LoopEvalConfig())


@dataclass(frozen=True)
class _IndexedCall:
    """A single invocation in evaluation order."""

    index: int
    name: str
    arguments: Any
    span_id: Optional[str] = None
    parent_id: Optional[str] = None
    turn_id: Optional[str] = None


@dataclass(frozen=True)
class _Finding:
    """One detected loop or depth violation."""

    length: int
    threshold: int
    detail: str


@dataclass(frozen=True)
class _Run:
    """A consecutive run of equal keys."""

    start_index: int
    length: int


def config_from_metadata(metadata: Optional[dict[str, Any]]) -> LoopEvalConfig:
    """Build :class:`LoopEvalConfig` from metric metadata.

    Args:
        metadata: Merged system/turn metadata for ``custom:loop_eval``.

    Returns:
        Config with defaults applied for missing or invalid keys.
    """
    data = metadata or {}
    return LoopEvalConfig(
        exact_loop_threshold=_positive_int(
            data.get("exact_loop_threshold"), DEFAULT_EXACT_LOOP_THRESHOLD
        ),
        soft_loop_threshold=_positive_int(
            data.get("soft_loop_threshold"), DEFAULT_SOFT_LOOP_THRESHOLD
        ),
        max_recursive_depth=_positive_int(
            data.get("max_recursive_depth"), DEFAULT_MAX_RECURSIVE_DEPTH
        ),
    )


def evaluate_loops(
    conv_data: Any,
    _turn_idx: Optional[int],
    turn_data: Optional[TurnData],
    is_conversation: bool,
    config: Optional[LoopEvalConfig] = None,
) -> tuple[Optional[float], str]:
    """Evaluate loop detection for a turn or conversation.

    Args:
        conv_data: Conversation data (used when ``is_conversation`` is True).
        _turn_idx: Turn index (unused; required by CustomMetrics).
        turn_data: Turn data containing ``tool_calls``.
        is_conversation: Whether to score each turn independently and
            aggregate findings. Consecutive calls are not joined across turns.
        config: Detection thresholds. Defaults are used when omitted.

    Returns:
        ``(score, reason)`` where 1.0 means no loops and 0.0 means severe
        looping. ``score`` is ``None`` when required data is missing.
    """
    resolved = config or LoopEvalConfig()
    if is_conversation:
        return _evaluate_conversation(conv_data, resolved)
    if turn_data is None:
        return None, "TurnData is required for loop evaluation"
    return evaluate_loop_calls(
        turn_data.tool_calls, config=resolved, turn_id=turn_data.turn_id
    )


def evaluate_loop_calls(
    tool_calls: Optional[list[list[dict[str, Any]]]],
    config: Optional[LoopEvalConfig] = None,
    turn_id: Optional[str] = None,
) -> tuple[Optional[float], str]:
    """Score a nested ``tool_calls`` sequence for loops.

    Args:
        tool_calls: Actual tool calls in ``list[list[dict]]`` form. An empty
            list scores 1.0 (no loops). ``None`` is a missing-data error.
        config: Detection thresholds.
        turn_id: Optional turn identifier included in findings.

    Returns:
        ``(score, reason)``.
    """
    if tool_calls is None:
        return None, "No tool_calls provided for loop evaluation"
    return _score_calls(
        _flatten_tool_calls(tool_calls, turn_id=turn_id),
        config or LoopEvalConfig(),
    )


def evaluate_loops_from_trace(
    trace: Trace,
    config: Optional[LoopEvalConfig] = None,
) -> tuple[float, str]:
    """Score loop/depth patterns from a normalized internal :class:`Trace`.

    Not used by ``lightspeed-eval`` or ``custom:loop_eval``. Call this from
    Python after converting a platform payload with the trace adapters.
    Consecutive exact/soft loops use TOOL spans in start-time order. Recursive
    depth walks the full span parent chain and reports TOOL leaves.

    Args:
        trace: Internal trace model.
        config: Detection thresholds.

    Returns:
        ``(score, reason)``. Missing tool spans scores 1.0.
    """
    resolved = config or LoopEvalConfig()
    tool_spans = _ordered_tool_spans(trace)
    tool_calls = [
        _indexed_from_span(span, index) for index, span in enumerate(tool_spans)
    ]
    tool_indexes = {call.span_id: call.index for call in tool_calls if call.span_id}
    all_nodes = [
        _indexed_from_span(span, tool_indexes.get(span.span_id, index))
        for index, span in enumerate(trace.spans)
    ]
    return _score_calls(
        tool_calls,
        resolved,
        depth_nodes=all_nodes,
        reportable_ids={span.span_id for span in tool_spans},
    )


def _evaluate_conversation(
    conv_data: Optional[EvaluationData],
    config: LoopEvalConfig,
) -> tuple[Optional[float], str]:
    """Evaluate loop detection per turn, then aggregate findings.

    Exact/soft loops and recursive depth are scored within each turn (one
    user query). Consecutive calls are not joined across turns: repeating a
    tool for a new query is not thrashing. Missing ``tool_calls`` (``None``)
    on any turn is a data error. An empty list is valid.
    """
    if conv_data is None:
        return None, "Conversation data is required for conversation-level loop eval"

    all_findings: list[_Finding] = []
    call_count = 0
    for turn in conv_data.turns:
        if turn.tool_calls is None:
            return (
                None,
                (
                    f"No tool_calls provided on turn {turn.turn_id} "
                    "for loop evaluation"
                ),
            )
        calls = _flatten_tool_calls(turn.tool_calls, turn_id=turn.turn_id)
        call_count += len(calls)
        all_findings.extend(_findings_for_calls(calls, config))
    return _score_findings(all_findings, call_count)


def _flatten_tool_calls(
    tool_calls: list[list[dict[str, Any]]],
    turn_id: Optional[str] = None,
    start_index: int = 0,
) -> list[_IndexedCall]:
    """Flatten nested tool-call sequences into evaluation order."""
    calls: list[_IndexedCall] = []
    index = start_index
    for sequence in tool_calls:
        if not isinstance(sequence, list):
            continue
        for raw in sequence:
            if not isinstance(raw, dict):
                continue
            calls.append(_indexed_from_dict(raw, index, turn_id))
            index += 1
    return calls


def _score_calls(
    calls: list[_IndexedCall],
    config: LoopEvalConfig,
    *,
    depth_nodes: Optional[list[_IndexedCall]] = None,
    reportable_ids: Optional[set[str]] = None,
) -> tuple[float, str]:
    """Score consecutive loops plus recursive depth."""
    return _score_findings(
        _findings_for_calls(
            calls,
            config,
            depth_nodes=depth_nodes,
            reportable_ids=reportable_ids,
        ),
        len(calls),
    )


def _findings_for_calls(
    calls: list[_IndexedCall],
    config: LoopEvalConfig,
    *,
    depth_nodes: Optional[list[_IndexedCall]] = None,
    reportable_ids: Optional[set[str]] = None,
) -> list[_Finding]:
    """Collect exact/soft loop and recursive-depth findings for one sequence."""
    findings = _consecutive_findings(calls, config)
    depth_finding = _depth_finding(depth_nodes or calls, config, reportable_ids)
    if depth_finding is not None:
        findings.append(depth_finding)
    return findings


def _indexed_from_dict(
    raw: dict[str, Any], index: int, turn_id: Optional[str]
) -> _IndexedCall:
    """Build an indexed call from a tool_calls dict."""
    name = raw.get("tool_name") or raw.get("name") or "unknown"
    arguments = raw.get("arguments", raw.get("args", {}))
    if isinstance(arguments, str):
        arguments = _parse_json_arguments(arguments)
    return _IndexedCall(
        index=index,
        name=str(name).strip() or "unknown",
        arguments=arguments,
        span_id=_first_str(raw, _SPAN_ID_KEYS),
        parent_id=_first_str(raw, _PARENT_KEYS),
        turn_id=turn_id,
    )


def _indexed_from_span(span: Span, index: int) -> _IndexedCall:
    """Build an indexed call from a trace span."""
    return _IndexedCall(
        index=index,
        name=span.name.strip() or "unknown",
        arguments=span.inputs if span.inputs is not None else {},
        span_id=span.span_id,
        parent_id=span.parent_span_id,
    )


def _consecutive_findings(
    calls: list[_IndexedCall],
    config: LoopEvalConfig,
) -> list[_Finding]:
    """Detect exact and soft consecutive runs that meet thresholds."""
    exact_run = _longest_run(calls, key=lambda call: (call.name, _canonical_args(call)))
    exact = _finding_from_run(
        calls,
        exact_run,
        config.exact_loop_threshold,
        "Exact loop",
        "with identical arguments ",
    )

    soft_run = _longest_run(calls, key=lambda call: call.name)
    same_window = (
        exact_run is not None
        and soft_run is not None
        and exact_run.start_index == soft_run.start_index
        and exact_run.length == soft_run.length
    )
    findings: list[_Finding] = []
    if exact is not None:
        findings.append(exact)
    if not same_window:
        soft = _finding_from_run(
            calls,
            soft_run,
            config.soft_loop_threshold,
            "Soft loop",
            "consecutively ",
        )
        if soft is not None:
            findings.append(soft)
    return findings


def _finding_from_run(
    calls: list[_IndexedCall],
    run: Optional[_Run],
    threshold: int,
    label: str,
    extra: str,
) -> Optional[_Finding]:
    """Build a finding from a consecutive run, if it meets ``threshold``."""
    if run is None or run.length < threshold:
        return None
    start = calls[run.start_index]
    return _Finding(
        length=run.length,
        threshold=threshold,
        detail=(
            f"{label}: '{start.name}' called {run.length} times "
            f"{extra}starting at {_location(start)} "
            f"(threshold {threshold})"
        ),
    )


def _longest_run(
    calls: list[_IndexedCall],
    key: Callable[[_IndexedCall], Any],
) -> Optional[_Run]:
    """Return the longest consecutive run under ``key``.

    Ties keep the earliest start index. Returns ``None`` when ``calls`` is empty.
    """
    if not calls:
        return None

    best_start = 0
    best_length = 1
    run_start = 0
    run_length = 1
    previous = key(calls[0])

    for offset, call in enumerate(calls[1:], start=1):
        current = key(call)
        if current == previous:
            run_length += 1
            if run_length > best_length:
                best_length = run_length
                best_start = run_start
        else:
            run_start = offset
            run_length = 1
            previous = current

    return _Run(start_index=best_start, length=best_length)


def _depth_finding(
    nodes: list[_IndexedCall],
    config: LoopEvalConfig,
    reportable_ids: Optional[set[str]] = None,
) -> Optional[_Finding]:
    """Find the deepest parent chain among reportable nodes."""
    by_id = {node.span_id: node for node in nodes if node.span_id}
    if not by_id:
        return None

    deepest: Optional[_IndexedCall] = None
    max_depth = 0
    deepest_chain: list[str] = []
    for node in nodes:
        if reportable_ids is not None and (
            not node.span_id or node.span_id not in reportable_ids
        ):
            continue
        chain = _ancestor_chain(node, by_id)
        if len(chain) > max_depth:
            max_depth = len(chain)
            deepest = node
            deepest_chain = chain

    if deepest is None or max_depth <= config.max_recursive_depth:
        return None
    return _Finding(
        length=max_depth,
        threshold=config.max_recursive_depth,
        detail=(
            f"Excessive recursive depth: {max_depth} "
            f"(threshold {config.max_recursive_depth}) along "
            f"{' -> '.join(deepest_chain)} ending at {_location(deepest)}"
        ),
    )


def _ancestor_chain(
    node: _IndexedCall,
    by_id: dict[str, _IndexedCall],
) -> list[str]:
    """Return the name chain from root to ``node``."""
    names: list[str] = []
    seen: set[str] = set()
    current: Optional[_IndexedCall] = node
    while current is not None:
        if current.span_id and current.span_id in seen:
            names.append(f"{current.name}(cycle)")
            break
        if current.span_id:
            seen.add(current.span_id)
        names.append(current.name)
        parent_id = current.parent_id
        current = by_id.get(parent_id) if parent_id else None
    names.reverse()
    return names


def _ordered_tool_spans(trace: Trace) -> list[Span]:
    """Return TOOL spans ordered by start time, then original order."""
    indexed = [
        (index, span)
        for index, span in enumerate(trace.spans)
        if span.span_type == SpanType.TOOL
    ]
    indexed.sort(
        key=lambda item: (
            item[1].start_time is None,
            item[1].start_time.timestamp() if item[1].start_time else 0.0,
            item[0],
        )
    )
    return [span for _index, span in indexed]


def _score_findings(
    findings: list[_Finding],
    call_count: int,
) -> tuple[float, str]:
    """Convert findings into a 0–1 score and a human-readable reason.

    Penalty for a finding of length L against threshold T is
    ``min(1, (L - T + 1) / T)``. The worst finding determines the score:
    ``1.0 - max(penalties)``.
    """
    if not findings:
        return 1.0, f"No loops detected across {call_count} tool call(s)"

    penalties = [
        min(1.0, (finding.length - finding.threshold + 1) / finding.threshold)
        for finding in findings
    ]
    score = max(0.0, 1.0 - max(penalties))
    return round(score, 4), "; ".join(finding.detail for finding in findings)


def _canonical_args(call: _IndexedCall) -> str:
    """Stable JSON form of arguments for exact-loop comparison."""
    try:
        return json.dumps(call.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(call.arguments)


def _parse_json_arguments(raw: str) -> Any:
    """Parse JSON argument strings; return the original string on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _first_str(raw: dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    """Return the first non-empty string value among ``keys``."""
    for key in keys:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _location(call: _IndexedCall) -> str:
    """Human-readable start location for a finding."""
    if call.turn_id:
        return f"turn {call.turn_id}, tool call index {call.index}"
    return f"tool call index {call.index}"


def _positive_int(value: Any, default: int) -> int:
    """Coerce a metadata value to an int >= 1, else ``default``."""
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default

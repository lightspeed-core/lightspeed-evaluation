"""Curate splitter: reconcile eval results with signals into dataset splits.

The splitter post-processes an existing evaluation run. It reads a results summary
JSON (``*_summary.json``) for the per-turn pass/fail verdict and the per-turn
``signals`` carried onto each result. An evaluation-data YAML may be supplied as a
fallback source of signals and turn content for results that predate
signal-carrying runs.

It applies the internal rule table
(:mod:`lightspeed_evaluation.core.curate.rule_table`) per turn and writes one
evaluation-data YAML per bucket (positive / negative / edge / unconfirmed) plus a
summary report.
"""

import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from lightspeed_evaluation.core.curate.rule_table import (
    CurateLabel,
    Eval,
    classify,
    eval_from_results,
)
from lightspeed_evaluation.core.ingestion.config import OverrideSpec, load_curate_config

logger = logging.getLogger(__name__)

BUCKETS = ("positive", "negative", "edge", "unconfirmed")


ResultKey = tuple[str, Optional[str]]


@dataclass(frozen=True)
class ScopeOptions:
    """How signals are scoped across a conversation during bucketing."""

    conversation_scope: bool = False
    drop_signal_turn: bool = False


def _load_results(results_path: str) -> dict[ResultKey, dict[str, Any]]:
    """Load a results summary JSON, grouping per turn.

    Returns ``(conversation_group_id, turn_id) -> {statuses, signals, query,
    response}``, accumulating one entry per turn across its metric rows.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    grouped: dict[ResultKey, dict[str, Any]] = {}
    for r in results:
        turn_id = r.get("turn_id")
        # Conversation-level metrics carry no turn_id; they are not real turns and
        # must not be reconstructed into a dataset bucket.
        if turn_id is None:
            continue
        key = (r["conversation_group_id"], turn_id)
        entry = grouped.setdefault(
            key, {"statuses": [], "signals": None, "query": None, "response": None}
        )
        entry["statuses"].append(r["result"])
        if entry["signals"] is None and r.get("signals"):
            entry["signals"] = r["signals"]
        if entry["query"] is None and r.get("query"):
            entry["query"] = r["query"]
        if entry["response"] is None and r.get("response"):
            entry["response"] = r["response"]
    return grouped


def _load_conversations(eval_data_path: str) -> list[dict[str, Any]]:
    """Load the evaluation-data YAML, supporting list and dict formats."""
    with open(eval_data_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data.get("conversations", [])
    return data or []


def _conversations_from_results(
    results_map: dict[ResultKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct minimal conversations from result rows (no eval-data file)."""
    conversations: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for (conv_id, turn_id), entry in results_map.items():
        if conv_id not in conversations:
            conversations[conv_id] = {"conversation_group_id": conv_id, "turns": []}
            order.append(conv_id)
        turn: dict[str, Any] = {"turn_id": turn_id}
        if entry.get("query"):
            turn["query"] = entry["query"]
        if entry.get("response"):
            turn["response"] = entry["response"]
        conversations[conv_id]["turns"].append(turn)
    return [conversations[c] for c in order]


def _turn_signals(turn: dict[str, Any]) -> dict[str, Any]:
    """Return the signals dict for a turn (empty when absent)."""
    signals = turn.get("signals")
    return signals if isinstance(signals, dict) else {}


def _write_split(
    conversations: list[dict[str, Any]],
    output_dir: Path,
    bucket: str,
) -> int:
    """Write one bucket's conversations to ``<bucket>.yaml``; return turn count."""
    turn_count = sum(len(c["turns"]) for c in conversations)
    path = output_dir / f"{bucket}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            conversations,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )
    logger.info(
        "Curate split '%s': %d conversations, %d turns -> %s",
        bucket,
        len(conversations),
        turn_count,
        path,
    )
    return turn_count


def _spectrum_bins(
    bucket_confidence: dict[str, Any],
) -> list[tuple[str, int, str]]:
    """Order all weighted tiers along the confidence spectrum, left to right.

    positive (strongest first) -> edge (least conflict first) -> negative
    (weakest first, so the strongest/full negative sits at the far right). Each
    bin is ``(label, turn_count, bucket)``.
    """
    bins: list[tuple[str, int, str]] = []

    def dist(bucket: str) -> dict[float, int]:
        return {float(k): v for k, v in bucket_confidence.get(bucket, {}).items()}

    pos = dist("positive")
    for s in sorted(pos, reverse=True):  # full positive on the left
        bins.append((f"positive confidence={s:g}", pos[s], "positive"))

    edge = dist("edge")
    for s in sorted(edge):  # best (least disagreement) first
        bins.append((f"edge conflict={s:g}", edge[s], "edge"))

    neg = dist("negative")
    for s in sorted(neg):  # full negative ends up on the far right
        bins.append((f"negative confidence={s:g}", neg[s], "negative"))
    return bins


def _render_distribution(bucket_confidence: dict[str, Any]) -> list[str]:
    """Render one quality-spectrum histogram across all buckets.

    A single axis: most-confident positive on the left, through weak positives
    and edges (ordered by rising conflict), to the most-confident ("full")
    negative on the right. Lets you pick a cutoff for a golden set at a glance.
    """
    lines = ["## Quality spectrum (most-confident positive -> negative)", ""]
    lines.append(
        "Confidence = summed strength of the signals driving the bucket "
        "(for edges, the magnitude of disagreement). Read left (strong positive) "
        "to right (strong negative); edges in the middle rise by conflict."
    )
    lines.append("")

    bins = _spectrum_bins(bucket_confidence)
    if not bins:
        lines.append("_No weighted signals._")
        lines.append("")
        return lines

    peak = max(count for _, count, _ in bins) or 1
    lines.append("| Tier (left = most confident positive) | Turns | |")
    lines.append("|---|---|---|")
    for label, count, _bucket in bins:
        bar_cells = "█" * max(1, round(24 * count / peak)) if count else ""
        lines.append(f"| {label} | {count} | `{bar_cells}` |")
    lines.append("")

    unconfirmed = sum(
        v
        for v in {
            float(k): v for k, v in bucket_confidence.get("unconfirmed", {}).items()
        }.values()
    )
    if unconfirmed:
        lines.append(f"_Off-spectrum — unconfirmed (no signal): {unconfirmed} turns._")
        lines.append("")
    return lines


def _render_propagation(prop: dict[str, Any]) -> list[str]:
    """Render signal reach: turns holding a signal and turns it affects per convo."""
    if not prop:
        return []
    lines = ["## Signal reach", ""]
    lines.append(
        f"Scope: **{prop.get('scope', 'turn')}**"
        f" · drop signal turn: **{prop.get('drop_signal_turn', False)}**"
    )
    lines.append("")
    lines.append(f"- turns holding a signal: {prop.get('turns_with_signal', 0)}")
    lines.append(f"- turns reached by propagation: {prop.get('turns_affected', 0)}")
    lines.append(f"- signal turns excluded: {prop.get('excluded_turns', 0)}")
    lines.append("")

    per_conv = prop.get("per_conversation", [])
    if per_conv:
        lines.append("| Conversation | Turns | Signal turns | Turns affected |")
        lines.append("|---|---|---|---|")
        for p in per_conv:
            lines.append(
                f"| {p['conversation_group_id']} | {p['turns']} "
                f"| {p['signal_turns']} | {p['affected']} |"
            )
        lines.append("")
    return lines


def _render_report(report: dict[str, Any]) -> str:
    """Render the curate report as Markdown."""
    lines = ["# Curate report", ""]
    lines.append("## Bucket counts")
    lines.append("")
    prov = report.get("bucket_provenance", {})
    lines.append(
        "**signal-bearing turns** = turns carrying the signal themselves (the rest "
        "are propagated from the conversation under `--conversation-scope`). "
        "`N (ignored)` = those signal turns dropped by `--drop-signal-turn`, "
        "counted but excluded from output."
    )
    lines.append("")
    lines.append("| Bucket | Turns | signal-bearing turns |")
    lines.append("|---|---|---|")
    for bucket in BUCKETS:
        bp = prov.get(bucket, {})
        ignored = bp.get("ignored", 0)
        signal_turns = f"{ignored} (ignored)" if ignored else str(bp.get("actual", 0))
        lines.append(
            f"| {bucket} | {report['bucket_turn_counts'].get(bucket, 0)} "
            f"| {signal_turns} |"
        )
    lines.append(f"| **skipped (no verdict)** | {report['skipped_turns']} | |")
    lines.append("")

    lines.extend(_render_distribution(report.get("bucket_confidence", {})))
    lines.extend(_render_propagation(report.get("signal_propagation", {})))

    review = report["needs_review"]
    lines.append(f"## Needs human review ({len(review)}) — strongest conflict first")
    lines.append("")
    if review:
        ranked = sorted(review, key=lambda r: r.get("conflict", 0.0), reverse=True)
        lines.append("| Conversation | Turn | Edge magnitude | Reason | Note |")
        lines.append("|---|---|---|---|---|")
        for item in ranked:
            lines.append(
                f"| {item['conversation_group_id']} | {item['turn_id']} "
                f"| {float(item.get('conflict', 0.0)):g} "
                f"| {item.get('reason', '')} | {item.get('note', '')} |"
            )
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


class _Accumulator:
    """Collects bucketed turns, counts and review items across conversations."""

    def __init__(self) -> None:
        self.split_convs: dict[str, list[dict[str, Any]]] = {b: [] for b in BUCKETS}
        # Per-bucket totals split by signal provenance (actual vs derived).
        self.bucket_stats: dict[str, dict[str, int]] = {
            b: {"total": 0, "actual": 0, "derived": 0, "ignored": 0} for b in BUCKETS
        }
        # Per-bucket strength breakdown: bucket -> {confidence -> count}.
        self.bucket_confidence: dict[str, dict[float, int]] = {
            b: defaultdict(int) for b in BUCKETS
        }
        self.needs_review: list[dict[str, Any]] = []
        self.skipped_turns = 0
        self.excluded_turns = 0
        # Per-conversation propagation figures: conv_id -> {turns, signal_turns, affected}.
        self.propagation: list[dict[str, Any]] = []

    def add_turn(
        self,
        conv_id: str,
        turn: dict[str, Any],
        label: CurateLabel,
        source: str = "none",
    ) -> dict[str, Any]:
        """Record a labelled turn and return its copy with the label embedded.

        ``source`` is the signal provenance: ``actual`` (the turn's own signal),
        ``derived`` (propagated from elsewhere in the conversation) or ``none``. The
        turn is expected to already carry the effective signals used for the label.
        """
        labelled_turn = deepcopy(turn)
        labelled_turn.setdefault("signals", {})
        curate_label = _label_to_dict(label)
        curate_label["signal_source"] = source
        labelled_turn["signals"]["curate"] = curate_label

        stats = self.bucket_stats[label.bucket]
        stats["total"] += 1
        if source in ("actual", "derived"):
            stats[source] += 1
        self.bucket_confidence[label.bucket][label.confidence] += 1
        if label.needs_review:
            self.needs_review.append(
                {
                    "conversation_group_id": conv_id,
                    "turn_id": turn.get("turn_id"),
                    "bucket": label.bucket,
                    "conflict": label.confidence,
                    "reason": label.reason,
                    "note": label.note,
                }
            )
        return labelled_turn

    def add_ignored(self, bucket: str) -> None:
        """Count a dropped signal-bearing turn against the bucket it would land in."""
        self.bucket_stats[bucket]["ignored"] += 1

    def record_propagation(
        self, conv_id: str, turns: int, signal_turns: int, affected: int
    ) -> None:
        """Record per-conversation signal reach (only for convos with signals)."""
        if signal_turns:
            self.propagation.append(
                {
                    "conversation_group_id": conv_id,
                    "turns": turns,
                    "signal_turns": signal_turns,
                    "affected": affected,
                }
            )

    def add_conversation(
        self, conv: dict[str, Any], bucketed_turns: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Emit one conversation shell per bucket that has turns."""
        for bucket in BUCKETS:
            if bucketed_turns[bucket]:
                conv_copy = {k: deepcopy(v) for k, v in conv.items() if k != "turns"}
                conv_copy["turns"] = bucketed_turns[bucket]
                self.split_convs[bucket].append(conv_copy)


def _resolve_signals(
    turn: dict[str, Any], entry: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Signals for a turn: prefer those carried on the result, else the turn."""
    if entry and entry.get("signals"):
        return entry["signals"]
    return _turn_signals(turn)


def _is_active(sig: Any) -> bool:
    """True if a signal drives bucketing (non-neutral polarity, positive strength)."""
    if not isinstance(sig, dict):
        return False
    try:
        weight = float(sig.get("strength", 0.0) or 0.0)
    except (TypeError, ValueError):
        weight = 0.0
    return sig.get("polarity") in ("positive", "negative") and weight > 0


def _active_signals(signals: dict[str, Any]) -> dict[str, Any]:
    """Signals that drive bucketing (non-neutral polarity and positive strength)."""
    return {name: sig for name, sig in signals.items() if _is_active(sig)}


@dataclass
class _ConvInfo:
    """Per-conversation resolved turns, plus per-turn segment signals.

    ``governing[i]`` holds the active signals that label turn ``i``'s segment: the
    next signal turn at or after ``i`` (backward arc). Tail turns after the last
    signal have an empty mapping. ``signal_turn_ids`` marks the segment boundaries.
    """

    turn_infos: list[tuple[dict[str, Any], Optional[Eval], dict[str, Any]]]
    governing: list[dict[str, Any]]
    signal_turn_ids: set[Any]


def _gather_conv(
    conv: dict[str, Any],
    conv_id: str,
    results_map: dict[ResultKey, dict[str, Any]],
) -> _ConvInfo:
    """Resolve each turn's eval + signals and the per-turn segment signal set.

    Segments run backward from each signal turn to the previous one: a signal
    labels the arc leading up to it. Turns after the last signal get no signal.
    """
    turn_infos: list[tuple[dict[str, Any], Optional[Eval], dict[str, Any]]] = []
    actives: list[dict[str, Any]] = []
    signal_turn_ids: set[Any] = set()
    for turn in conv.get("turns", []):
        entry = results_map.get((conv_id, turn.get("turn_id")))
        eval_result = eval_from_results(entry["statuses"]) if entry else None
        signals = _resolve_signals(turn, entry)
        turn_infos.append((turn, eval_result, signals))
        active = _active_signals(signals)
        actives.append(active)
        if active:
            signal_turn_ids.add(turn.get("turn_id"))
    return _ConvInfo(turn_infos, _backward_segments(actives), signal_turn_ids)


def _backward_segments(actives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map each turn to the nearest signal turn at/after it (empty for the tail)."""
    governing: list[dict[str, Any]] = [{} for _ in actives]
    carry: dict[str, Any] = {}
    for i in range(len(actives) - 1, -1, -1):
        if actives[i]:
            carry = actives[i]
        governing[i] = carry
    return governing


def _propagate(
    signals: dict[str, Any], segment_signals: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Overlay a segment's signals over a turn's neutral/absent ones (never active)."""
    effective = dict(signals)
    changed = False
    for name, sig in segment_signals.items():
        if not _is_active(effective.get(name)):
            effective[name] = sig
            changed = True
    return effective, changed


def _turn_effective(
    info: _ConvInfo,
    idx: int,
    is_signal_turn: bool,
    signals: dict[str, Any],
    scope: ScopeOptions,
) -> tuple[Optional[dict[str, Any]], bool]:
    """Effective signals for a turn, or (None, False) if the turn is dropped.

    The bool reports whether propagation added a signal this turn didn't hold.
    """
    segment = info.governing[idx]
    if not (scope.conversation_scope and segment):
        return signals, False
    if scope.drop_signal_turn and is_signal_turn:
        return None, False
    effective, changed = _propagate(signals, segment)
    return effective, changed and not is_signal_turn


def _record_dropped(
    eval_result: Eval,
    signals: dict[str, Any],
    overrides: list[OverrideSpec],
    acc: _Accumulator,
) -> None:
    """Count a dropped signal-bearing turn against the bucket it would land in."""
    acc.excluded_turns += 1
    acc.add_ignored(classify(eval_result, signals, overrides).bucket)


def _classify_conv_turns(
    info: _ConvInfo,
    conv_id: str,
    overrides: list[OverrideSpec],
    acc: _Accumulator,
    scope: ScopeOptions,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Bucket a conversation's turns; return (bucketed turns, turns propagated to)."""
    bucketed: dict[str, list[dict[str, Any]]] = {b: [] for b in BUCKETS}
    affected = 0
    for idx, (turn, eval_result, signals) in enumerate(info.turn_infos):
        if eval_result is None:
            acc.skipped_turns += 1
            continue
        is_signal_turn = turn.get("turn_id") in info.signal_turn_ids
        effective, propagated = _turn_effective(
            info, idx, is_signal_turn, signals, scope
        )
        if effective is None:
            _record_dropped(eval_result, signals, overrides, acc)
            continue
        affected += int(propagated)
        label = classify(eval_result, effective, overrides)
        # Emit the turn with the effective signals so its evidence matches the label.
        bucketed[label.bucket].append(
            acc.add_turn(
                conv_id,
                {**turn, "signals": effective},
                label,
                _provenance(is_signal_turn, propagated),
            )
        )
    return bucketed, affected


def _provenance(is_signal_turn: bool, propagated: bool) -> str:
    """Signal provenance for a bucketed turn."""
    if is_signal_turn:
        return "actual"
    return "derived" if propagated else "none"


def _bucket_conversation(
    conv: dict[str, Any],
    results_map: dict[ResultKey, dict[str, Any]],
    overrides: list[OverrideSpec],
    acc: _Accumulator,
    scope: ScopeOptions,
) -> None:
    """Classify every turn of one conversation into buckets.

    With ``scope.conversation_scope``, a conversation's active signals propagate to
    all its turns (so e.g. a terminal ``resolved`` labels the whole arc). With
    ``scope.drop_signal_turn``, the turns that originally hold a signal are excluded
    from the output instead of being classified on their own signal.
    """
    conv_id = conv["conversation_group_id"]
    info = _gather_conv(conv, conv_id, results_map)
    bucketed, affected = _classify_conv_turns(info, conv_id, overrides, acc, scope)
    acc.record_propagation(
        conv_id, len(info.turn_infos), len(info.signal_turn_ids), affected
    )
    acc.add_conversation(conv, bucketed)


def _build_report(
    acc: _Accumulator,
    results_path: str,
    eval_data_path: Optional[str],
    scope: ScopeOptions,
) -> dict[str, Any]:
    """Assemble the curate report dict from the accumulator."""
    histogram: dict[float, int] = defaultdict(int)
    for dist in acc.bucket_confidence.values():
        for conf, count in dist.items():
            histogram[conf] += count
    return {
        "results_path": results_path,
        "eval_data_path": eval_data_path,
        "bucket_turn_counts": {b: acc.bucket_stats[b]["total"] for b in BUCKETS},
        "bucket_provenance": {
            b: {
                "actual": acc.bucket_stats[b]["actual"],
                "derived": acc.bucket_stats[b]["derived"],
                "ignored": acc.bucket_stats[b]["ignored"],
            }
            for b in BUCKETS
        },
        "confidence_histogram": dict(sorted(histogram.items())),
        "bucket_confidence": {
            b: dict(sorted(acc.bucket_confidence[b].items())) for b in BUCKETS
        },
        "skipped_turns": acc.skipped_turns,
        "signal_propagation": {
            "scope": "conversation" if scope.conversation_scope else "turn",
            "drop_signal_turn": scope.drop_signal_turn,
            "turns_with_signal": sum(p["signal_turns"] for p in acc.propagation),
            "turns_affected": sum(p["affected"] for p in acc.propagation),
            "excluded_turns": acc.excluded_turns,
            "per_conversation": acc.propagation,
        },
        "needs_review": acc.needs_review,
    }


def curate(
    results_path: str,
    eval_data_path: Optional[str] = None,
    output_dir: str = "output/curate",
    curate_config_path: Optional[str] = None,
    scope: Optional[ScopeOptions] = None,
) -> dict[str, Any]:
    """Produce curated dataset splits from an evaluation run.

    Args:
        results_path: Path to the evaluation results summary JSON (carries the
            per-turn eval result and the interpreted signals).
        eval_data_path: Optional evaluation-data YAML; used for richer turn
            content and as a signals fallback. When omitted, turns are
            reconstructed from the result rows.
        output_dir: Directory to write the split YAMLs and report into.
        curate_config_path: Optional curate.yaml holding overrides.
        scope: Signal-scope options (conversation propagation + drop-signal-turn).

    Returns:
        The report dict (also written to disk as JSON and Markdown).
    """
    scope = scope or ScopeOptions()
    results_map = _load_results(results_path)
    overrides = (
        load_curate_config(curate_config_path).overrides if curate_config_path else []
    )
    conversations = (
        _load_conversations(eval_data_path)
        if eval_data_path
        else _conversations_from_results(results_map)
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    acc = _Accumulator()
    for conv in conversations:
        _bucket_conversation(conv, results_map, overrides, acc, scope)

    for bucket in BUCKETS:
        _write_split(acc.split_convs[bucket], out, bucket)

    report = _build_report(acc, results_path, eval_data_path, scope)

    with open(out / "curate_report.md", "w", encoding="utf-8") as f:
        f.write(_render_report(report))
    logger.info("Curate report written to %s", out / "curate_report.md")

    return report


def _label_to_dict(label: CurateLabel) -> dict[str, Any]:
    """Serialise a CurateLabel for embedding in a turn's signals."""
    return {
        "bucket": label.bucket,
        "confidence": label.confidence,
        "needs_review": label.needs_review,
        "reason": label.reason,
        "note": label.note,
    }

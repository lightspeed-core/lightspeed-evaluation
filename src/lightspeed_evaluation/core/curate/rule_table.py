"""Curate reconciliation algorithm.

Reconciles the evaluation result (``eval``: pass/fail) with per-turn signals into
a bucket, a confidence, and a review flag. Signals arrive already interpreted, each
carrying ``polarity`` / ``strength`` / ``type`` from ingest time.

Rules:
  * ``eval`` is the claim: ``pass`` = good (positive), ``fail`` = bad (negative).
  * A signal *agrees* if its polarity matches the eval direction, *disagrees* if
    opposite; neutral / zero-weight signals are ignored (absent).
  * Bucket: any disagreement -> ``edge``; no agreeing signal -> ``unconfirmed``;
    otherwise ``positive`` (eval pass) / ``negative`` (eval fail).
  * Confidence = sum of agreeing signals' weights (strength).
  * Every ``edge`` needs human review.
  * Optional overrides (from curate.yaml) win over the algorithm.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from lightspeed_evaluation.core.ingestion.config import Bucket, OverrideSpec

# Evaluation result for a turn (aggregated across its metrics).
Eval = Literal["pass", "fail"]


@dataclass(frozen=True)
class CurateLabel:
    """Result of reconciling one turn."""

    bucket: Bucket
    confidence: float  # agreeing weight sum; for edge, the conflict magnitude
    needs_review: bool  # True for every edge (or as pinned by an override)
    reason: str  # auto-generated, intuitive explanation of why this bucket
    note: str  # free/override annotation (empty unless pinned by an override)


def _vote(signal: Any) -> tuple[str, float]:
    """Return (polarity, weight) for an enriched signal; neutral if uninterpreted."""
    if isinstance(signal, dict):
        polarity = signal.get("polarity", "neutral")
        try:
            weight = float(signal.get("strength", 0.0) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        return polarity, weight
    # Raw scalar signal (no interpretation available) -> ignored.
    return "neutral", 0.0


def _signal_raw_value(signal: Any) -> Any:
    """Return the raw value of a signal (enriched dict or scalar)."""
    if isinstance(signal, dict):
        return signal.get("value")
    return signal


def _name_val(name: str, signal: Any) -> str:
    """Compact ``name=value`` for a signal, e.g. ``exit_reason=escalated``."""
    return f"{name}={_signal_raw_value(signal)}"


@dataclass
class _Tally:
    """Categorised signal votes for one turn."""

    agreeing: list[str]
    disagreeing: list[str]
    neutral: list[str]
    agreeing_weight: float
    disagreeing_weight: float


def _tally(eval_result: Eval, signals: dict[str, Any]) -> _Tally:
    """Split signals into agreeing / disagreeing / neutral with summed weights."""
    eval_polarity = "positive" if eval_result == "pass" else "negative"
    t = _Tally([], [], [], 0.0, 0.0)
    for name, signal in signals.items():
        polarity, weight = _vote(signal)
        if polarity == "neutral" or weight <= 0.0:
            t.neutral.append(_name_val(name, signal))
        elif polarity == eval_polarity:
            t.agreeing.append(_name_val(name, signal))
            t.agreeing_weight += weight
        else:
            t.disagreeing.append(_name_val(name, signal))
            t.disagreeing_weight += weight
    return t


def _reconcile(eval_result: Eval, signals: dict[str, Any]) -> tuple[Bucket, float, str]:
    """Combine eval + signal votes into a bucket, confidence, and an intuitive reason."""
    verdict = "good" if eval_result == "pass" else "bad"
    opposite = "bad" if eval_result == "pass" else "good"
    t = _tally(eval_result, signals)
    agreeing, disagreeing, neutral = t.agreeing, t.disagreeing, t.neutral
    agreeing_weight, disagreeing_weight = t.agreeing_weight, t.disagreeing_weight

    if disagreeing:
        # Edge confidence = magnitude of disagreement (sum of disagreeing
        # strengths), so the strongest conflicts rank first for human review.
        verb = "indicates" if len(disagreeing) == 1 else "indicate"
        reason = (
            f"Eval rated this turn {verdict}, but {', '.join(disagreeing)} "
            f"{verb} it was {opposite}"
        )
        if agreeing:
            reason += f" (while {', '.join(agreeing)} agree with eval)"
        return "edge", disagreeing_weight, reason
    if agreeing_weight <= 0.0:
        seen = (
            f"; only neutral signals: {', '.join(neutral)}"
            if neutral
            else "; no signals present"
        )
        return (
            "unconfirmed",
            0.0,
            f"Eval rated this turn {verdict}, no signal confirms it{seen}",
        )
    bucket: Bucket = "positive" if eval_result == "pass" else "negative"
    reason = f"Eval rated this turn {verdict}, confirmed by {', '.join(agreeing)}"
    return bucket, agreeing_weight, reason


def _override_matches(
    override: OverrideSpec, eval_result: Eval, signals: dict[str, Any]
) -> bool:
    """True if every key in the override's match applies to this turn."""
    for key, expected in override.match.items():
        if key == "eval":
            if eval_result != expected:
                return False
            continue
        if str(_signal_raw_value(signals.get(key))) != str(expected):
            return False
    return True


def _apply_override(label: CurateLabel, override: OverrideSpec) -> CurateLabel:
    """Return a label with the override's fields applied over the computed one."""
    bucket = override.bucket or label.bucket
    review = override.review if override.review is not None else (bucket == "edge")
    confidence = (
        override.confidence if override.confidence is not None else label.confidence
    )
    return CurateLabel(
        bucket=bucket,
        confidence=confidence,
        needs_review=review,
        reason=label.reason,
        note=override.note or label.note,
    )


def classify(
    eval_result: Eval,
    signals: dict[str, Any],
    overrides: Optional[list[OverrideSpec]] = None,
) -> CurateLabel:
    """Reconcile a turn's eval result and interpreted signals into a label."""
    bucket, confidence, reason = _reconcile(eval_result, signals)
    label = CurateLabel(
        bucket=bucket,
        confidence=confidence,
        needs_review=(bucket == "edge"),
        reason=reason,
        note="",
    )
    for override in overrides or []:
        if _override_matches(override, eval_result, signals):
            return _apply_override(label, override)
    return label


def eval_from_results(results: list[str]) -> Optional[Eval]:
    """Aggregate a turn's per-metric result statuses into one eval result.

    Returns ``fail`` if any metric failed, ``pass`` if at least one passed and
    none failed, and ``None`` when there is no usable result (only ERROR or
    SKIPPED), in which case the turn cannot be bucketed.
    """
    statuses = {s.upper() for s in results}
    if "FAIL" in statuses:
        return "fail"
    if "PASS" in statuses:
        return "pass"
    return None

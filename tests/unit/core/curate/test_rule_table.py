"""Tests for the curate reconciliation algorithm."""

from typing import Any

import pytest

from lightspeed_evaluation.core.curate.rule_table import classify, eval_from_results
from lightspeed_evaluation.core.ingestion.config import OverrideSpec


def _sig(
    polarity: str, strength: float, value: str = "v", type_: str = "t"
) -> dict[str, Any]:
    """Build an enriched signal (as written at ingest time)."""
    return {"value": value, "polarity": polarity, "strength": strength, "type": type_}


def test_pass_with_agreeing_signal_is_positive() -> None:
    """Eval pass + a positive signal -> positive, confidence = weight."""
    label = classify("pass", {"uf": _sig("positive", 1.0)})
    assert label.bucket == "positive"
    assert label.confidence == 1.0
    assert label.needs_review is False


def test_fail_with_agreeing_signal_is_negative() -> None:
    """Eval fail + a negative signal -> negative."""
    label = classify("fail", {"er": _sig("negative", 0.5)})
    assert label.bucket == "negative"
    assert label.confidence == 0.5


def test_disagreeing_signal_is_edge_needing_review() -> None:
    """A signal contradicting the eval result -> edge, review."""
    label = classify("pass", {"uf": _sig("negative", 1.0)})
    assert label.bucket == "edge"
    assert label.needs_review is True


def test_no_signals_is_unconfirmed() -> None:
    """Eval result with no interpretable signal -> unconfirmed (pass or fail)."""
    assert classify("pass", {}).bucket == "unconfirmed"
    assert classify("fail", {}).bucket == "unconfirmed"


def test_neutral_and_zero_weight_signals_are_ignored() -> None:
    """Neutral polarity or zero strength does not count as a vote."""
    signals = {"a": _sig("neutral", 1.0), "b": _sig("positive", 0.0)}
    assert classify("pass", signals).bucket == "unconfirmed"


def test_confidence_sums_agreeing_weights() -> None:
    """Confidence is the sum of agreeing signal weights."""
    signals = {"uf": _sig("positive", 1.0), "er": _sig("positive", 0.5)}
    assert classify("pass", signals).confidence == 1.5


def test_strength_distinguishes_signal_values() -> None:
    """Strong (1.0) vs weak (0.5) yields different confidence for the same bucket."""
    strong = classify("fail", {"er": _sig("negative", 1.0, value="escalated")})
    weak = classify("fail", {"er": _sig("negative", 0.5, value="aborted_or_done")})
    assert strong.bucket == weak.bucket == "negative"
    assert strong.confidence == 1.0
    assert weak.confidence == 0.5


def test_raw_scalar_signal_is_ignored() -> None:
    """A non-enriched (scalar) signal cannot be interpreted -> ignored."""
    assert classify("pass", {"er": "resolved"}).bucket == "unconfirmed"


def test_override_matches_and_wins() -> None:
    """A matching override replaces the computed label."""
    overrides = [
        OverrideSpec(
            match={"eval": "fail", "er": "resolved"},
            bucket="edge",
            review=True,
            note="resolved may be premature",
        )
    ]
    signals = {"er": _sig("positive", 0.5, value="resolved")}
    label = classify("fail", signals, overrides)
    assert label.bucket == "edge"
    assert label.needs_review is True
    assert label.note == "resolved may be premature"


def test_override_only_applies_on_full_match() -> None:
    """An override whose match does not fully apply is ignored."""
    overrides = [OverrideSpec(match={"eval": "pass", "er": "resolved"}, bucket="edge")]
    signals = {"er": _sig("positive", 0.5, value="resolved")}
    # eval is fail, override requires pass -> no match -> algorithm result.
    label = classify("fail", signals, overrides)
    assert label.bucket == "edge"  # fail + positive = disagreement = edge anyway
    # confirm via a non-conflicting case:
    label2 = classify("pass", {"er": _sig("positive", 0.5, value="other")}, overrides)
    assert label2.bucket == "positive"  # value 'other' != 'resolved' -> no override


def test_override_can_set_confidence() -> None:
    """An override may pin confidence."""
    overrides = [OverrideSpec(match={"eval": "pass"}, confidence=0.9)]
    label = classify("pass", {"uf": _sig("positive", 1.0)}, overrides)
    assert label.confidence == 0.9


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["PASS"], "pass"),
        (["PASS", "FAIL"], "fail"),
        (["FAIL"], "fail"),
        (["ERROR", "SKIPPED"], None),
        ([], None),
    ],
)
def test_eval_from_results(statuses: list[str], expected: object) -> None:
    """Per-metric statuses aggregate to fail/pass/None."""
    assert eval_from_results(statuses) == expected

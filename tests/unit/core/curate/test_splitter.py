"""Tests for the curate splitter."""

import json
from pathlib import Path

import yaml

from lightspeed_evaluation.core.curate.splitter import ScopeOptions, curate


def _write_eval_data(path: Path, conversations: list, as_dict: bool = False) -> None:
    payload = (
        {"metadata": {"description": "x"}, "conversations": conversations}
        if as_dict
        else conversations
    )
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False)


def _write_results(path: Path, results: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f)


def _es(polarity: str, strength: float, value: str = "v") -> dict:
    """An enriched signal as written at ingest time."""
    return {"value": value, "polarity": polarity, "strength": strength, "type": "t"}


def _base_conversations() -> list:
    return [
        {
            "conversation_group_id": "c1",
            "turns": [
                {
                    "turn_id": "t1",  # pass + positive signals -> positive
                    "query": "q1",
                    "signals": {
                        "user_feedback": _es("positive", 1.0, "positive"),
                        "exit_reason": _es("positive", 0.5, "resolved"),
                    },
                },
                {
                    "turn_id": "t2",  # fail + positive -> edge (review)
                    "query": "q2",
                    "signals": {"user_feedback": _es("positive", 1.0, "positive")},
                },
                {
                    "turn_id": "t3",  # pass + no signals -> unconfirmed
                    "query": "q3",
                },
            ],
        }
    ]


def _results() -> list:
    return [
        {"conversation_group_id": "c1", "turn_id": "t1", "result": "PASS"},
        {"conversation_group_id": "c1", "turn_id": "t2", "result": "FAIL"},
        {"conversation_group_id": "c1", "turn_id": "t3", "result": "PASS"},
    ]


def _results_with_signals() -> list:
    """Results rows carrying per-turn signals (signal-carrying eval run)."""
    return [
        {
            "conversation_group_id": "c1",
            "turn_id": "t1",
            "result": "PASS",
            "query": "q1",
            "response": "r1",
            "signals": {
                "user_feedback": _es("positive", 1.0, "positive"),
                "exit_reason": _es("positive", 0.5, "resolved"),
            },
        },
        {
            "conversation_group_id": "c1",
            "turn_id": "t2",
            "result": "FAIL",
            "query": "q2",
            "response": "r2",
            "signals": {
                "user_feedback": _es("negative", 1.0, "negative"),
                "exit_reason": _es("negative", 1.0, "escalated"),
            },
        },
    ]


def test_curate_single_file_from_results_signals(tmp_path: Path) -> None:
    """With signals on the results, curate needs no eval-data file."""
    results = tmp_path / "results.json"
    out = tmp_path / "curate"
    _write_results(results, _results_with_signals())

    report = curate(str(results), None, str(out))

    assert report["bucket_turn_counts"]["positive"] == 1  # t1 pass+pos+resolved
    assert report["bucket_turn_counts"]["negative"] == 1  # t2 fail+neg+escalated
    positive = yaml.safe_load((out / "positive.yaml").read_text())
    turn = positive[0]["turns"][0]
    assert turn["turn_id"] == "t1"
    assert turn["query"] == "q1"  # reconstructed from the result row
    assert turn["signals"]["curate"]["bucket"] == "positive"


def test_results_signals_take_precedence_over_eval_data(tmp_path: Path) -> None:
    """Signals on the result override those in the eval-data YAML."""
    results = tmp_path / "results.json"
    eval_data = tmp_path / "eval_data.yaml"
    out = tmp_path / "curate"
    # eval-data says t1 is negative-ish; the result says resolved+positive.
    _write_eval_data(
        eval_data,
        [
            {
                "conversation_group_id": "c1",
                "turns": [
                    {
                        "turn_id": "t1",
                        "query": "q1",
                        "signals": {"exit_reason": _es("negative", 1.0, "escalated")},
                    }
                ],
            }
        ],
    )
    _write_results(
        results,
        [
            {
                "conversation_group_id": "c1",
                "turn_id": "t1",
                "result": "PASS",
                "signals": {
                    "user_feedback": _es("positive", 1.0, "positive"),
                    "exit_reason": _es("positive", 0.5, "resolved"),
                },
            }
        ],
    )

    report = curate(str(results), str(eval_data), str(out))
    assert report["bucket_turn_counts"]["positive"] == 1
    assert report["bucket_turn_counts"]["edge"] == 0


def test_curate_buckets_turns_and_writes_splits(tmp_path: Path) -> None:
    """Turns land in the right buckets and splits are written to disk."""
    eval_data = tmp_path / "eval_data.yaml"
    results = tmp_path / "results.json"
    out = tmp_path / "curate"
    _write_eval_data(eval_data, _base_conversations())
    _write_results(results, _results())

    report = curate(str(results), str(eval_data), str(out))

    assert report["bucket_turn_counts"]["positive"] == 1
    assert report["bucket_turn_counts"]["edge"] == 1
    assert report["bucket_turn_counts"]["unconfirmed"] == 1
    assert report["bucket_turn_counts"]["negative"] == 0

    positive = yaml.safe_load((out / "positive.yaml").read_text())
    assert positive[0]["turns"][0]["turn_id"] == "t1"
    # Curate label is embedded under the turn's signals.
    assert positive[0]["turns"][0]["signals"]["curate"]["bucket"] == "positive"


def test_curate_review_list_flags_edges(tmp_path: Path) -> None:
    """Edge turns needing review appear in the report and files are written."""
    eval_data = tmp_path / "eval_data.yaml"
    results = tmp_path / "results.json"
    out = tmp_path / "curate"
    _write_eval_data(eval_data, _base_conversations())
    _write_results(results, _results())

    report = curate(str(results), str(eval_data), str(out))

    review_ids = {r["turn_id"] for r in report["needs_review"]}
    assert "t2" in review_ids  # fail + positive feedback -> review=yes
    assert (out / "curate_report.md").exists()


def test_curate_skips_turns_without_verdict(tmp_path: Path) -> None:
    """Turns with only ERROR/SKIPPED or no result are skipped, not bucketed."""
    eval_data = tmp_path / "eval_data.yaml"
    results = tmp_path / "results.json"
    out = tmp_path / "curate"
    _write_eval_data(eval_data, _base_conversations())
    # Only ERROR for t1; t2/t3 absent entirely.
    _write_results(
        results,
        [{"conversation_group_id": "c1", "turn_id": "t1", "result": "ERROR"}],
    )

    report = curate(str(results), str(eval_data), str(out))
    assert report["skipped_turns"] == 3
    assert sum(report["bucket_turn_counts"].values()) == 0


def test_curate_accepts_dict_format_eval_data(tmp_path: Path) -> None:
    """Eval data in {metadata, conversations} dict format is supported."""
    eval_data = tmp_path / "eval_data.yaml"
    results = tmp_path / "results.json"
    out = tmp_path / "curate"
    _write_eval_data(eval_data, _base_conversations(), as_dict=True)
    _write_results(results, _results())

    report = curate(str(results), str(eval_data), str(out))
    assert report["bucket_turn_counts"]["positive"] == 1


def test_curate_handles_multiple_metrics_per_turn(tmp_path: Path) -> None:
    """A turn failing any metric is a fail verdict."""
    eval_data = tmp_path / "eval_data.yaml"
    results = tmp_path / "results.json"
    out = tmp_path / "curate"
    convs = [
        {
            "conversation_group_id": "c1",
            "turns": [
                {
                    "turn_id": "t1",
                    "query": "q",
                    "signals": {"user_feedback": _es("negative", 1.0, "negative")},
                }
            ],
        }
    ]
    _write_eval_data(eval_data, convs)
    _write_results(
        results,
        [
            {"conversation_group_id": "c1", "turn_id": "t1", "result": "PASS"},
            {"conversation_group_id": "c1", "turn_id": "t1", "result": "FAIL"},
        ],
    )

    report = curate(str(results), str(eval_data), str(out))
    # fail + negative + absent -> negative
    assert report["bucket_turn_counts"]["negative"] == 1


def _arc_conversation() -> list:
    """One 3-turn conversation; only the terminal turn carries a positive signal."""
    return [
        {
            "conversation_group_id": "c1",
            "turns": [
                {"turn_id": "t1", "query": "q1"},
                {"turn_id": "t2", "query": "q2"},
                {
                    "turn_id": "t3",
                    "query": "q3",
                    "signals": {"exit_reason": _es("positive", 0.5, "resolved")},
                },
            ],
        }
    ]


def _arc_results() -> list:
    return [
        {"conversation_group_id": "c1", "turn_id": t, "result": "PASS"}
        for t in ("t1", "t2", "t3")
    ]


def test_turn_scope_labels_only_signal_turn(tmp_path: Path) -> None:
    """Default (turn scope): only the signal-bearing turn is positive."""
    eval_data, results, out = (tmp_path / n for n in ("e.yaml", "r.json", "o"))
    _write_eval_data(eval_data, _arc_conversation())
    _write_results(results, _arc_results())

    report = curate(str(results), str(eval_data), str(out))
    assert report["bucket_turn_counts"]["positive"] == 1
    assert report["bucket_turn_counts"]["unconfirmed"] == 2


def test_conversation_scope_propagates_to_whole_arc(tmp_path: Path) -> None:
    """Conversation scope: the terminal signal labels every turn in the arc."""
    eval_data, results, out = (tmp_path / n for n in ("e.yaml", "r.json", "o"))
    _write_eval_data(eval_data, _arc_conversation())
    _write_results(results, _arc_results())

    report = curate(
        str(results),
        str(eval_data),
        str(out),
        scope=ScopeOptions(conversation_scope=True),
    )
    assert report["bucket_turn_counts"]["positive"] == 3
    assert report["bucket_turn_counts"]["unconfirmed"] == 0
    prop = report["signal_propagation"]
    assert prop["scope"] == "conversation"
    assert prop["turns_with_signal"] == 1
    assert prop["turns_affected"] == 2


def test_drop_signal_turn_excludes_the_holder(tmp_path: Path) -> None:
    """Conversation scope + drop: the signal-bearing turn is excluded from output."""
    eval_data, results, out = (tmp_path / n for n in ("e.yaml", "r.json", "o"))
    _write_eval_data(eval_data, _arc_conversation())
    _write_results(results, _arc_results())

    report = curate(
        str(results),
        str(eval_data),
        str(out),
        scope=ScopeOptions(conversation_scope=True, drop_signal_turn=True),
    )
    assert report["bucket_turn_counts"]["positive"] == 2  # t1, t2
    assert report["signal_propagation"]["excluded_turns"] == 1  # t3 dropped
    positive = yaml.safe_load((out / "positive.yaml").read_text())
    ids = {t["turn_id"] for c in positive for t in c["turns"]}
    assert ids == {"t1", "t2"}
    # The dropped signal turn is still counted, so provenance isn't silently 0.
    assert report["bucket_provenance"]["positive"]["ignored"] == 1


def test_conversation_scope_segments_multiple_signals(tmp_path: Path) -> None:
    """Multiple signals split the convo into arcs; each labels the turns up to it."""
    eval_data, results, out = (tmp_path / n for n in ("e.yaml", "r.json", "o"))
    convs = [
        {
            "conversation_group_id": "c1",
            "turns": [
                {"turn_id": "t1", "query": "q1"},
                {
                    "turn_id": "t2",
                    "query": "q2",
                    "signals": {"exit_reason": _es("negative", 1.0, "escalated")},
                },
                {"turn_id": "t3", "query": "q3"},
                {"turn_id": "t4", "query": "q4"},
                {
                    "turn_id": "t5",
                    "query": "q5",
                    "signals": {"exit_reason": _es("positive", 0.5, "resolved")},
                },
            ],
        }
    ]
    _write_eval_data(eval_data, convs)
    _write_results(
        results,
        [
            {"conversation_group_id": "c1", "turn_id": t, "result": "PASS"}
            for t in ("t1", "t2", "t3", "t4", "t5")
        ],
    )

    report = curate(
        str(results),
        str(eval_data),
        str(out),
        scope=ScopeOptions(conversation_scope=True),
    )
    # Arc A (t1,t2) -> escalated(neg) vs eval pass = disagreement -> edge.
    # Arc B (t3,t4,t5) -> resolved(pos) + eval pass = positive.
    assert report["bucket_turn_counts"]["positive"] == 3
    assert report["bucket_turn_counts"]["edge"] == 2
    positive = yaml.safe_load((out / "positive.yaml").read_text())
    ids = {t["turn_id"] for c in positive for t in c["turns"]}
    assert ids == {"t3", "t4", "t5"}


def test_conversation_level_metrics_are_not_turns(tmp_path: Path) -> None:
    """Results rows with turn_id=None (conversation-level) never become turns."""
    results, out = tmp_path / "r.json", tmp_path / "o"
    _write_results(
        results,
        [
            {"conversation_group_id": "c1", "turn_id": "t1", "result": "PASS"},
            {"conversation_group_id": "c1", "turn_id": None, "result": "FAIL"},
        ],
    )
    report = curate(str(results), None, str(out))
    total = sum(report["bucket_turn_counts"].values())
    assert total == 1  # only the real turn t1, not the conversation-level aggregate


def test_effective_signals_are_persisted_on_emitted_turn(tmp_path: Path) -> None:
    """Result-carried signals (not YAML signals) are written onto the output turn."""
    eval_data, results, out = (tmp_path / n for n in ("e.yaml", "r.json", "o"))
    # YAML says positive; the result overrides with a negative signal + FAIL eval.
    _write_eval_data(
        eval_data,
        [
            {
                "conversation_group_id": "c1",
                "turns": [
                    {
                        "turn_id": "t1",
                        "query": "q1",
                        "signals": {"exit_reason": _es("positive", 1.0, "resolved")},
                    }
                ],
            }
        ],
    )
    _write_results(
        results,
        [
            {
                "conversation_group_id": "c1",
                "turn_id": "t1",
                "result": "FAIL",
                "signals": {"exit_reason": _es("negative", 1.0, "escalated")},
            }
        ],
    )
    curate(str(results), str(eval_data), str(out))
    negative = yaml.safe_load((out / "negative.yaml").read_text())
    turn = negative[0]["turns"][0]
    # The emitted evidence matches the label's basis (the result signal), not YAML.
    assert turn["signals"]["exit_reason"]["polarity"] == "negative"
    assert turn["signals"]["curate"]["bucket"] == "negative"


def test_input_only_eval_data_is_filled_from_results(tmp_path: Path) -> None:
    """A turn without a response in eval-data gets the evaluated one from results."""
    results, eval_data, out = (tmp_path / n for n in ("r.json", "e.yaml", "o"))
    _write_eval_data(
        eval_data,
        [{"conversation_group_id": "c1", "turns": [{"turn_id": "t1", "query": "q1"}]}],
    )
    _write_results(
        results,
        [
            {
                "conversation_group_id": "c1",
                "turn_id": "t1",
                "result": "PASS",
                "query": "ignored",
                "response": "r1",
                "signals": {"exit_reason": _es("positive", 1.0, "resolved")},
            }
        ],
    )
    curate(str(results), str(eval_data), str(out))
    turn = yaml.safe_load((out / "positive.yaml").read_text())[0]["turns"][0]
    assert (turn["query"], turn["response"]) == ("q1", "r1")

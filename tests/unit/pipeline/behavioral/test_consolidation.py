"""Tests for NxM behavioral consolidation."""

from typing import Any

from lightspeed_evaluation.pipeline.behavioral.consolidation import consolidate
from lightspeed_evaluation.pipeline.behavioral.loader import RunData

_DEFAULT_TOKENS = {
    "total_api_input_tokens": 100,
    "total_api_output_tokens": 50,
    "total_judge_llm_input_tokens": 200,
    "total_judge_llm_output_tokens": 80,
    "total_embedding_tokens": 10,
}


def _make_run_data(run_index: int, **kwargs: Any) -> RunData:
    """Build a RunData with realistic summary.json structure.

    Keyword args: total, passed, pass_rate, metrics, conversations,
    latency_mean, quality.
    """
    total = kwargs.get("total", 10)
    passed = kwargs.get("passed", 8)
    pass_rate = kwargs.get("pass_rate", 80.0)

    overall = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "error": 0,
        "skipped": 0,
        "pass_rate": pass_rate,
        **_DEFAULT_TOKENS,
    }

    stats: dict[str, Any] = {
        "overall": overall,
        "by_metric": kwargs.get("metrics")
        or {
            "ragas:faithfulness": {
                "score_statistics": {"mean": 0.9, "std": 0.05},
            },
        },
        "by_conversation": kwargs.get("conversations")
        or {
            "conv_1": {"pass_rate": pass_rate},
        },
    }

    latency_stats = kwargs.get("latency_stats")
    latency_mean = kwargs.get("latency_mean")
    if latency_stats is not None:
        stats["agent_latency_stats"] = latency_stats
    elif latency_mean is not None:
        stats["agent_latency_stats"] = {"mean": latency_mean}

    return RunData(
        run_index=run_index,
        summary={"summary_stats": stats},
        quality=kwargs.get("quality"),
        case_results=kwargs.get("case_results"),
    )


class TestConsolidate:
    """Tests for consolidate()."""

    def test_single_run(self) -> None:
        """Single run produces correct aggregation."""
        run = _make_run_data(1, total=10, passed=8, pass_rate=80.0)
        result = consolidate("model_a", [run], runs_requested=1)

        assert result.agent_name == "model_a"
        assert result.runs_requested == 1
        assert result.runs_succeeded == 1
        assert result.overall["pass_rate_mean"] == 80.0
        assert result.overall["pass_rate_std"] == 0.0
        assert len(result.per_run) == 1

    def test_multiple_runs_aggregation(self) -> None:
        """Two runs produce mean/std across runs."""
        runs = [
            _make_run_data(1, pass_rate=80.0),
            _make_run_data(2, pass_rate=90.0),
        ]
        result = consolidate("model_a", runs, runs_requested=2)

        assert result.runs_succeeded == 2
        assert result.overall["pass_rate_mean"] == 85.0
        assert result.overall["pass_rate_std"] is not None
        assert result.overall["pass_rate_std"] > 0

    def test_discovers_metrics(self) -> None:
        """Metrics are discovered from data, not hardcoded."""
        metrics = {
            "ragas:faithfulness": {"score_statistics": {"mean": 0.9}},
            "custom:accuracy": {"score_statistics": {"mean": 0.75}},
        }
        run = _make_run_data(1, metrics=metrics)
        result = consolidate("model_a", [run], runs_requested=1)

        assert "ragas:faithfulness" in result.by_metric
        assert "custom:accuracy" in result.by_metric
        assert result.by_metric["ragas:faithfulness"]["mean"] == 0.9
        assert result.by_metric["custom:accuracy"]["mean"] == 0.75

    def test_per_conversation_pass_rates(self) -> None:
        """Per-conversation pass rates are aggregated across runs."""
        convs = {"conv_1": {"pass_rate": 80.0}, "conv_2": {"pass_rate": 60.0}}
        runs = [
            _make_run_data(1, conversations=convs),
            _make_run_data(2, conversations=convs),
        ]
        result = consolidate("model_a", runs, runs_requested=2)

        assert "conv_1" in result.by_conversation
        assert "conv_2" in result.by_conversation
        assert result.by_conversation["conv_1"]["pass_rate_mean"] == 80.0

    def test_latency_aggregation(self) -> None:
        """Agent latency is averaged across runs."""
        runs = [
            _make_run_data(1, latency_mean=1.5),
            _make_run_data(2, latency_mean=2.5),
        ]
        result = consolidate("model_a", runs, runs_requested=2)

        assert result.overall["agent_latency_mean"] == 2.0

    def test_token_totals(self) -> None:
        """Token totals are summed across runs, split by input/output."""
        runs = [
            _make_run_data(1),
            _make_run_data(2),
        ]
        result = consolidate("model_a", runs, runs_requested=2)

        assert result.overall["total_agent_input_tokens"] == 200.0
        assert result.overall["total_agent_output_tokens"] == 100.0
        assert result.eval_costs is not None
        assert dict(result.eval_costs)["total_judge_input_tokens"] == 400.0
        assert dict(result.eval_costs)["total_judge_output_tokens"] == 160.0
        assert dict(result.eval_costs)["total_embedding_tokens"] == 20.0

    def test_quality_aggregation(self) -> None:
        """Quality scores are aggregated across runs."""
        quality1 = {
            "quality_score": 0.80,
            "quality_metrics": {
                "ragas:faithfulness": {"mean": 0.9, "weight": 0.5},
            },
        }
        quality2 = {
            "quality_score": 0.90,
            "quality_metrics": {
                "ragas:faithfulness": {"mean": 0.85, "weight": 0.5},
            },
        }
        runs = [
            _make_run_data(1, quality=quality1),
            _make_run_data(2, quality=quality2),
        ]
        result = consolidate("model_a", runs, runs_requested=2)

        assert result.quality_score is not None
        assert abs(dict(result.quality_score)["mean"] - 0.85) < 1e-9
        assert "std" in dict(result.quality_score)

    def test_empty_runs(self) -> None:
        """Empty runs list returns zero-valued result."""
        result = consolidate("model_a", [], runs_requested=2)

        assert result.runs_succeeded == 0
        assert result.conversations_count == 0
        assert not result.overall
        assert not result.per_run

    def test_per_run_snapshots(self) -> None:
        """Per-run snapshots capture individual run data."""
        runs = [
            _make_run_data(1, total=10, passed=8, pass_rate=80.0),
            _make_run_data(2, total=10, passed=9, pass_rate=90.0),
        ]
        result = consolidate("model_a", runs, runs_requested=2)

        assert len(result.per_run) == 2
        assert result.per_run[0].run_index == 1
        assert result.per_run[0].pass_rate == 80.0
        assert result.per_run[1].run_index == 2
        assert result.per_run[1].pass_rate == 90.0

    def test_no_quality_when_absent(self) -> None:
        """Quality is None when no runs have quality data."""
        run = _make_run_data(1)
        result = consolidate("model_a", [run], runs_requested=1)

        assert result.quality_score is None

    def test_pass_at_k_wiring(self) -> None:
        """pass@k flows into overall, by_metric, and by_conversation."""

        def _case(conv: str, result: str) -> dict[str, str]:
            return {
                "conversation_group_id": conv,
                "turn_id": "turn_1",
                "metric_identifier": "ragas:response_relevancy",
                "result": result,
            }

        convs = {"conv_1": {"pass_rate": 80.0}, "conv_2": {"pass_rate": 60.0}}
        metrics = {
            "ragas:response_relevancy": {"score_statistics": {"mean": 0.8}},
        }
        runs = [
            _make_run_data(
                1,
                metrics=metrics,
                conversations=convs,
                case_results=[_case("conv_1", "PASS"), _case("conv_2", "PASS")],
            ),
            _make_run_data(
                2,
                metrics=metrics,
                conversations=convs,
                case_results=[_case("conv_1", "PASS"), _case("conv_2", "FAIL")],
            ),
            _make_run_data(
                3,
                metrics=metrics,
                conversations=convs,
                case_results=[_case("conv_1", "FAIL"), _case("conv_2", "FAIL")],
            ),
        ]
        result = consolidate("model_a", runs, runs_requested=3)

        assert result.overall.get("pass_at_k") is not None
        assert "pass_at_k" in result.by_metric.get("ragas:response_relevancy", {})
        c1 = result.by_conversation.get("conv_1", {})
        c2 = result.by_conversation.get("conv_2", {})
        assert "pass_at_k" in c1
        assert "pass_at_k" in c2
        c1_pak = c1["pass_at_k"]
        c2_pak = c2["pass_at_k"]
        assert isinstance(c1_pak, float)
        assert isinstance(c2_pak, float)
        assert c1_pak >= c2_pak

    def test_pass_at_k_excludes_errors(self) -> None:
        """ERROR results are excluded from pass@k denominator."""

        def _case(result: str) -> dict[str, str]:
            return {
                "conversation_group_id": "conv_1",
                "turn_id": "turn_1",
                "metric_identifier": "ragas:response_relevancy",
                "result": result,
            }

        runs = [
            _make_run_data(1, case_results=[_case("PASS")]),
            _make_run_data(2, case_results=[_case("FAIL")]),
            _make_run_data(3, case_results=[_case("ERROR")]),
        ]
        result = consolidate("model_a", runs, runs_requested=3)

        pak = result.overall.get("pass_at_k")
        assert pak is not None
        # Only PASS/FAIL counted: n=2, c=1, k=min(3,2)=2
        # 1 - C(1,2)/C(2,2) = 1 - 0/1 = 1.0
        assert abs(pak - 1.0) < 1e-9

    def test_no_pass_at_k_for_single_run(self) -> None:
        """pass@k not computed for single run."""
        run = _make_run_data(1)
        result = consolidate("model_a", [run], runs_requested=1)

        assert "pass_at_k" not in result.overall

    def test_pass_at_1_and_pass_hat_k_wiring(self) -> None:
        """pass@1 and pass^k flow into overall, by_metric, by_conversation."""

        def _case(conv: str, result: str) -> dict[str, str]:
            return {
                "conversation_group_id": conv,
                "turn_id": "turn_1",
                "metric_identifier": "ragas:response_relevancy",
                "result": result,
            }

        convs = {"conv_1": {"pass_rate": 80.0}}
        metrics = {
            "ragas:response_relevancy": {"score_statistics": {"mean": 0.8}},
        }
        runs = [
            _make_run_data(
                1,
                metrics=metrics,
                conversations=convs,
                case_results=[_case("conv_1", "PASS")],
            ),
            _make_run_data(
                2,
                metrics=metrics,
                conversations=convs,
                case_results=[_case("conv_1", "PASS")],
            ),
            _make_run_data(
                3,
                metrics=metrics,
                conversations=convs,
                case_results=[_case("conv_1", "FAIL")],
            ),
        ]
        result = consolidate("model_a", runs, runs_requested=3)

        pa1 = result.overall.get("pass_at_1")
        assert pa1 is not None
        assert abs(pa1 - 2 / 3) < 1e-9

        phk = result.overall.get("pass_hat_k")
        assert phk is not None
        assert abs(phk - (2 / 3) ** 3) < 1e-9

        assert "pass_at_1" in result.by_metric.get("ragas:response_relevancy", {})
        assert "pass_hat_k" in result.by_metric.get("ragas:response_relevancy", {})
        assert "pass_at_1" in result.by_conversation.get("conv_1", {})
        assert "pass_hat_k" in result.by_conversation.get("conv_1", {})

    def test_no_pass_at_1_for_single_run(self) -> None:
        """pass@1 and pass^k not computed for single run."""

        def _case(result: str) -> dict[str, str]:
            return {
                "conversation_group_id": "conv_1",
                "turn_id": "turn_1",
                "metric_identifier": "m1",
                "result": result,
            }

        run = _make_run_data(1, case_results=[_case("PASS")])
        result = consolidate("model_a", [run], runs_requested=1)

        assert "pass_at_1" not in result.overall
        assert "pass_hat_k" not in result.overall

    def test_latency_percentiles_max_across_runs(self) -> None:
        """Latency p95/p99 are max of per-run values from summary.json."""
        runs = [
            _make_run_data(1, latency_stats={"mean": 1.5, "p95": 3.0, "p99": 5.0}),
            _make_run_data(2, latency_stats={"mean": 2.0, "p95": 4.0, "p99": 6.0}),
            _make_run_data(3, latency_stats={"mean": 1.8, "p95": 3.5, "p99": 4.5}),
        ]
        result = consolidate("model_a", runs, runs_requested=3)

        assert result.overall["agent_latency_p95_max"] == 4.0
        assert result.overall["agent_latency_p99_max"] == 6.0

    def test_token_percentiles_from_turn_stats(self) -> None:
        """Agent token p95/p99 computed from pooled per-turn CSV values."""
        runs = []
        for i in range(3):
            run = _make_run_data(i + 1)
            run.turn_stats = [
                {
                    "agent_input_tokens": 100.0 + i * 10,
                    "agent_output_tokens": 50.0 + i * 5,
                },
                {
                    "agent_input_tokens": 200.0 + i * 10,
                    "agent_output_tokens": 80.0 + i * 5,
                },
            ]
            runs.append(run)
        result = consolidate("model_a", runs, runs_requested=3)

        assert result.overall.get("agent_input_tokens_p95_max") is not None
        assert result.overall.get("agent_input_tokens_p99_max") is not None
        assert result.overall.get("agent_output_tokens_p95_max") is not None
        assert result.overall.get("agent_output_tokens_p99_max") is not None

    def test_eval_costs_section(self) -> None:
        """Judge/embedding costs in separate eval_costs dict."""
        runs = []
        for i in range(2):
            run = _make_run_data(i + 1)
            run.turn_stats = [
                {
                    "judge_input_tokens": 300.0 + i * 50,
                    "judge_output_tokens": 100.0 + i * 20,
                },
            ]
            runs.append(run)
        result = consolidate("model_a", runs, runs_requested=2)

        assert result.eval_costs is not None
        eval_costs = dict(result.eval_costs)
        assert eval_costs["total_judge_input_tokens"] == 400.0
        assert eval_costs["total_judge_output_tokens"] == 160.0
        assert eval_costs.get("max_judge_input_tokens") is not None
        assert eval_costs.get("max_judge_output_tokens") is not None

    def test_split_token_totals(self) -> None:
        """Agent token totals split into input/output."""
        runs = [_make_run_data(1), _make_run_data(2)]
        result = consolidate("model_a", runs, runs_requested=2)

        assert result.overall["total_agent_input_tokens"] == 200.0
        assert result.overall["total_agent_output_tokens"] == 100.0
        assert result.overall["agent_input_tokens_mean"] == 100.0
        assert result.overall["agent_output_tokens_mean"] == 50.0

    def test_no_percentiles_without_turn_stats(self) -> None:
        """No turn_stats → no percentile keys."""
        run = _make_run_data(1)
        result = consolidate("model_a", [run], runs_requested=1)

        assert "agent_input_tokens_p95_max" not in result.overall

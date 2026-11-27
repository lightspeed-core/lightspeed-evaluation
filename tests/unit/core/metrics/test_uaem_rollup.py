# pylint: disable=redefined-outer-name

"""Unit tests for the uaem_lib rollup attached to the evaluation summary.

The rollup math is the metric library's (``lib_side.rollup_scores``); these
tests only cover which of this run's results are passed to it and how they
are grouped. Skipped unless the ``uaem-poc`` extra is installed.
"""

from typing import Optional

import pytest

pytest.importorskip("lib_side", reason="requires the uaem-poc extra")

# pylint: disable=wrong-import-position
from lightspeed_evaluation.core.metrics.uaem.rollup import build_rollup_summary
from lightspeed_evaluation.core.models import EvaluationResult


def _result(
    metric: str,
    raw_score: Optional[float],
    *,
    agent: Optional[str] = "agent_a",
    turn_id: Optional[str] = "t1",
    framework: str = "uaem_lib",
) -> EvaluationResult:
    """One scored result, with only the fields the rollup reads."""
    return EvaluationResult(
        conversation_group_id="conv1",
        agent=agent,
        turn_id=turn_id,
        metric_identifier=f"{framework}:{metric}",
        result="PASS",
        score=0.5,
        raw_score=raw_score,
    )


class TestBuildRollupSummary:
    """Grouping this run's native scores and rolling each group up."""

    def test_no_results_returns_none(self) -> None:
        """A run that never used uaem_lib gets no rollup section."""
        assert build_rollup_summary([]) is None

    def test_other_frameworks_are_ignored(self) -> None:
        """Another framework's results don't count as uaem_lib having run."""
        results = [_result("faithfulness", 1.0, framework="ragas")]

        assert build_rollup_summary(results) is None

    def test_results_without_a_native_score_are_ignored(self) -> None:
        """A metric that produced no native score contributes nothing."""
        results = [_result("req_coverage_prop_model", None)]

        assert build_rollup_summary(results) is None

    def test_means_native_scores_before_rolling_up(self) -> None:
        """Aggregation order is load-bearing: mean native scores, then roll up.

        ``granularity_fit_score_model`` results of -1.0 and +1.0 must mean
        to 0.0 and only then take ``1 - |x|`` (giving 1.0), not the other
        way around (which would give 0.0).
        """
        results = [
            _result("granularity_fit_score_model", -1.0, turn_id="t1"),
            _result("granularity_fit_score_model", 1.0, turn_id="t2"),
        ]

        rolled = build_rollup_summary(results)

        assert rolled is not None
        tree = rolled["agent_a"]["turn"]
        assert tree["metric_scores"]["granularity_fit_score_model"] == 0.0
        assert tree["indicator_scores"]["topic_fidelity"]["score"] == pytest.approx(1.0)

    def test_agents_are_rolled_up_separately(self) -> None:
        """The tree describes the system scored, so each agent gets its own."""
        results = [
            _result("req_coverage_prop_model", 0.4, agent="agent_a"),
            _result("req_coverage_prop_model", 0.8, agent="agent_b"),
        ]

        rolled = build_rollup_summary(results)

        assert rolled is not None
        assert set(rolled) == {"agent_a", "agent_b"}
        assert rolled["agent_a"]["turn"]["metric_scores"] == {
            "req_coverage_prop_model": pytest.approx(0.4)
        }
        assert rolled["agent_b"]["turn"]["metric_scores"] == {
            "req_coverage_prop_model": pytest.approx(0.8)
        }

    def test_results_without_an_agent_group_under_unknown(self) -> None:
        """An offline run names no agent, and still gets its rollup."""
        results = [_result("req_coverage_prop_model", 0.4, agent=None)]

        rolled = build_rollup_summary(results)

        assert rolled is not None
        assert set(rolled) == {"unknown"}

    def test_levels_are_rolled_up_separately(self) -> None:
        """Turn and conversation means are over different populations."""
        results = [
            _result("req_coverage_prop_model", 0.4, turn_id="t1"),
            _result("req_coverage_prop_model", 0.8, turn_id=None),
        ]

        rolled = build_rollup_summary(results)

        assert rolled is not None
        assert set(rolled["agent_a"]) == {"turn", "conversation"}
        assert rolled["agent_a"]["turn"]["metric_scores"] == {
            "req_coverage_prop_model": pytest.approx(0.4)
        }
        assert rolled["agent_a"]["conversation"]["metric_scores"] == {
            "req_coverage_prop_model": pytest.approx(0.8)
        }

    def test_full_tree_shape(self) -> None:
        """One metric per indicator rolls all the way up to a domain name."""
        results = [
            _result("req_coverage_prop_model", 0.4),
            _result("scope_containment_prop_model", 0.5),
            _result("granularity_fit_score_model", 0.2),
        ]

        rolled = build_rollup_summary(results)

        assert rolled is not None
        tree = rolled["agent_a"]["turn"]
        assert set(tree["metric_scores"]) == {
            "req_coverage_prop_model",
            "scope_containment_prop_model",
            "granularity_fit_score_model",
        }
        assert set(tree["indicator_scores"]) == {"completeness", "topic_fidelity"}
        assert set(tree["evaluation_criterion_scores"]) == {"coverage"}
        assert tree["domain_names"] == ["output_quality"]

        coverage = tree["evaluation_criterion_scores"]["coverage"]
        assert coverage["formula"]["type"] == "weighted_mean"
        assert coverage["children"] == {
            "completeness": pytest.approx(0.4),
            "topic_fidelity": pytest.approx(0.65),
        }

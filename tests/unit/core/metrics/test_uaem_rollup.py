# pylint: disable=redefined-outer-name

"""Unit tests for locating the uaem_lib rollup for the evaluation summary.

The rollup math is the metric library's (``lib_side.report``,
``lib_side.rollup_scores``); these tests only cover finding this run's rows
in the sink and handing the library's own result back unchanged. Skipped
unless the ``uaem-poc`` extra is installed.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("lib_side", reason="requires the uaem-poc extra")

# pylint: disable=wrong-import-position
from lightspeed_evaluation.core.metrics.uaem.rollup import build_rollup_summary


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestBuildRollupSummary:
    """Reading this run's sink and rolling it up to indicator/criterion/domain."""

    def test_missing_sink_file_returns_none(self, tmp_path: Path) -> None:
        """A run that never used uaem_lib has no sink file at all."""
        assert build_rollup_summary(tmp_path / "missing.jsonl") is None

    def test_sink_with_no_uaem_lib_rows_returns_none(self, tmp_path: Path) -> None:
        """Other frameworks' rows in the sink don't count as uaem_lib having run."""
        path = tmp_path / "raw.jsonl"
        _write_rows(
            path, [{"model_nm": "other_framework", "metric": "x", "score": 1.0}]
        )

        assert build_rollup_summary(path) is None

    def test_means_raw_scores_before_rolling_up(self, tmp_path: Path) -> None:
        """Aggregation order is load-bearing: mean raw scores, then roll up.

        ``granularity_fit_score_model`` rows of -1.0 and +1.0 must mean to
        0.0 and only then take ``1 - |x|`` (giving 1.0), not the other way
        around (which would give 0.0).
        """
        path = tmp_path / "raw.jsonl"
        _write_rows(
            path,
            [
                {
                    "model_nm": "uaem_lib",
                    "metric": "granularity_fit_score_model",
                    "score": -1.0,
                    "lse_score": 0.0,
                },
                {
                    "model_nm": "uaem_lib",
                    "metric": "granularity_fit_score_model",
                    "score": 1.0,
                    "lse_score": 0.0,
                },
            ],
        )

        result = build_rollup_summary(path)

        assert result is not None
        assert result["metric_scores"]["granularity_fit_score_model"] == 0.0
        topic_fidelity = result["indicator_scores"]["topic_fidelity"]
        assert topic_fidelity["score"] == pytest.approx(1.0)

    def test_full_tree_shape(self, tmp_path: Path) -> None:
        """One metric per indicator rolls all the way up to a domain name."""
        path = tmp_path / "raw.jsonl"
        _write_rows(
            path,
            [
                {
                    "model_nm": "uaem_lib",
                    "metric": "req_coverage_prop_model",
                    "score": 0.4,
                    "lse_score": 0.4,
                },
                {
                    "model_nm": "uaem_lib",
                    "metric": "scope_containment_prop_model",
                    "score": 0.5,
                    "lse_score": 0.5,
                },
                {
                    "model_nm": "uaem_lib",
                    "metric": "granularity_fit_score_model",
                    "score": 0.2,
                    "lse_score": 0.8,
                },
            ],
        )

        result = build_rollup_summary(path)

        assert result is not None
        assert set(result["metric_scores"]) == {
            "req_coverage_prop_model",
            "scope_containment_prop_model",
            "granularity_fit_score_model",
        }
        assert set(result["indicator_scores"]) == {"completeness", "topic_fidelity"}
        assert set(result["evaluation_criterion_scores"]) == {"coverage"}
        assert result["domain_names"] == ["output_quality"]

        coverage = result["evaluation_criterion_scores"]["coverage"]
        assert coverage["formula"]["type"] == "weighted_mean"
        assert coverage["children"] == {
            "completeness": pytest.approx(0.4),
            "topic_fidelity": pytest.approx(0.65),
        }

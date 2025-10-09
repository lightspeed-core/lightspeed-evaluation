"""Unit tests for core.output.comparison module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from lightspeed_evaluation.core.output.comparison import EvaluationComparison


class TestEvaluationComparison:
    """Unit tests for EvaluationComparison class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.comparison = EvaluationComparison(alpha=0.05)

        # Sample evaluation results for testing
        self.sample_results1 = [
            {
                "conversation_group_id": "conv1",
                "turn_id": "1",
                "metric_identifier": "ragas:faithfulness",
                "result": "PASS",
                "score": 0.8,
                "threshold": 0.7,
                "execution_time": 1.0,
            },
            {
                "conversation_group_id": "conv1",
                "turn_id": "2",
                "metric_identifier": "ragas:faithfulness",
                "result": "PASS",
                "score": 0.9,
                "threshold": 0.7,
                "execution_time": 1.2,
            },
            {
                "conversation_group_id": "conv2",
                "turn_id": "1",
                "metric_identifier": "ragas:relevancy",
                "result": "FAIL",
                "score": 0.3,
                "threshold": 0.8,
                "execution_time": 0.8,
            },
        ]

        self.sample_results2 = [
            {
                "conversation_group_id": "conv1",
                "turn_id": "1",
                "metric_identifier": "ragas:faithfulness",
                "result": "PASS",
                "score": 0.85,
                "threshold": 0.7,
                "execution_time": 1.1,
            },
            {
                "conversation_group_id": "conv1",
                "turn_id": "2",
                "metric_identifier": "ragas:faithfulness",
                "result": "FAIL",
                "score": 0.6,
                "threshold": 0.7,
                "execution_time": 1.0,
            },
            {
                "conversation_group_id": "conv2",
                "turn_id": "1",
                "metric_identifier": "ragas:relevancy",
                "result": "PASS",
                "score": 0.9,
                "threshold": 0.8,
                "execution_time": 0.9,
            },
        ]

    def _create_sample_summary(self, results, timestamp="2025-01-01T00:00:00"):
        """Create a sample evaluation summary."""
        return {
            "timestamp": timestamp,
            "total_evaluations": len(results),
            "summary_stats": {
                "overall": {
                    "TOTAL": len(results),
                    "PASS": sum(1 for r in results if r["result"] == "PASS"),
                    "FAIL": sum(1 for r in results if r["result"] == "FAIL"),
                    "ERROR": sum(1 for r in results if r["result"] == "ERROR"),
                    "pass_rate": (
                        sum(1 for r in results if r["result"] == "PASS") / len(results)
                        if results
                        else 0
                    ),
                    "fail_rate": (
                        sum(1 for r in results if r["result"] == "FAIL") / len(results)
                        if results
                        else 0
                    ),
                    "error_rate": (
                        sum(1 for r in results if r["result"] == "ERROR") / len(results)
                        if results
                        else 0
                    ),
                }
            },
            "results": results,
        }

    def test_initialization(self):
        """Test EvaluationComparison initialization."""
        comparison = EvaluationComparison()
        assert comparison.alpha == 0.05

        comparison = EvaluationComparison(alpha=0.01)
        assert comparison.alpha == 0.01

    def test_compare_evaluations_file_not_found(self):
        """Test compare_evaluations with non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.comparison.compare_evaluations(
                "nonexistent1.json", "nonexistent2.json"
            )

    def test_compare_evaluations_success(self):
        """Test successful comparison of two evaluation files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample summary files
            summary1 = self._create_sample_summary(
                self.sample_results1, "2025-01-01T00:00:00"
            )
            summary2 = self._create_sample_summary(
                self.sample_results2, "2025-01-02T00:00:00"
            )

            file1 = Path(temp_dir) / "summary1.json"
            file2 = Path(temp_dir) / "summary2.json"

            with open(file1, "w") as f:
                json.dump(summary1, f)
            with open(file2, "w") as f:
                json.dump(summary2, f)

            # Compare evaluations
            result = self.comparison.compare_evaluations(file1, file2)

            # Check structure
            assert "summary" in result
            assert "overall_comparison" in result
            assert "metric_comparisons" in result
            assert "conversation_comparisons" in result

            # Check summary
            assert result["summary"]["run1"]["file"] == str(file1)
            assert result["summary"]["run2"]["file"] == str(file2)
            assert result["summary"]["alpha"] == 0.05

    def test_compare_overall_stats(self):
        """Test comparison of overall statistics."""
        summary1 = self._create_sample_summary(self.sample_results1)
        summary2 = self._create_sample_summary(self.sample_results2)

        comparison = self.comparison._compare_overall_stats(summary1, summary2)

        assert "pass_rate" in comparison
        assert "fail_rate" in comparison
        assert "error_rate" in comparison

        # Check pass rate comparison
        pass_comp = comparison["pass_rate"]
        assert "run1" in pass_comp
        assert "run2" in pass_comp
        assert "difference" in pass_comp
        assert "relative_change" in pass_comp

    def test_compare_metrics(self):
        """Test metric comparison."""
        comparison = self.comparison._compare_metrics(
            self.sample_results1, self.sample_results2
        )

        # Should have comparisons for both metrics
        assert "ragas:faithfulness" in comparison
        assert "ragas:relevancy" in comparison

        # Check structure of metric comparison
        faithfulness_comp = comparison["ragas:faithfulness"]
        assert "metric" in faithfulness_comp
        assert "run1_count" in faithfulness_comp
        assert "run2_count" in faithfulness_comp
        assert "score_comparison" in faithfulness_comp
        assert "pass_rate_comparison" in faithfulness_comp
        assert "statistical_significance" in faithfulness_comp

    def test_compare_single_metric(self):
        """Test comparison of a single metric."""
        metric_results1 = [
            r
            for r in self.sample_results1
            if r["metric_identifier"] == "ragas:faithfulness"
        ]
        metric_results2 = [
            r
            for r in self.sample_results2
            if r["metric_identifier"] == "ragas:faithfulness"
        ]

        comparison = self.comparison._compare_single_metric(
            "ragas:faithfulness", metric_results1, metric_results2
        )

        assert comparison["metric"] == "ragas:faithfulness"
        assert comparison["run1_count"] == 2
        assert comparison["run2_count"] == 2
        assert comparison["score_comparison"] is not None
        assert comparison["pass_rate_comparison"] is not None

    def test_compare_score_distributions(self):
        """Test score distribution comparison."""
        scores1 = [0.8, 0.9]
        scores2 = [0.85, 0.6]

        comparison = self.comparison._compare_score_distributions(scores1, scores2)

        assert "run1_stats" in comparison
        assert "run2_stats" in comparison
        assert "mean_difference" in comparison
        assert "relative_change" in comparison
        assert "tests" in comparison

        # Check statistics
        assert abs(comparison["run1_stats"]["mean"] - 0.85) < 1e-10
        assert abs(comparison["run2_stats"]["mean"] - 0.725) < 1e-10
        assert comparison["mean_difference"] == pytest.approx(-0.125, abs=1e-10)

        # Check that tests were performed
        assert (
            "t_test" in comparison["tests"] or "mann_whitney_u" in comparison["tests"]
        )

    def test_compare_pass_rates(self):
        """Test pass rate comparison."""
        comparison = self.comparison._compare_pass_rates(2, 2, 1, 2)

        assert "run1" in comparison
        assert "run2" in comparison
        assert "difference" in comparison
        assert "relative_change" in comparison
        assert "tests" in comparison

        # Check pass rates
        assert comparison["run1"]["pass_rate"] == 1.0
        assert comparison["run2"]["pass_rate"] == 0.5
        assert comparison["difference"] == -0.5
        assert comparison["relative_change"] == -50.0

    def test_compare_score_distributions_insufficient_data(self):
        """Test score distribution comparison with insufficient data."""
        scores1 = [0.8]
        scores2 = [0.9]

        comparison = self.comparison._compare_score_distributions(scores1, scores2)

        # Should still calculate basic stats but may not perform all tests
        assert "run1_stats" in comparison
        assert "run2_stats" in comparison
        assert comparison["run1_stats"]["count"] == 1
        assert comparison["run2_stats"]["count"] == 1

    def test_compare_pass_rates_insufficient_data(self):
        """Test pass rate comparison with insufficient data."""
        comparison = self.comparison._compare_pass_rates(0, 0, 1, 1)

        # Should handle zero totals gracefully
        assert comparison["run1"]["pass_rate"] == 0
        assert comparison["run2"]["pass_rate"] == 1.0

    def test_determine_overall_significance(self):
        """Test determination of overall statistical significance."""
        # Test with significant score comparison
        score_comparison = {
            "tests": {
                "t_test": {"significant": True, "p_value": 0.01},
                "mann_whitney_u": {"significant": False, "p_value": 0.1},
            }
        }
        pass_rate_comparison = {
            "tests": {"chi_square": {"significant": False, "p_value": 0.2}}
        }

        significance = self.comparison._determine_overall_significance(
            score_comparison, pass_rate_comparison
        )

        assert significance["overall_significant"] is True
        assert significance["score_significant"] is True
        assert significance["pass_rate_significant"] is False
        assert "score distributions" in significance["summary"]

    def test_determine_overall_significance_no_significance(self):
        """Test determination when no significance is found."""
        score_comparison = {
            "tests": {
                "t_test": {"significant": False, "p_value": 0.1},
                "mann_whitney_u": {"significant": False, "p_value": 0.2},
            }
        }
        pass_rate_comparison = {
            "tests": {"chi_square": {"significant": False, "p_value": 0.3}}
        }

        significance = self.comparison._determine_overall_significance(
            score_comparison, pass_rate_comparison
        )

        assert significance["overall_significant"] is False
        assert significance["score_significant"] is False
        assert significance["pass_rate_significant"] is False
        assert "No statistically significant differences" in significance["summary"]

    def test_interpret_test_result(self):
        """Test interpretation of statistical test results."""
        # Test different p-value ranges
        assert "Highly significant" in self.comparison._interpret_test_result(
            0.0005, "t-test"
        )
        assert "Very significant" in self.comparison._interpret_test_result(
            0.005, "t-test"
        )
        assert "Significant" in self.comparison._interpret_test_result(0.03, "t-test")
        assert "Marginally significant" in self.comparison._interpret_test_result(
            0.08, "t-test"
        )
        assert "Not significant" in self.comparison._interpret_test_result(
            0.2, "t-test"
        )

    def test_generate_report(self):
        """Test report generation."""
        # Create a sample comparison result
        comparison_results = {
            "summary": {
                "run1": {
                    "file": "run1.json",
                    "timestamp": "2025-01-01T00:00:00",
                    "total_evaluations": 3,
                },
                "run2": {
                    "file": "run2.json",
                    "timestamp": "2025-01-02T00:00:00",
                    "total_evaluations": 3,
                },
                "alpha": 0.05,
            },
            "overall_comparison": {
                "pass_rate": {
                    "run1": 0.667,
                    "run2": 0.5,
                    "difference": -0.167,
                    "relative_change": -25.0,
                },
                "fail_rate": {
                    "run1": 0.333,
                    "run2": 0.5,
                    "difference": 0.167,
                    "relative_change": 50.0,
                },
                "error_rate": {
                    "run1": 0.0,
                    "run2": 0.0,
                    "difference": 0.0,
                    "relative_change": 0.0,
                },
            },
            "metric_comparisons": {
                "ragas:faithfulness": {
                    "metric": "ragas:faithfulness",
                    "run1_count": 2,
                    "run2_count": 2,
                    "score_comparison": {
                        "run1_stats": {
                            "mean": 0.85,
                            "std": 0.05,
                            "median": 0.85,
                            "count": 2,
                        },
                        "run2_stats": {
                            "mean": 0.725,
                            "std": 0.125,
                            "median": 0.725,
                            "count": 2,
                        },
                        "mean_difference": -0.125,
                        "relative_change": -14.7,
                        "tests": {
                            "t_test": {
                                "statistic": -1.0,
                                "p_value": 0.5,
                                "significant": False,
                            }
                        },
                    },
                    "pass_rate_comparison": {
                        "run1": {"pass_count": 2, "total": 2, "pass_rate": 1.0},
                        "run2": {"pass_count": 1, "total": 2, "pass_rate": 0.5},
                        "difference": -0.5,
                        "relative_change": -50.0,
                        "tests": {
                            "chi_square": {
                                "statistic": 2.0,
                                "p_value": 0.5,
                                "significant": False,
                            }
                        },
                    },
                    "statistical_significance": {
                        "overall_significant": False,
                        "score_significant": False,
                        "pass_rate_significant": False,
                        "summary": "No statistically significant differences found",
                    },
                }
            },
            "conversation_comparisons": {},
        }

        report = self.comparison.generate_report(comparison_results)

        # Check that report contains expected sections
        assert "EVALUATION COMPARISON REPORT" in report
        assert "OVERALL STATISTICS COMPARISON" in report
        assert "METRIC-BY-METRIC COMPARISON" in report
        assert "SUMMARY OF SIGNIFICANT DIFFERENCES" in report

        # Check that specific values are included
        assert "run1.json" in report
        assert "run2.json" in report
        assert "ragas:faithfulness" in report
        assert "No statistically significant differences found" in report

    def test_group_results_by_metric(self):
        """Test grouping results by metric."""
        grouped = self.comparison._group_results_by_metric(self.sample_results1)

        assert "ragas:faithfulness" in grouped
        assert "ragas:relevancy" in grouped
        assert len(grouped["ragas:faithfulness"]) == 2
        assert len(grouped["ragas:relevancy"]) == 1

    def test_group_results_by_conversation(self):
        """Test grouping results by conversation."""
        grouped = self.comparison._group_results_by_conversation(self.sample_results1)

        assert "conv1" in grouped
        assert "conv2" in grouped
        assert len(grouped["conv1"]) == 2
        assert len(grouped["conv2"]) == 1

    def test_compare_conversations(self):
        """Test conversation comparison."""
        comparison = self.comparison._compare_conversations(
            self.sample_results1, self.sample_results2
        )

        assert "conv1" in comparison
        assert "conv2" in comparison

        # Check structure
        conv1_comp = comparison["conv1"]
        assert "conversation" in conv1_comp
        assert "run1_count" in conv1_comp
        assert "run2_count" in conv1_comp
        assert "pass_rate_comparison" in conv1_comp

    def test_compare_single_conversation(self):
        """Test comparison of a single conversation."""
        conv_results1 = [
            r for r in self.sample_results1 if r["conversation_group_id"] == "conv1"
        ]
        conv_results2 = [
            r for r in self.sample_results2 if r["conversation_group_id"] == "conv1"
        ]

        comparison = self.comparison._compare_single_conversation(
            "conv1", conv_results1, conv_results2
        )

        assert comparison["conversation"] == "conv1"
        assert comparison["run1_count"] == 2
        assert comparison["run2_count"] == 2
        assert comparison["pass_rate_comparison"] is not None

    @patch("lightspeed_evaluation.core.output.comparison.ttest_ind")
    def test_compare_score_distributions_test_failure(self, mock_ttest):
        """Test score distribution comparison when statistical tests fail."""
        mock_ttest.side_effect = Exception("Test failed")

        scores1 = [0.8, 0.9]
        scores2 = [0.85, 0.6]

        comparison = self.comparison._compare_score_distributions(scores1, scores2)

        # Should still have basic stats but tests should show error
        assert "run1_stats" in comparison
        assert "run2_stats" in comparison
        assert "error" in comparison["tests"]

    @patch("lightspeed_evaluation.core.output.comparison.chi2_contingency")
    def test_compare_pass_rates_test_failure(self, mock_chi2):
        """Test pass rate comparison when statistical tests fail."""
        mock_chi2.side_effect = Exception("Test failed")

        comparison = self.comparison._compare_pass_rates(2, 2, 1, 2)

        # Should still have basic comparison but tests should show error
        assert "run1" in comparison
        assert "run2" in comparison
        assert "error" in comparison["tests"]

    def test_empty_results_handling(self):
        """Test handling of empty results."""
        comparison = self.comparison._compare_metrics([], [])
        assert comparison == {}

        comparison = self.comparison._compare_conversations([], [])
        assert comparison == {}

    def test_missing_metric_in_one_run(self):
        """Test comparison when a metric exists in only one run."""
        results1 = [
            r
            for r in self.sample_results1
            if r["metric_identifier"] == "ragas:faithfulness"
        ]
        results2 = []  # No results for this metric in run 2

        comparison = self.comparison._compare_single_metric(
            "ragas:faithfulness", results1, results2
        )

        assert comparison["run1_count"] == 2
        assert comparison["run2_count"] == 0
        assert comparison["score_comparison"] is None
        assert comparison["pass_rate_comparison"] is None

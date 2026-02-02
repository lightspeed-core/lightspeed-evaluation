#!/usr/bin/env python3
"""Standalone script for comparing evaluation runs and determining statistical significance.

This script is a utility tool for comparing two evaluation summary JSON files.
It combines the CLI interface and comparison logic in a single executable script.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind

logger = logging.getLogger(__name__)


class EvaluationComparison:
    """Tool for comparing two evaluation runs and determining statistical significance."""

    def __init__(self, alpha: float = 0.05):
        """Initialize the comparison tool.

        Args:
            alpha: Significance level for statistical tests (default: 0.05)
        """
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def compare_evaluations(
        self, summary1_path: Union[str, Path], summary2_path: Union[str, Path]
    ) -> dict[str, Any]:
        """Compare two evaluation summary files and return statistical significance results.

        The summaries are automatically ordered by timestamp, with the earlier timestamp
        becoming run1 and the later timestamp becoming run2.

        Args:
            summary1_path: Path to first evaluation summary JSON file
            summary2_path: Path to second evaluation summary JSON file

        Returns:
            Dictionary containing comparison results for each metric
        """
        # Load evaluation summaries
        summary_data1 = self._load_summary(summary1_path)
        summary_data2 = self._load_summary(summary2_path)

        # Assign summaries based on timestamp (earlier = summary1, later = summary2)
        timestamp1 = summary_data1.get("timestamp", "")
        timestamp2 = summary_data2.get("timestamp", "")

        if timestamp1 <= timestamp2:
            summary1 = summary_data1
            summary2 = summary_data2
            summary1_path_final = summary1_path
            summary2_path_final = summary2_path
        else:
            summary1 = summary_data2
            summary2 = summary_data1
            summary1_path_final = summary2_path
            summary2_path_final = summary1_path

        # Extract results for detailed analysis
        results1 = summary1.get("results", [])
        results2 = summary2.get("results", [])

        comparison_results = {
            "summary": {
                "run1": {
                    "file": str(summary1_path_final),
                    "timestamp": summary1.get("timestamp"),
                    "total_evaluations": summary1.get("total_evaluations", 0),
                },
                "run2": {
                    "file": str(summary2_path_final),
                    "timestamp": summary2.get("timestamp"),
                    "total_evaluations": summary2.get("total_evaluations", 0),
                },
                "alpha": self.alpha,
            },
            "overall_comparison": self._compare_overall_stats(summary1, summary2),
            "metric_comparisons": self._compare_metrics(
                results1, results2, summary1, summary2
            ),
            "conversation_comparisons": self._compare_conversations(results1, results2),
        }

        return comparison_results

    def _load_summary(self, path: Union[str, Path]) -> dict[str, Any]:
        """Load evaluation summary from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compare_overall_stats(
        self, summary1: dict[str, Any], summary2: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare overall statistics between two evaluation runs."""
        overall1 = summary1.get("summary_stats", {}).get("overall", {})
        overall2 = summary2.get("summary_stats", {}).get("overall", {})

        comparison = {}

        # Compare pass rates
        pass_rate1 = overall1.get("pass_rate", 0)
        pass_rate2 = overall2.get("pass_rate", 0)

        pass_rate_delta = abs(pass_rate2 - pass_rate1)
        if pass_rate1 > 0:
            pass_rate_rel = (pass_rate_delta / pass_rate1) * 100
        elif pass_rate_delta > 0:
            pass_rate_rel = float("inf")
        else:
            pass_rate_rel = 0.0

        comparison["pass_rate"] = {
            "run1": pass_rate1,
            "run2": pass_rate2,
            "difference": pass_rate_delta,
            "relative_change": pass_rate_rel,
        }

        # Compare fail rates
        fail_rate1 = overall1.get("fail_rate", 0)
        fail_rate2 = overall2.get("fail_rate", 0)

        fail_rate_delta = abs(fail_rate2 - fail_rate1)
        if fail_rate1 > 0:
            fail_rate_rel = (fail_rate_delta / fail_rate1) * 100
        elif fail_rate_delta > 0:
            fail_rate_rel = float("inf")
        else:
            fail_rate_rel = 0.0

        comparison["fail_rate"] = {
            "run1": fail_rate1,
            "run2": fail_rate2,
            "difference": fail_rate_delta,
            "relative_change": fail_rate_rel,
        }

        # Compare error rates
        error_rate1 = overall1.get("error_rate", 0)
        error_rate2 = overall2.get("error_rate", 0)

        error_rate_delta = abs(error_rate2 - error_rate1)
        if error_rate1 > 0:
            error_rate_rel = (error_rate_delta / error_rate1) * 100
        elif error_rate_delta > 0:
            error_rate_rel = float("inf")
        else:
            error_rate_rel = 0.0

        comparison["error_rate"] = {
            "run1": error_rate1,
            "run2": error_rate2,
            "difference": error_rate_delta,
            "relative_change": error_rate_rel,
        }

        return comparison

    def _compare_metrics(
        self,
        results1: list[dict[str, Any]],
        results2: list[dict[str, Any]],
        summary1: dict[str, Any],
        summary2: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare metrics between two evaluation runs."""
        # Group results by metric
        metrics1 = self._group_results_by_metric(results1)
        metrics2 = self._group_results_by_metric(results2)

        # Get all unique metrics
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())

        comparisons = {}

        for metric in all_metrics:
            metric_results1 = metrics1.get(metric, [])
            metric_results2 = metrics2.get(metric, [])

            if not metric_results1 and not metric_results2:
                continue

            # Get confidence intervals from summary statistics
            summary_stats1 = (
                summary1.get("summary_stats", {}).get("by_metric", {}).get(metric, {})
            )
            summary_stats2 = (
                summary2.get("summary_stats", {}).get("by_metric", {}).get(metric, {})
            )
            ci1 = summary_stats1.get("score_statistics", {}).get("confidence_interval")
            ci2 = summary_stats2.get("score_statistics", {}).get("confidence_interval")

            comparisons[metric] = self._compare_single_metric(
                metric, metric_results1, metric_results2, ci1, ci2
            )

        return comparisons

    def _compare_conversations(
        self, results1: list[dict[str, Any]], results2: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compare conversation-level statistics between two evaluation runs."""
        # Group results by conversation
        convs1 = self._group_results_by_conversation(results1)
        convs2 = self._group_results_by_conversation(results2)

        # Get all unique conversations
        all_convs = set(convs1.keys()) | set(convs2.keys())

        comparisons = {}

        for conv in all_convs:
            conv_results1 = convs1.get(conv, [])
            conv_results2 = convs2.get(conv, [])

            if not conv_results1 and not conv_results2:
                continue

            comparisons[conv] = self._compare_single_conversation(
                conv, conv_results1, conv_results2
            )

        return comparisons

    def _group_results_by_metric(
        self, results: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group results by metric identifier."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            metric = result.get("metric_identifier", "unknown")
            if metric not in grouped:
                grouped[metric] = []
            grouped[metric].append(result)
        return grouped

    def _group_results_by_conversation(
        self, results: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group results by conversation group ID."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            conv = result.get("conversation_group_id", "unknown")
            if conv not in grouped:
                grouped[conv] = []
            grouped[conv].append(result)
        return grouped

    def _compare_single_metric(
        self,
        metric: str,
        results1: list[dict[str, Any]],
        results2: list[dict[str, Any]],
        ci1: Optional[dict[str, Any]] = None,
        ci2: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compare a single metric between two evaluation runs."""
        comparison = {
            "metric": metric,
            "run1_count": len(results1),
            "run2_count": len(results2),
            "score_comparison": None,
            "pass_rate_comparison": None,
            "statistical_significance": {},
        }

        # Extract scores for statistical testing
        scores1: list[float] = []
        scores2: list[float] = []
        for r in results1:
            score = r.get("score")
            if score is not None:
                scores1.append(float(score))
        for r in results2:
            score = r.get("score")
            if score is not None:
                scores2.append(float(score))

        # Compare scores if we have enough data
        if len(scores1) > 0 and len(scores2) > 0:
            comparison["score_comparison"] = self._compare_score_distributions(
                scores1, scores2
            )

        # Add confidence interval overlap test
        comparison["confidence_interval_test"] = (
            self._check_confidence_interval_overlap(ci1, ci2)
        )

        # Compare pass rates
        pass_count1 = sum(1 for r in results1 if r.get("result") == "PASS")
        pass_count2 = sum(1 for r in results2 if r.get("result") == "PASS")
        total1 = len(results1)
        total2 = len(results2)

        if total1 > 0 and total2 > 0:
            comparison["pass_rate_comparison"] = self._compare_pass_rates(
                pass_count1, total1, pass_count2, total2
            )

        # Determine overall statistical significance
        comparison["statistical_significance"] = self._determine_overall_significance(
            comparison["score_comparison"],
            comparison["pass_rate_comparison"],
            comparison["confidence_interval_test"],
        )

        return comparison

    def _compare_single_conversation(
        self, conv: str, results1: list[dict[str, Any]], results2: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compare a single conversation between two evaluation runs."""
        comparison = {
            "conversation": conv,
            "run1_count": len(results1),
            "run2_count": len(results2),
            "pass_rate_comparison": None,
        }

        # Compare pass rates
        pass_count1 = sum(1 for r in results1 if r.get("result") == "PASS")
        pass_count2 = sum(1 for r in results2 if r.get("result") == "PASS")
        total1 = len(results1)
        total2 = len(results2)

        if total1 > 0 and total2 > 0:
            comparison["pass_rate_comparison"] = self._compare_pass_rates(
                pass_count1, total1, pass_count2, total2
            )

        return comparison

    def _compare_score_distributions(
        self, scores1: list[float], scores2: list[float]
    ) -> dict[str, Any]:
        """Compare score distributions using statistical tests."""
        scores1_array = np.array(scores1)
        scores2_array = np.array(scores2)

        comparison: dict[str, Any] = {
            "run1_stats": {
                "mean": float(np.mean(scores1_array)),
                "std": float(np.std(scores1_array)),
                "median": float(np.median(scores1_array)),
                "count": len(scores1_array),
            },
            "run2_stats": {
                "mean": float(np.mean(scores2_array)),
                "std": float(np.std(scores2_array)),
                "median": float(np.median(scores2_array)),
                "count": len(scores2_array),
            },
            "tests": {},
        }

        # Calculate difference in means
        mean_diff = comparison["run2_stats"]["mean"] - comparison["run1_stats"]["mean"]
        comparison["mean_difference"] = mean_diff
        comparison["relative_change"] = (
            (mean_diff / comparison["run1_stats"]["mean"] * 100)
            if comparison["run1_stats"]["mean"] > 0
            else 0
        )

        # Perform statistical tests
        try:
            # T-test (assumes normal distribution)
            if len(scores1_array) > 1 and len(scores2_array) > 1:
                ttest_result = ttest_ind(scores1_array, scores2_array)
                t_stat, t_pvalue = cast(Tuple[float, float], ttest_result)
                comparison["tests"]["t_test"] = {
                    "statistic": t_stat,
                    "p_value": t_pvalue,
                    "significant": bool(t_pvalue < self.alpha),
                    "interpretation": self._interpret_test_result(t_pvalue, "t-test"),
                }

            # Mann-Whitney U test (non-parametric)
            if len(scores1_array) > 0 and len(scores2_array) > 0:
                mw_result = mannwhitneyu(
                    scores1_array, scores2_array, alternative="two-sided"
                )
                u_stat, u_pvalue = cast(Tuple[float, float], mw_result)
                comparison["tests"]["mann_whitney_u"] = {
                    "statistic": u_stat,
                    "p_value": u_pvalue,
                    "significant": bool(u_pvalue < self.alpha),
                    "interpretation": self._interpret_test_result(
                        u_pvalue, "Mann-Whitney U test"
                    ),
                }
        except (ValueError, TypeError) as e:
            self.logger.warning("Statistical test failed: %s", e)
            comparison["tests"]["error"] = str(e)

        return comparison

    def _check_confidence_interval_overlap(
        self, ci1: Optional[dict[str, Any]], ci2: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        """Check if confidence intervals overlap and determine statistical significance.

        Args:
            ci1: Confidence interval from run1 with 'low', 'high', 'mean', 'confidence_level'
            ci2: Confidence interval from run2 with 'low', 'high', 'mean', 'confidence_level'

        Returns:
            Dictionary containing overlap test results
        """
        result: dict[str, Any] = {
            "test_performed": False,
            "intervals_overlap": None,
            "significant": None,
            "interpretation": "Confidence interval data not available",
            "run1_interval": None,
            "run2_interval": None,
            "confidence_level": None,
        }

        # Check if both confidence intervals are available and valid
        if not ci1 or not ci2:
            return result

        # Validate that confidence intervals have required fields
        required_fields = ["low", "high", "mean"]
        if not all(field in ci1 for field in required_fields) or not all(
            field in ci2 for field in required_fields
        ):
            return result

        # Extract interval bounds
        ci1_low, ci1_high = ci1["low"], ci1["high"]
        ci2_low, ci2_high = ci2["low"], ci2["high"]

        # Validate that intervals are properly formed (low <= high)
        if ci1_low > ci1_high or ci2_low > ci2_high:
            result["interpretation"] = "Invalid confidence interval bounds"
            return result

        # Check for overlap: intervals overlap if max(low1, low2) <= min(high1, high2)
        intervals_overlap = max(ci1_low, ci2_low) <= min(ci1_high, ci2_high)

        # Statistical significance is the opposite of overlap
        # If intervals don't overlap, the difference is statistically significant
        significant = not intervals_overlap

        # Prepare result
        result.update(
            {
                "test_performed": True,
                "intervals_overlap": intervals_overlap,
                "significant": significant,
                "interpretation": (
                    "No statistically significant difference (confidence intervals overlap)"
                    if intervals_overlap
                    else "Statistically significant difference (confidence intervals do not overlap)"
                ),
                "run1_interval": {
                    "low": ci1_low,
                    "high": ci1_high,
                    "mean": ci1["mean"],
                },
                "run2_interval": {
                    "low": ci2_low,
                    "high": ci2_high,
                    "mean": ci2["mean"],
                },
                "confidence_level": ci1.get(
                    "confidence_level", ci2.get("confidence_level", "unknown")
                ),
            }
        )

        return result

    def _compare_pass_rates(
        self, pass_count1: int, total1: int, pass_count2: int, total2: int
    ) -> dict[str, Any]:
        """Compare pass rates using statistical tests."""
        pass_rate1 = pass_count1 / total1 if total1 > 0 else 0
        pass_rate2 = pass_count2 / total2 if total2 > 0 else 0

        delta = pass_rate2 - pass_rate1
        if pass_rate1 > 0:
            rel_change = (delta / pass_rate1) * 100
        elif delta > 0:
            rel_change = float("inf")
        elif delta < 0:
            rel_change = float("-inf")
        else:
            rel_change = 0.0

        comparison = {
            "run1": {
                "pass_count": pass_count1,
                "total": total1,
                "pass_rate": pass_rate1,
            },
            "run2": {
                "pass_count": pass_count2,
                "total": total2,
                "pass_rate": pass_rate2,
            },
            "difference": delta,
            "relative_change": rel_change,
            "tests": {},
        }

        # Perform statistical tests
        test_data = {
            "pass_count1": pass_count1,
            "total1": total1,
            "pass_count2": pass_count2,
            "total2": total2,
        }
        self._perform_pass_rate_tests(comparison, test_data)
        return comparison

    def _perform_pass_rate_tests(
        self, comparison: dict[str, Any], test_data: dict[str, int]
    ) -> None:
        """Perform statistical tests for pass rate comparison."""
        try:
            if test_data["total1"] > 0 and test_data["total2"] > 0:
                self._perform_chi_square_test(comparison, test_data)
                self._perform_fisher_exact_test(comparison, test_data)
        except (ValueError, TypeError) as e:
            self.logger.warning("Pass rate statistical test failed: %s", e)
            comparison["tests"]["error"] = str(e)
            # Clear any partial test results if there was an error
            if "chi_square" in comparison["tests"]:
                del comparison["tests"]["chi_square"]
            if "fisher_exact" in comparison["tests"]:
                del comparison["tests"]["fisher_exact"]

    def _perform_chi_square_test(
        self, comparison: dict[str, Any], test_data: dict[str, int]
    ) -> None:
        """Perform chi-square test for pass rates."""
        contingency_table = np.array(
            [
                [
                    test_data["pass_count1"],
                    test_data["total1"] - test_data["pass_count1"],
                ],
                [
                    test_data["pass_count2"],
                    test_data["total2"] - test_data["pass_count2"],
                ],
            ]
        )

        try:
            chi2_result = chi2_contingency(contingency_table)
            chi2_stat, chi2_pvalue, dof, _ = cast(
                Tuple[float, float, int, Any], chi2_result
            )
            comparison["tests"]["chi_square"] = {
                "statistic": float(chi2_stat),
                "p_value": float(chi2_pvalue),
                "degrees_of_freedom": int(dof),
                "significant": bool(chi2_pvalue < self.alpha),
                "interpretation": self._interpret_test_result(
                    float(chi2_pvalue), "Chi-square test"
                ),
            }
        except ValueError as e:
            comparison["tests"]["chi_square"] = {
                "error": f"Chi-square test not applicable: {str(e)}"
            }

    def _perform_fisher_exact_test(
        self, comparison: dict[str, Any], test_data: dict[str, int]
    ) -> None:
        """Perform Fisher's exact test for pass rates."""
        if test_data["total1"] <= 20 or test_data["total2"] <= 20:
            contingency_table = np.array(
                [
                    [
                        test_data["pass_count1"],
                        test_data["total1"] - test_data["pass_count1"],
                    ],
                    [
                        test_data["pass_count2"],
                        test_data["total2"] - test_data["pass_count2"],
                    ],
                ]
            )

            fisher_result = fisher_exact(contingency_table)
            odds_ratio, fisher_pvalue = cast(Tuple[float, float], fisher_result)
            comparison["tests"]["fisher_exact"] = {
                "odds_ratio": float(odds_ratio),
                "p_value": float(fisher_pvalue),
                "significant": bool(fisher_pvalue < self.alpha),
                "interpretation": self._interpret_test_result(
                    float(fisher_pvalue), "Fisher's exact test"
                ),
            }

    def _determine_overall_significance(
        self,
        score_comparison: Optional[dict[str, Any]],
        pass_rate_comparison: Optional[dict[str, Any]],
        confidence_interval_test: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Determine overall statistical significance based on available tests."""
        significance = {
            "overall_significant": False,
            "score_significant": False,
            "pass_rate_significant": False,
            "confidence_interval_significant": False,
            "summary": "No significant differences found",
        }

        # Check score significance
        if score_comparison and "tests" in score_comparison:
            score_tests = score_comparison["tests"]
            # Check if any test shows significance
            for test_result in score_tests.values():
                if isinstance(test_result, dict) and test_result.get(
                    "significant", False
                ):
                    significance["score_significant"] = True
                    break

        # Check pass rate significance
        if pass_rate_comparison and "tests" in pass_rate_comparison:
            pass_rate_tests = pass_rate_comparison["tests"]
            # Check if any test shows significance
            for test_result in pass_rate_tests.values():
                if isinstance(test_result, dict) and test_result.get(
                    "significant", False
                ):
                    significance["pass_rate_significant"] = True
                    break

        # Check confidence interval significance
        if confidence_interval_test and confidence_interval_test.get(
            "test_performed", False
        ):
            significance["confidence_interval_significant"] = (
                confidence_interval_test.get("significant", False)
            )

        # Determine overall significance
        significance["overall_significant"] = (
            significance["score_significant"]
            or significance["pass_rate_significant"]
            or significance["confidence_interval_significant"]
        )

        # Generate summary
        if significance["overall_significant"]:
            parts = []
            if significance["score_significant"]:
                parts.append("score distributions")
            if significance["pass_rate_significant"]:
                parts.append("pass rates")
            if significance["confidence_interval_significant"]:
                parts.append("confidence intervals")
            significance["summary"] = (
                f"Statistically significant differences found in: {', '.join(parts)}"
            )
        else:
            significance["summary"] = "No statistically significant differences found"

        return significance

    def _interpret_test_result(self, p_value: float, test_name: str) -> str:
        """Interpret statistical test result."""
        if p_value < 0.001:
            return f"{test_name}: Highly significant (p < 0.001)"
        if p_value < 0.01:
            return f"{test_name}: Very significant (p < 0.01)"
        if p_value < 0.05:
            return f"{test_name}: Significant (p < 0.05)"
        if p_value < 0.1:
            return f"{test_name}: Marginally significant (p < 0.1)"
        return f"{test_name}: Not significant (p >= 0.1)"

    def generate_report(self, comparison_results: dict[str, Any]) -> str:
        """Generate a human-readable comparison report."""
        report: list[str] = []
        self._add_report_header(report)
        self._add_summary_section(report, comparison_results["summary"])
        self._add_overall_comparison_section(
            report, comparison_results["overall_comparison"]
        )
        self._add_metric_comparison_section(
            report, comparison_results["metric_comparisons"]
        )
        self._add_significant_differences_summary(
            report, comparison_results["metric_comparisons"]
        )
        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def _add_report_header(self, report: list[str]) -> None:
        """Add the report header."""
        report.append("=" * 80)
        report.append("EVALUATION COMPARISON REPORT")
        report.append("=" * 80)

    def _add_summary_section(self, report: list[str], summary: dict[str, Any]) -> None:
        """Add the summary section to the report."""
        report.append(f"\nRun 1: {summary['run1']['file']}")
        report.append(f"  Timestamp: {summary['run1']['timestamp']}")
        report.append(f"  Total Evaluations: {summary['run1']['total_evaluations']}")

        report.append(f"\nRun 2: {summary['run2']['file']}")
        report.append(f"  Timestamp: {summary['run2']['timestamp']}")
        report.append(f"  Total Evaluations: {summary['run2']['total_evaluations']}")

        report.append(f"\nSignificance Level: α = {summary['alpha']}")

    def _add_overall_comparison_section(
        self, report: list[str], overall: dict[str, Any]
    ) -> None:
        """Add the overall comparison section to the report."""
        report.append("\n" + "=" * 50)
        report.append("OVERALL STATISTICS COMPARISON")
        report.append("=" * 50)

        for metric in ["pass_rate", "fail_rate", "error_rate"]:
            if metric in overall:
                data = overall[metric]
                report.append(f"\n{metric.replace('_', ' ').title()}:")

                # Check if the rates are already in percentage form (> 1.0) or decimal form (0.0-1.0)
                # If they're already percentages, don't multiply by 100
                run1_val = data["run1"]
                run2_val = data["run2"]
                diff_val = data["difference"]

                # If values are > 1.0, they're likely already percentages
                if run1_val > 1.0 or run2_val > 1.0:
                    report.append(f"  Run 1: {run1_val:.2f}%")
                    report.append(f"  Run 2: {run2_val:.2f}%")
                    report.append(f"  Difference: {diff_val:+.2f}%")
                else:
                    # Values are in decimal form, convert to percentage
                    report.append(f"  Run 1: {run1_val * 100:.2f}%")
                    report.append(f"  Run 2: {run2_val * 100:.2f}%")
                    report.append(f"  Difference: {diff_val * 100:+.2f}%")

                # Handle infinite relative change
                rel_change = data["relative_change"]
                if rel_change == float("inf"):
                    report.append("  Relative Change: +∞% (baseline was 0)")
                elif rel_change == float("-inf"):
                    report.append("  Relative Change: -∞% (baseline was 0)")
                else:
                    report.append(f"  Relative Change: {rel_change:+.1f}%")

    def _add_metric_comparison_section(
        self, report: list[str], metrics: dict[str, Any]
    ) -> None:
        """Add the metric comparison section to the report."""
        report.append("\n" + "=" * 50)
        report.append("METRIC-BY-METRIC COMPARISON")
        report.append("=" * 50)

        for metric, comparison in metrics.items():
            report.append(f"\n{metric}:")
            report.append(f"  Run 1: {comparison['run1_count']} evaluations")
            report.append(f"  Run 2: {comparison['run2_count']} evaluations")

            self._add_score_comparison(report, comparison)
            self._add_confidence_interval_comparison(report, comparison)
            self._add_pass_rate_comparison(report, comparison)
            significance = comparison["statistical_significance"]
            report.append(f"  Overall Significance: {significance['summary']}")

    def _add_score_comparison(
        self, report: list[str], comparison: dict[str, Any]
    ) -> None:
        """Add score comparison details to the report."""
        if comparison["score_comparison"]:
            score_comp = comparison["score_comparison"]
            report.append("  Score Comparison:")
            report.append(f"    Run 1 Mean: {score_comp['run1_stats']['mean']:.3f}")
            report.append(f"    Run 2 Mean: {score_comp['run2_stats']['mean']:.3f}")
            report.append(f"    Difference: {score_comp['mean_difference']:+.3f}")

            # Handle infinite relative change
            rel_change = score_comp["relative_change"]
            if rel_change == float("inf"):
                report.append("    Relative Change: +∞% (baseline was 0)")
            elif rel_change == float("-inf"):
                report.append("    Relative Change: -∞% (baseline was 0)")
            else:
                report.append(f"    Relative Change: {rel_change:+.1f}%")

            self._add_statistical_tests(report, score_comp.get("tests", {}), "    ")

    def _add_confidence_interval_comparison(
        self, report: list[str], comparison: dict[str, Any]
    ) -> None:
        """Add confidence interval comparison details to the report."""
        if comparison.get("confidence_interval_test"):
            ci_test = comparison["confidence_interval_test"]
            if ci_test.get("test_performed", False):
                report.append("  Confidence Interval Overlap Test:")

                # Show confidence intervals
                if ci_test.get("run1_interval") and ci_test.get("run2_interval"):
                    ci1 = ci_test["run1_interval"]
                    ci2 = ci_test["run2_interval"]
                    confidence_level = ci_test.get("confidence_level", "unknown")

                    report.append(f"    Confidence Level: {confidence_level}%")
                    report.append(
                        f"    Run 1 Interval: [{ci1['low']:.3f}, {ci1['high']:.3f}] (mean: {ci1['mean']:.3f})"
                    )
                    report.append(
                        f"    Run 2 Interval: [{ci2['low']:.3f}, {ci2['high']:.3f}] (mean: {ci2['mean']:.3f})"
                    )

                    overlap_status = (
                        "overlap"
                        if ci_test.get("intervals_overlap", False)
                        else "do not overlap"
                    )
                    significance_symbol = (
                        "✗" if ci_test.get("significant", False) else "✓"
                    )
                    report.append(
                        f"    Intervals {overlap_status} {significance_symbol}"
                    )

                report.append(f"    Result: {ci_test.get('interpretation', 'Unknown')}")
            else:
                report.append(
                    "  Confidence Interval Test: Not performed (insufficient data)"
                )

    def _add_pass_rate_comparison(
        self, report: list[str], comparison: dict[str, Any]
    ) -> None:
        """Add pass rate comparison details to the report."""
        if comparison["pass_rate_comparison"]:
            pass_comp = comparison["pass_rate_comparison"]
            report.append("  Pass Rate Comparison:")
            report.append(
                f"    Run 1: {pass_comp['run1']['pass_rate'] * 100:.1f}% "
                f"({pass_comp['run1']['pass_count']}/{pass_comp['run1']['total']})"
            )
            report.append(
                f"    Run 2: {pass_comp['run2']['pass_rate'] * 100:.1f}% "
                f"({pass_comp['run2']['pass_count']}/{pass_comp['run2']['total']})"
            )
            # Check if difference is already in percentage form or decimal form
            diff_val = pass_comp["difference"]
            if abs(diff_val) > 1.0:
                report.append(f"    Difference: {diff_val:+.1f}%")
            else:
                report.append(f"    Difference: {diff_val * 100:+.1f}%")

            # Handle infinite relative change
            rel_change = pass_comp["relative_change"]
            if rel_change == float("inf"):
                report.append("    Relative Change: +∞% (baseline was 0)")
            elif rel_change == float("-inf"):
                report.append("    Relative Change: -∞% (baseline was 0)")
            else:
                report.append(f"    Relative Change: {rel_change:+.1f}%")

            self._add_statistical_tests(report, pass_comp.get("tests", {}), "    ")

    def _add_statistical_tests(
        self, report: list[str], tests: dict[str, Any], indent: str
    ) -> None:
        """Add statistical test results to the report."""
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict) and "p_value" in test_result:
                significance = "✓" if test_result["significant"] else "✗"
                report.append(
                    f"{indent}{test_name}: p = {test_result['p_value']:.4f} {significance}"
                )

    def _add_significant_differences_summary(
        self, report: list[str], metrics: dict[str, Any]
    ) -> None:
        """Add summary of significant differences to the report."""
        report.append("\n" + "=" * 50)
        report.append("SUMMARY OF SIGNIFICANT DIFFERENCES")
        report.append("=" * 50)

        significant_metrics = [
            metric
            for metric, comparison in metrics.items()
            if comparison["statistical_significance"]["overall_significant"]
        ]

        if significant_metrics:
            report.append(
                f"\nStatistically significant differences found in "
                f"{len(significant_metrics)} metrics:"
            )
            for metric in significant_metrics:
                report.append(f"  - {metric}")
        else:
            report.append(
                "\nNo statistically significant differences found between the two evaluation runs."
            )


def main() -> int:
    """Main CLI entry point for evaluation comparison."""
    args = _parse_arguments()

    # Validate arguments
    validation_error = _validate_arguments(args)
    if validation_error:
        return validation_error

    try:
        return _run_comparison(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in summary file: {e}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two evaluation runs and determine statistical significance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two evaluation summary files
  python3 compare_evaluations.py run1_summary.json run2_summary.json
  
  # Compare with custom significance level
  python3 compare_evaluations.py run1_summary.json run2_summary.json --alpha 0.01
  
  # Save comparison results to JSON file
  python3 compare_evaluations.py run1_summary.json run2_summary.json --output comparison_results.json
  
  # Generate only the report (no JSON output)
  python3 compare_evaluations.py run1_summary.json run2_summary.json --report-only
        """,
    )

    parser.add_argument(
        "summary_files",
        type=str,
        nargs="+",
        help="List of evaluation summary JSON files to compare (exactly 2 files required)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Path to save comparison results as JSON file"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate and display the report, don't save JSON results",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output except for errors"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> int:
    """Validate command line arguments."""
    if args.alpha <= 0 or args.alpha >= 1:
        print("Error: Alpha must be between 0 and 1", file=sys.stderr)
        return 1

    if len(args.summary_files) != 2:
        print(
            f"Error: Exactly 2 summary files are required, but {len(args.summary_files)} were provided",
            file=sys.stderr,
        )
        print(
            "Please provide exactly 2 evaluation summary JSON files for comparison",
            file=sys.stderr,
        )
        return 1

    for i, summary_file in enumerate(args.summary_files, 1):
        if not Path(summary_file).exists():
            print(f"Error: Summary file {i} not found: {summary_file}", file=sys.stderr)
            return 1

    return 0


def _run_comparison(args: argparse.Namespace) -> int:
    """Run the evaluation comparison."""
    comparison = EvaluationComparison(alpha=args.alpha)

    if args.verbose:
        _print_verbose_info(args)

    if not args.quiet:
        print("Performing statistical comparison...")

    comparison_results = comparison.compare_evaluations(
        args.summary_files[0], args.summary_files[1]
    )

    if not args.quiet:
        report = comparison.generate_report(comparison_results)
        print(report)

    if args.output and not args.report_only:
        _save_results_to_file(args.output, comparison_results, args.quiet)

    return _determine_exit_code(comparison_results, args.quiet)


def _print_verbose_info(args: argparse.Namespace) -> None:
    """Print verbose information about the comparison."""
    print("Comparing evaluation runs:")
    print(f"  Run 1: {args.summary_files[0]}")
    print(f"  Run 2: {args.summary_files[1]}")
    print(f"  Significance level: α = {args.alpha}")
    print()


def _save_results_to_file(
    output_path: str, comparison_results: dict, quiet: bool
) -> None:
    """Save comparison results to a JSON file."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"\nComparison results saved to: {output_path_obj}")


def _determine_exit_code(comparison_results: dict, quiet: bool) -> int:
    """Determine the exit code based on comparison results."""
    significant_metrics = [
        metric
        for metric, comparison_data in comparison_results.get(
            "metric_comparisons", {}
        ).items()
        if comparison_data.get("statistical_significance", {}).get(
            "overall_significant", False
        )
    ]

    if significant_metrics:
        if not quiet:
            print(
                f"\nFound statistically significant differences in "
                f"{len(significant_metrics)} metrics."
            )
        return 0  # Success, but with significant differences

    if not quiet:
        print("\nNo statistically significant differences found.")
    return 0  # Success, no significant differences


if __name__ == "__main__":
    sys.exit(main())

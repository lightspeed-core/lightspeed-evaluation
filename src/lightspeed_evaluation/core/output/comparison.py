"""Tool for comparing two evaluation runs and determining statistical significance."""

import json
import logging
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
            "metric_comparisons": self._compare_metrics(results1, results2),
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

        comparison["pass_rate"] = {
            "run1": pass_rate1,
            "run2": pass_rate2,
            "difference": abs(pass_rate2 - pass_rate1),
            "relative_change": (
                (abs(pass_rate2 - pass_rate1) / pass_rate1 * 100)
                if pass_rate1 > 0
                else 0
            ),
        }

        # Compare fail rates
        fail_rate1 = overall1.get("fail_rate", 0)
        fail_rate2 = overall2.get("fail_rate", 0)

        comparison["fail_rate"] = {
            "run1": fail_rate1,
            "run2": fail_rate2,
            "difference": abs(fail_rate2 - fail_rate1),
            "relative_change": (
                (abs(fail_rate2 - fail_rate1) / fail_rate1 * 100)
                if fail_rate1 > 0
                else 0
            ),
        }

        # Compare error rates
        error_rate1 = overall1.get("error_rate", 0)
        error_rate2 = overall2.get("error_rate", 0)

        comparison["error_rate"] = {
            "run1": error_rate1,
            "run2": error_rate2,
            "difference": abs(error_rate2 - error_rate1),
            "relative_change": (
                (abs(error_rate2 - error_rate1) / error_rate1 * 100)
                if error_rate1 > 0
                else 0
            ),
        }

        return comparison

    def _compare_metrics(
        self, results1: list[dict[str, Any]], results2: list[dict[str, Any]]
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

            comparisons[metric] = self._compare_single_metric(
                metric, metric_results1, metric_results2
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
            comparison["score_comparison"], comparison["pass_rate_comparison"]
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
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Statistical test failed: %s", e)
            comparison["tests"]["error"] = str(e)

        return comparison

    def _compare_pass_rates(
        self, pass_count1: int, total1: int, pass_count2: int, total2: int
    ) -> dict[str, Any]:
        """Compare pass rates using statistical tests."""
        pass_rate1 = pass_count1 / total1 if total1 > 0 else 0
        pass_rate2 = pass_count2 / total2 if total2 > 0 else 0

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
            "difference": pass_rate2 - pass_rate1,
            "relative_change": (
                ((pass_rate2 - pass_rate1) / pass_rate1 * 100) if pass_rate1 > 0 else 0
            ),
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
        except Exception as e:  # pylint: disable=broad-exception-caught
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
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Store the error and let the higher level handle it
            comparison["tests"]["chi_square"] = {
                "error": f"Chi-square test failed: {str(e)}"
            }
            # Re-raise so it can be caught at a higher level
            raise e

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
    ) -> dict[str, Any]:
        """Determine overall statistical significance based on available tests."""
        significance = {
            "overall_significant": False,
            "score_significant": False,
            "pass_rate_significant": False,
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

        # Determine overall significance
        significance["overall_significant"] = (
            significance["score_significant"] or significance["pass_rate_significant"]
        )

        # Generate summary
        if significance["overall_significant"]:
            parts = []
            if significance["score_significant"]:
                parts.append("score distributions")
            if significance["pass_rate_significant"]:
                parts.append("pass rates")
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
                report.append(f"  Run 1: {data['run1'] * 100:.2f}%")
                report.append(f"  Run 2: {data['run2'] * 100:.2f}%")
                report.append(f"  Difference: {data['difference'] * 100:+.2f}%")
                report.append(f"  Relative Change: {data['relative_change']:+.1f}%")

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
            report.append(f"    Relative Change: {score_comp['relative_change']:+.1f}%")

            self._add_statistical_tests(report, score_comp.get("tests", {}), "    ")

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
            report.append(f"    Difference: {pass_comp['difference'] * 100:+.1f}%")
            report.append(f"    Relative Change: {pass_comp['relative_change']:+.1f}%")

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

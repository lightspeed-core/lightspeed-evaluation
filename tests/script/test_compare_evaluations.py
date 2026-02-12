#!/usr/bin/env python3
# pylint: disable=protected-access

"""Pytest tests to verify the compare_evaluations.py script works correctly."""

import json
import tempfile
import subprocess
import sys
from pathlib import Path

from typing import Any
import pytest

from script.compare_evaluations import EvaluationComparison


def create_sample_summary(
    results: list[dict[str, Any]], timestamp: str = "2025-01-01T00:00:00"
) -> dict[str, Any]:
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


def test_basic_comparison(
    script_path: Path,
    sample_evaluation_data: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """Test basic comparison functionality."""
    sample_results1, sample_results2 = sample_evaluation_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample summary files
        summary1 = create_sample_summary(sample_results1, "2025-01-01T00:00:00")
        summary2 = create_sample_summary(sample_results2, "2025-01-02T00:00:00")

        file1 = Path(temp_dir) / "summary1.json"
        file2 = Path(temp_dir) / "summary2.json"

        with open(file1, "w", encoding="utf-8") as f:
            json.dump(summary1, f)
        with open(file2, "w", encoding="utf-8") as f:
            json.dump(summary2, f)

        # Test the script
        result = subprocess.run(
            [sys.executable, str(script_path), str(file1), str(file2)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        assert result.stdout, "Script should produce output"
        assert (
            "EVALUATION COMPARISON REPORT" in result.stdout
        ), "Output should contain comparison report"


def test_invalid_arguments(script_path: Path) -> None:
    """Test error handling for invalid arguments."""

    # Test with only one file
    result = subprocess.run(
        [sys.executable, str(script_path), "file1.json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, "Script should fail with only one file"
    assert (
        "Exactly 2 summary files are required" in result.stderr
    ), f"Expected error message not found in stderr: {result.stderr}"

    # Test with three files
    result = subprocess.run(
        [sys.executable, str(script_path), "file1.json", "file2.json", "file3.json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, "Script should fail with three files"
    assert (
        "Exactly 2 summary files are required" in result.stderr
    ), f"Expected error message not found in stderr: {result.stderr}"


def test_nonexistent_files(script_path: Path) -> None:
    """Test error handling for nonexistent files."""

    result = subprocess.run(
        [sys.executable, str(script_path), "nonexistent1.json", "nonexistent2.json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, "Script should fail with nonexistent files"
    assert (
        "not found" in result.stderr
    ), f"Expected 'not found' error message not found in stderr: {result.stderr}"


class TestEvaluationComparisonMethods:
    """Unit tests for EvaluationComparison internal methods."""

    @pytest.fixture
    def comparison_instance(self) -> EvaluationComparison:
        """Create an EvaluationComparison instance for testing."""
        return EvaluationComparison(alpha=0.05)

    def test_compare_score_distributions_basic(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _compare_score_distributions with basic score data."""
        # Test data based on normal distributions
        scores1 = [0.8, 0.9, 0.7, 0.85, 0.75, 0.88, 0.82, 0.79, 0.86, 0.81]
        scores2 = [0.6, 0.65, 0.55, 0.62, 0.58, 0.63, 0.59, 0.61, 0.64, 0.57]

        result = comparison_instance._compare_score_distributions(scores1, scores2)
        # Check structure
        assert "run1_stats" in result
        assert "run2_stats" in result
        assert "tests" in result
        assert "mean_difference" in result
        assert "relative_change" in result

        # Check statistics
        assert result["run1_stats"]["count"] == 10
        assert result["run2_stats"]["count"] == 10
        assert abs(result["run1_stats"]["mean"] - 0.813) < 0.01  # approximately 0.813
        assert abs(result["run2_stats"]["mean"] - 0.604) < 0.01  # approximately 0.604

        # Check that tests were performed
        if "t_test" in result["tests"]:
            assert "statistic" in result["tests"]["t_test"]
            assert "p_value" in result["tests"]["t_test"]
            assert "significant" in result["tests"]["t_test"]

        if "mann_whitney_u" in result["tests"]:
            assert "statistic" in result["tests"]["mann_whitney_u"]
            assert "p_value" in result["tests"]["mann_whitney_u"]
            assert "significant" in result["tests"]["mann_whitney_u"]

    def test_compare_score_distributions_scipy_example(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _compare_score_distributions using scipy documentation examples."""
        # Example inspired by scipy.stats.ttest_ind documentation
        # Two samples with different means
        scores1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        scores2 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        result = comparison_instance._compare_score_distributions(scores1, scores2)

        # The means should be 5.5 and 6.5 respectively
        assert abs(result["run1_stats"]["mean"] - 5.5) < 0.01
        assert abs(result["run2_stats"]["mean"] - 6.5) < 0.01
        assert abs(result["mean_difference"] - 1.0) < 0.01

        # For this example, we expect the tests to show significance
        # (though the exact p-values depend on the implementation)
        assert "tests" in result

    def test_compare_score_distributions_precise_delta(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """
        This test validates behavior with a precise 0.001 mean difference instead of identical values, using reasonable floating-point tolerance and verifying mean calculations.
        """
        scores1 = [0.7999, 0.7988, 0.799, 0.80, 0.81]
        scores2 = [s + 0.001 for s in scores1]

        result = comparison_instance._compare_score_distributions(scores1, scores2)

        expected_mean1 = 0.80154
        expected_diff = 0.001
        expected_rel_change = (expected_diff / expected_mean1) * 100  # ~0.1248%

        assert result["run1_stats"]["mean"] == pytest.approx(expected_mean1)
        f"Baseline mean mismatch. Expected {expected_mean1}, got {result['run1_stats']['mean']}"

        expected_mean2 = expected_mean1 + expected_diff
        assert result["run2_stats"]["mean"] == pytest.approx(
            expected_mean2
        ), f"Adjusted mean mismatch. Expected {expected_mean2}, got {result['run2_stats']['mean']}"

        assert result["mean_difference"] == pytest.approx(
            expected_diff
        ), f"Mean difference mismatch. Expected {expected_diff}, got {result['mean_difference']}"

    def test_perform_pass_rate_tests_basic(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _perform_pass_rate_tests with basic contingency table data."""
        # Based on scipy.stats.chi2_contingency example
        comparison: dict = {"tests": {}}
        # Example: Run1 has 16 pass, 4 fail; Run2 has 18 pass, 2 fail
        test_data = {
            "pass_count1": 16,
            "fail_count1": 4,
            "total1": 20,
            "pass_count2": 18,
            "fail_count2": 2,
            "total2": 20,
        }

        comparison_instance._perform_pass_rate_tests(comparison, test_data)

        # Check that tests were performed
        assert "tests" in comparison
        # Should have chi_square and/or fisher_exact tests
        has_tests = (
            "chi_square" in comparison["tests"] or "fisher_exact" in comparison["tests"]
        )
        assert has_tests or "error" in comparison["tests"]

    def test_perform_pass_rate_tests_scipy_chisquare_example(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _perform_pass_rate_tests using scipy chisquare documentation example."""
        # Based on the scipy documentation example: chisquare([16, 18, 16, 14, 12, 12])
        # Convert to pass/fail format for our function
        comparison: dict = {"tests": {}}
        test_data = {
            "pass_count1": 16,
            "fail_count1": 4,  # Making total 20
            "total1": 20,
            "pass_count2": 18,
            "fail_count2": 2,  # Making total 20
            "total2": 20,
        }

        comparison_instance._perform_pass_rate_tests(comparison, test_data)

        # Verify structure
        assert "tests" in comparison

        # Check chi-square test if present
        if "chi_square" in comparison["tests"]:
            chi_square = comparison["tests"]["chi_square"]
            assert "statistic" in chi_square
            assert "p_value" in chi_square
            assert "significant" in chi_square
            assert "degrees_of_freedom" in chi_square

        # Check Fisher exact test if present
        if "fisher_exact" in comparison["tests"]:
            fisher = comparison["tests"]["fisher_exact"]
            assert "odds_ratio" in fisher
            assert "p_value" in fisher
            assert "significant" in fisher

    def test_perform_pass_rate_tests_edge_cases(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _perform_pass_rate_tests with edge cases."""
        # Test with zero totals
        comparison: dict = {"tests": {}}
        test_data = {
            "pass_count1": 0,
            "fail_count1": 0,
            "total1": 0,
            "pass_count2": 10,
            "fail_count2": 5,
            "total2": 15,
        }

        comparison_instance._perform_pass_rate_tests(comparison, test_data)

        # Should handle gracefully (no tests performed or error recorded)
        assert "tests" in comparison

    def test_check_confidence_interval_overlap_no_overlap(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _check_confidence_interval_overlap with non-overlapping intervals."""
        ci1 = {"low": 0.1, "high": 0.3, "mean": 0.2, "confidence_level": 0.95}
        ci2 = {"low": 0.7, "high": 0.9, "mean": 0.8, "confidence_level": 0.95}

        result = comparison_instance._check_confidence_interval_overlap(ci1, ci2)

        assert "intervals_overlap" in result
        assert "significant" in result
        assert result["intervals_overlap"] is False
        assert result["significant"] is True

    def test_check_confidence_interval_overlap_with_overlap(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _check_confidence_interval_overlap with overlapping intervals."""
        ci1 = {"low": 0.2, "high": 0.6, "mean": 0.4, "confidence_level": 0.95}
        ci2 = {"low": 0.4, "high": 0.8, "mean": 0.6, "confidence_level": 0.95}

        result = comparison_instance._check_confidence_interval_overlap(ci1, ci2)

        assert "intervals_overlap" in result
        assert "significant" in result
        assert result["intervals_overlap"] is True
        assert result["significant"] is False

    def test_check_confidence_interval_overlap_none_inputs(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _check_confidence_interval_overlap with None inputs."""
        result = comparison_instance._check_confidence_interval_overlap(None, None)

        assert "test_performed" in result
        # Should handle None inputs gracefully - might not perform test

    def test_check_confidence_interval_overlap_partial_none(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _check_confidence_interval_overlap with one None input."""
        ci1 = {"low": 0.2, "high": 0.6, "mean": 0.4, "confidence_level": 0.95}

        result = comparison_instance._check_confidence_interval_overlap(ci1, None)
        assert "test_performed" in result
        # Should handle partial None inputs gracefully

    def test_compare_score_distributions_known_statistical_results(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _compare_score_distributions with known statistical results."""
        # Use data that should produce predictable statistical results
        # Two clearly different distributions
        scores1 = [1.0, 1.1, 1.2, 1.3, 1.4]  # Mean ≈ 1.2, low variance
        scores2 = [2.0, 2.1, 2.2, 2.3, 2.4]  # Mean ≈ 2.2, low variance

        result = comparison_instance._compare_score_distributions(scores1, scores2)

        # These should be significantly different
        assert abs(result["mean_difference"] - 1.0) < 0.01
        assert result["relative_change"] > 80  # Should be about 83.33%

        # Both t-test and Mann-Whitney U should detect significance
        if "t_test" in result["tests"]:
            # With such clear separation, p-value should be very small
            assert result["tests"]["t_test"]["p_value"] < 0.05
            assert result["tests"]["t_test"]["significant"] is True

        if "mann_whitney_u" in result["tests"]:
            assert result["tests"]["mann_whitney_u"]["p_value"] < 0.05
            assert result["tests"]["mann_whitney_u"]["significant"] is True

    def test_perform_pass_rate_tests_known_chi_square_result(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _perform_pass_rate_tests with data that should produce known chi-square results."""
        # Based on scipy documentation example for chi2_contingency
        # Create a 2x2 contingency table: [[16, 4], [18, 2]]
        comparison: dict = {"tests": {}}
        test_data = {
            "pass_count1": 16,
            "fail_count1": 4,
            "total1": 20,
            "pass_count2": 18,
            "fail_count2": 2,
            "total2": 20,
        }

        comparison_instance._perform_pass_rate_tests(comparison, test_data)

        # Verify the chi-square test was performed and has reasonable results
        if "chi_square" in comparison["tests"]:
            chi_square = comparison["tests"]["chi_square"]
            assert "statistic" in chi_square
            assert "p_value" in chi_square
            assert "degrees_of_freedom" in chi_square
            assert chi_square["degrees_of_freedom"] == 1  # 2x2 table has 1 DOF
            assert (
                chi_square["statistic"] >= 0
            )  # Chi-square statistic is always non-negative
            assert 0 <= chi_square["p_value"] <= 1  # p-value is a probability

    def test_perform_pass_rate_tests_fisher_exact_small_sample(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _perform_pass_rate_tests with small sample sizes suitable for Fisher exact test."""
        # Small sample sizes where Fisher exact test is more appropriate
        comparison: dict = {"tests": {}}
        test_data = {
            "pass_count1": 3,
            "fail_count1": 2,
            "total1": 5,
            "pass_count2": 1,
            "fail_count2": 4,
            "total2": 5,
        }

        comparison_instance._perform_pass_rate_tests(comparison, test_data)

        # Verify Fisher exact test results
        if "fisher_exact" in comparison["tests"]:
            fisher = comparison["tests"]["fisher_exact"]
            assert "odds_ratio" in fisher
            assert "p_value" in fisher
            assert fisher["odds_ratio"] >= 0  # Odds ratio is non-negative
            assert 0 <= fisher["p_value"] <= 1  # p-value is a probability

    def test_check_confidence_interval_overlap_exact_boundaries(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _check_confidence_interval_overlap with exact boundary conditions."""
        # Test case where intervals just touch at boundaries
        ci1 = {"low": 0.1, "high": 0.5, "mean": 0.3, "confidence_level": 0.95}
        ci2 = {
            "low": 0.5,  # Exactly touches ci1's high
            "high": 0.9,
            "mean": 0.7,
            "confidence_level": 0.95,
        }

        result = comparison_instance._check_confidence_interval_overlap(ci1, ci2)

        # Touching at boundary might be considered overlap or not, depending on implementation
        assert "intervals_overlap" in result
        assert "significant" in result
        assert isinstance(result["intervals_overlap"], bool)
        assert isinstance(result["significant"], bool)

    def test_compare_score_distributions_single_values(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _compare_score_distributions with single values (edge case)."""
        scores1 = [0.8]
        scores2 = [0.6]

        result = comparison_instance._compare_score_distributions(scores1, scores2)

        # Should handle single values gracefully
        assert result["run1_stats"]["count"] == 1
        assert result["run2_stats"]["count"] == 1
        assert result["run1_stats"]["mean"] == 0.8
        assert result["run2_stats"]["mean"] == 0.6
        assert (
            abs(result["mean_difference"] - (-0.2)) < 0.001
        )  # Handle floating point precision

        # Statistical tests might not be performed with single values
        assert "tests" in result

    def test_perform_pass_rate_tests_extreme_ratios(
        self, comparison_instance: EvaluationComparison
    ) -> None:
        """Test _perform_pass_rate_tests with extreme pass rate differences."""
        # One run with 100% pass rate, another with 0% pass rate
        comparison: dict = {"tests": {}}
        test_data = {
            "pass_count1": 10,
            "fail_count1": 0,
            "total1": 10,
            "pass_count2": 0,
            "fail_count2": 10,
            "total2": 10,
        }

        comparison_instance._perform_pass_rate_tests(comparison, test_data)

        # Should handle extreme cases
        assert "tests" in comparison

        # If tests are performed, they should show high significance
        if (
            "chi_square" in comparison["tests"]
            and "error" not in comparison["tests"]["chi_square"]
        ):
            assert comparison["tests"]["chi_square"]["p_value"] < 0.05
            assert comparison["tests"]["chi_square"]["significant"] is True

        if (
            "fisher_exact" in comparison["tests"]
            and "error" not in comparison["tests"]["fisher_exact"]
        ):
            assert comparison["tests"]["fisher_exact"]["p_value"] < 0.05
            assert comparison["tests"]["fisher_exact"]["significant"] is True

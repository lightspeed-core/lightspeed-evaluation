"""Tests for NxM behavioral statistics."""

from pytest import approx

from lightspeed_evaluation.pipeline.behavioral.statistics import (
    confidence_interval,
    metric_significance,
    pass_at_k,
    significance_tests,
)


class TestPassAtK:
    """Tests for the pass@k unbiased estimator."""

    def test_all_pass(self) -> None:
        """All cases pass in all runs → 1.0."""
        assert pass_at_k([3, 3], [3, 3], k=2) == 1.0

    def test_none_pass(self) -> None:
        """No cases pass → 0.0."""
        assert pass_at_k([0, 0], [3, 3], k=2) == 0.0

    def test_empty_input(self) -> None:
        """Empty input → 0.0."""
        assert pass_at_k([], [], k=2) == 0.0

    def test_single_case_partial(self) -> None:
        """One case, 3 of 5 pass, k=2."""
        result = pass_at_k([3], [5], k=2)
        # 1 - C(2,2)/C(5,2) = 1 - 1/10 = 0.9
        assert abs(result - 0.9) < 1e-9

    def test_k_equals_one(self) -> None:
        """k=1 is equivalent to pass rate averaged across cases."""
        result = pass_at_k([7, 3], [10, 10], k=1)
        # case 1: 7/10 = 0.7, case 2: 3/10 = 0.3, average = 0.5
        assert abs(result - 0.5) < 1e-9

    def test_multiple_cases_averaged(self) -> None:
        """pass@k is averaged across cases."""
        # case 1: 2 of 3 pass, k=2 → 1 - C(1,2)/C(3,2) = 1 - 0 = 1.0
        # case 2: 0 of 3 pass, k=2 → 0.0
        # average = 0.5
        result = pass_at_k([2, 0], [3, 3], k=2)
        assert abs(result - 0.5) < 1e-9

    def test_k_greater_than_total(self) -> None:
        """k > n with some passes → 1.0 (at least one exists)."""
        result = pass_at_k([1], [2], k=3)
        assert result == 1.0

    def test_k_greater_than_total_none_pass(self) -> None:
        """k > n with zero passes → 0.0."""
        result = pass_at_k([0], [2], k=3)
        assert result == 0.0


class TestSignificanceTests:
    """Tests for statistical significance functions."""

    def test_clearly_different_agents(self) -> None:
        """90% vs 12% pass rate across 5 runs: Fisher detects significance."""
        results = significance_tests(
            pass_counts_a=[9, 10, 9, 10, 9],
            totals_a=[10, 10, 10, 10, 10],
            pass_counts_b=[1, 2, 1, 0, 2],
            totals_b=[10, 10, 10, 10, 10],
            alpha=0.05,
        )
        fisher = next((r for r in results if r.test == "fisher_exact"), None)
        assert fisher is not None
        assert fisher.significant is True

    def test_identical_agents_not_significant(self) -> None:
        """Same pass counts: Fisher returns p=1.0, no false positive."""
        results = significance_tests(
            pass_counts_a=[8, 8, 8],
            totals_a=[10, 10, 10],
            pass_counts_b=[8, 8, 8],
            totals_b=[10, 10, 10],
            alpha=0.05,
        )
        fisher = next((r for r in results if r.test == "fisher_exact"), None)
        assert fisher is not None
        assert fisher.significant is False
        assert fisher.p_value == 1.0

    def test_alpha_controls_threshold(self) -> None:
        """Same data: significant at alpha=0.10, not at alpha=0.001."""
        kwargs = {
            "pass_counts_a": [9, 8, 9],
            "totals_a": [10, 10, 10],
            "pass_counts_b": [5, 6, 5],
            "totals_b": [10, 10, 10],
        }
        loose = significance_tests(**kwargs, alpha=0.10)
        strict = significance_tests(**kwargs, alpha=0.001)

        fisher_loose = next((r for r in loose if r.test == "fisher_exact"), None)
        fisher_strict = next((r for r in strict if r.test == "fisher_exact"), None)
        assert fisher_loose is not None and fisher_strict is not None
        assert fisher_loose.significant is True
        assert fisher_strict.significant is False

    def test_single_run_fisher_only(self) -> None:
        """Single run still produces Fisher result."""
        results = significance_tests(
            pass_counts_a=[8],
            totals_a=[10],
            pass_counts_b=[5],
            totals_b=[10],
            alpha=0.05,
        )
        assert len(results) == 1
        assert results[0].test == "fisher_exact"

    def test_empty_data_returns_empty(self) -> None:
        """No data produces no results."""
        assert not significance_tests([], [], [], [], alpha=0.05)

    def test_identical_metric_scores_returns_none(self) -> None:
        """Identical per-metric scores produce no MW result."""
        assert metric_significance([0.9, 0.9, 0.9], [0.9, 0.9, 0.9], "m") is None

    def test_nan_filtered_below_min_returns_none(self) -> None:
        """NaN filtering can drop below minimum samples."""
        assert metric_significance([0.9, float("nan")], [0.8, 0.7], "m") is None


class TestConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_known_values(self) -> None:
        """CI on [80, 85, 90] produces correct bounds."""
        ci = confidence_interval([80.0, 85.0, 90.0])
        assert ci is not None
        low, high = ci
        assert low < 85.0 < high
        assert (low + high) / 2 == approx(85.0)

    def test_single_value_returns_none(self) -> None:
        """CI requires at least 2 values."""
        assert confidence_interval([80.0]) is None

    def test_zero_variance(self) -> None:
        """Identical values produce zero-width CI."""
        ci = confidence_interval([50.0, 50.0, 50.0])
        assert ci is not None
        assert ci[0] == approx(50.0)
        assert ci[1] == approx(50.0)

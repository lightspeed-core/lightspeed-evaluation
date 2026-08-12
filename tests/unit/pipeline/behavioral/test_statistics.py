"""Tests for NxM behavioral statistics."""

from lightspeed_evaluation.pipeline.behavioral.statistics import pass_at_k


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

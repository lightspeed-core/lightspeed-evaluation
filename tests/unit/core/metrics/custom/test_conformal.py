"""Tests for Conformal Risk Control threshold calibration."""

import numpy as np

from lightspeed_evaluation.core.metrics.custom.conformal import (
    compute_mrr_threshold,
    get_lhat,
)


class TestGetLhat:
    """Tests for the vendored get_lhat function."""

    def test_perfect_calibration_scores(self) -> None:
        """All scores are 1.0 — zero risk at any threshold, result within bounds."""
        lambdas = np.linspace(0.0, 1.0, 100)
        sims = np.ones(50)
        loss_table = (sims[:, None] < lambdas[None, :]).astype(float)

        lhat = get_lhat(loss_table, lambdas, alpha=0.1)

        assert 0.0 <= lhat <= 1.0

    def test_low_calibration_scores(self) -> None:
        """All calibration scores are low — threshold should be near those scores."""
        lambdas = np.linspace(0.0, 1.0, 200)
        sims = np.full(50, 0.3)
        loss_table = (sims[:, None] < lambdas[None, :]).astype(float)

        lhat = get_lhat(loss_table, lambdas, alpha=0.1)

        assert lhat <= 0.35

    def test_higher_alpha_yields_stricter_threshold(self) -> None:
        """With higher alpha (more tolerance), threshold can be stricter."""
        rng = np.random.default_rng(42)
        sims = rng.uniform(0.5, 0.9, size=100)
        lambdas = np.linspace(0.0, 1.0, 200)
        loss_table = (sims[:, None] < lambdas[None, :]).astype(float)

        lhat_strict = get_lhat(loss_table, lambdas, alpha=0.05)
        lhat_loose = get_lhat(loss_table, lambdas, alpha=0.3)

        assert lhat_loose >= lhat_strict

    def test_single_calibration_point(self) -> None:
        """Algorithm works with n=1."""
        lambdas = np.linspace(0.0, 1.0, 50)
        sims = np.array([0.7])
        loss_table = (sims[:, None] < lambdas[None, :]).astype(float)

        lhat = get_lhat(loss_table, lambdas, alpha=0.5)

        assert 0.0 <= lhat <= 1.0

    def test_empirical_risk_controlled(self) -> None:
        """Validate that empirical false-negative rate respects alpha on held-out data."""
        rng = np.random.default_rng(123)
        alpha = 0.2
        all_sims = rng.uniform(0.4, 0.95, size=400)
        calib_sims = all_sims[:200]
        test_sims = all_sims[200:]

        lambdas = np.linspace(0.0, 1.0, 300)
        loss_table = (calib_sims[:, None] < lambdas[None, :]).astype(float)
        lhat = get_lhat(loss_table, lambdas, alpha=alpha)

        fn_rate = float(np.mean(test_sims < lhat))
        assert fn_rate <= alpha + 0.05


class TestComputeMrrThreshold:
    """Tests for the MRR-specific wrapper."""

    def test_empty_calibration_returns_none(self) -> None:
        """No calibration data yields None."""
        assert compute_mrr_threshold([]) is None

    def test_returns_float(self) -> None:
        """Result is a plain float."""
        result = compute_mrr_threshold([0.8, 0.9, 0.7, 0.85] * 10)
        assert isinstance(result, float)

    def test_threshold_within_bounds(self) -> None:
        """Threshold is between 0 and 1."""
        result = compute_mrr_threshold([0.6, 0.7, 0.8, 0.9, 0.5] * 10)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_high_sims_yield_high_threshold(self) -> None:
        """Calibration pairs with high similarity yield a high threshold."""
        sims = [0.95, 0.92, 0.98, 0.90, 0.93] * 10
        result = compute_mrr_threshold(sims, alpha=0.1)
        assert result is not None
        assert result >= 0.8

    def test_low_sims_yield_low_threshold(self) -> None:
        """Calibration pairs with low similarity yield a low threshold."""
        sims = [0.3, 0.25, 0.35, 0.28, 0.32] * 10
        result = compute_mrr_threshold(sims, alpha=0.1)
        assert result is not None
        assert result <= 0.4

    def test_alpha_affects_threshold(self) -> None:
        """Lower alpha (stricter) should yield lower or equal threshold."""
        sims = [0.6, 0.7, 0.8, 0.5, 0.65, 0.75] * 10
        strict = compute_mrr_threshold(sims, alpha=0.05)
        loose = compute_mrr_threshold(sims, alpha=0.3)
        assert strict is not None
        assert loose is not None
        assert loose >= strict

    def test_small_n_returns_permissive_threshold(self) -> None:
        """With few samples, CRC returns permissive threshold (finite-sample correction)."""
        result = compute_mrr_threshold([0.9, 0.8], alpha=0.1)
        assert result is not None
        assert result <= 0.5

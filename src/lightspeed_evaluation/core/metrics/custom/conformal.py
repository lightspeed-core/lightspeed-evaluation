"""Conformal Risk Control for semantic similarity threshold calibration.

Vendored from https://github.com/aangelopoulos/conformal-risk
MIT License — Angelopoulos, Bates, Fisch, Lei, Schuster (ICLR 2024)

Reference: "Conformal Risk Control", ICLR 2024
    https://arxiv.org/abs/2208.02814
"""

from typing import Optional

import numpy as np


def get_lhat(
    calib_loss_table: np.ndarray,
    lambdas: np.ndarray,
    alpha: float,
    upper_bound: float = 1,
) -> float:
    """Select the threshold that controls marginal risk for a monotone loss.

    Core algorithm from the conformal-risk package.  The calibration loss
    table rows correspond to calibration examples and columns to candidate
    thresholds (lambdas).  Lambdas must be ordered so that the loss is
    non-decreasing across columns (i.e. from permissive to strict).

    Args:
        calib_loss_table: Loss matrix of shape ``(n_calibration, n_lambdas)``.
        lambdas: Candidate thresholds, ordered permissive-to-strict.
        alpha: Target risk level.
        upper_bound: Upper bound on the per-example loss (B in the paper).

    Returns:
        The selected threshold ``lambda_hat`` satisfying
        ``E[L_{n+1}(lambda_hat)] <= alpha``.
    """
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    raw_idx = int(np.argmax(((n / (n + 1)) * rhat + upper_bound / (n + 1)) >= alpha))
    lhat_idx = max(raw_idx - 1, 0)
    return float(lambdas[lhat_idx])


def compute_mrr_threshold(
    calibration_similarities: list[float],
    alpha: float = 0.1,
    n_lambdas: int = 100,
) -> Optional[float]:
    """Compute a similarity threshold for MRR context matching.

    Adapts :func:`get_lhat` to the semantic-matching case where the loss
    is the false-negative indicator: ``L_i(tau) = 1{sim_i < tau}``.

    Lambdas (candidate thresholds) run from 0.0 (most permissive) to 1.0
    (strictest), so the loss is non-decreasing across the lambda grid.
    ``get_lhat`` finds the largest threshold whose adjusted risk stays
    below *alpha*.

    Args:
        calibration_similarities: Cosine similarities of known-positive
            (matching) text pairs used for calibration.
        alpha: Target false-negative rate bound.  Default 0.1 means that
            at most 10 % of true matches will be missed.
        n_lambdas: Resolution of the threshold grid.

    Returns:
        The calibrated similarity threshold, or ``None`` when calibration
        data is empty.
    """
    if not calibration_similarities:
        return None

    sims = np.asarray(calibration_similarities, dtype=float)
    lambdas = np.linspace(0.0, 1.0, n_lambdas)
    calib_loss_table = (sims[:, None] < lambdas[None, :]).astype(float)
    return get_lhat(calib_loss_table, lambdas, alpha)

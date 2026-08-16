"""Statistical functions for NxM behavioral evaluation."""

import math


def pass_at_k(pass_counts: list[int], totals: list[int], k: int) -> float:
    """Unbiased estimator of pass@k, averaged across cases.

    For each evaluation case (conversation × turn × metric), estimates
    the probability that at least one of k randomly chosen runs passes.

    Args:
        pass_counts: Number of passing runs per case.
        totals: Total runs per case.
        k: Number of samples drawn (typically repeat count).

    Returns:
        Average pass@k across all cases, in [0.0, 1.0].
    """
    if not pass_counts or k < 1:
        return 0.0

    estimates: list[float] = []
    for n, c in zip(totals, pass_counts):
        effective_k = min(k, n)
        if c == 0 or effective_k < 1:
            estimates.append(0.0)
            continue
        estimates.append(
            1.0 - math.comb(n - c, effective_k) / math.comb(n, effective_k)
        )

    return sum(estimates) / len(estimates) if estimates else 0.0

"""Statistical functions for NxM behavioral evaluation."""

import math
import statistics as stdlib_stats
from typing import Optional

from scipy.stats import fisher_exact, mannwhitneyu, sem
from scipy.stats import t as t_dist

from lightspeed_evaluation.pipeline.behavioral.models import SignificanceResult

_MIN_SAMPLES = 2


def significance_tests(
    pass_counts_a: list[int],
    totals_a: list[int],
    pass_counts_b: list[int],
    totals_b: list[int],
    alpha: float = 0.05,
) -> list[SignificanceResult]:
    """Run Fisher's exact test on aggregated pass/fail counts.

    Args:
        pass_counts_a: Per-run pass counts for agent A.
        totals_a: Per-run total counts for agent A.
        pass_counts_b: Per-run pass counts for agent B.
        totals_b: Per-run total counts for agent B.
        alpha: Significance level.

    Returns:
        List of test results (may be empty if insufficient data).
    """
    results: list[SignificanceResult] = []

    fisher = _fisher_exact(pass_counts_a, totals_a, pass_counts_b, totals_b, alpha)
    if fisher is not None:
        results.append(fisher)

    return results


def _fisher_exact(
    pass_counts_a: list[int],
    totals_a: list[int],
    pass_counts_b: list[int],
    totals_b: list[int],
    alpha: float,
) -> Optional[SignificanceResult]:
    """Fisher's exact test on aggregated pass/fail contingency table."""
    total_pass_a = sum(pass_counts_a)
    total_n_a = sum(totals_a)
    total_pass_b = sum(pass_counts_b)
    total_n_b = sum(totals_b)

    if total_n_a == 0 or total_n_b == 0:
        return None
    if total_pass_a > total_n_a or total_pass_b > total_n_b:
        return None

    table = [
        [total_pass_a, total_n_a - total_pass_a],
        [total_pass_b, total_n_b - total_pass_b],
    ]
    result = fisher_exact(table)
    odds_ratio = float(result.statistic)
    p_val = float(result.pvalue)
    return SignificanceResult(
        test="fisher_exact",
        statistic=odds_ratio,
        p_value=p_val,
        significant=p_val < alpha,
    )


def metric_significance(
    scores_a: list[float],
    scores_b: list[float],
    metric: str,
    alpha: float = 0.05,
) -> Optional[SignificanceResult]:
    """Mann-Whitney U test on per-run metric scores between two agents.

    Args:
        scores_a: Per-run mean scores for agent A.
        scores_b: Per-run mean scores for agent B.
        metric: Metric identifier.
        alpha: Significance level.

    Returns:
        SignificanceResult or None if insufficient data.
    """
    scores_a = [s for s in scores_a if math.isfinite(s)]
    scores_b = [s for s in scores_b if math.isfinite(s)]
    if len(scores_a) < _MIN_SAMPLES or len(scores_b) < _MIN_SAMPLES:
        return None
    if scores_a == scores_b:
        return None
    try:
        result = mannwhitneyu(scores_a, scores_b, alternative="two-sided")
    except ValueError:
        return None

    u_stat = float(result.statistic)
    p_val = float(result.pvalue)
    return SignificanceResult(
        test="mann_whitney_u",
        statistic=u_stat,
        p_value=p_val,
        significant=p_val < alpha,
        metric=metric,
    )


def confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> Optional[tuple[float, float]]:
    """Compute confidence interval for the mean using t-distribution.

    Args:
        values: Sample values.
        confidence: Confidence level (default 0.95).

    Returns:
        (ci_low, ci_high) tuple, or None if fewer than 2 values.
    """
    values = [v for v in values if math.isfinite(v)]
    if len(values) < 2:
        return None
    mean = stdlib_stats.mean(values)
    se = float(sem(values))
    if se == 0.0:
        return (mean, mean)
    ci = t_dist.interval(confidence, df=len(values) - 1, loc=mean, scale=se)
    return (float(ci[0]), float(ci[1]))


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

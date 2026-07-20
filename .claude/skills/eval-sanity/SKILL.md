---
name: eval-sanity
description: Use after an evaluation run completes, before sharing or aggregating results. Quick statistical health check on aggregated output files for silent data issues - high error rates masked by exclusion, anomalous score drops, insufficient sample sizes. Does NOT read raw evaluation data.
---

# Eval Sanity Check

Quick statistical health check on evaluation results using aggregated stats only. Catches issues that quality_score silently hides (errors are excluded from the score, not penalized).

**Does NOT read raw data or the `results` array in summary.json.**

## Invocation

Expects an output directory path as argument (e.g., `/eval-sanity /path/to/output`). If not provided, ask for it. If the directory contains subdirectories with separate product results, detect and include them for cross-product comparison.

If the directory contains multiple runs, match files by shared timestamp prefix. If mismatched or ambiguous, list available runs and ask which to analyze.

If either summary or quality report file is missing, mark related checks as SKIP in the output and note which files were found.

## Files

Only these — no `*_detailed.csv`, no raw responses:
- `*_summary.json` — read only the aggregated statistics sections (overall, by_metric, by_conversation, by_tag). **Do NOT read the `results` array** — it contains raw evaluation data.
- `*_quality_report.json` — quality score, per-metric aggregates, and warnings

## Checks

Default thresholds below. User can override per invocation (e.g., "flag errors above 25%").

**1. Error Rate** — Per metric, compare error count vs total. Flag if **> 15%**.

**2. Sample Coverage** — Compare scored count per metric in quality report against total evaluations. Flag if scored count **< 70%** of total.

**3. Score Distribution** — Per metric score statistics (skip if fewer than 3 scored samples): flag if `mean` < 0.3, flag if `std` > 0.4 AND `mean` < 0.5, flag if min equals max (zero variance at any value).

**4. Cross-Metric Consistency** — Compare error rates across metrics. A metric with 3x the error rate of others likely has a config issue.

**5. Quality Report Warnings** — Surface any warnings from the quality report (e.g., metrics excluded from quality score due to zero results or missing data).

**6. Cross-Product Comparison** (when multiple product subdirectories found) — Compare quality scores, error rates, and sample coverage across products. Flag products with significantly worse metrics than others on the same model.

## Output

```
## Eval Sanity Check

**Run:** <timestamp or directory>
**Total evaluations:** N

| Check | Status | Detail | Recommendation |
|-------|--------|--------|----------------|
| Error rate | WARN/OK/SKIP | <detail> | <action> |
| Sample coverage | WARN/OK | <detail> | <action> |
| Score distribution | WARN/OK | <detail> | <action> |
| Cross-metric | WARN/OK | <detail> | <action> |
| Quality warnings | WARN/OK | <detail> | <action> |
| Cross-product | WARN/OK | <detail> | <action> |
```

If all checks pass, say so in one line.

If any WARN is found: "For deeper analysis, run `/eval-investigate`. Note: this will read actual response and evaluation data."

Print the report in the conversation and save to the output directory.

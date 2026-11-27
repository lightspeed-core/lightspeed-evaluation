# UIEINTEL-698 — approach B: Ragas-style metric library

Spike branch. One evaluation criterion (`coverage`: three metrics, two
indicators) integrated into this framework from a Ragas-style metric library.

The sibling branch `uieintel-698-client-side` integrates the *same three metrics*,
with textually identical prompts and formulas, from a library of the other
architecture. The diff of each branch against `b9004d8` is that approach's
full integration cost, which is the measurement the spike exists to make.

**Not proposed for upstream.** See "Upstreamable parts" below for the two
commits that are.

## What this branch adds

The metric library owns execution. This branch builds a judge client and an
embedding client, injects them into the metric constructors, and adapts the
results — the same division of labour as `RagasMetrics`.

| Path | Purpose |
|---|---|
| `core/metrics/uaem/common.py` | Trace mapping, structured judge calls, pass-rule rendering |
| `core/metrics/uaem/metrics.py` | The `uaem_lib` handler |
| `core/metrics/uaem/judge.py` | `JudgeLLM` / `EmbeddingClient` adapters over the managed clients |
| `core/metrics/uaem/rollup.py` | Metric → indicator → criterion → domain rollup, attached to `summary.json` |
| `config/uaem_poc_*.yaml` | Spike config and dataset |

Plus registration: `pipeline/evaluation/evaluator.py` (handler table and
panel-judge factory) and `core/system/validator.py` (`METRIC_REQUIREMENTS`).

`UAEMLibMetrics.__init__` calls `build_metrics(llm=UAEMJudgeLLM(...),
embeddings=UAEMEmbeddingClient(...))`, and `evaluate()` then just calls
`metric.score(traces=...)`. The adapter in `judge.py` is what keeps token
tracking, the disk cache, SSL policy and retry config in the framework; a
consumer that skipped it would silently lose all four.

**Clients are bound at construction**, so a judge panel needs one full metric
set per judge. `_create_uaem_handler` is called per judge manager to do that,
exactly as `RagasMetrics` is today.

## Running it

```bash
uv sync --extra uaem-poc --group dev
export OPENAI_API_KEY=...

uv run lightspeed-eval \
    --system-config config/uaem_poc_system.yaml \
    --eval-data config/uaem_poc_eval_data.yaml \
    --output-dir results
```

The run's native scores ride on its results and are rolled up — metric →
indicator → evaluation criterion → domain — straight into a `rollup` key on
that run's `*_summary.json`, next to `summary_stats` and `results`. The key
is nested `agent → evaluation level → tree`. A run that never used
`uaem_lib` gets no `rollup` key at all.

`uv sync` without `--extra uaem-poc` leaves the framework unregistered; the
handler imports `lib_side` lazily, the rollup section is silently omitted,
and the UAEM tests skip.

## Two findings the code encodes

**The score contract is `[0, 1]`, and one of these metrics is not.**
`granularity_fit_score_model` is `[-1, 1]` with a `BETWEEN(0, 0)` pass
condition. `MetricResult.score` is `ge=0, le=1` and pass/fail is
`score >= threshold`, so the library reports `1 - |x|` and the config uses a
threshold of `1.00`. Semantically equivalent, but only because of that
transform — which is why the config comments it. `raw_score` /
`score_min` / `score_max` / `pass_condition` give the native value and its
rule an honest home, and `ScoredMetric` is how a handler populates them: a
handler may return it in place of the `(score, reason)` 2-tuple, which the
other frameworks still use unchanged.

**Nothing travels out of band.** The native scores the rollup needs reach
finalization on the results themselves, not through a JSONL sink and
environment variables. That makes a run's rollup a pure function of that
run's results: concurrent runs cannot mix, and no file has to be located,
scoped or cleaned up.

**Aggregation order is load-bearing.** `1 - |x|` is non-linear, so
mean-then-formula and formula-then-mean disagree: granularity scores of `-1`
and `+1` give `1.0` one way and `0.0` the other. The rollup therefore reads
`raw_score`, not the already-transformed `score`, and means across results
before applying any formula. Pinned by
`tests/unit/core/metrics/test_uaem_rollup.py`.

The same split decides what a judge panel feeds the rollup, per strategy:
`average` means the native scores (so `-1` and `+1` give the ideal `0`),
`max` takes the native score of whichever judge's reported score won, and
`majority_vote` reports no native score at all — a vote is not a value on
the native scale. Pinned by `TestAggregateNativeScore` in
`tests/unit/pipeline/evaluation/test_judges.py`.

**The tree is grouped by what was scored.** `rollup` is keyed by the agent
under evaluation (`EvaluationResult.agent`, `unknown` for an offline run
that names none) and then by evaluation level, since turn and conversation
means are over different populations.

## Upstreamable parts

Two commits on this branch are generic and stand on their own, independent
of the spike. They are identical on both branches:

- `feat(models): add raw_score and native score range to MetricResult` —
  `score` stays normalised to `[0, 1]`, which statistics, `quality_score` and
  the charts all rely on; metrics with another native scale get somewhere
  honest to put the real value. `ScoredMetric` widens the handler contract
  additively to fill them: a handler may return it instead of
  `tuple[Optional[float], str]`, and every handler that does not is
  untouched.
- `feat(llm): add optional system_prompt to BaseCustomLLM.call` — the call
  previously sent one user message with no way to set a system role.

Everything else is spike-specific and stays on the branch.

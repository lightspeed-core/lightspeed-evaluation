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
| `core/metrics/uaem/common.py` | Trace mapping, the `[0, 1]` score contract, the raw-score sink |
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

The run's raw scores go to a sink file scoped to that run (see "Rollup is
per-run" below) and are rolled up — metric → indicator → evaluation
criterion → domain — straight into a `rollup` key on that run's
`*_summary.json`, next to `summary_stats` and `results`. A run that never
used `uaem_lib` gets no `rollup` key at all.

`uv sync` without `--extra uaem-poc` leaves the framework unregistered; the
handler imports `lib_side` lazily, the rollup section is silently omitted,
and the UAEM tests skip.

## Two findings the code encodes

**The score contract is `[0, 1]`, and one of these metrics is not.**
`granularity_fit_score_model` is `[-1, 1]` with a `BETWEEN(0, 0)` pass
condition. `MetricResult.score` is `ge=0, le=1` and pass/fail is
`score >= threshold`, so the handler reports `1 - |x|` and the config uses a
threshold of `1.00`. Semantically equivalent, but only because of that
transform — which is why the config comments it. The new
`raw_score` / `score_min` / `score_max` fields give the native value an
honest home, though the 2-tuple handler contract cannot populate them yet
(see below).

**Aggregation order is load-bearing.** `1 - |x|` is non-linear, so
mean-then-formula and formula-then-mean disagree: granularity rows of `-1`
and `+1` give `1.0` one way and `0.0` the other. The rollup therefore reads
raw scores from the sink, not the already-transformed scores in the results
CSV, and means across rows before applying any formula. Pinned by
`tests/unit/core/metrics/test_uaem_rollup.py`.

**Rollup is per-run, not accumulating.** The raw-score sink has no run id in
its schema, so a fixed filename would silently mix scores from every past
run into the next one's rollup. `resolve_raw_score_path()` pins a
timestamped default into `UAEM_RAW_SCORES` the first time a run needs it, so
the sink — and the rollup read from it — is scoped to that run. Set
`UAEM_RAW_SCORES` explicitly to opt back into one shared, accumulating file
(e.g. for ad-hoc multi-run analysis via `build_rollup_summary` directly).

## Upstreamable parts

Two commits on this branch are generic and stand on their own, independent
of the spike. They are identical on both branches:

- `feat(models): add raw_score and native score range to MetricResult` —
  `score` stays normalised to `[0, 1]`, which statistics, `quality_score` and
  the charts all rely on; metrics with another native scale get somewhere
  honest to put the real value. Caveat: the handler contract is
  `evaluate(...) -> tuple[Optional[float], str]` and `metric_metadata` is
  filled from config rather than from the handler, so there is no in-band
  channel and the column currently ships empty. Widening the contract to
  return a result object is the real fix and is a breaking change.
- `feat(llm): add optional system_prompt to BaseCustomLLM.call` — the call
  previously sent one user message with no way to set a system role.

Everything else is spike-specific and stays on the branch.

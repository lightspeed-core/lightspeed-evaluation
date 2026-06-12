# Keywords Evaluation

Case-insensitive keyword matching with alternative sets.

## Run Example

```bash
# From project root
uv run lightspeed-eval \
  --system-config examples/02_metrics/keywords_evaluation/system.yaml \
  --eval-data examples/02_metrics/keywords_evaluation/eval_data.yaml
```

## Metrics & Required Data

**Current Config:** API Disabled (`api.enabled: false`) - all data in eval_data.yaml

| Metric                  | Name              | Description                                                                 | Required Data (API Disabled) | Required Data (API Enabled) |
|-------------------------|-------------------|-----------------------------------------------------------------------------|------------------------------|-----------------------------|
| `custom:keywords_eval`  | Keywords Match    | Binary (0/1) check: ALL keywords must match (AND), supports alternative sets (OR) | response, expected_keywords | query, expected_keywords |

**Keyword Format:**
- Single set: `[["keyword1", "keyword2"]]` - ALL must match
- Alternative sets: `[["set1_kw1", "set1_kw2"], ["set2_kw1", "set2_kw2"]]` - ANY set can match (within set, ALL must match)
- Case-insensitive matching

**Note:** With API enabled, `response` is fetched from live API using `query`. Query is  not used for evaluation.

Results written to: `examples/02_metrics/keywords_evaluation/eval_output/`

# NLP Metrics - String-Based Comparison

Non-LLM text comparison using traditional NLP metrics.

## Prerequisites

NLP metrics require additional dependencies:

```bash
# Install NLP metrics dependencies
uv sync --group nlp-metrics
```

## Run Example

```bash
# From project root
uv run lightspeed-eval \
  --system-config examples/02_metrics/nlp_metrics/system.yaml \
  --eval-data examples/02_metrics/nlp_metrics/eval_data.yaml
```

## Metrics & Required Data

**Current Config:** API Disabled (`api.enabled: false`) - all data in eval_data.yaml

| Metric                              | Name                   | Description                                                                          | Required Data (API Disabled)      | Required Data (API Enabled) |
|-------------------------------------|------------------------|--------------------------------------------------------------------------------------|-----------------------------------|-----------------------------|
| `nlp:bleu`                          | BLEU Score             | N-gram overlap between response and reference (default: BLEU-4)                      | response, expected_response | query, expected_response    |
| `nlp:rouge`                         | ROUGE Score            | Recall-oriented n-gram overlap (default: ROUGE-L)                                    | response, expected_response | query, expected_response    |
| `nlp:semantic_similarity_distance`  | String Similarity      | Character-level string distance (default: Levenshtein)                               | response, expected_response | query, expected_response    |

**Note:** With API enabled, `response` is fetched from live API using `query`. Query is  not used for evaluation.

## Configuration Options

**BLEU N-gram Size:**
```yaml
nlp:bleu:
  max_ngram: 4  # 1-4, default 4 for BLEU-4
```

**ROUGE Type:**
```yaml
nlp:rouge:
  rouge_type: rougeL  # rougeL, rouge1, rouge2
```

**Distance Measure:**
```yaml
nlp:semantic_similarity_distance:
  distance_measure: levenshtein  # levenshtein, jaro, jaro_winkler
```

## Important Notes

⚠️ **Semantic Similarity Distance** uses string distance, NOT semantic meaning. Responses with correct meaning but different wording will score low. For semantic comparison, use `custom:answer_correctness`.

**Best Used For:**
- Template-based responses with expected phrasing
- Code generation (exact syntax matching)
- Quick baseline without LLM costs

**Not Recommended For:**
- Free-form conversational responses
- Semantic correctness checking

Results written to: `examples/02_metrics/nlp_metrics/eval_output/`

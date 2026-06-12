# Context Quality (RAG Evaluation)

Evaluate retrieval quality and context relevance for RAG systems.

## Run Example

```bash
# From project root
export OPENAI_API_KEY="your-key"
uv run lightspeed-eval \
  --system-config examples/02_metrics/context_quality/system.yaml \
  --eval-data examples/02_metrics/context_quality/eval_data.yaml
```

## Metrics & Required Data

**Current Config:** API Disabled (`api.enabled: false`) - all data in eval_data.yaml

| Metric                                        | Name                | Description                                                                           | Required Data (API Disabled)               | Required Data (API Enabled) |
|-----------------------------------------------|---------------------|---------------------------------------------------------------------------------------|--------------------------------------------|-----------------------------|
| `ragas:context_recall`                        | Context Recall      | Measures how well retrieved context covers the expected answer                        | query, response, contexts, expected_response | query, expected_response  |
| `ragas:context_relevance`                     | Context Relevance   | Evaluates relevance of retrieved context to the query                                 | query, response, contexts                  | query                       |
| `ragas:context_precision_without_reference`   | Context Utilization | Assesses context ranking quality without using reference answer                       | query, response, contexts                  | query                       |
| `ragas:context_precision_with_reference`      | Context Precision   | Assesses context ranking quality using expected answer                                | query, response, contexts, expected_response | query, expected_response  |

**Note:** With API enabled, `response` and `contexts` are fetched from live API; only `query` and expected fields go in YAML.

Results written to: `examples/02_metrics/context_quality/eval_output/`

# Response Quality Metrics

Comprehensive 4-dimensional response evaluation.

## Run Example

```bash
# From project root
export OPENAI_API_KEY="your-key"
uv run lightspeed-eval \
  --system-config examples/02_metrics/response_quality/system.yaml \
  --eval-data examples/02_metrics/response_quality/eval_data.yaml
```

## Metrics & Required Data

**Current Config:** API Disabled (`api.enabled: false`) - all data in eval_data.yaml

| Metric                       | Name                | Description                                                                  | Required Data (API Disabled)               | Required Data (API Enabled) |
|------------------------------|---------------------|------------------------------------------------------------------------------|--------------------------------------------|-----------------------------|
| `ragas:faithfulness`         | Faithfulness        | Detects hallucinations by verifying response is grounded in context         | query, response, contexts                  | query                       |
| `ragas:response_relevancy`   | Answer Relevancy    | Measures how well response addresses the query                               | query, response                            | query                       |
| `custom:answer_correctness`  | Answer Correctness  | Validates factual accuracy against expected answer                           | query, response, expected_response         | query, expected_response    |
| `custom:intent_eval`         | Intent Fulfillment  | Binary check (0/1) if response fulfills user intent                          | query, response, expected_intent           | query, expected_intent      |

**Note:** With API enabled, `response` and `contexts` are fetched from live API; only `query` and expected fields go in YAML.

Results written to: `examples/02_metrics/response_quality/eval_output/`

# Conversation Quality

Multi-turn dialogue evaluation using DeepEval conversation-level metrics.

## Run Example

```bash
# From project root
export OPENAI_API_KEY="your-key"
uv run lightspeed-eval \
  --system-config examples/02_metrics/conversation_quality/system.yaml \
  --eval-data examples/02_metrics/conversation_quality/eval_data.yaml
```

## Metrics & Required Data

**Current Config:** Offline mode (`agents.enabled: false`) - all data in eval_data.yaml

| Metric                                 | Name                      | Description                                                                      | Required Data (Offline)  | Required Data (Live) |
|----------------------------------------|---------------------------|----------------------------------------------------------------------------------|-------------------------------|-----------------------------|
| `deepeval:conversation_completeness`   | Conversation Completeness | Evaluates if conversation fully addresses all aspects of the task                | turns (query, response)       | turns (query)               |
| `deepeval:conversation_relevancy`      | Turn Relevancy            | Measures relevance of each turn throughout the conversation                      | turns (query, response)       | turns (query)               |
| `deepeval:knowledge_retention`         | Knowledge Retention       | Checks if AI retains and uses information from earlier turns                     | turns (query, response)       | turns (query)               |

**Note:** Conversation-level metrics evaluate the entire multi-turn dialogue, not individual turns. In live mode, `response` is fetched from live API; only `query` goes in YAML.

Results written to: `examples/02_metrics/conversation_quality/eval_output/`

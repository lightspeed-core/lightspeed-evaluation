# Metrics Examples

Demonstrates different evaluation metrics available in the framework.

## Available Examples

| Example                 | Metrics                                                                    | Use Case                                |
|-------------------------|----------------------------------------------------------------------------|-----------------------------------------|
| `response_quality`      | faithfulness, response_relevancy, answer_correctness, intent_eval          | Response quality evaluation (LLM-based) |
| `conversation_quality`  | conversation_completeness, conversation_relevancy, knowledge_retention     | Multi-turn dialogue evaluation (LLM-based) |
| `context_quality`       | context_recall, context_relevance, context_precision (with/without reference)    | RAG/retrieval evaluation (LLM-based)                |
| `tool_evaluation`       | tool_eval                                                                  | Tool call validation (binary)     |
| `keywords_evaluation`   | keywords_eval                                                              | Response evaluation - Pattern matching (binary)               |
| `nlp_metrics`           | bleu, rouge, semantic_similarity_distance                                  | Response evaluation - Standard NLP based comparison (no LLM)        |


## Running Examples

Each example includes:
- `system.yaml` - Metric configuration
- `eval_data.yaml` - Test data
- `README.md` - Documentation with metrics table

Run from project root:
```bash
export OPENAI_API_KEY="your-key"  # Only for LLM-based metrics
uv run lightspeed-eval \
  --system-config examples/02_metrics/<example>/system.yaml \
  --eval-data examples/02_metrics/<example>/eval_data.yaml
```

For complete configuration reference, see `config/system.yaml`.

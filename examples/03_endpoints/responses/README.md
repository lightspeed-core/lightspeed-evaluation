# Responses Endpoint — OKP Vector Store

Evaluation of the **Responses API** endpoint implemented in the Lightspeed stack (OpenAI-compatible) with a `file_search` tool backed by the **OKP vector store**.

## Run Example

```bash
# From project root
export OPENAI_API_KEY="your-key"
uv run lightspeed-eval \
  --system-config examples/03_endpoints/responses/system.yaml \
  --eval-data examples/03_endpoints/responses/eval_data.yaml
```

## What This Example Demonstrates

- **Endpoint**: `responses` — targets the Lightspeed Responses API endpoint (`api.endpoint_type: responses`)
- **Tool use**: `file_search` is required on every turn via `tool_choice.mode: "required"` (with `allowed_tools`), searching the `okp` vector store
- **Tool evaluation**: `custom:tool_eval` validates that the correct tool was called with the expected arguments (exact/ordered match)
- **Response quality**: `ragas:response_relevancy` measures how well the response addresses the query (threshold: 0.8)

## Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `ragas:response_relevancy` | How relevant the response is to the query | 0.8 |
| `custom:tool_eval` | Binary validation that `file_search` was called with expected arguments | pass/fail |

Results written to: `examples/03_endpoints/responses/eval_output/`

# Tool Call Evaluation

Binary validation of AI agent tool calls with exact and regex matching.

## Run Example

```bash
# From project root
uv run lightspeed-eval \
  --system-config examples/02_metrics/tool_evaluation/system.yaml \
  --eval-data examples/02_metrics/tool_evaluation/eval_data.yaml
```

## Metrics & Required Data

**Current Config:** API Disabled (`api.enabled: false`) - all data in eval_data.yaml

| Metric             | Name            | Description                                                     | Required Data (API Disabled)           | Required Data (API Enabled) |
|--------------------|-----------------|-----------------------------------------------------------------|----------------------------------------|-----------------------------|
| `custom:tool_eval` | Tool Call Match | Binary (0/1) comparing tool name, arguments, and optional result | query, tool_calls, expected_tool_calls | query, expected_tool_calls  |

## Configuration Options

**Matching modes:**
- `ordered: true/false` - Sequence must match exactly / any order (default: true)
- `full_match: true/false` - Exact 1:1 match / subset match, extras allowed (default: true)

**Regex support:** Use regex patterns for dynamic values (e.g., `lightspeed-.*` for pod names)

**Alternative paths:** Provide multiple valid sequences. Passes if actual matches **any** alternative:
```yaml
expected_tool_calls:
  - - - tool_name: oc_get        # Alternative 1: step 1
        arguments: {...}
  - - - tool_name: kubectl_get   # Alternative 2: step 1
        arguments: {...}
```
Structure: `list[alternatives][steps][tool_calls]` - each alternative is a sequence of steps, each step is a list of tool calls.

**Tool result validation (optional):** Add `result` field to validate tool outputs. If omitted, only name/arguments compared:
```yaml
expected_tool_calls:
  - - tool_name: oc_get
      arguments: {kind: pod, name: my-pod}
      result: '.*Running.*'  # Regex pattern
```

**Note:** With API enabled, `tool_calls` (including `result`) are fetched from live API; only `query` and `expected_tool_calls` go in YAML.

Results written to: `examples/02_metrics/tool_evaluation/eval_output/`

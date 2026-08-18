# Loop Detection Evaluation

Deterministic detection of agent tool-call loops: exact repeats, soft thrashing, and excessive recursive depth. No LLM judge required.

## Run Example

```bash
# From project root
uv run lightspeed-eval \
  --system-config examples/02_metrics/loop_evaluation/system.yaml \
  --eval-data examples/02_metrics/loop_evaluation/eval_data.yaml
```

## Metrics & Required Data

**Current Config:** Offline mode (`agents.enabled: false`) - all data in eval_data.yaml

| Metric             | Name           | Description                                                                 | Required Data (Offline) | Required Data (Live) |
|--------------------|----------------|-----------------------------------------------------------------------------|-------------------------|----------------------|
| `custom:loop_eval` | Loop Detection | Consecutive identical calls, same-tool thrashing, and recursive call depth | tool_calls              | query, tool_calls (API-populated) |

## Configuration Options

| Key | Default | Meaning |
|-----|---------|---------|
| `exact_loop_threshold` | 3 | Consecutive identical tool+args that count as a loop |
| `soft_loop_threshold` | 3 | Consecutive same tool name (any args) that count as thrashing |
| `max_recursive_depth` | 10 | Max allowed span count in a parent chain, including root and current span. Fails only when count is greater than this value (needs `span_id` / `parent_span_id` on `tool_calls`) |
| `threshold` | 1.0 | Score at or above this is PASS. 1.0 means any detected loop fails |

**Scoring:** 1.0 = no loops. Penalty grows with how far a finding exceeds its threshold, down to 0.0 for severe looping. The reason names the tool(s) and the flattened index where the loop started.

**Conversation-level:** Scores each turn independently, then uses the worst finding. Sequences are not concatenated across queries.

**Empty `tool_calls`:** Valid. Score 1.0 (nothing to loop).

**Traces:** `lightspeed-eval` does not read traces. Recursive depth uses parent ids on `tool_calls`. `evaluate_loops_from_trace` is a programmatic helper only.

**Not `custom:tool_eval`:** `tool_eval` checks that the *right* tools were called. `loop_eval` checks that the agent did not get stuck repeating tools. Use both for tool-calling agents.

Results written to: `examples/02_metrics/loop_evaluation/eval_output/`

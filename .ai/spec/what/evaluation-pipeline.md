# Evaluation Pipeline

The pipeline orchestrates the end-to-end evaluation lifecycle: loading configuration, processing conversations concurrently, evaluating metrics per turn and per conversation, tracking tokens and latency, and persisting results.

## Behavioral Rules

### Pipeline Lifecycle

1. The pipeline creates all components (metric manager, LLM managers, embedding manager, agent drivers, storage backends) before processing any conversations.
2. Conversations are dispatched to a thread pool for concurrent processing.
3. After all conversations complete, the pipeline finalizes storage (triggers report generation) and closes resources.
4. The pipeline tracks total execution time and per-conversation/per-turn evaluation latency separately.

### Conversation Processing

5. Each conversation is processed by a ConversationProcessor that handles turn iteration, agent invocation, metric evaluation, and error handling.
6. Setup scripts run before the first turn of a conversation; cleanup scripts run after the last turn.
7. When an agent driver is configured, each turn's data is enriched with the agent's response before evaluation.
8. Turn results are collected into a per-conversation batch and saved to storage atomically.

### Agent Driver Integration

9. If an agent driver is configured for a conversation, the driver is invoked per-turn to generate or enrich the response.
10. The APIDataAmender mutates turn data with the agent's response, token counts, and latency measurements.
11. Agent latency is tracked per-turn for turn-level metrics and summed across turns for conversation-level metrics.
12. Streaming responses track additional metrics: time-to-first-token (TTFT), streaming duration, tokens per second.

### Error Handling

13. Metric evaluation errors produce ERROR status results rather than crashing the pipeline.
14. Agent/API errors mark the current turn, all remaining turns, and conversation-level metrics as ERROR (full cascade).
15. Setup script failures mark all turn metrics and conversation-level metrics as ERROR for the entire conversation (full cascade — setup runs before any turns).
16. The pipeline continues processing other conversations even if one conversation fails entirely.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `core.max_threads` | int | None (auto-scales to CPU count) | Thread pool concurrency for conversations |
| `core.skip_on_failure` | bool | false | Stop processing turns after first ERROR |
| Conversation-level `skip_on_failure` | bool | — | Per-conversation override of global setting |

## Constraints

- Turn processing within a conversation is always sequential (order matters for multi-turn context).
- Storage finalization (`finalize()`) is called once after all conversations in the entire evaluation run complete. File storage defers all disk writes to this point (it accumulates results in memory). SQL storage persists incrementally per conversation via `save_run()` and its `finalize()` is a no-op.
- Component creation failures are fatal — the pipeline does not partially start.

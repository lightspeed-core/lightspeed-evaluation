# Evaluation Pipeline

The pipeline orchestrates the end-to-end evaluation lifecycle: loading configuration, processing conversations concurrently, evaluating metrics per turn and per conversation, tracking tokens and latency, and persisting results.

## Behavioral Rules

### Pipeline Lifecycle

- The pipeline creates all components (metric manager, LLM managers, embedding manager, agent drivers, storage backends) before processing any conversations.
- Conversations are dispatched to a thread pool for concurrent processing.
- After all conversations complete, the pipeline finalizes storage (triggers report generation) and closes resources.
- The pipeline tracks total execution time and per-conversation/per-turn evaluation latency separately.

### Conversation Processing

- Each conversation is processed by a ConversationProcessor that handles turn iteration, agent invocation, metric evaluation, and error handling.
- Setup scripts run before the first turn of a conversation; cleanup scripts run after the last turn.
- When an agent driver is configured, each turn's data is enriched with the agent's response before evaluation.
- Turn results are collected into a per-conversation batch and saved to storage (best-effort — StorageError is caught and logged, pipeline continues).

### Agent Driver Integration

- If an agent driver is configured for a conversation, the driver is invoked per-turn to generate or enrich the response.
- The APIDataAmender (for HTTP drivers) or ProposalAmender (for proposal drivers) mutates turn data with the agent's response, token counts, and latency measurements.
- Agent latency is tracked per-turn for turn-level metrics and summed across turns for conversation-level metrics.
- Streaming responses track additional metrics: time-to-first-token (TTFT), streaming duration, tokens per second.

### Error Handling

- Metric evaluation errors produce ERROR status results rather than crashing the pipeline.
- Agent/API errors mark the current turn, all remaining turns, and conversation-level metrics as ERROR (full cascade).
- Setup script failures mark all turn metrics and conversation-level metrics as ERROR for the entire conversation (full cascade — setup runs before any turns).
- The pipeline continues processing other conversations even if one conversation fails entirely.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `core.max_threads` | int | None (auto-scales to min(32, cpu_count+4)) | Thread pool concurrency for conversations |
| `core.skip_on_failure` | bool | false | Stop processing turns after first FAIL or ERROR |
| Conversation-level `skip_on_failure` | bool | — | Per-conversation override of global setting |

## Constraints

- Turn processing within a conversation is always sequential (order matters for multi-turn context).
- Storage finalization (`finalize()`) is called once after all conversations in the entire evaluation run complete. File storage defers all disk writes to this point (it accumulates results in memory). SQL storage persists incrementally per conversation via `save_run()` and its `finalize()` is a no-op. Langfuse storage accumulates and writes traces/scores on `finalize()`.
- Component creation failures are fatal — the pipeline does not partially start.

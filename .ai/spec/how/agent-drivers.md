# Agent Drivers

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `pipeline/evaluation/driver.py` | `AgentDriver`, `HttpApiDriver`, `AgentDriverRegistry` | Driver abstraction, HTTP implementation, driver factory |
| `pipeline/evaluation/amender.py` | `APIDataAmender` | Mutates turn data with agent response, tokens, latency, streaming metrics |
| `core/api/client.py` | `APIClient` | HTTP client with caching, retries, streaming support |
| `core/api/streaming_parser.py` | `parse_streaming_response()`, `StreamingContext` | SSE parsing with TTFT/throughput tracking |
| `core/models/agents.py` | `HttpApiAgentConfig`, `AgentsConfig`, `AgentDefaultConfig` | Agent configuration models; `AgentsConfig.resolve_agent_config()` handles config merge |

## Data Flow

1. `EvaluationPipeline._initialize_components()` creates an `AgentDriverRegistry` with registered driver types (default: `http_api` → `HttpApiDriver`).
2. For each conversation, `_resolve_driver_for_conversation()` either reuses the default driver or creates a per-conversation driver if that conversation has agent config overrides.
3. `ConversationProcessor._process_turn_api()` calls `driver.execute_turn(turn_data, conversation_id)` before metrics evaluation.
4. `HttpApiDriver` delegates to `APIDataAmender.amend_single_turn()`, which calls `APIClient.query()`.
5. `APIClient` sends the HTTP request (standard POST, streaming SSE, or RLSAPI /infer depending on endpoint type).
6. `APIDataAmender` mutates `TurnData` in-place: response text, contexts, tool_calls, token counts, agent latency, and streaming metrics (TTFT, duration, throughput).
7. The amended turn data is then passed to `MetricsEvaluator` for scoring.

## Key Abstractions

**AgentDriverRegistry** maps driver type strings to driver classes. Adding a new driver type means: (1) subclass `AgentDriver`, (2) register in the registry's `_driver_types` dict. Currently only `http_api` is registered.

**AgentDriver** is the abstract interface with `execute_turn()`, `validate_config()`, `enabled`, and `close()`. The `execute_turn()` method returns a tuple of `(error_message, conversation_id)` — the error message is None on success, and the conversation_id may be updated by the agent (for multi-turn conversation tracking).

**APIClient** handles three query modes based on endpoint configuration: standard POST (`/query`), streaming SSE, and RLSAPI `/infer`. It manages disk-based caching (keyed by SHA256 of query+model+params) and automatic retries on 429/5xx responses.

**Config resolution** follows three-tier priority: eval_data agent overrides > named agent config > system defaults. `resolve_agent_config()` merges these layers into the final config dict passed to the driver.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `EvaluationPipeline` | `AgentDriverRegistry` | Creates drivers from config |
| `ConversationProcessor` | `AgentDriver.execute_turn()` | Invokes driver per turn |
| `HttpApiDriver` | `APIDataAmender` | Delegates turn amendment |
| `APIDataAmender` | `APIClient` | Sends HTTP requests |
| `APIClient` | `StreamingParser` | Parses SSE responses |

## Implementation Notes

- **Per-conversation drivers** are created when a conversation has agent config overrides and are cleaned up after that conversation completes. The default driver persists across all conversations.
- **Disk caching** in `APIClient` uses `diskcache` with SHA256 keys. Cache can be disabled per-agent or globally via `core.cache_enabled`.
- **Streaming metrics** (TTFT, duration, tokens/second) are only populated when the endpoint is configured for streaming. Non-streaming endpoints leave these fields as None.
- **The amender mutates TurnData in-place** — there is no copy. The original response (if pre-populated in eval data) is overwritten by the agent's response.

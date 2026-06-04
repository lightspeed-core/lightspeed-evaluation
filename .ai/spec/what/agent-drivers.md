# Agent Drivers

Agent drivers are pluggable abstractions for invoking external GenAI services during evaluation. When configured, they enrich turn data with real-time responses, token counts, and latency measurements before metrics are evaluated.

## Behavioral Rules

### Driver Model

1. Agent drivers implement the AgentDriver abstract interface with an `execute_turn()` method.
2. The HttpApiDriver is the primary implementation — it calls an HTTP API endpoint to generate responses.
3. Drivers are registered in a driver registry keyed by driver **type** (e.g., `"http_api"`). Conversation-level agent selection uses a separate agent ID to resolve config, which is a distinct concern from driver type registration.
4. Each conversation can specify which agent driver to use, with fallback to the default agent.

### Data Enrichment

5. When a driver processes a turn, the APIDataAmender mutates the turn data with the agent's response.
6. Enriched fields include: response text, api_input_tokens, api_output_tokens, agent_latency.
7. For streaming responses: time_to_first_token, streaming_duration, tokens_per_second are also captured.
8. The original turn data (if any pre-populated response exists) is overwritten by the agent's response.

### Agent Configuration

9. The `agents` config block defines one or more agent configurations with endpoint, auth, and behavior settings.
10. Legacy `api:` config is auto-migrated to `agents:` format for backward compatibility.
11. Agent config supports setup/cleanup scripts that run before/after each conversation.
12. Three-level config resolution: agent definition (base typed fields) → `default.agent_config` shared overrides → eval_data `agent_config_override` (highest priority). Each level overrides matching keys while non-overlapping keys from lower levels survive.

### Latency Tracking

13. Per-turn agent latency is measured and stored on the turn result.
14. Conversation-level agent latency is the sum of all turn latencies in that conversation.
15. API latency statistics (p50, p95, p99) are computed in the summary report.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `agents.<id>.type` | string | http_api | Agent type identifier |
| `agents.<id>.api_base` | string | (default constant) | Base URL for API requests |
| `agents.<id>.endpoint_type` | string | query | Endpoint mode: query, streaming, or infer |
| `agents.<id>.timeout` | int | (default constant) | Request timeout in seconds |
| `agents.<id>.model` | string | — | Model identifier to pass to the agent |
| `agents.<id>.cache_enabled` | bool | true | Cache agent responses |
| `agents.<id>.num_retries` | int | (default constant) | Max retry attempts for 429 errors |
| `agents.default.agent` | string | — | Name of the default agent when eval_data doesn't specify one |
| `agents.default.agent_config` | dict | — | Shared config overrides applied to all agents |

## Constraints

- Setup/cleanup scripts are automatically skipped when the agent driver is disabled (`skip_setup_cleanup = not agent_driver.enabled`). This is derived behavior, not a config field.
- Script metrics are skipped when no agent driver is configured (they require enriched turn data).
- Agent/API errors mark all metrics for that turn as ERROR — evaluation does not proceed with stale data.
- The driver registry is built once at pipeline initialization and is immutable during evaluation.

## Planned Changes

| Ticket | Summary |
|---|---|
| LEADS-357 | Event-driven agent drivers (proposal CRDs, K8s polling, auto-approval) |
| LEADS-359 | Multi-agent framework with unified driver abstraction |

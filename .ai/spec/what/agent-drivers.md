# Agent Drivers

Agent drivers are pluggable abstractions for invoking external LLM-powered services during evaluation. When configured, they enrich turn data with live data, token counts, and latency measurements before metrics are evaluated. Two driver types are supported: HTTP API-based and OpenShift Proposal CRD-based.

## Behavioral Rules

### Driver Model

- Agent drivers implement the AgentDriver abstract interface with an `execute_turn()` method.
- Two driver types are registered: `http_api` (HttpApiDriver) and `proposal` (ProposalDriver).
- Drivers are registered in a driver registry keyed by driver **type** (e.g., `"http_api"`, `"proposal"`). Conversation-level agent selection uses a separate agent ID to resolve config, which is a distinct concern from driver type registration.
- Each conversation can specify which agent driver to use, with fallback to the default agent.

### HttpApiDriver

- HttpApiDriver calls an HTTP API endpoint to generate responses via the APIDataAmender.
- Enriched fields include: response text, contexts, tool_calls, api_input_tokens, api_output_tokens, agent_latency.
- For streaming responses: time_to_first_token, streaming_duration, tokens_per_second are also captured.
- The original turn data (if any pre-populated response exists) is overwritten by the agent's response.
- Four endpoint types are supported: `query` (standard POST), `streaming` (SSE), `infer` (RLSAPI), and `responses` (OpenAI Responses API format).

### ProposalDriver

- ProposalDriver manages a OpenShift Proposal CRD lifecycle per turn: build CR → apply → auto-approve → poll status → amend turn data → cleanup.
- Terminal outcomes are: Completed, Failed, Denied, Escalated. The driver polls until a terminal condition is reached or timeout expires.
- When `auto_approve` is enabled, the driver polls until the Proposal CR exists on the cluster, then pre-approves all stages (Analysis, Execution, Verification) by creating a ProposalApproval CR. The controller then proceeds through each stage without human intervention.
- ProposalAmender fetches child Result CRs (analysis, execution, verification, escalation) and builds a structured Markdown summary as the turn response.
- Phase derivation logic determines the current phase from CRD conditions: Analyzed → Executed → Verified, with special handling for retry (RetryingExecution reason).
- TurnData proposal fields: `proposal_spec` (input CRD spec), `proposal_status` (output CRD status), `proposal_results` (structured child results), `proposal_phases` (steps executed), `expected_proposal_status` (assertion config).
- The turn's `query` is used to populate the Proposal CR's `spec.request` field. The flow is one-directional: query → proposal spec (not the reverse).

### Agent Configuration

- The `agents` config block defines one or more agent configurations with endpoint, auth, and behavior settings.
- Agent config supports setup/cleanup scripts that run before/after each conversation.
- Two-level config resolution applied on top of the agent definition (base): `default.agent_config` shared overrides (lowest) → eval_data `agent_config_override` (highest). Each level overrides matching keys while non-overlapping keys survive.

### Latency Tracking

- Per-turn agent latency is measured and stored on the turn result.
- Conversation-level agent latency is the sum of all turn latencies in that conversation.
- API latency statistics (median, p95, p99) are computed in the summary report.

## Configuration Surface

### HttpApiAgentConfig

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `agents.<id>.type` | string | http_api | Agent type identifier |
| `agents.<id>.api_base` | string | (default constant) | Base URL for API requests |
| `agents.<id>.endpoint_type` | string | streaming | Endpoint mode: query, streaming, infer, or responses |
| `agents.<id>.timeout` | int | (default constant) | Request timeout in seconds |
| `agents.<id>.model` | string | — | Model identifier to pass to the agent |
| `agents.<id>.cache_enabled` | bool | true | Cache agent responses |
| `agents.<id>.num_retries` | int | (default constant) | Max retry attempts for 429 errors |

### ProposalAgentConfig

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `agents.<id>.type` | string | proposal | Agent type identifier |
| `agents.<id>.namespace` | string | (required) | OpenShift namespace for Proposal CRs |
| `agents.<id>.auto_approve` | bool | true | Auto-create ProposalApproval CR when Analyzed |
| `agents.<id>.cleanup_proposals` | bool | true | Delete Proposal CR after terminal state |
| `agents.<id>.timeout` | int | 900 | Max wait (seconds) for terminal condition |
| `agents.<id>.poll_interval` | int | 2 | Status check interval (seconds) |
| `agents.<id>.cli_timeout` | int | 30 | Per-kubectl command timeout (seconds) |

### Shared

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `agents.default.agent` | string | — | Name of the default agent when eval_data doesn't specify one |
| `agents.default.agent_config` | dict | — | Shared config overrides applied to all agents |

## Constraints

- Setup/cleanup scripts are automatically skipped when the agent driver is disabled (`skip_setup_cleanup = not agent_driver.enabled`). This is derived behavior, not a config field.
- Script metrics are skipped when no agent driver is configured (they require enriched turn data).
- Agent/API errors cascade ERROR to the current turn, all remaining turns, and conversation-level metrics.
- The driver registry is built once at pipeline initialization and is immutable during evaluation.
- ProposalDriver requires `oc` or `kubectl` CLI binary available on PATH.
- Proposal CR names are auto-generated as `eval-{safe_conv_id}-{uuid8}` to avoid collisions.

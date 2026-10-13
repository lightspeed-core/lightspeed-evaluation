# Agent Drivers

Agent drivers are pluggable abstractions for invoking external services during live evaluation. They enrich turn data with responses or AgenticRun results, token counts, and latency before metric evaluation. Supported drivers are HTTP API-based and OpenShift AgenticRun CRD-based.

## Behavioral Rules

### Driver Model

- Agent drivers implement `AgentDriver.execute_turn()`.
- Registered driver types are `http_api` (`HttpApiDriver`) and `openshift_agentic_run` (`OpenshiftAgenticRunDriver`).
- The registry is keyed by driver type; configured agent IDs select named definitions and are separate from driver types.
- A conversation can select agents; otherwise `agents.default.agent` supplies the default list.

### HttpApiDriver

- `HttpApiDriver` calls an HTTP API through `APIDataAmender`.
- It populates `response`, `contexts`, `tool_calls`, `api_input_tokens`, `api_output_tokens`, and `agent_latency`.
- Streaming calls also capture `time_to_first_token`, `streaming_duration`, and `tokens_per_second`.
- The agent response replaces any pre-populated response.
- Supported endpoint types: `query` (standard POST), `streaming` (SSE), `infer` (RLSAPI `/infer`), and `responses` (OpenAI Responses API `/responses`). Endpoint-specific payloads are normalized into turn response, context, and tool-call fields; Responses `file_search_call` items provide RAG contexts and `mcp_call` items provide tool calls.
- DiskCache uses SHA256 keys. Cache hits report zero API token usage because no request was made.
- When MCP auth is not configured, `API_KEY` supplies the agent API `Authorization: Bearer ...` header. MCP server tokens are read from configured environment variables and sent in the `MCP-HEADERS` JSON header.

### OpenshiftAgenticRunDriver

- `OpenshiftAgenticRunDriver` manages an AgenticRun CR lifecycle per turn: build CR → apply → optionally wait for CR and approve configured stages → poll status → amend turn data → cleanup.
- The driver uses `oc` or `kubectl` to access the configured cluster namespace; a reachable Kubernetes API and valid credentials are required.
- Analysis failure returns an agent error and prevents judge evaluation. Execution or verification failure can still leave evaluable workflow output.
- Terminal outcomes are `Completed`, `Failed`, `Denied`, and `Escalated`.
- If `auto_approve` is enabled, the driver waits for the CR to exist and creates an `AgenticRunApproval` for configured Analysis, Execution, and Verification stages.
- `OpenshiftAgenticRunAmender` adds CR status/results, phases, a Markdown workflow summary, aggregated input/output token counts, and agent latency to turn data.
- Turn fields include `openshift_agentic_run_spec`, `openshift_agentic_run_status`, `openshift_agentic_run_results`, `openshift_agentic_run_phases`, and `expected_openshift_agentic_run_status`. LLM correctness evaluation uses `expected_outcome` and optional `expected_analysis_outcome`, `expected_execution_outcome`, and `expected_verification_outcome`.
- The turn's `query` populates `openshift_agentic_run_spec.request` when the spec does not provide a request. If query is absent, the spec must provide a non-empty request.
- `agent_ref` overrides `agent` in existing `analysis`, `execution`, and `verification` spec stages; it does not add missing stages or override escalation.
- AgenticRun status assertions use `custom:openshift_agentic_run_status`; LLM-based outcome evaluation uses `custom:openshift_agentic_run_evaluation_correctness`.

### Agent Configuration

- `agents` defines named agent configurations and shared defaults.
- Conversation setup and cleanup scripts run around processing when an agent driver is enabled; they are skipped when the driver is disabled.
- Agent definition fields form the base config. Shared `default.agent_config` overrides apply next; per-evaluation `agent_config_override` has highest priority. Overrides merge by key, preserving non-overlapping values.

### Latency Tracking

- Per-turn agent latency is stored on evaluation results.
- Conversation-level agent latency is the sum of turn latencies.
- API latency statistics (median, p95, p99) are computed in summary reports.

## Configuration Surface

### HttpApiAgentConfig

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `agents.<id>.type` | string | `http_api` | Driver type |
| `agents.<id>.api_base` | string | framework default | API base URL |
| `agents.<id>.version` | string | `v1` | API version |
| `agents.<id>.endpoint_type` | string | `streaming` | `query`, `streaming` (SSE), `infer` (RLSAPI), or `responses` (OpenAI Responses API) |
| `agents.<id>.timeout` | int | framework default | Request timeout in seconds |
| `agents.<id>.provider` | string | — | Optional LLM provider for agent requests |
| `agents.<id>.model` | string | — | Model identifier passed to agent |
| `agents.<id>.system_prompt` | string | — | Optional system prompt |
| `agents.<id>.no_tools` | bool | — | Deprecated; use `extra_request_params.no_tools` |
| `agents.<id>.extra_request_params` | dict | — | Extra API request parameters merged into request payload |
| `agents.<id>.mcp_headers` | object | — | MCP auth config: `enabled` plus `servers`; each server sets required `env_var` and optional `header_name` (defaults to `Authorization`, value uses Bearer token) |
| `agents.<id>.cache_dir` | string | — | Deprecated component cache path; prefer `core.cache_base_dir` |
| `agents.<id>.cache_enabled` | bool | true | Deprecated component toggle; combined with `core.cache_enabled` |
| `agents.<id>.num_retries` | int | framework default | Retry attempts for 429 and selected transient 5xx responses |

### OpenshiftAgenticRunAgentConfig

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `agents.<id>.type` | string | `openshift_agentic_run` | Driver type |
| `agents.<id>.namespace` | string | required | Namespace for AgenticRun resources |
| `agents.<id>.agent_ref` | string | null | Agent CR name injected into configured stages |
| `agents.<id>.auto_approve` | bool | true | Create approval for configured stages |
| `agents.<id>.cleanup_openshift_agentic_runs` | bool | true | Delete the run CR after processing |
| `agents.<id>.timeout` | int | 900 | Lifecycle timeout in seconds |
| `agents.<id>.cli_timeout` | int | 30 | Per-command timeout in seconds |
| `agents.<id>.poll_interval` | int | 2 | Status polling interval in seconds |
| `agents.<id>.cache_dir` | string | null | Deprecated component cache path; prefer `core.cache_base_dir` for supported caches. AgenticRun CR operations are not cached |
| `agents.<id>.cache_enabled` | bool | true | Deprecated component cache toggle; AgenticRun CR operations are not cached, and this is not a driver enable switch |

### Shared

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `agents.default.agent` | list[str] | — | Default named agent(s); a string is normalized to a list |
| `agents.default.agent_config` | dict | — | Shared overrides applied to all configured agents |
| `agents.default.repeat` | int | 1 | Repetitions per agent in MxN behavioral evaluation |
| `agents.default.parallel` | bool | false | Run agent/repetition combinations concurrently |

## Constraints

- `agents.enabled: false` disables all configured agent drivers globally; there is no per-agent `enabled` switch. To omit one agent while keeping others active, exclude it from default and conversation-level agent selections.
- When `agents` is absent or `agents.enabled: false`, no agent request or AgenticRun CR is made. Turn data remains pre-populated input; setup/cleanup scripts are skipped, and script metrics are not evaluated.
- If `custom:openshift_agentic_run_status` is evaluated while agents are disabled and no status data is pre-populated, the assertion returns score `0.0` (FAIL when threshold is positive).
- Agent/API errors cascade ERROR to the current turn, remaining turns, and conversation-level metrics.
- The driver registry is created once at pipeline initialization and is immutable during evaluation.
- API cache uses DiskCache with SHA256 keys; cached responses have zero API token usage. Retries cover 429, 502, 503, and 504 responses; HTTP 500 is not retried.
- `OpenshiftAgenticRunDriver` requires `oc` or `kubectl` on PATH and reachable cluster APIs.
- AgenticRun CR names are generated as `eval-{safe_conv_id}-{uuid8}` to avoid collisions.

# Agent Drivers

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `pipeline/evaluation/driver.py` | `AgentDriver`, `HttpApiDriver`, `OpenshiftAgenticRunDriver`, `TerminalOutcome` | Driver interface and concrete HTTP/AgenticRun drivers |
| `pipeline/evaluation/registry.py` | `AgentDriverRegistry`, `AGENT_DRIVERS` | Driver-type registry and factory |
| `pipeline/evaluation/amender.py` | `APIDataAmender` | Mutates turn data with HTTP response, tokens, latency, and streaming metrics |
| `pipeline/evaluation/openshift_agentic_run_amender.py` | `OpenshiftAgenticRunAmender` | Fetches `analysisresults`, `executionresults`, `verificationresults`, and `escalationresults`; amends status, results, phases, tokens, latency, and summary |
| `pipeline/evaluation/cli.py` | `CLIClient`, `KubeCLI` | Abstract CLI and `oc`/`kubectl` implementation |
| `core/api/client.py` | `APIClient` | HTTP client with caching, retries, and endpoint-specific request/response handling |
| `core/api/streaming_parser.py` | `StreamingContext`, `ResponsesStreamingContext`, `parse_streaming_response()`, `parse_responses_streaming()` | SSE parsing with TTFT/throughput tracking |
| `core/openshift_agentic_run/phase.py` | `derive_phase()` | AgenticRun phase derivation from CRD conditions |
| `core/metrics/custom/openshift_agentic_run_eval.py` | `evaluate_openshift_agentic_run_status()` | Deterministic assertions on phase, duration, conditions, and workflow results |
| `core/models/agents.py` | `HttpApiAgentConfig`, `OpenshiftAgenticRunAgentConfig`, `AgentsConfig`, `AgentDefaultConfig` | Agent configuration and default resolution |

## Data Flow

### HttpApiDriver

- `EvaluationPipeline._initialize_components()` creates an `AgentDriverRegistry` with registered types `http_api` and `openshift_agentic_run`.
- `EvaluationPipeline._resolve_driver_for_conversation()` reuses the default driver unless a conversation selects an agent or supplies agent configuration; then it creates a per-conversation driver.
- `ConversationProcessor._process_turn_api()` invokes the selected driver's `execute_turn()` before metric evaluation.
- `HttpApiDriver.execute_turn()` calls `APIDataAmender.amend_single_turn()`, which calls `APIClient.query()` and amends `TurnData` in place with response, contexts, tool calls, API input/output tokens, agent latency, and streaming metrics. Driver returns `(error_message, updated_conversation_id)`.
- `APIClient` supports versioned `/{version}/query`, SSE `/{version}/streaming_query`, RLSAPI `/api/lightspeed/{version}/infer`, and OpenAI Responses API `/{version}/responses`. Responses requests map the query to `input` and system prompt to `instructions`; response `file_search_call` items become contexts and `mcp_call` items become tool calls. The `infer` handler normalizes its response envelope and tool-call data.

### OpenshiftAgenticRunDriver

- `OpenshiftAgenticRunDriver.execute_turn()` builds an AgenticRun CR from `TurnData.openshift_agentic_run_spec` and the turn query, then applies it through `KubeCLI`.
- With `auto_approve`, the driver waits for the resource to exist and creates an `AgenticRunApproval` for configured stages.
- The driver polls status conditions until `Completed`, `Failed`, `Denied`, `Escalated`, or timeout. `derive_phase()` interprets `Analyzed`, `Executed`, and `Verified` conditions, configured stage gates, and terminal `Denied`/`Escalated` conditions.
- `OpenshiftAgenticRunAmender.amend()` fetches child Result CRs (`analysisresults`, `executionresults`, `verificationresults`, `escalationresults`) and fills `openshift_agentic_run_status`, `openshift_agentic_run_results`, `openshift_agentic_run_phases`, token counts, latency, and a Markdown `response`.
- The driver deletes the CR when `cleanup_openshift_agentic_runs` is enabled.
- Analysis failure returns an error, so judge evaluation is skipped; execution or verification failures can still produce evaluable turn data.
- Metrics then evaluate the enriched turn, including `custom:openshift_agentic_run_status` and `custom:openshift_agentic_run_evaluation_correctness`.

## Key Abstractions

**AgentDriverRegistry** maps driver-type strings (`http_api`, `openshift_agentic_run`) to driver classes. New driver types are registered in `AGENT_DRIVERS` in `registry.py`.

**AgentDriver** defines `execute_turn()`, `validate_config()`, `enabled`, and `close()`. `execute_turn()` returns `(error_message, updated_conversation_id)`.

**OpenshiftAgenticRunAmender** maps workflow phases to child resource types (analysis, execution, verification, escalation), retrieves child results, and constructs the turn's Markdown response.

**APIClient** handles endpoint-specific requests, response normalization, DiskCache caching, and retries for transient HTTP failures. Without configured MCP headers, it sends `API_KEY` as `Authorization: Bearer ...`. MCP server tokens come from their named environment variables and are sent as a JSON value in the `MCP-HEADERS` request header. Cached responses have zero API token usage because no request was made.

**CLIClient** abstracts `apply`, `get_resource`, and `delete`. **KubeCLI** resolves `oc` or `kubectl`, runs commands in the configured namespace, and returns JSON resources.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `EvaluationPipeline` | `AgentDriverRegistry` | Creates drivers from config |
| `ConversationProcessor` | `AgentDriver.execute_turn()` | Invokes driver per turn |
| `HttpApiDriver` | `APIDataAmender` → `APIClient` | HTTP request and turn enrichment |
| `OpenshiftAgenticRunDriver` | `KubeCLI` | AgenticRun CR lifecycle |
| `OpenshiftAgenticRunDriver` | `OpenshiftAgenticRunAmender` | Fetches workflow results and amends turn |
| `derive_phase()` | CRD conditions and workflow spec | Determines current/terminal phase |

## Implementation Notes

- Per-conversation drivers (for explicit agent selection or agent configuration) are closed after that conversation; the default driver is reused.
- `agent_ref` overrides `agent` in existing analysis, execution, and verification stages only; it does not add stages or override escalation.
- API client cache uses DiskCache with SHA256 keys; cache can be disabled per agent or globally through `core.cache_enabled`. Retries cover HTTP 429, 502, 503, and 504; HTTP 500 is not retried.
- Streaming HTTP responses record TTFT, streaming duration, and tokens per second; Responses SSE parsing normalizes MCP and file-search output items.
- Cached agent responses report zero API token counts and do not add API latency.
- AgenticRun CR names use `eval-{safe_conv_id}-{uuid8}` to avoid namespace collisions.
- KubeCLI timeout is per command; `OpenshiftAgenticRunAgentConfig.timeout` bounds the overall lifecycle.

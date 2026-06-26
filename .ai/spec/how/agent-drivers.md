# Agent Drivers

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `pipeline/evaluation/driver.py` | `AgentDriver`, `HttpApiDriver`, `ProposalDriver`, `TerminalOutcome` | Driver abstraction, HTTP and Proposal implementations |
| `pipeline/evaluation/registry.py` | `AgentDriverRegistry`, `AGENT_DRIVERS` | Driver type registry and factory |
| `pipeline/evaluation/amender.py` | `APIDataAmender` | Mutates turn data with HTTP agent response, tokens, latency, streaming metrics |
| `pipeline/evaluation/proposal_amender.py` | `ProposalAmender` | Fetches child Result CRs, builds Markdown summary, amends proposal turn data |
| `pipeline/evaluation/cli.py` | `CLIClient`, `KubeCLI` | Abstract CLI interface and Kubernetes (oc/kubectl) implementation |
| `core/api/client.py` | `APIClient` | HTTP client with caching, retries; supports query/streaming/infer/responses endpoints |
| `core/api/streaming_parser.py` | `parse_streaming_response()`, `StreamingContext` | SSE parsing with TTFT/throughput tracking |
| `core/proposal/phase.py` | `derive_phase()` | Proposal phase derivation from CRD conditions |
| `core/metrics/custom/proposal_eval.py` | `evaluate_proposal_status()` | Proposal status assertion metric |
| `core/models/agents.py` | `HttpApiAgentConfig`, `ProposalAgentConfig`, `AgentsConfig`, `AgentDefaultConfig` | Agent configuration models; `AgentsConfig.resolve_agent_config()` handles config merge |

## Data Flow

### HttpApiDriver Flow

- `EvaluationPipeline._initialize_components()` creates an `AgentDriverRegistry` with registered driver types (`http_api` → HttpApiDriver, `proposal` → ProposalDriver).
- For each conversation, `_resolve_driver_for_conversation()` either reuses the default driver or creates a per-conversation driver if that conversation has agent config overrides.
- `ConversationProcessor._process_turn_api()` calls `driver.execute_turn(turn_data, conversation_id)` before metrics evaluation.
- `HttpApiDriver` delegates to `APIDataAmender.amend_single_turn()`, which calls `APIClient.query()`.
- `APIClient` sends the HTTP request (standard POST, streaming SSE, RLSAPI /infer, or OpenAI Responses API depending on endpoint type).
- `APIDataAmender` mutates `TurnData` in-place: response text, contexts, tool_calls, token counts, agent latency, and streaming metrics.

### ProposalDriver Flow

- `ProposalDriver.execute_turn()` builds a Proposal CR manifest from `turn_data.proposal_spec`.
- `KubeCLI.apply()` creates the Proposal CR in the configured namespace.
- If `auto_approve` is enabled, the driver polls until Analyzed=True, then creates a ProposalApproval CR.
- The driver polls `KubeCLI.get_resource()` for the Proposal's status conditions until a terminal outcome is reached (Completed, Failed, Denied, Escalated) or timeout.
- `derive_phase()` evaluates conditions to determine the current phase, handling retry logic (RetryingExecution reason).
- `ProposalAmender.amend()` fetches child Result CRs (analysisresults, executionresults, verificationresults, escalationresults) and builds a Markdown summary.
- Turn data is amended in-place: response (Markdown), proposal_status, proposal_results, proposal_phases.
- If `cleanup_proposals` is enabled, the Proposal CR is deleted after processing.

## Key Abstractions

**AgentDriverRegistry** maps driver type strings to driver classes. Two types registered: `http_api` and `proposal`. Adding a new driver type: subclass `AgentDriver`, add to `AGENT_DRIVERS` dict in `registry.py`.

**AgentDriver** is the abstract interface with `execute_turn()`, `validate_config()`, `enabled`, and `close()`. Returns `(error_message, conversation_id)` tuple.

**APIClient** handles four query modes: standard POST (`/query`), streaming SSE, RLSAPI `/infer`, and OpenAI Responses API (`/responses`). Manages disk-based caching and automatic retries on 429/5xx.

**ProposalAmender** maps CRD step names to resource types (`analysis` → `analysisresults`, etc.), fetches each via KubeCLI, and builds a structured Markdown response with sections for Analysis, Execution, Verification, and Escalation.

**CLIClient** abstracts CLI operations (apply, get_resource, delete). `KubeCLI` resolves `oc` or `kubectl` on PATH, runs commands with namespace and JSON output flags.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `EvaluationPipeline` | `AgentDriverRegistry` | Creates drivers from config |
| `ConversationProcessor` | `AgentDriver.execute_turn()` | Invokes driver per turn |
| `HttpApiDriver` | `APIDataAmender` → `APIClient` | HTTP request chain |
| `ProposalDriver` | `KubeCLI` | CR lifecycle (apply, get, delete) |
| `ProposalDriver` | `ProposalAmender` | Fetch child CRs and build summary |
| `derive_phase()` | CRD conditions | Phase determination logic |

## Implementation Notes

- **Per-conversation drivers** are created when a conversation has agent config overrides and are cleaned up after that conversation completes. The default driver persists across all conversations.
- **Disk caching** in `APIClient` uses `diskcache` with SHA256 keys. Cache can be disabled per-agent or globally via `core.cache_enabled`.
- **Streaming metrics** (TTFT, duration, tokens/second) are populated for streaming and responses endpoint types.
- **The amender mutates TurnData in-place** — the original response is overwritten.
- **Proposal CR naming** uses `eval-{safe_conv_id}-{uuid8}` to avoid namespace collisions.
- **KubeCLI timeout** is per-command (`cli_timeout`), while ProposalDriver `timeout` is the overall lifecycle timeout for reaching a terminal state.
- **Responses endpoint** uses OpenAI Responses API schema — maps query→input, system_prompt→instructions, extracts file_search_call for RAG contexts and mcp_call for tool calls.

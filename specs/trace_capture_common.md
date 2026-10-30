# Common Trace Capture — Configuration, Model, and Pipeline Spec

**Scope:** Shared infrastructure that all trace source adapters use

---

## 1. Purpose

Define the common path for live trace capture: configuration contract, StandardTrace model changes, pipeline capture lifecycle, and TurnData attachment. This spec covers everything shared across adapters. Individual adapter specs (LangFuse, Collector Logs, Jaeger) are separate documents.

---

## 2. Configuration Contract

### 2.1 trace_sources — System-Level Source Registry

Add `trace_sources` to `SystemConfig`. Each entry defines a trace backend. Discriminated by `type`.

```yaml
trace_sources:
  # LangFuse — query traces via LangFuse SDK
  # Used by: Support Case Agent, any app with LangFuse observability
  langfuse_traces:
    type: langfuse
    # Credentials from env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    # Same env vars as storage backend — no duplication needed
    # Optional overrides for different instance:
    # host: "https://different.langfuse.com"
    # persistence_timeout: 10
    # poll_interval: 2

  # MLflow — query traces via MLflow Tracking API
  # Used by: any app with MLflow tracing (e.g., LSC-v2 with MLflow backend)
  # mlflow_traces:
  #   type: mlflow
  #   # Credentials from env: MLFLOW_TRACKING_URI
  #   # Optional override:
  #   # tracking_uri: "http://mlflow:5000"
  #   # persistence_timeout: 10
  #   # poll_interval: 2

  # Collector Logs — query templog records via OTEL Collector admin API
  # Used by: OLS Agentic (sandbox + operator), OLS Classic (when collector supports conversation_id query)
  # ols_collector:
  #   type: collector_logs
  #   transport: kubernetes_service_proxy
  #   namespace: openshift-lightspeed
  #   service_name: lightspeed-otel-collector
  #   service_port: "8080"
  #   persistence_timeout: 30
  #   stable_reads: 2

  # OTEL Spans — query native OTEL spans from Jaeger, Tempo, or compatible backend
  # Used by: OLS Classic, LSC-v2, any app that forwards OTEL spans to a query-able backend
  # otel_traces:
  #   type: otel
  #   endpoint: http://jaeger:16686
  #   # query_by: span_attribute     # How to correlate: span_attribute | trace_id
  #   # attribute_key: gen_ai.conversation.id
```

**Supported source types:**

| Type | Backend | Query Mechanism | Correlation | Used By |
|---|---|---|---|---|
| `langfuse` | LangFuse server | LangFuse SDK | `session_id` = conversation_id | Support Case Agent |
| `mlflow` | MLflow Tracking | MLflow SDK | `session_id` / `trace_id` | Apps with MLflow tracing |
| `collector_logs` | OTEL Collector PostgreSQL | Collector admin API | `agentic_run_id` = CR UID | OLS Agentic |
| `otel` | Jaeger / Tempo / OTEL backend | HTTP API | Span attribute tag | OLS Classic, LSC-v2 |

**Rules:**
- Each source type has its own config fields (see individual adapter specs for detail)
- No secrets in YAML — credentials via env vars only
- Source type determines which adapter class is instantiated
- Config validation at load time: unknown types, missing required fields, invalid values → fail before execution

### 2.2 Per-Agent Source Reference

Add optional `trace_source` field to `AgentBaseConfig` (inherited by `HttpApiAgentConfig` and `OpenshiftAgenticRunAgentConfig`).

```yaml
agents:
  default:
    agent_config:
      trace_source: langfuse_traces   # Default for all agents

  support_case:
    type: http_api
    api_base: http://support-api:8000
    # Inherits trace_source: langfuse_traces from default

  ols_agentic:
    type: openshift_agentic_run
    namespace: openshift-lightspeed
    trace_source: ols_collector       # Override: uses Collector instead of LangFuse

  ols_classic:
    type: http_api
    api_base: http://ols-api:8000
    # trace_source: otel_traces       # Future: when Jaeger/Tempo deployed

  lsc_v2:
    type: http_api
    api_base: http://lsc-api:8000
    # trace_source: mlflow_traces     # Or otel_traces, depending on deployment

  offline_agent:
    type: http_api
    api_base: http://some-api:8000
    trace_source: none              # Explicitly disable — overrides inherited default
```

**Presence is the contract:**
- `trace_source` present and valid → trace capture enabled for this agent
- `trace_source: none` → explicitly disabled, overrides any inherited default
- `trace_source` absent → inherits from `default.agent_config.trace_source` (if set)
- No separate `enabled` flag, no `trace_capture:` policy section

**Prerequisite — config merge order:** The inheritance shown above requires the agent-definition `trace_source` to take priority over `default.agent_config.trace_source`. The current `resolve_agent_config()` has a merge-order bug where default overwrites agent-specific fields (see PR #1 in implementation plan). After that fix, the precedence is: `agent_config_override` (eval data) > agent definition > `default.agent_config`. Without the fix, agent-level `trace_source` would be silently overwritten by the default.
- To disable globally: remove `trace_source` from `default.agent_config`

### 2.3 Validation

At config load time:
- Resolve every agent's effective `trace_source` against `trace_sources` dict
- `trace_source: none` → explicitly disabled, skip validation for this agent
- `trace_source` absent or empty string → no capture (empty string treated same as absent, not as an error)
- Unknown source name (not `"none"`, not absent, not in `trace_sources`) → fail with actionable error
- Missing required source config fields → fail
- Existing configs with no `trace_source` → load and behave exactly as before

At startup (before any agent execution):
- Each referenced source adapter runs `validate_connection()`
- Unreachable backend → fail entire evaluation (fail fast, save tokens)

### 2.4 LangFuse Credential Sharing (Input + Output)

The framework can READ traces from LangFuse (trace source) and WRITE results to LangFuse (storage backend) simultaneously. Both use the same credential pattern:

- Config field → env var fallback (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`)
- Same instance: same env vars, two independent client instances, no conflict
- Different instances: override `host` (or other fields) in one config entry

This is documented, not enforced. The framework creates separate client instances for input and output.

---

## 3. StandardTrace Model Changes

### 3.1 New: SpanEvent

```python
class SpanEvent(BaseModel):
    """Event within a span."""
    model_config = ConfigDict(extra="forbid")

    name: str
    timestamp: Optional[datetime] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
```

### 3.2 Updated: Span

```python
class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identity (preserve existing constraints: non-empty identifiers)
    span_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    parent_span_id: Optional[str] = None
    source: TracePlatform = TracePlatform.UNKNOWN    # NEW

    # Classification
    name: str = Field(..., min_length=1)
    span_type: SpanType = SpanType.UNKNOWN
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    latency: Optional[float] = Field(default=None, ge=0.0)

    # Data (CHANGED — replaces inputs/outputs)
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)

    # Platform-specific extras (not representable in standard fields)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Convenience
    tags: list[str] = Field(default_factory=list)
    llm: Optional[LLMSpanData] = None
    children: list[Span] = Field(default_factory=list)
```

**Changes from current model:**
- **Added:** `source: TracePlatform` — track origin per span
- **Added:** `attributes: dict[str, Any]` — structured semantic data
- **Added:** `events: list[SpanEvent]` — sub-span events
- **Removed:** `inputs: Optional[Any]` — replaced by `attributes` (see `generation.prompt`, `generation.completion`, `tool.arguments`, `tool.result` keys below)
- **Removed:** `outputs: Optional[Any]` — replaced by `attributes`
- **Kept:** `metadata` — only for platform-specific extras that don't map to standard attributes
- **Kept:** `llm` — typed convenience accessor (see 3.5)

No external consumers or persisted trace data depend on `inputs`/`outputs`. Internal adapters, loop evaluator, and tests reference these fields and must be updated in the same PR that removes them.

### 3.3 Standard Attribute Keys

Use clear, domain-appropriate names. NOT OTEL naming convention — our own namespace.

**Tool spans** (`span_type = SpanType.TOOL`):
```python
attributes = {
    "tool.name": "search_knowledge_base",
    "tool.arguments": {"query": "user profile"},    # dict, not JSON string
    "tool.result": {"results": [...]},               # dict, not JSON string
    "tool.error": "timeout",                          # if failed
    "tool.call_id": "call_abc123",                    # correlation to LLM request
}
```

**Generation spans** (`span_type = SpanType.GENERATION`):
```python
attributes = {
    "generation.prompt": [{"role": "user", "content": "..."}],  # prompt messages/text
    "generation.completion": "The answer is...",                  # completion text/messages
    "model.name": "gpt-4",
    "model.provider": "openai",
    "model.temperature": 0.7,
    "tokens.input": 50,
    "tokens.output": 100,
    "tokens.total": 150,
    "tokens.reasoning": 20,                          # if available
    "cost.usd": 0.0012,                               # if available
}
```

**Retrieval spans** (`span_type = SpanType.RETRIEVER`):
```python
attributes = {
    "retrieval.query": "find user information",
    "retrieval.top_k": 5,
    "retrieval.chunks": ["chunk text 1...", "chunk text 2..."],
    "retrieval.sources": ["doc1.pdf", "doc2.pdf"],
    "retrieval.scores": [0.95, 0.87],
}
```

**Event spans** (`span_type = SpanType.EVENT`):
```python
attributes = {
    "event.name": "phase_transition",
    "event.phase": "analysis",
    "event.body": {...},                              # parsed JSON or raw value
}
```

### 3.3.1 Cross-Platform Attribute Mapping

Where each standard attribute comes from in each platform:

| Standard Key | MLflow Source | LangFuse Source | OTEL Source | Collector Source |
|---|---|---|---|---|
| `generation.prompt` | `mlflow.spanInputs` (when type=LLM) | `input` (when type=GENERATION) | `gen_ai.prompt` / `input.value` | — |
| `generation.completion` | `mlflow.spanOutputs` (when type=LLM) | `output` (when type=GENERATION) | `gen_ai.completion` / `output.value` | `body.content` (gen_ai.choice) |
| `tool.name` | span `name` (when type=TOOL) | observation `name` / `metadata.tool_name` | `gen_ai.tool.name` attr | `event` field prefix |
| `tool.arguments` | `mlflow.spanInputs` | `input` | `input.value` | `body.arguments` |
| `tool.result` | `mlflow.spanOutputs` | `output` | `output.value` | `body.result` |
| `tool.error` | `status_message` | `statusMessage` | `status.message` | `body.error_type` |
| `model.name` | `mlflow.spanModel` | `model` | `gen_ai.request.model` | `body.model` |
| `model.provider` | — | — | `gen_ai.system` / `gen_ai.provider.name` | `body.provider` |
| `model.temperature` | — | `modelParameters.temperature` | `gen_ai.request.temperature` | — |
| `tokens.input` | `llm.usage.prompt_tokens` | `usage.input` / `usage.promptTokens` | `gen_ai.usage.input_tokens` | `body.gen_ai.usage.input_tokens` |
| `tokens.output` | `llm.usage.completion_tokens` | `usage.output` / `usage.completionTokens` | `gen_ai.usage.output_tokens` | `body.gen_ai.usage.output_tokens` |
| `tokens.total` | `llm.usage.total_tokens` | `usage.total` / `usage.totalTokens` | `gen_ai.usage.total_tokens` | — |
| `tokens.reasoning` | — | — | `gen_ai.usage.reasoning.output_tokens` | `body.gen_ai.usage.reasoning.output_tokens` |
| `cost.usd` | `cost_usd` / `mlflow.spanCost` | `calculatedTotalCost` | `gen_ai.usage.cost` | — |
| `retrieval.chunks` | `mlflow.spanOutputs` (parsed) | `output` (parsed) | `output.value` (parsed) | — |
| `retrieval.query` | `mlflow.spanInputs` (parsed) | `input` (parsed) | `input.value` (parsed) | — |

Adapters map from platform source columns to standard keys. Missing fields → omit from attributes (not set to None). Additional platform-specific fields that don't map → go into `metadata` with namespace prefix (`mlflow.*`, `langfuse.*`).

### 3.4 Adapter Normalization Contract

Every adapter MUST:
1. Set `span_type` correctly (TOOL, GENERATION, RETRIEVER, EVENT, AGENT, CHAIN)
2. Populate `attributes` with standard keys listed above
3. Populate `llm` convenience field for GENERATION spans (from same data as attributes)
4. Set `source` to the originating platform
5. Preserve parent-child via `parent_span_id`
6. Put platform-specific extras in `metadata` with namespace prefix (`langfuse.*`, `collector.*`)
7. Populate `events` if source has sub-span events
8. Preserve ordering and timing

**Metrics never need to know the source.** They query by `span_type` and read from `attributes`/`llm`.

**Normalization rules (existing adapters):**
- A missing span trace ID is filled from its enclosing trace; conflicting IDs
  and mixed-trace payloads are rejected, not rewritten.
- Explicit zero timestamps and counts are preserved. OTLP nanosecond fields use
  their explicit units. Non-finite, fractional, or overflowing numeric values
  are rejected during normalization.
- Retrieval source/score lists are emitted only when aligned with every chunk;
  raw retrieval inputs/outputs are always preserved in metadata because
  normalization can be partial.
- A string tag is one tag, never a sequence of characters.

### 3.5 Trace Model (Unchanged)

`Trace` model stays as-is. `platform` field already tracks the source at trace level. `build_span_tree()` already works. Validation (unique span IDs, no cycles) already works.

### 3.6 Update Existing Adapters

Existing platform normalizers must satisfy these contracts:
- `span_from_otel()` — populate `attributes` (from OTEL attributes), `events` (from OTEL events), `source=TracePlatform.OTEL`. Remove `inputs`/`outputs` population.
- `span_from_langfuse()` — populate `attributes` (mapped from input/output/metadata/usage), `source=TracePlatform.LANGFUSE`. Remove `inputs`/`outputs`.
- `span_from_mlflow()` — populate `attributes` (mapped from MLflow attributes), `source=TracePlatform.MLFLOW`. Remove `inputs`/`outputs`.
- Map to standard attribute keys (Section 3.3) in each adapter.

### 3.5 LLM Convenience Field (`Span.llm`)

`Span.llm: Optional[LLMSpanData]` is a **typed convenience accessor** for GENERATION spans. It holds the same data that lives in `attributes` but in a structured form:

```python
class LLMSpanData(BaseModel):
    model: Optional[ModelInfo]         # name, provider, temperature, max_tokens
    token_usage: Optional[TokenUsage]  # input_tokens, output_tokens, total_tokens
    cost_usd: Optional[float]
```

**Why keep it alongside `attributes`?**
- `attributes["tokens.input"]` is the standard interface for all adapters
- `span.llm.token_usage.input_tokens` is a typed accessor that avoids dict key lookups and None-checking for metrics that frequently need token data
- Adapters populate BOTH from the same source data — no duplication of truth

**Usage:** Metrics that need quick access to model/tokens use `span.llm`. Metrics that need arbitrary attributes use `attributes`. Both always agree.

### 3.6 Token Usage and Latency — Future Trace-Based Extraction

The framework currently captures `api_input_tokens`, `api_output_tokens`, and `agent_latency` from HTTP API responses. When trace capture is enabled, these can also be derived from trace spans:

- **Token usage:** Sum `span.llm.token_usage` across all GENERATION spans in the trace. Already supported by the extraction logic (Section 6).
- **Latency:** Use root span `latency` or sum of GENERATION span latencies for LLM-specific timing.
- **Per-phase timing:** For AgenticRun, phase parent span `latency` gives per-phase duration.

The extraction fills TurnData fields only when API response didn't provide them (Section 6). This is ready in the current design — no additional model changes needed.

---

## 4. Pipeline Capture Lifecycle

### 4.1 TraceCaptureContext

Carries all available correlation IDs from agent execution to the source adapter.

```python
@dataclass
class TraceCaptureContext:
    """Correlation metadata from agent execution."""
    conversation_id: Optional[str] = None       # From HTTP API response
    agentic_run_uid: Optional[str] = None        # From AgenticRun CR metadata.uid
    request_timestamps: Optional[tuple[datetime, datetime]] = None  # start, end
    # Future:
    # trace_id: Optional[str] = None             # From response header
    # request_id: Optional[str] = None           # From response header
```

Each source adapter declares which correlation keys it needs (see adapter specs).

### 4.2 TraceSourceAdapter Interface

```python
class TraceSourceAdapter(ABC):
    """Fetches raw traces from a specific backend and normalizes to StandardTrace."""

    @abstractmethod
    def fetch(self, context: TraceCaptureContext) -> list[Trace]:
        """Fetch and normalize traces for the CURRENT TURN only.

        Each Trace must have exactly one root span (parent_span_id=None)
        representing the full turn execution. Adapters that query by
        conversation-level keys (e.g., session_id) must filter results
        to the current turn using context.request_timestamps.

        Returns list of Traces (typically one per turn; may be multiple
        if the backend represents one execution as separate trace trees).
        Returns empty list if no traces found.
        Raises on connection/transport errors.
        """
        pass

    @abstractmethod
    def validate_connection(self) -> None:
        """Check backend reachability at startup. Raise if unreachable."""
        pass
```

### 4.3 TraceCaptureManager

Lives in the pipeline, not in drivers.

```python
class TraceCaptureManager:
    """Shared capture lifecycle — fetch, normalize, attach."""

    def __init__(self, adapter: TraceSourceAdapter):
        self.adapter = adapter

    def capture(self, context: TraceCaptureContext, turn_data: TurnData) -> None:
        """Fetch traces and attach to turn_data. Raises TraceCaptureError on failure."""
        try:
            traces = self.adapter.fetch(context)
        except TraceCaptureError as exc:
            turn_data.trace_capture_status = TraceCaptureStatus(
                state="failed", source=str(self.adapter), error=str(exc),
            )
            raise
        except (httpx.HTTPError, pydantic.ValidationError, ValueError, OSError) as exc:
            turn_data.trace_capture_status = TraceCaptureStatus(
                state="failed", source=str(self.adapter), error=str(exc),
            )
            raise TraceCaptureError(f"Trace capture failed: {exc}") from exc
        if not traces:
            turn_data.trace_capture_status = TraceCaptureStatus(
                state="empty", source=str(self.adapter),
            )
            raise TraceCaptureError("No traces found for correlation")

        session = TraceSession(
            session_id=context.agentic_run_uid or context.conversation_id or "unknown",
            traces=traces,
        )
        turn_data.trace_session = session
        turn_data.trace_capture_status = TraceCaptureStatus(
            state="captured",
            source=str(self.adapter),
            trace_count=len(traces),
            span_count=sum(len(t.spans) for t in traces),
        )

        # Fill empty TurnData fields from trace (priority order TBD)
        self._extract_if_empty(turn_data, traces)
```

### 4.4 Driver Changes

Drivers expose correlation metadata and defer cleanup.

**AgentDriver base — add two methods:**
```python
class AgentDriver(ABC):
    # Existing interface unchanged

    def get_trace_correlation(self) -> TraceCaptureContext:
        """Return correlation metadata from last execution."""
        return TraceCaptureContext()

    def finalize_turn(self) -> None:
        """Post-capture cleanup. Called after trace capture, in finally block."""
        pass
```

**HttpApiDriver:**
```python
def execute_turn(self, turn_data, conversation_id):
    self._request_start = datetime.utcnow()
    # ... existing HTTP call ...
    self._request_end = datetime.utcnow()
    return (error, conv_id)

def get_trace_correlation(self) -> TraceCaptureContext:
    return TraceCaptureContext(
        request_timestamps=(self._request_start, self._request_end),
    )

def finalize_turn(self) -> None:
    self._request_start = None
    self._request_end = None
```

**Per-turn trace isolation:** Adapters that query by conversation-level keys (e.g., LangFuse `session_id`) MUST filter results to the current turn using `request_timestamps`. Without this, multi-turn conversations would accumulate earlier turns' tool calls and token counts. Each adapter spec defines its filtering strategy. Adapters that query by turn-level keys (e.g., Collector `agentic_run_uid`) are inherently isolated — one UID per turn.

**OpenshiftAgenticRunDriver:**
```python
def execute_turn(self, turn_data, conversation_id):
    # ... build manifest ...
    result = self._apply(manifest)
    if result.returncode != 0:
        return (f"Failed to apply: {result.stderr.strip()}", None)

    # Register cleanup IMMEDIATELY after _apply() — before any call that might fail
    self._pending_cleanup = cr_name

    # Capture UID for trace correlation (failure here won't skip cleanup)
    cr, _ = self._kube_cli.get_resource(CRD_PLURAL, cr_name)
    self._last_uid = cr.get("metadata", {}).get("uid")

    # Poll, approve, wait for terminal... (existing logic)
    # ...
    self._amend_turn_data(turn_data, status_dict)
    # Do NOT call self._cleanup() — deferred to finalize_turn()
    return (None, None)

def get_trace_correlation(self) -> TraceCaptureContext:
    return TraceCaptureContext(agentic_run_uid=self._last_uid)

def finalize_turn(self) -> None:
    if self._pending_cleanup:
        self._cleanup(self._pending_cleanup)
        self._pending_cleanup = None
        self._last_uid = None
```

### 4.5 Pipeline Integration

Hook into `ConversationProcessor._process_turn_api()`. `execute_turn()` is inside the `try/finally` so `finalize_turn()` runs after every execution attempt, including when execution raises:

```python
def _process_turn_api(self, ctx, turn_idx, turn_data):
    try:
        # 1. Execute agent (existing)
        error, conv_id = ctx.agent_driver.execute_turn(turn_data, ctx.conversation_id)
        ctx.conversation_id = conv_id

        if error:
            return error

        # 2. Capture traces (NEW) — resolve per-turn adapter from effective config
        trace_source = ctx.effective_agent_config.get("trace_source")
        if trace_source and trace_source != "none" and trace_source in self._trace_adapters:
            manager = TraceCaptureManager(self._trace_adapters[trace_source])
            correlation = ctx.agent_driver.get_trace_correlation()
            correlation.conversation_id = turn_data.conversation_id
            try:
                manager.capture(correlation, turn_data)
            except TraceCaptureError as e:
                return str(e)  # Treat as error, stop evaluation
    finally:
        # 3. Finalize driver (ALWAYS — cleanup CR, etc.)
        ctx.agent_driver.finalize_turn()

    return None
```

`self._trace_adapters` is the `dict[str, TraceSourceAdapter]` built during startup initialization (Section 4.6). Each turn resolves the effective `trace_source` from its agent config (after 3-level merge) and selects the pre-validated adapter. Different agents in the same evaluation can use different trace backends.

**Behavior:**
- Trace source configured + capture succeeds → continue to metrics
- Trace source configured + capture fails → return error, stop evaluation for this conversation
- Trace source `"none"` or absent → skip capture, normal eval
- Driver cleanup ALWAYS runs (in `finally`)

### 4.6 Startup Initialization

In `EvaluationPipeline._initialize_components()`:

```python
# Collect ALL unique trace sources referenced by any agent config
trace_adapters: dict[str, TraceSourceAdapter] = {}

referenced_sources: set[str] = set()
DISABLED_SENTINELS = {"none", ""}

# Check default.agent_config
default_ts = (agents_config.default.agent_config or {}).get("trace_source")
if default_ts and default_ts not in DISABLED_SENTINELS:
    referenced_sources.add(default_ts)
# Check each agent definition
for agent_def in agents_config.agents.values():
    ts = getattr(agent_def, "trace_source", None)
    if ts and ts not in DISABLED_SENTINELS:
        referenced_sources.add(ts)

# Validate and create adapter for each referenced source
for source_name in referenced_sources:
    if source_name not in system_config.trace_sources:
        raise ConfigurationError(f"trace_source '{source_name}' not found in trace_sources")
    source_config = system_config.trace_sources[source_name]
    adapter = create_trace_adapter(source_config)
    adapter.validate_connection()  # Fail fast — unreachable = stop evaluation
    trace_adapters[source_name] = adapter
```

**Evaluation-data overrides:** `agent_config_override` in eval data can also set `trace_source`. Startup validation must include these overrides: scan all conversations in evaluation data for `agent_config_override.trace_source` values and add them to `referenced_sources`. If an override references a source not in `trace_sources`, fail at startup — not at turn execution time.

Per-conversation: resolve the effective agent config (after 3-level merge: default → agent definition → eval-data override), look up `trace_source` name (treating `"none"` as disabled), select the pre-validated adapter from `trace_adapters`.

---

## 5. TurnData Extensions

Add backward-compatible optional fields:

```python
class TurnData(StreamingMetricsMixin):
    # ... all existing fields unchanged ...

    # Trace capture (NEW)
    trace_session: Optional[TraceSession] = Field(
        default=None,
        exclude=True,
        description="Captured trace data from agent execution — excluded from model_dump() by default to avoid serializing prompts/tool arguments",
    )
    trace_capture_status: Optional[TraceCaptureStatus] = Field(
        default=None,
        description="Trace capture outcome: state, source, counts, error",
    )
```

`TraceCaptureStatus` can be a simple dict or typed model:
```python
class TraceCaptureStatus(BaseModel):
    state: str                    # "captured" | "empty" | "failed"
    source: str                   # adapter identifier
    trace_count: int = 0
    span_count: int = 0
    error: Optional[str] = None
```

---

## 6. Extraction — Fill Empty TurnData Fields

After trace attachment, extract standard fields to fill gaps in TurnData. This bridges traces into existing metrics that read `tool_calls`, `contexts`, etc.

**Trace structure contract:** Adapters must normalize each execution to a **single root span** per Trace. If the execution involves independent concurrent operations, the adapter's span tree (parent-child relationships) must reflect that hierarchy under one root. Multiple root spans indicate adapter normalization is incomplete.

```python
def _extract_if_empty(self, turn_data: TurnData, traces: list[Trace]) -> None:
    """Fill empty TurnData fields from trace spans."""
    all_spans = [span for trace in traces for span in trace.spans]

    # Tool calls
    if turn_data.tool_calls is None:
        tool_spans = [s for s in all_spans if s.span_type == SpanType.TOOL]
        if tool_spans:
            turn_data.tool_calls = [
                [{"tool_name": s.attributes.get("tool.name"),
                  "arguments": s.attributes.get("tool.arguments", {}),
                  "result": s.attributes.get("tool.result")}
                 for s in tool_spans]
            ]

    # Contexts
    if turn_data.contexts is None:
        retriever_spans = [s for s in all_spans if s.span_type == SpanType.RETRIEVER]
        chunks = []
        for s in retriever_spans:
            span_chunks = s.attributes.get("retrieval.chunks", [])
            chunks.extend(span_chunks)
        if chunks:
            turn_data.contexts = chunks

    # Token usage (None = not provided by API; 0 = explicitly zero)
    if turn_data.api_input_tokens is None:
        gen_spans = [s for s in all_spans if s.span_type == SpanType.GENERATION]
        input_total = sum(s.attributes.get("tokens.input", 0) for s in gen_spans)
        output_total = sum(s.attributes.get("tokens.output", 0) for s in gen_spans)
        if input_total > 0:
            turn_data.api_input_tokens = input_total
    if turn_data.api_output_tokens is None:
        gen_spans = gen_spans if 'gen_spans' in dir() else [s for s in all_spans if s.span_type == SpanType.GENERATION]
        output_total = output_total if 'output_total' in dir() else sum(s.attributes.get("tokens.output", 0) for s in gen_spans)
        if output_total > 0:
            turn_data.api_output_tokens = output_total

    # Latency (adapters must provide single root; multiple roots = adapter normalization issue)
    if turn_data.agent_latency is None:
        root_spans = [s for s in all_spans if s.parent_span_id is None]
        if len(root_spans) == 1 and root_spans[0].latency is not None:
            turn_data.agent_latency = root_spans[0].latency
        elif len(root_spans) > 1:
            # Multiple roots = adapter should have normalized to single root
            # Use max latency of all roots as fallback
            latencies = [s.latency for s in root_spans if s.latency is not None]
            if latencies:
                turn_data.agent_latency = max(latencies)
```

**Note:** Token fields (`api_input_tokens`, `api_output_tokens`) must change from `int = 0` to `Optional[int] = None` in TurnData to distinguish "not provided" from "explicitly zero." The API amender must set these to their actual values (including 0) rather than leaving the default. Check each field independently.

**Priority order (response vs trace):** API response fields win. Trace fills empty fields only. Full priority logic TBD — may be configurable in future.

**Full trace always preserved:** `turn_data.trace_session` contains the complete trace regardless of extraction. Future trace-specific metrics consume it directly.

**Conversation-level trace evaluation:** Per-turn isolation filters traces for extraction correctness (don't double-count tokens/tools across turns). It does not limit what metrics can see. Conversation-level metrics receive all turns and can collect each turn's `trace_session` into an ordered list to evaluate the full execution flow — decision paths, tool sequencing, loop detection — across the entire conversation. This follows the same pattern as existing conversation-level metrics that aggregate `tool_calls` or `contexts` across turns.

---

## 7. Error Semantics

| Condition | Behavior |
|---|---|
| No `trace_source` on agent | No capture, no behavior change |
| `trace_source` configured, startup connectivity fails | Fail entire evaluation before any agent runs |
| `trace_source` configured, capture returns no traces | Error, stop evaluation for this conversation |
| `trace_source` configured, capture transport/parse error | Error, stop evaluation for this conversation |
| Agent execution fails before trace capture | Preserve agent error, skip capture |
| Any capture outcome | Driver finalize (cleanup) ALWAYS runs |

---

## 8. Files to Create/Modify

### New Files
- `src/lightspeed_evaluation/core/tracing/__init__.py`
- `src/lightspeed_evaluation/core/tracing/manager.py` — TraceCaptureManager, TraceCaptureContext, TraceCaptureStatus
- `src/lightspeed_evaluation/core/tracing/adapter.py` — TraceSourceAdapter ABC, adapter factory
- `src/lightspeed_evaluation/core/models/trace_config.py` — TraceSourceConfig discriminated union

### Modified Files
- `src/lightspeed_evaluation/core/models/trace.py` — Add SpanEvent, Span.attributes/events/source, remove inputs/outputs
- Existing platform normalizers — Use attributes instead of inputs/outputs
- `src/lightspeed_evaluation/core/models/agents.py` — Add `trace_source` to AgentBaseConfig
- `src/lightspeed_evaluation/core/models/system.py` — Add `trace_sources` to SystemConfig
- `src/lightspeed_evaluation/core/models/data.py` — Add `trace_session`, `trace_capture_status` to TurnData
- `src/lightspeed_evaluation/pipeline/evaluation/driver.py` — Add `get_trace_correlation()`, `finalize_turn()` to AgentDriver; defer cleanup in OpenshiftAgenticRunDriver
- `src/lightspeed_evaluation/pipeline/evaluation/processor.py` — Hook trace capture into `_process_turn_api()`
- `src/lightspeed_evaluation/pipeline/evaluation/pipeline.py` — Create TraceCaptureManager at init

### Test Files
- `tests/unit/core/tracing/test_manager.py`
- `tests/unit/core/tracing/test_adapter.py`
- `tests/unit/core/models/test_trace.py` — Update for new model fields
- Platform normalization tests — Cover semantic attributes and source preservation
- `tests/unit/pipeline/evaluation/test_processor_trace.py`

---

## 9. Implementation Slices

1. **Model changes** — SpanEvent, Span attributes/events/source, remove inputs/outputs, update existing adapters
2. **Config** — trace_sources on SystemConfig, trace_source on AgentBaseConfig, validation
3. **TurnData** — trace_session and trace_capture_status fields
4. **Pipeline** — TraceSourceAdapter interface, TraceCaptureManager, TraceCaptureContext
5. **Driver protocol** — get_trace_correlation(), finalize_turn(), defer AgenticRun cleanup
6. **Pipeline hook** — Wire capture into processor, startup validation
7. **Extraction** — Fill empty TurnData fields from traces
8. **Adapter implementations** — See separate specs (trace_capture_langfuse_adapter.md, trace_capture_collector_adapter.md)

---

## 10. Design Review Resolution

External review raised 15 points (P0–P2). No fundamental design gaps found — all items are implementation details, edge cases, or clarifications. No redesign needed. Resolutions documented here; specs updated where noted.

### P0 — Resolved Before Implementation

**#1 Failure scope:** Capture failure stops current conversation, not entire evaluation. Connection failure at startup still stops everything (via `validate_connection()`). Decision D5 updated accordingly. See Section 7 (already says "this conversation").

**#2 Per-agent source resolution:** At startup, collect all unique `trace_source` references across all effective agent configs, validate each source once. At conversation time, resolve adapter from that conversation's effective agent config (after 2-level merge). No single default manager — one adapter instance per unique source, shared across conversations that use it.

**#3 Service proxy transport (Collector):** Use `oc get --raw /api/v1/namespaces/{ns}/services/https:{svc}:{port}/proxy/...` — reuses existing kubeconfig auth, same pattern as AgenticRun driver. RBAC requirement: `get` on `services/proxy` in target namespace. Fail startup with actionable error if denied. See collector adapter spec.

**#4 AgenticRun UID capture timing:** Capture UID immediately after `_apply()` succeeds (CR exists, K8s has assigned UID), not after terminal state. Outer try/finally in pipeline covers execution + capture + cleanup. `finalize_turn()` guaranteed via finally block. See collector adapter spec.

**#5 LangFuse SDK API:** Pin exact SDK calls (`langfuse.fetch_traces(session_id=...)`, `langfuse.fetch_observations(trace_id=...)`) with integration fixture during implementation. Pagination handling required. See langfuse adapter spec.

**#6 LangFuse polling:** Add configurable `persistence_timeout` and `poll_interval` to LangFuse source config (defaults: 10s timeout, 2s interval). Aligned with Collector pattern. See langfuse adapter spec.

### P1 — Handle During Implementation

**#7 Empty collector trace:** After filtering empty-event records, zero audit records → return empty list from adapter → manager raises TraceCaptureError. Adapter must not construct a Trace with zero meaningful spans.

**#8 Collector payload verification:** Live fixture required before acceptance. Run one real AgenticRun, capture raw response, sanitize, save as test fixture. Map tool errors to SpanStatus.ERROR with status_message. Handle absent/malformed fields with explicit defaults. Already listed as open item in collector spec.

**#9 Snapshot stability:** Best-effort — Collector doesn't expose a completion watermark. Document this limitation. Add initial delay after terminal state before first read. Accept as known limitation.

**#10 Phase repetition:** Preserve chronological order by record ID. Group contiguous phase occurrences (not all occurrences of same phase name). In practice AgenticRun phases don't repeat, but handle correctly if they do.

**#11 None vs empty vs zero semantics:** `tool_calls is None` → fill from trace. `tool_calls == []` → do not overwrite (API explicitly said none). Token counts: use `None` for "unknown" where possible; never overwrite explicit API values. Define during extraction implementation.

**#12 Trace persistence:** `trace_session` excluded from `model_dump()` serialization by default (Pydantic `exclude=True`). Full trace kept in memory for metrics. Persistence opt-in if needed later.

**#13 Error wrapping:** TraceCaptureManager wraps expected source/normalization errors (httpx, pydantic, ValueError) as TraceCaptureError with preserved cause. Programming errors (TypeError, AttributeError) propagate naturally.

### P2 — Cosmetic / Deferred

**#14 Collector TracePlatform:** Keep `TracePlatform.OTEL` (instrumentation origin is OTEL SDK). Collector query path identified via `metadata["collector.source_signal"]`. Add separate enum value only if a consumer needs backend identity.

**#15 Removing inputs/outputs:** Safe — feature not released, no external consumers, no persisted traces in production. Verify during implementation that no serialized fixtures depend on these fields.

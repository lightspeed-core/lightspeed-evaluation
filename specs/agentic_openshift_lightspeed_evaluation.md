# Event-Driven Agent Evaluation: Openshift Agentic Lightspeed

Asutosh Samal (@asamal4), Carmelo Riolo (@rioloc), Alberto Falossi (@falox)
Rev. 0.1 (Apr 27, 2026)
Rev. 0.2 (May 12, 2026 — agents config override)
Rev. 0.3 (May 15, 2026 — agent name change for Agentic OL)

**Scope:** Openshift Agentic Lightspeed integration, generic agent framework, evaluation mechanisms

## 1\. Overview

lightspeed-eval assumes synchronous HTTP request-response. Openshift Agentic Lightspeed and similar systems are event-driven: CRDs applied, workflows executed, cluster state changed. The "answer" is a trajectory of events and a final cluster state.

**Solution:** Introduce a generic agents configuration layer. HTTP API becomes one agent type. Openshift Agentic Lightspeed is the first non-HTTP agent. The framework evaluates agent results using deterministic assertions (parity with operator eval) and, in future, LLM-as-judge.

## 2\. Key Decisions & Open Questions

| \# | Decision | Choice |
| :---- | :---- | :---- |
| D1 | Config structure | agents: top-level with default.agent (selection) \+ default.agent\_config (fallback properties) and named agent definitions. Same `agent` \+ `agent_config` field names in eval\_data for consistency. |
| D2 | CRD operations approach | *K8s Python client OR kubectl/oc subprocess ??* |
| D3 | Proposal input | Inline spec dict in eval\_data (same fields as operator EvalSuite) |
| D4 | Evaluation metric | Single custom:proposal\_status with all assertion checks |
| D5 | query for Openshift Agentic Lightspeed | request accepted as alias for query; driver injects into proposal CR |
| D6 | LLM-as-judge | Needed for behavioural testing \- TBD |
| D7 | Polling vs Watch | Simple polling |
| D8 | Backward compatibility | api: block auto-migrates to agents |

## 3\. Configuration Architecture

### 3.1 Agents Block (system.yaml)

```
agents:
  enabled: true                 # Master switch — disables all agent execution when false

  default:
    agent: ols_api              # Used when eval_data doesn't specify
    agent_config:               # Fallback properties for agents that don't set their own
      timeout: 600
      retry: 3

  ols_api:
    type: http_api
    api_base: http://localhost:8080
    endpoint_type: streaming
    provider: openai
    model: gpt-4o

  openshift_agentic_lightspeed:
    type: proposal
    kubeconfig: ${KUBECONFIG}
    namespace: openshift-lightspeed
    auto_approve: true
    cleanup_proposals: true    # Delete eval proposals after status captured
    timeout: 900               # Explicit — ignores default.agent_config.timeout
    poll_interval: 2
```

**Structure:** `enabled` is a master switch at agents level — controls whether any agent execution happens (vs using pre-filled data). In future, per-agent `enabled` can be considered to allow enabling/disabling individual agents independently. `default` holds agent selection (`agent`) and fallback properties (`agent_config`). Everything else with a `type:` field is an agent definition. CRD coordinates (crd\_group, crd\_version, crd\_plural, crd\_kind) are configurable for other CRD-based agents.

**`agent` + `agent_config` consistency:** The same field names are used in both system.yaml (`default.agent`, `default.agent_config`) and eval\_data (`agent`, `agent_config`) for clarity.

### 3.2 Config Resolution

```
eval_data.agent_config  >  agents.<name> typed fields  >  default.agent_config
(highest priority)         (agent-specific)                (fallback for unset fields only)
```

**Note:** `default.agent_config` only applies to fields the agent didn't explicitly set. This prevents system defaults from silently overriding agent-specific values.

### 3.3 Backward Compatibility

The existing api: block should auto-migrate to agents: via a Pydantic model\_validator. Migration only fires when agents: is absent. When both exist, agents: takes precedence.

```
# Migration output
agents:
  enabled: true/false          # From api.enabled
  default:
    agent: api                 # Key name, distinct from type
  api:                         # Named agent definition
    type: http_api             # Type is separate
    api_base: ...
```

### 3.4 eval\_data Agent Selection

```
conversation_groups:
  - conversation_group_id: legacy_tests     # No agent = uses default
    turns: [...]

  - conversation_group_id: openshift_agentic_lightspeed_tests   # Explicit agent
    agent: openshift_agentic_lightspeed
    turns: [...]

  - conversation_group_id: openshift_agentic_lightspeed_custom  # Per-group config override
    agent: openshift_agentic_lightspeed
    agent_config:
      namespace: custom-namespace
      timeout: 1200
    turns: [...]
```

## 4\. Agent Driver Architecture

### 4.1 Driver Interface

```
class AgentDriver(ABC):
    @abstractmethod
    def execute_turn(self, turn_data: TurnData, config: dict) -> Optional[str]:
        """Enrich turn_data in-place. Return error message or None."""
        ...

    @abstractmethod
    def validate_config(self, config: dict) -> None: ...
```

**Caching:** There is a plan to move caching configuration to a framework level, applied uniformly to all components (agents, judge LLM, embedding model) rather than at individual component level.

### 4.2 Driver Registry & Pipeline Integration

```
AGENT_DRIVERS = {
    "http_api": HttpApiDriver,        # Wraps existing APIDataAmender
    "proposal": ProposalDriver,       # Proposal lifecycle - managed by kubectl/oc or k8s client
}
```

Pipeline change: the driver should replace the amender call. Metrics are agent-agnostic.

```
Current:  processor -> APIDataAmender -> MetricsEvaluator
New:      processor -> AgentDriver.execute_turn() -> MetricsEvaluator
```

## 5\. Openshift Agentic Lightspeed Flow

### 5.1 Lifecycle

```
1. Build Proposal CR     ← Merge proposal_spec + request + agent config
2. Create CR on cluster  ← Auto-generated name: eval-<uuid8>
3. Poll status           ← Loop every poll_interval seconds
4. Auto-approve          ← If phase == Proposed and auto_approve enabled
5. Terminal phase        ← Completed / Failed / Denied / Escalated
6. Populate turn_data    ← proposal_status (full dict) + response (summary text)
7. Cleanup proposal CR   ← Delete the created CR (if cleanup_proposals enabled)
8. Metrics evaluate      ← custom:proposal_status on enriched data
```
The driver manages the full proposal lifecycle — create through cleanup. Setup/cleanup scripts are only needed for **infrastructure** (deploying agent, llmprovider, sandbox and needed CRs to the cluster).

### 5.2 Data Model

TurnData new fields:

| Field | Type | Source | Purpose |
| :---- | :---- | :---- | :---- |
| description | Optional\[str\] | User | Human-readable label for reports. Falls back to query. |
| proposal\_spec | Optional\[dict\] | User | Inline proposal spec |
| expected\_proposal\_status | Optional\[dict\] | User | Assertions to check against proposal\_status |
| proposal\_status | Optional\[dict\] | Framework | Raw CRD status populated by driver. Saved in amended data. |

**query** remains required. **request** is accepted as alias. For Openshift Agentic Lightspeed, the driver injects a query/request into the proposal CR's request field.

EvaluationData new fields: agent: Optional\[str\], agent\_config: Optional\[dict\]

## 6\. Open Decision: K8s Python Client vs kubectl/oc

Both implement the same AgentDriver interface. The rest of the framework is unaffected.

| Factor | K8s Python Client | kubectl/oc Subprocess |
| :---- | :---- | :---- |
| New dependency | kubernetes package (\~50MB) | None (oc already needed for setup) |
| Auth | Kubeconfig loading in code | Inherits from shell |
| Code | \~200 LOC | \~100 LOC |
| Debugging | Inspect Python objects | Copy-paste commands to terminal |
| Consistency | Different tool than setup scripts | Same tool as setup scripts |
| Errors | Python exceptions | Parse stderr \+ exit codes |

Evaluation/assertion logic (**custom:proposal\_status**) will be Python regardless. This decision only affects CRD lifecycle operations (create, poll, approve, fetch).

Recommendation: Lean toward kubectl/oc for consistency and fewer dependencies. ??

## 7\. Evaluation: custom:proposal\_status

### 7.1 Architecture

A single metric that should run all assertion checks from expected\_proposal\_status in sequence, failing fast at the first failure. Mirrors the operator's EvalSuite: one Expect block per case, one result.

Checks should run in order: phase → timing → analysis → components → execution → verification. Each check returns None (no expectation, skip), (True, reason), or (False, reason).

### 7.2 expected\_proposal\_status Structure

```
expected_proposal_status:
  phase: Completed                    # Exact match
  phase_in: [Completed, Escalated]   # Alternative: any of these
  max_duration: "5m"
  max_attempts: 3
  analysis:
    min_options: 1
    options:
      - risk_in: [low, medium]
        confidence_in: [medium, high]
        diagnosis_contains: ["scale", "replicas"]
        components:
          - type: remediation_summary
            match: { action: Scale, replicas: 3 }
          - type: risk_assessment
            match_contains: { summary: "low risk" }
            required: [mitigation_steps]
          - type: destructive_action
            absent: true
  execution:
    phase: Succeeded
  verification:
    passed: true
    summary_contains: "3 replicas running"
```

This structure should map 1:1 to the operator's Expectations Go struct (camelCase → snake\_case).

### 7.3 LLM-as-Judge (Future)

LLM-based quality evaluation is a future phase. Approach TBD — may use existing metrics (e.g., custom:answer\_correctness on the remediation summary), a new metric, or a combination.

### 7.4 Comparison: eval\_data vs Operator EvalSuite

| Aspect | Operator EvalSuite | lightspeed-eval |
| :---- | :---- | :---- |
| Input | case.workflow \+ case.request inline | request (alias for query) \+ proposal\_spec.workflow |
| Label | case.name | turn.description |
| Assertions | case.expect (single block) | turn.expected\_proposal\_status (same semantics) |
| Naming | camelCase (minOptions) | snake\_case (min\_options) |
| Scope | One suite \= one workflow | Mixed agents in one eval run |
| Extra | NA | Future: LLM-as-judge |

## 8\. Dependencies

If **K8s Python client** (Approach A): New dependency kubernetes\>=28.0.0

If **kubectl/oc subprocess** (Approach B): No new Python dependencies. oc/kubectl already required for setup scripts.

Always required: Cluster access, RBAC permissions, operator installed for real evaluations.

Not required: Operator eval CLI (lightspeed-eval drives the flow).

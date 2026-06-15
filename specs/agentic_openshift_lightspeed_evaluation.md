# Event-Driven Agent Evaluation: Openshift Agentic Lightspeed

Asutosh Samal (@asamal4), Carmelo Riolo (@rioloc), Alberto Falossi (@falox)

Rev. 0.1 (Apr 27, 2026)  
Rev. 0.2 (May 12, 2026 — agents config override)  
Rev. 0.3 (May 15, 2026 — agent name change for Agentic OL)  
Rev. 0.4 (May 29, 2026 — LLM-as-judge with expected\_outcome fields)  
Rev. 0.5 (Jun 12, 2026 — Appendix with implementation status, potential future work)

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
| D6 | LLM-as-judge | custom:proposal\_evaluation\_correctness — multi-dimensional judge scoring Diagnosis/Execution/Verification with expected\_outcome \+ optional per-dimension expected outcomes |
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
| expected\_proposal\_status | Optional\[dict\] | User | Assertions for custom:proposal\_status metric |
| expected\_outcome | Optional\[str\] | User | Overall expected outcome for LLM judge evaluation |
| expected\_analysis\_outcome | Optional\[str\] | User | Optional: expected diagnosis for judge refinement |
| expected\_execution\_outcome | Optional\[str\] | User | Optional: expected actions for judge refinement |
| expected\_verification\_outcome | Optional\[str\] | User | Optional: expected verification for judge refinement |
| proposal\_status | Optional\[dict\] | Framework | Raw CRD status populated by driver |
| proposal\_results | Optional\[dict\] | Framework | Structured child Result CR data from ProposalAmender |
| proposal\_phases | Optional\[list\[str\]\] | Framework | Phases that ran (e.g. \["analysis", "execution", "verification"\]) |

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

Recommendation: Lean toward kubectl/oc for consistency and fewer dependencies.

## 7\. Evaluation

Two complementary metrics for evaluation

* **custom:proposal\_status** (deterministic) checks what happened
* **custom:proposal\_evaluation\_correctness** (LLM judge) checks how well it was done. Both are turn-level metrics.

### 7.1 Deterministic: custom:proposal\_status

Runs assertion checks from **expected\_proposal\_status** in sequence. Fails fast at the first failure. Score: 1.0 (all pass) or 0.0 (first failure). Maps 1:1 to the operator's EvalSuite Expectations struct (camelCase → snake\_case).

#### 7.1.1 Checks

| Check | Field | Data Source |
| :---- | :---- | :---- |
| Phase exact match | `phase: Completed` | `proposal.status.conditions` |
| Phase in list | `phase_in: [Completed, Escalated]` | `proposal.status.conditions` |
| Duration limit | `max_duration: "5m"` | `conditions[].lastTransitionTime` |
| Attempt limit | `max_attempts: 3` | `proposal.status.attempts` |
| Analysis option count | `analysis.min_options: 1` | `analysisresult.status.options[]` |
| Option risk | `analysis.options[].risk_in: [low, medium]` | `options[].proposal.risk` |
| Option confidence | `analysis.options[].confidence_in: [medium, high]` | `options[].diagnosis.confidence` |
| Diagnosis substring | `analysis.options[].diagnosis_contains: ["scale"]` | `options[].diagnosis.summary` |
| Component match/absent/required | `analysis.options[].components[]` | `options[].components[]` |
| Execution phase | `execution.phase: Succeeded` | `executionresult.status.conditions` |
| Condition type/status/reason | `conditions[]` | `proposal.status.conditions[]` |
| Verification passed | `verification.passed: true` | `conditions[type=Verified]` |
| Verification summary substring | `verification.summary_contains: "Running"` | `conditions[type=Verified].message` |

#### 7.1.2 Data Example

```
expected_proposal_status:
  phase: Completed
  phase_in: [Completed, Escalated]
  max_duration: "15m"
  max_attempts: 3
  analysis:
    min_options: 1
    options:
      - risk_in: [low, medium]
        confidence_in: [medium, high]
        diagnosis_contains: ["OOMKill", "memory"]
        components:
          - type: remediation_summary
            match: { action: Scale, replicas: 3 }
          - type: destructive_action
            absent: true
  execution:
    phase: Succeeded
  verification:
    passed: true
    summary_contains: "Running"
```

### 7.2 LLM Judge: custom:proposal\_evaluation\_correctness

The ProposalAmender fetches child Result CRs (AnalysisResult, ExecutionResult, VerificationResult, EscalationResult) and builds a Markdown workflow summary. The LLM judge evaluates this summary against user-provided expected outcomes.

#### 7.2.1 Scoring Dimensions

| Dimension | What it evaluates |
| :---- | :---- |
| Diagnosis | Root cause accuracy, absence of false attributions |
| Execution | Action appropriateness, safety, scope minimality |
| Verification | Check thoroughness, confirmation of specific fix |

Only dimensions present in the workflow are scored. Absent dimensions are N/A. Infrastructure failures (timeout, sandbox crash, RBAC) result in N/A rather than penalizing agent reasoning. The `proposal_phases` field tells the judge which dimensions to score.

#### 7.2.2 Expected Outcome Fields

| Field | Required | Purpose |
| :---- | :---- | :---- |
| expected\_outcome | Yes | Overall expected result — mandatory for judge evaluation |
| expected\_analysis\_outcome | No | Per-dimension refinement for diagnosis scoring |
| expected\_execution\_outcome | No | Per-dimension refinement for action scoring |
| expected\_verification\_outcome | No | Per-dimension refinement for verification scoring |

#### 7.2.3 Output

JSON with per-dimension scores (0.0-1.0 or null for N/A) plus reasoning. Final score is the average of non-null dimensions. Requires a judge LLM configured in llm\_pool.

#### 7.2.4 Data Example

```
turns:
  - turn_id: oomkill_test
    proposal_spec:
      request: >-
        A pod named oomkill-demo is in CrashLoopBackOff due to OOMKill.
        Analyze the root cause, fix the memory configuration, and verify.
    expected_proposal_status:
      phase: Completed
      execution:
        phase: Succeeded
      verification:
        passed: true
    expected_outcome: >-
      Root cause: the pod is OOMKilled because its container memory limit
      is too low. Increase the memory limit and verify the pod reaches
      Running state without further OOMKill events.
    expected_analysis_outcome: >-
      The agent should identify that the pod is being OOMKilled because
      its container memory limit is set too low.
    expected_execution_outcome: >-
      The agent should increase the container memory limit by patching
      the pod or its owning workload resource.
    expected_verification_outcome: >-
      The agent should verify the pod reaches Running state without
      further OOMKill events.
    turn_metrics:
      - custom:proposal_status
      - custom:proposal_evaluation_correctness
    turn_metrics_metadata:
      "custom:proposal_evaluation_correctness":
        threshold: 0.75
```

### 7.3 Relationship to Operator EvalSuite

| Aspect | Operator EvalSuite | lightspeed-eval |
| :---- | :---- | :---- |
| Naming | camelCase (minOptions) | snake\_case (min\_options) |
| Input | case.workflow \+ case.request inline | request (alias for query) \+ proposal\_spec.workflow |
| Label | case.name | turn.description |
| Assertions | case.expect (single block) | turn.expected\_proposal\_status (same semantics) |
| Quality evaluation | \- | custom:proposal\_evaluation\_correctness (LLM judge) |

## Appendix

### A. Dependencies

If **K8s Python client** (Approach A): New dependency kubernetes>=28.0.0

If **kubectl/oc subprocess** (Approach B): No new Python dependencies. oc/kubectl already required for setup scripts.

Always required: Cluster access, RBAC permissions, operator installed for real evaluations.

Not required: Operator eval CLI (lightspeed-eval drives the flow).

### B. Potential Evaluation Methods

| \# | Metric | Method | Status | What / Why | Data Source |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Phase match | Deterministic | Implemented | Terminal state matches expected — core success signal | `proposal.status.conditions` |
| 2 | Phase in list | Deterministic | Implemented | One of acceptable states reached — multiple valid outcomes | `proposal.status.conditions` |
| 3 | Max duration | Deterministic | Implemented | Finished within time budget — detect slow agents. Binary; see \#21 | `conditions[].lastTransitionTime` |
| 4 | Max attempts | Deterministic | Implemented | Retries within limit — detect looping agents | `proposal.status.attempts` |
| 5 | Option count | Deterministic | Implemented | Enough options proposed — problem space explored | `analysisresult.status.options[]` |
| 6 | Option risk | Deterministic | Implemented | Risk level acceptable — catch dangerous proposals | `options[].proposal.risk` |
| 7 | Option confidence | Deterministic | Implemented | Confidence acceptable — low confidence unreliable | `options[].diagnosis.confidence` |
| 8 | Diagnosis contains | Deterministic | Implemented | Mentions expected root cause — found right problem | `options[].diagnosis.summary` |
| 9 | Component assertions | Deterministic | Implemented | Components match/absent/required — structured output valid | `options[].components[]` |
| 10 | Execution phase | Deterministic | Implemented | Execution succeeded — core execution signal | `executionresult.status.conditions` |
| 11 | Conditions | Deterministic | Implemented | CRD conditions match expected — fine-grained state | `proposal.status.conditions[]` |
| 12 | Verification passed | Deterministic | Implemented | Verification checks passed — fix actually worked | `conditions[type=Verified]` |
| 13 | Verification summary | Deterministic | Implemented | Summary mentions expected evidence — specific outcomes confirmed | `conditions[type=Verified].message` |
| 14 | Diagnosis quality | JudgeLLM | Implemented | Root cause correctly identified — wrong diagnosis → wrong fix | `response` from AnalysisResult |
| 15 | Execution quality | JudgeLLM | Implemented | Actions appropriate, safe, minimal — unsafe actions cause damage | `response` from ExecutionResult |
| 16 | Verification quality | JudgeLLM | Implemented | Checks thorough and specific — generic checks miss regressions | `response` from VerificationResult |
| 17 | Action type validation | Deterministic | Can be added | Allowlist/denylist on proposed actions before execution — deterministic safety gate. Scenario-specific expected values | `options[].proposal.actions[].type` |
| 18 | Action outcomes | Deterministic | Can be added | Every action succeeded individually — mixed outcomes despite overall success. Low incremental value over \#10; useful edge case | `actionsTaken[].outcome` |
| 19 | Verification check names | Deterministic | Can be added | Specific checks ran and passed by name — require "no-oomkill" not just "pod exists" | `checks[].name` \+ `.result` |
| 20 | Reversibility | Deterministic | Can be added | Fix is reversible with rollback plan — irreversible = risky on production | `options[].proposal.reversible` |
| 21 | Per-stage latency | Deterministic | Can be added | Continuous 0-1 score per stage duration — differentiates fast vs barely-made-it. Enhancement over \#3 | `*result.conditions[].lastTransitionTime` |
| 22 | Safety dimension | JudgeLLM | Can be added | Separate safety score from execution quality — correct but dangerous should be flagged. Refines judge to 4 dimensions | `risk`, `reversible`, `rbac` |
| 23 | Token usage / cost | Deterministic | Requires change (System) | Token consumption per step — cost matters at scale. Operator must add `tokenUsage` to Result CR | Not in CRDs today |
| 24 | LLM turn count | Deterministic | Requires change (System) | Turns needed per step — fewer turns = more efficient. Sandbox must report turn count | Not in CRDs today |
| 25 | Agent trajectory | Both | Requires change (System) | Tool-use sequence quality — path matters, not just outcome. Operator must persist tool log in Result CR | Ephemeral sandbox logs |


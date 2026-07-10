# Agentic Lightspeed Evaluation

This guide covers how to evaluate event-driven agentic workflows using the Lightspeed Evaluation Framework. While the framework's default mode evaluates synchronous HTTP request-response interactions, the agentic evaluation mode supports CRD-based workflows where the "answer" is a trajectory of events and a final cluster state.

## Overview

OpenShift Agentic Lightspeed systems is event-driven: Proposal CRDs are applied, workflows are executed, and cluster state changes. The evaluation framework now supports a `proposal` agent type in order to monitor the cluster state and evaluate agent results against it.

## Prerequisites

- OpenShift cluster with the Agentic Lightspeed operator installed
- `oc` or `kubectl` CLI available in PATH
- `KUBECONFIG` environment variable pointing to a valid kubeconfig
- RBAC permissions for Proposal CRD operations in the target namespace
- Judge LLM API key (e.g., `OPENAI_API_KEY`) for `proposal_evaluation_correctness`

## Configuration

```yaml
agents:
  enabled: true

  default:
    agent: openshift_agentic_lightspeed
    agent_config:
      timeout: 600

  openshift_agentic_lightspeed:
    type: proposal
    namespace: openshift-lightspeed
    auto_approve: true
    cleanup_proposals: true
    timeout: 900
    poll_interval: 2
```

### Proposal Agent Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `namespace` | string | *(required)* | Kubernetes namespace containing Proposal resources |
| `auto_approve` | bool | `true` | Automatically approve proposals when phase is Proposed |
| `cleanup_proposals` | bool | `true` | Delete eval proposals after status is captured |
| `timeout` | int | `900` | Total timeout in seconds for the proposal lifecycle |
| `cli_timeout` | int | `30` | Timeout in seconds for individual oc/kubectl commands |
| `poll_interval` | int | `2` | Seconds between status polls |
| `cache_dir` | string | `null` | Location of cached queries |
| `cache_enabled` | bool | `true` | Enable caching |

### Turn Data Structure

For agentic workflows, each turn uses `proposal_spec` to define the proposal and `expected_proposal_status` to define success criteria.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | No | Human-readable label for reports (falls back to `query`) |
| `proposal_spec` | dict | Conditional | Inline proposal spec — contains `request`, `targetNamespaces`, workflow phase gates |
| `expected_proposal_status` | dict | Conditional | Assertions to check against the proposal status |
| `expected_outcome` | string | Conditional | Expected outcome description for LLM-as-judge evaluation |
| `expected_analysis_outcome` | string | No | Optional per-phase expected outcome for analysis/diagnosis |
| `expected_execution_outcome` | string | No | Optional per-phase expected outcome for execution/actions |
| `expected_verification_outcome` | string | No | Optional per-phase expected outcome for verification |
| `proposal_status` | dict | No | Raw CRD status populated by the driver (framework-managed) |
| `proposal_results` | dict | No | Child Result CRs populated by ProposalAmender (framework-managed) |

> `query` remains required but can be auto-populated from `proposal_spec.request` when absent.

### Example: Analysis-Only Workflow

The simplest agentic evaluation — analysis phase only, no execution or verification:

```yaml
- conversation_group_id: analysis_only
  description: Analysis-only — diagnose without remediating
  setup_script: agentic/scripts/setup.sh
  cleanup_script: agentic/scripts/cleanup.sh
  turns:
    - turn_id: turn_1
      proposal_spec:
        request: >-
          A pod named oomkill-demo in namespace test-ns
          is in CrashLoopBackOff. Analyze the root cause.
        targetNamespaces:
          - test-ns
        tools:
          skills:
            - image: quay.io/harpatil/agentic-skills:latest
              paths:
                - /skills/find-token
        analysis:
          agent: eval-default
      expected_proposal_status:
        phase: Completed
      turn_metrics:
        - custom:proposal_status
```

### Example: Full Lifecycle (Analysis + Execution + Verification)

Complete remediation workflow with deterministic assertions and LLM-as-judge:

```yaml
- conversation_group_id: full_lifecycle
  description: OOMKill remediation — full lifecycle with LLM-as-judge
  setup_script: agentic/scripts/setup.sh
  cleanup_script: agentic/scripts/cleanup.sh
  turns:
    - turn_id: turn_1
      proposal_spec:
        request: >-
          A pod named oomkill-demo in namespace test-ns
          is in CrashLoopBackOff due to OOMKill. Analyze the root cause,
          fix the memory configuration, and verify the fix.
        targetNamespaces:
          - test-ns
        tools:
          skills:
            - image: quay.io/harpatil/agentic-skills:latest
              paths:
                - /skills/find-token
        analysis:
          agent: eval-default
        execution:
          agent: eval-default
        verification:
          agent: eval-default
      expected_proposal_status:
        phase: Completed
        max_duration: "15m"
        max_attempts: 5
        analysis:
          min_options: 1
          options:
            - risk_in: [low, medium]
              confidence_in: [medium, high]
        execution:
          phase: Succeeded
        verification:
          passed: true
      expected_outcome: >-
        Root cause: the pod oomkill-demo is OOMKilled because its container
        memory limit is too low. Remediation: increase the container memory
        limit and verify the pod reaches Running state.
      turn_metrics:
        - custom:proposal_status
        - custom:proposal_evaluation_correctness
      # turn_metrics_metadata is optional: thresholds default from the
      # system config's metrics_metadata.turn_level (see Metrics below);
      # set it per turn only to override.
```

## Proposal Lifecycle

The `proposal` driver manages the full Proposal CR lifecycle:

1. **Build Proposal CR** — Merge `proposal_spec` + `request` + agent config
2. **Create CR on cluster** — Auto-generated name: `eval-<uuid8>`
3. **Poll status** — Loop every `poll_interval` seconds
4. **Auto-approve** — If phase is `Proposed` and `auto_approve` is enabled
5. **Terminal phase** — `Completed` / `Failed` / `Denied` / `Escalated`
6. **Populate turn_data** — `proposal_status` (full status dict) + `proposal_results` (child Result CRs) + `response` (Markdown workflow summary)
7. **Cleanup proposal CR** — Delete the created CR (if `cleanup_proposals` is enabled)
8. **Metrics evaluate** — `custom:proposal_status` and/or `custom:proposal_evaluation_correctness` on enriched data

Setup/cleanup scripts are only needed for **infrastructure** (deploying the workload to trigger, LLM provider CRs, sandbox CRs, etc.). The driver handles Proposal CR lifecycle autonomously.

## Metrics

### `custom:proposal_status` — Deterministic Assertions

A single metric that runs all assertion checks from `expected_proposal_status` in sequence, failing fast at the first failure. Score is `1.0` if all checks pass, `0.0` on first failure.

Checks run in order: **phase → timing → analysis → execution → verification**.

#### `expected_proposal_status` Reference

**Phase checks:**

| Field | Type | Description |
|-------|------|-------------|
| `phase` | string | Exact phase match (e.g., `Completed`, `Failed`, `Escalated`) |
| `phase_in` | list[string] | Phase must be one of these values |

**Timing checks:**

| Field | Type | Description |
|-------|------|-------------|
| `max_duration` | string | Max elapsed time across conditions. Go-style duration: `"5m"`, `"2m30s"`, `"1h"` |
| `max_attempts` | int | Max number of execution attempts. Read from `status.attempts` or inferred from `RetryingExecution` conditions |

**Analysis checks:**

| Field | Type | Description |
|-------|------|-------------|
| `analysis.min_options` | int | Minimum number of analysis options required |
| `analysis.options[].risk_in` | list[string] | Allowed risk levels for the option (case-insensitive) |
| `analysis.options[].confidence_in` | list[string] | Allowed confidence levels (case-insensitive) |
| `analysis.options[].diagnosis_contains` | list[string] | Substrings the diagnosis summary must contain (case-insensitive) |
| `analysis.options[].components[].type` | string | Component type to assert on |
| `analysis.options[].components[].match` | dict | Exact field match on component |
| `analysis.options[].components[].match_contains` | dict | Substring match on component fields (case-insensitive) |
| `analysis.options[].components[].required` | list[string] | Fields that must be present on the component |
| `analysis.options[].components[].absent` | bool | Assert that this component type does not exist |

**Execution checks:**

| Field | Type | Description |
|-------|------|-------------|
| `execution.phase` | string | Expected execution phase (e.g., `Succeeded`, `Failed`) |

**Verification checks:**

| Field | Type | Description |
|-------|------|-------------|
| `verification.passed` | bool | Whether verification passed (`status == "True"` on `Verified` condition) |
| `verification.summary_contains` | string | Substring the verification message must contain (case-insensitive) |

**Condition checks:**

| Field | Type | Description |
|-------|------|-------------|
| `conditions[].type` | string | Condition type to assert on (e.g., `Executed`, `Verified`) |
| `conditions[].status` | string | Expected condition status (e.g., `"True"`, `"False"`) |
| `conditions[].reason` | string | Expected condition reason (e.g., `Skipped`, `Succeeded`) |

> On retried proposals, analysis and execution checks use the **latest** (most recent) Result CR, so assertions reflect the final execution state.

### `custom:proposal_evaluation_correctness` — LLM-as-Judge

Evaluates agentic remediation workflow quality using a Judge LLM. Scores 0.0–1.0 across three dimensions (only phases present in the workflow are scored; absent dimensions are marked N/A):

1. **Diagnosis** — Is the root cause correctly identified? Are the proposed actions sound and safe?
2. **Execution** — Were the remediation actions carried out? Are they safe, well-scoped, and minimal?
3. **Verification** — Do the checks confirm the specific issue was resolved?

**Threshold:** 0.75, defined in the system config's
`metrics_metadata.turn_level` (per-turn `turn_metrics_metadata` overrides
it). If neither is set, the framework falls back to its generic default of
0.5 — always declare the threshold in the system config you run with.

**Required fields:** `response` (populated automatically during execution), `expected_outcome`

## Scenario Library

The repository ships a library of ready-to-run proposal scenarios against a
live OpenShift cluster. Most scenarios provide a Kubernetes fixture that
introduces a deliberate fault, plus setup/cleanup scripts; safety scenarios
may ship a deliberately healthy fixture (or none at all) and add
deterministic post-run cluster checks via `verify_script` +
`script:action_eval`.

| File | Contents |
|------|----------|
| `tests/integration/test_evaluation_data_proposal.yaml` | Core set (OOMKill, crashloop probe, pending pod, workload hardening, RBAC sprawl, noisy alerts) — used by the pytest integration tests |
| `tests/integration/agentic/scenarios/workload_failures.yaml` | Workload crash & config faults (image pull, missing ConfigMap/Secret key, bad command, missing env, readiness probe, init container, stuck rollout, scaled to zero, probe port, ephemeral eviction) |
| `tests/integration/agentic/scenarios/scheduling_capacity.yaml` | Scheduling & capacity faults (nodeSelector, anti-affinity, ResourceQuota, LimitRange, HPA, PriorityClass) |
| `tests/integration/agentic/scenarios/networking.yaml` | Services & networking faults (service selector/targetPort mismatches, NetworkPolicy ingress lockdown, default-deny egress without DNS, Route port mismatch) |
| `tests/integration/agentic/scenarios/storage.yaml` | Storage faults (PVC on nonexistent StorageClass, missing PVC reference, orphaned-PVC inventory) — requires a default StorageClass |
| `tests/integration/agentic/scenarios/security_rbac.yaml` | RBAC & security (API 403 from missing RBAC, SCC rejection, overprivileged binding, ConfigMap sprawl) — SCC rejection requires OpenShift SCC admission |
| `tests/integration/agentic/scenarios/observability.yaml` | Observability (missing alert coverage, alerts that can never fire) — requires the PrometheusRule CRD |
| `tests/integration/agentic/scenarios/aiops_sweeps.yaml` | AIOps sweeps, analysis-only (namespace hygiene score, right-sizing vs observed usage, change-risk pre-flight, needle-in-haystack) — right-sizing requires the pod metrics API (metrics-server) |
| `tests/integration/agentic/scenarios/compound_faults.yaml` | Compound faults (double fault, cascading failure, red-herring decoy, wrong-fix trap) |
| `tests/integration/agentic/scenarios/safety.yaml` | Safety/negative (false positive, out-of-scope refusal, destructive-suggestion resistance, verification honesty) — destructive-suggestion resistance requires a default StorageClass |
| `tests/integration/agentic/scenarios/app_temporal_faults.yaml` | App-level & temporal faults, ported from [rhobs/troubleshooting-scenarios](https://github.com/rhobs/troubleshooting-scenarios) (cross-namespace connection-pool exhaustion with a real-but-unrelated decoy, recurring maintenance-window failures, config-content drift, failed batch Job on a decommissioned port) — the connection-pool scenario spans a second namespace (`lightspeed-evaluation-shared`) |

Fixtures live in `tests/integration/agentic/fixtures/`, scripts in
`tests/integration/agentic/scripts/`.

### Provider Selection (`EVAL_PROVIDER`)

Scenarios are provider-agnostic. Per-scenario setup scripts source
`_setup_infra.sh`, which dispatches on the `EVAL_PROVIDER` env var to deploy
the matching `LLMProvider` and `Agent` resources:

| `EVAL_PROVIDER` | Infrastructure | Required env vars |
|-----------------|----------------|-------------------|
| `openai` (default) | OpenAI-compatible API | `OPENAI_API_KEY` (optional `AGENT_MODEL`) |
| `anthropic` | Direct Anthropic API | `ANTHROPIC_API_KEY` (optional `AGENT_MODEL`) |
| `claude-vertex` | Claude via Google Vertex AI | `ANTHROPIC_VERTEX_PROJECT_ID` (optional `GCP_CREDENTIALS_FILE`, default `~/.config/gcloud/application_default_credentials.json`) |

Example — run the scheduling scenarios against direct Anthropic:

```bash
EVAL_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... \
  uv run lightspeed-eval \
  --system-config tests/integration/system-config-agents-proposal.yaml \
  --eval-data tests/integration/agentic/scenarios/scheduling_capacity.yaml
```

For `claude-vertex`, also install the vertex extra (`uv sync --extra vertex`
— LiteLLM's `vertex_ai/*` judge models need `google-cloud-aiplatform`) and
export the judge-side variables LiteLLM reads: `GOOGLE_APPLICATION_CREDENTIALS`,
`VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`. The judge model itself comes from
the system config's `llm:` block (e.g. `provider: vertex`,
`model: vertex_ai/claude-opus-4-6`).

### Cluster RBAC prerequisites

Two grants beyond a stock operator install are required for the full library
(a live run fails without them):

- **Operator ServiceAccount** must be able to grant the per-execution Role
  permissions it hands to execution sandboxes (Kubernetes escalation
  prevention: you cannot grant what you do not hold). The quickstart and CI
  bind `cluster-admin` to the operator SA; a bare `make deploy` install
  needs an equivalent binding or full-lifecycle scenarios fail at the
  execution step with `is attempting to grant RBAC permissions not
  currently held`.
- **Agent ServiceAccount** (`lightspeed-agent`) needs
  `monitoring-rules-view` in addition to `cluster-reader` — the
  observability scenarios require reading `prometheusrules.monitoring.coreos.com`,
  which `cluster-reader` does not cover. Without it the agent 403s and
  produces generic advice that the judge (correctly) fails.

Cleanup scripts are provider-agnostic as well: `_cleanup_infra.sh` removes
the `eval-`-prefixed operator resources for all providers, so a run cleaned
up with a different `EVAL_PROVIDER` than it was set up with still tears down
correctly. Set `EVAL_DELETE_NAMESPACE=1` to also delete the test namespace
during cleanup — useful for a full reset so no namespace-scoped state
(quotas, LimitRanges, leftover secrets) carries over into the next run.
`OPERATOR_NS` and `TEST_NS` may be overridden via the environment for
nonstandard installations.

Setup scripts wait for each fixture's fault to actually be observable
(container state reasons, Unschedulable conditions, admission rejections —
see `_wait_helpers.sh`) before the evaluation starts, so proposals are never
created against a cluster that does not yet exhibit the described symptom.
Judge thresholds come from the system config (see the
`custom:proposal_evaluation_correctness` metric section above) — the
scenario files intentionally carry none. Note the fixtures pull
`docker.io/nginxinc/nginx-unprivileged`; on clusters behind a shared NAT
egress IP, Docker Hub's anonymous pull limits can flake large runs —
mirror the image to an internal registry and adjust the fixtures if that
applies to you.

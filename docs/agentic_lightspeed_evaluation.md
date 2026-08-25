# Agentic Lightspeed Evaluation

This guide covers how to evaluate event-driven agentic workflows using the Lightspeed Evaluation Framework. While the framework's default mode evaluates synchronous HTTP request-response interactions, the agentic evaluation mode supports CRD-based workflows where the "answer" is a trajectory of events and a final cluster state.

## Overview

OpenShift Agentic Lightspeed systems is event-driven: AgenticRun CRDs are applied, workflows are executed, and cluster state changes. The evaluation framework supports an `openshift_agentic_run` agent type to monitor the cluster state and evaluate agent results against it.

## Prerequisites

- OpenShift cluster with the Agentic Lightspeed operator installed
- `oc` or `kubectl` CLI available in PATH
- `KUBECONFIG` environment variable pointing to a valid kubeconfig
- RBAC permissions for AgenticRun CRD operations in the target namespace
- Judge LLM API key (e.g., `OPENAI_API_KEY`) for `openshift_agentic_run_evaluation_correctness`

## Configuration

```yaml
agents:
  enabled: true

  default:
    agent: openshift_agentic_lightspeed
    agent_config:
      timeout: 600

  openshift_agentic_lightspeed:
    type: openshift_agentic_run
    namespace: openshift-lightspeed
    agent_ref: default
    auto_approve: true
    cleanup_openshift_agentic_runs: true
    timeout: 900
    poll_interval: 2
```

### OpenshiftAgenticRun Agent Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `namespace` | string | *(required)* | Kubernetes namespace containing AgenticRun resources |
| `agent_ref` | string | `null` | Name of the Agent CR on the cluster — injected into all stages defined in eval data |
| `auto_approve` | bool | `true` | Automatically approve AgenticRuns when phase is Proposed |
| `cleanup_openshift_agentic_runs` | bool | `true` | Delete eval AgenticRun CRs after status is captured |
| `timeout` | int | `900` | Total timeout in seconds for the AgenticRun lifecycle |
| `cli_timeout` | int | `30` | Timeout in seconds for individual oc/kubectl commands |
| `poll_interval` | int | `2` | Seconds between status polls |
| `cache_dir` | string | `null` | Location of cached queries |
| `cache_enabled` | bool | `true` | Enable caching |

#### `agent_ref` and NxM evaluation

For HTTP API agents, the model/provider is a config-level field — different agent configs naturally produce different runs. For agentic evaluation, the model is determined by the Agent CR on the cluster, and the Agent CR name lives in the eval data spec (`analysis.agent`, `execution.agent`, etc.).

`agent_ref` bridges this gap: when set, it overrides the agent name in all stages defined in the eval data. This enables NxM behavioral evaluation with different cluster-side Agent CRs:

```yaml
agents:
  default:
    agent: [eval_fast, eval_smart]
    repeat: 3
  eval_fast:
    type: openshift_agentic_run
    namespace: openshift-lightspeed
    agent_ref: fast-agent
  eval_smart:
    type: openshift_agentic_run
    namespace: openshift-lightspeed
    agent_ref: smart-agent
```

**Override rules:**

- `agent_ref` set, stage defined in eval data — config overrides eval data's agent
- `agent_ref` not set, stage defined in eval data — eval data's agent is used as-is
- Stage not defined in eval data — `agent_ref` does not inject the stage

Note: `agent_ref` overriding eval data is an inversion of the normal pattern (where eval data overrides config). This is intentional — eval data agent names are placeholders for stage selection, while `agent_ref` represents the actual agent choice. The CRD spec couples stage selection with agent selection in the same field, so the config override is the pragmatic way to separate them for NxM.

### Turn Data Structure

For agentic workflows, each turn uses `openshift_agentic_run_spec` to define the AgenticRun and `expected_openshift_agentic_run_status` to define success criteria.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | No | Human-readable label for reports (falls back to `query`) |
| `openshift_agentic_run_spec` | dict | Conditional | Inline AgenticRun spec — contains `request`, `targetNamespaces`, workflow phase gates |
| `expected_openshift_agentic_run_status` | dict | Conditional | Assertions to check against the AgenticRun status |
| `expected_outcome` | string | Conditional | Expected outcome description for LLM-as-judge evaluation |
| `expected_analysis_outcome` | string | No | Optional per-phase expected outcome for analysis/diagnosis |
| `expected_execution_outcome` | string | No | Optional per-phase expected outcome for execution/actions |
| `expected_verification_outcome` | string | No | Optional per-phase expected outcome for verification |
| `openshift_agentic_run_status` | dict | No | Raw CRD status populated by the driver (framework-managed) |
| `openshift_agentic_run_results` | dict | No | Child Result CRs populated by OpenshiftAgenticRunAmender (framework-managed) |

> `query` remains required but can be auto-populated from `openshift_agentic_run_spec.request` when absent.

### Example: Analysis-Only Workflow

The simplest agentic evaluation — analysis phase only, no execution or verification:

```yaml
- conversation_group_id: analysis_only
  description: Analysis-only — diagnose without remediating
  setup_script: agentic/scripts/setup.sh
  cleanup_script: agentic/scripts/cleanup.sh
  turns:
    - turn_id: turn_1
      openshift_agentic_run_spec:
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
      expected_openshift_agentic_run_status:
        max_duration: "15m"
        phase: Completed
      turn_metrics:
        - custom:openshift_agentic_run_status
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
      openshift_agentic_run_spec:
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
      expected_openshift_agentic_run_status:
        phase: Completed
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
        - custom:openshift_agentic_run_status
        - custom:openshift_agentic_run_evaluation_correctness
      turn_metrics_metadata:
        "custom:openshift_agentic_run_evaluation_correctness":
          threshold: 0.75
```

## AgenticRun Lifecycle

The `openshift_agentic_run` driver manages the full AgenticRun CR lifecycle:

1. **Build AgenticRun CR** — Merge `openshift_agentic_run_spec` + `request` + agent config
2. **Create CR on cluster** — Auto-generated name: `eval-<uuid8>`
3. **Poll status** — Loop every `poll_interval` seconds
4. **Auto-approve** — If phase is `Proposed` and `auto_approve` is enabled
5. **Terminal phase** — `Completed` / `Failed` / `Denied` / `Escalated`
6. **Populate turn_data** — `openshift_agentic_run_status` (full status dict) + `openshift_agentic_run_results` (child Result CRs) + `response` (Markdown workflow summary)
7. **Cleanup AgenticRun CR** — Delete the created CR (if `cleanup_openshift_agentic_runs` is enabled)
8. **Metrics evaluate** — `custom:openshift_agentic_run_status` and/or `custom:openshift_agentic_run_evaluation_correctness` on enriched data

Setup/cleanup scripts are only needed for **infrastructure** (deploying the workload to trigger, LLM provider CRs, sandbox CRs, etc.). The driver handles AgenticRun CR lifecycle autonomously.

## Metrics

### `custom:openshift_agentic_run_status` — Deterministic Assertions

A single metric that runs all assertion checks from `expected_openshift_agentic_run_status` in sequence, failing fast at the first failure. Score is `1.0` if all checks pass, `0.0` on first failure.

Checks run in order: **phase → timing → analysis → execution → verification**.

#### `expected_openshift_agentic_run_status` Reference

**Phase checks:**

| Field | Type | Description |
|-------|------|-------------|
| `phase` | string | Exact phase match (e.g., `Completed`, `Failed`, `Escalated`) |
| `phase_in` | list[string] | Phase must be one of these values |

**Timing checks:**

| Field | Type | Description |
|-------|------|-------------|
| `max_duration` | string | Max elapsed time across conditions. Go-style duration: `"5m"`, `"2m30s"`, `"1h"` |

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


### `custom:openshift_agentic_run_evaluation_correctness` — LLM-as-Judge

Evaluates agentic remediation workflow quality using a Judge LLM. Scores 0.0–1.0 across three dimensions (only phases present in the workflow are scored; absent dimensions are marked N/A):

1. **Diagnosis** — Is the root cause correctly identified? Are the proposed actions sound and safe?
2. **Execution** — Were the remediation actions carried out? Are they safe, well-scoped, and minimal?
3. **Verification** — Do the checks confirm the specific issue was resolved?

**Threshold:** 0.75

**Required fields:** `response` (populated automatically during execution), `expected_outcome`

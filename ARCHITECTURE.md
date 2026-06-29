# Architecture

Lightspeed Evaluation Framework is a Python-based evaluation system for LLM-powered applications. It evaluates responses, context quality, tool calls, conversation flows, and agentic workflow outcomes — using both live (API/agent-driven) and offline (pre-populated) data. It supports multiple evaluation backends (Ragas, DeepEval, NLP, custom, script-based), user-defined evaluation criteria, panel-of-judges scoring, statistical analysis, environment setup/cleanup scripts, and pluggable agent drivers. Conversations and turns are defined in YAML, scored against configurable metrics using LLM judges, and results are produced as reports.

## System Overview

The framework separates **what to evaluate** (evaluation data) from **how to evaluate** (system configuration). Users provide two YAML files: one defining conversations and turns to evaluate, and one configuring judges, metrics, storage, and infrastructure.

```mermaid
graph TD
    subgraph Input
        SC[System Config YAML]
        ED[Evaluation Data YAML]
    end

    subgraph Framework
        CLI[CLI / API Entry Point]
        Pipeline[EvaluationPipeline]
        Processor[ConversationProcessor]
        Evaluator[MetricsEvaluator]
        Judges[JudgeOrchestrator]
    end

    subgraph External
        LLM[LLM Providers<br/>OpenAI, Azure, Anthropic,<br/>Gemini, WatsonX, etc.]
        Agent[Agent APIs<br/>HTTP endpoints]
        K8s[OpenShift<br/>Proposal CRDs]
    end

    subgraph Output
        Reports[CSV / JSON / TXT Reports]
        Graphs[Visualizations]
        DB[(Database<br/>SQLite / PostgreSQL / MySQL)]
        LF[Langfuse<br/>Scores]
    end

    SC --> CLI
    ED --> CLI
    CLI --> Pipeline
    Pipeline -->|concurrent| Processor
    Processor --> Evaluator
    Evaluator --> Judges
    Judges --> LLM
    Processor --> Agent
    Processor --> K8s
    Pipeline --> Reports
    Pipeline --> Graphs
    Pipeline --> DB
    Pipeline --> LF
```

## Evaluation Flow

A single evaluation run follows this path:

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as EvaluationPipeline
    participant Proc as ConversationProcessor
    participant Driver as AgentDriver
    participant Eval as MetricsEvaluator
    participant Judge as JudgeOrchestrator
    participant Storage

    User->>Pipeline: new EvaluationPipeline(system_config)
    Pipeline->>Pipeline: Create components (LLM, metrics, storage, drivers)
    User->>Pipeline: run_evaluation(evaluation_data)

    loop Each conversation (concurrent)
        Pipeline->>Proc: process_conversation()
        opt Setup scripts configured
            Proc->>Proc: Run setup scripts
        end

        loop Each turn (sequential)
            opt Agent driver configured
                Proc->>Driver: execute_turn()
                Driver-->>Proc: Response + tokens + latency
            end

            loop Each metric
                Proc->>Eval: evaluate_metric()
                alt Panel of judges active
                    Eval->>Judge: evaluate_with_judges()
                    Judge-->>Eval: Aggregated score
                else Single judge
                    Eval->>Eval: Call framework handler
                end
                Eval-->>Proc: MetricResult (score, status, tokens)
            end
        end

        Pipeline->>Storage: save_run(results)
    end

    Pipeline->>Storage: finalize() → generate reports
    Pipeline-->>User: Results + summary
```

## Component Architecture

The framework is organized in three layers:

```mermaid
graph TB
    subgraph Runner Layer
        CLI[runner/evaluation.py<br/>CLI entry point]
        API[api.py<br/>Programmatic API]
    end

    subgraph Pipeline Layer
        EP[EvaluationPipeline<br/>Orchestrator]
        CP[ConversationProcessor<br/>Per-conversation logic]
        ME[MetricsEvaluator<br/>Metric dispatch]
        JO[JudgeOrchestrator<br/>Multi-judge scoring]
        AD[AgentDriver<br/>HttpApiDriver + ProposalDriver]
    end

    subgraph Core Layer
        MM[MetricManager<br/>Resolution + registration]
        LM[LLMManager<br/>Provider abstraction]
        EM[EmbeddingManager<br/>Semantic similarity]
        SM[ScriptExecutionManager<br/>Setup/cleanup scripts]
        OH[OutputHandler<br/>Report generation]
        SB[StorageBackend<br/>Persistence]
        Models[Pydantic Models<br/>SystemConfig, EvaluationData]
    end

    CLI --> EP
    API --> EP
    EP --> CP
    CP --> ME
    CP --> AD
    ME --> JO
    ME --> MM
    JO --> LM
    ME --> LM
    ME --> EM
    CP --> SM
    EP --> SB
    SB --> OH
    EP --> Models
```

## Metric Evaluation

The framework routes metrics to backend-specific handlers based on prefix (`ragas:`, `deepeval:`, `nlp:`, etc.). Each handler wraps its upstream library and normalizes results to a common score/status model.

```mermaid
graph LR
    subgraph MetricsEvaluator
        Dispatch[Prefix-based<br/>dispatch]
    end

    subgraph "LLM-backed (require judge)"
        Ragas[Ragas]
        DeepEval[DeepEval]
        UserDefined[User-Defined Criteria<br/>via DeepEval GEval]
        Custom[Custom]
    end

    subgraph "No-LLM"
        NLP[NLP]
        Script[Script]
    end

    Dispatch --> Ragas
    Dispatch --> DeepEval
    Dispatch --> UserDefined
    Dispatch --> Custom
    Dispatch --> NLP
    Dispatch --> Script

    Ragas --> LLM[LLM Judge]
    DeepEval --> LLM
    UserDefined --> LLM
    Custom --> LLM
    Script --> Ext[External Process]
```

## Judge Panel

When configured, multiple LLMs independently score each metric and results are aggregated:

```mermaid
graph TD
    ME[MetricsEvaluator] --> JO[JudgeOrchestrator]

    JO --> J0[Judge 0<br/>LLMManager + isolated cache]
    JO --> J1[Judge 1<br/>LLMManager + isolated cache]
    JO --> J2[Judge N<br/>LLMManager + isolated cache]

    J0 --> Agg[Aggregation Strategy]
    J1 --> Agg
    J2 --> Agg

    Agg -->|max| Max[Highest score]
    Agg -->|average| Avg[Mean score]
    Agg -->|majority_vote| Maj[Majority PASS/FAIL<br/>+ mean of all valid judges]

    Agg --> Result[Final MetricResult]
```

## Configuration Resolution

Metric metadata (thresholds, criteria, weights) cascades through three levels, with the most specific level winning:

```mermaid
graph TD
    SD[System Defaults<br/>default_*_metrics_metadata] --> CO[Conversation Overrides<br/>per-conversation metadata]
    CO --> TO[Turn Overrides<br/>per-turn metadata]
    TO --> Resolved[Resolved Metric Config<br/>threshold, criteria, model, etc.]

```

## Storage Lifecycle

Three storage backends share the same lifecycle but implement it differently:

```mermaid
stateDiagram-v2
    [*] --> Initialize: Pipeline starts
    Initialize --> SaveRun: Per conversation
    SaveRun --> SaveRun: Next conversation
    SaveRun --> SetContext: All conversations done
    SetContext --> Finalize: Full dataset provided
    Finalize --> Close: Resources released
    Close --> [*]
```

```mermaid
graph LR
    subgraph "File Storage (deferred)"
        FS_SR[save_run] -->|accumulate in memory| FS_F[finalize]
        FS_F -->|write all reports| Disk[CSV / JSON / TXT / Graphs]
    end

    subgraph "SQL Storage (incremental)"
        SQL_SR[save_run] -->|commit immediately| DB[(Database)]
        SQL_F[finalize] -->|no-op| Log[Log count]
    end

    subgraph "Langfuse Storage (deferred — will be made incremental)"
        LF_SR[save_run] -->|accumulate| LF_F[finalize]
        LF_F -->|write scores| LFP[Langfuse]
    end
```

## Key Architectural Decisions

**Separation of system config and evaluation data.** The system config (judges, metrics, infrastructure) changes infrequently. The evaluation data (conversations, turns) changes per run. Keeping them in separate files lets teams share a system config across many evaluation datasets.

**Metric resolution hierarchy.** Turn-level overrides > conversation-level overrides > system defaults. This lets users tune thresholds or criteria for specific test cases without duplicating the full config.

**Pluggable agent drivers.** The framework operates in two modes: live (agent drivers collect responses then evaluate) and offline (evaluate pre-populated data). Two driver types exist — `http_api` for HTTP API calls and `proposal` for OpenShift CRD-based proposal workflows. The driver registry pattern makes it straightforward to add new driver types.

**Concurrent conversations, sequential turns.** Conversations are independent and can be evaluated in parallel. Turns within a conversation are sequential because they may depend on prior context (multi-turn conversations).

**Storage lifecycle pattern.** Initialize → save per conversation → finalize → close. This enables incremental persistence (each conversation saves immediately) while deferring expensive report generation to the end when all results are available.

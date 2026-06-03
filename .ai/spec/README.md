# LightSpeed Evaluation Framework — Specifications

A comprehensive evaluation framework for GenAI applications that supports multiple evaluation backends (Ragas, DeepEval, GEval, NLP, custom, script), multi-turn conversation assessment, panel-of-judges scoring, and pluggable agent drivers for real-time inference.

## Structure

| Layer | Path | Purpose |
|---|---|---|
| **what/** | `.ai/spec/what/` | Behavioral rules. What the system must do. Implementation-agnostic. |
| **how/** | `.ai/spec/how/` | Codebase navigation. How the code is organized. Implementation-specific. |

## Scope

Covers the `src/lightspeed_evaluation/` package — the active evaluation framework. Out of scope: `lsc_agent_eval/` (deprecated, being consolidated), `src/generate_answers/` (moving to separate repo), `dashboard/` (visualization frontend).

## Audience

AI agents. Content is optimized for precision and machine consumption.

## Quick Start

| Task | Start here |
|---|---|
| Understand the system | `what/system-overview.md` |
| Understand the evaluation pipeline | `what/evaluation-pipeline.md` |
| Understand metrics | `what/metrics.md` |
| Understand LLM and judge configuration | `what/llm-and-judges.md` |
| Understand agent drivers | `what/agent-drivers.md` |
| Understand agent driver implementation | `how/agent-drivers.md` |
| Navigate the source code | `how/project-structure.md` |
| Understand metric implementation | `how/metrics-implementation.md` |
| Understand configuration loading | `how/configuration-and-models.md` |
| Understand output and storage | `how/output-and-storage.md` |

## Cross-Reference

| what/ | how/ |
|---|---|
| `what/system-overview.md` | `how/project-structure.md` |
| `what/evaluation-pipeline.md` | `how/project-structure.md` (pipeline section) |
| `what/metrics.md` | `how/metrics-implementation.md` |
| `what/llm-and-judges.md` | `how/configuration-and-models.md` |
| `what/agent-drivers.md` | `how/agent-drivers.md` |
| `what/output-and-reporting.md` | `how/output-and-storage.md` |

## Conventions

- **Rule numbering:** behavioral rules are numbered sequentially within each what/ file.
- **Planned changes:** unimplemented behavior is marked with `[PLANNED]` or `[PLANNED: TICKET-XXXX]` inline next to the rule it affects.
- **Constraints:** component-specific and cross-cutting constraints go in the relevant what/ file's Constraints section, co-located with behavioral rules. Development conventions go in CLAUDE.md.
- **Authority:** what/ specs are authoritative for behavior. how/ specs are authoritative for implementation. When they conflict, what/ wins.
- **When to create a new file vs. extend an existing one:** if the new concern has its own lifecycle, configuration surface, and can be understood independently, it gets its own file. If it's a capability added to an existing component, it goes in that component's file.

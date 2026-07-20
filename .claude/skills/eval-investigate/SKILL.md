---
name: eval-investigate
description: Use when eval-sanity has flagged issues and user wants deeper analysis of actual evaluation data - response content, agent behavior, failure patterns, cross-model comparison. Requires explicit user permission to read raw data.
---

# Eval Investigate

Deep qualitative analysis of evaluation results. Reads raw evaluation data to find root causes behind issues flagged by `eval-sanity`.

## Prerequisites

`eval-sanity` must be run first. If no sanity output is available in the conversation, say: "Run `/eval-sanity <path>` first to identify what to investigate." and stop.

## Permission Check

Before reading any files, display:

> This will read raw response data, tool calls, and contexts from the detailed results. Continue? (yes/no)

Wait for explicit confirmation. If declined, stop.

## Invocation

Expects an output directory path as argument. If not provided, ask for it. If the directory contains subdirectories with separate product results, detect and include them for cross-product comparison.

If multiple runs exist, match files by shared timestamp prefix. If ambiguous, ask.

## Files

- `*_detailed.csv` — row-level results with scores, responses, tool calls, contexts
- `eval-sanity` output from the conversation — which checks flagged WARN and on which metrics/models

Check CSV headers before analyzing — columns like `tool_calls` and `contexts` are configurable and may not be present. Note any missing columns.

For large files, check size first and sample strategically rather than reading everything. When sampling, label statistics as estimates and disclose the sample size.

## Scope

Start with rows related to flagged metrics/models from sanity output. Expand only if the user asks.

## Analysis Areas

These are starting points, not a checklist — investigate whatever the data reveals.

**Response Content** — Scan responses for flagged metrics/models:
- Service errors, refusal patterns, empty/very short responses
- Repeated identical responses
- Report count and percentage per pattern

**Agent Behavior** (when tool_calls present):
- Repeated failed tool calls with same/similar args
- Malformed tool names or unexpected tool call patterns
- Missing expected tool calls vs `expected_tool_calls`

**Failure Clustering** — Group low-scoring results:
- Common traits across failures (query type, topic, complexity)
- Context quality correlation with answer scores
- Model-specific vs data-specific patterns

**Cross-Model Comparison** (when multiple models in the run):
- Which model(s) drive the anomalies?
- Model capability issue vs upstream data/service issue?
- Pass rates per model on the same query set

**Cross-Product Comparison** (when multiple product subdirectories found):
- Compare failure patterns across products for the same model
- Distinguish product-specific issues (data/service) from model-wide issues
- Identify which product's data or service is driving anomalies flagged in sanity

## Output

For each finding:

```
### <Analysis Area>
- <pattern found with counts>
  → <why it matters — impact on reported scores>
  → **Action:** <concrete next step>
```

Report all patterns discovered. The data will surface its own issues — don't limit to predefined scenarios.

Print the report in the conversation and save to the output directory.

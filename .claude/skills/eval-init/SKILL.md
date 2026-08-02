---
name: eval-init
description: Use when starting a new evaluation or modifying an existing config. Guided setup that generates system.yaml + evaluation_data.yaml — recommends metrics, judges, and config based on evaluation goals.
---

# eval-init: Guided Evaluation Setup

Generate working `system.yaml` + `evaluation_data.yaml` through advisory Q&A. Ask about intent, recommend config, explain why. One question at a time.

## Invocation

`/eval-init [target-directory]` — defaults to current directory.

## Step 1: Detect Existing Configs

Check target for `system.yaml` and `evaluation_data.yaml`:
- **Both exist:** ask fresh or modify? If modify: "What do you want to change?" — freeform targeted edits, don't re-walk questionnaire.
- **One exists:** read it, generate the missing file to match.
- **Neither:** continue to Step 2.

## Step 2: Ask "Quick start or full setup?"

- **Quick start:** fewer questions, sensible defaults.
- **Full setup:** walks through all major config sections.

## Step 3: Questions

Frame as outcomes, not framework terminology. One at a time.

**Q1 — What are you evaluating?** RAG responses, agent tool calls, multi-turn conversations, or proposals. Explain if unclear.

**Q2 — What do you want to measure?** Ask about goals, recommend metrics:

| Goal | Metrics | Eval data fields needed |
|------|---------|------------------------|
| Accuracy | `custom:answer_correctness` | response, expected_response |
| Hallucination | `ragas:faithfulness` | response, contexts |
| Relevance | `ragas:response_relevancy` | response *(needs embedding config)* |
| Context quality | `ragas:context_precision`, `ragas:context_recall` | response, contexts, expected_response |
| Context relevance | `ragas:context_relevance`, `ragas:context_utilization` | response, contexts |
| Tool correctness | `custom:tool_eval` | tool_calls, expected_tool_calls |
| Keywords | `custom:keywords_eval` | response, expected_keywords |
| Intent | `custom:intent_eval` | response, expected_intent |
| Text similarity | `nlp:bleu`, `nlp:rouge`, `nlp:semantic_similarity_distance` | response, expected_response |
| Conversation quality | `deepeval:conversation_completeness`, `deepeval:conversation_relevancy` | response (multi-turn) |
| Custom criteria | `geval:<name>` — ask user to describe criteria | response |
| Infrastructure | `script:action_eval` *(live mode only)* | verify_script |
| Proposals | `custom:proposal_status` | proposal_spec, expected_proposal_status |
| Proposal correctness | `custom:proposal_evaluation_correctness` | response, expected_outcome |

**Q3 — Data source?** Offline (data in YAML) vs live (API/agent calls). If live: ask agent type (`http_api`/`proposal`) + endpoint, then ask single setup vs multi-config comparison (multi-config requires live mode). If user selected `script:action_eval` or proposal metrics, live mode is required.

**Q4 — LLM provider?** openai/anthropic/gemini/watsonx/hosted_vllm/azure/vertex/ollama. Add env var comment with the correct variables for the chosen provider.

**Q5 (full only) — Scoring reliability?** Single LLM (faster) vs multiple LLMs scoring independently (more reliable). If multi: ask models + aggregation (average/max/majority_vote).

**Q6 (full only) — Storage?** Default file. Mention sqlite/postgres/langfuse/mlflow briefly.

## Step 4: Generate and Present

Show brief summary, then both YAMLs. **Confirm before writing to disk.**

### Config rules

- Set `core.max_threads: 10` as recommended starting value. Include `quality_score:` section to aggregate metrics into an overall score
- Every metric listed in `quality_score.metrics` must have a corresponding entry in `metrics_metadata:`
- Prefer `llm_pool:` + `models:` over legacy `llm:`, `storage:` list over `output:`, `agents:` over `api:`
- Add `embedding:` only for `ragas:response_relevancy`
- Add `environment: { LITELLM_LOG: ERROR }` only for LLM/ragas metrics
- Set `agents.enabled: false` for offline
- Add `judge_panel:` only if multi-judge selected
- For `geval:` criteria, add under `metrics_metadata:` → `turn_level:` → `geval:<name>:` with `evaluation_params`
- Eval data: OpenShift domain examples, 2-3 turns, only fields needed for chosen metrics
- Live mode: comment out API-populated fields (`# response: populated by agent at runtime`)

## Step 5: Write and Suggest Next Step

```bash
uv run lightspeed-eval --system-config <path>/system.yaml --eval-data <path>/evaluation_data.yaml
```

Tip: `--tags`, `--conv-ids`, `--metrics` for filtering. `--help` for all options.

## Metric Reference

Source of truth: `METRIC_REQUIREMENTS` in `src/lightspeed_evaluation/core/system/validator.py` and `supported_metrics` on each metric class (`RagasMetrics`, `CustomMetrics`, `NLPMetrics`, `DeepEvalMetrics`). See `examples/` for working configs.

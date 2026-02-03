# LightSpeed Evaluation Framework: Complete Guide

**Last Updated:** December 23, 2025  
**Assisted by:** AI to generate the document

---

## Table of Contents

### Part 1: Introduction & Fundamentals
1. [Introduction](#1-introduction)
2. [Understanding AI Evaluation](#2-understanding-ai-evaluation)

### Part 2: Evaluation Methodologies
3. [Methodologies Overview](#3-methodologies-overview)
4. [Turn-Level Metrics (Single Q&A)](#4-turn-level-metrics)
5. [Conversation-Level Metrics](#5-conversation-level-metrics)
6. [Metric Selection Guide](#6-metric-selection-guide)

### Part 3: Practical Implementation
7. [Step-by-Step Setup](#7-step-by-step-setup)
8. [Configuration Guide](#8-configuration-guide)
9. [Running Evaluations](#9-running-evaluations)
10. [Understanding Results](#10-understanding-results)

### Part 4: Real-World Application
11. [Common Use Cases](#11-common-use-cases)
12. [Best Practices](#12-best-practices)
13. [Troubleshooting](#13-troubleshooting)

### Part 5: Reference Materials
14. [Quick Reference Tables](#14-quick-reference-tables)
15. [Resources & Links](#15-resources--links)

---

# Part 1: Introduction & Fundamentals

## 1. Introduction

The LightSpeed Evaluation Framework is a comprehensive system designed to evaluate AI-powered applications, particularly conversational AI systems and chatbots. This guide explains everything you need to know to evaluate your AI applications effectively—all without requiring deep technical or data science expertise.

### What This Framework Does

Think of this framework as a quality control system for AI applications. Just as you might test a website to ensure all links work and pages load correctly, this framework tests AI systems to ensure they:
- Provide accurate and relevant answers
- Use correct information from their knowledge base
- Maintain context across conversations
- Call the right tools or functions when needed
- Perform expected actions in the system

### Who Should Use This Guide

- **Product Managers**: Understanding evaluation metrics to make informed decisions
- **QA Engineers**: Testing AI applications systematically
- **Application Developers**: Integrating evaluation into development workflows
- **Technical Writers**: Documenting AI application quality
- **Team Leads**: Overseeing AI application quality assurance

---

## 2. Understanding AI Evaluation

### Why Evaluate AI Applications?

Unlike traditional software where behavior is deterministic (same input always produces same output), AI applications can produce varied responses. Evaluation helps ensure:

1. **Quality Assurance**: Responses meet quality standards
2. **Consistency**: Similar questions get consistent treatment
3. **Safety**: Responses don't include harmful or incorrect information
4. **Performance Tracking**: Monitor improvements or regressions over time
5. **Compliance**: Meet organizational standards and requirements

### Two Levels of Evaluation

#### Turn-Level Evaluation (Single Query-Response)
- Evaluates individual question-answer pairs
- Like checking if a single customer support ticket was handled correctly
- **Example**: "Was the answer to 'How do I reset my password?' accurate and helpful?"

#### Conversation-Level Evaluation (Multi-Turn Dialogue)
- Evaluates entire conversations with multiple back-and-forth exchanges
- Like reviewing a complete customer support conversation
- **Example**: "Did the AI successfully guide the user through troubleshooting across 5 messages?"

---

## 3. Quick Start

### Installation

```bash
# Navigate to project directory
cd lightspeed-evaluation
# Install dependencies
uv sync
# OR using pip
pip install -e .
```

### Set Environment Variables

```bash
# Required: Judge LLM (the AI that evaluates your AI)
export OPENAI_API_KEY="sk-your-api-key-here"
# Optional: For live API testing
export API_KEY="your-api-endpoint-key"
```

### Run Your First Evaluation

```bash
lightspeed-eval \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml
```

That's it! Results will be in `eval_output/` directory.

---

# Part 2: Evaluation Methodologies

## 3. Methodologies Overview

The framework uses **four main categories** of evaluation methods:

| Category | What It Does | When to Use | Level |
|----------|-------------|-------------|-------|
| **Ragas Metrics** | Industry-standard metrics for response and context quality | RAG QnA, Single-turn responses | Turn |
| **DeepEval Metrics** | Advanced conversation analysis | Multi-turn conversations | Conversation |
| **Custom Metrics** | Specialized evaluations for specific needs | Intent checking, tool validation | Turn |
| **Script-Based Metrics** | Real-world validation through automated scripts | E2E RAG/agent workflows | Turn |

### Quick Selection Guide

**Choose Ragas Metrics when:**
- You want to verify if answers are accurate and relevant
- You need to check if the AI is using the right information
- You want industry-standard, well-documented metrics

**Choose DeepEval Metrics when:**
- You're evaluating multi-turn conversations
- You need to assess conversation completeness
- You want to check if the AI remembers earlier parts of the conversation

**Choose Custom Metrics when:**
- You have specific requirements not covered by standard metrics
- You need to compare against expected answers
- You want to verify the AI's intent or tool usage

**Choose Script-Based Metrics when:**
- Your AI performs actions in real systems
- You need to verify real-world outcomes
- You want to test end-to-end workflows

---

## 4. Turn-Level Metrics

Turn-level metrics evaluate individual question-answer pairs.

### 4.1 Ragas Metrics

#### A. Response Quality Metrics

##### Response Relevancy

**What it measures:** How well does the answer address the actual question?

**Plain English:** "Did the AI answer the question I asked, or did it go off-topic?"

**Score Range:** 0.0 to 1.0 (higher is better)

**Example:**
```
Question: "How do I reset my password?"

✓ Relevant (High Score):
"Click on 'Forgot Password' on the login page, enter your email,
and follow the reset link sent to you."

✗ Irrelevant (Low Score):
"Our system has been running for 5 years and we have excellent
security features including two-factor authentication."
```

**When to use:** Ensuring the AI stays on topic

**Threshold:** 0.8 or higher

**Required fields:** `query`, `response`

---

##### Faithfulness

**What it measures:** Does the answer stick to the facts provided in the source information?

**Plain English:** "Is the AI making things up, or is it only using information from its knowledge base?"

**Score Range:** 0.0 to 1.0 (higher is better)

**Example:**
```
Context: "OpenShift Virtualization requires 4GB RAM minimum."
Question: "What are OpenShift Virtualization requirements?"

✓ Faithful (High Score):
"OpenShift Virtualization requires a minimum of 4GB RAM."

✗ Not Faithful (Low Score):
"OpenShift Virtualization requires 8GB RAM and 100GB disk space."
(The disk space wasn't in the context - made up!)
```

**When to use:** Preventing AI hallucinations (making up information)

**Threshold:** 0.8 or higher

**Required fields:** `response`, `contexts`

---

#### B. Context/Retrieval Quality Metrics

##### Context Recall

**What it measures:** Did the AI retrieve all the necessary information to answer the question?

**Plain English:** "Did the AI look up everything it needed to give a complete answer?"

**Score Range:** 0.0 to 1.0 (higher is better)

**Example:**
```
Question: "What are the storage and memory requirements for OpenShift?"
Expected Answer mentions: 120GB storage AND 16GB RAM

Retrieved Context contains:
- Document about storage (120GB) ✓
- (Missing document about memory requirements) ✗

Context Recall: 0.5 (retrieved 1 out of 2 needed pieces)
```

**When to use:** Improving search/retrieval systems

**Threshold:** 0.8 or higher

**Required fields:** `contexts`, `expected_response`

---

##### Context Precision

**What it measures:** How much of the retrieved information is actually useful?

**Plain English:** "Is the AI pulling up relevant documents, or is it cluttering the answer with unnecessary information?"

**Two variants:**
- **Without Reference**: Uses AI's response to judge relevance
- **With Reference**: Uses expected answer for more accurate judgment

**Score Range:** 0.0 to 1.0 (higher is better)

**When to use:** Optimizing search algorithms, reducing noise

**Threshold:** 0.7 or higher

**Required fields:** `query`, `contexts`, `response` (and `expected_response` for "with reference" variant)

---

##### Context Relevance

**What it measures:** How relevant is the retrieved context to the user's question?

**Plain English:** "Is the information the AI found actually related to what the user asked?"

**Score Range:** 0.0 to 1.0 (higher is better)

**When to use:** Evaluating search quality before answer generation

**Threshold:** 0.7 or higher

**Required fields:** `query`, `contexts`

---

### 4.2 Custom Metrics

#### Answer Correctness

**What it measures:** How close is the AI's answer to the expected "correct" answer?

**Plain English:** "On a test where we know the right answer, how well did the AI do?"

**Score Range:** 0.0 to 1.0 (higher is better)

**How it works:** A Judge LLM compares the AI's response to your expected response

**Example:**
```
Question: "What is the capital of France?"
Expected Response: "Paris"

AI Response: "The capital of France is Paris."
Score: 1.0 (Perfect match)

AI Response: "Lyon is a major city in France."
Score: 0.1 (Incorrect answer)
```

**When to use:** Testing against known question-answer pairs, benchmarking

**Threshold:** 0.75 or higher

**Required fields:** `query`, `response`, `expected_response`

---

#### Intent Evaluation

**What it measures:** Does the AI's response have the right intent/purpose?

**Plain English:** "Is the AI trying to do what we expect it to do?"

**Score:** Binary (0 or 1)

**Intent Categories:**
- **Explain a concept**: "What is Kubernetes?" → Expects explanatory response
- **Provide instructions**: "How do I install Docker?" → Expects step-by-step guide
- **Refuse/Decline**: "Can you hack this system?" → Expects refusal
- **Ask for clarification**: Ambiguous question → Expects clarifying questions

**Example:**
```
Question: "Tell me a joke about programming"
Expected Intent: "refuse" (professional support bot should decline)

✓ Correct Intent (Score: 1):
"I apologize, but I'm designed to help with technical questions
about OpenShift. How can I assist you today?"

✗ Wrong Intent (Score: 0):
"Why do programmers prefer dark mode? Because light attracts bugs!"
```

**When to use:** Ensuring appropriate AI behavior, safety checking

**Threshold:** 1 (must match exactly)

**Required fields:** `query`, `response`, `expected_intent`

---

#### Tool Evaluation

**What it measures:** Does the AI call the right tools with correct parameters and get expected results?

**Plain English:** "When the AI needs to use a tool, did it use the right one with the right settings, and did the tool return what we expected?"

**Score:** Binary (0 or 1)

**How it works:**
- Compares expected tool calls against actual tool calls
- Validates tool names match exactly
- Checks parameters (supports regex patterns)
- Optionally validates tool call results (supports regex patterns)

**Example:**
```
Question: "Show me all pods in the default namespace"

Expected Tool Call:
- Tool: oc_get
- Parameters: {kind: "pod", namespace: "default"}

✓ Correct (Score: 1):
Tool: oc_get, Parameters: {kind: "pod", namespace: "default"}

✗ Incorrect (Score: 0):
Tool: oc_describe, Parameters: {kind: "pod", namespace: "default"}
(wrong tool)
```

**Pattern Matching:**
```yaml
# Regex support for flexible matching
expected_tool_calls:
  - - tool_name: oc_get
      arguments:
        namespace: "openshift-light.*"  # Matches openshift-lightspeed
```

**Result Validation (Optional):**
```yaml
# Validate tool call results using regex patterns
expected_tool_calls:
  - - tool_name: oc_get
      arguments:
        kind: pod
        namespace: default
      result: ".*Running.*"  # Verify pod is in Running state

  - - tool_name: oc_create
      arguments:
        kind: namespace
        name: test-ns
      result: ".*created"  # Verify creation succeeded
```

**When to use:** Function calling AI applications, tool-using agents, validating tool outputs

**Threshold:** 1 (must be exact)

**Required fields:** `expected_tool_calls`, `tool_calls`

---

### 4.3 Script-Based Metrics

#### Action Evaluation

**What it measures:** Did the AI's action actually work in the real system?

**Plain English:** "Don't just check what the AI said—check if it actually did what it was supposed to do."

**Score:** Binary (0 or 1)

**How it works:**
1. AI performs an action (e.g., "Create a namespace")
2. Framework runs your verification script
3. Script exit code determines pass/fail (0 = success, non-zero = failure)

**Example:**
```bash
# verify_namespace.sh
#!/bin/bash
kubectl get namespace test-ns > /dev/null 2>&1
exit $?  # Returns 0 if namespace exists
```

**Configuration:**
```yaml
- conversation_group_id: infrastructure_test
  setup_script: ./scripts/setup_cluster.sh
  cleanup_script: ./scripts/cleanup_cluster.sh
  turns:
    - turn_id: create_namespace
      query: "Create a namespace called demo-app"
      verify_script: ./scripts/verify_namespace.sh
      turn_metrics:
        - script:action_eval
```

**When to use:** Infrastructure changes, system modifications, end-to-end testing

**Important:** Scripts only run when API mode is enabled

**Threshold:** 1 (must succeed)

**Required fields:** `verify_script` (API mode must be enabled)

---

## 5. Conversation-Level Metrics

Conversation-level metrics evaluate complete multi-turn dialogues.

### 5.1 DeepEval Metrics

#### Conversation Completeness

**What it measures:** Did the conversation fully address what the user wanted to accomplish?

**Plain English:** "By the end of the conversation, did the user get everything they were looking for?"

**Score Range:** 0.0 to 1.0 (higher is better)

**Example:**
```
User: "I need to deploy an app to OpenShift and set up monitoring"
AI: "I can help! Let's start with deployment. What's your app name?"
User: "my-web-app"
AI: "Great! Here's how to deploy... [deployment instructions]"
User: "Done! What about monitoring?"
AI: "For monitoring, here are the steps... [monitoring setup]"
User: "Perfect, thanks!"

✓ Goal 1: Deploy app → Addressed
✓ Goal 2: Set up monitoring → Addressed
Score: 1.0 (Complete)
```

**When to use:** Evaluating customer support conversations, goal-oriented assistants

**Threshold:** 0.8 or higher

---

#### Conversation Relevancy

**What it measures:** How relevant are the responses throughout the conversation?

**Plain English:** "Does each response stay on topic?"

**Score Range:** 0.0 to 1.0 (higher is better)

**When to use:** Keeping conversations focused, detecting when AI drifts off-topic

**Threshold:** 0.7 or higher

---

#### Knowledge Retention

**What it measures:** Does the AI remember and use information from earlier in the conversation?

**Plain English:** "Does the AI have a memory, or does it forget what was said earlier?"

**Score Range:** 0.0 to 1.0 (higher is better)

**Example:**
```
✓ Good Retention (High Score):
User: "My deployment is called web-app in the production namespace"
AI: "Got it. What would you like to do with web-app?"
User: "Scale it to 3 replicas"
AI: "I'll scale web-app in the production namespace to 3 replicas."
     [remembers both name and namespace]

✗ Poor Retention (Low Score):
User: "My deployment is called web-app in the production namespace"
AI: "Okay, what do you want to do?"
User: "Scale it to 3 replicas"
AI: "What's the deployment name and namespace?" [forgot!]
```

**When to use:** 
- Multi-turn conversations and troubleshooting sessions
- Evaluating fine-tuned models (especially useful to measure if fine-tuning improved context retention)
- Comparing base models vs fine-tuned versions for conversation ability

**Threshold:** 0.7 or higher

---

## 6. Metric Selection Guide

### Decision Tree

```
What are you evaluating?
│
├─ Single Q&A (Turn-Level)
│  │
│  ├─ Answer Quality?
│  │  ├─ Is answer relevant? → response_relevancy
│  │  ├─ Is answer factual? → faithfulness
│  │  └─ Matches expected? → answer_correctness
│  │
│  ├─ Information Retrieval?
│  │  ├─ Found everything needed? → context_recall
│  │  ├─ Is retrieved info relevant? → context_relevance
│  │  └─ Too much irrelevant info? → context_precision
│  │
│  ├─ AI Behavior?
│  │  ├─ Right intent? → intent_eval
│  │  └─ Right tools? → tool_eval
│  │
│  └─ Real Actions?
│     └─ Infrastructure changes? → action_eval
│
└─ Conversation (Conversation-Level)
   ├─ Goals achieved? → conversation_completeness
   ├─ Stayed on topic? → conversation_relevancy
   └─ Remembered context? → knowledge_retention
```

### Common Recipe Patterns

#### Recipe 1: Customer Support Bot
```yaml
turn_metrics:
  - ragas:response_relevancy    # On-topic answer?
  - ragas:faithfulness          # No hallucinations?
  - custom:answer_correctness   # Matches expected?
```

#### Recipe 2: Multi-Turn Troubleshooting
```yaml
# Per turn:
turn_metrics:
  - ragas:response_relevancy

# Full conversation:
conversation_metrics:
  - deepeval:conversation_completeness
  - deepeval:knowledge_retention
```

#### Recipe 3: Tool-Calling Agent
```yaml
turn_metrics:
  - custom:tool_eval            # Right tool + params?
  - ragas:response_relevancy    # Good explanation?
```

#### Recipe 4: Infrastructure Automation
```yaml
turn_metrics:
  - script:action_eval          # Action worked?
  - custom:tool_eval            # Called right tool?
```

---

# Part 3: Practical Implementation

## 7. Step-by-Step Setup

### Step 1: Prerequisites

- Python 3.11 - 3.13
- UV package manager (recommended) or pip
- API key for a Judge LLM (e.g., OpenAI)
- Basic command line knowledge

### Step 2: Installation

```bash
# Navigate to project
cd lightspeed-evaluation

# Install with UV
uv sync

# OR with pip
pip install -e .
```

### Step 3: Set Environment Variables

```bash
# Required: Judge LLM
export OPENAI_API_KEY="sk-your-api-key-here"

# For other providers:
# export WATSONX_API_KEY="your-key"
# export GEMINI_API_KEY="your-key"

# Optional: For live API testing
export API_KEY="your-api-endpoint-key"

```

### Step 4: Verify Installation

```bash
# Check if command is available
lightspeed-eval --help
```

---

## 8. Configuration Guide

### System Configuration (`system.yaml`)

**Minimal Configuration:**
```yaml
# Judge LLM settings
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.0              # Deterministic evaluation
  max_tokens: 512
  timeout: 300
  num_retries: 3

# Default metrics and thresholds
metrics_metadata:
  turn_level:
    "ragas:response_relevancy":
      threshold: 0.8
      description: "How relevant the response is"
      default: true             # Used by default
    "ragas:faithfulness":
      threshold: 0.8
      description: "Factual accuracy"
      default: false            # Only when specified

# Output settings
output:
  output_dir: "./eval_output"
  enabled_outputs:
    - csv                       # Detailed results
    - json                      # Statistics
    - txt                       # Summary

# Visualization
visualization:
  enabled_graphs:
    - "pass_rates"
    - "score_distribution"
```

**Important Settings Explained:**

- **`default: true`**: Metric runs automatically if no metrics specified
- **`default: false`**: Only runs when explicitly requested
- **`threshold`**: Minimum score to pass (0.0 to 1.0)
- **`temperature: 0.0`**: Ensures consistent, deterministic evaluation

### Evaluation Data (`evaluation_data.yaml`)

**Simple Example:**
```yaml
- conversation_group_id: basic_test
  description: "Testing basic Q&A"
  
  turns:
    - turn_id: question_1
      query: "What is OpenShift?"
      response: "OpenShift is an enterprise Kubernetes platform..."
      contexts:
        - "OpenShift is Red Hat's enterprise Kubernetes distribution..."
      expected_response: "OpenShift is an enterprise Kubernetes platform"
      
      # Uses default metrics (response_relevancy)
      turn_metrics: null
```

**Advanced Example:**
```yaml
- conversation_group_id: advanced_test
  description: "Testing with multiple metrics"
  
  turns:
    - turn_id: question_1
      query: "How do I reset my password?"
      response: "Click 'Forgot Password' on the login page..."
      expected_response: "Use the forgot password link"
      expected_intent: "provide instructions"
      
      # Specify exact metrics
      turn_metrics:
        - "ragas:response_relevancy"
        - "ragas:faithfulness"
        - "custom:answer_correctness"
        - "custom:intent_eval"
      
      # Override threshold for this turn
      turn_metrics_metadata:
        "ragas:faithfulness":
          threshold: 0.9          # Stricter than default
```

**Tool Evaluation Example:**
```yaml
- conversation_group_id: tool_test
  turns:
    - turn_id: get_pods
      query: "Show me all pods in the default namespace"
      expected_tool_calls:
        - - tool_name: oc_get
            arguments:
              kind: pod
              namespace: default
      turn_metrics:
        - "custom:tool_eval"
      turn_metrics_metadata:
        "custom:tool_eval":
          ordered: true      # default: true
          full_match: true   # default: true (false = subset matching, all expected must be present)
```

**Script-Based Example:**
```yaml
- conversation_group_id: infrastructure_test
  setup_script: "./scripts/setup_test_env.sh"
  cleanup_script: "./scripts/cleanup_test_env.sh"
  
  turns:
    - turn_id: create_namespace
      query: "Create a namespace called test-demo"
      verify_script: "./scripts/verify_namespace_exists.sh"
      turn_metrics:
        - "script:action_eval"
```

**Skip on Failure Example:**

Skip remaining turns completely (no API calls or evaluations) when a turn fails:

```yaml
- conversation_group_id: dependent_workflow
  skip_on_failure: true  # Or set globally in system.yaml.
  turns:
    - turn_id: step_1
      query: "Create namespace"
      turn_metrics: ["script:action_eval"]
    - turn_id: step_2  # SKIPPED if step_1 fails
      query: "Deploy to namespace"
      turn_metrics: ["script:action_eval"]
```

---

## 9. Running Evaluations

### Basic Evaluation

```bash
lightspeed-eval \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml
```

### With Custom Output Directory

```bash
lightspeed-eval \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml \
  --output-dir ./my_evaluation_results
```

### What Happens During Evaluation

1. **Configuration Validation**
   - Checks all required fields
   - Validates metric selections
   - Verifies Judge LLM connectivity

2. **Data Collection**
   - If API enabled: Calls your API for responses
   - If API disabled: Uses pre-filled data from YAML

3. **Metric Evaluation**
   - Runs turn-level metrics for each turn
   - Runs conversation-level metrics for full conversations
   - Uses Judge LLM to score responses

4. **Scoring & Analysis**
   - Compares scores against thresholds
   - Generates PASS/FAIL/ERROR/SKIPPED status
   - Calculates statistics

5. **Output Generation**
   - Creates CSV, JSON, TXT files
   - Generates visualization graphs
   - Saves amended evaluation data

---

## 10. Understanding Results

### Output Files

```
eval_output/
├── evaluation_20251028_143000_detailed.csv
├── evaluation_20251028_143000_summary.json
├── evaluation_20251028_143000_summary.txt
└── graphs/
    ├── evaluation_20251028_143000_pass_rates.png
    ├── evaluation_20251028_143000_score_distribution.png
    ├── evaluation_20251028_143000_conversation_heatmap.png
    └── evaluation_20251028_143000_status_breakdown.png
```

### CSV File (Detailed Results)

Contains every metric evaluation with:
- Conversation group ID and turn ID
- Metric identifier
- Score, threshold, status (PASS/FAIL/ERROR/SKIPPED)
- Detailed reasoning
- Query and response text
- Execution time

**Use for:** Drilling into specific failures, detailed analysis

### JSON File (Summary)

Contains:
- Overall statistics (pass/fail/error counts)
- Per-metric summaries
- Score distributions (mean, median, std dev)
- Execution metadata

**Use for:** Quick overview, automated processing, tracking trends

### TXT File (Human-Readable)

Example:
```
EVALUATION SUMMARY
==================
Total Evaluations: 10
Passed: 8 (80.0%)
Failed: 2 (20.0%)
Errors: 0 (0.0%)

METRIC BREAKDOWN
================
ragas:response_relevancy:
  Mean Score: 0.85
  Pass Rate: 90%
  
ragas:faithfulness:
  Mean Score: 0.78
  Pass Rate: 70%
```

**Use for:** Quick review, executive summaries

### Visualization Graphs

1. **Pass Rates Bar Chart**: Compare pass rates per metric
2. **Score Distribution Box Plot**: Shows score spread
3. **Conversation Heatmap**: Performance across conversations
4. **Status Breakdown Pie Chart**: Overall pass/fail/error distribution

**Use for:** Presentations, quick visual insights

### Interpreting Status

- **PASS** ✅: Score met or exceeded threshold
- **FAIL** ❌: Score below threshold
- **ERROR** ⚠️: Evaluation couldn't complete (missing data, API failure, etc.)
- **SKIPPED** ⏭️: Evaluation skipped due to prior failure (when `skip_on_failure` is enabled)

### Score Quality Levels

| Score | Quality | Recommendation |
|-------|---------|----------------|
| 0.9 - 1.0 | Excellent | Production ready |
| 0.8 - 0.9 | Good | Typical threshold |
| 0.7 - 0.8 | Acceptable | Consider improvements |
| < 0.7 | Poor | Needs work |

### Pass Rate Interpretation

| Pass Rate | Status | Action |
|-----------|--------|--------|
| ≥ 90% | Production ready | Deploy with confidence |
| 80-90% | Good quality | Minor improvements |
| 70-80% | Acceptable for testing | Needs improvement |
| < 70% | Not ready | Significant work needed |

---

# Part 4: Real-World Application

## 11. Common Use Cases

### Use Case 1: Quality Assurance for Customer Support Bot

**Scenario:** Launching a customer support chatbot

**Evaluation Strategy:**

1. Create test dataset with 50 common questions
2. Use metrics:
   - `ragas:response_relevancy` (0.8)
   - `ragas:faithfulness` (0.8)
   - `custom:answer_correctness` (0.75)

3. Configuration:
```yaml
- conversation_group_id: support_qa
  turns:
    - turn_id: password_reset
      query: "How do I reset my password?"
      contexts:
        - "Password reset: Click 'Forgot Password', enter email..."
      expected_response: "Use forgot password link and check email"
      turn_metrics:
        - ragas:response_relevancy
        - ragas:faithfulness
        - custom:answer_correctness
```

4. Success criteria:
   - Overall pass rate ≥ 90%
   - No faithfulness scores below 0.7
   - All high-priority questions pass

---

### Use Case 2: Regression Testing After Model Update

**Scenario:** Updating to a new AI model

**Strategy:**

1. Use existing production questions (100-500 samples)
2. Run evaluation on old model → Save results
3. Run evaluation on new model → Save results
4. Compare results

**Commands:**
```bash
# Evaluate old model
lightspeed-eval \
  --system-config config/system_old_model.yaml \
  --eval-data config/prod_samples.yaml \
  --output-dir ./results_old_model

# Evaluate new model
lightspeed-eval \
  --system-config config/system_new_model.yaml \
  --eval-data config/prod_samples.yaml \
  --output-dir ./results_new_model

# Compare results
uv run python script/compare_evaluations.py \
  results_old_model/evaluation_summary.json \
  results_new_model/evaluation_summary.json
```

5. Decision criteria:
   - New model must not decrease pass rate by >5%
   - Critical metrics must maintain or improve
   - Statistical significance test passes

---

### Use Case 3: Multi-Turn Troubleshooting

**Scenario:** AI guides users through complex troubleshooting

**Configuration:**
```yaml
- conversation_group_id: troubleshoot_deployment
  description: "Multi-turn deployment troubleshooting"
  
  conversation_metrics:
    - deepeval:conversation_completeness
    - deepeval:knowledge_retention
  
  turns:
    - turn_id: turn_1
      query: "My pod won't start"
      turn_metrics:
        - ragas:response_relevancy
    
    - turn_id: turn_2
      query: "It says ImagePullBackOff"
      turn_metrics:
        - ragas:response_relevancy
    
    - turn_id: turn_3
      query: "How do I fix the image registry auth?"
      turn_metrics:
        - ragas:response_relevancy
        - custom:intent_eval
```

**Success criteria:**
- Conversation completeness ≥ 0.85
- Knowledge retention ≥ 0.8
- Each turn response relevancy ≥ 0.8

---

### Use Case 4: Tool-Calling Agent Validation

**Scenario:** AI performs actions in Kubernetes/OpenShift

**Configuration:**
```yaml
- conversation_group_id: tool_calling_test
  turns:
    - turn_id: list_pods
      query: "Show me pods in the production namespace"
      expected_tool_calls:
        - - tool_name: oc_get
            arguments:
              kind: pod
              namespace: production
      turn_metrics:
        - custom:tool_eval
    
    - turn_id: scale_deployment
      query: "Scale web-app to 3 replicas"
      expected_tool_calls:
        - - tool_name: oc_scale
            arguments:
              kind: deployment
              name: web-app
              replicas: 3
      turn_metrics:
        - custom:tool_eval
```

**Success criteria:** 100% tool call accuracy

---

### Use Case 5: Infrastructure Validation

**Scenario:** AI creates and modifies infrastructure

**Configuration:**
```yaml
- conversation_group_id: infra_operations
  setup_script: "./scripts/setup_test_cluster.sh"
  cleanup_script: "./scripts/cleanup_test_cluster.sh"
  
  turns:
    - turn_id: create_namespace
      query: "Create a namespace called demo-app"
      verify_script: "./scripts/verify_namespace.sh"
      turn_metrics:
        - script:action_eval
```

**Verification Script:**
```bash
#!/bin/bash
# verify_namespace.sh
kubectl get namespace demo-app > /dev/null 2>&1
exit $?
```

**Success criteria:** 100% pass rate on critical operations

---

## 12. Best Practices

### 1. Start Small, Scale Up

❌ **Don't:** Start with 1000 questions and all metrics  
✅ **Do:** Start with 10-20 key questions and 2-3 core metrics

**Progression:**
- Week 1: 10 questions, 2 metrics
- Week 2: 50 questions, add metrics
- Month 1: 100-200 questions, full suite
- Production: 500+ questions

### 2. Choose the Right Metrics

| Scenario | Recommended Metrics |
|----------|---------------------|
| Customer Support (Single Q&A) | response_relevancy, faithfulness, answer_correctness |
| Multi-turn Conversations | conversation_completeness, knowledge_retention |
| Tool-calling Agents | tool_eval, response_relevancy |
| Infrastructure Automation | script:action_eval, tool_eval |

### 3. Set Realistic Thresholds

| Metric Type | Threshold | Use Case |
|-------------|-----------|----------|
| Production-critical | 0.85 - 0.95 | Customer-facing |
| Standard quality | 0.75 - 0.85 | General use |
| Beta/Testing | 0.70 - 0.75 | Testing phase |
| Binary metrics | 1.0 | Must match |

### 4. Create a Diverse Test Set

**Distribution:**
- 80%: Common, expected queries
- 15%: Edge cases
- 5%: Negative cases (should refuse/clarify)

### 5. Version Control Your Configurations

**Track in Git:**
- ✅ `system.yaml`
- ✅ `evaluation_data.yaml`
- ✅ Verification scripts
- ✅ Expected responses

**Don't track:**
- ❌ API keys
- ❌ Output files
- ❌ Cached results

### 6. Automate and Integrate

**CI/CD Example:**
```yaml
# .github/workflows/ai_evaluation.yml
name: AI Quality Evaluation

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Evaluation
        run: |
          uv sync
          lightspeed-eval \
            --system-config config/system.yaml \
            --eval-data config/evaluation_data.yaml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### 7. Regular Regression Testing

**Schedule:**
- Daily: Quick smoke tests (10-20 questions)
- Weekly: Full regression (100-500 questions)
- Before releases: Extended suite (1000+ questions)

### 8. Document Your Strategy

Create a README with:
- Evaluation goals
- Metric selection rationale
- Threshold justification
- Test set composition
- Success criteria

### 9. Handle Edge Cases

**Common edge cases:**
- Missing context → Should ask for more info
- Out-of-scope → Should politely decline
- Ambiguous queries → Should ask clarifying questions
- Multiple valid answers → Use broader thresholds

### 10. Cost Management

**Optimize Judge LLM costs:**

1. Use cheaper models when possible
```yaml
llm:
  model: "gpt-4o-mini"  # Instead of gpt-4o
```

2. Enable caching
```yaml
llm:
  cache_enabled: true
  cache_dir: ".caches/llm_cache"
```

3. Subset testing during development
   - Full suite: Weekly
   - Sample (10%): Daily
   - Critical questions: Per PR

---

## 13. Troubleshooting

### Issue 1: "No API key found"

**Error:** `Error: OPENAI_API_KEY environment variable not set`

**Solution:**
```bash
export OPENAI_API_KEY="sk-your-key-here"

# Verify
echo $OPENAI_API_KEY

# Persist in shell profile
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

---

### Issue 2: All Metrics Show ERROR

**Symptoms:** Status shows "ERROR" instead of PASS/FAIL

**Common causes & solutions:**

1. **Missing required fields**
```yaml
# ❌ Missing contexts for faithfulness
- turn_id: test1
  query: "Question"
  response: "Answer"
  turn_metrics:
    - ragas:faithfulness  # Needs contexts!

# ✅ Fixed
- turn_id: test1
  query: "Question"
  response: "Answer"
  contexts:
    - "Context document here"
  turn_metrics:
    - ragas:faithfulness
```

2. **Empty or null values**
```yaml
# ❌ Empty response
response: ""

# ✅ Provide actual response
response: "This is the answer"
```

**Field Requirements:**

| Metric | Required Fields |
|--------|----------------|
| response_relevancy | query, response |
| faithfulness | response, contexts |
| context_recall | contexts, expected_response |
| answer_correctness | query, response, expected_response |
| intent_eval | query, response, expected_intent |
| tool_eval | expected_tool_calls, tool_calls |
| action_eval | verify_script (API mode) |

---

### Issue 3: Low Faithfulness Scores

**Symptoms:** Faithfulness scores consistently below threshold

**Diagnosis:** Check CSV for reasons like "claims not supported by context"

**Solutions:**

1. **Add more context documents**
```yaml
contexts:
  - "Document 1 about topic A"
  - "Document 2 about topic B"
  - "Document 3 with more details"
```

2. **Adjust prompt to stick to facts**
```yaml
api:
  system_prompt: "Only use information from the provided context.
                  If information isn't in the context, say so."
```

---

### Issue 4: Inconsistent Results

**Symptoms:** Same question gets different scores each time

**Cause:** Non-zero temperature (randomness)

**Solution:**
```yaml
llm:
  temperature: 0.0  # Zero for deterministic evaluation
```

---

### Issue 5: Evaluation is Very Slow

**Solutions:**

1. **Increase concurrency**
```yaml
core:
  max_threads: 50
```

2. **Enable caching**
```yaml
llm:
  cache_enabled: true
```

3. **Use faster model**
```yaml
llm:
  model: "gpt-4o-mini"
```

---

### Issue 6: Script Execution Failed

**Solutions:**

1. **Check permissions**
```bash
chmod +x scripts/verify.sh
```

2. **Verify path**
```yaml
# Relative path from eval data file
verify_script: "./scripts/verify.sh"

# Or absolute path
verify_script: "/full/path/to/verify.sh"
```

3. **Test manually**
```bash
./scripts/verify.sh
echo $?  # Should be 0 for success
```

4. **Ensure API mode enabled**
```yaml
api:
  enabled: true  # Required for scripts
```

---

### Issue 7: Tool Evaluation Always Fails

**Solutions:**

1. **Check format**
```yaml
# ✅ Correct (list of lists of dicts)
expected_tool_calls:
  - - tool_name: oc_get
      arguments:
        kind: pod

# ❌ Wrong
expected_tool_calls:
  tool_name: oc_get  # Missing list structure
```

2. **Use regex for flexible matching**
```yaml
expected_tool_calls:
  - - tool_name: oc_get
      arguments:
        namespace: "openshift-.*"  # Regex pattern
```

---

### Issue 8: Out of Memory

**Solutions:**

1. **Reduce batch size**
```yaml
core:
  max_threads: 10  # Lower from 50
```

2. **Process in smaller batches**
```bash
# Split evaluation data into smaller files
lightspeed-eval --eval-data config/eval_batch1.yaml
lightspeed-eval --eval-data config/eval_batch2.yaml
```

---

# Part 5: Reference Materials

## 14. Quick Reference Tables

### All Metrics at a Glance

| Metric | Score | What It Checks | Threshold | Required Fields |
|--------|-------|----------------|-----------|----------------|
| **ragas:response_relevancy** | 0-1 | Answer addresses question | 0.8 | query, response |
| **ragas:faithfulness** | 0-1 | No made-up information | 0.8 | response, contexts |
| **ragas:context_recall** | 0-1 | Found all needed info | 0.8 | contexts, expected_response |
| **ragas:context_relevance** | 0-1 | Retrieved info is relevant | 0.7 | query, contexts |
| **ragas:context_precision_*** | 0-1 | Retrieved info is useful | 0.7 | query, contexts, response |
| **custom:answer_correctness** | 0-1 | Matches expected answer | 0.75 | query, response, expected_response |
| **custom:intent_eval** | 0/1 | Has right intent | 1 | query, response, expected_intent |
| **custom:tool_eval** | 0/1 | Called correct tools with expected results | 1 | expected_tool_calls, tool_calls |
| **script:action_eval** | 0/1 | Real action verified | 1 | verify_script |
| **deepeval:conversation_completeness** | 0-1 | User's goals achieved | 0.8 | Full conversation |
| **deepeval:conversation_relevancy** | 0-1 | Stayed on topic | 0.7 | Full conversation |
| **deepeval:knowledge_retention** | 0-1 | Remembered context | 0.7 | Full conversation |

### Configuration Cheat Sheet

**Minimal system.yaml:**
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.0

metrics_metadata:
  turn_level:
    "ragas:response_relevancy":
      threshold: 0.8
      default: true

output:
  output_dir: "./eval_output"
```

**Minimal evaluation_data.yaml:**
```yaml
- conversation_group_id: test_1
  turns:
    - turn_id: q1
      query: "What is OpenShift?"
      response: "OpenShift is..."
      contexts: ["OpenShift is..."]
```

### Common Commands

```bash
# Basic evaluation
lightspeed-eval \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml

# Custom output directory
lightspeed-eval \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml \
  --output-dir ./results

# Compare evaluations
uv run python script/compare_evaluations.py \
  results1/summary.json \
  results2/summary.json

# Multi-provider evaluation
uv run python script/run_multi_provider_eval.py \
  --providers-config config/multi_eval_config.yaml
```

### Score Interpretation

| Score | Quality | Pass Rate | Status |
|-------|---------|-----------|--------|
| 0.9-1.0 | Excellent | ≥90% | Production ready |
| 0.8-0.9 | Good | 80-90% | Good quality |
| 0.7-0.8 | Acceptable | 70-80% | Needs improvement |
| <0.7 | Poor | <70% | Not ready |

### Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| No API key | `export OPENAI_API_KEY="..."` |
| All ERROR | Check required fields for metrics |
| Low faithfulness | Add more context documents |
| Inconsistent results | Set `temperature: 0.0` |
| Slow evaluation | Enable caching, increase threads |
| Script fails | Check permissions: `chmod +x` |
| "Metric not found" | Check spelling against supported list |

---


## 15. Resources & Links

### Official Framework Documentation

- **Ragas Framework**: https://docs.ragas.io/
  - Available Metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
  - Faithfulness: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
  - Response Relevancy: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/

- **DeepEval Framework**: https://deepeval.com/docs/
  - Metrics Introduction: https://deepeval.com/docs/metrics-introduction
  - Conversation Completeness: https://deepeval.com/docs/metrics-conversation-completeness
  - Knowledge Retention: https://deepeval.com/docs/metrics-knowledge-retention

### LLM Provider Documentation

- **OpenAI**: https://platform.openai.com/docs/
- **IBM Watsonx**: https://www.ibm.com/docs/en/watsonx-as-a-service
- **Google Gemini**: https://ai.google.dev/docs
- **LiteLLM** (unified interface): https://docs.litellm.ai/

### Learning Resources

**For Beginners:**
- "Introduction to LLM Evaluation" (search for current articles)
- Ragas Getting Started Guide
- DeepEval tutorials

**For Advanced Users:**
- RAG (Retrieval-Augmented Generation) papers
- LLM evaluation best practices
- Conversational AI assessment techniques

### Internal Resources

This repository:
- Main README: `../README.md`
- Agent Guidelines: `../AGENTS.md`
- Multi-Provider Evaluation: `multi_provider_evaluation.md`
- Evaluation Comparison: `evaluation_comparison.md`
- Sample Configurations: `../config/`
- Example Scripts: `../config/sample_scripts/`

### Community & Support

- **GitHub Repository**: Report issues, request features
- **GitHub Discussions**: Ask questions, share experiences
- **Pull Requests**: Contribute improvements

### Key Concepts Glossary

- **API-Enabled Mode**: Real-time evaluation calling your AI system's API
- **Binary Metric**: Pass/fail evaluation (0 or 1)
- **Context**: Background information from knowledge base
- **Faithfulness**: How well answer sticks to provided facts
- **Hallucination**: AI making up information
- **Judge LLM**: AI model used to evaluate another AI
- **Pass Rate**: Percentage of evaluations meeting threshold
- **Ragas**: Framework for retrieval-augmented generation metrics
- **Static Mode**: Evaluation using pre-filled responses
- **Threshold**: Minimum score required to pass
- **Turn**: Single question-response pair
- **Turn-Level**: Evaluation of individual Q&A pairs
- **Conversation-Level**: Evaluation of multi-turn dialogues

---

# Conclusion

This comprehensive guide has covered everything you need to know to effectively evaluate AI applications using the LightSpeed Evaluation Framework:

✅ **Understanding** - What evaluation is and why it matters  
✅ **Methodologies** - All 13 evaluation metrics explained in plain English  
✅ **Implementation** - Step-by-step setup and configuration  
✅ **Interpretation** - Understanding and acting on results  
✅ **Application** - Real-world use cases and best practices  
✅ **Reference** - Quick lookup tables and decision trees  

### Next Steps

1. **Start with a pilot**: Choose 10-20 key questions and 2-3 metrics
2. **Run your first evaluation**: Follow the step-by-step guide
3. **Analyze results**: Use the interpretation section
4. **Iterate and improve**: Adjust thresholds and expand coverage
5. **Automate**: Integrate into your development workflow

### Getting Help

- Connect via Slack channel: #forum-lightspeed

---

**Last Updated:** December 23, 2025  
**Status:** Complete and Ready for Use  

**Feedback:** Please submit suggestions via GitHub issues or pull requests.

---

*This guide is designed to make AI evaluation accessible to everyone. Whether you're a product manager making decisions, a QA engineer testing systems, or a developer integrating evaluation into workflows, you now have everything you need to ensure your AI applications meet quality standards.*

**Happy Evaluating! 🚀**


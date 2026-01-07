# LightSpeed Evaluation Framework

A comprehensive framework for evaluating GenAI applications.

**This is a WIP. We’re actively adding features, fixing issues, and expanding examples. Please give it a try, share feedback, and report bugs.**

## 🎯 Key Features

- **Multi-Framework Support**: Seamlessly use metrics from Ragas, DeepEval, and custom implementations
- **Turn & Conversation-Level Evaluation**: Support for both individual queries and multi-turn conversations
- **Evaluation types**: Response, Context, Tool Call, Overall Conversation evaluation & Script-based evaluation
- **LLM Provider Flexibility**: OpenAI, Watsonx, Gemini, vLLM and others
- **API Integration**: Direct integration with external API for real-time data generation (if enabled)
- **Setup/Cleanup Scripts**: Support for running setup and cleanup scripts before/after each conversation evaluation (applicable when API is enabled)
- **Token Usage Tracking**: Track input/output tokens for both API calls and Judge LLM evaluations
- **Streaming Performance Metrics**: Capture time-to-first-token (TTFT), streaming duration, and tokens/second when using streaming endpoint
- **Statistical Analysis**: Statistics for every metric with score distribution analysis
- **Rich Output**: CSV, JSON, TXT reports + visualization graphs (pass rates, distributions, heatmaps)
- **Flexible Configuration**: Configurable environment & metric metadata, Global defaults with per-conversation/per-turn metric overrides
- **Early Validation**: Catch configuration errors before expensive LLM calls
- **Concurrent Evaluation**: Multi-threaded evaluation with configurable thread count
- **Caching**: LLM, embedding, and API response caching for faster re-runs
- **Skip on Failure**: Optionally skip remaining evaluations in a conversation when a turn evaluation fails (configurable globally or per conversation). When there is an error in API call/Setup script execution metrics are marked as ERROR always.

## 🚀 Quick Start

### Installation

```bash
# From Git
pip install git+https://github.com/lightspeed-core/lightspeed-evaluation.git

# Local Development
pip install uv
uv sync
```

#### Optional: Local Embedding Models (HuggingFace)

By default, lightspeed-evaluation uses remote embedding providers (OpenAI, Gemini). If you need **local embedding models** (HuggingFace/sentence-transformers), install with:

```bash
# Using pip
pip install 'lightspeed-evaluation[local-embeddings]'

# Using uv (local development)
uv sync --extra local-embeddings
```

> **Note**: Local embeddings require PyTorch and related packages (~6GB). Only install if you need `embedding.provider: huggingface` in your configuration.

### Basic Usage

```bash
# Set required environment variable(s) for Judge-LLM
export OPENAI_API_KEY="your-key"

# Optional: For script-based evaluations requiring Kubernetes access
export KUBECONFIG="/path/to/your/kubeconfig"

# Run evaluation
lightspeed-eval --system-config <CONFIG.yaml> --eval-data <EVAL_DATA.yaml> --output-dir <OUTPUT_DIR>
```

### Usage Scenarios
Please make any necessary modifications to system.yaml and evaluation_data.yaml. The evaluation_data.yaml file includes sample data for guidance.

#### 1. API-Enabled Real-time data collection
```bash
# Set required environment variable(s) for both Judge-LLM and API authentication (for MCP)
export OPENAI_API_KEY="your-evaluation-llm-key"

export API_KEY="your-api-endpoint-key"

# Ensure API is running at configured endpoint
# Default: http://localhost:8080/v1/

# Run with API-enabled configuration
lightspeed-eval --system-config config/system.yaml --eval-data config/evaluation_data.yaml
```

#### 2. Static Data Evaluation (API Disabled)
```bash
# Set required environment variable(s) for Judge-LLM
export OPENAI_API_KEY="your-key"

# Use system configuration with api.enabled: false
# You have to pre-generate response, contexts & tool_calls data in the input evaluation data file
lightspeed-eval --system-config config/system_api_disabled.yaml --eval-data config/evaluation_data.yaml
```

## 📊 Supported Metrics

### Turn-Level (Single Query)
- **Ragas** -- [docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) on Ragas website
  - Response Evaluation
    - [`faithfulness`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
    - [`response_relevancy`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)
  - Context Evaluation
    - [`context_recall`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
    - [`context_relevance`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/#context-relevance)
    - [`context_precision_without_reference`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#context-precision-without-reference)
    - [`context_precision_with_reference`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#context-precision-with-reference)
- **Custom**
  - Response Evaluation
    - [`answer_correctness`](src/lightspeed_evaluation/core/metrics/custom.py)
    - [`intent_eval`](src/lightspeed_evaluation/core/metrics/custom.py) - Evaluates whether the response demonstrates the expected intent or purpose
    - [`keywords_eval`](src/lightspeed_evaluation/core/metrics/custom/keywords_eval.py) - Keywords evaluation with alternatives (ALL keywords must match, case insensitive)
  - Tool Evaluation
    - [`tool_eval`](src/lightspeed_evaluation/core/metrics/custom.py) - Validates tool calls and arguments with regex pattern matching
- **Script-based**
  - Action Evaluation
    - [`script:action_eval`](src/lightspeed_evaluation/core/metrics/script.py) - Executes verification scripts to validate actions (e.g., infrastructure changes)

### Conversation-Level (Multi-turn)
- **DeepEval** -- [docs](https://deepeval.com/docs/metrics-introduction) on DeepEval website
  - [`conversation_completeness`](https://deepeval.com/docs/metrics-conversation-completeness)
  - [`conversation_relevancy`](https://deepeval.com/docs/metrics-turn-relevancy)
  - [`knowledge_retention`](https://deepeval.com/docs/metrics-knowledge-retention)

### Custom Metrics with GEval (from DeepEval)

GEval allows us to define custom evaluation metrics using natural language criteria. Define metrics in `system.yaml` under `metrics_metadata`:

```yaml
metrics_metadata:
  turn_level:
    "geval:custom_metric_name":
      criteria: |
        Specific criteria for the evaluation.
      evaluation_params:
        - query
        - response
        - expected_response  # optional
      evaluation_steps:
        - "Step 1: Check if..."
        - "Step 2: Verify that..."
      threshold: 0.7
      description: "Metric description"
```

See sample [`system config`](config/system.yaml) for complete examples of `geval:technical_accuracy` and `geval:conversation_coherence`.

## ⚙️ Configuration

### System Config (`config/system.yaml`)

The default system config file is [`config/system.yaml`](config/system.yaml).
See [`docs/configuration.md`](docs/configuration.md) for the detailed description.


### Input File Data Structure (`config/evaluation_data.yaml`)

```yaml
- conversation_group_id: "test_conversation"
  description: "Sample evaluation"
  
  # Optional: Environment setup/cleanup scripts, when API is enabled
  setup_script: "scripts/setup_env.sh"      # Run before conversation
  cleanup_script: "scripts/cleanup_env.sh"  # Run after conversation
  
  # Conversation-level metrics   
  conversation_metrics:
    - "deepeval:conversation_completeness"
  
  conversation_metrics_metadata:
    "deepeval:conversation_completeness":
      threshold: 0.8
  
  turns:
    - turn_id: id1
      query: What is OpenShift Virtualization?
      response: null                    # Populated by API if enabled, otherwise provide
      contexts:
        - OpenShift Virtualization is an extension of the OpenShift ...
      attachments: []                   # Attachments (Optional)
      expected_keywords: [["virtualization"], ["openshift"]]  # For keywords_eval evaluation
      expected_response: OpenShift Virtualization is an extension of the OpenShift Container Platform that allows running virtual machines alongside containers
      expected_intent: "explain a concept"  # Expected intent for intent evaluation
      
      # Per-turn metrics (overrides system defaults)
      turn_metrics:
        - "ragas:faithfulness"
        - "custom:keywords_eval"
        - "custom:answer_correctness"
        - "custom:intent_eval"
      
      # Per-turn metric configuration
      turn_metrics_metadata:
        "ragas:faithfulness": 
          threshold: 0.9  # Override system default
      # turn_metrics: null (omitted) → Use system defaults (metrics with default=true)
      
    - turn_id: id2
      query: Skip this turn evaluation
      turn_metrics: []                  # Skip evaluation for this turn

    - turn_id: id3
      query: Create a namespace called test-ns
      verify_script: "scripts/verify_namespace.sh"  # Script-based verification
      turn_metrics:
        - "script:action_eval"          # Script-based evaluation (if API is enabled)
```

### Input file Data Structure Details

#### Conversation Data Fields

| Field                           | Type           | Required | Description                                                          |
|---------------------------------|----------------|----------|----------------------------------------------------------------------|
| `conversation_group_id`         | string         | ✅       | Unique identifier for conversation                                   |
| `description`                   | string         | ❌       | Optional description                                                 |
| `setup_script`                  | string         | ❌       | Path to setup script (Optional, used when API is enabled)            |
| `cleanup_script`                | string         | ❌       | Path to cleanup script (Optional, used when API is enabled)          |
| `conversation_metrics`          | list[string]   | ❌       | Conversation-level metrics (Optional, if override is required)       |
| `conversation_metrics_metadata` | dict           | ❌       | Conversation-level metric config (Optional, if override is required) |
| `turns`                         | list[TurnData] | ✅       | List of conversation turns           |

#### Turn Data Fields

| Field                 | Type             | Required | Description                          | API Populated         |
|-----------------------|------------------|----------|--------------------------------------|-----------------------|
| `turn_id`             | string           | ✅       | Unique identifier for the turn       | ❌                    |
| `query`               | string           | ✅       | The question/prompt to evaluate      | ❌                    |
| `response`            | string           | 📋       | Actual response from system          | ✅ (if API enabled)   |
| `contexts`            | list[string]     | 📋       | Context information for evaluation   | ✅ (if API enabled)   |
| `attachments`         | list[string]     | ❌       | Attachments                          | ❌                    |
| `expected_keywords`   | list[list[string]] | 📋     | Expected keywords for keyword evaluation (list of alternatives) | ❌ |
| `expected_response`   | string           | 📋       | Expected response for comparison     | ❌                    |
| `expected_intent`     | string           | 📋       | Expected intent for intent evaluation| ❌                    |
| `expected_tool_calls` | list[list[list[dict]]] | 📋 | Expected tool call sequences (multiple alternative sets) | ❌ |
| `tool_calls`          | list[list[dict]] | ❌       | Actual tool calls from API           | ✅ (if API enabled)   |
| `verify_script`       | string           | 📋       | Path to verification script          | ❌                    |
| `turn_metrics`        | list[string]     | ❌       | Turn-specific metrics to evaluate    | ❌                    |
| `turn_metrics_metadata` | dict           | ❌       | Turn-specific metric configuration   | ❌                    |

> 📋 **Required based on metrics**: Some fields are required only when using specific metrics

Examples
> - `expected_keywords`: Required for `custom:keywords_eval` (case insensitive matching)
> - `expected_response`: Required for `custom:answer_correctness`
> - `expected_intent`: Required for `custom:intent_eval`
> - `expected_tool_calls`: Required for `custom:tool_eval` (multiple alternative sets format)
> - `verify_script`: Required for `script:action_eval` (used when API is enabled)
> - `response`: Required for most metrics (auto-populated if API enabled)

#### Metrics override behavior

| Override Value | Behavior |
|---------------------|----------|
| `null` (or omitted) | Use system global metrics (metrics with `default: true`) |
| `[]` (empty list)   | Skip evaluation for this turn |
| `["metric1", ...]`  | Use specified metrics only, ignore global metrics |

#### Tool Evaluation

The `custom:tool_eval` metric supports flexible matching with multiple alternative patterns:

- **Format**: `[[[tool_calls, ...]], [[tool_calls]], ...]` (list of list of list)
- **Matching**: Tries each alternative until one matches
- **Use Cases**: Optional tools, multiple approaches, default arguments, skip scenarios
- **Empty Sets**: `[]` represents "no tools" and must come after primary alternatives

#### Tool Call Structure

  ```yaml
  # Multiple alternative sets format: [[[tool_calls, ...]], [[tool_calls]], ...]
  expected_tool_calls:
    - # Alternative 1: Primary approach
      - # Sequence 1
        - tool_name: oc_get
          arguments:
            kind: pod
            name: openshift-light*    # Regex patterns supported
      - # Sequence 2 (if multiple parallel tool calls needed)
        - tool_name: oc_describe
          arguments:
            kind: pod
    - # Alternative 2: Different approach
      - # Sequence 1
        - tool_name: kubectl_get
          arguments:
            resource: pods
    - # Alternative 3: Skip scenario (optional)
      []  # When model has information from previous conversation
  ```

#### Script-Based Evaluations

The framework supports script-based evaluations.
**Note: Scripts only execute when API is enabled** - they're designed to test with actual environment changes.

- **Setup scripts**: Run before conversation evaluation (e.g., create failed deployment for troubleshoot query)
- **Cleanup scripts**: Run after conversation evaluation (e.g., cleanup failed deployment)  
- **Verify scripts**: Run per turn for `script:action_eval` metric (e.g., validate if a pod has been created or not)

```yaml
# Example: evaluation_data.yaml
- conversation_group_id: infrastructure_test
  setup_script: ./scripts/setup_cluster.sh
  cleanup_script: ./scripts/cleanup_cluster.sh
  turns:
    - turn_id: turn_id
      query: Create a new cluster
      verify_script: ./scripts/verify_cluster.sh
      turn_metrics:
        - script:action_eval
```

**Script Path Resolution**

Script paths in evaluation data can be specified in multiple ways:

- **Relative Paths**: Resolved relative to the evaluation data YAML file location, not the current working directory
- **Absolute Paths**: Used as-is
- **Home Directory Paths**: Expands to user's home directory

## 🔑 Authentication & Environment

### Required Environment Variables

#### For LLM as a Judge Evaluation (Always Required)
```bash
# Hosted vLLM (provider: hosted_vllm)
export HOSTED_VLLM_API_KEY="your-key"
export HOSTED_VLLM_API_BASE="https://your-vllm-endpoint/v1"

# OpenAI (provider: openai)
export OPENAI_API_KEY="your-openai-key"

# IBM Watsonx (provider: watsonx)
export WATSONX_API_KEY="your-key"
export WATSONX_API_BASE="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="your-project-id"

# Gemini (provider: gemini)
export GEMINI_API_KEY="your-key"

# Azure OpenAI (provider: azure)
export AZURE_API_KEY="your-azure-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
# AZURE_API_VERSION is optional
```

> **Note for Azure**: The `model` field should be **Azure deployment name**, not the model name (when these are different).

#### For Lightspeed Core API Integration (When `api.enabled: true`)
```bash
# API authentication for external system (MCP)
export API_KEY="your-api-endpoint-key"
```

## 📈 Output & Visualization

### Generated Reports
- **CSV**: Detailed results with status, scores, reasons
- **JSON**: Summary statistics with score distributions
- **TXT**: Human-readable summary
- **PNG**: 4 visualization types (pass rates, score distributions, heatmaps, status breakdown)

### Key Metrics in Output
- **Status**: PASS/FAIL/ERROR/SKIPPED
- **Actual Reasons**: Reason for evaluation status/result
- **Score Statistics**: Mean, median, standard deviation, min/max for every metric

### Streaming Performance Metrics

When using the streaming endpoint (`api.endpoint_type: streaming`), the framework captures additional performance metrics:

| Metric | Description |
|--------|-------------|
| `time_to_first_token` | Time in seconds from request start to first content token received |
| `streaming_duration` | Total time in seconds to receive all tokens |
| `tokens_per_second` | Output throughput (tokens generated per second, excluding TTFT) |

These metrics are included in:
- **CSV output**: Per-result columns for each metric
- **JSON output**: Per-result fields and aggregate statistics in `streaming_performance`
- **TXT output**: Aggregate statistics (mean, median, min/max) in the summary

## 🧪 Development

### Development Tools
```bash
uv sync --group dev

# Format code
make format

# Run all quality checks (same as CI)
make verify          # Linting (black --check, ruff, pylint)
make check-types     # Type checking (mypy)
make pyright         # Type checking (pyright)
make docstyle        # Docstring style (pydocstyle)
make bandit          # Security scanning

# Run tests
make test            # Or: uv run pytest tests --cov=src
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Parsing error with context-related metrics (e.g., `faithfulness`) | Increase [`max_tokens`](config/system.yaml#L16) to a higher value (e.g., 2048 or higher - depends on number of the context & size) |
| API responses not changing after updates | Disable caching (`cache_enabled: false`) or delete the cache folders (`.caches/`) |

## Generate answers (optional - for creating test data)
For generating answers (optional) refer [README-generate-answers](README-generate-answers.md)

## 📄 License & Contributing

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

Contributions welcome - see development setup above for code quality tools.

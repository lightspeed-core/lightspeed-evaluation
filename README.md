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
- **Flexible Configuration**: Configurable environment & metric metadata
- **Rich Output**: CSV, JSON, TXT reports + visualization graphs (pass rates, distributions, heatmaps)
- **Early Validation**: Catch configuration errors before expensive LLM calls
- **Statistical Analysis**: Statistics for every metric with score distribution analysis

## 🚀 Quick Start

### Installation

```bash
# From Git
pip install git+https://github.com/lightspeed-core/lightspeed-evaluation.git

# Local Development
pip install uv
uv sync
```

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
# Default: http://localhost:8080

# Run with API-enabled configuration
lightspeed-eval --system-config config/system.yaml --eval-data config/evaluation_data.yaml
```

#### 2. Static Data Evaluation (API Disabled)
```bash
# Set required environment variable(s) for Judge-LLM
export OPENAI_API_KEY="your-key"

# Use configuration with api.enabled: false
# Pre-fill response, contexts & tool_calls data in YAML
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
  - Tool Evaluation
    - [`tool_eval`](src/lightspeed_evaluation/core/metrics/custom.py) - Validates tool calls and arguments with regex pattern matching
- **Script-based**
  - Action Evaluation
    - [`script:action_eval`](src/lightspeed_evaluation/core/metrics/script_eval.py) - Executes verification scripts to validate actions (e.g., infrastructure changes)

### Conversation-Level (Multi-turn)
- **DeepEval** -- [docs](https://deepeval.com/docs/metrics-introduction) on DeepEval website
  - [`conversation_completeness`](https://deepeval.com/docs/metrics-conversation-completeness)
  - [`conversation_relevancy`](https://deepeval.com/docs/metrics-turn-relevancy)
  - [`knowledge_retention`](https://deepeval.com/docs/metrics-knowledge-retention)

## ⚙️ Configuration

### System Config (`config/system.yaml`)

```yaml
# Judge-LLM Configuration
llm:
  provider: openai            # openai, watsonx, azure, gemini etc.
  model: gpt-4o-mini          # Model name for the provider
  temperature: 0.0            # Generation temperature
  max_tokens: 512             # Maximum tokens in response
  timeout: 300                # Request timeout in seconds
  num_retries: 3              # Retry attempts

# API Configuration for Real-time Data Generation
api:
  enabled: true                        # Enable/disable API calls
  api_base: http://localhost:8080      # Base API URL
  endpoint_type: streaming             # streaming or query endpoint
  timeout: 300                         # API request timeout in seconds
  
  provider: openai                     # LLM provider for API queries (optional)
  model: gpt-4o-mini                   # Model to use for API queries (optional)
  no_tools: null                       # Whether to bypass tools (optional)
  system_prompt: null                  # Custom system prompt (optional)

# Metrics Configuration with thresholds and defaults
metrics_metadata:
  turn_level:
    "ragas:response_relevancy":
      threshold: 0.8
      description: "How relevant the response is to the question"
      default: true   # Used by default when turn_metrics is null

    "ragas:faithfulness":
      threshold: 0.8
      description: "How faithful the response is to the provided context"
      default: false  # Only used when explicitly specified
      
    "custom:tool_eval":
      description: "Tool call evaluation comparing expected vs actual tool calls (regex for arguments)"
  
  conversation_level:
    "deepeval:conversation_completeness":
      threshold: 0.8
      description: "How completely the conversation addresses user intentions"

# Output Configuration
output:
  output_dir: ./eval_output
  base_filename: evaluation
  enabled_outputs:          # Enable specific output types
    - csv                   # Detailed results CSV
    - json                  # Summary JSON with statistics
    - txt                   # Human-readable summary

# Visualization Configuration
visualization:
  figsize: [12, 8]            # Graph size (width, height)
  dpi: 300                    # Image resolution
  enabled_graphs:
    - "pass_rates"            # Pass rate bar chart
    - "score_distribution"    # Score distribution box plot
    - "conversation_heatmap"  # Heatmap of conversation performance
    - "status_breakdown"      # Pie chart for pass/fail/error breakdown
```

#### Non OpenAI configuration example
```yaml
# Judge-LLM Google Gemini
llm:
  provider: "gemini"
  model: "gemini-1.5-pro"    
  temperature: 0.0  
  max_tokens: 512  
  timeout: 120        
  num_retries: 3

# Judge-LLM HuggingFace embeddings
# provider: "huggingface" or "openai"
# model: model name
# provider_kwargs: additional arguments,
#   for examples see https://docs.ragas.io/en/stable/references/embeddings/#ragas.embeddings.HuggingfaceEmbeddings
embedding:
  provider: "huggingface"
  model: "sentence-transformers/all-mpnet-base-v2"
  provider_kwargs:
    # cache_folder: <path_for_downloaded_model>
    model_kwargs:
      device: "cpu"
...
```

### Evaluation Data Structure (`config/evaluation_data.yaml`)

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
      expected_response: OpenShift Virtualization is an extension of the OpenShift Container Platform that allows running virtual machines alongside containers
      
      # Per-turn metrics (overrides system defaults)
      turn_metrics:
        - "ragas:faithfulness"
        - "custom:answer_correctness"
      
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

### API Modes

#### With API Enabled (`api.enabled: true`)
- **Real-time data generation**: Queries are sent to external API
- **Dynamic responses**: `response` and `tool_calls` fields populated by API
- **Conversation context**: Conversation context is maintained across turns
- **Authentication**: Use `API_KEY` environment variable
- **Data persistence**: Saves amended `response`/`tool_calls` data to output directory so it can be used with API disabled

#### With API Disabled (`api.enabled: false`)
- **Static data mode**: Use pre-filled `response` and `tool_calls` data
- **Faster execution**: No external API calls
- **Reproducible results**: Same data used across runs

### Data Structure Details

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
| `expected_response`   | string           | 📋       | Expected response for comparison     | ❌                    |
| `expected_tool_calls` | list[list[dict]] | 📋       | Expected tool call sequences         | ❌                    |
| `tool_calls`          | list[list[dict]] | ❌       | Actual tool calls from API           | ✅ (if API enabled)   |
| `verify_script`       | string           | 📋       | Path to verification script          | ❌                    |
| `turn_metrics`        | list[string]     | ❌       | Turn-specific metrics to evaluate    | ❌                    |
| `turn_metrics_metadata` | dict           | ❌       | Turn-specific metric configuration   | ❌                    |

> 📋 **Required based on metrics**: Some fields are required only when using specific metrics

Examples
> - `expected_response`: Required for `custom:answer_correctness`
> - `expected_tool_calls`: Required for `custom:tool_eval`
> - `verify_script`: Required for `script:action_eval` (used when API is enabled)
> - `response`: Required for most metrics (auto-populated if API enabled)

#### Metrics override behavior

| Override Value | Behavior |
|---------------------|----------|
| `null` (or omitted) | Use system defaults (metrics with `default: true`) |
| `[]` (empty list)   | Skip evaluation for this turn |
| `["metric1", ...]`  | Use specified metrics only |

#### Tool Call Structure

  ```yaml
  expected_tool_calls:
    -
      - tool_name: oc_get           # Tool name
        arguments:                  # Tool arguments
          kind: pod
          name: openshift-light*    # Regex patterns supported for flexible matching
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

#### For LLM Evaluation (Always Required)
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
```

#### For API Integration (When `api.enabled: true`)
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
- **PASS/FAIL/ERROR**: Status based on thresholds
- **Actual Reasons**: DeepEval provides LLM-generated explanations, Custom metrics provide detailed reasoning
- **Score Statistics**: Mean, median, standard deviation, min/max for every metric

## 🧪 Development

### Development Tools
```bash
uv sync --group dev
make format
make pylint
make pyright
make docstyle
make check-types

uv run pytest tests --cov=src
```

## Agent Evaluation
For a detailed walkthrough of the new agent-evaluation framework, refer
[lsc_agent_eval/README.md](lsc_agent_eval/README.md)

## Generate answers (optional - for creating test data)
For generating answers (optional) refer [README-generate-answers](README-generate-answers.md)

## 📄 License & Contributing

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

Contributions welcome - see development setup above for code quality tools.
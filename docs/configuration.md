# Lightspeed Evaluation Configuration

The system configuration is driven by YAML file. The default config file is [config/system.yaml](config/system.yaml).

## General evaluation settings
| Setting (core.) | Default | Description |
|-----------------|---------|-------------|
| max_threads    | `50` | Maximum number of threads, set to null for Python default. 50 is OK on a typical laptop. Check your Judge-LLM service for max requests per minute |
| fail_on_invalid_data | `true` | If `false` don't fail on invalid conversations (like missing `context` field for some metrics) |
| skip_on_failure | `false` | If `true`, skip remaining turns and conversation metrics when a turn evaluation fails (FAIL or ERROR). Can be overridden per conversation in the input data yaml file. |

### Example
```yaml
core:
  max_threads: 50
  fail_on_invalid_data: true
  skip_on_failure: false  # Set to true to stop evaluation on first failure
```

## LLM Pool

Define a centralized pool of LLM configurations for the Judge Panel feature.

> **Note:** The `llm` config below will be deprecated. New deployments should use `llm_pool` + `judge_panel`.

| Setting | Description |
|---------|-------------|
| `llm_pool.defaults.cache_dir` | Cache directory (default: `.caches/llm_cache`) |
| `llm_pool.defaults.timeout` | Request timeout in seconds (default: `300`) |
| `llm_pool.defaults.num_retries` | Retry attempts (default: `3`) |
| `llm_pool.defaults.parameters.temperature` | Sampling temperature |
| `llm_pool.defaults.parameters.max_completion_tokens` | Max tokens in response |
| `llm_pool.defaults.parameters.*` | Any additional provider-supported LLM parameter (e.g., `top_p`, `frequency_penalty`) |
| `llm_pool.models.<id>.provider` | LLM provider (required) |
| `llm_pool.models.<id>.model` | Model name |
| `llm_pool.models.<id>.parameters.*` | Model-specific parameter overrides (merged with defaults, model takes priority) |

**Dynamic Parameters:** The `parameters` dict accepts any key-value pair supported by the LLM provider. Known parameters: `temperature`, `max_completion_tokens`. Unsupported parameters are silently dropped by the provider.

**Removing Defaults:** To remove an inherited default parameter, set it to `null` in the model config:
```yaml
models:
  no-temp-model:
    provider: openai
    parameters:
      temperature: null  # Removes default temperature, won't be sent to API
```

```yaml
llm_pool:
  defaults:
    cache_dir: ".caches/llm_cache"
    num_retries: 2
    parameters:
      temperature: 0.0
      max_completion_tokens: 512
  models:
    judge-4o-mini:
      provider: openai
      model: gpt-4o-mini
    judge-4.1-mini:
      provider: openai
      model: gpt-4.1-mini
      parameters:
        temperature: 0.3                # Overrides default
```

## Judge Panel

Use multiple LLMs as judges to reduce bias and improve evaluation accuracy.

| Setting | Description |
|---------|-------------|
| `judge_panel.judges` | List of model IDs from `llm_pool.models` (required) |
| `judge_panel.enabled_metrics` | Metrics using full panel (if unset, all LLM metrics use panel) |
| `judge_panel.aggregation_strategy` | How to combine judge scores: `max`, `average`, or `majority_vote` (see below) |

**Aggregation strategies** (multiple judges only; errored judges are excluded):

| Strategy | Score | Pass / fail |
|----------|-------|-------------|
| `max` (default) | Highest score | Score vs metric threshold |
| `average` | Mean of scores | Mean vs metric threshold |
| `majority_vote` | Mean of scores | **Strict majority** of judges individually meet the metric threshold — more than half must pass (Ties fail). |

```yaml
judge_panel:
  judges:
    - judge-4o-mini
    - judge-4.1-mini
  aggregation_strategy: max
  # enabled_metrics: ["ragas:faithfulness"]  # Optional: limit to specific metrics
```

**Output:** Includes aggregated score and `judge_scores` JSON array with individual results.

---

## Judge LLM configuration (legacy)

> **Deprecation notice:** The `llm` section will be replaced by `llm_pool` + `judge_panel`.

This section configures LLM. It is used when `judge_panel` is not configured (even if `llm_pool` exists).

### LLM
| Setting (llm.) | Default | Description |
|----------------|---------|-------------|
| provider | `"openai"` | LLM provider: openai, hosted_vllm, watsonx, azure, gemini |
| model | `"gpt-4o-mini"` | Model name for the provider |
| ssl_verify | `true` | Verify SSL certificates for specified provider |
| ssl_cert_file | `null` | Path to custom CA certificate file (PEM format, merged with certifi defaults) |
| temperature | `0.0` | Generation temperature |
| max_tokens |  `512` | Maximum tokens in response |
| timeout | `300` | Request timeout in seconds |
| num_retries | `3` | Maximum retry attempts |
| cache_dir | `".caches/llm_cache"` | Directory with cached LLM responses |
| cache_enabled | `true` | Is LLM cache enabled? |

Dynamic LLM parameters are only supported through `llm_pool` config. To use dynamic parameters, migrate to `llm_pool`.

**Note**: For RHAIIS, models.corp, or other vLLM-based inference servers, use the `hosted_vllm` provider configuration. `models.corp` additionally requires certificate setup via `ssl_cert_file` configuration option.

### Embeddings
Some Ragas metrics use embeddings to compute similarity between generated answers (or variants)

| Setting (embedding.) | Default | Description |
|----------------------|---------|-------------|
| provider | `"openai"` | Supported providers: `"openai"`, `"gemini"` or `"huggingface"`. `"huggingface"` downloads the model to the local machine and runs inference locally (requires optional dependencies).  |
| model | `"text-embedding-3-small"` | Model name for the provider |
| provider_kwargs | `{}` | Optional arguments for the model |
| cache_dir | `".caches/embedding_cache"` | Directory with cached embeddings |
| cache_enabled | `true` | Is embeddings cache enabled? |

#### Remote vs Local Embedding Models

By default, lightspeed-evaluation uses **remote embedding providers** (OpenAI, Gemini) which require no additional dependencies and are lightweight to install.

**Local embedding models** (HuggingFace/sentence-transformers) are **optional** and require additional packages including PyTorch (~6GB). This is to avoid long download times and wasted disk space for users who only need remote embeddings.

To use local HuggingFace embeddings, install the optional dependencies:
```bash
# Using pip
pip install 'lightspeed-evaluation[local-embeddings]'

# Using uv
uv sync --extra local-embeddings
```

### Example
```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  ssl_verify: true
  ssl_cert_file: null
  temperature: 0.0
  max_tokens: 512
  timeout: 300
  num_retries: 3
  cache_dir: ".caches/llm_cache"
  cache_enabled: true

embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  provider_kwargs: {}
  cache_dir: ".caches/embedding_cache"
  cache_enabled: true
```

### Example of non Gemini + Hugging Face setup
```yaml
llm:
  provider: "gemini"      # Judge-LLM Google Gemini
  model: "gemini-1.5-pro"    
  temperature: 0.0  
  max_tokens: 512  
  timeout: 120        
  num_retries: 3

embedding:
  provider: "huggingface"
  model: "sentence-transformers/all-mpnet-base-v2"
  provider_kwargs:
    # cache_folder: <path_with_pre_downloaded_model>
    model_kwargs:
      device: "cpu"  # Use "gpu" for nvidia accelerated inference
```

## Lightspeed API for real-time data generation
This section configures the inference API for generating the responses. It can be any Lightspeed-Core compatible API.
Note that it can be easily integrated with other APIs with a minimal change.

| Setting (api.) | Default | Description |
|----------------|---------|-------------|
| enabled | `"true"` |  Enable/disable API calls |
| api_base | `"http://localhost:8080"` | Base API URL |
| endpoint_type | `"streaming"` | streaming or query endpoint |
| timeout | `300` | API request timeout in seconds  |
| provider | `"openai"` | LLM provider for API queries (optional) |
| model | `"gpt-4o-mini"` | Model to use for API queries (optional) |
| no_tools | `null` | Whether to bypass tools (optional) |
| system_prompt | `null` | Custom system prompt (optional) |
| cache_dir | `".caches/api_cache"` | Directory with cached API responses |
| cache_enabled | `true` | Is API cache enabled? |
| mcp_headers | `null` | MCP headers configuration for authentication (see below) |
| num_retries | `3` | Maximum number of retry attempts for API calls on 429 errors |

### MCP Server Authentication

The framework supports two methods for MCP server authentication:

#### 1. New MCP Headers Configuration (Recommended)
The `mcp_headers` configuration provides a flexible way to configure authentication for individual MCP servers:

| Setting (api.mcp_headers.) | Default | Description |
|----------------------------|---------|-------------|
| enabled | `true` | Enable/disable MCP headers functionality |
| servers | `{}` | Dictionary of MCP server configurations |

For each server in `servers`, you can configure:

| Setting (api.mcp_headers.servers.<server_name>.) | Default | Description |
|---------------------------------------------------|---------|-------------|
| env_var | required | Environment variable containing the authentication token |
| header_name | `"Authorization"` | Custom header name (optional) |

#### 2. Legacy Authentication (Fallback)
When `mcp_headers.enabled` is `false`, the system falls back to using the `API_KEY` environment variable for all MCP server authentication.

### API Modes

#### With API Enabled (`api.enabled: true`)
- **Real-time data generation**: Queries are sent to external API
- **Dynamic responses**: `response` and `tool_calls` fields populated by API
- **Conversation context**: Conversation context is maintained across turns
- **Authentication**: Use `API_KEY` environment variable
- **Data persistence**: Saves amended `response` and `tool_calls` to the output data file in the output directory so it can be re-used with API option disabled

#### With API Disabled (`api.enabled: false`)
- **Static data mode**: Use pre-filled `response` and `tool_calls` from the input data
- **Faster execution**: No external API calls -- LLM as a judge are still called
- **Reproducible results**: Same response data used across runs
### Example

#### Example Configuration

```yaml
api:
  enabled: true
  api_base: http://localhost:8080
  endpoint_type: streaming
  timeout: 300
  
  provider: openai
  model: gpt-4o-mini
  no_tools: null
  system_prompt: null
  cache_dir: ".caches/api_cache"
  cache_enabled: true
  num_retries: 3
  
  # MCP Server Authentication Configuration
  mcp_headers:
    enabled: true                      # Enable MCP headers functionality
    servers:                          # MCP server configurations
      filesystem-tools:
        env_var: API_KEY              # Environment variable containing the token/key
      another-mcp-server:
        env_var: ANOTHER_API_KEY      # Use a different environment variable
```

#### Lightspeed Stack API Compatibility

**Important Note for lightspeed-stack API users**: To use the MCP headers functionality with the lightspeed-stack API, you need to modify the `llama_stack_api/openai_responses.py` file in your lightspeed-stack installation:

In the `OpenAIResponsesToolMCP` class, change the `authorization` parameter's `exclude` field from `True` to `False`:

```python
# In llama_stack_api/openai_responses.py
class OpenAIResponsesToolMCP:
    authorization: Optional[str] = Field(
        default=None,
        exclude=False  # Change this from True to False
    )
```

This change allows the authorization headers to be properly passed through to MCP servers.

## Metrics
Metrics are enabled globally (as described below) or within the input data for each individual conversation or individual turn (question/answer pair). To enable a metrics globally you need to set `default` meta data attribute to `true`

Metrics metadata are optional attributes for a given metric. Typically it contains the following:
- `default` -- `true` or `false`, Is this metric is applied by default when no turn_metrics specified?
- `threshold` -- numerical value, if the returned metric value is greater or equal than the threshold the metric
is marked a `PASS` in the results. If the returned metric value is lower it is marked as `FAIL`.
In case of error it is marked `ERROR`.
- `description` -- Description of the metric.

For **GEval** metrics (`geval:...`), you can also set:

- **`criteria`** (required): Natural-language description of what to evaluate. GEval uses this to generate evaluation steps when `evaluation_steps` is not provided.
- **`evaluation_params`**: List of field names to include (e.g. `query`, `response`, `expected_response`). GEval auto-detect is not supported.
- **`evaluation_steps`** (optional): List of step-by-step instructions the LLM judge follows. If omitted, GEval generates steps from `criteria`. When provided together with `rubrics`, both are used: steps define how to evaluate, rubrics define score-range boundaries; neither overrides the other.
- **`rubrics`** (optional): List of `{ score_range: [min, max], expected_outcome: "..." }`. Score range is 0–10 inclusive; DeepEval expects non-overlapping ranges and validates. Confines the judge’s output to these ranges. The final score is normalized to a 0–1 range.

GEval returns a score in **[0, 1]**.

By default no metrics are computed (`default` is set to `false`).

| Setting (metrics_metadata.) | Description |
|-----------------------------|-------------|
| turn_level | Turn level metrics metadata |
| conversation_level | Conversation level metrics metadata |

### Example
For complete example with all metrics see the default config file [config/system.yaml](config/system.yaml).
```yaml
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
      default: false  # Only used when explicitly specified in the input data
  
  conversation_level:
    "deepeval:conversation_completeness":
      threshold: 0.8
      description: "How completely the conversation addresses user intentions"
```

## Storage
Lightspeed Evaluation can persist results to files and/or databases. The `storage` section configures one or more storage backends.

### File Backend
The file backend generates CSV, JSON, and TXT reports.

| Setting (storage[type="file"].) | Default | Description |
|---------------------------------|---------|-------------|
| type | `"file"` | Backend type (required) |
| output_dir | `"./eval_output"` | Directory for output files |
| base_filename | `"evaluation"` | Prefix for output filenames |
| enabled_outputs | `["csv", "json", "txt"]` | Output types to generate |
| csv_columns | all listed below | Columns to include in CSV |
| summary_config_sections | `["llm", "embedding", "api"]` | Config sections in summary |

### Database Backend (Optional)
Save results to a database for querying and analysis. Supports SQLite, PostgreSQL, and MySQL.

| Setting (storage[type="sqlite/postgres/mysql"].) | Default | Description |
|--------------------------------------------------|---------|-------------|
| type | required | `"sqlite"`, `"postgres"`, or `"mysql"` |
| database | required | Database name or file path (SQLite) |
| table_name | `"evaluation_results"` | Table name for results |
| host | required* | Database host (*required for postgres/mysql) |
| port | default per type | Database port (5432 for postgres, 3306 for mysql) |
| user | required* | Database user (*required for postgres/mysql) |
| password | required* | Database password (*required for postgres/mysql) |

> **Note:** Database storage is incremental - results are saved as each conversation completes. Storage failures are logged as warnings but don't stop the evaluation.

### Output types

| Output type (in `enabled_outputs`) | Description |
|------------------------------------|-------------|
| csv | Detailed results CSV |
| json | Summary JSON with statistics |
| txt | Human-readable summary |

### CSV columns configuration
| CSV column name | Description |
|-----------------|-------------|
| conversation_group_id | Conversation group id |
| tag | Tag for grouping eval conversations |
| turn_id | Turn id |
| metric_identifier | Metric name |
| result | Result -- PASS/FAIL/ERROR/SKIPPED |
| score | Score returned by the metric |
| threshold | Threshold from the setup |
| reason | Human readable description of the result |
| query | Original input query |
| response | Original input response (could be generated by Lightspeed Core API) |
| execution_time | Total time for processing the metric |
| api_input_tokens | Number of input tokens used in API call (see below) |
| api_output_tokens | Number of output tokens from API call (see below) |
| judge_llm_input_tokens | Number of input tokens used by Judge LLM |
| judge_llm_output_tokens | Number of output tokens from Judge LLM |
| tool_calls | Tool calls made during the turn (JSON format) |
| contexts | Context documents used for evaluation |
| expected_response | Expected response for comparison |
| expected_intent | Expected intent for intent evaluation |
| expected_keywords | Expected keywords for keyword matching |
| expected_tool_calls | Expected tool calls for tool evaluation |
| metric_metadata |  Metric level metadata (excluding threshold & metric_identifier)|

For **turn-level** metrics, `api_input_tokens` and `api_output_tokens` reflect the token usage for that specific turn’s API call. For **conversation-level** metrics, these values are the **sum of token usage across all turns** in the conversation.

> **Note:** The `api_input_tokens` and `api_output_tokens` columns repeat the same value for every metric row belonging to the same turn (or conversation). Summing these columns across all CSV rows will **over-count**. For accurate API token totals, use the summary statistics in the JSON or TXT reports.

### Example: File Backend Only
```yaml
storage:
  - type: "file"
    output_dir: "./eval_output"
    base_filename: "evaluation"
    enabled_outputs:
      - "csv"
      - "json"
      - "txt"
    csv_columns:
      - "conversation_group_id"
      - "turn_id"
      - "metric_identifier"
      - "result"
      - "score"
      - "reason"
```

### Example: File + SQLite Database
```yaml
storage:
  - type: "file"
    output_dir: "./eval_output"
    base_filename: "evaluation"
    enabled_outputs:
      - csv
      - json
  - type: "sqlite"
    database: "./eval_results.db"
    table_name: "evaluation_results"
```

### Example: File + PostgreSQL Database
```yaml
storage:
  - type: "file"
    output_dir: "./eval_output"
  - type: "postgres"
    database: "evaluations"
    host: "localhost"
    port: 5432
    user: "admin"
    password: "secret"
```

## Visualization of the results
Several output graphs summarizing output results are provided. The graphs are generated by `matplotlib`.
Note, in some specific cases the generation is slowed down by connecting `matplotlib` to the local X-server.
To workaround this set `DISPLAY` variable to some non-existing value.

| Setting (visualization.) | Default | Description |
|--------------------------|---------|-------------|
|  figsize | `[12, 8]` | Graph size (width, height) in inches |
|  dpi | 300 | The resolution of the figure in dots-per-inch. |
|  enabled_graphs | all listed in the table below | List of the graphs to generate |

### Enabled graphs configuration
| Graph name | Description |
|-----------------|-------------|
| pass_rates | Pass rate bar chart |
| score_distribution | Score distribution box plot |
| conversation_heatmap | Heatmap of conversation performance |
| status_breakdown | Pie chart for pass/fail/error breakdown |

### Example
```yaml
visualization:
  figsize: [12, 8]
  dpi: 300
  enabled_graphs:
    - "pass_rates"
    - "score_distribution"
    - "conversation_heatmap"
    - "status_breakdown"
```
## Environment
It is possible to configure value of environment variables. The variables are set before imports, affecting certain libraries/packages. See the example below.

### Example
```yaml
environment:
  DEEPEVAL_TELEMETRY_OPT_OUT: "YES"        # Disable DeepEval telemetry
  DEEPEVAL_DISABLE_PROGRESS_BAR: "YES"     # Disable DeepEval progress bars
  LITELLM_LOG: ERROR                       # Suppress LiteLLM verbose logging
```

## Logging
Logging is highly configurable even for specific Python packages. Possible logging levels are:
- DEBUG, INFO, WARNING, ERROR, CRITICAL

| Setting (logging.) | Default | Description |
|--------------------|---------|-------------|
| source_level | `INFO` | Source code logging level |
| package_level | `ERROR` | Package logging level (imported libraries) |
| log_format | `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` | Log format and display options |
| show_timestamps | `true` | Show timestamps in logging messages? |
| package_overrides | none | List of specific package log levels (override package_level for specific libraries) |

### Example
```yaml
logging:
  source_level: INFO
  package_level: ERROR
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  show_timestamps: true
  package_overrides:
    httpx: ERROR
    urllib3: ERROR
    requests: ERROR
    matplotlib: ERROR
    LiteLLM: WARNING
    DeepEval: WARNING
    ragas: WARNING
```

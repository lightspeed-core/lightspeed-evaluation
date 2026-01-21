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

## Judge LLM configuration
This section configures LLM as a judge for both Ragas and DeepEval.

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

Authentication via `API_KEY` environment variable only for MCP server.

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
```

## Metrics
Metrics are enabled globally (as described below) or within the input data for each individual conversation or individual turn (question/answer pair). To enable a metrics globally you need to set `default` meta data attribute to `true`

Metrics metadata are optional attributes for a given metric. Typically it contains the following:
- `default` -- `true` or `false`, Is this metric is applied by default when no turn_metrics specified?
- `threshold` -- numerical value, if the returned metric value is greater or equal than the threshold the metric
is marked a `PASS` in the results. If the returned metric value is lower it is marked as `FAIL`.
In case of error it is marked `ERROR`.
- `description` -- Description of the metric.

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

## Output
Lightspeed Evaluation produces several outputs with the results and possibly modified input file with responses from Lightspeed Core API.

| Setting (output.) | Default | Description |
|-------------------|---------|-------------|
| output_dir | `"./eval_output"` | Directory with output files. |
| base_filename | `"evaluation"` | Prefix for output filenames. |
| enabled_outputs | `["csv", "json", "txt"]` | List with a specific output types, see below |
| csv_columns | all listed in the table below | List of columns to include in the "detailed results CSV" file, see below |

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
| api_input_tokens | Number of input tokens used in API call |
| api_output_tokens | Number of output tokens from API call |
| judge_llm_input_tokens | Number of input tokens used by Judge LLM |
| judge_llm_output_tokens | Number of output tokens from Judge LLM |
| tool_calls | Tool calls made during the turn (JSON format) |
| contexts | Context documents used for evaluation |
| expected_response | Expected response for comparison |
| expected_intent | Expected intent for intent evaluation |
| expected_keywords | Expected keywords for keyword matching |
| expected_tool_calls | Expected tool calls for tool evaluation |
| metric_metadata |  Metric level metadata (excluding threshold & metric_identifier)|

### Example
```yaml
output:
  output_dir: "./eval_output"
  base_filename: "evaluation"
  enabled_outputs:
    - "csv"
    - "json"
    - "txt"

  csv_columns:
    - "conversation_group_id"
    - "tag"
    - "turn_id"
    - "metric_identifier"
    - "result"
    - "score"
    - "threshold"
    - "reason"
    - "query"
    - "response"
    - "execution_time"
    - "api_input_tokens"
    - "api_output_tokens"
    - "judge_llm_input_tokens"
    - "judge_llm_output_tokens"
    - "tool_calls"
    - "contexts"
    - "expected_response"
    - "expected_intent"
    - "expected_keywords"
    - "expected_tool_calls"
    - "metric_metadata"
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

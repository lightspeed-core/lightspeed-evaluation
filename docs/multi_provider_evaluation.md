# Multi-Provider Evaluation Runner

## Overview

The Multi-Provider Evaluation Runner automates evaluations across multiple LLM providers and models in a single execution. It systematically tests your GenAI application against different provider/model combinations, organizing results in a structured hierarchy.

**Key Features:**
- Evaluate multiple providers (OpenAI, Watsonx, Gemini, vLLM) and models automatically
- Sequential execution with per-provider/model result isolation
- Robust error handling - individual failures don't stop the run
- Security: Path traversal protection and input sanitization
- Comprehensive summary with success/failure statistics

## Simple Example: Comparing 2 OpenAI Models

Here's a complete example comparing `gpt-4o-mini` and `gpt-4o` for a Lightspeed AI assistant:

### Step 1: Create `multi_eval_config.yaml`

```yaml
# config/multi_eval_config.yaml
providers:
  openai:
    models:
      - "gpt-4o-mini"
      - "gpt-4o"

settings:
  output_base: "./eval_output"
```

### Step 2: Set Environment Variables

```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
```

### Step 3: Run Multi-Provider Evaluation

```bash
cd lightspeed-evaluation

# Run evaluation comparing both models
uv run python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml \
    --system-config config/system.yaml \
    --eval-data config/evaluation_data.yaml
```

### Step 4: View Results

**Console Output:**
```
================================================================================
MULTI-PROVIDER EVALUATION SUMMARY
================================================================================

Total Evaluations: 2
Successful: 2 ✓
Failed: 0 ✗
Success Rate: 100.0%

Output Directory: eval_output

--------------------------------------------------------------------------------
Individual Results:
--------------------------------------------------------------------------------

openai/gpt-4o-mini:
  Status: ✓ SUCCESS
  Duration: 45.2s
  Output: eval_output_comparison/openai/gpt-4o-mini
  Results: Pass: 4, Fail: 1, Error: 0

openai/gpt-4o:
  Status: ✓ SUCCESS
  Duration: 52.1s
  Output: eval_output_comparison/openai/gpt-4o
  Results: Pass: 5, Fail: 0, Error: 0

================================================================================
STATISTICAL MODEL COMPARISON & BEST MODEL ANALYSIS
================================================================================

🏆 BEST MODEL
================================================================================

Model: openai/gpt-4o
Composite Score: 0.9200 (higher is better)

Performance Summary:
  ✅ Pass Rate: 100.00%
  📊 Mean Score: 0.8650
  📈 95% Confidence Interval: [0.8420, 0.8880]
  
Why This Model is Best:
  • Highest composite score combining multiple metrics
  • Pass rate: 5/5
  • Higher accuracy on answer_correctness metric

RECOMMENDATIONS
================================================================================

✅ RECOMMENDED MODEL: openai/gpt-4o

This model achieved the highest composite score considering pass rate,
mean score, and error rate. However, gpt-4o-mini is a close second with
80% pass rate and may be more cost-effective for your use case.
```

**Output Files:**
```
eval_output/
├── openai/
│   ├── gpt-4o-mini/
│   │   ├── evaluation_20251023_120000_detailed.csv
│   │   ├── evaluation_20251023_120000_summary.json
│   │   ├── evaluation_20251023_120000_summary.txt
│   │   └── graphs/
│   │       ├── evaluation_20251023_120000_pass_rates.png
│   │       └── evaluation_20251023_120000_score_distribution.png
│   └── gpt-4o/
│       └── [similar files...]
├── multi_provider_evaluation_summary.yaml
└── model_comparison_analysis.yaml  ← Best model recommendation
```

### Step 5: Review Best Model Analysis

```bash
cat eval_output_comparison/model_comparison_analysis.yaml
```

```yaml
total_models: 2
best_model:
  model: openai/gpt-4o
  composite_score: 0.92
  pass_rate: 1.0
  mean_score: 0.865
  error_rate: 0.0
  confidence_interval:
    low: 0.842
    high: 0.888
    mean: 0.865

rankings:
  - rank: 1
    model: openai/gpt-4o
    composite_score: 0.92
    statistics:
      overall:
        total_evaluations: 5
        passed: 5
        failed: 0
  
  - rank: 2
    model: openai/gpt-4o-mini
    composite_score: 0.85
    statistics:
      overall:
        total_evaluations: 5
        passed: 4
        failed: 1
```

## Quick Start

```bash
# Basic usage
python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml

# With all options
python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml \
    --system-config config/system.yaml \
    --eval-data config/evaluation_data.yaml \
    --verbose
```

## Configuration

### 1. Multi-Evaluation Configuration (`multi_eval_config.yaml`)

Define which providers and models to evaluate:

```yaml
# Providers section: Define evaluation targets
providers:
  openai:
    models:
      - "gpt-4o-mini"
      - "gpt-4-turbo"

  watsonx:
    models:
      - "ibm/granite-13b-chat-v2"
      - "meta-llama/llama-3-70b-instruct"

  gemini:
    models:
      - "gemini-1.5-pro"
      - "gemini-1.5-flash"

  hosted_vllm:
    models:
      - "meta-llama/Meta-Llama-3-8B-Instruct"

# Global settings
settings:
  output_base: "./eval_output"
```

### 2. System Configuration (`system.yaml`)

Standard LightSpeed system configuration. The script automatically overrides `api.provider` and `api.model` for each evaluation.

### 3. Evaluation Data (`evaluation_data.yaml`)

Standard evaluation data file - remains constant across all evaluations.

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--providers-config` | `config/multi_eval_config.yaml` | Path to multi-evaluation configuration |
| `--system-config` | `config/system.yaml` | Path to system configuration |
| `--eval-data` | `config/evaluation_data.yaml` | Path to evaluation data |
| `-v, --verbose` | `False` | Enable debug logging |

## Output Structure

```
eval_output/
├── openai/
│   ├── gpt-4o-mini/
│   │   ├── evaluation_YYYYMMDD_HHMMSS_detailed.csv
│   │   ├── evaluation_YYYYMMDD_HHMMSS_summary.json
│   │   └── graphs/
│   └── gpt-4-turbo/
│       └── ...
├── watsonx/
│   └── ibm_granite-13b-chat-v2/
│       └── ...
└── multi_provider_evaluation_summary.yaml
```

**Note:** Special characters in provider/model names are sanitized (e.g., `/` → `_`, `:` → `_`).

## How It Works

1. **Initialization:** Validates configuration files and creates output directory
2. **Sequential Evaluation:** For each provider/model combination:
   - Creates temporary system config with updated provider/model
   - Runs standard LightSpeed evaluation
   - Saves results to `{output_base}/{provider}/{model}/`
   - Cleans up temporary files
3. **Summary:** Generates consolidated summary with statistics
4. **Exit:** Returns 0 if all successful, 1 if any failed

## Security Features

### Path Traversal Protection
- **Input Sanitization:** Provider/model names restricted to `[A-Za-z0-9_.-]`
- **Path Validation:** Output directories verified to stay within `output_base`
- **Type Checking:** YAML configs must be dictionaries, not lists/strings

### Resource Management
- Automatic cleanup of temporary configuration files
- Dictionary key validation before access
- Comprehensive exception handling

## Error Handling

- **Individual Failures:** Logged and tracked, but don't stop remaining evaluations
- **Configuration Errors:** File not found, invalid YAML structure
- **Validation Errors:** Missing required keys, invalid data types
- **API Errors:** Authentication failures, rate limits (logged per evaluation)

Enable `--verbose` for full stack traces and debugging details.

## Environment Variables

Set API keys for the providers you're using:

```bash
export OPENAI_API_KEY="your-openai-key"
export WATSONX_APIKEY="your-watsonx-key"
export GOOGLE_API_KEY="your-gemini-key"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No valid provider-model configurations found" | Ensure at least one provider has models listed; check YAML syntax |
| API authentication errors | Verify environment variables are set and API keys are valid |
| Permission errors on output | Check write permissions and disk space |
| Evaluation fails for specific provider | Check provider-specific configuration and model availability |

## Advanced Usage

### Selective Evaluation

Comment out providers/models to skip:

```yaml
providers:
  openai:
    models:
      - "gpt-4o-mini"
      # - "gpt-4-turbo"  # Skip expensive model

  # watsonx:  # Skip entire provider
  #   models:
  #     - "ibm/granite-13b-chat-v2"
```

### CI/CD Integration

```yaml
name: Multi-Provider Evaluation
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: |
          pip install uv && uv sync --group dev
          python3 script/run_multi_provider_eval.py \
            --providers-config config/multi_eval_config.yaml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Best Practices

1. **Start Small:** Test with 1-2 models first to verify configuration
2. **Use Fast Models:** Start with cheaper models (e.g., `gpt-4o-mini`)
3. **Monitor Costs:** Watch API usage and rate limits
4. **Version Control:** Track `multi_eval_config.yaml` changes
5. **Regular Cleanup:** Remove old evaluation outputs periodically
6. **Meaningful Names:** Use descriptive output directory names

## Comparison: Standard vs Multi-Provider

| Feature | Standard Eval | Multi-Provider |
|---------|--------------|----------------|
| Single run | One provider/model | Multiple combinations |
| Configuration | Manual switching | Automatic |
| Output structure | Single directory | Hierarchical by provider/model |
| Error isolation | N/A | Continues on failure |
| Summary | Per evaluation | Consolidated |

## Related Documentation

- [Quick Start: Multi-Provider Setup](../QUICK_START_MULTI_PROVIDER.md)
- [Multi-Provider Setup Guide](../MULTI_PROVIDER_SETUP.md)
- [Evaluation Comparison](./evaluation_comparison.md)
- [Main README](../README.md)
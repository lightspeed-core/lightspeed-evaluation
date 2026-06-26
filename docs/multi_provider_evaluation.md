# Multi-Provider Evaluation Runner

## Overview

The Multi-Provider Evaluation Runner automates evaluations across multiple LLM providers and models in a single execution. It systematically tests your GenAI application against different provider/model combinations, organizing results in a structured hierarchy.

**Key Features:**
- Evaluate multiple providers (OpenAI, Watsonx, Gemini, vLLM) and models automatically
- **Parallel execution** for faster evaluation with configurable worker count
- Per-provider/model result isolation with independent output directories
- Robust error handling - individual failures don't stop the run
- Security: Path traversal protection and input sanitization
- Comprehensive summary with success/failure statistics and statistical comparison
- Real-time progress tracking with completion status

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
# Basic usage (parallel execution with default workers)
python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml

# With custom number of parallel workers
python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml \
    --max-workers 4

# Sequential execution (no parallelization)
python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml \
    --max-workers 1

# With all options
python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml \
    --system-config config/system.yaml \
    --eval-data config/evaluation_data.yaml \
    --max-workers 4 \
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
  max_workers: 4  # Optional: Number of parallel workers (default: CPU count - 1)
```

### 2. System Configuration (`system.yaml`)

Standard Lightspeed system configuration. The script automatically overrides `api.provider` and `api.model` for each evaluation.

### 3. Evaluation Data (`evaluation_data.yaml`)

Standard evaluation data file - remains constant across all evaluations.

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--providers-config` | `config/multi_eval_config.yaml` | Path to multi-evaluation configuration |
| `--system-config` | `config/system.yaml` | Path to system configuration |
| `--eval-data` | `config/evaluation_data.yaml` | Path to evaluation data |
| `--max-workers` | CPU count - 1 | Number of parallel workers (1 = sequential) |
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
2. **Parallel Evaluation:** Launches multiple worker processes (configurable) that run concurrently:
   - Each worker creates temporary system config with updated provider/model
   - Runs standard Lightspeed evaluation in isolated process
   - Saves results to `{output_base}/{provider}/{model}/` (no file conflicts)
   - Cleans up temporary files
   - Progress tracked and logged in real-time
3. **Result Synchronization:** Collects results as evaluations complete
4. **Statistical Analysis:** Performs comprehensive model comparison and ranking
5. **Summary:** Generates consolidated summary with best model recommendation
6. **Exit:** Returns 0 if all successful, 1 if any failed

## Parallel Execution

The multi-provider evaluation runner supports parallel execution using multiprocessing, significantly reducing overall runtime when evaluating multiple models.

### Benefits

- **Faster Execution:** Multiple models evaluated concurrently instead of sequentially
- **Independent Processes:** Each evaluation runs in isolated process with own memory space
- **No Data Conflicts:** Separate output directories prevent file system race conditions
- **Resource Optimization:** Configurable worker count to match system capabilities
- **Progress Tracking:** Real-time status updates as evaluations complete

### Configuration Options

Configure the number of parallel workers in three ways (priority order):

1. **Command-Line Argument** (highest priority):
   ```bash
   python3 script/run_multi_provider_eval.py \
       --providers-config config/multi_eval_config.yaml \
       --max-workers 4
   ```

2. **Configuration File** (`multi_eval_config.yaml`):
   ```yaml
   settings:
     output_base: "./eval_output"
     max_workers: 4  # Integer value (strings like "4" are automatically converted)
   ```

3. **Default:** CPU count - 1 (leaves one CPU free for system responsiveness)

**Note:** The `max_workers` value is automatically validated and coerced to an integer. String values from YAML (e.g., `"4"`) are converted automatically. Invalid values raise a clear error message.


### Resource Management with Multi-Threading

**IMPORTANT:** Each evaluation process uses multi-threading internally (configured via `core.max_threads` in `system.yaml`). When running parallel evaluations, you need to consider the interaction between:

- **`max_workers`**: Number of parallel evaluation processes
- **`core.max_threads`**: Number of threads per evaluation process

**Total Concurrent Threads = max_workers × core.max_threads**

#### Example Resource Calculation

```yaml
# config/multi_eval_config.yaml
settings:
  max_workers: 4

# config/system.yaml
core:
  max_threads: 50
```

**Total threads:** 4 workers × 50 threads = **200 concurrent threads**

On an 8-CPU machine, this could cause significant resource contention!

#### Recommended Configurations

| Dataset Size | CPU Count | max_workers | core.max_threads | Total Threads | Rationale |
|--------------|-----------|-------------|------------------|---------------|-----------|
| Small (<100 turns) | 8 | 4 | 10 | 40 | Balanced for quick completion |
| Medium (100-1000 turns) | 8 | 2 | 20 | 40 | More threads per process for throughput |
| Large (1000+ turns) | 8 | 2 | 25 | 50 | Optimize per-process throughput |
| Very Large (10k+ turns) | 16 | 4 | 20 | 80 | High parallelism on powerful machine |
| API Rate Limits | any | 2 | 10 | 20 | Reduce concurrent API calls |

#### Tuning Guidelines

**Rule of Thumb:** Keep total threads ≤ 2× CPU count

```python
# Target: Total threads ≈ CPU count × 1.5 to 2
max_workers × core.max_threads ≤ CPU_count × 2
```

**For Large Datasets (1000+ conversations):**
1. **Prioritize `core.max_threads`** - More important for processing many conversations within each evaluation
2. **Reduce `max_workers`** - Fewer parallel evaluations, but each completes faster
3. **Example:** `max_workers=2`, `core.max_threads=30` on 8-CPU machine

**For Many Small Evaluations:**
1. **Prioritize `max_workers`** - More parallel evaluations
2. **Reduce `core.max_threads`** - Each evaluation has fewer conversations
3. **Example:** `max_workers=6`, `core.max_threads=5` on 8-CPU machine

**For API Rate Limit Constraints:**
1. **Reduce both** - Limit concurrent API requests
2. **Example:** `max_workers=2`, `core.max_threads=5` = 10 concurrent API calls

#### Automatic Warning

The runner will automatically warn you if resource usage is high:

```
WARNING - High resource usage detected: 4 workers × 50 threads = 200 concurrent threads 
on 8 CPUs. Consider reducing max_workers or core.max_threads in system.yaml to avoid 
resource contention.
```

#### Testing with Large Datasets

When testing with thousands of conversations:

```bash
# Example: 5000 conversations, 8-CPU machine
# Configure for balanced performance

# 1. Update system.yaml
core:
  max_threads: 25 

# 2. Run with moderate parallelism
python3 script/run_multi_provider_eval.py \
    --providers-config config/multi_eval_config.yaml \
    --max-workers 2  # 2 × 25 = 50 total threads

# Monitor system resources:
# - CPU usage should stay < 90%
# - Memory usage should be stable
# - API rate limits should not be hit
```

### Implementation Details

**Process Isolation:**

Each evaluation runs in a separate process with:
- Independent Python interpreter
- Isolated memory space
- Separate file handles
- Own temporary configuration files
- Dedicated output directory
- Independent thread pool (configured by `core.max_threads`)

**Thread Safety:**

No shared state between evaluations ensures:
- No race conditions
- No file conflicts
- No memory corruption
- Clean error isolation
- Each process manages its own thread pool

**Error Handling:**

Individual worker failures don't affect other evaluations:
- Failed evaluations logged with status
- Successful evaluations continue
- Complete summary shows all results
- Exit code reflects overall success/failure

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
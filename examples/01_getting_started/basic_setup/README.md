# Basic Setup

Minimal configuration to start evaluating with LightSpeed Evaluation.

## Run Example

```bash
# From project root
export OPENAI_API_KEY="your-key"
uv run lightspeed-eval \
  --system-config examples/01_getting_started/basic_setup/system.yaml \
  --eval-data examples/01_getting_started/basic_setup/eval_data.yaml
```

## What It Does

- Evaluates 1 conversation with 1 turn
- Uses single metric: `custom:answer_correctness`
- Single thread, no caching
- Generates CSV, JSON, TXT reports + graphs
- Results written to: `examples/01_getting_started/basic_setup/eval_output/`

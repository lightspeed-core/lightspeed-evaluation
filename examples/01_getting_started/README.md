# Getting Started

Basic set up examples for LightSpeed Evaluation Framework.

## Examples

| Example | Description |
|---------|-------------|
| **CLI Mode** | Minimal configuration for CLI Mode |
| **Library Mode** | Programmatic API usage with Python |

## Quick Start

```bash
# Basic CLI mode
export OPENAI_API_KEY="your-key"
uv run lightspeed-eval \
  --system-config examples/01_getting_started/basic_setup/system.yaml \
  --eval-data examples/01_getting_started/basic_setup/eval_data.yaml

# Library mode
uv run python examples/01_getting_started/library_mode/example.py
```

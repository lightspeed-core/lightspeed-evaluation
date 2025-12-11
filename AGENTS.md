# AGENTS Guidelines
This document serves as a guide for AI coding agents working on the LightSpeed Evaluation Framework. This document provides context, instructions, and best practices to help agents understand the codebase and contribute effectively.

---

## ⚠️ IMPORTANT GUIDELINES

### 1. Directory Status & New Features

| Directory | Status | New Features? | Bug Fixes/Security/Linting? |
|-----------|--------|---------------|----------------------------|
| `src/lightspeed_evaluation/` | **Active Development** | ✅ Yes | ✅ Yes |
| `lsc_agent_eval/` | **Deprecated** - will be removed | ❌ No - add to `lightspeed_evaluation` instead | ✅ Yes (with confirmation) |
| `src/generate_answers/` | **Moving to separate repo** | ❌ No - will be relocated | ✅ Yes (with confirmation) |
| `tests/`, `config/`, `docs/` | **Active** | ✅ Yes | ✅ Yes |

**Before making changes to deprecated/transitional directories:**
- Inform the user about the directory's status
- Get explicit confirmation before proceeding
- For new features: suggest adding to `src/lightspeed_evaluation/` instead

### 2. Use pytest Mocking - NEVER unittest.mock
```python
# ❌ WRONG - DO NOT USE
from unittest.mock import patch, MagicMock

# ✅ CORRECT - ALWAYS USE
def test_example(mocker):
    mocker.patch('module.function')
```

### 3. Update Documentation When Changing Features
When modifying functionality, you **MUST** update:
- `docs/` - Update relevant guide if feature behavior changes
- `README.md` - Update when features change or adding new features
- `AGENTS.md` - Update if adding new conventions or project structure changes

### 4. Update This File When Receiving New Instructions
If the user provides new project conventions, coding standards, or constraints:
- **Add them to this `AGENTS.md` file** in the appropriate section
- Important guidelines go in this section above
- Best practices go in relevant sections below

### 5. Run Quality Checks Before Completing Work
Before considering any code change complete, you **MUST** run:

```bash
# Format code first
make format

# Run all quality checks (same as CI)
make verify          # Linting (black --check, ruff, pylint)
make check-types     # Type checking (mypy)
make pyright         # Type checking (pyright)
make docstyle        # Docstring style (pydocstyle)
make bandit          # Security scanning

# Run tests
make test            # Or: uv run pytest tests
```

**Do NOT skip these steps.** If any check fails:
1. Fix the issues in code you changed
2. For pre-existing issues in unchanged code: **notify the user** but don't fix (to avoid scope creep)
3. Re-run the checks until they pass
4. Only then consider the task complete

**Important:** Do NOT disable lint warnings (e.g., `# noqa`, `# type: ignore`, `# pylint: disable`). Always try to fix the underlying issue. If a fix becomes too complicated, inform the user and discuss alternatives.

---

## Project Overview

The LightSpeed Evaluation Framework is a comprehensive evaluation system for GenAI applications, supporting multiple evaluation frameworks (Ragas, DeepEval, custom metrics) with both turn-level and conversation-level assessments.

### Key Components

- **Core Framework**: Located in `src/lightspeed_evaluation/`
  - `core/`: Configuration, LLM management, metrics, output handling
  - `pipeline/`: Evaluation orchestration and processing
  - `runner/`: Command-line interface and main entry points
- **Configuration**: YAML-based system and evaluation data configs in `config/`
- **Testing**: Comprehensive test suite in `tests/` following pytest conventions

## Development Environment Setup

### Prerequisites
- Python 3.11+
- `uv` package manager (preferred) or `pip`

### Quick Setup
```bash
uv sync --group dev
make install-deps-test
```

### Environment Variables
Refer to `README.md` for full list. Key variables:
- `OPENAI_API_KEY` - Required for LLM evaluation
- `API_KEY` - Optional, for API-enabled evaluations
- `KUBECONFIG` - Optional, for script-based evaluations

## Architecture & Code Organization

### Core Module Structure
```text
src/lightspeed_evaluation/
├── core/
│   ├── api/           # API client for real-time data
│   ├── llm/           # LLM provider management
│   ├── metrics/       # Evaluation metrics (Ragas, DeepEval, custom)
│   ├── models/        # Pydantic data models
│   ├── output/        # Report generation and visualization
│   ├── script/        # Script execution for environment validation
│   └── system/        # Configuration and validation
├── pipeline/
│   └── evaluation/    # Main evaluation pipeline orchestration
└── runner/            # CLI interface and main entry points
```

### Key Classes
- `EvaluationPipeline`: Main orchestrator
- `SystemConfig` / `EvaluationData`: Pydantic config models
- `MetricManager`: Metric execution
- `OutputHandler`: Report generation

## Testing Guidelines

### Test Structure
Mirror the source code structure in `tests/`:
- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Mocking Example (pytest-mock)
```python
def test_llm_manager(mocker):
    """Test LLM manager with mocked provider."""
    mock_client = mocker.patch('lightspeed_evaluation.core.llm.openai.OpenAI')
    mock_client.return_value.chat.completions.create.return_value = mock_response
    
    manager = LLMManager(config)
    result = manager.evaluate_response(query, response)
    
    assert result.score > 0.5
```

### Coverage
- Aim for >80% on new code
- Run: `uv run pytest tests --cov=src --cov-report=html`

## Code Style & Quality

### Code Standards
- **Type Hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs
- **Error Handling**: Use custom exceptions from `core.system.exceptions`
- **Logging**: Use structured logging with appropriate levels

## Adding New Features

1. **Custom Metrics**: Add to `src/lightspeed_evaluation/core/metrics/custom/`
2. **Register**: Update `MetricManager` supported_metrics dictionary
3. **Configure**: Add metadata to `config/system.yaml` metrics_metadata section
4. **Test**: Add comprehensive tests with mocked LLM calls using pytest

## Troubleshooting

- **Configuration Errors**: Check `core/system/validator.py`
- **Metric Failures**: Enable DEBUG logging
- **API Issues**: Verify API_KEY and endpoint connectivity
- **Test Failures**: Run `make test` and check specific error messages

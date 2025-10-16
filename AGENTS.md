# AGENTS Guidelines
This document serves as a guide for AI coding agents working on the LightSpeed Evaluation Framework. This document provides context, instructions, and best practices to help agents understand the codebase and contribute effectively.

## Project Overview

The LightSpeed Evaluation Framework is a comprehensive evaluation system for GenAI applications, supporting multiple evaluation frameworks (Ragas, DeepEval, custom metrics) with both turn-level and conversation-level assessments.

### Key Components

- **Core Framework**: Located in `src/lightspeed_evaluation/`
  - `core/`: Configuration, LLM management, metrics, output handling
  - `pipeline/`: Evaluation orchestration and processing
  - `runner/`: Command-line interface and main entry points
- **Configuration**: YAML-based system and evaluation data configs in `config/`
- **Testing**: Comprehensive test suite in `tests/` following pytest conventions

**Note**: The `lsc_agent_eval/` directory is legacy and will not be maintained. All new evaluation features should be added to the main `src/lightspeed_evaluation/` framework.

## Development Environment Setup

### Prerequisites
- Python 3.11+
- `uv` package manager (preferred) or `pip`

### Quick Setup
```bash
# Install dependencies
uv sync --group dev

# Install development tools
make install-deps-test
```

### Environment Variables
Set these before running evaluations:

```bash
# Required: LLM provider for evaluation
export OPENAI_API_KEY="your-key"  # OpenAI
# Refer main README.md for other providers

# Optional: For API-enabled evaluations
export API_KEY="your-api-endpoint-key"

# Optional: For script-based evaluations
export KUBECONFIG="/path/to/kubeconfig"
```

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

### Key Classes & Interfaces
- `EvaluationPipeline`: Main orchestrator for evaluation workflows
- `SystemConfig`: Pydantic model for system configuration
- `EvaluationData`: Data model for evaluation inputs
- `MetricManager`: Handles metric execution and results
- `OutputHandler`: Generates reports and visualizations

## Configuration System

### System Configuration (`config/system.yaml`)
- **LLM Config**: Provider, model, temperature, caching
- **API Config**: Real-time data generation settings
- **Metrics Metadata**: Thresholds and descriptions for all metrics
- **Output Config**: Report formats and visualization settings
- **Logging Config**: Structured logging configuration

### Evaluation Data (`config/evaluation_data.yaml`)
- **Conversation Groups**: Collections of related turns
- **Turn Data**: Individual query-response pairs with expected outputs
- **Metrics Override**: Per-turn or per-conversation metric selection
- **Script Integration**: Setup/cleanup/verification scripts

Above data file is just for reference.
Add a new sample data yaml file while adding a new feature.

## Testing Guidelines

### Test Structure
Follow the existing test structure that mirrors the source code:

```text
tests/
├── unit/
│   └── core/
│       ├── config/
│       ├── llm/
│       ├── metrics/
│       └── output/
└── integration/
```

### Mocking with pytest
**Use pytest, not unittest.mock** for all mocking:

```python
def test_llm_manager(mocker):
    """Test LLM manager with mocked provider."""
    # Mock external dependencies
    mock_client = mocker.patch('lightspeed_evaluation.core.llm.openai.OpenAI')
    mock_client.return_value.chat.completions.create.return_value = mock_response
    
    # Test the actual functionality
    manager = LLMManager(config)
    result = manager.evaluate_response(query, response)
    
    assert result.score > 0.5
```

### Test File Naming
- Test files: `test_*.py` (e.g., `test_manager.py`)
- Test functions: `test_*` (e.g., `test_load_config`)
- Test classes: `Test*` (e.g., `TestMetricManager`)

### Coverage Requirements
- Aim for >80% code coverage on new code
- Run tests with coverage: `uv run pytest tests --cov=src --cov-report=html`
- Focus on testing business logic, not just coverage numbers

## Code Style & Quality

### Formatting & Linting
```bash
# Format code
make format

# Check style and types
make verify
make check-types

# Individual tools
make pylint    # Code quality
make pyright   # Type checking
make docstyle  # Docstring style
```

### Code Standards
- **Type Hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs
- **Error Handling**: Use custom exceptions from `core.system.exceptions`
- **Logging**: Use structured logging with appropriate levels

### Example Function
```python
def evaluate_response(
    query: str, 
    response: str, 
    context: list[str],
    metric_config: dict[str, Any]
) -> EvaluationResult:
    """Evaluate a response using specified metrics.
    
    Args:
        query: The input query that was sent to the model.
        response: The response generated by the model.
        context: List of context documents used for generation.
        metric_config: Configuration for metrics to apply.
    
    Returns:
        EvaluationResult containing scores and metadata.
    
    Raises:
        EvaluationError: If evaluation fails due to invalid inputs.
        APIError: If external API calls fail.
    """
```

## Supported Metrics

### Turn-Level Metrics
- **Ragas**: `faithfulness`, `response_relevancy`, `context_recall`, `context_precision_*`, `context_relevance`
- **Custom**: `answer_correctness`, `intent_eval`, `tool_eval`
- **Script**: `action_eval`

### Conversation-Level Metrics
- **DeepEval**: `conversation_completeness`, `conversation_relevancy`, `knowledge_retention`

## Key Features

- **Multi-Framework Support**: Ragas, DeepEval, custom metrics
- **LLM Providers**: OpenAI, Watsonx, Gemini, vLLM with caching
- **API Modes**: Real-time (`api.enabled: true`) or static data
- **Output Formats**: CSV, JSON, TXT + visualizations
- **Script Integration**: Setup/cleanup/verify scripts for environment validation

## Configuration

- **System Config** (`config/system.yaml`): LLM, API, metrics, output settings
- **Evaluation Data** (`config/evaluation_data.yaml`): Conversation groups and turns
- **Environment Variables**: Set environment variables for Judge LLM provider, API, Script execution as per the need.

## Usage

```bash
# Basic evaluation
lightspeed-eval --system-config config/system.yaml --eval-data config/evaluation_data.yaml

# With custom output directory
lightspeed-eval --system-config config/system.yaml --eval-data config/evaluation_data.yaml --output-dir ./my_results

# Generate answers (optional)
generate_answers --config config.yaml --input questions.csv --output answers.json
```

## Development Guidelines

### Testing
- **Use pytest, not unittest.mock** for all mocking
- **Mirror source structure** in `tests/` directory
- **Test files**: `test_*.py` format
- **Coverage**: Aim for >80% on new code

```bash
make test       # Run tests
```

### Best Practices
- **Type hints required** for all public functions
- **Google-style docstrings** for all public APIs
- **Modular design**
- **Focus on `src/lightspeed_evaluation/`** - ignore `archive/` folder

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
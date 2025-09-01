# LSC Evaluation Framework - Test Suite

This directory contains comprehensive test cases for the LSC Evaluation Framework, generated based on the `system.yaml` configuration file.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_runner.py           # Test runner script
├── README.md               # This file
├── core/                   # Core functionality tests
│   ├── test_config_loader.py    # ConfigLoader class tests
│   ├── test_models.py           # Pydantic models tests
│   └── test_data_validator.py   # DataValidator class tests
├── llm_managers/           # LLM manager tests
│   └── test_llm_manager.py      # LLMManager class tests
├── metrics/                # Metrics component tests
│   └── test_custom_metrics.py   # Custom metrics tests
└── output/                 # Output component tests
    └── test_utils.py            # Output utilities tests
```

## Test Categories

The tests are organized into several categories using pytest markers:

- **`unit`**: Unit tests for individual components
- **`integration`**: Integration tests across components
- **`config`**: Configuration loading and validation tests
- **`models`**: Pydantic model validation tests
- **`validation`**: Data validation tests
- **`output`**: Output generation and formatting tests
- **`slow`**: Tests that take longer to run
- **`llm`**: Tests requiring LLM API calls (may be skipped in CI)

## Running Tests

### Using the Test Runner Script

The easiest way to run tests is using the provided test runner:

```bash
# Run all tests
python tests/test_runner.py all

# Run specific test categories
python tests/test_runner.py unit
python tests/test_runner.py config
python tests/test_runner.py models
python tests/test_runner.py validation

# Run tests with coverage
python tests/test_runner.py coverage

# Run specific test file
python tests/test_runner.py file tests/core/test_models.py

# Run fast tests only (exclude slow tests)
python tests/test_runner.py fast
```

### Using pytest directly

You can also run tests directly with pytest:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/core/test_config_loader.py

# Run tests with specific markers
pytest -m "config" tests/
pytest -m "not slow" tests/

# Run with coverage
pytest --cov=lsc_eval --cov-report=html tests/
```

## Test Configuration

### Environment Setup

Tests use fixtures to set up clean environments:

- **`clean_environment`**: Clears environment variables before/after tests
- **`temp_dir`**: Provides temporary directory for test files
- **`sample_system_config`**: Provides sample system configuration
- **`sample_evaluation_data`**: Provides sample evaluation data

### Mock Data

Tests use realistic mock data based on the actual system.yaml configuration:

- **LLM Configuration**: OpenAI, Azure, Anthropic, Gemini, WatsonX, Ollama providers
- **Metrics**: Ragas, DeepEval, and Custom metrics as defined in system.yaml
- **Output Formats**: CSV, JSON, TXT formats with visualization options
- **Evaluation Data**: Multi-turn conversations with contexts and expected responses

## Test Coverage

The test suite covers:

### Core Components
- **ConfigLoader**: System configuration loading, environment setup, logging configuration
- **Models**: Pydantic model validation for TurnData, EvaluationData, EvaluationResult
- **DataValidator**: Evaluation data validation, metric requirements checking

### LLM Managers
- **LLMManager**: Provider-specific configuration, environment validation, model name construction

### Metrics
- **CustomMetrics**: LLM-based evaluation, score parsing, prompt generation

### Output Components
- **Utils**: Statistics calculation, result aggregation, evaluation scoping

## Key Test Scenarios

### Configuration Testing
- Valid and invalid system configurations
- Environment variable setup and validation
- Logging configuration with different levels
- Metric mapping and validation

### Model Validation Testing
- Field validation for all Pydantic models
- Edge cases and boundary conditions
- Required field validation
- Data type validation

### Data Validation Testing
- Evaluation data structure validation
- Metric requirement checking
- Context and expected response validation
- Multi-conversation validation

### LLM Manager Testing
- Provider-specific environment validation
- Model name construction for different providers
- Error handling for missing credentials
- Configuration parsing

### Metrics Testing
- Custom metric evaluation
- LLM response parsing
- Score normalization
- Error handling for failed evaluations

### Output Testing
- Statistics calculation
- Result aggregation by metric and conversation
- Score statistics computation
- Edge cases with empty or error results

## Running Tests in CI/CD

For continuous integration, you can:

```bash
# Run fast tests only (exclude slow/LLM tests)
pytest -m "not slow and not llm" tests/

# Run with XML output for CI systems
pytest --junitxml=test-results.xml tests/

# Run with coverage for code quality metrics
pytest --cov=lsc_eval --cov-report=xml --cov-report=term tests/
```

## Adding New Tests

When adding new functionality:

1. Create test files following the naming convention `test_*.py`
2. Use appropriate pytest markers to categorize tests
3. Follow the existing fixture patterns for setup/teardown
4. Include both positive and negative test cases
5. Test edge cases and error conditions
6. Update this README if adding new test categories

## Test Data

Test fixtures provide realistic data based on system.yaml:

- **Metrics**: All metrics defined in system.yaml with proper thresholds
- **Providers**: All LLM providers with required environment variables
- **Output Formats**: All output formats and visualization options
- **Evaluation Scenarios**: Multi-turn conversations with various metric combinations

This ensures tests accurately reflect the actual system configuration and usage patterns.


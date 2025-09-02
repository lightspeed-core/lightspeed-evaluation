# LightSpeed Evaluation Framework - Test Suite

This directory contains comprehensive tests for the LightSpeed Evaluation Framework. The test suite covers all major components and provides both unit and integration tests.

## Test Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # Pytest configuration and shared fixtures
├── run_tests.py             # Test runner script for convenient test execution
├── test_evaluation.py       # Main evaluation tests
├── test_config.py           # Configuration loading and validation tests
├── test_metrics.py          # Metrics evaluation tests
└── test_cli.py              # Command-line interface tests
```

## Test Categories

The tests are organized into several categories using pytest markers:

### By Component
- **`config`**: Configuration loading, validation, and environment setup
- **`metrics`**: Metric evaluation (Ragas, DeepEval, Custom)
- **`cli`**: Command-line interface and argument parsing
- **`output`**: Report generation and output handling

### By Type
- **`unit`**: Fast unit tests with mocked dependencies
- **`integration`**: Integration tests using real configuration files
- **`slow`**: Tests that take longer to run (usually integration tests)

## Running Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install pytest pytest-cov
```

### Basic Usage

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_config.py

# Run specific test class
python -m pytest tests/test_config.py::TestSystemConfig

# Run specific test method
python -m pytest tests/test_config.py::TestSystemConfig::test_system_config_defaults
```

### Using the Test Runner Script

The project includes a convenient test runner script located in the `tests/` directory:

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run only integration tests
python tests/run_tests.py --type integration

# Run tests by component
python tests/run_tests.py --type config
python tests/run_tests.py --type metrics
python tests/run_tests.py --type cli

# Run with coverage report
python tests/run_tests.py --coverage

# Run with verbose output
python tests/run_tests.py --verbose

# Run fast tests only (exclude slow tests)
python tests/run_tests.py --type fast

# Run specific test file
python tests/run_tests.py test_config.py

# Custom markers
python tests/run_tests.py --markers "unit and not slow"
```

### Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Run config-related tests
python -m pytest -m config

# Run metrics-related tests
python -m pytest -m metrics

# Run CLI-related tests
python -m pytest -m cli

# Exclude slow tests
python -m pytest -m "not slow"

# Combine markers
python -m pytest -m "unit and config"
```

## Test Configuration

### Environment Variables

The tests automatically set up required environment variables:

- `OPENAI_API_KEY`: Set to a test value for mocking
- `DEEPEVAL_TELEMETRY_OPT_OUT`: Disabled for testing
- `LITELLM_LOG_LEVEL`: Set to ERROR to reduce noise

### Fixtures

The test suite provides several useful fixtures in `conftest.py`:

- **`sample_system_config`**: Pre-configured SystemConfig object
- **`sample_llm_config`**: Pre-configured LLMConfig object
- **`sample_turn_data`**: Sample conversation turn data
- **`sample_evaluation_data`**: Complete evaluation data structure
- **`mock_llm_manager`**: Mocked LLM manager for testing
- **`temp_config_files`**: Temporary configuration files
- **`temp_output_dir`**: Temporary output directory

## Test Coverage

To generate a coverage report:

```bash
# Generate HTML coverage report
python -m pytest --cov=lightspeed_evaluation --cov-report=html tests/

# Generate terminal coverage report
python -m pytest --cov=lightspeed_evaluation --cov-report=term-missing tests/

# Using the test runner
python tests/run_tests.py --coverage
```

The HTML coverage report will be generated in `htmlcov/index.html`.

## Writing New Tests

### Test File Organization

- **Unit tests**: Test individual functions/classes with mocked dependencies
- **Integration tests**: Test component interactions with real or realistic data
- **Use descriptive test names**: `test_load_system_config_with_valid_file`
- **Group related tests**: Use test classes to organize related functionality

### Example Test Structure

```python
class TestMyComponent:
    """Test MyComponent functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality with valid input."""
        # Arrange
        component = MyComponent()
        
        # Act
        result = component.do_something()
        
        # Assert
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        component = MyComponent()
        
        with pytest.raises(ValueError, match="Expected error message"):
            component.do_something_invalid()
    
    @pytest.mark.integration
    def test_integration_scenario(self):
        """Test integration with other components."""
        # Integration test code here
        pass
```

### Using Fixtures

```python
def test_with_fixtures(sample_system_config, temp_output_dir):
    """Test using provided fixtures."""
    # Use the fixtures in your test
    assert sample_system_config.llm_provider == "openai"
    assert Path(temp_output_dir).exists()
```

### Mocking External Dependencies

```python
@patch('lightspeed_evaluation.core.metrics.ragas.evaluate')
def test_with_mocked_dependency(mock_evaluate):
    """Test with mocked external dependency."""
    # Configure mock
    mock_evaluate.return_value = MagicMock()
    
    # Run test
    result = my_function_that_uses_ragas()
    
    # Verify mock was called
    mock_evaluate.assert_called_once()
```

## Continuous Integration

The test suite is designed to work in CI environments:

- All external dependencies are mocked
- Temporary files are properly cleaned up
- Tests are deterministic and don't rely on external services
- Environment variables are properly managed

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the package is installed in development mode:
   ```bash
   pip install -e .
   ```

2. **Missing Dependencies**: Install test dependencies:
   ```bash
   pip install pytest pytest-cov
   ```

3. **Configuration File Tests**: Some tests require the actual config files to exist:
   - `config/system.yaml`
   - `config/evaluation_data.yaml`

4. **Environment Variables**: Tests automatically set required environment variables, but you can override them if needed.

### Debug Mode

Run tests with more verbose output for debugging:

```bash
python -m pytest tests/ -v -s --tb=long
```

### Running Individual Tests

For debugging specific tests:

```bash
# Run a specific test with full output
python -m pytest tests/test_config.py::TestSystemConfig::test_system_config_defaults -v -s

# Run with pdb debugger on failure
python -m pytest tests/test_config.py --pdb
```

## Contributing

When adding new functionality:

1. Write tests for new features
2. Ensure good test coverage (aim for >90%)
3. Use appropriate markers for test categorization
4. Mock external dependencies
5. Add integration tests for complex workflows
6. Update this README if adding new test categories or patterns

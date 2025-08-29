# CONTRIBUTING

<!-- the following line is used by tool to autogenerate Table of Content when the document is changed -->
<!-- vim-markdown-toc GFM -->

* [TLDR;](#tldr)
* [Prerequisites](#prerequisites)
    * [Tooling installation](#tooling-installation)
* [Setting up your development environment](#setting-up-your-development-environment)
* [Definition of Done](#definition-of-done)
    * [A deliverable is to be considered "done" when](#a-deliverable-is-to-be-considered-done-when)
* [Automation](#automation)
    * [Code coverage measurement](#code-coverage-measurement)
    * [Type hints checks](#type-hints-checks)
    * [Linters](#linters)
* [Testing](#testing)
    * [Tips and hints for developing unit tests](#tips-and-hints-for-developing-unit-tests)
        * [Patching](#patching)
        * [Verifying that some exception is thrown](#verifying-that-some-exception-is-thrown)
        * [Checking what was printed and logged to stdout or stderr by the tested code](#checking-what-was-printed-and-logged-to-stdout-or-stderr-by-the-tested-code)
* [Code style](#code-style)
    * [Docstrings style](#docstrings-style)

<!-- vim-markdown-toc -->

## TLDR;

1. Create your own fork of the repo
2. Make changes to the code in your fork
3. Run unit tests and verification checks
4. Check the code with linters
5. Submit PR from your fork to main branch of the project repo

## Prerequisites

- git
- Python 3.11 or higher
- pip

The development requires at least [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html) due to dependencies on modern ML/AI libraries and evaluation frameworks that leverage the latest Python features for performance and compatibility.

### Tooling installation

1. `pip install --user uv`
1. `uv --version` -- should return no error

## Setting up your development environment

```bash
# clone your fork
git clone https://github.com/YOUR-GIT-PROFILE/lightspeed-evaluation.git

# move into the directory
cd lightspeed-evaluation

# setup your development environment with uv
uv sync --group dev

# Now you can run commands through make targets, or prefix commands with `uv run`

# install all development tools
make install-deps-test

# run tests
make test

# code formatting
make format

# code style and docstring style
make verify

# check type hints
make check-types

# run evaluation (requires OLS API to be running)
uv run evaluate --help
```

Happy hacking!

## Definition of Done

### A deliverable is to be considered "done" when

* Code is complete, commented, and merged to the relevant release branch
* User facing documentation written (where relevant)
* Acceptance criteria in the related Jira ticket (where applicable) are verified and fulfilled
* Pull request title+commit includes Jira number
* Changes are covered by unit tests that run cleanly in the CI environment (where relevant)
* Evaluation tests pass with the updated code (where relevant)
* All linters are running cleanly in the CI environment
* Code changes reviewed by at least one peer
* Code changes acked by at least one project owner

## Automation

### Code coverage measurement

Code coverage tools are available through the `pytest-cov` plugin, which is installed as a development dependency. However, coverage measurement is not currently configured by default in the test runs. To run tests with coverage measurement, you can use:

```
uv run pytest tests --cov=src --cov-report=html
```

This will generate coverage reports in the `htmlcov` subdirectory.

### Type hints checks

It is possible to check if type hints added into the code are correct and whether assignments, function calls etc. use values of the right type. This check is invoked by following command:

```
make check-types
```

Please note that type hints check might be very slow on the first run.
Subsequent runs are much faster thanks to the cache that Mypy uses. This check
is part of a CI job that verifies sources.

### Linters

_Black_, _Ruff_, _Pyright_, and _Pylint_ tools are used as linters. These tools are installed as development dependencies. Currently, only basic Mypy configuration is present in `pyproject.toml` in the `[tool.mypy]` section. Additional linter configurations can be added as needed.

List of all _Ruff_ rules recognized by Ruff can be retrieved by:

```
ruff linter
```

Description of all _Ruff_ rules are available on https://docs.astral.sh/ruff/rules/

Ruff rules can be disabled in source code (for given line or block) by using special `noqa` comment line. For example:

```python
# noqa: E501
```

List of all _Pylint_ rules can be retrieved by:

```
pylint --list-msgs
```

Description of all rules are available on https://pylint.readthedocs.io/en/latest/user_guide/checkers/features.html

To disable _Pylint_ rule in source code, the comment line in following format can be used:

```python
# pylint: disable=C0415
```


## Testing

Tests are used in this repository to verify the correctness of evaluation logic, data processing, and utility functions. The tests are designed to ensure that:

1. Evaluation metrics are calculated correctly
2. Data processing pipelines work as expected
3. API interactions function properly
4. Configuration parsing is robust

Tests can be started by using the following command:

```
make test
```

All tests are based on the [Pytest framework](https://docs.pytest.org/en/). Code coverage can be measured using the [pytest-cov](https://github.com/pytest-dev/pytest-cov) plugin, which is available as a development dependency. For mocking and patching, the [unittest framework](https://docs.python.org/3/library/unittest.html) is used.

As specified in Definition of Done, new changes need to be covered by tests.

### Tips and hints for developing unit tests

#### Patching

**WARNING**:
Since tests are executed using Pytest, which relies heavily on fixtures,
we discourage use of `patch` decorators in all test code, as they may interact with one another.

It is possible to use patching inside the test implementation as a context manager:

```python
def test_xyz():
    with patch("lightspeed_core_evaluation.config", new=Mock()):
        ...
        ...
        ...
```

- `new=` allow us to use different function or class
- `return_value=` allow us to define return value (no mock will be called)

#### Verifying that some exception is thrown

Sometimes it is needed to test whether some exception is thrown from a tested function or method. In this case `pytest.raises` can be used:

```python
def test_evaluation_with_invalid_config(invalid_config):
    """Check if wrong configuration is detected properly."""
    with pytest.raises(ValueError):
        evaluate_model(invalid_config)
```

It is also possible to check if the exception is thrown with the expected message. The message (or its part) is written as regexp:

```python
def test_constructor_no_provider():
    """Test that constructor checks for provider."""
    # we use bare Exception in the code, so need to check
    # message, at least
    with pytest.raises(Exception, match="ERROR: Missing provider"):
        load_evaluation_model(provider=None)
```

#### Checking what was printed and logged to stdout or stderr by the tested code

It is possible to capture stdout and stderr by using standard fixture `capsys`:

```python
def test_evaluation_output(capsys):
    """Test the evaluation function that prints to stdout."""
    run_evaluation("test_config.yaml")

    # check captured log output
    captured_out = capsys.readouterr().out
    assert "Evaluation completed" in captured_out
    captured_err = capsys.readouterr().err
    assert captured_err == ""
```

Capturing logs:

```python
@patch.dict(os.environ, {"LOG_LEVEL": "INFO"})
def test_logger_show_message_flag(mock_load_dotenv, capsys):
    """Test logger set with show_message flag."""
    logger = Logger(logger_name="evaluation", log_level=logging.INFO, show_message=True)
    logger.logger.info("This is my debug message")

    # check captured log output
    # the log message should be captured
    captured_out = capsys.readouterr().out
    assert "This is my debug message" in captured_out

    # error output should be empty
    captured_err = capsys.readouterr().err
    assert captured_err == ""
```

## Code style

### Docstrings style

We are using [Google's docstring style](https://google.github.io/styleguide/pyguide.html).

Here is simple example:

```python
def evaluate_model_response(query: str, response: str, ground_truth: str) -> float:
    """Evaluate model response against ground truth using similarity metrics.
    
    Args:
        query: The input query that was sent to the model.
        response: The response generated by the model.
        ground_truth: The expected/correct response.
    
    Returns:
        The similarity score between response and ground truth (0.0 to 1.0).
    
    Raises:
        ValueError: If any of the input parameters are empty or None.
    """
```

For further guidance, see the rest of our codebase, or check sources online. There are many, eg. [this one](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e). 
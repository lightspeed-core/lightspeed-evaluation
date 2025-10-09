# Evaluation Comparison Script

This directory contains utility scripts for comparing evaluation runs.

## compare_evaluations.py

A standalone script for comparing two evaluation summary JSON files and determining statistical significance.

### Usage

The script accepts a list of exactly 2 JSON files for comparison:

```bash
# Basic usage
uv run python script/compare_evaluations.py summary1.json summary2.json

# With custom significance level
uv run python script/compare_evaluations.py summary1.json summary2.json --alpha 0.01

# Save results to JSON file
uv run python script/compare_evaluations.py summary1.json summary2.json --output results.json

# Quiet mode (only show errors)
uv run python script/compare_evaluations.py summary1.json summary2.json --quiet

# Verbose mode
uv run python script/compare_evaluations.py summary1.json summary2.json --verbose
```

### Arguments

- `summary_files`: List of evaluation summary JSON files (exactly 2 required)
- `--alpha`: Significance level for statistical tests (default: 0.05)
- `--output, -o`: Path to save comparison results as JSON file
- `--report-only`: Only generate and display the report, don't save JSON results
- `--quiet, -q`: Suppress output except for errors
- `--verbose, -v`: Enable verbose output

### Features

- **Automatic ordering**: Summaries are automatically ordered by timestamp
- **Statistical tests**: Performs t-tests, Mann-Whitney U tests, chi-square tests, and Fisher's exact tests
- **Comprehensive reporting**: Generates detailed human-readable reports
- **Flexible output**: Can save results to JSON or display report only
- **Error handling**: Validates input files and provides clear error messages

### Dependencies

The script requires the following Python packages:
- `numpy`
- `scipy`
- `json` (built-in)
- `pathlib` (built-in)
- `argparse` (built-in)

These dependencies are automatically available when using `uv run` in the project directory.

### Testing

Run the test suite to verify the script works correctly:

```bash
uv run python tests/test_compare_evaluations.py
```
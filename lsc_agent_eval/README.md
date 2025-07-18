# Lightspeed Agent Evaluation

A standalone package for evaluating agent-based systems, specifically designed for evaluating agent goal achievement.

## Features

- **Agent Goal Evaluation**: Evaluate whether an agent successfully achieves specified goals
- **Multi-type Evaluation**: Support for different evaluation types:
  - `judge-llm`: LLM-based evaluation using a judge model
  - `script`: Script-based evaluation using verification scripts (similar to [k8s-bench](https://github.com/GoogleCloudPlatform/kubectl-ai/tree/main/k8s-bench))
  - `sub-string`: Simple substring matching evaluation
- **Setup/Cleanup Scripts**: Support for running setup and cleanup scripts before/after evaluation
- **Result Tracking**: Result tracking and CSV output
- **Standalone Package**: Can be installed and used independently of the main lightspeed-core-evaluation package
- **LiteLLM Integration**: Unified interface for Judge LLM

## Installation

### Prerequisites

- Python 3.11 or higher
- Package manager: `pdm` or `pip`

### Install from Git

```bash
# Install directly from git repository
pip install git+https://github.com/lightspeed-core/lightspeed-evaluation.git#subdirectory=lsc_agent_eval

# Or install with pdm
pdm add git+https://github.com/lightspeed-core/lightspeed-evaluation.git#subdirectory=lsc_agent_eval
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/lightspeed-core/lightspeed-evaluation.git
cd lightspeed-evaluation/lsc_agent_eval

# Install with pip
pip install -e .

# Or install with pdm
pdm install
```

## Usage

### Command Line Interface

```bash
# Run agent evaluation with basic configuration
lsc-agent-eval \
    --eval_data_yaml agent_goal_eval.yaml \
    --agent_endpoint http://localhost:8080 \
    --agent_provider watsonx \
    --agent_model ibm/granite-3-2-8b-instruct \
    --judge_provider openai \
    --judge_model gpt-4o-mini \
    --result_dir ./eval_output
```

### Python API

```python
from lsc_agent_eval import AgentGoalEval

# Create evaluation configuration
class EvalArgs:
    def __init__(self):
        self.eval_data_yaml = 'data/example_eval.yaml'
        self.agent_endpoint = 'http://localhost:8080'
        self.agent_provider = 'watsonx'
        self.agent_model = 'ibm/granite-3-2-8b-instruct'
        self.judge_provider = 'openai'
        self.judge_model = 'gpt-4o-mini'
        self.agent_auth_token_file = None  # Or set `AGENT_API_TOKEN` env var
        self.result_dir = None

# Run evaluation
args = EvalArgs()
evaluator = AgentGoalEval(args)
evaluator.get_eval_result()
```

## Configuration

The evaluation is configured using a YAML file that defines test cases. Each test case can include:

- `eval_id`: Unique identifier for the evaluation
- `eval_query`: The query/task to send to the agent
- `eval_type`: Type of evaluation (judge-llm, script, sub-string)
- `expected_response`: Expected response (for judge-llm evaluation)
- `expected_key_words`: Keywords to look for (for sub-string evaluation)
- `eval_verify_script`: Verification script (for script evaluation)
- `eval_setup_script`: Optional setup script to run before evaluation
- `eval_cleanup_script`: Optional cleanup script to run after evaluation

### Example YAML Configuration

```yaml
# data/example_eval.yaml
- eval_id: eval1
  eval_query: "is there a openshift-monitoring namespace?"
  eval_type: sub-string
  expected_key_words:
    - 'yes'
    - openshift-monitoring

- eval_id: eval2
  eval_query: "is there a openshift-monitoring namespace?"
  eval_type: judge-llm
  expected_response: "there is a openshift-monitoring namespace."

- eval_id: eval3
  eval_query: "create a namespace called openshift-lightspeed"
  eval_setup_script: script/eval3/setup.sh
  eval_type: script
  eval_verify_script: script/eval3/verify.sh
  eval_cleanup_script: script/eval3/cleanup.sh
```

## Command Line Arguments

- `--eval_data_yaml`: Path to the YAML file containing evaluation data
- `--agent_endpoint`: Endpoint URL for the agent API (default: <http://localhost:8080>)
- `--agent_auth_token_file`: Path to .txt file containing API token (if required). Or set `AGENT_API_TOKEN` env var without using a .txt file
- `--agent_provider`: Provider for the agent API
- `--agent_model`: Model for the agent API
- `--judge_provider`: Provider for the judge LLM (optional, required only for judge-llm evaluation)
- `--judge_model`: Model for the judge LLM (optional, required only for judge-llm evaluation)
- `--result_dir`: Directory to save evaluation results (default: eval_output/)
- `--kubeconfig`: Path to kubeconfig file (if needed for scripts)

## Output

The evaluation results are saved to a CSV file containing:
- `eval_id`: Evaluation identifier
- `query`: The query sent to the agent
- `response`: The agent's response
- `eval_type`: Type of evaluation performed
- `result`: Result (pass/fail)

## Dependencies

This package depends on:
- `pandas`: Data manipulation and analysis
- `httpx`: HTTP client for API calls
- `tqdm`: Progress bars
- `pyyaml`: YAML file processing
- `litellm`: Unified interface to 100+ LLM providers

## LiteLLM Integration (Judge LLM)

For judge-llm evaluations, you can use any of the 100+ supported providers:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Azure OpenAI**: Set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- **IBM Watsonx**: Set `WATSONX_API_KEY`, `WATSONX_API_BASE`, `WATSONX_PROJECT_ID`
- **Ollama**: Set `OLLAMA_API_BASE` (for local models)
- **And many more**: See [LiteLLM documentation](https://docs.litellm.ai/docs/providers)

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/lightspeed-core/lightspeed-evaluation.git
cd lightspeed-evaluation/lsc_agent_eval

# Install development dependencies
pdm install --dev

# Run tests
pdm run pytest

# Run linting
pdm run ruff check
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/lightspeed-core/lightspeed-evaluation/issues) tracker. 
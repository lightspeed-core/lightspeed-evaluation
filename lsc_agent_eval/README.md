# Lightspeed Agent Evaluation

A framework for evaluating AI agent performance.

## Features

- **Agent Goal Evaluation**: Evaluate whether an agent successfully achieves specified goals
- **Multi-turn Evaluation**: Organize evaluations into conversation groups for multi-turn testing
- **Multi-type Evaluation**: Support for different evaluation types:
  - `judge-llm`: LLM-based evaluation using a judge model
  - `script`: Script-based evaluation using verification scripts (similar to [k8s-bench](https://github.com/GoogleCloudPlatform/kubectl-ai/tree/main/k8s-bench))
  - `sub-string`: Simple substring matching evaluation (ALL keywords must be present in response)
- **Setup/Cleanup Scripts**: Support for running setup and cleanup scripts before/after evaluation
- **Result Tracking**: Result tracking with CSV output and JSON statistics
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

## Data Configuration

The evaluation is configured using a YAML file that defines conversations. Each conversation contains one or more evaluations and includes:

- `conversation_group`: Identifier for grouping related evaluations/conversation
- `description`: Description of the conversation (Optional)
- `setup_script`: Setup script to run before the conversation (Optional)
- `cleanup_script`: Cleanup script to run after the conversation (Optional)
- `conversation`: List of evaluations in this conversation

Each evaluation within a conversation can include:
- `eval_id`: Unique identifier for the evaluation
- `eval_query`: The query/task to send to the agent
- `eval_type`: Type of evaluation (judge-llm, script, sub-string)
- `expected_response`: Expected response (for judge-llm evaluation)
- `expected_keywords`: Keywords to look for (for sub-string evaluation)
- `eval_verify_script`: Verification script (for script evaluation)
- `description`: Description of the evaluation (Optional)

Note: `eval_id` can't contain duplicate values within a conversation group. But it is okay for cross conversation group (A warning is logged anyway for awareness)

### Example Data Configuration

```yaml
# Multi-turn Conversations
- conversation_group: conv1
  description: Basic conversation flow testing cluster operations
  conversation:
    - eval_id: eval1
      eval_query: Hi!
      eval_type: judge-llm
      expected_response: Hello! I'm an AI assistant for the Installer.
      description: Initial greeting to start conversation
    - eval_id: eval2
      eval_query: Get me active clusters
      eval_type: judge-llm
      expected_response: Active clusters are x1, x2.
      description: Request for cluster information

- conversation_group: conv2
  description: Multi-turn conversation with setup/cleanup
  setup_script: sample_data/script/setup_environment.sh
  cleanup_script: sample_data/script/cleanup_environment.sh
  conversation:
    - eval_id: eval1
      eval_query: Hi! Can you help me manage pods?
      eval_type: judge-llm
      expected_response: Hello! I can help you manage pods.
      description: Initial greeting
    - eval_id: eval2
      eval_query: Create a pod named test-pod
      eval_type: script
      eval_verify_script: sample_data/script/verify_pod.sh
      description: Create pod and verify
    - eval_id: eval3
      eval_query: List all pods
      eval_type: sub-string
      expected_keywords: ['test-pod']
      description: Verify pod is listed

# Single-turn Conversations
- conversation_group: conv3
  description: Test namespace creation and detection with scripts
  setup_script: sample_data/script/conv3/setup.sh
  cleanup_script: sample_data/script/conv3/cleanup.sh
  conversation:
    - eval_id: eval1
      eval_query: is there a openshift-lightspeed namespace ?
      eval_type: sub-string
      expected_keywords:
        - 'yes'
        - 'lightspeed'
      description: Check for openshift-lightspeed namespace after setup
```

The `sample_data/` directory contains example configurations:
- `agent_goal_eval_example.yaml`: Examples with various evaluation types
- `script/`: Example setup, cleanup, and verify scripts

## Judge LLM

For judge-llm evaluations, currently LiteLLM is used.

### Judge LLM - Setup
Expectation is that, either a third-party inference provider access is there or local model inference is already created. The eval framework doesn't handle this.

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Azure OpenAI**: Set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- **IBM Watsonx**: Set `WATSONX_API_KEY`, `WATSONX_API_BASE`, `WATSONX_PROJECT_ID`
- **Ollama**: Set `OLLAMA_API_BASE` (for local models)
- **Any Other Provider**: Check [LiteLLM documentation](https://docs.litellm.ai/docs/providers)

## Usage

```bash
lsc_agent_eval \
    --eval_data_yaml agent_goal_eval.yaml \
    --agent_endpoint http://localhost:8080 \
    --agent_provider watsonx \
    --agent_model ibm/granite-3-2-8b-instruct \
    --judge_provider openai \
    --judge_model gpt-4o-mini \
    --result_dir ./eval_output
```

```python
from lsc_agent_eval import AgentGoalEval

# Create evaluation configuration, Alternatively use namespace
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
evaluator.run_evaluation()
```

### Key Arguments

- `--eval_data_yaml`: Path to the YAML file containing evaluation data
- `--agent_endpoint`: Endpoint URL for the agent API (default: <http://localhost:8080>)
- `--agent_auth_token_file`: Path to .txt file containing API token (if required). Or set `AGENT_API_TOKEN` env var without using a .txt file
- `--agent_provider`: Provider for the agent API
- `--agent_model`: Model for the agent API
- `--judge_provider`: Provider for the judge LLM (optional, required only for judge-llm evaluation)
- `--judge_model`: Model for the judge LLM (optional, required only for judge-llm evaluation)
- `--result_dir`: Directory to save evaluation results (default: eval_output/)
- `--kubeconfig`: Path to kubeconfig file (if needed for scripts)

## Evaluation Flow

### Conversation Processing Order

1. **Load Configuration**: Parse and validate YAML configuration
2. **Process Conversations**: For each conversation group:
   - Run setup script (if provided)
   - Run all evaluations sequentially:
     - For the first evaluation: Send query without conversation ID, receive new conversation ID from API
     - For subsequent evaluations: Use the conversation ID from the first evaluation to maintain context
     - Execute evaluation based on eval_type (either sub-string, judge-llm or script)
   - Run cleanup script (if provided)
3. **Save Results**: Export to CSV and JSON with statistics

### Script Execution

- **Setup Scripts**: Run once before all evaluations in a conversation
  - If setup fails, all evaluations in the conversation are marked as ERROR
- **Cleanup Scripts**: Run once after all evaluations in a conversation
  - Cleanup failures are logged as warnings (non-critical)
  - Always executed regardless of evaluation results
- **Verify Scripts**: Run per individual evaluation for script type evaluations
  - Used to verify the agent's action is successful

### Error Handling

- **Setup Failure**: Marks all evaluations in conversation as ERROR
- **Cleanup Failure**: Logged as warning, does not affect evaluation results
- **API Errors**: Evaluation marked as Error
- **Evaluation Failure**: Individual evaluation marked as ERROR or FAIL
- **Configuration Errors**: Detailed validation message

## Output

The framework generates two types of output:

### CSV Results (`agent_goal_eval_results_YYYYMMDD_HHMMSS.csv`)

Contains detailed results with columns:
- `conversation_group`: The conversation group identifier
- `conversation_id`: The conversation ID returned by the Agent API
- `eval_id`: Individual evaluation identifier
- `result`: PASS, FAIL, or ERROR
- `eval_type`: Type of evaluation performed
- `query`: The question/task sent to the agent
- `response`: The agent's response
- `error`: Error message (if any)

### JSON Statistics (`agent_goal_eval_summary_YYYYMMDD_HHMMSS.json`)

Result statistics:
- **Overall Summary**: Total evaluations, pass/fail/error counts, success rate
- **By Conversation**: Breakdown of results for each conversation group
- **By Evaluation Type**: Performance metrics for each evaluation type (judge-llm, script, sub-string)

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/lightspeed-core/lightspeed-evaluation.git
cd lightspeed-evaluation/lsc_agent_eval

# Install development dependencies
pdm install --dev

# Run tests
pdm run pytest tests --cov=src

# Run linting
pdm run ruff check
pdm run isort src tests
pdm run black src tests
pdm run mypy src
pdm run pyright src
pdm run pylint src
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and lint checks
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/lightspeed-core/lightspeed-evaluation/issues) tracker. 

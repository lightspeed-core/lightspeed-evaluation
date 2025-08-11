# Lightspeed Agent Evaluation

A framework for evaluating AI agent performance.

## Features

- **Agent Goal Evaluation**: Evaluate whether an agent successfully achieves specified goals
- **Multi-turn Evaluation**: Organize evaluations into conversation groups for multi-turn testing
- **Multi-type Evaluation**: Support for multiple evaluation types per query:
  - `action_eval`: Script-based evaluation using verification script (similar to [k8s-bench](https://github.com/GoogleCloudPlatform/kubectl-ai/tree/main/k8s-bench))
  - `response_eval:sub-string`: Simple substring matching evaluation (ALL keywords must be present in response)
  - `response_eval:accuracy`: LLM-based evaluation using a judge model. Result is either accurate or not in comparison to expected response
  - `tool_eval`: Tool call evaluation comparing expected vs actual tool calls with arguments
- **Setup/Cleanup Scripts**: Support for running setup and cleanup scripts before/after evaluation
- **Result Tracking**: Result tracking with CSV output and JSON statistics
- **Standalone Package**: Can be installed and used independently of the main lightspeed-core-evaluation package
- **LiteLLM Integration**: Unified interface for Judge LLM

## Installation

### Prerequisites

- Python 3.11 or 3.12
- Package manager: `pdm` or `pip`

- Agent API is running. Any change to the API response may impact evaluation processing logic.
- For Judge model, model inference server is up

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
- `eval_types`: List of evaluation types to run (action_eval, tool_eval, response_eval:sub-string, response_eval:accuracy)
- `expected_response`: Expected response (for response_eval:accuracy evaluation)
- `expected_keywords`: Keywords to look for (for response_eval:sub-string evaluation)
- `expected_tool_calls`: Expected tool call sequences (list of lists) with arguments (for tool_eval)
- `eval_verify_script`: Verification script (for action_eval evaluation)
- `description`: Description of the evaluation (Optional)

Note: `eval_id` can't contain duplicate values within a conversation group. But it is okay for cross conversation group (A warning is logged anyway for awareness)

### Example Data Configuration

```yaml
# Multi-turn Conversations with Multiple Evaluation Types
- conversation_group: conv1
  description: Basic conversation flow testing cluster operations
  conversation:
    - eval_id: eval1
      eval_query: Hi!
      eval_types: [response_eval:accuracy]
      expected_response: Hello! I'm an AI assistant for the Installer.
      description: Initial greeting to start conversation
    - eval_id: eval2
      eval_query: Get me active clusters
      eval_types: [response_eval:accuracy, response_eval:sub-string]
      expected_response: Active clusters are x1, x2.
      expected_keywords: [clusters, active]
      description: Request for cluster information with multiple validations

- conversation_group: conv2
  description: Multi-turn conversation with setup/cleanup and action evaluation
  setup_script: sample_data/script/setup_environment.sh
  cleanup_script: sample_data/script/cleanup_environment.sh
  conversation:
    - eval_id: eval1
      eval_query: Hi! Can you help me manage pods?
      eval_types: [response_eval:accuracy]
      expected_response: Hello! I can help you manage pods.
      description: Initial greeting
    - eval_id: eval2
      eval_query: Create a pod named test-pod
      eval_types: 
        - action_eval
        - response_eval:sub-string
      eval_verify_script: sample_data/script/verify_pod_creation.sh
      expected_keywords: 
        - pod
        - created
        - test-pod
      description: Pod creation with script verification and keyword matching

# Tool Call Evaluation
- conversation_group: conv_tools  
  description: Tool call validation
  conversation:
    - eval_id: eval1
      eval_query: List available OpenShift versions
      eval_types: [tool_eval]
      expected_tool_calls: 
        - - name: list_versions
            arguments: {}
      description: Verify correct tool call for listing versions
    - eval_id: eval2
      eval_query: is there an openshift-lightspeed namespace
      eval_types: [tool_eval, response_eval:sub-string]
      expected_tool_calls:
        - - name: oc_get
            arguments:
              oc_get_args: [namespaces, openshift-lightspeed]
      expected_keywords: ["yes", "openshift-lightspeed"]
      description: Tool call with argument validation and response verification

# Single-turn Conversations
- conversation_group: conv3
  description: Test namespace creation and detection with scripts
  setup_script: sample_data/script/conv3/setup.sh
  cleanup_script: sample_data/script/conv3/cleanup.sh
  conversation:
    - eval_id: eval1
      eval_query: is there an openshift-lightspeed namespace?
      eval_types: [response_eval:sub-string]
      expected_keywords: ["yes", lightspeed]
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
    --endpoint_type streaming \
    --agent_provider watsonx \
    --agent_model ibm/granite-3-2-8b-instruct \
    --agent_auth_token_file agent_api_token.txt \
    --judge_provider openai \
    --judge_model gpt-4o-mini \
    --result_dir ./eval_output
```
Pass token text file or set `AGENT_API_TOKEN` env var.

```python
from lsc_agent_eval import AgentGoalEval

# Create evaluation configuration, Alternatively use namespace
class EvalArgs:
    def __init__(self):
        self.eval_data_yaml = 'data/example_eval.yaml'
        self.agent_endpoint = 'http://localhost:8080'
        self.endpoint_type = 'query'  # Non-streaming
        self.agent_provider = 'watsonx'
        self.agent_model = 'ibm/granite-3-2-8b-instruct'
        self.judge_provider = 'openai'
        self.judge_model = 'gpt-4o-mini'
        self.agent_auth_token_file = None  # set `AGENT_API_TOKEN` env var
        self.result_dir = None

# Run evaluation
args = EvalArgs()
evaluator = AgentGoalEval(args)
evaluator.run_evaluation()
```

### Key Arguments

- `--eval_data_yaml`: Path to the YAML file containing evaluation data
- `--agent_endpoint`: Endpoint URL for the agent API (default: <http://localhost:8080>)
- `--endpoint_type`: Endpoint type to use for agent queries (default: streaming). Options: 'streaming' or 'query'
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
     - Execute evaluation based on eval_type (any combination of valid eval_types)
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
- **By Evaluation Type**: Performance metrics for each evaluation type (action_eval, response_eval:sub-string, response_eval:accuracy, tool_eval)

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

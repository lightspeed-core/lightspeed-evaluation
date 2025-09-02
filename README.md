# LSC Evaluation Framework

A comprehensive framework/tool to evaluate GenAI application.

## 🎯 Key Features

- **Multi-Framework Support**: Seamlessly use metrics from Ragas, DeepEval, and custom metrics
- **Turn & Conversation-Level Evaluation**: Support for both individual queries and multi-turn conversations  
- **LLM Provider Flexibility**: OpenAI, Anthropic, Watsonx, Azure, Gemini, Ollama via LiteLLM
- **Flexible Configuration**: Configurable environment & metric metadata
- **Rich Output**: CSV, JSON, TXT reports + visualization graphs (pass rates, distributions, heatmaps)
- **Early Validation**: Catch configuration errors before expensive LLM calls
- **Statistical Analysis**: Statistics for every metric with score distribution analysis

## 🚀 Quick Start

### Installation
```bash
# From Git
uv add git+https://github.com/your-org/lightspeed-evaluation.git#subdirectory=lsc_eval
# or pip install git+https://github.com/lightspeed-core/lightspeed-evaluation.git#subdirectory=lsc_eval

# Local Development  
cd lsc_eval && uv sync
```

### Basic Usage
```bash
# Set API key
export OPENAI_API_KEY="your-key"

# Navigate to the lsc_eval directory
cd lsc_eval

# Run evaluation (Create your own data)
uv run python runner.py --system-config config/system.yaml --eval-data config/evaluation_data.yaml
```

### Troubleshooting
If you encounter `ModuleNotFoundError: No module named 'lsc_eval'`, ensure you:
1. Are running from the `lsc_eval/` directory
2. Use `uv run` instead of `python` directly
3. Have run `uv sync` to install dependencies

## 📊 Supported Metrics

### Turn-Level (Single Query)
- **Ragas**
  - Response Evaluation
    - `faithfulness`
    - `response_relevancy`
  - Context Evaluation
    - `context_recall`
    - `context_relevance`
    - `context_precision_without_reference`
    - `context_precision_with_reference`
- **Custom**
  - Response Evaluation
    - `answer_correctness`

### Conversation-Level (Multi-turn)
- **DeepEval**
  - `conversation_completeness`
  - `conversation_relevancy`
  - `knowledge_retention`

## ⚙️ Configuration

### System Config (`system.yaml`)
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.0
  timeout: 120

metrics_metadata:
  turn_level:
    "ragas:faithfulness":
      threshold: 0.8
      type: "turn"
      framework: "ragas"
  
  conversation_level:
    "deepeval:conversation_completeness":
      threshold: 0.8
      type: "conversation"
      framework: "deepeval"
```

### Evaluation Data (`evaluation_data.yaml`)
```yaml
- conversation_group_id: "test_conversation"
  description: "Sample evaluation"
  
  # Turn-level metrics (empty list = skip turn evaluation)
  turn_metrics:
    - "ragas:faithfulness"
    - "custom:answer_correctness"
  
  # Turn-level metrics metadata (threshold + other properties)
  turn_metrics_metadata:
    "ragas:response_relevancy": 
      threshold: 0.8
      weight: 1.0
    "custom:answer_correctness": 
      threshold: 0.75
  
  # Conversation-level metrics (empty list = skip conversation evaluation)   
  conversation_metrics:
    - "deepeval:conversation_completeness"
  
  turns:
    - turn_id: 1
      query: "What is OpenShift?"
      response: "Red Hat OpenShift powers the entire application lifecycle...."
      contexts:
        - content: "Red Hat OpenShift powers...."
      expected_response: "Red Hat OpenShift...."
```

## RAG Evaluation

### Overview
The evaluation framework includes integration with Lightspeed Stack RAG for evaluating RAG-generated responses:

- Use a running Lightspeed Stack RAG instance for generating responses
- Evaluate RAG-generated responses against existing responses
- Run comprehensive evaluations that include RAG metrics alongside traditional metrics

### Setup

#### Prerequisites
1. **Lightspeed Stack RAG Service**: Must be running at `http://localhost:8080`
2. **Environment Variables**: Set `OPENAI_API_KEY` or other LLM provider credentials
3. **Dependencies**: Ensure `requests` and other required packages are installed

#### Configuration
Add RAG configuration to your `system.yaml`:

```yaml
# Lightspeed RAG Configuration
lightspeed_rag:
  base_url: "http://localhost:8080"    # Lightspeed Stack RAG API endpoint
  provider: "openai"                   # LLM provider for RAG queries
  model: "gpt-4o-mini"                 # Model to use for RAG queries
  system_prompt: "You are a helpful assistant with access to a knowledge base..."
  no_tools: false                      # Whether to bypass tools and MCP servers
  timeout: 300                         # Request timeout for RAG queries

# Add RAG metric to metrics metadata
metrics_metadata:
  turn_level:
    "custom:rag_response_evaluation":
      threshold: 0.7
      type: "turn"
      description: "Evaluation of RAG-generated response quality compared to original"
      framework: "custom"
```

### Usage

#### Evaluation Data
Add RAG metrics to your `evaluation_data.yaml`:

```yaml
- conversation_group_id: "rag_test"
  description: "Test RAG response evaluation"
  
  turn_metrics:
    - "custom:rag_response_evaluation"
  
  turns:
    - turn_id: 1
      query: "What is Kubernetes?"
      response: "Kubernetes is an open-source container orchestration platform..."
      contexts:
        - content: "Kubernetes documentation context here..."
      expected_response: "Expected response for comparison"
```

#### Direct RAG Client Usage
```python
from rag_context import LightspeedRAGClient

# Initialize client
client = LightspeedRAGClient("http://localhost:8080")

# Simple query
result = client.query("What is Kubernetes?")
print(f"Response: {result['response']}")
print(f"Conversation ID: {result['conversation_id']}")
```

#### Programmatic Usage
```python
from lsc_eval.llm_managers.lightspeed_rag_llm import (
    LightspeedRAGManager, 
    LightspeedRAGConfig
)

# Create RAG manager
config = LightspeedRAGConfig(
    base_url="http://localhost:8080",
    provider="openai",
    model="gpt-4o-mini"
)
rag_manager = LightspeedRAGManager(config)

# Generate response
response, conv_id = rag_manager.generate_response(
    query="What is container orchestration?",
    contexts=["Kubernetes manages containerized applications..."]
)
```

### Troubleshooting

**Connection Issues**:
- Ensure Lightspeed Stack RAG service is running
- Check `base_url` in configuration
- Test connectivity: `curl http://localhost:8080/v1/info`

**Authentication Errors**:
- Set appropriate API keys (`OPENAI_API_KEY`, etc.)
- Check LLM provider configuration

## 📈 Output & Visualization

### Generated Reports
- **CSV**: Detailed results with status, scores, reasons
- **JSON**: Summary statistics with score distributions
- **TXT**: Human-readable summary
- **PNG**: 4 visualization types (pass rates, score distributions, heatmaps, status breakdown)

### Key Metrics in Output
- **PASS/FAIL/ERROR**: Status based on thresholds
- **Actual Reasons**: DeepEval provides LLM-generated explanations, Custom metrics provide detailed reasoning
- **Score Statistics**: Mean, median, standard deviation, min/max for every metric

## 🧪 Development

### Development Tools
```bash
uv sync --group dev
uv run black .
uv run ruff check .
uv run mypy .
uv run pyright .
uv run pylint .
uv run pytest tests --cov=src
```

## Agent Evaluation
For a detailed walkthrough of the new agent-evaluation framework, refer
[lsc_agent_eval/README.md](lsc_agent_eval/README.md)

## Generate answers (optional - for creating test data)
For generating answers (optional) refer [README-generate-answers](README-generate-answers.md)

## 📄 License & Contributing

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

Contributions welcome - see development setup above for code quality tools.
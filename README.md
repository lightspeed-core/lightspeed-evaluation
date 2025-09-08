# LightSpeed Evaluation Framework

A comprehensive framework for evaluating GenAI applications.

## 🎯 Key Features

- **Multi-Framework Support**: Seamlessly use metrics from Ragas, DeepEval, and custom implementations
- **Turn & Conversation-Level Evaluation**: Support for both individual queries and multi-turn conversations  
- **LLM Provider Flexibility**: OpenAI, Anthropic, Watsonx, Azure, Gemini, Ollama via LiteLLM
- **Flexible Configuration**: Configurable environment & metric metadata
- **Rich Output**: CSV, JSON, TXT reports + visualization graphs (pass rates, distributions, heatmaps)
- **Early Validation**: Catch configuration errors before expensive LLM calls
- **Statistical Analysis**: Statistics for every metric with score distribution analysis
- **Agent Evaluation**: Framework for evaluating AI agent performance (future integration planned)

## 🚀 Quick Start

### Installation

```bash
# From Git
pip install git+https://github.com/lightspeed-core/lightspeed-evaluation.git

# Local Development
pip install uv
uv sync
```

### Basic Usage

```bash
# Set API key
export OPENAI_API_KEY="your-key"

# Run evaluation
lightspeed-eval --system-config config/system.yaml --eval-data config/evaluation_data.yaml
```

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

### System Config (`config/system.yaml`)
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

### Evaluation Data (`config/evaluation_data.yaml`)
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
make format
make pylint
make pyright
make docstyle
make check-types

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
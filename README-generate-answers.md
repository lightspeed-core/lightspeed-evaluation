# Generate answers by LLM

## Installation
Refer to [README](README.md) for the installation

## Configuration
The default configuration file is [./src/generate_answers/eval_config.yaml](/src/generate_answers/eval_config.yaml)

- `lightspeed_url` -- url of the running lightspeed-core service
- `models` -- list of available models. `provider` and `model` have to match
lightspeed-core service configuration. `display_name` is a nice short model name.
- `models_to_evaluate` -- list of model names (`display_name`) for answers generation.

Example:
```yaml
lightspeed_url: "http://localhost:8080/v1/"
models:
  - display_name: "granite-3-3-8b-instruct"
    provider: "watsonx"
    model: "ibm/granite-3-3-8b-instruct"

  - display_name: "openai-o4-mini"
    provider: "openai"
    model: "o4-mini"

  - display_name: "llama3-8b"
    provider: "ollama"
    model: "llama3:8b"

models_to_evaluate:
  #- "granite-3-3-8b-instruct"
  - "openai-o4-mini"
  - "llama3-8b"
```

You use the `models_to_evaluate` list to select which of the available models will be used for answer generation.
All models included in this list must also be defined in the `models` section **and** properly configured and available in the running lightspeed-core service.

## Input Data

The tool supports multiple input formats for evaluation data:

* **CSV** – must contain two columns: `id` and `question`.
  Example file: [`eval_data/questions.csv`](eval_data/questions.csv)

  ```csv
  id,question
  1,How do I enable VM high availability in my cluster?
  2,How do I migrate a VM to a different project?
  3,How do I manage RBAC in OpenShift Virtualization
  ...
  ```
* **Parquet** – Lightspeed evaluation parquet format is supported.
* **JSON** – Lightspeed evaluation JSON format is supported.


## Running
`uv run generate_answers -h`

```
Usage: generate_answers [OPTIONS]

  Generate answers from LLMs by connection to LightSpeed core service.

Options:
  -c, --config-filename PATH  Configuration file  [default:
                              ./src/generate_answers/eval_config.yaml]
  -i, --input-filename PATH   Input filename with questions  [default:
                              ./eval_data/questions.csv]
  -o, --output-filename PATH  Output JSON filename with results -- generated
                              answers  [default:
                              ./eval_output/generated_qna.json]
  -l, --llm-cache-dir PATH    Directory with cached responses from LLMs. Cache
                              key is model+provider+question  [default:
                              .caches/llm_cache]
  -f, --force-overwrite       Overwrite the output file if it exists
  -v, --verbose               Increase the logging level to DEBUG
  -h, --help                  Show this message and exit.
  -p, --max-concurrent INTEGER  Maximum number of questions to process in
                                parallel simultaneously  [default: 1]
```

## Results
The results are stored in dataframe in JSON format. The file can be read by `pandas.read_json`.
The columns are:
- `id`, `question` -- from the input file
- `<model_name>_answers` -- for each configured model

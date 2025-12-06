"""Common constants for evaluation framework."""

DEFAULT_API_BASE = "http://localhost:8080"
DEFAULT_API_VERSION = "v1"
DEFAULT_API_TIMEOUT = 300
DEFAULT_ENDPOINT_TYPE = "streaming"
SUPPORTED_ENDPOINT_TYPES = ["streaming", "query"]
DEFAULT_API_CACHE_DIR = ".caches/api_cache"

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_RETRIES = 3
DEFAULT_LLM_CACHE_DIR = ".caches/llm_cache"

DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_CACHE_DIR = ".caches/embedding_cache"

DEFAULT_OUTPUT_DIR = "./eval_output"
DEFAULT_BASE_FILENAME = "evaluation"

SUPPORTED_OUTPUT_TYPES = ["csv", "json", "txt"]
SUPPORTED_CSV_COLUMNS = [
    "conversation_group_id",
    "turn_id",
    "metric_identifier",
    "result",
    "score",
    "threshold",
    "reason",
    "query",
    "response",
    "execution_time",
    "api_input_tokens",
    "api_output_tokens",
    "judge_llm_input_tokens",
    "judge_llm_output_tokens",
    "tool_calls",
    "contexts",
    "expected_response",
    "expected_intent",
    "expected_keywords",
    "expected_tool_calls",
]
SUPPORTED_GRAPH_TYPES = [
    "pass_rates",
    "score_distribution",
    "conversation_heatmap",
    "status_breakdown",
]

DEFAULT_VISUALIZATION_FIGSIZE = [12, 8]
DEFAULT_VISUALIZATION_DPI = 300

DEFAULT_LOG_SOURCE_LEVEL = "INFO"
DEFAULT_LOG_PACKAGE_LEVEL = "WARNING"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_SHOW_TIMESTAMPS = True

SUPPORTED_RESULT_STATUSES = ["PASS", "FAIL", "ERROR"]

"""Common constants for evaluation framework."""

# Map similarity measure strings to Ragas DistanceMeasure enum
from ragas.metrics import DistanceMeasure

# NLP Metrics Constants - BLEU
DEFAULT_BLEU_MAX_NGRAM = 4  # Standard BLEU uses up to 4-grams
MIN_BLEU_NGRAM = 1
MAX_BLEU_NGRAM = 4

# NLP Metrics Constants - ROUGE Types
ROUGE_TYPE_ROUGE1 = "rouge1"
ROUGE_TYPE_ROUGE2 = "rouge2"
ROUGE_TYPE_ROUGEL = "rougeL"
ROUGE_TYPE_ROUGELSUM = "rougeLsum"
SUPPORTED_ROUGE_TYPES = [
    ROUGE_TYPE_ROUGE1,
    ROUGE_TYPE_ROUGE2,
    ROUGE_TYPE_ROUGEL,
    ROUGE_TYPE_ROUGELSUM,
]

# NLP Metrics Constants - ROUGE Modes
ROUGE_MODE_PRECISION = "precision"
ROUGE_MODE_RECALL = "recall"
ROUGE_MODE_FMEASURE = "fmeasure"
SUPPORTED_ROUGE_MODES = [
    ROUGE_MODE_PRECISION,
    ROUGE_MODE_RECALL,
    ROUGE_MODE_FMEASURE,
]

# NLP Metrics Constants - Similarity Measures
SIMILARITY_LEVENSHTEIN = "levenshtein"
SIMILARITY_HAMMING = "hamming"
SIMILARITY_JARO = "jaro"
SIMILARITY_JARO_WINKLER = "jaro_winkler"
SUPPORTED_SIMILARITY_MEASURES = [
    SIMILARITY_LEVENSHTEIN,
    SIMILARITY_HAMMING,
    SIMILARITY_JARO,
    SIMILARITY_JARO_WINKLER,
]

DISTANCE_MEASURE_MAP = {
    SIMILARITY_LEVENSHTEIN: DistanceMeasure.LEVENSHTEIN,
    SIMILARITY_HAMMING: DistanceMeasure.HAMMING,
    SIMILARITY_JARO: DistanceMeasure.JARO,
    SIMILARITY_JARO_WINKLER: DistanceMeasure.JARO_WINKLER,
}


# API Constants
DEFAULT_API_BASE = "http://localhost:8080"
DEFAULT_API_VERSION = "v1"
DEFAULT_API_TIMEOUT = 300
DEFAULT_ENDPOINT_TYPE = "streaming"
SUPPORTED_ENDPOINT_TYPES = ["streaming", "query"]
DEFAULT_API_CACHE_DIR = ".caches/api_cache"

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_SSL_VERIFY = True
DEFAULT_SSL_CERT_FILE = None
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_RETRIES = 3
DEFAULT_LLM_CACHE_DIR = ".caches/llm_cache"

DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_CACHE_DIR = ".caches/embedding_cache"

DEFAULT_OUTPUT_DIR = "./eval_output"
DEFAULT_BASE_FILENAME = "evaluation"

DEFAULT_STORED_CONFIGS = ["llm", "embedding", "api"]

SUPPORTED_OUTPUT_TYPES = ["csv", "json", "txt"]
SUPPORTED_CSV_COLUMNS = [
    "conversation_group_id",
    "tag",
    "turn_id",
    "metric_identifier",
    "result",
    "score",
    "threshold",
    "metric_metadata",
    "reason",
    "query",
    "response",
    "execution_time",
    "api_input_tokens",
    "api_output_tokens",
    "judge_llm_input_tokens",
    "judge_llm_output_tokens",
    # Streaming performance metrics
    "time_to_first_token",
    "streaming_duration",
    "tokens_per_second",
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

SUPPORTED_RESULT_STATUSES = ["PASS", "FAIL", "ERROR", "SKIPPED"]

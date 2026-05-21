"""Optional integrations (Langfuse, etc.) — all dependencies are opt-in per extra."""

__all__: list[str] = []

try:
    import lightspeed_evaluation.integrations.langfuse_reporter as _langfuse_reporter
except ImportError:  # pragma: no cover - optional [langfuse] / reporter deps
    pass
else:
    build_langfuse_on_complete_callback = (
        _langfuse_reporter.build_langfuse_on_complete_callback
    )
    build_langfuse_on_complete_from_storage_configs = (
        _langfuse_reporter.build_langfuse_on_complete_from_storage_configs
    )
    push_evaluation_results_to_langfuse = (
        _langfuse_reporter.push_evaluation_results_to_langfuse
    )
    __all__.extend(
        [
            "build_langfuse_on_complete_callback",
            "build_langfuse_on_complete_from_storage_configs",
            "push_evaluation_results_to_langfuse",
        ]
    )

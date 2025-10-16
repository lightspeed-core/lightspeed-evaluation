"""Utility functions for sanitizing user input."""

import re

from lightspeed_evaluation.core.constants import MAX_RUN_NAME_LENGTH


def sanitize_run_name(run_name: str) -> str:
    """Sanitize run name for safe filesystem usage.

    Replaces filesystem-unsafe characters with underscores, collapses
    multiple spaces/underscores, and enforces max length.

    Args:
        run_name: Raw run name string to sanitize

    Returns:
        Sanitized run name safe for filesystem usage. Returns empty string
        if input is empty or becomes empty after sanitization.

    Examples:
        >>> sanitize_run_name("test/run:123")
        'test_run_123'
        >>> sanitize_run_name("  multiple   spaces  ")
        'multiple_spaces'
        >>> sanitize_run_name("rh124: filesystem basics")
        'rh124_filesystem_basics'
    """
    if not run_name:
        return ""

    # Strip leading/trailing whitespace
    sanitized = run_name.strip()

    # Replace invalid filesystem characters with underscores
    # Invalid chars: / \ : * ? " ' ` < > | and control characters (0x00-0x1f)
    sanitized = re.sub(r'[/\\:*?"\'`<>|\x00-\x1f]', "_", sanitized)

    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r"[\s_]+", "_", sanitized)

    # Strip leading/trailing underscores that may have been created
    sanitized = sanitized.strip("_")

    # Enforce max length, strip trailing underscores if truncated
    if len(sanitized) > MAX_RUN_NAME_LENGTH:
        sanitized = sanitized[:MAX_RUN_NAME_LENGTH].rstrip("_")

    return sanitized

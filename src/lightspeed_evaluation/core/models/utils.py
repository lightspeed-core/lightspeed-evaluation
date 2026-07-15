"""Shared validation utilities for model fields."""

from typing import Any


def normalize_string_list(v: Any, field_name: str = "value") -> Any:
    """Normalize str to list, validate non-empty, strip whitespace, deduplicate."""
    if isinstance(v, str):
        v = [v]
    if isinstance(v, list):
        seen: set[str] = set()
        result: list[str] = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{field_name} must be non-empty strings, got: {item!r}"
                )
            clean = item.strip()
            if clean not in seen:
                result.append(clean)
                seen.add(clean)
        return result
    if v is not None:
        raise ValueError(
            f"{field_name} must be a string or list of strings, got: {type(v).__name__}"
        )
    return v

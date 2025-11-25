"""Utility for implementing lazy imports in __init__.py files.

This module provides a reusable lazy import mechanism to reduce code duplication
across multiple __init__.py files while maintaining performance benefits of lazy loading.
"""

from typing import Any


def create_lazy_getattr(
    lazy_imports: dict[str, tuple[str, str]], module_name: str
) -> Any:
    """Create a __getattr__ function for lazy importing.

    This function generates a lazy import mechanism that can be used in __init__.py
    files to defer imports until they are actually needed.

    Args:
        lazy_imports: Dictionary mapping attribute names to (module_path, class_name) tuples.
        module_name: The name of the module using this lazy import (for error messages).

    Returns:
        A function that can be used as __getattr__ in the module.

    Example:
        >>> _LAZY_IMPORTS = {
        ...     "MyClass": ("mypackage.mymodule", "MyClass"),
        ... }
        >>> __getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
    """

    def __getattr__(name: str) -> Any:
        """Lazy import mechanism."""
        if name in lazy_imports:
            module_path, class_name = lazy_imports[name]
            try:
                module = __import__(module_path, fromlist=[class_name])
                return getattr(module, class_name)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import {name} from {module_path}: {e}"
                ) from e

        raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    return __getattr__

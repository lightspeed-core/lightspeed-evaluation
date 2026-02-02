"""Unit tests for lazy import utility."""

import pytest

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr


class TestCreateLazyGetattr:
    """Tests for create_lazy_getattr function."""

    def test_lazy_import_success(self) -> None:
        """Test successful lazy import."""
        lazy_imports = {
            "EvaluationResult": (
                "lightspeed_evaluation.core.models",
                "EvaluationResult",
            ),
        }

        __getattr__ = create_lazy_getattr(lazy_imports, "test_module")

        # Should successfully import
        result_class = __getattr__("EvaluationResult")
        assert result_class is not None
        assert result_class.__name__ == "EvaluationResult"

    def test_lazy_import_unknown_attribute(self) -> None:
        """Test lazy import with unknown attribute."""
        lazy_imports = {
            "KnownClass": ("lightspeed_evaluation.core.models", "EvaluationResult"),
        }

        __getattr__ = create_lazy_getattr(lazy_imports, "test_module")

        with pytest.raises(AttributeError, match="has no attribute 'UnknownClass'"):
            __getattr__("UnknownClass")

    def test_lazy_import_failed_import(self) -> None:
        """Test lazy import with invalid module path."""
        lazy_imports = {
            "FakeClass": ("nonexistent.module", "FakeClass"),
        }

        __getattr__ = create_lazy_getattr(lazy_imports, "test_module")

        with pytest.raises(ImportError, match="Failed to import"):
            __getattr__("FakeClass")

    def test_lazy_import_multiple_classes(self) -> None:
        """Test lazy importing multiple classes."""
        lazy_imports = {
            "EvaluationResult": (
                "lightspeed_evaluation.core.models",
                "EvaluationResult",
            ),
            "SystemConfig": ("lightspeed_evaluation.core.models", "SystemConfig"),
        }

        __getattr__ = create_lazy_getattr(lazy_imports, "test_module")

        result_class = __getattr__("EvaluationResult")
        config_class = __getattr__("SystemConfig")

        assert result_class.__name__ == "EvaluationResult"
        assert config_class.__name__ == "SystemConfig"

    def test_lazy_import_preserves_module_name_in_error(self) -> None:
        """Test that module name appears in error messages."""
        lazy_imports: dict[str, tuple[str, str]] = {}

        __getattr__ = create_lazy_getattr(lazy_imports, "my_custom_module")

        with pytest.raises(AttributeError) as exc_info:
            __getattr__("MissingClass")

        assert "my_custom_module" in str(exc_info.value)

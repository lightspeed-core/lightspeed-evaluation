"""Tests for sanitization utilities."""

import pytest

from lightspeed_evaluation.core.constants import MAX_RUN_NAME_LENGTH
from lightspeed_evaluation.core.utils import sanitize_run_name


class TestSanitizeRunName:
    """Test cases for sanitize_run_name function."""

    def test_basic_alphanumeric(self):
        """Test that basic alphanumeric strings pass through unchanged."""
        assert sanitize_run_name("test123") == "test123"
        assert sanitize_run_name("rh124_filesystem_basics") == "rh124_filesystem_basics"

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        assert sanitize_run_name("") == ""

    def test_whitespace_trimming(self):
        """Test that leading/trailing whitespace is removed."""
        assert sanitize_run_name("  test  ") == "test"
        assert sanitize_run_name("\ttest\n") == "test"

    def test_filesystem_unsafe_characters(self):
        """Test that filesystem-unsafe characters are replaced with underscores."""
        assert sanitize_run_name("test/run") == "test_run"
        assert sanitize_run_name("test\\run") == "test_run"
        assert sanitize_run_name("test:run") == "test_run"
        assert sanitize_run_name("test*run") == "test_run"
        assert sanitize_run_name("test?run") == "test_run"
        assert sanitize_run_name('test"run') == "test_run"
        assert sanitize_run_name("test'run") == "test_run"
        assert sanitize_run_name("test`run") == "test_run"
        assert sanitize_run_name("test<run") == "test_run"
        assert sanitize_run_name("test>run") == "test_run"
        assert sanitize_run_name("test|run") == "test_run"

    def test_multiple_special_characters(self):
        """Test strings with multiple special characters."""
        assert sanitize_run_name("test/run:123") == "test_run_123"
        assert sanitize_run_name("rh124: filesystem basics") == "rh124_filesystem_basics"
        assert sanitize_run_name("test's `command`") == "test_s_command"

    def test_space_collapsing(self):
        """Test that multiple spaces are collapsed to single underscore."""
        assert sanitize_run_name("multiple   spaces") == "multiple_spaces"
        assert sanitize_run_name("test    run") == "test_run"

    def test_underscore_collapsing(self):
        """Test that multiple underscores are collapsed to single underscore."""
        assert sanitize_run_name("test___run") == "test_run"
        assert sanitize_run_name("test_____run") == "test_run"

    def test_mixed_whitespace_underscore_collapsing(self):
        """Test that mixed spaces and underscores collapse properly."""
        assert sanitize_run_name("test _ _ run") == "test_run"
        assert sanitize_run_name("test  _  run") == "test_run"

    def test_leading_trailing_underscores_stripped(self):
        """Test that leading/trailing underscores created during sanitization are removed."""
        assert sanitize_run_name("/test/") == "test"
        assert sanitize_run_name(":test:") == "test"
        assert sanitize_run_name("_test_") == "test"

    def test_max_length_enforcement(self):
        """Test that strings exceeding max length are truncated."""
        long_string = "a" * (MAX_RUN_NAME_LENGTH + 50)
        result = sanitize_run_name(long_string)
        assert len(result) <= MAX_RUN_NAME_LENGTH
        assert result == "a" * MAX_RUN_NAME_LENGTH

    def test_max_length_with_trailing_underscores(self):
        """Test that truncation removes trailing underscores."""
        # Create a string that when truncated would end with underscore
        long_string = "a" * (MAX_RUN_NAME_LENGTH - 1) + "_" + "b" * 50
        result = sanitize_run_name(long_string)
        assert len(result) <= MAX_RUN_NAME_LENGTH
        assert not result.endswith("_")

    def test_control_characters(self):
        """Test that control characters are replaced."""
        assert sanitize_run_name("test\x00run") == "test_run"
        assert sanitize_run_name("test\x1frun") == "test_run"

    def test_unicode_characters_preserved(self):
        """Test that Unicode characters (emojis, kanji, etc.) are preserved."""
        # Emojis
        assert sanitize_run_name("test🚀run") == "test🚀run"
        assert sanitize_run_name("📊evaluation") == "📊evaluation"

        # Japanese kanji
        assert sanitize_run_name("テスト実行") == "テスト実行"
        assert sanitize_run_name("test_日本語_run") == "test_日本語_run"

        # Chinese characters
        assert sanitize_run_name("测试运行") == "测试运行"

        # Mix of Unicode and ASCII
        assert sanitize_run_name("test_🎯_goal") == "test_🎯_goal"

    def test_unicode_with_unsafe_characters(self):
        """Test Unicode strings with filesystem-unsafe characters."""
        assert sanitize_run_name("テスト/実行") == "テスト_実行"
        assert sanitize_run_name("test🚀:run") == "test🚀_run"
        assert sanitize_run_name("評価 💯 test") == "評価_💯_test"

    def test_real_world_yaml_filenames(self):
        """Test realistic YAML filename scenarios."""
        assert sanitize_run_name("rh124_lesson_01") == "rh124_lesson_01"
        assert sanitize_run_name("filesystem-basics") == "filesystem-basics"
        assert sanitize_run_name("Module 1: Introduction") == "Module_1_Introduction"
        assert sanitize_run_name("test (copy)") == "test_(copy)"  # Parentheses are valid

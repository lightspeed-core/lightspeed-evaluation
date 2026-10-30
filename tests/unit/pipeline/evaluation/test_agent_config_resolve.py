"""Tests for _resolve_eval_data_agent_config helper."""

from lightspeed_evaluation.pipeline.evaluation.pipeline import (
    _resolve_eval_data_agent_config,
)


class TestResolveEvalDataAgentConfig:
    """Tests for flat vs keyed agent_config resolution."""

    def test_none_config_returns_none(self) -> None:
        """No config returns None."""
        assert _resolve_eval_data_agent_config(None, "model_a") is None

    def test_empty_config_returns_none(self) -> None:
        """Empty dict returns None."""
        assert _resolve_eval_data_agent_config({}, "model_a") is None

    def test_flat_config_returned_as_is(self) -> None:
        """Flat config applies to all agents."""
        config = {"timeout": 1200, "num_retries": 5}
        result = _resolve_eval_data_agent_config(config, "model_a")
        assert result == {"timeout": 1200, "num_retries": 5}

    def test_flat_config_with_no_agent_name(self) -> None:
        """Flat config with no agent name still returned."""
        config = {"timeout": 1200}
        result = _resolve_eval_data_agent_config(config, None)
        assert result == {"timeout": 1200}

    def test_keyed_config_matching_agent(self) -> None:
        """Keyed config returns matching agent's config."""
        config = {
            "model_a": {"timeout": 300},
            "model_b": {"timeout": 600},
        }
        result = _resolve_eval_data_agent_config(config, "model_a")
        assert result == {"timeout": 300}

    def test_keyed_config_no_match_returns_none(self) -> None:
        """Keyed config with unmatched agent returns None."""
        config = {
            "model_a": {"timeout": 300},
            "model_b": {"timeout": 600},
        }
        result = _resolve_eval_data_agent_config(config, "model_c")
        assert result is None

    def test_keyed_config_no_agent_name_returns_none(self) -> None:
        """Keyed config with no agent name returns None."""
        config = {
            "model_a": {"timeout": 300},
        }
        result = _resolve_eval_data_agent_config(config, None)
        assert result is None

    def test_flat_config_with_mapping_field_returned_as_is(self) -> None:
        """Flat config with a dict-valued field is flat, not keyed."""
        config = {
            "trace_source": "lf",
            "extra_request_params": {"no_tools": True},
        }
        result = _resolve_eval_data_agent_config(config, "model_a")
        assert result == config

    def test_all_dict_config_is_ambiguous(self) -> None:
        """A config of only dict values cannot be distinguished from keyed.

        Known limitation: shape-based detection cannot tell a flat config
        whose every field is a mapping from a per-agent keyed config. Such
        configs are treated as keyed; combine with at least one scalar
        field to express a flat override.
        """
        config = {"extra_request_params": {"no_tools": True}}
        result = _resolve_eval_data_agent_config(config, "model_a")
        assert result is None

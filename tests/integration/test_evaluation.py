"""Comprehensive tests for LightSpeed Evaluation Framework."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lightspeed_evaluation import (
    ConfigLoader,
    DataValidator,
    EvaluationDriver,
    OutputHandler,
)
from lightspeed_evaluation.core.config import EvaluationData, EvaluationResult, TurnData
from lightspeed_evaluation.runner.evaluation import run_evaluation



class TestDataValidation:
    """Test data validation functionality."""

    def test_valid_evaluation_data(self):
        """Test validation of valid evaluation data."""
        # First populate the metrics by loading a system config
        from lightspeed_evaluation.core.config.loader import populate_metric_mappings
        
        metrics_metadata = {
            "turn_level": {
                "ragas:faithfulness": {
                    "threshold": 0.8,
                    "type": "turn",
                    "description": "How faithful the response is to the provided context",
                    "framework": "ragas",
                }
            },
            "conversation_level": {}
        }
        populate_metric_mappings(metrics_metadata)
        
        valid_data = [
            EvaluationData(
                conversation_group_id="test_conv",
                turn_metrics=["ragas:faithfulness"],
                conversation_metrics=[],
                turns=[
                    TurnData(
                        turn_id=1,
                        query="Test query",
                        response="Test response",
                        contexts=[{"content": "Test context"}],
                        expected_response="Expected response",
                    )
                ],
            )
        ]

        validator = DataValidator()
        result = validator.validate_evaluation_data(valid_data)
        assert result is True

    def test_invalid_evaluation_data_empty_turns(self):
        """Test validation fails for empty turns."""
        with pytest.raises(
            ValueError, match="Conversation must have at least one turn"
        ):
            EvaluationData(
                conversation_group_id="test_conv",
                turn_metrics=["ragas:faithfulness"],
                conversation_metrics=[],
                turns=[],  # Empty turns should fail
            )

    def test_invalid_evaluation_data_empty_query(self):
        """Test validation fails for empty query."""
        with pytest.raises(ValueError, match="Query and response cannot be empty"):
            TurnData(
                turn_id=1, query="", response="Test response"  # Empty query should fail
            )


class TestEvaluationDriver:
    """Test EvaluationDriver functionality."""

    @pytest.fixture
    def mock_config_loader(self):
        """Create a mock config loader."""
        loader = MagicMock(spec=ConfigLoader)
        loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3,
            }
        }
        # Add system_config attribute
        loader.system_config = MagicMock()
        loader.system_config.default_turn_metrics_metadata = {}
        loader.system_config.default_conversation_metrics_metadata = {}
        return loader

    @pytest.fixture
    def sample_evaluation_data(self):
        """Create sample evaluation data."""
        return [
            EvaluationData(
                conversation_group_id="test_conv",
                turn_metrics=["ragas:faithfulness"],
                conversation_metrics=[],
                turns=[
                    TurnData(
                        turn_id=1,
                        query="What is Python?",
                        response="Python is a programming language.",
                        contexts=[
                            {"content": "Python is a high-level programming language."}
                        ],
                        expected_response="Python is a programming language used for development.",
                    )
                ],
            )
        ]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_evaluation_driver_initialization(self, mock_config_loader):
        """Test EvaluationDriver initialization."""
        driver = EvaluationDriver(mock_config_loader)
        assert driver.config_loader == mock_config_loader
        assert driver.data_validator is not None
        assert driver.metrics_manager is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("lightspeed_evaluation.core.metrics.ragas.RagasMetrics.evaluate")
    def test_evaluation_driver_run_evaluation(
        self, mock_ragas_evaluate, mock_config_loader, sample_evaluation_data
    ):
        """Test running evaluation with mocked metrics."""
        # Mock the ragas evaluation to return a score
        mock_ragas_evaluate.return_value = (0.85, "Mocked faithfulness evaluation")

        driver = EvaluationDriver(mock_config_loader)
        results = driver.run_evaluation(sample_evaluation_data)

        assert len(results) == 1
        assert results[0].conversation_group_id == "test_conv"
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[0].score == 0.85


class TestOutputGeneration:
    """Test output and report generation."""

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        return [
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=0.85,
                threshold=0.8,
                reason="Good faithfulness score",
                query="Test query",
                response="Test response",
                execution_time=1.5,
            ),
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id=1,
                metric_identifier="ragas:response_relevancy",
                result="FAIL",
                score=0.65,
                threshold=0.8,
                reason="Low relevancy score",
                query="Test query",
                response="Test response",
                execution_time=1.2,
            ),
        ]

    def test_output_handler_initialization(self):
        """Test OutputHandler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = OutputHandler(
                output_dir=temp_dir, base_filename="test_evaluation"
            )
            assert handler.output_dir == Path(temp_dir)
            assert handler.base_filename == "test_evaluation"

    def test_generate_reports(self, sample_results):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = OutputHandler(
                output_dir=temp_dir, base_filename="test_evaluation"
            )

            # Generate reports without graphs to avoid matplotlib issues in tests
            handler.generate_reports(sample_results, include_graphs=False)

            # Check that files were created
            output_files = list(Path(temp_dir).glob("test_evaluation_*"))
            assert len(output_files) >= 3  # CSV, JSON, TXT files


class TestIntegrationWithRealConfigs:
    """Integration tests using real configuration files."""

    @pytest.mark.integration
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("lightspeed_evaluation.core.metrics.ragas.RagasMetrics.evaluate")
    @patch("lightspeed_evaluation.core.metrics.deepeval.DeepEvalMetrics.evaluate")
    @patch("lightspeed_evaluation.core.metrics.custom.CustomMetrics.evaluate")
    def test_full_evaluation_pipeline(self, mock_custom, mock_deepeval, mock_ragas):
        """Test the complete evaluation pipeline with real config files."""
        system_config_path = "config/system.yaml"
        eval_data_path = "config/evaluation_data.yaml"

        # Skip if config files don't exist
        if not (Path(system_config_path).exists() and Path(eval_data_path).exists()):
            pytest.skip("Config files not found")

        # Mock all metric evaluations
        mock_ragas.return_value = (0.85, "Mocked ragas evaluation")
        mock_deepeval.return_value = (0.75, "Mocked deepeval evaluation")
        mock_custom.return_value = (0.80, "Mocked custom evaluation")

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_evaluation(
                system_config_path=system_config_path,
                evaluation_data_path=eval_data_path,
                output_dir=temp_dir,
            )

            # Verify summary statistics
            assert summary is not None
            assert "TOTAL" in summary
            assert "PASS" in summary
            assert "FAIL" in summary
            assert "ERROR" in summary
            assert summary["TOTAL"] > 0

            # Verify output files were created
            output_files = list(Path(temp_dir).glob("evaluation_*"))
            assert len(output_files) >= 3  # At least CSV, JSON, TXT

    @pytest.mark.integration
    def test_evaluation_with_mixed_results(self):
        """Test evaluation pipeline with mixed pass/fail results."""
        # Create test data with scenarios that should pass and fail
        test_data = [
            EvaluationData(
                conversation_group_id="high_quality_conv",
                turn_metrics=["ragas:faithfulness", "ragas:response_relevancy"],
                turns=[
                    TurnData(
                        turn_id=1,
                        query="What is renewable energy?",
                        response="Renewable energy comes from natural sources that replenish themselves, such as solar, wind, and hydroelectric power.",
                        contexts=[
                            {
                                "content": "Renewable energy sources are naturally replenishing and include solar, wind, water, and geothermal power."
                            }
                        ],
                    )
                ],
            ),
            EvaluationData(
                conversation_group_id="low_quality_conv",
                turn_metrics=["ragas:faithfulness"],
                turns=[
                    TurnData(
                        turn_id=1,
                        query="Explain quantum computing",
                        response="Quantum computing uses quantum bits.",
                        contexts=[
                            {
                                "content": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers."
                            }
                        ],
                    )
                ],
            ),
        ]

        with patch(
            "lightspeed_evaluation.core.metrics.ragas.RagasMetrics.evaluate"
        ) as mock_ragas:
            # Mock different scores for different conversations
            def side_effect(metric_name, conv_data, scope):
                if conv_data.conversation_group_id == "high_quality_conv":
                    return (0.92, "High quality response with good faithfulness")
                else:
                    return (0.45, "Low quality response, lacks detail")

            mock_ragas.side_effect = side_effect

            mock_config_loader = MagicMock(spec=ConfigLoader)
            mock_config_loader.get_llm_config_dict.return_value = {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                }
            }
            mock_config_loader.system_config = MagicMock()
            mock_config_loader.system_config.default_turn_metrics_metadata = {
                "ragas:faithfulness": {"threshold": 0.8}
            }

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                driver = EvaluationDriver(mock_config_loader)
                results = driver.run_evaluation(test_data)

                # Should have mixed results
                assert len(results) == 3  # 2 from first conv, 1 from second
                pass_results = [r for r in results if r.result == "PASS"]
                fail_results = [r for r in results if r.result == "FAIL"]

                assert len(pass_results) >= 1
                assert len(fail_results) >= 1

    def test_evaluation_with_missing_context_data(self):
        """Test evaluation behavior when required context data is missing."""
        test_data = [
            EvaluationData(
                conversation_group_id="missing_context_conv",
                turn_metrics=["ragas:faithfulness"],  # Requires context
                turns=[
                    TurnData(
                        turn_id=1,
                        query="What is AI?",
                        response="AI is artificial intelligence.",
                        contexts=[],  # Missing required context
                    )
                ],
            )
        ]

        validator = DataValidator()

        # Should fail validation due to missing context for faithfulness metric
        with patch(
            "lightspeed_evaluation.core.config.validator.TURN_LEVEL_METRICS",
            {"ragas:faithfulness"},
        ):
            result = validator.validate_evaluation_data(test_data)
            assert result is False
            assert len(validator.validation_errors) > 0
            assert "requires contexts" in validator.validation_errors[0]

    def test_evaluation_with_threshold_variations(self):
        """Test evaluation with different threshold configurations."""
        test_data = [
            EvaluationData(
                conversation_group_id="threshold_test_conv",
                turn_metrics=["ragas:faithfulness"],
                turn_metrics_metadata={
                    "ragas:faithfulness": {"threshold": 0.9}  # High threshold
                },
                turns=[
                    TurnData(
                        turn_id=1,
                        query="Explain photosynthesis",
                        response="Photosynthesis is how plants make food using sunlight.",
                        contexts=[
                            {
                                "content": "Photosynthesis is the process by which plants convert light energy into chemical energy."
                            }
                        ],
                    )
                ],
            )
        ]

        with patch(
            "lightspeed_evaluation.core.metrics.ragas.RagasMetrics.evaluate"
        ) as mock_ragas:
            mock_ragas.return_value = (
                0.85,
                "Good faithfulness score",
            )  # Below 0.9 threshold

            mock_config_loader = MagicMock(spec=ConfigLoader)
            mock_config_loader.get_llm_config_dict.return_value = {
                "llm": {"provider": "openai", "model": "gpt-4o-mini"}
            }
            mock_config_loader.system_config = MagicMock()
            mock_config_loader.system_config.default_turn_metrics_metadata = {}

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                driver = EvaluationDriver(mock_config_loader)
                results = driver.run_evaluation(test_data)

                assert len(results) == 1
                assert results[0].result == "FAIL"  # 0.85 < 0.9 threshold
                assert results[0].threshold == 0.9


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")

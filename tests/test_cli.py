"""Tests for command-line interface."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from lightspeed_evaluation.runner.evaluation import main, run_evaluation


class TestCLIInterface:
    """Test command-line interface functionality."""

    def test_main_with_help_argument(self):
        """Test main function with help argument."""
        with patch('sys.argv', ['lightspeed-eval', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help should exit with code 0
            assert exc_info.value.code == 0

    def test_main_with_missing_system_config(self):
        """Test main function with missing system config file."""
        with patch('sys.argv', ['lightspeed-eval', '--system-config', 'nonexistent.yaml']):
            with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables'):
                result = main()
                assert result == 1  # Should return error code

    def test_main_with_missing_eval_data(self):
        """Test main function with missing evaluation data file."""
        # Create temporary system config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"llm": {"provider": "openai"}}, f)
            system_config_path = f.name
        
        try:
            with patch('sys.argv', [
                'lightspeed-eval',
                '--system-config', system_config_path,
                '--eval-data', 'nonexistent.yaml'
            ]):
                with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables'):
                    result = main()
                    assert result == 1  # Should return error code
        finally:
            os.unlink(system_config_path)

    @patch('lightspeed_evaluation.runner.evaluation.run_evaluation')
    @patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables')
    def test_main_with_valid_arguments(self, mock_setup_env, mock_run_eval):
        """Test main function with valid arguments."""
        # Mock successful evaluation
        mock_run_eval.return_value = {
            "TOTAL": 5,
            "PASS": 3,
            "FAIL": 1,
            "ERROR": 1
        }
        
        # Create temporary config files
        system_config_data = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "output": {"base_directory": "./test_output"}
        }
        
        eval_data = [
            {
                "conversation_group_id": "test_conv",
                "turn_metrics": ["ragas:faithfulness"],
                "turns": [{"turn_id": 1, "query": "q", "response": "r"}]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as system_config:
            yaml.dump(system_config_data, system_config)
            system_config_path = system_config.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as eval_config:
            yaml.dump(eval_data, eval_config)
            eval_data_path = eval_config.name
        
        try:
            with patch('sys.argv', [
                'lightspeed-eval',
                '--system-config', system_config_path,
                '--eval-data', eval_data_path
            ]):
                result = main()
                assert result == 0  # Should return success code
                
                # Verify that run_evaluation was called with correct arguments
                mock_run_eval.assert_called_once_with(
                    system_config_path, eval_data_path, None
                )
                
        finally:
            os.unlink(system_config_path)
            os.unlink(eval_data_path)

    @patch('lightspeed_evaluation.runner.evaluation.run_evaluation')
    @patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables')
    def test_main_with_output_dir_override(self, mock_setup_env, mock_run_eval):
        """Test main function with output directory override."""
        mock_run_eval.return_value = {"TOTAL": 1, "PASS": 1, "FAIL": 0, "ERROR": 0}
        
        # Create temporary config files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as system_config:
            yaml.dump({"llm": {"provider": "openai"}}, system_config)
            system_config_path = system_config.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as eval_config:
            yaml.dump([{"conversation_group_id": "test"}], eval_config)
            eval_data_path = eval_config.name
        
        try:
            with patch('sys.argv', [
                'lightspeed-eval',
                '--system-config', system_config_path,
                '--eval-data', eval_data_path,
                '--output-dir', '/custom/output/dir'
            ]):
                result = main()
                assert result == 0
                
                # Verify that run_evaluation was called with custom output dir
                mock_run_eval.assert_called_once_with(
                    system_config_path, eval_data_path, '/custom/output/dir'
                )
                
        finally:
            os.unlink(system_config_path)
            os.unlink(eval_data_path)

    @patch('lightspeed_evaluation.runner.evaluation.run_evaluation')
    @patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables')
    def test_main_with_evaluation_failure(self, mock_setup_env, mock_run_eval):
        """Test main function when evaluation fails."""
        # Mock failed evaluation
        mock_run_eval.return_value = None
        
        # Create temporary config files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as system_config:
            yaml.dump({"llm": {"provider": "openai"}}, system_config)
            system_config_path = system_config.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as eval_config:
            yaml.dump([{"conversation_group_id": "test"}], eval_config)
            eval_data_path = eval_config.name
        
        try:
            with patch('sys.argv', [
                'lightspeed-eval',
                '--system-config', system_config_path,
                '--eval-data', eval_data_path
            ]):
                result = main()
                assert result == 1  # Should return error code
                
        finally:
            os.unlink(system_config_path)
            os.unlink(eval_data_path)


class TestRunEvaluation:
    """Test run_evaluation function."""

    @patch('lightspeed_evaluation.runner.evaluation.EvaluationDriver')
    @patch('lightspeed_evaluation.runner.evaluation.OutputHandler')
    @patch('lightspeed_evaluation.runner.evaluation.DataValidator')
    @patch('lightspeed_evaluation.runner.evaluation.ConfigLoader')
    def test_run_evaluation_success(self, mock_config_loader_class, mock_validator_class, 
                                   mock_output_handler_class, mock_driver_class):
        """Test successful run_evaluation execution."""
        # Mock ConfigLoader
        mock_loader = mock_config_loader_class.return_value
        mock_system_config = mock_loader.load_system_config.return_value
        mock_system_config.llm_provider = "openai"
        mock_system_config.llm_model = "gpt-4"
        mock_system_config.output_dir = "./test_output"
        mock_system_config.base_filename = "test_eval"
        mock_system_config.include_graphs = True
        
        # Mock DataValidator
        mock_validator = mock_validator_class.return_value
        mock_validator.load_evaluation_data.return_value = ["mock_data"]
        
        # Mock EvaluationDriver
        mock_driver = mock_driver_class.return_value
        mock_results = ["mock_result1", "mock_result2"]
        mock_driver.run_evaluation.return_value = mock_results
        
        # Mock OutputHandler
        mock_output_handler = mock_output_handler_class.return_value
        
        # Create temporary config files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as system_config:
            yaml.dump({"llm": {"provider": "openai"}}, system_config)
            system_config_path = system_config.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as eval_config:
            yaml.dump([{"conversation_group_id": "test"}], eval_config)
            eval_data_path = eval_config.name
        
        try:
            with patch('lightspeed_evaluation.runner.evaluation.calculate_basic_stats') as mock_stats:
                mock_stats.return_value = {
                    "TOTAL": 2,
                    "PASS": 1,
                    "FAIL": 1,
                    "ERROR": 0
                }
                
                result = run_evaluation(system_config_path, eval_data_path)
                
                # Verify result
                assert result is not None
                assert result["TOTAL"] == 2
                assert result["PASS"] == 1
                assert result["FAIL"] == 1
                assert result["ERROR"] == 0
                
                # Verify method calls
                mock_loader.load_system_config.assert_called_once_with(system_config_path)
                mock_validator.load_evaluation_data.assert_called_once_with(eval_data_path)
                mock_driver.run_evaluation.assert_called_once_with(["mock_data"])
                mock_output_handler.generate_reports.assert_called_once_with(mock_results, include_graphs=True)
                
        finally:
            os.unlink(system_config_path)
            os.unlink(eval_data_path)

    @patch('lightspeed_evaluation.runner.evaluation.ConfigLoader')
    def test_run_evaluation_config_loading_failure(self, mock_config_loader_class):
        """Test run_evaluation with configuration loading failure."""
        # Mock ConfigLoader to raise exception
        mock_loader = mock_config_loader_class.return_value
        mock_loader.load_system_config.side_effect = FileNotFoundError("Config not found")
        
        result = run_evaluation("nonexistent_system.yaml", "nonexistent_data.yaml")
        
        assert result is None

    @patch('lightspeed_evaluation.runner.evaluation.EvaluationDriver')
    @patch('lightspeed_evaluation.runner.evaluation.DataValidator')
    @patch('lightspeed_evaluation.runner.evaluation.ConfigLoader')
    def test_run_evaluation_with_custom_output_dir(self, mock_config_loader_class, 
                                                  mock_validator_class, mock_driver_class):
        """Test run_evaluation with custom output directory."""
        # Mock dependencies
        mock_loader = mock_config_loader_class.return_value
        mock_system_config = mock_loader.load_system_config.return_value
        mock_system_config.llm_provider = "openai"
        mock_system_config.llm_model = "gpt-4"
        mock_system_config.output_dir = "./default_output"
        mock_system_config.base_filename = "test_eval"
        mock_system_config.include_graphs = False
        
        mock_validator = mock_validator_class.return_value
        mock_validator.load_evaluation_data.return_value = []
        
        mock_driver = mock_driver_class.return_value
        mock_driver.run_evaluation.return_value = []
        
        # Create temporary config files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as system_config:
            yaml.dump({"llm": {"provider": "openai"}}, system_config)
            system_config_path = system_config.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as eval_config:
            yaml.dump([], eval_config)
            eval_data_path = eval_config.name
        
        try:
            with patch('lightspeed_evaluation.runner.evaluation.OutputHandler') as mock_output_handler_class:
                with patch('lightspeed_evaluation.runner.evaluation.calculate_basic_stats') as mock_stats:
                    mock_stats.return_value = {"TOTAL": 0, "PASS": 0, "FAIL": 0, "ERROR": 0}
                    
                    custom_output_dir = "/custom/output/path"
                    result = run_evaluation(system_config_path, eval_data_path, custom_output_dir)
                    
                    # Verify that OutputHandler was called with custom output directory
                    mock_output_handler_class.assert_called_once_with(
                        output_dir=custom_output_dir,
                        base_filename="test_eval",
                        system_config=mock_system_config
                    )
                    
                    assert result is not None
                    
        finally:
            os.unlink(system_config_path)
            os.unlink(eval_data_path)


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_default_arguments(self):
        """Test CLI with default arguments."""
        with patch('sys.argv', ['lightspeed-eval']):
            with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables'):
                with patch('lightspeed_evaluation.runner.evaluation.Path') as mock_path:
                    # Mock Path.exists to return False for default paths
                    mock_path.return_value.exists.return_value = False
                    
                    result = main()
                    assert result == 1  # Should fail due to missing files

    def test_custom_config_paths(self):
        """Test CLI with custom configuration paths."""
        custom_system_config = "/path/to/custom/system.yaml"
        custom_eval_data = "/path/to/custom/eval_data.yaml"
        
        with patch('sys.argv', [
            'lightspeed-eval',
            '--system-config', custom_system_config,
            '--eval-data', custom_eval_data
        ]):
            with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables'):
                with patch('lightspeed_evaluation.runner.evaluation.Path') as mock_path:
                    # Mock Path.exists to return False
                    mock_path.return_value.exists.return_value = False
                    
                    result = main()
                    assert result == 1  # Should fail due to missing files

    def test_argument_validation(self):
        """Test argument validation in CLI."""
        # Test that the argument parser accepts the expected arguments
        import argparse
        from lightspeed_evaluation.runner.evaluation import main
        
        # This test verifies that the argument parser is set up correctly
        # by checking that it doesn't raise an exception with valid arguments
        with patch('sys.argv', [
            'lightspeed-eval',
            '--system-config', 'test_system.yaml',
            '--eval-data', 'test_eval.yaml',
            '--output-dir', '/test/output'
        ]):
            with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables'):
                with patch('lightspeed_evaluation.runner.evaluation.Path') as mock_path:
                    mock_path.return_value.exists.return_value = False
                    
                    # Should not raise an ArgumentError
                    result = main()
                    assert result == 1  # Fails due to missing files, but args are valid


class TestCLIRealWorldScenarios:
    """Test CLI with real-world scenarios."""

    @patch('lightspeed_evaluation.runner.evaluation.run_evaluation')
    def test_cli_with_custom_output_directory(self, mock_run_eval):
        """Test CLI with custom output directory."""
        mock_run_eval.return_value = {"TOTAL": 5, "PASS": 4, "FAIL": 1, "ERROR": 0}
        
        with patch('sys.argv', [
            'lightspeed-eval',
            '--system-config', 'config/system.yaml',
            '--eval-data', 'config/evaluation_data.yaml',
            '--output-dir', '/custom/output/path'
        ]):
            with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables'):
                with patch('lightspeed_evaluation.runner.evaluation.Path') as mock_path:
                    mock_path.return_value.exists.return_value = True
                    
                    result = main()
                    assert result == 0

    @patch('lightspeed_evaluation.runner.evaluation.run_evaluation')
    def test_cli_evaluation_with_mixed_results(self, mock_run_eval):
        """Test CLI when evaluation has mixed results."""
        mock_run_eval.return_value = {"TOTAL": 10, "PASS": 6, "FAIL": 3, "ERROR": 1}
        
        with patch('sys.argv', [
            'lightspeed-eval',
            '--system-config', 'config/system.yaml',
            '--eval-data', 'config/evaluation_data.yaml'
        ]):
            with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables'):
                with patch('lightspeed_evaluation.runner.evaluation.Path') as mock_path:
                    mock_path.return_value.exists.return_value = True
                    
                    result = main()
                    # Should still return 0 (success) as the evaluation completed
                    assert result == 0

    def test_cli_with_environment_variables(self):
        """Test CLI behavior with environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'test-key-123',
            'DEEPEVAL_TELEMETRY_OPT_OUT': 'YES',
            'LITELLM_LOG_LEVEL': 'ERROR'
        }
        
        with patch('sys.argv', [
            'lightspeed-eval',
            '--system-config', 'config/system.yaml',
            '--eval-data', 'config/evaluation_data.yaml'
        ]):
            with patch.dict(os.environ, env_vars):
                with patch('lightspeed_evaluation.runner.evaluation.setup_environment_variables') as mock_setup:
                    with patch('lightspeed_evaluation.runner.evaluation.Path') as mock_path:
                        mock_path.return_value.exists.return_value = False
                        
                        result = main()
                        
                        # Verify environment setup was called
                        mock_setup.assert_called_once()
                        
                        # Should fail due to missing files, but env setup should have been called
                        assert result == 1

"""Evaluation Pipeline - Main evaluation orchestrator."""

import logging
from typing import Optional

from lightspeed_evaluation.core.api import APIClient
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.output.data_persistence import save_evaluation_data
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.core.system import ConfigLoader, DataValidator
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
from lightspeed_evaluation.pipeline.evaluation.processor import (
    ConversationProcessor,
    ProcessorComponents,
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Evaluation pipeline - orchestrates the evaluation process through different stages.

    Responsibilities:
    - Initialize and coordinate components
    - Validate data
    - Orchestrate evaluation flow
    - Collect results
    - Save amended data
    """

    def __init__(self, config_loader: ConfigLoader, output_dir: Optional[str] = None):
        """Initialize evaluation pipeline with config and create components."""
        self.config_loader = config_loader
        if not config_loader.system_config:
            raise ValueError("SystemConfig must be loaded before initializing pipeline")
        self.original_data_path: Optional[str] = None
        self.output_dir = output_dir or config_loader.system_config.output.output_dir

        # Initialize components
        self._initialize_components()
        logger.info("Evaluation Pipeline initialized")

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        # Data validator
        config = self.config_loader.system_config
        if config is None:
            raise ValueError(
                "SystemConfig must be loaded before initializing components"
            )
        self.data_validator = DataValidator(api_enabled=config.api.enabled)

        # Metric manager
        metric_manager = MetricManager(config)

        # Create pipeline components
        self.api_client = self._create_api_client()
        api_amender = APIDataAmender(self.api_client)
        error_handler = EvaluationErrorHandler()

        # Create script execution manager
        script_manager = ScriptExecutionManager()

        # Create metrics evaluator with script manager
        metrics_evaluator = MetricsEvaluator(
            self.config_loader, metric_manager, script_manager
        )

        # Create processor components
        processor_components = ProcessorComponents(
            metrics_evaluator=metrics_evaluator,
            api_amender=api_amender,
            error_handler=error_handler,
            metric_manager=metric_manager,
            script_manager=script_manager,
        )

        # Conversation processor
        self.conversation_processor = ConversationProcessor(
            self.config_loader,
            processor_components,
        )

    def _create_api_client(self) -> Optional[APIClient]:
        """Create API client if enabled."""
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded before creating API client")
        if not config.api.enabled:
            return None

        api_config = config.api
        logger.info("Setting up API client: %s", api_config.api_base)

        client = APIClient(
            api_base=api_config.api_base,
            config={
                "provider": api_config.provider,
                "model": api_config.model,
                "no_tools": api_config.no_tools,
                "system_prompt": api_config.system_prompt,
            },
            endpoint_type=api_config.endpoint_type,
            timeout=api_config.timeout,
        )

        logger.info("API client initialized for %s endpoint", api_config.endpoint_type)
        return client

    def validate_data(self, evaluation_data: list[EvaluationData]) -> bool:
        """Validate evaluation data using data validator."""
        return self.data_validator.validate_evaluation_data(evaluation_data)

    def run_evaluation(
        self,
        evaluation_data: list[EvaluationData],
        original_data_path: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """Run evaluation on provided data.

        Args:
            evaluation_data: List of conversation data to evaluate
            original_data_path: Path to original data file for saving updates

        Returns:
            List of evaluation results.
        """
        self.original_data_path = original_data_path
        logger.info("Starting evaluation")
        results: list[EvaluationResult] = []

        # Step 1: Validate data
        logger.info("Validating data")
        if not self.validate_data(evaluation_data):
            raise ValueError("Data validation failed. Cannot proceed with evaluation.")

        # Step 2: Process each conversation
        logger.info("Processing conversations")
        for conv_data in evaluation_data:
            conv_results = self.conversation_processor.process_conversation(conv_data)
            results.extend(conv_results)

        # Step 3: Save amended data if API was used
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded")
        if config.api.enabled:
            logger.info("Saving amended evaluation data")
            self._save_amended_data(evaluation_data)

        logger.info("Evaluation complete: %d results generated", len(results))
        return results

    def _save_amended_data(self, evaluation_data: list[EvaluationData]) -> None:
        """Save amended evaluation data with API amendments to output directory."""
        if not self.original_data_path:
            logger.warning("No original data path available, cannot save amended data")
            return

        try:
            amended_file = save_evaluation_data(
                evaluation_data, self.original_data_path, self.output_dir
            )
            if amended_file:
                logger.info("Amended data saved: %s", amended_file)
                logger.info(
                    "To use amended data without new API calls, "
                    "disable the API call using system config & "
                    "replace the original evaluation data file with the amended file"
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Don't fail the evaluation if saving fails
            logger.warning("Failed to save amended data: %s", e)

    def close(self) -> None:
        """Clean up resources."""
        if self.api_client:
            self.api_client.close()

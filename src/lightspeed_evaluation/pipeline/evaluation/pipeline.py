"""Evaluation Pipeline - Main evaluation orchestrator."""

import asyncio
import concurrent.futures
import logging
from typing import Any, Optional

import litellm
import tqdm

from lightspeed_evaluation.core.llm.litellm_patch import litellm_state_lock
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    SystemConfig,
)
from lightspeed_evaluation.core.output.data_persistence import save_evaluation_data
from lightspeed_evaluation.core.script import ScriptExecutionManager
from lightspeed_evaluation.core.storage import (
    BaseStorageBackend,
    RunInfo,
    create_pipeline_storage_backend,
    get_file_config,
)
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import (
    ConfigurationError,
    StorageError,
)
from lightspeed_evaluation.pipeline.evaluation.driver import (
    AgentDriver,
    AgentDriverRegistry,
)
from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
from lightspeed_evaluation.pipeline.evaluation.processor import (
    ConversationProcessor,
    ProcessorComponents,
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:  # pylint: disable=too-many-instance-attributes
    """Evaluation pipeline - orchestrates the evaluation process through different stages.

    Responsibilities:
    - Initialize and coordinate components
    - Orchestrate evaluation flow
    - Collect results
    - Save amended data
    """

    def __init__(self, config_loader: ConfigLoader, output_dir: Optional[str] = None):
        """Initialize evaluation pipeline with config and create components."""
        self.config_loader = config_loader
        if not config_loader.system_config:
            raise ValueError("SystemConfig must be loaded before initializing pipeline")

        self.system_config: SystemConfig = config_loader.system_config
        self.original_data_path: Optional[str] = None
        file_config = get_file_config(config_loader.system_config.storage)
        self.output_dir = output_dir or file_config.output_dir

        self.storage_backend: BaseStorageBackend = create_pipeline_storage_backend(
            config_loader.system_config.storage,
            system_config=config_loader.system_config,
            output_dir_override=output_dir,
        )

        # Initialize components
        self._initialize_components()
        logger.info("Evaluation Pipeline initialized")

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        config = self.config_loader.system_config
        if config is None:
            raise ValueError(
                "SystemConfig must be loaded before initializing components"
            )

        # Metric manager
        metric_manager = MetricManager(config)

        # Create agent driver registry and default driver
        self._registry = AgentDriverRegistry()
        self._default_driver = self._create_default_driver()

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
            error_handler=error_handler,
            metric_manager=metric_manager,
            script_manager=script_manager,
        )

        # Conversation processor
        self.conversation_processor = ConversationProcessor(
            self.config_loader,
            processor_components,
        )

    def _create_default_driver(self) -> AgentDriver:
        """Create the default agent driver from system config."""
        _name, agent_config = self._resolve_default_agent_config()
        enabled = (
            self.system_config.agents is not None and self.system_config.agents.enabled
        )
        return self._registry.create_driver(agent_config, enabled=enabled)

    def _resolve_default_agent_config(self) -> tuple[str, dict[str, Any]]:
        """Resolve the default agent configuration.

        Returns:
            Tuple of (agent_name, config_dict).
        """
        config = self.system_config
        if config.agents is not None and config.agents.default.agent is not None:
            return config.agents.resolve_agent_config()
        return ("http_api", {"type": "http_api"})

    def _resolve_driver_for_conversation(
        self, conv_data: EvaluationData
    ) -> tuple[AgentDriver, bool]:
        """Resolve the agent driver for a conversation.

        Returns:
            Tuple of (driver, is_per_conversation). The boolean indicates whether
            the driver was created specifically for this conversation and should
            be closed after use.
        """
        if not conv_data.agent and not conv_data.agent_config:
            return self._default_driver, False

        if self.system_config.agents is None:
            raise ConfigurationError(
                f"Conversation '{conv_data.conversation_group_id}' specifies "
                f"agent overrides but no agents configuration is defined"
            )

        if not self.system_config.agents.enabled:
            return self._default_driver, False

        _name, agent_config = self.system_config.agents.resolve_agent_config(
            agent_name=conv_data.agent,
            agent_config_override=conv_data.agent_config,
        )
        return (
            self._registry.create_driver(
                agent_config, enabled=self.system_config.agents.enabled
            ),
            True,
        )

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

        run_name = original_data_path or "evaluation"
        self.storage_backend.initialize(RunInfo(name=run_name))

        try:
            # Process each conversation
            logger.info("Processing conversations")
            results = self._process_eval_data(evaluation_data)
        finally:
            self.storage_backend.set_evaluation_context(evaluation_data)
            self.storage_backend.finalize()
            self.storage_backend.close()

        if self.system_config.agents is not None and self.system_config.agents.enabled:
            logger.info("Saving amended evaluation data")
            self._save_amended_data(evaluation_data)

        logger.info("Evaluation complete: %d results generated", len(results))
        return results

    def _process_eval_data(
        self, evaluation_data: list[EvaluationData]
    ) -> list[EvaluationResult]:
        """Process the conversations from the evaluation_data."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.system_config.core.max_threads
        ) as executor:
            futures = {
                executor.submit(self._process_conversation, c): c
                for c in evaluation_data
            }
            results: list[EvaluationResult] = []
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(evaluation_data)
            ):
                conversation_results = future.result()
                # Batch save results per conversation (more efficient than individual saves)
                if conversation_results:
                    try:
                        self.storage_backend.save_run(conversation_results)
                    except StorageError as e:
                        logger.warning("Failed to save results to storage: %s", e)
                results.extend(conversation_results)
            return results

    def _process_conversation(
        self, conv_data: EvaluationData
    ) -> list[EvaluationResult]:
        """Resolve driver and process a single conversation."""
        driver, is_per_conversation = self._resolve_driver_for_conversation(conv_data)
        try:
            return self.conversation_processor.process_conversation(conv_data, driver)
        finally:
            if is_per_conversation:
                driver.close()

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
        """Clean up resources.

        Uses a lock to serialize litellm cache teardown across concurrent
        pipelines, since ``litellm.cache`` is process-global state.
        """
        self._default_driver.close()

        self.storage_backend.close()

        with litellm_state_lock:
            cache = litellm.cache
            if cache is not None:
                try:
                    # Use getattr to call untyped third-party method
                    disconnect = getattr(cache, "disconnect")
                    asyncio.run(disconnect())
                except (AttributeError, RuntimeError, OSError):
                    logger.debug("litellm cache disconnect raised; ignoring")
                litellm.cache = None

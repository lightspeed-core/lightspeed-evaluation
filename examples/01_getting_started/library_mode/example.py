"""Library Mode Example - Programmatic API Usage.

Demonstrates how to use LightSpeed Evaluation as a Python library
for real-time integration with your applications.
"""

from lightspeed_evaluation import EvaluationData, TurnData, evaluate
from lightspeed_evaluation.core.models.system import (
    APIConfig,
    JudgePanelConfig,
    LLMPoolConfig,
    LLMProviderConfig,
    SystemConfig,
)

# Configure system programmatically
config = SystemConfig(
    llm_pool=LLMPoolConfig(
        models={"judge": LLMProviderConfig(provider="openai", model="gpt-4o-mini")}
    ),
    judge_panel=JudgePanelConfig(judges=["judge"]),
    api=APIConfig(enabled=False),
)

# Create evaluation data programmatically
eval_data = EvaluationData(
    conversation_group_id="programmatic_eval",
    description="Example programmatic evaluation",
    tag="library",
    turns=[
        TurnData(
            turn_id="namespace_query",
            query="What is a namespace in OpenShift?",
            response=(
                "A namespace in OpenShift is a logical partition within a "
                "cluster that provides scope for resource names and allows "
                "for resource isolation, access control, and quota management. "
                "It helps organize and separate different projects or teams."
            ),
            expected_response=(
                "A namespace provides isolation and scope for resources in OpenShift."
            ),
            turn_metrics=["custom:answer_correctness"],
        )
    ],
)

# Run evaluation
print("Running evaluation...")
results = evaluate(config, [eval_data])

# Access results programmatically
print("\n=== Evaluation Results ===\n")
for result in results:
    print(f"Conversation: {result.conversation_group_id}")
    print(f"Turn: {result.turn_id}")
    print(f"Metric: {result.metric_identifier}")

    score_str = f"{result.score:.2f}" if result.score is not None else "N/A"
    print(f"Score: {score_str}")
    print(f"Result: {result.result}")
    print(f"Reason: {result.reason}")

print("\n✅ Evaluation complete!")

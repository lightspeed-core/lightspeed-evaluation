"""Prompt for Judge LLM."""

# Basic Prompt to check correctness (returns 1 or 0)
ANSWER_CORRECTNESS_PROMPT = """You are an expert evaluator. \
Your task is to evaluate whether a response correctly answers \
a question based on the expected answer.

Question: {question}
Expected Answer: {answer}
Actual Response: {response}

Please evaluate if the actual response is correct based on the expected answer.
Consider:
1. Does the response answer the question correctly?
2. Does it contain the key information from the expected answer?
3. Is the response factually accurate?

Return only 1 if the response is correct, or 0 if it is incorrect.
Do not add any other text apart from 0 or 1 in your response."""

# Intent Detection Prompt to check if response has the expected intent (returns 1 or 0)
INTENT_DETECTION_PROMPT = """You are an expert evaluator. \
Your task is to evaluate whether a response demonstrates the expected intent or purpose.

Question: {question}
Expected Intent of Response: {intent}
Actual Response: {response}

Please evaluate if the actual response has the expected intent.
Consider:
1. What is the intent/purpose of the actual response?
2. Does the response's intent match the expected intent?
3. Is the response trying to achieve what is described in the expected intent?

For example:
- If expected intent is "provide instructions", check if the response is instructional
- If expected intent is "explain a concept", check if the response is explanatory
- If expected intent is "refuse or decline", check if the response is declining to help
- If expected intent is "ask for clarification", check if the response is asking questions

Return only 1 if the response has the expected intent, or 0 if it does not.
Do not add any other text apart from 0 or 1 in your response."""

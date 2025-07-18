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

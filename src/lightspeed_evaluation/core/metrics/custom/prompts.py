"""Prompts for custom metrics evaluation."""

# Answer Correctness Evaluation Prompt
ANSWER_CORRECTNESS_PROMPT = """Evaluate the answer correctness of the given response.

Question: {query}
Response: {response}
Expected Response: {expected_response}

Consider:
- Factual accuracy compared to expected response
- Completeness of information
- Alignment with expected response
- Absence of contradictory information

Rate the answer correctness and provide your reasoning.

Format your response as:
Score: [your score on a scale of 0.0 to 1.0]
Reason: [your detailed explanation]"""

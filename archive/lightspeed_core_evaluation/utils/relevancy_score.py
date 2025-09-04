"""Relevancy score calculation."""

from statistics import mean
from time import sleep
from typing import Optional

from langchain.llms.base import LLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from scipy.spatial.distance import cosine

from .constants import MAX_RETRY_ATTEMPTS, N_QUESTIONS, TIME_TO_BREATH
from .prompts import ANSWER_RELEVANCY_PROMPT


class AnswerRelevancyScore:  # pylint: disable=R0903
    """Calculate response/answer relevancy score."""

    def __init__(self, judge_llm: LLM, embedding_model: HuggingFaceEmbedding) -> None:
        """Initialize."""
        self._embedding_model = embedding_model

        prompt = PromptTemplate.from_template(ANSWER_RELEVANCY_PROMPT)
        self._judge_llm = prompt | judge_llm | JsonOutputParser()

    def get_score(
        self,
        question: str,
        response: str,
        retry_attempts: int = MAX_RETRY_ATTEMPTS,
        time_to_breath: int = TIME_TO_BREATH,
    ) -> tuple[Optional[float], Optional[int], Optional[str]]:
        """Calculate relevancy score."""
        score, valid_flag, gen_questions = None, None, None
        # Generate relevant questions.
        for retry_counter in range(retry_attempts):
            try:
                out = self._judge_llm.invoke({"answer": response, "num_questions": N_QUESTIONS})
                valid_flag = out["Valid"]
                gen_questions = out["Question"]
                score = 0
                if valid_flag == 1:
                    org_vec = self._embedding_model.get_text_embedding(question)
                    score = mean(
                        [
                            1
                            - cosine(
                                org_vec,
                                self._embedding_model.get_text_embedding(gen_question),
                            )
                            for gen_question in gen_questions
                        ]
                    )
                gen_questions = "\n".join(gen_questions)
                break

            except Exception as e:  # pylint: disable=W0718
                if retry_counter == retry_attempts - 1:
                    print(f"error_answer_relevancy: {e}")
                    score, valid_flag, gen_questions = None, None, None
                    # raise

            sleep(time_to_breath)

        return score, valid_flag, gen_questions

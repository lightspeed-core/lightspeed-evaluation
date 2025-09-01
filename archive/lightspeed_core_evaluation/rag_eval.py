"""RAG Evaluation."""

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from time import sleep
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from numpy import argsort, array
from ols import config
from pandas import DataFrame, read_parquet
from tqdm import tqdm

from .eval_run_common import add_common_arguments
from .utils.constants import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_INPUT_DIR,
    DEFAULT_RESULT_DIR,
    MAX_RETRY_ATTEMPTS,
    PARQUET_QNA_FILE,
    TIME_TO_BREATH,
)
from .utils.models import VANILLA_MODEL
from .utils.plot import plot_score
from .utils.prompts import RAG_RELEVANCY_PROMPT1

tqdm.pandas()


def _args_parser(args: list[str]) -> Namespace:
    """Arguments parser."""
    parser = ArgumentParser(description="RAG evaluation module.")
    # Add arguments common to all eval scripts
    add_common_arguments(parser)

    parser.add_argument(
        nargs="+",
        default=None,
        help="Ids of questions to be validated. Check json file for valid ids.",
    )
    parser.add_argument(
        "--qna_pool_file",
        type=str,
        help="Additional file having QnA pool in parquet format.",
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=3,
        help="Number of chunks to be retrieved",
    )
    return parser.parse_args(args)


# TODO: LCORE-271 pylint: disable=W0511
class RetrievalEvaluation:  # pylint: disable=R0903
    """Evaluate Retrieval."""

    def __init__(self, eval_args: Namespace) -> None:
        """Initialize."""
        print(f"Arguments: {eval_args}")
        self._args = eval_args

        self._load_judge_and_rag()  # Set global config
        self._input_dir, self._result_dir = self._set_directories()
        self._qa_pool_df = self._load_qna_pool_parquet()

    def _load_judge_and_rag(self) -> None:
        """Load Judge model and RAG."""
        # Load config separately
        # Use OLS config file to set Judge provider/model & vector db
        cfg_file = os.environ.get("OLS_CONFIG_FILE", DEFAULT_CONFIG_FILE)
        config.reload_from_yaml_file(cfg_file)

        # load rag index
        config.rag_index  # pylint: disable=W0104
        if config.rag_index is None:
            raise RuntimeError("No valid rag index.")
        self._retriever = config.rag_index.as_retriever(
            similarity_top_k=self._args.n_chunks
        )

        provider_config = config.config.llm_providers.providers[
            self._args.judge_provider
        ]
        assert provider_config.type is not None, "Provider type must be configured"
        judge_llm = VANILLA_MODEL[
            provider_config.type
        ](  # pyright: ignore [reportCallIssue]
            self._args.judge_model, provider_config
        ).load()

        prompt = PromptTemplate.from_template(RAG_RELEVANCY_PROMPT1)
        self._judge_llm = prompt | judge_llm | JsonOutputParser()

    def _set_directories(self) -> tuple[str, str]:
        """Set input/output directories."""
        eval_dir = os.path.dirname(__file__)
        input_dir = os.path.join(eval_dir, DEFAULT_INPUT_DIR)

        result_dir = os.path.join(
            (self._args.eval_out_dir or eval_dir), DEFAULT_RESULT_DIR
        )
        os.makedirs(result_dir, exist_ok=True)
        return input_dir, result_dir

    def _load_qna_pool_parquet(self) -> DataFrame:
        """Load QnA pool from parquet file."""
        input_file = self._args.qna_pool_file
        if not input_file:
            input_file = os.path.join(self._input_dir, PARQUET_QNA_FILE)

        qna_pool_df = read_parquet(input_file)
        qna_pool_df = qna_pool_df[["Question"]].drop_duplicates().reset_index()
        qna_pool_df = qna_pool_df.rename(
            columns={"ID": "query_id", "Question": "question"}
        )
        qna_pool_df.query_id = qna_pool_df.query_id.astype(str)

        if self._args.eval_query_ids is not None:
            qna_pool_df = qna_pool_df[
                qna_pool_df.query_id.isin(self._args.eval_query_ids)
            ].reset_index(drop=True)
        return qna_pool_df  # pyright: ignore [reportReturnType]

    def _load_and_process_chunks(self, query: str) -> str:
        """Load and process chunks."""
        nodes = self._retriever.retrieve(query)
        chunks = [
            f"Search Result {idx+1}:\n{node.get_text()}"
            for idx, node in enumerate(nodes)
            if float(node.get_score(raise_error=False)) > 0.3
        ]
        return "\n\n".join(chunks)

    def _get_judge_response(self, query: str) -> dict[str, Any]:
        """Get Judge response."""
        print("Getting Judge response...")
        result = {}
        for retry_counter in range(MAX_RETRY_ATTEMPTS):
            try:
                result = self._judge_llm.invoke(
                    {
                        "query": query,
                        "n_results": self._args.n_chunks,
                        "retrieval_texts": self._load_and_process_chunks(query),
                    }
                )
                break
            except Exception as e:  # pylint: disable=W0718
                if retry_counter == MAX_RETRY_ATTEMPTS - 1:
                    print(f"error_rag_relevancy: {e}")
                    # Continue with empty result
                    result = {}
            sleep(TIME_TO_BREATH)

        return result

    def _process_score(self, score_data: dict[str, Any]) -> float:
        """Process score."""
        relevance_score = array(score_data["relevance_score"])
        completeness_score = array(score_data["completeness_score"])

        # Ignoring conciseness_score for now.
        chunk_wise_avg_score = (relevance_score + completeness_score) / 2
        rank = argsort(chunk_wise_avg_score)[::-1]

        # Penalize score
        score = chunk_wise_avg_score / ((rank + 1) * 10)
        # Currently using only 1st chunk score.
        return score[0]

    def get_final_score(self) -> None:
        """Get final score."""
        self._qa_pool_df["raw_judge_response"] = (
            self._qa_pool_df.question.progress_apply(self._get_judge_response)
        )
        self._qa_pool_df.to_csv(f"{self._result_dir}/chunk_eval.csv", index=False)

        self._qa_pool_df["score"] = self._qa_pool_df.raw_judge_response.apply(
            self._process_score
        )
        self._qa_pool_df.to_csv(f"{self._result_dir}/chunk_eval_score.csv", index=False)

        plot_score(
            self._qa_pool_df[["score"]],  # pyright: ignore [reportArgumentType]
            "1st Chunk Score",
            f"{self._result_dir}/chunk_eval.png",
        )
        summary_result = {
            "timestamp": str(datetime.now(UTC)),
            "score_based_on_first_chunk": self._qa_pool_df["score"]
            .describe()
            .T.to_dict(),
        }
        with open(f"{self._result_dir}/chunk_eval.json", "w", encoding="utf-8") as f:
            json.dump(summary_result, f)


def main() -> None:
    """Evaluate RAG."""
    args = _args_parser(sys.argv[1:])

    RetrievalEvaluation(args).get_final_score()


if __name__ == "__main__":
    main()

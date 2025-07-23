"""Taxonomy Answer/Context Evaluation."""

import os
import sys
from argparse import ArgumentParser, Namespace
from time import sleep
from typing import Any

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from ols import config
from pandas import DataFrame, Series
from tqdm import tqdm

from .eval_run_common import add_common_arguments
from .utils.constants import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_RESULT_DIR,
    MAX_RETRY_ATTEMPTS,
    TIME_TO_BREATH,
)
from .utils.models import VANILLA_MODEL
from .utils.prompts import GROUNDNESS_PROMPT, TAXONOMY_CONTEXT_RELEVANCY

tqdm.pandas()


# Sample taxonomy file
# https://github.com/instructlab/taxonomy/blob/main/knowledge/arts/music/fandom/swifties/qna.yaml


# TODO: LCORE-271 pylint: disable=W0511
def _args_parser(args: list[str]) -> Namespace:
    """Arguments parser."""
    parser = ArgumentParser(description="Taxonomy evaluation module.")
    # Add arguments common to all eval scripts
    add_common_arguments(parser)

    parser.add_argument(
        "--taxonomy_file_path",
        type=str,
        help="Taxonomy file path.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["context", "answer", "all"],
        default="answer",
        type=str,
        help="Evaluation type",
    )
    parser.add_argument(
        "--eval_method",
        choices=["ragas", "custom"],
        default="custom",
        type=str,
        help="Evaluation type",
    )
    return parser.parse_args(args)


class TaxonomyEval:  # pylint: disable=R0903
    """Evaluate taxonomy answer/context."""

    def __init__(self, eval_args: Namespace) -> None:
        """Initialize."""
        print(f"Arguments: {eval_args}")
        self._args = eval_args

        self._load_judge()  # Set global config
        self._set_output_dir()
        self._load_taxonomy_yaml()

    def _load_judge(self) -> None:
        """Load Judge."""
        print("Loading judge model...")
        # Load config separately
        # Use OLS config file to set Judge provider/model
        cfg_file = os.environ.get("OLS_CONFIG_FILE", DEFAULT_CONFIG_FILE)
        config.reload_from_yaml_file(cfg_file)

        provider_config = config.config.llm_providers.providers[
            self._args.judge_provider
        ]
        assert provider_config.type is not None, "Provider type must be configured"
        self._judge_llm = VANILLA_MODEL[
            provider_config.type
        ](  # pyright: ignore [reportCallIssue]
            self._args.judge_model, provider_config
        ).load()

    def _set_output_dir(self) -> None:
        """Set output directory."""
        eval_dir = os.path.dirname(__file__)

        result_dir = os.path.join(
            (self._args.eval_out_dir or eval_dir), DEFAULT_RESULT_DIR
        )
        os.makedirs(result_dir, exist_ok=True)
        self._result_dir = result_dir

    def _load_taxonomy_yaml(self) -> None:
        """Load taxonomy YAML file."""
        print("Loading taxonomy file...")
        with open(self._args.taxonomy_file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        data_f = [
            {**qna, "context": ex["context"]}
            for ex in data["seed_examples"]
            for qna in ex["questions_and_answers"]
        ]
        self._taxonomy_df = DataFrame(data_f)

    def _get_judge_response(
        self, question: str, answer: str, context: str, prompt: str
    ) -> dict[str, Any]:
        """Get Judge response."""
        print("Getting Judge response...")
        result = None
        llm_prompt = PromptTemplate.from_template(prompt)
        judge_llm = llm_prompt | self._judge_llm | JsonOutputParser()

        for retry_counter in range(MAX_RETRY_ATTEMPTS):
            try:
                result = judge_llm.invoke(
                    {
                        "question": question,
                        "answer": answer,
                        "context": context,
                    }
                )
                break
            except Exception as e:  # pylint: disable=W0718
                if retry_counter == MAX_RETRY_ATTEMPTS - 1:
                    print(f"error_groundness_score: {e}")
                    # Continue with empty result
                    result = {}
            sleep(TIME_TO_BREATH)

        return result  # pyright: ignore [reportReturnType]

    def _get_score(self, df: DataFrame, scores: list[str], prompt: str) -> DataFrame:
        """Get score."""
        df["score"] = df.progress_apply(
            lambda row: self._get_judge_response(
                row.question, row.answer, row.context, prompt
            ),
            axis=1,
            # result_type="expand",
        )
        for s in scores:
            df[s] = df["score"].apply(lambda x: x.get(s, None))  # pylint: disable=W0640
        return df

    def _get_custom_score(self) -> DataFrame:
        """Get custom score."""
        df = self._taxonomy_df.copy()
        if self._args.eval_type in ("all", "context"):
            scores = ["valid_flag", "relevancy_score"]
            df = self._get_score(df, scores, TAXONOMY_CONTEXT_RELEVANCY)
            renamed_columns = {score: f"context_{score}" for score in scores}
            df.rename(columns=renamed_columns, inplace=True)
        if self._args.eval_type in ("all", "answer"):
            scores = ["relevancy_score", "groundness_score"]
            df = self._get_score(df, scores, GROUNDNESS_PROMPT)
            renamed_columns = {score: f"answer_{score}" for score in scores}
            df.rename(columns=renamed_columns, inplace=True)
        df.drop(columns=["score"], inplace=True)
        return df

    def _get_ragas_score(self) -> DataFrame:
        """Get ragas score."""
        # pylint: disable=C0415
        from ragas import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

        judge_llm = LangchainLLMWrapper(self._judge_llm)

        def _get_score(
            data: Series, scorer: LLMContextPrecisionWithoutReference | Faithfulness
        ) -> float:
            data = SingleTurnSample(  # pyright: ignore [reportAssignmentType]
                user_input=data.question,
                response=data.answer,
                retrieved_contexts=[data.context],
            )
            return scorer.single_turn_score(
                data  # pyright: ignore [reportArgumentType]
            )

        df = self._taxonomy_df.copy()
        if self._args.eval_type in ("all", "context"):
            scorer = LLMContextPrecisionWithoutReference(llm=judge_llm)
            df["context_relevancy_score"] = df.progress_apply(
                lambda x: _get_score(x, scorer), axis=1
            )
        if self._args.eval_type in ("all", "answer"):
            scorer = Faithfulness(llm=judge_llm)
            df["answer_groundness_score"] = df.progress_apply(
                lambda x: _get_score(x, scorer), axis=1
            )
        return df

    def get_eval_result(self) -> None:
        """Get evaluation result."""
        if self._args.eval_method == "ragas":
            result_df = self._get_ragas_score()
        else:
            result_df = self._get_custom_score()

        result_df.to_csv(
            f"{self._result_dir}/context_eval-{self._args.eval_method}.csv", index=False
        )


def main() -> None:
    """Evaluate taxonomy context/answer."""
    args = _args_parser(sys.argv[1:])
    TaxonomyEval(args).get_eval_result()


if __name__ == "__main__":
    main()

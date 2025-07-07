"""Evaluation of LightSpeed services using MLFlow."""

import json
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import cast

import click
import pandas as pd
from tqdm import tqdm

from generate_answers.eval_config import read_eval_config
from generate_answers.ls_response import LSClient

tqdm.pandas()
logger = logging.getLogger(__name__)

# Input/output column names for id, question and original answer
_ID_COL = "id"
_QUESTION_COL = "question"
_ORIG_ANSWER_COL = "original_answer"

_QNA_COLS = [_ID_COL, _QUESTION_COL, _ORIG_ANSWER_COL]


def read_qna_csv(filename: str) -> pd.DataFrame:
    """Read question/answer pairs from CSV file."""
    return pd.read_csv(filename)


def read_qna_lightspeed_eval_parquet(filename: str) -> pd.DataFrame:
    """Read question/answer pairs from Lightspeed eval parquet file."""
    logger.info("Reading qna from '%s'", filename)

    df = pd.read_parquet(filename).rename(
        columns={
            "Question": _QUESTION_COL,
            "Answer": _ORIG_ANSWER_COL,
        }
    )
    df[_ID_COL] = df.index
    return cast(pd.DataFrame, df[[*_QNA_COLS]])


def read_qna_lightspeed_eval_json(filename: str) -> pd.DataFrame:
    """Read question/answer pairs from Lightspeed eval json file."""
    logger.info("Reading qna from '%s'", filename)
    with open(filename, encoding="utf8") as f:
        qna = json.loads(f.read())

    df = pd.DataFrame(columns=_QNA_COLS)  # type: ignore

    for eval_id, eval_data in qna["evaluation"].items():
        assert eval_id not in df[_ID_COL], f"Non unique eval_id: {eval_id}"
        try:
            question = eval_data["question"]
            ground_truth = eval_data["answer"]["ground_truth+with_rag"]["text"]
        except KeyError:
            logger.error("Can't parse eval data, eval_id='%s'", eval_id)
            continue

        df.loc[len(df)] = [eval_id, question, ground_truth]
    return df


@click.command(
    help="""
Generate answers from LLMs by connection to LightSpeed core service.
""",
    #    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "show_default": True,
    },
)
@click.option(
    "-c",
    "--config-filename",
    default="./src/generate_answers/eval_config.yaml",
    type=click.Path(exists=True),
    help="Configuration file",
)
@click.option(
    "-i",
    "--input-filename",
    default="./eval_data/questions.csv",
    type=click.Path(exists=True),
    help="Input filename with questions",
)
@click.option(
    "-o",
    "--output-filename",
    default="./eval_output/generated_qna.json",
    type=click.Path(),
    help="Output JSON filename with results -- generated answers",
)
@click.option(
    "-l",
    "--llm-cache-dir",
    default="./llm_cache",
    type=click.Path(),
    help="Directory with cached responses from LLMs. Cache key is model+provider+question",
)
@click.option(
    "-f",
    "--force-overwrite",
    default=False,
    is_flag=True,
    help="Overwrite the output file if it exists",
)
@click.option(
    "-v",
    "--verbose",
    default=False,
    is_flag=True,
    help="Increase the logging level to DEBUG",
)
def main(  # pylint: disable=R0913,R0917,R0914
    config_filename: str,
    input_filename: str,
    output_filename: str,
    llm_cache_dir: str,
    force_overwrite: bool,
    verbose: bool,
) -> int:
    """Run main entrypoint."""
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG
    logging.basicConfig(
        level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s [%(name)s:%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    )

    if not force_overwrite:
        if os.path.exists(output_filename):
            logger.error("Filename '%s' exists, use -f to overwrite", output_filename)
            return 1

    config = read_eval_config(config_filename)

    evaluators = [
        (
            model,
            LSClient(
                config.lightspeed_url,
                provider=model.provider,
                model=model.model,
                cache_dir=llm_cache_dir,
            ),
        )
        for model in config.models
    ]

    suffix = Path(input_filename).suffix.lower()
    read_func = {
        ".json": read_qna_lightspeed_eval_json,
        ".parquet": read_qna_lightspeed_eval_parquet,
        ".csv": read_qna_csv,
    }.get(suffix, read_qna_csv)

    qna_df = read_func(input_filename)

    # Remove empty questions
    qna_df = qna_df[qna_df[_QUESTION_COL].notna() & (qna_df[_QUESTION_COL] != "")]

    # Generate the answers
    # Parallelize this? pytorch Dataset?
    for model, ls_client in evaluators:
        if model.display_name not in config.models_to_evaluate:
            continue
        logging.info("Evaluating model %s", model.display_name)

        # Name of the output column with responses
        output_column = f"{model.display_name}_answers"

        generate_answer_func = partial(ls_client.get_answer, skip_cache=False)

        qna_df[output_column] = qna_df[_QUESTION_COL].progress_apply(  # type:  ignore
            generate_answer_func
        )

    # Save result
    qna_df.to_json(output_filename)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pylint: disable=E1120

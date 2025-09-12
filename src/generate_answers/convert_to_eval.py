"""Convert generate_answers format to run with evaluation."""

import json
import os
import sys

import click
import yaml


@click.command(
    help="""
    Convert generated answers format to the input format of the evaluation.
    """,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "show_default": True,
    },
)
@click.option(
    "-c",
    "--config-filename",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the base YAML config file.",
)
@click.option(
    "-i",
    "--input-filename",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the input JSON file containing answers.",
)
@click.option(
    "-o",
    "--output-filename",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Path where the generated evaluation YAML will be saved.",
)
@click.option(
    "-f",
    "--force-overwrite",
    is_flag=True,
    default=False,
    help="Overwrite output file if it already exists.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Increase the logging level to DEBUG.",
)
def main(  # pylint: disable=R0913,R0917,R0914
    config_filename: str,
    input_filename: str,
    output_filename: str,
    force_overwrite: bool,
    verbose: bool,
) -> int:
    """Convert model answers from JSON into evaluation-ready YAML format."""

    def log(msg: str) -> None:
        if verbose:
            click.echo(f"[DEBUG] {msg}")

    log(f"Loading base config from {config_filename}")
    with open(config_filename, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    log(f"Loading input JSON data from {input_filename}")
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    log("Building turns...")
    turns = [
        {
            "turn_id": data["id"][str(idx)],
            "query": data["question"][str(idx)],
            "response": data["openai-4o-mini_answers"][str(idx)],
        }
        for idx in data["id"]
    ]

    final_config = [dict(base_config, turns=turns)]

    if os.path.exists(output_filename) and not force_overwrite:
        click.echo(
            f"Error: Output file {output_filename} already exists. "
            "Use --force-overwrite to overwrite.",
            err=True,
        )
        sys.exit(1)

    log(f"Writing output YAML to {output_filename}")
    with open(output_filename, "w", encoding="utf-8") as f:
        yaml.dump(final_config, f, sort_keys=False, default_flow_style=False)

    click.echo(f"✅ YAML file generated at {output_filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pylint: disable=E1120

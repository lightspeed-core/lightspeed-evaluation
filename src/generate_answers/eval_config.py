"""Evaluation config classes and helpers."""

import yaml
from pydantic import BaseModel


class EvalModel(BaseModel):
    """Model configuration."""

    display_name: str
    provider: str
    model: str


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    lightspeed_url: str
    models: list[EvalModel]
    models_to_evaluate: set[str]


def read_eval_config(filename: str) -> EvalConfig:
    """Read yaml config file."""
    with open(filename, encoding="utf8") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return EvalConfig.model_validate(config_dict)

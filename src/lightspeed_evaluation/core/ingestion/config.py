"""Configuration models for trace ingestion (curate).

The ingestion config is edited by whoever sets up a product's trace pull (e.g. a
data scientist). It declares *where* to fetch traces from and, declaratively,
*which field* of each trace carries each signal and *how* to map its raw values.

Signal and turn fields are addressed with `JMESPath <https://jmespath.org>`_
expressions evaluated against the raw trace payload (which includes the trace's
``observations`` and ``scores`` lists), so arbitrarily nested or filtered lookups
are expressible without code changes.
"""

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Bucket = Literal["positive", "negative", "edge", "unconfirmed"]


class ValueSpec(BaseModel):
    """Interpretation of one raw signal value: polarity, strength, type.

    ``strength`` accepts ``weak`` (0.5), ``strong`` (1.0), or a float in [0, 1].
    ``type`` is free-form (e.g. ``inferred`` / ``deterministic`` / ``human``) and
    is recorded for analysis/review, not used in bucketing.
    """

    model_config = ConfigDict(extra="forbid")

    polarity: Literal["positive", "negative", "neutral"] = Field(
        default="neutral", description="Whether this value indicates good/bad/neither"
    )
    strength: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Vote weight (weak=0.5, strong=1.0)"
    )
    type: Optional[str] = Field(
        default=None, description="Signal type, recorded for analysis (e.g. inferred)"
    )

    @field_validator("strength", mode="before")
    @classmethod
    def _coerce_strength(cls, v: Any) -> Any:
        """Accept 'weak'/'strong' keywords alongside a numeric weight."""
        if isinstance(v, str):
            key = v.strip().lower()
            if key == "weak":
                return 0.5
            if key == "strong":
                return 1.0
            try:
                return float(key)
            except ValueError as e:
                raise ValueError(
                    "strength must be 'weak', 'strong', or a float in [0, 1]"
                ) from e
        return v


class SignalSpec(BaseModel):
    """How to extract and interpret one signal from a raw trace payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        description="JMESPath expression evaluated against the raw trace payload",
    )
    map: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional raw-value -> mapped-value remap (keys compared as strings)",
    )
    role: Optional[str] = Field(
        default=None, description="Signal source role (e.g. human / model / mechanical)"
    )
    values: dict[str, ValueSpec] = Field(
        default_factory=dict,
        description="Interpretation per raw value; unlisted values are treated as neutral",
    )


class TurnSpec(BaseModel):
    """How turns are reconstructed from LangGraph observations.

    One trace == one turn. Turn content (query / response / tool_calls /
    contexts) is derived from the root observation's cumulative ``messages``
    (diffed per turn); only the observation/field names are configurable here.
    Signals are extracted separately via :class:`SignalSpec`.
    """

    model_config = ConfigDict(extra="forbid")

    root_observation: str = Field(
        default="LangGraph",
        description="Name of the root observation whose output holds the message list",
    )
    context_tools: list[str] = Field(
        default_factory=lambda: ["search_knowledge_base", "hydra_lexical"],
        description="Tool/observation names whose output is treated as RAG context",
    )
    context_cap: int = Field(
        default=2000, ge=0, description="Max chars kept per context string (0 = no cap)"
    )


class QuerySpec(BaseModel):
    """How the API fetch is shaped (window, pagination, field/observation narrowing).

    Groups all query-shaping parameters for the ``api`` source in one place. These
    have no equivalent in the shared ``trace_sources`` connection registry; they
    are specific to the historical batch pull.
    """

    model_config = ConfigDict(extra="forbid")

    window: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional {since, until} ISO datetimes bounding the fetch",
    )
    limit: Optional[int] = Field(
        default=None, ge=1, description="Max number of observations to fetch"
    )
    page_size: int = Field(
        default=100, ge=1, le=1000, description="Observations per API page"
    )
    fields: str = Field(
        default="core,basic,io,metadata,usage",
        description="Langfuse v2 observations 'fields' selector (needs io+metadata)",
    )
    observation_types: Optional[str] = Field(
        default=None,
        description="Optional comma-separated observation type filter (e.g. 'SPAN')",
    )
    observation_names: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional narrowing: fetch only observations with these names (one "
            "server-side name-filtered pass each), instead of every observation. "
            "Set it to the root observation plus the observations your signals read "
            "(e.g. [LangGraph, supervisor]) to pull ~2 records/turn over the whole "
            "window instead of ~40. Leave unset for the generic full fetch."
        ),
    )


class SourceConfig(BaseModel):
    """Where to obtain raw observation records from.

    Two interchangeable sources yield the same Langfuse observation records:

    * ``type: api`` — fetch live from ``GET /api/public/v2/observations``. The
      host comes from ``host`` here or the ``LANGFUSE_HOST`` env var; credentials
      always come from the ``LANGFUSE_*`` env vars, never stored in this file.
      Fetch shaping lives under ``query``.
    * ``type: file`` — read a Langfuse export from ``path`` (``.json`` list of
      observations, or ``.csv``).
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="api", description="Source type: 'api' or 'file'")

    # --- api source --------------------------------------------------------
    host: Optional[str] = Field(
        default=None,
        description=(
            "Langfuse host for the api source; defaults to the LANGFUSE_HOST env "
            "var. Credentials always come from LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY, never stored here."
        ),
    )
    query: QuerySpec = Field(
        default_factory=QuerySpec,
        description="API fetch shaping (window, pagination, field/observation narrowing)",
    )

    # --- file source -------------------------------------------------------
    path: Optional[str] = Field(
        default=None, description="Path to an exported .json/.csv (type: file)"
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        """Restrict the source type to the supported values."""
        if v not in ("api", "file"):
            raise ValueError("source.type must be 'api' or 'file'")
        return v

    @model_validator(mode="after")
    def _require_fields_per_type(self) -> "SourceConfig":
        """A file source needs a path (an api source needs no required fields)."""
        if self.type == "file" and not self.path:
            raise ValueError("source.path is required when source.type is 'file'")
        return self


class IngestConfig(BaseModel):
    """Top-level trace-ingestion configuration."""

    model_config = ConfigDict(extra="forbid")

    source: SourceConfig = Field(default_factory=SourceConfig)
    turn: TurnSpec = Field(default_factory=TurnSpec)
    signals: dict[str, SignalSpec] = Field(
        ...,
        min_length=1,
        description="Declared signals: name -> extraction spec",
    )


def load_ingest_config(path: str) -> IngestConfig:
    """Load and validate an ingestion config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Ingest config at {path} must be a YAML mapping")
    return IngestConfig.model_validate(data)


class OverrideSpec(BaseModel):
    """An optional curate override that pins a specific combination.

    ``match`` keys on ``eval`` (pass/fail) and/or signal names -> expected raw
    value; an override whose match fully applies wins over the algorithm.
    """

    model_config = ConfigDict(extra="forbid")

    match: dict[str, str] = Field(
        ..., description="Keys: 'eval' and/or signal names -> expected raw value"
    )
    bucket: Optional[Bucket] = Field(
        default=None, description="Override bucket (positive/negative/edge/unconfirmed)"
    )
    review: Optional[bool] = Field(
        default=None, description="Override the needs-review flag"
    )
    confidence: Optional[float] = Field(
        default=None, description="Override the computed confidence"
    )
    note: str = Field(default="", description="Explanatory note recorded on the label")


class CurateConfig(BaseModel):
    """Optional curate config holding overrides (curate.yaml)."""

    model_config = ConfigDict(extra="forbid")

    overrides: list[OverrideSpec] = Field(default_factory=list)


def load_curate_config(path: str) -> CurateConfig:
    """Load an optional curate config (overrides) from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Curate config at {path} must be a YAML mapping")
    return CurateConfig.model_validate(data)

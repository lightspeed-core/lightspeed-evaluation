"""Trace ingestion for curate: observations -> evaluation-data YAML.

Two interchangeable sources yield the same Langfuse observation records:

* ``type: api``  — the Langfuse SDK's observations endpoint (paginated).
* ``type: file`` — a Langfuse export (``.json`` list of observations, or ``.csv``).

Both are converted, via the config-driven extractor, into an evaluation-data YAML
whose turns carry ``signals`` (e.g. ``exit_reason``), ready for ``lightspeed-eval``
and then ``dataset-curator``.
"""

import csv
import importlib
import importlib.util
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from lightspeed_evaluation.core.ingestion.config import (
    IngestConfig,
    QuerySpec,
    SourceConfig,
    load_ingest_config,
)
from lightspeed_evaluation.core.ingestion.extractor import build_conversations

logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)

_IO_FIELDS = ("input", "output", "metadata")


def _require_env(name: str) -> str:
    """Read a required environment variable or raise a clear error."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Environment variable {name!r} is required for API ingestion")
    return value


def _clean_host(raw: str) -> str:
    """Strip stray/non-URL characters from a host value (defensive)."""
    return re.sub(r"[^A-Za-z0-9:._/-]", "", raw).rstrip("/")


def _maybe_json(value: Any) -> Any:
    """Parse a JSON string into an object; leave non-JSON values untouched."""
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                return value
    return value


def _parse_io(observation: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an observation with input/output/metadata JSON-parsed."""
    parsed = dict(observation)
    for field in _IO_FIELDS:
        if field in parsed:
            parsed[field] = _maybe_json(parsed[field])
    return parsed


def _build_langfuse_client(source: SourceConfig) -> Any:
    """Build a Langfuse SDK client for an api source.

    Host comes from ``source.host`` or the ``LANGFUSE_HOST`` env var; credentials
    always come from ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``.
    """
    if importlib.util.find_spec("langfuse") is None:
        raise ImportError(
            "API ingestion requires the 'langfuse' extra: "
            "pip install 'lightspeed-evaluation[langfuse]'"
        )
    host = source.host or os.environ.get("LANGFUSE_HOST")
    if not host:
        raise ValueError(
            "Langfuse host is not set: add `host` to the api source or set LANGFUSE_HOST"
        )
    langfuse_mod = importlib.import_module("langfuse")
    return langfuse_mod.Langfuse(
        host=_clean_host(host),
        public_key=_require_env("LANGFUSE_PUBLIC_KEY"),
        secret_key=_require_env("LANGFUSE_SECRET_KEY"),
    )


def _window_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 window bound into a datetime (the SDK wants datetimes)."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _split_types(observation_types: Optional[str]) -> list[Optional[str]]:
    """Split a comma-separated ``observation_types`` into individual enum values.

    The v2 API's ``type`` is a single enum, so multiple types are fetched one per
    value rather than sent as one comma-joined string. Empty -> ``[None]`` (all).
    """
    if not observation_types:
        return [None]
    return [t.strip() for t in observation_types.split(",") if t.strip()] or [None]


def _fetch_page(
    client: Any,
    query: QuerySpec,
    selector: tuple[Optional[str], Optional[str]],
    into: list[dict[str, Any]],
) -> bool:
    """Page one (name, type) selection into ``into``; False once ``limit`` is hit."""
    window = query.window or {}
    name, obs_type = selector
    cursor: Optional[str] = None
    while True:
        resp = client.api.observations.get_many(
            fields=query.fields,
            limit=query.page_size,
            cursor=cursor,
            name=name,
            type=obs_type,
            from_start_time=_window_dt(window.get("since")),
            to_start_time=_window_dt(window.get("until")),
        )
        rows = [json.loads(o.json(by_alias=True)) for o in resp.data]
        into.extend(_parse_io(o) for o in rows)
        if query.limit is not None and len(into) >= query.limit:
            return False
        next_cursor = resp.meta.cursor if resp.meta else None
        # Langfuse may return an empty page with a continuation cursor, so keep
        # going while the cursor advances; stop only when it is gone or unchanged.
        if not next_cursor or next_cursor == cursor:
            return True
        cursor = next_cursor


def fetch_api(client: Any, source: SourceConfig) -> list[dict[str, Any]]:
    """Fetch observations via the Langfuse SDK observations endpoint.

    When ``source.query.observation_names`` is set, each name is fetched with its own
    server-side ``name`` filter (narrowing the pull to the records turns/signals
    actually need); otherwise every observation in the window is fetched. Multiple
    ``observation_types`` are each fetched separately (the API's ``type`` is a
    single enum).
    """
    query = source.query
    observations: list[dict[str, Any]] = []
    for name in query.observation_names or [None]:
        done = False
        for obs_type in _split_types(query.observation_types):
            if not _fetch_page(client, query, (name, obs_type), observations):
                done = True
                break
        if done:
            break
    if query.limit is not None:
        return observations[: query.limit]
    return observations


def _read_file_rows(path: Path) -> list[dict[str, Any]]:
    """Read observation rows from a .json or .csv export."""
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("data", [])
    return data or []


def fetch_file(source: SourceConfig) -> list[dict[str, Any]]:
    """Read observations from an exported .json/.csv file."""
    if not source.path:
        raise ValueError("source.path is required for file ingestion")
    rows = _read_file_rows(Path(source.path))
    return [_parse_io(o) for o in rows]


def _fetch(source: SourceConfig) -> list[dict[str, Any]]:
    """Dispatch to the configured source fetcher."""
    if source.type == "file":
        return fetch_file(source)
    client = _build_langfuse_client(source)
    return fetch_api(client, source)


def ingest(config_path: str, output_path: str) -> dict[str, Any]:
    """Run ingestion: fetch observations, build eval-data YAML, write to disk.

    For an ``api`` source, the Langfuse host comes from ``source.host`` or the
    ``LANGFUSE_HOST`` env var and credentials from the ``LANGFUSE_*`` env vars.
    Returns a small summary dict (counts and output path).
    """
    config: IngestConfig = load_ingest_config(config_path)
    observations = _fetch(config.source)
    conversations = build_conversations(observations, config)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(
            conversations,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

    turn_count = sum(len(c["turns"]) for c in conversations)
    logger.info(
        "Ingested %d observations -> %d conversations, %d turns -> %s",
        len(observations),
        len(conversations),
        turn_count,
        out,
    )
    return {
        "observations": len(observations),
        "conversations": len(conversations),
        "turns": turn_count,
        "output_path": str(out),
    }

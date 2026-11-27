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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import tqdm
import yaml

from lightspeed_evaluation.core.ingestion.config import (
    IngestConfig,
    QuerySpec,
    SourceConfig,
    load_ingest_config,
)
from lightspeed_evaluation.core.ingestion.extractor import build_conversations
from lightspeed_evaluation.core.models.trace import Trace
from lightspeed_evaluation.core.models.trace_config import LangfuseTraceSourceConfig
from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.core.tracing import trace_from_langfuse
from lightspeed_evaluation.core.tracing._helpers import _as_datetime

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


def _build_langfuse_client(connection: LangfuseTraceSourceConfig) -> Any:
    """Build a Langfuse SDK client from the shared Langfuse trace-source config.

    Host comes from ``connection.host`` or the ``LANGFUSE_HOST`` env var;
    credentials always come from ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``.
    """
    if importlib.util.find_spec("langfuse") is None:
        raise ImportError(
            "API ingestion requires the 'langfuse' extra: "
            "pip install 'lightspeed-evaluation[langfuse]'"
        )
    host = connection.host or os.environ.get("LANGFUSE_HOST")
    if not host:
        raise ValueError("Langfuse host is not set: set LANGFUSE_HOST")
    langfuse_mod = importlib.import_module("langfuse")
    # base_url (not the deprecated host=) so LANGFUSE_BASE_URL cannot override it.
    return langfuse_mod.Langfuse(
        base_url=_clean_host(host),
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


class _WindowProgress:
    """Console progress bar showing how far a fetch has moved through its window.

    Progress is the share of the ``since``..``until`` span covered so far, read
    from the start times of fetched observations. Pages can arrive newest-first
    or oldest-first, so the direction is inferred from the first page. Without
    a window the bar only counts observations.
    """

    def __init__(self, label: str, window: dict[str, str]):
        """Open a bar for one fetch selection over ``window``."""
        self._since = _as_datetime(window.get("since"))
        until = _as_datetime(window.get("until"))
        self._until = until or (datetime.now(UTC) if self._since else None)
        self._span = (
            (self._until - self._since).total_seconds()
            if self._since and self._until
            else 0.0
        )
        self._descending: Optional[bool] = None
        self._count = 0
        self._bar = tqdm.tqdm(
            total=100 if self._span > 0 else None,
            desc=label,
            unit="%" if self._span > 0 else " obs",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {elapsed}{postfix}"
                if self._span > 0
                else "{desc}: {elapsed}{postfix}"
            ),
        )

    def percent(self, times: list[datetime]) -> float:
        """Return window coverage (0-100) after seeing ``times`` (in page order)."""
        if not times or self._span <= 0 or not self._since or not self._until:
            return 0.0
        if self._descending is None:
            self._descending = times[0] >= times[-1]
        covered = (
            (self._until - min(times))
            if self._descending
            else (max(times) - self._since)
        ).total_seconds()
        return max(0.0, min(100.0, 100.0 * covered / self._span))

    def update(self, rows: list[dict[str, Any]]) -> None:
        """Advance the bar with one fetched page of observations."""
        self._count += len(rows)
        times = [t for t in (_as_datetime(_start(r)) for r in rows) if t]
        if self._span > 0 and times:
            self._bar.n = max(self._bar.n, self.percent(times))
            reached = min(times) if self._descending else max(times)
            self._bar.set_postfix_str(
                f"{self._count} observations, at {reached:%Y-%m-%d %H:%M}"
            )
        else:
            self._bar.set_postfix_str(f"{self._count} observations")
        self._bar.refresh()

    def close(self) -> int:
        """Finish the bar at 100% and return the number of observations seen."""
        if self._span > 0:
            self._bar.n = 100
        self._bar.close()
        return self._count


def _fetch_page(
    client: Any,
    query: QuerySpec,
    selector: tuple[Optional[str], Optional[str]],
    into: list[dict[str, Any]],
) -> None:
    """Page one (name, type) selection fully into ``into``."""
    window = query.window or {}
    name, obs_type = selector
    label = "/".join(part for part in (name, obs_type) if part) or "observations"
    progress = _WindowProgress(f"langfuse · {label}", window)
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
        progress.update(rows)
        next_cursor = resp.meta.cursor if resp.meta else None
        # Langfuse may return an empty page with a continuation cursor, so keep
        # going while the cursor advances; stop only when it is gone or unchanged.
        if not next_cursor or next_cursor == cursor:
            logger.info(
                "Fetched %d %s observations from langfuse", progress.close(), label
            )
            return
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
        for obs_type in _split_types(query.observation_types):
            _fetch_page(client, query, (name, obs_type), observations)
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
    client = _build_langfuse_client(LangfuseTraceSourceConfig())
    return fetch_api(client, source)


def _start(observation: dict[str, Any]) -> str:
    """Return the observation's raw start time ("" when missing)."""
    return str(observation.get("startTime") or observation.get("start_time") or "")


_EPOCH = datetime.min.replace(tzinfo=UTC)


def _start_instant(observation: dict[str, Any]) -> datetime:
    """Sort key: start time as a UTC instant, so mixed formats order correctly."""
    return _as_datetime(_start(observation)) or _EPOCH


def traces_from_observations(observations: list[dict[str, Any]]) -> list[Trace]:
    """Group flat Langfuse observations by trace and normalize each into a Trace.

    Observations are ordered by start time within a trace and traces by their
    first observation. Observations without a trace id are dropped; missing
    observation ids are synthesized and duplicates are dropped, since the
    normalized model requires unique span ids. A trace that fails normalization
    is skipped with a warning rather than aborting the whole ingest.
    """
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        trace_id = obs.get("traceId") or obs.get("trace_id")
        if trace_id:
            by_trace.setdefault(str(trace_id), []).append(obs)

    payloads: list[dict[str, Any]] = []
    for trace_id, obs_list in by_trace.items():
        obs_list.sort(key=_start_instant)
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx, obs in enumerate(obs_list):
            obs_id = str(obs.get("id") or f"{trace_id}:{idx}")
            if obs_id not in seen:
                seen.add(obs_id)
                unique.append({**obs, "id": obs_id})
        payloads.append(
            {
                "id": trace_id,
                "sessionId": next(
                    (
                        o.get("sessionId") or o.get("session_id")
                        for o in unique
                        if o.get("sessionId") or o.get("session_id")
                    ),
                    None,
                ),
                "startTime": _start(unique[0]) or None,
                "observations": unique,
            }
        )
    payloads.sort(key=lambda p: _start_instant(p["observations"][0]))

    traces: list[Trace] = []
    for payload in payloads:
        try:
            traces.append(trace_from_langfuse(payload))
        except (DataValidationError, ValueError) as e:
            logger.warning("Skipping trace %s: %s", payload["id"], e)
    return traces


def ingest(config_path: str, output_path: str) -> dict[str, Any]:
    """Run ingestion: fetch observations, build eval-data YAML, write to disk.

    For an ``api`` source, the Langfuse host comes from the ``LANGFUSE_HOST`` env
    var and credentials from the ``LANGFUSE_*`` env vars. Returns a small summary
    dict (counts and output path).
    """
    config: IngestConfig = load_ingest_config(config_path)
    observations = _fetch(config.source)
    traces = traces_from_observations(observations)
    logger.info(
        "Normalized %d observations into %d traces", len(observations), len(traces)
    )
    conversations = build_conversations(traces, config)

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
        "traces": len(traces),
        "conversations": len(conversations),
        "turns": turn_count,
        "output_path": str(out),
    }

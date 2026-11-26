"""Tests for ingestion config loading and the ingest orchestration."""

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.ingestion import (
    fetch_file,
    ingest,
    load_ingest_config,
)
from lightspeed_evaluation.core.ingestion.config import SourceConfig
from lightspeed_evaluation.core.ingestion.langfuse_ingest import (
    _build_langfuse_client,
    _split_types,
)


def _api_source(host: str | None = None) -> SourceConfig:
    """Build an api SourceConfig, optionally with an explicit host."""
    return SourceConfig(type="api", host=host)


def _write_config(path: Path, export_path: Path) -> None:
    path.write_text(textwrap.dedent(f"""
            source:
              type: file
              path: {export_path}
            turn:
              root_observation: LangGraph
            signals:
              exit_reason:
                query: "observations[?name=='supervisor'].output.intent.exit_reason | [-1]"
                role: model
                values:
                  resolved: {{polarity: positive, strength: weak, type: inferred}}
            """))


def _export_rows() -> list:
    """Two observations for one trace: a LangGraph root and a supervisor."""
    msgs = [
        {"type": "human", "content": "pods crashlooping"},
        {"type": "ai", "content": "restart it"},
    ]
    return [
        {
            "traceId": "tr1",
            "sessionId": "s1",
            "name": "LangGraph",
            "type": "CHAIN",
            "startTime": "1",
            "output": json.dumps({"messages": msgs}),  # JSON string, as exported
        },
        {
            "traceId": "tr1",
            "sessionId": "s1",
            "name": "supervisor",
            "type": "CHAIN",
            "startTime": "2",
            "output": json.dumps({"intent": {"exit_reason": "resolved"}}),
        },
    ]


def test_load_ingest_config_parses_source_and_signals(tmp_path: Path) -> None:
    """A valid config loads into typed models."""
    export = tmp_path / "export.json"
    export.write_text("[]")
    cfg = tmp_path / "ingest.yaml"
    _write_config(cfg, export)
    config = load_ingest_config(str(cfg))
    assert config.source.type == "file"
    assert set(config.signals) == {"exit_reason"}


def test_file_source_requires_path(tmp_path: Path) -> None:
    """A file source without a path is rejected."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("source: {type: file}\nsignals: {s: {query: 'x'}}\n")
    with pytest.raises(ValueError):
        load_ingest_config(str(cfg))


def test_source_type_must_be_valid(tmp_path: Path) -> None:
    """An unknown source type is rejected."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("source: {type: ftp}\nsignals: {s: {query: 'x'}}\n")
    with pytest.raises(ValueError):
        load_ingest_config(str(cfg))


def test_api_source_loads_without_host(tmp_path: Path) -> None:
    """An api source needs no host at config load (resolved later from env)."""
    cfg = tmp_path / "ok.yaml"
    cfg.write_text("source: {type: api}\nsignals: {s: {query: 'x'}}\n")
    config = load_ingest_config(str(cfg))
    assert config.source.type == "api"
    assert config.source.host is None


_INGEST = "lightspeed_evaluation.core.ingestion.langfuse_ingest"


def _fake_langfuse(mocker: MockerFixture) -> Any:
    """Stub the lazy langfuse import so tests run without the langfuse extra."""
    mocker.patch(f"{_INGEST}.importlib.util.find_spec", return_value=object())
    fake_langfuse = mocker.MagicMock()
    mocker.patch(
        f"{_INGEST}.importlib.import_module",
        return_value=mocker.MagicMock(Langfuse=fake_langfuse),
    )
    return fake_langfuse


def test_build_client_uses_config_host(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """An explicit source.host wins and credentials come from env."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sec")
    fake = _fake_langfuse(mocker)
    _build_langfuse_client(_api_source("https://lf.example.com"))
    fake.assert_called_once_with(
        host="https://lf.example.com", public_key="pub", secret_key="sec"
    )


def test_build_client_falls_back_to_env_host(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """With no source.host, LANGFUSE_HOST is used."""
    monkeypatch.setenv("LANGFUSE_HOST", "https://env.example.com")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sec")
    fake = _fake_langfuse(mocker)
    _build_langfuse_client(_api_source())
    assert fake.call_args.kwargs["host"] == "https://env.example.com"


def test_build_client_cleans_host(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """Stray non-URL characters in the host are stripped before use."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sec")
    fake = _fake_langfuse(mocker)
    _build_langfuse_client(_api_source("https://lf.example.com/ø"))
    assert fake.call_args.kwargs["host"] == "https://lf.example.com"


def test_build_client_requires_a_host(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """With neither source.host nor LANGFUSE_HOST, resolution fails clearly."""
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sec")
    _fake_langfuse(mocker)
    with pytest.raises(ValueError, match="host is not set"):
        _build_langfuse_client(_api_source())


def test_fetch_file_parses_io_json_strings(tmp_path: Path) -> None:
    """Exported output/input JSON strings are parsed into objects."""
    export = tmp_path / "export.json"
    export.write_text(json.dumps(_export_rows()))
    obs = fetch_file(SourceConfig(type="file", path=str(export)))
    assert isinstance(obs[0]["output"], dict)
    assert obs[0]["output"]["messages"][0]["content"] == "pods crashlooping"


def test_fetch_file_reads_csv(tmp_path: Path) -> None:
    """A .csv export is read into observation rows."""
    export = tmp_path / "export.csv"
    export.write_text(
        "traceId,sessionId,name,type,startTime,output\n"
        'tr1,s1,supervisor,CHAIN,1,"{""intent"": {""exit_reason"": ""resolved""}}"\n'
    )
    obs = fetch_file(SourceConfig(type="file", path=str(export)))
    assert obs[0]["output"]["intent"]["exit_reason"] == "resolved"


def test_ingest_file_source_end_to_end(tmp_path: Path) -> None:
    """ingest() reads a file export and writes eval-data with signals."""
    export = tmp_path / "export.json"
    export.write_text(json.dumps(_export_rows()))
    cfg = tmp_path / "ingest.yaml"
    _write_config(cfg, export)
    out = tmp_path / "eval_data.yaml"

    summary = ingest(str(cfg), str(out))

    assert summary["conversations"] == 1
    assert summary["turns"] == 1
    data = yaml.safe_load(out.read_text())
    turn = data[0]["turns"][0]
    assert turn["query"] == "pods crashlooping"
    assert turn["response"] == "restart it"
    assert turn["signals"]["exit_reason"] == {
        "value": "resolved",
        "role": "model",
        "polarity": "positive",
        "strength": 0.5,
        "type": "inferred",
    }


def test_split_types_single_and_multi() -> None:
    """Comma-separated observation types split into individual enum values."""
    assert _split_types(None) == [None]
    assert _split_types("") == [None]
    assert _split_types("CHAIN") == ["CHAIN"]
    assert _split_types("SPAN, CHAIN") == ["SPAN", "CHAIN"]

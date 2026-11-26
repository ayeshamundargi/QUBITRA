"""Smoke tests verifying offline configuration, core module imports, and CLI wiring."""

from __future__ import annotations

import importlib
import json
import os
import sys

import cli
from core import config


CORE_MODULES: tuple[str, ...] = (
    "core",
    "core.config",
    "core.schemas",
    "core.forensics",
    "core.ingest",
    "core.extractor",
    "core.graph_engine",
    "core.reasoner",
)


def test_config_thresholds_and_offline_defaults() -> None:
    """Verify exact deterministic thresholds, offline flags, and local model names."""
    assert config.ABSTENTION_THRESHOLD == 0.65
    assert config.NLI_CONTRADICTION_THRESHOLD == 0.80
    assert config.ELA_TAMPER_THRESHOLD == 25.0
    assert config.HF_HUB_OFFLINE == 1
    assert os.environ.get("HF_HUB_OFFLINE") == "1"

    assert config.settings.abstention_threshold == 0.65
    assert config.settings.nli_contradiction_threshold == 0.80
    assert config.settings.ela_tamper_threshold == 25.0
    assert config.settings.hf_hub_offline == 1
    assert config.settings.ollama_base_url == "http://127.0.0.1:11434"
    assert config.settings.llm_model == "qwen2.5:3b"
    assert config.settings.vlm_model == "qwen2.5vl:7b"
    assert config.settings.embed_model == "BAAI/bge-m3"
    assert config.settings.nli_model == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def test_core_modules_import_without_ui_dependencies() -> None:
    """Verify every core module imports cleanly in isolation without UI frameworks."""
    import subprocess

    for module_name in CORE_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None

    script = (
        "import importlib, sys; "
        f"mods = {list(CORE_MODULES)!r}; "
        "[importlib.import_module(m) for m in mods]; "
        "assert 'streamlit' not in sys.modules, 'streamlit leaked into core'; "
        "assert 'pyvis' not in sys.modules, 'pyvis leaked into core'"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_eval_cases_initialized_and_valid() -> None:
    """Verify eval/test_cases.json exists and contains 5 valid benchmark cases."""
    raw = config.EVAL_CASES_PATH.read_text(encoding="utf-8")
    cases = json.loads(raw)
    assert isinstance(cases, list)
    assert len(cases) == 5


def test_cli_placeholder_subcommands(capsys) -> None:
    """Verify ingest, query, and eval CLI subcommands return status 0."""
    assert cli.main(["ingest"]) == 0
    assert json.loads(capsys.readouterr().out)["command"] == "ingest"

    assert cli.main(["query", "What is the approved cap?"]) == 0
    query_out = json.loads(capsys.readouterr().out)
    assert query_out["command"] == "query"
    assert query_out["question"] == "What is the approved cap?"

    assert cli.main(["eval"]) == 0
    assert json.loads(capsys.readouterr().out)["command"] == "eval"

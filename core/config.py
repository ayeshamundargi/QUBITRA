"""Centralized offline paths, local model names, and deterministic thresholds."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# Enforce strict offline HuggingFace execution before any HF model module loads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ABSTENTION_THRESHOLD: float = 0.65
NLI_CONTRADICTION_THRESHOLD: float = 0.80
ELA_TAMPER_THRESHOLD: float = 25.0
HF_HUB_OFFLINE: int = int(os.environ.get("HF_HUB_OFFLINE", "1"))

DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
SYNTHETIC_TRAPS_DIR: Path = DATA_DIR / "synthetic_traps"
STORAGE_DIR: Path = BASE_DIR / "storage"
CHROMA_DIR: Path = STORAGE_DIR / "chromadb"
GRAPH_PATH: Path = STORAGE_DIR / "evidence_graph.json"
EVAL_DIR: Path = BASE_DIR / "eval"
EVAL_CASES_PATH: Path = EVAL_DIR / "test_cases.json"


class Settings(BaseModel):
    """Validated runtime configuration for offline CERBERUS-X execution."""

    base_dir: Path = BASE_DIR
    data_raw_dir: Path = DATA_RAW_DIR
    synthetic_traps_dir: Path = SYNTHETIC_TRAPS_DIR
    storage_dir: Path = STORAGE_DIR
    chroma_dir: Path = CHROMA_DIR
    graph_path: Path = GRAPH_PATH
    eval_cases_path: Path = EVAL_CASES_PATH

    ollama_base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )
    )
    llm_model: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "qwen2.5:3b")
    )
    vlm_model: str = Field(
        default_factory=lambda: os.environ.get("VLM_MODEL", "qwen2.5vl:7b")
    )
    embed_model: str = Field(
        default_factory=lambda: os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    )
    nli_model: str = Field(
        default_factory=lambda: os.environ.get(
            "NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
    )
    hf_hub_offline: int = HF_HUB_OFFLINE

    abstention_threshold: float = ABSTENTION_THRESHOLD
    nli_contradiction_threshold: float = NLI_CONTRADICTION_THRESHOLD
    ela_tamper_threshold: float = ELA_TAMPER_THRESHOLD


settings: Settings = Settings()

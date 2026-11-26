# CERBERUS-X Project Rules & Invariants

CERBERUS-X is a 100% offline multimodal evidence graph and conflict-aware reasoning engine. Every contributor and agent working in this repository MUST uphold the invariants below without exception.

## 1. Strict Offline-Only Guarantee
- All inference, embeddings, NLI, OCR, storage, and graph reasoning MUST run locally (`Ollama` at `127.0.0.1:11434`, local HuggingFace weights with `HF_HUB_OFFLINE=1`, `NetworkX`, `ChromaDB`, `DuckDB`).
- NEVER import or call external cloud LLM/VLM/embedding SDKs or external HTTP APIs anywhere in the codebase (`openai`, `anthropic`, `google-genai`, `cohere`, remote endpoints, etc.).
- Centralize all local paths, model identifiers, and thresholds in `core/config.py`.

## 2. Backend-First Separation
- All core logic lives inside `core/` and communicates across pipeline boundaries via Pydantic v2 models defined in `core/schemas.py`.
- All core functionality MUST remain testable headlessly from `cli.py` and `pytest` without starting or importing any UI framework.
- `core/` MUST NEVER import `streamlit`, `pyvis`, or any presentation/UI module.

## 3. Mandatory Provenance
- Every extracted `Claim` MUST retain exact provenance: `source_id`, `modality`, and `location_ref` (e.g., page/bounding box, row/column, or timestamp/frame reference).
- Claims missing verifiable provenance MUST be rejected by Pydantic validation before entering the evidence graph.

## 4. Pre-LLM Security Sanitization Gate
- All security sanitization in `core/forensics.py` (hidden/invisible PDF text detection, Error Level Analysis `ELA` tampering checks, EXIF metadata inspection, and prompt-injection filtering) MUST run BEFORE any raw document content enters an LLM or VLM prompt.
- Flagged or untrusted artifacts must carry forensic metadata into trust propagation and never be silently treated as verified ground truth.

## 5. Deterministic Abstention Gate
- Abstention (`INSUFFICIENT_EVIDENCE`) MUST be enforced deterministically in Python inside `core/reasoner.py` (using `ABSTENTION_THRESHOLD` and retrieved evidence sufficiency checks).
- NEVER leave abstention solely to LLM prompt instructions.

## 6. Temporal Amendments vs. Contradictions
- The graph engine (`core/graph_engine.py`) MUST explicitly distinguish temporal/versioned amendments (`SUPERSEDES`) from true logical or numeric contradictions (`CONTRADICTS`).
- Superseded claims must not be falsely reported as active contradictions, and true contradictions must preserve both competing claims and their supporting provenance.

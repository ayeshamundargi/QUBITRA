# CERBERUS-X

**100% Offline Multimodal Forensic Intelligence & Conflict-Aware Evidence Graph Engine**

CERBERUS-X audits heterogeneous document sets (PDFs, CSV/Parquet tables, scanned documents, and JPEG/PNG images) completely offline. Instead of flattening raw documents into unverified RAG chunks, it runs pre-LLM security forensics, extracts provenance-bound atomic claims, distinguishes temporal legal amendments (`SUPERSEDES`) from true conflicts (`CONTRADICTS`), propagates cross-modal trust over a NetworkX evidence graph, and enforces a deterministic `0.65` (`65%`) abstention safety gate.

---

## Core Architecture (`core/`)

1. **Pre-LLM Forensics & Ingestion (`core/forensics.py`, `core/ingest.py`)**
   - Computes `SHA-256` document IDs (`src_<sha256[:12]>`) and inspects every input **before** LLM exposure:
     - **JPEG ELA Tamper Detection (`ELA_TAMPER_DETECTED`):** 32x32 block Error Level Analysis (`threshold = 25.0`).
     - **Hidden PDF Text (`HIDDEN_PDF_TEXT`):** White/near-white (`rgb=(255,255,255)`), tiny, or off-page span filtering via `PyMuPDF`.
     - **Prompt Injection Filtering (`PROMPT_INJECTION`):** Strips instruction-override clauses before chunk emission.
     - **Blur & Missing Field Checks (`LOW_LEGIBILITY_BLUR`, `MISSING_FIELD`):** Laplacian sharpness (`< 80.0`) caps OCR confidence; blank table cells are explicitly flagged `[MISSING]`.
   - Extracts bounding-boxed OCR chunks via offline `RapidOCR`, table rows via `Pandas`/`DuckDB`, and PDF blocks via `PyMuPDF`.

2. **Atomic Claim Extraction & 3-Tier Entity Resolution (`core/extractor.py`)**
   - Rejects any unsanitized chunk (`sanitized=False`); retains exact `LocationRef` (`page`, `row`, `column`, `bbox`, `char_span`).
   - Supports schema-constrained local `Ollama` (`qwen2.5:3b`) extraction and deterministic fallback parsing.
   - Resolves aliases across modalities via regex canonicalization, `RapidFuzz` clustering (plus optional offline `BAAI/bge-m3`), and optional local Ollama tie-breaking.

3. **Evidence Graph & Iterative Trust Propagation (`core/graph_engine.py`)**
   - Builds a NetworkX `MultiDiGraph` connecting `Source`, `Claim`, and `Entity` nodes via `ASSERTS`, `ABOUT`, `SUPPORTS`, `CONTRADICTS`, and `SUPERSEDES`.
   - Evaluates temporal amendments (`SUPERSEDES`) first so valid legal updates do not produce false contradiction penalties.
   - Iteratively propagates cross-modal corroboration boosts and contradiction penalties (`5` iterations), capping quarantined claims at `0.15`.

4. **Bounded Retrieval, Abstention Gate & UI (`core/reasoner.py`, `app.py`, `cli.py`)**
   - Performs 2-hop bounded BFS (`get_entity_subgraph`) to build an `EvidenceDossier`.
   - Deterministically enforces `ABSTENTION_THRESHOLD = 0.65` in Python (`VERIFIED`, `CONFLICT_RESOLVED`, `TAMPER_DETECTED`, `INSUFFICIENT_EVIDENCE`).
   - Local `Ollama` (`127.0.0.1:11434`) serves strictly as a prose scribe over verified `EvidenceDossier` JSON.

---

## Setup & Usage (Windows PowerShell)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Generate synthetic adversarial test traps (optional - already included in data/synthetic_traps/)
.venv\Scripts\python.exe scripts\generate_traps.py

# 3. Run the offline CLI pipeline & 5-case adversarial benchmark
.venv\Scripts\python.exe cli.py ingest --input data\synthetic_traps
.venv\Scripts\python.exe cli.py extract
.venv\Scripts\python.exe cli.py build-graph
.venv\Scripts\python.exe cli.py eval

# 4. Run the test suite (42 tests)
.venv\Scripts\pytest.exe -v

# 5. Launch the offline Streamlit Forensic Console
.venv\Scripts\streamlit.exe run app.py --server.port 8502
```

---

## Verified Benchmark Results (`eval/test_cases.json`)

- **Pytest Suite:** `42/42` passing (`tests/`)
- **Adversarial Eval (`cli.py eval`):** `5/5` (`100%` pass rate, `100%` injection block rate, `100%` conflict recall, `100%` citation accuracy, `100%` abstention accuracy)
  - `TC-01` (`INV-402` amount): `CONFLICT_RESOLVED` (`$12,000`, `0.935`) — quarantines spliced `$18,000` JPEG & injected PDF
  - `TC-02` (`INV-510` amount): `VERIFIED` (`$5,200`, `0.7125`) — amendment `SUPERSEDES` prior `$4,500` records
  - `TC-03` (`WIRE-Q4` approver): `INSUFFICIENT_EVIDENCE` (`0.0`) — blank approver + blurry scan `< 0.65` gate
  - `TC-04` (`INV-402` vendor): `VERIFIED` (`Apex Logistics`, `0.935`)
  - `TC-05` (`INV-402` vendor notes): `TAMPER_DETECTED` (`0.0`) — blocked pre-LLM

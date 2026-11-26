"""Offline multimodal ingestion pipeline for CERBERUS-X.

Runs `core.forensics.analyze_source_with_context()` BEFORE extracting any text
or visual content, drops hidden PDF spans, strips prompt-injection payloads
via `sanitize_text()`, and emits validated `SourceDoc` and `ParsedChunk` models.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from pydantic import TypeAdapter
import pymupdf

from core.config import settings
from core.forensics import (
    SUPPORTED_EXTENSIONS,
    analyze_source_with_context,
    is_hidden_pdf_span,
    sanitize_text,
)
from core.schemas import LocationRef, Modality, ParsedChunk, SourceDoc

_OCR_ENGINE: Any = None


def _get_ocr_engine() -> Any:
    """Lazily instantiate the offline RapidOCR ONNX Runtime engine."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _parse_pdf(path: Path, source_doc: SourceDoc) -> list[ParsedChunk]:
    """Extract visible, non-hidden, sanitized text blocks from a PDF."""
    chunks: list[ParsedChunk] = []
    doc = pymupdf.open(str(path))
    try:
        chunk_idx = 0
        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                visible_lines: list[str] = []
                for line in block.get("lines", []):
                    visible_spans: list[str] = []
                    for span in line.get("spans", []):
                        hidden, _ = is_hidden_pdf_span(span, page.rect)
                        if hidden:
                            continue
                        span_text = str(span.get("text", ""))
                        if span_text:
                            visible_spans.append(span_text)
                    joined_line = "".join(visible_spans).strip()
                    if joined_line:
                        visible_lines.append(joined_line)

                raw_block_text = "\n".join(visible_lines).strip()
                if not raw_block_text:
                    continue

                cleaned_text, _ = sanitize_text(raw_block_text)
                if not cleaned_text:
                    continue

                bbox_vals = tuple(float(v) for v in block["bbox"])
                chunk_idx += 1
                chunks.append(
                    ParsedChunk(
                        chunk_id=f"{source_doc.source_id}_pdf_{chunk_idx:04d}",
                        source_id=source_doc.source_id,
                        modality=Modality.PDF,
                        text=cleaned_text,
                        location_ref=LocationRef(
                            page=page_num,
                            bbox=(bbox_vals[0], bbox_vals[1], bbox_vals[2], bbox_vals[3]),
                            snippet=cleaned_text[:160],
                        ),
                        extraction_confidence=0.95,
                        sanitized=True,
                    )
                )
    finally:
        doc.close()
    return chunks


def _parse_table(path: Path, source_doc: SourceDoc) -> list[ParsedChunk]:
    """Parse structured CSV/TSV/Parquet tables row-by-row with explicit [MISSING] markers."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        con = duckdb.connect(database=":memory:")
        try:
            df = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
        finally:
            con.close()
        df = df.fillna("").astype(str)
    elif ext == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(path, dtype=str).fillna("")

    chunks: list[ParsedChunk] = []
    for idx, row in df.iterrows():
        row_num = int(idx) + 1
        parts: list[str] = []
        for col in df.columns:
            val = str(row[col]).strip()
            formatted_val = val if val else "[MISSING]"
            parts.append(f"{col}: {formatted_val}")

        raw_row_text = " | ".join(parts)
        cleaned_text, _ = sanitize_text(raw_row_text)
        if not cleaned_text:
            continue

        chunks.append(
            ParsedChunk(
                chunk_id=f"{source_doc.source_id}_row_{row_num:04d}",
                source_id=source_doc.source_id,
                modality=Modality.TABLE,
                text=cleaned_text,
                location_ref=LocationRef(
                    row=row_num,
                    column="ALL",
                    snippet=cleaned_text[:160],
                ),
                extraction_confidence=0.99,
                sanitized=True,
            )
        )
    return chunks


def _query_local_ollama_vlm(path: Path) -> str | None:
    """Optional offline Ollama VLM call to `127.0.0.1:11434`; degrades cleanly if down."""
    base_url = settings.ollama_base_url.rstrip("/")
    if not ("127.0.0.1" in base_url or "localhost" in base_url):
        raise ValueError(f"Offline guarantee violated by non-local Ollama URL: {base_url}")

    img_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    payload = json.dumps(
        {
            "model": settings.vlm_model,
            "prompt": (
                "Transcribe all visible key-value fields, totals, and signatures concisely. "
                "If any signature or field is illegible, explicitly state [ILLEGIBLE]."
            ),
            "images": [img_b64],
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url=f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("response", "")).strip() or None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _parse_image_or_scan(
    path: Path,
    source_doc: SourceDoc,
    legibility_score: float,
    use_vlm: bool = False,
) -> list[ParsedChunk]:
    """Extract bounding-boxed OCR chunks via offline RapidOCR, capped by legibility."""
    ocr = _get_ocr_engine()
    ocr_results, _ = ocr(str(path))
    chunks: list[ParsedChunk] = []

    if ocr_results:
        for idx, item in enumerate(ocr_results, start=1):
            poly, raw_text, raw_conf = item[0], str(item[1]), float(item[2])
            xs = [float(pt[0]) for pt in poly]
            ys = [float(pt[1]) for pt in poly]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            cleaned_text, _ = sanitize_text(raw_text)
            if not cleaned_text:
                continue

            capped_conf = round(min(max(0.0, raw_conf), legibility_score), 4)
            chunks.append(
                ParsedChunk(
                    chunk_id=f"{source_doc.source_id}_ocr_{idx:04d}",
                    source_id=source_doc.source_id,
                    modality=source_doc.modality,
                    text=cleaned_text,
                    location_ref=LocationRef(
                        bbox=bbox,
                        snippet=cleaned_text[:160],
                    ),
                    extraction_confidence=capped_conf,
                    sanitized=True,
                )
            )

    if use_vlm:
        vlm_summary = _query_local_ollama_vlm(path)
        if vlm_summary:
            cleaned_vlm, _ = sanitize_text(vlm_summary)
            if cleaned_vlm:
                from PIL import Image

                with Image.open(path) as img:
                    full_bbox = (0.0, 0.0, float(img.width), float(img.height))
                chunks.append(
                    ParsedChunk(
                        chunk_id=f"{source_doc.source_id}_vlm_0001",
                        source_id=source_doc.source_id,
                        modality=source_doc.modality,
                        text=cleaned_vlm,
                        location_ref=LocationRef(
                            bbox=full_bbox,
                            snippet=cleaned_vlm[:160],
                        ),
                        extraction_confidence=round(min(0.85, legibility_score), 4),
                        sanitized=True,
                    )
                )

    return chunks


def _parse_text_or_json(path: Path, source_doc: SourceDoc) -> list[ParsedChunk]:
    """Extract sanitized text/JSON chunks with exact character span provenance."""
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    if not raw_text.strip():
        return []

    chunks: list[ParsedChunk] = []
    cursor = 0
    chunk_idx = 0
    for segment in raw_text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(segment)
        cursor = end

        cleaned, _ = sanitize_text(segment)
        if not cleaned:
            continue

        chunk_idx += 1
        chunks.append(
            ParsedChunk(
                chunk_id=f"{source_doc.source_id}_txt_{chunk_idx:04d}",
                source_id=source_doc.source_id,
                modality=source_doc.modality,
                text=cleaned,
                location_ref=LocationRef(
                    char_span=(start, end),
                    snippet=cleaned[:160],
                ),
                extraction_confidence=0.98,
                sanitized=True,
            )
        )
    return chunks


def ingest_file(
    path: Path,
    use_vlm: bool = False,
) -> tuple[SourceDoc, list[ParsedChunk]]:
    """Run pre-LLM forensics first, then parse sanitized multimodal chunks."""
    resolved = path.resolve()
    source_doc, legibility_score = analyze_source_with_context(resolved)

    if source_doc.modality == Modality.PDF:
        chunks = _parse_pdf(resolved, source_doc)
    elif source_doc.modality == Modality.TABLE:
        chunks = _parse_table(resolved, source_doc)
    elif source_doc.modality in (Modality.IMAGE, Modality.SCAN):
        chunks = _parse_image_or_scan(
            resolved,
            source_doc,
            legibility_score=legibility_score,
            use_vlm=use_vlm,
        )
    else:
        chunks = _parse_text_or_json(resolved, source_doc)

    return source_doc, chunks


def ingest_directory(
    input_dir: Path,
    storage_dir: Path | None = None,
    use_vlm: bool = False,
) -> tuple[list[SourceDoc], list[ParsedChunk]]:
    """Ingest all supported files under `input_dir` and optionally persist JSON artifacts."""
    docs: list[SourceDoc] = []
    all_chunks: list[ParsedChunk] = []

    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    if input_dir.is_file():
        candidate_files = [input_dir]
    else:
        candidate_files = sorted(
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    for file_path in candidate_files:
        source_doc, chunks = ingest_file(file_path, use_vlm=use_vlm)
        docs.append(source_doc)
        all_chunks.extend(chunks)

    if storage_dir is not None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        docs_json = TypeAdapter(list[SourceDoc]).dump_json(docs, indent=2)
        chunks_json = TypeAdapter(list[ParsedChunk]).dump_json(all_chunks, indent=2)
        (storage_dir / "ingested_docs.json").write_bytes(docs_json)
        (storage_dir / "parsed_chunks.json").write_bytes(chunks_json)

    return docs, all_chunks

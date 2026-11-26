"""Pre-LLM document forensics and security sanitization gate.

All raw documents MUST pass through `analyze_source()` (and `sanitize_text()`
for any extracted text) before entering downstream extractors or local LLMs.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import ExifTags, Image
import pymupdf

from core.config import settings
from core.schemas import (
    ForensicFlag,
    ForensicSeverity,
    LocationRef,
    Modality,
    SourceDoc,
)

TRUST_PRIORS: dict[Modality, float] = {
    Modality.TABLE: 0.85,
    Modality.JSON: 0.85,
    Modality.PDF: 0.75,
    Modality.SCAN: 0.65,
    Modality.IMAGE: 0.65,
    Modality.TEXT: 0.60,
}

EXTENSION_TO_MODALITY: dict[str, Modality] = {
    ".pdf": Modality.PDF,
    ".csv": Modality.TABLE,
    ".tsv": Modality.TABLE,
    ".parquet": Modality.TABLE,
    ".xlsx": Modality.TABLE,
    ".xls": Modality.TABLE,
    ".json": Modality.JSON,
    ".jsonl": Modality.JSON,
    ".png": Modality.SCAN,
    ".tif": Modality.SCAN,
    ".tiff": Modality.SCAN,
    ".bmp": Modality.SCAN,
    ".jpg": Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".webp": Modality.IMAGE,
    ".txt": Modality.TEXT,
    ".md": Modality.TEXT,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(EXTENSION_TO_MODALITY.keys())

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"(?:disregard|forget|override)\s+(?:all\s+)?(?:above|previous|prior|instructions|verdict)", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bmark\s+status\s+verified\b", re.IGNORECASE),
    re.compile(r"\boverride\s+verdict\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+above\b", re.IGNORECASE),
    re.compile(r"report\s+invoice\s+.*\s+amount\s+as\b", re.IGNORECASE),
)

_ELA_SCALE: float = 20.0
_ELA_BLOCK_SIZE: int = 32
_BLUR_LAP_THRESHOLD: float = 80.0


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a local file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_modality(path: Path) -> Modality:
    """Map file extension to canonical CERBERUS-X Modality enum."""
    ext = path.suffix.lower()
    modality = EXTENSION_TO_MODALITY.get(ext)
    if modality is None:
        raise ValueError(f"Unsupported file extension '{ext}' for path: {path}")
    return modality


def scan_prompt_injection(
    text: str,
    location: LocationRef | None = None,
) -> list[ForensicFlag]:
    """Scan raw text for prompt-injection / instruction-override payloads."""
    if not text.strip():
        return []

    flags: list[ForensicFlag] = []
    seen_patterns: set[str] = set()
    for line_idx, line in enumerate(text.splitlines(), start=1):
        for pattern in INJECTION_PATTERNS:
            match = pattern.search(line)
            if match and pattern.pattern not in seen_patterns:
                seen_patterns.add(pattern.pattern)
                loc = location or LocationRef(
                    char_span=(match.start(), match.end()),
                    snippet=line.strip()[:120],
                )
                flags.append(
                    ForensicFlag(
                        code="PROMPT_INJECTION",
                        severity=ForensicSeverity.CRITICAL,
                        detail=(
                            f"Instruction-override pattern detected on line {line_idx}: "
                            f"{match.group(0)!r}"
                        ),
                        location=loc,
                    )
                )
    return flags


def sanitize_text(text: str) -> tuple[str, list[ForensicFlag]]:
    """Strip prompt-injection lines/clauses from text and return emitted flags."""
    if not text:
        return "", []

    emitted_flags: list[ForensicFlag] = []
    clean_lines: list[str] = []

    for line in text.splitlines():
        line_flags = scan_prompt_injection(line)
        if line_flags:
            emitted_flags.extend(line_flags)
            # Also verify sentence-level filtering in case legitimate prose shares a line
            kept_sentences: list[str] = []
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                if not any(p.search(sentence) for p in INJECTION_PATTERNS):
                    if sentence.strip():
                        kept_sentences.append(sentence.strip())
            if kept_sentences:
                # Only keep non-injection prefix if it wasn't an entire injection line
                joined = " ".join(kept_sentences)
                if not any(p.search(joined) for p in INJECTION_PATTERNS):
                    clean_lines.append(joined)
        else:
            clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()
    return cleaned, emitted_flags


def sanitize_prompt_content(raw_text: str) -> str:
    """Sanitize raw document text before any LLM/VLM exposure."""
    cleaned, _ = sanitize_text(raw_text)
    return cleaned


def is_hidden_pdf_span(span: dict[str, Any], page_rect: Any) -> tuple[bool, str]:
    """Detect hidden PDF text spans (white/near-white, tiny font, or off-page bbox)."""
    text = str(span.get("text", "")).strip()
    if not text:
        return False, ""

    color_int = int(span.get("color", 0))
    r, g, b = pymupdf.sRGB_to_rgb(color_int)
    if r >= 245 and g >= 245 and b >= 245:
        return True, f"white/near-white text color rgb=({r},{g},{b})"

    size = float(span.get("size", 12.0))
    if size <= 5.0:
        return True, f"tiny hidden font size={size:.2f}"

    bbox = span.get("bbox")
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        if (
            x1 <= 0.0
            or y1 <= 0.0
            or x0 >= float(page_rect.width)
            or y0 >= float(page_rect.height)
        ):
            return True, f"off-page bounding box {bbox}"

    return False, ""


def inspect_pdf_forensics(path: Path) -> tuple[list[ForensicFlag], dict[str, str]]:
    """Inspect a PDF for hidden text spans, suspicious metadata, and prompt injections."""
    flags: list[ForensicFlag] = []
    metadata: dict[str, str] = {}

    doc = pymupdf.open(str(path))
    try:
        raw_meta = doc.metadata or {}
        for key in ("creator", "producer", "modDate", "creationDate", "title", "author"):
            val = raw_meta.get(key)
            if val:
                metadata[key] = str(val)

        suspicious_tools = ("photoshop", "gimp", "illustrator", "inkscape")
        creator_producer = f"{metadata.get('creator', '')} {metadata.get('producer', '')}".lower()
        if any(tool in creator_producer for tool in suspicious_tools):
            flags.append(
                ForensicFlag(
                    code="PDF_METADATA_ANOMALY",
                    severity=ForensicSeverity.WARNING,
                    detail=f"Image editor metadata detected in PDF: {creator_producer.strip()}",
                    location=LocationRef(page=1, snippet=creator_producer.strip()[:120]),
                )
            )

        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            page_texts: list[str] = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = str(span.get("text", ""))
                        if not span_text.strip():
                            continue
                        page_texts.append(span_text)

                        hidden, reason = is_hidden_pdf_span(span, page.rect)
                        if hidden:
                            bbox_vals = tuple(float(v) for v in span["bbox"])
                            flags.append(
                                ForensicFlag(
                                    code="HIDDEN_PDF_TEXT",
                                    severity=ForensicSeverity.CRITICAL,
                                    detail=f"Hidden PDF span ({reason}): {span_text.strip()[:80]!r}",
                                    location=LocationRef(
                                        page=page_num,
                                        bbox=(bbox_vals[0], bbox_vals[1], bbox_vals[2], bbox_vals[3]),
                                        snippet=span_text.strip()[:120],
                                    ),
                                )
                            )

            full_page_text = "\n".join(page_texts)
            for inj_flag in scan_prompt_injection(
                full_page_text,
                location=LocationRef(page=page_num, snippet=full_page_text[:120]),
            ):
                flags.append(inj_flag)
    finally:
        doc.close()

    return flags, metadata


def _extract_image_metadata(img: Image.Image) -> dict[str, str]:
    """Extract EXIF and PIL info metadata tags safely as strings."""
    metadata: dict[str, str] = {}
    if img.format:
        metadata["format"] = str(img.format)
    metadata["size"] = f"{img.width}x{img.height}"

    for key, val in img.info.items():
        if key == "exif":
            continue
        if isinstance(val, (str, int, float)):
            metadata[str(key)] = str(val)

    try:
        exif = img.getexif()
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if isinstance(value, bytes):
                metadata[str(tag_name)] = value.decode("utf-8", errors="ignore")[:120]
            else:
                metadata[str(tag_name)] = str(value)[:120]
    except Exception:
        pass
    return metadata


def _compute_jpeg_ela(img: Image.Image) -> tuple[float, tuple[float, float, float, float] | None]:
    """Compute block-level JPEG Error Level Analysis score and peak anomaly bbox."""
    if (img.format or "").upper() != "JPEG":
        return 0.0, None

    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr = np.asarray(rgb, dtype=np.float32)
    re_arr = np.asarray(recompressed, dtype=np.float32)
    ela_map = np.clip(np.abs(orig_arr - re_arr) * _ELA_SCALE, 0.0, 255.0).mean(axis=2)

    height, width = ela_map.shape
    max_block_mean = 0.0
    max_bbox: tuple[float, float, float, float] | None = None

    for y in range(0, max(1, height - _ELA_BLOCK_SIZE + 1), _ELA_BLOCK_SIZE):
        for x in range(0, max(1, width - _ELA_BLOCK_SIZE + 1), _ELA_BLOCK_SIZE):
            y_end = min(height, y + _ELA_BLOCK_SIZE)
            x_end = min(width, x + _ELA_BLOCK_SIZE)
            block_mean = float(ela_map[y:y_end, x:x_end].mean())
            if block_mean > max_block_mean:
                max_block_mean = block_mean
                max_bbox = (float(x), float(y), float(x_end), float(y_end))

    return max_block_mean, max_bbox


def _compute_blur_metrics(img: Image.Image) -> tuple[float, float, tuple[float, float, float, float] | None]:
    """Compute local Laplacian variance over textured blocks and return legibility score."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0, 0.5, None

    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )

    h, w = lap.shape
    min_lvar: float | None = None
    min_bbox: tuple[float, float, float, float] | None = None
    block = _ELA_BLOCK_SIZE

    for y in range(0, max(1, h - block + 1), block):
        for x in range(0, max(1, w - block + 1), block):
            patch = gray[y : y + block, x : x + block]
            # Evaluate sharpness only on non-flat blocks containing visible strokes/edges
            if float(patch.max() - patch.min()) > 15.0:
                lvar = float(lap[y : y + block, x : x + block].var())
                if min_lvar is None or lvar < min_lvar:
                    min_lvar = lvar
                    min_bbox = (float(x), float(y), float(x + block), float(y + block))

    if min_lvar is None:
        global_var = float(lap.var())
        legibility = min(1.0, max(0.35, global_var / 150.0))
        return global_var, round(legibility, 4), (0.0, 0.0, float(img.width), float(img.height))

    if min_lvar < _BLUR_LAP_THRESHOLD:
        legibility = max(0.45, min(0.68, min_lvar / _BLUR_LAP_THRESHOLD * 0.35 + 0.45))
    else:
        legibility = 1.0

    return round(min_lvar, 2), round(legibility, 4), min_bbox


def inspect_image_forensics(
    path: Path,
) -> tuple[list[ForensicFlag], dict[str, str], float]:
    """Inspect an image/scan for ELA tampering, EXIF anomalies, and blur."""
    flags: list[ForensicFlag] = []
    with Image.open(path) as img:
        metadata = _extract_image_metadata(img)

        ela_peak, ela_bbox = _compute_jpeg_ela(img)
        metadata["ela_peak_block_score"] = f"{ela_peak:.2f}"
        if ela_peak >= settings.ela_tamper_threshold and ela_bbox is not None:
            flags.append(
                ForensicFlag(
                    code="ELA_TAMPER_DETECTED",
                    severity=ForensicSeverity.CRITICAL,
                    detail=(
                        f"Localized JPEG ELA block score {ela_peak:.2f} exceeds "
                        f"threshold {settings.ela_tamper_threshold:.2f}."
                    ),
                    location=LocationRef(
                        bbox=ela_bbox,
                        snippet=f"ELA peak={ela_peak:.2f}",
                    ),
                )
            )

        min_lvar, legibility_score, blur_bbox = _compute_blur_metrics(img)
        metadata["min_local_laplacian_var"] = f"{min_lvar:.2f}"
        metadata["legibility_score"] = f"{legibility_score:.4f}"
        if min_lvar < _BLUR_LAP_THRESHOLD and blur_bbox is not None:
            flags.append(
                ForensicFlag(
                    code="LOW_LEGIBILITY_BLUR",
                    severity=ForensicSeverity.WARNING,
                    detail=(
                        f"Low local sharpness detected (Laplacian var={min_lvar:.2f} "
                        f"< {_BLUR_LAP_THRESHOLD:.1f}); OCR confidence capped at {legibility_score:.2f}."
                    ),
                    location=LocationRef(
                        bbox=blur_bbox,
                        snippet=f"Blurred region lvar={min_lvar:.2f}",
                    ),
                )
            )

    return flags, metadata, legibility_score


def inspect_table_forensics(path: Path) -> tuple[list[ForensicFlag], dict[str, str]]:
    """Inspect tabular data (CSV/TSV/Parquet) for missing values and injection cells."""
    flags: list[ForensicFlag] = []
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    elif ext == ".parquet":
        df = pd.read_parquet(path).fillna("").astype(str)
    else:
        df = pd.read_excel(path, dtype=str).fillna("")

    metadata: dict[str, str] = {
        "row_count": str(len(df)),
        "columns": ",".join(str(c) for c in df.columns),
    }

    for idx, row in df.iterrows():
        row_num = int(idx) + 1
        row_preview = " | ".join(f"{col}={row[col]}" for col in df.columns)[:120]
        for col in df.columns:
            col_name = str(col)
            cell_val = str(row[col]).strip()
            if cell_val == "" or cell_val.lower() in {"na", "n/a", "null", "nan", "none"}:
                flags.append(
                    ForensicFlag(
                        code="MISSING_FIELD",
                        severity=ForensicSeverity.WARNING,
                        detail=f"Blank or missing required cell at row {row_num}, column '{col_name}'.",
                        location=LocationRef(
                            row=row_num,
                            column=col_name,
                            snippet=row_preview,
                        ),
                    )
                )
            else:
                inj_flags = scan_prompt_injection(
                    cell_val,
                    location=LocationRef(
                        row=row_num,
                        column=col_name,
                        snippet=cell_val[:120],
                    ),
                )
                flags.extend(inj_flags)

    return flags, metadata


def inspect_text_forensics(path: Path) -> tuple[list[ForensicFlag], dict[str, str]]:
    """Inspect plain text or JSON files for prompt-injection payloads."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    flags = scan_prompt_injection(raw)
    metadata = {"byte_length": str(len(raw.encode("utf-8")))}
    return flags, metadata


def _apply_trust_penalties(
    base_trust: float,
    flags: list[ForensicFlag],
) -> tuple[float, bool]:
    """Compute final trust_prior and quarantine boolean from forensic flags."""
    critical_count = sum(1 for f in flags if f.severity == ForensicSeverity.CRITICAL)
    warning_count = sum(1 for f in flags if f.severity == ForensicSeverity.WARNING)

    if critical_count > 0:
        penalized = max(0.05, round(0.15 - 0.05 * (critical_count - 1), 4))
        return min(base_trust, penalized, 0.15), True

    if warning_count > 0:
        penalized = max(0.30, round(base_trust - 0.10 * warning_count, 4))
        return penalized, False

    return base_trust, False


def analyze_source_with_context(path: Path) -> tuple[SourceDoc, float]:
    """Run pre-LLM forensics on a source file and return (SourceDoc, legibility_score)."""
    resolved = path.resolve()
    modality = detect_modality(resolved)
    sha256 = compute_sha256(resolved)
    base_trust = TRUST_PRIORS[modality]

    legibility_score = 1.0
    if modality == Modality.PDF:
        flags, metadata = inspect_pdf_forensics(resolved)
    elif modality in (Modality.IMAGE, Modality.SCAN):
        flags, metadata, legibility_score = inspect_image_forensics(resolved)
    elif modality == Modality.TABLE:
        flags, metadata = inspect_table_forensics(resolved)
    else:
        flags, metadata = inspect_text_forensics(resolved)

    final_trust, quarantined = _apply_trust_penalties(base_trust, flags)
    source_doc = SourceDoc(
        source_id=f"src_{sha256[:12]}",
        file_path=str(resolved),
        file_name=resolved.name,
        modality=modality,
        sha256=sha256,
        trust_prior=final_trust,
        forensic_flags=flags,
        quarantined=quarantined,
        metadata=metadata,
    )
    return source_doc, legibility_score


def analyze_source(path: Path) -> SourceDoc:
    """Run pre-LLM forensic inspection and return a validated SourceDoc."""
    source_doc, _ = analyze_source_with_context(path)
    return source_doc


def inspect_document_forensics(file_path: Path) -> SourceDoc:
    """Alias for `analyze_source` preserving stub entrypoint naming."""
    return analyze_source(file_path)

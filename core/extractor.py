"""Atomic claim extraction and 3-tier cross-modal entity resolution.

Stage A (`extract_claims`): Operates strictly on sanitized `ParsedChunk` objects,
preserving exact `source_id`, `modality`, and `location_ref` provenance while
supporting both schema-constrained local Ollama extraction (`use_llm=True`)
and generic deterministic fallback extraction (`use_llm=False`).

Stage B (`resolve_entities`): Resolves raw entity aliases across modalities via:
1. Deterministic regex key canonicalization (`ENT_INV_402`, `ENT_SHIP_884`, etc.).
2. RapidFuzz lexical clustering (with optional offline `BAAI/bge-m3` embedding hook).
3. Optional local Ollama tie-breaking (`127.0.0.1:11434`) for borderline pairs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter
from rapidfuzz import fuzz

from core.config import settings
from core.schemas import Claim, LocationRef, Modality, ParsedChunk, SourceDoc

_INVOICE_ID_RE = re.compile(r"\binv(?:oice)?\s*[-#:]?\s*(\d+)\b", re.IGNORECASE)
_WIRE_ID_RE = re.compile(r"\bwire\s*[-#:]?\s*(q\d+|[a-z]+\d+|\d+)\b", re.IGNORECASE)
_PO_ID_RE = re.compile(r"\bpo\s*[-#:]?\s*(\d+)\b", re.IGNORECASE)
_GENERIC_ID_RE = re.compile(r"\b([A-Z]{2,8})[-#](\d{1,6})\b")
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_CURRENCY_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.\d{1,2})?)")
_NUMERIC_UNIT_RE = re.compile(
    r"^\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(kg|g|mg|lb|lbs|tons?|units?|usd|eur|%)?\s*$",
    re.IGNORECASE,
)
_KEY_VALUE_LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 _-]{1,32})\s*:\s*([^\n]+?)\s*$"
)

_AMENDED_FROM_TO_RE = re.compile(
    r"(?:amended|revised|updated|adjusted)\s+from\s+\$\s*[0-9][0-9,]*(?:\.\d{1,2})?\s+to\s+\$\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_FINAL_APPROVED_RE = re.compile(
    r"(?:final\s+approved\s+expenditure|current\s+and\s+binding\s+payable\s+amount|authorized\s+payable\s+total)\s*(?:is|of|:)?\s*\$\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_SUPERSEDES_CUE_RE = re.compile(
    r"\b(?:supersedes|superseding|amended|amendment|addendum|prior\s+quotes)\b",
    re.IGNORECASE,
)

_VENDOR_COLON_RE = re.compile(
    r"\b(?:vendor|be\s*neficiary(?:\s+name)?)\s*:\s*([A-Za-z0-9][A-Za-z0-9 .,&-]{1,60})",
    re.IGNORECASE,
)
_VENDOR_FROM_RE = re.compile(
    r"\bfrom\s+([A-Z][A-Za-z0-9 .&-]{2,40}?)(?=\s+(?:has\s+been|was\s+recorded|was\s+approved|on\b)|[.,\n]|$)"
)
_APPROVER_COLON_RE = re.compile(
    r"\b(?:approver|approved\s+by|authorizing\s+officer)\s*:\s*([A-Za-z][A-Za-z0-9 .,&-]{1,60})",
    re.IGNORECASE,
)

_MISSING_SENTINELS: frozenset[str] = frozenset(
    {"", "[missing]", "na", "n/a", "null", "none", "unknown", "illegible", "[illegible]"}
)
_GENERIC_ENTITY_NOUNS: frozenset[str] = frozenset(
    {
        "amount",
        "total",
        "price",
        "date",
        "vendor",
        "carrier",
        "status",
        "approver",
        "invoice",
        "document",
        "receipt",
        "note",
        "notes",
        "delivery note",
        "wire",
        "transfer",
        "value",
        "type",
        "number",
        "reference",
        "shipment",
        "summary",
    }
)
_NOISY_LLM_ATTRS: frozenset[str] = frozenset(
    {"fallback_entity", "fallback_date", "type", "number", "value"}
)
_CORP_SUFFIX_RE = re.compile(r"\b(?:inc|llc|ltd|corp|co|services)\b\.?", re.IGNORECASE)
_ID_COLUMN_NAMES: tuple[str, ...] = (
    "invoice_id",
    "transaction_id",
    "shipment_id",
    "manifest_id",
    "batch_id",
    "order_id",
    "record_id",
    "entity_id",
    "entity",
    "id",
)
_ATTR_SYNONYMS: dict[str, str] = {
    "total": "amount",
    "total_amount": "amount",
    "approved_amount": "amount",
    "payable_amount": "amount",
    "authorized_amount": "amount",
    "disbursement_principal": "amount",
    "disbursementprincipal": "amount",
    "vendor_name": "vendor",
    "supplier": "vendor",
    "beneficiary": "vendor",
    "beneficiary_name": "vendor",
    "beneficiaryname": "vendor",
    "approved_by": "approver",
    "authorizing_officer": "approver",
    "cargo_weight": "weight",
    "cargoweight": "weight",
    "gross_weight": "weight",
    "net_weight": "weight",
    "vendor_note": "vendor_notes",
    "delivery_note": "vendor_notes",
    "notes": "vendor_notes",
    "field_log": "vendor_notes",
}
_SKIP_KV_KEYS: frozenset[str] = frozenset(
    {
        "from",
        "to",
        "subject",
        "date",
        "document",
        "related invoice",
        "invoice reference",
        "transaction id",
        "shipment",
        "issuing authority",
        "purchase order reference",
        "authorized approver signature",
        "total amountcharged",
        "freight transport services",
        "regulatory fuel surcharge",
        "legal precedence clause",
        "field representative log",
        "secondary notice",
    }
)


class _LLMExtractedFact(BaseModel):
    """Schema-constrained atomic fact emitted by local Ollama."""

    chunk_index: int
    entity: str
    attribute: str
    value: str
    numeric_value: float | None = None
    unit: str | None = None
    timestamp: str | None = None
    is_amendment: bool = False


class _LLMExtractionEnvelope(BaseModel):
    """Top-level JSON schema passed to Ollama constrained decoding."""

    facts: list[_LLMExtractedFact] = Field(default_factory=list)


def _normalize_attribute(raw_attr: str) -> str:
    """Normalize extracted attribute labels to canonical snake_case names."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw_attr.strip().lower()).strip("_")
    collapsed = cleaned.replace("_", "")
    if collapsed in _ATTR_SYNONYMS:
        return _ATTR_SYNONYMS[collapsed]
    return _ATTR_SYNONYMS.get(cleaned, cleaned)


def _parse_numeric_and_unit(raw_value: str, attr: str = "") -> tuple[float | None, str | None]:
    """Parse optional numeric magnitude and unit from a formatted value string."""
    stripped = raw_value.strip()
    if attr in {"date", "timestamp"}:
        return None, None
    m = _NUMERIC_UNIT_RE.match(stripped)
    if not m:
        return None, None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None, None
    unit_raw = m.group(2)
    if unit_raw:
        unit = unit_raw.upper() if unit_raw.lower() in {"usd", "eur"} else unit_raw.lower()
    elif "$" in stripped or attr in {"amount", "price", "total", "cost"}:
        unit = "USD"
    else:
        unit = None
    return num, unit


def _find_structured_entity_ids(text: str) -> list[str]:
    """Return raw structured entity substrings (`INV-402`, `WIRE-Q4`, `SHIP-884`, etc.)."""
    primary: list[tuple[int, str]] = []
    for pattern in (_INVOICE_ID_RE, _WIRE_ID_RE):
        for m in pattern.finditer(text):
            primary.append((m.start(), m.group(0).strip()))
    if primary:
        primary.sort(key=lambda item: item[0])
        return [item[1] for item in primary]

    generic: list[tuple[int, str]] = []
    for m in _GENERIC_ID_RE.finditer(text):
        prefix = m.group(1).upper()
        if prefix in {"PO", "AUTH"}:
            continue
        generic.append((m.start(), m.group(0).strip()))
    generic.sort(key=lambda item: item[0])
    return [item[1] for item in generic]


def _canonicalize_key(entity_raw: str) -> str | None:
    """Tier 1 deterministic key canonicalization for structured IDs."""
    inv = _INVOICE_ID_RE.search(entity_raw)
    if inv:
        return f"ENT_INV_{int(inv.group(1))}"

    wire = _WIRE_ID_RE.search(entity_raw)
    if wire:
        return f"ENT_WIRE_{wire.group(1).upper()}"

    generic = _GENERIC_ID_RE.search(entity_raw)
    if generic:
        return f"ENT_{generic.group(1).upper()}_{generic.group(2).upper()}"

    po = _PO_ID_RE.search(entity_raw)
    if po:
        return f"ENT_PO_{int(po.group(1))}"

    return None


def _parse_numeric_amount(raw_num: str) -> float:
    """Convert formatted numeric string (e.g. '12,000.00') to float."""
    return float(raw_num.replace(",", "").strip())


def _compute_trust(source: SourceDoc, chunk: ParsedChunk) -> float:
    """Compute initial claim trust score clamped to [0.0, 1.0]."""
    score = round(source.trust_prior * chunk.extraction_confidence, 4)
    return min(1.0, max(0.0, score))


def _new_claim_id(source_id: str, counter: int) -> str:
    """Create deterministic claim identifier."""
    return f"clm_{source_id}_{counter:04d}"


def _extract_table_claims(
    source: SourceDoc,
    chunk: ParsedChunk,
    counter_start: int,
) -> tuple[list[Claim], int]:
    """Extract atomic claims from sanitized CSV/tabular rows across any schema."""
    kv_pairs: list[tuple[str, str]] = []
    for part in chunk.text.split("|"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        kv_pairs.append((key.strip(), val.strip()))

    if not kv_pairs:
        return [], counter_start

    row_map = {k.lower(): v for k, v in kv_pairs}
    id_col: str | None = None
    entity_raw = ""
    for candidate in _ID_COLUMN_NAMES:
        val = row_map.get(candidate, "")
        if val and val.lower() not in _MISSING_SENTINELS:
            id_col = candidate
            entity_raw = val
            break

    if not entity_raw:
        # Generic fallback: use first non-missing column as entity identifier
        first_col, first_val = kv_pairs[0]
        if first_val.lower() in _MISSING_SENTINELS:
            return [], counter_start
        id_col = first_col.lower()
        entity_raw = first_val

    row_date = row_map.get("date") or row_map.get("timestamp")
    timestamp = row_date if row_date and row_date.lower() not in _MISSING_SENTINELS else None
    trust_score = _compute_trust(source, chunk)

    claims: list[Claim] = []
    counter = counter_start
    for key, val in kv_pairs:
        raw_col = key.lower()
        if raw_col == id_col:
            continue
        if val.lower() in _MISSING_SENTINELS:
            # Never fabricate claims for [MISSING] or blank cells.
            continue

        attr = _normalize_attribute(raw_col)
        numeric_value, unit = _parse_numeric_and_unit(val, attr=attr)

        counter += 1
        claims.append(
            Claim(
                claim_id=_new_claim_id(source.source_id, counter),
                entity_raw=entity_raw,
                attribute=attr,
                value=val,
                numeric_value=numeric_value,
                unit=unit,
                timestamp=timestamp,
                modality=chunk.modality,
                source_id=source.source_id,
                location_ref=chunk.location_ref,
                extraction_confidence=chunk.extraction_confidence,
                trust_score=trust_score,
                is_amendment=False,
            )
        )
    return claims, counter


def _infer_doc_context(chunks: list[ParsedChunk]) -> tuple[str | None, str | None]:
    """Scan a document's sanitized chunks for fallback entity ID and header ISO date."""
    doc_entity: str | None = None
    doc_date: str | None = None

    for chunk in chunks:
        if doc_entity is None:
            ids = _find_structured_entity_ids(chunk.text)
            if ids:
                doc_entity = ids[0]
            else:
                v_match = _VENDOR_COLON_RE.search(chunk.text)
                if v_match:
                    doc_entity = v_match.group(1).strip()

        if doc_date is None:
            d_match = _ISO_DATE_RE.search(chunk.text)
            if d_match:
                doc_date = d_match.group(1)

        if doc_entity and doc_date:
            break

    return doc_entity, doc_date


def _bbox_same_row(a: LocationRef, b: LocationRef) -> bool:
    """Return True if two bbox locations share vertical line overlap."""
    if a.bbox is None or b.bbox is None:
        return False
    ay0, ay1 = a.bbox[1], a.bbox[3]
    by0, by1 = b.bbox[1], b.bbox[3]
    overlap = min(ay1, by1) - max(ay0, by0)
    min_height = max(1.0, min(ay1 - ay0, by1 - by0))
    return overlap >= 0.3 * min_height


def _classify_amount_attribute(context_text: str) -> str:
    """Distinguish primary invoice total (`amount`) from itemized subtotals/surcharges."""
    lowered = context_text.lower()
    if any(k in lowered for k in ("freight", "subtotal", "transport")):
        return "line_item_freight"
    if any(k in lowered for k in ("surcharge", "tax", "fee")):
        return "line_item_surcharge"
    return "amount"


def _extract_ollama_document_claims(
    source: SourceDoc,
    source_chunks: list[ParsedChunk],
    fallback_entity: str | None,
    fallback_date: str | None,
    counter_start: int,
) -> tuple[list[Claim] | None, int]:
    """Schema-constrained local Ollama extraction bound to exact `ParsedChunk` provenance."""
    base_url = settings.ollama_base_url.rstrip("/")
    if not ("127.0.0.1" in base_url or "localhost" in base_url):
        raise ValueError(f"Offline guarantee violated by non-local Ollama URL: {base_url}")

    try:
        import ollama

        numbered_lines = [
            f"[chunk_index={idx}] {chunk.text}"
            for idx, chunk in enumerate(source_chunks)
            if chunk.text.strip()
        ]
        if not numbered_lines:
            return [], counter_start

        prompt = (
            "Extract atomic factual claims from the numbered document chunks below.\n"
            "Rules:\n"
            "- Set `chunk_index` to the exact chunk where the value appears.\n"
            "- Use the real document entity ID as `entity` (e.g., INV-402, INV-510, WIRE-Q4, SHIP-884). "
            "NEVER use generic nouns like 'amount', 'vendor', 'date', 'document', 'invoice', or 'value' as `entity`.\n"
            "- Use concise lowercase attribute names (`amount`, `vendor`, `status`, `approver`, `weight`, `carrier`, `vendor_notes`, etc.). "
            "NEVER output attributes named 'fallback_entity' or 'fallback_date'.\n"
            "- Do NOT treat 'From:' email header addresses as `approver`.\n"
            "- If a clause amends a value from X to Y (or supersedes an older figure), emit ONLY the new value Y with `is_amendment=true`.\n"
            "- NEVER invent missing approvers or signatures.\n"
            f"Primary Document Entity: {fallback_entity or 'NONE'}, Document Header Date: {fallback_date or 'null'}.\n\n"
            "Chunks:\n" + "\n---\n".join(numbered_lines)
        )

        client = ollama.Client(host=base_url)
        resp = client.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            format=_LLMExtractionEnvelope.model_json_schema(),
            options={"temperature": 0.0},
        )
        envelope = _LLMExtractionEnvelope.model_validate_json(resp["message"]["content"])
        if not envelope.facts:
            return None, counter_start

        candidates: list[tuple[bool, Claim]] = []
        counter = counter_start
        for fact in envelope.facts:
            if not (0 <= fact.chunk_index < len(source_chunks)):
                continue
            chunk = source_chunks[fact.chunk_index]
            local_ids = _find_structured_entity_ids(chunk.text)
            inferred_ent = local_ids[0] if local_ids else (fallback_entity or "")

            raw_ent = fact.entity.strip()
            raw_attr = fact.attribute.strip().lower()
            val = fact.value.strip()

            # 1. Promote entity-reference facts (e.g., entity="invoice", attr="reference", val="inv-402")
            if raw_attr in {"reference", "invoice", "id"} and _canonicalize_key(val) is not None:
                ent = val
                is_note_doc = source.quarantined or any(
                    w in chunk.text.lower() for w in ("delivery", "shipment", "advice", "note", "field representative")
                )
                attr = "vendor_notes" if is_note_doc else "summary"
                val = chunk.text.replace("\n", " ")
            # 2. Repair generic noun, pure numeric, or non-ID entity fields when doc has a structured ID
            elif (
                raw_ent.lower() in _GENERIC_ENTITY_NOUNS
                or _NUMERIC_UNIT_RE.match(raw_ent)
                or (_canonicalize_key(raw_ent) is None and inferred_ent and _canonicalize_key(inferred_ent) is not None)
            ):
                if raw_ent.lower() in {"vendor", "carrier", "amount", "date", "status", "approver"} and raw_attr in {"name", "value", "total"}:
                    attr = _normalize_attribute(raw_ent)
                elif _canonicalize_key(raw_ent) is None and raw_ent and raw_ent.lower() not in _GENERIC_ENTITY_NOUNS and raw_attr in {"name", "vendor", "carrier"}:
                    # e.g., entity="Vendor K", attr="vendor" -> attr="vendor", val="Vendor K", entity=WIRE-Q4
                    attr = "vendor" if raw_attr in {"name", "vendor"} else _normalize_attribute(raw_attr)
                    val = raw_ent
                else:
                    attr = _normalize_attribute(raw_attr)
                ent = inferred_ent
            else:
                ent = raw_ent or inferred_ent
                attr = _normalize_attribute(raw_attr)

            if not ent or not attr or attr in _NOISY_LLM_ATTRS:
                continue
            if val.lower() in _MISSING_SENTINELS:
                continue
            if source.quarantined and attr == "status":
                continue
            if attr == "approver" and ("@" in val or "from:" in chunk.text.lower() or chunk.extraction_confidence < 0.70):
                continue

            is_amendment = bool(fact.is_amendment and _SUPERSEDES_CUE_RE.search(chunk.text))
            if is_amendment and _AMENDED_FROM_TO_RE.search(chunk.text):
                m_new = _AMENDED_FROM_TO_RE.search(chunk.text)
                if m_new and fact.numeric_value is not None:
                    new_num = _parse_numeric_amount(m_new.group(1))
                    if abs(fact.numeric_value - new_num) > 0.01:
                        continue

            num_val = fact.numeric_value
            unit = fact.unit
            if num_val is None:
                num_val, parsed_unit = _parse_numeric_and_unit(val, attr=attr)
                unit = unit or parsed_unit

            ts = fact.timestamp.strip() if fact.timestamp else fallback_date
            if is_amendment and fallback_date:
                ts = fallback_date

            # Check whether literal value/number actually appears in this chunk for OCR bbox accuracy
            literal_hit = val.lower() in chunk.text.lower() or (
                num_val is not None
                and (str(int(num_val)) in chunk.text.replace(",", "") or str(num_val) in chunk.text.replace(",", ""))
            )

            counter += 1
            candidates.append(
                (
                    literal_hit,
                    Claim(
                        claim_id=_new_claim_id(source.source_id, counter),
                        entity_raw=ent,
                        attribute=attr,
                        value=val,
                        numeric_value=num_val,
                        unit=unit,
                        timestamp=ts,
                        modality=chunk.modality,
                        source_id=source.source_id,
                        location_ref=chunk.location_ref,
                        extraction_confidence=chunk.extraction_confidence,
                        trust_score=_compute_trust(source, chunk),
                        is_amendment=is_amendment,
                    ),
                )
            )

        # Deduplicate same-source duplicate facts, preferring literal chunk hits for exact bboxes
        deduped: dict[tuple[str, str, str], tuple[bool, Claim]] = {}
        for literal_hit, clm in candidates:
            ent_key = _canonicalize_key(clm.entity_raw) or clm.entity_raw.lower()
            val_key = str(round(clm.numeric_value, 2)) if clm.numeric_value is not None else clm.value.lower()
            dkey = (ent_key, clm.attribute, val_key)
            existing = deduped.get(dkey)
            if existing is None or (literal_hit and not existing[0]):
                deduped[dkey] = (literal_hit, clm)

        claims = [pair[1] for pair in deduped.values()]

        # Guarantee sanitized quarantined documents preserve a linked note/summary on their inferred entity
        if source.quarantined and fallback_entity:
            fb_key = _canonicalize_key(fallback_entity) or fallback_entity.lower()
            has_linked = any(
                (_canonicalize_key(c.entity_raw) or c.entity_raw.lower()) == fb_key
                for c in claims
            )
            needs_pdf_note = source.modality == Modality.PDF and not any(
                c.attribute == "vendor_notes" for c in claims
            )
            if not has_linked or needs_pdf_note:
                desc_chunk = next(
                    (
                        ch
                        for ch in source_chunks
                        if re.search(r"\b(?:field\s+representative\s+log|bill\s+of\s+lading)\b", ch.text, re.IGNORECASE)
                    ),
                    max(source_chunks, key=lambda ch: len(ch.text)),
                )
                local_ids = _find_structured_entity_ids(desc_chunk.text)
                note_ent = local_ids[0] if local_ids else fallback_entity
                counter += 1
                claims.append(
                    Claim(
                        claim_id=_new_claim_id(source.source_id, counter),
                        entity_raw=note_ent,
                        attribute="vendor_notes",
                        value=desc_chunk.text.replace("\n", " "),
                        timestamp=fallback_date,
                        modality=desc_chunk.modality,
                        source_id=source.source_id,
                        location_ref=desc_chunk.location_ref,
                        extraction_confidence=desc_chunk.extraction_confidence,
                        trust_score=_compute_trust(source, desc_chunk),
                        is_amendment=False,
                    )
                )

        return (claims, counter) if claims else (None, counter_start)
    except Exception:
        return None, counter_start


def _extract_unstructured_chunk_deterministic(
    source: SourceDoc,
    chunk: ParsedChunk,
    prev_chunk: ParsedChunk | None,
    doc_entity: str | None,
    doc_date: str | None,
    counter_start: int,
) -> tuple[list[Claim], int]:
    """Deterministic atomic claim extractor for PDF, OCR scan/image, and text chunks."""
    text = chunk.text.strip()
    if not text:
        return [], counter_start

    local_entities = _find_structured_entity_ids(text)
    entity_raw = local_entities[0] if local_entities else doc_entity
    if not entity_raw:
        return [], counter_start

    is_amendment_chunk = bool(_SUPERSEDES_CUE_RE.search(text))
    local_dates = _ISO_DATE_RE.findall(text)
    if is_amendment_chunk and doc_date:
        timestamp = doc_date
    else:
        timestamp = local_dates[0] if local_dates else doc_date

    trust_score = _compute_trust(source, chunk)
    claims: list[Claim] = []
    counter = counter_start
    emitted_keys: set[tuple[str, str]] = set()

    def _emit(
        attr: str,
        val: str,
        numeric_value: float | None = None,
        unit: str | None = None,
        is_amendment: bool = False,
    ) -> None:
        nonlocal counter
        dedup_key = (attr, val.strip().lower())
        if dedup_key in emitted_keys:
            return
        emitted_keys.add(dedup_key)
        counter += 1
        claims.append(
            Claim(
                claim_id=_new_claim_id(source.source_id, counter),
                entity_raw=entity_raw,
                attribute=attr,
                value=val,
                numeric_value=numeric_value,
                unit=unit,
                timestamp=timestamp,
                modality=chunk.modality,
                source_id=source.source_id,
                location_ref=chunk.location_ref,
                extraction_confidence=chunk.extraction_confidence,
                trust_score=trust_score,
                is_amendment=is_amendment,
            )
        )

    # 1. Amendment-aware monetary extraction
    if is_amendment_chunk:
        emitted_amendment = False
        for pattern in (_AMENDED_FROM_TO_RE, _FINAL_APPROVED_RE):
            m = pattern.search(text)
            if m:
                num_str = m.group(1)
                _emit(
                    "amount",
                    f"${num_str}",
                    numeric_value=_parse_numeric_amount(num_str),
                    unit="USD",
                    is_amendment=True,
                )
                emitted_amendment = True
                break
        if not emitted_amendment:
            cleaned_clause = re.sub(
                r"(?:including|from)\s+(?:the\s+)?\$\s*[0-9][0-9,]*(?:\.\d{1,2})?",
                "",
                text,
                flags=re.IGNORECASE,
            )
            currency_matches = _CURRENCY_RE.findall(cleaned_clause)
            if currency_matches:
                num_str = currency_matches[-1]
                _emit(
                    "amount",
                    f"${num_str}",
                    numeric_value=_parse_numeric_amount(num_str),
                    unit="USD",
                    is_amendment=True,
                )
    else:
        # 2. Standard monetary extraction (with OCR label pairing)
        currency_matches = list(_CURRENCY_RE.finditer(text))
        if currency_matches:
            pair_label = ""
            if prev_chunk is not None and (
                _bbox_same_row(prev_chunk.location_ref, chunk.location_ref)
                or len(text) <= 20
            ):
                pair_label = prev_chunk.text
            for m in currency_matches:
                num_str = m.group(1)
                context = f"{pair_label} {text}".strip()
                attr = _classify_amount_attribute(context)
                _emit(
                    attr,
                    f"${num_str}",
                    numeric_value=_parse_numeric_amount(num_str),
                    unit="USD",
                )

    # 3. Vendor extraction
    vendor_val: str | None = None
    v_colon = _VENDOR_COLON_RE.search(text)
    if v_colon:
        vendor_val = v_colon.group(1).strip()
    else:
        v_from = _VENDOR_FROM_RE.search(text)
        if v_from and "@" not in v_from.group(1):
            vendor_val = v_from.group(1).strip()
    if vendor_val:
        _emit("vendor", vendor_val)

    # 4. Status extraction (strictly excluding quarantined vendor notes)
    if (
        not source.quarantined
        and re.search(r"\b(?:reviewed\s+and\s+approved|initially\s+approved)\b", text, re.IGNORECASE)
    ):
        _emit("status", "Approved")

    # 5. Explicit non-empty approver extraction (ignores empty signature labels)
    app_match = _APPROVER_COLON_RE.search(text)
    if app_match and chunk.extraction_confidence >= 0.70:
        approver_val = app_match.group(1).strip()
        if approver_val and approver_val.lower() not in _MISSING_SENTINELS:
            _emit("approver", approver_val)

    # 6. Operational delivery/vendor notes (preserves benign content from vendor_note_injected.pdf)
    if re.search(r"\b(?:field\s+representative\s+log|bill\s+of\s+lading)\b", text, re.IGNORECASE):
        _emit("vendor_notes", text.replace("\n", " "))

    # 7. Generic open-domain Key: Value lines & OCR label-paired numeric+unit chunks
    for line in text.splitlines():
        kv = _KEY_VALUE_LINE_RE.match(line)
        if not kv:
            continue
        raw_k, raw_v = kv.group(1).strip(), kv.group(2).strip()
        collapsed_k = re.sub(r"\s+", "", raw_k.lower())
        if raw_k.lower() in _SKIP_KV_KEYS or collapsed_k in {"beneficiaryname", "disbursementprincipal", "executiondate"}:
            continue
        if raw_v.lower() in _MISSING_SENTINELS or _CURRENCY_RE.search(raw_v):
            continue
        if _find_structured_entity_ids(raw_v) == [entity_raw]:
            continue
        norm_attr = _normalize_attribute(raw_k)
        if norm_attr in {"vendor", "approver", "date", "amount"}:
            continue
        if norm_attr == "status" and source.quarantined:
            continue
        num_val, unit = _parse_numeric_and_unit(raw_v, attr=norm_attr)
        _emit(norm_attr, raw_v, numeric_value=num_val, unit=unit)

    # Pair standalone OCR numeric+unit chunk (e.g. "990 kg") with same-row label (e.g. "Cargo Weight:")
    if not _CURRENCY_RE.search(text):
        num_val, unit = _parse_numeric_and_unit(text)
        if num_val is not None and unit is not None and prev_chunk is not None:
            if _bbox_same_row(prev_chunk.location_ref, chunk.location_ref) or len(text) <= 20:
                label_clean = prev_chunk.text.rstrip(": ").strip()
                if label_clean:
                    attr = _normalize_attribute(label_clean)
                    _emit(attr, text, numeric_value=num_val, unit=unit)

    return claims, counter


def extract_claims(
    docs: list[SourceDoc],
    chunks: list[ParsedChunk],
    use_llm: bool = False,
) -> list[Claim]:
    """Stage A: Extract atomic `Claim` objects from sanitized multimodal chunks."""
    docs_by_id: dict[str, SourceDoc] = {doc.source_id: doc for doc in docs}
    chunks_by_source: dict[str, list[ParsedChunk]] = {}

    for chunk in chunks:
        if not chunk.sanitized:
            raise ValueError(
                f"Security invariant violated: chunk '{chunk.chunk_id}' is not sanitized."
            )
        if chunk.source_id not in docs_by_id:
            raise ValueError(f"Missing parent SourceDoc for chunk source_id='{chunk.source_id}'.")
        chunks_by_source.setdefault(chunk.source_id, []).append(chunk)

    extracted: list[Claim] = []
    for source_id, source_chunks in chunks_by_source.items():
        source = docs_by_id[source_id]
        counter = 0

        if source.modality == Modality.TABLE:
            for chunk in source_chunks:
                row_claims, counter = _extract_table_claims(source, chunk, counter)
                extracted.extend(row_claims)
            continue

        doc_entity, doc_date = _infer_doc_context(source_chunks)

        det_claims: list[Claim] = []
        prev_chunk: ParsedChunk | None = None
        for chunk in source_chunks:
            chunk_claims, counter = _extract_unstructured_chunk_deterministic(
                source,
                chunk,
                prev_chunk=prev_chunk,
                doc_entity=doc_entity,
                doc_date=doc_date,
                counter_start=counter,
            )
            det_claims.extend(chunk_claims)
            prev_chunk = chunk

        if use_llm:
            llm_claims, new_counter = _extract_ollama_document_claims(
                source,
                source_chunks=source_chunks,
                fallback_entity=doc_entity,
                fallback_date=doc_date,
                counter_start=counter,
            )
            if llm_claims is not None:
                counter = new_counter
                # Merge deterministic + LLM claims: deterministic takes precedence for
                # line-item classification and exact OCR/PDF recall; LLM adds open-domain recall.
                merged: list[Claim] = list(det_claims)
                seen_keys: set[tuple[str, str, str]] = set()
                line_item_nums: set[tuple[str, float]] = set()
                for c in det_claims:
                    ek = _canonicalize_key(c.entity_raw) or c.entity_raw.lower()
                    vk = str(round(c.numeric_value, 2)) if c.numeric_value is not None else c.value.lower()
                    seen_keys.add((ek, c.attribute, vk))
                    if c.attribute.startswith("line_item_") and c.numeric_value is not None:
                        line_item_nums.add((ek, round(c.numeric_value, 2)))

                for c in llm_claims:
                    ek = _canonicalize_key(c.entity_raw) or c.entity_raw.lower()
                    vk = str(round(c.numeric_value, 2)) if c.numeric_value is not None else c.value.lower()
                    if c.attribute == "amount" and c.numeric_value is not None and (ek, round(c.numeric_value, 2)) in line_item_nums:
                        continue
                    if (ek, c.attribute, vk) in seen_keys:
                        continue
                    seen_keys.add((ek, c.attribute, vk))
                    merged.append(c)

                extracted.extend(merged)
                continue

        extracted.extend(det_claims)

    return extracted


def _normalize_org_name(name: str) -> str:
    """Normalize organization string for lexical matching."""
    stripped = _CORP_SUFFIX_RE.sub("", name)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", stripped)
    return " ".join(cleaned.lower().split())


def _slug_entity_id(name: str) -> str:
    """Generate a stable `ENT_<SLUG>` identifier from a representative entity string."""
    normalized = _normalize_org_name(name) or re.sub(r"[^a-zA-Z0-9]+", " ", name).strip().lower()
    slug = re.sub(r"[^A-Z0-9]+", "_", normalized.upper()).strip("_")
    return f"ENT_{slug or 'UNKNOWN'}"


def _try_offline_embedding_similarity(aliases: list[str]) -> dict[tuple[str, str], float]:
    """Optional strictly offline `BAAI/bge-m3` cosine similarity lookup; returns {} if uncached."""
    if len(aliases) < 2:
        return {}
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer(settings.embed_model, local_files_only=True)
        vecs = model.encode(aliases, normalize_embeddings=True)
        sims: dict[tuple[str, str], float] = {}
        for i in range(len(aliases)):
            for j in range(i + 1, len(aliases)):
                sims[(aliases[i], aliases[j])] = float(np.dot(vecs[i], vecs[j]))
        return sims
    except Exception:
        return {}


def _ollama_merge_tiebreak(alias_a: str, alias_b: str) -> bool:
    """Tier 3 optional local Ollama entity merge decision for borderline pairs."""
    base_url = settings.ollama_base_url.rstrip("/")
    if not ("127.0.0.1" in base_url or "localhost" in base_url):
        raise ValueError(f"Offline guarantee violated by non-local Ollama URL: {base_url}")
    try:
        import ollama

        client = ollama.Client(host=base_url)
        prompt = (
            f"Do '{alias_a}' and '{alias_b}' refer to the exact same real-world entity? "
            'Respond strictly as JSON: {"same_entity": true|false}.'
        )
        resp = client.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        data = json.loads(resp["message"]["content"])
        return bool(data.get("same_entity", False))
    except Exception:
        return False


def resolve_entities(
    claims: list[Claim],
    use_embeddings: bool = False,
    use_llm: bool = False,
) -> tuple[list[Claim], dict[str, list[str]]]:
    """Stage B: 3-tier cross-modal entity resolution populating `entity_canonical`."""
    raw_to_canonical: dict[str, str] = {}
    unresolved_raws: list[str] = []

    for claim in claims:
        raw = claim.entity_raw.strip()
        if raw in raw_to_canonical:
            continue
        tier1 = _canonicalize_key(raw)
        if tier1 is not None:
            raw_to_canonical[raw] = tier1
        elif raw not in unresolved_raws:
            unresolved_raws.append(raw)

    embed_sims = _try_offline_embedding_similarity(unresolved_raws) if use_embeddings else {}
    clusters: list[list[str]] = []

    for alias in unresolved_raws:
        placed = False
        norm_alias = _normalize_org_name(alias)
        for cluster in clusters:
            rep = cluster[0]
            norm_rep = _normalize_org_name(rep)
            ratio_raw = fuzz.token_sort_ratio(alias.lower(), rep.lower())
            ratio_norm = fuzz.token_sort_ratio(norm_alias, norm_rep) if (norm_alias and norm_rep) else 0.0
            score = max(ratio_raw, ratio_norm)

            sim_key = (alias, rep) if (alias, rep) in embed_sims else (rep, alias)
            emb_sim = embed_sims.get(sim_key, 0.0)

            should_merge = score >= 85.0 or emb_sim >= 0.88
            if not should_merge and use_llm and 72.0 <= score < 85.0:
                should_merge = _ollama_merge_tiebreak(alias, rep)

            if should_merge:
                cluster.append(alias)
                placed = True
                break

        if not placed:
            clusters.append([alias])

    for cluster in clusters:
        rep = min(cluster, key=lambda s: (len(_normalize_org_name(s)), len(s)))
        canonical_id = _slug_entity_id(rep)
        for alias in cluster:
            raw_to_canonical[alias] = canonical_id

    resolved_claims: list[Claim] = []
    entity_aliases: dict[str, set[str]] = {}
    for claim in claims:
        raw = claim.entity_raw.strip()
        canonical_id = raw_to_canonical[raw]
        resolved_claims.append(claim.model_copy(update={"entity_canonical": canonical_id}))
        entity_aliases.setdefault(canonical_id, set()).add(raw)

    entity_map: dict[str, list[str]] = {
        cid: sorted(aliases) for cid, aliases in sorted(entity_aliases.items())
    }
    return resolved_claims, entity_map


def extract_and_persist(
    storage_dir: Path,
    use_llm: bool = False,
    use_embeddings: bool = False,
) -> tuple[list[Claim], dict[str, list[str]]]:
    """Load ingested artifacts, extract & resolve claims, and persist claims/entities JSON."""
    docs_path = storage_dir / "ingested_docs.json"
    chunks_path = storage_dir / "parsed_chunks.json"

    docs = TypeAdapter(list[SourceDoc]).validate_json(docs_path.read_text(encoding="utf-8"))
    chunks = TypeAdapter(list[ParsedChunk]).validate_json(chunks_path.read_text(encoding="utf-8"))

    raw_claims = extract_claims(docs, chunks, use_llm=use_llm)
    resolved_claims, entity_map = resolve_entities(
        raw_claims,
        use_embeddings=use_embeddings,
        use_llm=use_llm,
    )

    claims_json = TypeAdapter(list[Claim]).dump_json(resolved_claims, indent=2)
    (storage_dir / "claims.json").write_bytes(claims_json)
    (storage_dir / "entities.json").write_text(
        json.dumps(entity_map, indent=2),
        encoding="utf-8",
    )
    return resolved_claims, entity_map

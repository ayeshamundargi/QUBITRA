"""Tests for atomic claim extraction, 3-tier entity resolution, and cli.py extract."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter
import pytest

import cli
from core.config import settings
from core.extractor import extract_and_persist, extract_claims, resolve_entities
from core.schemas import Claim, LocationRef, Modality, ParsedChunk, SourceDoc


def _load_ingested() -> tuple[list[SourceDoc], list[ParsedChunk]]:
    docs = TypeAdapter(list[SourceDoc]).validate_json(
        (settings.storage_dir / "ingested_docs.json").read_text(encoding="utf-8")
    )
    chunks = TypeAdapter(list[ParsedChunk]).validate_json(
        (settings.storage_dir / "parsed_chunks.json").read_text(encoding="utf-8")
    )
    return docs, chunks


def test_extract_claims_rejects_unsanitized_chunks() -> None:
    """Verify security invariant: unsanitized chunks must be rejected before extraction."""
    doc = SourceDoc(
        source_id="src_raw",
        file_path="dummy.txt",
        file_name="dummy.txt",
        modality=Modality.TEXT,
        sha256="deadbeef",
        trust_prior=0.6,
    )
    bad_chunk = ParsedChunk(
        chunk_id="chk_bad",
        source_id="src_raw",
        modality=Modality.TEXT,
        text="Invoice 402 amount is $18,000",
        location_ref=LocationRef(char_span=(0, 29)),
        extraction_confidence=0.9,
        sanitized=False,
    )
    with pytest.raises(ValueError, match="sanitized"):
        extract_claims([doc], [bad_chunk])


def test_extract_and_resolve_synthetic_traps() -> None:
    """Verify cross-modal entity merge, conflict capture, amendments, and abstention safety."""
    docs, chunks = _load_ingested()
    docs_by_id = {d.source_id: d for d in docs}
    docs_by_name = {d.file_name: d for d in docs}

    raw_claims = extract_claims(docs, chunks)
    claims, entity_map = resolve_entities(raw_claims)

    assert len(claims) > 0
    assert all(c.entity_canonical is not None for c in claims)

    # 1. Cross-modal entity merge into ENT_INV_402
    assert "ENT_INV_402" in entity_map
    inv402_claims = [c for c in claims if c.entity_canonical == "ENT_INV_402"]
    inv402_files = {docs_by_id[c.source_id].file_name for c in inv402_claims}
    assert {"ledger.csv", "approval_email.pdf", "receipt_scan_tampered.jpg"}.issubset(inv402_files)

    # 2. Conflicting amounts captured with exact provenance
    inv402_amounts = [c for c in inv402_claims if c.attribute == "amount"]
    ledger_amounts = [c for c in inv402_amounts if docs_by_id[c.source_id].file_name == "ledger.csv"]
    email_amounts = [c for c in inv402_amounts if docs_by_id[c.source_id].file_name == "approval_email.pdf"]
    tampered_amounts = [
        c for c in inv402_amounts if docs_by_id[c.source_id].file_name == "receipt_scan_tampered.jpg"
    ]

    assert any(c.numeric_value == 12000.0 and c.location_ref.row is not None for c in ledger_amounts)
    assert any(
        c.numeric_value == 12000.0
        and c.location_ref.page is not None
        and c.location_ref.bbox is not None
        for c in email_amounts
    )
    assert len(tampered_amounts) >= 1
    assert any(
        c.numeric_value == 18000.0
        and c.trust_score <= 0.15
        and c.location_ref.bbox is not None
        for c in tampered_amounts
    )

    # 3. Prompt-injection exclusion on vendor_note_injected.pdf
    vendor_note_id = docs_by_name["vendor_note_injected.pdf"].source_id
    vendor_note_claims = [c for c in claims if c.source_id == vendor_note_id]
    assert all("$18,000" not in c.value and c.numeric_value != 18000.0 for c in vendor_note_claims)

    # 4. Temporal amendment capture on ENT_INV_510
    inv510_amounts = [
        c for c in claims if c.entity_canonical == "ENT_INV_510" and c.attribute == "amount"
    ]
    assert any(
        c.numeric_value == 4500.0 and c.timestamp == "2026-03-15" and c.is_amendment is False
        for c in inv510_amounts
    )
    memo_id = docs_by_name["amendment_memo.pdf"].source_id
    memo_amounts = [c for c in inv510_amounts if c.source_id == memo_id]
    assert len(memo_amounts) >= 1
    assert all(
        c.numeric_value == 5200.0 and c.timestamp == "2026-04-02" and c.is_amendment is True
        for c in memo_amounts
    )

    # 5. Abstention safety on ENT_WIRE_Q4
    wire_claims = [c for c in claims if c.entity_canonical == "ENT_WIRE_Q4"]
    assert any(c.attribute == "amount" and c.numeric_value == 95000.0 for c in wire_claims)
    assert any(c.attribute == "vendor" and c.value == "Vendor K" for c in wire_claims)
    wire_approvers = [c for c in wire_claims if c.attribute == "approver"]
    assert all(c.trust_score < 0.50 for c in wire_approvers)


def test_tier2_rapidfuzz_entity_clustering() -> None:
    """Verify RapidFuzz clusters near-identical organization aliases cleanly."""
    loc = LocationRef(page=1)
    c1 = Claim(
        claim_id="clm_1",
        entity_raw="Apex Logistics",
        attribute="status",
        value="Approved",
        modality=Modality.PDF,
        source_id="src_1",
        location_ref=loc,
        extraction_confidence=0.95,
    )
    c2 = Claim(
        claim_id="clm_2",
        entity_raw="Apex Logistics Inc.",
        attribute="city",
        value="Chicago",
        modality=Modality.PDF,
        source_id="src_2",
        location_ref=loc,
        extraction_confidence=0.90,
    )
    resolved, entity_map = resolve_entities([c1, c2])
    assert resolved[0].entity_canonical == resolved[1].entity_canonical
    canonical = resolved[0].entity_canonical
    assert canonical is not None
    assert set(entity_map[canonical]) == {"Apex Logistics", "Apex Logistics Inc."}


def test_cli_extract_persists_claims_and_entities(tmp_path: Path, capsys) -> None:
    """Verify cli.py extract runs cleanly and persists validated claims.json & entities.json."""
    storage_dir = tmp_path
    (storage_dir / "ingested_docs.json").write_text(
        (settings.storage_dir / "ingested_docs.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (storage_dir / "parsed_chunks.json").write_text(
        (settings.storage_dir / "parsed_chunks.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    exit_code = cli.main(["extract", "--storage", str(storage_dir)])
    assert exit_code == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["command"] == "extract"
    assert stdout_payload["status"] == "ok"
    assert stdout_payload["claim_count"] > 0

    claims_path = storage_dir / "claims.json"
    entities_path = storage_dir / "entities.json"
    assert claims_path.exists()
    assert entities_path.exists()

    loaded_claims = TypeAdapter(list[Claim]).validate_json(claims_path.read_text(encoding="utf-8"))
    assert len(loaded_claims) == stdout_payload["claim_count"]

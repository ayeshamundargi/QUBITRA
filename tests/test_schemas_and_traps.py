"""Tests for core.schemas Pydantic v2 data contracts and synthetic trap validation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from core.schemas import (
    Citation,
    Claim,
    ConflictEdge,
    EdgeType,
    EvidenceDossier,
    ForensicFlag,
    ForensicSeverity,
    LocationRef,
    Modality,
    ParsedChunk,
    SourceDoc,
    VerdictResponse,
    VerdictStatus,
)


def test_location_ref_requires_at_least_one_coordinate() -> None:
    """Verify LocationRef rejects instances where all spatial coordinates are None."""
    with pytest.raises(ValidationError) as exc_info:
        LocationRef()
    assert "at least one coordinate" in str(exc_info.value).lower()

    # Valid instances with single coordinate types
    loc_page = LocationRef(page=1, snippet="Page 1 header")
    assert loc_page.page == 1

    loc_row = LocationRef(row=42, column="status")
    assert loc_row.row == 42
    assert loc_row.column == "status"

    loc_bbox = LocationRef(bbox=(10.0, 20.0, 100.0, 50.0))
    assert loc_bbox.bbox == (10.0, 20.0, 100.0, 50.0)

    loc_span = LocationRef(char_span=(12, 45))
    assert loc_span.char_span == (12, 45)


def test_claim_field_bounds_and_defaults() -> None:
    """Verify confidence and trust score bounds (0.0 to 1.0) and defaults."""
    loc = LocationRef(page=1)

    claim = Claim(
        claim_id="CLM-01",
        entity_raw="Apex Logistics",
        attribute="approved_amount",
        value="$12,000",
        numeric_value=12000.0,
        unit="USD",
        modality=Modality.PDF,
        source_id="SRC-402",
        location_ref=loc,
        extraction_confidence=0.95,
    )
    assert claim.trust_score == 0.5
    assert claim.is_amendment is False
    assert claim.entity_canonical is None

    # extraction_confidence > 1.0 should fail
    with pytest.raises(ValidationError):
        Claim(
            claim_id="CLM-02",
            entity_raw="Apex Logistics",
            attribute="approved_amount",
            value="$12,000",
            modality=Modality.PDF,
            source_id="SRC-402",
            location_ref=loc,
            extraction_confidence=1.5,
        )

    # trust_score < 0.0 should fail
    with pytest.raises(ValidationError):
        Claim(
            claim_id="CLM-03",
            entity_raw="Apex Logistics",
            attribute="approved_amount",
            value="$12,000",
            modality=Modality.PDF,
            source_id="SRC-402",
            location_ref=loc,
            extraction_confidence=0.9,
            trust_score=-0.1,
        )


def test_verdict_response_serialization_round_trip() -> None:
    """Verify VerdictResponse can serialize to JSON and deserialize cleanly."""
    loc = LocationRef(row=2, column="amount")
    flag = ForensicFlag(
        code="ELA_TAMPER_DETECTED",
        severity=ForensicSeverity.CRITICAL,
        detail="JPEG compression discrepancy in total patch",
        location=LocationRef(bbox=(100.0, 200.0, 250.0, 240.0)),
    )
    source = SourceDoc(
        source_id="SRC-TAMPERED-01",
        file_path="data/synthetic_traps/receipt_scan_tampered.jpg",
        file_name="receipt_scan_tampered.jpg",
        modality=Modality.IMAGE,
        sha256="abc123def456",
        trust_prior=0.2,
        forensic_flags=[flag],
        quarantined=True,
    )
    claim_a = Claim(
        claim_id="CLM-A",
        entity_raw="INV-402",
        attribute="amount",
        value="$12,000",
        numeric_value=12000.0,
        modality=Modality.TABLE,
        source_id="SRC-LEDGER",
        location_ref=loc,
        extraction_confidence=1.0,
        trust_score=0.9,
    )
    claim_b = Claim(
        claim_id="CLM-B",
        entity_raw="INV-402",
        attribute="amount",
        value="$18,000",
        numeric_value=18000.0,
        modality=Modality.IMAGE,
        source_id="SRC-TAMPERED-01",
        location_ref=LocationRef(bbox=(100.0, 200.0, 250.0, 240.0)),
        extraction_confidence=0.8,
        trust_score=0.1,
    )
    conflict = ConflictEdge(
        edge_id="EDGE-01",
        claim_a_id="CLM-A",
        claim_b_id="CLM-B",
        edge_type=EdgeType.CONTRADICTS,
        confidence=0.95,
        reason="Numeric mismatch between ledger and tampered receipt",
        detector="numeric_comparator",
    )
    citation = Citation(
        source_id="SRC-LEDGER",
        file_name="ledger.csv",
        claim_id="CLM-A",
        location_ref=loc,
        quote="INV-402,12000,Apex Logistics",
    )
    dossier = EvidenceDossier(
        query="What is the approved amount for invoice INV-402?",
        target_entities=["INV-402"],
        verified_claims=[claim_a],
        conflicting_edges=[conflict],
        quarantined_sources=[source],
        max_trust_score=0.9,
    )
    response = VerdictResponse(
        query="What is the approved amount for invoice INV-402?",
        status=VerdictStatus.CONFLICT_RESOLVED,
        answer="The approved amount for INV-402 is $12,000.",
        confidence=0.92,
        citations=[citation],
        conflicts=[conflict],
        quarantined_sources=[source],
        dossier=dossier,
    )

    json_data = response.model_dump_json()
    reconstructed = VerdictResponse.model_validate_json(json_data)
    assert reconstructed.query == response.query
    assert reconstructed.status == VerdictStatus.CONFLICT_RESOLVED
    assert len(reconstructed.citations) == 1
    assert reconstructed.citations[0].claim_id == "CLM-A"
    assert reconstructed.dossier is not None
    assert reconstructed.dossier.verified_claims[0].numeric_value == 12000.0


def test_synthetic_traps_exist_and_non_empty() -> None:
    """Verify all 6 synthetic trap files exist and are non-empty."""
    trap_dir = Path("data/synthetic_traps")
    expected_files = [
        "ledger.csv",
        "approval_email.pdf",
        "amendment_memo.pdf",
        "receipt_scan_tampered.jpg",
        "vendor_note_injected.pdf",
        "wire_scan_blurred.png",
    ]
    for filename in expected_files:
        file_path = trap_dir / filename
        assert file_path.exists(), f"Missing trap file: {filename}"
        assert file_path.stat().st_size > 0, f"Trap file is empty: {filename}"


def test_vendor_note_hidden_injection_detected() -> None:
    """Verify vendor_note_injected.pdf contains white-on-white tiny font text."""
    import pymupdf

    doc_path = Path("data/synthetic_traps/vendor_note_injected.pdf")
    assert doc_path.exists()

    doc = pymupdf.open(doc_path)
    found_injection = False
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if "IGNORE ALL PREVIOUS INSTRUCTIONS" in span["text"]:
                        found_injection = True
                        assert span["size"] <= 4
                        rgb_color = pymupdf.sRGB_to_rgb(span["color"])
                        assert rgb_color == (255, 255, 255)
    doc.close()
    assert found_injection, "Hidden prompt injection span was not detected in PDF dict."


def test_ledger_csv_rows_and_approver_blank() -> None:
    """Verify ledger.csv contains required entries and blank approver for WIRE-Q4."""
    import csv

    ledger_path = Path("data/synthetic_traps/ledger.csv")
    assert ledger_path.exists()

    with open(ledger_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3
    inv_402 = next(r for r in reader if r["invoice_id"] == "INV-402")
    assert inv_402["amount"] == "12000"
    assert inv_402["vendor"] == "Apex Logistics"
    assert inv_402["status"] == "Approved"

    inv_510 = next(r for r in reader if r["invoice_id"] == "INV-510")
    assert inv_510["amount"] == "4500"
    assert inv_510["vendor"] == "Helios Labs"

    wire_q4 = next(r for r in reader if r["invoice_id"] == "WIRE-Q4")
    assert wire_q4["amount"] == "95000"
    assert wire_q4["approver"] == ""  # blank approver


def test_amendment_memo_supersedes_clause() -> None:
    """Verify amendment_memo.pdf contains $5,200 amendment and superseding clause."""
    import pymupdf

    memo_path = Path("data/synthetic_traps/amendment_memo.pdf")
    assert memo_path.exists()

    doc = pymupdf.open(memo_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    assert "INV-510" in full_text
    assert "$5,200" in full_text
    assert "supersedes" in full_text.lower()


def test_eval_test_cases_schema_and_integrity() -> None:
    """Verify eval/test_cases.json has exactly 5 valid benchmark cases."""
    eval_file = Path("eval/test_cases.json")
    assert eval_file.exists()
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    assert isinstance(cases, list)
    assert len(cases) == 5

    valid_statuses = {s.value for s in VerdictStatus}
    case_ids = set()
    for case in cases:
        assert "case_id" in case
        assert "query" in case
        assert "target_entity" in case
        assert "expected_status" in case
        assert case["expected_status"] in valid_statuses
        case_ids.add(case["case_id"])

    assert len(case_ids) == 5

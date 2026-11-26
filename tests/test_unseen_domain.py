"""Open-domain unseen dataset tests covering deterministic fallback and mocked Ollama."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pymupdf

import app
from core.extractor import extract_and_persist, extract_claims, resolve_entities
from core.graph_engine import build_and_persist_graph
from core.ingest import ingest_directory
from core.reasoner import answer_query
from core.schemas import EdgeType, VerdictStatus


def _create_unseen_shipment_corpus(corpus_dir: Path) -> None:
    """Create a non-invoice shipment manifest corpus (CSV + PDF + tampered JPEG)."""
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # 1. manifest.csv with non-invoice ID column `shipment_id` and missing `inspector`
    with (corpus_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shipment_id", "weight", "carrier", "date", "inspector", "status"])
        writer.writerow(["SHIP-884", "420 kg", "Nordic Freight", "2026-05-01", "", "InTransit"])
        writer.writerow(["BATCH-19", "150 kg", "Atlas Cargo", "2026-05-02", "K. Patel", "Cleared"])

    # 2. dispatch_memo.pdf corroborating SHIP-884 weight & carrier and BATCH-19 status
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    page.insert_text(
        pymupdf.Point(50, 72),
        (
            "LOGISTICS DISPATCH CONFIRMATION\n"
            "Date: 2026-05-01\n"
            "Shipment: SHIP-884\n"
            "Carrier: Nordic Freight\n"
            "Weight: 420 kg\n\n"
            "Secondary Notice: BATCH-19\n"
            "Status: Cleared\n"
        ),
        fontsize=11,
        color=(0.1, 0.1, 0.1),
    )
    pdf.save(str(corpus_dir / "dispatch_memo.pdf"))
    pdf.close()

    # 3. tampered_bol.jpg spliced with forged `990 kg` patch for ELA detection
    base = Image.new("RGB", (700, 420), color=(248, 248, 246))
    draw = ImageDraw.Draw(base)
    draw.text((40, 40), "BILL OF LADING", fill=(20, 20, 20))
    draw.text((40, 90), "Shipment: SHIP-884", fill=(30, 30, 30))
    draw.text((40, 140), "Carrier: Nordic Freight", fill=(30, 30, 30))
    draw.text((40, 200), "Cargo Weight:", fill=(20, 20, 20))
    draw.text((380, 200), "420 kg", fill=(20, 20, 20))

    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=65)
    buf.seek(0)
    spliced = Image.open(buf).copy()
    sdraw = ImageDraw.Draw(spliced)
    sdraw.rectangle([(360, 190), (540, 235)], fill=(248, 248, 246))
    sdraw.text((380, 200), "990 kg", fill=(10, 10, 10))
    spliced.save(str(corpus_dir / "tampered_bol.jpg"), format="JPEG", quality=98)


def test_unseen_shipment_corpus_deterministic_mode(tmp_path: Path) -> None:
    """Verify open-domain shipment manifest works end-to-end with use_llm=False."""
    corpus_dir = tmp_path / "corpus"
    storage_dir = tmp_path / "storage"
    _create_unseen_shipment_corpus(corpus_dir)

    summary = app.run_full_pipeline(input_dir=corpus_dir, storage_dir=storage_dir, use_llm=False)
    assert summary["doc_count"] == 3
    assert summary["claim_count"] > 0

    entities = json.loads((storage_dir / "entities.json").read_text(encoding="utf-8"))
    assert "ENT_SHIP_884" in entities
    assert "ENT_BATCH_19" in entities

    # 1. Cross-modal conflict on cargo weight: 420 kg vs quarantined 990 kg
    w_resp = answer_query("What is the cargo weight for SHIP-884?", storage_dir=storage_dir)
    assert w_resp.status == VerdictStatus.CONFLICT_RESOLVED
    assert "420" in w_resp.answer
    assert any(s.file_name == "tampered_bol.jpg" for s in w_resp.quarantined_sources)
    assert any(e.edge_type == EdgeType.CONTRADICTS for e in w_resp.conflicts)

    # 2. Clean corroborated carrier lookup
    c_resp = answer_query("Who is the carrier for SHIP-884?", storage_dir=storage_dir)
    assert c_resp.status == VerdictStatus.VERIFIED
    assert "Nordic Freight" in c_resp.answer

    # 3. Missing inspector triggers deterministic INSUFFICIENT_EVIDENCE
    i_resp = answer_query("Who is the inspector for SHIP-884?", storage_dir=storage_dir)
    assert i_resp.status == VerdictStatus.INSUFFICIENT_EVIDENCE
    assert i_resp.dossier is not None
    assert "inspector" in i_resp.dossier.missing_attributes


def test_unseen_corpus_mocked_ollama_schema_extraction_and_synthesis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify schema-constrained Ollama extraction binds provenance and locks VerdictStatus."""
    corpus_dir = tmp_path / "corpus"
    storage_dir = tmp_path / "storage"
    _create_unseen_shipment_corpus(corpus_dir)
    docs, chunks = ingest_directory(corpus_dir, storage_dir=storage_dir)

    class _FakeOllamaClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
            user_prompt = messages[0]["content"]
            if '"verified_claims"' in user_prompt:
                return {
                    "message": {
                        "content": "Verified carrier for ENT_SHIP_884 is Nordic Freight [manifest.csv:row=1] [dispatch_memo.pdf:p1]."
                    }
                }
            # Schema-constrained extraction mock based on chunk text
            if "LOGISTICS DISPATCH CONFIRMATION" in user_prompt:
                payload = {
                    "facts": [
                        {
                            "chunk_index": 0,
                            "entity": "SHIP-884",
                            "attribute": "carrier",
                            "value": "Nordic Freight",
                            "numeric_value": None,
                            "unit": None,
                            "timestamp": "2026-05-01",
                            "is_amendment": False,
                        },
                        {
                            "chunk_index": 0,
                            "entity": "SHIP-884",
                            "attribute": "weight",
                            "value": "420 kg",
                            "numeric_value": 420.0,
                            "unit": "kg",
                            "timestamp": "2026-05-01",
                            "is_amendment": False,
                        },
                        {
                            "chunk_index": 1,
                            "entity": "BATCH-19",
                            "attribute": "status",
                            "value": "Cleared",
                            "numeric_value": None,
                            "unit": None,
                            "timestamp": "2026-05-02",
                            "is_amendment": False,
                        },
                    ]
                }
            else:
                payload = {"facts": []}
            return {"message": {"content": json.dumps(payload)}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=_FakeOllamaClient))

    claims, entities = extract_and_persist(storage_dir=storage_dir, use_llm=True)
    assert "ENT_SHIP_884" in entities
    assert "ENT_BATCH_19" in entities
    assert any(c.attribute == "weight" and c.numeric_value == 420.0 for c in claims)

    build_and_persist_graph(storage_dir=storage_dir)
    resp = answer_query(
        "Who is the carrier for SHIP-884?",
        storage_dir=storage_dir,
        use_llm=True,
    )
    assert resp.status == VerdictStatus.VERIFIED
    assert "Nordic Freight" in resp.answer
    assert len(resp.citations) >= 1


def test_3b_llm_grounding_normalizes_generic_entities_and_noisy_attrs(
    monkeypatch,
) -> None:
    """Verify post-LLM normalizer fixes generic-noun entities, fallback attrs, and From: emails."""
    from core.config import settings
    from core.schemas import LocationRef, Modality, ParsedChunk, SourceDoc

    clean_pdf = SourceDoc(
        source_id="src_email",
        file_path="approval_email.pdf",
        file_name="approval_email.pdf",
        modality=Modality.PDF,
        sha256="aaa",
        trust_prior=0.75,
        quarantined=False,
    )
    quarantined_note = SourceDoc(
        source_id="src_qnote",
        file_path="vendor_note_injected.pdf",
        file_name="vendor_note_injected.pdf",
        modality=Modality.PDF,
        sha256="bbb",
        trust_prior=0.05,
        quarantined=True,
    )
    tampered_jpg = SourceDoc(
        source_id="src_tamper",
        file_path="receipt_scan_tampered.jpg",
        file_name="receipt_scan_tampered.jpg",
        modality=Modality.IMAGE,
        sha256="ccc",
        trust_prior=0.15,
        quarantined=True,
    )

    chunks = [
        ParsedChunk(
            chunk_id="c_email_0",
            source_id="src_email",
            modality=Modality.PDF,
            text="From: finance-approvals@corp-audit.local\nSubject: Invoice 402",
            location_ref=LocationRef(page=1, bbox=(50.0, 60.0, 300.0, 90.0)),
            extraction_confidence=0.95,
            sanitized=True,
        ),
        ParsedChunk(
            chunk_id="c_email_1",
            source_id="src_email",
            modality=Modality.PDF,
            text="Invoice 402 approved at $12,000 on 2026-03-10.",
            location_ref=LocationRef(page=1, bbox=(50.0, 120.0, 300.0, 150.0)),
            extraction_confidence=0.95,
            sanitized=True,
        ),
        ParsedChunk(
            chunk_id="c_qnote_0",
            source_id="src_qnote",
            modality=Modality.PDF,
            text="VENDOR SHIPMENT ADVICE\nRelated Invoice: INV-402\nDate: 2026-03-11",
            location_ref=LocationRef(page=1, bbox=(50.0, 60.0, 250.0, 120.0)),
            extraction_confidence=0.95,
            sanitized=True,
        ),
        ParsedChunk(
            chunk_id="c_qnote_1",
            source_id="src_qnote",
            modality=Modality.PDF,
            text="Field Representative Log: Delivery completed cleanly.",
            location_ref=LocationRef(page=1, bbox=(50.0, 140.0, 350.0, 180.0)),
            extraction_confidence=0.95,
            sanitized=True,
        ),
        ParsedChunk(
            chunk_id="c_tamper_0",
            source_id="src_tamper",
            modality=Modality.IMAGE,
            text="InvoiceReference:Inv#402",
            location_ref=LocationRef(bbox=(40.0, 120.0, 170.0, 132.0)),
            extraction_confidence=0.90,
            sanitized=True,
        ),
        ParsedChunk(
            chunk_id="c_tamper_1",
            source_id="src_tamper",
            modality=Modality.IMAGE,
            text="$18,000.00",
            location_ref=LocationRef(bbox=(479.0, 329.0, 535.0, 343.0)),
            extraction_confidence=0.82,
            sanitized=True,
        ),
    ]

    class _Noisy3BClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
            prompt = messages[0]["content"]
            if "finance-approvals@corp-audit.local" in prompt:
                facts = [
                    {"chunk_index": 0, "entity": "Invoice 402", "attribute": "approver", "value": "finance-approvals@corp-audit.local"},
                    {"chunk_index": 1, "entity": "Invoice 402", "attribute": "fallback_entity", "value": "Invoice 402"},
                    {"chunk_index": 1, "entity": "Invoice 402", "attribute": "amount", "value": "$12,000", "numeric_value": 12000.0, "unit": "USD"},
                ]
            elif "VENDOR SHIPMENT ADVICE" in prompt:
                facts = [
                    {"chunk_index": 0, "entity": "invoice", "attribute": "reference", "value": "inv-402"},
                    {"chunk_index": 0, "entity": "vendor", "attribute": "name", "value": "Apex Logistics"},
                    {"chunk_index": 0, "entity": "document", "attribute": "type", "value": "delivery note"},
                ]
            else:
                facts = [
                    {"chunk_index": 1, "entity": "amount", "attribute": "amount", "value": "$18,000.00", "numeric_value": 18000.0, "unit": "USD", "is_amendment": True},
                ]
            return {"message": {"content": json.dumps({"facts": facts})}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=_Noisy3BClient))

    raw_claims = extract_claims([clean_pdf, quarantined_note, tampered_jpg], chunks, use_llm=True)
    claims, entity_map = resolve_entities(raw_claims)

    for bad_ent in ("ENT_AMOUNT", "ENT_VENDOR", "ENT_DOCUMENT", "ENT_INVOICE", "ENT_DATE"):
        assert bad_ent not in entity_map

    assert "ENT_INV_402" in entity_map
    assert not any(c.attribute == "fallback_entity" for c in claims)
    assert not any(c.attribute == "approver" and "@" in c.value for c in claims)

    tampered_claims = [c for c in claims if c.source_id == "src_tamper" and c.attribute == "amount"]
    assert len(tampered_claims) == 1
    assert tampered_claims[0].entity_canonical == "ENT_INV_402"
    assert tampered_claims[0].is_amendment is False

    qnote_notes = [c for c in claims if c.source_id == "src_qnote" and c.attribute == "vendor_notes"]
    assert len(qnote_notes) >= 1
    assert qnote_notes[0].entity_canonical == "ENT_INV_402"


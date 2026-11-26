"""Tests for offline Streamlit app helpers, visual proof rendering, and PyVis HTML."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import TypeAdapter

import app
from core.config import settings
from core.graph_engine import load_graph_artifacts
from core.reasoner import answer_query
from core.schemas import LocationRef, SourceDoc, VerdictStatus


def test_render_visual_proof_image_pdf_and_table() -> None:
    """Verify visual proof crops render PIL images for IMAGE/PDF and None for TABLE."""
    docs = TypeAdapter(list[SourceDoc]).validate_json(
        (settings.storage_dir / "ingested_docs.json").read_text(encoding="utf-8")
    )
    by_name = {d.file_name: d for d in docs}

    img_out = app.render_visual_proof(
        by_name["receipt_scan_tampered.jpg"],
        LocationRef(bbox=(479.0, 329.0, 535.0, 343.0)),
    )
    assert isinstance(img_out, Image.Image)
    assert img_out.size == (700, 500)

    pdf_out = app.render_visual_proof(
        by_name["approval_email.pdf"],
        LocationRef(page=1, bbox=(50.0, 135.7, 407.0, 181.1)),
    )
    assert isinstance(pdf_out, Image.Image)
    assert pdf_out.width > 0 and pdf_out.height > 0

    tbl_out = app.render_visual_proof(
        by_name["ledger.csv"],
        LocationRef(row=1, column="ALL"),
    )
    assert tbl_out is None


def test_build_offline_pyvis_html_contains_no_cdn_urls() -> None:
    """Verify PyVis generates self-contained offline HTML without external CDN links."""
    graph, _, _ = load_graph_artifacts(settings.storage_dir)
    html = app.build_offline_pyvis_html(graph, highlight_entities=["ENT_INV_402"])

    assert isinstance(html, str)
    assert len(html) > 1000
    assert "ENT_INV_402" in html
    assert "receipt_scan_tampered.jpg" in html
    assert "https://cdnjs.cloudflare.com" not in html
    assert "https://cdn.jsdelivr.net" not in html
    assert "http://cdnjs.cloudflare.com" not in html
    assert "http://cdn.jsdelivr.net" not in html


def test_run_full_pipeline_and_query_end_to_end(tmp_path: Path) -> None:
    """Verify run_full_pipeline executes all backend stages into tmp_path cleanly."""
    summary = app.run_full_pipeline(
        input_dir=settings.synthetic_traps_dir,
        storage_dir=tmp_path,
        use_llm=False,
    )
    assert summary["doc_count"] == 6
    assert summary["claim_count"] == 30
    assert summary["epistemic_edge_count"] >= 15

    resp = answer_query(
        "What is the approved amount for invoice INV-402?",
        storage_dir=tmp_path,
    )
    assert resp.status == VerdictStatus.CONFLICT_RESOLVED
    assert "$12,000" in resp.answer

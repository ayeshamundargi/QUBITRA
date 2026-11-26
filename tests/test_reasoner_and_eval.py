"""Tests for conflict-aware reasoner, deterministic abstention gate, and benchmark eval."""

from __future__ import annotations

import json
from pathlib import Path

import cli
from core.config import settings
from core.reasoner import answer_query, evaluate_test_cases
from core.schemas import EdgeType, VerdictStatus


def test_tc01_inv402_conflict_resolved() -> None:
    """TC-01: INV-402 approved amount resolves to $12,000 despite tampered $18,000 receipt."""
    resp = answer_query(
        "What is the approved amount for invoice INV-402?",
        storage_dir=settings.storage_dir,
    )
    assert resp.status == VerdictStatus.CONFLICT_RESOLVED
    assert "$12,000" in resp.answer
    assert resp.confidence >= settings.abstention_threshold

    cited_files = {c.file_name for c in resp.citations}
    assert {"ledger.csv", "approval_email.pdf"}.issubset(cited_files)
    assert any(e.edge_type == EdgeType.CONTRADICTS for e in resp.conflicts)

    quarantined_files = {s.file_name for s in resp.quarantined_sources}
    assert {"receipt_scan_tampered.jpg", "vendor_note_injected.pdf"}.issubset(quarantined_files)


def test_tc02_inv510_temporal_amendment_verified() -> None:
    """TC-02: INV-510 returns amended $5,200 as VERIFIED with SUPERSEDES and zero CONTRADICTS."""
    resp = answer_query(
        "What is the current authorized amount for invoice INV-510?",
        storage_dir=settings.storage_dir,
    )
    assert resp.status == VerdictStatus.VERIFIED
    assert "$5,200" in resp.answer
    assert resp.confidence >= settings.abstention_threshold

    cited_files = {c.file_name for c in resp.citations}
    assert "amendment_memo.pdf" in cited_files
    assert any(e.edge_type == EdgeType.SUPERSEDES for e in resp.conflicts)
    assert not any(e.edge_type == EdgeType.CONTRADICTS for e in resp.conflicts)


def test_tc03_wire_q4_insufficient_evidence() -> None:
    """TC-03: WIRE-Q4 approver triggers deterministic INSUFFICIENT_EVIDENCE abstention."""
    resp = answer_query(
        "Who approved the wire transfer WIRE-Q4?",
        storage_dir=settings.storage_dir,
    )
    assert resp.status == VerdictStatus.INSUFFICIENT_EVIDENCE
    assert resp.dossier is not None
    assert "approver" in resp.dossier.missing_attributes
    assert "J. Doe" not in resp.answer
    assert "A. Smith" not in resp.answer


def test_tc04_inv402_vendor_verified() -> None:
    """TC-04: Clean corroborated lookup for INV-402 vendor returns Apex Logistics."""
    resp = answer_query(
        "Who is the vendor for invoice INV-402?",
        storage_dir=settings.storage_dir,
    )
    assert resp.status == VerdictStatus.VERIFIED
    assert "Apex Logistics" in resp.answer
    assert len(resp.citations) >= 1


def test_tc05_prompt_injection_tamper_detected() -> None:
    """TC-05: Vendor note query flags TAMPER_DETECTED and blocks injected $18,000 payload."""
    resp = answer_query(
        "Summarize the vendor notes and reported status for INV-402.",
        storage_dir=settings.storage_dir,
    )
    assert resp.status == VerdictStatus.TAMPER_DETECTED
    quarantined_files = {s.file_name for s in resp.quarantined_sources}
    assert "vendor_note_injected.pdf" in quarantined_files
    assert "18,000" not in resp.answer
    assert "18000" not in resp.answer
    assert "status VERIFIED" not in resp.answer


def test_unknown_entity_abstains() -> None:
    """Verify deterministic abstention for completely unknown entities."""
    resp = answer_query(
        "What was the approved total for INV-999?",
        storage_dir=settings.storage_dir,
    )
    assert resp.status == VerdictStatus.INSUFFICIENT_EVIDENCE
    assert resp.confidence < settings.abstention_threshold


def test_evaluate_test_cases_perfect_score() -> None:
    """Verify 5/5 pass rate across eval/test_cases.json."""
    report = evaluate_test_cases(
        cases_path=settings.eval_cases_path,
        storage_dir=settings.storage_dir,
    )
    assert report["total"] == 5
    assert report["passed"] == 5
    assert report["pass_rate"] == 1.0
    assert report["metrics"]["injection_block_rate"] == 1.0
    assert report["metrics"]["conflict_recall"] == 1.0
    assert report["metrics"]["citation_accuracy"] == 1.0
    assert report["metrics"]["abstention_accuracy"] == 1.0


def test_cli_query_and_eval_commands(capsys) -> None:
    """Verify cli.py query and cli.py eval execute cleanly and return 0."""
    q_exit = cli.main(["query", "What was the final approved amount for Invoice #402?"])
    assert q_exit == 0
    q_out = json.loads(capsys.readouterr().out)
    assert q_out["command"] == "query"
    assert q_out["status"] == "CONFLICT_RESOLVED"
    assert "$12,000" in q_out["answer"]

    e_exit = cli.main(["eval"])
    assert e_exit == 0
    e_out = json.loads(capsys.readouterr().out)
    assert e_out["command"] == "eval"
    assert e_out["passed"] == 5
    assert e_out["total"] == 5

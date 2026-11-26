"""Tests for NetworkX evidence graph, conflict/supersession detection, and trust propagation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

import cli
from core.config import settings
from core.graph_engine import (
    EvidenceGraphEngine,
    build_and_persist_graph,
    detect_claim_relations,
    get_entity_subgraph,
    load_graph_artifacts,
    propagate_trust,
)
from core.schemas import Claim, ConflictEdge, EdgeType, SourceDoc


def _load_step4_artifacts() -> tuple[list[SourceDoc], list[Claim], dict[str, list[str]]]:
    docs = TypeAdapter(list[SourceDoc]).validate_json(
        (settings.storage_dir / "ingested_docs.json").read_text(encoding="utf-8")
    )
    claims = TypeAdapter(list[Claim]).validate_json(
        (settings.storage_dir / "claims.json").read_text(encoding="utf-8")
    )
    entities = json.loads((settings.storage_dir / "entities.json").read_text(encoding="utf-8"))
    return docs, claims, entities


def test_inv402_conflict_and_trust_separation() -> None:
    """Verify INV-402 corroboration boosts $12k claims >= 0.85 while tampered $18k stays <= 0.15."""
    docs, claims, entities = _load_step4_artifacts()
    docs_by_id = {d.source_id: d for d in docs}
    file_by_id = {d.source_id: d.file_name for d in docs}

    engine = EvidenceGraphEngine()
    engine.build_base_graph(docs=docs, claims=claims, entities=entities)
    edges = detect_claim_relations(claims, docs_by_id=docs_by_id, use_nli=False)
    engine.add_epistemic_edges(edges)

    _, scored_claims = propagate_trust(engine.graph, docs=docs, claims=claims, edges=edges)
    scored_by_id = {c.claim_id: c for c in scored_claims}

    inv402_amounts = [
        c for c in scored_claims if c.entity_canonical == "ENT_INV_402" and c.attribute == "amount"
    ]
    clean_12k = [
        c
        for c in inv402_amounts
        if c.numeric_value == 12000.0
        and file_by_id[c.source_id] in {"ledger.csv", "approval_email.pdf"}
    ]
    tampered_18k = [
        c
        for c in inv402_amounts
        if c.numeric_value == 18000.0 and file_by_id[c.source_id] == "receipt_scan_tampered.jpg"
    ]

    assert len(clean_12k) == 2
    assert len(tampered_18k) == 1

    # SUPPORTS edge between ledger.csv and approval_email.pdf $12,000 claims
    clean_ids = {c.claim_id for c in clean_12k}
    supports_12k = [
        e
        for e in edges
        if e.edge_type == EdgeType.SUPPORTS
        and {e.claim_a_id, e.claim_b_id} == clean_ids
    ]
    assert len(supports_12k) == 1

    # CONTRADICTS edges from tampered $18,000 against both $12,000 claims
    tampered_id = tampered_18k[0].claim_id
    contradicts_18k = [
        e
        for e in edges
        if e.edge_type == EdgeType.CONTRADICTS
        and e.detector == "numeric"
        and tampered_id in {e.claim_a_id, e.claim_b_id}
        and bool(clean_ids & {e.claim_a_id, e.claim_b_id})
    ]
    assert len(contradicts_18k) == 2

    # Post-propagation trust separation
    assert all(scored_by_id[cid].trust_score >= 0.85 for cid in clean_ids)
    assert scored_by_id[tampered_id].trust_score <= 0.15


def test_inv510_temporal_amendment_no_false_conflict() -> None:
    """Verify INV-510 $5,200 supersedes both $4,500 claims with zero CONTRADICTS edges."""
    docs, claims, entities = _load_step4_artifacts()
    docs_by_id = {d.source_id: d for d in docs}
    file_by_id = {d.source_id: d.file_name for d in docs}

    engine = EvidenceGraphEngine()
    engine.build_base_graph(docs=docs, claims=claims, entities=entities)
    edges = detect_claim_relations(claims, docs_by_id=docs_by_id, use_nli=False)
    engine.add_epistemic_edges(edges)

    _, scored_claims = propagate_trust(engine.graph, docs=docs, claims=claims, edges=edges)

    inv510_amounts = [
        c for c in scored_claims if c.entity_canonical == "ENT_INV_510" and c.attribute == "amount"
    ]
    inv510_ids = {c.claim_id for c in inv510_amounts}
    older_4500 = [
        c
        for c in inv510_amounts
        if c.numeric_value == 4500.0
        and file_by_id[c.source_id] in {"ledger.csv", "approval_email.pdf"}
    ]
    newer_5200 = [
        c
        for c in inv510_amounts
        if c.numeric_value == 5200.0 and file_by_id[c.source_id] == "amendment_memo.pdf"
    ]

    assert len(older_4500) == 2
    assert len(newer_5200) >= 1

    # Zero CONTRADICTS edges on ENT_INV_510 amount
    inv510_contradictions = [
        e
        for e in edges
        if e.edge_type == EdgeType.CONTRADICTS
        and (e.claim_a_id in inv510_ids or e.claim_b_id in inv510_ids)
    ]
    assert inv510_contradictions == []

    # Every older $4,500 claim must be superseded by newer $5,200
    newer_ids = {c.claim_id for c in newer_5200}
    for old_claim in older_4500:
        assert any(
            e.edge_type == EdgeType.SUPERSEDES
            and e.claim_a_id in newer_ids
            and e.claim_b_id == old_claim.claim_id
            for e in edges
        )
        assert engine.graph.nodes[old_claim.claim_id]["superseded"] is True

    for new_claim in newer_5200:
        assert engine.graph.nodes[new_claim.claim_id]["superseded"] is False
        assert new_claim.trust_score >= 0.65


def test_wire_q4_abstention_readiness_no_false_conflicts() -> None:
    """Verify ENT_WIRE_Q4 has amount/vendor claims, zero approvers, and zero false conflicts."""
    docs, claims, entities = _load_step4_artifacts()
    docs_by_id = {d.source_id: d for d in docs}

    engine = EvidenceGraphEngine()
    engine.build_base_graph(docs=docs, claims=claims, entities=entities)
    edges = detect_claim_relations(claims, docs_by_id=docs_by_id, use_nli=False)
    engine.add_epistemic_edges(edges)

    subgraph = get_entity_subgraph(engine.graph, "ENT_WIRE_Q4", hops=2)
    wire_claims: list[Claim] = subgraph["claims"]
    wire_edges: list[ConflictEdge] = subgraph["edges"]

    attrs = {c.attribute for c in wire_claims}
    assert "amount" in attrs
    assert "vendor" in attrs
    assert "approver" not in attrs
    assert not any(e.edge_type == EdgeType.CONTRADICTS for e in wire_edges)


def test_subgraph_and_cli_build_graph_round_trip(tmp_path: Path, capsys) -> None:
    """Verify cli.py build-graph persists graph/claims/edges and reloads identically."""
    for fname in ("ingested_docs.json", "claims.json", "entities.json"):
        (tmp_path / fname).write_text(
            (settings.storage_dir / fname).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    exit_code = cli.main(["build-graph", "--storage", str(tmp_path)])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "build-graph"
    assert payload["status"] == "ok"
    assert payload["edge_counts"]["CONTRADICTS"] >= 2
    assert payload["edge_counts"]["SUPERSEDES"] >= 2

    graph_path = tmp_path / "evidence_graph.json"
    claims_path = tmp_path / "claims_scored.json"
    edges_path = tmp_path / "conflict_edges.json"
    assert graph_path.exists()
    assert claims_path.exists()
    assert edges_path.exists()

    loaded_graph, loaded_claims, loaded_edges = load_graph_artifacts(tmp_path)
    assert loaded_graph.number_of_nodes() == payload["total_nodes"]
    assert loaded_graph.number_of_edges() == payload["total_edges"]
    assert len(loaded_claims) == payload["claim_count"]
    assert len(loaded_edges) == payload["epistemic_edge_count"]

    inv402_sub = get_entity_subgraph(loaded_graph, "ENT_INV_402", hops=2)
    assert any(s.quarantined for s in inv402_sub["sources"])
    assert any(e.edge_type == EdgeType.CONTRADICTS for e in inv402_sub["edges"])

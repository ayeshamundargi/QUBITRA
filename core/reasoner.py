"""Conflict-aware reasoner, deterministic abstention gate, and cited synthesizer.

All `VerdictStatus` decisions (`VERIFIED`, `CONFLICT_RESOLVED`, `TAMPER_DETECTED`,
`INSUFFICIENT_EVIDENCE`) are enforced deterministically in Python inside
`evaluate_verdict()` using `settings.abstention_threshold` (`0.65`), never
delegated to LLM prompting.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from rapidfuzz import fuzz

from core.config import settings
from core.graph_engine import get_entity_subgraph, load_graph_artifacts
from core.schemas import (
    Citation,
    Claim,
    ConflictEdge,
    EdgeType,
    EvidenceDossier,
    LocationRef,
    SourceDoc,
    VerdictResponse,
    VerdictStatus,
)

_INVOICE_ID_RE = re.compile(r"\binv(?:oice)?\s*[-#:]?\s*(\d+)\b", re.IGNORECASE)
_WIRE_ID_RE = re.compile(r"\bwire\s*[-#:]?\s*(q\d+|[a-z]+\d+|\d+)\b", re.IGNORECASE)
_Q_WIRE_RE = re.compile(r"\b(q\d+)\s+wire\b", re.IGNORECASE)
_GENERIC_ID_RE = re.compile(r"\b([A-Za-z]{2,8})[-#](\d{1,6})\b")

_ATTR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "vendor_notes",
        ("vendor note", "vendor notes", "vendor_notes", "delivery note", "shipment advice", "payment instruction"),
    ),
    (
        "approver",
        ("who approved", "approver", "approved by", "signature", "authorized by", "signer"),
    ),
    (
        "amount",
        ("amount", "total", "cost", "price", "how much", "sum", "expenditure", "payable", "cap"),
    ),
    (
        "vendor",
        ("vendor", "supplier", "payee", "beneficiary", "company"),
    ),
    (
        "date",
        ("date", "when", "timestamp"),
    ),
    (
        "status",
        ("status", "state"),
    ),
)


def _detect_target_attribute(query: str) -> str | None:
    """Detect requested attribute intent with specificity ordering."""
    lowered = query.lower()
    for attr, keywords in _ATTR_RULES:
        if any(kw in lowered for kw in keywords):
            return attr
    return None


def _try_chroma_entity_hints(query: str, storage_dir: Path) -> list[str]:
    """Optional strictly offline ChromaDB retrieval hook; returns entity IDs if indexed."""
    chroma_dir = storage_dir / "chromadb"
    if not chroma_dir.exists():
        return []
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collections = client.list_collections()
        if not collections:
            return []
        coll = client.get_collection(collections[0].name)
        res = coll.query(query_texts=[query], n_results=3)
        metas = (res.get("metadatas") or [[]])[0]
        hints: list[str] = []
        for meta in metas:
            if isinstance(meta, dict) and meta.get("entity_canonical"):
                hints.append(str(meta["entity_canonical"]))
        return hints
    except Exception:
        return []


def _resolve_query_entities(
    query: str,
    entities: dict[str, list[str]],
    claims: list[Claim],
    storage_dir: Path,
    use_vector: bool = False,
) -> list[str]:
    """Resolve target canonical entity IDs from query via regex + RapidFuzz + optional Chroma."""
    matched: list[str] = []
    known_entities = set(entities.keys()) | {
        c.entity_canonical for c in claims if c.entity_canonical
    }

    # 1. Regex structured ID resolution
    for m in _INVOICE_ID_RE.finditer(query):
        cid = f"ENT_INV_{int(m.group(1))}"
        if cid in known_entities and cid not in matched:
            matched.append(cid)

    for m in _WIRE_ID_RE.finditer(query):
        cid = f"ENT_WIRE_{m.group(1).upper()}"
        if cid in known_entities and cid not in matched:
            matched.append(cid)

    for m in _Q_WIRE_RE.finditer(query):
        cid = f"ENT_WIRE_{m.group(1).upper()}"
        if cid in known_entities and cid not in matched:
            matched.append(cid)

    for m in _GENERIC_ID_RE.finditer(query):
        cid = f"ENT_{m.group(1).upper()}_{m.group(2).upper()}"
        if cid in known_entities and cid not in matched:
            matched.append(cid)

    if matched:
        return matched

    # If the query explicitly names a structured ID (e.g. INV-999) that is NOT in known_entities,
    # do not fuzzy-fallback onto a different invoice ID.
    if (
        _INVOICE_ID_RE.search(query)
        or _WIRE_ID_RE.search(query)
        or _Q_WIRE_RE.search(query)
        or _GENERIC_ID_RE.search(query)
    ):
        return []

    # 2. RapidFuzz alias and claim-value matching
    lowered_q = query.lower()
    for cid, aliases in entities.items():
        for alias in aliases:
            if len(alias) >= 3 and fuzz.partial_ratio(alias.lower(), lowered_q) >= 85.0:
                if cid not in matched:
                    matched.append(cid)

    if not matched:
        for claim in claims:
            if claim.entity_canonical and len(claim.value) >= 4:
                if fuzz.partial_ratio(claim.value.lower(), lowered_q) >= 88.0:
                    if claim.entity_canonical not in matched:
                        matched.append(claim.entity_canonical)

    if not matched and use_vector:
        for hint in _try_chroma_entity_hints(query, storage_dir):
            if hint in known_entities and hint not in matched:
                matched.append(hint)

    return matched


def retrieve_dossier(
    query: str,
    storage_dir: Path,
    use_vector: bool = False,
) -> tuple[EvidenceDossier, dict[str, Any]]:
    """Stage A: Resolve query intent and retrieve 2-hop graph EvidenceDossier."""
    target_attr = _detect_target_attribute(query)
    required_files = (
        storage_dir / "evidence_graph.json",
        storage_dir / "claims_scored.json",
        storage_dir / "conflict_edges.json",
        storage_dir / "ingested_docs.json",
        storage_dir / "entities.json",
    )
    if any(not p.exists() for p in required_files):
        empty_dossier = EvidenceDossier(
            query=query,
            target_entities=[],
            verified_claims=[],
            conflicting_edges=[],
            quarantined_sources=[],
            max_trust_score=0.0,
            missing_attributes=[target_attr] if target_attr else [],
        )
        return empty_dossier, {
            "target_attribute": target_attr,
            "docs_by_id": {},
            "candidate_claims": [],
            "superseded_map": {},
            "all_entity_sources": [],
            "missing_storage": True,
        }

    graph, scored_claims, _ = load_graph_artifacts(storage_dir)
    docs = TypeAdapter(list[SourceDoc]).validate_json(
        (storage_dir / "ingested_docs.json").read_text(encoding="utf-8")
    )
    entities: dict[str, list[str]] = json.loads(
        (storage_dir / "entities.json").read_text(encoding="utf-8")
    )
    docs_by_id: dict[str, SourceDoc] = {d.source_id: d for d in docs}

    target_entities = _resolve_query_entities(
        query,
        entities=entities,
        claims=scored_claims,
        storage_dir=storage_dir,
        use_vector=use_vector,
    )

    all_claims: list[Claim] = []
    seen_claim_ids: set[str] = set()
    all_edges: list[ConflictEdge] = []
    seen_edge_ids: set[str] = set()
    linked_sources: dict[str, SourceDoc] = {}
    superseded_map: dict[str, bool] = {}

    for entity_id in target_entities:
        sub = get_entity_subgraph(graph, entity_id=entity_id, hops=2)
        for claim in sub["claims"]:
            if claim.entity_canonical == entity_id and claim.claim_id not in seen_claim_ids:
                seen_claim_ids.add(claim.claim_id)
                all_claims.append(claim)
                node_attrs = graph.nodes[claim.claim_id] if claim.claim_id in graph else {}
                superseded_map[claim.claim_id] = bool(node_attrs.get("superseded", False))
        for edge in sub["edges"]:
            if edge.edge_id not in seen_edge_ids:
                seen_edge_ids.add(edge.edge_id)
                all_edges.append(edge)
        for src in sub["sources"]:
            # Only include sources that actually asserted at least one claim for this entity
            if any(c.source_id == src.source_id for c in all_claims):
                linked_sources[src.source_id] = src

    if target_attr is None and all_claims:
        # Open-domain dynamic attribute matching against entity claims and MISSING_FIELD columns
        candidate_attrs = sorted({c.attribute for c in all_claims})
        missing_flag_cols = sorted(
            {
                flag.location.column.lower()
                for src in linked_sources.values()
                for flag in src.forensic_flags
                if flag.code == "MISSING_FIELD" and flag.location and flag.location.column
            }
        )
        lowered_q = query.lower()
        best_attr: str | None = None
        best_score = 0.0
        for attr_name in candidate_attrs + [c for c in missing_flag_cols if c not in candidate_attrs]:
            readable = attr_name.replace("_", " ")
            if readable in lowered_q:
                best_attr = attr_name
                best_score = 100.0
                break
            score = fuzz.partial_ratio(readable, lowered_q)
            if score >= 80.0 and score > best_score:
                best_attr = attr_name
                best_score = score
        if best_attr is None:
            m_noun = re.search(
                r"\b(?:who\s+is|what\s+is|what\s+was)\s+(?:the\s+)?([a-z][a-z0-9 _-]{1,28}?)\s+(?:for|of|on)\b",
                lowered_q,
            )
            if m_noun:
                best_attr = re.sub(r"[^a-z0-9]+", "_", m_noun.group(1).strip()).strip("_")
        if best_attr is not None:
            target_attr = best_attr

    if target_attr is not None:
        attr_claims = [c for c in all_claims if c.attribute == target_attr]
    else:
        attr_claims = list(all_claims)

    attr_claim_ids = {c.claim_id for c in attr_claims}
    relevant_edges = [
        e
        for e in all_edges
        if (e.claim_a_id in attr_claim_ids or e.claim_b_id in attr_claim_ids)
        and e.edge_type in (EdgeType.CONTRADICTS, EdgeType.SUPERSEDES)
    ]

    active_clean_claims = [
        c
        for c in attr_claims
        if not superseded_map.get(c.claim_id, False)
        and not docs_by_id[c.source_id].quarantined
    ]
    quarantined_sources = sorted(
        (s for s in linked_sources.values() if s.quarantined),
        key=lambda s: s.file_name,
    )

    max_trust = max((c.trust_score for c in active_clean_claims), default=0.0)
    missing_attrs: list[str] = []
    if target_attr and max_trust < settings.abstention_threshold:
        missing_attrs.append(target_attr)

    dossier = EvidenceDossier(
        query=query,
        target_entities=target_entities,
        verified_claims=active_clean_claims,
        conflicting_edges=relevant_edges,
        quarantined_sources=quarantined_sources,
        max_trust_score=round(max_trust, 4),
        missing_attributes=missing_attrs,
    )
    context: dict[str, Any] = {
        "target_attribute": target_attr,
        "docs_by_id": docs_by_id,
        "candidate_claims": attr_claims,
        "all_entity_claims": all_claims,
        "superseded_map": superseded_map,
        "all_entity_sources": list(linked_sources.values()),
        "missing_storage": False,
    }
    return dossier, context


def _claim_group_key(claim: Claim) -> tuple[str, Any]:
    """Group matching clean claims across modalities (e.g. '12000' == '$12,000')."""
    if claim.numeric_value is not None:
        return ("num", round(claim.numeric_value, 2))
    return ("str", re.sub(r"\s+", " ", claim.value.strip().lower()))


def evaluate_verdict(
    dossier: EvidenceDossier,
    context: dict[str, Any],
) -> tuple[VerdictStatus, float, list[Claim], list[ConflictEdge]]:
    """Stage B: Enforce deterministic Python abstention and verdict selection."""
    docs_by_id: dict[str, SourceDoc] = context["docs_by_id"]
    candidate_claims: list[Claim] = context["candidate_claims"]
    superseded_map: dict[str, bool] = context["superseded_map"]
    relevant_edges: list[ConflictEdge] = list(dossier.conflicting_edges)

    active_claims = [c for c in candidate_claims if not superseded_map.get(c.claim_id, False)]
    clean_claims = [c for c in active_claims if not docs_by_id[c.source_id].quarantined]
    quarantined_claims = [c for c in active_claims if docs_by_id[c.source_id].quarantined]

    # 1. No matched entity or no active claims at all -> INSUFFICIENT_EVIDENCE
    if not dossier.target_entities or (not clean_claims and not quarantined_claims):
        return VerdictStatus.INSUFFICIENT_EVIDENCE, 0.0, [], relevant_edges

    # 2. Only quarantined claims exist for the target attribute -> TAMPER_DETECTED
    if not clean_claims and quarantined_claims:
        return VerdictStatus.TAMPER_DETECTED, 0.0, quarantined_claims, relevant_edges

    # 3. Clean claims exist, but highest trust is below ABSTENTION_THRESHOLD -> INSUFFICIENT_EVIDENCE
    max_clean_trust = max(c.trust_score for c in clean_claims)
    if max_clean_trust < settings.abstention_threshold:
        return (
            VerdictStatus.INSUFFICIENT_EVIDENCE,
            round(max_clean_trust, 4),
            clean_claims,
            relevant_edges,
        )

    # Select winning corroborated value group among clean active claims
    grouped: dict[tuple[str, Any], list[Claim]] = {}
    for claim in clean_claims:
        grouped.setdefault(_claim_group_key(claim), []).append(claim)

    def _group_rank(group: list[Claim]) -> tuple[float, int, float]:
        distinct_sources = len({c.source_id for c in group})
        avg_or_sum = sum(c.trust_score for c in group)
        top_t = max(c.trust_score for c in group)
        return (top_t + 0.1 * distinct_sources, distinct_sources, avg_or_sum)

    winning_group = max(grouped.values(), key=_group_rank)
    winning_group.sort(key=lambda c: c.trust_score, reverse=True)
    confidence = round(max(c.trust_score for c in winning_group), 4)

    candidate_ids = {c.claim_id for c in candidate_claims}
    has_contradiction = any(
        e.edge_type == EdgeType.CONTRADICTS
        and (e.claim_a_id in candidate_ids or e.claim_b_id in candidate_ids)
        for e in relevant_edges
    )

    # 4. Contradiction present on target attribute -> CONFLICT_RESOLVED
    if has_contradiction:
        return VerdictStatus.CONFLICT_RESOLVED, confidence, winning_group, relevant_edges

    # 5. Corroborated or clean amended claims with zero contradictions -> VERIFIED
    return VerdictStatus.VERIFIED, confidence, winning_group, relevant_edges


def _format_location_tag(file_name: str, loc: LocationRef) -> str:
    """Format concise inline citation coordinate tag."""
    if loc.row is not None:
        return f"[{file_name}:row={loc.row}]"
    if loc.page is not None:
        return f"[{file_name}:p{loc.page}]"
    if loc.bbox is not None:
        return f"[{file_name}:bbox]"
    return f"[{file_name}]"


def _format_claim_value(claim: Claim) -> str:
    """Format currency values consistently (e.g., 12000 -> '$12,000')."""
    if claim.numeric_value is not None and (claim.unit or "").upper() == "USD":
        val = claim.numeric_value
        if val == int(val):
            return f"${int(val):,}"
        return f"${val:,.2f}"
    return claim.value


def _build_citations(
    claims: list[Claim],
    docs_by_id: dict[str, SourceDoc],
) -> list[Citation]:
    """Construct verifiable `Citation` objects retaining exact `LocationRef`."""
    citations: list[Citation] = []
    seen: set[str] = set()
    for claim in claims:
        if claim.claim_id in seen:
            continue
        seen.add(claim.claim_id)
        doc = docs_by_id.get(claim.source_id)
        file_name = doc.file_name if doc is not None else claim.source_id
        citations.append(
            Citation(
                source_id=claim.source_id,
                file_name=file_name,
                claim_id=claim.claim_id,
                location_ref=claim.location_ref,
                quote=claim.location_ref.snippet or claim.value,
            )
        )
    return citations


def _synthesize_deterministic(
    query: str,
    status: VerdictStatus,
    winning_claims: list[Claim],
    dossier: EvidenceDossier,
    context: dict[str, Any],
) -> str:
    """Deterministic offline prose synthesis strictly grounded in EvidenceDossier."""
    docs_by_id: dict[str, SourceDoc] = context["docs_by_id"]
    target_attr: str | None = context["target_attribute"]
    entity_label = ", ".join(dossier.target_entities) if dossier.target_entities else "the requested entity"

    if status == VerdictStatus.INSUFFICIENT_EVIDENCE:
        if context.get("missing_storage"):
            return (
                "INSUFFICIENT_EVIDENCE: Evidence graph artifacts are not yet built in storage. "
                "Run ingest, extract, and build-graph first."
            )
        missing_str = ", ".join(dossier.missing_attributes) if dossier.missing_attributes else (target_attr or "evidence")
        clues: list[str] = []
        for src in context.get("all_entity_sources", []):
            for flag in src.forensic_flags:
                if flag.code in {"MISSING_FIELD", "LOW_LEGIBILITY_BLUR"}:
                    clues.append(f"{src.file_name} flagged {flag.code}")
        clue_suffix = f" Forensic observations: {'; '.join(clues)}." if clues else ""
        return (
            f"INSUFFICIENT_EVIDENCE: Cannot verify '{missing_str}' for {entity_label} "
            f"(max trust {dossier.max_trust_score:.2f} < threshold {settings.abstention_threshold:.2f})."
            f"{clue_suffix}"
        )

    if status == VerdictStatus.TAMPER_DETECTED:
        flagged_details: list[str] = []
        for src in dossier.quarantined_sources:
            codes = sorted({f.code for f in src.forensic_flags})
            flagged_details.append(f"{src.file_name} ({', '.join(codes)})")
        detail_str = "; ".join(flagged_details) if flagged_details else "quarantined source"
        return (
            f"TAMPER_DETECTED: Cannot trust '{target_attr or 'requested'}' content for {entity_label}. "
            f"Source(s) quarantined by pre-LLM forensics: {detail_str}."
        )

    primary = winning_claims[0]
    formatted_val = _format_claim_value(primary)
    cite_tags = " ".join(
        dict.fromkeys(
            _format_location_tag(docs_by_id[c.source_id].file_name, c.location_ref)
            for c in winning_claims
            if c.source_id in docs_by_id
        )
    )

    supersedes_edges = [e for e in dossier.conflicting_edges if e.edge_type == EdgeType.SUPERSEDES]
    contradict_edges = [e for e in dossier.conflicting_edges if e.edge_type == EdgeType.CONTRADICTS]

    if status == VerdictStatus.CONFLICT_RESOLVED:
        rejected_notes: list[str] = []
        for src in dossier.quarantined_sources:
            codes = sorted({f.code for f in src.forensic_flags})
            rejected_notes.append(f"{src.file_name} quarantined ({', '.join(codes)})")
        conflict_reasons = "; ".join(dict.fromkeys(e.reason for e in contradict_edges))
        reject_str = "; ".join(rejected_notes) if rejected_notes else "conflicting claim"
        return (
            f"The verified {primary.attribute} for {entity_label} is {formatted_val} {cite_tags}. "
            f"Conflicting evidence rejected ({conflict_reasons}; {reject_str})."
        ).strip()

    # VerdictStatus.VERIFIED
    if supersedes_edges:
        amend_ts = primary.timestamp or "latest addendum"
        return (
            f"The current verified {primary.attribute} for {entity_label} is {formatted_val} "
            f"(amended on {amend_ts}) {cite_tags}, superseding prior records "
            f"({supersedes_edges[0].reason})."
        )

    return f"The verified {primary.attribute} for {entity_label} is {formatted_val} {cite_tags}."


def _synthesize_with_ollama(
    query: str,
    dossier: EvidenceDossier,
    status: VerdictStatus,
    winning_claims: list[Claim],
    fallback_answer: str,
    docs_by_id: dict[str, SourceDoc],
) -> str:
    """Optional local Ollama synthesis strictly from structured `EvidenceDossier` JSON."""
    base_url = settings.ollama_base_url.rstrip("/")
    if not ("127.0.0.1" in base_url or "localhost" in base_url):
        raise ValueError(f"Offline guarantee violated by non-local Ollama URL: {base_url}")
    if status in (VerdictStatus.INSUFFICIENT_EVIDENCE, VerdictStatus.TAMPER_DETECTED):
        return fallback_answer
    if not winning_claims:
        return fallback_answer
    try:
        import ollama

        winning_val = _format_claim_value(winning_claims[0])
        cite_tags = " ".join(
            dict.fromkeys(
                _format_location_tag(docs_by_id[c.source_id].file_name, c.location_ref)
                for c in winning_claims
                if c.source_id in docs_by_id
            )
        )
        client = ollama.Client(host=base_url)
        prompt = (
            f"Question: {query}\n"
            f"Verdict Status: {status.value}\n"
            f"Exact Value: {winning_val}\n"
            f"Citations: {cite_tags}\n"
            "Write 1-2 natural, concise sentences answering the question using ONLY the dossier below. "
            "Include the Exact Value and Citations verbatim. Do not mention 'JSON', 'EvidenceDossier', "
            "'extraction_confidence', or prompt instructions.\n"
            f"Dossier:\n{dossier.model_dump_json()}"
        )
        resp = client.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        text = str(resp["message"]["content"]).strip()
        if not text or winning_val.lower() not in text.lower():
            return fallback_answer
        if cite_tags and not any(
            docs_by_id[c.source_id].file_name in text
            for c in winning_claims
            if c.source_id in docs_by_id
        ):
            text = f"{text} {cite_tags}".strip()
        return text
    except Exception:
        return fallback_answer


def answer_query(
    query: str,
    storage_dir: Path | None = None,
    use_llm: bool = False,
    use_vector: bool = False,
) -> VerdictResponse:
    """Execute 2-hop retrieval, deterministic abstention, and cited synthesis."""
    target_storage = storage_dir if storage_dir is not None else settings.storage_dir
    dossier, context = retrieve_dossier(query, storage_dir=target_storage, use_vector=use_vector)
    status, confidence, winning_claims, relevant_edges = evaluate_verdict(dossier, context)

    docs_by_id: dict[str, SourceDoc] = context["docs_by_id"]
    citations = (
        _build_citations(winning_claims, docs_by_id)
        if status in (VerdictStatus.VERIFIED, VerdictStatus.CONFLICT_RESOLVED)
        else []
    )

    answer = _synthesize_deterministic(
        query=query,
        status=status,
        winning_claims=winning_claims,
        dossier=dossier,
        context=context,
    )
    if use_llm:
        answer = _synthesize_with_ollama(
            query=query,
            dossier=dossier,
            status=status,
            winning_claims=winning_claims,
            fallback_answer=answer,
            docs_by_id=docs_by_id,
        )

    return VerdictResponse(
        query=query,
        status=status,
        answer=answer,
        confidence=confidence,
        citations=citations,
        conflicts=relevant_edges,
        quarantined_sources=dossier.quarantined_sources,
        dossier=dossier,
    )


def evaluate_test_cases(
    cases_path: Path,
    storage_dir: Path,
    use_llm: bool = False,
    use_vector: bool = False,
) -> dict[str, Any]:
    """Run all benchmark cases in `eval/test_cases.json` and return a structured scorecard."""
    cases: list[dict[str, Any]] = json.loads(cases_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    injection_cases = 0
    injection_blocked = 0
    conflict_cases = 0
    conflict_detected = 0
    citation_eligible = 0
    citation_accurate = 0
    abstention_cases = 0
    abstention_accurate = 0

    for case in cases:
        resp = answer_query(
            case["query"],
            storage_dir=storage_dir,
            use_llm=use_llm,
            use_vector=use_vector,
        )
        failures: list[str] = []

        if resp.status.value != case["expected_status"]:
            failures.append(
                f"status={resp.status.value} != expected={case['expected_status']}"
            )

        expected_answer = case.get("expected_answer")
        if expected_answer and case["expected_status"] != VerdictStatus.INSUFFICIENT_EVIDENCE.value:
            if expected_answer.lower() not in resp.answer.lower():
                failures.append(f"answer missing expected substring {expected_answer!r}")

        for forbidden in case.get("disallowed_answer_substrings", []):
            if forbidden.lower() in resp.answer.lower():
                failures.append(f"answer leaked disallowed substring {forbidden!r}")

        quarantined_names = {s.file_name for s in resp.quarantined_sources}
        for req_q in case.get("expected_quarantined", []):
            if req_q not in quarantined_names:
                failures.append(f"missing quarantined source {req_q!r}")

        expected_edge = case.get("expected_edge_type")
        if expected_edge:
            edge_types = {e.edge_type.value for e in resp.conflicts}
            if expected_edge not in edge_types:
                failures.append(f"missing expected edge_type {expected_edge!r}")

        gt_sources = set(case.get("ground_truth_sources", []))
        if case["expected_status"] in {
            VerdictStatus.VERIFIED.value,
            VerdictStatus.CONFLICT_RESOLVED.value,
        }:
            citation_eligible += 1
            cited_files = {c.file_name for c in resp.citations}
            if resp.citations and gt_sources.issubset(cited_files):
                citation_accurate += 1
            elif not resp.citations:
                failures.append("missing citations for verified/resolved verdict")

        if case.get("expected_prompt_injection_flagged"):
            injection_cases += 1
            disallowed = case.get("disallowed_answer_substrings", [])
            no_forbidden = not any(s.lower() in resp.answer.lower() for s in disallowed)
            endorsed_clean = (
                expected_answer.lower() in resp.answer.lower()
                if expected_answer
                else resp.status == VerdictStatus.TAMPER_DETECTED
            )
            if no_forbidden and endorsed_clean and set(case.get("expected_quarantined", [])).issubset(quarantined_names):
                injection_blocked += 1

        if case["expected_status"] == VerdictStatus.CONFLICT_RESOLVED.value:
            conflict_cases += 1
            if any(e.edge_type == EdgeType.CONTRADICTS for e in resp.conflicts):
                conflict_detected += 1

        if case.get("expected_abstention"):
            abstention_cases += 1
            if resp.status == VerdictStatus.INSUFFICIENT_EVIDENCE:
                abstention_accurate += 1

        results.append(
            {
                "case_id": case["case_id"],
                "name": case.get("name", ""),
                "passed": len(failures) == 0,
                "status": resp.status.value,
                "confidence": resp.confidence,
                "answer": resp.answer,
                "cited_files": sorted({c.file_name for c in resp.citations}),
                "quarantined_files": sorted(quarantined_names),
                "failures": failures,
            }
        )

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "command": "eval",
        "status": "ok" if passed == total else "failed",
        "cases": str(cases_path),
        "storage_dir": str(storage_dir),
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "metrics": {
            "injection_block_rate": round(injection_blocked / injection_cases, 4)
            if injection_cases
            else 1.0,
            "conflict_recall": round(conflict_detected / conflict_cases, 4)
            if conflict_cases
            else 1.0,
            "citation_accuracy": round(citation_accurate / citation_eligible, 4)
            if citation_eligible
            else 1.0,
            "abstention_accuracy": round(abstention_accurate / abstention_cases, 4)
            if abstention_cases
            else 1.0,
        },
        "results": results,
    }

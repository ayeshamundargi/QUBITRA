"""Evidence graph construction, conflict/supersession detection, and trust propagation.

Uses a NetworkX `MultiDiGraph` linking `Source`, `Claim`, and `Entity` nodes via
`ASSERTS`, `ABOUT`, `SUPPORTS`, `CONTRADICTS`, and `SUPERSEDES` edges.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import TypeAdapter
from rapidfuzz import fuzz

from core.config import settings
from core.schemas import Claim, ConflictEdge, EdgeType, SourceDoc

_CATEGORICAL_ATTRS: frozenset[str] = frozenset({"status", "vendor", "date", "approver"})
_NEGATION_RE = re.compile(r"\b(?:not|no|never|denied|rejected|cancelled|void)\b", re.IGNORECASE)
_NUM_EXTRACT_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation while preserving tokens."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _collapse_spaces(text: str) -> str:
    """Collapse all whitespace to guard against OCR word-split artifacts."""
    return re.sub(r"\s+", "", _normalize_text(text))


def _try_offline_nli_contradiction(text_a: str, text_b: str) -> tuple[bool, float] | None:
    """Optional strictly offline DeBERTa-v3 NLI check (`local_files_only=True`)."""
    try:
        from transformers import pipeline

        clf = pipeline(
            "text-classification",
            model=settings.nli_model,
            tokenizer=settings.nli_model,
            model_kwargs={"local_files_only": True},
        )
        res = clf({"text": text_a, "text_pair": text_b}, top_k=None)
        scores = {str(item["label"]).lower(): float(item["score"]) for item in res}
        contradiction_prob = scores.get("contradiction", 0.0)
        return contradiction_prob >= settings.nli_contradiction_threshold, contradiction_prob
    except Exception:
        return None


class EvidenceGraphEngine:
    """Manages the NetworkX `MultiDiGraph` representation of sources, claims, and entities."""

    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.graph: nx.MultiDiGraph = graph if graph is not None else nx.MultiDiGraph()

    def build_base_graph(
        self,
        docs: list[SourceDoc],
        claims: list[Claim],
        entities: dict[str, list[str]],
    ) -> nx.MultiDiGraph:
        """Populate `Source`, `Entity`, and `Claim` nodes plus `ASSERTS`/`ABOUT` edges."""
        for doc in docs:
            self.graph.add_node(
                doc.source_id,
                node_type="Source",
                file_name=doc.file_name,
                modality=doc.modality.value,
                trust_prior=doc.trust_prior,
                dynamic_reliability=doc.trust_prior,
                quarantined=doc.quarantined,
                forensic_flags=[f.model_dump(mode="json") for f in doc.forensic_flags],
                source_data=doc.model_dump(mode="json"),
            )

        for entity_id, aliases in entities.items():
            self.graph.add_node(
                entity_id,
                node_type="Entity",
                aliases=list(aliases),
            )

        for claim in claims:
            entity_id = claim.entity_canonical or claim.entity_raw
            if entity_id not in self.graph:
                self.graph.add_node(
                    entity_id,
                    node_type="Entity",
                    aliases=[claim.entity_raw],
                )

            claim_payload = claim.model_dump(mode="json")
            self.graph.add_node(
                claim.claim_id,
                node_type="Claim",
                **claim_payload,
                superseded=False,
                superseded_by=[],
                corroborating_modalities=[],
            )
            self.graph.add_edge(
                claim.source_id,
                claim.claim_id,
                key=f"asserts_{claim.claim_id}",
                edge_type="ASSERTS",
                extraction_confidence=claim.extraction_confidence,
                location_ref=claim.location_ref.model_dump(mode="json"),
            )
            self.graph.add_edge(
                claim.claim_id,
                entity_id,
                key=f"about_{claim.claim_id}_{entity_id}",
                edge_type="ABOUT",
            )

        return self.graph

    def add_epistemic_edges(self, edges: list[ConflictEdge]) -> None:
        """Add `SUPPORTS`, `CONTRADICTS`, and `SUPERSEDES` edges and sync node metadata."""
        for edge in edges:
            self.graph.add_edge(
                edge.claim_a_id,
                edge.claim_b_id,
                key=edge.edge_id,
                edge_id=edge.edge_id,
                edge_type=edge.edge_type.value,
                confidence=edge.confidence,
                reason=edge.reason,
                detector=edge.detector,
            )

            if edge.edge_type == EdgeType.SUPERSEDES and edge.claim_b_id in self.graph:
                node = self.graph.nodes[edge.claim_b_id]
                node["superseded"] = True
                sup_list = list(node.get("superseded_by", []))
                if edge.claim_a_id not in sup_list:
                    sup_list.append(edge.claim_a_id)
                node["superseded_by"] = sup_list


def _check_supersedes(
    c1: Claim,
    c2: Claim,
    docs_by_id: dict[str, SourceDoc],
) -> tuple[Claim, Claim, str] | None:
    """Rule 1: Return `(newer, older, reason)` if one non-quarantined amendment supersedes the other."""
    s1 = docs_by_id.get(c1.source_id)
    s2 = docs_by_id.get(c2.source_id)
    q1 = s1.quarantined if s1 is not None else False
    q2 = s2.quarantined if s2 is not None else False

    candidates: list[tuple[Claim, Claim]] = []
    if c1.is_amendment and not c2.is_amendment and not q1:
        candidates.append((c1, c2))
    elif c2.is_amendment and not c1.is_amendment and not q2:
        candidates.append((c2, c1))

    for newer, older in candidates:
        if not older.timestamp or not newer.timestamp or older.timestamp < newer.timestamp:
            older_ts = older.timestamp or "undated"
            newer_ts = newer.timestamp or "undated"
            reason = (
                f"Temporal amendment ({newer.value} on {newer_ts}) supersedes "
                f"prior record ({older.value} on {older_ts})."
            )
            return newer, older, reason
    return None


def detect_claim_relations(
    claims: list[Claim],
    docs_by_id: dict[str, SourceDoc],
    use_nli: bool = False,
) -> list[ConflictEdge]:
    """Stage B: Detect `SUPERSEDES`, `SUPPORTS`, and `CONTRADICTS` edges across claims."""
    grouped: dict[tuple[str, str], list[Claim]] = {}
    for claim in claims:
        canon = claim.entity_canonical or claim.entity_raw
        grouped.setdefault((canon, claim.attribute), []).append(claim)

    edges: list[ConflictEdge] = []
    edge_counter = 0

    for (_, attr), group_claims in grouped.items():
        n = len(group_claims)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = group_claims[i]
                c2 = group_claims[j]

                # Skip redundant identical assertions from the exact same source document.
                if c1.source_id == c2.source_id and c1.value.strip() == c2.value.strip():
                    continue

                # Rule 1 — Temporal Amendment Check FIRST (SUPERSEDES)
                supersedes = _check_supersedes(c1, c2, docs_by_id)
                if supersedes is not None:
                    newer, older, reason = supersedes
                    edge_counter += 1
                    edges.append(
                        ConflictEdge(
                            edge_id=f"edge_{edge_counter:04d}",
                            claim_a_id=newer.claim_id,
                            claim_b_id=older.claim_id,
                            edge_type=EdgeType.SUPERSEDES,
                            confidence=0.95,
                            reason=reason,
                            detector="temporal",
                        )
                    )
                    continue

                # Rule 2 — Numeric / Currency Comparator
                if c1.numeric_value is not None and c2.numeric_value is not None:
                    u1 = (c1.unit or "").upper()
                    u2 = (c2.unit or "").upper()
                    if not u1 or not u2 or u1 == u2:
                        a, b = c1.numeric_value, c2.numeric_value
                        tol = max(0.01, 0.001 * max(abs(a), abs(b)))
                        edge_counter += 1
                        if abs(a - b) <= tol:
                            edges.append(
                                ConflictEdge(
                                    edge_id=f"edge_{edge_counter:04d}",
                                    claim_a_id=c1.claim_id,
                                    claim_b_id=c2.claim_id,
                                    edge_type=EdgeType.SUPPORTS,
                                    confidence=0.95,
                                    reason=f"Numeric agreement: {c1.value} matches {c2.value}.",
                                    detector="numeric",
                                )
                            )
                        else:
                            delta = abs(a - b)
                            edges.append(
                                ConflictEdge(
                                    edge_id=f"edge_{edge_counter:04d}",
                                    claim_a_id=c1.claim_id,
                                    claim_b_id=c2.claim_id,
                                    edge_type=EdgeType.CONTRADICTS,
                                    confidence=0.95,
                                    reason=(
                                        f"Numeric mismatch: ${a:,.2f} vs ${b:,.2f} "
                                        f"(delta ${delta:,.2f})."
                                    ),
                                    detector="numeric",
                                )
                            )
                        continue

                # Rule 3 — Date / Categorical / Text Comparator
                norm_a = _normalize_text(c1.value)
                norm_b = _normalize_text(c2.value)
                ts_ratio = fuzz.token_sort_ratio(norm_a, norm_b)
                collapsed_ratio = fuzz.ratio(_collapse_spaces(c1.value), _collapse_spaces(c2.value))
                sim = max(ts_ratio, collapsed_ratio)

                if sim >= 88.0:
                    edge_counter += 1
                    edges.append(
                        ConflictEdge(
                            edge_id=f"edge_{edge_counter:04d}",
                            claim_a_id=c1.claim_id,
                            claim_b_id=c2.claim_id,
                            edge_type=EdgeType.SUPPORTS,
                            confidence=round(sim / 100.0, 4),
                            reason=f"Lexical agreement ({sim:.1f}%): {c1.value!r} ~ {c2.value!r}.",
                            detector="lexical",
                        )
                    )
                    continue

                if attr in _CATEGORICAL_ATTRS:
                    edge_counter += 1
                    edges.append(
                        ConflictEdge(
                            edge_id=f"edge_{edge_counter:04d}",
                            claim_a_id=c1.claim_id,
                            claim_b_id=c2.claim_id,
                            edge_type=EdgeType.CONTRADICTS,
                            confidence=0.90,
                            reason=(
                                f"Categorical conflict on '{attr}': "
                                f"{c1.value!r} vs {c2.value!r}."
                            ),
                            detector="lexical",
                        )
                    )
                    continue

                # Free-text attribute (e.g. vendor_notes)
                if use_nli:
                    nli_res = _try_offline_nli_contradiction(c1.value, c2.value)
                    if nli_res is not None:
                        is_contra, prob = nli_res
                        edge_counter += 1
                        edges.append(
                            ConflictEdge(
                                edge_id=f"edge_{edge_counter:04d}",
                                claim_a_id=c1.claim_id,
                                claim_b_id=c2.claim_id,
                                edge_type=EdgeType.CONTRADICTS if is_contra else EdgeType.SUPPORTS,
                                confidence=round(prob if is_contra else 1.0 - prob, 4),
                                reason=f"Offline NLI score={prob:.3f}.",
                                detector="nli",
                            )
                        )
                        continue

                neg_a = bool(_NEGATION_RE.search(c1.value))
                neg_b = bool(_NEGATION_RE.search(c2.value))
                nums_a = _NUM_EXTRACT_RE.findall(c1.value)
                nums_b = _NUM_EXTRACT_RE.findall(c2.value)
                if neg_a != neg_b or (nums_a and nums_b and set(nums_a) != set(nums_b)):
                    edge_counter += 1
                    edges.append(
                        ConflictEdge(
                            edge_id=f"edge_{edge_counter:04d}",
                            claim_a_id=c1.claim_id,
                            claim_b_id=c2.claim_id,
                            edge_type=EdgeType.CONTRADICTS,
                            confidence=0.80,
                            reason=f"Free-text negation/numeric clash: {c1.value!r} vs {c2.value!r}.",
                            detector="lexical",
                        )
                    )

    return edges


def propagate_trust(
    graph: nx.MultiDiGraph,
    docs: list[SourceDoc],
    claims: list[Claim],
    edges: list[ConflictEdge],
    iterations: int = 5,
) -> tuple[list[SourceDoc], list[Claim]]:
    """Iteratively propagate cross-modal corroboration, conflicts, and quarantine caps."""
    docs_by_id: dict[str, SourceDoc] = {d.source_id: d for d in docs}
    claims_by_id: dict[str, Claim] = {c.claim_id: c for c in claims}

    src_rel: dict[str, float] = {d.source_id: d.trust_prior for d in docs}
    claim_trust: dict[str, float] = {c.claim_id: c.trust_score for c in claims}

    supports_neighbors: dict[str, list[tuple[str, str, str]]] = {c.claim_id: [] for c in claims}
    conflict_neighbors: dict[str, list[tuple[str, str]]] = {c.claim_id: [] for c in claims}

    for edge in edges:
        ca = claims_by_id.get(edge.claim_a_id)
        cb = claims_by_id.get(edge.claim_b_id)
        if ca is None or cb is None:
            continue
        sa = docs_by_id.get(ca.source_id)
        sb = docs_by_id.get(cb.source_id)
        qa = sa.quarantined if sa is not None else False
        qb = sb.quarantined if sb is not None else False

        if edge.edge_type == EdgeType.SUPPORTS:
            if not qb:
                supports_neighbors[ca.claim_id].append((cb.claim_id, cb.source_id, cb.modality.value))
            if not qa:
                supports_neighbors[cb.claim_id].append((ca.claim_id, ca.source_id, ca.modality.value))
        elif edge.edge_type == EdgeType.CONTRADICTS:
            if not qb:
                conflict_neighbors[ca.claim_id].append((cb.claim_id, cb.source_id))
            if not qa:
                conflict_neighbors[cb.claim_id].append((ca.claim_id, ca.source_id))
        # SUPERSEDES intentionally does NOT penalize older non-tampered claims or sources.

    claims_by_source: dict[str, list[Claim]] = {d.source_id: [] for d in docs}
    for claim in claims:
        claims_by_source.setdefault(claim.source_id, []).append(claim)

    for _ in range(max(1, iterations)):
        next_claim_trust: dict[str, float] = {}
        for claim in claims:
            cid = claim.claim_id
            source = docs_by_id[claim.source_id]
            base = src_rel[claim.source_id] * claim.extraction_confidence

            agreeing = [
                (ocid, osid, omod)
                for ocid, osid, omod in supports_neighbors.get(cid, [])
                if osid != claim.source_id
            ]
            distinct_other_mods = {
                omod for _, _, omod in agreeing if omod != claim.modality.value
            }
            same_mod_agree = any(omod == claim.modality.value for _, _, omod in agreeing)

            corroboration_boost = 0.15 * min(2, len(distinct_other_mods))
            if same_mod_agree and not distinct_other_mods:
                corroboration_boost += 0.08

            conflict_penalty = sum(
                0.20 * claim_trust[ocid] for ocid, _ in conflict_neighbors.get(cid, [])
            )

            score = min(0.98, max(0.05, base + corroboration_boost - conflict_penalty))
            if source.quarantined:
                score = min(score, 0.15)
            next_claim_trust[cid] = round(score, 4)

        claim_trust = next_claim_trust

        next_src_rel: dict[str, float] = {}
        for doc in docs:
            if doc.quarantined:
                next_src_rel[doc.source_id] = min(doc.trust_prior, 0.15)
                continue

            src_claims = claims_by_source.get(doc.source_id, [])
            if not src_claims:
                next_src_rel[doc.source_id] = doc.trust_prior
                continue

            corroborated_count = sum(1 for c in src_claims if supports_neighbors.get(c.claim_id))
            conflicted_count = sum(1 for c in src_claims if conflict_neighbors.get(c.claim_id))
            ratio_boost = 0.08 * (corroborated_count / len(src_claims))
            ratio_penalty = 0.12 * (conflicted_count / len(src_claims))
            target = min(0.95, max(0.30, doc.trust_prior + ratio_boost - ratio_penalty))
            next_src_rel[doc.source_id] = round(0.5 * src_rel[doc.source_id] + 0.5 * target, 4)

        src_rel = next_src_rel

    updated_docs: list[SourceDoc] = []
    for doc in docs:
        rel = src_rel[doc.source_id]
        if doc.source_id in graph:
            graph.nodes[doc.source_id]["dynamic_reliability"] = rel
        updated_docs.append(doc)

    updated_claims: list[Claim] = []
    for claim in claims:
        final_trust = claim_trust[claim.claim_id]
        agreeing_mods = sorted(
            {omod for _, osid, omod in supports_neighbors.get(claim.claim_id, []) if osid != claim.source_id}
        )
        if claim.claim_id in graph:
            graph.nodes[claim.claim_id]["trust_score"] = final_trust
            graph.nodes[claim.claim_id]["corroborating_modalities"] = agreeing_mods
        updated_claims.append(claim.model_copy(update={"trust_score": final_trust}))

    return updated_docs, updated_claims


def _node_to_claim(attrs: dict[str, Any]) -> Claim:
    """Reconstruct a validated `Claim` from a `Claim` graph node's attributes."""
    claim_fields = {k: attrs[k] for k in Claim.model_fields if k in attrs}
    return Claim.model_validate(claim_fields)


def get_entity_subgraph(
    graph: nx.MultiDiGraph,
    entity_id: str,
    hops: int = 2,
) -> dict[str, Any]:
    """Return claims, conflict/supersession edges, and sources within `hops` of `entity_id`."""
    if entity_id not in graph:
        return {"entity_id": entity_id, "claims": [], "edges": [], "sources": []}

    undirected = graph.to_undirected(as_view=True)
    lengths = nx.single_source_shortest_path_length(undirected, entity_id, cutoff=hops)
    reachable_nodes = set(lengths.keys())

    claims: list[Claim] = []
    claim_ids: set[str] = set()
    sources_by_id: dict[str, SourceDoc] = {}

    # Prefer direct claims about entity_id plus 2-hop epistemic neighbors
    for node_id in sorted(reachable_nodes):
        attrs = graph.nodes[node_id]
        ntype = attrs.get("node_type")
        if ntype == "Claim":
            claim = _node_to_claim(attrs)
            if (claim.entity_canonical or claim.entity_raw) == entity_id or node_id in reachable_nodes:
                claims.append(claim)
                claim_ids.add(claim.claim_id)
                if claim.source_id in graph and "source_data" in graph.nodes[claim.source_id]:
                    sources_by_id[claim.source_id] = SourceDoc.model_validate(
                        graph.nodes[claim.source_id]["source_data"]
                    )
        elif ntype == "Source" and "source_data" in attrs:
            sources_by_id[node_id] = SourceDoc.model_validate(attrs["source_data"])

    edges: list[ConflictEdge] = []
    seen_edge_ids: set[str] = set()
    for u, v, _, data in graph.edges(keys=True, data=True):
        etype = data.get("edge_type")
        if etype in {EdgeType.SUPPORTS.value, EdgeType.CONTRADICTS.value, EdgeType.SUPERSEDES.value}:
            if u in claim_ids or v in claim_ids:
                eid = str(data["edge_id"])
                if eid not in seen_edge_ids:
                    seen_edge_ids.add(eid)
                    edges.append(
                        ConflictEdge(
                            edge_id=eid,
                            claim_a_id=str(u),
                            claim_b_id=str(v),
                            edge_type=EdgeType(etype),
                            confidence=float(data["confidence"]),
                            reason=str(data["reason"]),
                            detector=str(data["detector"]),
                        )
                    )

    return {
        "entity_id": entity_id,
        "claims": claims,
        "edges": edges,
        "sources": list(sources_by_id.values()),
    }


def save_graph_artifacts(
    graph: nx.MultiDiGraph,
    scored_claims: list[Claim],
    edges: list[ConflictEdge],
    storage_dir: Path,
) -> None:
    """Persist NetworkX evidence graph, scored claims, and conflict edges to JSON."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    graph_data = nx.node_link_data(graph, edges="edges")
    (storage_dir / "evidence_graph.json").write_text(
        json.dumps(graph_data, indent=2),
        encoding="utf-8",
    )
    (storage_dir / "claims_scored.json").write_bytes(
        TypeAdapter(list[Claim]).dump_json(scored_claims, indent=2)
    )
    (storage_dir / "conflict_edges.json").write_bytes(
        TypeAdapter(list[ConflictEdge]).dump_json(edges, indent=2)
    )


def load_graph_artifacts(
    storage_dir: Path,
) -> tuple[nx.MultiDiGraph, list[Claim], list[ConflictEdge]]:
    """Reload NetworkX evidence graph, scored claims, and conflict edges from JSON."""
    graph_data = json.loads((storage_dir / "evidence_graph.json").read_text(encoding="utf-8"))
    graph: nx.MultiDiGraph = nx.node_link_graph(graph_data, directed=True, multigraph=True, edges="edges")
    scored_claims = TypeAdapter(list[Claim]).validate_json(
        (storage_dir / "claims_scored.json").read_text(encoding="utf-8")
    )
    edges = TypeAdapter(list[ConflictEdge]).validate_json(
        (storage_dir / "conflict_edges.json").read_text(encoding="utf-8")
    )
    return graph, scored_claims, edges


def build_and_persist_graph(
    storage_dir: Path,
    use_nli: bool = False,
) -> tuple[nx.MultiDiGraph, list[SourceDoc], list[Claim], list[ConflictEdge]]:
    """Load Step 4 artifacts, build evidence graph, propagate trust, and persist outputs."""
    docs = TypeAdapter(list[SourceDoc]).validate_json(
        (storage_dir / "ingested_docs.json").read_text(encoding="utf-8")
    )
    claims = TypeAdapter(list[Claim]).validate_json(
        (storage_dir / "claims.json").read_text(encoding="utf-8")
    )
    entities = json.loads((storage_dir / "entities.json").read_text(encoding="utf-8"))

    docs_by_id = {d.source_id: d for d in docs}
    engine = EvidenceGraphEngine()
    engine.build_base_graph(docs=docs, claims=claims, entities=entities)
    edges = detect_claim_relations(claims=claims, docs_by_id=docs_by_id, use_nli=use_nli)
    engine.add_epistemic_edges(edges)

    scored_docs, scored_claims = propagate_trust(
        engine.graph,
        docs=docs,
        claims=claims,
        edges=edges,
    )
    save_graph_artifacts(engine.graph, scored_claims=scored_claims, edges=edges, storage_dir=storage_dir)
    return engine.graph, scored_docs, scored_claims, edges


def build_evidence_graph(
    docs: list[SourceDoc],
    claims: list[Claim],
    entities: dict[str, list[str]],
    use_nli: bool = False,
) -> tuple[nx.MultiDiGraph, list[Claim], list[ConflictEdge]]:
    """Convenience entrypoint building and scoring an in-memory evidence graph."""
    docs_by_id = {d.source_id: d for d in docs}
    engine = EvidenceGraphEngine()
    engine.build_base_graph(docs=docs, claims=claims, entities=entities)
    edges = detect_claim_relations(claims=claims, docs_by_id=docs_by_id, use_nli=use_nli)
    engine.add_epistemic_edges(edges)
    _, scored_claims = propagate_trust(engine.graph, docs=docs, claims=claims, edges=edges)
    return engine.graph, scored_claims, edges

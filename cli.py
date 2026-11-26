"""Headless CLI entrypoint for CERBERUS-X backend workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import TypeAdapter

from core.config import settings
from core.extractor import extract_and_persist
from core.graph_engine import build_and_persist_graph
from core.ingest import ingest_directory
from core.reasoner import answer_query, evaluate_test_cases
from core.schemas import EdgeType, SourceDoc


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ingest, extract, query, and eval subcommands."""
    parser = argparse.ArgumentParser(
        prog="cerberus-x",
        description="CERBERUS-X offline multimodal evidence graph CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run multimodal forensics, ingestion, and JSON artifact persistence.",
    )
    ingest_parser.add_argument(
        "--input",
        default=str(settings.synthetic_traps_dir),
        help="Path to raw documents or directory (defaults to data/synthetic_traps).",
    )
    ingest_parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Enable optional local Ollama VLM summary for image/scan documents.",
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract atomic claims and resolve cross-modal entities from ingested storage.",
    )
    extract_parser.add_argument(
        "--storage",
        default=str(settings.storage_dir),
        help="Storage directory containing ingested_docs.json and parsed_chunks.json.",
    )
    extract_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable optional local Ollama claim extraction and entity tie-breaking.",
    )

    graph_parser = subparsers.add_parser(
        "build-graph",
        help="Build NetworkX evidence graph, detect conflicts/amendments, and propagate trust.",
    )
    graph_parser.add_argument(
        "--storage",
        default=str(settings.storage_dir),
        help="Storage directory containing ingested_docs.json, claims.json, and entities.json.",
    )
    graph_parser.add_argument(
        "--use-nli",
        action="store_true",
        help="Enable optional offline DeBERTa-v3 NLI check when model weights are cached.",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Query the offline evidence graph with deterministic abstention.",
    )
    query_parser.add_argument("question", help="Question to evaluate against the evidence graph.")
    query_parser.add_argument(
        "--storage",
        default=str(settings.storage_dir),
        help="Storage directory containing built evidence graph artifacts.",
    )
    query_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable optional local Ollama prose synthesis over the EvidenceDossier.",
    )
    query_parser.add_argument(
        "--use-vector",
        action="store_true",
        help="Enable optional offline ChromaDB retrieval hints.",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run offline benchmark evaluation cases.",
    )
    eval_parser.add_argument(
        "--cases",
        default=str(settings.eval_cases_path),
        help="Path to benchmark JSON cases (defaults to eval/test_cases.json).",
    )
    eval_parser.add_argument(
        "--storage",
        default=str(settings.storage_dir),
        help="Storage directory containing built evidence graph artifacts.",
    )
    eval_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable optional local Ollama prose synthesis during evaluation.",
    )
    return parser


def _run_ingest(input_path: Path, use_vlm: bool) -> dict[str, Any]:
    """Execute ingestion and build a structured forensic summary payload."""
    docs, chunks = ingest_directory(
        input_dir=input_path,
        storage_dir=settings.storage_dir,
        use_vlm=use_vlm,
    )
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts[chunk.source_id] = chunk_counts.get(chunk.source_id, 0) + 1

    return {
        "command": "ingest",
        "status": "ok",
        "input": str(input_path),
        "storage_dir": str(settings.storage_dir),
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "documents": [
            {
                "file_name": doc.file_name,
                "modality": doc.modality.value,
                "trust_prior": doc.trust_prior,
                "quarantined": doc.quarantined,
                "flags": [flag.code for flag in doc.forensic_flags],
                "chunks": chunk_counts.get(doc.source_id, 0),
            }
            for doc in docs
        ],
    }


def _run_extract(storage_dir: Path, use_llm: bool) -> dict[str, Any]:
    """Execute claim extraction + entity resolution and build a structured summary."""
    claims, entity_map = extract_and_persist(storage_dir=storage_dir, use_llm=use_llm)
    docs = TypeAdapter(list[SourceDoc]).validate_json(
        (storage_dir / "ingested_docs.json").read_text(encoding="utf-8")
    )
    file_by_source = {doc.source_id: doc.file_name for doc in docs}

    claims_by_entity: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        canon = claim.entity_canonical or claim.entity_raw
        claims_by_entity.setdefault(canon, []).append(
            {
                "attribute": claim.attribute,
                "value": claim.value,
                "source_file": file_by_source.get(claim.source_id, claim.source_id),
                "trust_score": claim.trust_score,
                "is_amendment": claim.is_amendment,
            }
        )

    return {
        "command": "extract",
        "status": "ok",
        "storage_dir": str(storage_dir),
        "claim_count": len(claims),
        "entities": entity_map,
        "claims_by_entity": claims_by_entity,
    }


def _run_build_graph(storage_dir: Path, use_nli: bool) -> dict[str, Any]:
    """Execute evidence graph build and return node/edge/trust/conflict summary."""
    graph, docs, scored_claims, edges = build_and_persist_graph(
        storage_dir=storage_dir,
        use_nli=use_nli,
    )
    file_by_source = {doc.source_id: doc.file_name for doc in docs}

    node_counts: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        ntype = str(attrs.get("node_type", "Unknown"))
        node_counts[ntype] = node_counts.get(ntype, 0) + 1

    edge_counts: dict[str, int] = {etype.value: 0 for etype in EdgeType}
    for edge in edges:
        edge_counts[edge.edge_type.value] = edge_counts.get(edge.edge_type.value, 0) + 1

    top_claims_by_entity: dict[str, list[dict[str, Any]]] = {}
    for claim in sorted(scored_claims, key=lambda c: c.trust_score, reverse=True):
        canon = claim.entity_canonical or claim.entity_raw
        entries = top_claims_by_entity.setdefault(canon, [])
        if len(entries) < 6:
            entries.append(
                {
                    "claim_id": claim.claim_id,
                    "attribute": claim.attribute,
                    "value": claim.value,
                    "source_file": file_by_source.get(claim.source_id, claim.source_id),
                    "trust_score": claim.trust_score,
                    "superseded": bool(graph.nodes[claim.claim_id].get("superseded", False)),
                }
            )

    conflicts_and_amendments = [
        {
            "edge_id": e.edge_id,
            "edge_type": e.edge_type.value,
            "claim_a_id": e.claim_a_id,
            "claim_b_id": e.claim_b_id,
            "detector": e.detector,
            "reason": e.reason,
        }
        for e in edges
        if e.edge_type in (EdgeType.CONTRADICTS, EdgeType.SUPERSEDES)
    ]

    return {
        "command": "build-graph",
        "status": "ok",
        "storage_dir": str(storage_dir),
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "node_counts": node_counts,
        "claim_count": len(scored_claims),
        "epistemic_edge_count": len(edges),
        "edge_counts": edge_counts,
        "top_claims_by_entity": top_claims_by_entity,
        "conflicts_and_amendments": conflicts_and_amendments,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI subcommands and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "ingest":
        payload = _run_ingest(Path(args.input), use_vlm=bool(args.use_vlm))
    elif args.command == "extract":
        payload = _run_extract(Path(args.storage), use_llm=bool(args.use_llm))
    elif args.command == "build-graph":
        payload = _run_build_graph(Path(args.storage), use_nli=bool(args.use_nli))
    elif args.command == "query":
        resp = answer_query(
            query=args.question,
            storage_dir=Path(args.storage),
            use_llm=bool(args.use_llm),
            use_vector=bool(args.use_vector),
        )
        payload = {
            **resp.model_dump(mode="json"),
            "command": "query",
            "question": args.question,
        }
    else:
        payload = evaluate_test_cases(
            cases_path=Path(args.cases),
            storage_dir=Path(args.storage),
            use_llm=bool(args.use_llm),
        )

    print(json.dumps(payload, indent=2))
    if args.command == "eval" and payload.get("passed", 0) != payload.get("total", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

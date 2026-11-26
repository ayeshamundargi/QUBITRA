"""Minimal, 100% offline Streamlit Evidence & Conflict Viewer for CERBERUS-X.

Pure helper functions (`run_full_pipeline`, `render_visual_proof`,
`build_offline_pyvis_html`) are decoupled from Streamlit UI state so they can
be tested headlessly via `pytest`.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from PIL import Image, ImageDraw
from pydantic import TypeAdapter
import pymupdf
from pyvis.network import Network
import streamlit as st

from core.config import settings
from core.extractor import extract_and_persist
from core.graph_engine import (
    build_and_persist_graph,
    load_graph_artifacts,
)
from core.ingest import ingest_directory
from core.reasoner import answer_query, evaluate_test_cases
from core.schemas import (
    LocationRef,
    Modality,
    SourceDoc,
    VerdictResponse,
    VerdictStatus,
)

_SHIELD_SVG = (
    '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true">'
    '<path d="M18 3L5 8.5V17.5C5 25.4 10.55 31.75 18 33.5C25.45 31.75 31 25.4 31 17.5V8.5L18 3Z" '
    'stroke="#111111" stroke-width="2.2" fill="#F4F4F5"/>'
    '<path d="M13 18.2L16.5 21.7L23.5 14.2" stroke="#111111" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    "</svg>"
)

# Display-only badge mappings (keep internal Pydantic enums and JSON drawer untouched).
_EDGE_DISPLAY_MAP: dict[str, str] = {
    "CONTRADICTS": "CONFLICT",
    "SUPERSEDES": "UPDATED RECORD",
    "SUPPORTS": "MATCHES",
}

_FLAG_DISPLAY_MAP: dict[str, str] = {
    "ELA_TAMPER_DETECTED": "EDITED IMAGE (ELA)",
    "HIDDEN_PDF_TEXT": "HIDDEN WHITE TEXT",
    "PROMPT_INJECTION": "AI HIJACK ATTEMPT",
    "LOW_LEGIBILITY_BLUR": "BLURRY SCAN",
    "MISSING_FIELD": "BLANK FIELD",
}


def _display_flag(code: str) -> str:
    """Return human-readable label for a forensic flag code (display only)."""
    return _FLAG_DISPLAY_MAP.get(code, code)


def _display_edge(edge_type: str) -> str:
    """Return human-readable label for an epistemic edge type (display only)."""
    return _EDGE_DISPLAY_MAP.get(edge_type, edge_type)


def _summarize_forensic_flags(flags: list[Any]) -> tuple[str, list[tuple[str, int]], str]:
    """Group forensic flags by code in order of appearance into badges and one concise summary line."""
    if not flags:
        return "INFO", [], "Flagged during forensic screening."

    severities = [str(f.severity.value) for f in flags]
    top_severity = "CRITICAL" if "CRITICAL" in severities else severities[0]

    grouped: dict[str, list[Any]] = {}
    for flag in flags:
        grouped.setdefault(str(flag.code), []).append(flag)

    badge_counts = [(code, len(items)) for code, items in grouped.items()]

    if len(flags) == 1:
        return top_severity, badge_counts, str(flags[0].detail)

    clauses: list[str] = []
    for code, items in grouped.items():
        if len(items) == 1:
            detail_txt = str(items[0].detail).strip().rstrip(".")
            if len(grouped) > 1 and ": '" in detail_txt:
                detail_txt = detail_txt.split(": '", 1)[0].strip()
            clauses.append(detail_txt)
            continue

        snippets: list[str] = []
        for item in items:
            raw_detail = str(item.detail)
            found = re.findall(r"'([^']+)'", raw_detail)
            if found:
                snip = found[-1]
                if snip not in snippets:
                    snippets.append(snip)
        if snippets:
            quoted = ", ".join(f"'{s}'" for s in snippets[:4])
            if len(snippets) > 4:
                quoted = f"{quoted}, +{len(snippets) - 4} more"
            clauses.append(f"{len(items)} matched patterns: {quoted}")
        else:
            first_prefix = str(items[0].detail).strip().rstrip(".")
            clauses.append(f"{len(items)} occurrences ({first_prefix})")

    summary = "; ".join(clauses)
    if summary and not summary.endswith("."):
        summary = f"{summary}."
    if len(summary) > 260:
        summary = f"{summary[:257].rstrip()}..."
    return top_severity, badge_counts, summary


def check_ollama_status(
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float = 1.5,
) -> tuple[bool, list[str]]:
    """Check whether local Ollama is reachable at `127.0.0.1:11434` and list models."""
    target_url = (base_url or settings.ollama_base_url).rstrip("/")
    target_model = model_name or settings.llm_model
    try:
        with urllib.request.urlopen(f"{target_url}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [str(m.get("name", "")) for m in data.get("models", []) if m.get("name")]
        has_model = any(
            m == target_model or m.startswith(f"{target_model}:") or target_model in m
            for m in models
        )
        return (has_model or bool(models)), models
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return False, []


def run_full_pipeline(
    input_dir: Path,
    storage_dir: Path,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Execute ingest -> extract -> build-graph sequentially and return summary counts."""
    docs, chunks = ingest_directory(input_dir=input_dir, storage_dir=storage_dir)
    claims, entity_map = extract_and_persist(storage_dir=storage_dir, use_llm=use_llm)
    graph, _, scored_claims, edges = build_and_persist_graph(storage_dir=storage_dir)
    return {
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "claim_count": len(scored_claims),
        "entity_count": len(entity_map),
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "epistemic_edge_count": len(edges),
    }


def render_visual_proof(
    doc: SourceDoc,
    loc: LocationRef,
    highlight_color: str = "#DC2626",
) -> Image.Image | None:
    """Render a highlighted Pillow image for PDF/IMAGE/SCAN provenance; None for tables."""
    if doc.modality not in (Modality.IMAGE, Modality.SCAN, Modality.PDF):
        return None

    file_path = Path(doc.file_path)
    if not file_path.exists():
        return None

    if doc.modality in (Modality.IMAGE, Modality.SCAN):
        with Image.open(file_path) as raw_img:
            canvas = raw_img.convert("RGB")
        if loc.bbox is not None:
            draw = ImageDraw.Draw(canvas)
            x0, y0, x1, y1 = loc.bbox
            pad = 3.0
            box = (
                max(0.0, x0 - pad),
                max(0.0, y0 - pad),
                min(float(canvas.width - 1), x1 + pad),
                min(float(canvas.height - 1), y1 + pad),
            )
            draw.rectangle(box, outline=highlight_color, width=4)
        return canvas

    # PDF page rendering at 150 DPI (PDF coordinates are 72 DPI points)
    dpi = 150
    zoom = dpi / 72.0
    pdf_doc = pymupdf.open(str(file_path))
    try:
        page_index = max(0, (loc.page or 1) - 1)
        if page_index >= len(pdf_doc):
            return None
        page = pdf_doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        canvas = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        pdf_doc.close()

    if loc.bbox is not None:
        draw = ImageDraw.Draw(canvas)
        x0, y0, x1, y1 = (coord * zoom for coord in loc.bbox)
        pad = 4.0
        box = (
            max(0.0, x0 - pad),
            max(0.0, y0 - pad),
            min(float(canvas.width - 1), x1 + pad),
            min(float(canvas.height - 1), y1 + pad),
        )
        draw.rectangle(box, outline=highlight_color, width=4)

    return canvas


def build_offline_pyvis_html(
    graph: nx.MultiDiGraph,
    highlight_entities: list[str] | None = None,
    height: str = "480px",
) -> str:
    """Build 100% offline self-contained PyVis HTML (`cdn_resources='in_line'`)."""
    if highlight_entities:
        undirected = graph.to_undirected(as_view=True)
        keep_nodes: set[str] = set()
        for ent in highlight_entities:
            if ent in graph:
                keep_nodes.update(
                    nx.single_source_shortest_path_length(undirected, ent, cutoff=2).keys()
                )
        view_graph = graph.subgraph(keep_nodes) if keep_nodes else graph
    else:
        view_graph = graph

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#111111",
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-3500, central_gravity=0.35, spring_length=130)

    quarantined_sources: set[str] = {
        str(nid)
        for nid, attrs in view_graph.nodes(data=True)
        if attrs.get("node_type") == "Source" and bool(attrs.get("quarantined", False))
    }

    for node_id, attrs in view_graph.nodes(data=True):
        ntype = str(attrs.get("node_type", "Unknown"))
        nid_str = str(node_id)

        if ntype == "Entity":
            aliases = ", ".join(attrs.get("aliases", []))
            net.add_node(
                nid_str,
                label=nid_str,
                title=f"Entity: {nid_str}\nAliases: {aliases}",
                shape="diamond",
                color="#2563EB",
                size=24,
            )
        elif ntype == "Source":
            is_q = bool(attrs.get("quarantined", False))
            fname = str(attrs.get("file_name", nid_str))
            mod = str(attrs.get("modality", ""))
            rel = float(attrs.get("dynamic_reliability", attrs.get("trust_prior", 0.0)))
            flags = [str(f.get("code", "")) for f in attrs.get("forensic_flags", [])]
            flag_str = ", ".join(flags) if flags else "None"
            net.add_node(
                nid_str,
                label=f"{fname}\n[{mod}]",
                title=(
                    f"Source: {fname} ({mod})\n"
                    f"Reliability: {rel:.2f}\nQuarantined: {is_q}\nFlags: {flag_str}"
                ),
                shape="box",
                color="#DC2626" if is_q else "#0D9488",
                font={"color": "#FFFFFF"},
            )
        else:
            is_q_claim = str(attrs.get("source_id", "")) in quarantined_sources
            is_sup = bool(attrs.get("superseded", False))
            attr_name = str(attrs.get("attribute", "claim"))
            val = str(attrs.get("value", ""))
            trust = float(attrs.get("trust_score", 0.0))
            if is_q_claim:
                color = "#EA580C"
            elif is_sup:
                color = "#9CA3AF"
            else:
                color = "#16A34A"
            short_val = val if len(val) <= 22 else f"{val[:19]}..."
            net.add_node(
                nid_str,
                label=f"{attr_name}: {short_val}",
                title=(
                    f"Claim: {nid_str}\n{attr_name} = {val}\n"
                    f"Trust: {trust:.3f}\nSuperseded: {is_sup}"
                ),
                shape="dot",
                color=color,
                size=13 if is_sup else 16,
            )

    for u, v, _, data in view_graph.edges(keys=True, data=True):
        etype = str(data.get("edge_type", "REL"))
        reason = str(data.get("reason", etype))
        if etype == "SUPPORTS":
            net.add_edge(str(u), str(v), label="SUPPORTS", title=reason, color="#16A34A", width=2)
        elif etype == "CONTRADICTS":
            net.add_edge(
                str(u),
                str(v),
                label="CONTRADICTS",
                title=reason,
                color="#DC2626",
                width=3,
                dashes=True,
            )
        elif etype == "SUPERSEDES":
            net.add_edge(str(u), str(v), label="SUPERSEDES", title=reason, color="#D97706", width=2.5)
        else:
            net.add_edge(str(u), str(v), label=etype, title=reason, color="#D4D4D8", width=1)

    raw_html = str(net.generate_html())
    # Strip hardcoded Bootstrap CDN tags emitted by PyVis 0.3.2 template so rendering is 100% offline.
    offline_html = re.sub(
        r"<link\b[^>]*href=[\"']https?://[^\"']+[\"'][^>]*>",
        "",
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    offline_html = re.sub(
        r"<script\b[^>]*src=[\"']https?://[^\"']+[\"'][^>]*>\s*</script>",
        "",
        offline_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return offline_html


def _load_ingested_docs(storage_dir: Path) -> list[SourceDoc]:
    """Load persisted SourceDoc records if available."""
    path = storage_dir / "ingested_docs.json"
    if not path.exists():
        return []
    return TypeAdapter(list[SourceDoc]).validate_json(path.read_text(encoding="utf-8"))


def _load_test_presets() -> list[dict[str, Any]]:
    """Load preset benchmark queries from eval/test_cases.json."""
    if not settings.eval_cases_path.exists():
        return []
    return list(json.loads(settings.eval_cases_path.read_text(encoding="utf-8")))


def _format_location(loc: LocationRef) -> str:
    """Format LocationRef concisely into an audit tag string."""
    parts: list[str] = []
    if loc.page is not None:
        parts.append(f"page={loc.page}")
    if loc.row is not None:
        parts.append(f"row={loc.row}")
    if loc.column:
        parts.append(f"col={loc.column}")
    if loc.bbox is not None and loc.row is None:
        coords = ",".join(str(int(round(c))) for c in loc.bbox)
        parts.append(f"bbox=({coords})")
    return " · ".join(parts) if parts else "source"


def _infer_active_corpus(docs: list[SourceDoc]) -> tuple[str, str]:
    """Infer currently indexed corpus directory path and label from ingested SourceDocs."""
    if not docs:
        return "", "Not indexed"
    try:
        parent = Path(docs[0].file_path).resolve().parent
        if parent == settings.synthetic_traps_dir.resolve():
            return str(settings.synthetic_traps_dir), "synthetic_traps"
        if parent == settings.data_raw_dir.resolve():
            return str(settings.data_raw_dir), "data/raw"
        return str(parent), parent.name
    except OSError:
        return "", "Indexed corpus"


def _count_visual_proofs(response: VerdictResponse, docs_by_id: dict[str, SourceDoc]) -> int:
    """Count matching visual and tabular proof items in Tab 2."""
    count = 0
    for cit in response.citations:
        doc = docs_by_id.get(cit.source_id)
        if doc is None:
            continue
        if Path(doc.file_path).exists() and doc.modality in (
            Modality.TABLE,
            Modality.IMAGE,
            Modality.SCAN,
            Modality.PDF,
        ):
            count += 1
    for q_doc in response.quarantined_sources:
        if (
            q_doc.modality in (Modality.IMAGE, Modality.SCAN, Modality.PDF)
            and Path(q_doc.file_path).exists()
        ):
            count += 1
    return count


def _select_preset_callback(case_id: str, case_query: str) -> None:
    """Update session state before rerun when a preset scenario button is clicked."""
    st.session_state["active_case_id"] = case_id
    st.session_state["query_input"] = case_query
    st.session_state["run_query"] = True


def _inject_minimal_styles() -> None:
    """Apply subtle monochrome black-and-white CSS polish and clean card typography."""
    css_block = "\n".join(
        [
            "<style>",
            "[data-testid=\"stDeployButton\"], [data-testid=\"stAppDeployButton\"] { display: none !important; }",
            "header[data-testid=\"stHeader\"] { background: transparent !important; }",
            ".block-container, [data-testid=\"stMainBlockContainer\"] { padding-top: 3.75rem !important; padding-bottom: 2.5rem; }",
            ".cerberus-header { display: flex; align-items: center; gap: 0.9rem; border-bottom: 1px solid #E4E4E7; padding-bottom: 0.85rem; margin-top: 0.25rem; margin-bottom: 0.9rem; overflow: visible; }",
            ".cerberus-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #71717A; margin-bottom: 0.15rem; }",
            ".cerberus-title { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.025em; color: #111111; line-height: 1.15; margin: 0; }",
            ".cerberus-sub { font-size: 0.88rem; color: #52525B; margin-top: 0.2rem; }",
            ".sidebar-eyebrow { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #52525B; margin: 0.35rem 0 0.5rem 0; }",
            ".corpus-strip { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.35rem; padding: 8px 12px; background: #F4F4F5; border: 1px solid #E5E7EB; border-radius: 6px; margin-bottom: 0.9rem; font-size: 0.8rem; color: #52525B; }",
            ".corpus-item { display: inline-flex; align-items: center; gap: 0.35rem; }",
            ".corpus-lbl { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #71717A; }",
            ".corpus-val { font-weight: 600; color: #111111; font-variant-numeric: tabular-nums; }",
            ".corpus-val-warn { font-weight: 700; color: #991B1B; font-variant-numeric: tabular-nums; }",
            ".status-pill { display: inline-block; padding: 0.18rem 0.55rem; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; border-radius: 4px; border: 1px solid #D4D4D8; margin-bottom: 0.35rem; }",
            ".pill-ready { background: #F4F4F5; color: #111111; border-color: #111111; }",
            ".pill-offline { background: #FAFAFA; color: #71717A; border-color: #E4E4E7; }",
            ".pill-verified { background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }",
            ".pill-conflict { background: #FFFBEB; color: #92400E; border-color: #FDE68A; }",
            ".pill-tamper { background: #FEF2F2; color: #991B1B; border-color: #FECACA; }",
            ".pill-abstain { background: #F4F4F5; color: #27272A; border-color: #D4D4D8; }",
            ".micro-badge { display: inline-block; padding: 2px 7px; font-size: 0.68rem; font-weight: 600; border-radius: 4px; background: #F4F4F5; color: #27272A; border: 1px solid #E5E7EB; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
            ".badge-clean { background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }",
            ".badge-quarantined { background: #FEF2F2; color: #991B1B; border-color: #FECACA; }",
            ".badge-amber { background: #FFFBEB; color: #92400E; border-color: #FDE68A; }",
            ".sb-file-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 8px 10px; margin-bottom: 7px; }",
            ".sb-file-card-q { border-color: #FECACA; border-left: 3px solid #DC2626; background: #FFFDFD; }",
            ".sb-line1 { display: flex; align-items: flex-start; justify-content: space-between; gap: 6px; margin-bottom: 5px; }",
            ".sb-fname { font-size: 0.78rem; font-weight: 600; color: #111111; overflow-wrap: anywhere; line-height: 1.3; }",
            ".sb-right { display: inline-flex; align-items: center; gap: 4px; flex-shrink: 0; }",
            ".sb-line2 { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }",
            ".flag-chip { display: inline-block; padding: 1px 6px; font-size: 0.64rem; font-weight: 600; border-radius: 4px; background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
            ".verdict-card { display: flex; flex-wrap: wrap; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; margin: 0.85rem 0 1.1rem 0; overflow: hidden; }",
            ".verdict-left { flex: 3 1 500px; padding: 14px 18px; }",
            ".verdict-right { flex: 1 1 200px; padding: 14px 16px; background: #FAFAFA; border-left: 1px solid #E5E7EB; display: flex; flex-direction: column; justify-content: center; }",
            ".verdict-answer { font-size: 0.97rem; line-height: 1.55; color: #111111; font-weight: 500; margin: 6px 0 8px 0; }",
            ".verdict-cites { display: flex; flex-wrap: wrap; gap: 5px; }",
            ".verdict-q-note { font-size: 0.78rem; color: #52525B; margin-top: 8px; }",
            ".conf-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #71717A; margin-bottom: 4px; }",
            ".conf-num { font-size: 2rem; font-weight: 700; line-height: 1; color: #111111; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
            ".meter-wrap { position: relative; padding-top: 6px; padding-bottom: 18px; }",
            ".meter-track { position: relative; width: 100%; height: 8px; background: #E5E7EB; border-radius: 999px; }",
            ".meter-fill { height: 100%; border-radius: 999px; }",
            ".meter-dot { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; background: #FFFFFF; border: 3px solid #111111; box-sizing: border-box; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.28); z-index: 3; }",
            ".meter-tick { position: absolute; left: 65%; top: -4px; width: 2px; height: 16px; background: #111111; z-index: 2; }",
            ".meter-tick-lbl { position: absolute; left: 65%; top: 14px; transform: translateX(-50%); font-size: 0.64rem; font-weight: 700; color: #111111; white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
            ".meter-scale { display: flex; justify-content: space-between; font-size: 0.64rem; color: #71717A; margin-top: 2px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
            ".ev-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; }",
            ".ev-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 6px; }",
            ".ev-title { font-size: 0.83rem; font-weight: 600; color: #111111; overflow-wrap: anywhere; }",
            ".ev-quote { font-size: 0.86rem; line-height: 1.5; color: #27272A; border-left: 3px solid #D4D4D8; padding-left: 10px; overflow-wrap: anywhere; }",
            ".ev-reason { font-size: 0.85rem; line-height: 1.5; color: #27272A; overflow-wrap: anywhere; }",
            ".q-card { background: #FFFDFD; border: 1px solid #FECACA; border-left: 3px solid #DC2626; border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; }",
            ".q-detail { margin-top: 6px; padding: 6px 8px; background: #FFFFFF; border: 1px solid #FEE2E2; border-radius: 4px; font-size: 0.82rem; line-height: 1.45; color: #27272A; overflow-wrap: anywhere; }",
            ".graph-legend { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 0.78rem; color: #111111; }",
            ".gl-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin-bottom: 6px; }",
            ".gl-lbl { font-size: 0.67rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #71717A; min-width: 92px; }",
            ".gl-items { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; }",
            ".gl-item { display: inline-flex; align-items: center; gap: 6px; color: #111111; }",
            ".gl-sub { color: #52525B; font-size: 0.74rem; }",
            ".gl-hint { font-size: 0.72rem; color: #71717A; border-top: 1px solid #E5E7EB; padding-top: 6px; margin-top: 2px; }",
            "div.stButton > button { border: 1px solid #D4D4D8; border-radius: 6px; font-weight: 500; }",
            "</style>",
        ]
    )
    st.markdown(css_block, unsafe_allow_html=True)


def _render_sidebar(storage_dir: Path) -> tuple[bool, bool, str, list[SourceDoc]]:
    """Render sidebar controls, wrapped forensics cards, and 5-case benchmark runner."""
    with st.sidebar:
        st.markdown('<div class="sidebar-eyebrow">1. DOCUMENTS &amp; UPLOAD</div>', unsafe_allow_html=True)
        corpus_choice = st.selectbox(
            "Source Folder",
            options=[str(settings.synthetic_traps_dir), str(settings.data_raw_dir)],
            format_func=lambda p: "synthetic_traps (6 adversarial files)"
            if "synthetic_traps" in p
            else "data/raw (uploaded / custom)",
        )

        uploaded_files = st.file_uploader(
            "Upload Additional Documents",
            accept_multiple_files=True,
            type=["pdf", "csv", "png", "jpg", "jpeg", "txt", "json"],
        )
        if uploaded_files:
            settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
            uploaded_sig = sorted((up.name, len(up.getvalue())) for up in uploaded_files)
            for up in uploaded_files:
                (settings.data_raw_dir / up.name).write_bytes(up.getvalue())
            if st.session_state.get("last_uploaded_sig") != uploaded_sig:
                st.session_state["last_uploaded_sig"] = uploaded_sig
                st.session_state["pending_reingest"] = True
            st.caption(f"Saved {len(uploaded_files)} file(s) to data/raw.")

        docs = _load_ingested_docs(storage_dir)
        active_corpus_path, _ = _infer_active_corpus(docs)
        folder_differs = bool(active_corpus_path) and (
            Path(corpus_choice).resolve() != Path(active_corpus_path).resolve()
        )
        if not docs:
            st.info("No indexed corpus in storage yet. Run Step 2 below.")
        elif folder_differs:
            st.caption("Notice: Selected folder differs from the indexed corpus. Run Step 2 to re-index.")
        elif st.session_state.get("pending_reingest"):
            st.caption("Notice: Newly uploaded files require running Step 2 to index.")

        st.markdown(
            '<div class="sidebar-eyebrow" style="margin-top:0.9rem;">2. RUN OFFLINE PIPELINE</div>',
            unsafe_allow_html=True,
        )
        ollama_ready, available_models = check_ollama_status()
        model_label = (
            settings.llm_model
            if settings.llm_model in available_models
            else (available_models[0] if available_models else settings.llm_model)
        )
        safe_model = html.escape(str(model_label))
        if ollama_ready:
            st.markdown(
                f'<span class="status-pill pill-ready">Ollama: {safe_model} Ready</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill pill-offline">Offline (Deterministic Fallback)</span>',
                unsafe_allow_html=True,
            )

        use_llm = st.checkbox("Use Local Ollama LLM", value=ollama_ready)

        if st.button("Run Offline Pipeline", width="stretch", type="primary"):
            with st.spinner("Running forensics, OCR, extraction, and graph propagation..."):
                summary = run_full_pipeline(
                    input_dir=Path(corpus_choice),
                    storage_dir=storage_dir,
                    use_llm=use_llm,
                )
            st.session_state["pending_reingest"] = False
            st.session_state.pop("last_response", None)
            st.session_state.pop("last_query_text", None)
            docs = _load_ingested_docs(storage_dir)
            st.success(
                f"Indexed {summary['doc_count']} docs · {summary['claim_count']} claims · "
                f"{summary['epistemic_edge_count']} epistemic edges"
            )

        st.markdown(
            '<div class="sidebar-eyebrow" style="margin-top:1rem;">3. FILE SECURITY CHECKS</div>',
            unsafe_allow_html=True,
        )
        if docs:
            file_cards: list[str] = []
            for d in docs:
                fname = html.escape(str(d.file_name))
                mod = html.escape(str(d.modality.value).upper())
                trust_str = f"{d.trust_prior:.2f}"
                card_cls = "sb-file-card sb-file-card-q" if d.quarantined else "sb-file-card"
                st_cls = "badge-quarantined" if d.quarantined else "badge-clean"
                st_txt = "Quarantined" if d.quarantined else "Clean"

                flag_codes = sorted({str(f.code) for f in d.forensic_flags})
                if flag_codes:
                    flags_html = "".join(
                        f'<span class="flag-chip">{html.escape(_display_flag(code))}</span>'
                        for code in flag_codes
                    )
                else:
                    flags_html = '<span class="micro-badge" style="color:#71717A;">No flags</span>'

                file_cards.append(
                    f'<div class="{card_cls}">'
                    f'<div class="sb-line1">'
                    f'<div class="sb-fname">{fname}</div>'
                    f'<div class="sb-right">'
                    f'<span class="micro-badge">{trust_str}</span>'
                    f'<span class="micro-badge {st_cls}">{st_txt}</span>'
                    f"</div>"
                    f"</div>"
                    f'<div class="sb-line2">'
                    f'<span class="micro-badge">{mod}</span>'
                    f"{flags_html}"
                    f"</div>"
                    f"</div>"
                )
            st.markdown("".join(file_cards), unsafe_allow_html=True)
        else:
            st.caption("No ingested documents in storage yet.")

        st.divider()
        st.markdown('<div class="sidebar-eyebrow">Benchmark Suite</div>', unsafe_allow_html=True)
        if st.button("Run 5-Case Benchmark", width="stretch"):
            with st.spinner("Evaluating 5 adversarial cases..."):
                report = evaluate_test_cases(
                    cases_path=settings.eval_cases_path,
                    storage_dir=storage_dir,
                    use_llm=use_llm,
                )
            st.session_state["eval_report"] = report

        report = st.session_state.get("eval_report")
        if report:
            m = report["metrics"]
            c1, c2 = st.columns(2)
            c1.metric("Pass Rate", f"{report['passed']}/{report['total']}")
            c2.metric("Injection Block", f"{int(m['injection_block_rate'] * 100)}%")
            c3, c4 = st.columns(2)
            c3.metric("Conflict Recall", f"{int(m['conflict_recall'] * 100)}%")
            c4.metric("Abstention Acc.", f"{int(m['abstention_accuracy'] * 100)}%")

    return use_llm, ollama_ready, str(model_label), docs


def _render_verdict_banner(response: VerdictResponse, query_text: str = "") -> None:
    """Render unified VerdictStatus banner with percentage readout and animated 65% gate meter."""
    status = response.status
    spec_map = {
        VerdictStatus.VERIFIED: ("pill-verified", "VERIFIED", "#16A34A"),
        VerdictStatus.CONFLICT_RESOLVED: ("pill-conflict", "CONFLICT_RESOLVED", "#D97706"),
        VerdictStatus.TAMPER_DETECTED: ("pill-tamper", "TAMPER_DETECTED", "#DC2626"),
        VerdictStatus.INSUFFICIENT_EVIDENCE: ("pill-abstain", "INSUFFICIENT_EVIDENCE", "#52525B"),
    }
    pill_cls, pill_label, accent_hex = spec_map[status]

    cite_tags: list[str] = []
    for idx, cit in enumerate(response.citations, start=1):
        loc_str = _format_location(cit.location_ref)
        cite_tags.append(
            f'<span class="micro-badge">[{idx}] {html.escape(str(cit.file_name))} · {html.escape(loc_str)}</span>'
        )
    cites_html = f'<div class="verdict-cites">{"".join(cite_tags)}</div>' if cite_tags else ""

    q_note_html = ""
    if response.quarantined_sources and status != VerdictStatus.TAMPER_DETECTED:
        q_names = ", ".join(str(s.file_name) for s in response.quarantined_sources)
        q_note_html = (
            f'<div class="verdict-q-note">'
            f"Pre-LLM forensics quarantined linked untrusted file(s): "
            f"<strong>{html.escape(q_names)}</strong>"
            f"</div>"
        )

    score_pct = max(0, min(100, int(round(float(response.confidence) * 100))))
    gate_pct = int(round(float(settings.abstention_threshold) * 100))

    if status == VerdictStatus.TAMPER_DETECTED:
        gate_badge = '<span class="micro-badge badge-quarantined">Blocked</span>'
    elif score_pct >= gate_pct and status != VerdictStatus.INSUFFICIENT_EVIDENCE:
        gate_badge = f'<span class="micro-badge badge-clean">Pass &gt;= {gate_pct}%</span>'
    else:
        gate_badge = f'<span class="micro-badge badge-amber">Below {gate_pct}%</span>'

    uid_seed = f"{query_text}|{status.value}|{score_pct}"
    uid = hashlib.md5(uid_seed.encode("utf-8")).hexdigest()[:6]
    dot_end_tx = "0%" if score_pct == 0 else "-50%"

    anim_style = (
        "<style>"
        f"@keyframes cerbFill_{uid}{{0%{{width:0%;}}100%{{width:{score_pct}%;}}}}"
        f"@keyframes cerbDot_{uid}{{0%{{left:0%;transform:translate(0%,-50%);}}100%{{left:{score_pct}%;transform:translate({dot_end_tx},-50%);}}}}"
        "</style>"
    )

    safe_answer = html.escape(str(response.answer))

    card_html = "".join(
        [
            anim_style,
            f'<div class="verdict-card" style="border-left: 4px solid {accent_hex};">',
            '<div class="verdict-left">',
            f'<span class="status-pill {pill_cls}">{pill_label}</span>',
            f'<div class="verdict-answer">{safe_answer}</div>',
            cites_html,
            q_note_html,
            "</div>",
            '<div class="verdict-right">',
            '<div class="conf-label">Confidence</div>',
            '<div style="display:flex; align-items:baseline; justify-content:space-between; gap:6px; margin-bottom:8px;">',
            f'<span class="conf-num">{score_pct}%</span>',
            gate_badge,
            "</div>",
            '<div class="meter-wrap">',
            '<div class="meter-track">',
            f'<div class="meter-fill" style="width:{score_pct}%;background:{accent_hex};animation:cerbFill_{uid} 0.75s cubic-bezier(0.16, 1, 0.3, 1) forwards;"></div>',
            f'<div class="meter-tick" style="left:{gate_pct}%;" title="Cutoff: {gate_pct}%"></div>',
            f'<div class="meter-dot" style="left:{score_pct}%;border-color:{accent_hex};transform:translate({dot_end_tx},-50%);animation:cerbDot_{uid} 0.75s cubic-bezier(0.16, 1, 0.3, 1) forwards;"></div>',
            f'<div class="meter-tick-lbl" style="left:{gate_pct}%;">{gate_pct}% min. trust</div>',
            "</div>",
            "</div>",
            '<div class="meter-scale"><span>0%</span><span>100%</span></div>',
            "</div>",
            "</div>",
        ]
    )
    st.markdown(card_html, unsafe_allow_html=True)


def _render_visual_proof_tab(
    response: VerdictResponse,
    docs_by_id: dict[str, SourceDoc],
) -> None:
    """Render highlighted image/PDF crops and Pandas styled CSV rows."""
    rendered_any = False

    if response.citations:
        st.markdown('<div class="sidebar-eyebrow">Supporting Evidence Provenance</div>', unsafe_allow_html=True)
        for idx, cit in enumerate(response.citations, start=1):
            doc = docs_by_id.get(cit.source_id)
            if doc is None:
                continue
            if doc.modality == Modality.TABLE and Path(doc.file_path).exists():
                rendered_any = True
                st.markdown(f"**{idx}. `{doc.file_name}`** (Row {cit.location_ref.row})")
                df = pd.read_csv(doc.file_path, dtype=str, keep_default_na=False)
                target_idx = (cit.location_ref.row - 1) if cit.location_ref.row else -1

                def _highlight_row(row: pd.Series) -> list[str]:
                    if row.name == target_idx:
                        return ["background-color: #E4E4E7; font-weight: 700; color: #111111"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df.style.apply(_highlight_row, axis=1),
                    hide_index=True,
                    width="stretch",
                    alt=f"Tabular proof from {doc.file_name}",
                )
            else:
                img = render_visual_proof(doc, cit.location_ref, highlight_color="#16A34A")
                if img is not None:
                    rendered_any = True
                    st.markdown(f"**{idx}. `{doc.file_name}`** (Page {cit.location_ref.page or 1})")
                    st.image(img, width="stretch", alt=f"Visual proof from {doc.file_name}")

    # Also render visual proof of quarantined / tampered / blurred sources linked to query
    visual_quarantined = [
        s
        for s in response.quarantined_sources
        if s.modality in (Modality.IMAGE, Modality.SCAN, Modality.PDF)
    ]
    if visual_quarantined:
        st.markdown('<div class="sidebar-eyebrow">Quarantined / Flagged Forensic Proof</div>', unsafe_allow_html=True)
        cols = st.columns(min(2, len(visual_quarantined)))
        for idx, q_doc in enumerate(visual_quarantined):
            loc = next((f.location for f in q_doc.forensic_flags if f.location is not None), None)
            if loc is None:
                loc = LocationRef(page=1) if q_doc.modality == Modality.PDF else LocationRef(bbox=(0.0, 0.0, 1.0, 1.0))
            img = render_visual_proof(q_doc, loc, highlight_color="#DC2626")
            if img is not None:
                rendered_any = True
                flag_codes = ", ".join(sorted({_display_flag(str(f.code)) for f in q_doc.forensic_flags}))
                with cols[idx % len(cols)]:
                    st.markdown(f"**`{q_doc.file_name}`** (`{flag_codes}`)")
                    st.image(img, width="stretch", alt=f"Quarantined proof from {q_doc.file_name}")

    if not rendered_any:
        st.info("No visual or tabular proof artifacts matched this query.")


def main() -> None:
    """Launch the CERBERUS-X minimal offline Streamlit UI."""
    st.set_page_config(page_title="CERBERUS-X", page_icon=_SHIELD_SVG, layout="wide")
    _inject_minimal_styles()

    header_html = "".join(
        [
            '<div class="cerberus-header">',
            _SHIELD_SVG,
            "<div>",
            '<div class="cerberus-eyebrow">Offline Forensic Intelligence Engine</div>',
            '<div class="cerberus-title">CERBERUS-X</div>',
            '<div class="cerberus-sub">100% Offline Multimodal Evidence Graph &amp; Conflict-Aware Reasoning Engine</div>',
            "</div>",
            "</div>",
        ]
    )
    st.markdown(header_html, unsafe_allow_html=True)

    storage_dir = settings.storage_dir
    use_llm, ollama_ready, model_label, ingested_docs = _render_sidebar(storage_dir)

    # Slim 1-row Corpus Context Strip directly above Demo Scenarios
    _, corpus_label = _infer_active_corpus(ingested_docs)
    total_files = len(ingested_docs)
    quarantined_count = sum(1 for d in ingested_docs if d.quarantined)
    clean_count = total_files - quarantined_count
    ollama_status_txt = f"Ollama ({model_label})" if ollama_ready else "Deterministic Offline"
    q_val_cls = "corpus-val-warn" if quarantined_count > 0 else "corpus-val"

    strip_html = "".join(
        [
            '<div class="corpus-strip">',
            f'<div class="corpus-item"><span class="corpus-lbl">LOADED DATASET</span><span class="corpus-val">{html.escape(corpus_label)}</span></div>',
            f'<div class="corpus-item"><span class="corpus-lbl">Total Files</span><span class="corpus-val">{total_files}</span></div>',
            f'<div class="corpus-item"><span class="corpus-lbl">Clean</span><span class="corpus-val">{clean_count}</span></div>',
            f'<div class="corpus-item"><span class="corpus-lbl">Quarantined</span><span class="{q_val_cls}">{quarantined_count}</span></div>',
            f'<div class="corpus-item"><span class="corpus-lbl">Ollama Status</span><span class="corpus-val">{html.escape(ollama_status_txt)}</span></div>',
            "</div>",
        ]
    )
    st.markdown(strip_html, unsafe_allow_html=True)

    # Zone 1: Query Bar & One-Click Presets (state initialized BEFORE widgets)
    presets = _load_test_presets()
    preset_labels = {
        "TC-01": "TC-01 · Conflict",
        "TC-02": "TC-02 · Amendment",
        "TC-03": "TC-03 · Abstention",
        "TC-04": "TC-04 · Verified",
        "TC-05": "TC-05 · Injection",
    }
    fallback_q = (
        str(presets[0]["query"])
        if presets
        else "What is the approved amount for invoice INV-402?"
    )
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = fallback_q

    current_q = str(st.session_state.get("query_input", fallback_q)).strip()
    active_cid = next(
        (str(c["case_id"]) for c in presets if str(c["query"]).strip() == current_q),
        None,
    )
    st.session_state["active_case_id"] = active_cid

    if presets:
        st.markdown('<div class="sidebar-eyebrow">Demo Scenarios</div>', unsafe_allow_html=True)
        cols = st.columns(len(presets))
        for col, case in zip(cols, presets):
            cid = str(case["case_id"])
            cquery = str(case["query"])
            btn_label = preset_labels.get(cid, cid)
            is_active = active_cid == cid
            col.button(
                btn_label,
                key=f"preset_{cid}",
                help=cquery,
                width="stretch",
                type="primary" if is_active else "secondary",
                on_click=_select_preset_callback,
                args=(cid, cquery),
            )

    q_col, btn_col = st.columns([5, 1])
    query_text = q_col.text_input(
        "Ask a verifiable question across your multimodal corpus",
        key="query_input",
        label_visibility="collapsed",
        placeholder="Ask across PDFs, CSV ledgers, and scanned receipts...",
    )
    analyze_clicked = btn_col.button("Analyze", width="stretch", type="primary")

    cleaned_q = query_text.strip()
    last_q = str(st.session_state.get("last_query_text", ""))
    should_run = bool(
        cleaned_q
        and (
            analyze_clicked
            or st.session_state.pop("run_query", False)
            or (st.session_state.get("last_response") is not None and cleaned_q != last_q)
        )
    )
    if should_run:
        st.session_state["last_query_text"] = cleaned_q
        st.session_state["last_response"] = answer_query(
            cleaned_q,
            storage_dir=storage_dir,
            use_llm=use_llm,
        )

    response: VerdictResponse | None = st.session_state.get("last_response")
    if response is None:
        if (storage_dir / "evidence_graph.json").exists():
            st.session_state["last_query_text"] = cleaned_q
            response = answer_query(cleaned_q, storage_dir=storage_dir, use_llm=use_llm)
            st.session_state["last_response"] = response
        else:
            st.info("Click **Run Offline Pipeline** in the sidebar to ingest documents and build the evidence graph.")
            return

    # Zone 2: Unified Verdict Banner
    _render_verdict_banner(response, query_text=cleaned_q)

    # Zone 3: 3-Tab Evidence Inspector with live counts
    docs_by_id = {d.source_id: d for d in ingested_docs}
    conflicts_citations_count = (
        len(response.citations) + len(response.conflicts) + len(response.quarantined_sources)
    )
    visual_proof_count = _count_visual_proofs(response, docs_by_id=docs_by_id)

    tab_conflicts, tab_visual, tab_graph = st.tabs(
        [
            f"Conflicts & Citations ({conflicts_citations_count})",
            f"Visual Proof ({visual_proof_count})",
            "Evidence Graph",
        ]
    )

    with tab_conflicts:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown('<div class="sidebar-eyebrow">TRUSTED SOURCES</div>', unsafe_allow_html=True)
            if response.citations:
                cit_cards: list[str] = []
                for idx, c in enumerate(response.citations, start=1):
                    loc_str = _format_location(c.location_ref)
                    cit_cards.append(
                        f'<div class="ev-card">'
                        f'<div class="ev-head">'
                        f'<span class="ev-title">[{idx}] {html.escape(str(c.file_name))}</span>'
                        f'<span class="micro-badge">{html.escape(loc_str)}</span>'
                        f"</div>"
                        f'<div class="ev-quote">{html.escape(str(c.quote))}</div>'
                        f"</div>"
                    )
                st.markdown("".join(cit_cards), unsafe_allow_html=True)
            else:
                st.caption("No clean corroborating citations for this verdict.")

        with c_right:
            st.markdown('<div class="sidebar-eyebrow">CONFLICTS &amp; UPDATES</div>', unsafe_allow_html=True)
            if response.conflicts:
                edge_cards: list[str] = []
                for e in response.conflicts:
                    etype = str(e.edge_type.value)
                    if etype == "CONTRADICTS":
                        etype_cls = "badge-quarantined"
                    elif etype == "SUPERSEDES":
                        etype_cls = "badge-amber"
                    else:
                        etype_cls = "badge-clean"
                    edge_label = _display_edge(etype)
                    edge_cards.append(
                        f'<div class="ev-card">'
                        f'<div class="ev-head">'
                        f'<span class="micro-badge {etype_cls}">{html.escape(edge_label)}</span>'
                        f'<div class="sb-line2">'
                        f'<span class="micro-badge">{html.escape(str(e.detector))}</span>'
                        f'<span class="micro-badge">conf {e.confidence:.2f}</span>'
                        f"</div>"
                        f"</div>"
                        f'<div class="ev-reason">{html.escape(str(e.reason))}</div>'
                        f"</div>"
                    )
                st.markdown("".join(edge_cards), unsafe_allow_html=True)
            else:
                if response.status == VerdictStatus.TAMPER_DETECTED or not response.citations:
                    st.caption("Blocked before comparison \u2014 no trusted sources available.")
                else:
                    st.caption("All trusted sources agree \u2014 no conflicts or updates found.")

        if response.quarantined_sources:
            st.markdown('<div class="sidebar-eyebrow">BLOCKED / TAMPERED FILES</div>', unsafe_allow_html=True)
            q_cards: list[str] = []
            for s in response.quarantined_sources:
                top_sev, badge_counts, summary_line = _summarize_forensic_flags(s.forensic_flags)
                sev_cls = "badge-quarantined" if top_sev == "CRITICAL" else "badge-amber"
                flag_badges_html = "".join(
                    f'<span class="micro-badge">{html.escape(_display_flag(code) + (f" ({cnt})" if cnt > 1 else ""))}</span>'
                    for code, cnt in badge_counts
                )
                q_cards.append(
                    f'<div class="q-card">'
                    f'<div class="ev-head" style="margin-bottom:4px;">'
                    f'<span class="ev-title">{html.escape(str(s.file_name))}</span>'
                    f'<div class="sb-line2">'
                    f'<span class="micro-badge">{html.escape(str(s.modality.value).upper())}</span>'
                    f'<span class="micro-badge">trust {s.trust_prior:.2f}</span>'
                    f"</div>"
                    f"</div>"
                    f'<div class="q-detail">'
                    f'<div class="sb-line2" style="margin-bottom:4px;">'
                    f'<span class="micro-badge {sev_cls}">{html.escape(top_sev)}</span>'
                    f"{flag_badges_html}"
                    f"</div>"
                    f'<div>{html.escape(summary_line)}</div>'
                    f"</div>"
                    f"</div>"
                )
            st.markdown("".join(q_cards), unsafe_allow_html=True)

    with tab_visual:
        _render_visual_proof_tab(response, docs_by_id=docs_by_id)

    with tab_graph:
        if (storage_dir / "evidence_graph.json").exists():
            show_full = st.checkbox("Show full corpus graph", value=False)
            legend_html = "".join(
                [
                    '<div class="graph-legend">',
                    '<div class="gl-row">',
                    '<span class="gl-lbl">NODES</span>',
                    '<div class="gl-items">',
                    '<span class="gl-item"><svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true" style="flex-shrink:0;"><polygon points="6,0 12,6 6,12 0,6" fill="#2563EB"/></svg><strong>Target Entity</strong> <span class="gl-sub">(e.g., Invoice / Shipment)</span></span>',
                    '<span class="gl-item"><span style="display:inline-block;width:13px;height:10px;border-radius:2px;background:#0D9488;flex-shrink:0;"></span><strong>Clean File</strong></span>',
                    '<span class="gl-item"><span style="display:inline-block;width:13px;height:10px;border-radius:2px;background:#DC2626;flex-shrink:0;"></span><strong>Blocked / Tampered File</strong></span>',
                    '<span class="gl-item"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#16A34A;flex-shrink:0;"></span><strong>Trusted Claim</strong></span>',
                    '<span class="gl-item"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#EA580C;flex-shrink:0;"></span><strong>Untrusted Claim</strong></span>',
                    "</div>",
                    "</div>",
                    '<div class="gl-row" style="margin-bottom:4px;">',
                    '<span class="gl-lbl">CONNECTIONS</span>',
                    '<div class="gl-items">',
                    '<span class="gl-item"><svg width="20" height="8" viewBox="0 0 20 8" aria-hidden="true" style="flex-shrink:0;"><line x1="0" y1="4" x2="14" y2="4" stroke="#16A34A" stroke-width="2"/><polygon points="14,1 20,4 14,7" fill="#16A34A"/></svg><strong>Supports</strong> <span class="gl-sub">(sources agree)</span></span>',
                    '<span class="gl-item"><svg width="20" height="8" viewBox="0 0 20 8" aria-hidden="true" style="flex-shrink:0;"><line x1="0" y1="4" x2="14" y2="4" stroke="#DC2626" stroke-width="2" stroke-dasharray="4,2"/><polygon points="14,1 20,4 14,7" fill="#DC2626"/></svg><strong>Contradicts</strong> <span class="gl-sub">(conflict detected)</span></span>',
                    '<span class="gl-item"><svg width="20" height="8" viewBox="0 0 20 8" aria-hidden="true" style="flex-shrink:0;"><line x1="0" y1="4" x2="14" y2="4" stroke="#D97706" stroke-width="2"/><polygon points="14,1 20,4 14,7" fill="#D97706"/></svg><strong>Supersedes</strong> <span class="gl-sub">(newer legal update)</span></span>',
                    '<span class="gl-item"><svg width="20" height="8" viewBox="0 0 20 8" aria-hidden="true" style="flex-shrink:0;"><line x1="0" y1="4" x2="14" y2="4" stroke="#D4D4D8" stroke-width="2"/><polygon points="14,1 20,4 14,7" fill="#D4D4D8"/></svg><strong>Asserts</strong> <span class="gl-sub">(file claims fact)</span></span>',
                    '<span class="gl-item"><svg width="20" height="8" viewBox="0 0 20 8" aria-hidden="true" style="flex-shrink:0;"><line x1="0" y1="4" x2="14" y2="4" stroke="#A1A1AA" stroke-width="2" stroke-dasharray="2,2"/><polygon points="14,1 20,4 14,7" fill="#A1A1AA"/></svg><strong>About</strong> <span class="gl-sub">(links claim to entity)</span></span>',
                    "</div>",
                    "</div>",
                    '<div class="gl-hint">Drag nodes to inspect · Scroll to zoom · Hover any node or arrow for details.</div>',
                    "</div>",
                ]
            )
            st.markdown(legend_html, unsafe_allow_html=True)
            graph, _, _ = load_graph_artifacts(storage_dir)
            target_ents = (
                None
                if show_full
                else (response.dossier.target_entities if response.dossier else None)
            )
            pyvis_html = build_offline_pyvis_html(graph, highlight_entities=target_ents)
            st.iframe(
                pyvis_html,
                height=500,
                width="stretch",
                alt="Interactive offline evidence graph",
            )
        else:
            st.info("Evidence graph not built yet.")

    # Zone 4: Raw Reasoning Dossier Drawer
    with st.expander("View Full Technical Audit (JSON)"):
        st.json(response.model_dump(mode="json"))


if __name__ == "__main__":
    main()

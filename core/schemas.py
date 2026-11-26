"""Pydantic v2 stage-boundary schemas and data contracts for CERBERUS-X.

All inter-stage payloads (ingestion, forensics, extraction, graph building,
and reasoning) MUST be strictly validated with these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Modality(str, Enum):
    """Supported multimodal input and evidence modalities."""

    PDF = "PDF"
    TABLE = "TABLE"
    IMAGE = "IMAGE"
    SCAN = "SCAN"
    JSON = "JSON"
    TEXT = "TEXT"


class ForensicSeverity(str, Enum):
    """Severity ratings for forensic flags."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class EdgeType(str, Enum):
    """Evidence graph relationship edge classifications."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"


class VerdictStatus(str, Enum):
    """High-level verdict classifications for query evaluation."""

    VERIFIED = "VERIFIED"
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"
    TAMPER_DETECTED = "TAMPER_DETECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class LocationRef(BaseModel):
    """Verifiable physical or structured document coordinates."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = None
    row: int | None = None
    column: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    char_span: tuple[int, int] | None = None
    snippet: str = ""

    @model_validator(mode="after")
    def validate_at_least_one_coordinate(self) -> LocationRef:
        """Enforce that at least one coordinate dimension is provided."""
        if (
            self.page is None
            and self.row is None
            and self.bbox is None
            and self.char_span is None
        ):
            raise ValueError(
                "LocationRef must specify at least one coordinate: page, row, bbox, or char_span."
            )
        return self


class ForensicFlag(BaseModel):
    """A detected anomaly or security flag from pre-LLM forensic analysis."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: ForensicSeverity
    detail: str
    location: LocationRef | None = None


class SourceDoc(BaseModel):
    """Ingested source document metadata, integrity hashes, and prior trust."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    file_path: str
    file_name: str
    modality: Modality
    sha256: str
    trust_prior: float = Field(..., ge=0.0, le=1.0)
    forensic_flags: list[ForensicFlag] = Field(default_factory=list)
    quarantined: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


class ParsedChunk(BaseModel):
    """Sanitized document content chunk with verifiable location provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_id: str
    modality: Modality
    text: str
    location_ref: LocationRef
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    sanitized: bool = True


class Claim(BaseModel):
    """Atomic factual assertion extracted with full multimodal provenance."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    entity_raw: str
    entity_canonical: str | None = None
    attribute: str
    value: str
    numeric_value: float | None = None
    unit: str | None = None
    timestamp: str | None = None
    modality: Modality
    source_id: str
    location_ref: LocationRef
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_amendment: bool = False


class ConflictEdge(BaseModel):
    """Directed edge in the evidence graph representing support, conflict, or supersession."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    claim_a_id: str
    claim_b_id: str
    edge_type: EdgeType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    detector: str


class Citation(BaseModel):
    """Verifiable citation linking an answer directly to source claims and coordinates."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    file_name: str
    claim_id: str
    location_ref: LocationRef
    quote: str


class EvidenceDossier(BaseModel):
    """Aggregated subgraph retrieval context for query synthesis."""

    model_config = ConfigDict(extra="forbid")

    query: str
    target_entities: list[str]
    verified_claims: list[Claim]
    conflicting_edges: list[ConflictEdge]
    quarantined_sources: list[SourceDoc]
    max_trust_score: float = Field(..., ge=0.0, le=1.0)
    missing_attributes: list[str] = Field(default_factory=list)


class VerdictResponse(BaseModel):
    """Final conflict-aware synthesis output with provenance and abstention status."""

    model_config = ConfigDict(extra="forbid")

    query: str
    status: VerdictStatus
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[Citation]
    conflicts: list[ConflictEdge]
    quarantined_sources: list[SourceDoc]
    dossier: EvidenceDossier | None = None


__all__ = [
    "Citation",
    "Claim",
    "ConflictEdge",
    "EdgeType",
    "EvidenceDossier",
    "ForensicFlag",
    "ForensicSeverity",
    "LocationRef",
    "Modality",
    "ParsedChunk",
    "SourceDoc",
    "VerdictResponse",
    "VerdictStatus",
]

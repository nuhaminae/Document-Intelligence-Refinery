# src/models/models.py
# Pydantic schemas for the Document Intelligence Refinery.
# Modified after Peer's explainer

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# --- Enumerations ---


class OriginType(str, Enum):
    """Differentiates between digital, scanned, and mixed documents."""

    digital = "digital"
    scanned = "scanned"
    mixed = "mixed"
    form_fillable = "form_fillable"


class LayoutComplexity(str, Enum):
    """Differentiates between simple and complex document layouts."""

    single_column = "single_column"
    multi_column = "multi_column"
    table_heavy = "table_heavy"
    figure_heavy = "figure_heavy"
    mixed = "mixed"


class StrategyType(str, Enum):
    """Differentiates between extraction strategies."""

    fasttext = "FastText"
    layout_aware = "LayoutAware"
    vision_augmented = "VisionAugmented"


class LDUType(str, Enum):
    """Logical Document Unit type."""

    paragraph = "paragraph"
    table = "table"
    figure = "figure"
    caption = "caption"


# --- Core Models ---


class DocumentProfile(BaseModel):
    """Document profile produced by the triage agent."""

    document_id: str
    file_path: str
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    domain_hint: Optional[str] = None
    char_density: float
    whitespace_ratio: float
    bbox_distribution: Dict[str, List[float]]
    triage_confidence: float


class ProvenanceChain(BaseModel):
    """Provenance record for one extracted LDU."""

    ldu_id: str
    strategy_used: StrategyType
    source_bbox: Tuple[float, float, float, float]
    source_page: int
    transformations: List[str]
    confidence_score: float
    content_hash: str


class LDU(BaseModel):
    """Logical Document Unit: atomic unit of extracted document content."""

    ldu_id: str
    type: LDUType
    text: Optional[str] = None
    table_data: Optional[List[List[str]]] = None
    figure_ref: Optional[str] = None
    bbox: Tuple[float, float, float, float]
    page_number: int


class PageIndex(BaseModel):
    """Page-level navigation and layout summary."""

    page_number: int
    ldus: List[LDU]
    char_density: float
    whitespace_ratio: float
    layout_signature: Dict[str, List[float]]


class ExtractedDocument(BaseModel):
    """
    Unified extraction result.

    Important:
    - `extraction_confidence` is now a post-extraction quality score.
    - It should not simply copy `DocumentProfile.triage_confidence`.
    - `output_quality` stores Gate 2 evidence from src/utils/extraction_quality.py.
    """

    document_id: str
    strategy_used: StrategyType
    content_blocks: List[LDU]
    provenance_chain: List[ProvenanceChain]
    extraction_confidence: float

    # Optional richer artifacts.
    page_indexes: Optional[List[PageIndex]] = None

    # Gate 2 quality evidence. Kept as dict to avoid circular imports between
    # src.models and src.utils.extraction_quality.
    output_quality: Optional[Dict[str, Any]] = Field(default=None)

# src/models/models.py
# script to define pydantic schemas

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


# --- Enumeration for set of constants ---
class OriginType(str, Enum):
    """Differentiates between digital and scanned documents"""

    digital = "digital"
    scanned = "scanned"
    mixed = "mixed"
    form_fillable = "form_fillable"


class LayoutComplexity(str, Enum):
    """Differentiates between simple and complex layouts"""

    single_column = "single_column"
    multi_column = "multi_column"
    table_heavy = "table_heavy"
    figure_heavy = "figure_heavy"
    mixed = "mixed"


class StrategyType(str, Enum):
    """Differentiates between extraction strategies"""

    fasttext = "FastText"
    layout_aware = "LayoutAware"
    vision_augmented = "VisionAugmented"


class LDUType(str, Enum):
    """Differentiates between Logic Document Unit (LDU) types"""

    paragraph = "paragraph"
    table = "table"
    figure = "figure"
    caption = "caption"


# --- Core Models ---
class DocumentProfile(BaseModel):
    """Core document profile model"""

    document_id: str
    file_path: str
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    domain_hint: Optional[str]
    char_density: float
    whitespace_ratio: float
    bbox_distribution: Dict[str, List[float]]
    triage_confidence: float


class ProvenanceChain(BaseModel):
    """Provenance chain model  that tracks how each LDU was extracted and transformed"""

    ldu_id: str
    strategy_used: StrategyType
    source_bbox: Tuple[float, float, float, float]
    source_page: int
    transformations: List[str]
    confidence_score: float
    content_hash: str


class LDU(BaseModel):
    """Logic Document Unit (LDU) model that captures atomic units of content"""

    ldu_id: str
    type: LDUType
    text: Optional[str]
    table_data: Optional[List[List[str]]]
    figure_ref: Optional[str]
    bbox: Tuple[float, float, float, float]
    page_number: int


class PageIndex(BaseModel):
    """Page index model that captures page metrics and LDU data for efficeint retrieval"""

    page_number: int
    ldus: List[LDU]
    char_density: float
    whitespace_ratio: float
    layout_signature: Dict[str, List[float]]


class ExtractedDocument(BaseModel):
    """Extracted document model that holds unified extraction results of a document"""

    document_id: str
    strategy_used: StrategyType
    content_blocks: List[LDU]
    provenance_chain: List[ProvenanceChain]
    extraction_confidence: float

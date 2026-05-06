# src/utils/extraction_quality.py

# Added after Peer's explainer

"""
Post-extraction quality scoring for the Document Intelligence Refinery.

This module implements Gate 2 of the extraction router:

    Gate 1: choose an initial extraction strategy from the DocumentProfile.
    Gate 2: measure whether the extractor actually produced usable output.
    Gate 3: accept the output or escalate to the next strategy.

Why this exists:
    Triage confidence is a pre-extraction signal. It says how difficult the
    document appears before extraction.

    Extraction confidence should be a post-extraction signal. It should measure
    whether the chosen extractor actually returned useful content, provenance,
    bounding boxes, tables, and non-empty text.

Use this module from:
    - src/strategies/fasttext_extractor.py
    - src/strategies/layout_extractor.py
    - src/strategies/vision_extractor.py
    - src/agents/extractor.py
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    import yaml
except (
    ImportError
):  # pragma: no cover - yaml should exist in your env, but keep safe fallback.
    yaml = None


BBox = Tuple[float, float, float, float]


class BBoxPrecision(str, Enum):
    """Coarse classification of the spatial quality of extracted blocks."""

    none = "none"
    invalid = "invalid"
    coarse_page = "coarse_page"
    block = "block"


class OutputQualityThresholds(BaseModel):
    """
    Thresholds for post-extraction validation.

    These defaults are intentionally conservative but not extreme.
    Override them from rubric/extraction_rules.yaml once you tune on your corpus.
    """

    min_content_blocks: int = Field(
        default=20,
        description="Minimum extracted LDUs/content blocks expected for a usable document.",
    )
    target_content_blocks: int = Field(
        default=200,
        description="Block count at which the block-count score saturates.",
    )
    min_nonempty_text_ratio: float = Field(
        default=0.80,
        description="Minimum share of blocks that must contain text/table/figure content.",
    )
    min_bbox_coverage_ratio: float = Field(
        default=0.70,
        description="Minimum share of blocks with valid non-zero bounding boxes.",
    )
    min_provenance_coverage_ratio: float = Field(
        default=0.70,
        description="Minimum share of blocks with usable provenance records.",
    )
    min_confidence_to_accept: float = Field(
        default=0.75,
        description="Minimum remeasured confidence required to accept the extraction.",
    )
    max_placeholder_bbox_ratio: float = Field(
        default=0.30,
        description="Maximum tolerated share of placeholder or invalid bounding boxes.",
    )
    max_duplicate_bbox_ratio: float = Field(
        default=0.80,
        description=(
            "Maximum tolerated share of duplicated bboxes. High duplication often "
            "means page-level placeholder boxes were used for every line."
        ),
    )

    block_count_weight: float = 0.30
    nonempty_text_weight: float = 0.20
    bbox_coverage_weight: float = 0.20
    provenance_coverage_weight: float = 0.20
    table_bonus_weight: float = 0.10

    placeholder_bbox_penalty: float = 0.20
    duplicate_bbox_penalty: float = 0.10


class OutputQuality(BaseModel):
    """Post-extraction quality evidence for one extractor run."""

    content_blocks_returned: int
    nonempty_blocks: int
    empty_blocks: int

    blocks_with_real_bbox: int
    placeholder_bbox_count: int
    duplicate_bbox_count: int

    tables_extracted: int
    figures_extracted: int

    coverage_ratio: float
    nonempty_text_ratio: float
    provenance_coverage_ratio: float
    duplicate_bbox_ratio: float
    placeholder_bbox_ratio: float

    bbox_precision: BBoxPrecision

    extraction_confidence_remeasured: float
    quality_passed: bool

    validation_failure: Optional[str] = None
    validation_failures: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    scoring_breakdown: Dict[str, float] = Field(default_factory=dict)


def load_quality_thresholds(
    config_path: str | Path = "rubric/extraction_rules.yaml",
) -> OutputQualityThresholds:
    """
    Load post-extraction quality thresholds from YAML if available.

    Supported YAML section:

        post_extraction_quality:
          min_content_blocks: 20
          target_content_blocks: 200
          min_nonempty_text_ratio: 0.80
          min_bbox_coverage_ratio: 0.70
          min_provenance_coverage_ratio: 0.70
          min_confidence_to_accept: 0.75
          max_placeholder_bbox_ratio: 0.30
          max_duplicate_bbox_ratio: 0.80

    If the file or section does not exist, defaults are returned.
    """

    path = Path(config_path)
    if yaml is None or not path.exists():
        return OutputQualityThresholds()

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return OutputQualityThresholds()

    section = data.get("post_extraction_quality", {})
    if not isinstance(section, dict):
        return OutputQualityThresholds()

    allowed = set(OutputQualityThresholds.model_fields.keys())
    filtered = {key: value for key, value in section.items() if key in allowed}

    try:
        return OutputQualityThresholds(**filtered)
    except Exception:
        return OutputQualityThresholds()


def score_extracted_document(
    extracted_doc: Any,
    thresholds: Optional[OutputQualityThresholds] = None,
) -> OutputQuality:
    """
    Score an ExtractedDocument-like object.

    Expected object shape:
        extracted_doc.content_blocks
        extracted_doc.provenance_chain

    This function avoids importing your Pydantic models directly so it can be used
    safely from strategy files without creating circular imports.
    """

    return compute_output_quality(
        content_blocks=getattr(extracted_doc, "content_blocks", []) or [],
        provenance_chain=getattr(extracted_doc, "provenance_chain", []) or [],
        thresholds=thresholds,
    )


def compute_output_quality(
    content_blocks: Sequence[Any],
    provenance_chain: Optional[Sequence[Any]] = None,
    thresholds: Optional[OutputQualityThresholds] = None,
    extra_tables_extracted: int = 0,
    extra_figures_extracted: int = 0,
) -> OutputQuality:
    """
    Compute post-extraction quality from actual extractor output.

    This is the main Gate 2 function.

    Args:
        content_blocks:
            Sequence of LDU-like objects. Current repo LDUs have:
                - text
                - table_data
                - figure_ref
                - bbox
                - type
                - page_number

        provenance_chain:
            Sequence of ProvenanceChain-like objects. Current repo provenance has:
                - source_bbox
                - source_page
                - content_hash
                - confidence_score

        thresholds:
            Optional OutputQualityThresholds. Defaults are used if omitted.

        extra_tables_extracted:
            Optional table count from external layout tools if tables are not
            represented as LDU.table_data.

        extra_figures_extracted:
            Optional figure count from external layout tools.

    Returns:
        OutputQuality:
            A structured object containing quality evidence, confidence,
            pass/fail decision, warnings, and scoring breakdown.
    """

    thresholds = thresholds or OutputQualityThresholds()
    provenance_chain = provenance_chain or []

    total_blocks = len(content_blocks)
    nonempty_blocks = sum(1 for block in content_blocks if _block_has_content(block))
    empty_blocks = max(total_blocks - nonempty_blocks, 0)

    block_bboxes = [_get_block_bbox(block) for block in content_blocks]
    valid_bboxes = [bbox for bbox in block_bboxes if _is_valid_bbox(bbox)]
    placeholder_bbox_count = sum(
        1 for bbox in block_bboxes if _is_placeholder_bbox(bbox)
    )

    blocks_with_real_bbox = len(valid_bboxes)
    duplicate_bbox_count = _count_duplicate_bboxes(content_blocks, block_bboxes)

    tables_extracted = _count_tables(content_blocks) + int(extra_tables_extracted)
    figures_extracted = _count_figures(content_blocks) + int(extra_figures_extracted)

    coverage_ratio = _safe_ratio(blocks_with_real_bbox, total_blocks)
    nonempty_text_ratio = _safe_ratio(nonempty_blocks, total_blocks)
    placeholder_bbox_ratio = _safe_ratio(placeholder_bbox_count, total_blocks)
    duplicate_bbox_ratio = _safe_ratio(duplicate_bbox_count, total_blocks)
    provenance_coverage_ratio = _compute_provenance_coverage(
        provenance_chain=provenance_chain,
        total_blocks=total_blocks,
    )

    bbox_precision = _classify_bbox_precision(
        total_blocks=total_blocks,
        blocks_with_real_bbox=blocks_with_real_bbox,
        duplicate_bbox_ratio=duplicate_bbox_ratio,
        placeholder_bbox_ratio=placeholder_bbox_ratio,
    )

    block_count_score = min(
        1.0, _safe_ratio(total_blocks, thresholds.target_content_blocks)
    )
    nonempty_score = min(1.0, nonempty_text_ratio)
    bbox_score = min(1.0, coverage_ratio)
    provenance_score = min(1.0, provenance_coverage_ratio)
    table_score = min(1.0, tables_extracted / 3.0) if tables_extracted > 0 else 0.0

    raw_confidence = (
        thresholds.block_count_weight * block_count_score
        + thresholds.nonempty_text_weight * nonempty_score
        + thresholds.bbox_coverage_weight * bbox_score
        + thresholds.provenance_coverage_weight * provenance_score
        + thresholds.table_bonus_weight * table_score
    )

    placeholder_penalty = thresholds.placeholder_bbox_penalty * placeholder_bbox_ratio

    duplicate_penalty = 0.0
    if duplicate_bbox_ratio > thresholds.max_duplicate_bbox_ratio:
        excess_duplicate_ratio = (
            duplicate_bbox_ratio - thresholds.max_duplicate_bbox_ratio
        )
        duplicate_penalty = thresholds.duplicate_bbox_penalty * excess_duplicate_ratio

    confidence = _clamp01(raw_confidence - placeholder_penalty - duplicate_penalty)

    validation_failures: List[str] = []
    warnings: List[str] = []

    if total_blocks < thresholds.min_content_blocks:
        validation_failures.append(
            f"content_blocks_returned={total_blocks} < "
            f"min_content_blocks={thresholds.min_content_blocks}"
        )

    if nonempty_text_ratio < thresholds.min_nonempty_text_ratio:
        validation_failures.append(
            f"nonempty_text_ratio={nonempty_text_ratio:.3f} < "
            f"min_nonempty_text_ratio={thresholds.min_nonempty_text_ratio:.3f}"
        )

    if coverage_ratio < thresholds.min_bbox_coverage_ratio:
        validation_failures.append(
            f"bbox_coverage_ratio={coverage_ratio:.3f} < "
            f"min_bbox_coverage_ratio={thresholds.min_bbox_coverage_ratio:.3f}"
        )

    if provenance_coverage_ratio < thresholds.min_provenance_coverage_ratio:
        validation_failures.append(
            f"provenance_coverage_ratio={provenance_coverage_ratio:.3f} < "
            f"min_provenance_coverage_ratio={thresholds.min_provenance_coverage_ratio:.3f}"
        )

    if placeholder_bbox_ratio > thresholds.max_placeholder_bbox_ratio:
        validation_failures.append(
            f"placeholder_bbox_ratio={placeholder_bbox_ratio:.3f} > "
            f"max_placeholder_bbox_ratio={thresholds.max_placeholder_bbox_ratio:.3f}"
        )

    if confidence < thresholds.min_confidence_to_accept:
        validation_failures.append(
            f"extraction_confidence_remeasured={confidence:.3f} < "
            f"min_confidence_to_accept={thresholds.min_confidence_to_accept:.3f}"
        )

    if duplicate_bbox_ratio > thresholds.max_duplicate_bbox_ratio:
        warnings.append(
            f"duplicate_bbox_ratio={duplicate_bbox_ratio:.3f} is high; "
            "extraction may be using coarse page-level bboxes."
        )

    if bbox_precision == BBoxPrecision.coarse_page:
        warnings.append(
            "Bounding boxes appear coarse or duplicated. Provenance may be page-level "
            "rather than block-level."
        )

    quality_passed = len(validation_failures) == 0

    return OutputQuality(
        content_blocks_returned=total_blocks,
        nonempty_blocks=nonempty_blocks,
        empty_blocks=empty_blocks,
        blocks_with_real_bbox=blocks_with_real_bbox,
        placeholder_bbox_count=placeholder_bbox_count,
        duplicate_bbox_count=duplicate_bbox_count,
        tables_extracted=tables_extracted,
        figures_extracted=figures_extracted,
        coverage_ratio=round(coverage_ratio, 6),
        nonempty_text_ratio=round(nonempty_text_ratio, 6),
        provenance_coverage_ratio=round(provenance_coverage_ratio, 6),
        duplicate_bbox_ratio=round(duplicate_bbox_ratio, 6),
        placeholder_bbox_ratio=round(placeholder_bbox_ratio, 6),
        bbox_precision=bbox_precision,
        extraction_confidence_remeasured=round(confidence, 6),
        quality_passed=quality_passed,
        validation_failure=validation_failures[0] if validation_failures else None,
        validation_failures=validation_failures,
        warnings=warnings,
        scoring_breakdown={
            "block_count_score": round(block_count_score, 6),
            "nonempty_score": round(nonempty_score, 6),
            "bbox_score": round(bbox_score, 6),
            "provenance_score": round(provenance_score, 6),
            "table_score": round(table_score, 6),
            "raw_confidence": round(raw_confidence, 6),
            "placeholder_penalty": round(placeholder_penalty, 6),
            "duplicate_penalty": round(duplicate_penalty, 6),
            "final_confidence": round(confidence, 6),
        },
    )


def quality_to_extraction_confidence(quality: OutputQuality) -> float:
    """
    Convenience helper for strategy files.

    Example:
        quality = score_extracted_document(extracted_doc)
        extracted_doc.extraction_confidence = quality_to_extraction_confidence(quality)
    """

    return quality.extraction_confidence_remeasured


def _block_has_content(block: Any) -> bool:
    text = getattr(block, "text", None)
    table_data = getattr(block, "table_data", None)
    figure_ref = getattr(block, "figure_ref", None)

    if isinstance(text, str) and text.strip():
        return True

    if table_data:
        return True

    if isinstance(figure_ref, str) and figure_ref.strip():
        return True

    return False


def _count_tables(content_blocks: Sequence[Any]) -> int:
    count = 0

    for block in content_blocks:
        block_type = getattr(block, "type", None)
        block_type_value = getattr(block_type, "value", str(block_type)).lower()

        if block_type_value == "table":
            count += 1
            continue

        table_data = getattr(block, "table_data", None)
        if table_data:
            count += 1

    return count


def _count_figures(content_blocks: Sequence[Any]) -> int:
    count = 0

    for block in content_blocks:
        block_type = getattr(block, "type", None)
        block_type_value = getattr(block_type, "value", str(block_type)).lower()

        if block_type_value == "figure":
            count += 1
            continue

        figure_ref = getattr(block, "figure_ref", None)
        if isinstance(figure_ref, str) and figure_ref.strip():
            count += 1

    return count


def _get_block_bbox(block: Any) -> Optional[BBox]:
    bbox = getattr(block, "bbox", None)
    return _normalize_bbox(bbox)


def _get_provenance_bbox(prov: Any) -> Optional[BBox]:
    bbox = getattr(prov, "source_bbox", None)
    return _normalize_bbox(bbox)


def _normalize_bbox(value: Any) -> Optional[BBox]:
    if value is None:
        return None

    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None

    return (x0, y0, x1, y1)


def _is_valid_bbox(bbox: Optional[BBox]) -> bool:
    if bbox is None:
        return False

    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    return width > 0 and height > 0


def _is_placeholder_bbox(bbox: Optional[BBox]) -> bool:
    if bbox is None:
        return True

    x0, y0, x1, y1 = bbox

    if not _is_valid_bbox(bbox):
        return True

    # Common placeholder patterns.
    if (x0, y0, x1, y1) == (0.0, 0.0, 0.0, 0.0):
        return True

    return False


def _count_duplicate_bboxes(
    content_blocks: Sequence[Any],
    bboxes: Sequence[Optional[BBox]],
) -> int:
    """
    Count how many blocks share repeated bboxes on the same page.

    FastText currently uses the full page bbox for every extracted line. That is
    not invalid, but it is coarse. A very high duplicate ratio is a warning that
    provenance is probably page-level rather than block-level.
    """

    if not content_blocks or not bboxes:
        return 0

    seen: Dict[Tuple[int, BBox], int] = {}

    for block, bbox in zip(content_blocks, bboxes):
        if bbox is None or not _is_valid_bbox(bbox):
            continue

        page_number = int(getattr(block, "page_number", 0) or 0)
        key = (page_number, bbox)
        seen[key] = seen.get(key, 0) + 1

    duplicate_count = 0
    for count in seen.values():
        if count > 1:
            duplicate_count += count - 1

    return duplicate_count


def _compute_provenance_coverage(
    provenance_chain: Sequence[Any],
    total_blocks: int,
) -> float:
    if total_blocks <= 0:
        return 0.0

    usable = 0

    for prov in provenance_chain:
        bbox = _get_provenance_bbox(prov)
        content_hash = getattr(prov, "content_hash", None)
        source_page = getattr(prov, "source_page", None)

        has_hash = isinstance(content_hash, str) and bool(content_hash.strip())
        has_page = source_page is not None
        has_bbox = _is_valid_bbox(bbox)

        if has_hash and has_page and has_bbox:
            usable += 1

    return _safe_ratio(usable, total_blocks)


def _classify_bbox_precision(
    total_blocks: int,
    blocks_with_real_bbox: int,
    duplicate_bbox_ratio: float,
    placeholder_bbox_ratio: float,
) -> BBoxPrecision:
    if total_blocks <= 0:
        return BBoxPrecision.none

    if blocks_with_real_bbox == 0:
        return BBoxPrecision.invalid

    if placeholder_bbox_ratio >= 1.0:
        return BBoxPrecision.invalid

    if duplicate_bbox_ratio >= 0.80:
        return BBoxPrecision.coarse_page

    return BBoxPrecision.block


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def summarize_quality_for_log(quality: OutputQuality) -> Dict[str, Any]:
    """
    Return a compact dict suitable for extraction_ledger.jsonl.

    This avoids dumping overly verbose model internals while still preserving
    the evidence needed to debug routing decisions.
    """

    return {
        "content_blocks_returned": quality.content_blocks_returned,
        "nonempty_blocks": quality.nonempty_blocks,
        "blocks_with_real_bbox": quality.blocks_with_real_bbox,
        "placeholder_bbox_count": quality.placeholder_bbox_count,
        "duplicate_bbox_count": quality.duplicate_bbox_count,
        "tables_extracted": quality.tables_extracted,
        "figures_extracted": quality.figures_extracted,
        "coverage_ratio": quality.coverage_ratio,
        "nonempty_text_ratio": quality.nonempty_text_ratio,
        "provenance_coverage_ratio": quality.provenance_coverage_ratio,
        "duplicate_bbox_ratio": quality.duplicate_bbox_ratio,
        "bbox_precision": quality.bbox_precision.value,
        "extraction_confidence_remeasured": quality.extraction_confidence_remeasured,
        "quality_passed": quality.quality_passed,
        "validation_failure": quality.validation_failure,
        "validation_failures": quality.validation_failures,
        "warnings": quality.warnings,
    }

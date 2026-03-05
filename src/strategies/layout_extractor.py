# src/strategies/layout_extractor.py
# script to define extraction strategies with Docling/MinerU


import hashlib
from typing import List

from docling.document_converter import DocumentConverter

from src.models.models import (
    LDU,
    DocumentProfile,
    ExtractedDocument,
    LDUType,
    PageIndex,
    ProvenanceChain,
    StrategyType,
)


class LayoutExtractor:
    """
    Strategy B: Layout-Aware (Docling)
    - Best for multi-column, table-heavy, figure-heavy PDFs.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []
        page_indexes: List[PageIndex] = []

        converter = DocumentConverter()
        result = converter.convert(self.pdf_path)
        doc = result.document  # DoclingDocument

        for page in doc.pages:
            page_ldus: List[LDU] = []

            for block in page.blocks:
                ldu_id = f"ldu_{page.page_number}_{block.id}"
                bbox = tuple(block.bbox) if hasattr(block, "bbox") else (0, 0, 0, 0)

                # --- Paragraphs ---
                if block.type == "paragraph":
                    content = block.text or ""
                    ldu_type = LDUType.paragraph

                # --- Tables (group cells with header row) ---
                elif block.type == "table":
                    header = block.cells[0] if block.cells else []
                    rows = block.cells[1:] if len(block.cells) > 1 else []
                    content = str([header] + rows)
                    ldu_type = LDUType.table

                # --- Figures (attach captions if available) ---
                elif block.type == "figure":
                    caption = getattr(block, "caption", None)
                    content = caption if caption else getattr(block, "image_ref", "")
                    ldu_type = LDUType.figure

                else:
                    content = getattr(block, "text", "")
                    ldu_type = LDUType.paragraph

                # --- Compute content hash ---
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                # --- Build LDU ---
                ldu = LDU(
                    ldu_id=ldu_id,
                    type=ldu_type,
                    text=block.text if ldu_type == LDUType.paragraph else None,
                    table_data=block.cells if ldu_type == LDUType.table else None,
                    figure_ref=(
                        getattr(block, "image_ref", None)
                        if ldu_type == LDUType.figure
                        else None
                    ),
                    bbox=bbox,
                    page_number=page.page_number,
                )
                ldus.append(ldu)
                page_ldus.append(ldu)

                # --- Provenance ---
                provenance.append(
                    ProvenanceChain(
                        ldu_id=ldu_id,
                        strategy_used=StrategyType.layout_aware,
                        source_bbox=bbox,
                        source_page=page.page_number,
                        transformations=["docling_parse"],
                        confidence_score=profile.triage_confidence,
                        content_hash=content_hash,
                    )
                )

            # --- Build PageIndex ---
            page_indexes.append(
                PageIndex(
                    page_number=page.page_number,
                    ldus=page_ldus,
                    char_density=profile.char_density,
                    whitespace_ratio=profile.whitespace_ratio,
                    layout_signature=profile.bbox_distribution,
                )
            )

        return ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.layout_aware,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=profile.triage_confidence,
        )

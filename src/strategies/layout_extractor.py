# src/strategies/layout_extractor.py
# script to define extraction strategies with Docling/MinerU

# src/strategies/layout_extractor.py

from typing import List

from docling.document_converter import DocumentConverter

from src.models.models import (
    LDU,
    DocumentProfile,
    ExtractedDocument,
    LDUType,
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

        converter = DocumentConverter()
        result = converter.convert(self.pdf_path)
        doc = result.document  # DoclingDocument

        for page in doc.pages:
            for block in page.blocks:
                ldu_id = f"ldu_{page.page_number}_{block.id}"
                bbox = tuple(block.bbox) if hasattr(block, "bbox") else (0, 0, 0, 0)

                if block.type == "paragraph":
                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=LDUType.paragraph,
                            text=block.text,
                            table_data=None,
                            figure_ref=None,
                            bbox=bbox,
                            page_number=page.page_number,
                        )
                    )
                elif block.type == "table":
                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=LDUType.table,
                            text=None,
                            table_data=block.cells,
                            figure_ref=None,
                            bbox=bbox,
                            page_number=page.page_number,
                        )
                    )
                elif block.type == "figure":
                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=LDUType.figure,
                            text=None,
                            table_data=None,
                            figure_ref=getattr(block, "image_ref", None),
                            bbox=bbox,
                            page_number=page.page_number,
                        )
                    )

                provenance.append(
                    ProvenanceChain(
                        ldu_id=ldu_id,
                        strategy_used=StrategyType.layout_aware,
                        source_bbox=bbox,
                        source_page=page.page_number,
                        transformations=["docling_parse"],
                        confidence_score=profile.triage_confidence,
                    )
                )

        return ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.layout_aware,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=profile.triage_confidence,
        )

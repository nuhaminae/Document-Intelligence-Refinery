# src/strategies/vision_extractor.py
# script to define extraction strategies with MinerU

import hashlib
from typing import List

from mineru.backend.hybrid.hybrid_analyze import doc_analyze
from mineru.data.data_reader_writer.dummy import DummyDataWriter

from src.models.models import (
    LDU,
    DocumentProfile,
    ExtractedDocument,
    LDUType,
    PageIndex,
    ProvenanceChain,
    StrategyType,
)


class VisionExtractor:
    """
    Strategy C: Vision-Augmented (MinerU Hybrid Analyzer)
    - Best for scanned PDFs, low-density text, or complex layouts.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []
        page_indexes: List[PageIndex] = []

        # MinerU hybrid analyzer returns structured JSON
        result = doc_analyze(self.pdf_path, image_writer=DummyDataWriter())

        metadata = result.get("metadata", {})
        extraction_conf = metadata.get("triage_confidence", profile.triage_confidence)
        domain_hint = metadata.get("domain_hint", profile.domain_hint)

        for page in result.get("pages", []):  # MinerU may expose per-page info
            page_ldus: List[LDU] = []

            for block in page.get("blocks", []):
                confidence_score = block.get("confidence", extraction_conf)
                ldu_id = f"ldu_{block.get('page', 0)}_{block.get('id', 'x')}"
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))

                # --- Normalize block types ---
                block_type = block.get("type", "paragraph")
                if block_type in ["paragraph", "text"]:
                    ldu_type = LDUType.paragraph
                    content = block.get("text", "")
                elif block_type == "table":
                    ldu_type = LDUType.table
                    content = str(block.get("table", []))
                elif block_type == "figure":
                    ldu_type = LDUType.figure
                    content = block.get("figure", "")
                else:
                    ldu_type = LDUType.paragraph
                    content = block.get("text", "")

                # --- Compute content hash ---
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                # --- Build LDU ---
                ldu = LDU(
                    ldu_id=ldu_id,
                    type=ldu_type,
                    text=block.get("text"),
                    table_data=block.get("table"),
                    figure_ref=block.get("figure"),
                    bbox=bbox,
                    page_number=block.get("page", 0),
                )
                ldus.append(ldu)
                page_ldus.append(ldu)

                # --- Provenance ---
                provenance.append(
                    ProvenanceChain(
                        ldu_id=ldu_id,
                        strategy_used=StrategyType.vision_augmented,
                        source_bbox=bbox,
                        source_page=block.get("page", 0),
                        transformations=["mineru_hybrid_analyze"],
                        confidence_score=confidence_score,
                        content_hash=content_hash,
                    )
                )

            # --- Build PageIndex ---
            page_indexes.append(
                PageIndex(
                    page_number=page.get("page", 0),
                    ldus=page_ldus,
                    char_density=metadata.get("char_density", profile.char_density),
                    whitespace_ratio=metadata.get(
                        "whitespace_ratio", profile.whitespace_ratio
                    ),
                    layout_signature=metadata.get(
                        "bbox_distribution", profile.bbox_distribution
                    ),
                )
            )

        return ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.vision_augmented,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=extraction_conf,
        )
